from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from quant.core.paths import PROJECT_ROOT
from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.research.byd_daily_t_plan import build_daily_t_plan
from quant.strategies.custom.byd_minute_t import (
    BydHolding,
    build_minute_payload,
    load_daily_qfq,
)


WorkspacePayload = dict[str, Any]
SnapshotReader = Callable[..., WorkspacePayload | None]
SnapshotWriter = Callable[..., None]
FrameLoader = Callable[[], pd.DataFrame]
PayloadBuilder = Callable[..., WorkspacePayload]


def normalize_byd_daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Normalize canonical or legacy BYD daily bars for the T-plan calculators."""

    if daily is None or daily.empty:
        return pd.DataFrame()
    out = daily.copy()
    parsed_date = pd.Series(pd.NaT, index=out.index)
    if "date" in out.columns:
        parsed_date = pd.to_datetime(out["date"], errors="coerce")
    if "trade_date" in out.columns:
        trade_date = pd.to_datetime(
            out["trade_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        parsed_date = parsed_date.fillna(trade_date)
    out["date"] = parsed_date
    if "vol" in out.columns and "volume" not in out.columns:
        out["volume"] = out["vol"]
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    required = ["date", "open", "high", "low", "close"]
    if any(column not in out.columns for column in required):
        return pd.DataFrame()
    return (
        out.dropna(subset=required)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def load_byd_daily_frame(
    daily_dir: Path = PROJECT_ROOT / "data/raw/daily",
    cache_dir: Path = PROJECT_ROOT / "data/cache",
) -> pd.DataFrame:
    """Prefer canonical refreshed daily data and fall back to the local qfq cache."""

    try:
        store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
        daily = store.read_frame(daily_dir.name, "002594.SZ")
        normalized = normalize_byd_daily_frame(daily)
        if not normalized.empty:
            return normalized
    except Exception:
        pass
    return load_daily_qfq(cache_dir)


@lru_cache(maxsize=1)
def load_byd_intraday_validation_frame(
    cache_dir: Path = PROJECT_ROOT / "data/cache",
) -> pd.DataFrame:
    """Load the longest local five-minute history used only for validation."""

    paths = list(cache_dir.glob("baostock_002594_5min_*_qfq.parquet"))
    if not paths:
        raise FileNotFoundError("缺少比亚迪历史5分钟验证数据")
    path = max(paths, key=lambda item: item.stat().st_size)
    return pd.read_parquet(path)


@dataclass(frozen=True)
class BydWorkspaceDependencies:
    """External collaborators used by the BYD daily workspace."""

    read_snapshot: SnapshotReader
    write_snapshot: SnapshotWriter
    load_daily: FrameLoader = load_byd_daily_frame
    load_intraday: FrameLoader = load_byd_intraday_validation_frame
    build_minute_payload: PayloadBuilder = build_minute_payload
    build_daily_plan: PayloadBuilder = build_daily_t_plan


def build_byd_daily_strategy(
    shares: int = 10_000,
    cost: float = 110.6061,
    refresh: bool = False,
    *,
    dependencies: BydWorkspaceDependencies,
) -> WorkspacePayload:
    """Build BYD's fixed pre-market daily T plan and persist its workspace snapshot."""

    params = {
        "plan_version": 3,
        "shares": int(shares),
        "cost": round(float(cost), 6),
    }
    if not refresh:
        cached = dependencies.read_snapshot(
            "byd_daily_plan",
            params=params,
            allow_sql=False,
        )
        if cached is not None:
            return cached

    daily = dependencies.load_daily()
    holding = BydHolding(
        shares=max(int(shares), 0),
        cost=float(cost),
        full_shares=10_000,
    )
    payload = dependencies.build_minute_payload(
        daily=daily,
        minutes=pd.DataFrame(),
        holding=holding,
        data_status="盘前固定日线计划",
    )
    daily_plan = dependencies.build_daily_plan(
        daily=daily,
        intraday=dependencies.load_intraday(),
        shares=holding.shares,
        full_shares=holding.full_shares,
    )
    positive = daily_plan["positive"]
    if positive["execution_enabled"]:
        primary_action = {
            "action": "BUY_LIMIT",
            "title": f"正T挂买单：{positive['shares']}股 × {positive['buy_price']:.2f}",
            "detail": f"{positive['entry_rule']} {positive['exit_rule']}",
            "shares_delta": positive["shares"],
        }
    else:
        primary_action = {
            "action": "WAIT",
            "title": "今日不挂正T买单",
            "detail": (
                "历史规则本身已通过验证，但下一交易日质量分未过闸门；"
                "不为增加次数而追价。反T仍暂停。"
            ),
            "shares_delta": 0,
        }
    payload.update(
        {
            "daily_t_plan": daily_plan,
            "planned_t": daily_plan,
            "validation": daily_plan["validation"],
            "primary_action": primary_action,
            "stage": {
                "key": (
                    "OVERWEIGHT"
                    if holding.shares > holding.full_shares
                    else "UNDERWEIGHT"
                    if holding.shares < 8_000
                    else "BALANCED"
                ),
                "label": (
                    f"高于满仓 {holding.shares - holding.full_shares} 股"
                    if holding.shares > holding.full_shares
                    else f"低于合理仓 {8_000 - holding.shares} 股"
                    if holding.shares < 8_000
                    else "合理仓位"
                ),
                "goal_shares": holding.full_shares,
                "mode": daily_plan["inventory"]["note"],
            },
            "alerts": [],
            "playbook": [
                positive["entry_rule"],
                positive["exit_rule"],
                positive["no_fill_rule"],
                daily_plan["reverse"]["reason"],
                daily_plan["inventory"]["note"],
            ],
        }
    )
    for obsolete_key in [
        "today_t",
        "recent_minutes",
        "intraday_dynamic",
        "indicators",
        "validated_positive_t",
        "intraday_levels",
    ]:
        payload.pop(obsolete_key, None)
    dependencies.write_snapshot(
        "byd_daily_plan",
        daily_plan["signal_date"],
        payload,
        params=params,
        write_sql=refresh,
    )
    return payload
