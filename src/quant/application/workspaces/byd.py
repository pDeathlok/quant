from __future__ import annotations

import os
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
SnapshotValidator = Callable[[WorkspacePayload], bool]
DEFAULT_INTRADAY_TRAINING_MAX_STALENESS_DAYS = 60


def _accept_snapshot(_: WorkspacePayload) -> bool:
    return True


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
    normalized = (
        out.dropna(subset=required)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    normalized.attrs.update(getattr(daily, "attrs", {}))
    return normalized


def _normalized_date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).replace("-", "")
    parsed = (
        pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        if len(text) == 8 and text.isdigit()
        else pd.to_datetime(value, errors="coerce")
    )
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _latest_frame_date(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Timestamp | None:
    if frame is None or frame.empty:
        return None
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column]
        text = values.astype(str).str.replace("-", "", regex=False)
        parsed = (
            pd.to_datetime(text, format="%Y%m%d", errors="coerce")
            if text.str.fullmatch(r"\d{8}").all()
            else pd.to_datetime(values, errors="coerce")
        )
        latest = parsed.max()
        if pd.notna(latest):
            return pd.Timestamp(latest).normalize()
    return None


def _validated_daily_frame(
    daily: pd.DataFrame,
    *,
    expected_date: pd.Timestamp | None,
    source: str,
) -> pd.DataFrame:
    normalized = normalize_byd_daily_frame(daily)
    actual_date = _latest_frame_date(normalized, ("date", "trade_date"))
    if actual_date is None:
        raise RuntimeError(f"BYD daily source has no valid feature date: source={source}")
    if expected_date is not None and actual_date != expected_date:
        raise RuntimeError(
            "BYD daily feature date mismatch: "
            f"source={source} expected={expected_date.date().isoformat()} "
            f"actual={actual_date.date().isoformat()}"
        )
    normalized.attrs.update(
        {
            "daily_feature_source": source,
            "daily_feature_date": actual_date.date().isoformat(),
            "expected_trade_date": (
                expected_date.date().isoformat()
                if expected_date is not None
                else actual_date.date().isoformat()
            ),
        }
    )
    return normalized


def load_byd_daily_frame(
    daily_dir: Path = PROJECT_ROOT / "data/raw/daily",
    cache_dir: Path = PROJECT_ROOT / "data/cache",
    expected_trade_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load current BYD daily features; only use a date-verified qfq fallback."""

    expected_date = _normalized_date(expected_trade_date)
    store: MarketDataStore | None = None
    try:
        store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
        if expected_date is None:
            expected_date = _normalized_date(store.latest_dataset_trade_date(daily_dir.name))
        daily = store.read_frame(daily_dir.name, "002594.SZ")
        return _validated_daily_frame(
            daily,
            expected_date=expected_date,
            source="canonical_market_store",
        )
    except Exception as canonical_error:
        try:
            fallback = load_daily_qfq(cache_dir)
        except Exception as fallback_error:
            raise RuntimeError(
                "BYD daily data unavailable from canonical and qfq sources: "
                f"canonical={canonical_error}; fallback={fallback_error}"
            ) from fallback_error
        if expected_date is None:
            raise RuntimeError(
                "BYD qfq fallback cannot be used without an expected trade date: "
                f"canonical={canonical_error}"
            ) from canonical_error
        return _validated_daily_frame(
            fallback,
            expected_date=expected_date,
            source="local_qfq_fallback",
        )


@lru_cache(maxsize=1)
def load_byd_intraday_validation_frame(
    cache_dir: Path = PROJECT_ROOT / "data/cache",
) -> pd.DataFrame:
    """Load historical five-minute samples used for model training and validation."""

    paths = list(cache_dir.glob("baostock_002594_5min_*_qfq.parquet"))
    if not paths:
        raise FileNotFoundError("缺少比亚迪历史5分钟验证数据")
    path = max(paths, key=lambda item: item.stat().st_size)
    frame = pd.read_parquet(path)
    frame.attrs["intraday_training_source"] = str(path)
    return frame


def _intraday_training_sla_days(configured: int | None) -> int:
    raw_value: Any = (
        configured
        if configured is not None
        else os.getenv(
            "BYD_INTRADAY_TRAINING_MAX_STALENESS_DAYS",
            str(DEFAULT_INTRADAY_TRAINING_MAX_STALENESS_DAYS),
        )
    )
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "BYD intraday training max staleness must be an integer number of days"
        ) from exc
    if value < 0:
        raise ValueError("BYD intraday training max staleness must be non-negative")
    return value


def _feature_freshness(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    *,
    expected_trade_date: str | pd.Timestamp | None,
    intraday_training_max_staleness_days: int,
) -> dict[str, Any]:
    daily_feature_date = _latest_frame_date(daily, ("date", "trade_date"))
    if daily_feature_date is None:
        raise RuntimeError("BYD daily feature frame has no valid date")
    expected_date = (
        _normalized_date(expected_trade_date)
        or _normalized_date(daily.attrs.get("expected_trade_date"))
        or daily_feature_date
    )
    if daily_feature_date != expected_date:
        raise RuntimeError(
            "BYD daily feature date mismatch: "
            f"expected={expected_date.date().isoformat()} "
            f"actual={daily_feature_date.date().isoformat()}"
        )
    intraday_max_date = _latest_frame_date(
        intraday,
        ("datetime", "date", "trade_date"),
    )
    if intraday_max_date is None:
        raise RuntimeError("BYD intraday training history has no valid maximum date")
    if intraday_max_date > daily_feature_date:
        raise RuntimeError(
            "BYD intraday training history extends beyond the daily decision date: "
            f"daily={daily_feature_date.date().isoformat()} "
            f"intraday={intraday_max_date.date().isoformat()}"
        )
    staleness_days = int((daily_feature_date - intraday_max_date).days)
    if staleness_days > intraday_training_max_staleness_days:
        raise RuntimeError(
            "BYD intraday training history exceeds freshness SLA: "
            f"daily={daily_feature_date.date().isoformat()} "
            f"intraday={intraday_max_date.date().isoformat()} "
            f"staleness_days={staleness_days} "
            f"maximum={intraday_training_max_staleness_days}"
        )
    return {
        "daily_feature_date": daily_feature_date.date().isoformat(),
        "expected_daily_feature_date": expected_date.date().isoformat(),
        "daily_feature_current": True,
        "daily_feature_source": daily.attrs.get("daily_feature_source") or "provided_dependency",
        "intraday_training_max_date": intraday_max_date.date().isoformat(),
        "intraday_training_role": "historical_model_training_and_validation_samples",
        "intraday_is_current_feature": False,
        "intraday_training_staleness_days": staleness_days,
        "intraday_training_max_staleness_days": intraday_training_max_staleness_days,
        "intraday_training_within_sla": True,
        "staleness_unit": "calendar_days",
    }


@dataclass(frozen=True)
class BydWorkspaceDependencies:
    """External collaborators used by the BYD daily workspace."""

    read_snapshot: SnapshotReader
    write_snapshot: SnapshotWriter
    is_snapshot_current: SnapshotValidator = _accept_snapshot
    load_daily: FrameLoader = load_byd_daily_frame
    load_intraday: FrameLoader = load_byd_intraday_validation_frame
    build_minute_payload: PayloadBuilder = build_minute_payload
    build_daily_plan: PayloadBuilder = build_daily_t_plan


def build_byd_daily_strategy(
    shares: int = 10_000,
    cost: float = 110.6061,
    refresh: bool = False,
    *,
    expected_trade_date: str | pd.Timestamp | None = None,
    intraday_training_max_staleness_days: int | None = None,
    dependencies: BydWorkspaceDependencies,
) -> WorkspacePayload:
    """Build BYD's fixed pre-market daily T plan and persist its workspace snapshot."""

    training_sla_days = _intraday_training_sla_days(
        intraday_training_max_staleness_days
    )
    params = {
        "plan_version": 4,
        "shares": int(shares),
        "cost": round(float(cost), 6),
        "intraday_training_max_staleness_days": training_sla_days,
    }
    if not refresh:
        cached = dependencies.read_snapshot(
            "byd_daily_plan",
            params=params,
            allow_sql=False,
        )
        if cached is not None and dependencies.is_snapshot_current(cached):
            return cached

    daily = normalize_byd_daily_frame(dependencies.load_daily())
    intraday = dependencies.load_intraday()
    feature_freshness = _feature_freshness(
        daily,
        intraday,
        expected_trade_date=expected_trade_date,
        intraday_training_max_staleness_days=training_sla_days,
    )
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
        intraday=intraday,
        shares=holding.shares,
        full_shares=holding.full_shares,
    )
    plan_signal_date = _normalized_date(daily_plan.get("signal_date"))
    daily_feature_date = _normalized_date(feature_freshness["daily_feature_date"])
    if plan_signal_date != daily_feature_date:
        raise RuntimeError(
            "BYD daily plan signal date does not match its feature date: "
            f"signal={daily_plan.get('signal_date')} "
            f"feature={feature_freshness['daily_feature_date']}"
        )
    daily_plan["feature_freshness"] = feature_freshness
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
            "feature_freshness": feature_freshness,
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
