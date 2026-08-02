from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.data.atomic_io import atomic_write_json
from quant.data.tushare_fetcher import TushareDataFetcher
from quant.routine.paths import PROJECT_ROOT
from quant.strategies.convertible_bond.backtest import _active_basic, _active_call, _prepare_basic, _prepare_call
from quant.strategies.convertible_bond.grid import (
    HoldingGridConfig,
    HoldingGridStrategy,
    add_low_position_features,
    add_market_state_features,
)
from quant.strategies.convertible_bond.trend_enhanced import add_trend_enhanced_features


CB_DATA_DIR = PROJECT_ROOT / "data/convertible_bond/tushare"
CB_DAILY_PATH = CB_DATA_DIR / "cb_daily_20180101_20260616.parquet"
CB_BASIC_PATH = CB_DATA_DIR / "cb_basic_all.parquet"
CB_CALL_PATH = CB_DATA_DIR / "cb_call_20180101_20260616.parquet"
ITERATION_SUMMARY_PATH = PROJECT_ROOT / "reports/convertible_bond/ladder_grid_iteration/iteration_summary.csv"
GRID_TREND_OVERLAY_SUMMARY_PATH = PROJECT_ROOT / "reports/convertible_bond/grid_trend_overlay/iteration_summary.csv"
CB_PREMIUM_MIN_COVERAGE = 0.90


def default_convertible_bond_grid_config() -> HoldingGridConfig:
    """Guarded low-position grid used by the web operation desk."""
    return HoldingGridConfig(
        name="strict_dynamic_floor98_grid",
        top_n=None,
        max_holdings=22,
        max_total_weight=0.95,
        max_position_weight=0.10,
        max_entry_price=118.0,
        min_premium_rate=-3.0,
        max_premium_rate=28.0,
        max_double_low=144.0,
        max_price_position_252=0.40,
        min_drawdown_from_252_high=0.06,
        min_amount=1_500.0,
        min_remaining_size=0.8,
        min_momentum_20d=-0.12,
        min_credit_rating="A+",
        market_risk_mode="block_downtrend",
        exit_price=134.0,
        exit_premium_rate=48.0,
        exit_double_low=170.0,
        exit_price_position_252=0.84,
        exit_floor_price=98.0,
        initial_entry_fraction=0.40,
        add_on_drawdown_step=0.035,
        add_position_fraction=0.20,
        max_grid_position_fraction=1.00,
        dynamic_grid=True,
        low_risk_grid_step=0.030,
        medium_risk_grid_step=0.040,
        high_risk_grid_step=0.055,
        low_risk_position_scale=1.00,
        medium_risk_position_scale=0.80,
        high_risk_position_scale=0.55,
        take_profit_1=0.07,
        take_profit_1_keep_fraction=0.55,
        take_profit_2=0.18,
    )


def convertible_bond_grid_strategy_configs() -> list[tuple[HoldingGridConfig, dict[str, Any]]]:
    strict = default_convertible_bond_grid_config()
    stable_gate = HoldingGridConfig(
        name="core_market_scaled_market_gate",
        top_n=8,
        max_holdings=12,
        max_total_weight=0.95,
        max_position_weight=0.18,
        max_entry_price=116.0,
        min_premium_rate=0.0,
        max_premium_rate=24.0,
        max_double_low=138.0,
        max_price_position_252=0.35,
        min_drawdown_from_252_high=0.07,
        min_amount=3_000.0,
        min_momentum_20d=-0.08,
        market_risk_mode="scale_downtrend",
        market_entry_scale_weak=0.35,
        market_entry_scale_strong=1.0,
        exit_price=132.0,
        exit_premium_rate=45.0,
        exit_double_low=165.0,
        exit_price_position_252=0.82,
        max_entry_market_median_double_low=145.0,
        min_entry_market_trend_20d=-0.02,
        min_entry_market_trend_breadth=0.15,
    )
    rebound_overlay = HoldingGridConfig(
        name="return_core_trend_rebound",
        top_n=8,
        max_holdings=14,
        max_total_weight=1.00,
        max_position_weight=0.18,
        max_entry_price=118.0,
        min_premium_rate=-1.0,
        max_premium_rate=26.0,
        max_double_low=142.0,
        max_price_position_252=0.38,
        min_drawdown_from_252_high=0.06,
        min_amount=3_000.0,
        min_momentum_20d=-0.10,
        exit_price=140.0,
        exit_premium_rate=55.0,
        exit_double_low=178.0,
        exit_price_position_252=0.90,
        min_entry_trend_strength=50.0,
        min_entry_six_sword=3,
        min_entry_return_5d=-4.0,
        max_entry_return_5d=10.0,
        min_entry_return_1d=-3.0,
        max_entry_return_1d=5.0,
        max_entry_price_position_60d=0.82,
        max_entry_market_median_double_low=145.0,
        min_entry_market_trend_20d=-0.04,
        min_entry_market_trend_breadth=0.10,
    )
    return [
        (
            stable_gate,
            {
                "name": "双低网格·市场闸门",
                "description": "正式稳健版：保留双低低吸网格，只在全市场双低不过热、趋势未明显转弱、趋势广度足够时开新仓。",
                "style": "稳健主推",
                "overlay": "market_gate",
            },
        ),
        (
            rebound_overlay,
            {
                "name": "双低网格·反弹增强",
                "description": "正式进攻版：在双低低位基础上加入温和反弹过滤，保留低位弹性，同时避开仍在明显走弱的标的。",
                "style": "收益进攻",
                "overlay": "trend_rebound",
            },
        ),
        (
            strict,
            {
                "name": "严格动态网格",
                "description": "只展示真实满足低位条件的转债；弱市暂停新开仓；按风险使用 3% / 4% / 5.5% 动态网格。",
                "style": "保守默认",
            },
        ),
        (
            replace(strict, name="strict_dynamic_no_floor_grid", exit_floor_price=None),
            {
                "name": "债性修复网格",
                "description": "跌破 98 不机械清仓，把 98 视为风险复核线；更重视债性修复，但长周期尾部回撤更高。",
                "style": "修复持有",
            },
        ),
        (
            HoldingGridConfig(
                name="floor98_expanded_ladder_block",
                top_n=14,
                max_holdings=22,
                max_total_weight=0.95,
                max_position_weight=0.10,
                max_entry_price=118.0,
                min_premium_rate=-3.0,
                max_premium_rate=28.0,
                max_double_low=144.0,
                max_price_position_252=0.40,
                min_drawdown_from_252_high=0.06,
                min_amount=1_500.0,
                min_remaining_size=0.8,
                min_momentum_20d=-0.12,
                min_credit_rating="A+",
                market_risk_mode="block_downtrend",
                exit_price=136.0,
                exit_premium_rate=52.0,
                exit_double_low=176.0,
                exit_price_position_252=0.86,
                exit_floor_price=98.0,
                initial_entry_fraction=0.40,
                add_on_drawdown_step=0.035,
                add_position_fraction=0.20,
                max_grid_position_fraction=1.00,
                take_profit_1=0.07,
                take_profit_1_keep_fraction=0.55,
                take_profit_2=0.18,
            ),
            {
                "name": "扩池收益基准",
                "description": "固定 3.5% 网格，候选池更宽，收益弹性高于严格动态版，但回撤和单券风险更高。",
                "style": "收益基准",
            },
        ),
        (
            HoldingGridConfig(
                name="defensive_low_price_dynamic_grid",
                top_n=None,
                max_holdings=16,
                max_total_weight=0.80,
                max_position_weight=0.12,
                max_entry_price=112.0,
                min_premium_rate=-3.0,
                max_premium_rate=18.0,
                max_double_low=130.0,
                max_price_position_252=0.28,
                min_drawdown_from_252_high=0.08,
                min_amount=2_500.0,
                min_remaining_size=1.0,
                min_momentum_20d=-0.08,
                min_credit_rating="AA-",
                market_risk_mode="block_downtrend",
                exit_price=128.0,
                exit_premium_rate=38.0,
                exit_double_low=158.0,
                exit_price_position_252=0.76,
                exit_floor_price=100.0,
                initial_entry_fraction=0.45,
                add_on_drawdown_step=0.030,
                add_position_fraction=0.20,
                max_grid_position_fraction=1.00,
                dynamic_grid=True,
                low_risk_grid_step=0.025,
                medium_risk_grid_step=0.035,
                high_risk_grid_step=0.050,
                low_risk_position_scale=1.00,
                medium_risk_position_scale=0.75,
                high_risk_position_scale=0.45,
                take_profit_1=0.06,
                take_profit_1_keep_fraction=0.55,
                take_profit_2=0.14,
            ),
            {
                "name": "防守低价网格",
                "description": "更严格限制价格、溢价、双低和流动性，允许长期空仓；目标是极低回撤和高确定性。",
                "style": "防守观察",
            },
        ),
    ]


def _backtest_summary(config_name: str) -> dict[str, Any]:
    summary_path = (
        GRID_TREND_OVERLAY_SUMMARY_PATH
        if GRID_TREND_OVERLAY_SUMMARY_PATH.exists()
        else ITERATION_SUMMARY_PATH
    )
    if not summary_path.exists():
        return {}
    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        return {}
    rows = summary[summary["variant"].astype(str).eq(config_name)].copy()
    if rows.empty and ITERATION_SUMMARY_PATH.exists() and summary_path != ITERATION_SUMMARY_PATH:
        try:
            legacy_summary = pd.read_csv(ITERATION_SUMMARY_PATH)
            rows = legacy_summary[legacy_summary["variant"].astype(str).eq(config_name)].copy()
        except Exception:
            rows = pd.DataFrame()
    result: dict[str, Any] = {}
    for label, start_prefix in [("from_2020", "2020"), ("from_2024", "2024")]:
        match = rows[rows["config"].astype(str).str.contains(f"'start_date': '{start_prefix}0101'", regex=False)]
        if match.empty:
            match = rows[rows["start_date"].astype(str).str.startswith(start_prefix)]
        if match.empty:
            continue
        if "rebalance" in match.columns:
            weekly = match[match["rebalance"].astype(str).eq("weekly")]
            if not weekly.empty:
                match = weekly
        row = match.iloc[-1]
        result[label] = {
            "total_return": _json_value(row.get("total_return")),
            "annual_return": _json_value(row.get("annual_return")),
            "max_drawdown": _json_value(row.get("max_drawdown")),
            "sharpe": _json_value(row.get("sharpe")),
            "position_win_rate": _json_value(row.get("position_win_rate")),
            "position_profit_factor": _json_value(row.get("position_profit_factor")),
            "average_exposure": _json_value(row.get("average_exposure")),
            "trade_count": _json_value(row.get("trade_count")),
            "rebalance": _json_value(row.get("rebalance")),
        }
    return result


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _requested_trade_date(daily: pd.DataFrame, trade_date: str | None) -> str:
    dates = daily["trade_date"].dropna().astype(str).sort_values().unique().tolist()
    if not dates:
        raise ValueError("No convertible-bond daily data available")
    if not trade_date:
        return dates[-1]
    normalized = str(trade_date).replace("-", "")
    eligible = [item for item in dates if item <= normalized]
    if not eligible:
        raise ValueError(f"No convertible-bond data before {trade_date}")
    return eligible[-1]


def _premium_coverage_for_date(daily: pd.DataFrame, trade_date: str) -> float:
    premium_column = "premium_rate" if "premium_rate" in daily.columns else "bond_over_rate"
    if premium_column not in daily.columns:
        return 0.0
    day = daily[daily["trade_date"].astype(str).eq(str(trade_date))]
    close = pd.to_numeric(day.get("close"), errors="coerce")
    premium = pd.to_numeric(day[premium_column], errors="coerce")
    eligible = close.notna()
    total = int(eligible.sum())
    return 0.0 if total == 0 else float((premium.notna() & eligible).sum() / total)


def _resolve_trade_date(daily: pd.DataFrame, trade_date: str | None) -> str:
    requested = _requested_trade_date(daily, trade_date)
    dates = daily["trade_date"].dropna().astype(str).sort_values().unique().tolist()
    usable = [
        item
        for item in dates
        if item <= requested and _premium_coverage_for_date(daily, item) >= CB_PREMIUM_MIN_COVERAGE
    ]
    if not usable:
        raise ValueError(
            f"No convertible-bond data with at least {CB_PREMIUM_MIN_COVERAGE:.0%} premium coverage before {requested}"
        )
    return usable[-1]


def _normalize_trade_date(value: str | int | pd.Timestamp) -> str:
    parsed = pd.to_datetime(str(value).replace("-", ""), format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid trade date: {value}")
    return parsed.strftime("%Y%m%d")


def _merge_parquet(path: Path, frame: pd.DataFrame, keys: list[str]) -> int:
    if frame is None or frame.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        merged = pd.concat([existing, frame], ignore_index=True, sort=False)
    else:
        merged = frame.copy()
    for key in keys:
        if key in merged.columns:
            merged[key] = merged[key].astype(str)
    available_keys = [key for key in keys if key in merged.columns]
    if available_keys:
        merged = merged.drop_duplicates(available_keys, keep="last")
    sort_keys = [key for key in keys if key in merged.columns]
    if sort_keys:
        merged = merged.sort_values(sort_keys)
    merged.to_parquet(path, index=False)
    return int(len(frame))


def refresh_convertible_bond_daily(
    trade_date: str,
    retries: int = 5,
    sleep_seconds: float = 8.0,
    fetcher: TushareDataFetcher | None = None,
) -> dict[str, Any]:
    """Refresh one convertible-bond daily snapshot with empty-result retries."""

    normalized = _normalize_trade_date(trade_date)
    fetcher = fetcher or TushareDataFetcher(cache_dir=CB_DATA_DIR / "tushare_cache")
    attempts: list[dict[str, Any]] = []
    daily = pd.DataFrame()
    last_error = ""
    fields = (
        "ts_code,trade_date,pre_close,open,high,low,close,change,pct_chg,"
        "vol,amount,bond_value,bond_over_rate,stock_price,stock_over_rate,"
        "turnover_rate"
    )
    for attempt in range(1, max(1, retries) + 1):
        try:
            frame = fetcher.pro.cb_daily(trade_date=normalized, fields=fields)
            daily = frame if frame is not None else pd.DataFrame()
            rows = int(len(daily))
            attempts.append({"attempt": attempt, "rows": rows, "error": None})
            if rows > 0:
                break
            last_error = "Tushare cb_daily returned empty"
        except Exception as exc:
            last_error = str(exc)
            attempts.append({"attempt": attempt, "rows": 0, "error": last_error})
        if attempt < max(1, retries) and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    basic_rows = 0
    call_rows = 0
    try:
        basic = fetcher.pro.cb_basic(fields=(
            "ts_code,bond_short_name,stk_code,stk_short_name,list_date,delist_date,"
            "remain_size,conv_start_date,conv_end_date,issue_rating,newest_rating"
        ))
        if basic is not None and not basic.empty:
            basic.to_parquet(CB_BASIC_PATH, index=False)
            basic_rows = int(len(basic))
    except Exception as exc:
        attempts.append({"attempt": "cb_basic", "rows": 0, "error": str(exc)})

    try:
        call = fetcher.pro.cb_call(ann_date=normalized)
        if call is not None and not call.empty:
            call_rows = _merge_parquet(
                CB_CALL_PATH,
                call,
                ["ts_code", "ann_date", "call_type", "call_price", "is_call"],
            )
    except Exception as exc:
        attempts.append({"attempt": "cb_call", "rows": 0, "error": str(exc)})

    if daily.empty:
        return {
            "status": "no_data",
            "trade_date": normalized,
            "daily_rows": 0,
            "basic_rows": basic_rows,
            "call_rows": call_rows,
            "attempts": attempts,
            "error": last_error,
        }

    daily["trade_date"] = daily["trade_date"].astype(str)
    written_rows = _merge_parquet(CB_DAILY_PATH, daily, ["trade_date", "ts_code"])
    return {
        "status": "success",
        "trade_date": normalized,
        "daily_rows": written_rows,
        "basic_rows": basic_rows,
        "call_rows": call_rows,
        "attempts": attempts,
        "path": str(CB_DAILY_PATH),
    }


def _parts(weight: float) -> int:
    return max(int(round(float(weight) * 100)), 1)


def _price(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 3)


def _build_grid_plan(row: dict[str, Any], config: HoldingGridConfig) -> dict[str, Any]:
    close = float(row["close"])
    target_weight = float(row["target_weight"])
    max_parts = _parts(target_weight)
    first_parts = max(1, int(round(max_parts * config.initial_entry_fraction)))
    add_parts = max(1, int(round(max_parts * config.add_position_fraction)))
    keep_parts = max(1, int(round(max_parts * config.take_profit_1_keep_fraction)))
    grid_step = float(row.get("grid_step_pct") or config.add_on_drawdown_step or 0.035)
    stop_loss = float(config.stop_loss_from_entry or 0.08)
    buy_levels = []
    fraction = float(config.initial_entry_fraction)
    level_index = 1
    while fraction < float(config.max_grid_position_fraction) - 1e-9:
        fraction = min(float(config.max_grid_position_fraction), fraction + float(config.add_position_fraction))
        level_price = close * (1.0 - grid_step * level_index)
        buy_levels.append(
            {
                "level": level_index,
                "trigger_price": _price(level_price),
                "trigger_pct": -round(grid_step * level_index * 100, 2),
                "buy_parts": add_parts,
                "target_total_parts": max(1, int(round(max_parts * fraction))),
                "condition": _entry_condition_text(config),
            }
        )
        level_index += 1
        if level_index > 6:
            break

    take_profit_1 = close * (1.0 + float(config.take_profit_1 or 0.07))
    take_profit_2 = close * (1.0 + float(config.take_profit_2 or 0.18))
    stop_price = min(close * (1.0 - stop_loss), 98.0) if close > 100 else close * (1.0 - stop_loss)
    return {
        "mode": "低位分批网格",
        "unit_definition": "1份 = 账户权益 1%",
        "risk_level": row.get("risk_level") or "standard",
        "max_parts": max_parts,
        "first_buy_parts": first_parts,
        "grid_type": "percentage",
        "grid_step_pct": round(grid_step * 100, 2),
        "grid_step_price": _price(close * grid_step),
        "buy_levels": buy_levels,
        "sell_levels": [
            {
                "level": 1,
                "trigger_price": _price(take_profit_1),
                "trigger_pct": round(float(config.take_profit_1 or 0.07) * 100, 2),
                "sell_parts": max(max_parts - keep_parts, 1),
                "target_remaining_parts": keep_parts,
                "condition": "第一档止盈，先锁定利润，保留尾仓观察",
            },
            {
                "level": 2,
                "trigger_price": _price(take_profit_2),
                "trigger_pct": round(float(config.take_profit_2 or 0.18) * 100, 2),
                "sell_parts": keep_parts,
                "target_remaining_parts": 0,
                "condition": "第二档止盈，清仓或仅留观察仓",
            },
        ],
        "risk_controls": [
            {
                "name": "硬风控线",
                "trigger_price": _price(stop_price),
                "action": "停止加仓，至少减半；若正股继续走弱则清仓",
            },
            {
                "name": "强赎/到期风险",
                "trigger_price": None,
                "action": "出现强赎风险或不可转股，直接退出候选池",
            },
        ],
    }


def _entry_condition_text(config: HoldingGridConfig) -> str:
    conditions = ["仍满足低价、低溢价、低双低"]
    if config.max_entry_market_median_double_low is not None:
        conditions.append(f"市场双低中位 <= {config.max_entry_market_median_double_low:g}")
    if config.min_entry_market_trend_20d is not None:
        conditions.append(f"市场20日趋势 >= {config.min_entry_market_trend_20d:.1%}")
    if config.min_entry_market_trend_breadth is not None:
        conditions.append(f"趋势广度 >= {config.min_entry_market_trend_breadth:.0%}")
    if config.min_entry_six_sword is not None:
        conditions.append(f"个券六脉 >= {config.min_entry_six_sword}")
    return "、".join(conditions) + "时执行"


def _action_from_plan(row: dict[str, Any], plan: dict[str, Any]) -> str:
    close = float(row["close"])
    if close <= float(plan["risk_controls"][0]["trigger_price"] or 0):
        return "触发风控，不新买"
    if close <= float(row.get("grid_full_price", 106.0) or 106.0):
        return "深低位，可按首批+一档网格执行"
    return "首批观察买入，等待低位网格加仓"


def _risk_label(value: Any) -> str:
    return {
        "low": "低风险",
        "medium": "中风险",
        "high": "高风险",
        "standard": "标准",
    }.get(str(value or "standard"), "标准")


def _overlay_entry_permission(market: dict[str, Any], config: HoldingGridConfig) -> tuple[bool, str]:
    reasons: list[str] = []
    median_double_low = _safe_numeric(market.get("market_median_double_low"))
    trend_20d = _safe_numeric(market.get("market_trend_20d"))
    trend_breadth = _safe_numeric(market.get("market_trend_breadth"))
    if (
        config.max_entry_market_median_double_low is not None
        and median_double_low is not None
        and median_double_low > config.max_entry_market_median_double_low
    ):
        reasons.append(f"市场双低中位 {median_double_low:.1f} > {config.max_entry_market_median_double_low:g}")
    if (
        config.min_entry_market_trend_20d is not None
        and trend_20d is not None
        and trend_20d < config.min_entry_market_trend_20d
    ):
        reasons.append(f"市场20日趋势 {trend_20d:.2%} < {config.min_entry_market_trend_20d:.2%}")
    if (
        config.min_entry_market_trend_breadth is not None
        and trend_breadth is not None
        and trend_breadth < config.min_entry_market_trend_breadth
    ):
        reasons.append(f"趋势广度 {trend_breadth:.1%} < {config.min_entry_market_trend_breadth:.0%}")
    if reasons:
        return False, "趋势闸门暂停新建仓：" + "；".join(reasons)
    return True, "允许新建仓"


def _safe_numeric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _build_single_strategy_plan(
    *,
    config: HoldingGridConfig,
    meta: dict[str, Any],
    day: pd.DataFrame,
    active_basic: pd.DataFrame,
    active_call: pd.DataFrame,
    resolved_trade_date: str,
    limit: int,
) -> dict[str, Any]:
    strategy = HoldingGridStrategy(config)
    universe = strategy.selector.build_universe(daily=day, basic=active_basic, call=active_call)
    if not universe.empty:
        universe = strategy._attach_grid_columns(universe)
    entry_scale = strategy._market_entry_scale(universe) if not universe.empty else 0.0
    target = strategy.target_portfolio(daily=day, basic=active_basic, call=active_call)
    if not target.empty and config.top_n is not None:
        target = target.head(limit).copy()

    candidates: list[dict[str, Any]] = []
    market = day.iloc[-1].to_dict() if not day.empty else {}
    overlay_entry_allowed, overlay_entry_permission = _overlay_entry_permission(market, config)
    for row in _records(target):
        grid_plan = _build_grid_plan(row, config)
        close = float(row["close"])
        execution_enabled = entry_scale > 0 and overlay_entry_allowed
        if not execution_enabled:
            grid_plan["first_buy_parts"] = 0
            for level in grid_plan["buy_levels"]:
                level["buy_parts"] = 0
                level["condition"] = f"{overlay_entry_permission}；已有实际持仓才继续按账户计划和单券风控执行"
        row.update(
            {
                "bond_name": row.get("bond_short_name") or row.get("bond_full_name") or row.get("ts_code"),
                "stock_code": row.get("stk_code"),
                "stock_name": row.get("stk_short_name"),
                "action": _action_from_plan(row, grid_plan) if execution_enabled else "弱市暂停新建仓；未成交网格暂停",
                "strategy_mode": grid_plan["mode"],
                "risk_level": row.get("risk_level") or "standard",
                "risk_label": _risk_label(row.get("risk_level")),
                "target_weight": _json_value(row.get("target_weight")),
                "first_buy_weight": (
                    round(float(row.get("target_weight") or 0) * config.initial_entry_fraction, 4)
                    if execution_enabled
                    else 0.0
                ),
                "execution_enabled": execution_enabled,
                "reference_price": _price(close),
                "operation_plan": grid_plan,
            }
        )
        candidates.append(row)

    entry_permission = (
        overlay_entry_permission
        if entry_scale > 0
        else "弱市暂停新建仓"
    )
    return {
        "trade_date": resolved_trade_date,
        "strategy": {
            "key": config.name,
            "name": meta["name"],
            "description": meta["description"],
            "style": meta.get("style"),
            "overlay": meta.get("overlay") or "base_grid",
            "config": asdict(config),
            "backtest": _backtest_summary(config.name),
        },
        "market_state": {
            "median_double_low": _json_value(market.get("market_median_double_low")),
            "price_position_252": _json_value(market.get("market_price_position_252")),
            "trend_20d": _json_value(market.get("market_trend_20d")),
            "trend_breadth": _json_value(market.get("market_trend_breadth")),
            "entry_scale": entry_scale,
            "entry_permission": entry_permission,
            "existing_grid_permission": "未成交网格暂停；已有实际持仓的网格继续按单券风控管理",
            "weak_market_basis": "全市场可转债双低中位数低于20日均线" if entry_scale <= 0 else "全市场可转债双低中位数不低于20日均线",
        },
        "unit_definition": "1份 = 账户权益 1%；示例：100万账户，1份约1万元。",
        "candidates": candidates,
        "notes": [
            "候选基于本地 Tushare 可转债缓存，不联网刷新。",
            "所有网格为收盘价参考计划，盘中执行需结合流动性和正股走势。",
            "扩池策略收益弹性更高，但必须严格执行硬风控线，避免深跌债拖累。",
        ],
    }


def build_convertible_bond_grid_plan(
    trade_date: str | None = None,
    limit: int = 18,
    config: HoldingGridConfig | None = None,
) -> dict[str, Any]:
    daily = pd.read_parquet(CB_DAILY_PATH)
    basic = pd.read_parquet(CB_BASIC_PATH)
    call = pd.read_parquet(CB_CALL_PATH) if CB_CALL_PATH.exists() else pd.DataFrame()
    requested_trade_date = _requested_trade_date(daily, trade_date)
    resolved_trade_date = _resolve_trade_date(daily, trade_date)
    premium_coverage = _premium_coverage_for_date(daily, resolved_trade_date)

    daily = daily[daily["trade_date"].astype(str) <= resolved_trade_date].copy()
    featured = add_trend_enhanced_features(add_market_state_features(add_low_position_features(daily)))
    day = featured[featured["trade_date"].astype(str) == resolved_trade_date].copy()
    prepared_basic = _prepare_basic(basic)
    prepared_call = _prepare_call(call)
    active_basic = _active_basic(prepared_basic, resolved_trade_date)
    active_call = _active_call(prepared_call, resolved_trade_date)

    if config is not None:
        configs = [(config, {"name": "可转债自定义网格", "description": "测试或脚本传入的自定义可转债网格策略。", "style": "自定义"})]
    else:
        configs = convertible_bond_grid_strategy_configs()

    plans = [
        _build_single_strategy_plan(
            config=item_config,
            meta=meta,
            day=day,
            active_basic=active_basic,
            active_call=active_call,
            resolved_trade_date=resolved_trade_date,
            limit=limit,
        )
        for item_config, meta in configs
    ]
    primary = plans[0] if plans else {}
    payload = dict(primary)
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["trade_date"] = resolved_trade_date
    payload["data_quality"] = {
        "status": "success",
        "requested_trade_date": requested_trade_date,
        "resolved_trade_date": resolved_trade_date,
        "premium_coverage": round(premium_coverage, 6),
        "minimum_premium_coverage": CB_PREMIUM_MIN_COVERAGE,
        "stale": requested_trade_date != resolved_trade_date,
        "reason": (
            "latest_trade_date_missing_premium_data"
            if requested_trade_date != resolved_trade_date
            else "complete"
        ),
    }
    payload["strategy_plans"] = plans
    payload["strategy_pool"] = [
        {
            "key": plan["strategy"]["key"],
            "name": plan["strategy"]["name"],
            "description": plan["strategy"]["description"],
            "style": plan["strategy"].get("style"),
            "overlay": plan["strategy"].get("overlay"),
            "candidate_count": len(plan.get("candidates", [])),
            "entry_permission": plan.get("market_state", {}).get("entry_permission"),
            "backtest": plan["strategy"].get("backtest", {}),
        }
        for plan in plans
    ]
    return payload


def write_convertible_bond_grid_plan(output_path: Path | None = None, trade_date: str | None = None) -> Path:
    payload = build_convertible_bond_grid_plan(trade_date=trade_date)
    output_path = output_path or PROJECT_ROOT / "data/web/convertible_bond_grid_plan.json"
    return atomic_write_json(payload, output_path)
