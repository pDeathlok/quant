"""Build the daily medium/long-horizon workspace for blood-chip repair events."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quant.research.blood_chip import (
    BloodChipBacktestConfig,
    BloodChipBacktestResult,
    BloodChipSignalConfig,
    add_blood_chip_path_features,
    build_blood_chip_features,
    generate_blood_chip_signals,
)
from quant.research.blood_chip_scale_in import (
    DEFAULT_SCALE_IN_POLICIES,
    run_blood_chip_scale_in_backtest,
)
from quant.research.blood_chip_kdj import attach_blood_chip_kdj_path


BLOOD_CHIP_SIGNAL_CONFIG = BloodChipSignalConfig(
    maximum_return_120d=0.50,
    minimum_market_return_60d=-0.15,
)
BLOOD_CHIP_BACKTEST_CONFIG = BloodChipBacktestConfig(
    maximum_positions=10,
    stop_loss=0.10,
    maximum_holding_days=120,
    allow_reentry_after_stop=True,
    require_new_event_for_reentry=True,
)
MAXIMUM_REBOUND_FROM_EVENT_LOW = 0.15
BLOOD_CHIP_SCALE_IN_POLICY = DEFAULT_SCALE_IN_POLICIES["increasing_survival"]
BLOOD_CHIP_LONG_SCHEMA_VERSION = "blood_chip_long_v3_kdj_path_annotation"


def _number(value: Any, digits: int = 6) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or not np.isfinite(float(number)):
        return None
    return round(float(number), digits)


def _date(value: Any) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).date().isoformat()


def _profiles(stock_basic: pd.DataFrame | None) -> dict[str, dict[str, str]]:
    if stock_basic is None or stock_basic.empty or "ts_code" not in stock_basic:
        return {}
    frame = stock_basic.copy()
    for column in ("name", "industry"):
        if column not in frame:
            frame[column] = ""
    def text(value: Any) -> str:
        return "" if pd.isna(value) else str(value)

    return {
        str(row.ts_code): {
            "name": text(row.name),
            "industry": text(row.industry),
        }
        for row in frame[["ts_code", "name", "industry"]].itertuples(index=False)
    }


def select_blood_chip_long_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen path filter and the same-day low-volatility ranking."""

    if signals.empty:
        out = signals.copy()
        out["selection_rank"] = pd.Series(dtype="Int64")
        return out
    out = signals.copy()
    rebound = pd.to_numeric(out["rebound_from_event_low"], errors="coerce")
    out = out.loc[rebound.le(MAXIMUM_REBOUND_FROM_EVENT_LOW)].copy()
    out["volatility_60d"] = pd.to_numeric(out["volatility_60d"], errors="coerce")
    out = out.sort_values(
        ["signal_date", "volatility_60d", "signal_score", "ts_code"],
        ascending=[True, True, False, True],
        na_position="last",
    )
    out["selection_rank"] = (
        out.groupby("signal_date", observed=True).cumcount() + 1
    )
    # The execution engine sorts descending; negative volatility preserves the
    # empirically stronger preference for the calmer confirmation path.
    out["signal_score"] = -out["volatility_60d"].fillna(999.0)
    return out.reset_index(drop=True)


def _candidate_row(
    row: pd.Series,
    profile: dict[str, str],
    *,
    rank: int,
    is_reentry: bool,
) -> dict[str, Any]:
    raw_kdj_state = row.get("shock_kdj_state")
    kdj_state = str(raw_kdj_state) if pd.notna(raw_kdj_state) else "unavailable"
    negative_count = _number(row.get("shock_kdj_negative_count"), 0)
    kdj_labels = {
        "triple_oversold": "深度带血筹（三周期超跌）",
        "daily_weekly_oversold": "深度带血筹（日周超跌）",
        "partial_oversold": "带血筹（局部周期超跌）",
        "not_oversold": "带血筹（非KDJ超跌）",
        "unavailable": "带血筹（KDJ数据不足）",
    }
    return {
        "rank": rank,
        "ts_code": str(row["ts_code"]),
        "name": profile.get("name", ""),
        "industry": profile.get("industry", ""),
        "signal_date": _date(row.get("signal_date")),
        "shock_date": _date(row.get("shock_date")),
        "shock_event_id": int(row.get("shock_event_id") or 0),
        "shock_score": _number(row.get("shock_score")),
        "absorption_score": _number(row.get("absorption_score")),
        "volatility_60d": _number(row.get("volatility_60d")),
        "return_120d": _number(row.get("return_120d")),
        "market_return_60d": _number(row.get("market_return_60d")),
        "rebound_from_event_low": _number(row.get("rebound_from_event_low")),
        "shock_volatility_expansion_ratio": _number(
            row.get("shock_volatility_expansion_ratio")
        ),
        "confirmation_amount_vs_prior_ratio": _number(
            row.get("confirmation_amount_vs_prior_ratio")
        ),
        "blood_chip_subtype": kdj_labels.get(kdj_state, "带血筹（KDJ待核对）"),
        "shock_kdj_state": kdj_state,
        "shock_kdj_daily_j": _number(row.get("shock_kdj_daily_j"), 2),
        "shock_kdj_weekly_j": _number(row.get("shock_kdj_weekly_j"), 2),
        "shock_kdj_monthly_j": _number(row.get("shock_kdj_monthly_j"), 2),
        "shock_kdj_negative_count": int(negative_count or 0),
        "confirmation_kdj_daily_j": _number(row.get("confirmation_kdj_daily_j"), 2),
        "confirmation_kdj_weekly_j": _number(row.get("confirmation_kdj_weekly_j"), 2),
        "confirmation_kdj_monthly_j": _number(row.get("confirmation_kdj_monthly_j"), 2),
        "kdj_policy": "冲击日路径标签；不作为当前硬门槛或排序权重",
        "action": "NEW_EVENT_REENTRY_WATCH" if is_reentry else "NEXT_OPEN_WATCH",
        "initial_tranche_fraction": 0.20,
        "target_position_fraction": 0.10,
        "execution_rule": "首仓 20%：下一交易日开盘缺口在 -7% 至 +7%，且未一字涨停时买入",
        "risk_rule": "实际成交价下方 10% 止损；最多持有 120 个交易日",
    }


def _position_row(row: pd.Series, profile: dict[str, str]) -> dict[str, Any]:
    entry_fill = _number(row.get("entry_fill"))
    tranches_filled = int(row.get("tranches_filled") or 1)
    stage_labels = {
        1: "首仓已完成（累计 20%）",
        2: "第二段已完成（累计 50%）",
        3: "第三段已完成（累计 100%）",
    }
    if tranches_filled <= 1:
        next_addition_fraction = 0.30
        next_trigger = "至少持有 5 日；收盘不低于信号价 95%；三日残差收益不低于 0"
    elif tranches_filled == 2:
        next_addition_fraction = 0.50
        next_trigger = "至少持有 10 日；收盘不低于信号价；三日残差收益大于 0"
    else:
        next_addition_fraction = None
        next_trigger = "三段建仓已经完成；继续执行首仓止损与 120 日到期规则"
    stop_price = _number(row.get("stop_price"))
    if stop_price is None and entry_fill is not None:
        stop_price = round(entry_fill * 0.90, 6)
    return {
        "ts_code": str(row["ts_code"]),
        "name": profile.get("name", ""),
        "industry": profile.get("industry", ""),
        "signal_date": _date(row.get("signal_date")),
        "entry_date": _date(row.get("entry_date")),
        "entry_fill": entry_fill,
        "current_estimated_price": _number(row.get("exit_fill")),
        "estimated_net_return": _number(row.get("net_return")),
        "holding_sessions": int(row.get("holding_sessions") or 0),
        "stop_price": stop_price,
        "signal_close": _number(row.get("signal_close")),
        "current_residual_return_3d": _number(
            row.get("current_residual_return_3d")
        ),
        "tranches_filled": tranches_filled,
        "stage_label": stage_labels.get(tranches_filled, "阶段状态待核对"),
        "tranche_dates": str(row.get("tranche_dates") or ""),
        "deployed_fraction": _number(row.get("deployed_fraction")),
        "next_addition_fraction": next_addition_fraction,
        "next_trigger": next_trigger,
        "next_stage_ready": bool(row.get("next_stage_ready") or False),
        "shock_event_id": int(row.get("shock_event_id") or 0),
        "reentry_number": int(row.get("reentry_number") or 0),
        "status": "SIMULATED_HOLDING",
    }


def _exit_row(row: pd.Series, profile: dict[str, str]) -> dict[str, Any]:
    return {
        "ts_code": str(row["ts_code"]),
        "name": profile.get("name", ""),
        "exit_date": _date(row.get("exit_date")),
        "exit_reason": str(row.get("exit_reason") or ""),
        "net_return": _number(row.get("net_return")),
        "holding_sessions": int(row.get("holding_sessions") or 0),
        "shock_event_id": int(row.get("shock_event_id") or 0),
        "reentry_number": int(row.get("reentry_number") or 0),
    }


def compose_blood_chip_long_plan(
    *,
    signal_date: str,
    signals: pd.DataFrame,
    result: BloodChipBacktestResult,
    stock_basic: pd.DataFrame | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Compose a JSON-ready daily workspace from point-in-time research outputs."""

    asof = pd.Timestamp(signal_date).normalize()
    profiles = _profiles(stock_basic)
    trades = result.trades.copy()
    if not trades.empty:
        trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    stopped_events: dict[str, int] = {}
    if not trades.empty:
        stopped = trades.loc[trades["exit_reason"].eq("stop_loss")]
        stopped_events = (
            stopped.groupby("ts_code")["shock_event_id"].max().astype(int).to_dict()
            if not stopped.empty
            else {}
        )

    pending = signals.copy()
    if not pending.empty:
        if "selection_rank" not in pending:
            pending = select_blood_chip_long_signals(pending)
        pending["signal_date"] = pd.to_datetime(pending["signal_date"], errors="coerce")
        pending = pending.loc[
            pending["signal_date"].eq(asof) & pending["entry_date"].isna()
        ].sort_values(["selection_rank", "ts_code"])
    candidates: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(pending.iterrows(), start=1):
        symbol = str(row["ts_code"])
        event_id = int(row.get("shock_event_id") or 0)
        stopped_event = stopped_events.get(symbol)
        is_reentry = stopped_event is not None and event_id > stopped_event
        candidates.append(
            _candidate_row(
                row,
                profiles.get(symbol, {}),
                rank=rank,
                is_reentry=is_reentry,
            )
        )

    active = (
        trades.loc[trades["exit_reason"].eq("end_of_data")].copy()
        if not trades.empty
        else pd.DataFrame()
    )
    simulated_positions = [
        _position_row(row, profiles.get(str(row["ts_code"]), {}))
        for _, row in active.sort_values(["entry_date", "ts_code"]).iterrows()
    ]
    exits = (
        trades.loc[
            trades["exit_date"].eq(asof)
            & ~trades["exit_reason"].eq("end_of_data")
        ].copy()
        if not trades.empty
        else pd.DataFrame()
    )
    recent_exits = [
        _exit_row(row, profiles.get(str(row["ts_code"]), {}))
        for _, row in exits.sort_values(["exit_reason", "ts_code"]).iterrows()
    ]
    stopped_today = sum(row["exit_reason"] == "stop_loss" for row in recent_exits)
    reentry_candidates = sum(
        row["action"] == "NEW_EVENT_REENTRY_WATCH" for row in candidates
    )
    deep_kdj_candidates = sum(
        row.get("shock_kdj_state") in {"triple_oversold", "daily_weekly_oversold"}
        for row in candidates
    )

    return {
        "schema_version": BLOOD_CHIP_LONG_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "signal_date": asof.date().isoformat(),
        "variant": "blood_chip",
        "strategy": {
            "name": "带血筹修复",
            "horizon": "1–6 个月",
            "status": "当前执行",
            "maximum_positions": BLOOD_CHIP_BACKTEST_CONFIG.maximum_positions,
            "stop_loss": BLOOD_CHIP_BACKTEST_CONFIG.stop_loss,
            "maximum_holding_sessions": BLOOD_CHIP_BACKTEST_CONFIG.maximum_holding_days,
            "scale_in_policy": BLOOD_CHIP_SCALE_IN_POLICY.key,
            "tranche_fractions": list(BLOOD_CHIP_SCALE_IN_POLICY.fractions),
            "first_tranche_rule": "信号次日开盘执行目标仓位的 20%",
            "second_tranche_rule": "至少持有 5 日、守住信号价 95% 且三日残差收益转正后，次日开盘加 30%",
            "third_tranche_rule": "至少持有 10 日、站回信号价且三日残差收益继续为正后，次日开盘加 50%",
            "reentry_rule": "止损后允许再入，但必须出现新的冲击事件",
            "ranking_rule": "同日候选按 60 日残差波动率由低到高",
            "kdj_overlay": "冲击日月/周/日J作为路径标签展示，不改变当前排序或仓位",
        },
        "summary": {
            "new_candidates": len(candidates),
            "simulated_active_positions": len(simulated_positions),
            "stopped_today": stopped_today,
            "reentry_candidates": reentry_candidates,
            "deep_kdj_candidates": deep_kdj_candidates,
        },
        "candidates": candidates,
        "simulated_positions": simulated_positions,
        "recent_exits": recent_exits,
        "research_evidence": {
            "development_2014_2019": {
                "trades": 243,
                "win_rate": 0.3004,
                "capital_weighted_trade_return": 0.0268,
                "capital_profit_factor": 1.35,
                "total_return": 0.6466,
                "maximum_drawdown": -0.6145,
            },
            "iteration_2020_2022": {
                "trades": 117,
                "win_rate": 0.3077,
                "capital_weighted_trade_return": 0.0471,
                "capital_profit_factor": 1.68,
                "total_return": 0.5140,
                "maximum_drawdown": -0.1509,
            },
            "seen_diagnostic_2023_2026": {
                "trades": 117,
                "win_rate": 0.3077,
                "capital_weighted_trade_return": 0.0239,
                "capital_profit_factor": 1.33,
                "total_return": 0.1873,
                "maximum_drawdown": -0.3305,
                "benchmark_total_return": 0.2125,
            },
            "warning": "当前已直接采用生存确认 20%/30%/50% 执行；2023–2026 已参与案例诊断，不是未见样本。",
            "kdj_overlay": {
                "deployment": "annotation_only",
                "development_baseline_capital_pf": 1.3542,
                "development_soft_overlay_capital_pf": 1.3071,
                "validation_baseline_total_return": 0.5140,
                "validation_soft_overlay_total_return": 0.5057,
                "reason": "冲击日KDJ软排序未同时改善开发期与验证期；硬过滤样本过少且研发期为负",
            },
        },
        "notes": [
            "量价只能识别疑似受迫卖出与卖盘衰竭，不能证明卖方身份。",
            "前期允许高波动；确认阶段不设硬波动上限，改为同日低波动优先。",
            "失败信号仅承担 20% 首仓；只有经过 5 日与 10 日生存确认才逐步提高仓位。",
            "月/周/日J用于描述冲击发生时的超跌深度；现阶段不改变低波动排序、首仓或加仓规则。",
            "所有持仓均为统一资金与交易成本假设下的策略模拟，不代表真实账户。",
        ],
    }


def build_blood_chip_daily_iteration(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe the candidate-set delta against the preceding daily snapshot."""

    current_symbols = {str(row.get("ts_code")) for row in current.get("candidates") or []}
    previous_symbols = {
        str(row.get("ts_code")) for row in (previous or {}).get("candidates") or []
    }
    current_positions = {
        str(row.get("ts_code")): row
        for row in current.get("simulated_positions") or []
    }
    previous_positions = {
        str(row.get("ts_code")): row
        for row in (previous or {}).get("simulated_positions") or []
    }
    advanced_positions = []
    for symbol in sorted(current_positions.keys() & previous_positions.keys()):
        current_stage = int(current_positions[symbol].get("tranches_filled") or 0)
        previous_stage = int(previous_positions[symbol].get("tranches_filled") or 0)
        if current_stage > previous_stage:
            advanced_positions.append(
                {
                    "ts_code": symbol,
                    "from_stage": previous_stage,
                    "to_stage": current_stage,
                }
            )
    return {
        "previous_signal_date": (previous or {}).get("signal_date"),
        "added_candidates": sorted(current_symbols - previous_symbols),
        "removed_candidates": sorted(previous_symbols - current_symbols),
        "continued_candidates": sorted(current_symbols & previous_symbols),
        "new_positions": sorted(current_positions.keys() - previous_positions.keys()),
        "closed_positions": sorted(previous_positions.keys() - current_positions.keys()),
        "advanced_positions": advanced_positions,
        "ready_additions": sorted(
            symbol
            for symbol, row in current_positions.items()
            if bool(row.get("next_stage_ready"))
        ),
        "policy_version": BLOOD_CHIP_LONG_SCHEMA_VERSION,
        "policy_changed": bool(
            previous
            and previous.get("schema_version") != current.get("schema_version")
        ),
    }


def build_blood_chip_long_plan(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    signal_date: str,
    stock_basic: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build features, signals and the recent strategy-state reconstruction."""

    features = build_blood_chip_features(daily, benchmark)
    signals = generate_blood_chip_signals(
        features,
        BLOOD_CHIP_SIGNAL_CONFIG,
        include_pending_entry=True,
    )
    signals = add_blood_chip_path_features(features, signals)
    signals = attach_blood_chip_kdj_path(features, signals)
    signals = select_blood_chip_long_signals(signals)
    concrete = signals.dropna(subset=["entry_date", "entry_open"]).copy()
    asof = pd.Timestamp(signal_date).normalize()
    entry_start = asof - pd.Timedelta(days=260)
    result = run_blood_chip_scale_in_backtest(
        features,
        concrete,
        BLOOD_CHIP_BACKTEST_CONFIG,
        BLOOD_CHIP_SCALE_IN_POLICY,
        entry_start.date().isoformat(),
        asof.date().isoformat(),
    )
    return compose_blood_chip_long_plan(
        signal_date=asof.date().isoformat(),
        signals=signals,
        result=result,
        stock_basic=stock_basic,
    )
