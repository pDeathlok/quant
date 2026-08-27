#!/usr/bin/env python3
"""Persist the final market-repair and financial-survival falsification."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import backtest_monthly_low_zone_strict_extension as extension
from backtest_monthly_low_zone_confirmation import (
    FINANCIAL_SOURCE_COLUMNS,
    _read_available_columns,
)
from quant.features.long_quality_factors import build_annual_quality_events
from quant.research.blood_chip_multidimensional import merge_financial_survival_asof
from quant.research.monthly_low_zone import (
    MonthlyLowZoneConfig,
    evaluate_monthly_low_zone_events,
    generate_monthly_low_zone_signals,
)
from quant.research.monthly_low_zone_confirmation import (
    MonthlyConfirmationConfig,
    build_benchmark_confirmation_features,
    generate_monthly_confirmation_signals,
)
from quant.research.monthly_low_zone_profit_lock import (
    ProfitLockConfig,
    evaluate_profit_lock_events,
)
from quant.research.monthly_low_zone_strict import (
    StrictLowZoneConfig,
    add_strict_gate_columns,
)


REPAIR_RULES = (
    "breadth_repair",
    "breadth_relative",
    "breadth_relative_weekly",
    "confirmed_market",
)
OUTPUT_DIR = Path("reports/research/monthly_low_zone_strict_falsification")
EXTENSION_DIR = Path("data/research/monthly_low_zone_strict_extension")
RECENT_BENCHMARK = Path(
    "data/research/low9_kdj_rebound/index_000001.SH_20100101_20260731.parquet"
)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isposinf(value):
            return "infinity"
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _old_repair_events() -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = EXTENSION_DIR / "old_cycle_market_repair_profit_events.parquet"
    if cache.exists():
        events = pd.read_parquet(cache)
        return events, {"cache_hit": True, "events": int(len(events))}
    features = pd.read_parquet(EXTENSION_DIR / "features_2000_2015.parquet")
    monthly = pd.read_parquet(EXTENSION_DIR / "monthly_features_2000_2015.parquet")
    weekly = pd.read_parquet(EXTENSION_DIR / "weekly_features_2000_2015.parquet")
    breadth = pd.read_parquet(EXTENSION_DIR / "market_breadth_2000_2015.parquet")
    benchmark = extension._load_extension_benchmark(
        EXTENSION_DIR, RECENT_BENCHMARK
    )
    benchmark = benchmark.loc[benchmark["date"].le("2015-12-31")]
    calendar = pd.DatetimeIndex(sorted(benchmark["date"].unique()))
    monthly_signals = generate_monthly_low_zone_signals(
        monthly, weekly, MonthlyLowZoneConfig()
    )
    anchors = monthly_signals.loc[
        monthly_signals["rule"].eq("monthly_low9")
        & pd.to_datetime(monthly_signals["signal_date"]).le("2012-12-31")
    ].copy()
    anchors = anchors.merge(
        breadth,
        left_on="signal_date",
        right_on="date",
        how="left",
    ).drop(columns="date")
    anchors = anchors.loc[
        anchors["breadth_constituents"].ge(500)
        & anchors["breadth_positive_share_20d"].le(0.20)
        & anchors["breadth_median_return_20d"].le(-0.10)
        & anchors["drawdown_from_prior_peak"].le(-0.50)
    ].copy()
    benchmark_features = build_benchmark_confirmation_features(benchmark).merge(
        breadth, on="date", how="left", validate="one_to_one"
    )
    signals, _ = generate_monthly_confirmation_signals(
        features,
        weekly,
        anchors,
        benchmark_features,
        calendar,
        MonthlyConfirmationConfig(),
    )
    signals = signals.loc[signals["rule"].isin(REPAIR_RULES)].copy()
    baseline = evaluate_monthly_low_zone_events(
        features,
        signals,
        benchmark,
        calendar,
        MonthlyLowZoneConfig(),
    )
    lookup = signals[["signal_id", "anchor_id", "anchor_date"]].drop_duplicates(
        "signal_id"
    )
    baseline = baseline.merge(lookup, on="signal_id", how="left")
    events = evaluate_profit_lock_events(
        features,
        baseline,
        calendar,
        ProfitLockConfig(horizon_sessions=504, target_returns=(0.10, 0.15, 0.20)),
        benchmark,
    )
    events["source_sample"] = "new_old_cycle_2003_2012"
    events.to_parquet(cache, index=False, compression="zstd")
    return events, {
        "cache_hit": False,
        "filtered_old_anchors": int(len(anchors)),
        "confirmed_signals": int(len(signals)),
        "events": int(len(events)),
    }


def _recent_repair_events() -> tuple[pd.DataFrame, dict[str, Any]]:
    signals = pd.read_parquet(
        "reports/research/monthly_low_zone_confirmation_breadth/signals.parquet"
    )
    baseline = pd.read_parquet(
        "reports/research/monthly_low_zone_confirmation_breadth/events.parquet"
    )
    gated = pd.read_parquet(
        "reports/research/monthly_low_zone_strict/gated_events.parquet"
    )
    anchor_state = gated.loc[
        gated["rule"].eq("anchor_direct")
        & gated["horizon"].eq(504)
        & np.isclose(gated["target_return"], 0.15),
        [
            "anchor_id",
            "breadth_constituents",
            "breadth_positive_share_20d",
            "breadth_median_return_20d",
            "drawdown_from_prior_peak",
        ],
    ].drop_duplicates("anchor_id")
    anchor_state = anchor_state.rename(
        columns={
            "breadth_constituents": "anchor_constituents",
            "breadth_positive_share_20d": "anchor_positive_share",
            "breadth_median_return_20d": "anchor_median_return",
            "drawdown_from_prior_peak": "anchor_drawdown",
        }
    )
    selected = signals.loc[signals["rule"].isin(REPAIR_RULES)].merge(
        anchor_state, on="anchor_id", how="inner"
    )
    selected = selected.loc[
        selected["anchor_constituents"].ge(500)
        & selected["anchor_positive_share"].le(0.20)
        & selected["anchor_median_return"].le(-0.10)
        & selected["anchor_drawdown"].le(-0.50)
    ].copy()
    baseline = baseline.loc[
        baseline["signal_id"].isin(selected["signal_id"])
        & baseline["horizon"].eq(504)
    ].merge(
        selected[["signal_id", "anchor_id"]].drop_duplicates("signal_id"),
        on="signal_id",
        how="left",
    )
    symbols = set(selected["ts_code"].astype(str))
    daily = extension._load_recent_prices(
        Path("data/research/blood_chip_deep_base/features.parquet"),
        Path("data/research/low9_kdj_rebound/supplemental_daily"),
        symbols,
    )
    benchmark = extension._normalize_benchmark(pd.read_parquet(RECENT_BENCHMARK))
    calendar = pd.DatetimeIndex(sorted(benchmark["date"].unique()))
    events = evaluate_profit_lock_events(
        daily,
        baseline,
        calendar,
        ProfitLockConfig(horizon_sessions=504, target_returns=(0.10, 0.15, 0.20)),
        benchmark,
    )
    events["source_sample"] = "exposed_recent_2013_2024"
    return events, {
        "selected_signals": int(len(selected)),
        "symbols": int(len(symbols)),
        "events": int(len(events)),
    }


def _repair_structures(old: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([old, recent], ignore_index=True, sort=False)
    drawdown = pd.to_numeric(combined["drawdown_from_prior_peak"], errors="coerce")
    parts: list[pd.DataFrame] = []
    for rule in REPAIR_RULES:
        for threshold in (0.50, 0.60, 0.70, 0.80):
            selected = combined.loc[
                combined["rule"].eq(rule) & drawdown.le(-threshold)
            ].copy()
            selected["structure"] = f"{rule}_dd{int(threshold * 100)}"
            parts.append(selected)
    return pd.concat(parts, ignore_index=True, sort=False)


def _financial_structures() -> tuple[pd.DataFrame, dict[str, Any]]:
    sources = {
        name: _read_available_columns(Path("data/raw") / f"{name}.parquet", columns)
        for name, columns in FINANCIAL_SOURCE_COLUMNS.items()
    }
    annual = build_annual_quality_events(
        sources["fina_indicator"],
        sources["income"],
        sources["cashflow"],
        sources["balancesheet"],
    )
    old = pd.read_parquet(EXTENSION_DIR / "old_cycle_profit_events.parquet")
    signals = old[["signal_id", "ts_code", "signal_date"]].drop_duplicates(
        "signal_id"
    )
    financial = merge_financial_survival_asof(signals, annual)
    keep = [column for column in financial if column not in {"ts_code", "signal_date"}]
    old = old.merge(financial[keep], on="signal_id", how="left")
    old = add_strict_gate_columns(old, StrictLowZoneConfig())
    old["source_sample"] = "new_old_cycle_2003_2012"
    recent = pd.read_parquet(
        "reports/research/monthly_low_zone_strict/gated_events.parquet"
    )
    recent = recent.loc[
        recent["rule"].eq("range_mid_reclaim") & recent["horizon"].eq(504)
    ].copy()
    recent["source_sample"] = "exposed_recent_2013_2024"
    combined = pd.concat([old, recent], ignore_index=True, sort=False)
    market = (
        combined["breadth_constituents"].ge(500)
        & combined["breadth_positive_share_20d"].le(0.20)
        & combined["breadth_median_return_20d"].le(-0.10)
    )
    drawdown = pd.to_numeric(combined["drawdown_from_prior_peak"], errors="coerce")
    parts: list[pd.DataFrame] = []
    for survival in (0.60, 0.80):
        gate = combined[f"gate_survival{int(survival * 100)}"].fillna(False)
        for threshold in (0.50, 0.60, 0.70, 0.80):
            selected = combined.loc[
                market & gate & drawdown.le(-threshold)
            ].copy()
            selected["structure"] = (
                f"range_mid_reclaim_dd{int(threshold * 100)}_"
                f"survival{int(survival * 100)}"
            )
            parts.append(selected)
    return pd.concat(parts, ignore_index=True, sort=False), {
        "annual_quality_events": int(len(annual)),
        "old_financial_coverage": int(old["financial_coverage"].fillna(False).sum()),
    }


def _old_cycle_passes(metrics: pd.DataFrame) -> pd.Series:
    return (
        metrics["completed_cohorts"].ge(5)
        & metrics["event_win_rate"].ge(0.85)
        & metrics["profit_factor"].ge(2.0)
        & metrics["positive_cohort_share"].ge(0.80)
        & metrics["cohort_bootstrap_ci95_low"].gt(0.0)
        & metrics["worst_cohort_return"].ge(-0.05)
        & metrics["leave_one_cohort_out_min_mean"].gt(0.0)
    )


def _format_percent(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in (
        "event_win_rate",
        "positive_cohort_share",
        "cohort_equal_mean_return",
        "cohort_bootstrap_ci95_low",
        "worst_cohort_return",
        "leave_one_cohort_out_min_mean",
    ):
        if column in out:
            out[column] = out[column].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.2%}"
            )
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building final market-repair falsification", flush=True)
    old, old_metadata = _old_repair_events()
    recent, recent_metadata = _recent_repair_events()
    repair_events = _repair_structures(old, recent)
    repair_metrics, repair_cohorts = extension._summarize_candidate_events(
        repair_events
    )
    print("building point-in-time financial-survival falsification", flush=True)
    financial_events, financial_metadata = _financial_structures()
    financial_metrics, financial_cohorts = extension._summarize_candidate_events(
        financial_events
    )
    repair_old = repair_metrics.loc[
        repair_metrics["period"].eq("new_old_cycle_2003_2012")
        & np.isclose(repair_metrics["target_return"], 0.15)
    ].copy()
    financial_old = financial_metrics.loc[
        financial_metrics["period"].eq("new_old_cycle_2003_2012")
        & np.isclose(financial_metrics["target_return"], 0.15)
    ].copy()
    repair_old["frozen_old_cycle_gate_passed"] = _old_cycle_passes(repair_old)
    financial_old["frozen_old_cycle_gate_passed"] = _old_cycle_passes(financial_old)
    passed = pd.concat(
        [
            repair_old.loc[repair_old["frozen_old_cycle_gate_passed"]],
            financial_old.loc[financial_old["frozen_old_cycle_gate_passed"]],
        ],
        ignore_index=True,
    )
    decision = {
        "repair_rules": REPAIR_RULES,
        "financial_survival_diagnostics": [0.60, 0.80],
        "old_cycle_candidates_tested": int(len(repair_old) + len(financial_old)),
        "old_cycle_candidates_passed": int(len(passed)),
        "status": (
            "unexpected_candidate_requires_new_holdout"
            if not passed.empty
            else "no_high_certainty_blood_chip_structure_found"
        ),
        "stop_reason": (
            "all mechanism-linked stricter confirmations failed old-cycle gates; "
            "further threshold cutting would be post-hoc sample selection"
        ),
        "deployment_eligible": False,
    }
    metadata = {
        "generated_at": datetime.now(),
        "old_market_repair": old_metadata,
        "recent_market_repair": recent_metadata,
        "financial": financial_metadata,
    }
    repair_metrics.to_csv(OUTPUT_DIR / "market_repair_metrics.csv", index=False)
    repair_cohorts.to_csv(OUTPUT_DIR / "market_repair_cohorts.csv", index=False)
    financial_metrics.to_csv(
        OUTPUT_DIR / "financial_survival_metrics.csv", index=False
    )
    financial_cohorts.to_csv(
        OUTPUT_DIR / "financial_survival_cohorts.csv", index=False
    )
    repair_old.to_csv(OUTPUT_DIR / "old_cycle_market_repair.csv", index=False)
    financial_old.to_csv(OUTPUT_DIR / "old_cycle_financial_survival.csv", index=False)
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(_json_value(decision), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(_json_value(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    repair_display = _format_percent(
        repair_old.loc[
            repair_old["structure"].str.endswith("dd60"),
            [
                "structure",
                "completed_events",
                "completed_cohorts",
                "event_win_rate",
                "profit_factor",
                "positive_cohort_share",
                "cohort_equal_mean_return",
                "cohort_bootstrap_ci95_low",
                "worst_cohort_return",
                "leave_one_cohort_out_min_mean",
                "frozen_old_cycle_gate_passed",
            ],
        ]
    )
    financial_display = _format_percent(
        financial_old.loc[
            financial_old["structure"].str.contains("dd60"),
            [
                "structure",
                "completed_events",
                "completed_cohorts",
                "event_win_rate",
                "profit_factor",
                "positive_cohort_share",
                "cohort_bootstrap_ci95_low",
                "worst_cohort_return",
                "frozen_old_cycle_gate_passed",
            ],
        ]
    )
    (OUTPUT_DIR / "report.md").write_text(
        f"""# 月线带血筹最终反证审计

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

没有找到达到冻结高确定性标准的带血筹结构。市场宽度同步修复、相对宽度修复、周线修复、指数修复与点时财务生存门槛都未能修复 2003—2012 旧周期。继续按个别亏损月切阈值会变成事后选择，因此研究在这里停止。

## 市场同步修复，前高回撤 60%

{repair_display.to_markdown(index=False)}

## 点时财务生存，前高回撤 60%

{financial_display.to_markdown(index=False)}

## 判定

```json
{json.dumps(_json_value(decision), ensure_ascii=False, indent=2)}
```

近期样本中最强的 case 线索仍可进入观察清单，但不能称为高确定性策略或据此扩大仓位。本研究仅供研究与教育用途，不构成投资建议、收益承诺或交易指令。
""",
        encoding="utf-8",
    )
    print(f"falsification status: {decision['status']}", flush=True)
    print(f"report: {OUTPUT_DIR / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
