#!/usr/bin/env python3
"""Test causal bottom-pattern constraints on strict monthly low-zone anchors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.research.blood_chip_deep_base import build_deep_base_features
from quant.research.monthly_low_zone import (
    MonthlyLowZoneConfig,
    evaluate_monthly_low_zone_events,
    generate_monthly_low_zone_signals,
)
from quant.research.monthly_low_zone_patterns import (
    PATTERN_RULES,
    ChartPatternConfig,
    generate_chart_pattern_signals,
)
from quant.research.monthly_low_zone_profit_lock import (
    ProfitLockConfig,
    evaluate_profit_lock_events,
)
from quant.research.monthly_low_zone_strict import (
    cohort_bootstrap_interval,
    leave_one_cohort_out_minimum,
    wilson_interval,
)


TARGETS = (0.10, 0.15, 0.20)
DRAWDOWN_NEIGHBORHOOD = (0.50, 0.60, 0.70, 0.80)
BASELINE_RULE = "range_mid_reclaim"
PERIODS = {
    "old_cycle_2003_2012": (
        pd.Timestamp("2003-01-01"),
        pd.Timestamp("2012-12-31"),
    ),
    "recent_exposed_2013_2024": (
        pd.Timestamp("2013-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
    "combined_2003_2024": (
        pd.Timestamp("2003-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
}
DAILY_COLUMNS = [
    "ts_code",
    "date",
    "open",
    "high",
    "low",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "prior_amount_median_20d",
    "sessions_since_new_low",
    "return_20d",
    "base_position",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension-data-dir",
        type=Path,
        default=Path("data/research/monthly_low_zone_strict_extension"),
    )
    parser.add_argument(
        "--recent-signals",
        type=Path,
        default=Path(
            "reports/research/monthly_low_zone_confirmation_breadth/signals.parquet"
        ),
    )
    parser.add_argument(
        "--recent-gated-events",
        type=Path,
        default=Path("reports/research/monthly_low_zone_strict/gated_events.parquet"),
    )
    parser.add_argument(
        "--recent-feature-cache",
        type=Path,
        default=Path("data/research/blood_chip_deep_base/features.parquet"),
    )
    parser.add_argument(
        "--recent-supplemental-root",
        type=Path,
        default=Path("data/research/low9_kdj_rebound/supplemental_daily"),
    )
    parser.add_argument(
        "--recent-benchmark",
        type=Path,
        default=Path(
            "data/research/low9_kdj_rebound/index_000001.SH_20100101_20260731.parquet"
        ),
    )
    parser.add_argument(
        "--baseline-events",
        type=Path,
        default=Path(
            "reports/research/monthly_low_zone_strict_extension/combined_candidate_events.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/research/monthly_low_zone_chart_patterns"),
    )
    return parser.parse_args()


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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame["trade_date"] if "trade_date" in frame else frame["date"]
    compact = pd.to_datetime(
        source.astype("string").str.replace(r"\.0$", "", regex=True).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    parsed = pd.to_datetime(source, errors="coerce")
    out = frame.copy()
    out["date"] = compact.fillna(parsed).dt.normalize()
    return (
        out.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )


def _market_gate(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame["breadth_constituents"], errors="coerce").ge(500)
        & pd.to_numeric(
            frame["breadth_positive_share_20d"], errors="coerce"
        ).le(0.20)
        & pd.to_numeric(frame["breadth_median_return_20d"], errors="coerce").le(
            -0.10
        )
        & pd.to_numeric(frame["drawdown_from_prior_peak"], errors="coerce").le(
            -0.50
        )
    ).fillna(False)


def _load_old_anchors(data_dir: Path) -> pd.DataFrame:
    monthly = pd.read_parquet(data_dir / "monthly_features_2000_2015.parquet")
    weekly = pd.read_parquet(data_dir / "weekly_features_2000_2015.parquet")
    breadth = pd.read_parquet(data_dir / "market_breadth_2000_2015.parquet")
    monthly_signals = generate_monthly_low_zone_signals(
        monthly, weekly, MonthlyLowZoneConfig()
    )
    anchors = monthly_signals.loc[
        monthly_signals["rule"].eq("monthly_low9")
        & pd.to_datetime(
            monthly_signals["signal_date"], errors="coerce"
        ).le("2012-12-31")
    ].merge(
        breadth,
        left_on="signal_date",
        right_on="date",
        how="left",
        validate="many_to_one",
    )
    anchors = anchors.loc[_market_gate(anchors)].drop(columns="date").copy()
    anchors = anchors.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)
    anchors["source_sample"] = "old_cycle_2003_2012"
    return anchors


def _load_recent_anchors(signals_path: Path, gated_path: Path) -> pd.DataFrame:
    signals = pd.read_parquet(signals_path)
    anchors = signals.loc[
        signals["rule"].eq("anchor_direct")
        & pd.to_datetime(signals["signal_date"], errors="coerce").between(
            "2013-01-01", "2024-12-31", inclusive="both"
        )
    ].copy()
    gate_columns = [
        "signal_id",
        "horizon",
        "target_return",
        "breadth_constituents",
        "breadth_median_return_20d",
        "breadth_positive_share_20d",
    ]
    state = pd.read_parquet(gated_path, columns=gate_columns)
    state = state.loc[
        pd.to_numeric(state["horizon"], errors="coerce").eq(504)
        & np.isclose(pd.to_numeric(state["target_return"], errors="coerce"), 0.15)
    ].drop_duplicates("signal_id")
    rename = {
        column: f"{column}_anchor"
        for column in (
            "breadth_constituents",
            "breadth_median_return_20d",
            "breadth_positive_share_20d",
        )
    }
    anchors = anchors.merge(
        state.rename(columns=rename).drop(columns=["horizon", "target_return"]),
        on="signal_id",
        how="left",
        validate="one_to_one",
    )
    for column in rename:
        anchors[column] = anchors[f"{column}_anchor"]
    anchors = anchors.loc[_market_gate(anchors)].copy()
    anchors["rule"] = "monthly_low9"
    anchors["source_sample"] = "recent_exposed_2013_2024"
    return anchors.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)


def _load_old_daily(data_dir: Path, symbols: set[str]) -> pd.DataFrame:
    return pd.read_parquet(
        data_dir / "features_2000_2015.parquet",
        columns=DAILY_COLUMNS,
        filters=[("ts_code", "in", sorted(symbols))],
    )


def _load_recent_daily(
    feature_cache: Path,
    supplemental_root: Path,
    symbols: set[str],
) -> pd.DataFrame:
    primary = pd.read_parquet(
        feature_cache,
        columns=DAILY_COLUMNS,
        filters=[("ts_code", "in", sorted(symbols))],
    )
    raw_columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "vol",
        "amount",
    ]
    parts = []
    for symbol in sorted(symbols):
        path = supplemental_root / f"{symbol}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path, columns=raw_columns))
    supplemental = (
        build_deep_base_features(pd.concat(parts, ignore_index=True, sort=False))[
            DAILY_COLUMNS
        ]
        if parts
        else pd.DataFrame(columns=DAILY_COLUMNS)
    )
    combined = pd.concat([primary, supplemental], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
    return combined.sort_values(["ts_code", "date"]).drop_duplicates(
        ["ts_code", "date"], keep="last"
    )


def _load_old_benchmark(data_dir: Path, recent_path: Path) -> pd.DataFrame:
    parts = [
        pd.read_parquet(path)
        for path in sorted(data_dir.glob("index_000001.SH_*.parquet"))
    ]
    parts.append(pd.read_parquet(recent_path))
    benchmark = _normalize_benchmark(pd.concat(parts, ignore_index=True, sort=False))
    return benchmark.loc[benchmark["date"].le("2015-12-31")].copy()


def _evaluate_patterns(
    daily: pd.DataFrame,
    anchors: pd.DataFrame,
    benchmark: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    source_sample: str,
    config: ChartPatternConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals, diagnostics = generate_chart_pattern_signals(
        daily, anchors, calendar, config
    )
    signals["source_sample"] = source_sample
    diagnostics["source_sample"] = source_sample
    if signals.empty:
        return signals, diagnostics, pd.DataFrame()
    baseline = evaluate_monthly_low_zone_events(
        daily,
        signals,
        benchmark,
        calendar,
        MonthlyLowZoneConfig(),
    )
    lookup_columns = [
        "signal_id",
        "anchor_id",
        "anchor_date",
        "anchor_drawdown_from_prior_peak",
        "pattern_start_date",
        "pattern_middle_date",
        "pattern_end_date",
        "pattern_start_price",
        "pattern_middle_price",
        "pattern_end_price",
        "pattern_neckline",
        "pattern_rebound",
        "pattern_symmetry",
        "pattern_head_depth",
        "confirmation_wait_sessions",
    ]
    baseline = baseline.merge(
        signals[lookup_columns].drop_duplicates("signal_id"),
        on="signal_id",
        how="left",
        validate="many_to_one",
    )
    events = evaluate_profit_lock_events(
        daily,
        baseline,
        calendar,
        ProfitLockConfig(horizon_sessions=504, target_returns=TARGETS),
        benchmark,
    )
    events["source_sample"] = source_sample
    return signals, diagnostics, events


def _materialize_pattern_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.assign(structure=pd.Series(dtype="string"))
    drawdown = pd.to_numeric(events["drawdown_from_prior_peak"], errors="coerce")
    parts: list[pd.DataFrame] = []
    for rule in PATTERN_RULES:
        for threshold in DRAWDOWN_NEIGHBORHOOD:
            selected = events.loc[
                events["rule"].eq(rule) & drawdown.le(-threshold)
            ].copy()
            selected["structure"] = f"{rule}_dd{int(threshold * 100)}"
            parts.append(selected)
    return pd.concat(parts, ignore_index=True, sort=False)


def _load_baseline_events(path: Path) -> pd.DataFrame:
    baseline = pd.read_parquet(path)
    baseline["structure"] = baseline["structure"].astype("string").str.replace(
        "^case_range_reclaim_", "range_mid_reclaim_", regex=True
    )
    baseline["rule"] = BASELINE_RULE
    baseline["source_sample"] = baseline["source_sample"].replace(
        {
            "new_old_cycle_2003_2012": "old_cycle_2003_2012",
            "exposed_recent_2013_2024": "recent_exposed_2013_2024",
        }
    )
    return baseline


def _structure_catalog() -> list[str]:
    return [
        *[f"{BASELINE_RULE}_dd{int(value * 100)}" for value in DRAWDOWN_NEIGHBORHOOD],
        *[
            f"{rule}_dd{int(value * 100)}"
            for rule in PATTERN_RULES
            for value in DRAWDOWN_NEIGHBORHOOD
        ],
    ]


def _summarize_events(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = events.copy()
    frame["anchor_month"] = pd.PeriodIndex(
        frame["month_period"], freq="M"
    ).to_timestamp(how="end").normalize()
    completed = frame.loc[
        frame["entry_status"].eq("accepted")
        & frame["outcome_completed"].fillna(False)
        & frame["baseline_outcome_completed"].fillna(False)
    ].copy()
    completed["net_return"] = pd.to_numeric(completed["net_return"], errors="coerce")
    completed = completed.dropna(subset=["net_return"])
    cohort = (
        completed.groupby(
            ["structure", "target_return", "anchor_month"],
            observed=True,
            sort=True,
        )
        .agg(
            cohort_return=("net_return", "mean"),
            events=("net_return", "size"),
            winners=("net_return", lambda values: int(values.gt(0.0).sum())),
            worst_event_return=("net_return", "min"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for structure in _structure_catalog():
        for target in TARGETS:
            group = completed.loc[
                completed["structure"].eq(structure)
                & np.isclose(completed["target_return"], target)
            ]
            matching = cohort.loc[
                cohort["structure"].eq(structure)
                & np.isclose(cohort["target_return"], target)
            ]
            for period, (start, end) in PERIODS.items():
                scoped = group.loc[group["anchor_month"].between(start, end)]
                scoped_cohort = matching.loc[
                    matching["anchor_month"].between(start, end)
                ]
                returns = scoped["net_return"]
                cohort_returns = scoped_cohort.set_index("anchor_month")[
                    "cohort_return"
                ]
                gains = float(returns.loc[returns > 0.0].sum())
                losses = float(-returns.loc[returns <= 0.0].sum())
                event_low = wilson_interval(
                    int(returns.gt(0.0).sum()), len(returns)
                )[0]
                mean, lower, upper = cohort_bootstrap_interval(
                    cohort_returns, iterations=10_000, seed=20260826
                )
                rows.append(
                    {
                        "period": period,
                        "structure": structure,
                        "target_return": target,
                        "completed_events": int(len(returns)),
                        "completed_cohorts": int(len(cohort_returns)),
                        "event_win_rate": (
                            float(returns.gt(0.0).mean()) if len(returns) else np.nan
                        ),
                        "event_win_wilson_low": event_low,
                        "median_event_return": (
                            float(returns.median()) if len(returns) else np.nan
                        ),
                        "profit_factor": (
                            gains / losses
                            if losses > 0.0
                            else np.inf
                            if gains > 0.0
                            else np.nan
                        ),
                        "positive_cohort_share": (
                            float(cohort_returns.gt(0.0).mean())
                            if len(cohort_returns)
                            else np.nan
                        ),
                        "cohort_equal_mean_return": mean,
                        "cohort_bootstrap_ci95_low": lower,
                        "cohort_bootstrap_ci95_high": upper,
                        "worst_cohort_return": (
                            float(cohort_returns.min())
                            if len(cohort_returns)
                            else np.nan
                        ),
                        "leave_one_cohort_out_min_mean": (
                            leave_one_cohort_out_minimum(cohort_returns)
                        ),
                    }
                )
    return pd.DataFrame(rows), cohort


def _conversion_metrics(
    anchors: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = [
        "old_cycle_2003_2012",
        "recent_exposed_2013_2024",
        "combined_2003_2024",
    ]
    for source in sources:
        anchor_scope = (
            anchors
            if source == "combined_2003_2024"
            else anchors.loc[anchors["source_sample"].eq(source)]
        )
        signal_scope = (
            signals
            if source == "combined_2003_2024"
            else signals.loc[signals["source_sample"].eq(source)]
        )
        for threshold in DRAWDOWN_NEIGHBORHOOD:
            eligible = anchor_scope.loc[
                pd.to_numeric(
                    anchor_scope["drawdown_from_prior_peak"], errors="coerce"
                ).le(-threshold)
            ]
            for rule in PATTERN_RULES:
                confirmed = signal_scope.loc[
                    signal_scope["rule"].eq(rule)
                    & pd.to_numeric(
                        signal_scope["drawdown_from_prior_peak"], errors="coerce"
                    ).le(-threshold)
                ]
                rows.append(
                    {
                        "period": source,
                        "rule": rule,
                        "drawdown_threshold": threshold,
                        "eligible_anchors": int(len(eligible)),
                        "confirmed_patterns": int(len(confirmed)),
                        "independent_anchor_months": int(
                            confirmed["month_period"].nunique()
                        ),
                        "conversion_rate": (
                            float(len(confirmed) / len(eligible))
                            if len(eligible)
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _metric_row(
    metrics: pd.DataFrame,
    structure: str,
    period: str,
) -> pd.Series:
    selected = metrics.loc[
        metrics["structure"].eq(structure)
        & metrics["period"].eq(period)
        & np.isclose(metrics["target_return"], 0.15)
    ]
    return selected.iloc[0] if not selected.empty else pd.Series(dtype="object")


def _decide(metrics: pd.DataFrame) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    baseline_old = _metric_row(metrics, "range_mid_reclaim_dd60", "old_cycle_2003_2012")
    for rule in PATTERN_RULES:
        primary_structure = f"{rule}_dd60"
        old = _metric_row(metrics, primary_structure, "old_cycle_2003_2012")
        combined = _metric_row(metrics, primary_structure, "combined_2003_2024")
        sample_checks = {
            "old_cohorts_at_least_3": bool(old.get("completed_cohorts", 0) >= 3),
            "combined_cohorts_at_least_8": bool(
                combined.get("completed_cohorts", 0) >= 8
            ),
        }
        old_checks = {
            "event_win_rate_at_least_85pct": bool(
                old.get("event_win_rate", np.nan) >= 0.85
            ),
            "profit_factor_at_least_2": bool(
                old.get("profit_factor", np.nan) >= 2.0
            ),
            "positive_cohort_share_at_least_80pct": bool(
                old.get("positive_cohort_share", np.nan) >= 0.80
            ),
            "bootstrap_lower_positive": bool(
                old.get("cohort_bootstrap_ci95_low", np.nan) > 0.0
            ),
            "worst_cohort_no_worse_than_minus_5pct": bool(
                old.get("worst_cohort_return", np.nan) >= -0.05
            ),
        }
        combined_checks = {
            "bootstrap_lower_positive": bool(
                combined.get("cohort_bootstrap_ci95_low", np.nan) > 0.0
            ),
            "leave_one_cohort_out_positive": bool(
                combined.get("leave_one_cohort_out_min_mean", np.nan) > 0.0
            ),
        }
        comparison_checks = {
            "old_bootstrap_lower_improves_baseline": bool(
                old.get("cohort_bootstrap_ci95_low", np.nan)
                > baseline_old.get("cohort_bootstrap_ci95_low", np.nan)
            ),
            "old_worst_cohort_improves_baseline": bool(
                old.get("worst_cohort_return", np.nan)
                > baseline_old.get("worst_cohort_return", np.nan)
            ),
        }
        neighborhood = metrics.loc[
            metrics["period"].eq("combined_2003_2024")
            & metrics["structure"].str.startswith(rule)
            & np.isclose(metrics["target_return"], 0.15)
        ]
        stable = (
            neighborhood["cohort_bootstrap_ci95_low"].gt(0.0)
            & neighborhood["leave_one_cohort_out_min_mean"].gt(0.0)
        )
        neighborhood_checks = {
            "three_of_four_drawdown_neighbors_robust": bool(stable.sum() >= 3),
            "robust_neighbors": neighborhood.loc[stable, "structure"].tolist(),
        }
        passed = (
            all(sample_checks.values())
            and all(old_checks.values())
            and all(combined_checks.values())
            and all(comparison_checks.values())
            and neighborhood_checks["three_of_four_drawdown_neighbors_robust"]
        )
        if passed:
            status = "historical_pattern_increment"
        elif not all(sample_checks.values()):
            status = "pattern_sample_insufficient"
        else:
            status = "pattern_failed_cross_cycle_increment"
        decisions[rule] = {
            "primary_structure": primary_structure,
            "sample_checks": sample_checks,
            "old_cycle_checks": old_checks,
            "combined_checks": combined_checks,
            "baseline_comparison_checks": comparison_checks,
            "drawdown_neighborhood_checks": neighborhood_checks,
            "status": status,
        }
    any_increment = any(
        item["status"] == "historical_pattern_increment"
        for item in decisions.values()
    )
    return {
        "patterns": decisions,
        "overall_status": (
            "at_least_one_historical_pattern_increment"
            if any_increment
            else "no_proven_cross_cycle_pattern_increment"
        ),
        "deployment_eligible": False,
        "head_shoulders_top_entry_decision": (
            "rejected_by_direction: bearish top is not a low-entry confirmation"
        ),
        "stop_boundary": (
            "do not tune pivot radius, symmetry, spacing, or neckline buffers after results"
        ),
    }


def _case_tables(
    baseline: pd.DataFrame,
    pattern_events: pd.DataFrame,
    pattern_signals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_primary = baseline.loc[
        baseline["structure"].eq("range_mid_reclaim_dd60")
        & np.isclose(baseline["target_return"], 0.15)
        & baseline["entry_status"].eq("accepted")
        & baseline["outcome_completed"].fillna(False)
        & baseline["baseline_outcome_completed"].fillna(False)
    ].copy()
    baseline_primary["anchor_key"] = (
        baseline_primary["source_sample"].astype("string")
        + "|"
        + baseline_primary["ts_code"].astype("string")
        + "|"
        + baseline_primary["month_period"].astype("string")
    )
    exclusion_parts: list[pd.DataFrame] = []
    for rule in PATTERN_RULES:
        confirmed = pattern_signals.loc[
            pattern_signals["rule"].eq(rule)
            & pd.to_numeric(
                pattern_signals["drawdown_from_prior_peak"], errors="coerce"
            ).le(-0.60)
        ].copy()
        confirmed_keys = set(
            confirmed["source_sample"].astype("string")
            + "|"
            + confirmed["ts_code"].astype("string")
            + "|"
            + confirmed["month_period"].astype("string")
        )
        part = baseline_primary.copy()
        part["pattern_rule"] = rule
        part["pattern_confirmed"] = part["anchor_key"].isin(confirmed_keys)
        part["baseline_winner"] = pd.to_numeric(
            part["net_return"], errors="coerce"
        ).gt(0.0)
        exclusion_parts.append(part)
    exclusions = pd.concat(exclusion_parts, ignore_index=True, sort=False)
    negative_cases = pattern_events.loc[
        pattern_events["structure"].str.endswith("dd60")
        & np.isclose(pattern_events["target_return"], 0.15)
        & pattern_events["entry_status"].eq("accepted")
        & pattern_events["outcome_completed"].fillna(False)
        & pd.to_numeric(pattern_events["net_return"], errors="coerce").le(0.0)
    ].copy()
    return exclusions, negative_cases


def _format_percent(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = out[column].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.2%}"
            )
    return out


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    conversions: pd.DataFrame,
    exclusions: pd.DataFrame,
    negative_cases: pd.DataFrame,
    decision: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    primary_structures = [
        "range_mid_reclaim_dd60",
        "double_bottom_breakout_dd60",
        "inverse_head_shoulders_breakout_dd60",
    ]
    primary = metrics.loc[
        metrics["structure"].isin(primary_structures)
        & np.isclose(metrics["target_return"], 0.15)
    ].copy()
    neighborhood = metrics.loc[
        metrics["period"].eq("combined_2003_2024")
        & np.isclose(metrics["target_return"], 0.15)
    ].copy()
    percent_columns = [
        "target_return",
        "event_win_rate",
        "event_win_wilson_low",
        "median_event_return",
        "positive_cohort_share",
        "cohort_equal_mean_return",
        "cohort_bootstrap_ci95_low",
        "cohort_bootstrap_ci95_high",
        "worst_cohort_return",
        "leave_one_cohort_out_min_mean",
    ]
    primary = _format_percent(primary, percent_columns)
    neighborhood = _format_percent(neighborhood, percent_columns)
    conversion_display = _format_percent(
        conversions.loc[np.isclose(conversions["drawdown_threshold"], 0.60)].copy(),
        ["drawdown_threshold", "conversion_rate"],
    )
    exclusion_summary = (
        exclusions.groupby(
            ["pattern_rule", "source_sample", "pattern_confirmed"],
            observed=True,
            sort=True,
        )
        .agg(
            baseline_events=("net_return", "size"),
            baseline_winners=("baseline_winner", "sum"),
            baseline_mean_return=("net_return", "mean"),
            baseline_worst_return=("net_return", "min"),
        )
        .reset_index()
    )
    exclusion_summary = _format_percent(
        exclusion_summary, ["baseline_mean_return", "baseline_worst_return"]
    )
    negative_columns = [
        "source_sample",
        "structure",
        "ts_code",
        "month_period",
        "signal_date",
        "pattern_start_date",
        "pattern_middle_date",
        "pattern_end_date",
        "entry_date",
        "exit_date",
        "exit_reason",
        "net_return",
        "mae",
    ]
    negative_display = _format_percent(
        negative_cases.reindex(columns=negative_columns).sort_values("net_return"),
        ["net_return", "mae"],
    )
    conclusion = (
        "至少一种形态在冻结判据下形成了历史增益，但独立恐慌月没有增加，仍不能称为高确定性。"
        if decision["overall_status"]
        == "at_least_one_historical_pattern_increment"
        else "双底和头肩底都没有证明跨周期增益；停止继续微调图形参数。"
    )
    path.write_text(
        f"""# 月线低位区间：前高跌幅与技术形态约束回验

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

{conclusion}

本轮以月线低9、市场上涨家数不高于 20%、市场 20 日收益中位数不高于 -10%、个股前高回撤至少 60% 为主锚。双底和头肩底都必须等右侧拐点可见，并在锚后 126 个市场日内收盘突破 101% 颈线；15% 止盈、504 日上限、20bp 成本。

`头肩顶` 是看跌形态，不能作为低位建仓确认。本轮按方向性原则排除；若验证它，应作为入场后的独立退出研究，而不是抄底加分项。

## 60% 回撤、15% 止盈主比较

{primary.to_markdown(index=False)}

## 形态转化率

{conversion_display.to_markdown(index=False)}

## 合并样本的 50%/60%/70%/80% 邻域

{neighborhood.to_markdown(index=False)}

## 形态删掉了哪些原策略事件

{exclusion_summary.to_markdown(index=False) if not exclusion_summary.empty else '_没有可比较事件_'}

## 形态确认后仍亏损的案例

{negative_display.to_markdown(index=False) if not negative_display.empty else '_没有完成亏损事件_'}

## 冻结判定

```json
{json.dumps(_json_value(decision), ensure_ascii=False, indent=2)}
```

## 数据审计

```json
{json.dumps(_json_value(metadata), ensure_ascii=False, indent=2)}
```

本研究仅供研究与教育用途，不构成投资建议、收益承诺或交易指令。
""",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ChartPatternConfig()

    print("loading strict old and recent monthly anchors", flush=True)
    old_anchors = _load_old_anchors(args.extension_data_dir)
    recent_anchors = _load_recent_anchors(
        args.recent_signals, args.recent_gated_events
    )
    all_anchors = pd.concat(
        [old_anchors, recent_anchors], ignore_index=True, sort=False
    )
    print(
        f"eligible anchors: old={len(old_anchors):,}, recent={len(recent_anchors):,}",
        flush=True,
    )

    old_benchmark = _load_old_benchmark(
        args.extension_data_dir, args.recent_benchmark
    )
    old_calendar = pd.DatetimeIndex(sorted(old_benchmark["date"].unique()))
    old_daily = _load_old_daily(
        args.extension_data_dir, set(old_anchors["ts_code"].astype(str))
    )
    print("generating causal old-cycle chart patterns", flush=True)
    old_signals, old_diagnostics, old_events = _evaluate_patterns(
        old_daily,
        old_anchors,
        old_benchmark,
        old_calendar,
        "old_cycle_2003_2012",
        config,
    )

    recent_benchmark = _normalize_benchmark(pd.read_parquet(args.recent_benchmark))
    recent_calendar = pd.DatetimeIndex(sorted(recent_benchmark["date"].unique()))
    recent_daily = _load_recent_daily(
        args.recent_feature_cache,
        args.recent_supplemental_root,
        set(recent_anchors["ts_code"].astype(str)),
    )
    print("generating causal recent-period chart patterns", flush=True)
    recent_signals, recent_diagnostics, recent_events = _evaluate_patterns(
        recent_daily,
        recent_anchors,
        recent_benchmark,
        recent_calendar,
        "recent_exposed_2013_2024",
        config,
    )

    pattern_signals = pd.concat(
        [old_signals, recent_signals], ignore_index=True, sort=False
    )
    pattern_diagnostics = pd.concat(
        [old_diagnostics, recent_diagnostics], ignore_index=True, sort=False
    )
    raw_pattern_events = pd.concat(
        [old_events, recent_events], ignore_index=True, sort=False
    )
    pattern_events = _materialize_pattern_events(raw_pattern_events)
    baseline = _load_baseline_events(args.baseline_events)
    all_events = pd.concat([baseline, pattern_events], ignore_index=True, sort=False)
    metrics, cohorts = _summarize_events(all_events)
    conversions = _conversion_metrics(all_anchors, pattern_signals)
    decision = _decide(metrics)
    exclusions, negative_cases = _case_tables(
        baseline, pattern_events, pattern_signals
    )

    metadata = {
        "analysis_cutoff": pd.Timestamp("2026-07-31"),
        "pattern_config": config.to_dict(),
        "drawdown_neighborhood": DRAWDOWN_NEIGHBORHOOD,
        "target_neighborhood": TARGETS,
        "old_eligible_anchors_dd50": int(len(old_anchors)),
        "recent_eligible_anchors_dd50": int(len(recent_anchors)),
        "old_independent_anchor_months": int(old_anchors["month_period"].nunique()),
        "recent_independent_anchor_months": int(
            recent_anchors["month_period"].nunique()
        ),
        "old_pattern_signals": int(len(old_signals)),
        "recent_pattern_signals": int(len(recent_signals)),
        "completed_pattern_event_rows": int(
            pattern_events["outcome_completed"].fillna(False).sum()
        )
        if not pattern_events.empty
        else 0,
        "claim_boundary": (
            "both historical periods have now been exposed; no out-of-sample claim"
        ),
    }
    pattern_signals.to_parquet(
        args.output_dir / "pattern_signals.parquet",
        index=False,
        compression="zstd",
    )
    pattern_diagnostics.to_parquet(
        args.output_dir / "pattern_diagnostics.parquet",
        index=False,
        compression="zstd",
    )
    pattern_events.to_parquet(
        args.output_dir / "pattern_events.parquet",
        index=False,
        compression="zstd",
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    cohorts.to_csv(args.output_dir / "cohort_metrics.csv", index=False)
    conversions.to_csv(args.output_dir / "conversion_metrics.csv", index=False)
    exclusions.to_parquet(
        args.output_dir / "baseline_exclusion_cases.parquet",
        index=False,
        compression="zstd",
    )
    negative_cases.to_parquet(
        args.output_dir / "negative_pattern_cases.parquet",
        index=False,
        compression="zstd",
    )
    _write_json(args.output_dir / "decision.json", decision)
    _write_json(args.output_dir / "metadata.json", metadata)
    _write_report(
        args.output_dir / "report.md",
        metrics,
        conversions,
        exclusions,
        negative_cases,
        decision,
        metadata,
    )
    print(f"pattern validation status: {decision['overall_status']}", flush=True)
    print(f"report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
