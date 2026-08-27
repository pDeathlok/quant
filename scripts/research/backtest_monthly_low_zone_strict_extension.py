#!/usr/bin/env python3
"""Validate the case-corrected strict structure on 2000-2012 old cycles."""

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
    build_monthly_weekly_features,
    evaluate_monthly_low_zone_events,
    generate_monthly_low_zone_signals,
)
from quant.research.monthly_low_zone_confirmation import (
    MonthlyConfirmationConfig,
    build_benchmark_confirmation_features,
    build_market_breadth_features,
    generate_monthly_confirmation_signals,
)
from quant.research.monthly_low_zone_profit_lock import (
    ProfitLockConfig,
    ProfitLockPortfolioConfig,
    evaluate_profit_lock_events,
    simulate_profit_lock_portfolio,
)
from quant.research.monthly_low_zone_strict import (
    cohort_bootstrap_interval,
    leave_one_cohort_out_minimum,
    wilson_interval,
)


TARGETS = (0.10, 0.15, 0.20)
DRAWDOWN_NEIGHBORHOOD = (0.50, 0.60, 0.70, 0.80)
PRIMARY_STRUCTURE = "case_range_reclaim_dd60"
SUMMARY_PERIODS = {
    "new_old_cycle_2003_2012": (
        pd.Timestamp("2003-01-01"),
        pd.Timestamp("2012-12-31"),
    ),
    "exposed_recent_2013_2024": (
        pd.Timestamp("2013-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
    "combined_2003_2024": (
        pd.Timestamp("2003-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
}
PORTFOLIO_PERIODS = {
    "new_old_cycle_2003_2012_held_to_2015": (
        pd.Timestamp("2003-01-01"),
        pd.Timestamp("2015-12-31"),
    ),
    "development_2013_2016": (
        pd.Timestamp("2013-01-01"),
        pd.Timestamp("2016-12-31"),
    ),
    "exposed_validation_2017_2020": (
        pd.Timestamp("2017-01-01"),
        pd.Timestamp("2020-12-31"),
    ),
    "seen_diagnostic_2021_2024": (
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension-data-dir",
        type=Path,
        default=Path("data/research/monthly_low_zone_strict_extension"),
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
        "--output-dir",
        type=Path,
        default=Path("reports/research/monthly_low_zone_strict_extension"),
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


def _a_share_mask(codes: pd.Series) -> pd.Series:
    text = codes.astype("string")
    return (
        text.str.match(r"^(000|001|002|003|300|301)\d{3}\.SZ$", na=False)
        | text.str.match(r"^(600|601|603|605|688|689)\d{3}\.SH$", na=False)
        | text.str.match(r"^(4|8|9)\d{5}\.BJ$", na=False)
    )


def _normalize_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame["trade_date"] if "trade_date" in frame else frame["date"]
    compact = pd.to_datetime(
        source.astype("string").str.replace(r"\.0$", "", regex=True).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    fallback = pd.to_datetime(source, errors="coerce")
    out = frame.copy()
    out["date"] = compact.fillna(fallback).dt.normalize()
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates(
        "date", keep="last"
    )


def _load_extension_benchmark(
    data_dir: Path,
    recent_path: Path,
) -> pd.DataFrame:
    extension_paths = sorted(data_dir.glob("index_000001.SH_*.parquet"))
    parts = [pd.read_parquet(path) for path in extension_paths]
    parts.append(pd.read_parquet(recent_path))
    return _normalize_benchmark(pd.concat(parts, ignore_index=True, sort=False))


def _load_or_build_extension_features(
    data_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = data_dir / "features_2000_2015.parquet"
    if cache.exists():
        print(f"loading extension feature cache: {cache}", flush=True)
        features = pd.read_parquet(cache)
        return features, {"cache_hit": True, "rows": int(len(features))}
    paths = sorted(data_dir.glob("market_daily_20*.parquet"))
    if not paths:
        raise FileNotFoundError("no market_daily yearly files found")
    print(f"loading {len(paths)} extension yearly files", flush=True)
    raw = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
        sort=False,
    )
    raw = raw.loc[_a_share_mask(raw["ts_code"])].copy()
    raw = (
        raw.sort_values(["ts_code", "trade_date"])
        .drop_duplicates(["ts_code", "trade_date"], keep="last")
        .reset_index(drop=True)
    )
    print(f"building causal extension features: {len(raw):,} A-share rows", flush=True)
    features = build_deep_base_features(raw)
    features.to_parquet(cache, index=False, compression="zstd")
    return features, {
        "cache_hit": False,
        "raw_rows": int(len(raw)),
        "raw_symbols": int(raw["ts_code"].nunique()),
        "rows": int(len(features)),
    }


def _load_or_build_multitimeframe(
    features: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    monthly_path = data_dir / "monthly_features_2000_2015.parquet"
    weekly_path = data_dir / "weekly_features_2000_2015.parquet"
    if monthly_path.exists() and weekly_path.exists():
        return pd.read_parquet(monthly_path), pd.read_parquet(weekly_path), True
    print("building old-cycle monthly and weekly bars", flush=True)
    monthly, weekly = build_monthly_weekly_features(features, calendar)
    monthly.to_parquet(monthly_path, index=False, compression="zstd")
    weekly.to_parquet(weekly_path, index=False, compression="zstd")
    return monthly, weekly, False


def _build_old_profit_events(
    features: pd.DataFrame,
    benchmark: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    data_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    event_cache = data_dir / "old_cycle_profit_events.parquet"
    if event_cache.exists():
        print(f"loading old-cycle profit event cache: {event_cache}", flush=True)
        events = pd.read_parquet(event_cache)
        return events, {"event_cache_hit": True, "profit_event_rows": int(len(events))}
    monthly, weekly, timeframe_cache_hit = _load_or_build_multitimeframe(
        features, calendar, data_dir
    )
    low_config = MonthlyLowZoneConfig()
    monthly_signals = generate_monthly_low_zone_signals(monthly, weekly, low_config)
    anchors = monthly_signals.loc[
        monthly_signals["rule"].eq("monthly_low9")
        & pd.to_datetime(monthly_signals["signal_date"]).le("2012-12-31")
    ].copy()
    confirmation_config = MonthlyConfirmationConfig()
    breadth_path = data_dir / "market_breadth_2000_2015.parquet"
    if breadth_path.exists():
        breadth = pd.read_parquet(breadth_path)
    else:
        print("building old-cycle full-market breadth", flush=True)
        breadth = build_market_breadth_features(features, confirmation_config)
        breadth.to_parquet(breadth_path, index=False, compression="zstd")
    benchmark_features = build_benchmark_confirmation_features(benchmark).merge(
        breadth,
        on="date",
        how="left",
        validate="one_to_one",
    )
    print(f"generating range-reclaim confirmations for {len(anchors):,} old anchors", flush=True)
    confirmation_signals, diagnostics = generate_monthly_confirmation_signals(
        features,
        weekly,
        anchors,
        benchmark_features,
        calendar,
        confirmation_config,
    )
    range_signals = confirmation_signals.loc[
        confirmation_signals["rule"].eq("range_mid_reclaim")
        & pd.to_datetime(confirmation_signals["anchor_date"]).le("2012-12-31")
    ].copy()
    print(f"evaluating old-cycle baseline paths: {len(range_signals):,}", flush=True)
    baseline = evaluate_monthly_low_zone_events(
        features,
        range_signals,
        benchmark,
        calendar,
        low_config,
    )
    lookup = range_signals[
        ["signal_id", "anchor_id", "anchor_date", "anchor_drawdown_from_prior_peak"]
    ].drop_duplicates("signal_id")
    baseline = baseline.merge(
        lookup,
        on="signal_id",
        how="left",
        validate="many_to_one",
    )
    print("applying fixed 10/15/20% profit locks to old paths", flush=True)
    events = evaluate_profit_lock_events(
        features,
        baseline,
        calendar,
        ProfitLockConfig(horizon_sessions=504, target_returns=TARGETS),
        benchmark,
    )
    anchor_state = (
        anchors[["signal_id", "signal_date", "month_period"]]
        .rename(columns={"signal_id": "anchor_id", "signal_date": "anchor_date"})
        .merge(
            breadth,
            left_on="anchor_date",
            right_on="date",
            how="left",
            validate="many_to_one",
        )
        .drop(columns="date")
    )
    events = events.merge(
        lookup[["signal_id", "anchor_id", "anchor_date"]],
        on="signal_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_lookup"),
    )
    if "anchor_id_lookup" in events:
        events["anchor_id"] = events["anchor_id"].fillna(events["anchor_id_lookup"])
        events = events.drop(columns="anchor_id_lookup")
    if "anchor_date_lookup" in events:
        events["anchor_date"] = events["anchor_date"].fillna(events["anchor_date_lookup"])
        events = events.drop(columns="anchor_date_lookup")
    events = events.merge(
        anchor_state.drop(columns=["anchor_date", "month_period"]),
        on="anchor_id",
        how="left",
        validate="many_to_one",
    )
    events["source_sample"] = "new_old_cycle_2003_2012"
    events.to_parquet(event_cache, index=False, compression="zstd")
    diagnostics.to_parquet(
        data_dir / "old_cycle_confirmation_diagnostics.parquet",
        index=False,
        compression="zstd",
    )
    return events, {
        "event_cache_hit": False,
        "timeframe_cache_hit": timeframe_cache_hit,
        "monthly_rows": int(len(monthly)),
        "weekly_rows": int(len(weekly)),
        "monthly_low9_anchors_through_2012": int(len(anchors)),
        "range_reclaim_signals": int(len(range_signals)),
        "baseline_event_rows": int(len(baseline)),
        "profit_event_rows": int(len(events)),
    }


def _recent_case_events(path: Path) -> pd.DataFrame:
    events = pd.read_parquet(path)
    events = events.loc[
        events["rule"].eq("range_mid_reclaim")
        & events["horizon"].eq(504)
        & events["baseline_outcome_completed"].fillna(False)
    ].copy()
    events["anchor_date"] = pd.PeriodIndex(
        events["month_period"], freq="M"
    ).to_timestamp(how="end").normalize()
    events["source_sample"] = "exposed_recent_2013_2024"
    return events


def _materialize_candidate_events(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    deep_market = (
        pd.to_numeric(frame["breadth_constituents"], errors="coerce").ge(500)
        & pd.to_numeric(frame["breadth_positive_share_20d"], errors="coerce").le(0.20)
        & pd.to_numeric(frame["breadth_median_return_20d"], errors="coerce").le(-0.10)
    )
    drawdown = pd.to_numeric(frame["drawdown_from_prior_peak"], errors="coerce")
    parts: list[pd.DataFrame] = []
    for threshold in DRAWDOWN_NEIGHBORHOOD:
        selected = frame.loc[deep_market & drawdown.le(-threshold)].copy()
        selected["structure"] = f"case_range_reclaim_dd{int(threshold * 100)}"
        parts.append(selected)
    return pd.concat(parts, ignore_index=True, sort=False)


def _summarize_candidate_events(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = events.copy()
    frame["anchor_month"] = pd.PeriodIndex(frame["month_period"], freq="M").to_timestamp(
        how="end"
    ).normalize()
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
    for (structure, target), group in completed.groupby(
        ["structure", "target_return"], observed=True, sort=True
    ):
        matching = cohort.loc[
            cohort["structure"].eq(structure)
            & np.isclose(cohort["target_return"], float(target))
        ]
        for period, (start, end) in SUMMARY_PERIODS.items():
            scoped = group.loc[group["anchor_month"].between(start, end)]
            scoped_cohort = matching.loc[matching["anchor_month"].between(start, end)]
            returns = scoped["net_return"]
            cohort_returns = scoped_cohort.set_index("anchor_month")["cohort_return"]
            gains = float(returns.loc[returns > 0.0].sum())
            losses = float(-returns.loc[returns <= 0.0].sum())
            event_wilson = wilson_interval(int(returns.gt(0.0).sum()), len(returns))[0]
            cohort_wilson = wilson_interval(
                int(cohort_returns.gt(0.0).sum()), len(cohort_returns)
            )[0]
            mean, lower, upper = cohort_bootstrap_interval(
                cohort_returns, iterations=10_000, seed=20260826
            )
            rows.append(
                {
                    "period": period,
                    "structure": structure,
                    "target_return": float(target),
                    "completed_events": int(len(returns)),
                    "completed_cohorts": int(len(cohort_returns)),
                    "event_win_rate": float(returns.gt(0.0).mean()) if len(returns) else np.nan,
                    "event_win_wilson_low": event_wilson,
                    "median_event_return": float(returns.median()) if len(returns) else np.nan,
                    "profit_factor": gains / losses if losses > 0.0 else (np.inf if gains > 0.0 else np.nan),
                    "positive_cohort_share": float(cohort_returns.gt(0.0).mean()) if len(cohort_returns) else np.nan,
                    "positive_cohort_wilson_low": cohort_wilson,
                    "cohort_equal_mean_return": mean,
                    "cohort_bootstrap_ci95_low": lower,
                    "cohort_bootstrap_ci95_high": upper,
                    "worst_cohort_return": float(cohort_returns.min()) if len(cohort_returns) else np.nan,
                    "leave_one_cohort_out_min_mean": leave_one_cohort_out_minimum(cohort_returns),
                }
            )
    return pd.DataFrame(rows), cohort


def _load_recent_prices(
    feature_cache: Path,
    supplemental_root: Path,
    symbols: set[str],
) -> pd.DataFrame:
    columns = [
        "ts_code",
        "date",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
    ]
    primary = pd.read_parquet(
        feature_cache,
        columns=columns,
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
        build_deep_base_features(pd.concat(parts, ignore_index=True, sort=False))[columns]
        if parts
        else pd.DataFrame(columns=columns)
    )
    combined = pd.concat([primary, supplemental], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
    return combined.sort_values(["ts_code", "date"]).drop_duplicates(
        ["ts_code", "date"], keep="last"
    )


def _curve_metrics(curve: pd.DataFrame) -> dict[str, float]:
    nav = pd.to_numeric(curve["nav"], errors="coerce")
    dates = pd.to_datetime(curve["date"], errors="coerce")
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    years = max(float((dates.iloc[-1] - dates.iloc[0]).days / 365.25), 1 / 365.25)
    drawdown = nav / nav.cummax() - 1.0
    rolling = nav / nav.shift(504) - 1.0
    return {
        "total_return": total,
        "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1.0),
        "maximum_drawdown": float(drawdown.min()),
        "worst_rolling_24m_return": (
            float(rolling.dropna().min()) if rolling.notna().any() else total
        ),
        "mean_invested_fraction": float(
            pd.to_numeric(curve["invested_fraction"], errors="coerce").mean()
        ),
    }


def _simulate_one_period(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    period: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    renamed = events.copy()
    renamed["source_rule"] = renamed["rule"]
    renamed["rule"] = "anchor_direct"
    curve, trades, audit = simulate_profit_lock_portfolio(
        daily,
        renamed,
        calendar,
        start_date=start,
        end_date=end,
        target_return=0.15,
        add_rule=None,
        config=ProfitLockPortfolioConfig(
            maximum_anchors=20,
            target_anchor_fraction=0.10,
            probe_budget_fraction=0.25,
        ),
    )
    curve["period"] = period
    trades["period"] = period
    return curve, trades, {"period": period, **_curve_metrics(curve), **audit}


def _run_portfolios(
    extension_features: pd.DataFrame,
    extension_calendar: pd.DatetimeIndex,
    old_primary: pd.DataFrame,
    recent_primary: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    old_curve, old_trades, old_metrics = _simulate_one_period(
        extension_features,
        old_primary,
        extension_calendar,
        "new_old_cycle_2003_2012_held_to_2015",
        *PORTFOLIO_PERIODS["new_old_cycle_2003_2012_held_to_2015"],
    )
    curve_parts.append(old_curve)
    trade_parts.append(old_trades)
    metric_rows.append(old_metrics)
    recent_symbols = set(recent_primary["ts_code"].astype(str))
    recent_daily = _load_recent_prices(
        args.recent_feature_cache,
        args.recent_supplemental_root,
        recent_symbols,
    )
    recent_benchmark = _normalize_benchmark(pd.read_parquet(args.recent_benchmark))
    recent_calendar = pd.DatetimeIndex(sorted(recent_benchmark["date"].unique()))
    for period in list(PORTFOLIO_PERIODS)[1:]:
        curve, trades, period_metrics = _simulate_one_period(
            recent_daily,
            recent_primary,
            recent_calendar,
            period,
            *PORTFOLIO_PERIODS[period],
        )
        curve_parts.append(curve)
        trade_parts.append(trades)
        metric_rows.append(period_metrics)
    return (
        pd.concat(curve_parts, ignore_index=True, sort=False),
        pd.concat(trade_parts, ignore_index=True, sort=False),
        pd.DataFrame(metric_rows),
    )


def _decide(metrics: pd.DataFrame, portfolio: pd.DataFrame) -> dict[str, Any]:
    primary = metrics.loc[
        metrics["structure"].eq(PRIMARY_STRUCTURE)
        & np.isclose(metrics["target_return"], 0.15)
    ].set_index("period")
    old = primary.reindex(["new_old_cycle_2003_2012"]).iloc[0]
    recent = primary.reindex(["exposed_recent_2013_2024"]).iloc[0]
    combined = primary.reindex(["combined_2003_2024"]).iloc[0]

    def quality(row: pd.Series, *, minimum_cohorts: int) -> dict[str, bool]:
        return {
            "minimum_cohorts": bool(row.get("completed_cohorts", 0) >= minimum_cohorts),
            "event_win_rate_at_least_85pct": bool(row.get("event_win_rate", np.nan) >= 0.85),
            "profit_factor_at_least_two": bool(row.get("profit_factor", np.nan) >= 2.0),
            "positive_cohort_share_at_least_80pct": bool(row.get("positive_cohort_share", np.nan) >= 0.80),
            "cohort_bootstrap_lower_positive": bool(row.get("cohort_bootstrap_ci95_low", np.nan) > 0.0),
            "worst_cohort_no_worse_than_minus_5pct": bool(row.get("worst_cohort_return", np.nan) >= -0.05),
            "leave_one_cohort_out_positive": bool(row.get("leave_one_cohort_out_min_mean", np.nan) > 0.0),
        }

    old_checks = quality(old, minimum_cohorts=5)
    recent_checks = quality(recent, minimum_cohorts=5)
    combined_checks = quality(combined, minimum_cohorts=12)
    neighborhood = metrics.loc[
        metrics["period"].eq("combined_2003_2024")
        & np.isclose(metrics["target_return"], 0.15)
    ].copy()
    stable = (
        neighborhood["cohort_bootstrap_ci95_low"].gt(0.0)
        & neighborhood["leave_one_cohort_out_min_mean"].gt(0.0)
    )
    neighborhood_checks = {
        "three_of_four_drawdown_neighbors_robust": bool(stable.sum() >= 3),
        "robust_neighbors": neighborhood.loc[stable, "structure"].tolist(),
    }
    portfolio_checks = {
        "all_periods_present": bool(
            set(PORTFOLIO_PERIODS).issubset(set(portfolio["period"]))
        ),
        "positive_cagr_each": bool(portfolio["cagr"].gt(0.0).all()),
        "maximum_drawdown_no_worse_than_10pct_each": bool(
            portfolio["maximum_drawdown"].ge(-0.10).all()
        ),
        "worst_rolling_24m_no_worse_than_10pct_each": bool(
            portfolio["worst_rolling_24m_return"].ge(-0.10).all()
        ),
    }
    all_passed = (
        all(old_checks.values())
        and all(recent_checks.values())
        and all(combined_checks.values())
        and neighborhood_checks["three_of_four_drawdown_neighbors_robust"]
        and all(portfolio_checks.values())
    )
    if all_passed:
        status = "reasonable_cross_cycle_historical_research_structure"
    elif not old_checks["minimum_cohorts"] or not combined_checks["minimum_cohorts"]:
        status = "promising_but_independent_sample_insufficient"
    else:
        status = "case_corrected_structure_failed_old_cycle_validation"
    return {
        "primary_structure": PRIMARY_STRUCTURE,
        "market_gate": "breadth_positive_share_20d<=20% and median_return_20d<=-10%",
        "confirmation": "range_mid_reclaim",
        "horizon_sessions": 504,
        "target_return": 0.15,
        "old_cycle_checks": old_checks,
        "recent_checks": recent_checks,
        "combined_checks": combined_checks,
        "neighborhood_checks": neighborhood_checks,
        "portfolio_checks": portfolio_checks,
        "status": status,
        "deployment_eligible": False,
        "deployment_blocker": (
            "case-corrected thresholds were proposed after inspecting 2013-2024; "
            "require newly completed live cohorts"
        ),
    }


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
    cohorts: pd.DataFrame,
    portfolio: pd.DataFrame,
    decision: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    primary = metrics.loc[
        metrics["structure"].eq(PRIMARY_STRUCTURE)
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
        "positive_cohort_wilson_low",
        "cohort_equal_mean_return",
        "cohort_bootstrap_ci95_low",
        "cohort_bootstrap_ci95_high",
        "worst_cohort_return",
        "leave_one_cohort_out_min_mean",
    ]
    primary_display = _format_percent(primary, percent_columns)
    neighborhood_display = _format_percent(neighborhood, percent_columns)
    portfolio_display = _format_percent(
        portfolio,
        [
            "total_return",
            "cagr",
            "maximum_drawdown",
            "worst_rolling_24m_return",
            "mean_invested_fraction",
        ],
    )
    primary_cohorts = cohorts.loc[
        cohorts["structure"].eq(PRIMARY_STRUCTURE)
        & np.isclose(cohorts["target_return"], 0.15)
    ].copy()
    primary_cohorts = _format_percent(
        primary_cohorts, ["cohort_return", "worst_event_return"]
    )
    conclusion = {
        "reasonable_cross_cycle_historical_research_structure": (
            "case 修正版通过旧周期、近期、合并批次、阈值邻域和日度组合检查，"
            "可作为跨周期历史研究结构；仍不是可部署策略。"
        ),
        "promising_but_independent_sample_insufficient": (
            "case 修正版收益方向较好，但新增或合并独立恐慌月仍不足。"
        ),
        "case_corrected_structure_failed_old_cycle_validation": (
            "case 修正版未通过旧周期或合并稳健性，不应继续使用。"
        ),
    }[decision["status"]]
    path.write_text(
        f"""# 月线带血筹旧周期扩展验证

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

{conclusion}

主结构：市场上涨家数不高于 20%、横截面 20 日中位跌幅至少 10%，个股从前高跌幅至少 60%，等待价格重新站回因果底部区间中轴；每锚净值 2.5%，最多 20 锚，504 日上限，15% 止盈，20bp 成本。

2000—2012 是在 case 修正版冻结后补取的旧周期；2013—2015 的扩展数据只用于闭合旧锚路径。由于修正规则来自已见的 2013—2024 case，`deployment_eligible` 仍为 `{str(decision['deployment_eligible']).lower()}`。

## 主结构跨周期结果

{primary_display.to_markdown(index=False) if not primary_display.empty else '_没有完成事件_'}

## 50%/60%/70%/80% 前高回撤邻域

{neighborhood_display.to_markdown(index=False) if not neighborhood_display.empty else '_没有邻域结果_'}

## 独立恐慌月

{primary_cohorts.to_markdown(index=False) if not primary_cohorts.empty else '_没有完成恐慌月_'}

## 日度盯市组合

{portfolio_display.to_markdown(index=False) if not portfolio_display.empty else '_没有组合结果_'}

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
    benchmark = _load_extension_benchmark(
        args.extension_data_dir, args.recent_benchmark
    )
    benchmark = benchmark.loc[benchmark["date"].le("2015-12-31")].copy()
    calendar = pd.DatetimeIndex(sorted(benchmark["date"].unique()))
    features, feature_metadata = _load_or_build_extension_features(
        args.extension_data_dir
    )
    old_events, old_metadata = _build_old_profit_events(
        features,
        benchmark,
        calendar,
        args.extension_data_dir,
    )
    recent_events = _recent_case_events(args.recent_gated_events)
    old_candidate = _materialize_candidate_events(old_events)
    recent_candidate = _materialize_candidate_events(recent_events)
    candidate = pd.concat(
        [old_candidate, recent_candidate], ignore_index=True, sort=False
    )
    metrics, cohorts = _summarize_candidate_events(candidate)
    old_primary = old_candidate.loc[
        old_candidate["structure"].eq(PRIMARY_STRUCTURE)
        & np.isclose(old_candidate["target_return"], 0.15)
    ].copy()
    recent_primary = recent_candidate.loc[
        recent_candidate["structure"].eq(PRIMARY_STRUCTURE)
        & np.isclose(recent_candidate["target_return"], 0.15)
    ].copy()
    portfolio_curves, portfolio_trades, portfolio_metrics = _run_portfolios(
        features,
        calendar,
        old_primary,
        recent_primary,
        args,
    )
    decision = _decide(metrics, portfolio_metrics)
    metadata = {
        "analysis_cutoff": pd.Timestamp("2026-07-31"),
        "new_signal_period_end": pd.Timestamp("2012-12-31"),
        "extension_path_end": pd.Timestamp("2015-12-31"),
        "primary_structure": PRIMARY_STRUCTURE,
        "drawdown_neighborhood": DRAWDOWN_NEIGHBORHOOD,
        "targets": TARGETS,
        "features": feature_metadata,
        "old_cycle": old_metadata,
        "old_candidate_rows": int(len(old_candidate)),
        "recent_candidate_rows": int(len(recent_candidate)),
        "combined_candidate_rows": int(len(candidate)),
    }
    old_events.to_parquet(
        args.output_dir / "old_cycle_profit_events.parquet",
        index=False,
        compression="zstd",
    )
    candidate.to_parquet(
        args.output_dir / "combined_candidate_events.parquet",
        index=False,
        compression="zstd",
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    cohorts.to_csv(args.output_dir / "cohort_metrics.csv", index=False)
    portfolio_curves.to_parquet(
        args.output_dir / "portfolio_curves.parquet",
        index=False,
        compression="zstd",
    )
    portfolio_trades.to_parquet(
        args.output_dir / "portfolio_trades.parquet",
        index=False,
        compression="zstd",
    )
    portfolio_metrics.to_csv(args.output_dir / "portfolio_metrics.csv", index=False)
    _write_json(args.output_dir / "decision.json", decision)
    _write_json(args.output_dir / "metadata.json", metadata)
    _write_report(
        args.output_dir / "report.md",
        metrics,
        cohorts,
        portfolio_metrics,
        decision,
        metadata,
    )
    print(f"extension status: {decision['status']}", flush=True)
    print(f"report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
