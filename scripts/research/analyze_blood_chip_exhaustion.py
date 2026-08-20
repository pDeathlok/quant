#!/usr/bin/env python3
"""Compare static volatility caps with causal shock-to-confirmation exhaustion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from quant.research.blood_chip import (
    BloodChipBacktestConfig,
    BloodChipSignalConfig,
    add_blood_chip_path_features,
    analyze_blood_chip_cases,
    generate_blood_chip_signals,
    load_benchmark,
    run_blood_chip_backtest,
    summarize_blood_chip_result,
)


DEVELOPMENT = ("development_2014_2019", "2014-01-01", "2019-12-31")
ITERATION = ("iteration_2020_2022", "2020-01-01", "2022-12-30")
SEEN_HOLDOUT = ("seen_holdout_2023_2026", "2023-01-03", "2026-02-06")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether volatility and sell pressure decay after a blood-chip shock."
    )
    parser.add_argument(
        "--features", default="data/research/blood_chip/features.parquet"
    )
    parser.add_argument(
        "--benchmark", default="data/raw/index_000300.SH.parquet"
    )
    parser.add_argument(
        "--output-dir", default="reports/research/blood_chip_exhaustion"
    )
    parser.add_argument("--reuse-signals", action="store_true")
    parser.add_argument("--include-seen-holdout", action="store_true")
    parser.add_argument("--seen-holdout-only", action="store_true")
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _positive_year_share(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    years = pd.to_datetime(trades["entry_date"]).dt.year
    annual = pd.to_numeric(trades["net_return"], errors="coerce").groupby(years).mean()
    return float((annual > 0).mean()) if not annual.empty else np.nan


def _development_thresholds(signals: pd.DataFrame) -> dict[str, float]:
    entry_dates = pd.to_datetime(signals["entry_date"])
    development = signals.loc[
        entry_dates.between(DEVELOPMENT[1], DEVELOPMENT[2], inclusive="both")
    ]
    quantiles = {
        "shock_volatility_q25": (
            "shock_realized_volatility_5d",
            0.25,
        ),
        "shock_volatility_expansion_q50": (
            "shock_volatility_expansion_ratio",
            0.50,
        ),
        "shock_volatility_expansion_q75": (
            "shock_volatility_expansion_ratio",
            0.75,
        ),
        "shock_amount_expansion_q50": ("shock_amount_expansion_ratio", 0.50),
        "confirmation_amount_vs_prior_q25": (
            "confirmation_amount_vs_prior_ratio",
            0.25,
        ),
        "confirmation_amount_vs_prior_q50": (
            "confirmation_amount_vs_prior_ratio",
            0.50,
        ),
        "volatility_decay_q50": ("volatility_decay_ratio", 0.50),
        "volatility_decay_q75": ("volatility_decay_ratio", 0.75),
        "range_decay_q50": ("range_decay_ratio", 0.50),
        "amount_decay_q75": ("amount_decay_ratio", 0.75),
        "downside_amount_decay_q50": ("downside_amount_decay_ratio", 0.50),
        "downside_amount_decay_q75": ("downside_amount_decay_ratio", 0.75),
        "sell_pressure_decay_q50": ("sell_pressure_decay_ratio", 0.50),
        "sell_pressure_decay_q75": ("sell_pressure_decay_ratio", 0.75),
        "confirmation_downside_share_q50": (
            "confirmation_downside_amount_share",
            0.50,
        ),
    }
    return {
        name: float(pd.to_numeric(development[column], errors="coerce").quantile(q))
        for name, (column, q) in quantiles.items()
    }


def _candidate_filters(
    thresholds: dict[str, float],
) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    static_control = lambda frame: frame["volatility_60d"].le(0.55)
    controlled_rebound = lambda frame: (
        static_control(frame) & frame["rebound_from_event_low"].le(0.15)
    )
    return {
        "static_quality_control": static_control,
        "rank_low_chronic_vol": controlled_rebound,
        "rank_low_vol_no_cap": lambda frame: frame["rebound_from_event_low"].le(0.15),
        "rank_acute_stable_no_cap": lambda frame: frame[
            "rebound_from_event_low"
        ].le(0.15),
    }


def _apply_ranking(rule: str, signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    if rule in {"rank_low_chronic_vol", "rank_low_vol_no_cap"}:
        out["signal_score"] = -pd.to_numeric(out["volatility_60d"], errors="coerce")
    elif rule == "rank_acute_stable_no_cap":
        groups = out.groupby("entry_date", observed=True, sort=False)
        low_volatility = 1.0 - groups["volatility_60d"].rank(pct=True)
        shock_expansion = groups["shock_volatility_expansion_ratio"].rank(pct=True)
        rebound_distance = (out["rebound_from_event_low"] - 0.10).abs()
        controlled_rebound = 1.0 - rebound_distance.groupby(
            out["entry_date"], observed=True, sort=False
        ).rank(pct=True)
        retained_turnover = groups["confirmation_amount_vs_prior_ratio"].rank(pct=True)
        out["signal_score"] = (
            0.40 * low_volatility
            + 0.30 * shock_expansion
            + 0.20 * controlled_rebound
            + 0.10 * retained_turnover
        )
    return out


def _yearly_cases(trades: pd.DataFrame, rule: str, period: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["entry_year"] = pd.to_datetime(frame["entry_date"]).dt.year
    yearly = frame.groupby("entry_year", observed=True).agg(
        trades=("net_return", "size"),
        win_rate=("net_return", lambda values: float((values > 0).mean())),
        average_net_return=("net_return", "mean"),
        median_net_return=("net_return", "median"),
        stop_rate=("exit_reason", lambda values: float((values == "stop_loss").mean())),
    )
    yearly = yearly.reset_index()
    yearly.insert(0, "period", period)
    yearly.insert(0, "rule", rule)
    return yearly


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading features: {args.features}", flush=True)
    features = pd.read_parquet(args.features)
    benchmark = load_benchmark(args.benchmark, "20130104", "20260807")
    signal_config = BloodChipSignalConfig(
        maximum_return_120d=0.50,
        minimum_market_return_60d=-0.15,
    )
    signal_path = output_dir / "path_signals.parquet"
    if args.reuse_signals and signal_path.exists():
        print(f"loading cached path signals: {signal_path}", flush=True)
        signals = pd.read_parquet(signal_path)
    else:
        print("generating path-base signals without a static volatility cap", flush=True)
        signals = generate_blood_chip_signals(features, signal_config)
        signals = add_blood_chip_path_features(features, signals)
        signals.to_parquet(signal_path, index=False, compression="zstd")
    thresholds = _development_thresholds(signals)
    _write_json(
        output_dir / "development_thresholds.json",
        {
            "thresholds": thresholds,
            "derivation": "quantiles of all development-period signals, not outcomes",
            "signal_config": asdict(signal_config),
        },
    )

    periods = [SEEN_HOLDOUT] if args.seen_holdout_only else [DEVELOPMENT, ITERATION]
    if args.include_seen_holdout and not args.seen_holdout_only:
        periods.append(SEEN_HOLDOUT)
    backtest_config = BloodChipBacktestConfig()
    metric_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    filters = _candidate_filters(thresholds)
    for rule, filter_function in filters.items():
        mask = filter_function(signals).fillna(False)
        candidate_signals = _apply_ranking(rule, signals.loc[mask])
        print(f"{rule}: {len(candidate_signals):,} signals", flush=True)
        for period, start, end in periods:
            result = run_blood_chip_backtest(
                features,
                candidate_signals,
                backtest_config,
                start,
                end,
            )
            metrics = summarize_blood_chip_result(result, benchmark)
            metrics.update(
                {
                    "rule": rule,
                    "period": period,
                    "entry_start": start,
                    "entry_end": end,
                    "positive_year_share": _positive_year_share(result.trades),
                    "signals_all_dates": int(len(candidate_signals)),
                }
            )
            metric_rows.append(metrics)
            trades = result.trades.copy()
            trades["rule"] = rule
            trades["period"] = period
            trades.to_parquet(
                output_dir / f"{rule}_{period}_trades.parquet",
                index=False,
                compression="zstd",
            )
            analyze_blood_chip_cases(trades).assign(
                rule=rule,
                period=period,
            ).to_csv(output_dir / f"{rule}_{period}_cases.csv", index=False)
            yearly_frames.append(_yearly_cases(trades, rule, period))
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(output_dir / "candidate_metrics.csv", index=False)
    _write_json(output_dir / "candidate_metrics.json", {"metrics": metric_rows})
    pd.concat(yearly_frames, ignore_index=True, sort=False).to_csv(
        output_dir / "yearly_cases.csv", index=False
    )
    print(
        metric_frame[
            [
                "rule",
                "period",
                "trades",
                "win_rate",
                "average_net_return",
                "profit_factor",
                "total_return",
                "maximum_drawdown",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
