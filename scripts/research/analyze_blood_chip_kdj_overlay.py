#!/usr/bin/env python3
"""Evaluate causal multi-timeframe KDJ as an overlay to blood-chip signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.research.blood_chip import (
    BloodChipBacktestConfig,
    load_benchmark,
    summarize_blood_chip_result,
)
from quant.research.blood_chip_kdj import attach_blood_chip_kdj_path, apply_kdj_overlay
from quant.research.blood_chip_scale_in import (
    DEFAULT_SCALE_IN_POLICIES,
    run_blood_chip_scale_in_backtest,
)


PERIODS = (
    ("development_2014_2019", "2014-01-01", "2019-12-31"),
    ("validation_2020_2022", "2020-01-01", "2022-12-30"),
    ("seen_diagnostic_2023_2026", "2023-01-03", "2026-02-06"),
    ("recent_audit_2026", "2026-02-09", "2026-08-07"),
)
MODES = (
    "baseline_low_vol",
    "kdj_soft_priority",
    "daily_weekly_only",
    "triple_only",
)
FEATURE_COLUMNS = (
    "ts_code",
    "trade_date",
    "date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_chg",
    "vol",
    "amount",
    "adjustment_factor",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "residual_return_3d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/research/blood_chip/features.parquet")
    parser.add_argument(
        "--signals",
        default="reports/research/blood_chip_exhaustion/path_signals.parquet",
    )
    parser.add_argument("--benchmark", default="data/raw/index_000300.SH.parquet")
    parser.add_argument(
        "--output-dir", default="reports/research/blood_chip_kdj_overlay"
    )
    parser.add_argument("--reuse-enriched-signals", action="store_true")
    return parser.parse_args()


def _capital_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "capital_weighted_trade_return": np.nan,
            "capital_profit_factor": np.nan,
        }
    pnl = (
        pd.to_numeric(trades["exit_value"], errors="coerce")
        - pd.to_numeric(trades["fees"], errors="coerce")
        - pd.to_numeric(trades["entry_value"], errors="coerce")
    )
    invested = pd.to_numeric(trades["invested_value"], errors="coerce")
    loss = float(-pnl.loc[pnl <= 0].sum())
    return {
        "capital_weighted_trade_return": float(pnl.sum() / invested.sum()),
        "capital_profit_factor": float(pnl.loc[pnl > 0].sum() / loss) if loss > 0 else np.nan,
    }


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _decision(metrics: pd.DataFrame) -> dict[str, object]:
    indexed = metrics.set_index(["mode", "period"])
    baseline_dev = indexed.loc[("baseline_low_vol", "development_2014_2019")]
    soft_dev = indexed.loc[("kdj_soft_priority", "development_2014_2019")]
    baseline_val = indexed.loc[("baseline_low_vol", "validation_2020_2022")]
    soft_val = indexed.loc[("kdj_soft_priority", "validation_2020_2022")]
    checks = {
        "development_capital_pf_not_worse": soft_dev["capital_profit_factor"] >= baseline_dev["capital_profit_factor"] * 0.95,
        "validation_capital_pf_improves": soft_val["capital_profit_factor"] > baseline_val["capital_profit_factor"],
        "validation_total_return_improves": soft_val["total_return"] > baseline_val["total_return"],
        "validation_drawdown_not_worse_3pp": soft_val["maximum_drawdown"] >= baseline_val["maximum_drawdown"] - 0.03,
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "selected_mode": "kdj_soft_priority" if passed else "baseline_low_vol",
        "deployment": "ranking_overlay" if passed else "annotation_only",
        "checks": checks,
        "selection_uses_periods": ["development_2014_2019", "validation_2020_2022"],
        "seen_diagnostic_excluded_from_selection": True,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enriched_path = output / "signals_with_kdj.parquet"
    print("loading cached blood-chip features", flush=True)
    features = pd.read_parquet(args.features, columns=list(FEATURE_COLUMNS))
    signals = pd.read_parquet(args.signals)
    if args.reuse_enriched_signals and enriched_path.exists():
        signals = pd.read_parquet(enriched_path)
    else:
        print("attaching causal completed-period KDJ at confirmation and shock", flush=True)
        signals = attach_blood_chip_kdj_path(features, signals)
        for name in ("daily_j", "weekly_j", "monthly_j"):
            signals[f"kdj_{name}"] = signals[f"shock_kdj_{name}"]
        signals["kdj_negative_count"] = signals["shock_kdj_negative_count"]
        signals["kdj_state"] = signals["shock_kdj_state"]
        signals.to_parquet(enriched_path, index=False, compression="zstd")
    signals = signals.loc[
        pd.to_numeric(signals["rebound_from_event_low"], errors="coerce").le(0.15)
    ].copy()
    benchmark = load_benchmark(args.benchmark, "20130104", "20260807")
    config = BloodChipBacktestConfig(maximum_positions=10)
    policy = DEFAULT_SCALE_IN_POLICIES["increasing_survival"]
    rows: list[dict[str, object]] = []
    trades: list[pd.DataFrame] = []
    for mode in MODES:
        candidates = apply_kdj_overlay(signals, mode)
        print(f"{mode}: {len(candidates):,} candidate signals", flush=True)
        for period, start, end in PERIODS:
            result = run_blood_chip_scale_in_backtest(
                features,
                candidates.dropna(subset=["entry_date", "entry_open"]),
                config,
                policy,
                start,
                end,
            )
            metrics = summarize_blood_chip_result(result, benchmark)
            metrics.update(_capital_metrics(result.trades))
            period_signals = candidates.loc[
                pd.to_datetime(candidates["entry_date"], errors="coerce").between(
                    start, end, inclusive="both"
                )
            ]
            metrics.update(
                {
                    "mode": mode,
                    "period": period,
                    "candidate_signals": int(len(period_signals)),
                    "triple_signal_share": float(period_signals["kdj_state"].eq("triple_oversold").mean()) if len(period_signals) else np.nan,
                    "daily_weekly_signal_share": float(period_signals["kdj_state"].isin(["triple_oversold", "daily_weekly_oversold"]).mean()) if len(period_signals) else np.nan,
                }
            )
            rows.append(metrics)
            trade = result.trades.copy()
            trade["mode"] = mode
            trade["period"] = period
            trades.append(trade)
    metric_frame = pd.DataFrame(rows)
    decision = _decision(metric_frame)
    metric_frame.to_csv(output / "metrics.csv", index=False)
    pd.concat(trades, ignore_index=True, sort=False).to_parquet(
        output / "trades.parquet", index=False, compression="zstd"
    )
    (output / "decision.json").write_text(
        json.dumps(_json_value(decision), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    columns = [
        "mode",
        "period",
        "candidate_signals",
        "trades",
        "win_rate",
        "capital_weighted_trade_return",
        "capital_profit_factor",
        "total_return",
        "maximum_drawdown",
    ]
    print(metric_frame[columns].to_string(index=False), flush=True)
    print(json.dumps(_json_value(decision), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
