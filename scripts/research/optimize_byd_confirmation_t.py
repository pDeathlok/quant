#!/usr/bin/env python
"""Frozen-grid BYD T confirmation research; 2026 never ranks parameters."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data.byd_intraday import TRAINING_FILENAME, read_byd_training_frame
from quant.infrastructure.publication import PublicationStore, publication_context
from quant.research.byd_t_backtest import ChinaAStockFees, prepare_byd_intraday_bars
from quant.research.byd_t_confirmation import (
    ConfirmationTConfig,
    confirmation_metrics,
    confirmation_sessions,
    selection_qualified,
    simulate_confirmation_t,
)
from quant.routine.paths import PROJECT_ROOT, load_project_env


def monthly_profit_interval(trades: pd.DataFrame, first: str, last: str) -> list[float]:
    """Conditional 90% monthly-block interval; not a selection-adjusted CI."""
    dates = pd.to_datetime(trades["date"])
    selected = trades.loc[dates.between(pd.Timestamp(first), pd.Timestamp(last))]
    monthly = selected.groupby(pd.to_datetime(selected["date"]).dt.to_period("M"))["pnl"].sum()
    values = monthly.reindex(pd.period_range(first, last, freq="M"), fill_value=0).to_numpy(dtype=float)
    rng = np.random.default_rng(20261101)
    bootstraps = rng.choice(values, size=(1000, len(values)), replace=True).sum(axis=1)
    return np.quantile(bootstraps, [.05, .95]).round(2).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/byd_002594/t_confirmation_20261101")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_project_env()
    view = PublicationStore(PROJECT_ROOT, (f"data/cache/{TRAINING_FILENAME}",)).view()
    with publication_context(view):
        bars = read_byd_training_frame()
    sessions = confirmation_sessions(prepare_byd_intraday_bars(bars))
    selection = [day for day in sessions if pd.Timestamp("2022-01-01") <= day.date < pd.Timestamp("2026-01-01")]
    configs = [ConfirmationTConfig(
        direction=direction, entry_mode=mode, entry_atr=entry, target_atr=target, stop_atr=stop,
        latest_entry_minute=latest, regime=regime,
    ) for direction, mode, entry, target, stop, latest, regime in product(
        ["positive", "reverse"], ["limit", "confirmed"], [.4, .6], [.4, .6], [.3, .5], [660, 840], ["balanced", "bounded"],
    )]
    records = []
    stress_fees = replace(ChinaAStockFees(), slippage_rate=.0002)
    for index, config in enumerate(configs):
        result = simulate_confirmation_t(selection, config)
        development = confirmation_metrics(result, "2023-01-01", "2023-12-31")
        validation = confirmation_metrics(result, "2024-01-01", "2025-12-31")
        annual = [confirmation_metrics(result, f"{year}-01-01", f"{year}-12-31") for year in (2024, 2025)]
        interval = monthly_profit_interval(result.trades, "2024-01-01", "2025-12-31")
        valid_trades = result.trades.loc[pd.to_datetime(result.trades["date"]).between("2024-01-01", "2025-12-31")]
        best_trade = float(valid_trades["pnl"].max()) if len(valid_trades) else 0.
        basic = selection_qualified(development, validation, annual)
        stress_metrics = None
        latency_metrics = None
        if basic:
            stressed = simulate_confirmation_t(selection, config, stress_fees)
            stress_metrics = [confirmation_metrics(stressed, "2023-01-01", "2023-12-31"),
                              confirmation_metrics(stressed, "2024-01-01", "2025-12-31")]
            if config.entry_mode == "confirmed":
                delayed = simulate_confirmation_t(selection, replace(config, confirmation_delay_bars=2))
                latency_metrics = confirmation_metrics(delayed, "2024-01-01", "2025-12-31")
        eligible = bool(basic and interval[0] > 0 and validation["net_pnl"] - best_trade > 0
                        and stress_metrics and all(item["net_pnl"] > 0 for item in stress_metrics)
                        and (config.entry_mode == "limit" or latency_metrics and latency_metrics["net_pnl"] > 0))
        records.append({
            "config": asdict(config), "basic_qualified": basic, "eligible_before_neighbors": eligible,
            "development": development, "validation": validation, "annual_validation": annual,
            "monthly_profit_interval_90": interval, "validation_without_best_trade": validation["net_pnl"] - best_trade,
            "stress_selection": stress_metrics, "latency_validation": latency_metrics,
            "risk_score": min(development["net_pnl"] / max(development["mtm_drawdown_yuan"], 1),
                              validation["net_pnl"] / max(validation["mtm_drawdown_yuan"], 1)),
        })
        print(f"candidate {index+1}/{len(configs)} basic={basic} eligible={eligible}", flush=True)
    dimensions = ["entry_atr", "target_atr", "stop_atr", "latest_entry_minute", "regime"]
    for record in records:
        neighbors = [other for other in records if other["config"]["direction"] == record["config"]["direction"]
                     and other["config"]["entry_mode"] == record["config"]["entry_mode"]
                     and sum(other["config"][key] != record["config"][key] for key in dimensions) == 1]
        positive_neighbors = sum(other["development"]["net_pnl"] > 0 and other["validation"]["net_pnl"] > 0 for other in neighbors)
        record["neighbor_positive_rate"] = positive_neighbors / len(neighbors) if neighbors else 0.
        record["eligible"] = bool(record["eligible_before_neighbors"] and record["neighbor_positive_rate"] >= .60)
    def rank(item: dict) -> tuple:
        coverage = min(item["development"]["cycles"] / 40, item["validation"]["cycles"] / 25, 1.)
        return item["eligible"], item["basic_qualified"], coverage, item["risk_score"]
    ranked = sorted(records, key=rank, reverse=True)
    frozen = ranked[0]
    config = ConfirmationTConfig(**frozen["config"])
    # 2026 outcomes are evaluated only after the pre-2026 grid and ranking freeze.
    complete = [day for day in sessions if day.date >= pd.Timestamp("2022-01-01")]
    audit = simulate_confirmation_t(complete, config)
    stress_audit = simulate_confirmation_t(complete, config, stress_fees)
    monitoring = confirmation_metrics(audit, "2026-01-01", "2026-12-31")
    monitoring_stress = confirmation_metrics(stress_audit, "2026-01-01", "2026-12-31")
    factor = monitoring["profit_factor"]
    monitoring_passed = (monitoring["cycles"] >= 10 and monitoring["win_rate"] >= .55
                         and monitoring["net_pnl"] > 0 and (factor is None or factor >= 1.1)
                         and monitoring_stress["net_pnl"] > 0)
    audit.trades.to_csv(args.output_dir / "frozen_trades.csv", index=False)
    audit.equity.to_parquet(args.output_dir / "frozen_equity.parquet", index=False)
    annual = [{"year": year, **confirmation_metrics(audit, f"{year}-01-01", f"{year}-12-31")} for year in range(2022, 2027)]
    report = {
        "data": {"input_generation": view.generation, "intraday_end": str(bars.datetime.max()),
                 "fingerprint": bars.attrs.get("intraday_training_fingerprint")},
        "candidate_count": len(records), "basic_qualified_count": sum(item["basic_qualified"] for item in records),
        "qualified_count": sum(item["eligible"] for item in records),
        "selection_used_2026": False, "frozen_candidate": frozen,
        "monitoring": monitoring, "monitoring_stress": monitoring_stress, "annual": annual,
        "monitoring_passed": bool(monitoring_passed),
        "research_evidence_passed": bool(frozen["eligible"] and monitoring_passed),
        "execution_enabled": False,
        "decision": "Research requires a fresh forward paper-trading sample and current intraday confirmation; no automatic entry is authorized.",
        "limitations": ["2026 was previously inspected, and is monitoring rather than untouched test data",
                        "Monthly interval is conditional and not adjusted for repeated candidate searches",
                        "Marked drawdown uses five-minute closes, excludes base holding and within-bar extremes",
                        "Fees are constant research assumptions on qfq bars; no queue or daily-limit fill reconstruction",
                        "Risk budget is a sizing target, not a guaranteed maximum gap loss"],
    }
    report["family_diagnostics"] = [
        next(item for item in ranked if item["config"]["direction"] == direction
             and item["config"]["entry_mode"] == mode)
        for direction, mode in product(["positive", "reverse"], ["limit", "confirmed"])
    ]
    (args.output_dir / "candidates.json").write_text(json.dumps(ranked, ensure_ascii=False, indent=2))
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
