#!/usr/bin/env python
"""Select BYD daily-T parameters before 2026, then audit the frozen choice."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

from quant.application.workspaces.byd import load_byd_daily_frame
from quant.data.byd_intraday import read_byd_training_frame
from quant.data.byd_intraday import TRAINING_FILENAME
from quant.infrastructure.publication import PublicationStore, publication_context
from quant.research.byd_daily_t_plan import (
    POSITIVE_T_CONFIG,
    build_daily_t_plan,
    daily_from_intraday,
    daily_t_features,
    evaluate_positive_t,
    sideways_feature_mask,
    simulate_daily_t,
    walk_forward_scores,
)
from quant.routine.paths import PROJECT_ROOT, load_project_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/byd_002594/t_optimization_20261101")
    args = parser.parse_args()
    load_project_env()
    # Match the input actually served by the application, including when a
    # committed generation is newer than the legacy root cache.
    view = PublicationStore(PROJECT_ROOT, (f"data/cache/{TRAINING_FILENAME}",)).view()
    with publication_context(view):
        bars = read_byd_training_frame()
    daily = load_byd_daily_frame()
    selection_bars = bars.loc[bars["datetime"] < "2026-01-01"]
    selection_daily = daily_from_intraday(selection_bars)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    configs = [replace(POSITIVE_T_CONFIG, entry_deviation=entry, target_deviation=target,
                       stop_loss=stop, score_threshold=threshold)
               for entry, target, stop, threshold in product(
        [.006, .008, .010], [.008, .010, .012], [None, .02], [10., 20.]
    )]
    configs.extend(replace(POSITIVE_T_CONFIG, model_max_depth=depth, score_threshold=threshold)
                   for depth, threshold in product([1, 2], [10., 20.]))
    for config in configs:
        result = evaluate_positive_t(selection_bars, selection_daily, config)
        metrics = result["results"][:2]
        eligible = bool(result["selection_passed"] and all(
            item["net_pnl"] > 0 for item in result["stress_results"][:2]
        ))
        risk_score = min(item["net_pnl"] / max(item["cycle_drawdown_yuan"], 1.) for item in metrics)
        candidates.append({"config": asdict(config), "eligible": eligible,
                           "selection_risk_score": risk_score, "segments": metrics,
                           "stress_segments": result["stress_results"][:2]})
        print(f"candidate {len(candidates)}/{len(configs)} eligible={eligible} risk_score={risk_score:.3f}", flush=True)
    ranked = sorted(candidates, key=lambda item: (
        item["eligible"], item["selection_risk_score"],
        item["segments"][1]["net_pnl"],
    ), reverse=True)
    winner = ranked[0]
    config = replace(POSITIVE_T_CONFIG, **winner["config"])
    audit = evaluate_positive_t(bars, daily, config)
    frozen_reference = evaluate_positive_t(bars, daily, replace(
        POSITIVE_T_CONFIG, entry_deviation=.008, target_deviation=.010,
        stop_loss=None, score_threshold=10.,
    ))
    outcomes = simulate_daily_t(bars, config)
    features = daily_t_features(daily_from_intraday(bars)).shift(1)
    samples = outcomes.join(features).dropna(subset=list(features.columns))
    scored = samples.join(walk_forward_scores(samples, config), how="inner")
    selected = scored.loc[scored["score"].gt(config.score_threshold) & sideways_feature_mask(scored)]
    selected.to_csv(args.output_dir / "selected_cycles.csv", index_label="date")
    report = {
        "data": {"intraday_start": str(bars.datetime.min()), "intraday_end": str(bars.datetime.max()),
                 "daily_end": str(daily.date.max()), "sessions": bars.date.nunique(),
                 "input_generation": view.generation,
                 "fingerprint": bars.attrs.get("intraday_training_fingerprint")},
        "selection_cutoff": "2025-12-31", "selection_used_2026": False,
        "monitoring_note": "2026 has already been inspected in earlier research; no untouched holdout claim.",
        "selected_config": asdict(config), "eligible": winner["eligible"],
        "promotion_approved": bool(winner["eligible"] and audit["passed"]),
        "production_config": asdict(POSITIVE_T_CONFIG),
        "evaluation": audit, "corrected_reference": frozen_reference,
        "ranking": "Minimum development/validation net-PnL to settled-cycle drawdown ratio, after fixed quality gates and double-slippage checks.",
        "limitations": ["500 settled shares and buy-side cash must be reserved; no account-wide cash ledger",
                        "Cycle drawdown excludes base holding and intraday mark-to-market losses",
                        "Constant current fee assumptions on qfq history; no corporate-action cash/share ledger",
                        "Five-minute touches do not guarantee executable queue fills; no daily-limit order-book model"],
    }
    (args.output_dir / "candidates.json").write_text(json.dumps(ranked, ensure_ascii=False, indent=2))
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    # The production default is updated only after reviewing this frozen audit.
    plan = build_daily_t_plan(daily, bars, shares=10000)
    (args.output_dir / "production_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print(json.dumps({"selected_config": asdict(config), "evaluation": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
