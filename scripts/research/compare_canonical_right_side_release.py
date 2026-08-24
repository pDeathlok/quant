#!/usr/bin/env python
"""Compare the canonical right-side ranker with the preserved production model."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
    RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION,
)
from quant.research.right_side_paired_comparison import paired_model_comparisons


DEFAULT_NEW = (
    PROJECT_ROOT
    / "reports/research/right_side_unified_canonical_v5_rule113/test_predictions.parquet"
)
DEFAULT_OLD = (
    PROJECT_ROOT
    / "reports/research/right_side_unified_v2_118/test_predictions.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "reports/research/right_side_unified_canonical_v5_rule113"
)


def compare(args: argparse.Namespace) -> dict[str, object]:
    keys = ["symbol", "date", "fold", "entry_mode", "horizon", "label"]
    new = pd.read_parquet(
        args.new_predictions,
        columns=[
            *keys,
            "good_path5",
            "terminal_return",
            "pred_unified_long_task_deep",
        ],
    )
    old = pd.read_parquet(
        args.old_predictions,
        columns=[*keys, "pred_unified_long_task_deep_rule105"],
    ).rename(
        columns={"pred_unified_long_task_deep_rule105": "pred_current_production"}
    )
    new = new[new["fold"].isin(["A", "B"])].copy()
    old = old[old["fold"].isin(["A", "B"])].copy()
    paired = new.merge(old, on=keys, how="inner", validate="one_to_one")
    if len(paired) != len(new) or len(paired) != len(old):
        raise RuntimeError(
            "canonical/production prediction coverage is not exact; "
            f"new={len(new)} old={len(old)} paired={len(paired)}"
        )
    comparison = paired_model_comparisons(
        paired,
        candidate_experiments=("unified_long_task_deep",),
        baseline_experiment="current_production",
        bootstrap_iterations=args.bootstrap_iterations,
        top_fraction=0.10,
        daily_top_k=10,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_root / "canonical_vs_production_ranking_ab.csv"
    atomic_write_csv(comparison, comparison_path, index=False)
    ordered = comparison.sort_values("fold")
    required_folds = {"A", "B"}
    observed_folds = set(ordered["fold"].astype(str))
    passed = bool(
        observed_folds == required_folds
        and ordered["status"].isin(["ok", "exact_only"]).all()
        and ordered["delta_pr_auc"].gt(0).all()
        and ordered["delta_top_lift"].gt(0).all()
    )
    decision = {
        "schema_version": "right-side-production-replacement-decision-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision_metric": "ranking_only_no_returns",
        "folds_read": ["A", "B"],
        "fold_c_read": False,
        "selected_research_candidate": "unified_long_task_deep",
        "selected_score_column": "pred_unified_long_task_deep",
        "canonical_factor_gate": {
            "passed": passed,
            "feature_schema_version": RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION,
            "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
            "exact_overlap_coverage": 1.0,
            "delta_pr_auc_by_fold": {
                str(row.fold): float(row.delta_pr_auc)
                for row in ordered.itertuples()
            },
            "delta_top_lift_by_fold": {
                str(row.fold): float(row.delta_top_lift)
                for row in ordered.itertuples()
            },
            "source": comparison_path.name,
        },
        "shadow_candidate": passed,
        "replace_online": passed,
        "decision_reason": (
            "canonical_model_ranked_better_than_current_production_in_both_A_and_B"
            if passed
            else "canonical_model_did_not_rank_better_in_every_required_fold"
        ),
        "rollback_artifact_policy": "preserve_old_production_artifact_unchanged",
    }
    decision_path = args.output_root / "production_replacement_decision_ab.json"
    atomic_write_json(decision, decision_path)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-predictions", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--old-predictions", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    print(compare(args))


if __name__ == "__main__":
    main()
