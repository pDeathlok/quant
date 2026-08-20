from __future__ import annotations

import pandas as pd
import pytest

from quant.research.right_side_legacy_ranking_comparison import (
    compare_candidate_with_legacy_artifact,
)


def _candidate() -> pd.DataFrame:
    rows = []
    for fold, year in (("A", 2024), ("B", 2025)):
        for index in range(12):
            rows.append(
                {
                    "symbol": f"S{index}",
                    "date": pd.Timestamp(year, index // 4 + 1, index % 4 + 1),
                    "fold": fold,
                    "entry_mode": "next_close",
                    "horizon": 5,
                    "label": "good_path5",
                    "good_path5": index % 2,
                    "pred_unified_long_task_deep": 0.9 if index % 2 else 0.1,
                }
            )
    return pd.DataFrame(rows)


def _legacy() -> pd.DataFrame:
    candidate = _candidate().iloc[:-2].copy()
    candidate["entry_mode"] = "next_open"
    candidate["good_path5"] = [index % 2 for index in range(len(candidate))]
    candidate["legacy_quality_score"] = [
        0.8 if value == 0 else 0.2 for value in candidate["good_path5"]
    ]
    candidate["legacy_signal_timing_match_any"] = [index % 3 != 0 for index in range(len(candidate))]
    candidate["legacy_temporal_status"] = "seen_artifact_period"
    return candidate.drop(columns=["label", "pred_unified_long_task_deep"])


def test_legacy_comparison_is_paired_reports_coverage_and_no_extrapolation() -> None:
    result = compare_candidate_with_legacy_artifact(
        _candidate(), _legacy(), bootstrap_iterations=0, daily_top_k=1
    )
    assert set(result["fold"]) == {"A", "B"}
    assert set(result["comparison_scope"]) == {
        "all_exact_symbol_date_overlap",
        "legacy_signal_timing_match_any",
    }
    assert set(result["comparison_label"]) == {
        "candidate_training_contract_good_path5",
        "legacy_next_open_good_path5",
    }
    assert not result["extrapolation_allowed"].any()
    all_overlap = result[
        result["comparison_scope"].eq("all_exact_symbol_date_overlap")
        & result["comparison_label"].eq("candidate_training_contract_good_path5")
    ]
    assert all_overlap.set_index("fold").loc["A", "legacy_overlap_coverage"] == pytest.approx(1.0)
    assert all_overlap.set_index("fold").loc["B", "legacy_overlap_coverage"] == pytest.approx(10 / 12)
    assert all_overlap["delta_pr_auc"].gt(0).all()
    assert set(result["candidate_training_entry_mode"]) == {"next_close"}


def test_legacy_comparison_rejects_c_period() -> None:
    with pytest.raises(ValueError, match="A/B only"):
        compare_candidate_with_legacy_artifact(
            _candidate(), _legacy(), folds=("C",), bootstrap_iterations=0
        )
