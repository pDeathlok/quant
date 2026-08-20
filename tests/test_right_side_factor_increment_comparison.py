from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.research.right_side_factor_increment_comparison import (
    compare_rule_feature_versions,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, year in (("A", 2024), ("B", 2025)):
        for month in (1, 2, 3):
            for day, target, candidate, baseline in (
                (2, 1, 0.9, 0.1),
                (2, 0, 0.1, 0.9),
                (3, 1, 0.8, 0.2),
                (3, 0, 0.2, 0.8),
            ):
                rows.append(
                    {
                        "date": pd.Timestamp(year, month, day),
                        "entry_mode": "next_close",
                        "horizon": 5,
                        "label": "good_path5",
                        "fold": fold,
                        "good_path5": target,
                        "pred_unified_long_task_deep": candidate,
                        "pred_unified_long_task_deep_rule105": baseline,
                    }
                )
    return pd.DataFrame(rows)


def test_better_ranking_has_positive_paired_deltas() -> None:
    result = compare_rule_feature_versions(
        _frame(), bootstrap_iterations=50, top_fraction=0.5, daily_top_k=1
    )
    assert set(result["fold"]) == {"A", "B"}
    for metric in (
        "pr_auc",
        "roc_auc",
        "top10_precision",
        "top10_lift",
        "daily_top_k_label_hit_rate",
    ):
        assert result[f"delta_{metric}"].gt(0).all()
        assert result[f"delta_{metric}_ci_low"].gt(0).all()
    assert result["coverage"].eq(1.0).all()


def test_identical_scores_have_zero_deltas() -> None:
    frame = _frame()
    frame["pred_unified_long_task_deep"] = frame[
        "pred_unified_long_task_deep_rule105"
    ]
    result = compare_rule_feature_versions(frame, bootstrap_iterations=20)
    columns = [column for column in result if column.startswith("delta_") and not column.endswith(("_low", "_high", "_valid"))]
    for column in columns:
        if "bootstrap" not in column:
            assert result[column].eq(0.0).all()


def test_fold_and_label_are_not_mixed_and_invalid_rows_reduce_coverage() -> None:
    frame = _frame()
    frame.loc[0, "pred_unified_long_task_deep"] = np.nan
    result = compare_rule_feature_versions(frame, bootstrap_iterations=0)
    assert len(result) == 2
    assert result.set_index("fold").loc["A", "coverage"] == pytest.approx(11 / 12)
    assert result.set_index("fold").loc["B", "coverage"] == pytest.approx(1.0)
    assert set(result["status"]) == {"exact_only"}


def test_comparison_has_no_return_dependency() -> None:
    frame = _frame()
    assert not any("return" in column for column in frame)
    assert not compare_rule_feature_versions(frame, bootstrap_iterations=0).empty


def test_script_decision_has_top_level_replace_contract_and_beam_fallback(
    tmp_path: Path,
) -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "research"
        / "validate_unified_right_side_models.py"
    )
    spec = importlib.util.spec_from_file_location("right_side_increment_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = _frame()
    frame["symbol"] = [f"S{index}" for index in range(len(frame))]
    frame["pred_unified_long_task_deep_beam"] = frame[
        "pred_unified_long_task_deep"
    ]
    for signal in module.RIGHT_SIDE_SIGNALS:
        frame[signal] = signal == "B2"
    predictions = tmp_path / "predictions.parquet"
    frame.to_parquet(predictions, index=False)
    comparison = tmp_path / "comparison.csv"
    decision_path = tmp_path / "decision.json"
    args = argparse.Namespace(
        predictions=predictions,
        folds=["A", "B"],
        label="good_path5",
        candidate_experiments=[
            "unified_long_task_deep",
            "unified_long_task_deep_beam",
        ],
        baseline_experiment="unified_long_task_deep_rule105",
        top_fraction=0.5,
        bootstrap_iterations=0,
        random_seed=42,
        daily_top_k=1,
        confidence_level=0.95,
        comparison_out=comparison,
        decision_out=decision_path,
        beam_report_root=tmp_path / "missing_beam",
    )
    decision = module.compare_factor_increment(args)
    assert decision["replace_online"] is True
    assert decision["selected_candidate"] == "unified_long_task_deep"
    assert decision["decision_reason"] == "ranking_gate_passed"
    assert not decision["candidates"]["unified_long_task_deep_beam"][
        "all_candidate_gates_passed"
    ]


def test_beam_promotion_requires_v3_provenance_permutation_and_pipeline_select(
    tmp_path: Path,
) -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "research"
        / "validate_unified_right_side_models.py"
    )
    spec = importlib.util.spec_from_file_location("right_side_beam_gate_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = _frame()
    frame["symbol"] = [f"S{index}" for index in range(len(frame))]
    frame["pred_unified_long_task_deep_beam"] = frame[
        "pred_unified_long_task_deep"
    ]
    for signal in module.RIGHT_SIDE_SIGNALS:
        frame[signal] = signal == "B2"
    predictions = tmp_path / "predictions.parquet"
    frame.to_parquet(predictions, index=False)
    beam_root = tmp_path / "beam"
    selected = tuple(module.ADDED_RULE_FEATURE_COLUMNS_V2[:6])
    for fold in ("A", "B"):
        fold_root = beam_root / fold
        fold_root.mkdir(parents=True)
        (fold_root / "search_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": module.BEAM_SCHEMA_VERSION,
                    "test_data_used": False,
                    "test_data_used_for_search": False,
                    "candidate_features": list(module.ADDED_RULE_FEATURE_COLUMNS_V2),
                    "selected_features": list(selected),
                    "selected_features_sha256": module.beam_feature_columns_sha256(
                        selected
                    ),
                    "permutation_gate_passed": True,
                    "pipeline_select_gate_passed": True,
                    "pipeline_select": {"test_data_used": False},
                }
            ),
            encoding="utf-8",
        )
    args = argparse.Namespace(
        predictions=predictions,
        folds=["A", "B"],
        label="good_path5",
        candidate_experiments=["unified_long_task_deep_beam"],
        baseline_experiment="unified_long_task_deep_rule105",
        top_fraction=0.5,
        bootstrap_iterations=0,
        random_seed=42,
        daily_top_k=1,
        confidence_level=0.95,
        comparison_out=tmp_path / "comparison.csv",
        decision_out=tmp_path / "decision.json",
        beam_report_root=beam_root,
    )
    decision = module.compare_factor_increment(args)
    beam = decision["candidates"]["unified_long_task_deep_beam"]
    assert beam["beam_provenance_gate"]["passed"]
    assert beam["beam_permutation_gate"]["passed"]
    assert beam["beam_pipeline_select_gate"]["passed"]
    assert beam["all_candidate_gates_passed"]
    assert decision["selected_candidate"] == "unified_long_task_deep_beam"
    assert decision["replace_online"]

    # An otherwise-identical old/bounded manifest must fail closed.
    old_manifest = json.loads(
        (beam_root / "A" / "search_manifest.json").read_text(encoding="utf-8")
    )
    old_manifest["schema_version"] = "right-side-beam-residual-v1"
    (beam_root / "A" / "search_manifest.json").write_text(
        json.dumps(old_manifest), encoding="utf-8"
    )
    rejected = module.compare_factor_increment(args)
    assert not rejected["candidates"]["unified_long_task_deep_beam"][
        "all_candidate_gates_passed"
    ]
    assert not rejected["replace_online"]


def test_beam_residual_xgboost_uses_row_aligned_base_margin() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "research"
        / "validate_unified_right_side_models.py"
    )
    spec = importlib.util.spec_from_file_location("right_side_beam_margin_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train_x = np.linspace(-2.0, 2.0, 120)
    early_x = np.linspace(-1.8, 1.8, 40)
    train = pd.DataFrame({"x": train_x, "target": (train_x > 0).astype(int)})
    early = pd.DataFrame({"x": early_x, "target": (early_x > 0).astype(int)})
    classifier_spec = replace(
        module.DEFAULT_XGB_CLASSIFIER_SPEC,
        n_estimators=12,
        max_depth=2,
        min_child_weight=1.0,
        early_stopping_rounds=3,
    )
    model = module._fit_residual_base_classifier(
        train,
        early,
        ["x"],
        "target",
        train_base_margin=np.full(len(train), 0.2),
        early_stop_base_margin=np.full(len(early), 0.2),
        n_jobs=1,
        classifier_spec=classifier_spec,
    )
    first = module._predict_residual_raw_margin(
        model, early, base_margin=np.full(len(early), 0.2)
    )
    second = module._predict_residual_raw_margin(
        model, early, base_margin=np.full(len(early), 0.7)
    )
    np.testing.assert_allclose(second - first, 0.5, atol=1e-6)


def test_beam_control_and_residual_helpers_run_end_to_end() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "research"
        / "validate_unified_right_side_models.py"
    )
    spec = importlib.util.spec_from_file_location("right_side_beam_helper_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = 180
    dates = pd.bdate_range("2023-01-02", periods=rows)
    signal_names = module.RIGHT_SIDE_SIGNALS
    frame = pd.DataFrame(
        {
            "symbol": [f"S{index:04d}" for index in range(rows)],
            "date": dates,
            "context": np.linspace(-2.0, 2.0, rows),
            "increment": np.sin(np.arange(rows)),
            "target": (np.arange(rows) % 3 == 0).astype(int),
        }
    )
    for position, signal in enumerate(signal_names):
        frame[signal] = np.arange(rows) % len(signal_names) == position
    train = frame.iloc[:100].copy()
    early = frame.iloc[100:125].copy()
    calibration = frame.iloc[125:150].copy()
    evaluation = frame.iloc[150:].copy()
    classifier_spec = replace(
        module.DEFAULT_XGB_CLASSIFIER_SPEC,
        n_estimators=10,
        max_depth=2,
        min_child_weight=1.0,
        early_stopping_rounds=3,
    )
    control = module._fit_beam_control(
        train,
        early,
        calibration,
        ["context"],
        "target",
        n_jobs=1,
        classifier_spec=classifier_spec,
    )
    candidate = module._fit_beam_residual_candidate(
        train,
        early,
        control,
        ["context", "increment"],
        "target",
        n_jobs=1,
        classifier_spec=classifier_spec,
    )
    baseline, residual = module._beam_residual_event_probability(
        candidate,
        control,
        evaluation,
        candidate_common_features=["context", "increment"],
    )
    assert baseline.shape == residual.shape == (len(evaluation),)
    assert np.isfinite(baseline).all()
    assert np.isfinite(residual).all()
