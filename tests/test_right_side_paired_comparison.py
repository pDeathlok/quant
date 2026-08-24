from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.research.right_side_paired_comparison import (
    _prepare_rank_inputs,
    _weighted_rank_metrics,
    paired_model_comparisons,
)
from quant.research.right_side_targets import (
    TERMINAL_NET_POSITIVE_15BPS,
    target_metadata,
)
from quant.research.right_side_unified import binary_metrics


def _paired_prediction_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in (1, 2, 3):
        for day, label, terminal, unified, independent in (
            (2, 1, 0.05, 0.90, 0.10),
            (2, 0, -0.02, 0.10, 0.90),
            (3, 1, 0.04, 0.80, 0.20),
            (3, 0, -0.01, 0.20, 0.80),
        ):
            rows.append(
                {
                    "date": pd.Timestamp(2024, month, day),
                    "entry_mode": "next_open",
                    "horizon": 5,
                    "label": "good_path5",
                    "fold": "A",
                    "good_path5": label,
                    "terminal_return": terminal,
                    "pred_independent": independent,
                    "pred_unified_with_signal_id": unified,
                    "pred_unified_balanced": unified,
                    "pred_unified_long_task": unified,
                    "pred_unified_long_task_balanced": unified,
                    "pred_unified_long_task_deep": unified,
                }
            )
    return pd.DataFrame(rows)


def test_paired_comparison_reports_exact_fold_deltas_and_month_ci() -> None:
    result = paired_model_comparisons(
        _paired_prediction_frame(),
        bootstrap_iterations=100,
        daily_top_k=1,
        top_fraction=0.5,
        random_state=7,
    )

    assert set(result["candidate"]) == {
        "unified_with_signal_id",
        "unified_balanced",
    }
    row = result.set_index("candidate").loc["unified_with_signal_id"]
    assert row["status"] == "ok"
    assert row["paired_rows"] == 12
    assert row["month_blocks"] == 3
    assert row["delta_pr_auc"] > 0
    assert row["delta_top_lift"] == pytest.approx(2.0)
    assert row["candidate_daily_top_k_avg_terminal_return"] == pytest.approx(0.045)
    assert row["baseline_daily_top_k_avg_terminal_return"] == pytest.approx(-0.015)
    assert row["delta_daily_top_k_avg_terminal_return"] == pytest.approx(0.06)
    assert row["delta_pr_auc_ci_low"] > 0
    assert row["delta_top_lift_ci_low"] > 0
    assert row["delta_daily_top_k_avg_terminal_return_ci_low"] > 0
    assert row["delta_pr_auc_bootstrap_valid"] == 100


def test_paired_comparison_uses_intersection_of_valid_test_rows() -> None:
    frame = _paired_prediction_frame()
    frame.loc[0, "pred_unified_balanced"] = np.nan

    result = paired_model_comparisons(frame, bootstrap_iterations=0)
    rows = result.set_index("candidate")

    assert rows.loc["unified_with_signal_id", "paired_rows"] == 12
    assert rows.loc["unified_balanced", "paired_rows"] == 11
    assert set(rows["status"]) == {"exact_only"}
    assert rows["delta_pr_auc_ci_low"].isna().all()


def test_paired_comparison_keeps_fold_and_label_scopes_separate() -> None:
    first = _paired_prediction_frame()
    second = first.copy()
    second["fold"] = "B"
    second["label"] = "hit_up5"
    second["hit_up5"] = 1 - second["good_path5"]
    frame = pd.concat([first, second], ignore_index=True, sort=False)

    result = paired_model_comparisons(frame, bootstrap_iterations=0)

    assert len(result) == 4
    assert set(zip(result["label"], result["fold"])) == {
        ("good_path5", "A"),
        ("hit_up5", "B"),
    }
    good_path = result[result["label"].eq("good_path5")]
    hit_up = result[result["label"].eq("hit_up5")]
    assert good_path["delta_pr_auc"].gt(0).all()
    assert hit_up["delta_pr_auc"].lt(0).all()


def test_paired_comparison_validates_required_prediction_arms() -> None:
    frame = _paired_prediction_frame().drop(columns="pred_independent")

    with pytest.raises(ValueError, match="pred_independent"):
        paired_model_comparisons(frame)


def test_month_weighted_rank_metrics_match_explicit_block_replication() -> None:
    labels = np.array([1, 0, 0, 1, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    block_ids = np.array([0, 0, 1, 1, 1, 1])
    block_counts = np.array([2, 1])
    prepared = _prepare_rank_inputs(labels, probabilities, block_ids)

    actual_ap, actual_lift = _weighted_rank_metrics(
        prepared,
        block_counts,
        top_fraction=0.5,
    )
    repeated_indices = np.r_[
        np.flatnonzero(block_ids == 0),
        np.flatnonzero(block_ids == 0),
        np.flatnonzero(block_ids == 1),
    ]
    expected = binary_metrics(
        labels[repeated_indices],
        probabilities[repeated_indices],
        top_fraction=0.5,
    )

    assert actual_ap == pytest.approx(expected["average_precision"])
    assert actual_lift == pytest.approx(expected["top_lift"])


def test_validation_report_writes_paired_comparison_artifact(tmp_path: Path) -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "research"
        / "validate_unified_right_side_models.py"
    )
    spec = importlib.util.spec_from_file_location("validate_unified_right_side_models", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metrics_path = tmp_path / "metrics.csv"
    metric_row = {
        "entry_mode": "next_open",
        "horizon": 5,
        "label": TERMINAL_NET_POSITIVE_15BPS,
        "fold": "A",
        "experiment": "independent",
        "signal": "ALL",
        "rows": 12,
        "roc_auc": 0.5,
        "average_precision": 0.5,
        "top_lift": 1.0,
        "brier": 0.25,
        "average_net_return": 0.0,
        "profit_factor": 1.0,
        **target_metadata(TERMINAL_NET_POSITIVE_15BPS),
    }
    pd.DataFrame([metric_row]).to_csv(metrics_path, index=False)
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(
        json.dumps(
            {
                "unique_events": 12,
                "multi_hit_events": 0,
                "locked_limit_rows": 0,
                "mature_rows": 12,
            }
        ),
        encoding="utf-8",
    )
    factor_path = tmp_path / "factors.csv"
    pd.DataFrame({"signal": ["ALL"], "feature": ["x"], "status": ["ok"]}).to_csv(
        factor_path,
        index=False,
    )
    predictions_path = tmp_path / "predictions.parquet"
    predictions = _paired_prediction_frame()
    predictions[TERMINAL_NET_POSITIVE_15BPS] = predictions["good_path5"]
    predictions["label"] = TERMINAL_NET_POSITIVE_15BPS
    for column, value in target_metadata(TERMINAL_NET_POSITIVE_15BPS).items():
        predictions[column] = value
    predictions["independent_model_available"] = True
    predictions.loc[0, "independent_model_available"] = False
    predictions.to_parquet(predictions_path, index=False)
    paired_path = tmp_path / "paired.csv"
    report_path = tmp_path / "report.md"
    args = argparse.Namespace(
        metrics=metrics_path,
        sample_audit=sample_path,
        factor_audit=factor_path,
        predictions=predictions_path,
        paired_comparison_out=paired_path,
        paired_bootstrap_iterations=20,
        paired_confidence_level=0.95,
        paired_random_state=42,
        paired_top_fraction=0.5,
        daily_top_k=1,
        report_out=report_path,
    )

    assert module.render_report(args) == report_path
    paired = pd.read_csv(paired_path)
    report = report_path.read_text(encoding="utf-8")
    assert len(paired) == 10
    assert set(paired["comparison_scope"]) == {
        "all_events",
        "independent_model_rows",
    }
    assert set(paired["candidate"]) == {
        "unified_with_signal_id",
        "unified_balanced",
        "unified_long_task",
        "unified_long_task_balanced",
        "unified_long_task_deep",
    }
    assert set(paired.groupby("comparison_scope")["paired_rows"].first()) == {11, 12}
    assert "统一模型相对独立模型的测试集配对差值" in report
    assert "independent_model_rows" in report
    assert "unified_long_task" in report
    assert "unified_long_task_balanced" in report
    assert "unified_long_task_deep" in report
    assert "Δ PR-AUC [95% CI]" in report
    assert "terminal_return > 0.0015" in report
    assert "equality is negative" in report


def test_dataset_audit_requires_every_signal_and_complete_mature_labels(
    tmp_path: Path,
) -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "research"
        / "validate_unified_right_side_models.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_unified_right_side_models_audit_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    events = pd.DataFrame(
        {
            "symbol": [f"S{index:02d}" for index in range(len(module.RIGHT_SIDE_SIGNALS))],
            "date": pd.Timestamp("2024-01-02"),
        }
    )
    for signal in module.RIGHT_SIDE_SIGNALS:
        events[signal] = False
    for index, signal in enumerate(module.RIGHT_SIDE_SIGNALS):
        events.loc[index, signal] = True
    events = pd.concat(
        [
            events,
            pd.DataFrame(
                {
                    feature: np.full(len(events), float(feature_index + 1))
                    for feature_index, feature in enumerate(module.RULE_FEATURE_COLUMNS)
                }
            ),
        ],
        axis=1,
    )
    for feature in {
        feature
        for requirements in module.SIGNAL_FEATURE_REQUIREMENTS.values()
        for feature in requirements
    }:
        if feature not in events:
            events[feature] = 1.0

    labels = events[["symbol", "date"]].copy()
    labels["entry_mode"] = "next_open"
    labels["horizon"] = 5
    labels["entry_date"] = labels["date"] + pd.Timedelta(days=1)
    labels["label_end_date"] = labels["date"] + pd.Timedelta(days=8)
    labels["mature"] = True
    labels["locked_limit_up"] = False
    labels["mfe"] = 0.06
    labels["mae"] = -0.01
    labels["terminal"] = 0.03
    labels["terminal_return"] = 0.03
    labels["hit_up3"] = True
    labels["hit_up5"] = True
    labels["hit_up8"] = False
    labels["hit_down3"] = False
    labels["good_path5"] = True

    events_path = tmp_path / "events.parquet"
    labels_path = tmp_path / "labels.parquet"
    events.to_parquet(events_path, index=False)
    labels.to_parquet(labels_path, index=False)
    args = argparse.Namespace(dataset=events_path, labels=labels_path)

    summary = module.audit_dataset(args)
    assert summary["signals_without_events"] == []
    assert summary["signal_event_rows"] == {
        signal: 1 for signal in module.RIGHT_SIDE_SIGNALS
    }
    assert summary["mature_with_missing_label"] == 0

    missing_signal_events = events.copy()
    missing_signal_events["YUEYUE"] = False
    missing_signal_events.to_parquet(events_path, index=False)
    with pytest.raises(RuntimeError, match="signals_without_events.*YUEYUE"):
        module.audit_dataset(args)

    events.to_parquet(events_path, index=False)
    incomplete_labels = labels.copy()
    incomplete_labels["good_path5"] = incomplete_labels["good_path5"].astype(
        "boolean"
    )
    incomplete_labels.loc[0, "good_path5"] = pd.NA
    incomplete_labels.to_parquet(labels_path, index=False)
    with pytest.raises(RuntimeError, match="mature_with_missing_label.*1"):
        module.audit_dataset(args)
