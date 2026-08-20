from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _load_training_script() -> object:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "research"
        / "validate_unified_right_side_models.py"
    )
    spec = importlib.util.spec_from_file_location(
        "right_side_beam_v3_integration_script", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_beam_inputs(module: object, root: Path) -> tuple[Path, Path, pd.DataFrame]:
    date_blocks = (
        pd.bdate_range("2020-01-02", periods=160),
        pd.bdate_range("2023-01-03", periods=100),
        pd.bdate_range("2024-01-03", periods=40),
    )
    dates = date_blocks[0].append(date_blocks[1]).append(date_blocks[2])
    row_number = np.arange(len(dates))
    target = ((row_number + row_number // 7) % 2).astype(int)
    events = pd.DataFrame(
        {
            "symbol": [f"S{index:06d}" for index in row_number],
            "date": dates,
        }
    )
    for position, signal in enumerate(module.RIGHT_SIDE_SIGNALS):
        events[signal] = signal == "B2"
    for column in module.SUBVARIANT_COLUMNS:
        events[column] = column == module.SUBVARIANT_COLUMNS[0]
    events["signal_count"] = 1
    events["has_right_signal"] = True
    events["has_mixed_signal"] = False

    # Keep every required schema column populated and non-constant so the
    # smoke test exercises the same admission and model-input contracts as the
    # full experiment without depending on production parquet files.
    all_factors = tuple(
        dict.fromkeys(
            [*module.PROJECT_FACTOR_COLUMNS, *module.RULE_FEATURE_COLUMNS]
        )
    )
    for position, column in enumerate(all_factors):
        events[column] = (
            np.sin(row_number / (3.0 + position % 11))
            + (position % 5) * 0.01
        ).astype(np.float32)

    labels = pd.DataFrame(
        {
            "symbol": events["symbol"],
            "date": dates,
            "entry_mode": "next_close",
            "horizon": 5,
            "label_end_date": dates + pd.offsets.BDay(1),
            "mature": True,
            "locked_limit_up": False,
            "good_path5": target,
            "terminal_return": np.where(target == 1, 0.06, -0.02),
            "mfe": np.where(target == 1, 0.08, 0.01),
            "mae": np.where(target == 1, -0.01, -0.04),
        }
    )
    dataset_path = root / "events.parquet"
    labels_path = root / "labels.parquet"
    events.to_parquet(dataset_path, index=False)
    labels.to_parquet(labels_path, index=False)
    return dataset_path, labels_path, events


def test_beam_v3_full_helper_chain_persists_final_model_and_manifest(
    tmp_path: Path,
) -> None:
    module = _load_training_script()
    dataset_path, labels_path, events = _synthetic_beam_inputs(module, tmp_path)
    tiny_spec = replace(
        module.LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC,
        n_estimators=5,
        max_depth=2,
        min_child_weight=1.0,
        early_stopping_rounds=2,
    )
    module.LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC = tiny_spec
    model_root = tmp_path / "models"
    report_root = tmp_path / "reports"
    args = argparse.Namespace(
        dataset=dataset_path,
        labels=labels_path,
        entry_mode="next_close",
        horizon=5,
        label="good_path5",
        folds=["A"],
        experiments=["unified_long_task_deep_beam"],
        minimum_local_rows=20,
        minimum_project_feature_coverage=0.5,
        model_jobs=1,
        daily_top_k=3,
        round_trip_cost_bps=15.0,
        model_root=model_root,
        metrics_out=report_root / "metrics.csv",
        signal_metrics_out=report_root / "signal_metrics.csv",
        predictions_out=report_root / "predictions.parquet",
        beam_report_root=report_root / "beam",
        beam_width=1,
        beam_min_features=len(module.ADDED_RULE_FEATURE_COLUMNS_V2),
        beam_max_remove=0,
        beam_permutation_rounds=1,
        beam_maximum_permutation_p_value=0.99,
        beam_search_estimators=3,
        beam_search_max_depth=2,
        beam_search_early_stopping_rounds=2,
        beam_history_max_rows=10_000,
        beam_evaluation_max_rows=10_000,
    )

    result = module.train_models(args)

    assert result["trained_folds"] == ["A"]
    assert result["trained_experiments"] == ["unified_long_task_deep_beam"]
    search_manifest_path = report_root / "beam" / "A" / "search_manifest.json"
    search_manifest = json.loads(search_manifest_path.read_text(encoding="utf-8"))
    assert search_manifest["schema_version"] == module.BEAM_SCHEMA_VERSION
    assert search_manifest["test_data_used"] is False
    assert search_manifest["test_data_used_for_search"] is False
    assert len(search_manifest["rolling_windows"]) == 3
    assert search_manifest["permutation"]["rounds"] == 1
    assert search_manifest["pipeline_select"]["status"] == "evaluated_after_top1_freeze"
    assert search_manifest["pipeline_select"]["test_data_used"] is False
    assert search_manifest["selected_features"] == list(
        module.ADDED_RULE_FEATURE_COLUMNS_V2
    )

    model_path = (
        model_root
        / "next_close"
        / "h5"
        / "good_path5"
        / "A"
        / "unified_long_task_deep_beam.joblib"
    )
    final_manifest_path = model_path.with_suffix(".manifest.json")
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    assert final_manifest["beam_schema_version"] == module.BEAM_SCHEMA_VERSION
    assert final_manifest["fold"]["test_year"] == 2024
    assert final_manifest["beam_search"]["outer_fold"] == "A"
    assert final_manifest["rule_feature_count"] == len(module.RULE_FEATURE_COLUMNS)

    persisted_model = joblib.load(model_path)
    test_events = events[pd.to_datetime(events["date"]).dt.year.eq(2024)]
    probability = persisted_model.predict_proba(test_events)[:, 1]
    assert probability.shape == (len(test_events),)
    assert np.isfinite(probability).all()
    predictions = pd.read_parquet(report_root / "predictions.parquet")
    assert set(predictions["fold"]) == {"A"}
    assert "C" not in set(predictions["fold"])
    assert predictions["pred_unified_long_task_deep_beam"].notna().all()


def test_beam_runtime_files_do_not_import_cross_module_private_names() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "src" / "quant" / "research" / "right_side_beam_feature_selection.py",
        root / "scripts" / "research" / "validate_unified_right_side_models.py",
    )
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for imported in node.names:
                if imported.name.startswith("_"):
                    violations.append(
                        f"{path.relative_to(root)}:{node.lineno}:{imported.name}"
                    )
    assert violations == []
