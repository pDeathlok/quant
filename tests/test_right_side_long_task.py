from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from quant.research.right_side_long_task import (
    ACTIVE_TASK_COLUMN,
    DEFAULT_XGB_CLASSIFIER_SPEC,
    EVENT_POSITION_COLUMN,
    LONG_TASK_ARM_SPECS,
    LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC,
    LONG_TASK_FEATURE_COLUMNS,
    LongTaskUnifiedModel,
    aggregate_long_task_predictions,
    expand_long_task_rows,
    inverse_sqrt_task_weights,
    merge_prediction_artifacts,
)
from quant.research.right_side_unified import RIGHT_SIDE_SIGNALS
from quant.research.right_side_unified_features import (
    LEGACY_RULE_FEATURE_COLUMNS_V1,
    RULE_FEATURE_COLUMNS,
)


def _events() -> pd.DataFrame:
    frame = pd.DataFrame(False, index=range(3), columns=list(RIGHT_SIDE_SIGNALS))
    frame.loc[0, ["B2", "B3"]] = True
    frame.loc[1, "YUEYUE"] = True
    frame.loc[2, "KEY_K"] = True
    frame["factor"] = [1.0, 2.0, 3.0]
    frame["target"] = [1, 0, 1]
    frame["future_only"] = [99.0, 98.0, 97.0]
    return frame


def test_expand_long_task_rows_duplicates_multi_hits_with_one_active_task() -> None:
    expanded = expand_long_task_rows(
        _events(),
        retained_columns=["factor", "target"],
    )

    assert expanded[EVENT_POSITION_COLUMN].tolist() == [0, 0, 1, 2]
    assert expanded[ACTIVE_TASK_COLUMN].tolist() == ["B2", "B3", "YUEYUE", "KEY_K"]
    assert expanded[list(LONG_TASK_FEATURE_COLUMNS)].sum(axis=1).eq(1).all()
    assert expanded.loc[0, "task_B2"]
    assert expanded.loc[1, "task_B3"]
    assert expanded.loc[2, "task_YUEYUE"]
    assert expanded.loc[3, "task_KEY_K"]
    assert "future_only" not in expanded.columns
    assert not set(RIGHT_SIDE_SIGNALS).intersection(expanded.columns)


def test_aggregate_long_task_predictions_uses_event_level_max() -> None:
    expanded = expand_long_task_rows(_events(), retained_columns=["factor"])

    actual = aggregate_long_task_predictions(
        expanded,
        [0.20, 0.80, 0.60, 0.40],
        event_count=3,
    )

    assert actual.tolist() == pytest.approx([0.80, 0.60, 0.40])


def test_inverse_sqrt_task_weights_are_mean_one_and_strictly_clipped() -> None:
    expanded = pd.DataFrame(
        {
            ACTIVE_TASK_COLUMN: ["B2"] * 10_000 + ["PINGHANG"] * 2_500 + ["CHANGAN"],
        }
    )

    weights = inverse_sqrt_task_weights(expanded)
    by_task = weights.groupby(expanded[ACTIVE_TASK_COLUMN]).first()

    assert weights.index.equals(expanded.index)
    assert weights.mean() == pytest.approx(1.0, abs=1e-12)
    assert weights.between(0.5, 3.0).all()
    assert by_task["B2"] < by_task["PINGHANG"] < by_task["CHANGAN"]
    assert by_task["CHANGAN"] == pytest.approx(3.0)


def test_inverse_sqrt_task_weights_validate_bounds_and_empty_input() -> None:
    empty = pd.DataFrame({ACTIVE_TASK_COLUMN: pd.Series(dtype=str)})

    assert inverse_sqrt_task_weights(empty).empty
    with pytest.raises(ValueError, match="lower_bound"):
        inverse_sqrt_task_weights(
            pd.DataFrame({ACTIVE_TASK_COLUMN: ["B2"]}),
            lower_bound=1.0,
        )


def test_long_task_arm_specs_change_only_preregistered_dimensions() -> None:
    arms = {arm.experiment: arm for arm in LONG_TASK_ARM_SPECS}

    assert set(arms) == {
        "unified_long_task",
        "unified_long_task_balanced",
        "unified_long_task_deep_rule105",
        "unified_long_task_deep",
    }
    assert arms["unified_long_task"].task_weighting == "one_vote"
    assert (
        arms["unified_long_task_balanced"].task_weighting
        == "inverse_sqrt_task_frequency_clip_0p5_3p0"
    )
    assert (
        arms["unified_long_task_balanced"].classifier_spec
        == DEFAULT_XGB_CLASSIFIER_SPEC
    )
    default = asdict(DEFAULT_XGB_CLASSIFIER_SPEC)
    deep = asdict(LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC)
    changed = {key for key in default if default[key] != deep[key]}
    assert changed == {"max_depth", "min_child_weight"}
    assert deep["max_depth"] == 6
    assert deep["min_child_weight"] == pytest.approx(6.0)
    assert (
        arms["unified_long_task_deep_rule105"].classifier_spec
        == arms["unified_long_task_deep"].classifier_spec
    )
    assert (
        arms["unified_long_task_deep_rule105"].task_weighting
        == arms["unified_long_task_deep"].task_weighting
        == "one_vote"
    )
    assert len(arms["unified_long_task_deep_rule105"].rule_feature_columns) == len(
        LEGACY_RULE_FEATURE_COLUMNS_V1
    )
    assert len(arms["unified_long_task_deep"].rule_feature_columns) == len(
        RULE_FEATURE_COLUMNS
    )


def test_long_task_expansion_accepts_an_explicit_left_side_task_contract() -> None:
    signals = ("DUICHEN_VA", "NANA", "YIDONG_DILIAN")
    tasks = tuple(f"task_{signal}" for signal in signals)
    events = pd.DataFrame(
        {
            "DUICHEN_VA": [True, False],
            "NANA": [True, False],
            "YIDONG_DILIAN": [False, True],
            "factor": [1.0, 2.0],
        }
    )

    expanded = expand_long_task_rows(
        events,
        retained_columns=("factor",),
        signal_columns=signals,
        task_feature_columns=tasks,
    )

    assert expanded[ACTIVE_TASK_COLUMN].tolist() == [
        "DUICHEN_VA",
        "NANA",
        "YIDONG_DILIAN",
    ]
    assert expanded[list(tasks)].sum(axis=1).eq(1).all()


class _TaskScoreModel:
    feature_names_in_ = ["factor", *LONG_TASK_FEATURE_COLUMNS]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        score = (
            frame["task_B2"].astype(float) * 0.20
            + frame["task_B3"].astype(float) * 0.80
            + frame["task_KEY_K"].astype(float) * 0.40
            + frame["task_YUEYUE"].astype(float) * 0.60
        ).to_numpy()
        return np.column_stack([1.0 - score, score])


class _IdentityLogitCalibrator:
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=float).reshape(-1)
        probability = 1.0 / (1.0 + np.exp(-values))
        return np.column_stack([1.0 - probability, probability])


def test_long_task_model_calibrates_after_event_level_max() -> None:
    model = LongTaskUnifiedModel(
        base_model=_TaskScoreModel(),
        event_calibrator=_IdentityLogitCalibrator(),
        common_features=("factor",),
    )

    probability = model.predict_proba(_events())[:, 1]

    assert probability.tolist() == pytest.approx([0.80, 0.60, 0.40])


def test_merge_prediction_artifacts_replaces_only_requested_arm_and_fold() -> None:
    existing = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "date": pd.to_datetime(["2024-01-02", "2025-01-02"]),
            "fold": ["A", "B"],
            "entry_mode": ["next_close", "next_close"],
            "horizon": [5, 5],
            "label": ["good_path5", "good_path5"],
            "pred_independent": [0.10, 0.20],
            "pred_unified_long_task": [0.30, 0.40],
        }
    )
    replacement = existing.iloc[[0]].copy()
    replacement["pred_unified_long_task"] = 0.90
    replacement = replacement.drop(columns="pred_independent")

    merged = merge_prediction_artifacts(
        existing,
        replacement,
        replace_columns=["pred_unified_long_task"],
    ).sort_values("fold")

    assert merged["pred_independent"].tolist() == pytest.approx([0.10, 0.20])
    assert merged["pred_unified_long_task"].tolist() == pytest.approx([0.90, 0.40])
