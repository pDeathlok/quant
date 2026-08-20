"""Shared long-table task model for right-side signal candidates.

Every event is expanded to one row per active signal.  The shared estimator
receives the common causal factor set plus exactly one task one-hot.  At
inference, task-row scores are collapsed back to the original event with a
maximum, matching the operational aggregation used by independent models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant.research.right_side_unified import RIGHT_SIDE_SIGNALS
from quant.research.right_side_unified_features import (
    LEGACY_RULE_FEATURE_COLUMNS_V1,
    LEGACY_RULE_FEATURE_SCHEMA_VERSION_V1,
    RULE_FEATURE_COLUMNS,
    RULE_FEATURE_SCHEMA_VERSION,
)


ACTIVE_TASK_COLUMN = "active_task"
EVENT_POSITION_COLUMN = "_event_position"
LONG_TASK_SCHEMA_VERSION = "right-side-long-task-v1-event-max"
TASK_WEIGHT_LOWER_BOUND = 0.5
TASK_WEIGHT_UPPER_BOUND = 3.0
LONG_TASK_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"task_{signal}" for signal in RIGHT_SIDE_SIGNALS
)
PREDICTION_KEY_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "fold",
    "entry_mode",
    "horizon",
    "label",
)


@dataclass(frozen=True)
class XGBClassifierSpec:
    """Pre-registered XGBoost structure shared by all research arms."""

    n_estimators: int = 450
    max_depth: int = 4
    learning_rate: float = 0.04
    min_child_weight: float = 10.0
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.10
    reg_lambda: float = 3.0
    max_bin: int = 128
    early_stopping_rounds: int = 40

    def classifier_kwargs(self, *, n_jobs: int) -> dict[str, Any]:
        return {
            **asdict(self),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": int(n_jobs),
            "random_state": 42,
        }


DEFAULT_XGB_CLASSIFIER_SPEC = XGBClassifierSpec()
LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC = replace(
    DEFAULT_XGB_CLASSIFIER_SPEC,
    max_depth=6,
    min_child_weight=6.0,
)


@dataclass(frozen=True)
class LongTaskArmSpec:
    experiment: str
    task_weighting: str
    classifier_spec: XGBClassifierSpec
    rule_feature_schema_version: str
    rule_feature_columns: tuple[str, ...]


LONG_TASK_ARM_SPECS: tuple[LongTaskArmSpec, ...] = (
    LongTaskArmSpec(
        "unified_long_task",
        "one_vote",
        DEFAULT_XGB_CLASSIFIER_SPEC,
        RULE_FEATURE_SCHEMA_VERSION,
        RULE_FEATURE_COLUMNS,
    ),
    LongTaskArmSpec(
        "unified_long_task_balanced",
        "inverse_sqrt_task_frequency_clip_0p5_3p0",
        DEFAULT_XGB_CLASSIFIER_SPEC,
        RULE_FEATURE_SCHEMA_VERSION,
        RULE_FEATURE_COLUMNS,
    ),
    LongTaskArmSpec(
        "unified_long_task_deep_rule105",
        "one_vote",
        LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC,
        LEGACY_RULE_FEATURE_SCHEMA_VERSION_V1,
        LEGACY_RULE_FEATURE_COLUMNS_V1,
    ),
    LongTaskArmSpec(
        "unified_long_task_deep",
        "one_vote",
        LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC,
        RULE_FEATURE_SCHEMA_VERSION,
        RULE_FEATURE_COLUMNS,
    ),
)


def expand_long_task_rows(
    events: pd.DataFrame,
    *,
    retained_columns: Sequence[str],
    signal_columns: Sequence[str] = RIGHT_SIDE_SIGNALS,
) -> pd.DataFrame:
    """Expand events to one row per active signal with a unique task one-hot.

    ``retained_columns`` is explicit so outcome-only or identity columns cannot
    accidentally enter the training frame.  Original multi-hot signal columns
    are used only to construct the task rows and are not returned.
    """

    signals = tuple(signal_columns)
    retained = tuple(dict.fromkeys(retained_columns))
    missing = set(signals).union(retained) - set(events.columns)
    if missing:
        raise ValueError(f"long-task frame missing columns: {sorted(missing)}")
    leaked_identity = set(retained).intersection(signals)
    if leaked_identity:
        raise ValueError(
            "retained_columns must not include multi-hot signal identity: "
            f"{sorted(leaked_identity)}"
        )

    active = events[list(signals)].fillna(False).astype(bool).to_numpy()
    event_positions, task_positions = np.nonzero(active)
    output_columns = [
        *retained,
        EVENT_POSITION_COLUMN,
        ACTIVE_TASK_COLUMN,
        *LONG_TASK_FEATURE_COLUMNS,
    ]
    if len(event_positions) == 0:
        return pd.DataFrame(columns=output_columns)

    expanded = events.iloc[event_positions][list(retained)].reset_index(drop=True)
    expanded[EVENT_POSITION_COLUMN] = event_positions.astype(np.int64, copy=False)
    task_names = np.asarray(signals, dtype=object)[task_positions]
    expanded[ACTIVE_TASK_COLUMN] = task_names

    task_matrix = np.zeros(
        (len(expanded), len(RIGHT_SIDE_SIGNALS)),
        dtype=bool,
    )
    canonical_position = {signal: position for position, signal in enumerate(RIGHT_SIDE_SIGNALS)}
    for row_position, task_name in enumerate(task_names):
        if task_name not in canonical_position:
            raise ValueError(f"unknown long-task signal: {task_name}")
        task_matrix[row_position, canonical_position[str(task_name)]] = True
    task_frame = pd.DataFrame(task_matrix, columns=LONG_TASK_FEATURE_COLUMNS)
    return pd.concat([expanded, task_frame], axis=1)


def inverse_sqrt_task_weights(
    expanded: pd.DataFrame,
    *,
    task_column: str = ACTIVE_TASK_COLUMN,
    lower_bound: float = TASK_WEIGHT_LOWER_BOUND,
    upper_bound: float = TASK_WEIGHT_UPPER_BOUND,
) -> pd.Series:
    """Return inverse-sqrt task-frequency weights with exact mean-one scaling.

    A simple normalize-then-clip sequence does not preserve both the requested
    bounds and mean one.  We therefore solve a single monotone scale factor by
    bisection and apply ``clip(raw * scale, lower, upper)``.  This preserves
    inverse-frequency ordering while strictly capping extremely rare tasks.
    """

    if task_column not in expanded.columns:
        raise ValueError(f"long-task frame missing {task_column}")
    if not (
        np.isfinite(lower_bound)
        and np.isfinite(upper_bound)
        and 0.0 < lower_bound < 1.0 < upper_bound
    ):
        raise ValueError(
            "task weight bounds must satisfy 0 < lower_bound < 1 < upper_bound"
        )
    if expanded.empty:
        return pd.Series(dtype=float, index=expanded.index)
    tasks = expanded[task_column]
    if tasks.isna().any():
        raise ValueError("active task contains missing values")
    counts = tasks.value_counts(dropna=False)
    raw = tasks.map(1.0 / np.sqrt(counts.astype(float))).to_numpy(dtype=float)

    scale_low = 0.0
    scale_high = 1.0
    while float(np.clip(raw * scale_high, lower_bound, upper_bound).mean()) < 1.0:
        scale_high *= 2.0
    for _ in range(100):
        scale = (scale_low + scale_high) / 2.0
        current_mean = float(
            np.clip(raw * scale, lower_bound, upper_bound).mean()
        )
        if current_mean < 1.0:
            scale_low = scale
        else:
            scale_high = scale
    weights = np.clip(
        raw * ((scale_low + scale_high) / 2.0),
        lower_bound,
        upper_bound,
    )
    return pd.Series(weights, index=expanded.index, dtype=float)


def aggregate_long_task_predictions(
    expanded: pd.DataFrame,
    probabilities: Sequence[float],
    *,
    event_count: int,
    event_position_column: str = EVENT_POSITION_COLUMN,
) -> np.ndarray:
    """Collapse task-row scores to events using the independent-arm max rule."""

    if event_count < 0:
        raise ValueError("event_count must be non-negative")
    if event_position_column not in expanded.columns:
        raise ValueError(f"expanded frame missing {event_position_column}")
    values = np.asarray(probabilities, dtype=float)
    if len(expanded) != len(values):
        raise ValueError("prediction length does not match expanded long-task rows")
    positions = pd.to_numeric(
        expanded[event_position_column], errors="raise"
    ).to_numpy(dtype=np.int64)
    if len(positions) and (positions.min() < 0 or positions.max() >= event_count):
        raise ValueError("long-task event position is outside the source frame")

    aggregated = np.full(event_count, -np.inf, dtype=float)
    finite = np.isfinite(values)
    if finite.any():
        np.maximum.at(aggregated, positions[finite], values[finite])
    aggregated[np.isneginf(aggregated)] = np.nan
    return aggregated


def apply_event_calibrator(
    probabilities: Sequence[float],
    calibrator: object,
) -> np.ndarray:
    """Apply an event-level probability calibrator while preserving NaNs."""

    raw = np.asarray(probabilities, dtype=float)
    calibrated = np.full(len(raw), np.nan, dtype=float)
    valid = np.isfinite(raw)
    if not valid.any():
        return calibrated
    clipped = np.clip(raw[valid], 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrated[valid] = np.asarray(
        calibrator.predict_proba(logits), dtype=float
    )[:, 1]
    return calibrated


@dataclass
class LongTaskUnifiedModel:
    """One shared task-conditioned estimator with event-level calibration."""

    base_model: object
    event_calibrator: object
    common_features: tuple[str, ...]

    @property
    def feature_names_in_(self) -> list[str]:
        return [*self.common_features, *LONG_TASK_FEATURE_COLUMNS]

    @property
    def selected_features_(self) -> list[str]:
        return list(self.feature_names_in_)

    @property
    def factor_schema_version_(self) -> str | None:
        return getattr(self.base_model, "factor_schema_version_", None)

    def predict_proba(self, events: pd.DataFrame) -> np.ndarray:
        expanded = expand_long_task_rows(
            events,
            retained_columns=self.common_features,
        )
        if expanded.empty:
            probability = np.full(len(events), np.nan, dtype=float)
        else:
            task_probability = np.asarray(
                self.base_model.predict_proba(expanded), dtype=float
            )[:, 1]
            raw_event_probability = aggregate_long_task_predictions(
                expanded,
                task_probability,
                event_count=len(events),
            )
            probability = apply_event_calibrator(
                raw_event_probability,
                self.event_calibrator,
            )
        return np.column_stack([1.0 - probability, probability])

    def predict(self, events: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(events)[:, 1] >= 0.5).astype(int)


def merge_prediction_artifacts(
    existing: pd.DataFrame,
    replacement: pd.DataFrame,
    *,
    replace_columns: Sequence[str],
    key_columns: Sequence[str] = PREDICTION_KEY_COLUMNS,
) -> pd.DataFrame:
    """Upsert selected prediction columns without deleting other model arms."""

    keys = tuple(key_columns)
    replaced = tuple(dict.fromkeys(replace_columns))
    missing_existing = set(keys) - set(existing.columns)
    missing_replacement = set(keys).union(replaced) - set(replacement.columns)
    if missing_existing or missing_replacement:
        raise ValueError(
            "prediction artifact columns missing; "
            f"existing={sorted(missing_existing)} "
            f"replacement={sorted(missing_replacement)}"
        )
    if existing.duplicated(list(keys)).any() or replacement.duplicated(list(keys)).any():
        raise ValueError("prediction artifacts contain duplicate event keys")

    old = existing.set_index(list(keys))
    new = replacement.set_index(list(keys))
    union = old.index.union(new.index, sort=False)
    merged = old.reindex(union)
    for column in new.columns:
        incoming = new[column].reindex(union)
        if column in replaced:
            if column not in merged.columns:
                merged[column] = np.nan
            present = union.isin(new.index)
            merged.loc[present, column] = incoming.loc[present].to_numpy()
        elif column not in merged.columns:
            merged[column] = incoming
        else:
            merged[column] = merged[column].combine_first(incoming)
    return (
        merged.reset_index()
        .sort_values(["entry_mode", "horizon", "label", "fold", "date", "symbol"], kind="stable")
        .reset_index(drop=True)
    )
