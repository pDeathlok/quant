"""Absolute market-state gate for left-side ranking research.

The cross-sectional ranker answers *which* left-side candidates are preferable.
This module answers the separate question of *whether* the current market state
has historically paid for taking those candidates at all.  Keeping the two
questions separate prevents a daily percentile from manufacturing a high score
on uniformly weak days.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LEFT_SIDE_REGIME_GATE_SCHEMA_VERSION = "left-side-regime-gate-v1"
DEFAULT_ADMISSION_QUANTILE = 0.40
DEFAULT_RIDGE_ALPHA = 10.0
DEFAULT_ROUND_TRIP_COST = 20.0 / 10_000.0


@dataclass
class LeftSideRegimeGate:
    """Persisted absolute-return model with a validation-frozen threshold."""

    pipeline: Pipeline
    feature_names_in_: list[str]
    threshold: float
    admission_quantile: float
    target_name: str = "daily_average_net_return"
    schema_version: str = LEFT_SIDE_REGIME_GATE_SCHEMA_VERSION

    def predict_score(self, values: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.feature_names_in_) - set(values.columns))
        if missing:
            raise ValueError(f"left-side regime input misses columns: {missing}")
        score = np.asarray(
            self.pipeline.predict(values[self.feature_names_in_]), dtype=float
        )
        if score.ndim != 1 or len(score) != len(values) or not np.isfinite(score).all():
            raise ValueError("left-side regime model produced invalid scores")
        return score

    def predict_open(self, values: pd.DataFrame) -> np.ndarray:
        """Admit rows whose absolute expected return clears the frozen threshold."""

        return self.predict_score(values) >= float(self.threshold)


def materialize_daily_regime_targets(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
) -> pd.DataFrame:
    """Collapse eligible signal events to one causal market-state row per date."""

    feature_names = list(dict.fromkeys(str(value) for value in features))
    required = {"date", "terminal_return", *feature_names}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"left-side regime target frame misses columns: {missing}")
    if not feature_names:
        raise ValueError("left-side regime target requires market-state features")
    if not np.isfinite(float(round_trip_cost)) or float(round_trip_cost) < 0.0:
        raise ValueError("round_trip_cost must be finite and non-negative")

    work = frame[["date", "terminal_return", *feature_names]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    work["terminal_return"] = pd.to_numeric(
        work["terminal_return"], errors="coerce"
    )
    if work["terminal_return"].isna().any():
        raise ValueError("left-side regime target contains missing terminal returns")

    # Every gate feature is a full-market value and therefore must be identical
    # for all stock events on the same date.  Fail closed on accidental leakage
    # or a bad many-to-one join instead of silently taking an arbitrary row.
    inconsistent = []
    grouped = work.groupby("date", sort=True)
    for column in feature_names:
        if grouped[column].nunique(dropna=False).gt(1).any():
            inconsistent.append(column)
    if inconsistent:
        raise ValueError(
            "left-side regime features vary within date: "
            f"{sorted(inconsistent)}"
        )

    aggregations: dict[str, tuple[str, str]] = {
        column: (column, "first") for column in feature_names
    }
    daily = (
        work.groupby("date", sort=True)
        .agg(
            **aggregations,
            average_terminal_return=("terminal_return", "mean"),
            event_count=("terminal_return", "size"),
        )
        .reset_index()
    )
    daily["daily_average_net_return"] = (
        daily["average_terminal_return"] - float(round_trip_cost)
    )
    return daily


def fit_left_side_regime_gate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    features: Sequence[str],
    admission_quantile: float = DEFAULT_ADMISSION_QUANTILE,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    target: str = "daily_average_net_return",
) -> LeftSideRegimeGate:
    """Fit on training dates and freeze an absolute threshold on validation."""

    feature_names = list(dict.fromkeys(str(value) for value in features))
    if not 0.0 < float(admission_quantile) < 1.0:
        raise ValueError("admission_quantile must be in (0, 1)")
    if not np.isfinite(float(ridge_alpha)) or float(ridge_alpha) <= 0.0:
        raise ValueError("ridge_alpha must be finite and positive")
    for name, current in (("train", train), ("validation", validation)):
        missing = sorted(set((*feature_names, target)) - set(current.columns))
        if missing:
            raise ValueError(f"{name} regime frame misses columns: {missing}")
        target_values = pd.to_numeric(current[target], errors="coerce")
        if target_values.isna().any() or current.empty:
            raise ValueError(f"{name} regime target must be non-empty and finite")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(ridge_alpha))),
        ]
    )
    pipeline.fit(train[feature_names], train[target].astype(float))
    validation_score = np.asarray(
        pipeline.predict(validation[feature_names]), dtype=float
    )
    if not np.isfinite(validation_score).all():
        raise ValueError("left-side validation regime scores are not finite")
    threshold = float(np.quantile(validation_score, float(admission_quantile)))
    return LeftSideRegimeGate(
        pipeline=pipeline,
        feature_names_in_=feature_names,
        threshold=threshold,
        admission_quantile=float(admission_quantile),
        target_name=target,
    )


def compose_absolute_display_score(
    cross_section_percentile: Sequence[float] | np.ndarray,
    gate_open: Sequence[bool] | np.ndarray,
) -> np.ndarray:
    """Map closed days to [0, 50) and admitted days to [50, 100]."""

    percentile = np.asarray(cross_section_percentile, dtype=float)
    admitted = np.asarray(gate_open, dtype=bool)
    if percentile.ndim != 1 or admitted.ndim != 1 or len(percentile) != len(admitted):
        raise ValueError("left-side absolute score inputs must be aligned vectors")
    if not np.isfinite(percentile).all() or ((percentile < 0.0) | (percentile > 1.0)).any():
        raise ValueError("cross-section percentile must be finite and within [0, 1]")
    return np.where(admitted, 50.0 + 50.0 * percentile, 50.0 * percentile)


__all__ = [
    "DEFAULT_ADMISSION_QUANTILE",
    "DEFAULT_RIDGE_ALPHA",
    "DEFAULT_ROUND_TRIP_COST",
    "LEFT_SIDE_REGIME_GATE_SCHEMA_VERSION",
    "LeftSideRegimeGate",
    "compose_absolute_display_score",
    "fit_left_side_regime_gate",
    "materialize_daily_regime_targets",
]
