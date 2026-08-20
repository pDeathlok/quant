"""Shared contracts for leakage-resistant unified right-side model research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    roc_auc_score,
)

from quant.research.validation import purge_overlapping_training_events

EntryMode = Literal["next_open", "next_close"]

RIGHT_SIDE_SIGNALS: tuple[str, ...] = (
    "B2",
    "B3",
    "KEY_K",
    "VIOLENCE_K",
    "PINGHANG",
    "DOUBLE_GUN",
    "CHANGAN",
    "KENGQI",
    "VEGAS",
    "TRIPLE_VOLUME_BREAKOUT",
    "GOLDEN_BOWL",
    "ZAIHOU",
    "BREATHING",
    "YUEYUE",
)

RIGHT_ONLY_SIGNALS: frozenset[str] = frozenset(
    {
        "B2",
        "B3",
        "KEY_K",
        "VIOLENCE_K",
        "PINGHANG",
        "DOUBLE_GUN",
        "CHANGAN",
        "KENGQI",
        "VEGAS",
        "TRIPLE_VOLUME_BREAKOUT",
    }
)
MIXED_SIGNALS: frozenset[str] = frozenset(RIGHT_SIDE_SIGNALS) - RIGHT_ONLY_SIGNALS

B2_SOURCE_COLUMNS: tuple[str, ...] = (
    "b2_any_pchg4_vol15",
    "b2_oversold_pchg3_vol12",
    "b2_bbi_reclaim_vol12",
    "b2_pchg4_vol15",
)
B3_SOURCE_COLUMNS: tuple[str, ...] = (
    "b3_broad_small_pos",
    "b3_broad_calm_pullback",
    "b3_small_pos_amp7",
)
FAMILY_DIRECT_COLUMNS: Mapping[str, str] = {
    "VEGAS": "signal_vegas_tunnel",
    "TRIPLE_VOLUME_BREAKOUT": "signal_tvb_merged",
}
Z_DIRECT_SIGNALS: tuple[str, ...] = tuple(
    signal
    for signal in RIGHT_SIDE_SIGNALS
    if signal not in {"B2", "B3", *FAMILY_DIRECT_COLUMNS}
)


@dataclass(frozen=True)
class YearFold:
    """One expanding train/validation/test split with explicit calendar years."""

    name: str
    train_start_year: int
    train_end_year: int
    validation_year: int
    test_year: int


DEFAULT_YEAR_FOLDS: tuple[YearFold, ...] = (
    YearFold("A", 2020, 2022, 2023, 2024),
    YearFold("B", 2020, 2023, 2024, 2025),
    YearFold("C", 2020, 2024, 2025, 2026),
)


@dataclass(frozen=True)
class SplitFrames:
    """Chronological frames after purging labels that cross a boundary."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class Float32NaNTransformer:
    """Sklearn-compatible no-imputation transformer for XGBoost research.

    XGBoost handles missing values natively.  Keeping them as NaN avoids a
    large float64 imputation copy for the million-row pooled sample while still
    giving :class:`~quant.ml.xgb_research.XGBResearchModel` a persisted,
    deterministic transform object.
    """

    def fit(self, values: object, labels: object | None = None) -> "Float32NaNTransformer":
        del values, labels
        return self

    def transform(self, values: object) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            array = values.to_numpy(dtype=np.float32, na_value=np.nan)
        else:
            array = np.asarray(values, dtype=np.float32)
        array = np.array(array, dtype=np.float32, copy=True)
        array[~np.isfinite(array)] = np.nan
        return array

    def fit_transform(self, values: object, labels: object | None = None) -> np.ndarray:
        return self.fit(values, labels).transform(values)


@dataclass
class ProbabilityCalibratedModel:
    """Persist a research model with a validation-fitted Platt calibrator."""

    base_model: object
    calibrator: object

    @property
    def feature_names_in_(self) -> list[str]:
        return list(getattr(self.base_model, "feature_names_in_"))

    @property
    def selected_features_(self) -> list[str]:
        return list(getattr(self.base_model, "selected_features_"))

    @property
    def factor_schema_version_(self) -> str | None:
        return getattr(self.base_model, "factor_schema_version_", None)

    def predict_proba(self, values: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.base_model.predict_proba(values), dtype=float)[:, 1]
        raw = np.clip(raw, 1e-6, 1.0 - 1e-6)
        logit = np.log(raw / (1.0 - raw)).reshape(-1, 1)
        calibrated = np.asarray(self.calibrator.predict_proba(logit), dtype=float)[:, 1]
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, values: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(values)[:, 1] >= 0.5).astype(int)


def _read_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    available = set(pd.read_parquet(path, columns=[]).columns)
    # pandas/pyarrow returns no schema columns for columns=[] on some versions.
    if not available:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
    missing = set(columns) - available
    if missing:
        raise ValueError(f"{path} missing signal columns: {sorted(missing)}")
    return pd.read_parquet(path, columns=list(columns))


def load_signal_universe(
    z_skill_cache: Path | str,
    family_cache: Path | str,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load and OR-aggregate all 14 production right/mixed signal members.

    The historical training loader used ``drop_duplicates(keep='last')`` and
    therefore erased overlapping signals.  This function intentionally performs
    one boolean OR per symbol/date and retains the complete multi-hot identity.
    """

    z_path = Path(z_skill_cache)
    family_path = Path(family_cache)
    z = _read_columns(z_path, ("symbol", "date", *Z_DIRECT_SIGNALS))
    family_source = (
        "symbol",
        "date",
        *B2_SOURCE_COLUMNS,
        *B3_SOURCE_COLUMNS,
        *FAMILY_DIRECT_COLUMNS.values(),
    )
    family = _read_columns(family_path, family_source)

    z["date"] = pd.to_datetime(z["date"], errors="coerce")
    family["date"] = pd.to_datetime(family["date"], errors="coerce")
    z = z.dropna(subset=["symbol", "date"])
    family = family.dropna(subset=["symbol", "date"])

    family_out = family[["symbol", "date"]].copy()
    family_out["B2"] = family[list(B2_SOURCE_COLUMNS)].fillna(False).astype(bool).any(axis=1)
    family_out["B3"] = family[list(B3_SOURCE_COLUMNS)].fillna(False).astype(bool).any(axis=1)
    for target, source in FAMILY_DIRECT_COLUMNS.items():
        family_out[target] = family[source].fillna(False).astype(bool)
    for source in (*B2_SOURCE_COLUMNS, *B3_SOURCE_COLUMNS):
        family_out[source] = family[source].fillna(False).astype(bool)

    merged = z.merge(family_out, on=["symbol", "date"], how="outer")
    boolean_columns = [
        *RIGHT_SIDE_SIGNALS,
        *B2_SOURCE_COLUMNS,
        *B3_SOURCE_COLUMNS,
    ]
    for column in boolean_columns:
        if column not in merged.columns:
            merged[column] = False
        merged[column] = merged[column].fillna(False).astype(bool)

    aggregations = {column: "max" for column in boolean_columns}
    merged = (
        merged.groupby(["symbol", "date"], as_index=False, sort=False)
        .agg(aggregations)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    merged = merged[merged[list(RIGHT_SIDE_SIGNALS)].any(axis=1)].copy()
    if start_date is not None:
        merged = merged[merged["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        merged = merged[merged["date"] <= pd.Timestamp(end_date)]
    merged["signal_count"] = merged[list(RIGHT_SIDE_SIGNALS)].sum(axis=1).astype("int16")
    merged["has_right_signal"] = merged[list(RIGHT_ONLY_SIGNALS)].any(axis=1)
    merged["has_mixed_signal"] = merged[list(MIXED_SIGNALS)].any(axis=1)
    return merged.reset_index(drop=True)


def split_by_year_fold(
    data: pd.DataFrame,
    fold: YearFold,
    *,
    date_column: str = "date",
    label_end_column: str = "label_end_date",
) -> SplitFrames:
    """Build one train/validation/test fold and purge crossing label windows."""

    required = {date_column, label_end_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"dataset missing split columns: {sorted(missing)}")
    # Keep normalized timestamps as lightweight Series rather than copying the
    # complete (potentially million-row, hundreds-of-columns) model frame.
    # Each returned subset is copied below, preserving the caller's data.
    dates = pd.to_datetime(data[date_column], errors="coerce")
    label_ends = pd.to_datetime(data[label_end_column], errors="coerce")
    if dates.isna().any() or label_ends.isna().any():
        raise ValueError("date and label_end_date must be mature valid timestamps")

    validation_start = pd.Timestamp(f"{fold.validation_year}-01-01")
    test_start = pd.Timestamp(f"{fold.test_year}-01-01")
    test_end = pd.Timestamp(f"{fold.test_year}-12-31")
    train_mask = dates.dt.year.between(fold.train_start_year, fold.train_end_year)
    train = data.loc[train_mask].copy()
    train[date_column] = dates.loc[train_mask].to_numpy()
    train[label_end_column] = label_ends.loc[train_mask].to_numpy()
    train = purge_overlapping_training_events(
        train,
        test_start=validation_start,
        label_end_column=label_end_column,
    )
    validation_mask = dates.dt.year.eq(fold.validation_year)
    validation = data.loc[validation_mask].copy()
    validation[date_column] = dates.loc[validation_mask].to_numpy()
    validation[label_end_column] = label_ends.loc[validation_mask].to_numpy()
    validation = purge_overlapping_training_events(
        validation,
        test_start=test_start,
        label_end_column=label_end_column,
    )
    test_mask = dates.between(test_start, test_end, inclusive="both")
    test = data.loc[test_mask].copy()
    test[date_column] = dates.loc[test_mask].to_numpy()
    test[label_end_column] = label_ends.loc[test_mask].to_numpy()
    return SplitFrames(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def balanced_sample_weights(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    signal_columns: Sequence[str] = RIGHT_SIDE_SIGNALS,
) -> pd.Series:
    """Return weights balanced by trade date and inverse signal frequency.

    Multi-hit rows receive the mean inverse frequency of their active signals,
    so they are not counted once per signal.  The result is normalized to mean 1.
    """

    missing = {date_column, *signal_columns} - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing weight columns: {sorted(missing)}")
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    active = frame[list(signal_columns)].fillna(False).astype(bool)
    counts = active.sum(axis=0).replace(0, np.nan)
    inverse = 1.0 / counts
    signal_weight = active.mul(inverse, axis=1).sum(axis=1) / active.sum(axis=1).replace(0, np.nan)
    date_count = frame.groupby(pd.to_datetime(frame[date_column]).dt.normalize())[date_column].transform("size")
    date_weight = 1.0 / date_count.replace(0, np.nan)
    weight = (signal_weight * date_weight).replace([np.inf, -np.inf], np.nan)
    weight = weight.fillna(weight.dropna().median() if weight.notna().any() else 1.0)
    weight = (weight / weight.mean()).clip(lower=0.5, upper=3.0)
    return (weight / weight.mean()).astype(float)


def daily_top_k_trading_metrics(
    frame: pd.DataFrame,
    probabilities: Sequence[float],
    *,
    return_column: str = "terminal_return",
    date_column: str = "date",
    top_k: int = 10,
    round_trip_cost_bps: float = 15.0,
) -> dict[str, float | int]:
    """Evaluate trade-level outcomes for the highest-ranked daily events.

    One candidate row is expected per symbol/date for the selected entry mode
    and horizon.  The statistics deliberately do *not* compound horizon returns
    on their signal dates: events overlap and a terminal-label table cannot
    produce a valid marked-to-market capital curve.  Portfolio return and
    drawdown therefore require a separate position-aware backtest.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be non-negative")
    if len(frame) != len(probabilities):
        raise ValueError("prediction length does not match frame")
    missing = {date_column, return_column} - set(frame.columns)
    if missing:
        raise ValueError(f"trading frame missing columns: {sorted(missing)}")
    work = frame[[date_column, return_column]].copy()
    work["_probability"] = np.asarray(probabilities, dtype=float)
    work[return_column] = pd.to_numeric(work[return_column], errors="coerce")
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[date_column, return_column, "_probability"])
    if work.empty:
        return {"trades": 0, "trading_days": 0}
    selected = (
        work.sort_values([date_column, "_probability"], ascending=[True, False], kind="stable")
        .groupby(date_column, sort=True, group_keys=False)
        .head(top_k)
        .copy()
    )
    cost = float(round_trip_cost_bps) / 10_000.0
    selected["_net_return"] = selected[return_column] - cost
    positive = selected.loc[selected["_net_return"] > 0, "_net_return"].sum()
    negative = -selected.loc[selected["_net_return"] < 0, "_net_return"].sum()
    selected_dates = selected[date_column].nunique()
    return {
        "trades": int(len(selected)),
        "trading_days": int(selected_dates),
        "average_net_return": float(selected["_net_return"].mean()),
        "median_net_return": float(selected["_net_return"].median()),
        "win_rate": float(selected["_net_return"].gt(0).mean()),
        "profit_factor": float(positive / negative) if negative > 0 else np.inf,
        "top_k": int(top_k),
        "round_trip_cost_bps": float(round_trip_cost_bps),
    }


def choose_validation_threshold(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    minimum_selection_rate: float = 0.02,
    maximum_selection_rate: float = 0.20,
) -> float:
    """Choose a probability threshold using validation precision and coverage only."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if len(y) == 0:
        raise ValueError("cannot choose a threshold from an empty validation set")
    candidates = np.unique(np.quantile(p, np.linspace(0.80, 0.98, 19)))
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        selected = p >= threshold
        rate = float(selected.mean())
        if rate < minimum_selection_rate or rate > maximum_selection_rate or not selected.any():
            continue
        precision = float(y[selected].mean())
        lift = precision / max(float(y.mean()), 1e-12)
        score = (lift, precision, -rate)
        if best is None or score > best[:3]:
            best = (*score, float(threshold))
    if best is None:
        return float(np.quantile(p, 0.90))
    return float(best[3])


def binary_metrics(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    threshold: float | None = None,
    top_fraction: float = 0.10,
) -> dict[str, float | int]:
    """Calculate ranking, calibration, and fixed-selection binary metrics."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if len(y) == 0:
        return {"rows": 0}
    base_rate = float(y.mean())
    top_n = max(1, int(np.ceil(len(y) * top_fraction)))
    top_idx = np.argsort(-p, kind="stable")[:top_n]
    top_precision = float(y[top_idx].mean())
    if np.allclose(p, p[0]):
        top_precision = base_rate
    result: dict[str, float | int] = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": base_rate,
        "average_precision": float(average_precision_score(y, p)) if np.unique(y).size > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "top_fraction": float(top_fraction),
        "top_precision": top_precision,
        "top_lift": top_precision / max(base_rate, 1e-12),
    }
    if threshold is not None:
        selected = p >= threshold
        result.update(
            {
                "threshold": float(threshold),
                "selection_rate": float(selected.mean()),
                "selected": int(selected.sum()),
                "threshold_precision": float(precision_score(y, selected, zero_division=0)),
            }
        )
    return result


def signal_metrics(
    frame: pd.DataFrame,
    probabilities: Sequence[float],
    *,
    label_column: str,
    threshold: float | None = None,
    signal_columns: Sequence[str] = RIGHT_SIDE_SIGNALS,
) -> pd.DataFrame:
    """Evaluate the same predictions on every active member slice."""

    if len(frame) != len(probabilities):
        raise ValueError("prediction length does not match frame")
    rows: list[dict[str, object]] = []
    values = np.asarray(probabilities, dtype=float)
    for signal in signal_columns:
        if signal not in frame.columns:
            continue
        mask = frame[signal].fillna(False).astype(bool).to_numpy()
        metrics = binary_metrics(
            frame.loc[mask, label_column].astype(int),
            values[mask],
            threshold=threshold,
        )
        rows.append({"signal": signal, **metrics})
    return pd.DataFrame(rows)


def aggregate_independent_predictions(
    frame: pd.DataFrame,
    predictions_by_signal: Mapping[str, Sequence[float]],
    *,
    signal_columns: Sequence[str] = RIGHT_SIDE_SIGNALS,
) -> np.ndarray:
    """Combine per-signal model predictions without duplicating multi-hit rows.

    Each independent model is only eligible on rows that hit its signal.  The
    maximum score is used as the operational candidate score, matching the idea
    that any one strategy model may approve a multi-hit opportunity.
    """

    matrix = np.full((len(frame), len(signal_columns)), np.nan, dtype=float)
    for position, signal in enumerate(signal_columns):
        if signal not in predictions_by_signal or signal not in frame.columns:
            continue
        prediction = np.asarray(predictions_by_signal[signal], dtype=float)
        if len(prediction) != len(frame):
            raise ValueError(f"prediction length mismatch for {signal}")
        active = frame[signal].fillna(False).astype(bool).to_numpy()
        matrix[active, position] = prediction[active]
    eligible = np.isfinite(matrix).any(axis=1)
    combined = np.full(len(frame), np.nan, dtype=float)
    if eligible.any():
        combined[eligible] = np.nanmax(matrix[eligible], axis=1)
    return combined


def ensure_model_features(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    minimum_coverage: float = 0.01,
) -> list[str]:
    """Validate and return numeric, populated model feature columns."""

    requested = list(dict.fromkeys(feature_columns))
    missing = set(requested) - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing model features: {sorted(missing)}")
    admitted: list[str] = []
    rejected: list[str] = []
    for column in requested:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().mean() < minimum_coverage or values.nunique(dropna=True) < 2:
            rejected.append(column)
        else:
            admitted.append(column)
    if not admitted:
        raise ValueError(f"no usable model features; rejected={rejected}")
    return admitted
