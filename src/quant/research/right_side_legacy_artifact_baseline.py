"""Overlap-only baseline for the legacy per-signal Z-skill artifacts.

The June-2026 ``models/research/z_skill`` artifacts are still consumed by the
daily selector, but they are not directly comparable with the unified
right-side research models:

* they require the legacy ``project-v1-latest-scale-global-rank`` factors;
* they were trained on the historical signal cache (including its five-session
  de-duplication and now-drifted detector predicates);
* their labels did not reject next-session locked limit-ups, did not use a
  market calendar, and encoded incomplete future tails as negatives;
* dates before 2025 participated in fitting or early stopping, while 2025+
  was an OOT report set subsequently used to choose operational playbooks.

This module therefore never scores a legacy artifact on the identically named
current causal factors.  It joins canonical events to the persisted legacy
factor dataset by ``symbol/date`` and scores only that intersection.  Coverage
and exact historical-signal timing overlap are first-class outputs so the
result cannot silently masquerade as a full, clean walk-forward baseline.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost.callback import TrainingCallback

from quant.features.project_factor_layer import (
    LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION,
)


LEGACY_Z_ARTIFACT_SIGNALS: tuple[str, ...] = (
    "B2",
    "BREATHING",
    "GOLDEN_BOWL",
    "KEY_K",
    "VIOLENCE_K",
    "YUEYUE",
    "ZAIHOU",
)

LEGACY_ARTIFACT_LABELS: tuple[str, ...] = ("up5", "up8", "down3")

LEGACY_LABEL_CONTRACT: Mapping[str, str] = {
    "up5": (
        "T+1 local-row open to maximum high over local rows T+2..T+6 >=5%; "
        "near-equivalent to next_open/horizon=5 hit_up5 only after the new "
        "maturity and tradability gates"
    ),
    "up8": (
        "T+1 local-row open to maximum high over local rows T+2..T+6 >=8%; "
        "near-equivalent to next_open/horizon=5 hit_up8 only after the new "
        "maturity and tradability gates"
    ),
    "down3": (
        "minimum low over local rows T+2..T+6 at least 3% below signal-day "
        "low; incompatible with the new entry-price MAE/hit_down3 label"
    ),
}

PRIMARY_NEW_TARGET_BY_ARTIFACT: Mapping[str, str | None] = {
    "up5": "hit_up5",
    "up8": "hit_up8",
    "down3": None,
}


class _LegacyAucGapEarlyStopping(TrainingCallback):
    """Unpickle-only shim for artifacts that persisted a ``__main__`` class.

    Prediction never invokes the callback.  The historical training script was
    executed as ``__main__``, so without this alias otherwise healthy artifacts
    cannot be loaded from a library or notebook process.
    """


@contextmanager
def _legacy_main_callback_alias() -> Iterator[None]:
    import __main__

    sentinel = object()
    previous = getattr(__main__, "AucGapEarlyStopping", sentinel)
    if previous is sentinel:
        setattr(__main__, "AucGapEarlyStopping", _LegacyAucGapEarlyStopping)
    try:
        yield
    finally:
        if previous is sentinel:
            delattr(__main__, "AucGapEarlyStopping")


@dataclass(frozen=True)
class LegacyArtifactContract:
    signal: str
    model_label: str
    path: Path
    sha256: str
    modified_at: pd.Timestamp
    inferred_factor_schema: str
    raw_feature_count: int
    selected_feature_count: int
    best_iteration: int | None


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_legacy_z_artifacts(
    model_dir: Path | str,
    *,
    signals: Sequence[str] = LEGACY_Z_ARTIFACT_SIGNALS,
) -> tuple[dict[tuple[str, str], object], pd.DataFrame]:
    """Load and audit the 21 persisted per-signal artifacts.

    A missing schema attribute is interpreted as legacy v1 because that is the
    builder contract in effect for these artifacts.  An explicit non-legacy
    schema fails closed.
    """

    root = Path(model_dir)
    models: dict[tuple[str, str], object] = {}
    contracts: list[LegacyArtifactContract] = []
    with _legacy_main_callback_alias():
        for signal in signals:
            if signal not in LEGACY_Z_ARTIFACT_SIGNALS:
                raise ValueError(f"unsupported legacy Z-skill signal: {signal}")
            for model_label in LEGACY_ARTIFACT_LABELS:
                path = root / f"{signal}_{model_label}.joblib"
                if not path.exists():
                    raise FileNotFoundError(f"missing legacy artifact: {path}")
                model = joblib.load(path)
                features = list(getattr(model, "feature_names_in_", ()))
                if not features:
                    raise RuntimeError(f"legacy artifact declares no features: {path}")
                declared_schema = getattr(model, "factor_schema_version_", None)
                inferred_schema = (
                    str(declared_schema)
                    if declared_schema
                    else LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
                )
                if inferred_schema != LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"artifact is not a legacy-factor baseline: {path}; "
                        f"schema={inferred_schema}"
                    )
                selected = list(getattr(model, "selected_features_", ()))
                stat = path.stat()
                contracts.append(
                    LegacyArtifactContract(
                        signal=signal,
                        model_label=model_label,
                        path=path,
                        sha256=_file_sha256(path),
                        modified_at=pd.Timestamp(stat.st_mtime, unit="s"),
                        inferred_factor_schema=inferred_schema,
                        raw_feature_count=len(features),
                        selected_feature_count=len(selected),
                        best_iteration=(
                            int(model.best_iteration)
                            if getattr(model, "best_iteration", None) is not None
                            else None
                        ),
                    )
                )
                models[(signal, model_label)] = model
    return models, pd.DataFrame([vars(contract) for contract in contracts])


def required_legacy_features(models: Mapping[tuple[str, str], object]) -> list[str]:
    """Return the stable union of artifact input columns."""

    return list(
        dict.fromkeys(
            feature
            for model in models.values()
            for feature in getattr(model, "feature_names_in_", ())
        )
    )


def _normalise_keys(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not {"symbol", "date"} <= set(frame.columns):
        raise ValueError(f"{name} requires symbol/date columns")
    out = frame.copy()
    out["symbol"] = out["symbol"].astype("string").str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if out["symbol"].isna().any() or out["symbol"].eq("").any() or out["date"].isna().any():
        raise ValueError(f"{name} contains invalid symbol/date keys")
    return out


def build_legacy_overlap_rows(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    legacy_factor_rows: pd.DataFrame,
    models: Mapping[tuple[str, str], object],
    *,
    entry_mode: str = "next_open",
    horizon: int = 5,
    targets: Sequence[str] = ("hit_up5", "hit_up8", "good_path5"),
    signals: Sequence[str] = LEGACY_Z_ARTIFACT_SIGNALS,
) -> pd.DataFrame:
    """Create one row per executable canonical event and active legacy member.

    ``legacy_factor_rows`` must be the persisted legacy Z-skill model dataset,
    not the current unified event factors.  Rows without an exact key match are
    retained with ``legacy_factor_available=False`` for coverage accounting and
    are skipped by :func:`score_legacy_overlap_rows`.
    """

    events = _normalise_keys(events, name="events")
    labels = _normalise_keys(labels, name="labels")
    legacy = _normalise_keys(legacy_factor_rows, name="legacy_factor_rows")
    signals = tuple(signals)
    missing_signals = set(signals) - set(events.columns)
    if missing_signals:
        raise ValueError(f"events missing canonical signals: {sorted(missing_signals)}")
    required_label_columns = {
        "entry_mode",
        "horizon",
        "mature",
        "locked_limit_up",
        *targets,
    }
    missing_labels = required_label_columns - set(labels.columns)
    if missing_labels:
        raise ValueError(f"labels missing columns: {sorted(missing_labels)}")

    selected_labels = labels[
        labels["entry_mode"].astype(str).eq(entry_mode)
        & pd.to_numeric(labels["horizon"], errors="coerce").eq(int(horizon))
    ].copy()
    selected_labels = selected_labels[
        selected_labels["mature"].fillna(False).astype(bool)
        & ~selected_labels["locked_limit_up"].fillna(False).astype(bool)
    ]
    if selected_labels.duplicated(["symbol", "date"]).any():
        raise ValueError("labels are not unique after entry_mode/horizon selection")
    label_keep = [
        "symbol",
        "date",
        "entry_mode",
        "horizon",
        *targets,
        *[
            column
            for column in ("entry_date", "terminal_return", "mfe", "mae")
            if column in selected_labels.columns
        ],
    ]
    base = events[["symbol", "date", *signals]].merge(
        selected_labels[label_keep],
        on=["symbol", "date"],
        how="inner",
        validate="one_to_one",
    )

    active_parts: list[pd.DataFrame] = []
    base_keep = [
        "symbol",
        "date",
        "entry_mode",
        "horizon",
        *targets,
        *[column for column in ("entry_date", "terminal_return", "mfe", "mae") if column in base.columns],
    ]
    for signal in signals:
        active = base[signal].fillna(False).astype(bool)
        if active.any():
            part = base.loc[active, base_keep].copy()
            part["signal"] = signal
            active_parts.append(part)
    if not active_parts:
        return pd.DataFrame(columns=[*base_keep, "signal", "legacy_factor_available"])
    long = pd.concat(active_parts, ignore_index=True, sort=False)

    feature_columns = required_legacy_features(models)
    missing_features = set(feature_columns) - set(legacy.columns)
    if missing_features:
        raise ValueError(
            "legacy factor dataset is incompatible with artifacts; "
            f"missing features={sorted(missing_features)}"
        )
    legacy_signal_columns = [signal for signal in signals if signal in legacy.columns]
    legacy_keep = ["symbol", "date", *legacy_signal_columns, *feature_columns]
    legacy = legacy[legacy_keep].copy()
    if legacy.duplicated(["symbol", "date"]).any():
        raise ValueError("legacy factor dataset contains duplicate symbol/date rows")
    legacy = legacy.rename(
        columns={signal: f"legacy_signal__{signal}" for signal in legacy_signal_columns}
    )
    merged = long.merge(
        legacy,
        on=["symbol", "date"],
        how="left",
        validate="many_to_one",
        indicator="_legacy_join",
    )
    merged["legacy_factor_available"] = merged.pop("_legacy_join").eq("both")
    merged["legacy_signal_timing_match"] = False
    for signal in signals:
        column = f"legacy_signal__{signal}"
        mask = merged["signal"].eq(signal)
        if column in merged.columns:
            merged.loc[mask, "legacy_signal_timing_match"] = (
                merged.loc[mask, column].fillna(False).astype(bool).to_numpy()
            )
    merged["legacy_temporal_status"] = np.where(
        merged["date"].dt.year < 2025,
        "fit_or_internal_early_stop_period",
        "training_oot_but_seen_in_artifact_selection",
    )
    merged["fold"] = merged["date"].dt.year.map(
        {2024: "A", 2025: "B", 2026: "C"}
    ).fillna("outside_unified_test_folds")
    return merged.sort_values(["date", "symbol", "signal"], kind="stable").reset_index(drop=True)


def score_legacy_overlap_rows(
    overlap_rows: pd.DataFrame,
    models: Mapping[tuple[str, str], object],
) -> pd.DataFrame:
    """Score available overlap rows with the matching per-signal artifacts."""

    out = overlap_rows.copy()
    if "legacy_factor_available" not in out.columns:
        raise ValueError("overlap rows are missing legacy_factor_available")
    for model_label in LEGACY_ARTIFACT_LABELS:
        out[f"legacy_pred_{model_label}"] = np.nan
    for signal in LEGACY_Z_ARTIFACT_SIGNALS:
        active = out["signal"].eq(signal) & out["legacy_factor_available"].fillna(False)
        if not active.any():
            continue
        for model_label in LEGACY_ARTIFACT_LABELS:
            model = models.get((signal, model_label))
            if model is None:
                continue
            features = list(getattr(model, "feature_names_in_", ()))
            missing = set(features) - set(out.columns)
            if missing:
                raise ValueError(
                    f"overlap rows missing {signal}/{model_label} features: {sorted(missing)}"
                )
            out.loc[active, f"legacy_pred_{model_label}"] = model.predict_proba(
                out.loc[active, features]
            )[:, 1]
    required_predictions = [f"legacy_pred_{label}" for label in LEGACY_ARTIFACT_LABELS]
    out["legacy_scored"] = out[required_predictions].notna().all(axis=1)
    out["legacy_quality_score"] = (
        out["legacy_pred_up5"]
        + out["legacy_pred_up8"]
        + (1.0 - out["legacy_pred_down3"])
    ) / 3.0
    return out


def aggregate_legacy_event_predictions(scored_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-hit events by the strongest compatible member score."""

    scored = scored_rows[scored_rows["legacy_scored"].fillna(False)].copy()
    if scored.empty:
        return pd.DataFrame()
    key_columns = [
        column
        for column in (
            "symbol",
            "date",
            "entry_mode",
            "horizon",
            "fold",
            "legacy_temporal_status",
            "entry_date",
            "hit_up5",
            "hit_up8",
            "good_path5",
            "terminal_return",
            "mfe",
            "mae",
        )
        if column in scored.columns
    ]
    probability_columns = [
        "legacy_pred_up5",
        "legacy_pred_up8",
        "legacy_pred_down3",
        "legacy_quality_score",
    ]
    aggregate = scored.groupby(key_columns, as_index=False, dropna=False)[
        probability_columns
    ].max()
    details = scored.groupby(key_columns, as_index=False, dropna=False).agg(
        legacy_covered_signal_count=("signal", "nunique"),
        legacy_signals=("signal", lambda values: ",".join(sorted(set(values)))),
        legacy_signal_timing_match_any=("legacy_signal_timing_match", "max"),
    )
    return aggregate.merge(details, on=key_columns, validate="one_to_one")


def legacy_overlap_coverage(overlap_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize factor-row and exact historical signal timing overlap."""

    rows: list[dict[str, object]] = []
    for signal, part in overlap_rows.groupby("signal", sort=False):
        available = part["legacy_factor_available"].fillna(False).astype(bool)
        timing = part["legacy_signal_timing_match"].fillna(False).astype(bool)
        rows.append(
            {
                "signal": signal,
                "canonical_rows": int(len(part)),
                "legacy_factor_rows": int(available.sum()),
                "legacy_factor_coverage": float(available.mean()) if len(part) else np.nan,
                "legacy_signal_timing_match_rows": int((available & timing).sum()),
                "legacy_signal_timing_match_rate": (
                    float(timing.loc[available].mean()) if available.any() else np.nan
                ),
                "date_min": part["date"].min(),
                "date_max": part["date"].max(),
            }
        )
    return pd.DataFrame(rows)


def _binary_metrics(labels: pd.Series, probability: pd.Series, *, top_fraction: float) -> dict[str, float | int]:
    valid = labels.notna() & probability.notna()
    y = labels.loc[valid].astype(bool).astype(int)
    p = pd.to_numeric(probability.loc[valid], errors="coerce")
    finite = np.isfinite(p.to_numpy(dtype=float))
    y = y.loc[finite]
    p = p.loc[finite]
    if y.empty:
        return {
            "rows": 0,
            "positive_rate": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "brier": np.nan,
            "top_precision": np.nan,
            "top_lift": np.nan,
        }
    base = float(y.mean())
    top_n = max(1, int(np.ceil(len(y) * top_fraction)))
    top = y.loc[p.sort_values(ascending=False, kind="stable").index[:top_n]]
    top_precision = float(top.mean())
    return {
        "rows": int(len(y)),
        "positive_rate": base,
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if y.nunique() == 2 else np.nan,
        "brier": float(brier_score_loss(y, p.clip(0.0, 1.0))),
        "top_precision": top_precision,
        "top_lift": top_precision / base if base > 0 else np.nan,
    }


def evaluate_legacy_event_predictions(
    event_predictions: pd.DataFrame,
    *,
    targets: Sequence[str] = ("hit_up5", "hit_up8", "good_path5"),
    top_fraction: float = 0.10,
) -> pd.DataFrame:
    """Evaluate raw legacy probabilities and the fixed untuned quality score."""

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    score_columns = (
        "legacy_pred_up5",
        "legacy_pred_up8",
        "legacy_pred_down3",
        "legacy_quality_score",
    )
    rows: list[dict[str, object]] = []
    group_columns = [column for column in ("fold", "entry_mode", "horizon") if column in event_predictions.columns]
    grouped = event_predictions.groupby(group_columns, dropna=False, sort=False) if group_columns else [((), event_predictions)]
    for group_key, part in grouped:
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_columns, values))
        for target in targets:
            if target not in part.columns:
                continue
            for score_column in score_columns:
                model_label = score_column.removeprefix("legacy_pred_")
                if score_column == "legacy_quality_score":
                    semantic_match = "diagnostic_composite_no_retuning"
                elif PRIMARY_NEW_TARGET_BY_ARTIFACT.get(model_label) == target:
                    semantic_match = "near_equivalent_after_new_sample_gates"
                elif model_label == "down3" and target == "hit_down3":
                    semantic_match = "incompatible_label_do_not_compare"
                else:
                    semantic_match = "cross_target_diagnostic_only"
                rows.append(
                    {
                        **group_values,
                        "target": target,
                        "score": score_column,
                        "semantic_match": semantic_match,
                        **_binary_metrics(part[target], part[score_column], top_fraction=top_fraction),
                    }
                )
    return pd.DataFrame(rows)


__all__ = [
    "LEGACY_ARTIFACT_LABELS",
    "LEGACY_LABEL_CONTRACT",
    "LEGACY_Z_ARTIFACT_SIGNALS",
    "PRIMARY_NEW_TARGET_BY_ARTIFACT",
    "aggregate_legacy_event_predictions",
    "build_legacy_overlap_rows",
    "evaluate_legacy_event_predictions",
    "legacy_overlap_coverage",
    "load_legacy_z_artifacts",
    "required_legacy_features",
    "score_legacy_overlap_rows",
]
