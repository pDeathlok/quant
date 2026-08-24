"""Narrow, versioned datasets for the right-side playbook layer.

The event table owns all T-close factors and the frozen first-layer OOS score.
The outcome table owns only event keys, action parameters, T+1 execution facts,
and realized outcomes.  Keeping those tables separate prevents the 8 trade
actions plus ``NO_TRADE`` from multiplying every wide factor row nine times on
disk.

This module is research-only.  It deliberately rejects fold C so the current
development experiment remains A (earlier first-layer OOS scores) -> B (test).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.right_side_playbook_model import (
    EVENT_KEY_COLUMNS,
    FIRST_LAYER_FOLD_COLUMN,
    FIRST_LAYER_PROVENANCE_COLUMN,
    FIRST_LAYER_SCORE_COLUMN,
    validate_first_layer_score_contract,
)
from quant.research.right_side_playbook_policy import (
    DEFAULT_PLAYBOOK_CATALOG,
    NO_TRADE_PLAYBOOK_ID,
    PLAYBOOK_POLICY_VERSION,
    playbook_catalog_hash,
)
from quant.research.right_side_unified import (
    B2_SOURCE_COLUMNS,
    B3_SOURCE_COLUMNS,
    DEFAULT_YEAR_FOLDS,
    RIGHT_SIDE_SIGNALS,
)
from quant.research.right_side_unified_features import (
    RULE_FEATURE_COLUMNS,
    RULE_FEATURE_COLUMNS_SHA256,
    RULE_FEATURE_SCHEMA_VERSION,
)


PLAYBOOK_DATASET_SCHEMA_VERSION = "right-side-playbook-narrow-v2-118"
PLAYBOOK_DEVELOPMENT_FOLDS: tuple[str, str] = ("A", "B")
PLAYBOOK_TRAIN_FOLD = "A"
PLAYBOOK_EVALUATION_FOLD = "B"

SUBVARIANT_COLUMNS: tuple[str, ...] = (*B2_SOURCE_COLUMNS, *B3_SOURCE_COLUMNS)
IDENTITY_COLUMNS: tuple[str, ...] = (
    *RIGHT_SIDE_SIGNALS,
    *SUBVARIANT_COLUMNS,
    "signal_count",
    "has_right_signal",
    "has_mixed_signal",
)
EVENT_FACTOR_COLUMNS: tuple[str, ...] = (
    *PROJECT_FACTOR_COLUMNS,
    *RULE_FEATURE_COLUMNS,
    *IDENTITY_COLUMNS,
)

PLAYBOOK_OUTCOME_COLUMNS: tuple[str, ...] = (
    "fold",
    "event_id",
    "symbol",
    "date",
    "playbook_id",
    "playbook_version",
    "entry_mode",
    "exit_policy_id",
    "eligible",
    "eligibility_reason",
    "mature",
    "maturity_reason",
    "entry_date",
    "entry_price",
    "entry_raw_price",
    "exit_date",
    "exit_price",
    "exit_raw_price",
    "exit_reason",
    "gross_return",
    "net_return",
    "mae",
    "holding_sessions",
    "ambiguous_bar",
    "round_trip_cost",
    "round_trip_cost_bps",
    "locked_limit_up",
    "locked_limit_source",
    "open_at_up_limit",
    "close_at_up_limit",
    "open_gap",
)

_OUTCOME_STRING_COLUMNS = frozenset(
    {
        "fold",
        "event_id",
        "symbol",
        "playbook_id",
        "playbook_version",
        "entry_mode",
        "exit_policy_id",
        "eligibility_reason",
        "maturity_reason",
        "exit_reason",
        "locked_limit_source",
    }
)
_OUTCOME_DATE_COLUMNS = frozenset({"date", "entry_date", "exit_date"})
_OUTCOME_BOOL_COLUMNS = frozenset(
    {
        "eligible",
        "mature",
        "ambiguous_bar",
        "locked_limit_up",
        "open_at_up_limit",
        "close_at_up_limit",
    }
)
_OUTCOME_FLOAT_COLUMNS = frozenset(PLAYBOOK_OUTCOME_COLUMNS) - (
    _OUTCOME_STRING_COLUMNS
    | _OUTCOME_DATE_COLUMNS
    | _OUTCOME_BOOL_COLUMNS
    | {"holding_sessions"}
)


@dataclass(frozen=True)
class FirstLayerPredictionContract:
    """Exact first-layer score slice admitted into the second layer."""

    score_column: str = "pred_unified_long_task_deep"
    entry_mode: str = "next_close"
    horizon: int = 5
    label: str = "good_path5"
    train_fold: str = PLAYBOOK_TRAIN_FOLD
    evaluation_fold: str = PLAYBOOK_EVALUATION_FOLD
    selected_candidate: str | None = None

    def __post_init__(self) -> None:
        if not self.score_column:
            raise ValueError("score_column cannot be empty")
        if self.selected_candidate is not None:
            expected_score_column = f"pred_{self.selected_candidate}"
            if self.score_column != expected_score_column:
                raise ValueError(
                    "selected first-layer candidate does not match score column: "
                    f"candidate={self.selected_candidate} "
                    f"score_column={self.score_column}"
                )
        if self.entry_mode not in {"next_open", "next_close"}:
            raise ValueError(f"unsupported first-layer entry mode: {self.entry_mode}")
        if self.horizon <= 0:
            raise ValueError("first-layer horizon must be positive")
        if (self.train_fold, self.evaluation_fold) != PLAYBOOK_DEVELOPMENT_FOLDS:
            raise ValueError("playbook development is frozen to A -> B; fold C is forbidden")


@dataclass
class OutcomeBuildAudit:
    """Streaming audit state for event-by-action outcome generation."""

    event_rows: int = 0
    outcome_rows: int = 0
    eligible_regular_rows: int = 0
    mature_regular_rows: int = 0
    ambiguous_rows: int = 0
    locked_limit_rows: int = 0
    duplicate_action_keys: int = 0
    action_counts: Counter[str] | None = None
    eligibility_reasons: Counter[str] | None = None
    maturity_reasons: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.action_counts is None:
            self.action_counts = Counter()
        if self.eligibility_reasons is None:
            self.eligibility_reasons = Counter()
        if self.maturity_reasons is None:
            self.maturity_reasons = Counter()

    def update(self, outcomes: pd.DataFrame) -> None:
        _require_columns(
            outcomes,
            {"fold", "event_id", "playbook_id", "eligible", "mature"},
            name="playbook outcomes",
        )
        self.outcome_rows += int(len(outcomes))
        self.event_rows += int(
            outcomes[list(EVENT_KEY_COLUMNS)].drop_duplicates().shape[0]
        )
        self.duplicate_action_keys += int(
            outcomes.duplicated([*EVENT_KEY_COLUMNS, "playbook_id"]).sum()
        )
        regular = ~outcomes["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
        eligible = outcomes["eligible"].fillna(False).astype(bool)
        mature = outcomes["mature"].fillna(False).astype(bool)
        self.eligible_regular_rows += int((regular & eligible).sum())
        self.mature_regular_rows += int((regular & eligible & mature).sum())
        if "ambiguous_bar" in outcomes:
            self.ambiguous_rows += int(
                outcomes["ambiguous_bar"].fillna(False).astype(bool).sum()
            )
        if "locked_limit_up" in outcomes:
            self.locked_limit_rows += int(
                outcomes["locked_limit_up"].fillna(False).astype(bool).sum()
            )
        assert self.action_counts is not None
        assert self.eligibility_reasons is not None
        assert self.maturity_reasons is not None
        self.action_counts.update(
            outcomes["playbook_id"].astype(str).value_counts().to_dict()
        )
        if "eligibility_reason" in outcomes:
            self.eligibility_reasons.update(
                outcomes["eligibility_reason"].astype(str).value_counts().to_dict()
            )
        if "maturity_reason" in outcomes:
            self.maturity_reasons.update(
                outcomes["maturity_reason"].astype(str).value_counts().to_dict()
            )

    def finalize(self, *, expected_events: int) -> dict[str, Any]:
        action_ids = [spec.playbook_id for spec in DEFAULT_PLAYBOOK_CATALOG]
        expected_rows = expected_events * len(action_ids)
        regular_rows = expected_events * (len(action_ids) - 1)
        failures: list[str] = []
        if self.event_rows != expected_events:
            failures.append(
                f"stream event count {self.event_rows} != expected {expected_events}"
            )
        if self.outcome_rows != expected_rows:
            failures.append(
                f"outcome rows {self.outcome_rows} != expected {expected_rows}"
            )
        if self.duplicate_action_keys:
            failures.append(f"duplicate action keys={self.duplicate_action_keys}")
        assert self.action_counts is not None
        for playbook_id in action_ids:
            if self.action_counts[playbook_id] != expected_events:
                failures.append(
                    f"action {playbook_id} count={self.action_counts[playbook_id]} "
                    f"expected={expected_events}"
                )
        unexpected = set(self.action_counts) - set(action_ids)
        if unexpected:
            failures.append(f"unexpected actions={sorted(unexpected)}")
        if failures:
            raise ValueError("playbook outcome streaming audit failed: " + "; ".join(failures))
        ineligible_regular_rows = regular_rows - self.eligible_regular_rows
        eligible_immature_regular_rows = (
            self.eligible_regular_rows - self.mature_regular_rows
        )
        known_utility_regular_rows = (
            ineligible_regular_rows + self.mature_regular_rows
        )
        return {
            "event_rows": int(expected_events),
            "outcome_rows": int(self.outcome_rows),
            "actions_per_event": len(action_ids),
            "action_counts": dict(self.action_counts),
            "eligible_regular_rows": int(self.eligible_regular_rows),
            "ineligible_cancel_zero_utility_rows": int(ineligible_regular_rows),
            "mature_regular_rows": int(self.mature_regular_rows),
            "eligible_immature_null_utility_rows": int(
                eligible_immature_regular_rows
            ),
            "known_utility_regular_rows": int(known_utility_regular_rows),
            "known_utility_regular_coverage": (
                float(known_utility_regular_rows / regular_rows)
                if regular_rows
                else np.nan
            ),
            "mature_regular_coverage": (
                float(self.mature_regular_rows / self.eligible_regular_rows)
                if self.eligible_regular_rows
                else np.nan
            ),
            "ambiguous_rows": int(self.ambiguous_rows),
            "locked_limit_rows": int(self.locked_limit_rows),
            "eligibility_reasons": dict(self.eligibility_reasons or {}),
            "maturity_reasons": dict(self.maturity_reasons or {}),
        }


class StreamingParquetWriter:
    """Append homogeneous frames and atomically publish one parquet file."""

    def __init__(self, path: Path, *, compression: str = "zstd") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        self.compression = compression
        self._writer: Any | None = None
        self._schema: Any | None = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                self.temp_path,
                self._schema,
                compression=self.compression,
            )
        else:
            if table.column_names != self._schema.names:
                raise ValueError("streaming parquet columns changed between batches")
            if not table.schema.equals(self._schema, check_metadata=False):
                table = table.cast(self._schema)
        self._writer.write_table(table)
        self.rows += int(len(frame))

    def close(self, *, commit: bool) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if commit:
            if self.rows <= 0 or not self.temp_path.is_file():
                raise ValueError("cannot publish an empty streaming parquet file")
            os.replace(self.temp_path, self.path)
        else:
            self.temp_path.unlink(missing_ok=True)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _normalize_dates(values: pd.Series, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        samples = values.loc[parsed.isna()].astype(str).head(3).tolist()
        raise ValueError(f"{name} contains invalid dates: {samples}")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize()


def stable_event_ids(symbol: pd.Series, date: pd.Series) -> pd.Series:
    """Return the stable cross-table event identifier."""

    return (
        symbol.astype("string").str.strip()
        + "|"
        + _normalize_dates(date, name="event date").dt.strftime("%Y%m%d")
    ).astype("string")


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a versioned source or output artifact without loading it in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_first_layer_predictions(
    path: Path,
    *,
    contract: FirstLayerPredictionContract = FirstLayerPredictionContract(),
) -> pd.DataFrame:
    """Read only the frozen A/B first-layer score slice; fold C is never read."""

    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    required = {
        "symbol",
        "date",
        "fold",
        "entry_mode",
        "horizon",
        "label",
        contract.score_column,
    }
    missing = required - available
    if missing:
        raise ValueError(f"first-layer predictions missing columns: {sorted(missing)}")
    columns = list(required)
    predictions = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("entry_mode", "==", contract.entry_mode),
            ("horizon", "==", contract.horizon),
            ("label", "==", contract.label),
            ("fold", "in", list(PLAYBOOK_DEVELOPMENT_FOLDS)),
        ],
    )
    if predictions.empty:
        raise ValueError("first-layer A/B prediction slice is empty")
    for column, expected in (
        ("entry_mode", contract.entry_mode),
        ("horizon", contract.horizon),
        ("label", contract.label),
    ):
        if set(predictions[column].dropna().astype(str)) != {str(expected)}:
            raise ValueError(f"first-layer {column} slice drifted from {expected}")
    predictions["symbol"] = predictions["symbol"].astype("string").str.strip()
    predictions["date"] = _normalize_dates(predictions["date"], name="prediction date")
    predictions["fold"] = predictions["fold"].astype("string")
    if set(predictions["fold"]) != set(PLAYBOOK_DEVELOPMENT_FOLDS):
        raise ValueError("first-layer predictions must contain exactly folds A and B")
    if predictions.duplicated(["fold", "symbol", "date"]).any():
        raise ValueError("first-layer prediction keys are duplicated")
    if predictions.duplicated(["symbol", "date"]).any():
        raise ValueError("one symbol/date appears in multiple first-layer folds")
    score = pd.to_numeric(predictions[contract.score_column], errors="coerce")
    if not np.isfinite(score.to_numpy(dtype=float)).all() or not score.between(0, 1).all():
        raise ValueError("first-layer A/B scores must be finite probabilities")
    year_by_fold = {fold.name: fold.test_year for fold in DEFAULT_YEAR_FOLDS}
    invalid_year = predictions["date"].dt.year.ne(
        predictions["fold"].map(year_by_fold).astype(int)
    )
    if invalid_year.any():
        raise ValueError("first-layer prediction dates do not match declared test folds")
    predictions[FIRST_LAYER_SCORE_COLUMN] = score.astype("float32")
    predictions[FIRST_LAYER_FOLD_COLUMN] = predictions["fold"].astype("string")
    predictions[FIRST_LAYER_PROVENANCE_COLUMN] = np.where(
        predictions["fold"].eq(contract.train_fold),
        "oof",
        "test",
    )
    return predictions[
        [
            "fold",
            "symbol",
            "date",
            FIRST_LAYER_SCORE_COLUMN,
            FIRST_LAYER_PROVENANCE_COLUMN,
            FIRST_LAYER_FOLD_COLUMN,
        ]
    ].sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def prepare_playbook_events(
    factors: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    factor_columns: Sequence[str] = EVENT_FACTOR_COLUMNS,
) -> pd.DataFrame:
    """Join one factor row to every frozen A/B first-layer prediction row."""

    selected_factors = tuple(dict.fromkeys(factor_columns))
    _require_columns(factors, {"symbol", "date", *selected_factors}, name="factor events")
    _require_columns(
        predictions,
        {
            "fold",
            "symbol",
            "date",
            FIRST_LAYER_SCORE_COLUMN,
            FIRST_LAYER_PROVENANCE_COLUMN,
            FIRST_LAYER_FOLD_COLUMN,
        },
        name="first-layer predictions",
    )
    left = factors[["symbol", "date", *selected_factors]].copy()
    left["symbol"] = left["symbol"].astype("string").str.strip()
    left["date"] = _normalize_dates(left["date"], name="factor date")
    if left.duplicated(["symbol", "date"]).any():
        raise ValueError("factor events contain duplicate symbol/date rows")
    right = predictions.copy()
    right["symbol"] = right["symbol"].astype("string").str.strip()
    right["date"] = _normalize_dates(right["date"], name="prediction date")
    if right.duplicated(["symbol", "date"]).any():
        raise ValueError("first-layer predictions contain duplicate symbol/date rows")
    if set(right["fold"].astype(str)) != set(PLAYBOOK_DEVELOPMENT_FOLDS):
        raise ValueError("playbook events require exactly A/B predictions; fold C is forbidden")
    merged = right.merge(
        left,
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = merged["_merge"].ne("both")
    if missing.any():
        samples = merged.loc[missing, ["symbol", "date"]].head(3).to_dict("records")
        raise ValueError(f"first-layer predictions lack factor rows: {samples}")
    merged = merged.drop(columns="_merge")
    merged["event_id"] = stable_event_ids(merged["symbol"], merged["date"])
    validate_first_layer_score_contract(merged)
    if len(RULE_FEATURE_COLUMNS) != len(set(RULE_FEATURE_COLUMNS)):
        raise ValueError("playbook rule-factor contract contains duplicates")
    return merged[
        [
            "fold",
            "event_id",
            "symbol",
            "date",
            FIRST_LAYER_SCORE_COLUMN,
            FIRST_LAYER_PROVENANCE_COLUMN,
            FIRST_LAYER_FOLD_COLUMN,
            *selected_factors,
        ]
    ].sort_values(["fold", "date", "symbol"], kind="stable").reset_index(drop=True)


def audit_reusable_playbook_events(
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    events_path: Path,
    predictions_path: Path,
    factor_dataset_path: Path,
    available_event_columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fail closed before resuming outcomes from an atomically published event table."""

    required = {
        "fold",
        "event_id",
        "symbol",
        "date",
        FIRST_LAYER_SCORE_COLUMN,
        FIRST_LAYER_PROVENANCE_COLUMN,
        FIRST_LAYER_FOLD_COLUMN,
        *EVENT_FACTOR_COLUMNS,
    }
    available = (
        set(events.columns)
        if available_event_columns is None
        else set(available_event_columns)
    )
    missing = required - available
    if missing:
        raise ValueError(
            "reusable playbook events missing required columns: "
            f"{sorted(missing)}"
        )
    _require_columns(
        predictions,
        {
            "fold",
            "symbol",
            "date",
            FIRST_LAYER_SCORE_COLUMN,
            FIRST_LAYER_PROVENANCE_COLUMN,
            FIRST_LAYER_FOLD_COLUMN,
        },
        name="first-layer predictions",
    )
    work = events.copy()
    work["fold"] = work["fold"].astype("string")
    work["event_id"] = work["event_id"].astype("string")
    work["symbol"] = work["symbol"].astype("string").str.strip()
    work["date"] = _normalize_dates(work["date"], name="reusable event date")
    if set(work["fold"].astype(str)) != set(PLAYBOOK_DEVELOPMENT_FOLDS):
        raise ValueError("reusable event table must contain exactly A/B and never C")
    if len(work) != len(predictions):
        raise ValueError(
            f"reusable event rows={len(work)} differ from predictions={len(predictions)}"
        )
    if work.duplicated(["fold", "event_id"]).any():
        raise ValueError("reusable event table contains duplicate fold/event keys")
    if work.duplicated(["symbol", "date"]).any():
        raise ValueError("reusable event table contains duplicate symbol/date keys")
    expected_ids = stable_event_ids(work["symbol"], work["date"])
    if not work["event_id"].astype(str).eq(expected_ids.astype(str)).all():
        raise ValueError("reusable event IDs do not match stable symbol/date IDs")
    validate_first_layer_score_contract(work)

    prediction_contract = predictions[
        [
            "fold",
            "symbol",
            "date",
            FIRST_LAYER_SCORE_COLUMN,
            FIRST_LAYER_PROVENANCE_COLUMN,
            FIRST_LAYER_FOLD_COLUMN,
        ]
    ].copy()
    prediction_contract["symbol"] = (
        prediction_contract["symbol"].astype("string").str.strip()
    )
    prediction_contract["date"] = _normalize_dates(
        prediction_contract["date"], name="prediction date"
    )
    comparison = work[
        [
            "fold",
            "symbol",
            "date",
            FIRST_LAYER_SCORE_COLUMN,
            FIRST_LAYER_PROVENANCE_COLUMN,
            FIRST_LAYER_FOLD_COLUMN,
        ]
    ].merge(
        prediction_contract,
        on=["fold", "symbol", "date"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_event", "_prediction"),
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError("reusable event keys do not exactly cover A/B predictions")
    event_score = pd.to_numeric(
        comparison[f"{FIRST_LAYER_SCORE_COLUMN}_event"], errors="coerce"
    ).to_numpy(dtype=float)
    prediction_score = pd.to_numeric(
        comparison[f"{FIRST_LAYER_SCORE_COLUMN}_prediction"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.array_equal(event_score, prediction_score, equal_nan=False):
        raise ValueError("reusable event first-layer scores differ from selected predictions")
    for column in (FIRST_LAYER_PROVENANCE_COLUMN, FIRST_LAYER_FOLD_COLUMN):
        if not comparison[f"{column}_event"].astype(str).eq(
            comparison[f"{column}_prediction"].astype(str)
        ).all():
            raise ValueError(f"reusable event {column} differs from predictions")

    factor_schema_versions = (
        sorted(work["factor_schema_version"].dropna().astype(str).unique())
        if "factor_schema_version" in work
        else []
    )
    return {
        "rows": int(len(work)),
        "prediction_coverage": 1.0,
        "fold_rows": {
            str(key): int(value)
            for key, value in work["fold"].astype(str).value_counts().sort_index().items()
        },
        "first_layer_provenance_rows": {
            str(key): int(value)
            for key, value in work[FIRST_LAYER_PROVENANCE_COLUMN]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "date_min": pd.Timestamp(work["date"].min()),
        "date_max": pd.Timestamp(work["date"].max()),
        "project_factor_count": len(PROJECT_FACTOR_COLUMNS),
        "event_factor_count": len(EVENT_FACTOR_COLUMNS),
        "factor_schema_versions": factor_schema_versions,
        "reused_atomically_published_events": True,
        "reuse_validation": {
            "keys": "exact_A_B_one_to_one",
            "rows": "exact_prediction_row_count",
            "selected_score": "exact_float_match",
            "provenance": "exact_oof_test_and_fold_match",
            "event_sha256": file_sha256(events_path),
            "prediction_sha256": file_sha256(predictions_path),
            "factor_source_sha256": file_sha256(factor_dataset_path),
        },
    }


def narrow_playbook_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated event factors and normalize the on-disk outcome schema."""

    _require_columns(outcomes, PLAYBOOK_OUTCOME_COLUMNS, name="playbook outcomes")
    narrowed = outcomes[list(PLAYBOOK_OUTCOME_COLUMNS)].copy()
    for column in _OUTCOME_STRING_COLUMNS:
        narrowed[column] = narrowed[column].astype("string")
    for column in _OUTCOME_DATE_COLUMNS:
        narrowed[column] = pd.to_datetime(narrowed[column], errors="coerce")
    for column in _OUTCOME_BOOL_COLUMNS:
        narrowed[column] = narrowed[column].fillna(False).astype(bool)
    narrowed["holding_sessions"] = pd.array(
        narrowed["holding_sessions"], dtype="Int64"
    )
    for column in _OUTCOME_FLOAT_COLUMNS:
        narrowed[column] = pd.to_numeric(narrowed[column], errors="coerce").astype(
            "float64"
        )
    return narrowed


def join_playbook_event_outcomes(
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    event_feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Materialize an in-memory model frame while storage remains normalized."""

    features = tuple(dict.fromkeys(event_feature_columns))
    event_columns = [
        *EVENT_KEY_COLUMNS,
        "symbol",
        "date",
        FIRST_LAYER_SCORE_COLUMN,
        FIRST_LAYER_PROVENANCE_COLUMN,
        FIRST_LAYER_FOLD_COLUMN,
        *features,
    ]
    _require_columns(events, event_columns, name="playbook events")
    _require_columns(outcomes, {*EVENT_KEY_COLUMNS, "symbol", "date"}, name="playbook outcomes")
    if events.duplicated(list(EVENT_KEY_COLUMNS)).any():
        raise ValueError("playbook events contain duplicate fold/event keys")
    if outcomes.duplicated([*EVENT_KEY_COLUMNS, "playbook_id"]).any():
        raise ValueError("playbook outcomes contain duplicate fold/event/action keys")
    event_payload = events[event_columns].copy()
    joined = outcomes.merge(
        event_payload,
        on=[*EVENT_KEY_COLUMNS, "symbol", "date"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise ValueError("playbook outcomes contain keys absent from event factors")
    joined = joined.drop(columns="_merge")
    validate_first_layer_score_contract(joined)
    return joined


def audit_narrow_playbook_tables(
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> dict[str, Any]:
    """Fail closed on normalized keys, action completeness, masks and folds."""

    _require_columns(
        events,
        {
            *EVENT_KEY_COLUMNS,
            "symbol",
            "date",
            *EVENT_FACTOR_COLUMNS,
            FIRST_LAYER_SCORE_COLUMN,
            FIRST_LAYER_PROVENANCE_COLUMN,
            FIRST_LAYER_FOLD_COLUMN,
        },
        name="playbook events",
    )
    _require_columns(outcomes, PLAYBOOK_OUTCOME_COLUMNS, name="playbook outcomes")
    if set(events["fold"].astype(str)) != set(PLAYBOOK_DEVELOPMENT_FOLDS):
        raise ValueError("playbook event table must contain exactly A/B and never C")
    validate_first_layer_score_contract(events)
    if events.duplicated(list(EVENT_KEY_COLUMNS)).any():
        raise ValueError("playbook events contain duplicate keys")
    if outcomes.duplicated([*EVENT_KEY_COLUMNS, "playbook_id"]).any():
        raise ValueError("playbook outcomes contain duplicate action keys")
    repeated_wide = set(EVENT_FACTOR_COLUMNS) & set(outcomes.columns)
    if repeated_wide:
        raise ValueError(
            "wide event factors must not be repeated in outcomes: "
            f"{sorted(repeated_wide)}"
        )
    expected_actions = {spec.playbook_id for spec in DEFAULT_PLAYBOOK_CATALOG}
    actual_actions = set(outcomes["playbook_id"].astype(str))
    if actual_actions != expected_actions:
        raise ValueError(
            f"playbook action catalog drifted: expected={sorted(expected_actions)} "
            f"actual={sorted(actual_actions)}"
        )
    event_count = int(len(events))
    counts = outcomes.groupby("playbook_id", sort=False).size()
    if any(int(counts.get(action, 0)) != event_count for action in expected_actions):
        raise ValueError("every event must have exactly one row for every playbook action")
    outcome_events = outcomes[list(EVENT_KEY_COLUMNS)].drop_duplicates()
    coverage = outcome_events.merge(
        events[list(EVENT_KEY_COLUMNS)],
        on=list(EVENT_KEY_COLUMNS),
        how="outer",
        indicator=True,
    )
    if not coverage["_merge"].eq("both").all():
        raise ValueError("event/outcome key coverage is not one-to-one")
    no_trade = outcomes["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
    eligible = outcomes["eligible"].fillna(False).astype(bool)
    mature = outcomes["mature"].fillna(False).astype(bool)
    if not (eligible.loc[no_trade] & mature.loc[no_trade]).all():
        raise ValueError("NO_TRADE must always be eligible and mature")
    for column in ("net_return", "mae", "round_trip_cost"):
        values = pd.to_numeric(outcomes.loc[no_trade, column], errors="coerce")
        if not values.eq(0.0).all():
            raise ValueError(f"NO_TRADE {column} must be zero")
    regular = ~no_trade
    if ((~eligible | ~mature) & regular & outcomes["net_return"].notna()).any():
        raise ValueError("ineligible or immature regular actions cannot have net_return")
    regular_rows = int(regular.sum())
    ineligible_regular = regular & ~eligible
    eligible_mature_regular = regular & eligible & mature
    eligible_immature_regular = regular & eligible & ~mature
    known_utility_regular = ineligible_regular | eligible_mature_regular
    return {
        "schema_version": PLAYBOOK_DATASET_SCHEMA_VERSION,
        "event_rows": event_count,
        "outcome_rows": int(len(outcomes)),
        "fold_event_rows": {
            str(key): int(value)
            for key, value in events["fold"].astype(str).value_counts().sort_index().items()
        },
        "actions_per_event": len(expected_actions),
        "eligible_regular_rows": int((regular & eligible).sum()),
        "ineligible_cancel_zero_utility_rows": int(ineligible_regular.sum()),
        "eligible_mature_regular_rows": int(eligible_mature_regular.sum()),
        "eligible_immature_null_utility_rows": int(eligible_immature_regular.sum()),
        "known_utility_regular_rows": int(known_utility_regular.sum()),
        "known_utility_regular_coverage": (
            float(known_utility_regular.sum() / regular_rows)
            if regular_rows
            else np.nan
        ),
        "eligibility_reasons": {
            str(key): int(value)
            for key, value in outcomes.loc[regular, "eligibility_reason"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "maturity_reasons": {
            str(key): int(value)
            for key, value in outcomes.loc[regular, "maturity_reason"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "rule_feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
        "rule_feature_count": len(RULE_FEATURE_COLUMNS),
        "rule_feature_columns_sha256": RULE_FEATURE_COLUMNS_SHA256,
        "first_layer_oof_contract": {
            "A": "earlier-model test score admitted as layer-2 OOF training input",
            "B": "first-layer test score used for layer-2 evaluation",
            "C": "forbidden_not_read",
        },
        "storage_contract": "wide event factors stored once; outcomes contain no event factors",
    }


def dataset_manifest_payload(
    *,
    events_path: Path,
    outcomes_path: Path,
    predictions_path: Path,
    factor_dataset_path: Path,
    prediction_contract: FirstLayerPredictionContract,
    event_summary: dict[str, Any],
    outcome_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable provenance payload after both files are published."""

    return {
        "schema_version": PLAYBOOK_DATASET_SCHEMA_VERSION,
        "release_status": "research_only_not_production",
        "event_table": {
            "path": str(events_path),
            "sha256": file_sha256(events_path),
            **event_summary,
        },
        "outcome_table": {
            "path": str(outcomes_path),
            "sha256": file_sha256(outcomes_path),
            **outcome_summary,
        },
        "first_layer_prediction_source": {
            "path": str(predictions_path),
            "sha256": file_sha256(predictions_path),
            **asdict(prediction_contract),
            "fold_provenance": {
                "A": "oof_for_layer2",
                "B": "test",
                "C": "forbidden_not_read",
            },
        },
        "factor_source": {
            "path": str(factor_dataset_path),
            "sha256": file_sha256(factor_dataset_path),
            "rule_feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
            "rule_feature_count": len(RULE_FEATURE_COLUMNS),
            "rule_feature_columns_sha256": RULE_FEATURE_COLUMNS_SHA256,
        },
        "playbook_catalog": {
            "version": PLAYBOOK_POLICY_VERSION,
            "sha256": playbook_catalog_hash(),
            "action_count": len(DEFAULT_PLAYBOOK_CATALOG),
            "action_ids": [spec.playbook_id for spec in DEFAULT_PLAYBOOK_CATALOG],
            "round_trip_cost_bps": {
                spec.playbook_id: float(spec.round_trip_cost_bps)
                for spec in DEFAULT_PLAYBOOK_CATALOG
            },
        },
        "warning": (
            "event-level counterfactual outcomes with overlapping holding windows; "
            "not a capital curve and not production approval"
        ),
    }


__all__ = [
    "EVENT_FACTOR_COLUMNS",
    "FirstLayerPredictionContract",
    "IDENTITY_COLUMNS",
    "OutcomeBuildAudit",
    "PLAYBOOK_DATASET_SCHEMA_VERSION",
    "PLAYBOOK_DEVELOPMENT_FOLDS",
    "PLAYBOOK_EVALUATION_FOLD",
    "PLAYBOOK_OUTCOME_COLUMNS",
    "PLAYBOOK_TRAIN_FOLD",
    "StreamingParquetWriter",
    "SUBVARIANT_COLUMNS",
    "audit_narrow_playbook_tables",
    "audit_reusable_playbook_events",
    "dataset_manifest_payload",
    "file_sha256",
    "join_playbook_event_outcomes",
    "load_first_layer_predictions",
    "narrow_playbook_outcomes",
    "prepare_playbook_events",
    "stable_event_ids",
]
