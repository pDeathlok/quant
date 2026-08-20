"""Leakage-resistant shared model for event-by-playbook research.

The first-layer model decides candidate quality.  This second layer receives a
frozen first-layer OOF/test score and predicts the pre-registered utility of
each eligible executable playbook.  ``NO_TRADE`` is never model-predicted: its
utility is fixed at zero and it wins whenever every executable action scores
at or below zero.

This module is research-only.  It does not build counterfactual outcomes and
does not modify the production selector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.right_side_playbook_policy import (
    NO_TRADE_PLAYBOOK_ID,
    playbook_catalog_hash,
    serialize_playbook_catalog,
)
from quant.research.right_side_unified import DEFAULT_YEAR_FOLDS, RIGHT_SIDE_SIGNALS, YearFold
from quant.research.right_side_unified_features import RULE_FEATURE_COLUMNS


PLAYBOOK_MODEL_SCHEMA_VERSION = "right-side-playbook-shared-v1"
UTILITY_VERSION = "planned-action-utility-with-ineligible-cancel-v2"
UTILITY_DRAWDOWN_PENALTY = 0.25
UTILITY_FORMULA = (
    "0 if no_trade or T+1 ineligible; otherwise "
    "net_return - 0.25 * abs(min(mae, 0))"
)
FIRST_LAYER_SCORE_COLUMN = "first_layer_score"
FIRST_LAYER_PROVENANCE_COLUMN = "first_layer_score_provenance"
FIRST_LAYER_FOLD_COLUMN = "first_layer_score_fold"
ALLOWED_FIRST_LAYER_PROVENANCE = frozenset({"oof", "test"})

EVENT_KEY_COLUMNS: tuple[str, ...] = ("fold", "event_id")
ACTION_ID_COLUMNS: tuple[str, ...] = (
    "playbook_id",
    "entry_mode",
    "exit_policy_id",
)
PLAYBOOK_PARAMETER_COLUMNS: tuple[str, ...] = ("round_trip_cost",)
DEFAULT_EVENT_FEATURE_COLUMNS: tuple[str, ...] = (
    *PROJECT_FACTOR_COLUMNS,
    *RULE_FEATURE_COLUMNS,
    *RIGHT_SIDE_SIGNALS,
)

# Outcome labels, realized execution fields, and T+1-derived eligibility facts
# may be used for target construction or masking, never for action scoring.
FORBIDDEN_MODEL_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "eligible",
        "eligibility_reason",
        "mature",
        "maturity_reason",
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "exit_reason",
        "gross_return",
        "net_return",
        "mae",
        "holding_sessions",
        "ambiguous_bar",
        "locked_limit_up",
        "open_at_up_limit",
        "close_at_up_limit",
        "open_gap",
        "utility",
        "predicted_utility",
    }
)

REQUIRED_OUTCOME_COLUMNS: frozenset[str] = frozenset(
    {
        "event_id",
        "symbol",
        "date",
        "fold",
        "playbook_id",
        "entry_mode",
        "exit_policy_id",
        "eligible",
        "mature",
        "net_return",
        "mae",
        "round_trip_cost",
        FIRST_LAYER_SCORE_COLUMN,
        FIRST_LAYER_PROVENANCE_COLUMN,
        FIRST_LAYER_FOLD_COLUMN,
    }
)


@dataclass(frozen=True)
class PlaybookRegressorSpec:
    """Fixed first-pass estimator; no C/test-driven hyperparameter search."""

    learning_rate: float = 0.05
    max_iter: int = 160
    max_leaf_nodes: int = 31
    max_depth: int = 5
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    random_state: int = 42

    def build(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(**asdict(self))


DEFAULT_PLAYBOOK_REGRESSOR_SPEC = PlaybookRegressorSpec()


@dataclass(frozen=True)
class PlaybookTimeSplit:
    fold: str
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class StaticPlaybookChoice:
    """One action learned from an earlier outcome fold only."""

    playbook_id: str
    mean_utility: float
    training_rows: int


@dataclass(frozen=True)
class StaticPerSignalPlaybookPolicy:
    """Deterministic signal-to-action mapping fitted on fold A outcomes."""

    choices: tuple[tuple[str, StaticPlaybookChoice], ...]

    def as_dict(self) -> dict[str, StaticPlaybookChoice]:
        return dict(self.choices)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def validate_playbook_feature_columns(
    feature_columns: Sequence[str],
) -> tuple[str, ...]:
    """Validate an explicit T-day event feature allowlist."""

    columns = tuple(dict.fromkeys(str(column) for column in feature_columns))
    if not columns:
        raise ValueError("event feature columns must not be empty")
    forbidden = set(columns) & FORBIDDEN_MODEL_FEATURE_COLUMNS
    if forbidden:
        raise ValueError(
            "future/outcome/eligibility columns are forbidden model features: "
            f"{sorted(forbidden)}"
        )
    reserved = set(columns) & set(ACTION_ID_COLUMNS)
    if reserved:
        raise ValueError(
            "action identifiers are encoded by the model and cannot be event features: "
            f"{sorted(reserved)}"
        )
    first_layer_reserved = {
        FIRST_LAYER_SCORE_COLUMN,
        FIRST_LAYER_PROVENANCE_COLUMN,
        FIRST_LAYER_FOLD_COLUMN,
    }
    leaked = set(columns) & first_layer_reserved
    if leaked:
        raise ValueError(
            "first-layer score/provenance columns are managed by the stacking contract: "
            f"{sorted(leaked)}"
        )
    return columns


def admit_default_event_features(
    frame: pd.DataFrame,
    *,
    minimum_project_coverage: float = 0.50,
) -> tuple[str, ...]:
    """Admit covered project factors while mandating rule/identity columns.

    Several registered project factors can be structurally unavailable in an
    older research slice. They are admitted from the layer-2 training fold
    only; all causal rule factors and 14 identities remain mandatory.
    """

    if not 0.0 <= minimum_project_coverage <= 1.0:
        raise ValueError("minimum_project_coverage must be in [0, 1]")
    mandatory = (*RULE_FEATURE_COLUMNS, *RIGHT_SIDE_SIGNALS)
    _require_columns(frame, mandatory, name="playbook event factors")
    unusable_mandatory: list[str] = []
    for column in mandatory:
        numeric = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if not numeric.notna().any():
            unusable_mandatory.append(column)
    if unusable_mandatory:
        raise ValueError(
            "mandatory rule/identity features are all non-finite: "
            f"{sorted(unusable_mandatory)}"
        )
    admitted_project: list[str] = []
    for column in PROJECT_FACTOR_COLUMNS:
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if (
            numeric.notna().mean() >= minimum_project_coverage
            and numeric.nunique(dropna=True) >= 2
        ):
            admitted_project.append(column)
    return tuple((*admitted_project, *mandatory))


def validate_first_layer_score_contract(frame: pd.DataFrame) -> None:
    """Reject in-sample or fold-mismatched first-layer scores.

    ``oof`` is required on rows used to train the second layer. ``test`` is
    required on the target fold's test year and its declared score fold must
    equal that row's playbook fold.  Event copies across actions must carry an
    identical score and provenance.
    """

    required = {
        *EVENT_KEY_COLUMNS,
        FIRST_LAYER_SCORE_COLUMN,
        FIRST_LAYER_PROVENANCE_COLUMN,
        FIRST_LAYER_FOLD_COLUMN,
    }
    _require_columns(frame, required, name="playbook frame")
    score = pd.to_numeric(frame[FIRST_LAYER_SCORE_COLUMN], errors="coerce")
    if not np.isfinite(score.to_numpy(dtype=float)).all():
        raise ValueError("first-layer score must be finite")
    if not score.between(0.0, 1.0).all():
        raise ValueError("first-layer score must be in [0, 1]")
    provenance = frame[FIRST_LAYER_PROVENANCE_COLUMN].astype(str)
    invalid = set(provenance) - ALLOWED_FIRST_LAYER_PROVENANCE
    if invalid:
        raise ValueError(
            "first-layer score provenance must be oof or test; "
            f"invalid={sorted(invalid)}"
        )
    score_fold = frame[FIRST_LAYER_FOLD_COLUMN].astype(str)
    if score_fold.eq("").any() or score_fold.eq("nan").any():
        raise ValueError("first-layer score fold must be explicit")
    if not score_fold.eq(frame["fold"].astype(str)).all():
        raise ValueError("first-layer score fold differs from playbook fold")

    consistency = frame.assign(
        _score=score,
        _provenance=provenance,
        _score_fold=score_fold,
    ).groupby(list(EVENT_KEY_COLUMNS), sort=False).agg(
        score_values=("_score", "nunique"),
        provenance_values=("_provenance", "nunique"),
        score_fold_values=("_score_fold", "nunique"),
    )
    if consistency.gt(1).any(axis=None):
        raise ValueError("first-layer score/provenance is inconsistent across event actions")


def attach_playbook_utility(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach planned-action utility while preserving unknown future tails.

    A T+1-ineligible plan is cancelled to ``NO_TRADE`` and realizes zero. It
    remains a valid target: dropping it would condition training on future
    executability. Eligible actions with incomplete exit windows stay null.
    """

    _require_columns(
        frame,
        {"playbook_id", "mature", "net_return", "mae"},
        name="playbook outcomes",
    )
    out = frame.copy()
    net = pd.to_numeric(out["net_return"], errors="coerce")
    mae = pd.to_numeric(out["mae"], errors="coerce")
    mature = out["mature"].fillna(False).astype(bool)
    eligible = (
        out["eligible"].fillna(False).astype(bool)
        if "eligible" in out
        else pd.Series(True, index=out.index, dtype=bool)
    )
    no_trade = out["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
    utility = pd.Series(np.nan, index=out.index, dtype=float)
    regular = ~no_trade
    regular_valid = eligible & mature & regular & net.notna() & mae.notna()
    regular_valid &= np.isfinite(net.to_numpy(dtype=float)) & np.isfinite(
        mae.to_numpy(dtype=float)
    )
    downside = np.abs(np.minimum(mae.loc[regular_valid], 0.0))
    utility.loc[regular_valid] = (
        net.loc[regular_valid] - UTILITY_DRAWDOWN_PENALTY * downside
    )
    utility.loc[regular & ~eligible] = 0.0
    utility.loc[mature & no_trade] = 0.0
    out["utility"] = utility
    return out


def audit_playbook_model_dataset(
    frame: pd.DataFrame,
    *,
    event_feature_columns: Sequence[str],
) -> dict[str, Any]:
    """Fail closed on keys, masks, utility, stacking provenance and features."""

    features = validate_playbook_feature_columns(event_feature_columns)
    _require_columns(frame, REQUIRED_OUTCOME_COLUMNS | set(features), name="playbook dataset")
    if frame.duplicated([*EVENT_KEY_COLUMNS, "playbook_id"]).any():
        raise ValueError("playbook dataset contains duplicate fold/event/action keys")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("playbook dataset contains invalid dates")
    validate_first_layer_score_contract(frame)
    eligible = frame["eligible"].fillna(False).astype(bool)
    mature = frame["mature"].fillna(False).astype(bool)
    no_trade = frame["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
    if not (eligible.loc[no_trade] & mature.loc[no_trade]).all():
        raise ValueError("NO_TRADE must always be eligible and mature")
    no_trade_net = pd.to_numeric(frame.loc[no_trade, "net_return"], errors="coerce")
    no_trade_mae = pd.to_numeric(frame.loc[no_trade, "mae"], errors="coerce")
    no_trade_cost = pd.to_numeric(
        frame.loc[no_trade, "round_trip_cost"], errors="coerce"
    )
    if not (
        no_trade_net.eq(0.0).all()
        and no_trade_mae.eq(0.0).all()
        and no_trade_cost.eq(0.0).all()
    ):
        raise ValueError("NO_TRADE net_return, mae, and round_trip_cost must be zero")
    no_trade_counts = no_trade.groupby(
        [frame[column] for column in EVENT_KEY_COLUMNS], sort=False
    ).sum()
    if not no_trade_counts.eq(1).all():
        raise ValueError("every fold/event must contain exactly one NO_TRADE action")
    if ((~eligible | ~mature) & ~no_trade & frame["net_return"].notna()).any():
        raise ValueError("ineligible/immature regular actions must not carry net_return")
    expected_return_rows = eligible & mature & ~no_trade
    expected_net = pd.to_numeric(
        frame.loc[expected_return_rows, "net_return"], errors="coerce"
    )
    expected_mae = pd.to_numeric(
        frame.loc[expected_return_rows, "mae"], errors="coerce"
    )
    if not (
        np.isfinite(expected_net.to_numpy(dtype=float)).all()
        and np.isfinite(expected_mae.to_numpy(dtype=float)).all()
    ):
        raise ValueError("eligible mature regular actions require finite net_return and mae")

    expected_target_rows = ~no_trade & (~eligible | (eligible & mature))
    non_numeric: list[str] = []
    unusable: list[str] = []
    feature_frame = frame.drop_duplicates(list(EVENT_KEY_COLUMNS), keep="first")
    for column in features:
        numeric = pd.to_numeric(feature_frame[column], errors="coerce")
        if feature_frame[column].notna().any() and numeric.notna().sum() == 0:
            non_numeric.append(column)
        if not np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            unusable.append(column)
    if non_numeric:
        raise ValueError(f"event features must be numeric: {sorted(non_numeric)}")
    if unusable:
        raise ValueError(f"event features are entirely non-finite: {sorted(unusable)}")
    return {
        "schema_version": PLAYBOOK_MODEL_SCHEMA_VERSION,
        "rows": int(len(frame)),
        "events": int(frame[list(EVENT_KEY_COLUMNS)].drop_duplicates().shape[0]),
        "folds": sorted(frame["fold"].astype(str).unique()),
        "actions": sorted(frame["playbook_id"].astype(str).unique()),
        "event_feature_count": len(features),
        "eligible_regular_rows": int((eligible & ~no_trade).sum()),
        "planned_action_training_rows": int(expected_target_rows.sum()),
        "eligible_mature_training_rows": int(expected_return_rows.sum()),
        "ineligible_cancel_training_rows": int((~eligible & ~no_trade).sum()),
        "utility_version": UTILITY_VERSION,
        "utility_formula": UTILITY_FORMULA,
    }


def split_playbook_time_fold(frame: pd.DataFrame, fold: YearFold) -> PlaybookTimeSplit:
    """Split earlier first-layer OOS folds from one later evaluation fold.

    With today's artifacts, the valid development experiment is A (2024 OOS)
    -> B (2025 OOS).  A+B -> C is descriptive only because the overall system
    has already inspected C; callers must not describe it as untouched.
    """

    _require_columns(
        frame,
        {"fold", "date", FIRST_LAYER_PROVENANCE_COLUMN, FIRST_LAYER_FOLD_COLUMN},
        name="playbook dataset",
    )
    fold_order = {item.name: position for position, item in enumerate(DEFAULT_YEAR_FOLDS)}
    target_position = fold_order[fold.name]
    training_folds = {
        name for name, position in fold_order.items() if position < target_position
    }
    frame_fold = frame["fold"].astype(str)
    train = frame[frame_fold.isin(training_folds)].copy()
    test = frame[frame_fold.eq(fold.name)].copy()
    if train.empty or test.empty:
        raise ValueError(f"fold {fold.name} has empty train or test events")
    train_dates = pd.to_datetime(train["date"], errors="coerce")
    test_dates = pd.to_datetime(test["date"], errors="coerce")
    if train_dates.isna().any() or test_dates.isna().any():
        raise ValueError("playbook fold contains invalid dates")
    if train_dates.max() >= test_dates.min():
        raise ValueError(f"fold {fold.name} first-layer OOS folds are not chronological")
    if not set(train[FIRST_LAYER_PROVENANCE_COLUMN].astype(str)) <= ALLOWED_FIRST_LAYER_PROVENANCE:
        raise ValueError(f"fold {fold.name} training scores are not OOF/test provenance")
    if set(test[FIRST_LAYER_PROVENANCE_COLUMN].astype(str)) != {"test"}:
        raise ValueError(f"fold {fold.name} test rows require first-layer test scores")
    if set(test[FIRST_LAYER_FOLD_COLUMN].astype(str)) != {fold.name}:
        raise ValueError(f"fold {fold.name} test scores declare a different fold")
    # Every action copy of one event is selected by the same event date, so no
    # event can be divided across train and test.
    overlap = set(train["event_id"].astype(str)) & set(test["event_id"].astype(str))
    if overlap:
        raise ValueError(f"fold {fold.name} event IDs cross train/test boundary")
    return PlaybookTimeSplit(fold=fold.name, train=train, test=test)


def _category_values(frame: pd.DataFrame, column: str) -> tuple[str, ...]:
    values = frame[column].fillna("").astype(str)
    return tuple(sorted(value for value in values.unique() if value))


@dataclass
class SharedPlaybookModel:
    """Persistable shared action-conditional utility regressor."""

    estimator: Any
    event_feature_columns: tuple[str, ...]
    playbook_ids: tuple[str, ...]
    entry_modes: tuple[str, ...]
    exit_policy_ids: tuple[str, ...]
    numeric_action_columns: tuple[str, ...]
    evaluation_fold: str
    training_score_folds: tuple[str, ...]
    trained_through: str
    training_rows: int = 0
    eligible_mature_training_rows: int = 0
    ineligible_cancel_training_rows: int = 0
    schema_version: str = PLAYBOOK_MODEL_SCHEMA_VERSION
    utility_version: str = UTILITY_VERSION

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        event_feature_columns: Sequence[str],
        fold: str,
        regressor_spec: PlaybookRegressorSpec = DEFAULT_PLAYBOOK_REGRESSOR_SPEC,
    ) -> "SharedPlaybookModel":
        features = validate_playbook_feature_columns(event_feature_columns)
        audit_playbook_model_dataset(frame, event_feature_columns=features)
        fold_order = {item.name: position for position, item in enumerate(DEFAULT_YEAR_FOLDS)}
        if fold not in fold_order:
            raise ValueError(f"unknown evaluation fold: {fold}")
        training_score_folds = tuple(sorted(
            set(frame["fold"].astype(str)), key=lambda value: fold_order.get(value, 999)
        ))
        invalid_training_folds = [
            value
            for value in training_score_folds
            if value not in fold_order or fold_order[value] >= fold_order[fold]
        ]
        if invalid_training_folds:
            raise ValueError(
                "second-layer training requires earlier first-layer OOS folds; "
                f"invalid={invalid_training_folds} evaluation_fold={fold}"
            )
        eligible = frame["eligible"].fillna(False).astype(bool)
        mature = frame["mature"].fillna(False).astype(bool)
        regular = ~frame["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
        net = pd.to_numeric(frame["net_return"], errors="coerce")
        mae = pd.to_numeric(frame["mae"], errors="coerce")
        executable_target = eligible & mature & regular & net.notna() & mae.notna()
        cancelled_target = ~eligible & regular
        mask = (
            executable_target | cancelled_target
        )
        if not mask.any():
            raise ValueError("no known regular planned-action utilities for training")
        training_index = frame.index[mask]
        utility_series = pd.Series(0.0, index=training_index, dtype=float)
        executable_index = frame.index[executable_target]
        downside = np.abs(
            np.minimum(mae.loc[executable_index].to_numpy(dtype=float), 0.0)
        )
        utility_series.loc[executable_index] = (
            net.loc[executable_index].to_numpy(dtype=float)
            - UTILITY_DRAWDOWN_PENALTY * downside
        )
        utility = utility_series.to_numpy(dtype=float)
        playbook_ids = tuple(
            value
            for value in _category_values(frame, "playbook_id")
            if value != NO_TRADE_PLAYBOOK_ID
        )
        model = cls(
            estimator=regressor_spec.build(),
            event_feature_columns=features,
            playbook_ids=playbook_ids,
            entry_modes=_category_values(frame.loc[~frame["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)], "entry_mode"),
            exit_policy_ids=_category_values(frame.loc[~frame["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)], "exit_policy_id"),
            numeric_action_columns=tuple(
                column for column in PLAYBOOK_PARAMETER_COLUMNS if column in frame.columns
            ),
            evaluation_fold=str(fold),
            training_score_folds=training_score_folds,
            trained_through=str(
                pd.to_datetime(frame.loc[training_index, "date"]).max().date()
            ),
            training_rows=int(mask.sum()),
            eligible_mature_training_rows=int(executable_target.sum()),
            ineligible_cancel_training_rows=int(cancelled_target.sum()),
        )
        matrix = model._transform_index(frame, training_index)
        model.estimator.fit(matrix, utility)
        return model

    @classmethod
    def fit_normalized(
        cls,
        events: pd.DataFrame,
        outcomes: pd.DataFrame,
        *,
        event_feature_columns: Sequence[str],
        fold: str,
        regressor_spec: PlaybookRegressorSpec = DEFAULT_PLAYBOOK_REGRESSOR_SPEC,
        scratch_dir: Path | None = None,
    ) -> "SharedPlaybookModel":
        """Fit all known action targets without materializing a 9x wide frame.

        Event factors remain one row per event.  Each regular action is joined
        and transformed separately into a disk-backed float32 design matrix.
        T+1-ineligible plans are retained with utility zero; eligible actions
        with incomplete exit windows remain unknown and are not written.
        """

        features = validate_playbook_feature_columns(event_feature_columns)
        event_required = {
            *EVENT_KEY_COLUMNS,
            "symbol",
            "date",
            FIRST_LAYER_SCORE_COLUMN,
            FIRST_LAYER_PROVENANCE_COLUMN,
            FIRST_LAYER_FOLD_COLUMN,
            *features,
        }
        outcome_required = {
            *EVENT_KEY_COLUMNS,
            "symbol",
            "date",
            *ACTION_ID_COLUMNS,
            "eligible",
            "mature",
            "net_return",
            "mae",
            *PLAYBOOK_PARAMETER_COLUMNS,
        }
        _require_columns(events, event_required, name="normalized playbook events")
        _require_columns(outcomes, outcome_required, name="normalized playbook outcomes")
        if events.duplicated(list(EVENT_KEY_COLUMNS)).any():
            raise ValueError("normalized playbook events contain duplicate keys")
        if outcomes.duplicated([*EVENT_KEY_COLUMNS, "playbook_id"]).any():
            raise ValueError("normalized playbook outcomes contain duplicate action keys")
        validate_first_layer_score_contract(events)
        if set(events["fold"].astype(str)) != set(outcomes["fold"].astype(str)):
            raise ValueError("normalized event and outcome folds differ")

        fold_order = {item.name: position for position, item in enumerate(DEFAULT_YEAR_FOLDS)}
        if fold not in fold_order:
            raise ValueError(f"unknown evaluation fold: {fold}")
        training_score_folds = tuple(
            sorted(
                set(events["fold"].astype(str)),
                key=lambda value: fold_order.get(value, 999),
            )
        )
        invalid_training_folds = [
            value
            for value in training_score_folds
            if value not in fold_order or fold_order[value] >= fold_order[fold]
        ]
        if invalid_training_folds:
            raise ValueError(
                "second-layer training requires earlier first-layer OOS folds; "
                f"invalid={invalid_training_folds} evaluation_fold={fold}"
            )

        no_trade = outcomes["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
        no_trade_counts = no_trade.groupby(
            [outcomes[column] for column in EVENT_KEY_COLUMNS], sort=False
        ).sum()
        if not no_trade_counts.eq(1).all():
            raise ValueError("every normalized event requires exactly one NO_TRADE action")
        eligible = outcomes["eligible"].fillna(False).astype(bool)
        mature = outcomes["mature"].fillna(False).astype(bool)
        net = pd.to_numeric(outcomes["net_return"], errors="coerce")
        mae = pd.to_numeric(outcomes["mae"], errors="coerce")
        regular = ~no_trade
        executable_target = eligible & mature & regular & net.notna() & mae.notna()
        cancelled_target = ~eligible & regular
        known_target = executable_target | cancelled_target
        training_rows = int(known_target.sum())
        if training_rows <= 0:
            raise ValueError("no known regular planned-action utilities for training")
        invalid_known = executable_target & (
            ~np.isfinite(net.to_numpy(dtype=float))
            | ~np.isfinite(mae.to_numpy(dtype=float))
        )
        if invalid_known.any():
            raise ValueError("eligible mature normalized actions require finite outcomes")

        regular_outcomes = outcomes.loc[regular]
        model = cls(
            estimator=regressor_spec.build(),
            event_feature_columns=features,
            playbook_ids=tuple(
                value
                for value in _category_values(regular_outcomes, "playbook_id")
                if value != NO_TRADE_PLAYBOOK_ID
            ),
            entry_modes=_category_values(regular_outcomes, "entry_mode"),
            exit_policy_ids=_category_values(regular_outcomes, "exit_policy_id"),
            numeric_action_columns=tuple(
                column for column in PLAYBOOK_PARAMETER_COLUMNS if column in outcomes.columns
            ),
            evaluation_fold=str(fold),
            training_score_folds=training_score_folds,
            trained_through=str(pd.to_datetime(events["date"]).max().date()),
            training_rows=training_rows,
            eligible_mature_training_rows=int(executable_target.sum()),
            ineligible_cancel_training_rows=int(cancelled_target.sum()),
        )

        event_payload = events[
            [
                *EVENT_KEY_COLUMNS,
                "symbol",
                "date",
                FIRST_LAYER_SCORE_COLUMN,
                FIRST_LAYER_PROVENANCE_COLUMN,
                FIRST_LAYER_FOLD_COLUMN,
                *features,
            ]
        ]
        scratch_root = Path(scratch_dir) if scratch_dir is not None else None
        if scratch_root is not None:
            scratch_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="right-side-playbook-", dir=scratch_root) as temp:
            temp_root = Path(temp)
            design = np.memmap(
                temp_root / "design.float32",
                mode="w+",
                dtype="float32",
                shape=(training_rows, len(model.feature_names_in_)),
            )
            target = np.memmap(
                temp_root / "target.float32",
                mode="w+",
                dtype="float32",
                shape=(training_rows,),
            )
            offset = 0
            for _, action_rows in regular_outcomes.groupby("playbook_id", sort=True):
                action_known = known_target.loc[action_rows.index]
                selected = action_rows.loc[action_known].copy()
                if selected.empty:
                    continue
                joined = selected.merge(
                    event_payload,
                    on=[*EVENT_KEY_COLUMNS, "symbol", "date"],
                    how="left",
                    validate="many_to_one",
                    indicator=True,
                )
                if joined["_merge"].ne("both").any():
                    raise ValueError("normalized outcomes contain keys absent from events")
                joined = joined.drop(columns="_merge")
                count = len(joined)
                stop = offset + count
                matrix = model.transform(joined)
                design[offset:stop] = matrix.to_numpy(dtype=np.float32, copy=False)
                joined_eligible = joined["eligible"].fillna(False).astype(bool)
                utility = np.zeros(count, dtype=np.float32)
                if joined_eligible.any():
                    joined_net = pd.to_numeric(
                        joined.loc[joined_eligible, "net_return"], errors="raise"
                    ).to_numpy(dtype=float)
                    joined_mae = pd.to_numeric(
                        joined.loc[joined_eligible, "mae"], errors="raise"
                    ).to_numpy(dtype=float)
                    utility[joined_eligible.to_numpy()] = (
                        joined_net
                        - UTILITY_DRAWDOWN_PENALTY
                        * np.abs(np.minimum(joined_mae, 0.0))
                    ).astype(np.float32)
                target[offset:stop] = utility
                offset = stop
                del matrix, joined, utility, selected
            if offset != training_rows:
                raise RuntimeError(
                    f"normalized training rows written={offset} expected={training_rows}"
                )
            design.flush()
            target.flush()
            model.estimator.fit(design, target)
            del design, target
        return model

    @property
    def feature_names_in_(self) -> tuple[str, ...]:
        return (
            *self.event_feature_columns,
            FIRST_LAYER_SCORE_COLUMN,
            *self.numeric_action_columns,
            *(f"playbook__{value}" for value in self.playbook_ids),
            *(f"entry__{value}" for value in self.entry_modes),
            *(f"exit__{value}" for value in self.exit_policy_ids),
        )

    def _transform_index(
        self,
        frame: pd.DataFrame,
        index: pd.Index,
    ) -> pd.DataFrame:
        required = {
            *self.event_feature_columns,
            *ACTION_ID_COLUMNS,
            *self.numeric_action_columns,
            FIRST_LAYER_SCORE_COLUMN,
        }
        _require_columns(frame, required, name="playbook scoring frame")
        unknown = set(frame["playbook_id"].astype(str)) - {
            *self.playbook_ids,
            NO_TRADE_PLAYBOOK_ID,
        }
        if unknown:
            raise ValueError(f"unknown playbook IDs at scoring: {sorted(unknown)}")
        columns: dict[str, pd.Series] = {}
        for column in (*self.event_feature_columns, FIRST_LAYER_SCORE_COLUMN, *self.numeric_action_columns):
            columns[column] = (
                pd.to_numeric(frame.loc[index, column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .astype("float32")
            )
        playbook = frame.loc[index, "playbook_id"].astype(str)
        entry = frame.loc[index, "entry_mode"].fillna("").astype(str)
        exit_policy = frame.loc[index, "exit_policy_id"].fillna("").astype(str)
        for value in self.playbook_ids:
            columns[f"playbook__{value}"] = playbook.eq(value).astype("float32")
        for value in self.entry_modes:
            columns[f"entry__{value}"] = entry.eq(value).astype("float32")
        for value in self.exit_policy_ids:
            columns[f"exit__{value}"] = exit_policy.eq(value).astype("float32")
        matrix = pd.DataFrame(columns, index=index)
        return matrix.loc[:, list(self.feature_names_in_)]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._transform_index(frame, frame.index)

    def score_actions(self, frame: pd.DataFrame) -> pd.DataFrame:
        validate_first_layer_score_contract(frame)
        if set(frame["fold"].astype(str)) != {self.evaluation_fold}:
            raise ValueError(
                "scoring frame must contain exactly the model evaluation fold "
                f"{self.evaluation_fold}"
            )
        out = frame.copy()
        no_trade = out["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
        predicted = pd.Series(np.nan, index=out.index, dtype=float)
        # This is a T-close policy decision. Realized T+1 eligibility is not a
        # mask here: using it would retrospectively switch from a failed open
        # plan to a close plan. Execution is gated only after precommitment.
        regular = ~no_trade
        if regular.any():
            matrix = self.transform(out.loc[regular])
            if not hasattr(self.estimator, "feature_names_in_"):
                prediction_input: Any = matrix.to_numpy(dtype=np.float32, copy=False)
            else:
                prediction_input = matrix
            predicted.loc[regular] = self.estimator.predict(prediction_input)
        predicted.loc[no_trade] = 0.0
        out["predicted_utility"] = predicted
        return out

    def manifest(self, *, playbook_catalog_version: str, data_cutoff: str) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_kind": "shared_event_by_playbook_utility_regressor",
            "evaluation_fold": self.evaluation_fold,
            "training_first_layer_score_folds": list(self.training_score_folds),
            "trained_through": self.trained_through,
            "training_rows": int(self.training_rows),
            "eligible_mature_training_rows": int(
                self.eligible_mature_training_rows
            ),
            "ineligible_cancel_training_rows": int(
                self.ineligible_cancel_training_rows
            ),
            "data_cutoff": str(data_cutoff),
            "event_features": list(self.event_feature_columns),
            "first_layer_score_contract": {
                "score_column": FIRST_LAYER_SCORE_COLUMN,
                "provenance_column": FIRST_LAYER_PROVENANCE_COLUMN,
                "fold_column": FIRST_LAYER_FOLD_COLUMN,
                "training_provenance": "earlier_fold_oof_or_test",
                "test_provenance": "test",
            },
            "action_features": {
                "playbook_ids": list(self.playbook_ids),
                "entry_modes": list(self.entry_modes),
                "exit_policy_ids": list(self.exit_policy_ids),
                "numeric_columns": list(self.numeric_action_columns),
            },
            "playbook_catalog_version": str(playbook_catalog_version),
            "playbook_catalog_sha256": playbook_catalog_hash(),
            "playbook_catalog": serialize_playbook_catalog(),
            "round_trip_cost_source": "pre_registered_playbook_parameter",
            "utility_version": self.utility_version,
            "utility_formula": UTILITY_FORMULA,
            "no_trade": {
                "playbook_id": NO_TRADE_PLAYBOOK_ID,
                "fixed_utility": 0.0,
                "selection_rule": "choose NO_TRADE when best regular predicted utility <= 0",
            },
            "release_status": "research_only_not_production",
        }


def select_planned_playbook(scored: pd.DataFrame) -> pd.DataFrame:
    """Precommit one T-known action per event before observing T+1 execution."""

    _require_columns(
        scored,
        {*EVENT_KEY_COLUMNS, "playbook_id", "predicted_utility"},
        name="scored playbooks",
    )
    work = scored.copy()
    no_trade = work["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
    counts = no_trade.groupby(
        [work[column] for column in EVENT_KEY_COLUMNS], sort=False
    ).sum()
    if not counts.eq(1).all() or not pd.to_numeric(
        work.loc[no_trade, "predicted_utility"], errors="coerce"
    ).eq(0.0).all():
        raise ValueError("every event requires exactly one zero-utility NO_TRADE action")
    regular = work.loc[~no_trade].copy()
    regular["predicted_utility"] = pd.to_numeric(
        regular["predicted_utility"], errors="coerce"
    )
    regular = regular.dropna(subset=["predicted_utility"])
    best_regular = (
        regular.sort_values(
            [*EVENT_KEY_COLUMNS, "predicted_utility", "playbook_id"],
            ascending=[True, True, True, True],
            kind="stable",
        )
        .groupby(list(EVENT_KEY_COLUMNS), sort=False, as_index=False)
        .tail(1)
    )
    abstain_keys = best_regular.loc[
        best_regular["predicted_utility"].gt(0.0), list(EVENT_KEY_COLUMNS)
    ]
    positive = best_regular.merge(
        abstain_keys,
        on=list(EVENT_KEY_COLUMNS),
        how="inner",
    )
    no_trade_rows = work.loc[no_trade].copy()
    positive_keys = set(map(tuple, positive[list(EVENT_KEY_COLUMNS)].to_numpy()))
    select_no_trade = no_trade_rows[list(EVENT_KEY_COLUMNS)].apply(tuple, axis=1).map(
        lambda key: key not in positive_keys
    )
    chosen = pd.concat([positive, no_trade_rows.loc[select_no_trade]], ignore_index=True, sort=False)
    if chosen.duplicated(list(EVENT_KEY_COLUMNS)).any():
        raise RuntimeError("playbook selection produced multiple actions for one event")
    chosen["abstained"] = chosen["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
    chosen["planned_playbook_id"] = chosen["playbook_id"].astype(str)
    chosen["execution_status"] = np.where(
        chosen["abstained"], "planned_no_trade", "planned_regular"
    )
    return chosen.sort_values(list(EVENT_KEY_COLUMNS), kind="stable").reset_index(drop=True)


def apply_execution_gate(
    planned: pd.DataFrame,
    action_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Cancel an ineligible precommitted plan without ex-post action switching."""

    _require_columns(
        planned,
        {*EVENT_KEY_COLUMNS, "playbook_id", "planned_playbook_id", "eligible"},
        name="planned playbooks",
    )
    _require_columns(
        action_rows,
        {*EVENT_KEY_COLUMNS, "playbook_id", "eligible"},
        name="playbook action rows",
    )
    if planned.duplicated(list(EVENT_KEY_COLUMNS)).any():
        raise ValueError("planned playbooks contain multiple actions for one event")
    no_trade = action_rows[
        action_rows["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
    ].copy()
    if no_trade.duplicated(list(EVENT_KEY_COLUMNS)).any():
        raise ValueError("action rows contain duplicate NO_TRADE actions")
    canceled = (
        ~planned["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
        & ~planned["eligible"].fillna(False).astype(bool)
    )
    kept = planned.loc[~canceled].copy()
    if not kept.empty:
        kept["execution_status"] = np.where(
            kept["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID),
            "no_trade",
            "executed",
        )
    if canceled.any():
        cancelled_keys = planned.loc[
            canceled, [*EVENT_KEY_COLUMNS, "planned_playbook_id"]
        ]
        replacement = no_trade.merge(
            cancelled_keys,
            on=list(EVENT_KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
            suffixes=("", "_planned"),
        )
        if len(replacement) != int(canceled.sum()):
            raise ValueError("missing NO_TRADE row for a canceled planned action")
        replacement["abstained"] = True
        replacement["execution_status"] = "cancelled_ineligible"
        kept = pd.concat([kept, replacement], ignore_index=True, sort=False)
    if kept.duplicated(list(EVENT_KEY_COLUMNS)).any():
        raise RuntimeError("execution gate produced multiple actions for one event")
    return kept.sort_values(list(EVENT_KEY_COLUMNS), kind="stable").reset_index(drop=True)


def _eligible_mature_regular_utility(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = attach_playbook_utility(frame)
    mask = (
        ~enriched["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
        & enriched["utility"].notna()
    )
    return enriched.loc[mask].copy()


def fit_static_global_playbook(frame: pd.DataFrame) -> StaticPlaybookChoice:
    """Fit one globally fixed action from an earlier outcome fold."""

    _require_columns(
        frame,
        {"playbook_id", "eligible", "mature", "net_return", "mae"},
        name="static global training outcomes",
    )
    training = _eligible_mature_regular_utility(frame)
    if training.empty:
        return StaticPlaybookChoice(NO_TRADE_PLAYBOOK_ID, 0.0, 0)
    summary = (
        training.groupby("playbook_id", sort=False)["utility"]
        .agg(["mean", "size"])
        .reset_index()
        .sort_values(
            ["mean", "playbook_id"],
            ascending=[True, True],
            kind="stable",
        )
    )
    best = summary.iloc[-1]
    if float(best["mean"]) <= 0.0:
        return StaticPlaybookChoice(NO_TRADE_PLAYBOOK_ID, 0.0, int(best["size"]))
    return StaticPlaybookChoice(
        str(best["playbook_id"]),
        float(best["mean"]),
        int(best["size"]),
    )


def score_static_global_playbook(
    frame: pd.DataFrame,
    choice: StaticPlaybookChoice,
) -> pd.DataFrame:
    """Score a later fold without consulting its eligibility or outcomes."""

    _require_columns(
        frame,
        {*EVENT_KEY_COLUMNS, "playbook_id"},
        name="static global scoring outcomes",
    )
    out = frame.copy()
    no_trade = out["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
    predicted = pd.Series(-np.inf, index=out.index, dtype=float)
    predicted.loc[no_trade] = 0.0
    if choice.playbook_id != NO_TRADE_PLAYBOOK_ID and choice.mean_utility > 0.0:
        predicted.loc[out["playbook_id"].astype(str).eq(choice.playbook_id)] = float(
            choice.mean_utility
        )
    out["predicted_utility"] = predicted
    return out


def fit_static_per_signal_playbooks(
    frame: pd.DataFrame,
    *,
    signal_columns: Sequence[str] = RIGHT_SIDE_SIGNALS,
) -> StaticPerSignalPlaybookPolicy:
    """Fit one fixed action for each strategy identity using fold A only."""

    signals = tuple(dict.fromkeys(signal_columns))
    _require_columns(frame, signals, name="static per-signal training outcomes")
    choices: list[tuple[str, StaticPlaybookChoice]] = []
    for signal in signals:
        active = frame[signal].fillna(False).astype(bool)
        choice = fit_static_global_playbook(frame.loc[active])
        choices.append((signal, choice))
    return StaticPerSignalPlaybookPolicy(tuple(choices))


def score_static_per_signal_playbooks(
    frame: pd.DataFrame,
    policy: StaticPerSignalPlaybookPolicy,
) -> pd.DataFrame:
    """Apply the frozen A-fold mappings to B before the T+1 execution gate."""

    choices = policy.as_dict()
    _require_columns(
        frame,
        {*EVENT_KEY_COLUMNS, "playbook_id", *choices},
        name="static per-signal scoring outcomes",
    )
    out = frame.copy()
    playbook = out["playbook_id"].astype(str)
    no_trade = playbook.eq(NO_TRADE_PLAYBOOK_ID)
    predicted = pd.Series(-np.inf, index=out.index, dtype=float)
    predicted.loc[no_trade] = 0.0
    for signal, choice in choices.items():
        if choice.playbook_id == NO_TRADE_PLAYBOOK_ID or choice.mean_utility <= 0.0:
            continue
        matching = (
            out[signal].fillna(False).astype(bool)
            & playbook.eq(choice.playbook_id)
        )
        if matching.any():
            predicted.loc[matching] = np.maximum(
                predicted.loc[matching].to_numpy(dtype=float),
                float(choice.mean_utility),
            )
    out["predicted_utility"] = predicted
    return out


def select_no_trade_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exactly one zero-return NO_TRADE row per event."""

    _require_columns(
        frame,
        {*EVENT_KEY_COLUMNS, "playbook_id"},
        name="no-trade baseline outcomes",
    )
    selected = frame[
        frame["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
    ].copy()
    if selected.duplicated(list(EVENT_KEY_COLUMNS)).any() or len(selected) != frame[
        list(EVENT_KEY_COLUMNS)
    ].drop_duplicates().shape[0]:
        raise ValueError("NO_TRADE baseline requires exactly one row per event")
    selected["predicted_utility"] = 0.0
    selected["planned_playbook_id"] = NO_TRADE_PLAYBOOK_ID
    selected["abstained"] = True
    selected["execution_status"] = "no_trade"
    return selected.sort_values(list(EVENT_KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def select_oracle_playbook(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the best realized action per event as a non-executable upper bound."""

    _require_columns(
        frame,
        {
            *EVENT_KEY_COLUMNS,
            "playbook_id",
            "eligible",
            "mature",
            "net_return",
            "mae",
        },
        name="oracle outcomes",
    )
    enriched = attach_playbook_utility(frame)
    no_trade = enriched["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
    admissible = no_trade | (
        enriched["eligible"].fillna(False).astype(bool)
        & enriched["mature"].fillna(False).astype(bool)
        & enriched["utility"].notna()
    )
    candidates = enriched.loc[admissible].copy()
    selected = (
        candidates.sort_values(
            [*EVENT_KEY_COLUMNS, "utility", "playbook_id"],
            ascending=[True, True, True, True],
            kind="stable",
        )
        .groupby(list(EVENT_KEY_COLUMNS), sort=False, as_index=False)
        .tail(1)
        .copy()
    )
    expected = frame[list(EVENT_KEY_COLUMNS)].drop_duplicates().shape[0]
    if len(selected) != expected:
        raise ValueError("oracle selection failed to cover every event")
    selected["predicted_utility"] = selected["utility"]
    selected["planned_playbook_id"] = selected["playbook_id"].astype(str)
    selected["abstained"] = selected["playbook_id"].astype(str).eq(
        NO_TRADE_PLAYBOOK_ID
    )
    selected["execution_status"] = "oracle_upper_bound"
    return selected.sort_values(list(EVENT_KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def evaluate_playbook_selections(
    selected: pd.DataFrame,
    *,
    arm: str,
) -> dict[str, Any]:
    """Return event-level reward/risk diagnostics, never a capital curve."""

    _require_columns(
        selected,
        {
            *EVENT_KEY_COLUMNS,
            "date",
            "playbook_id",
            "planned_playbook_id",
            "execution_status",
            "net_return",
            "mae",
        },
        name="selected playbooks",
    )
    if selected.duplicated(list(EVENT_KEY_COLUMNS)).any():
        raise ValueError("selection metrics require exactly one row per event")
    work = selected.copy()
    dates = pd.to_datetime(work["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("selection metrics contain invalid dates")
    net = pd.to_numeric(work["net_return"], errors="coerce")
    mae = pd.to_numeric(work["mae"], errors="coerce")
    planned_trade = ~work["planned_playbook_id"].astype(str).eq(
        NO_TRADE_PLAYBOOK_ID
    )
    executed_trade = ~work["playbook_id"].astype(str).eq(NO_TRADE_PLAYBOOK_ID)
    evaluated = net.notna() & np.isfinite(net.to_numpy(dtype=float))
    evaluated_trade = executed_trade & evaluated
    unevaluated_trade = executed_trade & ~evaluated
    covered_net = net.loc[evaluated]
    trade_net = net.loc[evaluated_trade]
    trade_mae = mae.loc[evaluated_trade]
    positive = float(trade_net.loc[trade_net > 0].sum())
    negative = float(-trade_net.loc[trade_net < 0].sum())

    daily = pd.DataFrame({"date": dates, "net": net}).loc[evaluated]
    daily_mean = daily.groupby("date", sort=True)["net"].mean()
    cumulative_proxy = daily_mean.cumsum()
    drawdown_proxy = cumulative_proxy - cumulative_proxy.cummax()
    max_drawdown_proxy = (
        float(-drawdown_proxy.min()) if len(drawdown_proxy) else np.nan
    )
    planned_counts = work["planned_playbook_id"].astype(str).value_counts()
    executed_counts = work["playbook_id"].astype(str).value_counts()
    total = len(work)
    return {
        "arm": str(arm),
        "events": int(total),
        "covered_events": int(evaluated.sum()),
        "unevaluated_events": int((~evaluated).sum()),
        "event_coverage": float(evaluated.mean()) if total else np.nan,
        "planned_trades": int(planned_trade.sum()),
        "executed_trades": int(executed_trade.sum()),
        "evaluated_trades": int(evaluated_trade.sum()),
        "unevaluated_trades": int(unevaluated_trade.sum()),
        "planned_trade_rate": float(planned_trade.mean()) if total else np.nan,
        "executed_trade_rate": float(executed_trade.mean()) if total else np.nan,
        "cancellation_rate": (
            float(work["execution_status"].astype(str).eq("cancelled_ineligible").mean())
            if total
            else np.nan
        ),
        "execution_status_counts": {
            str(key): int(value)
            for key, value in work["execution_status"].astype(str).value_counts().items()
        },
        "average_event_net_return": (
            float(covered_net.mean()) if len(covered_net) else np.nan
        ),
        "sum_event_net_return": (
            float(covered_net.sum()) if len(covered_net) else np.nan
        ),
        "average_trade_net_return": (
            float(trade_net.mean()) if len(trade_net) else np.nan
        ),
        "win_rate": float(trade_net.gt(0).mean()) if len(trade_net) else np.nan,
        "p10_trade_net_return": (
            float(trade_net.quantile(0.10)) if len(trade_net) else np.nan
        ),
        "average_trade_mae": (
            float(trade_mae.mean()) if len(trade_mae) else np.nan
        ),
        "profit_factor": positive / negative if negative > 0 else np.nan,
        "daily_equal_event_cumsum_final_proxy": (
            float(cumulative_proxy.iloc[-1]) if len(cumulative_proxy) else np.nan
        ),
        "daily_equal_event_max_drawdown_proxy": max_drawdown_proxy,
        "planned_playbook_selection_rate": {
            str(key): float(value / total) for key, value in planned_counts.items()
        },
        "executed_playbook_selection_rate": {
            str(key): float(value / total) for key, value in executed_counts.items()
        },
        "warning": (
            "event-level overlapping outcomes; daily equal-event cumulative sum and "
            "drawdown are diagnostics, not a capital curve"
        ),
    }


def select_best_playbook(scored: pd.DataFrame) -> pd.DataFrame:
    """Compatibility alias for the T-close planned-action selection only."""

    return select_planned_playbook(scored)


def fold_by_name(name: str) -> YearFold:
    for fold in DEFAULT_YEAR_FOLDS:
        if fold.name == name:
            return fold
    raise ValueError(f"unknown playbook time fold: {name}")


__all__ = [
    "ACTION_ID_COLUMNS",
    "ALLOWED_FIRST_LAYER_PROVENANCE",
    "DEFAULT_EVENT_FEATURE_COLUMNS",
    "DEFAULT_PLAYBOOK_REGRESSOR_SPEC",
    "FIRST_LAYER_FOLD_COLUMN",
    "FIRST_LAYER_PROVENANCE_COLUMN",
    "FIRST_LAYER_SCORE_COLUMN",
    "FORBIDDEN_MODEL_FEATURE_COLUMNS",
    "NO_TRADE_PLAYBOOK_ID",
    "PLAYBOOK_MODEL_SCHEMA_VERSION",
    "PlaybookRegressorSpec",
    "PlaybookTimeSplit",
    "StaticPerSignalPlaybookPolicy",
    "StaticPlaybookChoice",
    "SharedPlaybookModel",
    "UTILITY_DRAWDOWN_PENALTY",
    "UTILITY_FORMULA",
    "UTILITY_VERSION",
    "attach_playbook_utility",
    "admit_default_event_features",
    "apply_execution_gate",
    "audit_playbook_model_dataset",
    "fold_by_name",
    "evaluate_playbook_selections",
    "fit_static_global_playbook",
    "fit_static_per_signal_playbooks",
    "score_static_global_playbook",
    "score_static_per_signal_playbooks",
    "select_best_playbook",
    "select_no_trade_baseline",
    "select_oracle_playbook",
    "select_planned_playbook",
    "split_playbook_time_fold",
    "validate_first_layer_score_contract",
    "validate_playbook_feature_columns",
]
