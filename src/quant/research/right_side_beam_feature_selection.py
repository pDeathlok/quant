"""Leakage-safe backward Beam selection for the 13 new right-side factors.

The 105 legacy rule factors, admitted project factors, and task identity form
an immutable context.  Only the 13 v2 increment columns may be removed.  Beam
evaluation is delegated to a matched R0/R1 callback so this module can enforce
search, temporal-scoring, permutation, and provenance contracts independently
of the estimator implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from statistics import median, pstdev
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


BEAM_SCHEMA_VERSION = (
    "right-side-beam-residual-v3-adaptation-105-plus-v2-increment"
)


@dataclass(frozen=True)
class BeamSettings:
    width: int = 4
    min_features: int = 6
    # The source Beam Residual v3 contract allows ten removals.  With this
    # project's 13 candidates and six-feature floor the effective depth is
    # seven, but keeping the declared budget at ten makes the adaptation
    # auditable instead of silently presenting a shallower search as v3.
    max_remove: int = 10
    positive_fold_requirement: int = 2
    minimum_fold_pr_auc_lift: float = -0.002
    pr_auc_std_penalty: float = 0.5
    negative_fold_penalty: float = 2.0
    # Ranking is the first-layer objective in this project, so PR-AUC replaces
    # source-v3 ROC-AUC as the primary lift.  The v3 probability-stability term
    # is retained at its original coefficient; ROC-AUC remains descriptive.
    roc_auc_weight: float = 0.0
    logloss_weight: float = 5.0
    feature_penalty: float = 0.00002
    permutation_rounds: int = 20
    maximum_permutation_p_value: float = 0.05

    def validate(self, candidate_count: int) -> None:
        if candidate_count <= 0:
            raise ValueError("beam candidate features must not be empty")
        if self.width <= 0:
            raise ValueError("beam width must be positive")
        if not 0 < self.min_features <= candidate_count:
            raise ValueError("beam min_features is outside candidate range")
        if self.max_remove < 0:
            raise ValueError("beam max_remove must be non-negative")
        if self.positive_fold_requirement <= 0:
            raise ValueError("positive_fold_requirement must be positive")
        if self.permutation_rounds <= 0:
            raise ValueError("permutation_rounds must be positive")
        if not 0 < self.maximum_permutation_p_value < 1:
            raise ValueError("maximum_permutation_p_value must be in (0, 1)")


@dataclass(frozen=True)
class BeamFoldLift:
    fold: str
    delta_pr_auc: float
    delta_roc_auc: float
    logloss_improvement: float


@dataclass(frozen=True)
class BeamCandidateScore:
    score: float
    eligible: bool
    positive_folds: int
    median_delta_pr_auc: float
    median_delta_roc_auc: float
    median_logloss_improvement: float
    pr_auc_lift_std: float
    minimum_delta_pr_auc: float
    n_features: int


@dataclass(frozen=True)
class BeamEvaluation:
    removed: tuple[str, ...]
    selected: tuple[str, ...]
    candidate_score: BeamCandidateScore
    fold_lifts: tuple[BeamFoldLift, ...]
    fold_predictions: tuple[Mapping[str, Any], ...]

    @property
    def score(self) -> float:
        return self.candidate_score.score

    @property
    def eligible(self) -> bool:
        return self.candidate_score.eligible


@dataclass(frozen=True)
class BeamSearchResult:
    best: BeamEvaluation
    evaluations: tuple[BeamEvaluation, ...]
    layers: tuple[tuple[tuple[str, ...], ...], ...]


@dataclass(frozen=True)
class RollingWindow:
    name: str
    history_dates: tuple[pd.Timestamp, ...]
    evaluation_dates: tuple[pd.Timestamp, ...]


def deterministic_stratified_event_sample(
    frame: pd.DataFrame,
    *,
    maximum_rows: int,
    label_column: str,
    signal_columns: Sequence[str],
    random_seed: int = 42,
    date_column: str = "date",
    symbol_column: str = "symbol",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cap Beam search data by month, primary signal and target label.

    Sampling is deterministic and used only for development search.  It never
    changes final selected-subset fitting or outer-test evaluation.
    """

    required = {
        date_column,
        symbol_column,
        label_column,
        *signal_columns,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"beam sampling frame missing columns: {sorted(missing)}")
    if maximum_rows <= 0:
        raise ValueError("beam maximum_rows must be positive")
    original_rows = len(frame)
    metadata: dict[str, Any] = {
        "method": "deterministic_month_primary_signal_label_stratified_cap",
        "random_seed": int(random_seed),
        "maximum_rows": int(maximum_rows),
        "original_rows": int(original_rows),
        "primary_signal_rule": "first_true_in_declared_signal_columns",
        "primary_signal_priority": list(signal_columns),
        "stable_hash_keys": [symbol_column, date_column, "random_seed"],
    }
    if original_rows <= maximum_rows:
        metadata["sampled_rows"] = int(original_rows)
        metadata["sampling_applied"] = False
        return frame.copy(), metadata

    work = frame.copy()
    active = work[list(signal_columns)].fillna(False).astype(bool).to_numpy()
    primary_positions = np.argmax(active, axis=1)
    has_active = active.any(axis=1)
    names = np.asarray(signal_columns, dtype=object)
    primary = np.where(has_active, names[primary_positions], "NO_ACTIVE_SIGNAL")
    work["_beam_month"] = pd.to_datetime(work[date_column]).dt.to_period("M").astype(str)
    work["_beam_primary_signal"] = primary
    work["_beam_label"] = pd.to_numeric(work[label_column], errors="raise").astype(int)
    # Stable hash priority avoids sensitivity to source row order and does not
    # inspect factor values or labels beyond the declared stratum.
    hash_frame = pd.DataFrame(
        {
            "symbol": work[symbol_column].astype(str),
            "date": pd.to_datetime(work[date_column]).astype("int64").astype(str),
            "seed": str(int(random_seed)),
        }
    )
    work["_beam_hash"] = pd.util.hash_pandas_object(
        hash_frame, index=False, hash_key="0123456789abcdef"
    ).to_numpy(dtype=np.uint64)
    group_columns = ["_beam_month", "_beam_primary_signal", "_beam_label"]
    group_sizes = work.groupby(group_columns, sort=True, observed=True).size()
    exact_quota = group_sizes.astype(float) / float(original_rows) * maximum_rows
    quotas = np.floor(exact_quota).astype(int)
    nonempty = group_sizes.gt(0)
    quotas.loc[nonempty] = quotas.loc[nonempty].clip(lower=1)
    while int(quotas.sum()) > maximum_rows:
        reducible = quotas[quotas.gt(1)]
        if reducible.empty:
            break
        key = max(reducible.index, key=lambda value: (quotas.loc[value], str(value)))
        quotas.loc[key] -= 1
    remaining = maximum_rows - int(quotas.sum())
    if remaining > 0:
        fractional = (exact_quota - np.floor(exact_quota)).sort_values(
            ascending=False, kind="stable"
        )
        for key in fractional.index:
            if remaining <= 0:
                break
            if quotas.loc[key] < group_sizes.loc[key]:
                quotas.loc[key] += 1
                remaining -= 1
    pieces: list[pd.DataFrame] = []
    for key, group in work.groupby(group_columns, sort=True, observed=True):
        count = int(quotas.get(key, 0))
        if count:
            pieces.append(group.sort_values("_beam_hash", kind="stable").head(count))
    sampled = pd.concat(pieces, ignore_index=False, sort=False)
    sampled = sampled.sort_values([date_column, symbol_column], kind="stable")
    sampled = sampled.drop(
        columns=["_beam_month", "_beam_primary_signal", "_beam_label", "_beam_hash"]
    ).reset_index(drop=True)
    metadata["sampled_rows"] = int(len(sampled))
    metadata["sampling_applied"] = True
    metadata["strata"] = int(len(group_sizes))
    return sampled, metadata


def feature_columns_sha256(columns: Sequence[str]) -> str:
    payload = "\n".join(str(column) for column in columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_validation_quarter_windows(
    training_dates: Sequence[object],
    validation_dates: Sequence[object],
) -> tuple[RollingWindow, ...]:
    """Build B1->B2, B1+B2->B3, B1+B2+B3->B4 windows.

    All outer-training dates are prepended to every history.  The outer test
    year is not accepted by this API, making accidental test-window use harder.
    """

    prior = pd.DatetimeIndex(pd.to_datetime(pd.Index(training_dates))).dropna().unique().sort_values()
    development = pd.DatetimeIndex(pd.to_datetime(pd.Index(validation_dates))).dropna().unique().sort_values()
    if len(prior) < 30 or len(development) < 40:
        raise ValueError("beam rolling windows need sufficient train/validation dates")
    blocks = [pd.DatetimeIndex(block) for block in np.array_split(development, 4)]
    if any(len(block) == 0 for block in blocks):
        raise ValueError("beam validation quarter block is empty")
    windows: list[RollingWindow] = []
    for index in range(1, 4):
        history = prior.append(blocks[0])
        for block in blocks[1:index]:
            history = history.append(block)
        evaluation = blocks[index]
        if history.max() >= evaluation.min():
            raise ValueError("beam rolling dates are not strictly chronological")
        windows.append(
            RollingWindow(
                name=f"validation_roll_{index}",
                history_dates=tuple(pd.Timestamp(value) for value in history),
                evaluation_dates=tuple(pd.Timestamp(value) for value in evaluation),
            )
        )
    return tuple(windows)


def build_validation_search_and_pipeline_select(
    training_dates: Sequence[object],
    validation_dates: Sequence[object],
) -> tuple[tuple[RollingWindow, ...], RollingWindow]:
    """Reserve an independent tail before constructing the three Beam folds.

    The first four chronological fifths of the outer validation period are
    treated as B1..B4 and passed to the standard B1→B2, B1+B2→B3,
    B1+B2+B3→B4 search.  The final fifth is never available to Beam ranking or
    the permutation test; it is the independent ``pipeline_select`` window
    used only after Top1 has been frozen.
    """

    prior = (
        pd.DatetimeIndex(pd.to_datetime(pd.Index(training_dates)))
        .dropna()
        .unique()
        .sort_values()
    )
    validation = (
        pd.DatetimeIndex(pd.to_datetime(pd.Index(validation_dates)))
        .dropna()
        .unique()
        .sort_values()
    )
    if len(prior) < 30 or len(validation) < 50:
        raise ValueError("beam search/pipeline-select needs sufficient dates")
    blocks = [pd.DatetimeIndex(block) for block in np.array_split(validation, 5)]
    if any(len(block) == 0 for block in blocks):
        raise ValueError("beam search/pipeline-select block is empty")
    search_dates = blocks[0]
    for block in blocks[1:4]:
        search_dates = search_dates.append(block)
    windows = build_validation_quarter_windows(prior, search_dates)
    pipeline_history = prior.append(search_dates)
    pipeline = RollingWindow(
        name="pipeline_select",
        history_dates=tuple(pd.Timestamp(value) for value in pipeline_history),
        evaluation_dates=tuple(pd.Timestamp(value) for value in blocks[4]),
    )
    if max(pipeline.history_dates) >= min(pipeline.evaluation_dates):
        raise ValueError("beam pipeline_select is not strictly chronological")
    return windows, pipeline


def score_candidate(
    fold_lifts: Sequence[BeamFoldLift],
    *,
    n_features: int,
    settings: BeamSettings,
) -> BeamCandidateScore:
    if not fold_lifts:
        raise ValueError("beam candidate needs temporal fold lifts")
    if n_features <= 0:
        raise ValueError("beam candidate needs at least one feature")
    pr = [float(item.delta_pr_auc) for item in fold_lifts]
    roc = [float(item.delta_roc_auc) for item in fold_lifts]
    losses = [float(item.logloss_improvement) for item in fold_lifts]
    median_pr = float(median(pr))
    median_roc = float(median(roc))
    median_loss = float(median(losses))
    pr_std = float(pstdev(pr)) if len(pr) > 1 else 0.0
    minimum_pr = float(min(pr))
    positive = int(sum(value > 0 for value in pr))
    score = (
        median_pr
        + settings.roc_auc_weight * median_roc
        + settings.logloss_weight * median_loss
        - settings.pr_auc_std_penalty * pr_std
        - settings.negative_fold_penalty * max(0.0, -minimum_pr)
        - settings.feature_penalty * n_features
    )
    eligible = (
        positive >= settings.positive_fold_requirement
        and median_loss >= 0.0
        and minimum_pr >= settings.minimum_fold_pr_auc_lift
    )
    return BeamCandidateScore(
        score=float(score),
        eligible=bool(eligible),
        positive_folds=positive,
        median_delta_pr_auc=median_pr,
        median_delta_roc_auc=median_roc,
        median_logloss_improvement=median_loss,
        pr_auc_lift_std=pr_std,
        minimum_delta_pr_auc=minimum_pr,
        n_features=int(n_features),
    )


def ranking_lift(
    labels: Sequence[int | bool],
    baseline_probability: Sequence[float],
    candidate_probability: Sequence[float],
    *,
    fold: str,
) -> BeamFoldLift:
    labels_array = np.asarray(labels, dtype=int)
    baseline = np.clip(np.asarray(baseline_probability, dtype=float), 1e-6, 1 - 1e-6)
    candidate = np.clip(np.asarray(candidate_probability, dtype=float), 1e-6, 1 - 1e-6)
    if not (len(labels_array) == len(baseline) == len(candidate)):
        raise ValueError("beam matched prediction lengths differ")
    if len(np.unique(labels_array)) < 2:
        raise ValueError("beam evaluation fold lacks both label classes")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("beam probabilities must be finite")
    return BeamFoldLift(
        fold=str(fold),
        delta_pr_auc=float(
            average_precision_score(labels_array, candidate)
            - average_precision_score(labels_array, baseline)
        ),
        delta_roc_auc=float(
            roc_auc_score(labels_array, candidate)
            - roc_auc_score(labels_array, baseline)
        ),
        logloss_improvement=float(
            log_loss(labels_array, baseline, labels=[0, 1])
            - log_loss(labels_array, candidate, labels=[0, 1])
        ),
    )


def combine_residual_probabilities(
    baseline_probability: Sequence[float],
    baseline_raw_margin: Sequence[float],
    candidate_raw_margin: Sequence[float],
    *,
    reliability: float | Sequence[float] = 1.0,
) -> np.ndarray:
    """Apply the Beam Residual v3 offset equation exactly.

    ``candidate_raw_margin`` is the total R1 margin from a matched model fitted
    with the R0 margin as ``base_margin``.  Subtracting the R0 raw margin
    isolates the learned residual.  The residual is then added to the logit of
    the calibrated R0 probability, preserving the baseline calibration while
    allowing the candidate subset to change ranking.

    The face-quality source uses a 1.0/0.5 quality gate.  This right-side
    adaptation has no homologous ex-ante quality gate, so production search
    declares and uses constant reliability 1.0.  The parameter remains
    explicit to make that adaptation testable and manifest-visible.
    """

    baseline = np.asarray(baseline_probability, dtype=float)
    baseline_margin = np.asarray(baseline_raw_margin, dtype=float)
    candidate_margin = np.asarray(candidate_raw_margin, dtype=float)
    weight = np.asarray(reliability, dtype=float)
    if not (
        baseline.shape == baseline_margin.shape == candidate_margin.shape
    ):
        raise ValueError("beam residual inputs must have identical shapes")
    if weight.ndim and weight.shape != baseline.shape:
        raise ValueError("beam residual reliability must be scalar or row-aligned")
    if not (
        np.isfinite(baseline).all()
        and np.isfinite(baseline_margin).all()
        and np.isfinite(candidate_margin).all()
        and np.isfinite(weight).all()
    ):
        raise ValueError("beam residual inputs must be finite")
    if np.any((weight < 0.0) | (weight > 1.0)):
        raise ValueError("beam residual reliability must be in [0, 1]")
    clipped = np.clip(baseline, 1e-6, 1.0 - 1e-6)
    baseline_logit = np.log(clipped / (1.0 - clipped))
    increment = weight * (candidate_margin - baseline_margin)
    combined_margin = baseline_logit + increment
    return 1.0 / (1.0 + np.exp(-np.clip(combined_margin, -80.0, 80.0)))


def evaluate_pipeline_select_variants(
    labels: Sequence[int | bool],
    baseline_probability: Sequence[float],
    *,
    soft_gate_probability: Sequence[float],
    no_gate_probability: Sequence[float],
) -> dict[str, Any]:
    """Freeze the residual policy on an independent validation tail.

    ``baseline_only`` is always available with zero lift.  In the absence of a
    homologous ex-ante quality signal for stocks, the source-v3 hard gate is
    explicitly unavailable; the adaptation compares a constant 0.5 soft
    residual with the constant 1.0 no-gate residual.  Ranking remains primary,
    while non-negative log-loss improvement is required before a residual
    variant may beat the baseline-only fallback.
    """

    variants = {
        "soft_gate_0p5": ranking_lift(
            labels,
            baseline_probability,
            soft_gate_probability,
            fold="pipeline_select",
        ),
        "no_gate_1p0": ranking_lift(
            labels,
            baseline_probability,
            no_gate_probability,
            fold="pipeline_select",
        ),
    }
    eligible = {
        name: lift
        for name, lift in variants.items()
        if lift.delta_pr_auc > 0.0 and lift.logloss_improvement >= 0.0
    }
    if eligible:
        selected_name, selected_lift = max(
            eligible.items(),
            key=lambda item: (
                item[1].delta_pr_auc,
                item[1].logloss_improvement,
                item[1].delta_roc_auc,
            ),
        )
        gate_passed = True
    else:
        selected_name = "baseline_only"
        selected_lift = BeamFoldLift(
            fold="pipeline_select",
            delta_pr_auc=0.0,
            delta_roc_auc=0.0,
            logloss_improvement=0.0,
        )
        gate_passed = False
    return {
        "selection_metric": "delta_pr_auc_then_logloss_no_returns",
        "hard_gate": "unavailable_no_homologous_ex_ante_quality_signal",
        "selected_variant": selected_name,
        "gate_passed": gate_passed,
        "selected_lift": asdict(selected_lift),
        "variants": {
            "baseline_only": asdict(
                BeamFoldLift("pipeline_select", 0.0, 0.0, 0.0)
            ),
            **{name: asdict(lift) for name, lift in variants.items()},
        },
    }


def _ranking_key(item: BeamEvaluation) -> tuple[int, float, int]:
    return (int(item.eligible), float(item.score), len(item.removed))


def backward_beam_search(
    *,
    candidate_features: Sequence[str],
    evaluate: Callable[[tuple[str, ...]], BeamEvaluation],
    settings: BeamSettings = BeamSettings(),
) -> BeamSearchResult:
    features = tuple(dict.fromkeys(str(value) for value in candidate_features))
    if len(features) != len(candidate_features):
        raise ValueError("beam candidate features must be unique")
    settings.validate(len(features))
    maximum_depth = min(settings.max_remove, len(features) - settings.min_features)
    cache: dict[tuple[str, ...], BeamEvaluation] = {}

    def cached(state: tuple[str, ...]) -> BeamEvaluation:
        if state not in cache:
            result = evaluate(state)
            expected_selected = tuple(feature for feature in features if feature not in state)
            if result.removed != state or result.selected != expected_selected:
                raise ValueError("beam evaluator returned mismatched feature state")
            cache[state] = result
        return cache[state]

    cached(())
    beam: tuple[tuple[str, ...], ...] = ((),)
    layers: list[tuple[tuple[str, ...], ...]] = [beam]
    for depth in range(1, maximum_depth + 1):
        states: set[tuple[str, ...]] = set()
        for removed in beam:
            removed_set = set(removed)
            for feature in features:
                if feature not in removed_set:
                    state = tuple(sorted((*removed_set, feature)))
                    if len(state) == depth:
                        states.add(state)
        evaluations = [cached(state) for state in sorted(states)]
        evaluations.sort(key=_ranking_key, reverse=True)
        beam = tuple(item.removed for item in evaluations[: settings.width])
        if not beam:
            break
        layers.append(beam)
    visited = tuple(cache.values())
    return BeamSearchResult(
        best=max(visited, key=_ranking_key),
        evaluations=visited,
        layers=tuple(layers),
    )


def visited_candidate_permutation_test(
    evaluations: Sequence[BeamEvaluation],
    *,
    settings: BeamSettings,
    observed_best_score: float,
    random_seed: int = 42,
) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("permutation test needs visited beam candidates")
    rng = np.random.default_rng(random_seed)
    maxima: list[float] = []
    for _ in range(settings.permutation_rounds):
        permuted_by_fold = {
            str(record["fold"]): rng.permutation(np.asarray(record["labels"], dtype=int))
            for record in evaluations[0].fold_predictions
        }
        round_scores: list[tuple[bool, float]] = []
        for evaluation in evaluations:
            lifts = [
                ranking_lift(
                    permuted_by_fold[str(record["fold"])],
                    record["baseline_probability"],
                    record["candidate_probability"],
                    fold=str(record["fold"]),
                )
                for record in evaluation.fold_predictions
            ]
            candidate_score = score_candidate(
                lifts,
                n_features=len(evaluation.selected),
                settings=settings,
            )
            round_scores.append((candidate_score.eligible, candidate_score.score))
        maxima.append(max(round_scores, key=lambda value: (int(value[0]), value[1]))[1])
    null = np.asarray(maxima, dtype=float)
    exceedances = int(np.sum(null >= observed_best_score))
    return {
        "method": "visited_candidate_validation_label_permutation",
        "rounds": settings.permutation_rounds,
        "observed_best_score": float(observed_best_score),
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "null_max": float(null.max()),
        "exceedances": exceedances,
        "empirical_p_value": float((1 + exceedances) / (settings.permutation_rounds + 1)),
    }


def evaluation_record(evaluation: BeamEvaluation, *, rank: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rank": int(rank),
        "removed_features": "|".join(evaluation.removed),
        "selected_features": "|".join(evaluation.selected),
        "selected_features_sha256": feature_columns_sha256(evaluation.selected),
        **asdict(evaluation.candidate_score),
    }
    for fold in evaluation.fold_lifts:
        prefix = fold.fold
        record[f"{prefix}_delta_pr_auc"] = fold.delta_pr_auc
        record[f"{prefix}_delta_roc_auc"] = fold.delta_roc_auc
        record[f"{prefix}_logloss_improvement"] = fold.logloss_improvement
    return record


def search_manifest(
    result: BeamSearchResult,
    *,
    settings: BeamSettings,
    outer_fold: str,
    context_features: Sequence[str],
    candidate_features: Sequence[str],
    rolling_windows: Sequence[RollingWindow],
    permutation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": BEAM_SCHEMA_VERSION,
        "method": "beam_residual_v3_backward_search",
        "adaptation": {
            "source_contract": "Beam Residual v3",
            "candidate_universe": "13 right-side v2 incremental rule factors",
            "fixed_context": "admitted project + 105 legacy rule + task one-hot",
            "primary_ranking_metric": "pr_auc_instead_of_source_roc_auc",
            "reliability_policy": "constant_1_no_homologous_quality_gate",
            "search_space_reduction": "none",
        },
        "outer_fold": str(outer_fold),
        "test_data_used": False,
        "test_data_used_for_search": False,
        "ranking_objective": "median_delta_pr_auc_primary_no_returns",
        "settings": asdict(settings),
        "context_features": list(context_features),
        "context_features_sha256": feature_columns_sha256(context_features),
        "candidate_features": list(candidate_features),
        "candidate_features_sha256": feature_columns_sha256(candidate_features),
        "selected_features": list(result.best.selected),
        "selected_features_sha256": feature_columns_sha256(result.best.selected),
        "selected_score": asdict(result.best.candidate_score),
        "visited_combinations": len(result.evaluations),
        "layers": [[list(state) for state in layer] for layer in result.layers],
        "rolling_windows": [
            {
                "name": window.name,
                "history_date_min": str(min(window.history_dates).date()),
                "history_date_max": str(max(window.history_dates).date()),
                "evaluation_date_min": str(min(window.evaluation_dates).date()),
                "evaluation_date_max": str(max(window.evaluation_dates).date()),
            }
            for window in rolling_windows
        ],
        "permutation": dict(permutation),
    }


__all__ = [
    "BEAM_SCHEMA_VERSION",
    "BeamCandidateScore",
    "BeamEvaluation",
    "BeamFoldLift",
    "BeamSearchResult",
    "BeamSettings",
    "RollingWindow",
    "backward_beam_search",
    "build_validation_quarter_windows",
    "build_validation_search_and_pipeline_select",
    "combine_residual_probabilities",
    "deterministic_stratified_event_sample",
    "evaluate_pipeline_select_variants",
    "evaluation_record",
    "feature_columns_sha256",
    "ranking_lift",
    "score_candidate",
    "search_manifest",
    "visited_candidate_permutation_test",
]
