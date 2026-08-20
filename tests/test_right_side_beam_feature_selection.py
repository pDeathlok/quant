from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.right_side_beam_feature_selection import (
    BEAM_SCHEMA_VERSION,
    BeamEvaluation,
    BeamFoldLift,
    BeamSettings,
    backward_beam_search,
    build_validation_quarter_windows,
    build_validation_search_and_pipeline_select,
    combine_residual_probabilities,
    deterministic_stratified_event_sample,
    evaluate_pipeline_select_variants,
    feature_columns_sha256,
    ranking_lift,
    search_manifest,
    score_candidate,
    visited_candidate_permutation_test,
)


def test_quarter_windows_never_accept_or_reach_outer_test_dates() -> None:
    train = pd.bdate_range("2020-01-01", "2022-12-31")
    validation = pd.bdate_range("2023-01-01", "2023-12-31")
    windows = build_validation_quarter_windows(train, validation)
    assert len(windows) == 3
    assert min(windows[0].evaluation_dates) > max(windows[0].history_dates)
    assert max(windows[-1].evaluation_dates).year == 2023
    assert all(max(window.evaluation_dates).year < 2024 for window in windows)


def test_pipeline_select_tail_is_disjoint_from_all_search_windows() -> None:
    train = pd.bdate_range("2020-01-01", "2022-12-31")
    validation = pd.bdate_range("2023-01-01", "2023-12-31")
    windows, pipeline = build_validation_search_and_pipeline_select(train, validation)
    search_evaluation_dates = {
        value
        for window in windows
        for value in window.evaluation_dates
    }
    assert search_evaluation_dates.isdisjoint(pipeline.evaluation_dates)
    assert max(window.evaluation_dates[-1] for window in windows) < min(
        pipeline.evaluation_dates
    )
    assert max(pipeline.evaluation_dates).year == 2023


def test_development_sampling_is_deterministic_stratified_and_capped() -> None:
    rows = 200
    frame = pd.DataFrame(
        {
            "symbol": [f"S{index:03d}" for index in range(rows)],
            "date": pd.bdate_range("2023-01-02", periods=rows),
            "target": np.arange(rows) % 2,
            "B2": np.arange(rows) % 3 == 0,
            "B3": np.arange(rows) % 3 != 0,
        }
    )
    first, metadata = deterministic_stratified_event_sample(
        frame,
        maximum_rows=80,
        label_column="target",
        signal_columns=("B2", "B3"),
        random_seed=42,
    )
    shuffled, _ = deterministic_stratified_event_sample(
        frame.sample(frac=1.0, random_state=9),
        maximum_rows=80,
        label_column="target",
        signal_columns=("B2", "B3"),
        random_seed=42,
    )
    assert len(first) == 80
    assert first[["symbol", "date"]].equals(shuffled[["symbol", "date"]])
    assert metadata["original_rows"] == 200
    assert metadata["sampled_rows"] == 80
    assert metadata["sampling_applied"]
    assert set(first["target"]) == {0, 1}


def test_score_is_pr_auc_primary_and_stability_gated() -> None:
    settings = BeamSettings(permutation_rounds=3)
    strong = score_candidate(
        [
            BeamFoldLift("f1", 0.03, 0.01, 0.001),
            BeamFoldLift("f2", 0.02, 0.01, 0.001),
            BeamFoldLift("f3", 0.01, 0.01, 0.001),
        ],
        n_features=8,
        settings=settings,
    )
    unstable = score_candidate(
        [
            BeamFoldLift("f1", 0.05, 0.10, 0.10),
            BeamFoldLift("f2", -0.01, 0.10, 0.10),
            BeamFoldLift("f3", -0.01, 0.10, 0.10),
        ],
        n_features=8,
        settings=settings,
    )
    assert strong.eligible
    assert not unstable.eligible


def test_v3_adaptation_searches_full_13_feature_universe_to_effective_depth_7() -> None:
    settings = BeamSettings()
    settings.validate(13)
    assert "v3-adaptation" in BEAM_SCHEMA_VERSION
    assert settings.width == 4
    assert settings.min_features == 6
    assert settings.max_remove == 10
    assert min(settings.max_remove, 13 - settings.min_features) == 7
    assert settings.logloss_weight == 5.0
    assert settings.roc_auc_weight == 0.0


def test_residual_probability_matches_v3_offset_equation() -> None:
    baseline_probability = np.array([0.2, 0.5, 0.8])
    baseline_margin = np.array([-1.0, 0.0, 1.0])
    candidate_margin = baseline_margin + np.array([0.4, -0.2, 0.1])
    combined = combine_residual_probabilities(
        baseline_probability,
        baseline_margin,
        candidate_margin,
    )
    baseline_logit = np.log(baseline_probability / (1.0 - baseline_probability))
    expected = 1.0 / (1.0 + np.exp(-(baseline_logit + candidate_margin - baseline_margin)))
    np.testing.assert_allclose(combined, expected)

    half = combine_residual_probabilities(
        baseline_probability,
        baseline_margin,
        candidate_margin,
        reliability=0.5,
    )
    expected_half = 1.0 / (
        1.0 + np.exp(-(baseline_logit + 0.5 * (candidate_margin - baseline_margin)))
    )
    np.testing.assert_allclose(half, expected_half)


def test_pipeline_select_falls_back_to_baseline_when_residual_is_unstable() -> None:
    labels = np.array([0, 1] * 20)
    baseline = np.where(labels == 1, 0.65, 0.35)
    worse = 1.0 - baseline
    decision = evaluate_pipeline_select_variants(
        labels,
        baseline,
        soft_gate_probability=worse,
        no_gate_probability=np.full(len(labels), 0.5),
    )
    assert decision["selected_variant"] == "baseline_only"
    assert not decision["gate_passed"]


def test_pipeline_select_chooses_positive_stable_residual() -> None:
    labels = np.array([0, 1, 0, 1] * 20)
    baseline = np.linspace(0.55, 0.45, len(labels))
    soft = np.where(labels == 1, 0.65, 0.35)
    no_gate = np.where(labels == 1, 0.85, 0.15)
    decision = evaluate_pipeline_select_variants(
        labels,
        baseline,
        soft_gate_probability=soft,
        no_gate_probability=no_gate,
    )
    assert decision["selected_variant"] == "no_gate_1p0"
    assert decision["gate_passed"]


def test_v3_stability_gate_requires_nonnegative_median_logloss() -> None:
    settings = BeamSettings(permutation_rounds=3)
    score = score_candidate(
        [
            BeamFoldLift("f1", 0.03, 0.10, -0.01),
            BeamFoldLift("f2", 0.02, 0.10, -0.01),
            BeamFoldLift("f3", 0.01, 0.10, 0.00),
        ],
        n_features=8,
        settings=settings,
    )
    assert not score.eligible


def test_backward_beam_caches_duplicate_paths_and_selects_global_best() -> None:
    features = ("a", "b", "c", "d")
    settings = BeamSettings(
        width=3,
        min_features=2,
        max_remove=2,
        positive_fold_requirement=1,
        permutation_rounds=3,
    )
    calls: list[tuple[str, ...]] = []

    def evaluate(removed: tuple[str, ...]) -> BeamEvaluation:
        calls.append(removed)
        selected = tuple(feature for feature in features if feature not in removed)
        value = 0.04 if removed == ("d",) else 0.01 - len(removed) * 0.001
        lifts = tuple(BeamFoldLift(f"f{i}", value, 0.0, 0.0) for i in range(3))
        score = score_candidate(lifts, n_features=len(selected), settings=settings)
        predictions = tuple(
            {
                "fold": f"f{i}",
                "labels": np.array([0, 1, 0, 1]),
                "baseline_probability": np.array([0.4, 0.6, 0.4, 0.6]),
                "candidate_probability": np.array([0.3, 0.7, 0.3, 0.7]),
            }
            for i in range(3)
        )
        return BeamEvaluation(removed, selected, score, lifts, predictions)

    result = backward_beam_search(
        candidate_features=features, evaluate=evaluate, settings=settings
    )
    assert result.best.removed == ("d",)
    assert len(calls) == len(set(calls)) == len(result.evaluations)


def test_default_v3_search_reaches_full_effective_depth_without_prefilter() -> None:
    features = tuple(f"f{index}" for index in range(13))
    settings = BeamSettings(permutation_rounds=3)

    def evaluate(removed: tuple[str, ...]) -> BeamEvaluation:
        selected = tuple(feature for feature in features if feature not in removed)
        lift_value = 0.01 + len(removed) * 0.0001
        lifts = tuple(
            BeamFoldLift(f"roll_{index}", lift_value, 0.0, 0.001)
            for index in range(3)
        )
        score = score_candidate(lifts, n_features=len(selected), settings=settings)
        return BeamEvaluation(removed, selected, score, lifts, ())

    result = backward_beam_search(
        candidate_features=features,
        evaluate=evaluate,
        settings=settings,
    )
    assert len(result.layers) == 8
    assert all(len(state) == 7 for state in result.layers[-1])
    assert len(result.best.selected) == settings.min_features
    assert len(result.evaluations) > len(features) + 1


def test_ranking_lift_and_permutation_use_only_labels_and_scores() -> None:
    labels = np.array([0, 1, 0, 1] * 4)
    baseline = np.linspace(0.7, 0.3, len(labels))
    candidate = np.where(labels == 1, 0.8, 0.2)
    lift = ranking_lift(labels, baseline, candidate, fold="f1")
    assert lift.delta_pr_auc > 0
    assert lift.delta_roc_auc > 0
    settings = BeamSettings(
        width=1,
        min_features=1,
        max_remove=0,
        positive_fold_requirement=1,
        permutation_rounds=5,
    )
    score = score_candidate([lift], n_features=1, settings=settings)
    evaluation = BeamEvaluation(
        (),
        ("x",),
        score,
        (lift,),
        (
            {
                "fold": "f1",
                "labels": labels,
                "baseline_probability": baseline,
                "candidate_probability": candidate,
            },
        ),
    )
    result = visited_candidate_permutation_test(
        [evaluation],
        settings=settings,
        observed_best_score=evaluation.score,
        random_seed=7,
    )
    assert result["rounds"] == 5
    assert 0 < result["empirical_p_value"] <= 1


def test_manifest_freezes_v3_provenance_and_test_exclusion() -> None:
    settings = BeamSettings(
        width=1,
        min_features=1,
        max_remove=0,
        positive_fold_requirement=1,
        permutation_rounds=3,
    )
    lifts = (BeamFoldLift("roll_1", 0.01, 0.0, 0.001),)
    score = score_candidate(lifts, n_features=1, settings=settings)
    evaluation = BeamEvaluation((), ("new_factor",), score, lifts, ())
    search = type("Search", (), {})()
    search.best = evaluation
    search.evaluations = (evaluation,)
    search.layers = (((),),)
    window = type("Window", (), {})()
    window.name = "roll_1"
    window.history_dates = (pd.Timestamp("2023-01-02"),)
    window.evaluation_dates = (pd.Timestamp("2023-02-01"),)
    manifest = search_manifest(
        search,
        settings=settings,
        outer_fold="A",
        context_features=("context",),
        candidate_features=("new_factor",),
        rolling_windows=(window,),
        permutation={"empirical_p_value": 0.01},
    )
    assert manifest["schema_version"] == BEAM_SCHEMA_VERSION
    assert manifest["test_data_used"] is False
    assert manifest["test_data_used_for_search"] is False
    assert manifest["adaptation"]["search_space_reduction"] == "none"
    assert manifest["selected_features_sha256"] == feature_columns_sha256(
        ("new_factor",)
    )
