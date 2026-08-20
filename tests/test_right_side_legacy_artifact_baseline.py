from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from quant.research.right_side_legacy_artifact_baseline import (
    aggregate_legacy_event_predictions,
    build_legacy_overlap_rows,
    evaluate_legacy_event_predictions,
    legacy_overlap_coverage,
    load_legacy_z_artifacts,
    score_legacy_overlap_rows,
)


@dataclass
class _FakeArtifact:
    feature_names_in_: list[str]
    selected_features_: list[str]
    best_iteration: int = 3
    factor_schema_version_: str | None = None
    offset: float = 0.0

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.clip(frame[self.feature_names_in_[0]].to_numpy(dtype=float) + self.offset, 0.01, 0.99)
        return np.column_stack([1.0 - positive, positive])


def _models() -> dict[tuple[str, str], _FakeArtifact]:
    return {
        (signal, label): _FakeArtifact(["legacy_factor"], ["legacy_factor"], offset=offset)
        for signal, offset in (("B2", 0.0), ("KEY_K", 0.1))
        for label in ("up5", "up8", "down3")
    }


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2024-01-02", "2025-01-02", "2026-01-02"])
    events = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "date": dates,
            "B2": [True, True, True],
            "KEY_K": [False, True, False],
        }
    )
    labels = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "date": dates,
            "entry_mode": ["next_open"] * 3,
            "horizon": [5] * 3,
            "mature": [True, True, True],
            "locked_limit_up": [False, False, True],
            "hit_up5": [True, False, True],
            "hit_up8": [False, False, True],
            "good_path5": [True, False, True],
            "terminal_return": [0.06, -0.02, 0.10],
        }
    )
    legacy = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "date": dates[:2],
            "B2": [True, False],
            "KEY_K": [False, True],
            "legacy_factor": [0.8, 0.2],
        }
    )
    return events, labels, legacy


def test_overlap_uses_legacy_factors_and_discloses_signal_timing() -> None:
    events, labels, legacy = _frames()
    overlap = build_legacy_overlap_rows(
        events,
        labels,
        legacy,
        _models(),
        signals=("B2", "KEY_K"),
    )

    assert len(overlap) == 3
    assert overlap["legacy_factor_available"].all()
    assert overlap.loc[overlap["signal"].eq("B2"), "legacy_signal_timing_match"].tolist() == [True, False]
    assert overlap.loc[overlap["signal"].eq("KEY_K"), "legacy_signal_timing_match"].tolist() == [True]
    assert set(overlap["fold"]) == {"A", "B"}
    assert "C" not in overlap["symbol"].tolist()  # locked next-session limit-up is excluded


def test_scoring_and_event_aggregation_keep_multi_hit_event_once() -> None:
    events, labels, legacy = _frames()
    overlap = build_legacy_overlap_rows(
        events,
        labels,
        legacy,
        _models(),
        signals=("B2", "KEY_K"),
    )
    scored = score_legacy_overlap_rows(overlap, _models())
    event = aggregate_legacy_event_predictions(scored)

    assert scored["legacy_scored"].all()
    assert len(event) == 2
    b = event[event["symbol"].eq("B")].iloc[0]
    assert b["legacy_covered_signal_count"] == 2
    assert b["legacy_signals"] == "B2,KEY_K"
    assert np.isclose(b["legacy_pred_up5"], 0.3)
    metrics = evaluate_legacy_event_predictions(event, top_fraction=0.5)
    primary = metrics[
        metrics["target"].eq("hit_up5")
        & metrics["score"].eq("legacy_pred_up5")
        & metrics["fold"].eq("A")
    ].iloc[0]
    assert primary["semantic_match"] == "near_equivalent_after_new_sample_gates"


def test_coverage_denominator_is_canonical_event_signal_rows() -> None:
    events, labels, legacy = _frames()
    overlap = build_legacy_overlap_rows(
        events,
        labels,
        legacy,
        _models(),
        signals=("B2", "KEY_K"),
    )
    coverage = legacy_overlap_coverage(overlap).set_index("signal")

    assert coverage.loc["B2", "canonical_rows"] == 2
    assert coverage.loc["B2", "legacy_factor_coverage"] == 1.0
    assert coverage.loc["B2", "legacy_signal_timing_match_rate"] == 0.5
    assert coverage.loc["KEY_K", "legacy_signal_timing_match_rate"] == 1.0


def test_loader_infers_legacy_schema_for_unversioned_artifacts(tmp_path) -> None:
    for label in ("up5", "up8", "down3"):
        joblib.dump(
            _FakeArtifact(["legacy_factor"], ["legacy_factor"]),
            tmp_path / f"B2_{label}.joblib",
        )

    models, contracts = load_legacy_z_artifacts(tmp_path, signals=("B2",))

    assert len(models) == 3
    assert len(contracts) == 3
    assert contracts["inferred_factor_schema"].eq(
        "project-v1-latest-scale-global-rank"
    ).all()
    assert contracts["sha256"].str.len().eq(64).all()
