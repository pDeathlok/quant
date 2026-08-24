from __future__ import annotations

import numpy as np

from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
    Lifecycle,
)
from quant.routine.left_side_unified_production import (
    normalize_daily_percentile,
    validate_left_side_production_artifact,
)


def test_daily_percentile_normalization_is_monotonic_and_tie_stable() -> None:
    scores = np.asarray([0.2, 0.1, 0.2, 0.9], dtype=float)
    normalized = normalize_daily_percentile(scores)

    assert normalized.tolist() == [50.0, 0.0, 50.0, 100.0]
    assert normalize_daily_percentile(np.asarray([0.4])).tolist() == [50.0]
    assert normalize_daily_percentile(np.asarray([])).tolist() == []


def test_promoted_left_artifact_and_daily_registry_are_consistent() -> None:
    bundle = validate_left_side_production_artifact()
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    short = set(registry.required_node_ids("short"))

    assert bundle["production_threshold_mode"] == "none_rank_only"
    assert "score.left_side_unified" in short
    assert "score.z_skill" not in short
    assert "score.b1" not in short
    assert registry.nodes["score.z_skill"].lifecycle == Lifecycle.RETIRED
    assert registry.nodes["score.b1"].lifecycle == Lifecycle.RETIRED
