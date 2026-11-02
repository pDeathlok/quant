from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.left_side_regime_gate import (
    compose_absolute_display_score,
    fit_left_side_regime_gate,
    materialize_daily_regime_targets,
)


def test_materialize_daily_regime_targets_uses_all_eligible_events() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02", "2025-01-03"],
            "terminal_return": [0.01, -0.002, 0.02],
            "breadth": [0.4, 0.4, 0.6],
        }
    )

    daily = materialize_daily_regime_targets(
        frame, features=["breadth"], round_trip_cost=0.002
    )

    assert daily["event_count"].tolist() == [2, 1]
    assert daily["daily_average_net_return"].tolist() == pytest.approx(
        [0.002, 0.018]
    )


def test_materialize_daily_regime_targets_rejects_non_market_feature() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "terminal_return": [0.01, 0.02],
            "not_market_state": [0.4, 0.5],
        }
    )

    with pytest.raises(ValueError, match="vary within date"):
        materialize_daily_regime_targets(frame, features=["not_market_state"])


def test_regime_threshold_is_frozen_from_validation_only() -> None:
    train = pd.DataFrame(
        {
            "breadth": [0.1, 0.2, 0.8, 0.9],
            "daily_average_net_return": [-0.02, -0.01, 0.01, 0.02],
        }
    )
    validation = pd.DataFrame(
        {
            "breadth": [0.25, 0.5, 0.75],
            "daily_average_net_return": [-0.01, 0.0, 0.01],
        }
    )
    gate = fit_left_side_regime_gate(
        train,
        validation,
        features=["breadth"],
        admission_quantile=0.5,
        ridge_alpha=1.0,
    )

    validation_score = gate.predict_score(validation)
    assert gate.threshold == pytest.approx(float(np.quantile(validation_score, 0.5)))
    assert gate.predict_open(validation).tolist() == [False, True, True]


def test_absolute_display_score_reserves_upper_half_for_open_regime() -> None:
    score = compose_absolute_display_score(
        np.asarray([0.0, 0.8, 0.0, 1.0]),
        np.asarray([False, False, True, True]),
    )

    assert score.tolist() == pytest.approx([0.0, 40.0, 50.0, 100.0])
