import numpy as np
import pandas as pd
import pytest

from quant.research.tail_overnight_selection import (
    choose_candidates,
    selection_metrics,
    take_profit_return,
)


def test_take_profit_gap_touch_and_no_touch():
    frame = pd.DataFrame(
        {
            "y_open": [0.03, 0.0, 0.0, np.nan],
            "y_high": [0.04, 0.02, 0.005, np.nan],
            "y_close": [0.01, -0.04, -0.01, np.nan],
        }
    )
    result = take_profit_return(frame, 0.01)
    assert result.iloc[:3].tolist() == pytest.approx([0.028, 0.008, -0.012])
    assert np.isnan(result.iloc[3])


def test_selection_requires_score_risk_and_preserves_cash():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "date": [dates[0]] * 3,
            "predicted_net": [0.02, 0.04, 0.03],
            "predicted_risk": [0.1, 0.8, np.nan],
            "target_net": [0.01, -0.1, 0.01],
        }
    )
    picks = choose_candidates(frame, pool="union", top_k=2, minimum_net=0.01, maximum_risk=0.3)
    assert picks.symbol.tolist() == ["A"]
    result = selection_metrics(picks, dates, top_k=2)
    assert result["daily_mark_mean"] == pytest.approx(0.0025)
    assert result["allocation_fraction"] == 0.25
    assert result["active_days"] == 1


def test_empty_selection_and_missing_outcome_are_distinct():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    frame = pd.DataFrame({"symbol": ["A"], "date": [dates[0]], "target_net": [np.nan]})
    assert selection_metrics(frame.iloc[:0], dates, top_k=1)["daily_mark_mean"] == 0
    result = selection_metrics(frame, dates, top_k=1)
    assert result["missing_outcomes"] == 1
    assert result["daily_mark_mean"] == pytest.approx(-0.05)


def test_selection_does_not_consult_future_outcome():
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "date": pd.to_datetime(["2025-01-02"] * 2),
            "predicted_net": [0.02, 0.01],
            "predicted_risk": [0.1, 0.1],
            "target_net": [-0.5, 0.5],
        }
    )
    policy = dict(pool="union", top_k=1, minimum_net=0.0, maximum_risk=0.4)
    assert choose_candidates(frame, **policy).symbol.tolist() == ["A"]
    frame["target_net"] = [0.5, -0.5]
    assert choose_candidates(frame, **policy).symbol.tolist() == ["A"]


def test_calibration_ties_use_raw_model_rank():
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "date": pd.to_datetime(["2025-01-02"] * 2),
            "predicted_net": [0.01, 0.01],
            "raw_predicted_net": [0.015, 0.02],
            "predicted_risk": [0.1, 0.1],
        }
    )
    picks = choose_candidates(frame, pool="union", top_k=1, minimum_net=0.0, maximum_risk=0.4)
    assert picks.symbol.tolist() == ["B"]
