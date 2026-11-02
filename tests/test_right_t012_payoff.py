import numpy as np
import pandas as pd
import pytest
from train_right_t012_payoff_v3 import (
    calibrate_scale,
    payoff_selection,
    payoff_targets,
    recent_mask,
)


def test_payoff_decomposition_preserves_tails_and_unknowns() -> None:
    gross = np.array([.03, -.25, 0., np.nan])
    gain, loss = payoff_targets(gross)
    assert np.allclose(gain-loss, gross-.004, equal_nan=True)
    assert loss[1] == pytest.approx(.254)
    assert gain[2] == 0


def test_recent_window_uses_market_sessions_and_purges_exit() -> None:
    days = pd.date_range("2025-01-01", periods=6)
    frame = pd.DataFrame(dict(date=days[[0, 3, 4]], planned_exit_date=days[[1, 4, 5]]))
    assert recent_mask(frame, days, "2025-01-06", 2).tolist() == [False, True, False]


def test_calibration_is_shrunk_and_bounded() -> None:
    assert calibrate_scale(np.array([2.]), np.array([1.]), np.ones(1), 20) == pytest.approx(1.5)
    assert calibrate_scale(np.array([100.]), np.array([1.]), np.ones(1), 20) == 2.


def test_payoff_selection_ignores_realized_labels() -> None:
    frame = pd.DataFrame(dict(symbol=["a", "b"], date=pd.to_datetime(["2025-01-01"]*2),
        screen_to_entry_return=[.01, .02], entry_right=[True, True], gain=[.03, .025], loss=[.02, .005], gross=[.9, -.9]))
    rule = dict(pool="all_right", entry="any", penalty=1., min_mu=0., top_k=1)
    assert payoff_selection(frame, rule).tolist() == [1]
    frame["gross"] *= -1
    assert payoff_selection(frame, rule).tolist() == [1]
