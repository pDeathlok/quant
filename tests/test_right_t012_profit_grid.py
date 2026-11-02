import numpy as np
import pandas as pd
import pytest
from run_right_t012_profit_grid import cashbook, exit_returns, pick_indices, portfolio_metrics


def bars() -> pd.DataFrame:
    return pd.DataFrame(dict(entry_close=[10.] * 3, next_open=[10., 9., 10.],
        next_high=[10.21, 10.20, 10.01], next_low=[9.5, 9., 9.], next_close=[10.1, 9., 9.],
        next_valid_bar=[True] * 3, down_limit=[9.] * 3, price_limit_inconsistent=[False] * 3))


def test_limit_requires_penetration_and_keeps_unknown_fallback() -> None:
    result = exit_returns(bars())
    assert result[0, 3] == pytest.approx(.02)
    assert np.isnan(result[1, 3])
    assert np.isnan(result[2, 1])
    assert result[0, 6] == pytest.approx(.015)


def test_gap_open_exit_cannot_assume_lower_limit_fill() -> None:
    result = exit_returns(bars())
    assert np.isnan(result[1, 7])
    assert result[0, 7] == pytest.approx(.01)


def test_selection_only_consults_supplied_rank_mask_and_date() -> None:
    dates = np.array([0, 0, 1, 1])
    order = np.array([1, 0, 3, 2])
    assert pick_indices(order, np.array([True, True, True, False]), dates, 1).tolist() == [1, 2]
    assert pick_indices(order, np.array([True, True, True, False]), dates, 3).tolist() == [1, 0, 2]


def test_cash_slots_and_unknown_stress_are_not_renormalized() -> None:
    gross = np.array([.02, np.nan])
    result, curve = portfolio_metrics(gross, np.array([0, 1]), 3, 5, 40, -.10)
    assert curve.tolist() == pytest.approx([.0016, -.01, 0.])
    assert result["unknown"] == 1
    assert result["net_return_scenario"] == pytest.approx(-.0084)
    assert result["max_drawdown_scenario"] > 0


def test_cashbook_never_uses_same_close_proceeds_for_new_purchase() -> None:
    days = pd.date_range("2025-01-01", periods=4)
    rows = pd.DataFrame(dict(date=days[:3], planned_exit_date=days[1:], gross=[-.2, 0., 0.]))
    result, nav = cashbook(rows, days, 1, 0, -.3)
    assert nav.tolist() == pytest.approx([1., .9, .9, .9])
    assert result["cashbook_cash"] == pytest.approx(.9)


def test_unknown_exit_does_not_release_cash() -> None:
    days = pd.date_range("2025-01-01", periods=3)
    rows = pd.DataFrame(dict(date=days[:2], planned_exit_date=days[1:], gross=[np.nan, 0.]))
    result, _ = cashbook(rows, days, 1, 0, -.3)
    assert result["cashbook_locked_value"] == pytest.approx(.35)
    assert result["cashbook_cash"] == pytest.approx(.5)
    assert result["cashbook_return"] == pytest.approx(-.15)
