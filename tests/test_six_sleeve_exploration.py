"""Checks of time alignment, self-financing, and risk budgeting for the experiment."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "scripts/research/explore_six_sleeves.py"
spec = importlib.util.spec_from_file_location("six_sleeve_exploration", PATH)
lab = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lab
spec.loader.exec_module(lab)


def panel(values):
    prices = pd.DataFrame(values, index=pd.date_range("2021-01-01", periods=len(values)))
    return prices, prices.notna()


def test_signal_cannot_earn_execution_day_return():
    p, v = panel([[100], [200], [300], [330]])
    r = lab.simulate(p, v, {0: np.array([1.])}, cost=0, lag=1)
    np.testing.assert_allclose(r.returns, [0, 0, .5, .1], atol=1e-12)


def test_cash_ledger_charges_buy_and_sell_without_borrowing():
    p, v = panel([[100], [100], [100], [100]])
    r = lab.simulate(p, v, {0: np.array([1.]), 2: np.array([0.])}, cost=.01)
    assert abs((1+r.returns).prod() - .99/1.01) < 1e-12
    assert r.weights.sum(axis=1).max() <= 1


def test_positions_drift_without_hidden_daily_rebalancing():
    p, v = panel([[100, 100], [100, 100], [200, 100], [400, 100]])
    r = lab.simulate(p, v, {0: np.array([.5, .5])}, cost=0)
    assert abs((1+r.returns).prod() - 2.5) < 1e-12
    np.testing.assert_allclose(r.weights.iloc[-1], [.8, .2])


def test_missing_quote_blocks_trade_and_retains_existing_holding():
    p, v = panel([[100], [100], [100], [110]])
    v.iloc[2, 0] = False
    r = lab.simulate(p, v, {0: np.array([1.]), 1: np.array([0.])}, cost=0)
    assert len(r.skipped) == 1
    assert abs(r.returns.iloc[-1]-.1) < 1e-12


def test_erc_accounts_for_correlations():
    cov = np.array([[.04, .006, -.002], [.006, .01, .001], [-.002, .001, .0225]])
    w = lab.erc(cov)
    rc = w*(cov@w)/(w@cov@w)
    np.testing.assert_allclose(rc, np.ones(3)/3, atol=1e-7)
    assert w.min() > 0 and abs(w.sum()-1) < 1e-12


def test_drawdown_includes_first_day_loss():
    r = pd.Series([-.1, .05], index=pd.date_range("2021-01-01", periods=2))
    assert abs(lab.metrics(r)["max_drawdown"]+.1) < 1e-12


def test_future_price_changes_cannot_change_past_strategy_signals():
    dates = pd.bdate_range("2018-01-01", "2023-12-31")
    rng = np.random.default_rng(12)
    prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(.0002, .01, (len(dates), 10)), axis=0)),
                          index=dates, columns=lab.CODES)
    valid = pd.DataFrame(True, index=dates, columns=lab.CODES)
    before, _, alloc_before = lab.experiment(prices, valid)
    changed = prices.copy()
    cutoff = pd.Timestamp("2023-01-01")
    changed.loc[changed.index >= cutoff] *= 2
    after, _, alloc_after = lab.experiment(changed, valid)
    for name in before:
        np.testing.assert_allclose(before[name].returns.loc[:"2022-12-30"],
                                   after[name].returns.loc[:"2022-12-30"], atol=1e-12)
    pd.testing.assert_frame_equal(alloc_before[alloc_before.date < "2023-01-01"],
                                  alloc_after[alloc_after.date < "2023-01-01"])
