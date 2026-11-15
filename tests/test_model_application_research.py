import numpy as np
import pandas as pd

from scripts.research.validate_model_applications import Policy, correlated, simulate


def market_fixture():
    shape = (30, 2)
    price = np.full(shape, 100.)
    market = {
        'price': price, 'mark': price.copy(), 'buyable': np.ones(shape, dtype=bool),
        'sellable': np.ones(shape, dtype=bool), 'observed': np.ones(shape, dtype=bool),
        'gap': np.zeros(shape), 'returns': np.zeros(shape),
    }
    group = pd.DataFrame({'sid': [0], 'score': [1.], 'volatility_20d': [.5], 'side': ['left']}).to_records(index=False)
    return market, {0: group}


def test_next_session_entry_and_five_session_exit_conserve_cash():
    market, groups = market_fixture()
    metrics, nav, trades = simulate(Policy(positions=1), groups, market, 1, 30, 1.)
    assert trades.iloc[0].entry_day == 1
    assert trades.iloc[0].exit_day == 6
    assert np.isclose(metrics['return'], .999 / 1.001 - 1)
    assert np.isclose(metrics['return'], -metrics['fees'])
    assert np.isclose(nav.iloc[-1].cash, nav.iloc[-1].nav)


def test_untradeable_sale_defers_without_fabricating_cash():
    market, groups = market_fixture()
    market['sellable'][6, 0] = False
    metrics, nav, trades = simulate(Policy(positions=1), groups, market, 1, 30, 1.)
    assert trades.iloc[0].exit_day == 7
    assert metrics['deferred_sale_days'] == 1
    assert nav.loc[nav.day.eq(6), 'cash'].iloc[0] == 0


def test_unfilled_entry_does_not_create_position_or_fee():
    market, groups = market_fixture()
    market['buyable'][1, 0] = False
    metrics, _, trades = simulate(Policy(positions=1), groups, market, 1, 30, 1.)
    assert trades.empty
    assert metrics['return'] == metrics['fees'] == 0
    assert metrics['no_fills'] == 1


def test_future_prices_do_not_change_past_orders_or_nav():
    market, groups = market_fixture()
    _, original, _ = simulate(Policy(positions=1), groups, market, 1, 30, 1.)
    market['price'][15:] *= 2
    market['mark'][15:] *= 2
    _, revised, _ = simulate(Policy(positions=1), groups, market, 1, 30, 1.)
    pd.testing.assert_frame_equal(original.loc[original.day.lt(15)], revised.loc[revised.day.lt(15)])


def test_correlation_uses_only_signal_day_and_earlier_returns():
    rng = np.random.default_rng(4)
    returns = rng.normal(size=(100, 2))
    result = correlated(0, [1], returns, 40)
    returns[41:, 1] = returns[41:, 0]
    assert correlated(0, [1], returns, 40) == result
