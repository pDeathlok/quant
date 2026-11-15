"""Counterfactual label accounting, maturity purge and causal state features."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from test_index_futures_funded import row
import evaluate_index_futures_exit_value as research


def paths(roll=False):
    previous = row(price=1000, entry="20220104", exit="20220105")
    current = row(price=1000, entry="20220105", exit="20220106")
    current.trade_date = "20220104"
    current.exit_open = 1100
    if roll:
        current.ts_code = "IF2207"
    opens = {("20220105", "IF2206"): 1000}
    closes = {("20220105", current.ts_code): 900}
    return [previous, current], opens, closes


def test_incremental_label_cancels_sunk_cost_and_immediate_close_fee():
    rows, opens, closes = paths()
    value = research.hold_path(rows, 1, 1, opens, closes, .0003, 1)[0]
    assert value[0] == "20220106"
    assert np.isclose(value[1], 30000-(330000-300000)*.0003)
    assert np.isclose(value[2], 30000-(300000-270000)*.0003)
    rows[0].entry_open = 1  # old entry PnL must not enter a future incremental label
    assert research.hold_path(rows, 1, 1, opens, closes, .0003, 1)[0] == value


def test_roll_fees_and_short_path_are_exact():
    rows, opens, closes = paths(roll=True)
    long = research.hold_path(rows, 1, 1, opens, closes, .0003, 1)[0]
    short = research.hold_path(rows, 1, -1, opens, closes, .0003, 1)[0]
    assert np.isclose(long[1], 30000-90-99)
    assert np.isclose(short[1], -30000-90-99)
    assert np.isclose(long[2], 30000+90+81)
    assert np.isclose(short[2], 30000+90+99)


def test_flat_no_roll_has_zero_incremental_cost_not_double_roundtrip():
    rows, opens, closes = paths()
    rows[1].exit_open = 1000
    closes["20220105", "IF2206"] = 1000
    assert research.hold_path(rows, 1, 1, opens, closes, .0006, 1)[0][1:] == (0., -0.)


def test_targets_crossing_training_boundary_are_purged():
    states = pd.DataFrame(dict(direction=[1]*4, signal_date=["20211201"]*4,
                               maturity=["20211230", "20211231", "20220105", None]))
    for target in research.TARGETS:
        states[target] = 0.
    train = research.training_before(states, "20211231", 1)
    assert train.index.tolist() == [0]


def test_episode_weights_do_not_treat_nineteen_states_as_nineteen_trades():
    train = pd.DataFrame(dict(episode=["a", "a", "b"], maturity=["20211201"]*3))
    weights = research.state_weights(train, "20220101")
    assert np.isclose(weights[:2].sum(), weights[2])
    assert np.isclose(weights.mean(), 1)


def test_callback_two_confirmations_reset_on_entry_and_warmup():
    row_ = SimpleNamespace(symbol="IF")
    def state(net, risk=.02):
        return SimpleNamespace(direction=1, pred_net_remaining_3=net, pred_risk_remaining_3=risk, notional=1000000)
    lookup = {("a", "1", "IF"): state(.005), ("a", "2", "IF"): state(.005),
              ("b", "3", "IF"): state(.005), ("b", "4", "IF"): state(np.nan),
              ("b", "5", "IF"): state(.005)}
    utility = research.exit_callback(lookup, "utility_remaining", .0003)
    assert not utility("1", "a", row_, 1)[0]
    assert utility("2", "a", row_, 1)[0]
    assert not utility("3", "b", row_, 1)[0]
    assert not utility("4", "b", row_, 1)[0]
    assert not utility("5", "b", row_, 1)[0]
    mean = research.exit_callback(lookup, "mean_remaining", .0003)
    assert not mean("1", "a", row_, 1)[0]
    assert not mean("2", "a", row_, 1)[0]


def test_candidate_state_features_ignore_unknown_open_and_later_prices():
    rows, opens, closes = paths()
    frame = pd.DataFrame([vars(r) for r in rows])
    # Include only two rows: the state is at Jan 4 close, decision Jan 5 open.
    index = pd.MultiIndex.from_tuples([("20220104", "IF2206"), ("20220105", "IF2206"),
                                       ("20220106", "IF2206")], names=["trade_date", "ts_code"])
    bars = pd.DataFrame(dict(open=[1000, 1000, 1100], close=[980, 900, 1100]), index=index)
    a = research.build_states(frame, bars, [])
    frame.loc[0, "exit_open"] = 1200
    frame.loc[1, ["entry_open", "exit_open"]] = [1200, 2000]
    bars.loc[("20220105", "IF2206"), "open"] = 1200
    bars.loc[("20220105", "IF2206"), "close"] = 1900
    b = research.build_states(frame, bars, [])
    np.testing.assert_allclose(a[research.STATE_FEATURES], b[research.STATE_FEATURES])
    assert np.isclose(a.marked_return.iloc[0], -.02)


def test_fixed5_supplement_uses_age_not_model_scores():
    state = SimpleNamespace(direction=-1, age=4)
    lookup = {("entry", "today", "IF"): state}
    choose = research.exit_callback(lookup, "fixed5", .0003)
    assert not choose("today", "entry", SimpleNamespace(symbol="IF"), -1)[0]
    state.age = 5
    result, detail = choose("today", "entry", SimpleNamespace(symbol="IF"), -1)
    assert result and detail["exit_action"] == "fixed5_exit"
