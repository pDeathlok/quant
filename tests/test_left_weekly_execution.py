import numpy as np
import pandas as pd

from quant.research.left_weekly_execution import (
    DEVELOPMENT,
    development_rank,
    materialize,
    schedule_paths,
)
from quant.research.sequence_strategy_validation import capital_backtest


def path(values):
    close = np.full((1, 22), 100.0)
    close[0, :len(values)] = values
    return close


def test_stop_confirms_at_close_then_fills_next_close_gap():
    close = path([100, 100, 91, 80])
    ma = np.full_like(close, 95)
    start, end, confirmation, trigger = schedule_paths(close, ma, ma, "next_close", "stop8")
    assert (start[0], end[0], confirmation[0], trigger[0]) == (1, 3, 0, 2)
    assert close[0, end[0]] / close[0, start[0]] - 1 == -0.19999999999999996


def test_wait_confirmation_executes_next_session_and_expires():
    close = np.concatenate([path([100, 99, 98, 101, 105]), path([100, 99, 98, 97, 110])])
    ma = np.full_like(close, 99)
    start, end, confirmation, trigger = schedule_paths(close, ma, ma, "wait_green", "fixed20")
    assert start.tolist() == [4, -1]
    assert confirmation.tolist() == [3, -1]
    assert end.tolist() == [21, 21]
    assert trigger.tolist() == [-1, -1]


def test_reclaim_requires_cross_and_already_known_ma():
    close = path([100, 102, 99, 101])
    ma = np.full_like(close, 100)
    start, _, confirmation, _ = schedule_paths(close, ma, ma, "reclaim_ma5", "fixed20")
    assert start[0] == 2 and confirmation[0] == 1
    close[0, 0] = 101
    start, _, confirmation, _ = schedule_paths(close, ma, ma, "reclaim_ma5", "fixed20")
    assert start[0] == 4 and confirmation[0] == 3


def test_later_prices_cannot_change_confirmed_entry_or_triggered_exit():
    close = path([100, 99, 101, 102, 90, 80])
    ma = np.full_like(close, 100)
    first = schedule_paths(close, ma, ma, "wait_green", "stop8")
    close[:, 6:] *= 2
    second = schedule_paths(close, ma, ma, "wait_green", "stop8")
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert first[1][0] == 5 and first[3][0] == 4


def test_trailing_stop_uses_entry_onward_peak_and_next_close():
    close = path([150, 100, 109, 99, 90])
    ma = np.full_like(close, 100)
    _, end, _, trigger = schedule_paths(close, ma, ma, "next_close", "trail8")
    assert end[0] == 4 and trigger[0] == 3


def test_materialize_fixed_twenty_and_confirmation_dedup():
    calendar = pd.bdate_range("2025-01-01", periods=30)
    events = pd.DataFrame({
        "symbol": ["A", "A"], "date": calendar[:2],
        "entry_date": calendar[1:3], "exit_date": calendar[21:23],
    })
    close = np.vstack([path([100, 99, 101])[0], path([99, 101])[0]])
    paths = {"close": close, "ma5": np.full_like(close, 100), "ma10": np.full_like(close, 100)}
    fixed, _ = materialize(events, paths, calendar, "next_close", "fixed20")
    assert set(fixed.entry_date) == set(calendar[1:3])
    np.testing.assert_allclose(
        fixed.sort_values("origin_date").net_return,
        [100 / 99 - 1 - 0.002, 100 / 101 - 1 - 0.002],
    )
    waited, audit = materialize(events, paths, calendar, "wait_green", "fixed20")
    assert len(waited) == 1 and audit["duplicate_confirmations_removed"] == 1
    assert waited.iloc[0].origin_date == calendar[1]
    assert waited.iloc[0].entry_date == calendar[3]


def test_capital_engine_defers_stop_on_blocked_sell():
    calendar = pd.bdate_range("2025-01-01", periods=6)
    events = pd.DataFrame({"symbol": ["A"], "date": [calendar[0]],
                           "entry_date": [calendar[1]], "exit_date": [calendar[3]],
                           "execution_score": [1.0]})
    bars = pd.DataFrame({"date": calendar, "close": [100, 100, 91, 80, 75, 90],
                         "buy_ok": True, "sell_ok": [True, True, True, False, True, True]})
    _, trades, _ = capital_backtest(events, {"A": bars}, calendar, slots=1)
    assert trades.iloc[0].exit_date == calendar[4]
    assert trades.iloc[0].net_return < -0.25


def test_development_ranking_does_not_read_test_and_requires_activity():
    record = {p: {"trades": 60, "max_drawdown": -0.1, "total_return": 0.1} for p in DEVELOPMENT}
    a = development_rank(record)
    record["test"] = {"total_return": 100}
    assert development_rank(record) == a
    record[DEVELOPMENT[0]]["trades"] = 39
    assert development_rank(record)[0] == -float("inf")


def test_component_only_ranking_keeps_final_drawdown_constraint():
    record = {p: {"trades": 60, "max_drawdown": -0.36, "total_return": 0.1} for p in DEVELOPMENT}
    assert development_rank(record)[0] == -float("inf")
    assert development_rank(record, enforce_drawdown=False)[0] == 0.1
    record[DEVELOPMENT[0]]["trades"] = 39
    assert development_rank(record, enforce_drawdown=False)[0] == -float("inf")
