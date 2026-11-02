from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant.research.byd_t_backtest import ChinaAStockFees
from quant.research.byd_t_confirmation import (
    ConfirmationTConfig,
    confirmation_metrics,
    confirmation_sessions,
    risk_sized_shares,
    selection_qualified,
    simulate_confirmation_t,
)


ZERO_FEES = ChinaAStockFees(0, 0, 0, 0, 0)


def prepared_day() -> pd.DataFrame:
    frame = pd.DataFrame([
        ("2025-01-06 09:55", 99., 99., 98.6, 98.8, False),
        ("2025-01-06 10:00", 98.9, 99.2, 98.9, 99.1, True),
        ("2025-01-06 10:05", 99.1, 99.6, 99., 99.4, False),
        ("2025-01-06 10:10", 99.4, 100.4, 99.3, 100.1, False),
        ("2025-01-06 14:50", 100., 100.2, 99.8, 100., False),
        ("2025-01-06 14:55", 100., 120., 99., 119., False),
    ], columns=["datetime", "open", "high", "low", "close", "turn_up"])
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["date"] = frame["datetime"].dt.normalize()
    frame["minute"] = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute
    for name, value in {
        "previous_close": 100., "prior_atr_pct": .02, "prior_return_60": 0.,
        "prior_ma20_slope_5": 0., "prior_ma20_ma60_gap": 0.,
        "prior_range_width_60": .20, "prior_range_position": .4, "vwap": 100.2,
    }.items():
        frame[name] = value
    return frame


def test_confirmed_order_uses_next_open_and_restores_settled_inventory() -> None:
    result = simulate_confirmation_t(confirmation_sessions(prepared_day()), ConfirmationTConfig(), ZERO_FEES)
    trade = result.trades.iloc[0]

    assert len(result.trades) == 1
    assert trade["signal_time"] == pd.Timestamp("2025-01-06 10:00")
    assert trade["entry_bar_end"] == pd.Timestamp("2025-01-06 10:05")
    assert trade["entry_price"] == 99.1
    assert trade["position_after"] == 10000
    assert trade["planned_loss"] <= 200


def test_pending_confirmation_cannot_cross_lunch() -> None:
    frame = prepared_day()
    frame.loc[1, "datetime"] = pd.Timestamp("2025-01-06 11:30")
    frame.loc[2, "datetime"] = pd.Timestamp("2025-01-06 13:05")
    frame = frame.loc[[1, 2, 4, 5]].copy()
    frame["minute"] = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute

    result = simulate_confirmation_t(confirmation_sessions(frame), replace(ConfirmationTConfig(), latest_entry_minute=840))

    assert result.trades.empty


def test_same_bar_target_and_stop_uses_stop_and_counts_first_loss_drawdown() -> None:
    frame = prepared_day()
    frame.loc[3, ["high", "low"]] = [102., 97.]
    result = simulate_confirmation_t(confirmation_sessions(frame), ConfirmationTConfig(), ZERO_FEES)
    trade = result.trades.iloc[0]
    metrics = confirmation_metrics(result, "2025-01-01", "2025-12-31")

    assert trade["exit_reason"] == "STOP"
    assert trade["exit_price"] == 98.5
    assert trade["pnl"] < 0
    assert metrics["mtm_drawdown_yuan"] >= -trade["pnl"]


def test_later_bar_high_cannot_change_scheduled_exit() -> None:
    frame = prepared_day()
    frame.loc[3:4, ["open", "high", "low", "close"]] = [99.4, 99.6, 99.2, 99.4]
    result = simulate_confirmation_t(confirmation_sessions(frame), ConfirmationTConfig(), ZERO_FEES)

    assert result.trades.iloc[0]["exit_price"] == 99.4
    assert result.trades.iloc[0]["exit_reason"] == "SESSION_EXIT"
    assert result.trades.iloc[0]["exit_bar_end"] == pd.Timestamp("2025-01-06 14:50")


def test_actual_fees_and_cash_limit_participate_in_risk_sizing() -> None:
    config = replace(ConfirmationTConfig(), stop_atr=.5)

    assert risk_sized_shares(100, 99, config, ZERO_FEES, 50000) == 200
    assert risk_sized_shares(100, 99, config, ChinaAStockFees(), 50000) == 100
    assert risk_sized_shares(100, 99, config, ZERO_FEES, 9999) == 0


def test_gap_through_stop_is_reported_as_budget_exceeded() -> None:
    frame = prepared_day()
    frame.loc[3, ["open", "high", "low", "close"]] = [95., 96., 94., 95.]
    result = simulate_confirmation_t(confirmation_sessions(frame), ConfirmationTConfig(), ZERO_FEES)

    assert result.trades.iloc[0]["exit_price"] == 95.
    assert result.trades.iloc[0]["budget_exceeded"]


def test_future_prices_do_not_change_historical_entry_decisions() -> None:
    frame = prepared_day()
    before = simulate_confirmation_t(confirmation_sessions(frame), ConfirmationTConfig())
    frame.loc[4:, ["open", "high", "low", "close"]] = [100., 200., 50., 150.]
    after = simulate_confirmation_t(confirmation_sessions(frame), ConfirmationTConfig())

    pd.testing.assert_series_equal(before.trades.iloc[0], after.trades.iloc[0])


def test_missing_exit_session_is_rejected_and_zero_trades_cannot_qualify() -> None:
    frame = prepared_day().iloc[:4]
    with pytest.raises(ValueError, match="missing scheduled exit"):
        simulate_confirmation_t(confirmation_sessions(frame), ConfirmationTConfig())
    empty = {"cycles": 0, "win_rate": 0., "net_pnl": 0., "profit_factor": 0.}
    assert not selection_qualified(empty, empty, [])


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="risk_budget_yuan"):
        ConfirmationTConfig(risk_budget_yuan=np.nan)


def test_confirmation_can_follow_touch_after_price_leaves_extreme_zone() -> None:
    frame = prepared_day()
    frame.loc[1, ["open", "high", "low", "close"]] = [99.2, 99.7, 99.2, 99.6]
    frame.loc[2, ["open", "high", "low", "close"]] = [99.6, 99.9, 99.5, 99.8]
    result = simulate_confirmation_t(confirmation_sessions(frame), ConfirmationTConfig(), ZERO_FEES)

    assert result.trades.iloc[0]["signal_time"] == pd.Timestamp("2025-01-06 10:00")
    assert result.trades.iloc[0]["entry_price"] == 99.6


def test_reverse_t_is_covered_by_settled_shares_and_restores_inventory() -> None:
    frame = prepared_day()
    original = frame[["open", "high", "low", "close"]].copy()
    frame["open"] = 200 - original["open"]
    frame["high"] = 200 - original["low"]
    frame["low"] = 200 - original["high"]
    frame["close"] = 200 - original["close"]
    frame["vwap"] = 200 - frame["vwap"]
    frame["prior_range_position"] = 1 - frame["prior_range_position"]

    result = simulate_confirmation_t(confirmation_sessions(frame), replace(ConfirmationTConfig(), direction="reverse"), ZERO_FEES)
    trade = result.trades.iloc[0]

    assert trade["direction"] == "reverse"
    assert trade["target_price"] < trade["entry_price"] < trade["stop_price"]
    assert trade["pnl"] > 0
    assert trade["position_after"] == 10000
    assert trade["planned_loss"] <= 200


def test_known_next_open_entry_can_take_profit_later_in_its_entry_bar() -> None:
    frame = prepared_day()
    frame.loc[2, ["high", "close"]] = [101., 100.5]
    result = simulate_confirmation_t(confirmation_sessions(frame), ConfirmationTConfig(), ZERO_FEES)

    assert result.trades.iloc[0]["exit_reason"] == "TARGET"
    assert result.trades.iloc[0]["exit_bar_end"] == result.trades.iloc[0]["entry_bar_end"]


def test_intrabar_limit_touch_cannot_use_an_earlier_high_as_profit() -> None:
    frame = prepared_day()
    frame.loc[1, ["open", "high", "low", "close"]] = [100., 102., 99., 99.1]
    frame.loc[2:4, ["open", "high", "low", "close"]] = [99.3, 99.5, 99.2, 99.4]
    result = simulate_confirmation_t(confirmation_sessions(frame), replace(ConfirmationTConfig(), entry_mode="limit"), ZERO_FEES)

    assert result.trades.iloc[0]["exit_reason"] == "SESSION_EXIT"
