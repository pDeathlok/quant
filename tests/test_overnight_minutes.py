import numpy as np
import pandas as pd
import pytest

from quant.research.overnight_minutes import (
    close_check_exit,
    daily_exit_labels,
    daily_limit_exit,
    lagged_factor_row,
    minute_exit,
    preclose_snapshot,
    session_grid,
)


def test_preset_limit_requires_penetration_and_caps_opening_advantage():
    missed = daily_limit_exit(100., 100., 100.5, 98., .005)
    assert missed["net"] == pytest.approx(-.022)
    reached = daily_limit_exit(100., 100., 110., 109., .005)
    assert reached["net"] == pytest.approx(.003)
    assert reached["limit_price"] == 100.5
    rounded = daily_limit_exit(10.01, 10.01, 11., 10., .005)
    assert rounded["limit_price"] == 10.07
    assert daily_limit_exit(100., 50., 51., 50., .005)["status"] == "corporate_action_unresolved"


def test_daily_exit_labels_keep_gap_losses_and_independent_missing_prices():
    result = daily_exit_labels(100., 100., 90., 101.)
    assert result["target_net_open"] == pytest.approx(-.102)
    assert result["target_net_close"] == pytest.approx(.008)
    result = daily_exit_labels(100., 100., np.nan, 101.)
    assert np.isnan(result["target_net_open"])
    assert result["target_net_close"] == pytest.approx(.008)
    result = daily_exit_labels(100., 50., 51., 52.)
    assert np.isnan(result["target_net_open"])
    assert result["label_quality_close"] == "corporate_action_unresolved"


def test_lagged_factors_ignore_same_day_and_future_changes():
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
                          "factor": [1., 2., 3.]})
    date = pd.Timestamp("2025-01-03")
    assert lagged_factor_row(frame, date)["factor"] == 1.
    frame.loc[1:, "factor"] = 1e9
    assert lagged_factor_row(frame, date)["factor"] == 1.
    assert lagged_factor_row(frame, pd.Timestamp("2025-01-02")) is None
    with pytest.raises(ValueError, match="unique"):
        lagged_factor_row(pd.concat([frame, frame]), date)


def bars():
    times = session_grid(pd.Timestamp("2025-01-02"))
    return pd.DataFrame({"trade_time": times, "open": 100., "high": 100.2,
                         "low": 99.8, "close": 100., "vol": 1000., "amount": 100000.})


def test_no_stop_policy_can_recover_but_does_not_cap_large_losses():
    frame = bars().iloc[1:].reset_index(drop=True)
    frame.loc[1, "close"] = 98.
    frame.loc[2, "open"] = 98.
    frame.loc[5, "close"] = 101.
    frame.loc[6, "open"] = 100.8
    stopped = close_check_exit(frame, 100., .005, .01)
    unstopped = close_check_exit(frame, 100., .005, .01, use_stop=False)
    assert stopped["net"] == pytest.approx(-.022)
    assert unstopped["net"] == pytest.approx(.006)
    frame["close"] = 90.
    frame.loc[len(frame) - 1, "close"] = 80.
    assert close_check_exit(frame, 100., .005, .01, use_stop=False)["net"] == pytest.approx(-.202)


def test_future_bar_changes_cannot_change_preclose_features():
    frame = bars()
    cutoff = pd.Timestamp("2025-01-02 14:55")
    before = preclose_snapshot(frame, 99., cutoff)
    frame.loc[frame.trade_time.gt("2025-01-02 14:50"), ["high", "close", "vol", "amount"]] = 1e9
    assert before == preclose_snapshot(frame, 99., cutoff)
    assert before["observed_bars"] == 47


def test_missing_preclose_bar_rejected_but_missing_future_bar_allowed():
    frame = bars()
    cutoff = pd.Timestamp("2025-01-02 14:55")
    assert preclose_snapshot(frame.iloc[:-1], 99., cutoff) is not None
    assert preclose_snapshot(frame.drop(index=10), 99., cutoff) is None


def test_minute_sequence_resolves_daily_double_touch():
    frame = bars()
    frame.loc[1, "high"] = 102.
    frame.loc[5, "low"] = 97.
    result = minute_exit(frame, 100., .01, .01)
    assert result["net_lower"] == pytest.approx(.008)
    assert result["ambiguous"] is False


def test_same_bar_double_touch_remains_uncertain():
    frame = bars()
    frame.loc[1, ["high", "low"]] = [102., 97.]
    result = minute_exit(frame, 100., .01, .01)
    assert result["net_lower"] == pytest.approx(-.012)
    assert result["net_upper"] == pytest.approx(.008)
    assert result["ambiguous"] is True


def test_opening_gap_not_capped_and_incomplete_session_not_filled():
    frame = bars()
    frame.loc[0, ["open", "high", "low", "close"]] = [95., 95.2, 94.8, 95.]
    assert minute_exit(frame, 100., .01, .01)["net_lower"] == pytest.approx(-.052)
    assert minute_exit(frame.iloc[:-1], 100., .01, .01)["status"] == "incomplete_session"
    frame.loc[10, "close"] = np.nan
    assert preclose_snapshot(frame, 100., pd.Timestamp("2025-01-02 14:55")) is None


def test_baostock_grid_is_explicit_not_silently_filled():
    frame = bars().iloc[1:].copy()
    cutoff = pd.Timestamp("2025-01-02 14:55")
    assert len(session_grid(cutoff, include_auction=False)) == 48
    assert preclose_snapshot(frame, 100., cutoff) is None
    assert preclose_snapshot(frame, 100., cutoff, include_auction=False)["observed_bars"] == 46
    assert minute_exit(frame, 100., .01, .01, include_auction=False)["status"] == "close_price_proxy"


def test_close_check_does_not_sell_on_same_bar_or_intrabar_high():
    frame = bars().iloc[1:].reset_index(drop=True)
    frame.loc[0, "high"] = 103.
    frame.loc[1, ["close", "high"]] = [101., 102.]
    frame.loc[2, ["open", "high"]] = [102., 102.]
    result = close_check_exit(frame, 100., .01, .01)
    assert result["trigger_time"] == str(frame.trade_time.iloc[1])
    assert result["execution_bar"] == str(frame.trade_time.iloc[2])
    assert result["net"] == pytest.approx(.018)
    frame[["high", "low"]] = np.nan
    assert close_check_exit(frame, 100., .01, .01) == result


def test_close_check_preserves_gap_loss_and_unexecuted_state():
    frame = bars().iloc[1:].reset_index(drop=True)
    frame.loc[1, ["close", "low"]] = [99., 98.]
    frame.loc[2, ["open", "low"]] = [95., 94.]
    assert close_check_exit(frame, 100., .01, .01)["net"] == pytest.approx(-.052)
    frame.loc[2:, "vol"] = 0.
    assert close_check_exit(frame, 100., .01, .01)["status"] == "no_observed_execution_volume"


def test_close_check_uses_precommitted_close_when_no_signal():
    frame = bars().iloc[1:].copy()
    result = close_check_exit(frame, 100., .01, .01)
    assert result["status"] == "precommitted_close_price_proxy"
    assert result["net"] == pytest.approx(-.002)
