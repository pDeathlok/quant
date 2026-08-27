from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.monthly_low_zone_patterns import (
    ChartPatternConfig,
    causal_pivot_lows,
    generate_chart_pattern_signals,
)


def _prices(length: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=length)
    close = np.full(length, 10.0)
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "date": dates,
            "adjusted_close": close,
            "adjusted_low": close - 0.2,
            "prior_amount_median_20d": 50_000.0,
            "sessions_since_new_low": 30.0,
            "return_20d": 0.05,
            "base_position": 0.60,
        }
    )


def _anchor(prices: pd.DataFrame, position: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_id": [17],
            "ts_code": ["000001.SZ"],
            "signal_date": [prices.loc[position, "date"]],
            "month_period": [prices.loc[position, "date"].to_period("M")],
            "adjusted_close": [8.5],
            "prior_peak": [20.0],
            "drawdown_from_prior_peak": [-0.575],
            "monthly_j": [-25.0],
            "weekly_j": [-15.0],
            "monthly_low9": [True],
            "monthly_low9_count": [9],
            "median_daily_amount": [50_000.0],
        }
    )


def test_causal_pivot_low_has_right_side_recognition_delay() -> None:
    prices = _prices(12)
    prices.loc[5, "adjusted_low"] = 8.0
    pivots = causal_pivot_lows(prices, radius=3)
    match = pivots.loc[pivots["position"].eq(5)].iloc[0]
    assert match["pivot_date"] == prices.loc[5, "date"]
    assert match["recognition_date"] == prices.loc[8, "date"]


def test_double_bottom_requires_causal_neckline_breakout() -> None:
    prices = _prices()
    prices.loc[10, "adjusted_low"] = 8.0
    prices.loc[35, "adjusted_low"] = 8.2
    prices.loc[45, "adjusted_close"] = 10.2
    prices.loc[45, "adjusted_low"] = 9.9
    anchors = _anchor(prices, 40)

    signals, diagnostics = generate_chart_pattern_signals(
        prices,
        anchors,
        pd.DatetimeIndex(prices["date"]),
        ChartPatternConfig(),
    )

    signal = signals.loc[signals["rule"].eq("double_bottom_breakout")].iloc[0]
    assert signal["signal_date"] == prices.loc[45, "date"]
    assert signal["pattern_start_date"] == prices.loc[10, "date"]
    assert signal["pattern_end_date"] == prices.loc[35, "date"]
    assert signal["pattern_neckline"] == 10.0
    assert signal["anchor_id"] == 17
    status = diagnostics.loc[
        diagnostics["rule"].eq("double_bottom_breakout"), "confirmation_status"
    ].iloc[0]
    assert status == "confirmed"


def test_double_bottom_does_not_use_breakout_before_second_pivot_is_visible() -> None:
    prices = _prices()
    prices.loc[10, "adjusted_low"] = 8.0
    prices.loc[35, "adjusted_low"] = 8.2
    prices.loc[36, "adjusted_close"] = 10.2
    prices.loc[36, "adjusted_low"] = 9.9
    anchors = _anchor(prices, 30)

    signals, diagnostics = generate_chart_pattern_signals(
        prices,
        anchors,
        pd.DatetimeIndex(prices["date"]),
        ChartPatternConfig(),
    )

    assert signals.loc[signals["rule"].eq("double_bottom_breakout")].empty
    status = diagnostics.loc[
        diagnostics["rule"].eq("double_bottom_breakout"), "confirmation_status"
    ].iloc[0]
    assert status == "expired"


def test_inverse_head_shoulders_requires_deep_head_and_matching_shoulders() -> None:
    prices = _prices()
    prices.loc[8, "adjusted_low"] = 9.0
    prices.loc[25, "adjusted_low"] = 7.8
    prices.loc[42, "adjusted_low"] = 9.2
    prices.loc[50, "adjusted_close"] = 10.2
    prices.loc[50, "adjusted_low"] = 9.9
    anchors = _anchor(prices, 35)

    signals, _ = generate_chart_pattern_signals(
        prices,
        anchors,
        pd.DatetimeIndex(prices["date"]),
        ChartPatternConfig(),
    )

    signal = signals.loc[
        signals["rule"].eq("inverse_head_shoulders_breakout")
    ].iloc[0]
    assert signal["signal_date"] == prices.loc[50, "date"]
    assert signal["pattern_start_date"] == prices.loc[8, "date"]
    assert signal["pattern_middle_date"] == prices.loc[25, "date"]
    assert signal["pattern_end_date"] == prices.loc[42, "date"]
    assert signal["pattern_head_depth"] >= 0.08


def test_inverse_head_shoulders_rejects_shallow_head() -> None:
    prices = _prices()
    prices.loc[8, "adjusted_low"] = 9.0
    prices.loc[25, "adjusted_low"] = 8.7
    prices.loc[42, "adjusted_low"] = 9.2
    prices.loc[50, "adjusted_close"] = 10.2
    prices.loc[50, "adjusted_low"] = 9.9
    anchors = _anchor(prices, 35)

    signals, _ = generate_chart_pattern_signals(
        prices,
        anchors,
        pd.DatetimeIndex(prices["date"]),
        ChartPatternConfig(),
    )

    assert signals.loc[
        signals["rule"].eq("inverse_head_shoulders_breakout")
    ].empty


def test_pattern_confirmation_expires_after_frozen_wait_window() -> None:
    prices = _prices(100)
    prices.loc[10, "adjusted_low"] = 8.0
    prices.loc[35, "adjusted_low"] = 8.2
    prices.loc[61, "adjusted_close"] = 10.2
    prices.loc[61, "adjusted_low"] = 9.9
    anchors = _anchor(prices, 40)
    config = ChartPatternConfig(maximum_wait_sessions=20)

    signals, _ = generate_chart_pattern_signals(
        prices,
        anchors,
        pd.DatetimeIndex(prices["date"]),
        config,
    )

    assert signals.empty


def test_pattern_breakout_must_also_reclaim_causal_base_midpoint() -> None:
    prices = _prices()
    prices.loc[10, "adjusted_low"] = 8.0
    prices.loc[35, "adjusted_low"] = 8.2
    prices.loc[45, "adjusted_close"] = 10.2
    prices.loc[45, "adjusted_low"] = 9.9
    prices.loc[45, "base_position"] = 0.49
    anchors = _anchor(prices, 40)

    signals, _ = generate_chart_pattern_signals(
        prices,
        anchors,
        pd.DatetimeIndex(prices["date"]),
        ChartPatternConfig(),
    )

    assert signals.loc[signals["rule"].eq("double_bottom_breakout")].empty
