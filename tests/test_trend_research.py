# ruff: noqa: E402
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from quant.research.trend_research import (
    DAILY_FEATURES,
    WEEKLY_FEATURES,
    WEEKLY_GATES,
    direction_features,
    forward_event_labels,
    recency_weights,
    recent_window,
)


def raw_bars(n=900):
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 10 * np.exp(np.arange(n) * 0.0005 + np.sin(np.arange(n) / 10) * 0.02)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "pre_close": np.r_[10.0, close[:-1]],
            "vol": 1000 + 100 * np.cos(np.arange(n)),
        }
    )


def test_weekly_and_daily_features_are_prefix_invariant():
    raw = raw_bars()
    origin = raw.date.iloc[750]
    before = direction_features(raw[raw.date <= origin])
    changed = raw.copy()
    future = changed.date > origin
    changed.loc[future, ["open", "high", "low", "close", "pre_close", "vol"]] *= 10
    after = direction_features(changed)
    columns = ["date", *DAILY_FEATURES, *WEEKLY_FEATURES, *WEEKLY_GATES]
    pd.testing.assert_frame_equal(
        before[columns], after[after.date <= origin][columns].reset_index(drop=True)
    )
    active = before[before.weekly_source_date.notna()]
    assert (
        active.weekly_source_date.dt.to_period("W-SUN") < active.date.dt.to_period("W-SUN")
    ).all()


def test_missing_reference_or_price_does_not_become_an_executable_bar():
    raw = raw_bars(100)
    raw.loc[50, "close"] = 0
    frame = direction_features(raw)
    assert raw.date.iloc[50] not in set(frame.date)


def test_exact_exchange_horizon_and_suspension_rejection():
    raw = raw_bars(50)
    features = direction_features(raw)
    calendar = pd.DatetimeIndex(raw.date)
    events = pd.DataFrame({"date": [calendar[10], calendar[11]], "symbol": ["X", "X"]})
    labels = forward_event_labels(features, events, calendar, 5)
    assert labels.entry_date.tolist() == list(calendar[[11, 12]])
    assert labels.exit_date.tolist() == list(calendar[[16, 17]])
    expected = features.close.iloc[16] / features.close.iloc[11] - 1 - 0.002
    assert labels.net_return.iloc[0] == pytest.approx(expected)
    features.loc[features.date.eq(calendar[16]), "volume"] = 0
    assert forward_event_labels(features, events, calendar, 5).empty
    assert forward_event_labels(
        direction_features(raw), events, calendar, 20
    ).exit_date.tolist() == list(calendar[[31, 32]])


def test_weights_and_window_only_use_mature_past():
    dates = pd.Series(pd.to_datetime(["2025-09-01", "2025-12-01", "2025-12-31"]))
    weights = recency_weights(dates, pd.Timestamp("2026-01-01"), 90)
    assert np.diff(weights).min() > 0 and weights.mean() == pytest.approx(1)
    balanced = recency_weights(
        pd.Series([dates[0], dates[0], dates[1]]),
        pd.Timestamp("2026-01-01"),
        None,
        equal_dates=True,
    )
    assert balanced[0] + balanced[1] == pytest.approx(balanced[2])
    with pytest.raises(ValueError):
        recency_weights(dates, dates.iloc[-1], 90)
    frame = pd.DataFrame(
        {
            "date": dates,
            "exit_date": dates + pd.Timedelta(days=8),
            "net_return": [0.01, 0.02, 100.0],
        }
    )
    window = recent_window(frame, pd.Timestamp("2026-01-01"), 6)
    assert window.index.tolist() == [0, 1]
    changed = frame.copy()
    changed.loc[2, "net_return"] = -100.0
    pd.testing.assert_frame_equal(window, recent_window(changed, pd.Timestamp("2026-01-01"), 6))


def test_weekly_market_breadth_excludes_unfinished_week_and_future_stock():
    from quant.research.trend_research import weekly_market_breadth

    raw = raw_bars(150)
    prices = pd.concat(
        [raw[["date", "close"]].assign(symbol="A"), raw[["date", "close"]].assign(symbol="B")],
        ignore_index=True,
    )
    origin = raw.date.iloc[120]
    first = weekly_market_breadth(prices[prices.date <= origin])
    future = pd.DataFrame({"symbol": "NEW", "date": raw.date[raw.date > origin], "close": 1.0})
    later = weekly_market_breadth(pd.concat([prices, future], ignore_index=True))
    pd.testing.assert_frame_equal(first, later[later.date <= origin].reset_index(drop=True))
    known = first[first.market_source_date.notna()]
    assert (known.market_source_date.dt.to_period("W-SUN") < known.date.dt.to_period("W-SUN")).all()
