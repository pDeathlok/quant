from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.features.candlestick_context import (
    CANDLE_CONTEXT_FEATURE_COLUMNS,
    compute_candlestick_context_features,
)
from quant.research.right_side_unified_features import (
    compute_right_side_rule_features,
)


def _daily(rows: int = 25) -> pd.DataFrame:
    dates = pd.bdate_range("2026-07-27", periods=rows)
    close = np.full(rows, 6.77)
    frame = pd.DataFrame(
        {
            "date": dates,
            "trade_date": dates.strftime("%Y%m%d"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "pre_close": close,
            "volume": np.full(rows, 100.0),
        }
    )
    return frame


def test_ocean_king_style_candle_context_uses_explicit_denominators() -> None:
    daily = _daily(21)
    daily.loc[20, ["open", "high", "low", "close", "pre_close", "volume"]] = [
        6.80,
        7.33,
        6.77,
        6.93,
        6.77,
        123.0,
    ]

    result = compute_candlestick_context_features(daily)
    latest = result.iloc[-1]

    assert tuple(result.columns) == CANDLE_CONTEXT_FEATURE_COLUMNS
    assert latest["rs_upper_shadow_pct"] == pytest.approx(0.40 / 6.77 * 100.0)
    assert latest["rs_upper_shadow_range_share"] == pytest.approx(0.40 / 0.56)
    assert latest["rs_upper_shadow_body_ratio"] == pytest.approx(0.40 / 0.13)
    assert latest["rs_volume_ratio_prev20"] == pytest.approx(1.23)


def test_candle_context_requires_full_prior_volume_window_and_handles_doji() -> None:
    daily = _daily(21)
    daily.loc[20, ["high", "close", "open"]] = [7.0, 6.8, 6.8]

    result = compute_candlestick_context_features(daily)

    assert result["rs_volume_ratio_prev20"].iloc[:20].isna().all()
    assert pd.isna(result["rs_upper_shadow_body_ratio"].iloc[-1])


def test_candle_context_features_are_prefix_causal() -> None:
    daily = _daily(25)
    daily["open"] += np.arange(len(daily)) * 0.01
    daily["high"] = daily[["open", "close"]].max(axis=1) + 0.20
    daily["low"] = daily[["open", "close"]].min(axis=1) - 0.10
    daily["volume"] = np.arange(len(daily), dtype=float) + 100.0

    full = compute_candlestick_context_features(daily)
    prefix = compute_candlestick_context_features(daily.iloc[:22])

    pd.testing.assert_frame_equal(full.loc[prefix.index], prefix)


def test_existing_upper_shadow_feature_keeps_right_rule_parity() -> None:
    daily = _daily(25)
    daily["open"] += np.arange(len(daily)) * 0.01
    daily["high"] = daily[["open", "close"]].max(axis=1) + 0.20
    daily["low"] = daily[["open", "close"]].min(axis=1) - 0.10

    context = compute_candlestick_context_features(daily)
    rules = compute_right_side_rule_features(daily)

    pd.testing.assert_series_equal(
        context["rs_upper_shadow_pct"].iloc[1:],
        rules["rs_upper_shadow_pct"].iloc[1:],
    )
