# ruff: noqa: E402
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from quant.research.weekly_trend_quality import (
    QUALITY_FEATURES,
    add_outcomes,
    cross_sectional_strength,
    quality_features,
    shortlist,
    trend_rules,
)


def weekly(close):
    close = np.asarray(close)
    dates = pd.date_range("2021-01-04", periods=len(close), freq="W-MON")
    return pd.DataFrame(
        {
            "available_at": dates,
            "date": dates - pd.Timedelta(days=3),
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000.0,
        }
    )


def features(close):
    frame = quality_features(weekly(close))
    frame["symbol"] = "X"
    return cross_sectional_strength(frame)


def test_future_append_does_not_change_features_or_rules():
    close = np.exp(np.arange(150) * 0.005) * 100
    before = features(close[:120])
    after = features(np.r_[close[:120], close[120:] * 4])
    cols = [c for c in QUALITY_FEATURES if c in before] + ["trend_stage", "legacy_macd_boll"]
    pd.testing.assert_frame_equal(before[cols], after.iloc[:120][cols].reset_index(drop=True))
    for key, value in trend_rules(before).items():
        pd.testing.assert_series_equal(
            value, trend_rules(after)[key].iloc[:120].reset_index(drop=True)
        )


def test_rebound_is_not_long_term_uptrend_and_smooth_trend_is():
    rising = features(np.exp(np.arange(150) * 0.005) * 100)
    assert trend_rules(rising)["long52_r2_60"].iloc[-1]
    rebound = features(np.r_[np.linspace(200, 60, 140), np.linspace(62, 80, 10)])
    assert not trend_rules(rebound)["long52"].iloc[-1]
    assert rebound.trend_stage.iloc[-1] in ["rebound", "transition"]


def test_sawtooth_has_lower_trend_quality_and_overheated_is_excluded():
    t = np.arange(150)
    smooth = features(100 * np.exp(t * 0.005))
    noisy = features(100 * np.exp(t * 0.005 + np.where(t % 2, 0.06, -0.06)))
    assert smooth.w_lr13_r2.iloc[-1] > noisy.w_lr13_r2.iloc[-1]
    assert smooth.w_efficiency13.iloc[-1] > noisy.w_efficiency13.iloc[-1]
    hot = features(np.r_[100 * np.exp(np.arange(140) * 0.005), np.linspace(220, 450, 10)])
    assert trend_rules(hot)["long52"].iloc[-1]
    assert not trend_rules(hot)["long52_healthy"].iloc[-1]


def test_live_guard_can_veto_breakdown_in_current_week():
    frame = features(100 * np.exp(np.arange(150) * 0.005))
    frame["current_close"] = frame.w_ema26 * 0.8
    assert trend_rules(frame)["legacy_macd_boll"].iloc[-1]
    assert not trend_rules(frame)["macd_boll_live_guard"].iloc[-1]


def test_outcomes_require_exact_future_market_weeks_and_do_not_change_gates():
    frame = features(np.exp(np.arange(100) * 0.005) * 100)
    labeled = add_outcomes(frame, pd.DatetimeIndex(frame.available_at))
    assert labeled.future_return4.iloc[80] == pytest.approx(np.exp(0.02) - 1)
    assert labeled.label_end4.iloc[80] == frame.available_at.iloc[84]
    assert labeled.future_return4.tail(4).isna().all()
    for key, value in trend_rules(frame).items():
        pd.testing.assert_series_equal(value, trend_rules(labeled)[key])
    missing = frame.drop(index=82)
    labels = add_outcomes(missing, pd.DatetimeIndex(frame.available_at))
    assert pd.isna(
        labels.loc[labels.available_at.eq(frame.available_at.iloc[80]), "future_return4"].iloc[0]
    )


def test_shortlist_cannot_be_influenced_by_test_returns():
    def record(name, mean, test):
        row = {"rule": name, "test": {"average_net_return": test}}
        for p in ["development_2024", "development_2025H1", "selection"]:
            row[p] = {"trades": 200, "average_net_return": mean, "win_rate": 0.55}
        return row

    first = shortlist([record("A", 0.01, -100), record("B", -0.01, 100)], count=1)
    second = shortlist([record("A", 0.01, 100), record("B", -0.01, -100)], count=1)
    assert first == second and first[0] == "A"


def test_lifecycle_age_and_runup_are_causal_and_reset_after_breakdown():
    from quant.research.weekly_trend_quality import lifecycle_features

    frame = features(
        np.r_[
            100 * np.exp(np.arange(130) * 0.005),
            np.linspace(180, 100, 10),
            np.linspace(101, 140, 10),
        ]
    )
    before = lifecycle_features(frame.iloc[:125])
    after = lifecycle_features(frame)
    columns = ["trend_age_weeks", "episode_runup", "episode_drawdown"]
    pd.testing.assert_frame_equal(before[columns], after.iloc[:125][columns])
    assert after.trend_age_weeks.iloc[129] > 20
    assert after.trend_age_weeks.iloc[139] == 0
    after["current_close"] = after.w_close
    assert not trend_rules(after)["age_2_13"].iloc[129]
    assert not trend_rules(after)["age_2_13"].iloc[139]
