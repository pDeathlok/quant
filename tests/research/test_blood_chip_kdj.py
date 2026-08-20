from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.blood_chip_kdj import (
    attach_blood_chip_kdj_path,
    attach_completed_kdj,
    apply_kdj_overlay,
)


def _features(periods: int = 520) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=periods)
    base = 20.0 + np.sin(np.arange(periods) / 15.0) * 2.0 - np.arange(periods) * 0.01
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "date": dates,
            "adjusted_high": base * 1.01,
            "adjusted_low": base * 0.99,
            "adjusted_close": base,
        }
    )


def test_future_rows_do_not_change_attached_completed_kdj() -> None:
    features = _features()
    signal_date = features.iloc[-25]["date"]
    signals = pd.DataFrame({"ts_code": ["000001.SZ"], "signal_date": [signal_date]})

    full = attach_completed_kdj(features, signals)
    truncated = attach_completed_kdj(
        features.loc[features["date"].le(signal_date)], signals
    )

    for column in ("kdj_daily_j", "kdj_weekly_j", "kdj_monthly_j"):
        assert full.iloc[0][column] == truncated.iloc[0][column]


def test_incomplete_month_is_not_exposed_before_month_end() -> None:
    features = _features()
    targets = features.loc[features["date"].dt.day.between(10, 20)].tail(2)["date"]
    signals = pd.DataFrame(
        {"ts_code": "000001.SZ", "signal_date": targets.to_list()}
    )

    original = attach_completed_kdj(features, signals)
    modified = features.copy()
    month_start = targets.iloc[0].to_period("M").start_time
    in_current_month = modified["date"].between(month_start, targets.iloc[-1])
    modified.loc[in_current_month, "adjusted_low"] *= 0.5
    changed = attach_completed_kdj(modified, signals)

    pd.testing.assert_series_equal(
        original["kdj_monthly_j"], changed["kdj_monthly_j"], check_names=False
    )


def test_soft_overlay_preserves_candidates_and_hard_overlay_filters() -> None:
    signals = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
            "signal_date": pd.Timestamp("2026-08-13"),
            "volatility_60d": [0.30, 0.20, 0.40],
            "kdj_daily_j": [-5.0, 5.0, -3.0],
            "kdj_weekly_j": [-4.0, 4.0, -2.0],
            "kdj_monthly_j": [-1.0, 3.0, 2.0],
        }
    )

    soft = apply_kdj_overlay(signals, "kdj_soft_priority")
    hard = apply_kdj_overlay(signals, "triple_only")

    assert len(soft) == 3
    assert soft["ts_code"].tolist() == ["000002.SZ", "000001.SZ", "600000.SH"]
    assert hard["ts_code"].tolist() == ["000001.SZ"]


def test_path_attachment_keeps_confirmation_date_and_labels_shock_separately() -> None:
    features = _features()
    signals = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "shock_date": [features.iloc[-40]["date"]],
            "signal_date": [features.iloc[-35]["date"]],
        }
    )

    result = attach_blood_chip_kdj_path(features, signals)

    assert result.iloc[0]["signal_date"] == signals.iloc[0]["signal_date"]
    assert pd.notna(result.iloc[0]["shock_kdj_monthly_j"])
    assert pd.notna(result.iloc[0]["confirmation_kdj_daily_j"])
