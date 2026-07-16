from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from quant.research.similar_patterns_validation import (
    apply_expanding_calibration,
    apply_global_expanding_calibration,
    build_market_regime,
    filter_cases_mature_at_signal,
    summarize_walk_forward_records,
)


def test_build_market_regime_uses_only_trailing_prices() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    close = np.r_[np.linspace(100, 120, 60), np.linspace(119, 80, 40)]
    frame = pd.DataFrame({"trade_date": dates.strftime("%Y%m%d"), "close": close})

    regime = build_market_regime(frame)

    assert regime.iloc[40]["market_regime"] == "risk_on"
    assert regime.iloc[-1]["market_regime"] == "risk_off"
    assert list(regime.columns) == ["date", "market_regime", "market_ret_20d", "market_vol_20d"]


def test_filter_cases_mature_at_signal_uses_horizon_specific_cutoff() -> None:
    cases = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
            "fwd_1d": [0.01, 0.02, 0.03],
            "fwd_20d": [0.04, 0.05, 0.06],
        }
    )

    next_day = filter_cases_mature_at_signal(cases, pd.Timestamp("2024-02-02"), "next_1d")
    next_month = filter_cases_mature_at_signal(cases, pd.Timestamp("2024-02-01"), "next_1m")

    assert list(next_day["date"]) == list(pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]))
    assert list(next_month["date"]) == [pd.Timestamp("2024-01-01")]


def test_expanding_calibration_never_uses_unmatured_outcomes() -> None:
    records = pd.DataFrame(
        {
            "symbol": ["A"] * 5,
            "horizon": ["next_1d"] * 5,
            "signal_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]),
            "outcome_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]),
            "raw_up_probability": [40.0, 45.0, 55.0, 60.0, 65.0],
            "actual_return": [-0.01, -0.02, 0.01, 0.02, 0.03],
        }
    )

    calibrated, calibrations = apply_expanding_calibration(records, min_samples=2)

    assert calibrated.loc[0, "calibration_samples"] == 0
    assert calibrated.loc[1, "calibration_samples"] == 1
    assert calibrated.loc[3, "calibration_samples"] == 3
    assert calibrated["calibrated_up_probability"].between(0, 100).all()
    assert "A" in calibrations


def test_global_expanding_calibration_pools_symbols_without_future_outcomes() -> None:
    records = pd.DataFrame(
        {
            "symbol": ["A", "B", "A", "B"],
            "horizon": ["next_1d"] * 4,
            "signal_date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-03", "2025-01-03"]),
            "outcome_date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-06", "2025-01-06"]),
            "raw_up_probability": [40.0, 60.0, 45.0, 55.0],
            "actual_return": [-0.01, 0.01, -0.02, 0.02],
        }
    )

    calibrated, calibrations = apply_global_expanding_calibration(records, min_samples=2)

    assert calibrated.loc[0, "calibration_samples"] == 0
    assert calibrated.loc[1, "calibration_samples"] == 0
    assert calibrated.loc[2, "calibration_samples"] == 2
    assert calibrated.loc[3, "calibration_samples"] == 2
    assert set(calibrations) == {"next_1d"}


def test_summarize_walk_forward_records_reports_coverage_accuracy_and_costs() -> None:
    records = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "horizon": ["next_1d"] * 4,
            "signal_date": pd.bdate_range("2025-01-02", periods=4),
            "signal": ["bullish", "observe", "bearish", "bullish"],
            "actual_return": [0.02, -0.01, -0.03, -0.02],
            "calibrated_up_probability": [60.0, 51.0, 40.0, 58.0],
        }
    )

    summary = summarize_walk_forward_records(records, transaction_cost=0.001)

    assert summary.iloc[0]["signals"] == 4
    assert summary.iloc[0]["actionable_signals"] == 3
    assert summary.iloc[0]["coverage"] == 75.0
    assert summary.iloc[0]["direction_accuracy"] == round(2 / 3 * 100, 2)
    assert summary.iloc[0]["cost_adjusted_return"] < summary.iloc[0]["gross_directional_return"]
