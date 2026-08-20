from __future__ import annotations

import pandas as pd
import pytest

from quant.research.blood_chip import BloodChipBacktestConfig
from quant.research.blood_chip_scale_in import (
    DEFAULT_SCALE_IN_POLICIES,
    run_blood_chip_scale_in_backtest,
)


def _daily(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=["trade_date", "open", "high", "low", "close", "residual_return_3d"],
    )
    frame["ts_code"] = "000001.SZ"
    frame["pre_close"] = frame["close"].shift(1).fillna(frame["close"])
    frame["pct_chg"] = (frame["close"] / frame["pre_close"] - 1.0) * 100.0
    frame["vol"] = 1_000_000.0
    frame["amount"] = 100_000.0
    frame["adjustment_factor"] = 1.0
    for column in ("open", "high", "low", "close"):
        frame[f"adjusted_{column}"] = frame[column]
    frame["date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
    return frame


def _signal() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "signal_date": pd.Timestamp("2024-01-02"),
                "entry_date": pd.Timestamp("2024-01-03"),
                "shock_date": pd.Timestamp("2023-12-29"),
                "shock_event_id": 1,
                "signal_close": 10.0,
                "signal_score": -0.30,
                "rebound_from_event_low": 10.0 / 9.0 - 1.0,
                "volatility_60d": 0.30,
            }
        ]
    )


def test_increasing_confirmed_builds_20_30_50_on_successive_next_opens() -> None:
    daily = _daily(
        [
            ("20240102", 10.0, 10.1, 9.9, 10.0, 0.00),
            ("20240103", 10.0, 10.1, 9.5, 9.6, -0.005),
            ("20240104", 9.6, 10.2, 9.5, 10.1, 0.020),
            ("20240105", 10.1, 10.5, 10.0, 10.4, 0.030),
            ("20240108", 10.4, 10.6, 10.3, 10.5, 0.020),
        ]
    )

    result = run_blood_chip_scale_in_backtest(
        daily,
        _signal(),
        BloodChipBacktestConfig(maximum_positions=1, maximum_holding_days=20),
        DEFAULT_SCALE_IN_POLICIES["increasing_confirmed"],
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["tranches_filled"] == 3
    assert trade["tranche_dates"] == "2024-01-03|2024-01-04|2024-01-05"
    assert trade["planned_fractions"] == "0.2000|0.3000|0.5000"
    assert trade["deployed_fraction"] == pytest.approx(1.0, abs=0.02)


def test_stop_is_processed_before_a_pending_addition() -> None:
    daily = _daily(
        [
            ("20240102", 10.0, 10.1, 9.9, 10.0, 0.00),
            ("20240103", 10.0, 10.1, 9.5, 9.6, -0.005),
            ("20240104", 8.8, 8.9, 8.7, 8.8, -0.040),
            ("20240105", 8.8, 9.0, 8.7, 8.9, 0.000),
        ]
    )

    result = run_blood_chip_scale_in_backtest(
        daily,
        _signal(),
        BloodChipBacktestConfig(maximum_positions=1, maximum_holding_days=20),
        DEFAULT_SCALE_IN_POLICIES["increasing_confirmed"],
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_date"] == pd.Timestamp("2024-01-04")
    assert trade["tranches_filled"] == 1


def test_price_only_additions_are_also_causal_and_capped_at_one_slot() -> None:
    daily = _daily(
        [
            ("20240102", 10.0, 10.1, 9.9, 10.0, 0.00),
            ("20240103", 10.0, 10.1, 9.4, 9.5, -0.020),
            ("20240104", 9.5, 9.6, 9.05, 9.1, -0.030),
            ("20240105", 9.1, 9.4, 9.05, 9.3, 0.010),
            ("20240108", 9.3, 9.5, 9.2, 9.4, 0.020),
        ]
    )

    result = run_blood_chip_scale_in_backtest(
        daily,
        _signal(),
        BloodChipBacktestConfig(maximum_positions=1, maximum_holding_days=20),
        DEFAULT_SCALE_IN_POLICIES["increasing_price_only"],
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["tranches_filled"] == 3
    assert trade["tranche_dates"] == "2024-01-03|2024-01-04|2024-01-05"
    assert trade["deployed_fraction"] <= 1.0


def test_open_plan_limit_can_be_separated_from_full_position_budget() -> None:
    second = _signal().copy()
    second["ts_code"] = "000002.SZ"
    second["shock_event_id"] = 2
    daily = pd.concat(
        [
            _daily(
                [
                    ("20240102", 10.0, 10.1, 9.9, 10.0, 0.00),
                    ("20240103", 10.0, 10.1, 9.8, 10.0, 0.00),
                    ("20240104", 10.0, 10.1, 9.9, 10.0, 0.00),
                ]
            ),
            _daily(
                [
                    ("20240102", 10.0, 10.1, 9.9, 10.0, 0.00),
                    ("20240103", 10.0, 10.1, 9.8, 10.0, 0.00),
                    ("20240104", 10.0, 10.1, 9.9, 10.0, 0.00),
                ]
            ).assign(ts_code="000002.SZ"),
        ],
        ignore_index=True,
    )

    result = run_blood_chip_scale_in_backtest(
        daily,
        pd.concat([_signal(), second], ignore_index=True),
        BloodChipBacktestConfig(maximum_positions=1, maximum_holding_days=20),
        DEFAULT_SCALE_IN_POLICIES["increasing_confirmed"],
        "2024-01-01",
        "2024-01-31",
        maximum_open_plans=2,
        target_position_fraction=0.50,
    )

    assert len(result.trades) == 2
    assert result.trades["deployed_fraction"].between(0.18, 0.21).all()


def test_survival_policy_waits_five_and_ten_sessions_before_adding() -> None:
    dates = pd.bdate_range("2024-01-02", periods=14)
    rows = []
    for index, date in enumerate(dates):
        close = 10.0 if index < 6 else 10.1
        rows.append(
            (
                date.strftime("%Y%m%d"),
                close,
                close + 0.1,
                close - 0.1,
                close,
                0.01,
            )
        )
    result = run_blood_chip_scale_in_backtest(
        _daily(rows),
        _signal(),
        BloodChipBacktestConfig(maximum_positions=1, maximum_holding_days=20),
        DEFAULT_SCALE_IN_POLICIES["increasing_survival"],
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["tranches_filled"] == 3
    assert trade["tranche_dates"] == "2024-01-03|2024-01-11|2024-01-18"


def test_survival_position_exports_next_stage_state_at_latest_close() -> None:
    dates = pd.bdate_range("2024-01-02", periods=7)
    rows = [
        (
            date.strftime("%Y%m%d"),
            10.0,
            10.2,
            9.9,
            10.0,
            0.01,
        )
        for date in dates
    ]

    result = run_blood_chip_scale_in_backtest(
        _daily(rows),
        _signal(),
        BloodChipBacktestConfig(maximum_positions=1, maximum_holding_days=20),
        DEFAULT_SCALE_IN_POLICIES["increasing_survival"],
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["tranches_filled"] == 1
    assert trade["entry_fill"] == pytest.approx(10.005)
    assert trade["signal_close"] == pytest.approx(10.0)
    assert trade["stop_price"] == pytest.approx(9.0045)
    assert trade["current_residual_return_3d"] == pytest.approx(0.01)
    assert bool(trade["next_stage_ready"]) is True


def test_risk_capped_survival_policy_raises_stop_after_addition() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    rows = []
    for index, date in enumerate(dates):
        open_price = 10.0
        high = 10.2
        low = 9.9
        close = 10.1
        if index == 7:
            open_price, high, low, close = 10.5, 10.6, 10.4, 10.5
        elif index == 8:
            open_price, high, low, close = 9.3, 9.4, 9.15, 9.2
        rows.append(
            (date.strftime("%Y%m%d"), open_price, high, low, close, 0.01)
        )

    result = run_blood_chip_scale_in_backtest(
        _daily(rows),
        _signal(),
        BloodChipBacktestConfig(maximum_positions=1, maximum_holding_days=20),
        DEFAULT_SCALE_IN_POLICIES["increasing_survival_risk_capped"],
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["tranches_filled"] == 2
    assert trade["exit_date"] == pd.Timestamp("2024-01-12")
    assert trade["exit_reason"] == "stop_loss"
