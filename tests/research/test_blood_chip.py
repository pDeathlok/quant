from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.blood_chip import (
    BloodChipBacktestConfig,
    BloodChipBacktestResult,
    BloodChipSignalConfig,
    add_blood_chip_path_features,
    analyze_blood_chip_cases,
    build_blood_chip_features,
    generate_blood_chip_signals,
    run_blood_chip_backtest,
    summarize_blood_chip_result,
)


def _feature_inputs(periods: int = 150) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(["000001.SZ", "000002.SZ", "600000.SH"]):
        close = 10.0 + symbol_index + np.arange(periods) * (0.01 + symbol_index * 0.001)
        for index, date in enumerate(dates):
            previous = close[index - 1] if index else close[index]
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": date.strftime("%Y%m%d"),
                    "open": close[index] * 0.998,
                    "high": close[index] * 1.01,
                    "low": close[index] * 0.99,
                    "close": close[index],
                    "pre_close": previous,
                    "pct_chg": (close[index] / previous - 1.0) * 100.0,
                    "vol": 1_000_000.0 + index * 1_000.0,
                    "amount": 100_000.0 + index * 100.0,
                }
            )
    daily = pd.DataFrame(rows)
    benchmark_close = 4_000.0 + np.arange(periods) * 2.0
    benchmark = pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y%m%d"),
            "close": benchmark_close,
            "pct_chg": pd.Series(benchmark_close).pct_change().fillna(0.0).to_numpy() * 100.0,
        }
    )
    return daily, benchmark


def _execution_daily(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=["trade_date", "open", "high", "low", "close"],
    )
    frame["ts_code"] = "000001.SZ"
    frame["pre_close"] = frame["close"].shift(1).fillna(frame["close"])
    frame["pct_chg"] = (frame["close"] / frame["pre_close"] - 1.0) * 100.0
    frame["vol"] = 1_000_000.0
    frame["amount"] = 100_000.0
    for column in ("open", "high", "low", "close"):
        frame[f"adjusted_{column}"] = frame[column]
    frame["date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
    return frame


def _signal(
    signal_date: str,
    entry_date: str,
    event_id: int,
    *,
    score: float = 0.9,
) -> dict[str, object]:
    return {
        "ts_code": "000001.SZ",
        "signal_date": pd.Timestamp(signal_date),
        "entry_date": pd.Timestamp(entry_date),
        "shock_date": pd.Timestamp(signal_date) - pd.Timedelta(days=2),
        "shock_event_id": event_id,
        "signal_score": score,
        "shock_score": score,
        "absorption_score": score,
        "return_120d": -0.10,
        "volatility_60d": 0.02,
        "market_return_60d": 0.01,
        "impact_decay": 0.5,
        "rebound_from_event_low": 0.04,
        "clv_3d": 0.2,
    }


def test_future_rows_do_not_change_existing_feature_values() -> None:
    daily, benchmark = _feature_inputs(150)
    original = build_blood_chip_features(daily, benchmark)
    extended_daily, extended_benchmark = _feature_inputs(160)
    extended = build_blood_chip_features(extended_daily, extended_benchmark)

    columns = [
        "adjusted_close",
        "market_beta_60d",
        "residual_return_5d",
        "amount_ratio_5d",
        "impact_ratio_5d",
        "return_120d",
    ]
    left = original.loc[original["ts_code"].eq("000001.SZ"), columns].reset_index(drop=True)
    right = extended.loc[
        extended["ts_code"].eq("000001.SZ"), columns
    ].iloc[: len(left)].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_build_features_rejects_missing_terminal_benchmark_observation() -> None:
    daily, benchmark = _feature_inputs(150)
    expected_date = daily["trade_date"].max()
    benchmark = benchmark.loc[benchmark["trade_date"] != expected_date].copy()

    with pytest.raises(ValueError, match="benchmark does not cover terminal daily date"):
        build_blood_chip_features(daily, benchmark)


def test_build_features_rejects_missing_terminal_benchmark_return() -> None:
    daily, benchmark = _feature_inputs(150)
    benchmark.loc[benchmark.index[-1], "pct_chg"] = np.nan

    with pytest.raises(ValueError, match="terminal benchmark return is missing"):
        build_blood_chip_features(daily, benchmark)


def test_signal_requires_shock_then_absorption_and_deduplicates_event() -> None:
    dates = pd.bdate_range("2024-01-02", periods=15)
    features = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "date": dates,
            "adjusted_open": 10.0,
            "adjusted_high": 10.2,
            "adjusted_low": 9.8,
            "adjusted_close": 10.0,
            "history_days": 200,
            "prior_amount_median_20d": 100_000.0,
            "residual_5d_percentile": 0.50,
            "amount_ratio_5d": 1.0,
            "impact_ratio_5d": 1.0,
            "drawdown_20d_percentile": 0.50,
            "shock_score": 0.50,
            "residual_return_3d": -0.01,
            "down_impact_3d": 1.0,
            "clv_3d": -0.5,
            "return_120d": -0.1,
            "volatility_60d": 0.03,
            "market_return_60d": 0.01,
        }
    )
    features.loc[3, [
        "residual_5d_percentile",
        "amount_ratio_5d",
        "impact_ratio_5d",
        "drawdown_20d_percentile",
        "shock_score",
    ]] = [0.01, 2.0, 3.0, 0.02, 0.95]
    features.loc[5:7, "adjusted_low"] = [9.0, 9.1, 9.2]
    features.loc[5:7, "adjusted_close"] = [9.2, 9.45, 9.55]
    features.loc[5:7, "residual_return_3d"] = [0.01, 0.02, 0.03]
    features.loc[5:7, "down_impact_3d"] = [0.5, 0.4, 0.3]
    features.loc[5:7, "clv_3d"] = [0.1, 0.2, 0.3]

    signals = generate_blood_chip_signals(features, BloodChipSignalConfig())

    assert len(signals) == 1
    assert signals.iloc[0]["signal_date"] > signals.iloc[0]["shock_date"]
    assert signals.iloc[0]["entry_date"] > signals.iloc[0]["signal_date"]


def test_latest_confirmation_is_only_returned_as_explicit_pending_entry() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    features = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "date": dates,
            "adjusted_open": 10.0,
            "adjusted_high": 10.2,
            "adjusted_low": 9.8,
            "adjusted_close": 10.0,
            "history_days": 200,
            "prior_amount_median_20d": 100_000.0,
            "residual_5d_percentile": 0.50,
            "amount_ratio_5d": 1.0,
            "impact_ratio_5d": 1.0,
            "drawdown_20d_percentile": 0.50,
            "shock_score": 0.50,
            "residual_return_3d": -0.01,
            "down_impact_3d": 1.0,
            "clv_3d": -0.5,
            "return_120d": -0.1,
            "volatility_60d": 0.30,
            "market_return_60d": 0.01,
        }
    )
    features.loc[3, [
        "residual_5d_percentile",
        "amount_ratio_5d",
        "impact_ratio_5d",
        "drawdown_20d_percentile",
        "shock_score",
    ]] = [0.01, 2.0, 3.0, 0.02, 0.95]
    features.loc[7, [
        "adjusted_low",
        "adjusted_close",
        "residual_return_3d",
        "down_impact_3d",
        "clv_3d",
    ]] = [9.0, 9.5, 0.03, 0.3, 0.3]

    default_signals = generate_blood_chip_signals(features, BloodChipSignalConfig())
    pending_signals = generate_blood_chip_signals(
        features,
        BloodChipSignalConfig(),
        include_pending_entry=True,
    )

    assert default_signals.empty
    assert len(pending_signals) == 1
    assert pending_signals.iloc[0]["signal_date"] == dates[-1]
    assert pd.isna(pending_signals.iloc[0]["entry_date"])
    assert pd.isna(pending_signals.iloc[0]["entry_open"])


def test_signal_quality_filters_use_only_entry_time_features() -> None:
    dates = pd.bdate_range("2024-01-02", periods=15)
    features = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "date": dates,
            "adjusted_open": 10.0,
            "adjusted_high": 10.2,
            "adjusted_low": 9.8,
            "adjusted_close": 10.0,
            "history_days": 200,
            "prior_amount_median_20d": 100_000.0,
            "residual_5d_percentile": 0.50,
            "amount_ratio_5d": 1.0,
            "impact_ratio_5d": 1.0,
            "drawdown_20d_percentile": 0.50,
            "shock_score": 0.50,
            "residual_return_3d": -0.01,
            "down_impact_3d": 1.0,
            "clv_3d": -0.5,
            "return_120d": 0.80,
            "volatility_60d": 0.65,
            "market_return_60d": -0.25,
        }
    )
    features.loc[3, [
        "residual_5d_percentile",
        "amount_ratio_5d",
        "impact_ratio_5d",
        "drawdown_20d_percentile",
        "shock_score",
    ]] = [0.01, 2.0, 3.0, 0.02, 0.95]
    features.loc[5:7, "adjusted_low"] = [9.0, 9.1, 9.2]
    features.loc[5:7, "adjusted_close"] = [9.2, 9.45, 9.55]
    features.loc[5:7, "residual_return_3d"] = [0.01, 0.02, 0.03]
    features.loc[5:7, "down_impact_3d"] = [0.5, 0.4, 0.3]
    features.loc[5:7, "clv_3d"] = [0.1, 0.2, 0.3]

    signals = generate_blood_chip_signals(
        features,
        BloodChipSignalConfig(
            maximum_return_120d=0.75,
            maximum_volatility_60d=0.60,
            minimum_market_return_60d=-0.20,
        ),
    )

    assert signals.empty


def test_path_features_capture_exhaustion_without_future_rows() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    features = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "date": dates,
            "adjusted_high": [10.2, 10.3, 10.1, 9.8, 9.2, 9.15, 9.2, 9.25, 20.0, 30.0],
            "adjusted_low": [9.8, 9.7, 9.4, 8.9, 8.0, 8.85, 8.95, 9.0, 1.0, 1.0],
            "adjusted_close": [10.0, 10.0, 9.6, 9.1, 8.4, 9.0, 9.1, 9.2, 10.0, 10.0],
            "amount": [100.0, 120.0, 200.0, 300.0, 500.0, 90.0, 80.0, 70.0, 1_000.0, 1_000.0],
            "residual_return_1d": [0.0, 0.01, -0.04, -0.06, -0.10, 0.02, 0.01, 0.01, -0.50, 0.50],
            "volatility_60d": 0.60,
        }
    )
    signal = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "shock_date": [dates[4]],
            "signal_date": [dates[7]],
        }
    )

    path = add_blood_chip_path_features(features, signal)
    truncated = add_blood_chip_path_features(features.iloc[:8], signal)

    assert path.iloc[0]["volatility_decay_ratio"] < 1.0
    assert path.iloc[0]["shock_volatility_expansion_ratio"] > 1.0
    assert path.iloc[0]["amount_decay_ratio"] < 1.0
    assert path.iloc[0]["confirmation_amount_vs_prior_ratio"] < 1.0
    assert path.iloc[0]["downside_amount_decay_ratio"] == pytest.approx(0.0)
    assert path.iloc[0]["confirmation_downside_amount_share"] == pytest.approx(0.0)
    assert path.iloc[0]["event_low_age_sessions"] == 3
    pd.testing.assert_series_equal(path.iloc[0], truncated.iloc[0], check_names=False)


def test_entry_uses_next_available_open() -> None:
    daily = _execution_daily(
        [
            ("20240102", 10.0, 10.2, 9.8, 10.0),
            ("20240103", 10.5, 10.7, 10.4, 10.6),
            ("20240104", 10.6, 10.8, 10.5, 10.7),
            ("20240105", 10.7, 10.8, 10.6, 10.7),
        ]
    )
    signals = pd.DataFrame([_signal("2024-01-02", "2024-01-03", 1)])

    result = run_blood_chip_backtest(
        daily,
        signals,
        BloodChipBacktestConfig(maximum_holding_days=1),
        "2024-01-01",
        "2024-01-31",
    )

    assert result.trades.iloc[0]["entry_raw"] == pytest.approx(10.5)
    assert result.trades.iloc[0]["entry_date"] == pd.Timestamp("2024-01-03")


def test_t_plus_one_blocks_same_day_stop() -> None:
    daily = _execution_daily(
        [
            ("20240102", 10.0, 10.1, 9.9, 10.0),
            ("20240103", 10.0, 10.2, 8.0, 10.0),
            ("20240104", 10.0, 10.2, 9.8, 10.1),
            ("20240105", 10.1, 10.2, 10.0, 10.1),
        ]
    )
    signals = pd.DataFrame([_signal("2024-01-02", "2024-01-03", 1)])

    result = run_blood_chip_backtest(
        daily,
        signals,
        BloodChipBacktestConfig(stop_loss=0.10, maximum_holding_days=1),
        "2024-01-01",
        "2024-01-31",
    )

    assert result.trades.iloc[0]["exit_reason"] == "time_exit"
    assert result.trades.iloc[0]["exit_date"] == pd.Timestamp("2024-01-04")


def test_gap_through_stop_fills_at_open() -> None:
    daily = _execution_daily(
        [
            ("20240102", 10.0, 10.1, 9.9, 10.0),
            ("20240103", 10.0, 10.2, 9.8, 10.0),
            ("20240104", 8.5, 8.7, 8.4, 8.6),
            ("20240105", 8.6, 8.7, 8.5, 8.6),
        ]
    )
    signals = pd.DataFrame([_signal("2024-01-02", "2024-01-03", 1)])

    result = run_blood_chip_backtest(
        daily,
        signals,
        BloodChipBacktestConfig(stop_loss=0.10, maximum_holding_days=10),
        "2024-01-01",
        "2024-01-31",
    )

    assert result.trades.iloc[0]["exit_reason"] == "stop_loss"
    assert result.trades.iloc[0]["exit_raw"] == pytest.approx(8.5)


def test_locked_limit_down_delays_stop_exit() -> None:
    daily = _execution_daily(
        [
            ("20240102", 10.0, 10.1, 9.9, 10.0),
            ("20240103", 10.0, 10.2, 9.8, 10.0),
            ("20240104", 9.0, 9.0, 9.0, 9.0),
            ("20240105", 8.8, 9.0, 8.7, 8.9),
            ("20240108", 8.9, 9.0, 8.8, 8.9),
        ]
    )
    signals = pd.DataFrame([_signal("2024-01-02", "2024-01-03", 1)])

    result = run_blood_chip_backtest(
        daily,
        signals,
        BloodChipBacktestConfig(stop_loss=0.10, maximum_holding_days=10),
        "2024-01-01",
        "2024-01-31",
    )

    assert result.trades.iloc[0]["exit_date"] == pd.Timestamp("2024-01-05")
    assert result.trades.iloc[0]["exit_raw"] == pytest.approx(8.8)


def test_stop_does_not_permanently_blacklist_symbol() -> None:
    daily = _execution_daily(
        [
            ("20240102", 10.0, 10.1, 9.9, 10.0),
            ("20240103", 10.0, 10.1, 9.8, 10.0),
            ("20240104", 8.8, 9.0, 8.7, 8.9),
            ("20240105", 9.0, 9.2, 8.9, 9.1),
            ("20240108", 9.1, 9.4, 9.0, 9.3),
            ("20240109", 9.3, 9.5, 9.2, 9.4),
        ]
    )
    signals = pd.DataFrame(
        [
            _signal("2024-01-02", "2024-01-03", 1),
            _signal("2024-01-04", "2024-01-05", 2),
        ]
    )

    result = run_blood_chip_backtest(
        daily,
        signals,
        BloodChipBacktestConfig(stop_loss=0.10, maximum_holding_days=2),
        "2024-01-01",
        "2024-01-31",
    )

    assert result.trades["shock_event_id"].tolist() == [1, 2]
    assert result.trades["reentry_number"].tolist() == [0, 1]


def test_reentry_requires_a_new_shock_event() -> None:
    daily = _execution_daily(
        [
            ("20240102", 10.0, 10.1, 9.9, 10.0),
            ("20240103", 10.0, 10.1, 9.8, 10.0),
            ("20240104", 8.8, 9.0, 8.7, 8.9),
            ("20240105", 9.0, 9.2, 8.9, 9.1),
            ("20240108", 9.1, 9.4, 9.0, 9.3),
        ]
    )
    signals = pd.DataFrame(
        [
            _signal("2024-01-02", "2024-01-03", 1),
            _signal("2024-01-04", "2024-01-05", 1),
        ]
    )

    result = run_blood_chip_backtest(
        daily,
        signals,
        BloodChipBacktestConfig(stop_loss=0.10, maximum_holding_days=2),
        "2024-01-01",
        "2024-01-31",
    )

    assert result.trades["shock_event_id"].tolist() == [1]
    assert "event_reused" in result.rejected_entries["reason"].tolist()


def test_round_trip_costs_include_sell_stamp_tax() -> None:
    daily = _execution_daily(
        [
            ("20240102", 10.0, 10.1, 9.9, 10.0),
            ("20240103", 10.0, 10.1, 9.9, 10.0),
            ("20240104", 10.0, 10.1, 9.9, 10.0),
        ]
    )
    signals = pd.DataFrame([_signal("2024-01-02", "2024-01-03", 1)])

    result = run_blood_chip_backtest(
        daily,
        signals,
        BloodChipBacktestConfig(maximum_holding_days=1, slippage=0.0),
        "2024-01-01",
        "2024-01-31",
    )
    trade = result.trades.iloc[0]

    assert trade["gross_return"] == pytest.approx(0.0)
    assert trade["net_return"] < -0.001
    assert trade["fees"] > trade["entry_value"] * 0.0005


def test_summary_total_return_includes_first_portfolio_day() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "cash": [900.0, 900.0],
            "positions": [0, 0],
            "equity": [900.0, 900.0],
            "daily_return": [-0.10, 0.0],
        }
    )
    benchmark = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "close": [100.0, 100.0],
            "pct_chg": [0.0, 0.0],
        }
    )

    metrics = summarize_blood_chip_result(
        BloodChipBacktestResult(equity, pd.DataFrame(), pd.DataFrame()),
        benchmark,
    )

    assert metrics["total_return"] == pytest.approx(-0.10)


def test_case_analysis_uses_only_entry_time_features() -> None:
    trades = pd.DataFrame(
        {
            "net_return": [0.20, -0.10, -0.15],
            "exit_reason": ["time_exit", "stop_loss", "stop_loss"],
            "reentry_number": [0, 0, 1],
            "return_120d": [0.1, -0.2, -0.4],
            "volatility_60d": [0.02, 0.04, 0.08],
            "market_return_60d": [0.05, -0.02, -0.10],
            "shock_score": [0.8, 0.9, 0.95],
            "absorption_score": [0.9, 0.7, 0.6],
            "impact_decay": [0.4, 0.8, 1.0],
            "rebound_from_event_low": [0.08, 0.03, 0.02],
            "future_leak": [999.0, 999.0, 999.0],
        }
    )

    cases = analyze_blood_chip_cases(trades)

    assert "future_leak" not in cases["feature"].tolist()
    assert {"winner", "loss", "stop_loss", "failed_reentry"} <= set(cases["case_type"])
