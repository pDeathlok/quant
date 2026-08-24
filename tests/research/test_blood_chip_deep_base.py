from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.blood_chip_deep_base import (
    DeepBaseExecutionConfig,
    DeepBaseSignalConfig,
    build_deep_base_features,
    generate_deep_base_signals,
    run_deep_base_backtest,
)


def _daily_from_close(
    closes: list[float],
    *,
    amounts: list[float] | None = None,
    start: str = "2020-01-02",
    symbol: str = "000001.SZ",
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    close = pd.Series(closes, dtype=float)
    previous = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame(
        {
            "ts_code": symbol,
            "trade_date": dates.strftime("%Y%m%d"),
            "open": close,
            "high": close * 1.006,
            "low": close * 0.994,
            "close": close,
            "pre_close": previous,
            "pct_chg": (close / previous - 1.0) * 100.0,
            "vol": 1_000_000.0,
            "amount": amounts or [100_000.0] * len(closes),
        }
    )
    return frame


def _deep_base_daily(*, healthy: bool = True) -> pd.DataFrame:
    peak = [100.0 + 0.2 * np.sin(index / 4.0) for index in range(60)]
    decline = np.linspace(99.0, 39.0, 50).tolist()
    base = [39.6 + 0.35 * np.sin(index / 3.0) for index in range(78)]
    closes = peak + decline + base
    amounts: list[float] = []
    for index, value in enumerate(closes):
        if index < 110:
            amounts.append(180_000.0)
        elif index == 110 or value >= closes[index - 1]:
            amounts.append(80_000.0)
        else:
            amounts.append(18_000.0 if healthy else 120_000.0)
    frame = _daily_from_close(closes, amounts=amounts, start="2019-01-02")
    if not healthy:
        tail = frame.index[-60:]
        oscillation = np.resize(np.array([0.82, 1.18]), len(tail))
        frame.loc[tail, "close"] = frame.loc[tail, "close"].to_numpy() * oscillation
        frame.loc[tail, "open"] = frame.loc[tail, "close"]
        frame.loc[tail, "high"] = frame.loc[tail, "close"] * 1.02
        frame.loc[tail, "low"] = frame.loc[tail, "close"] * 0.98
        frame["pre_close"] = frame["close"].shift(1).fillna(frame["close"])
        frame["pct_chg"] = (frame["close"] / frame["pre_close"] - 1.0) * 100.0
    return frame


def _signal_config(**overrides: object) -> DeepBaseSignalConfig:
    values: dict[str, object] = {
        "minimum_history_days": 150,
        "minimum_peak_age_sessions": 80,
        "minimum_deep_drawdown_sessions": 35,
        "signal_cooldown_sessions": 120,
    }
    values.update(overrides)
    return DeepBaseSignalConfig(**values)


def _execution_daily(
    closes: list[float],
    *,
    opens: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    frame = _daily_from_close(closes, start="2024-01-02")
    if opens is not None:
        frame["open"] = opens
    if lows is not None:
        frame["low"] = lows
    frame["high"] = np.maximum(frame["high"], frame[["open", "close"]].max(axis=1))
    frame["low"] = np.minimum(frame["low"], frame[["open", "close"]].min(axis=1))
    frame["date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
    frame["adjustment_factor"] = 1.0
    for column in ("open", "high", "low", "close"):
        frame[f"adjusted_{column}"] = frame[column]
    frame["return_20d"] = frame["adjusted_close"].pct_change(20, fill_method=None)
    return frame


def _signal(entry_date: str = "2024-01-03") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "base_event_id": 1,
                "signal_date": pd.Timestamp("2024-01-02"),
                "entry_date": pd.Timestamp(entry_date),
                "entry_open": 10.0,
                "signal_close": 10.0,
                "signal_score": 1.0,
                "base_low": 9.0,
                "base_high": 11.0,
                "base_mid": 10.0,
                "base_position": 0.5,
                "drawdown_from_peak": -0.70,
                "prior_peak": 33.333333,
                "peak_age_sessions": 300,
            }
        ]
    )


def _execution_config(**overrides: object) -> DeepBaseExecutionConfig:
    values: dict[str, object] = {
        "maximum_positions": 1,
        "target_position_fraction": 1.0,
        "maximum_holding_sessions": 80,
    }
    values.update(overrides)
    return DeepBaseExecutionConfig(**values)


def test_deep_drawdown_uses_only_prior_visible_peak() -> None:
    frame = _daily_from_close([10.0, 12.0, 9.0, 20.0, 18.0])

    features = build_deep_base_features(frame)

    assert pd.isna(features.loc[0, "prior_peak"])
    assert features.loc[1, "prior_peak"] == pytest.approx(10.06)
    assert features.loc[3, "prior_peak"] == pytest.approx(12.072)
    assert features.loc[4, "prior_peak"] == pytest.approx(20.12)


def test_future_rows_do_not_change_existing_deep_base_features() -> None:
    daily = _deep_base_daily()
    prefix = daily.iloc[:-8].copy()
    extended = pd.concat(
        [daily, _daily_from_close([180.0] * 8, start="2025-01-02")],
        ignore_index=True,
    )

    before = build_deep_base_features(prefix).set_index(["ts_code", "date"])
    after = build_deep_base_features(extended).set_index(["ts_code", "date"])
    columns = [
        "prior_peak",
        "drawdown_from_peak",
        "base_low",
        "base_high",
        "base_range",
        "recent_down_amount_share",
        "volatility_contraction_ratio",
    ]

    pd.testing.assert_frame_equal(before[columns], after.loc[before.index, columns])


def test_signal_requires_threshold_duration_and_sixty_session_base() -> None:
    features = build_deep_base_features(_deep_base_daily())

    signals = generate_deep_base_signals(features, _signal_config())

    assert len(signals) == 1
    signal = signals.iloc[0]
    assert signal["drawdown_from_peak"] <= -0.50
    assert signal["deep_drawdown_sessions"] >= 35
    assert signal["base_range"] <= 0.25
    assert signal["entry_date"] > signal["signal_date"]


def test_falling_price_or_wide_range_does_not_signal() -> None:
    features = build_deep_base_features(_deep_base_daily(healthy=False))

    signals = generate_deep_base_signals(features, _signal_config())

    assert signals.empty


def test_recent_down_amount_and_volatility_must_contract() -> None:
    features = build_deep_base_features(_deep_base_daily(healthy=False))
    tail = features.index[-60:]
    assert features.loc[tail, "volatility_contraction_ratio"].median() > 0.85

    signals = generate_deep_base_signals(features, _signal_config())

    assert signals.empty


def test_signal_cooldown_suppresses_duplicate_base_entries() -> None:
    features = build_deep_base_features(_deep_base_daily())

    signals = generate_deep_base_signals(
        features,
        _signal_config(signal_cooldown_sessions=120),
    )

    assert len(signals) == 1


def test_signal_ranking_rewards_base_quality_not_deeper_drawdown() -> None:
    left = build_deep_base_features(_deep_base_daily())
    baseline = generate_deep_base_signals(left, _signal_config())
    signal_date = baseline.iloc[0]["signal_date"]
    right = left.copy()
    right["ts_code"] = "000002.SZ"
    right.loc[
        right["date"].eq(signal_date),
        "drawdown_from_peak",
    ] -= 0.20
    combined = pd.concat([left, right], ignore_index=True)

    signals = generate_deep_base_signals(combined, _signal_config())

    scores = signals.set_index("ts_code")["signal_score"]
    assert scores["000001.SZ"] == pytest.approx(scores["000002.SZ"])


def test_first_second_and_third_tranches_trade_on_successive_next_opens() -> None:
    closes = [10.0] * 31
    closes[10:20] = [9.7] * 10
    closes[20:] = np.linspace(9.7, 10.4, 11).tolist()
    result = run_deep_base_backtest(
        _execution_daily(closes),
        _signal(),
        _execution_config(),
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["tranches_filled"] == 3
    assert trade["planned_fractions"] == "0.2000|0.3000|0.5000"
    assert trade["deployed_fraction"] == pytest.approx(1.0, abs=0.03)
    tranche_dates = trade["tranche_dates"].split("|")
    assert tranche_dates[0] == "2024-01-03"
    assert pd.Timestamp(tranche_dates[1]) > pd.Timestamp(tranche_dates[0])
    assert pd.Timestamp(tranche_dates[2]) > pd.Timestamp(tranche_dates[1])


def test_unconfirmed_base_keeps_partial_position() -> None:
    closes = [10.0] + [10.7] * 29
    result = run_deep_base_backtest(
        _execution_daily(closes),
        _signal(),
        _execution_config(maximum_holding_sessions=20),
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["tranches_filled"] == 1
    assert trade["deployed_fraction"] == pytest.approx(0.20, abs=0.03)


def test_additions_never_chase_above_original_base() -> None:
    closes = [10.0] * 12 + [11.5] * 20
    opens = closes.copy()
    opens[12] = 11.5
    result = run_deep_base_backtest(
        _execution_daily(closes, opens=opens),
        _signal(),
        _execution_config(maximum_holding_sessions=25),
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["tranches_filled"] == 1


def test_structural_break_exits_on_next_open() -> None:
    closes = [10.0, 10.0, 8.65, 8.60, 8.55, 8.50]
    opens = [10.0, 10.0, 8.7, 8.62, 8.55, 8.50]
    lows = [9.9, 9.9, 8.55, 8.50, 8.45, 8.40]
    result = run_deep_base_backtest(
        _execution_daily(closes, opens=opens, lows=lows),
        _signal(),
        _execution_config(maximum_holding_sessions=30),
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "structural_break"
    assert trade["exit_date"] == pd.Timestamp("2024-01-08")


def test_gap_through_hard_stop_uses_open_and_t_plus_one() -> None:
    closes = [10.0, 10.0, 7.8, 7.9]
    opens = [10.0, 10.0, 7.8, 7.9]
    lows = [9.9, 7.5, 7.7, 7.8]
    result = run_deep_base_backtest(
        _execution_daily(closes, opens=opens, lows=lows),
        _signal(),
        _execution_config(maximum_holding_sessions=30),
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "hard_stop"
    assert trade["exit_date"] == pd.Timestamp("2024-01-04")
    assert trade["exit_fill"] == pytest.approx(7.8 * (1.0 - 0.0005))


def test_long_hold_exits_at_configured_session_count() -> None:
    closes = [10.0] * 12
    result = run_deep_base_backtest(
        _execution_daily(closes),
        _signal(),
        _execution_config(maximum_holding_sessions=5),
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "time_exit"
    assert trade["holding_sessions"] == 5


def test_sixty_missing_market_sessions_write_position_off() -> None:
    first = _execution_daily([10.0, 10.0])
    market_only = _execution_daily(
        [20.0] * 4,
    ).assign(ts_code="000002.SZ")
    daily = pd.concat([first, market_only.iloc[2:]], ignore_index=True)
    result = run_deep_base_backtest(
        daily,
        _signal(),
        _execution_config(
            maximum_holding_sessions=30,
            maximum_missing_market_sessions=2,
        ),
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "missing_bar_writeoff"
    assert trade["exit_value"] == 0.0
    assert trade["net_return"] == pytest.approx(-1.0)


def test_retest_reclaim_waits_while_price_remains_in_lower_half() -> None:
    closes = [10.0] + [9.6] * 29
    result = run_deep_base_backtest(
        _execution_daily(closes),
        _signal(),
        _execution_config(
            stage_policy="retest_reclaim",
            structural_exit_enabled=False,
            maximum_holding_sessions=20,
        ),
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["tranches_filled"] == 1


def test_retest_reclaim_adds_after_midline_recovery() -> None:
    closes = [10.0] * 10 + [9.6] * 11 + [10.1, 10.2, 10.5, 10.7, 10.8, 10.9] * 3
    result = run_deep_base_backtest(
        _execution_daily(closes),
        _signal(),
        _execution_config(
            stage_policy="retest_reclaim",
            structural_exit_enabled=False,
            maximum_holding_sessions=30,
        ),
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["tranches_filled"] == 3
    assert trade["deployed_fraction"] == pytest.approx(1.0, abs=0.03)


def test_hard_stop_only_policy_ignores_shallow_structural_break() -> None:
    closes = [10.0, 10.0, 8.65, 8.60, 9.0, 9.4, 9.8, 10.0]
    opens = [10.0, 10.0, 8.7, 8.62, 9.0, 9.4, 9.8, 10.0]
    lows = [9.9, 9.9, 8.55, 8.50, 8.9, 9.3, 9.7, 9.9]
    result = run_deep_base_backtest(
        _execution_daily(closes, opens=opens, lows=lows),
        _signal(),
        _execution_config(
            stage_policy="retest_reclaim",
            structural_exit_enabled=False,
            maximum_holding_sessions=6,
        ),
        "2024-01-01",
        "2024-01-31",
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "time_exit"
