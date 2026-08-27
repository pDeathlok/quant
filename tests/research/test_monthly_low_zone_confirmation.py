from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.monthly_low_zone import (
    MonthlyLowZoneConfig,
    evaluate_monthly_low_zone_events,
)
from quant.research.monthly_low_zone_confirmation import (
    MonthlyConfirmationConfig,
    build_benchmark_confirmation_features,
    build_market_breadth_features,
    generate_monthly_confirmation_signals,
)


def _anchors(
    date: pd.Timestamp,
    *,
    symbol: str = "000001.SZ",
    anchor_id: int = 1,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_id": [anchor_id],
            "ts_code": [symbol],
            "signal_date": [date],
            "rule": ["monthly_low9"],
            "month_period": [date.to_period("M").strftime("%Y-%m")],
            "adjusted_close": [50.0],
            "prior_peak": [100.0],
            "drawdown_from_prior_peak": [-0.50],
            "monthly_j": [-5.0],
            "weekly_j": [-5.0],
            "monthly_low9": [True],
            "monthly_low9_count": [9],
            "median_daily_amount": [100_000.0],
        }
    )


def _daily_features(
    dates: pd.DatetimeIndex,
    *,
    symbol: str = "000001.SZ",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": symbol,
            "date": dates,
            "adjusted_close": 50.0,
            "adjusted_low": 49.0,
            "prior_amount_median_20d": 100_000.0,
            "sessions_since_new_low": 30,
            "return_20d": 0.10,
            "base_position": 0.60,
            "down_amount_share_ratio": 0.80,
            "volatility_contraction_ratio": 0.80,
        }
    )


def _benchmark_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "benchmark_close": 100.0,
            "benchmark_return_20d": 0.02,
            "benchmark_ma120": 99.0,
            "breadth_constituents": 1_000,
            "breadth_median_return_20d": 0.02,
            "breadth_positive_share_20d": 0.60,
        }
    )


def _config(**overrides: object) -> MonthlyConfirmationConfig:
    values: dict[str, object] = {
        "minimum_wait_sessions": 5,
        "maximum_wait_sessions": 126,
    }
    values.update(overrides)
    return MonthlyConfirmationConfig(**values)


def test_confirmation_never_precedes_anchor_or_minimum_wait() -> None:
    calendar = pd.bdate_range("2020-01-02", periods=140)
    signals, _ = generate_monthly_confirmation_signals(
        _daily_features(calendar),
        pd.DataFrame(),
        _anchors(calendar[0]),
        _benchmark_features(calendar),
        calendar,
        _config(),
    )

    direct = signals[signals["rule"].eq("anchor_direct")].iloc[0]
    confirmed = signals[signals["rule"].eq("no_new_low_20")].iloc[0]
    assert direct["signal_date"] == calendar[0]
    assert confirmed["signal_date"] == calendar[5]
    assert confirmed["confirmation_wait_sessions"] == 5


def test_confirmation_uses_first_eligible_day_and_expires_after_126_sessions() -> None:
    calendar = pd.bdate_range("2020-01-02", periods=150)
    daily = _daily_features(calendar)
    daily["sessions_since_new_low"] = 0
    daily.loc[daily["date"].isin([calendar[10], calendar[20]]), "sessions_since_new_low"] = 30
    second = _daily_features(calendar, symbol="000002.SZ")
    second["sessions_since_new_low"] = 0
    second.loc[second["date"].eq(calendar[130]), "sessions_since_new_low"] = 30
    anchors = pd.concat(
        [_anchors(calendar[0]), _anchors(calendar[0], symbol="000002.SZ", anchor_id=2)],
        ignore_index=True,
    )

    signals, diagnostics = generate_monthly_confirmation_signals(
        pd.concat([daily, second], ignore_index=True),
        pd.DataFrame(),
        anchors,
        _benchmark_features(calendar),
        calendar,
        _config(),
    )

    first = signals[
        signals["ts_code"].eq("000001.SZ") & signals["rule"].eq("no_new_low_20")
    ].iloc[0]
    expired = diagnostics[
        diagnostics["ts_code"].eq("000002.SZ")
        & diagnostics["rule"].eq("no_new_low_20")
    ].iloc[0]
    assert first["signal_date"] == calendar[10]
    assert expired["confirmation_status"] == "expired"


def test_weekly_state_cannot_use_incomplete_week() -> None:
    calendar = pd.bdate_range("2024-01-22", periods=20)
    weekly = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "weekly_available_date": [calendar[4], calendar[9]],
            "weekly_j": [-15.0, 5.0],
            "weekly_prev_j": [-20.0, -15.0],
        }
    )

    signals, _ = generate_monthly_confirmation_signals(
        _daily_features(calendar),
        weekly,
        _anchors(calendar[0]),
        _benchmark_features(calendar),
        calendar,
        _config(),
    )

    weekly_signal = signals[signals["rule"].eq("range_mid_weekly")].iloc[0]
    assert weekly_signal["signal_date"] == calendar[9]
    assert weekly_signal["weekly_available_date"] == calendar[9]


def test_relative_strength_uses_only_same_day_known_benchmark_history() -> None:
    dates = pd.bdate_range("2019-01-02", periods=150)
    benchmark = pd.DataFrame(
        {
            "trade_date": dates,
            "open": np.linspace(90.0, 110.0, len(dates)),
            "close": np.linspace(90.0, 110.0, len(dates)),
        }
    )
    target = dates[130]

    full = build_benchmark_confirmation_features(benchmark)
    truncated = build_benchmark_confirmation_features(
        benchmark[benchmark["trade_date"].le(target)]
    )
    left = full[full["date"].eq(target)].iloc[0]
    right = truncated[truncated["date"].eq(target)].iloc[0]

    assert left["benchmark_return_20d"] == right["benchmark_return_20d"]
    assert left["benchmark_ma120"] == right["benchmark_ma120"]


def test_confirmation_preserves_anchor_peak_and_limits_rebound_drawdown() -> None:
    calendar = pd.bdate_range("2020-01-02", periods=140)
    daily = _daily_features(calendar)
    daily.loc[daily["date"].between(calendar[5], calendar[9]), "adjusted_close"] = 65.0
    daily.loc[daily["date"].eq(calendar[10]), "adjusted_close"] = 60.0

    signals, _ = generate_monthly_confirmation_signals(
        daily,
        pd.DataFrame(),
        _anchors(calendar[0]),
        _benchmark_features(calendar),
        calendar,
        _config(),
    )

    confirmed = signals[signals["rule"].eq("no_new_low_20")].iloc[0]
    assert confirmed["signal_date"] == calendar[10]
    assert confirmed["anchor_prior_peak"] == 100.0
    assert confirmed["confirmation_drawdown_from_anchor_peak"] == -0.40


def test_waiting_path_drawdown_is_measured_from_anchor_close() -> None:
    calendar = pd.bdate_range("2020-01-02", periods=140)
    daily = _daily_features(calendar)
    daily.loc[daily["date"].eq(calendar[3]), "adjusted_low"] = 40.0

    signals, _ = generate_monthly_confirmation_signals(
        daily,
        pd.DataFrame(),
        _anchors(calendar[0]),
        _benchmark_features(calendar),
        calendar,
        _config(),
    )

    confirmed = signals[signals["rule"].eq("no_new_low_20")].iloc[0]
    assert confirmed["waiting_path_drawdown"] == pytest.approx(-0.20)


def _execution_daily(dates: pd.DatetimeIndex, closes: list[float]) -> pd.DataFrame:
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "date": dates,
            "open": close * 1.001,
            "high": close * 1.02,
            "low": close * 0.98,
            "adjusted_open": close * 1.001,
            "adjusted_high": close * 1.02,
            "adjusted_low": close * 0.98,
            "adjusted_close": close,
        }
    )


def test_anchor_direct_matches_existing_monthly_event_entry() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=20)
    signals, _ = generate_monthly_confirmation_signals(
        _daily_features(calendar),
        pd.DataFrame(),
        _anchors(calendar[0]),
        _benchmark_features(calendar),
        calendar,
        _config(),
    )
    benchmark = pd.DataFrame({"trade_date": calendar, "open": 100.0, "close": 100.0})
    events = evaluate_monthly_low_zone_events(
        _execution_daily(calendar[1:10], list(np.linspace(10.0, 12.0, 9))),
        signals[signals["rule"].eq("anchor_direct")],
        benchmark,
        calendar,
        MonthlyLowZoneConfig(horizons=(2, 3, 4)),
    )

    assert events.iloc[0]["signal_date"] == calendar[0]
    assert events.iloc[0]["entry_date"] == calendar[1]


def test_target_suspension_uses_recovery_open_through_existing_evaluator() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=20)
    signals, _ = generate_monthly_confirmation_signals(
        _daily_features(calendar),
        pd.DataFrame(),
        _anchors(calendar[0]),
        _benchmark_features(calendar),
        calendar,
        _config(),
    )
    benchmark = pd.DataFrame({"trade_date": calendar, "open": 100.0, "close": 100.0})
    events = evaluate_monthly_low_zone_events(
        _execution_daily(
            pd.DatetimeIndex([calendar[1], calendar[2], calendar[6], calendar[7]]),
            [10.0, 10.5, 12.0, 12.5],
        ),
        signals[signals["rule"].eq("anchor_direct")],
        benchmark,
        calendar,
        MonthlyLowZoneConfig(horizons=(3, 4, 5)),
    )
    delayed = events[events["horizon"].eq(3)].iloc[0]

    assert delayed["exit_reason"] == "next_open_after_target_suspension"
    assert delayed["exit_date"] == calendar[6]


def test_breadth_uses_only_same_date_liquid_cross_section() -> None:
    date = pd.Timestamp("2024-01-31")
    daily = pd.DataFrame(
        {
            "ts_code": ["A", "B", "C", "D"],
            "date": [date] * 4,
            "return_20d": [-0.10, 0.10, 0.20, 1.00],
            "prior_amount_median_20d": [50_000.0, 60_000.0, 70_000.0, 1_000.0],
        }
    )

    breadth = build_market_breadth_features(
        daily,
        _config(minimum_breadth_constituents=2),
    ).iloc[0]

    assert breadth["breadth_constituents"] == 3
    assert breadth["breadth_median_return_20d"] == pytest.approx(0.10)
    assert breadth["breadth_positive_share_20d"] == pytest.approx(2 / 3)


def test_future_cross_section_rows_do_not_change_prior_breadth() -> None:
    first = pd.Timestamp("2024-01-31")
    second = pd.Timestamp("2024-02-01")
    daily = pd.DataFrame(
        {
            "ts_code": ["A", "B", "A", "B"],
            "date": [first, first, second, second],
            "return_20d": [-0.10, 0.20, 1.00, 2.00],
            "prior_amount_median_20d": 100_000.0,
        }
    )
    config = _config(minimum_breadth_constituents=2)

    full = build_market_breadth_features(daily, config)
    truncated = build_market_breadth_features(daily[daily["date"].eq(first)], config)

    assert full.iloc[0].to_dict() == truncated.iloc[0].to_dict()


def test_breadth_confirmation_waits_until_positive_share_reaches_55pct() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=30)
    benchmark = _benchmark_features(calendar)
    benchmark.loc[
        benchmark["date"].between(calendar[5], calendar[11]),
        "breadth_positive_share_20d",
    ] = 0.50

    signals, _ = generate_monthly_confirmation_signals(
        _daily_features(calendar),
        pd.DataFrame(),
        _anchors(calendar[0]),
        benchmark,
        calendar,
        _config(minimum_breadth_constituents=1),
    )

    breadth_signal = signals[signals["rule"].eq("breadth_repair")].iloc[0]
    assert breadth_signal["signal_date"] == calendar[12]
