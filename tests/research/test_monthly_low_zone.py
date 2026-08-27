from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.monthly_low_zone import (
    MonthlyLowZoneConfig,
    build_monthly_weekly_features,
    evaluate_monthly_low_zone_events,
    generate_monthly_low_zone_signals,
)


def _daily_frame(dates: pd.DatetimeIndex, closes: list[float], symbol: str = "000001.SZ") -> pd.DataFrame:
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "ts_code": symbol,
            "date": dates,
            "open": close * 1.001,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "pre_close": np.r_[close[0], close[:-1]],
            "amount": 100_000.0,
            "adjusted_open": close * 1.001,
            "adjusted_high": close * 1.02,
            "adjusted_low": close * 0.98,
            "adjusted_close": close,
        }
    )


def _eligible_monthly(dates: pd.DatetimeIndex, monthly_j: list[float]) -> pd.DataFrame:
    periods = dates.to_period("M")
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "month_period": periods.astype(str),
            "signal_date": dates,
            "last_trade_date": dates,
            "adjusted_open": 40.0,
            "adjusted_high": 42.0,
            "adjusted_low": 38.0,
            "adjusted_close": 40.0,
            "median_daily_amount": 100_000.0,
            "history_months": np.arange(1, len(dates) + 1) + 40,
            "prior_peak": 100.0,
            "drawdown_from_prior_peak": -0.60,
            "signal_staleness_sessions": 0,
            "monthly_k": 10.0,
            "monthly_d": 20.0,
            "monthly_j": monthly_j,
            "monthly_low9_count": 0,
            "monthly_low9": False,
        }
    )


def _config(**overrides: object) -> MonthlyLowZoneConfig:
    values: dict[str, object] = {
        "minimum_history_months": 1,
        "minimum_drawdown_from_prior_peak": 0.01,
        "minimum_median_daily_amount_thousand": 0.0,
        "horizons": (2, 3, 4),
    }
    values.update(overrides)
    return MonthlyLowZoneConfig(**values)


def test_monthly_low9_is_nine_completed_months_below_four_month_lag() -> None:
    dates = pd.date_range("2020-01-31", periods=13, freq="ME")
    daily = _daily_frame(dates, [100.0 - index * 4.0 for index in range(13)])

    monthly, weekly = build_monthly_weekly_features(daily, pd.DatetimeIndex(dates))
    signals = generate_monthly_low_zone_signals(monthly, weekly, _config())
    low9 = signals[signals["rule"].eq("monthly_low9")]

    assert len(low9) == 1
    assert low9.iloc[0]["signal_date"] == dates[-1]
    assert low9.iloc[0]["monthly_low9_count"] == 9


def test_future_month_rows_do_not_change_prior_month_signal() -> None:
    dates = pd.date_range("2018-01-31", periods=50, freq="ME")
    closes = 100.0 - np.arange(50) * 0.8
    daily = _daily_frame(dates, closes.tolist())
    target = dates[-2]

    full, _ = build_monthly_weekly_features(daily, pd.DatetimeIndex(dates))
    truncated, _ = build_monthly_weekly_features(
        daily[daily["date"].le(target)], pd.DatetimeIndex(dates[dates <= target])
    )
    left = full[full["signal_date"].eq(target)].iloc[0]
    right = truncated[truncated["signal_date"].eq(target)].iloc[0]

    for column in ("monthly_j", "prior_peak", "drawdown_from_prior_peak", "monthly_low9_count"):
        assert left[column] == right[column]


def test_midweek_month_end_cannot_see_incomplete_week() -> None:
    signal_date = pd.Timestamp("2024-01-31")  # Wednesday
    monthly = _eligible_monthly(pd.DatetimeIndex([signal_date]), [-15.0])
    weekly = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "weekly_available_date": pd.to_datetime(["2024-01-26", "2024-02-02"]),
            "weekly_j": [-12.0, 50.0],
            "weekly_prev_j": [-8.0, -12.0],
        }
    )

    signals = generate_monthly_low_zone_signals(monthly, weekly, _config())
    row = signals[signals["rule"].eq("monthly_j_le_minus10")].iloc[0]

    assert row["weekly_available_date"] == pd.Timestamp("2024-01-26")
    assert row["weekly_j"] == -12.0


def test_monthly_j_threshold_triggers_on_state_onset_with_twelve_month_cooldown() -> None:
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    values = [-5.0] * 24
    values[1:4] = [-15.0] * 3
    values[7:9] = [-15.0] * 2
    values[14:16] = [-15.0] * 2
    monthly = _eligible_monthly(dates, values)

    signals = generate_monthly_low_zone_signals(monthly, pd.DataFrame(), _config())
    selected = signals[signals["rule"].eq("monthly_j_le_minus10")]

    assert selected["signal_date"].tolist() == [dates[1], dates[14]]


def test_weekly_reclaim_requires_a_completed_cross_within_signal_month() -> None:
    signal_date = pd.Timestamp("2024-01-31")
    monthly = _eligible_monthly(pd.DatetimeIndex([signal_date]), [-15.0])
    weekly = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "weekly_available_date": pd.to_datetime(["2023-12-29", "2024-01-12", "2024-02-02"]),
            "weekly_j": [-15.0, -5.0, 10.0],
            "weekly_prev_j": [-8.0, -15.0, -5.0],
        }
    )

    signals = generate_monthly_low_zone_signals(monthly, weekly, _config())

    assert "monthly_j_le_minus10_weekly_reclaim" in set(signals["rule"])


def test_drawdown_uses_only_prior_completed_month_peak() -> None:
    dates = pd.date_range("2022-01-31", periods=4, freq="ME")
    daily = _daily_frame(dates, [100.0, 80.0, 50.0, 200.0])

    monthly, _ = build_monthly_weekly_features(daily, pd.DatetimeIndex(dates))

    march = monthly[monthly["signal_date"].eq(dates[2])].iloc[0]
    april = monthly[monthly["signal_date"].eq(dates[3])].iloc[0]
    assert march["prior_peak"] == monthly.iloc[0]["adjusted_high"]
    assert march["drawdown_from_prior_peak"] < -0.45
    assert april["prior_peak"] == monthly.iloc[0]["adjusted_high"]


def test_entry_uses_next_open_and_rejects_long_suspension_or_one_price_bar() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=20)
    benchmark = pd.DataFrame({"trade_date": calendar, "open": 100.0, "close": 100.0})
    signals = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "signal_date": [calendar[0]] * 3,
            "rule": ["monthly_low9"] * 3,
        }
    )
    accepted = _daily_frame(pd.DatetimeIndex([calendar[1], calendar[2], calendar[3], calendar[4]]), [10, 11, 12, 13])
    one_price = _daily_frame(pd.DatetimeIndex([calendar[1], calendar[2], calendar[3], calendar[4]]), [10, 11, 12, 13], "000002.SZ")
    one_price.loc[one_price.index[0], ["open", "high", "low"]] = 10.0
    delayed = _daily_frame(pd.DatetimeIndex([calendar[8], calendar[9], calendar[10], calendar[11]]), [10, 11, 12, 13], "000003.SZ")

    events = evaluate_monthly_low_zone_events(
        pd.concat([accepted, one_price, delayed], ignore_index=True),
        signals,
        benchmark,
        pd.DatetimeIndex(calendar),
        _config(),
    )

    status = events.groupby("ts_code")["entry_status"].first().to_dict()
    assert status == {
        "000001.SZ": "accepted",
        "000002.SZ": "one_price_entry",
        "000003.SZ": "entry_delay_exceeded",
    }
    first = events[(events["ts_code"].eq("000001.SZ")) & events["horizon"].eq(2)].iloc[0]
    assert first["entry_date"] == calendar[1]
    assert first["entry_open"] == accepted.iloc[0]["adjusted_open"]


def test_unresolved_horizon_is_excluded_and_sixty_session_disappearance_is_written_off() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=100)
    benchmark = pd.DataFrame({"trade_date": calendar, "open": 100.0, "close": 100.0})
    daily = _daily_frame(pd.DatetimeIndex([calendar[1], calendar[2]]), [10.0, 10.5])
    signals = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "signal_date": [calendar[0]],
            "rule": ["monthly_low9"],
        }
    )
    config = _config(horizons=(70, 150, 200), maximum_missing_market_sessions=60)

    events = evaluate_monthly_low_zone_events(
        daily,
        signals,
        benchmark,
        pd.DatetimeIndex(calendar),
        config,
    )
    writeoff = events[events["horizon"].eq(70)].iloc[0]
    unresolved = events[events["horizon"].eq(150)].iloc[0]

    assert bool(writeoff["outcome_completed"])
    assert writeoff["exit_reason"] == "missing_bar_writeoff"
    assert writeoff["net_return"] == -1.0
    assert not bool(unresolved["outcome_completed"])
    assert unresolved["exit_reason"] == "unresolved_at_cutoff"


def test_target_date_suspension_exits_at_first_recovery_open() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=20)
    benchmark = pd.DataFrame(
        {"trade_date": calendar, "open": np.arange(100.0, 120.0), "close": 100.0}
    )
    daily = _daily_frame(
        pd.DatetimeIndex([calendar[1], calendar[2], calendar[6], calendar[7]]),
        [10.0, 10.5, 12.0, 12.5],
    )
    signals = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "signal_date": [calendar[0]],
            "rule": ["monthly_low9"],
        }
    )

    events = evaluate_monthly_low_zone_events(
        daily,
        signals,
        benchmark,
        pd.DatetimeIndex(calendar),
        _config(horizons=(3, 4, 5)),
    )
    delayed = events[events["horizon"].eq(3)].iloc[0]

    assert bool(delayed["outcome_completed"])
    assert delayed["target_date"] == calendar[3]
    assert delayed["exit_date"] == calendar[6]
    assert delayed["exit_reason"] == "next_open_after_target_suspension"
    entry_open = daily.loc[daily["date"].eq(calendar[1]), "adjusted_open"].iloc[0]
    recovery_open = daily.loc[
        daily["date"].eq(calendar[6]), "adjusted_open"
    ].iloc[0]
    assert delayed["gross_return"] == pytest.approx(recovery_open / entry_open - 1.0)
