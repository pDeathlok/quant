from __future__ import annotations

import pandas as pd
import pytest

from quant.research.monthly_low_zone_continuous import (
    ContinuousConfig,
    ContinuousPolicy,
    simulate_continuous_anchor,
    simulate_continuous_portfolio,
)


def _daily(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    close = [10.0] * len(calendar)
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "date": calendar,
            "open": close,
            "high": [10.2] * len(calendar),
            "low": [9.8] * len(calendar),
            "adjusted_open": close,
            "adjusted_high": [10.2] * len(calendar),
            "adjusted_low": [9.8] * len(calendar),
            "adjusted_close": close,
            "base_low": [9.0] * len(calendar),
            "base_position": [0.60] * len(calendar),
            "sessions_since_new_low": [30.0] * len(calendar),
            "return_20d": [0.05] * len(calendar),
            "prior_amount_median_20d": [50_000.0] * len(calendar),
            "prior_peak": [20.0] * len(calendar),
        }
    )


def _event(
    calendar: pd.DatetimeIndex,
    *,
    baseline_exit_reason: str = "time_exit",
) -> pd.Series:
    return pd.Series(
        {
            "signal_id": 11,
            "anchor_id": 7,
            "ts_code": "000001.SZ",
            "signal_date": calendar[0],
            "month_period": "2024-01",
            "source_sample": "test",
            "entry_status": "accepted",
            "entry_date": calendar[1],
            "entry_open": 10.0,
            "target_date": calendar[9],
            "baseline_exit_date": calendar[9],
            "baseline_exit_reason": baseline_exit_reason,
            "baseline_outcome_completed": True,
            "net_return": 0.098,
            "median_daily_amount": 50_000.0,
        }
    )


def _config(**overrides: object) -> ContinuousConfig:
    values: dict[str, object] = {
        "horizon_sessions": 9,
        "minimum_reentry_wait_sessions": 2,
    }
    values.update(overrides)
    return ContinuousConfig(**values)


def test_lump_sum_reproduces_fixed_take_profit_return() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=12)
    daily = _daily(calendar)
    daily.loc[4, ["high", "adjusted_high"]] = 11.6

    result, path, trades = simulate_continuous_anchor(
        daily,
        _event(calendar),
        calendar,
        ContinuousPolicy("lump", 1.0),
        _config(),
    )

    assert result["target_hit"]
    assert result["budget_return"] == pytest.approx(0.148)
    assert result["exit_date"] == calendar[4]
    assert path.iloc[-1]["state"] == "completed"
    assert list(trades["action"]) == ["buy", "sell"]


def test_unfilled_grid_cash_reduces_budget_return_without_fake_full_exposure() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=12)
    daily = _daily(calendar)
    daily.loc[3, ["high", "adjusted_high"]] = 11.6
    policy = ContinuousPolicy("grid", 0.40, (0.10, 0.20), (0.30, 0.30))

    result, path, _ = simulate_continuous_anchor(
        daily, _event(calendar), calendar, policy, _config()
    )

    assert result["grid_add_count"] == 0
    assert result["budget_return"] == pytest.approx(0.0592)
    assert result["maximum_invested_fraction"] < 0.45
    assert path.iloc[-1]["inner_cash"] == pytest.approx(1.0592)


def test_grid_adds_at_frozen_levels_and_uses_weighted_average_target() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=12)
    daily = _daily(calendar)
    daily.loc[2, ["low", "adjusted_low", "adjusted_close"]] = [8.9, 8.9, 9.2]
    daily.loc[3, ["low", "adjusted_low", "adjusted_close"]] = [7.9, 7.9, 8.2]
    daily.loc[4, ["high", "adjusted_high", "adjusted_close"]] = [10.5, 10.5, 10.3]
    policy = ContinuousPolicy("grid", 0.40, (0.10, 0.20), (0.30, 0.30))

    result, _, trades = simulate_continuous_anchor(
        daily, _event(calendar), calendar, policy, _config()
    )

    buys = trades.loc[trades["action"].eq("buy")]
    assert list(buys["price"]) == pytest.approx([10.0, 9.0, 8.0])
    assert result["grid_add_count"] == 2
    assert result["budget_return"] == pytest.approx(0.148)
    assert result["exit_date"] == calendar[4]


def test_grid_cannot_take_profit_on_same_day_as_new_add() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=12)
    daily = _daily(calendar)
    daily.loc[2, ["high", "low", "adjusted_high", "adjusted_low"]] = [
        11.6,
        8.9,
        11.6,
        8.9,
    ]
    daily.loc[3, ["high", "adjusted_high"]] = 11.6
    policy = ContinuousPolicy("grid", 0.40, (0.10, 0.20), (0.30, 0.30))

    result, _, _ = simulate_continuous_anchor(
        daily, _event(calendar), calendar, policy, _config()
    )

    assert result["grid_add_count"] == 1
    assert result["exit_date"] == calendar[3]


def test_structural_stop_exits_next_open_then_waits_for_causal_reentry() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=14)
    daily = _daily(calendar)
    daily.loc[2, ["close", "adjusted_close", "low", "adjusted_low"]] = [
        8.8,
        8.8,
        8.7,
        8.7,
    ]
    daily.loc[3, ["open", "adjusted_open", "high", "low"]] = [8.7, 8.7, 8.9, 8.6]
    daily.loc[3, ["adjusted_high", "adjusted_low", "adjusted_close"]] = [8.9, 8.6, 8.8]
    daily.loc[4, ["sessions_since_new_low", "return_20d"]] = [10.0, -0.02]
    daily.loc[5, ["open", "adjusted_open", "high", "adjusted_high"]] = [
        9.0,
        9.0,
        10.5,
        10.5,
    ]
    event = _event(calendar)
    event["target_date"] = calendar[11]
    event["baseline_exit_date"] = calendar[11]
    policy = ContinuousPolicy(
        "stop_reentry", 1.0, structural_stop=True, maximum_reentries=2
    )

    result, _, trades = simulate_continuous_anchor(
        daily, event, calendar, policy, _config()
    )

    stop_sale = trades.loc[trades["trade_label"].eq("structural_stop")].iloc[0]
    reentry = trades.loc[trades["trade_label"].eq("reentry_initial")].iloc[0]
    assert stop_sale["date"] == calendar[3]
    assert reentry["date"] == calendar[6]
    assert result["stop_count"] == 1
    assert result["reentries"] == 1


def test_missing_bar_writeoff_preserves_total_loss_for_fully_invested_policy() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=12)
    daily = _daily(calendar).iloc[:5].copy()
    event = _event(calendar, baseline_exit_reason="missing_bar_writeoff")

    result, _, _ = simulate_continuous_anchor(
        daily,
        event,
        calendar,
        ContinuousPolicy("lump", 1.0),
        _config(),
    )

    assert result["exit_reason"] == "missing_bar_writeoff"
    assert result["budget_return"] == -1.0


def test_reentry_policy_stops_after_two_reentries() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=24)
    daily = _daily(calendar)
    for position in range(2, 24):
        daily.loc[position, ["open", "adjusted_open"]] = 8.8
        daily.loc[position, ["high", "adjusted_high"]] = 9.0
        daily.loc[position, ["low", "adjusted_low"]] = 8.2
        daily.loc[position, ["close", "adjusted_close"]] = 8.7
    daily.loc[2, ["close", "adjusted_close"]] = 8.8
    daily.loc[3, ["open", "adjusted_open"]] = 8.7
    daily.loc[5, "base_low"] = 8.5
    daily.loc[6, ["open", "adjusted_open"]] = 8.8
    daily.loc[7, ["close", "adjusted_close"]] = 8.4
    daily.loc[8, ["open", "adjusted_open"]] = 8.3
    daily.loc[10, "base_low"] = 8.0
    daily.loc[11, ["open", "adjusted_open"]] = 8.2
    daily.loc[12, ["close", "adjusted_close"]] = 7.9
    daily.loc[13, ["open", "adjusted_open"]] = 7.8
    event = _event(calendar)
    event["target_date"] = calendar[20]
    event["baseline_exit_date"] = calendar[20]
    policy = ContinuousPolicy(
        "stop_reentry", 1.0, structural_stop=True, maximum_reentries=2
    )

    result, _, trades = simulate_continuous_anchor(
        daily, event, calendar, policy, _config()
    )

    assert result["reentries"] == 2
    assert result["stop_count"] == 3
    assert result["exit_reason"] == "structural_stop_final"
    assert trades["trade_label"].eq("reentry_initial").sum() == 2


def test_portfolio_reports_reserved_grid_cash_separately_from_stock_exposure() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=12)
    daily = _daily(calendar)
    daily.loc[3, ["high", "adjusted_high"]] = 11.6
    policy = ContinuousPolicy("grid", 0.40, (0.10, 0.20), (0.30, 0.30))
    result, path, _ = simulate_continuous_anchor(
        daily, _event(calendar), calendar, policy, _config()
    )
    results = pd.DataFrame([result])

    curve, trades, audit = simulate_continuous_portfolio(
        path,
        results,
        calendar,
        policy="grid",
        start_date=calendar[0],
        end_date=calendar[0].to_period("M").to_timestamp(how="end"),
    )

    entered = curve.loc[curve["active_anchors"].eq(1)].iloc[0]
    assert entered["reserved_fraction"] > entered["invested_fraction"]
    assert audit["entered_anchors"] == 1
    assert set(trades["action"]) == {"enter_anchor", "exit_anchor"}
