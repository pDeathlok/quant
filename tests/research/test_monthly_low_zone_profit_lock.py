from __future__ import annotations

import pandas as pd
import pytest

from quant.research.monthly_low_zone_profit_lock import (
    ProfitLockConfig,
    ProfitLockPortfolioConfig,
    assemble_staged_anchor_events,
    evaluate_profit_lock_events,
    simulate_profit_lock_portfolio,
    summarize_profit_lock_events,
)


def _daily(dates: pd.DatetimeIndex, highs: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "date": dates,
            "adjusted_high": highs,
        }
    )


def _baseline(
    calendar: pd.DatetimeIndex,
    *,
    net_return: float = 0.20,
    exit_reason: str = "time_exit",
    outcome_completed: bool = True,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_id": [11],
            "anchor_id": [7],
            "ts_code": ["000001.SZ"],
            "rule": ["anchor_direct"],
            "signal_date": [calendar[0]],
            "horizon": [5],
            "entry_status": ["accepted"],
            "entry_date": [calendar[1]],
            "entry_open": [10.0],
            "target_date": [calendar[5]],
            "exit_date": [calendar[5]],
            "exit_reason": [exit_reason],
            "outcome_completed": [outcome_completed],
            "gross_return": [net_return + 0.002 if net_return > -1.0 else -1.0],
            "net_return": [net_return],
            "benchmark_return": [0.05],
            "excess_net_return": [net_return - 0.05],
            "mae": [-0.10],
            "mfe": [0.30],
        }
    )


def _config(**overrides: object) -> ProfitLockConfig:
    values: dict[str, object] = {
        "horizon_sessions": 5,
        "target_returns": (0.10,),
    }
    values.update(overrides)
    return ProfitLockConfig(**values)


def test_take_profit_fills_at_frozen_threshold_not_daily_high() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=8)
    events = evaluate_profit_lock_events(
        _daily(calendar[1:6], [10.2, 11.4, 12.0, 12.5, 13.0]),
        _baseline(calendar),
        calendar,
        _config(),
    )

    event = events.iloc[0]
    assert event["exit_reason"] == "take_profit"
    assert event["exit_date"] == calendar[2]
    assert event["gross_return"] == pytest.approx(0.10)
    assert event["net_return"] == pytest.approx(0.098)
    assert event["holding_sessions"] == 2


def test_price_after_baseline_exit_cannot_trigger_take_profit() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=8)
    baseline = _baseline(calendar)
    baseline["exit_date"] = calendar[3]
    baseline["target_date"] = calendar[3]
    daily = _daily(calendar[1:7], [10.1, 10.2, 10.3, 12.0, 12.0, 12.0])

    event = evaluate_profit_lock_events(
        daily, baseline, calendar, _config()
    ).iloc[0]

    assert event["exit_reason"] == "time_exit"
    assert event["net_return"] == pytest.approx(0.20)
    assert event["exit_date"] == calendar[3]


def test_take_profit_before_later_writeoff_is_a_completed_exit() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=8)
    baseline = _baseline(calendar, net_return=-1.0, exit_reason="missing_bar_writeoff")
    event = evaluate_profit_lock_events(
        _daily(calendar[1:3], [10.2, 11.0]),
        baseline,
        calendar,
        _config(),
    ).iloc[0]

    assert event["exit_reason"] == "take_profit"
    assert event["net_return"] == pytest.approx(0.098)
    assert event["baseline_exit_reason"] == "missing_bar_writeoff"


def test_never_hit_target_inherits_baseline_writeoff() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=8)
    baseline = _baseline(calendar, net_return=-1.0, exit_reason="missing_bar_writeoff")
    event = evaluate_profit_lock_events(
        _daily(calendar[1:3], [10.2, 10.5]),
        baseline,
        calendar,
        _config(),
    ).iloc[0]

    assert event["exit_reason"] == "missing_bar_writeoff"
    assert event["net_return"] == -1.0
    assert not event["target_hit"]


def test_incomplete_baseline_remains_unresolved_even_if_later_data_exists() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=8)
    baseline = _baseline(calendar, outcome_completed=False)
    event = evaluate_profit_lock_events(
        _daily(calendar[1:7], [12.0] * 6),
        baseline,
        calendar,
        _config(),
    ).iloc[0]

    assert not event["outcome_completed"]
    assert pd.isna(event["net_return"])


def test_portfolio_mode_can_resolve_observed_target_before_cutoff() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=8)
    baseline = _baseline(calendar, outcome_completed=False)
    event = evaluate_profit_lock_events(
        _daily(calendar[1:7], [10.2, 11.0, 12.0, 12.0, 12.0, 12.0]),
        baseline,
        calendar,
        _config(),
        resolve_observed_targets=True,
    ).iloc[0]

    assert event["target_hit"]
    assert event["outcome_completed"]
    assert not event["baseline_outcome_completed"]


def _profit_events(
    calendar: pd.DatetimeIndex,
    *,
    confirmation_date: pd.Timestamp,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_id": [1, 2],
            "anchor_id": [99, 99],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "rule": ["anchor_direct", "no_new_low_20"],
            "signal_date": [calendar[0], confirmation_date],
            "target_return": [0.10, 0.10],
            "horizon": [504, 504],
            "entry_status": ["accepted", "accepted"],
            "outcome_completed": [True, True],
            "entry_date": [calendar[1], confirmation_date],
            "exit_date": [calendar[3], calendar[6]],
            "exit_reason": ["take_profit", "take_profit"],
            "net_return": [0.098, 0.198],
            "holding_sessions": [3, 4],
        }
    )


def test_confirmation_after_probe_exit_does_not_reopen_position() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=10)
    staged = assemble_staged_anchor_events(
        _profit_events(calendar, confirmation_date=calendar[4]),
        add_rule="no_new_low_20",
    ).iloc[0]

    assert staged["add_status"] == "confirmation_after_probe_exit"
    assert staged["build_fraction"] == pytest.approx(0.25)
    assert staged["budget_return"] == pytest.approx(0.25 * 0.098)
    assert staged["committed_return"] == pytest.approx(0.098)


def test_confirmation_before_probe_exit_adds_second_quarter() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=10)
    staged = assemble_staged_anchor_events(
        _profit_events(calendar, confirmation_date=calendar[2]),
        add_rule="no_new_low_20",
    ).iloc[0]

    assert staged["add_status"] == "added_before_probe_exit"
    assert staged["build_fraction"] == pytest.approx(0.50)
    assert staged["budget_return"] == pytest.approx(0.25 * 0.098 + 0.25 * 0.198)
    assert staged["committed_return"] == pytest.approx((0.098 + 0.198) / 2)


def test_summary_labels_2025_as_time_out_diagnostic() -> None:
    calendar = pd.bdate_range("2025-01-02", periods=8)
    events = evaluate_profit_lock_events(
        _daily(calendar[1:6], [10.2, 11.4, 12.0, 12.5, 13.0]),
        _baseline(calendar),
        calendar,
        _config(),
    )
    events["signal_date"] = pd.Timestamp("2025-01-31")

    summary = summarize_profit_lock_events(events)

    assert summary.iloc[0]["period"] == "time_out_diagnostic_2025"
    assert summary.iloc[0]["win_rate"] == 1.0


def _portfolio_event(
    calendar: pd.DatetimeIndex,
    *,
    anchor_id: int,
    symbol: str,
    amount: float,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp | pd.NaT,
    rule: str = "anchor_direct",
    net_return: float = 0.098,
    completed: bool = True,
) -> dict[str, object]:
    return {
        "signal_id": anchor_id * 10 + (0 if rule == "anchor_direct" else 1),
        "anchor_id": anchor_id,
        "ts_code": symbol,
        "rule": rule,
        "signal_date": calendar[0] if rule == "anchor_direct" else calendar[1],
        "median_daily_amount": amount,
        "target_return": 0.10,
        "horizon": 504,
        "entry_status": "accepted",
        "outcome_completed": completed,
        "entry_date": entry_date,
        "entry_open": 10.0,
        "exit_date": exit_date,
        "exit_reason": "take_profit" if completed else "unresolved_at_cutoff",
        "net_return": net_return if completed else float("nan"),
    }


def _portfolio_daily(
    calendar: pd.DatetimeIndex,
    symbols: tuple[str, ...] = ("A",),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        for date in calendar:
            rows.append(
                {
                    "ts_code": symbol,
                    "date": date,
                    "adjusted_close": 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_portfolio_counts_idle_cash_and_realized_probe_return() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=6)
    events = pd.DataFrame(
        [
            _portfolio_event(
                calendar,
                anchor_id=1,
                symbol="A",
                amount=100.0,
                entry_date=calendar[1],
                exit_date=calendar[3],
            )
        ]
    )
    curve, trades, _ = simulate_profit_lock_portfolio(
        _portfolio_daily(calendar),
        events,
        calendar,
        start_date=calendar[0],
        end_date=calendar[-1],
        target_return=0.10,
        add_rule=None,
        config=ProfitLockPortfolioConfig(
            initial_cash=1_000.0,
            maximum_anchors=1,
            target_anchor_fraction=1.0,
            probe_budget_fraction=0.25,
            add_budget_fraction=0.25,
        ),
    )

    assert trades.iloc[0]["allocated_cash"] == pytest.approx(250.0)
    assert curve.iloc[-1]["nav"] == pytest.approx(1_024.5)
    assert curve.iloc[-1]["cash"] == pytest.approx(1_024.5)
    assert curve.iloc[-1]["active_anchors"] == 0


def test_capacity_uses_same_day_liquidity_without_future_return() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=6)
    events = pd.DataFrame(
        [
            _portfolio_event(
                calendar,
                anchor_id=1,
                symbol="A",
                amount=100.0,
                entry_date=calendar[1],
                exit_date=pd.NaT,
                completed=False,
            ),
            _portfolio_event(
                calendar,
                anchor_id=2,
                symbol="B",
                amount=200.0,
                entry_date=calendar[1],
                exit_date=pd.NaT,
                completed=False,
            ),
        ]
    )
    _, trades, audit = simulate_profit_lock_portfolio(
        _portfolio_daily(calendar, ("A", "B")),
        events,
        calendar,
        start_date=calendar[0],
        end_date=calendar[-1],
        target_return=0.10,
        add_rule=None,
        config=ProfitLockPortfolioConfig(maximum_anchors=1),
    )

    assert trades.loc[trades["action"].eq("entry"), "ts_code"].tolist() == ["B"]
    assert audit["skipped_capacity"] == 1


def test_portfolio_adds_only_while_probe_anchor_is_active() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=8)
    early = pd.DataFrame(
        [
            _portfolio_event(
                calendar,
                anchor_id=1,
                symbol="A",
                amount=100.0,
                entry_date=calendar[1],
                exit_date=calendar[4],
            ),
            _portfolio_event(
                calendar,
                anchor_id=1,
                symbol="A",
                amount=100.0,
                entry_date=calendar[2],
                exit_date=calendar[5],
                rule="no_new_low_20",
            ),
        ]
    )
    _, early_trades, _ = simulate_profit_lock_portfolio(
        _portfolio_daily(calendar),
        early,
        calendar,
        start_date=calendar[0],
        end_date=calendar[-1],
        target_return=0.10,
        add_rule="no_new_low_20",
        config=ProfitLockPortfolioConfig(),
    )
    late = early.copy()
    late.loc[late["rule"].eq("no_new_low_20"), "entry_date"] = calendar[5]
    _, late_trades, late_audit = simulate_profit_lock_portfolio(
        _portfolio_daily(calendar),
        late,
        calendar,
        start_date=calendar[0],
        end_date=calendar[-1],
        target_return=0.10,
        add_rule="no_new_low_20",
        config=ProfitLockPortfolioConfig(),
    )

    assert early_trades["action"].tolist().count("add") == 1
    assert late_trades["action"].tolist().count("add") == 0
    assert late_audit["skipped_inactive_confirmation"] == 1


def test_unresolved_position_is_marked_to_period_end_close() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=6)
    events = pd.DataFrame(
        [
            _portfolio_event(
                calendar,
                anchor_id=1,
                symbol="A",
                amount=100.0,
                entry_date=calendar[1],
                exit_date=pd.NaT,
                completed=False,
            )
        ]
    )
    daily = _portfolio_daily(calendar)
    daily.loc[daily["date"].eq(calendar[-1]), "adjusted_close"] = 12.0
    curve, _, _ = simulate_profit_lock_portfolio(
        daily,
        events,
        calendar,
        start_date=calendar[0],
        end_date=calendar[-1],
        target_return=0.10,
        add_rule=None,
        config=ProfitLockPortfolioConfig(
            initial_cash=1_000.0,
            maximum_anchors=1,
            target_anchor_fraction=1.0,
            probe_budget_fraction=0.25,
            add_budget_fraction=0.25,
        ),
    )

    assert curve.iloc[-1]["nav"] == pytest.approx(1_050.0)
    assert curve.iloc[-1]["active_anchors"] == 1


def test_profit_lock_can_leave_a_runner_that_uses_baseline_exit() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=8)
    event = _portfolio_event(
        calendar,
        anchor_id=1,
        symbol="A",
        amount=100.0,
        entry_date=calendar[1],
        exit_date=calendar[3],
    )
    event.update(
        {
            "target_hit": True,
            "baseline_outcome_completed": True,
            "baseline_exit_date": calendar[5],
            "baseline_net_return": 0.50,
            "baseline_exit_reason": "time_exit",
        }
    )
    _, trades, _ = simulate_profit_lock_portfolio(
        _portfolio_daily(calendar),
        pd.DataFrame([event]),
        calendar,
        start_date=calendar[0],
        end_date=calendar[-1],
        target_return=0.10,
        add_rule=None,
        config=ProfitLockPortfolioConfig(
            initial_cash=1_000.0,
            maximum_anchors=1,
            target_anchor_fraction=1.0,
            probe_budget_fraction=0.25,
            add_budget_fraction=0.25,
            profit_lock_fraction=0.90,
        ),
    )

    assert trades["action"].tolist() == ["entry", "exit_core", "exit_runner"]
    exits = trades[trades["action"].str.startswith("exit")]
    assert exits.iloc[0]["allocated_cash"] == pytest.approx(225.0)
    assert exits.iloc[1]["allocated_cash"] == pytest.approx(25.0)
