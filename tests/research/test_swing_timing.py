from __future__ import annotations

import pandas as pd
import pytest

from quant.backtest import AShareExecutionConfig
from quant.research.swing_timing import (
    SwingExitRule,
    apply_same_symbol_cooldown,
    select_rule_without_holdout,
    simulate_swing_exits,
    summarize_swing_trades,
    wilson_lower_bound,
)


def _path_frame(
    bars: list[tuple[float, float, float, float]],
) -> pd.DataFrame:
    row: dict[str, object] = {
        "date": pd.Timestamp("2026-01-02"),
        "ts_code": "000001.SZ",
        "close": 10.0,
        "entry_open": bars[0][0],
    }
    dates = pd.bdate_range("2026-01-05", periods=len(bars))
    for day, ((open_, high, low, close), date) in enumerate(
        zip(bars, dates),
        start=1,
    ):
        row[f"date_t{day}"] = date
        row[f"open_t{day}"] = open_
        row[f"high_t{day}"] = high
        row[f"low_t{day}"] = low
        row[f"close_t{day}"] = close
    return pd.DataFrame([row])


def _zero_cost_execution() -> AShareExecutionConfig:
    return AShareExecutionConfig(
        commission_rate=0,
        stamp_tax_rate=0,
        transfer_fee_rate=0,
        min_commission=0,
        slippage=0,
    )


def test_t_plus_one_ignores_entry_day_barriers() -> None:
    frame = _path_frame(
        [
            (10.0, 11.0, 9.0, 10.0),
            (10.0, 10.7, 9.9, 10.6),
        ]
    )
    rule = SwingExitRule(
        "t1",
        take_profit=0.05,
        stop_loss=0.05,
        hold_days=1,
        exit_grace_days=0,
    )

    result = simulate_swing_exits(
        frame,
        rule,
        execution=_zero_cost_execution(),
    )

    assert result.iloc[0]["exit_type"] == "take_profit"
    assert result.iloc[0]["exit_day"] == 2
    assert result.iloc[0]["net_return"] == pytest.approx(0.05)


def test_same_day_stop_and_target_assumes_stop_first() -> None:
    frame = _path_frame(
        [
            (10.0, 10.1, 9.9, 10.0),
            (10.0, 10.6, 9.4, 10.1),
        ]
    )
    rule = SwingExitRule(
        "collision",
        take_profit=0.05,
        stop_loss=0.05,
        hold_days=1,
        exit_grace_days=0,
    )

    result = simulate_swing_exits(
        frame,
        rule,
        execution=_zero_cost_execution(),
    )

    assert result.iloc[0]["exit_type"] == "stop_loss"
    assert result.iloc[0]["net_return"] == pytest.approx(-0.05)


def test_gap_through_stop_fills_at_open() -> None:
    frame = _path_frame(
        [
            (10.0, 10.1, 9.9, 10.0),
            (9.2, 9.4, 9.1, 9.3),
        ]
    )
    rule = SwingExitRule(
        "gap",
        take_profit=0.05,
        stop_loss=0.05,
        hold_days=1,
        exit_grace_days=0,
    )

    result = simulate_swing_exits(
        frame,
        rule,
        execution=_zero_cost_execution(),
    )

    assert result.iloc[0]["exit_raw"] == pytest.approx(9.2)
    assert result.iloc[0]["net_return"] == pytest.approx(-0.08)


def test_one_price_limit_down_delays_expiry_exit() -> None:
    frame = _path_frame(
        [
            (10.0, 10.1, 9.9, 10.0),
            (9.5, 9.5, 9.5, 9.5),
            (9.4, 9.6, 9.3, 9.5),
        ]
    )
    rule = SwingExitRule(
        "delay",
        take_profit=0.20,
        stop_loss=0.20,
        hold_days=1,
        exit_grace_days=1,
    )

    result = simulate_swing_exits(
        frame,
        rule,
        execution=_zero_cost_execution(),
    )

    assert result.iloc[0]["exit_type"] == "delayed_expiry"
    assert result.iloc[0]["exit_day"] == 3
    assert result.iloc[0]["exit_raw"] == pytest.approx(9.4)


def test_same_symbol_cooldown_is_independent_by_symbol() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["A", "A", "B", "A"],
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-20", "2026-01-20", "2026-02-26"]
            ),
        }
    )

    result = apply_same_symbol_cooldown(frame, 56)

    assert list(zip(result["ts_code"], result["date"].dt.day)) == [
        ("A", 1),
        ("B", 20),
        ("A", 26),
    ]


def test_wilson_bound_penalizes_small_samples() -> None:
    assert wilson_lower_bound(6, 10) < wilson_lower_bound(60, 100)


def test_execution_costs_reduce_target_return() -> None:
    frame = _path_frame(
        [
            (10.0, 10.1, 9.9, 10.0),
            (10.0, 10.7, 9.9, 10.6),
        ]
    )
    rule = SwingExitRule(
        "costs",
        take_profit=0.05,
        stop_loss=0.05,
        hold_days=1,
        exit_grace_days=0,
    )

    result = simulate_swing_exits(
        frame,
        rule,
        execution=AShareExecutionConfig(slippage=0),
    )

    assert result.iloc[0]["gross_return"] == pytest.approx(0.05)
    assert 0 < result.iloc[0]["net_return"] < 0.05


def test_one_year_rule_expires_after_244_sellable_sessions() -> None:
    bars = [(10.0, 10.1, 9.9, 10.0)] * 245
    frame = _path_frame(bars)
    frame["entry_date"] = frame["date_t1"]
    rule = SwingExitRule(
        "one_year",
        take_profit=0.20,
        stop_loss=0.10,
        hold_days=244,
        exit_grace_days=0,
    )

    result = simulate_swing_exits(
        frame,
        rule,
        execution=_zero_cost_execution(),
        result_columns=["date", "entry_date", "ts_code"],
    )

    assert result.iloc[0]["exit_type"] == "expiry"
    assert result.iloc[0]["exit_day"] == 245
    assert "open_t244" not in result.columns

    summary = summarize_swing_trades(result)
    assert summary["average_holding_sessions"] == pytest.approx(244)


def test_selection_uses_only_named_period() -> None:
    summary = pd.DataFrame(
        [
            {
                "rule_id": "A",
                "period": "selection_2020_2023",
                "trades": 100,
                "avg_net_return": 0.01,
                "profit_factor": 1.4,
                "positive_year_share": 1.0,
                "win_rate_wilson_lower_95": 0.55,
            },
            {
                "rule_id": "B",
                "period": "selection_2020_2023",
                "trades": 100,
                "avg_net_return": 0.01,
                "profit_factor": 1.4,
                "positive_year_share": 1.0,
                "win_rate_wilson_lower_95": 0.50,
            },
            {
                "rule_id": "B",
                "period": "reused_2024_2025",
                "trades": 100,
                "avg_net_return": 0.20,
                "profit_factor": 9.0,
                "positive_year_share": 1.0,
                "win_rate_wilson_lower_95": 0.90,
            },
        ]
    )

    selected, passed = select_rule_without_holdout(summary)

    assert passed is True
    assert selected["rule_id"] == "A"
