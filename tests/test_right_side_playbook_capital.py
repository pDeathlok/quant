from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.right_side_playbook_capital import (
    CapitalBacktestSpec,
    build_continuous_daily_marks,
    prepare_capital_candidates,
    simulate_capital_constrained_policy,
)


def _candidate(
    event_id: str,
    *,
    symbol: str,
    entry: str,
    exit_date: str,
    score: float,
    gross: float = 0.0015,
    raw_price: float = 1.0,
) -> dict[str, object]:
    return {
        "arm": "shared_playbook_model",
        "fold": "B",
        "event_id": event_id,
        "symbol": symbol,
        "playbook_id": "next_open__expiry_t3_close",
        "entry_mode": "next_open",
        "first_layer_score": score,
        "entry_date": pd.Timestamp(entry),
        "exit_date": pd.Timestamp(exit_date),
        "entry_raw_price": raw_price,
        "gross_return": gross,
        "source_net_return": gross - 0.0015,
        "source_round_trip_cost": 0.0015,
        "capital_evaluable": True,
        "outcome_ambiguous_bar": False,
    }


def _marks(symbols: tuple[str, ...], dates: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "date": pd.Timestamp(date),
                "continuous_open": 1.0,
                "continuous_close": 1.0,
            }
            for symbol in symbols
            for date in dates
        ]
    )


def test_continuous_marks_remove_ex_date_price_discontinuity() -> None:
    daily = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "open": [10.0, 9.0],
            "close": [10.0, 9.0],
            "pre_close": [10.0, 9.0],
        }
    )
    marks = build_continuous_daily_marks(daily)

    assert marks["continuous_close"].iloc[0] == pytest.approx(10.0)
    assert marks["continuous_close"].iloc[1] == pytest.approx(10.0)


def test_capital_simulator_ranks_stably_and_reuses_only_completed_exit_cash() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                "lower",
                symbol="000002.SZ",
                entry="2025-01-02",
                exit_date="2025-01-03",
                score=0.8,
            ),
            _candidate(
                "higher",
                symbol="000001.SZ",
                entry="2025-01-02",
                exit_date="2025-01-03",
                score=0.9,
                gross=0.1015,
            ),
            _candidate(
                "after_exit",
                symbol="000003.SZ",
                entry="2025-01-03",
                exit_date="2025-01-06",
                score=0.7,
            ),
        ]
    )
    spec = CapitalBacktestSpec(
        scenario_id="test",
        initial_capital=1_000.0,
        target_position_cash=500.0,
        max_concurrent_positions=1,
        max_new_positions_per_session=1,
        board_lot_size=100,
    )
    result = simulate_capital_constrained_policy(
        candidates,
        _marks(
            ("000001.SZ", "000002.SZ", "000003.SZ"),
            ("2025-01-02", "2025-01-03", "2025-01-06"),
        ),
        arm="shared_playbook_model",
        spec=spec,
    )
    orders = result.orders.set_index("event_id")

    assert orders.loc["higher", "status"] == "executed"
    assert orders.loc["lower", "reason"] == "daily_new_position_limit"
    assert orders.loc["after_exit", "status"] == "executed"
    assert result.metrics["executed_trades"] == 2
    assert result.metrics["final_equity"] == pytest.approx(1_050.0)
    assert result.metrics["minimum_cash"] >= 0.0
    assert result.metrics["no_leverage_cash_non_negative"]
    assert result.metrics["final_open_positions"] == 0
    assert result.metrics["accounting_reconciliation_pass"]
    assert result.curve["open_positions"].max() == 1


def test_board_lot_rejection_and_missing_price_equal_slot_fallback_are_explicit() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                "too_expensive",
                symbol="000001.SZ",
                entry="2025-01-02",
                exit_date="2025-01-03",
                score=0.9,
                raw_price=6.0,
            ),
            _candidate(
                "fallback",
                symbol="000002.SZ",
                entry="2025-01-02",
                exit_date="2025-01-03",
                score=0.8,
                raw_price=np.nan,
            ),
        ]
    )
    spec = CapitalBacktestSpec(
        scenario_id="test",
        initial_capital=1_000.0,
        target_position_cash=500.0,
        max_concurrent_positions=2,
        max_new_positions_per_session=2,
        board_lot_size=100,
    )
    result = simulate_capital_constrained_policy(
        candidates,
        _marks(
            ("000001.SZ", "000002.SZ"),
            ("2025-01-02", "2025-01-03"),
        ),
        arm="shared_playbook_model",
        spec=spec,
    )
    orders = result.orders.set_index("event_id")

    assert orders.loc["too_expensive", "reason"] == "insufficient_cash_or_board_lot"
    assert orders.loc["fallback", "status"] == "executed"
    assert orders.loc["fallback", "allocation_mode"] == "equal_slot_price_fallback"
    assert result.metrics["equal_slot_price_fallback_allocations"] == 1


def test_prepare_candidates_preserves_frozen_score_outcome_and_rejects_c() -> None:
    selections = pd.DataFrame(
        {
            "arm": ["shared_playbook_model"],
            "fold": ["B"],
            "event_id": ["e1"],
            "symbol": ["000001.SZ"],
            "date": pd.to_datetime(["2025-01-02"]),
            "playbook_id": ["next_open__expiry_t3_close"],
            "entry_mode": ["next_open"],
            "eligible": [True],
            "mature": [True],
            "entry_date": pd.to_datetime(["2025-01-03"]),
            "exit_date": pd.to_datetime(["2025-01-06"]),
            "net_return": [0.01],
            "round_trip_cost": [0.0015],
            "ambiguous_bar": [False],
        }
    )
    events = pd.DataFrame(
        {"fold": ["B"], "event_id": ["e1"], "first_layer_score": [0.75]}
    )
    outcomes = pd.DataFrame(
        {
            "fold": ["B"],
            "event_id": ["e1"],
            "playbook_id": ["next_open__expiry_t3_close"],
            "entry_raw_price": [10.0],
            "exit_raw_price": [10.1],
            "gross_return": [0.0115],
            "net_return": [0.01],
            "round_trip_cost": [0.0015],
            "entry_date": pd.to_datetime(["2025-01-03"]),
            "exit_date": pd.to_datetime(["2025-01-06"]),
            "ambiguous_bar": [False],
        }
    )
    prepared = prepare_capital_candidates(selections, events, outcomes)

    assert prepared.loc[0, "first_layer_score"] == pytest.approx(0.75)
    assert prepared.loc[0, "capital_evaluable"]
    contaminated = selections.copy()
    contaminated["fold"] = "C"
    with pytest.raises(ValueError, match="B only"):
        prepare_capital_candidates(contaminated, events, outcomes)
