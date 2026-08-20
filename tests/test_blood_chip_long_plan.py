from __future__ import annotations

import pandas as pd
import pytest

from quant.application.blood_chip_long_plan import (
    build_blood_chip_daily_iteration,
    build_blood_chip_long_plan,
    compose_blood_chip_long_plan,
)
from quant.research.blood_chip import BloodChipBacktestResult


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000002.SZ",
                "shock_event_id": 3,
                "shock_date": pd.Timestamp("2026-08-05"),
                "signal_date": pd.Timestamp("2026-08-10"),
                "entry_date": pd.NaT,
                "entry_open": float("nan"),
                "signal_close": 12.0,
                "signal_score": 0.88,
                "shock_score": 0.92,
                "absorption_score": 0.83,
                "rebound_from_event_low": 0.08,
                "volatility_60d": 0.42,
                "market_return_60d": 0.02,
                "return_120d": -0.18,
                "shock_volatility_expansion_ratio": 2.4,
                "confirmation_amount_vs_prior_ratio": 0.72,
            },
            {
                "ts_code": "000001.SZ",
                "shock_event_id": 5,
                "shock_date": pd.Timestamp("2026-08-04"),
                "signal_date": pd.Timestamp("2026-08-10"),
                "entry_date": pd.NaT,
                "entry_open": float("nan"),
                "signal_close": 10.0,
                "signal_score": 0.86,
                "shock_score": 0.90,
                "absorption_score": 0.81,
                "rebound_from_event_low": 0.06,
                "volatility_60d": 0.25,
                "market_return_60d": 0.02,
                "return_120d": -0.12,
                "shock_volatility_expansion_ratio": 2.0,
                "confirmation_amount_vs_prior_ratio": 0.65,
                "shock_kdj_state": "triple_oversold",
                "shock_kdj_daily_j": -8.0,
                "shock_kdj_weekly_j": -5.0,
                "shock_kdj_monthly_j": -2.0,
                "shock_kdj_negative_count": 3,
            },
        ]
    )


def _backtest_result() -> BloodChipBacktestResult:
    trades = pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "shock_event_id": 2,
                "signal_date": pd.Timestamp("2026-07-20"),
                "entry_date": pd.Timestamp("2026-07-21"),
                "exit_date": pd.Timestamp("2026-08-10"),
                "entry_fill": 9.0,
                "exit_fill": 9.8,
                "signal_close": 9.2,
                "stop_price": 8.1,
                "current_residual_return_3d": 0.025,
                "next_stage_ready": True,
                "tranches_filled": 2,
                "tranche_dates": "2026-07-21|2026-07-29",
                "planned_fractions": "0.2000|0.3000|0.5000",
                "deployed_fraction": 0.49,
                "scale_in_policy": "increasing_survival",
                "holding_sessions": 14,
                "exit_reason": "end_of_data",
                "net_return": 0.085,
                "reentry_number": 1,
            },
            {
                "ts_code": "000003.SZ",
                "shock_event_id": 1,
                "signal_date": pd.Timestamp("2026-08-01"),
                "entry_date": pd.Timestamp("2026-08-04"),
                "exit_date": pd.Timestamp("2026-08-10"),
                "entry_fill": 8.0,
                "exit_fill": 7.2,
                "holding_sessions": 4,
                "exit_reason": "stop_loss",
                "net_return": -0.102,
                "reentry_number": 0,
            },
        ]
    )
    return BloodChipBacktestResult(pd.DataFrame(), trades, pd.DataFrame())


def test_compose_plan_ranks_lower_residual_volatility_and_maps_simulated_state() -> None:
    stock_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"],
            "name": ["平安银行", "万科A", "测试股", "浦发银行"],
            "industry": ["银行", "地产", "测试", "银行"],
        }
    )

    payload = compose_blood_chip_long_plan(
        signal_date="2026-08-10",
        signals=_signals(),
        result=_backtest_result(),
        stock_basic=stock_basic,
        generated_at="2026-08-10T18:00:00",
    )

    assert [row["ts_code"] for row in payload["candidates"]] == ["000001.SZ", "000002.SZ"]
    assert payload["candidates"][0]["rank"] == 1
    assert payload["candidates"][0]["action"] == "NEXT_OPEN_WATCH"
    assert payload["candidates"][0]["initial_tranche_fraction"] == 0.20
    assert payload["candidates"][0]["blood_chip_subtype"] == "深度带血筹（三周期超跌）"
    assert payload["candidates"][0]["shock_kdj_negative_count"] == 3
    assert "首仓 20%" in payload["candidates"][0]["execution_rule"]
    assert payload["simulated_positions"][0]["name"] == "浦发银行"
    assert payload["simulated_positions"][0]["stop_price"] == 8.1
    assert payload["simulated_positions"][0]["reentry_number"] == 1
    assert payload["simulated_positions"][0]["tranches_filled"] == 2
    assert payload["simulated_positions"][0]["stage_label"] == "第二段已完成（累计 50%）"
    assert payload["simulated_positions"][0]["next_addition_fraction"] == 0.50
    assert payload["simulated_positions"][0]["next_stage_ready"] is True
    assert "至少持有 10 日" in payload["simulated_positions"][0]["next_trigger"]
    assert payload["strategy"]["scale_in_policy"] == "increasing_survival"
    assert payload["strategy"]["tranche_fractions"] == [0.20, 0.30, 0.50]
    assert payload["recent_exits"][0]["exit_reason"] == "stop_loss"
    assert payload["summary"] == {
        "new_candidates": 2,
        "simulated_active_positions": 1,
        "stopped_today": 1,
        "reentry_candidates": 0,
        "deep_kdj_candidates": 1,
    }


def test_daily_iteration_compares_with_previous_snapshot() -> None:
    current = {
        "signal_date": "2026-08-10",
        "candidates": [{"ts_code": "000001.SZ"}, {"ts_code": "000002.SZ"}],
        "simulated_positions": [
            {"ts_code": "600000.SH", "tranches_filled": 2, "next_stage_ready": True},
            {"ts_code": "000004.SZ", "tranches_filled": 1, "next_stage_ready": False},
        ],
    }
    previous = {
        "signal_date": "2026-08-07",
        "candidates": [{"ts_code": "000002.SZ"}, {"ts_code": "000003.SZ"}],
        "simulated_positions": [
            {"ts_code": "600000.SH", "tranches_filled": 1, "next_stage_ready": False},
            {"ts_code": "000005.SZ", "tranches_filled": 3, "next_stage_ready": False},
        ],
    }

    iteration = build_blood_chip_daily_iteration(current, previous)

    assert iteration["previous_signal_date"] == "2026-08-07"
    assert iteration["added_candidates"] == ["000001.SZ"]
    assert iteration["removed_candidates"] == ["000003.SZ"]
    assert iteration["continued_candidates"] == ["000002.SZ"]
    assert iteration["new_positions"] == ["000004.SZ"]
    assert iteration["closed_positions"] == ["000005.SZ"]
    assert iteration["advanced_positions"] == [
        {"ts_code": "600000.SH", "from_stage": 1, "to_stage": 2}
    ]
    assert iteration["ready_additions"] == ["600000.SH"]


def test_long_plan_rejects_benchmark_stale_for_actual_daily_end() -> None:
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20260811", "20260812"],
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "pre_close": [10.0, 10.1],
            "pct_chg": [1.0, 0.99],
            "vol": [1_000_000.0, 1_100_000.0],
            "amount": [100_000.0, 110_000.0],
        }
    )
    benchmark = pd.DataFrame(
        {
            "trade_date": ["20260811"],
            "close": [4_000.0],
            "pct_chg": [0.5],
        }
    )

    with pytest.raises(ValueError, match="daily=2026-08-12 benchmark=2026-08-11"):
        build_blood_chip_long_plan(
            daily,
            benchmark,
            signal_date="2026-08-12",
        )
