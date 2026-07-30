import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.strategies.custom.chan_daily import add_chan_daily_signals, summarize_chan_daily
from scripts.research.backtest_chan_daily import ExitRule, build_topn_portfolio, simulate_exit


def _make_chan_third_buy_frame(name: str = "测试股份") -> pd.DataFrame:
    prefix = list(np.linspace(8.0, 10.0, 70))
    pivots = [
        10.0,
        12.0,
        10.5,
        12.2,
        10.7,
        12.4,
        13.4,
        12.75,
        13.75,
    ]
    swings = []
    for left, right in zip(pivots, pivots[1:]):
        leg = np.linspace(left, right, 5, endpoint=False).tolist()
        swings.extend(leg)
    swings.append(pivots[-1])
    values = prefix + swings
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    close = np.array(values, dtype=float)
    open_ = close - 0.04
    high = close + 0.05
    low = close - 0.05
    volume = np.full(len(values), 1000.0)

    for idx in range(70, len(values)):
        volume[idx] = 1200 + (idx - 70) * 20

    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST.SZ",
            "name": name,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pd.Series(close).shift(1).fillna(close[0]),
            "volume": volume,
        }
    )


def test_chan_daily_detects_daily_third_buy_after_center_pullback():
    out = add_chan_daily_signals(_make_chan_third_buy_frame())

    signal_idx = out.index[out["signal_chan_daily_long"] == 1][-1]

    assert out.loc[signal_idx, "chan_buy3_confirm"] == 1
    assert out.loc[signal_idx, "chan_signal_name"] == "三买确认"
    assert out.loc[signal_idx, "chan_center_low"] > 0
    assert out.loc[signal_idx, "chan_center_high"] > out.loc[signal_idx, "chan_center_low"]
    assert out.loc[signal_idx, "chan_score"] >= 90
    assert out.loc[signal_idx, "chan_buy_plan"]
    assert out.loc[signal_idx, "chan_sell_plan"]


def test_chan_daily_excludes_risk_names_from_final_long_signal():
    out = add_chan_daily_signals(_make_chan_third_buy_frame(name="*ST测试"))

    assert out["chan_buy3_confirm"].sum() >= 1
    assert out["signal_chan_daily_long"].sum() == 0


def test_summarize_chan_daily_counts_signals():
    summary = summarize_chan_daily(_make_chan_third_buy_frame())

    assert summary["rows"] == 111
    assert summary["long_signals"] >= 1
    assert summary["buy3_confirm"] >= 1


def test_trailing_stop_uses_peak_confirmed_before_current_daily_bar():
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")],
            "symbol": ["TEST.SZ"],
            "name": ["测试"],
            "close": [100.0],
            "entry_open": [100.0],
            "date_t1": [pd.Timestamp("2026-01-02")],
            "open_t2": [100.0],
            "high_t2": [120.0],
            "low_t2": [110.0],
            "close_t2": [118.0],
            "date_t2": [pd.Timestamp("2026-01-05")],
        }
    )
    rule = ExitRule("trail", "trailing", hold_days=1, take_profit=0.10, trail_drawdown=0.05)

    result = simulate_exit(frame, rule)

    assert result.iloc[0]["exit_type"] == "expiry"
    assert np.isclose(result.iloc[0]["return_pct"], 18.0)


def test_topn_portfolio_respects_cash_and_overlapping_positions():
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "entry_date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "exit_date": pd.to_datetime(["2026-01-10", "2026-01-06"]),
            "symbol": ["A.SZ", "B.SZ"],
            "chan_signal_name": ["三买确认", "三买确认"],
            "chan_score": [99.0, 98.0],
            "return_pct": [10.0, 100.0],
        }
    )

    equity = build_topn_portfolio(trades, top_n=1, round_trip_cost_pct=0.0)

    assert list(equity["opened_positions"].iloc[:2]) == [1, 0]
    assert equity.iloc[-1]["equity"] == 1.1
