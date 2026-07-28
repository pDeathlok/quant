from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data.factors import KDJ
from quant.features.b1_gate import (
    B1_GATE_METRIC_COLUMNS,
    calculate_b1_gate,
)
from quant.features.variable_library import build_continuous_ohlc, calc_bbi
from scripts.research import rebuild_strategy_signal_cache as signal_cache


def test_b1_gate_reuses_shared_factors_by_row_position() -> None:
    rows = 90
    close = (
        np.linspace(12.0, 9.0, rows) + np.sin(np.arange(rows) / 4) * 0.3
    )
    pre_close = np.concatenate(([close[0]], close[:-1]))
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=rows),
            "open": close * 1.001,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "pre_close": pre_close,
        },
        index=pd.RangeIndex(10, 10 + rows),
    )
    price = build_continuous_ohlc(daily)
    shared = pd.DataFrame(
        {
            "bbi": calc_bbi(price["close"]).to_numpy(),
            "kdj_d_j": KDJ().compute(price)["J"].to_numpy(),
        }
    )

    direct = calculate_b1_gate(daily)
    reused = calculate_b1_gate(daily, shared_factors=shared)

    pd.testing.assert_frame_equal(direct, reused)


def test_signal_gate_uses_the_original_450_day_b1_window(
    monkeypatch,
) -> None:
    dates = pd.bdate_range("2024-01-01", periods=500)
    close = np.linspace(10.0, 15.0, len(dates))
    frame = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "symbol": "000001.SZ",
            "trade_date": dates.strftime("%Y%m%d"),
            "date": dates,
            "name": "测试股份",
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "pre_close": np.concatenate(([close[0]], close[:-1])),
            "vol": 1_000_000,
        }
    )
    rebuild_start = dates[-1]
    captured: dict[str, pd.Timestamp] = {}

    def fake_gate(daily: pd.DataFrame) -> pd.DataFrame:
        captured["min"] = daily["date"].min()
        captured["max"] = daily["date"].max()
        return pd.DataFrame(
            {
                "b1_gate": False,
                **{
                    column: np.nan
                    for column in B1_GATE_METRIC_COLUMNS
                },
            },
            index=daily.index,
        )

    monkeypatch.setattr(signal_cache, "calculate_b1_gate", fake_gate)
    result = signal_cache._build_b1_gate_rows(
        "000001.SZ",
        frame,
        rebuild_start.strftime("%Y-%m-%d"),
    )

    assert result.empty
    assert captured["min"] >= rebuild_start - pd.Timedelta(days=450)
    assert captured["max"] == rebuild_start
