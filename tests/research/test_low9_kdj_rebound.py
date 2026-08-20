from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.low9_kdj_rebound import (
    SymbolSignalState,
    benjamini_hochberg,
    newey_west_mean_test,
)


def _bar(date: pd.Timestamp, close: float) -> dict[str, object]:
    return {
        "date": date,
        "open": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "pre_close": close + 1.0,
        "amount": 100_000.0,
        "name": "测试股",
    }


def test_completed_low9_uses_next_bar_as_entry() -> None:
    dates = pd.bdate_range("2026-01-01", periods=16)
    closes = [20.0 - index for index in range(16)]
    state = SymbolSignalState(
        symbol="000001.SZ",
        horizons=(1, 3),
        min_history_bars=1,
    )
    resolved: list[dict[str, object]] = []
    for date, close in zip(dates, closes):
        resolved.extend(
            state.process_bar(
                _bar(date, close),
                {"open": 100.0, "close": 100.0},
            )
        )

    assert len(resolved) == 2
    first = resolved[0]
    assert first["signal_date"] == dates[12]
    assert first["entry_date"] == dates[13]
    assert first["horizon"] == 1
    assert resolved[1]["horizon"] == 3


def test_causal_continuity_does_not_treat_split_as_price_crash() -> None:
    state = SymbolSignalState(symbol="000001.SZ")
    first = state._continuous_prices(
        raw_open=100.0,
        raw_high=101.0,
        raw_low=99.0,
        raw_close=100.0,
        pre_close=99.0,
    )
    second = state._continuous_prices(
        raw_open=50.0,
        raw_high=51.0,
        raw_low=49.0,
        raw_close=50.0,
        pre_close=50.0,
    )

    assert first[3] == 100.0
    assert second[3] == 100.0


def test_benjamini_hochberg_is_monotone_in_rank() -> None:
    p_values = np.array([0.04, 0.001, 0.02, np.nan])
    adjusted = benjamini_hochberg(p_values)
    order = np.argsort(p_values[:3])

    assert np.all(np.diff(adjusted[:3][order]) >= 0)
    assert np.isnan(adjusted[3])


def test_newey_west_mean_test_detects_positive_constant_shift() -> None:
    values = np.linspace(0.005, 0.015, 200)
    result = newey_west_mean_test(values, lag=5)

    assert result["mean"] > 0
    assert result["p"] < 0.01
