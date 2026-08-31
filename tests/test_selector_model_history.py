from __future__ import annotations

import runpy
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/research/build_selector_model_history.py"


def test_group_realized_history_waits_for_label_horizon() -> None:
    module = runpy.run_path(str(SCRIPT))
    sessions = [pd.Timestamp(date(2024, 1, 1) + timedelta(days=index)) for index in range(12)]
    history = pd.DataFrame(
        {
            "symbol": [f"stock-{index}" for index in range(12)],
            "date": sessions,
            "matched_groups": [["group-a"]] * 12,
            "future_return_t5_pct": [10.0] * 12,
            "future_max_high_t5_pct": [12.0] * 12,
        }
    )

    result = module["add_group_realized_history"](history, sessions)

    assert np.isnan(result.loc[4, "selector_group_hold_realized_20d"])
    assert result.loc[9, "selector_group_hold_realized_20d"] == 10.0
    assert result.loc[9, "selector_group_buy_realized_20d"] == 12.0


def test_selector_history_materializes_candlestick_context_contract() -> None:
    module = runpy.run_path(str(SCRIPT))
    rows = 21
    frame = pl.DataFrame(
        {
            "symbol": ["002724.SZ"] * rows,
            "open": [6.77] * 20 + [6.80],
            "high": [6.77] * 20 + [7.33],
            "low": [6.77] * 20 + [6.77],
            "close": [6.77] * 20 + [6.93],
            "volume": [100.0] * 20 + [123.0],
        }
    )

    result = frame.with_columns(module["candlestick_context_expressions"]())
    latest = result.row(-1, named=True)

    assert latest["rs_upper_shadow_range_share"] == pytest.approx(0.40 / 0.56)
    assert latest["rs_upper_shadow_body_ratio"] == pytest.approx(0.40 / 0.13)
    assert latest["rs_volume_ratio_prev20"] == pytest.approx(1.23)
