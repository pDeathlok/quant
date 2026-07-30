from __future__ import annotations

import runpy
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


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
