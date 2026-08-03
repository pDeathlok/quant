from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.long_entry_v2 import (
    add_entry_labels,
    cooldown_cases,
    month_end_week_mask,
    select_industry_capped,
)


def test_entry_labels_reward_multi_horizon_return_and_shallower_drawdown() -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "date": pd.Timestamp("2024-01-05"),
                "ts_code": f"{index:06d}.SZ",
                "industry": "行业A" if index < 5 else "行业B",
                "excess_return_13w": index / 100,
                "excess_return_26w": index / 50,
                "excess_return_52w": index / 25,
                "mae_13w": -0.20 + index / 50,
                "mae_26w": -0.30 + index / 40,
            }
        )
    labelled = add_entry_labels(pd.DataFrame(rows))
    assert labelled.loc[9, "label_entry_utility"] > labelled.loc[0, "label_entry_utility"]
    assert labelled.loc[9, "label_entry_utility_26w"] > labelled.loc[0, "label_entry_utility_26w"]
    assert labelled.loc[9, "label_entry_utility_13w"] > labelled.loc[0, "label_entry_utility_13w"]
    assert labelled.loc[9, "label_entry_success"] == 1.0
    assert labelled.loc[0, "label_entry_success"] == 0.0


def test_industry_cap_and_month_end_week_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-05")] * 5,
            "ts_code": list("ABCDE"),
            "industry": ["X", "X", "X", "Y", "Z"],
            "score": [5, 4, 3, 2, 1],
        }
    )
    selected = select_industry_capped(frame, score_column="score", top_n=3, max_per_industry=1)
    assert selected["ts_code"].tolist() == ["A", "D", "E"]

    weeks = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-01-26", "2024-02-02"])})
    assert month_end_week_mask(weeks).tolist() == [False, True, True]


def test_case_cooldown_removes_adjacent_weekly_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-12", "2024-05-10"]),
            "ts_code": ["A", "A", "A"],
            "severity": [3.0, 2.0, 1.0],
        }
    )
    result = cooldown_cases(frame, severity_column="severity", cooldown_days=91)
    assert result["date"].tolist() == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-05-10")]
