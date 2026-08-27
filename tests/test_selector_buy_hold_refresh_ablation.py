from __future__ import annotations

import pandas as pd
import pytest

from quant.research.selector_buy_hold_refresh_ablation import (
    choose_maintenance_aware_candidate,
    history_start,
    refresh_periods,
)


def test_refresh_periods_cover_range_without_overlap() -> None:
    periods = refresh_periods(
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-08-14"),
        3,
    )

    assert [(item.cutoff.date().isoformat(), item.end.date().isoformat()) for item in periods] == [
        ("2025-01-01", "2025-03-31"),
        ("2025-04-01", "2025-06-30"),
        ("2025-07-01", "2025-08-14"),
    ]


def test_history_start_supports_rolling_and_expanding_windows() -> None:
    cutoff = pd.Timestamp("2025-01-01")
    dataset_start = pd.Timestamp("2020-01-02")

    assert history_start(cutoff, 36, dataset_start) == pd.Timestamp("2022-01-01")
    assert history_start(cutoff, None, dataset_start) == dataset_start


def test_choose_candidate_prefers_lower_maintenance_inside_tolerance() -> None:
    rows = [
        {
            "frequency_months": 1,
            "metrics": {"development": {"mean_daily_spearman": 0.120}},
        },
        {
            "frequency_months": 3,
            "metrics": {"development": {"mean_daily_spearman": 0.117}},
        },
        {
            "frequency_months": 6,
            "metrics": {"development": {"mean_daily_spearman": 0.110}},
        },
    ]

    selected = choose_maintenance_aware_candidate(
        rows,
        complexity_key=lambda row: -float(row["frequency_months"]),
        tolerance=0.005,
    )

    assert selected["frequency_months"] == 3


def test_refresh_periods_reject_non_positive_frequency() -> None:
    with pytest.raises(ValueError, match="positive"):
        refresh_periods(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01"), 0)
