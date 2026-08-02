from __future__ import annotations

import pandas as pd
import pytest

from quant.research.validation import (
    PurgedWalkForwardSplitter,
    purge_overlapping_training_events,
)


def test_purged_walk_forward_has_explicit_gaps_and_non_overlapping_tests() -> None:
    dates = pd.bdate_range("2026-01-01", periods=15)
    splitter = PurgedWalkForwardSplitter(
        train_periods=5,
        test_periods=2,
        purge_periods=1,
        embargo_periods=1,
    )

    splits = splitter.split(dates)

    assert [(item.train_start, item.train_end, item.test_start, item.test_end) for item in splits] == [
        (dates[0], dates[4], dates[6], dates[7]),
        (dates[3], dates[7], dates[9], dates[10]),
        (dates[6], dates[10], dates[12], dates[13]),
    ]
    assert all(item.train_end < item.test_start for item in splits)
    assert splits[0].test_end < splits[1].test_start


def test_purged_walk_forward_sorts_and_deduplicates_dates() -> None:
    dates = ["2026-01-05", "2026-01-02", "2026-01-02", "2026-01-06"]
    splitter = PurgedWalkForwardSplitter(
        train_periods=2,
        test_periods=1,
        expanding=True,
    )

    splits = splitter.split(dates)

    assert len(splits) == 1
    assert splits[0].train_start == pd.Timestamp("2026-01-02")
    assert splits[0].train_end == pd.Timestamp("2026-01-05")
    assert splits[0].test_start == pd.Timestamp("2026-01-06")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_periods": 0, "test_periods": 1},
        {"train_periods": 2, "test_periods": 0},
        {"train_periods": 2, "test_periods": 1, "purge_periods": -1},
        {"train_periods": 2, "test_periods": 1, "embargo_periods": -1},
    ],
)
def test_purged_walk_forward_rejects_invalid_windows(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        PurgedWalkForwardSplitter(**kwargs)


def test_purge_overlapping_training_events_uses_label_end_time() -> None:
    events = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "label_end": pd.to_datetime(["2026-01-05", "2026-01-08", "2026-01-07"]),
            "value": [1, 2, 3],
        }
    )

    purged = purge_overlapping_training_events(
        events,
        test_start="2026-01-07",
        label_end_column="label_end",
    )

    assert purged["value"].tolist() == [1]
