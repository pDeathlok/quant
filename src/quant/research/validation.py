"""Leakage-resistant chronological validation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class PurgedWalkForwardSplitter:
    """Build ordered train/test windows with explicit purge and embargo gaps."""

    def __init__(
        self,
        *,
        train_periods: int,
        test_periods: int,
        purge_periods: int = 0,
        embargo_periods: int = 0,
        expanding: bool = False,
    ) -> None:
        if train_periods <= 0:
            raise ValueError("train_periods must be positive")
        if test_periods <= 0:
            raise ValueError("test_periods must be positive")
        if purge_periods < 0:
            raise ValueError("purge_periods must be non-negative")
        if embargo_periods < 0:
            raise ValueError("embargo_periods must be non-negative")
        self.train_periods = int(train_periods)
        self.test_periods = int(test_periods)
        self.purge_periods = int(purge_periods)
        self.embargo_periods = int(embargo_periods)
        self.expanding = bool(expanding)

    @staticmethod
    def _dates(dates: Sequence[object]) -> pd.DatetimeIndex:
        normalized = pd.to_datetime(pd.Index(dates), errors="coerce")
        if normalized.isna().any():
            raise ValueError("dates contain invalid timestamps")
        return pd.DatetimeIndex(normalized.unique()).sort_values()

    def split(self, dates: Sequence[object]) -> list[TimeSplit]:
        ordered = self._dates(dates)
        first_test = self.train_periods + self.purge_periods
        results: list[TimeSplit] = []
        test_start_index = first_test
        fold = 1
        while test_start_index + self.test_periods <= len(ordered):
            train_end_exclusive = test_start_index - self.purge_periods
            train_start_index = (
                0
                if self.expanding
                else train_end_exclusive - self.train_periods
            )
            if train_start_index < 0 or train_end_exclusive <= train_start_index:
                break
            test_end_exclusive = test_start_index + self.test_periods
            results.append(
                TimeSplit(
                    fold=fold,
                    train_start=ordered[train_start_index],
                    train_end=ordered[train_end_exclusive - 1],
                    test_start=ordered[test_start_index],
                    test_end=ordered[test_end_exclusive - 1],
                )
            )
            fold += 1
            test_start_index = test_end_exclusive + self.embargo_periods
        return results

    def split_indices(
        self,
        dates: Sequence[object],
    ) -> list[tuple[np.ndarray, np.ndarray, TimeSplit]]:
        original = pd.to_datetime(pd.Index(dates), errors="coerce")
        if original.isna().any():
            raise ValueError("dates contain invalid timestamps")
        output: list[tuple[np.ndarray, np.ndarray, TimeSplit]] = []
        for split in self.split(dates):
            train = np.flatnonzero(
                (original >= split.train_start) & (original <= split.train_end)
            )
            test = np.flatnonzero(
                (original >= split.test_start) & (original <= split.test_end)
            )
            output.append((train, test, split))
        return output


def purge_overlapping_training_events(
    events: pd.DataFrame,
    *,
    test_start: str | pd.Timestamp,
    label_end_column: str,
) -> pd.DataFrame:
    """Remove training labels whose outcome window reaches the test period."""

    if label_end_column not in events.columns:
        raise ValueError(f"events missing label end column: {label_end_column}")
    label_end = pd.to_datetime(events[label_end_column], errors="coerce")
    if label_end.isna().any():
        raise ValueError("label end column contains invalid timestamps")
    boundary = pd.Timestamp(test_start)
    return events.loc[label_end.lt(boundary)].copy().reset_index(drop=True)
