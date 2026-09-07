"""Shared rule-backtest helpers without importing model-training scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.core.paths import PROJECT_ROOT

DEFAULT_DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/b1/research/xgb_project_vars_strategy"


def drop_overlapping_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Keep the first signal per stock until that trade exits."""
    if trades.empty:
        return trades
    required = {"symbol", "date", "exit_date"}
    if not required <= set(trades.columns):
        missing = sorted(required - set(trades.columns))
        raise ValueError(f"Cannot enforce non-overlap without columns: {missing}")

    ordered = trades.sort_values(["symbol", "date", "exit_date"]).reset_index()
    entry_ns = pd.to_datetime(ordered["date"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    exit_ns = pd.to_datetime(ordered["exit_date"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    keep_mask = np.zeros(len(ordered), dtype=bool)

    symbols = ordered["symbol"].to_numpy()
    starts = np.r_[0, np.flatnonzero(symbols[1:] != symbols[:-1]) + 1]
    ends = np.r_[starts[1:], len(ordered)]
    nat = np.iinfo("int64").min
    for start, end in zip(starts, ends):
        current_exit = nat
        for pos in range(start, end):
            if exit_ns[pos] == nat:
                continue
            if entry_ns[pos] > current_exit:
                keep_mask[pos] = True
                current_exit = exit_ns[pos]

    kept_index = ordered.loc[keep_mask, "index"].to_numpy()
    return trades.loc[kept_index].sort_values(["date", "symbol"]).reset_index(drop=True)
