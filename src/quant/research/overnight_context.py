"""Same-cutoff watchlist context, not a whole-market breadth indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_watchlist_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only a whitelist of observed pre-close inputs by date."""
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate stock/date context keys")
    dates = pd.to_datetime(frame.date)
    if not pd.to_datetime(frame.cutoff).eq(dates + pd.Timedelta(hours=14, minutes=55)).all():
        raise ValueError("Context requires a common 14:55 cutoff")
    names = ["return_so_far", "last30_return", "vwap_deviation", "range_so_far", "amount_so_far"]
    columns = [f"x_min_{name}" for name in names]
    part = frame[["date", *columns]].copy()
    part[columns] = part[columns].replace([np.inf, -np.inf], np.nan)
    valid = part[columns].notna().all(axis=1) & part.x_min_amount_so_far.gt(0)
    part = part.loc[valid].copy()
    part["positive"] = part.x_min_return_so_far.gt(0).astype(float)
    part["up1"] = part.x_min_return_so_far.ge(.01).astype(float)
    part["down1"] = part.x_min_return_so_far.le(-.01).astype(float)
    part["above_vwap"] = part.x_min_vwap_deviation.gt(0).astype(float)
    daily = part.groupby("date").agg(
        x_ctx_count=("positive", "size"), x_ctx_return_mean=("x_min_return_so_far", "mean"),
        x_ctx_return_median=("x_min_return_so_far", "median"), x_ctx_return_std=("x_min_return_so_far", "std"),
        x_ctx_positive_fraction=("positive", "mean"), x_ctx_up1_fraction=("up1", "mean"),
        x_ctx_down1_fraction=("down1", "mean"), x_ctx_above_vwap_fraction=("above_vwap", "mean"),
        x_ctx_last30_mean=("x_min_last30_return", "mean"), x_ctx_range_mean=("x_min_range_so_far", "mean"))
    result = frame.copy()
    for col in daily:
        result[col] = result.date.map(daily[col])
    result["x_ctx_relative_return"] = result.x_min_return_so_far - result.x_ctx_return_mean
    result["x_ctx_relative_last30"] = result.x_min_last30_return - result.x_ctx_last30_mean
    result["x_ctx_return_rank"] = np.nan
    result.loc[part.index, "x_ctx_return_rank"] = part.groupby("date").x_min_return_so_far.rank(pct=True)
    return result
