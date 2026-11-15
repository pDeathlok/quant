"""Causal option-volatility and price-stress features; price stress is not VIX."""
from __future__ import annotations

import numpy as np
import pandas as pd

QVIX_FEATURES = [f"panic_{scope}_{name}" for scope in ("common", "own")
                 for name in ("rank", "change5", "premium", "missing")]
STRESS_FEATURES = ["panic_downside_rank", "panic_drawdown_rank",
                   "panic_sell_pressure_rank", "panic_score_change5"]
QVIX_MAP = {"IH": "50etf", "IF": "300etf", "IC": "500etf", "IM": "1000index"}


def past_percentile(series: pd.Series, window: int = 252, minimum: int = 60) -> pd.Series:
    """Rank current value against preceding observations only; ties at mid-rank."""
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        past = values[max(0, i-window):i]
        past = past[np.isfinite(past)]
        if np.isfinite(value) and len(past) >= minimum:
            result[i] = (np.sum(past < value) + .5*np.sum(past == value))/len(past)
    return pd.Series(result, index=series.index)


def price_stress(spot: pd.DataFrame) -> pd.DataFrame:
    """Closing-price/volume stress, available after this session's close."""
    frame = spot.sort_index()
    if frame.index.duplicated().any():
        raise ValueError("Duplicate spot dates")
    ret = np.log(frame.close.where(frame.close.gt(0))).diff()
    downside = np.sqrt(ret.clip(upper=0).pow(2).rolling(20).mean()*252)
    drawdown = 1-frame.close/frame.close.rolling(60).max()
    volume_ratio = frame.vol/frame.vol.shift().rolling(20).mean().replace(0, np.nan)
    sell_pressure = ((-ret).clip(lower=0)*volume_ratio).rolling(5).sum()
    result = pd.DataFrame(index=frame.index)
    result["panic_downside_rank"] = past_percentile(downside)
    result["panic_drawdown_rank"] = past_percentile(drawdown)
    result["panic_sell_pressure_rank"] = past_percentile(sell_pressure)
    result["panic_price_score"] = 100*result.mean(axis=1, skipna=False)
    result["panic_score_change5"] = result.panic_price_score.diff(5)/100
    result["realized_vol20"] = ret.rolling(20).std()*np.sqrt(252)
    return result.replace([np.inf, -np.inf], np.nan)


def option_features(close: pd.Series, realized: pd.Series, calendar: pd.Index) -> pd.DataFrame:
    """No forward/backfill; add one full session lag beyond signal-date availability.

    Feed the underlying's complete trading calendar. A missing session remains
    missing, including after a discontinued source. This is a frozen historical
    snapshot, not proof of publication timestamps or unrevised vintages.
    """
    if close.index.duplicated().any():
        raise ValueError("Duplicate QVIX dates must be quarantined before joining")
    q = close.reindex(calendar).where(lambda s: s.gt(0))
    result = pd.DataFrame(index=calendar)
    result["level"] = q
    result["rank"] = past_percentile(q)
    # Require six contiguous observations: don't bridge a source outage.
    result["change5"] = (q/q.shift(5)-1).where(q.rolling(6).count().eq(6))
    result["premium"] = q/100-realized.reindex(calendar)
    result = result.shift(1)
    result["source_date"] = pd.Series(calendar, index=calendar).where(q.notna()).shift(1)
    result["missing"] = result[["rank", "change5", "premium"]].isna().any(axis=1).astype(float)
    return result.replace([np.inf, -np.inf], np.nan)
