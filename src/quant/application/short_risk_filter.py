"""Shared risk filter for the short-term selector."""

from __future__ import annotations

from math import isfinite

import pandas as pd


LOOKBACK_SESSIONS = 5
SHOCK_DROP_PCT = 7.0
SHOCK_VOLUME_MULTIPLE = 2.0
FILTER_VERSION = "unreclaimed-volume-selloff-v1"
FILTER_LABEL = "近期放量大跌，还没收回长阴低点"


def unreclaimed_selloff_symbols(
    daily: pd.DataFrame, symbols: set[str], signal_date: str
) -> set[str]:
    """Detect a selloff in the previous five market sessions as of the signal close."""
    if daily.empty or not symbols:
        return set()

    target = signal_date.replace("-", "")
    dates = daily["trade_date"].astype(str).str.replace("-", "", regex=False)
    dates = dates.where(dates <= target)
    sessions = sorted(dates.dropna().unique())[-(LOOKBACK_SESSIONS + 1):]
    if len(sessions) < 2 or sessions[-1] != target:
        return set()

    candidates = daily.loc[
        dates.notna() & daily["ts_code"].isin(symbols),
        ["ts_code", "close", "low", "pre_close", "pct_chg", "volume"],
    ].copy()
    candidates["trade_date"] = dates.loc[candidates.index]
    candidates = candidates.sort_values("trade_date").drop_duplicates(
        ["ts_code", "trade_date"], keep="last"
    )
    for column in ("close", "low", "pre_close", "pct_chg", "volume"):
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce").astype(float)
    candidates["prior_mean_volume"] = candidates.groupby("ts_code")["volume"].transform(
        lambda volume: volume.shift().rolling(LOOKBACK_SESSIONS).mean()
    )
    bars = {
        (bar.ts_code, bar.trade_date): bar
        for bar in candidates[candidates["trade_date"].isin(sessions)].itertuples(index=False)
    }

    excluded = set()
    for symbol in symbols:
        current = bars.get((symbol, target))
        if current is None or not (isfinite(current.close) and current.close > 0):
            continue
        adjustment = 1.0
        for day in reversed(sessions[:-1]):
            previous = bars.get((symbol, day))
            if previous is None or not all(
                isfinite(value) and value > 0 for value in (current.pre_close, previous.close)
            ):
                break
            # Express the historical low on the signal day's price scale.
            adjustment *= current.pre_close / previous.close
            if (
                previous.pct_chg <= -SHOCK_DROP_PCT
                and previous.prior_mean_volume > 0
                and previous.volume >= SHOCK_VOLUME_MULTIPLE * previous.prior_mean_volume
                and isfinite(previous.low)
                and previous.low > 0
                and bars[(symbol, target)].close < previous.low * adjustment - 1e-9
            ):
                excluded.add(symbol)
                break
            current = previous
    return excluded
