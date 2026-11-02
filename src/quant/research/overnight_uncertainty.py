"""Descriptive date-block uncertainty for locked picks, not post-search proof."""

from __future__ import annotations

import numpy as np
import pandas as pd


def date_block_audit(picks: pd.DataFrame, calendar: pd.DatetimeIndex, *,
                     block_days: int = 5, draws: int = 2000, seed: int = 20260912) -> dict:
    """Circular moving blocks preserve same-date stock clustering.

    Intervals are percentile summaries, conditional on this historical sample.
    They do not correct model/threshold search or establish independent evidence.
    Unknown outcomes count as failures in the all-signal win fraction; their
    returns remain unknown and are not assigned a fictitious PnL.
    """
    calendar = pd.DatetimeIndex(calendar)
    if (calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing
            or block_days < 1 or block_days > len(calendar) or draws < 100):
        raise ValueError("Invalid calendar or resampling parameters")
    frame = picks[["date", "net"]].copy()
    frame["date"] = pd.to_datetime(frame.date)
    if not frame.date.isin(calendar).all() or np.isinf(frame.net.to_numpy(float)).any():
        raise ValueError("Pick date outside calendar or infinite return")
    frame["win"] = frame.net.gt(0).astype(int)
    frame["known"] = frame.net.notna().astype(int)
    daily = frame.groupby("date").agg(signals=("win", "size"), wins=("win", "sum"),
        known=("known", "sum"), known_net_sum=("net", "sum")).reindex(calendar, fill_value=0)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(calendar), size=(draws, int(np.ceil(len(calendar) / block_days))))
    indices = ((starts[:, :, None] + np.arange(block_days)) % len(calendar)).reshape(draws, -1)[:, :len(calendar)]
    totals = daily.to_numpy(float)[indices].sum(axis=1)
    win = np.divide(totals[:, 1], totals[:, 0], out=np.full(draws, np.nan), where=totals[:, 0] > 0)
    mean = np.divide(totals[:, 3], totals[:, 2], out=np.full(draws, np.nan), where=totals[:, 2] > 0)

    def interval(values: np.ndarray) -> list[float] | None:
        finite = values[np.isfinite(values)]
        return np.quantile(finite, [.025, .975]).tolist() if len(finite) else None

    counts = daily.signals.sort_values(ascending=False)
    total = len(frame)
    concentration = float(counts.iloc[:2].sum() / total) if total else None
    shares = counts / total if total else counts.astype(float)
    active_dates = int(counts.gt(0).sum())
    return {"signals": total, "active_dates": active_dates,
            "known": int(frame.known.sum()), "unknown": int(frame.net.isna().sum()),
            "win_missing_as_failure": float(frame.win.mean()) if total else None,
            "known_mean_net": float(frame.net.mean()) if frame.known.sum() else None,
            "largest_two_dates_share": concentration,
            "effective_signal_dates": float(1 / np.square(shares).sum()) if total else 0.,
            "block_days": block_days, "draws": draws,
            "nonempty_draws": int(np.isfinite(win).sum()),
            "win_missing_as_failure_percentile95": interval(win),
            "known_mean_net_percentile95": interval(mean),
            "degenerate_win_sample": bool(frame.win.nunique() <= 1),
            "warning": "Descriptive only; known-return interval excludes unresolved PnL;"
            " no search correction or independent confirmation; degenerate samples cannot establish certainty."}
