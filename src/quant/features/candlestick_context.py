"""Causal candlestick context factors shared by short-side rankers.

These factors are materialized beside the frozen production contracts so they
can be audited and backtested before a checksum-pinned model is retrained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CANDLE_CONTEXT_RESEARCH_FEATURE_COLUMNS: tuple[str, ...] = (
    "rs_upper_shadow_range_share",
    "rs_upper_shadow_body_ratio",
    "rs_volume_ratio_prev20",
)
CANDLE_CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (
    "rs_upper_shadow_pct",
    *CANDLE_CONTEXT_RESEARCH_FEATURE_COLUMNS,
)
CANDLE_CONTEXT_FEATURE_SCHEMA_VERSION = "candle_context_features_v1_4_20260831"


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce").div(
        pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    )


def compute_candlestick_context_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Return exact, causal upper-shadow and prior-volume context factors.

    ``rs_volume_ratio_prev20`` deliberately excludes the current session from
    its denominator.  This differs from the canonical
    ``volume_relative_20d``, whose rolling mean includes the current session.
    A complete 20-session prior window is required.
    """

    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"candlestick context input misses columns: {missing}")

    frame = daily.copy()
    if "date" in frame.columns:
        frame = frame.sort_values("date", kind="stable")
    elif "trade_date" in frame.columns:
        frame = frame.sort_values("trade_date", kind="stable")

    open_ = pd.to_numeric(frame["open"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    pre_close = (
        pd.to_numeric(frame["pre_close"], errors="coerce")
        if "pre_close" in frame.columns
        else close.shift(1)
    )

    candle_top = pd.concat([open_, close], axis=1).max(axis=1)
    upper_shadow = high - candle_top
    candle_range = high - low
    body = (close - open_).abs()
    previous_volume_mean_20 = volume.shift(1).rolling(20, min_periods=20).mean()

    return pd.DataFrame(
        {
            "rs_upper_shadow_pct": _safe_div(upper_shadow, pre_close) * 100.0,
            "rs_upper_shadow_range_share": _safe_div(upper_shadow, candle_range),
            "rs_upper_shadow_body_ratio": _safe_div(upper_shadow, body),
            "rs_volume_ratio_prev20": _safe_div(volume, previous_volume_mean_20),
        },
        index=frame.index,
    ).loc[:, CANDLE_CONTEXT_FEATURE_COLUMNS]


__all__ = [
    "CANDLE_CONTEXT_FEATURE_COLUMNS",
    "CANDLE_CONTEXT_FEATURE_SCHEMA_VERSION",
    "CANDLE_CONTEXT_RESEARCH_FEATURE_COLUMNS",
    "compute_candlestick_context_features",
]
