"""Causal key-index moving-average positions and consecutive-session states.

Counts use exchange sessions, not calendar days. Missing sessions break rolling
history and streaks; equality to a moving average resets both streaks. Index
backcasts before public launch are excluded, including from MA warmup.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quant.features.active_market_value import KEY_INDEX_SPECS

INDEX_MA_STATE_SCHEMA_VERSION = "index_ma_state_v1_20261101"
MA_WINDOWS = (20, 60, 120, 250)
CORE_INDEX_SLUGS = tuple(
    spec.slug for spec in KEY_INDEX_SPECS if spec.slug not in {"star50", "bse50"}
)
# Use first live session conservatively; do not backfill then-unpublished index
# history into pre-launch model rows (SSE/BSE official launch announcements).
PUBLIC_START_DATES = {"star50": "2020-07-23", "bse50": "2022-11-21"}
MA_SUFFIXES = (
    "value", "distance_pct", "slope5_pct", "above", "below",
    "above_days", "below_days", "cross_up", "cross_down",
)
STATE_SUFFIXES = tuple(f"ma{w}_{s}" for w in MA_WINDOWS for s in MA_SUFFIXES) + ("trend_state",)
INDEX_MA_STATE_FEATURE_COLUMNS = tuple(
    f"index_{spec.slug}_{suffix}" for spec in KEY_INDEX_SPECS for suffix in STATE_SUFFIXES
) + tuple(
    f"market_index_ma{w}_{suffix}" for w in MA_WINDOWS for suffix in ("above_ratio", "eligible_count")
)
ANNUAL_MODEL_COLUMNS = tuple(
    f"index_{slug}_ma250_{suffix}" for slug in CORE_INDEX_SLUGS
    for suffix in MA_SUFFIXES if suffix != "value"
) + ("market_index_ma250_above_ratio", "market_index_ma250_eligible_count")
ALL_MA_MODEL_COLUMNS = tuple(
    f"index_{slug}_{suffix}" for slug in CORE_INDEX_SLUGS for suffix in STATE_SUFFIXES
    if not suffix.endswith("_value")
) + tuple(
    f"market_index_ma{w}_{suffix}" for w in MA_WINDOWS for suffix in ("above_ratio", "eligible_count")
)


def _dates(values: pd.Series | Sequence) -> pd.DatetimeIndex:
    text = pd.Series(values).astype(str).str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(text.where(~compact), errors="coerce")
    parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError("index MA dates contain invalid values")
    return pd.DatetimeIndex(parsed).normalize()


def _streak(flag: pd.Series, valid: pd.Series) -> pd.Series:
    active = flag & valid
    groups = active.ne(active.shift(fill_value=False)).cumsum()
    counts = active.groupby(groups).cumsum().astype(float)
    return counts.where(valid)


def compute_index_ma_state_features(
    index_daily: pd.DataFrame,
    *,
    market_calendar: Sequence | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One date-keyed row with 9-index MA states and 7-index consensus.

    Input requires ts_code/trade_date/close (symbol/date aliases accepted).
    Data is available at signal close and intended for next-session decisions.
    Consensus is NaN unless all seven fixed core indices have a valid MA.
    trend_state is +1/-1 for close and MA60/120/250 with rising/falling MA250;
    all other fully observed configurations are 0, and unavailable ones NaN.
    """
    aliases = {}
    if "ts_code" not in index_daily and "symbol" in index_daily:
        aliases["symbol"] = "ts_code"
    if "trade_date" not in index_daily and "date" in index_daily:
        aliases["date"] = "trade_date"
    frame = index_daily.rename(columns=aliases).copy()
    required = {"ts_code", "trade_date", "close"}
    if not required.issubset(frame):
        raise ValueError(f"index MA input missing {sorted(required-set(frame))}")
    frame["date"] = _dates(frame["trade_date"]).to_numpy()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame.loc[~np.isfinite(frame["close"]) | frame["close"].le(0), "close"] = np.nan
    cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else frame["date"].max()
    frame = frame[frame.date.le(cutoff)]
    if frame.duplicated(["ts_code", "date"]).any():
        raise ValueError("index MA input contains duplicate index/date keys")
    if market_calendar is None:
        calendar = pd.DatetimeIndex(frame.date.unique()).sort_values()
    else:
        calendar = _dates(market_calendar)
        if calendar.has_duplicates or not calendar.is_monotonic_increasing:
            raise ValueError("market calendar must be unique and increasing")
        calendar = calendar[calendar <= cutoff]
        if not frame.date.isin(calendar).all():
            raise ValueError("index observations fall outside market calendar")
    output: dict[str, object] = {"date": calendar}
    for spec in KEY_INDEX_SPECS:
        codes = (spec.ts_code, *spec.aliases)
        sub = frame[frame.ts_code.isin(codes)]
        if sub.date.duplicated().any():
            raise ValueError(f"duplicate alias observations for {spec.slug}")
        close = sub.set_index("date")["close"].reindex(calendar)
        if spec.slug in PUBLIC_START_DATES:
            close = close.where(close.index >= pd.Timestamp(PUBLIC_START_DATES[spec.slug]))
        prefix = f"index_{spec.slug}_"
        averages = {}
        for window in MA_WINDOWS:
            ma = close.rolling(window, min_periods=window).mean()
            averages[window] = ma
            valid = close.notna() & ma.notna()
            above, below = close.gt(ma), close.lt(ma)
            previous_valid = valid.shift(1, fill_value=False)
            data = {
                "value": ma,
                "distance_pct": close.div(ma).sub(1).mul(100),
                "slope5_pct": ma.div(ma.shift(5)).sub(1).mul(100),
                "above": above.astype(float).where(valid),
                "below": below.astype(float).where(valid),
                "above_days": _streak(above, valid),
                "below_days": _streak(below, valid),
                "cross_up": (above & close.shift(1).le(ma.shift(1))).astype(float).where(valid & previous_valid),
                "cross_down": (below & close.shift(1).ge(ma.shift(1))).astype(float).where(valid & previous_valid),
            }
            for suffix, values in data.items():
                output[f"{prefix}ma{window}_{suffix}"] = values.to_numpy()
        slope = averages[250].div(averages[250].shift(5)).sub(1)
        valid = close.notna() & slope.notna()
        bull = close.gt(averages[250]) & averages[60].gt(averages[120]) & averages[120].gt(averages[250]) & slope.gt(0)
        bear = close.lt(averages[250]) & averages[60].lt(averages[120]) & averages[120].lt(averages[250]) & slope.lt(0)
        output[f"{prefix}trend_state"] = pd.Series(np.select([bull, bear], [1.0, -1.0], default=0.0), index=calendar).where(valid).to_numpy()
    result = pd.DataFrame(output)
    for window in MA_WINDOWS:
        states = result[[f"index_{slug}_ma{window}_above" for slug in CORE_INDEX_SLUGS]]
        count = states.notna().sum(axis=1)
        result[f"market_index_ma{window}_eligible_count"] = count.astype(float)
        result[f"market_index_ma{window}_above_ratio"] = states.mean(axis=1).where(count.eq(len(CORE_INDEX_SLUGS)))
    return result[["date", *INDEX_MA_STATE_FEATURE_COLUMNS]]
