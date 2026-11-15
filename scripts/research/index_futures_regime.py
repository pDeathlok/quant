"""Causal spot-index trend states; no future turning-point labels."""
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = {"IF": "000300.SH", "IH": "000016.SH", "IC": "000905.SH", "IM": "000852.SH"}
MODES = ("none", "entry", "exit", "both")


def trend_states(bars: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """State is known at this row's close; consumers execute next session."""
    bars = bars.sort_values("trade_date").copy()
    if bars.trade_date.duplicated().any():
        raise ValueError("Duplicate index dates")
    previous = bars.close.shift()
    tr = pd.concat([bars.high-bars.low, (bars.high-previous).abs(),
                    (bars.low-previous).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20, min_periods=20).mean()
    ma = bars.close.rolling(window, min_periods=window).mean()
    slope = ma-ma.shift(20)
    ready = ma.notna() & slope.notna() & atr.gt(0)
    state = pd.Series(np.nan, index=bars.index)
    state.loc[ready] = 0
    state.loc[ready & bars.close.gt(ma+atr) & slope.gt(0)] = 1
    state.loc[ready & bars.close.lt(ma-atr) & slope.lt(0)] = -1
    return pd.DataFrame(dict(trade_date=bars.trade_date, close=bars.close, ma=ma,
                             atr20=atr, slope20=slope, distance_atr=(bars.close-ma)/atr,
                             state=state))


def load_states(window: int = 200, end: str = "20261106") -> pd.DataFrame:
    parts = []
    for symbol, code in TARGETS.items():
        bars = pd.read_parquet(Path("data/raw")/f"index_{code}.parquet")
        bars = bars[bars.trade_date.le(end)]
        parts.append(trend_states(bars, window).assign(symbol=symbol, index_code=code))
    return pd.concat(parts, ignore_index=True)


def entry_allowed(state: int, direction: int) -> bool:
    return state != -direction


def exit_required(state: int, direction: int, continuation: float) -> bool:
    return state != direction and np.isfinite(continuation) and continuation <= 0
