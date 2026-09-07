"""Finite rule context, applied only after feature histories are computed.

Rows mean valid observations, not calendar days. Recursive indicators and
continuous-price state must be evaluated on the unchanged full input first.
A correction/deletion replaces all outputs from the earliest affected date;
these windows limit *prior context*, never the forward propagation of changes.
Unknown rules fail conservatively to full evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

RULE_WINDOW_VERSION = "rule-tail-v1"


@dataclass(frozen=True)
class RuleHistory:
    previous_rows: int | None
    feature_state: str


# Composed shifts/rolls: B3 needs B2(3) + B1(3) + yin(3);
# washout needs a ten-row prior low followed by three prior washout rows.
B1_RULE_HISTORY = {
    **{name: RuleHistory(6, "full-input base factors and continuous OHLC") for name in (
        "b2_pchg3_vol12", "b2_pchg4_vol15", "b2_pchg5_vol15",
    )},
    "b2_any_pchg4_vol15": RuleHistory(0, "full-input base factors"),
    "b2_oversold_pchg3_vol12": RuleHistory(5, "full-input KDJ"),
    "b2_bbi_reclaim_vol12": RuleHistory(1, "full-input BBI and continuous OHLC"),
    **{name: RuleHistory(9, "full-input base factors and MA60") for name in (
        "b3_small_pos_amp7", "b3_small_pos_amp5", "b3_calm_pullback",
        "b3_broad_small_pos", "b3_broad_calm_pullback",
    )},
    **{name: RuleHistory(3, "full-input base factors and continuous OHLC") for name in (
        "sb1_range10_vol12", "sb1_range7_vol15", "sb1_range5_vol15_j10",
    )},
    **{name: RuleHistory(13, "full-input base factors and continuous OHLC") for name in (
        "super_washout_vol12", "super_washout_vol15_j0", "super_washout_j10_closepos40",
    )},
    "signal_vegas_tunnel": RuleHistory(None, "full-input EMA and whole-series percentile ranks"),
    "signal_tvb_merged": RuleHistory(None, "full-input anchor state; not optimized here"),
}

# Five prior RAW flags suppress repeated signals, so add that context after
# composing each underlying rule's rolling/shift dependencies.
Z_RULE_HISTORY = {
    name: RuleHistory(rows + 5, "full normalized-input Z EMA/KDJ and rolling features")
    for name, rows in {
        "CHANGAN": 2, "PINGHANG": 8, "DOUBLE_GUN": 8,
        "YIDONG_DILIAN": 59, "NANA": 15, "GOLDEN_BOWL": 0,
        "BREATHING": 6, "KENGQI": 32, "DUICHEN_VA": 21,
        "ZAIHOU": 20, "YUEYUE": 29, "KEY_K": 20, "VIOLENCE_K": 20,
    }.items()
}


def rule_tail_start(
    frame: pd.DataFrame,
    output_start: str | pd.Timestamp | None,
    rules: Iterable[str],
    contracts: Mapping[str, RuleHistory],
) -> int:
    """Return a safe positional start, or zero without a finite contract."""
    if output_start is None or frame.empty:
        return 0
    histories = [contracts.get(rule) for rule in rules]
    if not histories or any(item is None or item.previous_rows is None for item in histories):
        return 0
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or not dates.is_monotonic_increasing or not dates.is_unique:
        return 0
    start = pd.Timestamp(output_start)
    if pd.isna(start):
        raise ValueError("output_start must be a valid date")
    first = int(dates.searchsorted(start))
    context = max(item.previous_rows for item in histories)
    return max(0, first - context)
