"""B1/B2/B3/SB1/Super-B1 signal family.

The rules in this module translate the B1 family into daily-data predicates
that can be backtested without minute/L2 data. They are strategy-family
signals, not production trading advice by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.features.daily_factor_layer import attach_daily_base_factors
from quant.features.variable_library import build_continuous_ohlc


@dataclass(frozen=True)
class B1FamilySignalSpec:
    name: str
    description: str
    column: str


B1_FAMILY_SIGNAL_SPECS = [
    B1FamilySignalSpec("B1", "J<=-10、缩量回调、未连续四阴", "signal_b1"),
    B1FamilySignalSpec("B2", "B1 后 3 日内放量长阳确认", "signal_b2"),
    B1FamilySignalSpec("B3", "B2 后分歧转一致小阳/十字星", "signal_b3"),
    B1FamilySignalSpec("SB1", "横盘后放量下破洗盘，次日观察", "signal_sb1"),
    B1FamilySignalSpec("SUPER_B1", "放量下杀后缩量企稳且 J 仍为负", "signal_super_b1"),
]


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    if "volume" not in out.columns and "vol" in out.columns:
        out = out.rename(columns={"vol": "volume"})
    needed = {
        "kdj_d_j",
        "bbi",
        "volume_relative_5d",
        "volume_relative_20d",
        "post_yidong_shrink",
        "s1_distribution",
    }
    if not needed <= set(out.columns):
        symbol = ""
        for column in ("ts_code", "symbol"):
            if column in out.columns and out[column].notna().any():
                symbol = str(out.loc[out[column].notna(), column].iloc[-1])
                break
        out = attach_daily_base_factors(
            out,
            symbol=symbol,
            compute_if_missing=True,
            persist_missing=False,
        )
    return out


def add_b1_family_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add zettaranc-inspired B1/B2/B3/SB1/Super-B1 signal columns."""
    out = _ensure_features(df).sort_values("date").reset_index(drop=True)
    price = build_continuous_ohlc(out)
    close = price["close"]
    open_ = price["open"]
    high = price["high"]
    low = price["low"]
    pct_chg = out["pct_chg"] if "pct_chg" in out.columns else close.pct_change() * 100
    amplitude = (high - low) / close.shift(1).replace(0, np.nan) * 100
    is_yinxian = close < open_
    recent_yin_count = is_yinxian.rolling(4, min_periods=1).sum()
    if "ma_60" not in out.columns:
        out["ma_60"] = close.rolling(60, min_periods=20).mean()
    support_ok = (close >= out["bbi"] * 0.97) | (close >= out["ma_60"] * 0.97)

    out["signal_b1"] = (
        (out["kdj_d_j"] <= -10)
        & (out["volume_relative_20d"] < 1.0)
        & (recent_yin_count < 4)
        & support_ok
    ).astype(int)

    b1_recent = out["signal_b1"].shift(1).rolling(3, min_periods=1).max() > 0
    close_pos = (close - low) / (high - low).replace(0, np.nan)
    pre_oversold_recent = out["kdj_d_j"].lt(0).shift(1).rolling(5, min_periods=1).max() > 0
    b2_from_b1 = (
        b1_recent
        & (pct_chg >= 4)
        & (out["volume_relative_5d"] >= 1.5)
        & (high <= close * 1.01)
        & (out["kdj_d_j"] < 55)
    )
    b2_any = (
        (pct_chg >= 4)
        & (out["volume_relative_5d"] >= 1.5)
        & (close_pos >= 0.75)
        & (out["kdj_d_j"] < 80)
    )
    b2_oversold = (
        pre_oversold_recent
        & (pct_chg >= 3)
        & (out["volume_relative_5d"] >= 1.2)
        & (close_pos >= 0.70)
        & (out["kdj_d_j"] < 80)
    )
    b2_bbi_reclaim = (
        (close.shift(1) <= out["bbi"].shift(1) * 1.01)
        & (close > out["bbi"])
        & (pct_chg >= 3)
        & (out["volume_relative_5d"] >= 1.2)
        & (close_pos >= 0.65)
        & (out["kdj_d_j"] < 80)
    )
    out["signal_b2_from_b1"] = b2_from_b1.astype(int)
    out["signal_b2_any"] = b2_any.astype(int)
    out["signal_b2_oversold"] = b2_oversold.astype(int)
    out["signal_b2_bbi_reclaim"] = b2_bbi_reclaim.astype(int)
    out["signal_b2"] = (b2_from_b1 | b2_any | b2_oversold | b2_bbi_reclaim).astype(int)

    b2_recent = (b2_from_b1 | b2_any | b2_oversold | b2_bbi_reclaim).shift(1).rolling(3, min_periods=1).max() > 0
    b3_small_pos = (
        b2_recent
        & (pct_chg > 0)
        & (pct_chg < 2)
        & (amplitude < 7)
        & (close_pos >= 0.50)
    )
    b3_calm_pullback = (
        b2_recent
        & (pct_chg >= -1)
        & (pct_chg < 2)
        & (amplitude < 7)
        & (out["volume_relative_5d"] <= 1.3)
    )
    out["signal_b3_small_pos"] = b3_small_pos.astype(int)
    out["signal_b3_calm_pullback"] = b3_calm_pullback.astype(int)
    out["signal_b3"] = (b3_small_pos | b3_calm_pullback).astype(int)

    sideways_range = (
        high.shift(1).rolling(3, min_periods=3).max()
        / low.shift(1).rolling(3, min_periods=3).min().replace(0, np.nan)
        - 1
    )
    out["signal_sb1"] = (
        (sideways_range < 0.07)
        & is_yinxian
        & (out["volume_relative_5d"] >= 1.5)
        & (low < low.shift(1).rolling(3, min_periods=3).min())
        & (out["kdj_d_j"] < 0)
    ).astype(int)

    washout = (
        is_yinxian
        & (pct_chg > -9.5)
        & (out["volume_relative_5d"] >= 1.5)
        & (low < low.shift(1).rolling(10, min_periods=5).min())
    )
    washout_recent = washout.shift(1).rolling(3, min_periods=1).max() > 0
    small_reversal = (
        (amplitude < 7)
        & (pct_chg > -2)
        & (pct_chg < 2.5)
        & (out["volume_relative_20d"] < 0.9)
    )
    out["signal_super_b1"] = (
        washout_recent
        & small_reversal
        & (out["kdj_d_j"] < 0)
    ).astype(int)

    return out


def summarize_b1_family_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return simple counts for the B1 family signals."""
    out = add_b1_family_signals(df)
    rows = []
    for spec in B1_FAMILY_SIGNAL_SPECS:
        rows.append({
            "signal": spec.name,
            "description": spec.description,
            "count": int(out[spec.column].sum()) if spec.column in out.columns else 0,
        })
    return pd.DataFrame(rows)
