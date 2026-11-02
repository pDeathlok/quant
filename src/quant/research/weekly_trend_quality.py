"""Point-in-time weekly trend geometry, stages, and fixed interpretable definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.kronos_multiscale import period_bars

QUALITY_FEATURES = (
    "w_close",
    "w_ema26",
    "w_ema52",
    "w_ema26_slope4",
    "w_ema52_slope4",
    "w_lr13_slope",
    "w_lr13_r2",
    "w_lr26_slope",
    "w_lr26_r2",
    "w_efficiency13",
    "w_above26_share13",
    "w_higher_high4",
    "w_higher_low4",
    "w_drawdown13",
    "w_drawdown26",
    "w_extension26",
    "w_extension_atr",
    "w_return4",
    "w_return13",
    "w_return26",
    "w_high13",
    "w_rs13_percentile",
    "w_relative_return13",
)
DEVELOPMENT_PERIODS = ("development_2024", "development_2025H1", "selection")


def lifecycle_features(frame: pd.DataFrame) -> pd.DataFrame:
    """One stock: causal age and run-up since the current trend episode began."""
    frame = frame.sort_values("available_at").copy()
    base = frame.legacy_boll_up & frame.w_close.gt(frame.w_ema26) & frame.w_ema26_slope4.gt(0)
    episode = base.ne(base.shift(fill_value=False)).cumsum()
    starts = base & ~base.shift(fill_value=False)
    anchor = frame.w_close.shift(1).where(starts).ffill()
    peak = frame.high.groupby(episode).cummax()
    frame["trend_age_weeks"] = base.astype(int).groupby(episode).cumsum()
    frame["episode_runup"] = (frame.w_close / anchor - 1).where(base)
    frame["episode_drawdown"] = (frame.w_close / peak - 1).where(base)
    return frame


def log_regression(close: pd.Series, length: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.log(close.to_numpy(float))
    x = np.arange(length, dtype=float)
    sxx = float(((x - x.mean()) ** 2).sum())
    mean = pd.Series(values).rolling(length).mean().to_numpy()
    syy = pd.Series(values).rolling(length).var(ddof=0).to_numpy() * length
    weighted = np.full(len(close), np.nan)
    if len(close) >= length:
        weighted[length - 1 :] = np.convolve(values, x[::-1], mode="valid")
    sxy = weighted - x.sum() * mean
    slope = np.expm1(sxy / sxx)
    r2 = np.clip(np.divide(sxy * sxy, sxx * syy, out=np.zeros(len(close)), where=syy > 1e-15), 0, 1)
    r2[~np.isfinite(mean)] = np.nan
    return slope, r2


def quality_features(weekly: pd.DataFrame) -> pd.DataFrame:
    frame = weekly.sort_values("available_at").reset_index(drop=True).copy()
    close = frame.close
    ema10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    ema52 = close.ewm(span=52, adjust=False, min_periods=52).mean()
    dif = close.ewm(span=12, adjust=False, min_periods=12).mean() - ema26
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    mid = close.rolling(20).mean()
    frame["legacy_macd_boll"] = dif.gt(dea) & dif.gt(0) & close.gt(mid) & mid.gt(mid.shift(4))
    frame["legacy_boll_up"] = close.gt(mid) & mid.gt(mid.shift(4))
    frame["w_close"] = close
    frame["w_ema26"] = ema26
    frame["w_ema52"] = ema52
    frame["w_ema26_slope4"] = ema26 / ema26.shift(4) - 1
    frame["w_ema52_slope4"] = ema52 / ema52.shift(4) - 1
    for length in (13, 26):
        slope, r2 = log_regression(close, length)
        frame[f"w_lr{length}_slope"] = slope
        frame[f"w_lr{length}_r2"] = r2
    frame["w_efficiency13"] = (close - close.shift(13)).abs() / close.diff().abs().rolling(
        13
    ).sum().replace(0, np.nan)
    frame["w_above26_share13"] = (
        close.gt(ema26).astype(float).where(ema26.notna()).rolling(13).mean()
    )
    frame["w_higher_high4"] = frame.high.rolling(4).max().gt(frame.high.shift(4).rolling(4).max())
    frame["w_higher_low4"] = frame.low.rolling(4).min().gt(frame.low.shift(4).rolling(4).min())
    frame["w_high13"] = frame.high.rolling(13).max()
    frame["w_drawdown13"] = close / frame.w_high13 - 1
    frame["w_drawdown26"] = close / frame.high.rolling(26).max() - 1
    frame["w_extension26"] = close / ema26 - 1
    tr = pd.concat(
        [
            frame.high - frame.low,
            (frame.high - close.shift(1)).abs(),
            (frame.low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["w_extension_atr"] = (close - ema26) / tr.rolling(13).mean().replace(0, np.nan)
    for n in (4, 13, 26):
        frame[f"w_return{n}"] = close.pct_change(n, fill_method=None)
    long_base = close.gt(ema26) & ema26.gt(ema52) & frame.w_ema52_slope4.gt(0)
    transition = close.gt(ema26) & frame.w_ema26_slope4.gt(0) & ~long_base
    rebound = close.gt(ema10) & ~long_base & ~transition
    extended = long_base & (frame.w_extension26.gt(0.25) | frame.w_extension_atr.gt(4))
    fading = long_base & (
        frame.w_lr13_slope.le(0) | frame.w_lr13_r2.lt(0.2) | ema10.lt(ema10.shift(4))
    )
    pullback = (
        long_base & ~extended & ~fading & frame.w_return4.le(0) & frame.w_drawdown13.ge(-0.15)
    )
    frame["trend_stage"] = np.select(
        [rebound, transition, long_base, extended, fading, pullback],
        ["rebound", "transition", "established", "extended", "fading", "pullback"],
        default="other",
    )
    # np.select takes its FIRST match; apply the more specific long-trend stages last.
    frame.loc[extended, "trend_stage"] = "extended"
    frame.loc[fading, "trend_stage"] = "fading"
    frame.loc[pullback, "trend_stage"] = "pullback"
    return frame


def from_daily(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.sort_values("date").copy()
    if frame.date.duplicated().any():
        raise ValueError("duplicate daily observations")
    frame = frame[frame[["open", "high", "low", "close", "pre_close"]].gt(0).all(axis=1)]
    close = np.exp(np.log(frame.close / frame.pre_close).cumsum()) * 100
    daily = frame[["date", "open", "high", "low", "close"]].copy()
    daily[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].mul(
        close / frame.close, axis=0
    )
    daily["volume"] = frame.vol.to_numpy()
    return quality_features(period_bars(daily, "weekly"))


def cross_sectional_strength(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["w_rs13_percentile"] = frame.groupby("available_at").w_return13.rank(pct=True)
    frame["w_relative_return13"] = frame.w_return13 - frame.groupby(
        "available_at"
    ).w_return13.transform("median")
    return frame


def trend_rules(frame: pd.DataFrame) -> dict[str, pd.Series]:
    long_base = (
        frame.w_close.gt(frame.w_ema26)
        & frame.w_ema26.gt(frame.w_ema52)
        & frame.w_ema52_slope4.gt(0)
    )
    smooth = frame.w_lr13_slope.gt(0) & frame.w_lr13_r2.ge(0.4)
    healthy = (
        frame.w_extension26.le(0.25) & frame.w_drawdown13.ge(-0.15) & frame.w_ema26_slope4.gt(0)
    )
    structure = frame.w_higher_high4 & frame.w_higher_low4
    rs = frame.w_rs13_percentile.ge(0.6)
    early = frame.w_close.gt(frame.w_ema26) & frame.w_ema26_slope4.gt(0.01) & smooth
    old = frame.legacy_macd_boll
    masks = {
        "legacy_macd_boll": old,
        "legacy_boll_up": frame.legacy_boll_up,
        "long52": long_base,
        "long52_slope2": long_base & frame.w_ema26_slope4.ge(0.02),
        "long52_persistent70": long_base & frame.w_above26_share13.ge(0.7),
        "long52_r2_40": long_base & smooth,
        "long52_r2_60": long_base & frame.w_lr13_slope.gt(0) & frame.w_lr13_r2.ge(0.6),
        "long52_efficiency30": long_base & frame.w_efficiency13.ge(0.3),
        "long52_efficiency50": long_base & frame.w_efficiency13.ge(0.5),
        "long52_structure": long_base & structure,
        "long52_rs60": long_base & rs,
        "long52_rs80": long_base & frame.w_rs13_percentile.ge(0.8),
        "long52_rs60_smooth": long_base & rs & smooth,
        "long52_healthy": long_base & healthy,
        "long52_healthy_rs60": long_base & healthy & rs,
        "long52_pullback": long_base & healthy & frame.w_return4.le(0),
        "early_transition": frame.w_close.gt(frame.w_ema26)
        & frame.w_ema26_slope4.gt(0)
        & frame.w_ema52_slope4.le(0)
        & frame.w_higher_low4
        & frame.w_return4.gt(0),
        "early_confirmed": early,
        "early_confirmed_rs60": early & rs,
        "macd_boll_healthy": old & healthy,
        "macd_boll_persistent70": old & frame.w_above26_share13.ge(0.7),
        "macd_boll_smooth": old & smooth,
        "macd_boll_rs60": old & rs,
        "macd_boll_structure": old & structure,
    }
    if "current_close" in frame:
        live = frame.current_close.ge(frame.w_ema26) & frame.current_close.ge(frame.w_high13 * 0.85)
        masks["macd_boll_live_guard"] = old & live
        masks["long52_live_guard"] = long_base & live
    if "trend_age_weeks" in frame:
        age = frame.trend_age_weeks
        fresh = age.between(2, 13)
        masks.update(
            {
                "age_2_8": age.between(2, 8),
                "age_2_13": fresh,
                "age_4_20": age.between(4, 20),
                "age_2_13_runup40": fresh & frame.episode_runup.le(0.4),
                "age_2_13_structure": fresh & structure,
                "age_2_13_smooth": fresh & smooth,
                "age_2_13_retest": fresh & frame.episode_drawdown.between(-0.12, -0.02),
            }
        )
        if "current_close" in frame:
            masks["age_2_13_live_guard"] = fresh & live
    return {key: value.fillna(False).astype(bool) for key, value in masks.items()}


def add_outcomes(frame: pd.DataFrame, market_weeks: pd.DatetimeIndex) -> pd.DataFrame:
    full = frame.set_index("available_at").reindex(market_weeks)
    output = frame.copy().set_index("available_at")
    for n in (4, 8):
        lows = pd.concat([full.low.shift(-i) for i in range(1, n + 1)], axis=1)
        volume = pd.concat([full.volume.shift(-i) for i in range(1, n + 1)], axis=1)
        end = pd.Series(market_weeks, index=market_weeks).shift(-n)
        valid = lows.notna().all(axis=1) & volume.gt(0).all(axis=1) & full.close.notna()
        ret = (full.close.shift(-n) / full.close - 1).where(valid)
        mae = (lows.min(axis=1) / full.close - 1).where(valid)
        output[f"future_return{n}"] = ret.reindex(output.index)
        output[f"future_mae{n}"] = mae.reindex(output.index)
        output[f"label_end{n}"] = end.reindex(output.index).where(valid.reindex(output.index))
        output[f"continued_up{n}"] = (
            (ret.gt(0) & mae.ge(-0.08)).astype(float).where(valid).reindex(output.index)
        )
    return output.reset_index()


def shortlist(records: list[dict], count: int = 6) -> list[str]:
    def key(record):
        parts = [record[p] for p in DEVELOPMENT_PERIODS]
        if min(x["trades"] for x in parts) < 100:
            return (-float("inf"), -float("inf"), -float("inf"))
        return (
            min(x["average_net_return"] for x in parts),
            min(x["win_rate"] for x in parts),
            sum(x["average_net_return"] for x in parts),
        )

    ordered = sorted(records, key=key, reverse=True)
    return list(
        dict.fromkeys([r["rule"] for r in ordered[:count]] + ["legacy_macd_boll", "legacy_boll_up"])
    )
