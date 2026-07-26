#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backtest z-skill daily pattern tactics into operational playbooks.

The first pass keeps one daily-rule signal per tactic, then crosses:
- T+1 open filters
- fixed take-profit/stop-loss exits
- trailing exits after reaching a target
- expiry exits

Signal date is T+0, entry is T+1 open, exits are evaluated from T+2 onward.
The non-overlap policy keeps the first trade per stock until the previous trade
has exited, which matches the selector's practical "no repeated buy while held"
assumption.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_b1_entry_exit_grid import ExitRule, add_future_prices, simulate_exit, summarize_returns
from analyze_b1_xgb_entry_exit_grid import DEFAULT_DAILY_DIR, DEFAULT_OUTPUT_DIR, drop_overlapping_trades
from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.data.atomic_io import atomic_link_or_copy, atomic_write_parquet
from quant.features.daily_factor_layer import Z_CONSUMER_ALIASES, attach_z_skill_base_factors
from quant.features.variable_library import build_continuous_ohlc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_CACHE = PROJECT_ROOT / "data/features/z_skill_daily_candidates.parquet"


@dataclass(frozen=True)
class SignalSpec:
    key: str
    name: str
    timing: str
    description: str


@dataclass(frozen=True)
class OpenFilter:
    name: str
    description: str
    min_gap_pct: float | None = None
    max_gap_pct: float | None = None
    min_close_pos: float | None = None


def build_signal_specs() -> list[SignalSpec]:
    return [
        SignalSpec("CHANGAN", "长安战法", "日线三日确认，T+1 开盘观察", "B1痕迹后放量长阳，第三日缩量小阳分歧转一致。"),
        SignalSpec("PINGHANG", "平行重炮", "日线右侧确认，T+1 开盘观察", "两根放量阳线夹缩量整理，第二炮强确认。"),
        SignalSpec("DOUBLE_GUN", "双枪战法", "日线右侧确认，T+1 开盘观察", "两根放量阳线，中间缩量整理，第二枪前有低位痕迹。"),
        SignalSpec("YIDONG_DILIAN", "异动地量", "日线低吸观察，T+1 开盘/回踩观察", "前期放量上涨后缩量回调到低量、低位区域。"),
        SignalSpec("NANA", "娜娜图形", "日线回调观察，T+1 开盘/回踩观察", "连续放量上涨后无巨量阴线，随后缩量回调且 J 低位。"),
        SignalSpec("GOLDEN_BOWL", "黄金碗", "日线支撑观察，T+1 不破黄线观察", "白线强于黄线，价格回到碗底附近。"),
        SignalSpec("BREATHING", "呼吸结构", "日线节奏观察，T+1 回踩观察", "放量涨、缩量跌、再放量涨，低点不破。"),
        SignalSpec("KENGQI", "坑里起好货", "日线填坑观察，T+1 回踩观察", "放量挖坑后缩量填坑，回到坑沿附近。"),
        SignalSpec("DUICHEN_VA", "对称VA", "日线企稳观察，T+1 低吸观察", "上涨后的回落在时间/空间上接近对称，低位缩量企稳。"),
        SignalSpec("ZAIHOU", "灾后重建", "日线回踩观察，T+1 不破BBI观察", "放量大阳后缩量回踩上行 BBI。"),
        SignalSpec("YUEYUE", "跃跃欲试", "日线平台观察，优先等突破", "横盘平台内多次巨量试盘，阳线占比更高。"),
        SignalSpec("KEY_K", "关键K", "日线关键位，T+1 实体中位观察", "关键位置出现放量强阳 K。"),
        SignalSpec("VIOLENCE_K", "暴力K", "日线底部异动，T+1 回踩观察", "底部区域倍量大实体 K，资金强行改变节奏。"),
    ]


def build_open_filters() -> list[OpenFilter]:
    return [
        OpenFilter("all", "不限制 T+1 开盘涨跌幅"),
        OpenFilter("gap_le_1", "T+1 开盘涨幅 <=1%", None, 1.0),
        OpenFilter("gap_0_to_2", "T+1 开盘 0%-2%", 0.0, 2.0),
        OpenFilter("gap_-1_to_2", "T+1 开盘 -1%-2%", -1.0, 2.0),
        OpenFilter("gap_-2_to_1", "T+1 开盘 -2%-1%", -2.0, 1.0),
        OpenFilter("gap_le_2_strong_close", "T+1 开盘涨幅<=2%，信号日收盘位置>=70%", None, 2.0, 0.70),
    ]


def build_exit_rules() -> list[ExitRule]:
    rules: list[ExitRule] = []
    for label, hold_days in [("T3", 2), ("T5", 4), ("T7", 6)]:
        rules.append(ExitRule(f"expiry_{label}_close", "expiry", hold_days))
    for label, hold_days in [("T5", 4), ("T7", 6)]:
        for tp in [0.03, 0.04, 0.06, 0.08]:
            for sl in [0.01, 0.015, 0.02]:
                for trigger in ["intraday", "close"]:
                    rules.append(ExitRule(f"fixed_tp{tp:.1%}_sl{sl:.1%}_{trigger}_{label}", "fixed", hold_days, tp, sl, stop_trigger=trigger))
        for target in [0.03, 0.04, 0.06]:
            for trail in [0.015, 0.02]:
                for sl in [0.01, 0.015]:
                    for trigger in ["intraday", "close"]:
                        rules.append(
                            ExitRule(
                                f"trail_target{target:.1%}_dd{trail:.1%}_sl{sl:.1%}_{trigger}_{label}",
                                "trailing",
                                hold_days,
                                target,
                                sl,
                                trail,
                                trigger,
                            )
                        )
    return rules


def _normalize_daily(
    path: Path,
    start_date: str,
    source_frame: pd.DataFrame | None = None,
    *,
    factors_attached: bool = False,
) -> pd.DataFrame:
    df = source_frame.copy() if source_frame is not None else pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "trade_date" in out.columns:
        trade_date = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        out["date"] = trade_date
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    elif "volume" in out.columns and "vol" in out.columns:
        out["volume"] = out["volume"].fillna(out["vol"])
    if "ts_code" not in out.columns:
        out["ts_code"] = path.stem
    out["symbol"] = out["ts_code"].fillna(path.stem).astype(str)
    if "name" not in out.columns:
        out["name"] = ""
    needed = ["date", "open", "high", "low", "close", "volume"]
    if not set(needed) <= set(out.columns):
        return pd.DataFrame()
    out = out.dropna(subset=needed).sort_values("date").reset_index(drop=True)
    history_start = pd.to_datetime(start_date) - pd.Timedelta(days=450)
    out = out[out["date"] >= history_start].reset_index(drop=True)
    if len(out) < 130:
        return pd.DataFrame()
    st_mask = out["name"].fillna("").astype(str).str.upper().str.contains("ST") | out["name"].fillna("").astype(str).str.contains("退")
    out = out[~st_mask].reset_index(drop=True)
    if len(out) < 130:
        return pd.DataFrame()
    symbol = str(out["ts_code"].dropna().iloc[-1]) if out["ts_code"].notna().any() else path.stem
    if factors_attached:
        for target, source in Z_CONSUMER_ALIASES.items():
            if source in out.columns:
                out[target] = out[source]
    else:
        out = attach_z_skill_base_factors(out, symbol=symbol, persist_missing=False)
    price = build_continuous_ohlc(out)
    for col in ["open", "high", "low", "close"]:
        out[col] = price[col]
    out["pre_close"] = out["close"].shift(1)
    out["pct_chg"] = out["close"].pct_change() * 100
    out["pct_chg"] = out["pct_chg"].fillna(0)
    if "amplitude" not in out.columns:
        out["amplitude"] = (out["high"] - out["low"]) / out["pre_close"].replace(0, np.nan) * 100
    if "close_pos" not in out.columns:
        out["close_pos"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)
    if "vol_ratio_prev" not in out.columns:
        out["vol_ratio_prev"] = out["volume"] / out["volume"].shift(1).replace(0, np.nan)
    out["vol_ma5_prev"] = out["volume"].shift(1).rolling(5, min_periods=2).mean()
    if "vol_ratio_5" not in out.columns:
        out["vol_ratio_5"] = out["volume"] / out["vol_ma5_prev"].replace(0, np.nan)
    if "vol_ma10" not in out.columns:
        out["vol_ma10"] = out["volume"].rolling(10, min_periods=3).mean()
    if "vol_ma20" not in out.columns:
        out["vol_ma20"] = out["volume"].rolling(20, min_periods=5).mean()
    if "is_rise" not in out.columns:
        out["is_rise"] = out["close"] > out["open"]
    if "is_big_yin" not in out.columns:
        out["is_big_yin"] = (out["close"] < out["open"]) & (out["vol_ratio_5"] >= 1.5) & (out["pct_chg"] <= -2)
    for window, min_periods in ((3, 1), (6, 2), (12, 4), (24, 8)):
        if f"ma{window}" not in out.columns:
            out[f"ma{window}"] = out["close"].rolling(window, min_periods=min_periods).mean()
    if "bbi" not in out.columns:
        out["bbi"] = (out["ma3"] + out["ma6"] + out["ma12"] + out["ma24"]) / 4
    out["zg_white"] = out["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    if "dg_yellow" not in out.columns:
        out["dg_yellow"] = (
            out["close"].rolling(14, min_periods=8).mean()
            + out["close"].rolling(28, min_periods=14).mean()
            + out["close"].rolling(57, min_periods=28).mean()
            + out["close"].rolling(114, min_periods=60).mean()
        ) / 4
    out["kdj_j"] = _calculate_kdj_j(out)
    return out


def _calculate_kdj_j(df: pd.DataFrame) -> pd.Series:
    low9 = df["low"].rolling(9, min_periods=3).min()
    high9 = df["high"].rolling(9, min_periods=3).max()
    rsv = (df["close"] - low9) / (high9 - low9).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    return 3 * k - 2 * d


def _two_yang_pattern(df: pd.DataFrame, strict_pinghang: bool) -> pd.Series:
    signal = np.zeros(len(df), dtype=bool)
    yang = (df["is_rise"] & (df["pct_chg"] >= 3) & (df["vol_ratio_5"] >= 1.5)).to_numpy()
    for pos in range(8, len(df)):
        recent = [idx for idx in range(max(0, pos - 7), pos + 1) if yang[idx]]
        if len(recent) < 2 or recent[-1] != pos:
            continue
        y1, y2 = recent[-2], recent[-1]
        middle = df.iloc[y1 + 1 : y2]
        if len(middle) < 2:
            continue
        yin_count = int((middle["close"] <= middle["open"]).sum())
        max_mid_vol = float(middle["volume"].max())
        if strict_pinghang:
            ok = (
                yin_count >= len(middle) * 0.5
                and df.iloc[y1]["volume"] >= max_mid_vol * 1.15
                and df.iloc[y2]["volume"] >= max_mid_vol * 1.15
                and df.iloc[y2]["volume"] >= df.iloc[y1]["volume"] * 0.9
                and df.iloc[y2]["pct_chg"] >= 4
                and df.iloc[y2]["kdj_j"] < 55
            )
        else:
            avg_mid_ratio = float(middle["vol_ratio_prev"].mean())
            j_before = float(df.iloc[y2 - 1]["kdj_j"]) if y2 > 0 else 99
            ok = (
                3 <= y2 - y1 <= 10
                and avg_mid_ratio < 1.2
                and j_before < 20
                and df.iloc[y2]["vol_ratio_prev"] >= 1.8
            )
        signal[pos] = bool(ok)
    return pd.Series(signal, index=df.index)


def compute_z_skill_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct = out["pct_chg"]
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]
    close_pos = out["close_pos"]
    j = out["kdj_j"]

    flags = pd.DataFrame(index=out.index)
    flags["CHANGAN"] = (
        (j.shift(2) < -13)
        & (pct.shift(1) >= 4)
        & out["is_rise"].shift(1).fillna(False)
        & (out["vol_ratio_5"].shift(1) >= 1.4)
        & (j.shift(1) > j.shift(2))
        & (pct > 0)
        & (pct < 2.2)
        & (out["amplitude"] < 7)
        & (volume <= volume.shift(1) * 0.55)
    )
    flags["PINGHANG"] = _two_yang_pattern(out, strict_pinghang=True)
    flags["DOUBLE_GUN"] = _two_yang_pattern(out, strict_pinghang=False)

    yidong = out["is_rise"] & (pct >= 2.5) & (out["vol_ratio_5"] >= 1.8)
    yidong_recent = yidong.shift(2).rolling(9, min_periods=1).max().fillna(False).astype(bool)
    yidong_vol = volume.where(yidong).shift(1).rolling(10, min_periods=1).max()
    shrink_ok = volume <= yidong_vol * 0.75
    ground_ok = volume <= volume.rolling(60, min_periods=20).quantile(0.25)
    low_absorb = (pct >= -2.5) & (pct <= 1.0) & (close <= out["bbi"] * 1.01)
    flags["YIDONG_DILIAN"] = yidong_recent & (shrink_ok | ground_ok) & low_absorb & (j < 25)

    build_rise = ((out["is_rise"]) & (volume > volume.shift(1))).rolling(8, min_periods=1).sum()
    shrink_count = (volume < volume.shift(1)).rolling(7, min_periods=1).sum()
    big_yin_recent = out["is_big_yin"].rolling(15, min_periods=1).max().fillna(False).astype(bool)
    flags["NANA"] = (build_rise.shift(7) >= 4) & (shrink_count >= 3) & ~big_yin_recent & low_absorb & (j < 8)

    white = out["zg_white"]
    yellow = out["dg_yellow"]
    near_bowl = (yellow <= close) & (close <= (white + yellow) / 2) & (((close - yellow) / yellow) <= 0.04)
    flags["GOLDEN_BOWL"] = (white > yellow * 1.005) & near_bowl & (j < 80)

    phase_exhale = (pct > 0) & (out["vol_ratio_prev"] > 1)
    phase_inhale = (pct < 0) & (out["vol_ratio_prev"] < 1)
    n_type = low > low.shift(3).rolling(2, min_periods=1).min() * 0.98
    flags["BREATHING"] = (
        phase_exhale
        & (phase_exhale.rolling(7, min_periods=1).sum() >= 2)
        & (phase_inhale.rolling(7, min_periods=1).sum() >= 2)
        & n_type
        & (pct >= 2)
        & (out["vol_ratio_prev"] >= 1.5)
        & (close_pos >= 0.75)
    )

    pre_high = high.shift(13).rolling(5, min_periods=3).max()
    pit_low = low.rolling(18, min_periods=10).min()
    pit_idx_depth = (pre_high - pit_low) / pre_high
    fill_ratio = (close - pit_low) / (pre_high - pit_low).replace(0, np.nan)
    pit_day = (low <= low.shift(1).rolling(18, min_periods=8).min()) & (close < out["open"]) & (out["vol_ratio_prev"] >= 1.25)
    pit_recent = pit_day.shift(3).rolling(12, min_periods=1).max().fillna(False).astype(bool)
    post_vol = volume.rolling(5, min_periods=2).mean()
    pre_vol = volume.shift(10).rolling(5, min_periods=2).mean()
    flags["KENGQI"] = (pit_idx_depth >= 0.12) & pit_recent & (fill_ratio >= 0.78) & (fill_ratio <= 1.12) & (post_vol < pre_vol * 0.8) & (pct <= 3)

    peak20 = high.rolling(22, min_periods=12).max()
    trough20 = low.rolling(22, min_periods=12).min()
    up_pct = (peak20 - trough20) / trough20
    down_pct = (peak20 - close) / peak20
    space_sym = down_pct / up_pct.replace(0, np.nan)
    flags["DUICHEN_VA"] = (space_sym >= 0.45) & (space_sym <= 1.0) & (out["vol_ratio_prev"] < 0.65) & low_absorb & (j < 15)

    fangliang = (pct > 5) & (volume > volume.shift(1).rolling(5, min_periods=3).mean() * 1.5)
    fangliang_vol = volume.where(fangliang).shift(1).rolling(15, min_periods=1).max()
    bbi_up = out["bbi"] > out["bbi"].shift(5)
    near_bbi = ((close - out["bbi"]).abs() / out["bbi"]) < 0.025
    fangliang_recent = fangliang.shift(3).rolling(10, min_periods=1).max().fillna(False).astype(bool)
    flags["ZAIHOU"] = fangliang_recent & bbi_up & near_bbi & (volume < fangliang_vol * 0.55) & (pct >= -2.5) & (pct <= 1.5)

    amp20 = (high.rolling(20, min_periods=15).max() - low.rolling(20, min_periods=15).min()) / low.rolling(20, min_periods=15).min()
    huge = volume > volume.shift(1).rolling(10, min_periods=5).mean() * 2
    huge_count = huge.rolling(20, min_periods=10).sum()
    huge_yang = (huge & out["is_rise"]).rolling(20, min_periods=10).sum()
    near_platform_high = close >= high.rolling(20, min_periods=15).max() * 0.94
    flags["YUEYUE"] = (amp20 <= 0.16) & (huge_count >= 2) & (huge_yang / huge_count.replace(0, np.nan) >= 0.5) & near_platform_high & (pct >= 0) & (close_pos >= 0.6)

    body_pct = (close - out["open"]).abs() / out["pre_close"].replace(0, np.nan) * 100
    high20 = high.shift(1).rolling(20, min_periods=10).max()
    low20 = low.shift(1).rolling(20, min_periods=10).min()
    at_key = (high >= high20 * 0.98) | (low <= low20 * 1.02)
    vol_threshold = np.where(body_pct >= 7, 1.1, 1.3)
    flags["KEY_K"] = (body_pct >= 3) & out["is_rise"] & (close_pos >= 0.75) & (pct >= 2) & (out["vol_ratio_5"] >= vol_threshold) & at_key

    prev_body = body_pct.shift(1).rolling(5, min_periods=2).mean()
    at_bottom = low <= low20 * 1.05
    flags["VIOLENCE_K"] = (
        at_bottom
        & out["is_rise"]
        & (pct > 0)
        & (close_pos >= 0.7)
        & (body_pct >= 5)
        & (body_pct > prev_body * 2)
        & (out["vol_ratio_5"] >= 2)
    )

    result_cols = ["symbol", "date", "open", "high", "low", "close", "pct_chg", "close_pos", "kdj_j"]
    result = out[result_cols].copy()
    if "name" in out.columns:
        result["name"] = out["name"]
    for col in [spec.key for spec in build_signal_specs()]:
        raw_flag = flags[col].fillna(False).astype(bool)
        recent_same_signal = raw_flag.shift(1).rolling(5, min_periods=1).max().fillna(False).astype(bool)
        result[col] = (raw_flag & ~recent_same_signal).astype(bool)
    return result[result["date"] >= pd.Timestamp("2020-01-01")].copy()


def process_file(path: Path, start_date: str) -> pd.DataFrame | None:
    try:
        df = _normalize_daily(path, start_date)
        if df.empty:
            return None
        signals = compute_z_skill_flags(df)
        signal_cols = [spec.key for spec in build_signal_specs()]
        signals = signals[signals[signal_cols].any(axis=1)].copy()
        return signals if not signals.empty else None
    except Exception as exc:
        print(f"skip {path.name}: {exc}", flush=True)
        return None


def process_frame(
    symbol: str,
    frame: pd.DataFrame,
    start_date: str,
    *,
    factors_attached: bool = False,
    raise_errors: bool = False,
) -> pd.DataFrame | None:
    """Build z-skill signals from an in-memory canonical symbol slice."""

    path = Path(f"{symbol}.parquet")
    try:
        df = _normalize_daily(
            path,
            start_date,
            source_frame=frame,
            factors_attached=factors_attached,
        )
        if df.empty:
            return None
        signals = compute_z_skill_flags(df)
        signal_cols = [spec.key for spec in build_signal_specs()]
        signals = signals[signals[signal_cols].any(axis=1)].copy()
        return signals if not signals.empty else None
    except Exception as exc:
        if raise_errors:
            raise RuntimeError(f"{symbol}: {exc}") from exc
        print(f"skip {symbol}: {exc}", flush=True)
        return None


def _parse_cache_start_date(value: str) -> pd.Timestamp:
    if value.isdigit() and len(value) == 8:
        return pd.to_datetime(value, format="%Y%m%d")
    return pd.to_datetime(value)


def build_signal_candidates(
    daily_dir: Path,
    start_date: str,
    force_refresh: bool,
    workers: int,
    reuse_signal_cache: bool = False,
) -> pd.DataFrame:
    start_ts = _parse_cache_start_date(start_date)
    cached: pd.DataFrame | None = None
    if SIGNAL_CACHE.exists() and not force_refresh:
        cached = pd.read_parquet(SIGNAL_CACHE)
        cached["date"] = pd.to_datetime(cached["date"])
        expected = {spec.key for spec in build_signal_specs()}
        if expected <= set(cached.columns):
            cached = cached.dropna(subset=["symbol", "date"])
        else:
            print("z-skill signal cache missing expected columns; rebuilding", flush=True)
            cached = None
    if reuse_signal_cache:
        if cached is None:
            raise RuntimeError("--reuse-signal-cache requires a valid z-skill signal cache")
        return (
            cached[pd.to_datetime(cached["date"]) >= start_ts]
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )

    history_start = start_ts - pd.Timedelta(days=450)
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=daily_dir.parent))
    market = store.read_market_range(daily_dir.name, start_date=history_start.strftime("%Y%m%d"))
    if market.empty:
        raise RuntimeError(f"No canonical daily rows found for {history_start:%Y-%m-%d}+")
    tasks = [
        (str(symbol), group.reset_index(drop=True))
        for symbol, group in market.groupby("ts_code", sort=False)
    ]
    frames = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(process_frame, symbol, frame, start_date) for symbol, frame in tasks]
        for n, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None and not result.empty:
                result = result[pd.to_datetime(result["date"]) >= start_ts].copy()
                if not result.empty:
                    frames.append(result)
            if n % 500 == 0 or n == len(futures):
                print(f"  z-skill signals: {n}/{len(futures)} symbols", flush=True)
    if cached is not None:
        old = cached[pd.to_datetime(cached["date"]) < start_ts].copy()
        if frames:
            recent = pd.concat(frames, ignore_index=True)
            combined = pd.concat([old, recent], ignore_index=True)
        else:
            combined = old
    elif frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        raise RuntimeError("No z-skill signal candidates built")
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
    atomic_write_parquet(combined, SIGNAL_CACHE, index=False)
    return combined[combined["date"] >= start_ts].copy()


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = np.select(
        [
            out["date"] < pd.Timestamp("2024-01-01"),
            (out["date"] >= pd.Timestamp("2024-01-01")) & (out["date"] < pd.Timestamp("2025-01-01")),
            out["date"] >= pd.Timestamp("2025-01-01"),
        ],
        ["train", "test", "oot"],
        default="unknown",
    )
    return out


def apply_open_filter(df: pd.DataFrame, rule: OpenFilter) -> pd.Series:
    gap = (df["entry_open"] / df["close"] - 1) * 100
    mask = pd.Series(True, index=df.index)
    if rule.min_gap_pct is not None:
        mask &= gap >= rule.min_gap_pct
    if rule.max_gap_pct is not None:
        mask &= gap <= rule.max_gap_pct
    if rule.min_close_pos is not None:
        mask &= df["close_pos"] >= rule.min_close_pos
    return mask


def evaluate(candidates: pd.DataFrame, min_entry_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {spec.key: spec for spec in build_signal_specs()}
    open_filters = build_open_filters()
    exit_rules = build_exit_rules()
    rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []

    for signal_key, spec in specs.items():
        base = candidates[candidates[signal_key]].copy()
        if base.empty:
            continue
        print(f"signal {signal_key}: raw candidates={len(base):,}", flush=True)
        for open_filter in open_filters:
            entry_df = base[apply_open_filter(base, open_filter)].copy()
            if len(entry_df) < min_entry_rows:
                continue
            for exit_rule in exit_rules:
                trades = simulate_exit(entry_df, exit_rule)
                if trades.empty:
                    continue
                meta_cols = ["date", "symbol", "split", "close", "entry_open", "close_pos", "kdj_j"]
                trades = trades.merge(entry_df[meta_cols], on=["date", "symbol"], how="left")
                raw_trades = len(trades)
                trades = drop_overlapping_trades(trades)
                skipped = raw_trades - len(trades)
                if exit_rule.name in {"expiry_T5_close", "fixed_tp4.0%_sl1.5%_intraday_T5", "trail_target4.0%_dd2.0%_sl1.5%_intraday_T7"}:
                    sample = trades.copy()
                    sample["signal"] = signal_key
                    sample["strategy_name"] = spec.name
                    sample["open_filter"] = open_filter.name
                    sample["exit_rule"] = exit_rule.name
                    trade_frames.append(sample)
                for split in ["train", "test", "oot"]:
                    part = trades[trades["split"] == split]
                    metrics = summarize_returns(part)
                    if not metrics:
                        continue
                    rows.append(
                        {
                            "signal": signal_key,
                            "strategy_name": spec.name,
                            "timing": spec.timing,
                            "signal_description": spec.description,
                            "open_filter": open_filter.name,
                            "open_filter_description": open_filter.description,
                            "exit_rule": exit_rule.name,
                            "exit_kind": exit_rule.kind,
                            "hold_days": exit_rule.hold_days,
                            "take_profit": exit_rule.take_profit,
                            "stop_loss": exit_rule.stop_loss,
                            "stop_trigger": exit_rule.stop_trigger,
                            "trail_drawdown": exit_rule.trail_drawdown,
                            "split": split,
                            "raw_trades": raw_trades,
                            "skipped_overlaps": skipped,
                            "overlap_skip_rate": skipped / raw_trades if raw_trades else np.nan,
                            "min_return_pct": float(part["return_pct"].min()) if not part.empty else np.nan,
                            "max_return_pct": float(part["return_pct"].max()) if not part.empty else np.nan,
                            **metrics,
                        }
                    )
    detail = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return pd.DataFrame(rows), detail


def _score_row(row: pd.Series) -> float:
    trades = float(row.get("trades") or 0)
    avg_return = float(row.get("avg_return_pct") or 0)
    win_rate = float(row.get("win_rate") or 0)
    pf = float(row.get("profit_factor") or 0)
    dd = abs(float(row.get("max_drawdown_pct") or 0))
    min_ret = abs(float(row.get("min_return_pct") or 0))
    sample = min(1.0, np.sqrt(max(trades, 0) / 80))
    return (avg_return * 0.45 + min(pf, 4) * 1.3 + win_rate * 2.5 - dd / 18 - min_ret / 20) * (0.45 + 0.55 * sample)


def choose_playbooks(summary: pd.DataFrame) -> pd.DataFrame:
    oot = summary[summary["split"] == "oot"].copy()
    if oot.empty:
        return oot
    oot["selection_score"] = oot.apply(_score_row, axis=1)
    rows = []
    for signal, part in oot.groupby("signal"):
        min_trades = 20 if len(part) and part["trades"].max() < 80 else 50
        tradable = part[
            (part["trades"] >= min_trades)
            & (part["avg_return_pct"] > 0)
            & (part["profit_factor"].fillna(0) >= 1.5)
            & (part["max_drawdown_pct"] >= -35)
            & (part["min_return_pct"] >= -20)
        ].copy()
        if not tradable.empty:
            tradable["risk_score"] = (
                tradable["profit_factor"].fillna(0).clip(upper=4) * 1.4
                + tradable["avg_return_pct"].fillna(0) * 0.35
                + tradable["win_rate"].fillna(0) * 2
                + tradable["max_drawdown_pct"].fillna(-100) / 25
                + tradable["min_return_pct"].fillna(-100) / 30
            )
            eligible = tradable.sort_values(["risk_score", "selection_score"], ascending=[False, False]).head(1).copy()
            eligible["action_level"] = "可小仓实操"
        else:
            eligible = part[(part["trades"] >= min_trades) & (part["profit_factor"].fillna(0) >= 1.0)].copy()
            if eligible.empty:
                eligible = part.sort_values(["selection_score", "profit_factor", "avg_return_pct"], ascending=[False, False, False]).head(1).copy()
                eligible["action_level"] = "仅观察"
            else:
                eligible = eligible.sort_values(["selection_score", "profit_factor", "avg_return_pct"], ascending=[False, False, False]).head(1).copy()
                eligible["action_level"] = "谨慎观察"
        if "action_level" not in eligible.columns:
            eligible["action_level"] = "谨慎观察"
        rows.append(eligible)
    level_order = {"可小仓实操": 0, "谨慎观察": 1, "仅观察": 2}
    out = pd.concat(rows, ignore_index=True)
    out["action_order"] = out["action_level"].map(level_order).fillna(9)
    return out.sort_values(["action_order", "selection_score"], ascending=[True, False]).drop(columns=["action_order"])


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}%"


def fmt_rate(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def markdown_table(rows: list[dict], headers: list[str]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(header, "")) for header in headers) + "|")
    return "\n".join(lines)


def _format_summary_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "战法": row["strategy_name"],
                "信号": row["signal"],
                "开盘过滤": row["open_filter"],
                "卖出": row["exit_rule"],
                "交易数": fmt_num(row["trades"]),
                "均值": fmt_pct(row["avg_return_pct"]),
                "胜率": fmt_rate(row["win_rate"]),
                "最大回撤": fmt_pct(row["max_drawdown_pct"]),
                "PF": f"{row['profit_factor']:.2f}" if pd.notna(row["profit_factor"]) else "",
                "最差单笔": fmt_pct(row["min_return_pct"]),
            }
        )
    return rows


def write_report(summary: pd.DataFrame, playbooks: pd.DataFrame, output_dir: Path, timestamp: str) -> Path:
    path = output_dir / f"z_skill_entry_exit_backtest_{timestamp}.md"
    oot = summary[(summary["split"] == "oot") & (summary["trades"] >= 30)].copy()
    oot["selection_score"] = oot.apply(_score_row, axis=1)
    top_pf = oot.sort_values(["profit_factor", "avg_return_pct", "max_drawdown_pct"], ascending=[False, False, False]).head(40)
    top_score = oot.sort_values(["selection_score", "profit_factor", "avg_return_pct"], ascending=[False, False, False]).head(40)

    with path.open("w", encoding="utf-8") as f:
        f.write("# z-skill 战法买入/卖出策略回测与实操方案\n\n")
        f.write("## 回测口径\n\n")
        f.write("- 数据：Tushare 日线 OHLCV，信号从 2020-01-01 起评估。\n")
        f.write("- 切分：2020-2023 为 train，2024 为 test，2025 以后为 OOT。\n")
        f.write("- 买入：T+0 收盘生成信号，T+1 开盘买入；交叉不同开盘涨跌幅过滤。\n")
        f.write("- 卖出：T+2 起检查止盈/止损/回撤止盈；到期未触发则到期收盘卖出。\n")
        f.write("- 持仓：同一股票未卖出前不重复买入，避免连续信号放大样本。\n")
        f.write("- 说明：这里是日线实操回测，盘中战法仍需分钟数据做二次验证。\n\n")

        f.write("## 推荐实操清单\n\n")
        headers = ["级别", "战法", "开盘过滤", "卖出", "交易数", "均值", "胜率", "最大回撤", "PF", "实操结论"]
        playbook_rows = []
        for _, row in playbooks.iterrows():
            playbook_rows.append(
                {
                    "级别": row["action_level"],
                    "战法": row["strategy_name"],
                    "开盘过滤": row["open_filter_description"],
                    "卖出": row["exit_rule"],
                    "交易数": fmt_num(row["trades"]),
                    "均值": fmt_pct(row["avg_return_pct"]),
                    "胜率": fmt_rate(row["win_rate"]),
                    "最大回撤": fmt_pct(row["max_drawdown_pct"]),
                    "PF": f"{row['profit_factor']:.2f}" if pd.notna(row["profit_factor"]) else "",
                    "实操结论": "进入前端跟踪/小仓验证" if row["action_level"] == "可小仓实操" else "先观察，不作为默认买入",
                }
            )
        f.write(markdown_table(playbook_rows, headers))
        f.write("\n\n")

        f.write("## OOT 综合分 Top 40\n\n")
        f.write(markdown_table(_format_summary_rows(top_score), ["战法", "信号", "开盘过滤", "卖出", "交易数", "均值", "胜率", "最大回撤", "PF", "最差单笔"]))
        f.write("\n\n")

        f.write("## OOT PF Top 40\n\n")
        f.write(markdown_table(_format_summary_rows(top_pf), ["战法", "信号", "开盘过滤", "卖出", "交易数", "均值", "胜率", "最大回撤", "PF", "最差单笔"]))
        f.write("\n\n")

        f.write("## 分战法执行说明\n\n")
        spec_map = {spec.key: spec for spec in build_signal_specs()}
        for _, row in playbooks.iterrows():
            spec = spec_map[row["signal"]]
            f.write(f"### {row['strategy_name']}\n\n")
            f.write(f"- 信号逻辑：{spec.description}\n")
            f.write(f"- 买入计划：{row['open_filter_description']}；符合信号后 T+1 开盘执行，不符合则空仓。\n")
            f.write(f"- 卖出计划：`{row['exit_rule']}`；止损口径见规则名中的 `intraday/close`。\n")
            f.write(
                f"- OOT 表现：{fmt_num(row['trades'])} 笔，均值 {fmt_pct(row['avg_return_pct'])}，"
                f"胜率 {fmt_rate(row['win_rate'])}，最大回撤 {fmt_pct(row['max_drawdown_pct'])}，PF {row['profit_factor']:.2f}。\n"
            )
            f.write(f"- 当前结论：{row['action_level']}。\n\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="z-skill entry/exit strategy backtest")
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--reuse-signal-cache",
        action="store_true",
        help="Use the existing validated signal cache without rebuilding raw indicators.",
    )
    parser.add_argument("--min-entry-rows", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("building/loading z-skill signal candidates", flush=True)
    candidates = build_signal_candidates(
        args.daily_dir,
        args.start_date,
        args.force_refresh,
        args.workers,
        reuse_signal_cache=args.reuse_signal_cache,
    )
    candidates = add_split(candidates)
    print(f"candidate rows: {len(candidates):,}", flush=True)

    max_hold_days = max(rule.hold_days for rule in build_exit_rules())
    print("adding future prices", flush=True)
    candidates = add_future_prices(candidates, args.daily_dir, max_hold_days=max_hold_days)
    candidates = candidates.dropna(subset=["entry_open"]).copy()
    print(f"tradable rows: {len(candidates):,}", flush=True)

    print("evaluating z-skill entry/exit grid", flush=True)
    summary, details = evaluate(candidates, args.min_entry_rows)
    playbooks = choose_playbooks(summary)

    csv_path = args.output_dir / f"z_skill_entry_exit_backtest_{timestamp}.csv"
    latest_csv = args.output_dir / "latest_z_skill_entry_exit_backtest.csv"
    playbook_path = args.output_dir / f"z_skill_operational_playbook_{timestamp}.csv"
    latest_playbook = args.output_dir / "latest_z_skill_operational_playbook.csv"
    detail_path = args.output_dir / f"z_skill_trade_samples_{timestamp}.csv"
    latest_detail = args.output_dir / "latest_z_skill_trade_samples.csv"

    summary.to_csv(csv_path, index=False)
    atomic_link_or_copy(csv_path, latest_csv)
    playbooks.to_csv(playbook_path, index=False)
    atomic_link_or_copy(playbook_path, latest_playbook)
    if not details.empty:
        details.to_csv(detail_path, index=False)
        atomic_link_or_copy(detail_path, latest_detail)
    else:
        latest_detail.unlink(missing_ok=True)

    report_path = write_report(summary, playbooks, args.output_dir, timestamp)
    latest_report = args.output_dir / "latest_z_skill_entry_exit_backtest.md"
    atomic_link_or_copy(report_path, latest_report)

    print(f"summary: {csv_path}", flush=True)
    print(f"playbook: {playbook_path}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
