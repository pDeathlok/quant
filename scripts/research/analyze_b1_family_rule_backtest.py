#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backtest B2/B3/SB1/Super-B1 rule families before modeling.

The goal of this script is to answer whether each zettaranc-style tactic has a
reasonable daily-rule threshold before creating dedicated ML labels/models.
Intraday tactics are explicitly marked and evaluated with a daily approximation
only, because real early/tail execution needs minute data.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_b1_entry_exit_grid import ExitRule, add_future_prices, simulate_exit, summarize_returns
from analyze_b1_xgb_entry_exit_grid import DEFAULT_DAILY_DIR, DEFAULT_OUTPUT_DIR, drop_overlapping_trades
from quant.features.variable_library import calculate_project_extra_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_CACHE = PROJECT_ROOT / "data/features/b1/b1_family_rule_candidates.parquet"


@dataclass(frozen=True)
class SignalSpec:
    name: str
    family: str
    timeframe: str
    ideal_timing: str
    daily_backtest_timing: str
    description: str


@dataclass(frozen=True)
class OpenFilter:
    name: str
    description: str
    min_gap_pct: float | None = None
    max_gap_pct: float | None = None


def build_signal_specs() -> list[SignalSpec]:
    return [
        SignalSpec(
            "b2_pchg3_vol12",
            "B2",
            "日线级",
            "收盘确认，次日开盘观察",
            "T+1 开盘买入",
            "B1后三日内涨幅>=3%、5日量比>=1.2、收盘位置强、J<60",
        ),
        SignalSpec(
            "b2_pchg4_vol15",
            "B2",
            "日线级",
            "收盘确认，次日开盘观察",
            "T+1 开盘买入",
            "B1后三日内涨幅>=4%、5日量比>=1.5、上影较小、J<55",
        ),
        SignalSpec(
            "b2_pchg5_vol15",
            "B2",
            "日线级",
            "收盘确认，次日开盘观察",
            "T+1 开盘买入",
            "B1后三日内涨幅>=5%、5日量比>=1.5、收盘位置强、J<60",
        ),
        SignalSpec(
            "b2_any_pchg4_vol15",
            "B2",
            "日线级",
            "收盘确认，次日开盘观察",
            "T+1 开盘买入",
            "独立右侧B2：涨幅>=4%、5日量比>=1.5、收盘位置>=75%、J<80",
        ),
        SignalSpec(
            "b2_oversold_pchg3_vol12",
            "B2",
            "日线级",
            "收盘确认，次日开盘观察",
            "T+1 开盘买入",
            "近5日J<0后右侧确认：涨幅>=3%、5日量比>=1.2、收盘位置>=70%",
        ),
        SignalSpec(
            "b2_bbi_reclaim_vol12",
            "B2",
            "日线级",
            "收盘确认，次日开盘观察",
            "T+1 开盘买入",
            "BBI收复确认：昨日收盘不高于BBI、今日涨幅>=3%、5日量比>=1.2",
        ),
        SignalSpec(
            "b3_small_pos_amp7",
            "B3",
            "日线级",
            "B2后分歧转一致，收盘确认",
            "T+1 开盘买入",
            "B2后三日内小阳，涨幅0%-2%、振幅<7%",
        ),
        SignalSpec(
            "b3_small_pos_amp5",
            "B3",
            "日线级",
            "B2后分歧转一致，收盘确认",
            "T+1 开盘买入",
            "B2后三日内小阳，涨幅0%-2%、振幅<5%",
        ),
        SignalSpec(
            "b3_calm_pullback",
            "B3",
            "日线级",
            "B2后缩量分歧，收盘确认",
            "T+1 开盘买入",
            "B2后三日内-1%到2%、振幅<7%、5日量比<=1.3",
        ),
        SignalSpec(
            "b3_broad_small_pos",
            "B3",
            "日线级",
            "宽口径B2后分歧转一致，收盘确认",
            "T+1 开盘买入",
            "宽口径B2后三日内小阳，涨幅0%-2%、振幅<7%、收盘在中位以上",
        ),
        SignalSpec(
            "b3_broad_calm_pullback",
            "B3",
            "日线级",
            "宽口径B2后缩量分歧，收盘确认",
            "T+1 开盘买入",
            "宽口径B2后三日内-1%到2%、振幅<7%、5日量比<=1.3",
        ),
        SignalSpec(
            "sb1_range10_vol12",
            "SB1",
            "盘中级（日线近似）",
            "横盘后盘中下破洗盘，尾盘/次日早盘确认",
            "日线收盘信号，T+1 开盘近似买入",
            "前三日横盘区间<10%、放量阴线下破、J<0",
        ),
        SignalSpec(
            "sb1_range7_vol15",
            "SB1",
            "盘中级（日线近似）",
            "横盘后盘中下破洗盘，尾盘/次日早盘确认",
            "日线收盘信号，T+1 开盘近似买入",
            "前三日横盘区间<7%、5日量比>=1.5、阴线下破、J<0",
        ),
        SignalSpec(
            "sb1_range5_vol15_j10",
            "SB1",
            "盘中级（日线近似）",
            "横盘后盘中下破洗盘，尾盘/次日早盘确认",
            "日线收盘信号，T+1 开盘近似买入",
            "前三日横盘区间<5%、5日量比>=1.5、阴线下破、J<-10",
        ),
        SignalSpec(
            "super_washout_vol12",
            "SUPER_B1",
            "盘中级（日线近似）",
            "放量下杀后缩量企稳，尾盘/次日早盘确认",
            "日线收盘信号，T+1 开盘近似买入",
            "近三日放量下杀，随后缩量企稳，J<0",
        ),
        SignalSpec(
            "super_washout_vol15_j0",
            "SUPER_B1",
            "盘中级（日线近似）",
            "放量下杀后缩量企稳，尾盘/次日早盘确认",
            "日线收盘信号，T+1 开盘近似买入",
            "近三日5日量比>=1.5下杀，随后20日量比<0.9，J<0",
        ),
        SignalSpec(
            "super_washout_j10_closepos40",
            "SUPER_B1",
            "盘中级（日线近似）",
            "放量下杀后尾盘收回，次日早盘确认",
            "日线收盘信号，T+1 开盘近似买入",
            "近三日放量下杀，随后J<-10且收盘位置>=40%",
        ),
    ]


def build_open_filters() -> list[OpenFilter]:
    return [
        OpenFilter("all", "不限制 T+1 开盘涨跌幅"),
        OpenFilter("gap_up", "T+1 不低开", 0.0, None),
        OpenFilter("gap_0_to_1", "T+1 开盘 0%-1%", 0.0, 1.0),
        OpenFilter("gap_0_to_2", "T+1 开盘 0%-2%", 0.0, 2.0),
        OpenFilter("gap_le_1", "T+1 开盘不超过 1%", None, 1.0),
    ]


def build_exit_rules() -> list[ExitRule]:
    return [
        ExitRule("fixed_tp10_sl15_T7", "fixed", 6, 0.10, 0.015, stop_trigger="intraday"),
        ExitRule("fixed_tp8_sl15_T7", "fixed", 6, 0.08, 0.015, stop_trigger="intraday"),
        ExitRule("fixed_tp6_sl15_T7", "fixed", 6, 0.06, 0.015, stop_trigger="intraday"),
        ExitRule("fixed_tp4_sl15_T5", "fixed", 4, 0.04, 0.015, stop_trigger="intraday"),
        ExitRule("trail_target4_dd2_sl15_T7", "trailing", 6, 0.04, 0.015, 0.02, "intraday"),
        ExitRule("expiry_T5_close", "expiry", 4),
        ExitRule("expiry_T7_close", "expiry", 6),
    ]


def normalize_daily(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "trade_date" in df.columns:
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "vol" in df.columns and "volume" not in df.columns:
        df["volume"] = df["vol"]
    df["symbol"] = path.stem
    if "ts_code" not in df.columns:
        df["ts_code"] = path.stem
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return df


def compute_signal_flags(df: pd.DataFrame) -> pd.DataFrame:
    extra = calculate_project_extra_features(df)
    out = df.copy()
    for col in extra.columns:
        if col not in out.columns:
            out[col] = extra[col]

    close = out["close"]
    open_ = out["open"]
    high = out["high"]
    low = out["low"]
    pct = out["pct_chg"] if "pct_chg" in out.columns else close.pct_change() * 100
    prev_close = close.shift(1).replace(0, np.nan)
    amplitude = (high - low) / prev_close * 100
    close_pos = (close - low) / (high - low).replace(0, np.nan)
    is_yin = close < open_
    recent_yin_count = is_yin.rolling(4, min_periods=1).sum()
    ma60 = out["ma_60"] if "ma_60" in out.columns else close.rolling(60, min_periods=20).mean()
    bbi = out["bbi"]
    support_ok = (close >= bbi * 0.97) | (close >= ma60 * 0.97)

    b1_base = (
        (out["kdj_d_j"] <= -10)
        & (out["volume_relative_20d"] < 1.0)
        & (recent_yin_count < 4)
        & support_ok
    )
    b1_recent = b1_base.shift(1).rolling(3, min_periods=1).max() > 0

    flags = pd.DataFrame(index=out.index)
    flags["b2_pchg3_vol12"] = (
        b1_recent
        & (pct >= 3)
        & (out["volume_relative_5d"] >= 1.2)
        & (close_pos >= 0.70)
        & (out["kdj_d_j"] < 60)
    )
    flags["b2_pchg4_vol15"] = (
        b1_recent
        & (pct >= 4)
        & (out["volume_relative_5d"] >= 1.5)
        & (high <= close * 1.01)
        & (out["kdj_d_j"] < 55)
    )
    flags["b2_pchg5_vol15"] = (
        b1_recent
        & (pct >= 5)
        & (out["volume_relative_5d"] >= 1.5)
        & (close_pos >= 0.75)
        & (out["kdj_d_j"] < 60)
    )
    pre_oversold_recent = out["kdj_d_j"].lt(0).shift(1).rolling(5, min_periods=1).max() > 0
    flags["b2_any_pchg4_vol15"] = (
        (pct >= 4)
        & (out["volume_relative_5d"] >= 1.5)
        & (close_pos >= 0.75)
        & (out["kdj_d_j"] < 80)
    )
    flags["b2_oversold_pchg3_vol12"] = (
        pre_oversold_recent
        & (pct >= 3)
        & (out["volume_relative_5d"] >= 1.2)
        & (close_pos >= 0.70)
        & (out["kdj_d_j"] < 80)
    )
    flags["b2_bbi_reclaim_vol12"] = (
        (close.shift(1) <= bbi.shift(1) * 1.01)
        & (close > bbi)
        & (pct >= 3)
        & (out["volume_relative_5d"] >= 1.2)
        & (close_pos >= 0.65)
        & (out["kdj_d_j"] < 80)
    )

    b2_reference = flags["b2_pchg3_vol12"] | flags["b2_pchg4_vol15"] | flags["b2_pchg5_vol15"]
    b2_broad_reference = (
        b2_reference
        | flags["b2_any_pchg4_vol15"]
        | flags["b2_oversold_pchg3_vol12"]
        | flags["b2_bbi_reclaim_vol12"]
    )
    b2_recent = b2_reference.shift(1).rolling(3, min_periods=1).max() > 0
    b2_broad_recent = b2_broad_reference.shift(1).rolling(3, min_periods=1).max() > 0
    flags["b3_small_pos_amp7"] = b2_recent & (pct > 0) & (pct < 2) & (amplitude < 7)
    flags["b3_small_pos_amp5"] = b2_recent & (pct > 0) & (pct < 2) & (amplitude < 5)
    flags["b3_calm_pullback"] = (
        b2_recent
        & (pct >= -1)
        & (pct < 2)
        & (amplitude < 7)
        & (out["volume_relative_5d"] <= 1.3)
    )
    flags["b3_broad_small_pos"] = b2_broad_recent & (pct > 0) & (pct < 2) & (amplitude < 7) & (close_pos >= 0.50)
    flags["b3_broad_calm_pullback"] = (
        b2_broad_recent
        & (pct >= -1)
        & (pct < 2)
        & (amplitude < 7)
        & (out["volume_relative_5d"] <= 1.3)
    )

    range3 = high.shift(1).rolling(3, min_periods=3).max() / low.shift(1).rolling(3, min_periods=3).min().replace(0, np.nan) - 1
    flags["sb1_range10_vol12"] = (
        (range3 < 0.10)
        & is_yin
        & (out["volume_relative_5d"] >= 1.2)
        & (low < low.shift(1).rolling(3, min_periods=3).min())
        & (out["kdj_d_j"] < 0)
    )
    flags["sb1_range7_vol15"] = (
        (range3 < 0.07)
        & is_yin
        & (out["volume_relative_5d"] >= 1.5)
        & (low < low.shift(1).rolling(3, min_periods=3).min())
        & (out["kdj_d_j"] < 0)
    )
    flags["sb1_range5_vol15_j10"] = (
        (range3 < 0.05)
        & is_yin
        & (out["volume_relative_5d"] >= 1.5)
        & (low < low.shift(1).rolling(3, min_periods=3).min())
        & (out["kdj_d_j"] < -10)
    )

    washout12 = (
        is_yin
        & (pct > -9.5)
        & (out["volume_relative_5d"] >= 1.2)
        & (low < low.shift(1).rolling(10, min_periods=5).min())
    )
    washout15 = washout12 & (out["volume_relative_5d"] >= 1.5)
    small_reversal_loose = (amplitude < 8) & (pct > -2.5) & (pct < 3.0) & (out["volume_relative_20d"] < 1.0)
    small_reversal_tight = (amplitude < 7) & (pct > -2.0) & (pct < 2.5) & (out["volume_relative_20d"] < 0.9)
    flags["super_washout_vol12"] = (
        (washout12.shift(1).rolling(3, min_periods=1).max() > 0)
        & small_reversal_loose
        & (out["kdj_d_j"] < 0)
    )
    flags["super_washout_vol15_j0"] = (
        (washout15.shift(1).rolling(3, min_periods=1).max() > 0)
        & small_reversal_tight
        & (out["kdj_d_j"] < 0)
    )
    flags["super_washout_j10_closepos40"] = (
        (washout12.shift(1).rolling(3, min_periods=1).max() > 0)
        & (amplitude < 8)
        & (pct > -2.5)
        & (pct < 3.0)
        & (out["kdj_d_j"] < -10)
        & (close_pos >= 0.40)
    )

    result = out[["symbol", "date", "open", "high", "low", "close", "pct_chg", "kdj_d_j"]].copy()
    if "name" in out.columns:
        result["name"] = out["name"]
    for col in flags.columns:
        result[col] = flags[col].fillna(False).astype(bool)
    return result


def process_file(path: Path) -> pd.DataFrame | None:
    try:
        df = normalize_daily(path)
        if len(df) < 160:
            return None
        if "name" in df.columns:
            names = df["name"].fillna("").astype(str)
            if names.str.upper().str.contains("ST").any() or names.str.contains("退").any():
                df = df[~names.str.upper().str.contains("ST") & ~names.str.contains("退")].copy()
        signals = compute_signal_flags(df)
        signal_cols = [spec.name for spec in build_signal_specs()]
        mask = signals[signal_cols].any(axis=1)
        signals = signals[mask].copy()
        return signals if not signals.empty else None
    except Exception as exc:
        print(f"skip {path.name}: {exc}", flush=True)
        return None


def build_signal_candidates(force_refresh: bool = False, workers: int = 32) -> pd.DataFrame:
    if SIGNAL_CACHE.exists() and not force_refresh:
        cached = pd.read_parquet(SIGNAL_CACHE)
        cached["date"] = pd.to_datetime(cached["date"])
        expected = {spec.name for spec in build_signal_specs()}
        if expected <= set(cached.columns):
            return cached.dropna(subset=["symbol", "date"])
        print("signal cache is missing new columns; rebuilding", flush=True)

    files = sorted(DEFAULT_DAILY_DIR.glob("*.parquet"))
    frames = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(process_file, path) for path in files]
        for n, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None and not result.empty:
                frames.append(result)
            if n % 500 == 0 or n == len(futures):
                print(f"  family signals: {n}/{len(futures)} files", flush=True)
    if not frames:
        raise RuntimeError("No family signal candidates built")
    combined = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    SIGNAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SIGNAL_CACHE, index=False)
    return combined


def apply_open_filter(df: pd.DataFrame, rule: OpenFilter) -> pd.Series:
    gap = (df["entry_open"] / df["close"] - 1) * 100
    mask = pd.Series(True, index=df.index)
    if rule.min_gap_pct is not None:
        mask &= gap >= rule.min_gap_pct
    if rule.max_gap_pct is not None:
        mask &= gap <= rule.max_gap_pct
    return mask


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


def evaluate(candidates: pd.DataFrame) -> pd.DataFrame:
    specs = {spec.name: spec for spec in build_signal_specs()}
    open_filters = build_open_filters()
    exit_rules = build_exit_rules()
    rows = []
    for signal_name, spec in specs.items():
        base = candidates[candidates[signal_name]].copy()
        if base.empty:
            continue
        for open_filter in open_filters:
            entry_df = base[apply_open_filter(base, open_filter)].copy()
            if len(entry_df) < 20:
                continue
            for exit_rule in exit_rules:
                trades = simulate_exit(entry_df, exit_rule)
                if trades.empty:
                    continue
                trades = trades.merge(entry_df[["date", "symbol", "split"]], on=["date", "symbol"], how="left")
                raw_trades = len(trades)
                trades = drop_overlapping_trades(trades)
                skipped = raw_trades - len(trades)
                for split in ["train", "test", "oot"]:
                    part = trades[trades["split"] == split]
                    metrics = summarize_returns(part)
                    if not metrics:
                        continue
                    rows.append(
                        {
                            "signal": signal_name,
                            "family": spec.family,
                            "timeframe": spec.timeframe,
                            "ideal_timing": spec.ideal_timing,
                            "daily_backtest_timing": spec.daily_backtest_timing,
                            "signal_description": spec.description,
                            "open_filter": open_filter.name,
                            "open_filter_description": open_filter.description,
                            "exit_rule": exit_rule.name,
                            "exit_kind": exit_rule.kind,
                            "split": split,
                            "raw_trades": raw_trades,
                            "skipped_overlaps": skipped,
                            "overlap_skip_rate": skipped / raw_trades if raw_trades else np.nan,
                            "min_return_pct": float(part["return_pct"].min()) if not part.empty else np.nan,
                            "max_return_pct": float(part["return_pct"].max()) if not part.empty else np.nan,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


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


def format_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "战法": row["family"],
                "级别": row["timeframe"],
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


def write_report(summary: pd.DataFrame, output_dir: Path, timestamp: str) -> Path:
    path = output_dir / f"b1_family_rule_backtest_{timestamp}.md"
    oot = summary[(summary["split"] == "oot") & (summary["trades"] >= 30)].copy()
    top = oot.sort_values(["profit_factor", "avg_return_pct", "max_drawdown_pct"], ascending=[False, False, False]).head(30)
    low_dd = oot[oot["trades"] >= 100].sort_values(["max_drawdown_pct", "profit_factor"], ascending=[False, False]).head(30)

    with path.open("w", encoding="utf-8") as f:
        f.write("# B2 / B3 / SB1 / 超级B1 规则回测\n\n")
        f.write("本轮目标是先用规则阈值判断战法是否值得继续优化或建模。未使用专门模型分。\n\n")
        f.write("执行口径：T+0 出信号，T+1 开盘买入，T+2 起检查卖出；同一股票未卖出前不重复买入；盘中止损采用 gap-aware 口径。\n\n")
        f.write("## 战法级别标注\n\n")
        f.write(markdown_table(
            [
                {"战法": "B2", "级别": "日线级", "说明": "B1 后放量长阳确认，收盘后可确认，次日开盘观察。"},
                {"战法": "B3", "级别": "日线级", "说明": "B2 后分歧转一致，收盘后可确认，次日开盘观察。"},
                {"战法": "SB1", "级别": "盘中级（日线近似）", "说明": "横盘后下破洗盘，真实执行通常需要尾盘或次日早盘确认；本轮只用日线近似。"},
                {"战法": "超级B1", "级别": "盘中级（日线近似）", "说明": "放量下杀后企稳，真实执行需要尾盘/早盘分钟确认；本轮只用日线近似。"},
            ],
            ["战法", "级别", "说明"],
        ))
        f.write("\n\n")

        f.write("## OOT PF Top 30\n\n")
        f.write(markdown_table(format_rows(top), ["战法", "级别", "信号", "开盘过滤", "卖出", "交易数", "均值", "胜率", "最大回撤", "PF", "最差单笔"]))
        f.write("\n\n")

        f.write("## OOT 低回撤 Top 30\n\n")
        f.write(markdown_table(format_rows(low_dd), ["战法", "级别", "信号", "开盘过滤", "卖出", "交易数", "均值", "胜率", "最大回撤", "PF", "最差单笔"]))
        f.write("\n\n")

        f.write("## 分战法最佳观察\n\n")
        for family in ["B2", "B3", "SB1", "SUPER_B1"]:
            part = oot[oot["family"] == family].copy()
            if part.empty:
                f.write(f"### {family}\n\n无满足最低样本的 OOT 结果。\n\n")
                continue
            best = part.sort_values(["profit_factor", "avg_return_pct", "max_drawdown_pct"], ascending=[False, False, False]).iloc[0]
            f.write(f"### {family}\n\n")
            f.write(
                f"- 最优观察：`{best['signal']} + {best['open_filter']} + {best['exit_rule']}`，"
                f"级别 `{best['timeframe']}`，OOT 交易数 {fmt_num(best['trades'])}，"
                f"均值 {fmt_pct(best['avg_return_pct'])}，胜率 {fmt_rate(best['win_rate'])}，"
                f"最大回撤 {fmt_pct(best['max_drawdown_pct'])}，PF {best['profit_factor']:.2f}。\n"
            )
            f.write(f"- 阈值含义：{best['signal_description']}\n")
            f.write(f"- 执行说明：{best['ideal_timing']}；本轮回测为 `{best['daily_backtest_timing']}`。\n\n")

        f.write("## 初步建模建议\n\n")
        f.write("- 若某战法在 OOT 中交易数足够、PF>1.5 且回撤可控，可进入下一阶段：构建该战法专属标签与模型。\n")
        f.write("- 对 SB1/超级B1，即使日线近似表现不错，也必须先补分钟数据验证早盘/尾盘执行点，再决定是否建模。\n")
        f.write("- 若 OOT 样本少于 30 或 train/test 与 OOT 明显背离，暂不建模，只保留规则观察。\n")
    return path


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("building/loading family signal candidates", flush=True)
    candidates = build_signal_candidates()
    candidates = add_split(candidates)
    print(f"signal rows: {len(candidates):,}", flush=True)
    max_hold_days = max(rule.hold_days for rule in build_exit_rules())
    print("adding future prices", flush=True)
    candidates = add_future_prices(candidates, DEFAULT_DAILY_DIR, max_hold_days=max_hold_days)
    print("evaluating family rule grid", flush=True)
    summary = evaluate(candidates)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DEFAULT_OUTPUT_DIR / f"b1_family_rule_backtest_{timestamp}.csv"
    latest_csv = DEFAULT_OUTPUT_DIR / "latest_b1_family_rule_backtest.csv"
    summary.to_csv(csv_path, index=False)
    summary.to_csv(latest_csv, index=False)
    report_path = write_report(summary, DEFAULT_OUTPUT_DIR, timestamp)
    latest_report = DEFAULT_OUTPUT_DIR / "latest_b1_family_rule_backtest.md"
    latest_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"summary: {csv_path}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
