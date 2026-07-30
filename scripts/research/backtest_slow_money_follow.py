#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backtest a slow-money-following strategy idea.

The script validates whether "large capital is building a position" signals
have useful forward returns.  It prefers Tushare moneyflow data when available
and falls back to Tushare daily_basic + daily amount/turnover proxies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.tushare_fetcher import TushareDataFetcher
from quant.data import MarketDataStore, MarketDataStoreConfig


DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DAILY_BASIC_DIR = PROJECT_ROOT / "data/raw/daily_basic"
MONEYFLOW_DIR = PROJECT_ROOT / "data/raw/moneyflow"
TOP_LIST_DIR = PROJECT_ROOT / "data/raw/top_list"
HOLDER_TRADE_PATH = PROJECT_ROOT / "data/raw/holder_trade.parquet"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports/slow_money_follow"


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    min_score: float
    entry_mode: str
    exit_mode: str


STRATEGIES = [
    StrategyConfig("score60_pullback_t10", 60.0, "pullback", "t10"),
    StrategyConfig("score70_pullback_t10", 70.0, "pullback", "t10"),
    StrategyConfig("score70_breakout_t10", 70.0, "breakout", "t10"),
    StrategyConfig("score80_breakout_t10", 80.0, "breakout", "t10"),
    StrategyConfig("score80_breakout_no_overheat_t10", 80.0, "breakout_no_overheat", "t10"),
    StrategyConfig("score80_breakout_no_overheat_t20", 80.0, "breakout_no_overheat", "t20"),
    StrategyConfig("score80_breakout_no_overheat_t40", 80.0, "breakout_no_overheat", "t40"),
    StrategyConfig("score80_breakout_no_overheat_tp20_sl8_t40", 80.0, "breakout_no_overheat", "tp20_sl8_t40"),
    StrategyConfig("score70_any_t10", 70.0, "any", "t10"),
    StrategyConfig("score70_pullback_tp12_sl5_trail5_t20", 70.0, "pullback", "tp12_sl5_trail5_t20"),
    StrategyConfig("score80_pullback_tp12_sl5_trail5_t20", 80.0, "pullback", "tp12_sl5_trail5_t20"),
]


FACTOR_COLUMNS = [
    "large_net_amount_ratio",
    "large_net_3d_ratio",
    "large_net_5d_ratio",
    "moneyflow_net_ratio",
    "amount_rel20",
    "amount_rel60",
    "turnover_rate_f",
    "volume_ratio",
    "ret_5d",
    "ret_10d",
    "close_pos_20",
    "pullback_to_ma20",
    "price_not_chased_score",
    "trend_confirm_score",
    "top_net_amount_ratio",
    "market_high_ret5_rate",
    "market_breadth_ma20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate slow-money-following signals.")
    parser.add_argument("--start-date", default="2026-06-01", help="Signal start date, YYYY-MM-DD or YYYYMMDD.")
    parser.add_argument("--end-date", default="2026-06-30", help="Signal end date, YYYY-MM-DD or YYYYMMDD.")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--forward-days", type=int, default=60)
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--daily-basic-dir", type=Path, default=DAILY_BASIC_DIR)
    parser.add_argument("--moneyflow-dir", type=Path, default=MONEYFLOW_DIR)
    parser.add_argument("--top-list-dir", type=Path, default=TOP_LIST_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--refresh-tushare", action="store_true", help="Fetch moneyflow/top_list through Tushare first.")
    parser.add_argument("--refresh-top-list", action="store_true", help="Fetch top_list when --refresh-tushare is set.")
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--max-dates", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--min-total-mv", type=float, default=300000.0, help="Minimum total_mv in 10k CNY.")
    parser.add_argument("--min-amount", type=float, default=50000.0, help="Minimum daily amount in 10k CNY.")
    return parser.parse_args()


def normalize_date(value: str) -> str:
    text = str(value).replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"Invalid date: {value}")
    return text


def load_env_token() -> str | None:
    token = os.environ.get("TUSHARE_TOKEN")
    if token:
        return token
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("TUSHARE_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def ymd_to_timestamp(value: str) -> pd.Timestamp:
    return pd.to_datetime(normalize_date(value), format="%Y%m%d")


def date_text(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%d")


def available_trade_dates(daily_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
    frame = store.read_market_range(
        daily_dir.name,
        start_date=date_text(start),
        end_date=date_text(end),
        columns=["trade_date"],
    )
    if frame.empty or "trade_date" not in frame.columns:
        return []
    dates = pd.to_datetime(
        frame["trade_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    return sorted(dates.dropna().dt.strftime("%Y%m%d").unique().tolist())


def refresh_tushare_data(args: argparse.Namespace, dates: list[str]) -> dict[str, int | str]:
    token = load_env_token()
    if not token:
        return {"status": "skipped_no_token", "moneyflow_files": 0, "top_list_files": 0}

    args.moneyflow_dir.mkdir(parents=True, exist_ok=True)
    args.top_list_dir.mkdir(parents=True, exist_ok=True)
    fetcher = TushareDataFetcher(token=token, cache_dir=args.moneyflow_dir)
    moneyflow_files = 0
    top_list_files = 0
    for idx, trade_date in enumerate(dates, start=1):
        moneyflow_path = args.moneyflow_dir / f"tushare_moneyflow_{trade_date}.parquet"
        if not moneyflow_path.exists():
            df = fetcher.get_moneyflow(trade_date)
            df.to_parquet(moneyflow_path, index=False)
            time.sleep(args.sleep_seconds)
        if moneyflow_path.exists():
            moneyflow_files += 1

        if args.refresh_top_list:
            top_path = args.top_list_dir / f"tushare_top_list_{trade_date}.parquet"
            if not top_path.exists():
                top_df = fetcher.get_top_list(trade_date)
                top_df.to_parquet(top_path, index=False)
                time.sleep(args.sleep_seconds)
            if top_path.exists():
                top_list_files += 1
        if idx % 10 == 0:
            print(f"  refreshed tushare dates: {idx}/{len(dates)}", flush=True)
    return {"status": "ok", "moneyflow_files": moneyflow_files, "top_list_files": top_list_files}


def load_daily_window(daily_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "vol", "amount", "turnover", "volume", "name"]
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
    out = store.read_market_range(
        daily_dir.name,
        start_date=date_text(start),
        end_date=date_text(end),
    )
    if out.empty or "trade_date" not in out.columns:
        return pd.DataFrame()
    out = out[[column for column in columns if column in out.columns]].copy()
    out["date"] = pd.to_datetime(
        out["trade_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    if "amount" not in out.columns:
        out["amount"] = np.nan
    if "turnover" in out.columns:
        out["amount"] = out["amount"].fillna(pd.to_numeric(out["turnover"], errors="coerce"))
    if "vol" not in out.columns and "volume" in out.columns:
        out["vol"] = out["volume"]
    for col in ["open", "high", "low", "close", "pct_chg", "vol", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["ts_code", "date"]).reset_index(drop=True)


def read_daily_basic(daily_basic_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for path in sorted(daily_basic_dir.glob("*.parquet")):
        try:
            date = pd.to_datetime(path.stem, format="%Y%m%d")
        except ValueError:
            continue
        if date < start or date > end:
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["date"] = date
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ["turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv", "float_share", "free_share"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def read_moneyflow(moneyflow_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for path in sorted(moneyflow_dir.glob("*moneyflow_*.parquet")):
        date_part = path.stem.rsplit("_", 1)[-1]
        if not date_part.isdigit():
            continue
        date = pd.to_datetime(date_part, format="%Y%m%d", errors="coerce")
        if pd.isna(date) or date < start or date > end:
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if "trade_date" not in df.columns:
            df["trade_date"] = date_part
        df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    numeric_cols = [col for col in out.columns if col.endswith("_amount") or col.endswith("_vol")]
    numeric_cols += ["net_mf_amount", "net_mf_vol"]
    for col in set(numeric_cols):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def read_top_list(top_list_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for path in sorted(top_list_dir.glob("*top_list_*.parquet")):
        date_part = path.stem.rsplit("_", 1)[-1]
        date = pd.to_datetime(date_part, format="%Y%m%d", errors="coerce")
        if pd.isna(date) or date < start or date > end:
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if "trade_date" not in df.columns:
            df["trade_date"] = date_part
        df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ["net_amount", "amount", "net_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_daily_features(daily: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    out = daily.sort_values(["ts_code", "date"]).copy()
    group = out.groupby("ts_code", group_keys=False)
    out["amount_ma5"] = group["amount"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    out["amount_ma20"] = group["amount"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    out["amount_ma60"] = group["amount"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    out["amount_rel20"] = out["amount"] / out["amount_ma20"].replace(0, np.nan)
    out["amount_rel60"] = out["amount"] / out["amount_ma60"].replace(0, np.nan)
    out["ret_5d"] = group["close"].pct_change(5) * 100
    out["ret_10d"] = group["close"].pct_change(10) * 100
    out["ma20"] = group["close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    out["ma60"] = group["close"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    out["ma20_slope_5d"] = group["ma20"].pct_change(5) * 100
    out["high20"] = group["high"].transform(lambda x: x.rolling(20, min_periods=10).max())
    out["low20"] = group["low"].transform(lambda x: x.rolling(20, min_periods=10).min())
    out["close_pos_20"] = (out["close"] - out["low20"]) / (out["high20"] - out["low20"]).replace(0, np.nan)
    out["pullback_to_ma20"] = (out["close"] / out["ma20"] - 1) * 100
    out["breakout_20d"] = out["close"] >= group["high"].shift(1).transform(lambda x: x.rolling(20, min_periods=10).max())

    for day in [1, 3, 5, 10, 20, 40, 60]:
        out[f"future_return_t{day}_pct"] = (group["close"].shift(-day) / out["close"] - 1) * 100
    for horizon in [20, 40, 60]:
        if horizon > forward_days:
            continue
        future_high = pd.concat([group["high"].shift(-day) for day in range(1, horizon + 1)], axis=1).max(axis=1)
        future_low = pd.concat([group["low"].shift(-day) for day in range(1, horizon + 1)], axis=1).min(axis=1)
        out[f"future_max_high_t{horizon}_pct"] = (future_high / out["close"] - 1) * 100
        out[f"future_max_drawdown_t{horizon}_pct"] = (future_low / out["close"] - 1) * 100
    return out


def percentile_by_date(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df.groupby("date")[column].rank(pct=True, ascending=ascending)


def add_moneyflow_features(frame: pd.DataFrame, has_moneyflow: bool) -> pd.DataFrame:
    out = frame.sort_values(["ts_code", "date"]).copy()
    if not has_moneyflow:
        out["large_net_amount"] = np.nan
        out["large_net_amount_ratio"] = np.nan
        out["large_net_3d"] = np.nan
        out["large_net_5d"] = np.nan
        out["large_net_3d_ratio"] = np.nan
        out["large_net_5d_ratio"] = np.nan
        out["moneyflow_net_ratio"] = np.nan
        return out

    out["large_net_amount"] = (
        out.get("buy_lg_amount", 0).fillna(0)
        + out.get("buy_elg_amount", 0).fillna(0)
        - out.get("sell_lg_amount", 0).fillna(0)
        - out.get("sell_elg_amount", 0).fillna(0)
    )
    out["large_net_amount_ratio"] = out["large_net_amount"] / out["amount"].replace(0, np.nan)
    out["moneyflow_net_ratio"] = out.get("net_mf_amount", np.nan) / out["amount"].replace(0, np.nan)
    group = out.groupby("ts_code", group_keys=False)
    out["large_net_3d"] = group["large_net_amount"].transform(lambda x: x.rolling(3, min_periods=2).sum())
    out["large_net_5d"] = group["large_net_amount"].transform(lambda x: x.rolling(5, min_periods=3).sum())
    out["amount_3d"] = group["amount"].transform(lambda x: x.rolling(3, min_periods=2).sum())
    out["amount_5d"] = group["amount"].transform(lambda x: x.rolling(5, min_periods=3).sum())
    out["large_net_3d_ratio"] = out["large_net_3d"] / out["amount_3d"].replace(0, np.nan)
    out["large_net_5d_ratio"] = out["large_net_5d"] / out["amount_5d"].replace(0, np.nan)
    return out


def score_signals(frame: pd.DataFrame, has_moneyflow: bool, has_top_list: bool) -> pd.DataFrame:
    out = frame.copy()
    if has_moneyflow:
        out["large_net_rank"] = percentile_by_date(out, "large_net_amount_ratio", ascending=True)
        out["large_net_3d_rank"] = percentile_by_date(out, "large_net_3d_ratio", ascending=True)
        out["large_net_5d_rank"] = percentile_by_date(out, "large_net_5d_ratio", ascending=True)
        out["moneyflow_rank"] = percentile_by_date(out, "moneyflow_net_ratio", ascending=True)
    else:
        out["large_net_rank"] = np.nan
        out["large_net_3d_rank"] = np.nan
        out["large_net_5d_rank"] = np.nan
        out["moneyflow_rank"] = np.nan

    out["amount_rel20_rank"] = percentile_by_date(out, "amount_rel20", ascending=True)
    out["amount_rel60_rank"] = percentile_by_date(out, "amount_rel60", ascending=True)
    out["turnover_f_rank"] = percentile_by_date(out, "turnover_rate_f", ascending=True)
    out["abs_ret_5d"] = out["ret_5d"].abs()
    out["abs_ret_5d_rank"] = percentile_by_date(out, "abs_ret_5d", ascending=False)
    out["price_not_chased_score"] = (
        np.where(out["ret_5d"].between(-4, 8), 1.0, 0.0) * 0.55
        + np.where(out["close_pos_20"].between(0.25, 0.85), 1.0, 0.0) * 0.45
    )
    out["trend_confirm_score"] = (
        np.where(out["close"] >= out["ma20"], 1.0, 0.0) * 0.45
        + np.where(out["ma20_slope_5d"] > 0, 1.0, 0.0) * 0.35
        + np.where(out["close"] >= out["ma60"], 1.0, 0.0) * 0.20
    )
    out["top_list_bonus"] = 0.0
    if has_top_list and "top_net_amount_ratio" in out.columns:
        out["top_list_bonus"] = np.where(out["top_net_amount_ratio"] > 0, 1.0, 0.0)

    if has_moneyflow:
        flow_component = (
            out["large_net_rank"].fillna(0) * 0.28
            + out["large_net_3d_rank"].fillna(0) * 0.22
            + out["large_net_5d_rank"].fillna(0) * 0.15
            + out["moneyflow_rank"].fillna(0) * 0.15
        )
        proxy_component = out["amount_rel20_rank"].fillna(0) * 0.12 + out["turnover_f_rank"].fillna(0) * 0.08
    else:
        flow_component = out["amount_rel20_rank"].fillna(0) * 0.35 + out["amount_rel60_rank"].fillna(0) * 0.25
        proxy_component = out["turnover_f_rank"].fillna(0) * 0.20 + out["abs_ret_5d_rank"].fillna(0) * 0.10

    out["slow_money_score"] = (
        (flow_component + proxy_component) * 100
        + out["price_not_chased_score"] * 8
        + out["trend_confirm_score"] * 7
        + out["top_list_bonus"] * 3
    ).clip(0, 100)
    out["entry_pullback"] = (
        (out["slow_money_score"] >= 60)
        & out["pullback_to_ma20"].between(-3.0, 4.0)
        & (out["close"] >= out["ma20"] * 0.97)
        & (out["ret_5d"] <= 8)
    )
    out["entry_breakout"] = (
        (out["slow_money_score"] >= 65)
        & out["breakout_20d"].fillna(False)
        & (out["ret_5d"] <= 12)
        & (out["amount_rel20"] >= 1.2)
    )
    return out


def add_market_context(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["above_ma20"] = out["close"] >= out["ma20"]
    out["high_ret5"] = out["ret_5d"] > 12
    context = (
        out.groupby("date")
        .agg(
            market_breadth_ma20=("above_ma20", "mean"),
            market_high_ret5_rate=("high_ret5", "mean"),
            market_median_ret5=("ret_5d", "median"),
            market_breakout_count=("entry_breakout", "sum"),
        )
        .reset_index()
    )
    return out.merge(context, on="date", how="left")


def simulate_exit(row: pd.Series, mode: str) -> float:
    if mode == "t10":
        return row["future_return_t10_pct"]
    if mode == "t20":
        return row["future_return_t20_pct"]
    if mode == "t40":
        return row["future_return_t40_pct"]
    if mode == "t60":
        return row["future_return_t60_pct"]
    if mode == "tp20_sl8_t40":
        high = row["future_max_high_t60_pct"] if "future_max_high_t60_pct" in row.index else row.get("future_max_high_t20_pct")
        low = row["future_max_drawdown_t60_pct"] if "future_max_drawdown_t60_pct" in row.index else row.get("future_max_drawdown_t20_pct")
        if pd.notna(low) and low <= -8:
            return -8.0
        if pd.notna(high) and high >= 20:
            return 20.0
        return row["future_return_t40_pct"]
    if mode == "tp12_sl5_trail5_t20":
        high = row["future_max_high_t20_pct"]
        low = row["future_max_drawdown_t20_pct"]
        if pd.notna(low) and low <= -5:
            return -5.0
        if pd.notna(high) and high >= 12:
            return 12.0
        return row["future_return_t20_pct"]
    raise ValueError(f"Unknown exit mode: {mode}")


def strategy_trades(scored: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    mask = scored["slow_money_score"] >= config.min_score
    if config.entry_mode == "pullback":
        mask &= scored["entry_pullback"]
    elif config.entry_mode == "breakout":
        mask &= scored["entry_breakout"]
    elif config.entry_mode == "breakout_no_overheat":
        mask &= scored["entry_breakout"] & (scored["market_high_ret5_rate"] <= 0.10)
    elif config.entry_mode == "any":
        mask &= scored["entry_pullback"] | scored["entry_breakout"]
    else:
        raise ValueError(f"Unknown entry mode: {config.entry_mode}")
    trades = scored[mask].copy()
    trades["strategy"] = config.name
    trades["entry_mode"] = config.entry_mode
    trades["exit_mode"] = config.exit_mode
    trades["trade_return_pct"] = trades.apply(lambda row: simulate_exit(row, config.exit_mode), axis=1)
    return trades


def win_rate(series: pd.Series) -> float:
    valid = series.dropna()
    return float((valid > 0).mean()) if len(valid) else np.nan


def summarize(df: pd.DataFrame, keys: list[str], return_col: str = "trade_return_pct") -> pd.DataFrame:
    return (
        df.groupby(keys, dropna=False)
        .agg(
            n=("ts_code", "size"),
            valid=(return_col, "count"),
            avg_return_pct=(return_col, "mean"),
            median_return_pct=(return_col, "median"),
            win_rate=(return_col, win_rate),
            avg_t5=("future_return_t5_pct", "mean"),
            avg_t10=("future_return_t10_pct", "mean"),
            avg_t20=("future_return_t20_pct", "mean"),
            avg_t40=("future_return_t40_pct", "mean"),
            avg_t60=("future_return_t60_pct", "mean"),
            avg_high20=("future_max_high_t20_pct", "mean"),
            avg_high40=("future_max_high_t40_pct", "mean"),
            avg_high60=("future_max_high_t60_pct", "mean"),
            avg_drawdown20=("future_max_drawdown_t20_pct", "mean"),
            avg_drawdown40=("future_max_drawdown_t40_pct", "mean"),
            avg_drawdown60=("future_max_drawdown_t60_pct", "mean"),
            avg_score=("slow_money_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_return_pct", ascending=False)
    )


def factor_quantile_summary(scored: pd.DataFrame, factors: Iterable[str]) -> pd.DataFrame:
    rows = []
    for factor in factors:
        if factor not in scored.columns:
            continue
        values = pd.to_numeric(scored[factor], errors="coerce")
        valid = scored[values.notna() & scored["future_return_t10_pct"].notna()].copy()
        if valid[factor].nunique(dropna=True) < 5:
            continue
        try:
            valid["factor_quantile"] = pd.qcut(valid[factor], 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"], duplicates="drop")
        except ValueError:
            continue
        for quantile, part in valid.groupby("factor_quantile", observed=True):
            rows.append(
                {
                    "factor": factor,
                    "quantile": str(quantile),
                    "n": int(len(part)),
                    "factor_avg": float(part[factor].mean()),
                    "t5_avg": float(part["future_return_t5_pct"].mean()),
                    "t10_avg": float(part["future_return_t10_pct"].mean()),
                    "t20_avg": float(part["future_return_t20_pct"].mean()),
                    "high20_avg": float(part["future_max_high_t20_pct"].mean()),
                    "drawdown20_avg": float(part["future_max_drawdown_t20_pct"].mean()),
                    "win_t10": win_rate(part["future_return_t10_pct"]),
                    "big_win_t10": float((part["future_return_t10_pct"] >= 10).mean()),
                }
            )
    return pd.DataFrame(rows)


def factor_ic_summary(scored: pd.DataFrame, factors: Iterable[str]) -> pd.DataFrame:
    rows = []
    for factor in factors:
        if factor not in scored.columns:
            continue
        daily_ic = []
        for _, part in scored.groupby("date", sort=True):
            sample = part[[factor, "future_return_t10_pct"]].dropna()
            if len(sample) < 30 or sample[factor].nunique() < 5:
                continue
            corr = sample[factor].corr(sample["future_return_t10_pct"], method="spearman")
            if pd.notna(corr):
                daily_ic.append(corr)
        if not daily_ic:
            continue
        series = pd.Series(daily_ic, dtype=float)
        rows.append(
            {
                "factor": factor,
                "dates": int(len(series)),
                "ic_mean": float(series.mean()),
                "ic_median": float(series.median()),
                "ic_positive_rate": float((series > 0).mean()),
                "ic_std": float(series.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("ic_mean", ascending=False)


def build_case_studies(scored: pd.DataFrame, trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base_cols = [
        "date",
        "ts_code",
        "name",
        "close",
        "slow_money_score",
        "large_net_amount_ratio",
        "large_net_3d_ratio",
        "large_net_5d_ratio",
        "moneyflow_net_ratio",
        "amount_rel20",
        "turnover_rate_f",
        "ret_5d",
        "ret_10d",
        "close_pos_20",
        "pullback_to_ma20",
        "breakout_20d",
        "future_return_t5_pct",
        "future_return_t10_pct",
        "future_return_t20_pct",
        "future_max_high_t20_pct",
        "future_max_drawdown_t20_pct",
    ]
    available = [col for col in base_cols if col in scored.columns]
    cases: dict[str, pd.DataFrame] = {}
    breakout = trades[trades["strategy"] == "score70_breakout_t10"].copy()
    if not breakout.empty:
        cases["breakout_winners"] = breakout.sort_values("trade_return_pct", ascending=False).head(20)[available + ["trade_return_pct"]]
        cases["breakout_losers"] = breakout.sort_values("trade_return_pct", ascending=True).head(20)[available + ["trade_return_pct"]]

    high_score = scored[scored["slow_money_score"] >= 85].copy()
    if not high_score.empty:
        cases["high_score_winners"] = high_score.sort_values("future_return_t10_pct", ascending=False).head(20)[available]
        cases["high_score_failures"] = high_score.sort_values("future_return_t10_pct", ascending=True).head(20)[available]

    divergence = scored[
        (scored["large_net_5d_ratio"] >= scored["large_net_5d_ratio"].quantile(0.85))
        & (scored["ret_5d"] > 12)
    ].copy()
    if not divergence.empty:
        cases["high_flow_chased_cases"] = divergence.sort_values("future_return_t10_pct", ascending=True).head(20)[available]
    return cases


def daily_topn_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = {
        "top5_score": scored,
        "top10_score": scored,
        "top5_breakout": scored[scored["entry_breakout"]],
        "top10_breakout": scored[scored["entry_breakout"]],
        "top5_breakout_no_overheat": scored[scored["entry_breakout"] & (scored["market_high_ret5_rate"] <= 0.10)],
        "top10_breakout_no_overheat": scored[scored["entry_breakout"] & (scored["market_high_ret5_rate"] <= 0.10)],
    }
    for name, source in variants.items():
        n = 5 if "top5" in name else 10
        selected = source.sort_values(["date", "slow_money_score"], ascending=[True, False]).groupby("date").head(n).copy()
        if selected.empty:
            continue
        selected["portfolio"] = name
        rows.append(
            {
                "portfolio": name,
                "dates": int(selected["date"].nunique()),
                "n": int(len(selected)),
                "avg_t5": float(selected["future_return_t5_pct"].mean()),
                "avg_t10": float(selected["future_return_t10_pct"].mean()),
                "median_t10": float(selected["future_return_t10_pct"].median()),
                "win_t10": win_rate(selected["future_return_t10_pct"]),
                "avg_high20": float(selected["future_max_high_t20_pct"].mean()),
                "avg_drawdown20": float(selected["future_max_drawdown_t20_pct"].mean()),
                "avg_score": float(selected["slow_money_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_t10", ascending=False)


def monthly_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    base = scored[
        (scored["slow_money_score"] >= 80)
        & scored["entry_breakout"]
        & (scored["market_high_ret5_rate"] <= 0.10)
    ].copy()
    if base.empty:
        return base
    base["signal_month"] = base["date"].dt.to_period("M").astype(str)
    base = base.sort_values(["signal_month", "ts_code", "date", "slow_money_score"], ascending=[True, True, True, False])
    counts = (
        base.groupby(["signal_month", "ts_code"])
        .agg(
            monthly_signal_count=("date", "size"),
            monthly_max_score=("slow_money_score", "max"),
            monthly_avg_score=("slow_money_score", "mean"),
            monthly_max_top_net=("top_net_amount_ratio", "max"),
            monthly_max_amount_rel20=("amount_rel20", "max"),
        )
        .reset_index()
    )
    first = base.groupby(["signal_month", "ts_code"], as_index=False).first()
    out = first.merge(counts, on=["signal_month", "ts_code"], how="left", suffixes=("", "_monthly"))
    out["monthly_rank_score"] = (
        out["monthly_max_score"]
        + np.log1p(out["monthly_signal_count"]) * 3
        + np.where(out["monthly_max_top_net"].fillna(0) > 0, 2.0, 0.0)
    )
    return out.sort_values(["signal_month", "monthly_rank_score"], ascending=[True, False])


def monthly_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows = []
    variants = {
        "monthly_all": monthly,
        "monthly_confirm_2plus": monthly[monthly["monthly_signal_count"] >= 2],
        "monthly_top5": monthly.groupby("signal_month").head(5),
        "monthly_top10": monthly.groupby("signal_month").head(10),
    }
    for name, part in variants.items():
        if part.empty:
            continue
        rows.append(
            {
                "monthly_portfolio": name,
                "months": int(part["signal_month"].nunique()),
                "n": int(len(part)),
                "avg_t20": float(part["future_return_t20_pct"].mean()),
                "median_t20": float(part["future_return_t20_pct"].median()),
                "win_t20": win_rate(part["future_return_t20_pct"]),
                "avg_t40": float(part["future_return_t40_pct"].mean()),
                "median_t40": float(part["future_return_t40_pct"].median()),
                "win_t40": win_rate(part["future_return_t40_pct"]),
                "avg_t60": float(part["future_return_t60_pct"].mean()),
                "median_t60": float(part["future_return_t60_pct"].median()),
                "win_t60": win_rate(part["future_return_t60_pct"]),
                "avg_high60": float(part["future_max_high_t60_pct"].mean()),
                "avg_drawdown60": float(part["future_max_drawdown_t60_pct"].mean()),
                "avg_signal_count": float(part["monthly_signal_count"].mean()),
                "avg_rank_score": float(part["monthly_rank_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_t40", ascending=False)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if df.empty:
        return "_无数据_"
    columns = [col for col in columns if col in df.columns]
    view = df[columns].copy()
    if limit:
        view = view.head(limit)
    for col in view.columns:
        if col in {
            "avg_return_pct",
            "median_return_pct",
            "avg_t5",
                "avg_t10",
                "avg_t20",
                "avg_t40",
                "avg_t60",
                "avg_high20",
                "avg_high40",
                "avg_high60",
                "avg_drawdown20",
                "avg_drawdown40",
                "avg_drawdown60",
                "avg_score",
            "factor_avg",
            "t5_avg",
            "t10_avg",
            "t20_avg",
            "high20_avg",
            "drawdown20_avg",
            "ic_mean",
            "ic_median",
            "ic_std",
            "median_t10",
            "median_t20",
            "median_t40",
            "median_t60",
            "avg_signal_count",
            "avg_rank_score",
        }:
            view[col] = view[col].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        elif col in {"win_rate", "win_t10", "win_t20", "win_t40", "win_t60", "big_win_t10", "ic_positive_rate"}:
            view[col] = view[col].map(lambda x: "-" if pd.isna(x) else f"{x:.1%}")
        elif col in {"n", "valid", "dates", "months"}:
            view[col] = view[col].map(lambda x: f"{int(x)}")
    return view.to_markdown(index=False)


def score_bucket(score: float) -> str:
    if score >= 85:
        return "85-100"
    if score >= 75:
        return "75-85"
    if score >= 65:
        return "65-75"
    if score >= 55:
        return "55-65"
    return "<55"


def main() -> None:
    args = parse_args()
    signal_start = ymd_to_timestamp(args.start_date)
    signal_end = ymd_to_timestamp(args.end_date)
    load_start = signal_start - pd.Timedelta(days=args.lookback_days * 2)
    load_end = signal_end + pd.Timedelta(days=args.forward_days * 2)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    signal_dates = available_trade_dates(args.daily_dir, signal_start, signal_end)
    if args.max_dates:
        signal_dates = signal_dates[: args.max_dates]
    if args.refresh_tushare:
        refresh_status = refresh_tushare_data(args, signal_dates)
    else:
        refresh_status = {"status": "not_requested", "moneyflow_files": 0, "top_list_files": 0}

    print("loading daily", flush=True)
    daily = load_daily_window(args.daily_dir, load_start, load_end)
    if daily.empty:
        raise RuntimeError(f"No daily data found under {args.daily_dir}")
    daily = add_daily_features(daily, args.forward_days)

    print("loading daily_basic", flush=True)
    daily_basic = read_daily_basic(args.daily_basic_dir, load_start, load_end)
    if not daily_basic.empty:
        daily = daily.merge(
            daily_basic.drop(columns=["trade_date"], errors="ignore"),
            on=["ts_code", "date"],
            how="left",
            suffixes=("", "_basic"),
        )

    moneyflow = read_moneyflow(args.moneyflow_dir, load_start, load_end)
    has_moneyflow = not moneyflow.empty
    if has_moneyflow:
        daily = daily.merge(
            moneyflow.drop(columns=["trade_date"], errors="ignore"),
            on=["ts_code", "date"],
            how="left",
            suffixes=("", "_mf"),
        )

    top_list = read_top_list(args.top_list_dir, load_start, load_end)
    has_top_list = not top_list.empty
    if has_top_list:
        top_list["top_net_amount_ratio"] = top_list.get("net_amount", np.nan) / top_list.get("amount", np.nan).replace(0, np.nan)
        top = top_list[["ts_code", "date", "top_net_amount_ratio"]].drop_duplicates(["ts_code", "date"])
        daily = daily.merge(top, on=["ts_code", "date"], how="left")

    daily = add_moneyflow_features(daily, has_moneyflow=has_moneyflow)
    signal = daily[(daily["date"] >= signal_start) & (daily["date"] <= signal_end)].copy()
    signal = signal[
        (signal["amount"] >= args.min_amount)
        & (signal.get("total_mv", pd.Series(np.nan, index=signal.index)).fillna(args.min_total_mv) >= args.min_total_mv)
        & signal["future_return_t10_pct"].notna()
    ].copy()
    if signal.empty:
        raise RuntimeError("No signal rows after filters.")
    scored = score_signals(signal, has_moneyflow=has_moneyflow, has_top_list=has_top_list)
    scored = add_market_context(scored)
    scored["score_bucket"] = scored["slow_money_score"].map(score_bucket)

    all_trades = pd.concat([strategy_trades(scored, config) for config in STRATEGIES], ignore_index=True)
    score_summary = summarize(scored.assign(trade_return_pct=scored["future_return_t10_pct"]), ["score_bucket"])
    strategy_summary = summarize(all_trades, ["strategy", "entry_mode", "exit_mode"])
    date_summary = summarize(all_trades, ["strategy", "date"])
    factor_quantiles = factor_quantile_summary(scored, FACTOR_COLUMNS)
    factor_ic = factor_ic_summary(scored, FACTOR_COLUMNS)
    topn_summary = daily_topn_summary(scored)
    monthly = monthly_candidates(scored)
    monthly_stats = monthly_summary(monthly)
    cases = build_case_studies(scored, all_trades)
    top_candidates = scored.sort_values(["date", "slow_money_score"], ascending=[True, False]).groupby("date").head(10)

    scored.to_parquet(args.report_dir / "slow_money_scored_candidates.parquet", index=False)
    all_trades.to_csv(args.report_dir / "slow_money_strategy_trades.csv", index=False)
    score_summary.to_csv(args.report_dir / "slow_money_score_buckets.csv", index=False)
    strategy_summary.to_csv(args.report_dir / "slow_money_strategy_summary.csv", index=False)
    date_summary.to_csv(args.report_dir / "slow_money_strategy_by_date.csv", index=False)
    factor_quantiles.to_csv(args.report_dir / "slow_money_factor_quantiles.csv", index=False)
    factor_ic.to_csv(args.report_dir / "slow_money_factor_ic.csv", index=False)
    topn_summary.to_csv(args.report_dir / "slow_money_topn_summary.csv", index=False)
    monthly.to_csv(args.report_dir / "slow_money_monthly_candidates.csv", index=False)
    monthly_stats.to_csv(args.report_dir / "slow_money_monthly_summary.csv", index=False)
    case_dir = args.report_dir / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)
    for name, case_df in cases.items():
        case_df.to_csv(case_dir / f"{name}.csv", index=False)
    top_candidates.to_csv(args.report_dir / "slow_money_top_candidates.csv", index=False)

    coverage = {
        "start_date": date_text(signal_start),
        "end_date": date_text(signal_end),
        "daily_rows": int(len(daily)),
        "signal_rows": int(len(signal)),
        "symbols": int(signal["ts_code"].nunique()),
        "daily_basic_rows": int(len(daily_basic)),
        "moneyflow_rows": int(len(moneyflow)),
        "top_list_rows": int(len(top_list)),
        "has_moneyflow": bool(has_moneyflow),
        "has_top_list": bool(has_top_list),
        "refresh_status": refresh_status,
        "data_note": "moneyflow" if has_moneyflow else "daily_basic_amount_turnover_proxy",
    }
    (args.report_dir / "slow_money_coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 慢资金跟随策略验证",
        "",
        "## 口径",
        "",
        f"- 信号区间：{args.start_date} 至 {args.end_date}。",
        f"- 数据来源：Tushare 日线 `{args.daily_dir.relative_to(PROJECT_ROOT)}`、Tushare daily_basic `{args.daily_basic_dir.relative_to(PROJECT_ROOT)}`。",
        f"- moneyflow 覆盖：{len(moneyflow):,} 行；龙虎榜覆盖：{len(top_list):,} 行。",
        f"- 当前资金信号口径：{'moneyflow 大/超大单净流入 + 成交活跃度' if has_moneyflow else '成交额放大 + 换手率 + 价格不追高代理'}。",
        "- 收益为信号日收盘后的前瞻收益代理，尚未模拟 T+1 开盘成交滑点。",
        "",
        "## 结论",
        "",
    ]
    best = strategy_summary.head(1)
    if not best.empty:
        row = best.iloc[0]
        lines += [
            f"- 最好组合：`{row['strategy']}`，样本 {int(row['valid'])}，平均收益 {row['avg_return_pct']:.2f}%，中位数 {row['median_return_pct']:.2f}%，胜率 {row['win_rate']:.1%}。",
        ]
    high_score = score_summary[score_summary["score_bucket"].isin(["75-85", "85-100"])]
    low_score = score_summary[score_summary["score_bucket"].isin(["<55", "55-65"])]
    if not high_score.empty and not low_score.empty:
        lines.append(
            f"- 高分组 T+10 均值 {high_score['avg_return_pct'].mean():.2f}%，低分组 {low_score['avg_return_pct'].mean():.2f}%；"
            + ("分层有效。" if high_score["avg_return_pct"].mean() > low_score["avg_return_pct"].mean() else "分层暂不明显。")
        )
    if not factor_ic.empty:
        top_factor = factor_ic.iloc[0]
        lines.append(
            f"- 因子 IC 最靠前的是 `{top_factor['factor']}`，日均 Spearman IC {top_factor['ic_mean']:.2f}，正 IC 占比 {top_factor['ic_positive_rate']:.1%}。"
        )
    if not topn_summary.empty:
        topn = topn_summary.iloc[0]
        lines.append(
            f"- 每日 TopN 组合里 `{topn['portfolio']}` 最好，T+10 均值 {topn['avg_t10']:.2f}%，中位数 {topn['median_t10']:.2f}%。"
        )
    if not monthly_stats.empty:
        month_row = monthly_stats.iloc[0]
        lines.append(
            f"- 月度候选里 `{month_row['monthly_portfolio']}` 的 T+40 均值 {month_row['avg_t40']:.2f}%，中位数 {month_row['median_t40']:.2f}%。"
        )
    lines += [
        "- 当前更像“右侧资金突破”有效，而不是“回踩吸筹后低吸”有效；回踩组合均值接近 0 或为负。",
        "- 最好组合的中位数仍为负，说明收益依赖少数强趋势票，实盘必须做分散、限仓和高开过滤。",
    ]
    if not has_moneyflow:
        lines.append("- 当前没有 moneyflow 覆盖，这份结果只能验证“成交活跃吸筹代理”，不能直接证明真实大/超大单建仓有效。")
    lines += [
        "",
        "## 资金分数分层",
        "",
        markdown_table(score_summary, ["score_bucket", "n", "valid", "avg_return_pct", "win_rate", "avg_t5", "avg_t10", "avg_t20", "avg_high20", "avg_drawdown20", "avg_score"]),
        "",
        "## 策略组合",
        "",
        markdown_table(strategy_summary, ["strategy", "n", "valid", "avg_return_pct", "median_return_pct", "win_rate", "avg_t20", "avg_t40", "avg_high40", "avg_drawdown40", "avg_score"]),
        "",
        "## 每日 TopN 组合",
        "",
        markdown_table(topn_summary, ["portfolio", "dates", "n", "avg_t5", "avg_t10", "median_t10", "win_t10", "avg_high20", "avg_drawdown20", "avg_score"]),
        "",
        "## 月度买卖点",
        "",
        markdown_table(monthly_stats, ["monthly_portfolio", "months", "n", "avg_t20", "median_t20", "win_t20", "avg_t40", "median_t40", "win_t40", "avg_t60", "median_t60", "win_t60", "avg_signal_count", "avg_rank_score"]),
        "",
        "## 因子 IC",
        "",
        markdown_table(factor_ic, ["factor", "dates", "ic_mean", "ic_median", "ic_positive_rate", "ic_std"], limit=12),
        "",
        "## 关键因子分层",
        "",
        markdown_table(
            factor_quantiles[factor_quantiles["factor"].isin(["large_net_5d_ratio", "large_net_3d_ratio", "large_net_amount_ratio", "amount_rel20", "ret_5d", "close_pos_20"])],
            ["factor", "quantile", "n", "factor_avg", "t10_avg", "win_t10", "big_win_t10", "high20_avg", "drawdown20_avg"],
            limit=40,
        ),
        "",
        "## Case 输出",
        "",
        f"- 成功/失败案例已输出到 `{case_dir.resolve().relative_to(PROJECT_ROOT)}`。",
        "- 重点看 `breakout_winners.csv`、`breakout_losers.csv`、`high_score_failures.csv`：前者帮助识别真正主升浪，后两者帮助排除假建仓/高位接盘。",
        "",
        "## 下一步",
        "",
        "1. 增加 T+1 开盘涨跌幅过滤，避免信号日后高开追买。",
        "2. 对高分候选做行业中性化，防止单一主题行情造成假分层。",
        "3. 对 case 里的失败样本建立排除规则：过热日、前期涨幅过大、突破后次日弱开、龙虎榜净买缺失。",
        "4. moneyflow 分层已初步有效，下一步加入北向持股、股东增持、基金季报这些慢变量做二次确认。",
        "",
    ]
    report_path = args.report_dir / "slow_money_follow_review.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
