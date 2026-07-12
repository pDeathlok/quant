"""Market sentiment feature builders for daily stock-selection research."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize_ts_code(symbol: str) -> str:
    text = str(symbol)
    if "." in text:
        return text
    if text.startswith(("6", "9")):
        return f"{text}.SH"
    if text.startswith(("0", "2", "3")):
        return f"{text}.SZ"
    if text.startswith(("4", "8")):
        return f"{text}.BJ"
    return text


def build_limit_proxy_features(daily_dir: Path, start: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Build market mood proxies from local daily OHLCV files.

    This does not replace exchange/Tushare limit-list data. It is a robust local
    fallback that approximates broad sentiment with daily return thresholds.
    """
    start_ts = pd.to_datetime(start) if start is not None else None
    rows: list[pd.DataFrame] = []
    for path in sorted(daily_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(path, columns=["date", "close", "volume"])
        except Exception:
            continue
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").dropna(subset=["date", "close"])
        if start_ts is not None:
            df = df[df["date"] >= start_ts]
        if len(df) < 2:
            continue
        df["ret_1d_pct"] = df["close"].pct_change() * 100
        rows.append(df[["date", "ret_1d_pct", "volume"]])
    if not rows:
        return pd.DataFrame()

    all_daily = pd.concat(rows, ignore_index=True)
    all_daily["limit_up_proxy"] = all_daily["ret_1d_pct"] >= 9.5
    all_daily["limit_down_proxy"] = all_daily["ret_1d_pct"] <= -9.5
    all_daily["strong_up_proxy"] = all_daily["ret_1d_pct"] >= 5.0
    all_daily["strong_down_proxy"] = all_daily["ret_1d_pct"] <= -5.0
    all_daily["above_zero"] = all_daily["ret_1d_pct"] > 0
    market = (
        all_daily.groupby("date")
        .agg(
            market_stock_count=("ret_1d_pct", "count"),
            limit_up_count_proxy=("limit_up_proxy", "sum"),
            limit_down_count_proxy=("limit_down_proxy", "sum"),
            strong_up_count_proxy=("strong_up_proxy", "sum"),
            strong_down_count_proxy=("strong_down_proxy", "sum"),
            market_up_ratio=("above_zero", "mean"),
            market_median_ret_1d=("ret_1d_pct", "median"),
            market_total_volume=("volume", "sum"),
        )
        .reset_index()
    )
    denom = market["market_stock_count"].replace(0, np.nan)
    market["limit_up_ratio_proxy"] = market["limit_up_count_proxy"] / denom
    market["limit_down_ratio_proxy"] = market["limit_down_count_proxy"] / denom
    market["strong_up_ratio_proxy"] = market["strong_up_count_proxy"] / denom
    market["strong_down_ratio_proxy"] = market["strong_down_count_proxy"] / denom
    market["market_sentiment_5d"] = market["limit_up_ratio_proxy"].rolling(5, min_periods=2).mean()
    market["market_panic_5d"] = market["limit_down_ratio_proxy"].rolling(5, min_periods=2).mean()
    return market.replace([np.inf, -np.inf], np.nan)


def read_top_list_features(top_list_dir: Path, start: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Read local Tushare top-list files and derive stock/date features."""
    start_ts = pd.to_datetime(start) if start is not None else None
    frames: list[pd.DataFrame] = []
    for path in sorted(top_list_dir.glob("*top_list_*.parquet")):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if "trade_date" not in df.columns:
            date_part = path.stem.rsplit("_", 1)[-1]
            df["trade_date"] = date_part
        df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        if start_ts is not None:
            df = df[df["date"] >= start_ts]
        if df.empty or "ts_code" not in df.columns:
            continue
        for col in ["net_amount", "amount", "buy", "sell", "net_rate", "pct_change"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw["top_net_amount_ratio"] = raw.get("net_amount", np.nan) / raw.get("amount", np.nan).replace(0, np.nan)
    agg = (
        raw.groupby(["ts_code", "date"])
        .agg(
            top_list_count=("ts_code", "size"),
            top_net_amount=("net_amount", "sum"),
            top_amount=("amount", "sum"),
            top_net_rate=("net_rate", "mean"),
            top_pct_change=("pct_change", "mean"),
        )
        .reset_index()
    )
    agg["top_net_amount_ratio"] = agg["top_net_amount"] / agg["top_amount"].replace(0, np.nan)
    return agg.replace([np.inf, -np.inf], np.nan)
