"""Market sentiment feature builders for daily stock-selection research."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.data.source_merge import normalize_ts_code as _normalize_ts_code

def normalize_ts_code(symbol: str) -> str:
    """Compatibility export for the project's canonical symbol normalizer."""
    return _normalize_ts_code(symbol)


def build_limit_proxy_features(daily_dir: Path, start: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Build market mood proxies from local daily OHLCV files.

    This does not replace exchange/Tushare limit-list data. It is a robust local
    fallback that approximates broad sentiment with daily return thresholds.
    """
    start_ts = pd.to_datetime(start) if start is not None else None
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
    read_start = (start_ts - pd.Timedelta(days=10)).strftime("%Y%m%d") if start_ts is not None else None
    all_daily = store.read_market_range(
        daily_dir.name,
        start_date=read_start,
        columns=["ts_code", "trade_date", "date", "close", "volume"],
    )
    if all_daily.empty:
        return pd.DataFrame()
    all_daily["date"] = pd.to_datetime(all_daily["date"], errors="coerce")
    all_daily = all_daily.sort_values(["ts_code", "date"]).dropna(subset=["date", "close"])
    all_daily["ret_1d_pct"] = all_daily.groupby("ts_code")["close"].pct_change() * 100
    if start_ts is not None:
        all_daily = all_daily[all_daily["date"] >= start_ts]
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
