"""Current-day broad-market context, with causal rolling summaries."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ASHARE = r"(?:(?:00[0-3]|30[01])\d{3}\.SZ|(?:60[0135]|68[89])\d{3}\.SH|(?:[48]\d{5}|92\d{4})\.BJ)"


def market_context(bars: pd.DataFrame, minimum_stocks: int = 1000) -> pd.DataFrame:
    bars = bars.copy()
    if "volume" in bars:
        bars["vol"] = bars["vol"].fillna(bars["volume"])
    if bars.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Duplicate market bars")
    valid = bars.ts_code.str.fullmatch(ASHARE) & bars[["open", "high", "low", "close", "pre_close", "vol"]].gt(0).all(axis=1)
    valid &= bars.high.ge(bars[["open", "close"]].max(axis=1) - 1e-8)
    valid &= bars.low.le(bars[["open", "close"]].min(axis=1) + 1e-8)
    b = bars.loc[valid].copy()
    b["ret"] = b.close / b.pre_close - 1
    b["gap"] = b.open / b.pre_close - 1
    b["intraday"] = b.close / b.open - 1
    b["range"] = (b.high - b.low) / b.pre_close
    b["position"] = (b.close - b.low) / (b.high - b.low).replace(0, np.nan)
    b["positive"] = b.ret.gt(0)
    b["up5"] = b.ret.ge(.05)
    b["down5"] = b.ret.le(-.05)
    g = b.groupby("trade_date", sort=True)
    context = g.agg(stocks=("ts_code", "nunique"), breadth=("positive", "mean"),
        return_mean=("ret", "mean"), return_median=("ret", "median"), return_std=("ret", "std"),
        gap_mean=("gap", "mean"), intraday_mean=("intraday", "mean"), range_mean=("range", "mean"),
        position_mean=("position", "mean"), up5_fraction=("up5", "mean"), down5_fraction=("down5", "mean"))
    context["return_q10"] = g.ret.quantile(.1)
    context["return_q90"] = g.ret.quantile(.9)
    observed_dates = set(bars.loc[bars.ts_code.str.fullmatch(ASHARE), "trade_date"])
    if context.empty or observed_dates != set(context.index) or context.stocks.lt(minimum_stocks).any():
        raise ValueError("Insufficient broad-market coverage")
    for window in [5, 20]:
        context[f"breadth_mean_{window}"] = context.breadth.rolling(window, min_periods=window).mean()
        context[f"return_mean_{window}"] = context.return_mean.rolling(window, min_periods=window).mean()
    context["return_time_std20"] = context.return_mean.rolling(20, min_periods=20).std()
    context.index = pd.to_datetime(context.index)
    context.index.name = "date"
    return context.add_prefix("today_market_").reset_index()


def attach_market_context(frame: pd.DataFrame, root: Path, output: Path) -> tuple[pd.DataFrame, list[str]]:
    paths = [p for p in sorted((root / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
             if "202310" <= p.parent.name.split("=")[1] <= "202609"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    raw = ds.dataset([str(p) for p in paths], format="parquet")
    bars = raw.to_table(columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "volume"],
        filter=(ds.field("trade_date") >= "20231001") & (ds.field("trade_date") <= "20260904")).to_pandas()
    context = market_context(bars)
    joined = frame.merge(context, on="date", how="left", validate="many_to_one")
    pd.testing.assert_frame_equal(frame, joined[frame.columns])
    cols = [c for c in context if c.startswith("today_market_")]
    if joined[cols].isna().any().any():
        raise ValueError("Missing market day or rolling warmup")
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Market source changed")
    context.to_parquet(output / "market_context.parquet", index=False)
    (output / "market_context_sources.json").write_text(json.dumps({"files": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "columns": cols,
        "population": "all locally observed A-share daily bars passing positive OHLC/preclose/volume and OHLC checks",
        "volume_alias": "use volume only where vol is missing; no synthetic values or zero-volume override",
        "timing": "current EOD plus trailing windows; 14:30 is user deployment approximation",
        "caveat": "minimum1000 stocks is a coverage guard, not proof of an exhaustive historical market"}, indent=2))
    return joined, cols
