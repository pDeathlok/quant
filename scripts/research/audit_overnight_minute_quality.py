"""Snapshot available minute chunks and audit them without filtering trades."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from quant.research.overnight_minutes import session_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "reports/research/overnight_baostock_2024_2026"
    paths = sorted(source.glob("*/[0-9][0-9][0-9][0-9].parquet"))
    if not paths:
        raise ValueError("No completed minute chunks")
    args.output.mkdir(exist_ok=False)
    manifest = {"scope": "available complete chunks only, not the full requested pool",
                "use": "quality diagnosis, never a future-outcome trade-selection gate",
                "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    symbols = sorted({p.parent.name for p in paths})
    raw = ds.dataset(ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
    daily = raw.to_table(columns=["ts_code", "trade_date", "open", "high", "low", "close", "volume", "amount"],
        filter=ds.field("ts_code").isin(symbols) & (ds.field("trade_date") >= "20240101")
        & (ds.field("trade_date") <= "20260911")).to_pandas()
    daily["date"] = pd.to_datetime(daily.trade_date)
    summaries, details = [], []
    for path in paths:
        frame = pd.read_parquet(path)
        symbol, year = path.parent.name, int(path.stem)
        if frame.empty:
            summaries.append({"symbol": symbol, "year": year, "minute_days": 0})
            continue
        frame["date"] = pd.to_datetime(frame.trade_time).dt.normalize()
        groups = frame.groupby("date", sort=True)
        grouped = groups.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                             close=("close", "last"), volume=("vol", "sum"), amount=("amount", "sum"))
        grouped["complete_grid"] = groups.trade_time.apply(
            lambda x: pd.DatetimeIndex(x).equals(session_grid(x.iloc[0], include_auction=False)))
        reference = daily[daily.ts_code.eq(symbol) & daily.date.dt.year.eq(year)]
        paired = grouped.reset_index().merge(reference, on="date", suffixes=("_minute", "_daily"),
                                             how="outer", validate="one_to_one", indicator=True)
        paired["symbol"] = symbol
        row = {"symbol": symbol, "year": year, "minute_days": len(grouped),
               "reference_days": len(reference), "matched_days": int(paired._merge.eq("both").sum()),
               "complete_grids": int(grouped.complete_grid.sum())}
        matched = paired._merge.eq("both")
        for col in ("open", "high", "low", "close"):
            delta = paired[f"{col}_minute"] - paired[f"{col}_daily"]
            paired[f"delta_{col}"] = delta
            row[f"{col}_mismatches"] = int(delta[matched].abs().gt(.011).sum())
            row[f"{col}_max_abs_diff"] = float(delta[matched].abs().max())
        paired["volume_ratio"] = paired.volume_minute / (paired.volume_daily * 100).replace(0, np.nan)
        paired["amount_ratio"] = paired.amount_minute / (paired.amount_daily * 1000).replace(0, np.nan)
        row["volume_ratio_median"] = float(paired.volume_ratio[matched].median())
        row["amount_ratio_median"] = float(paired.amount_ratio[matched].median())
        summaries.append(row)
        details.append(paired)
    result = pd.DataFrame(summaries)
    result.to_csv(args.output / "summary.csv", index=False)
    if details:
        pd.concat(details, ignore_index=True).to_parquet(args.output / "daily_comparison.parquet", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
