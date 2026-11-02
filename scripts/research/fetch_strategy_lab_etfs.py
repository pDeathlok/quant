"""Fetch historical ETF market bars and adjustment factors from Tushare."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/strategy_lab_20260930/source"
SYMBOLS = ("511010.SH", "518880.SH", "513100.SH", "513500.SH", "159920.SZ")


def main() -> None:
    token = dotenv_values(ROOT / ".env").get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN missing")
    pro = ts.pro_api(token)
    OUT.mkdir(parents=True, exist_ok=True)
    for code in SYMBOLS:
        path = OUT / f"{code.replace('.', '_')}.parquet"
        if path.exists():
            print(f"{code}: cached", flush=True)
            continue
        bars, factors = [], []
        for year in range(2017, 2027):
            lo, hi = f"{year}0101", min(f"{year}1231", "20260924")
            if lo > hi:
                continue
            bars.append(pro.fund_daily(ts_code=code, start_date=lo, end_date=hi,
                                       fields="ts_code,trade_date,close,amount"))
            factors.append(pro.fund_adj(ts_code=code, start_date=lo, end_date=hi))
        daily = pd.concat(bars, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
        adj = pd.concat(factors, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
        frame = daily.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
        frame = frame.sort_values("trade_date")
        if frame.empty or frame.adj_factor.isna().any():
            raise RuntimeError(f"Incomplete Tushare fund_adj for {code}")
        frame.to_parquet(path, index=False)
        print(f"{code}: {frame.trade_date.min()} to {frame.trade_date.max()}, {len(frame)} rows", flush=True)


if __name__ == "__main__":
    main()
