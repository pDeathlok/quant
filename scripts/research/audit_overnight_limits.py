"""Fetch exact historical price limits for locked picks; never infer fills."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "reports/research/overnight_direct_v1"
    output = source / "limit_audit"
    output.mkdir(exist_ok=True)
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not token:
        raise RuntimeError("Tushare credential unavailable")
    picks = pd.read_parquet(source / "tp1_locked_picks.parquet")
    symbols = sorted(picks.symbol.unique())
    # Scope is the selected stock-days, not a new screen of confirmation outcomes.
    collected = []
    with httpx.Client(timeout=25) as client:
        for number, symbol in enumerate(symbols, 1):
            path = output / f"{symbol}.parquet"
            dates = picks.loc[picks.symbol.eq(symbol), "date"].dt.strftime("%Y%m%d")
            if path.exists():
                table = pd.read_parquet(path)
            else:
                for attempt in range(3):
                    try:
                        response = client.post("https://api.tushare.pro", json={
                            "api_name": "stk_limit", "token": token,
                            "params": {"ts_code": symbol, "start_date": dates.min(), "end_date": dates.max()},
                            "fields": "ts_code,trade_date,pre_close,up_limit,down_limit",
                        })
                        if response.status_code != 200:
                            raise RuntimeError(f"HTTP status {response.status_code}")
                        body = response.json()
                        if body.get("code") != 0 or body.get("data") is None:
                            raise RuntimeError(f"Tushare result code {body.get('code')}")
                        data = body["data"]
                        table = pd.DataFrame(data["items"], columns=data["fields"])
                        if table.duplicated(["ts_code", "trade_date"]).any():
                            raise RuntimeError("duplicate price limit keys")
                        table.to_parquet(path, index=False)
                        break
                    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                        if attempt == 2:
                            raise RuntimeError(f"Limit query failed for {symbol}; {type(exc).__name__}") from None
                        time.sleep(1)
                time.sleep(.25)
            collected.append(table)
            if number % 10 == 0:
                print(f"limits fetched {number}/{len(symbols)}", flush=True)
    limits = pd.concat(collected, ignore_index=True).rename(columns={"ts_code": "symbol"})
    limits["date"] = pd.to_datetime(limits.trade_date)
    raw = ds.dataset(ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
    bars = raw.to_table(columns=["ts_code", "trade_date", "close", "open", "high", "low"],
                        filter=ds.field("ts_code").isin(symbols)).to_pandas()
    bars["date"] = pd.to_datetime(bars.trade_date)
    bars = bars.rename(columns={"ts_code": "symbol", "close": "entry_close"})
    audited = picks.merge(limits[["symbol", "date", "up_limit", "down_limit"]],
                         on=["symbol", "date"], how="left", validate="one_to_one")
    audited = audited.merge(bars[["symbol", "date", "entry_close"]], on=["symbol", "date"],
                            how="left", validate="one_to_one")
    audited["entry_at_upper_limit"] = audited.entry_close.ge(audited.up_limit - .0051)
    audited["entry_limit_known"] = audited.entry_close.notna() & audited.up_limit.gt(0)
    audited.to_parquet(output / "audited_picks.parquet", index=False)
    groups = []
    for name, mask in [("all", pd.Series(True, index=audited.index)),
                       ("at_upper_limit", audited.entry_limit_known & audited.entry_at_upper_limit),
                       ("below_upper_limit", audited.entry_limit_known & ~audited.entry_at_upper_limit),
                       ("unknown_entry_limit", ~audited.entry_limit_known)]:
        part = audited[mask]
        groups.append({"group": name, "n": len(part), "known_outcome": int(part.net.notna().sum()),
                       "win_known": float(part.net.dropna().gt(0).mean()), "mean_net_known": float(part.net.mean())})
    summary = {"source": "Tushare stk_limit plus canonical raw close",
               "note": "at limit is a fill risk, not proof of no fill; below limit is not a fill guarantee",
               "groups": groups}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
