"""Official daily price proxies for pre-set next-session limit exits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from quant.research.overnight_minutes import daily_limit_exit

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "reports/research/overnight_minute_dataset_v1"
    output = ROOT / "reports/research/overnight_limit_exit_dataset_v1"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["smoke_only"] or not (source / "complete.json").exists():
        raise ValueError("Complete full parent required")
    frame = pd.read_parquet(source / "dataset.parquet")
    symbols = sorted(frame.symbol.unique())
    raw = ds.dataset(ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
    daily = raw.to_table(columns=["ts_code", "trade_date", "high", "close", "pre_close", "vol"],
        filter=ds.field("ts_code").isin(symbols) & (ds.field("trade_date") >= "20240101")
        & (ds.field("trade_date") <= "20260911")).to_pandas()
    daily["target_date"] = pd.to_datetime(daily.trade_date)
    daily = daily.rename(columns={"ts_code": "symbol", "high": "next_high", "close": "next_close",
                                  "pre_close": "next_pre_close", "vol": "next_vol"})
    limits = pd.concat([pd.read_parquet(ROOT / f"reports/research/overnight_watchlist_limits_2024_2026/{s}.parquet")
                        for s in symbols], ignore_index=True)
    limits["target_date"] = pd.to_datetime(limits.trade_date)
    limits = limits.rename(columns={"ts_code": "symbol"})
    joined = frame.merge(daily.drop(columns="trade_date"), on=["symbol", "target_date"], how="left", validate="many_to_one")
    joined = joined.merge(limits[["symbol", "target_date", "down_limit"]], on=["symbol", "target_date"],
                           how="left", validate="many_to_one")
    pd.testing.assert_frame_equal(frame, joined[frame.columns])
    records = []
    for row in joined.itertuples():
        record = {}
        for name, threshold in (("tp05_limit", .005), ("tp1_limit", .01)):
            result = daily_limit_exit(row.entry_close_label, row.next_pre_close, row.next_high, row.next_close, threshold)
            status = result["status"]
            if status == "ok":
                if not np.isfinite(row.next_vol) or row.next_vol <= 0:
                    status = "no_observed_daily_volume"
                elif not result["penetrated"] and (not np.isfinite(row.down_limit)
                        or row.next_close <= row.down_limit + .0051):
                    status = "close_exit_limit_risk"
            record[f"target_net_{name}"] = result["net"] if status == "ok" else np.nan
            record[f"net_proxy_{name}"] = result["net"]
            record[f"label_quality_{name}"] = status
        records.append(record)
    result = pd.concat([frame, pd.DataFrame(records)], axis=1)
    output.mkdir(exist_ok=False)
    manifest.update(parent_dataset_sha256=hashlib.sha256((source / "dataset.parquet").read_bytes()).hexdigest(),
        label_source="official daily high/close, exact next market date",
        limit_policy="round sell limit upward to 0.01; require high>=limit+entry*0.001; no opening improvement;"
        " no intraday stop; precommitted close fallback; close at lower limit remains unresolved",
        execution="reachability proxy only; penetration does not guarantee queue fills")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    joined[["symbol", "date", "target_date", "next_high", "next_close", "next_pre_close", "next_vol", "down_limit"]].to_parquet(
        output / "daily_label_source.parquet", index=False)
    result.to_parquet(output / "dataset.parquet", index=False)
    summary = {"rows": len(result), "stocks": len(symbols), "eligible": int(result.decision_eligible.sum()),
        "eligible_known": {name: int(result.loc[result.decision_eligible, f"target_net_{name}"].notna().sum())
                           for name in ("tp05_limit", "tp1_limit")}}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
