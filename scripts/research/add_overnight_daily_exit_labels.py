"""Add official next-session open/close labels without changing preclose inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from quant.research.overnight_minutes import daily_exit_labels

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "reports/research/overnight_minute_dataset_v1"
    output = ROOT / "reports/research/overnight_daily_exit_dataset_v1"
    if not (source / "complete.json").exists():
        raise ValueError("Full source dataset incomplete")
    frame = pd.read_parquet(source / "dataset.parquet")
    symbols = sorted(frame.symbol.unique())
    raw = ds.dataset(ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
    daily = raw.to_table(columns=["ts_code", "trade_date", "open", "close", "pre_close"],
        filter=ds.field("ts_code").isin(symbols) & (ds.field("trade_date") >= "20240101")
        & (ds.field("trade_date") <= "20260911")).to_pandas()
    daily["target_date"] = pd.to_datetime(daily.trade_date)
    daily = daily.rename(columns={"ts_code": "symbol", "open": "next_open", "close": "next_close",
                                  "pre_close": "next_pre_close"})
    joined = frame.merge(daily.drop(columns="trade_date"), on=["symbol", "target_date"],
                         how="left", validate="many_to_one", sort=False)
    pd.testing.assert_frame_equal(frame, joined[frame.columns])
    labels = pd.DataFrame([daily_exit_labels(row.entry_close_label, row.next_pre_close, row.next_open, row.next_close)
                           for row in joined.itertuples()])
    result = pd.concat([frame, labels], axis=1)
    output.mkdir(exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest.update(parent_dataset_sha256=hashlib.sha256((source / "dataset.parquet").read_bytes()).hexdigest(),
        label_source="canonical daily_partitioned exact next market date, no minute-price agreement gate",
        label_note="20bps price proxies; corporate actions unresolved; no actual fill guarantee")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    daily.to_parquet(output / "daily_label_source.parquet", index=False)
    result.to_parquet(output / "dataset.parquet", index=False)
    summary = {"rows": len(result), "stocks": len(symbols), "eligible": int(result.decision_eligible.sum()),
        "eligible_known": {name: int(result.loc[result.decision_eligible, f"target_net_{name}"].notna().sum())
                           for name in ("open", "close")}}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
