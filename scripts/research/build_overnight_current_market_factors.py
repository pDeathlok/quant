"""Recompute market-derived daily factors on every historical candidate key."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from quant.data.source_merge import normalize_tushare_daily
from quant.features.project_factor_layer import calculate_project_market_factors
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "project-v5-canonical-alias-free"


def compute(task: tuple[str, pd.DataFrame, pd.DataFrame, Path]) -> dict:
    symbol, history, keys, output = task
    history = normalize_tushare_daily(history, symbol).sort_values("date").reset_index(drop=True)
    if history.duplicated("date").any():
        raise ValueError(f"Duplicate history: {symbol}")
    factors = calculate_project_market_factors(history, symbol, factor_schema_version=SCHEMA)
    cols = [c for c in PROJECT_FACTOR_COLUMNS if c in factors]
    factors["market_row_present"] = True
    result = keys.merge(factors[["symbol", "date", "market_row_present", *cols]],
                        on=["symbol", "date"], how="left", validate="one_to_one")
    result[cols] = result[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    path = output / f"{symbol}.parquet"
    result.to_parquet(path, index=False)
    return {"symbol": symbol, "rows": len(result), "market_rows": int(result.market_row_present.eq(True).sum()),
            "features": len(cols), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unfiltered", action="store_true", help="Use all observed daily keys, with unchanged formulas and warmup")
    parser.add_argument("--event-extension", action="store_true", help="Locked September candidate extension with historical boundary keys")
    args = parser.parse_args()
    if args.unfiltered and args.event_extension:
        parser.error("Choose one candidate universe")
    source = ROOT / "reports/research/tail_overnight/dataset.parquet"
    output = ROOT / "reports/research/overnight_current_market_factors_v1"
    if args.unfiltered:
        source = ROOT / "reports/research/overnight_unfiltered_universe_v1/daily_bars.parquet"
        output = ROOT / "reports/research/overnight_unfiltered_market_factors_v1"
    history_end = "2026-09-04"
    extra_sources = {}
    if args.event_extension:
        source = ROOT / "reports/research/overnight_event_extension_pool_v1/entry_pool.parquet"
        output = ROOT / "reports/research/overnight_event_extension_market_factors_v1"
        check = json.loads((source.parent / "complete.json").read_text())
        if hashlib.sha256(source.read_bytes()).hexdigest() != check["dataset_sha256"]:
            raise ValueError("Extension pool differs from audited keys")
        boundary_path = source.parent / "boundary_comparison.parquet"
        extra_sources[str(boundary_path)] = hashlib.sha256(boundary_path.read_bytes()).hexdigest()
        history_end = "2026-09-10"
    output.mkdir(exist_ok=False)
    shards = output / "by_symbol"
    shards.mkdir()
    keys = pd.read_parquet(source, columns=["symbol", "date"])
    if args.event_extension:
        boundary = pd.read_parquet(boundary_path)
        if not boundary._merge.eq("both").all():
            raise ValueError("Unaudited boundary membership")
        keys = pd.concat([boundary[["symbol", "date"]], keys], ignore_index=True)
    if keys.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate candidate keys")
    raw_root = ROOT / "data/raw/daily_partitioned"
    paths = [p for p in sorted(raw_root.glob("year_month=*/data.parquet"))
             if "201801" <= p.parent.name.split("=")[1] <= "202609"]
    manifest = {"candidate_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "schema": SCHEMA,
                "history_start": "2018-01-01", "history_end": history_end, "candidate_rows": len(keys),
                "stocks": int(keys.symbol.nunique()), "workers": 4,
                "stage": "OHLCV-derived factors only; daily_basic merge still required",
                "timing": "current EOD daily research, user substitutes14:30 daily snapshot; no minute inputs",
                "missing": "retain keys without raw bars or insufficient history; no future carry/backfill",
                "limitations": "six-year minimum warmup; vendor revisions and historical ST-name metadata not fully audited",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "formula_files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                    ROOT / "src/quant/features/project_factor_layer.py",
                    ROOT / "src/quant/features/variable_library.py",
                    ROOT / "src/quant/data/source_merge.py"]},
                "daily_files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    if extra_sources:
        manifest["extra_candidate_sources"] = extra_sources
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    raw = ds.dataset([str(p) for p in paths], format="parquet")
    daily = raw.to_table(columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
        "vol", "volume", "amount", "pct_chg", "change"], filter=(ds.field("trade_date") >= "20180101")
        & (ds.field("trade_date") <= history_end.replace("-", "")) & ds.field("ts_code").isin(keys.symbol.unique())).to_pandas()
    print(f"loaded {len(daily)} daily bars for {keys.symbol.nunique()} candidate stocks", flush=True)
    if daily.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Duplicate raw daily keys")
    key_groups = {s: g for s, g in keys.groupby("symbol", sort=False)}
    completed = []
    # Bound outstanding tasks so large history frames are not all duplicated in the queue.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = []
        for symbol, history in daily.groupby("ts_code", sort=True):
            pending.append(pool.submit(compute, (symbol, history.copy(), key_groups[symbol], shards)))
            if len(pending) == 8:
                completed.extend(future.result() for future in pending)
                pending.clear()
                if len(completed) % 80 == 0:
                    print(f"market factor shards: {len(completed)}/{len(key_groups)}", flush=True)
        completed.extend(future.result() for future in pending)
    missing_symbols = set(key_groups) - {r["symbol"] for r in completed}
    if missing_symbols:
        raise ValueError(f"No raw history for candidate symbols: {sorted(missing_symbols)}")
    for path, digest in {**manifest["daily_files"], **extra_sources}.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Source changed during build")
    pd.DataFrame(completed).to_csv(output / "coverage.csv", index=False)
    summary = {"stage": "market factors only; NOT complete145-factor training dataset",
               "stocks": len(completed), "candidate_rows": sum(r["rows"] for r in completed),
               "market_rows": sum(r["market_rows"] for r in completed),
               "feature_counts": sorted({r["features"] for r in completed})}
    (output / "market_complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
