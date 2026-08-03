#!/usr/bin/env python
"""Backfill Tushare datasets needed by long-horizon factor research."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import tushare as ts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.long_factor_backfill import (
    PROJECT_START_DATE,
    RequestPolicy,
    backfill_holder_trade,
    backfill_pledge_by_symbol,
    backfill_stock_universe,
    backfill_trade_date_partitions,
    make_audit_directory,
    publish_manifest,
)


RAW_DIR = PROJECT_ROOT / "data/raw"
AUDIT_ROOT = RAW_DIR / "source_audit"
DATASETS = (
    "stock_universe",
    "margin_detail",
    "moneyflow",
    "top_list",
    "holder_trade",
    "pledge_stat",
    "pledge_detail",
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--start", default=PROJECT_START_DATE)
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-retry-wait", type=float, default=60.0)
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--max-years", type=int, default=None)
    parser.add_argument("--symbol-offset", type=int, default=0)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--skip-compact", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = perf_counter()
    args = parse_args()
    load_env_file(PROJECT_ROOT / ".env")
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    pro = ts.pro_api(token)
    policy = RequestPolicy(
        retries=max(1, args.retries),
        sleep_seconds=max(0.0, args.sleep),
        max_retry_wait_seconds=max(0.0, args.max_retry_wait),
    )
    audit_dir = make_audit_directory(AUDIT_ROOT)
    results: dict[str, dict] = {}
    universe = None
    if "stock_universe" in args.datasets:
        universe, results["stock_universe"] = backfill_stock_universe(
            pro,
            RAW_DIR,
            policy=policy,
        )
        print(json.dumps(results["stock_universe"], ensure_ascii=False), flush=True)
    for dataset in ("margin_detail", "moneyflow", "top_list"):
        if dataset not in args.datasets:
            continue
        results[dataset] = backfill_trade_date_partitions(
            pro,
            dataset,
            args.start,
            args.end,
            RAW_DIR,
            audit_dir,
            policy=policy,
            force=args.force,
            max_dates=args.max_dates,
        )
        print(json.dumps(results[dataset], ensure_ascii=False), flush=True)
    if "holder_trade" in args.datasets:
        results["holder_trade"] = backfill_holder_trade(
            pro,
            args.start,
            args.end,
            RAW_DIR,
            audit_dir,
            policy=policy,
            force=args.force,
            max_years=args.max_years,
        )
        print(json.dumps(results["holder_trade"], ensure_ascii=False), flush=True)
    pledge_datasets = [
        dataset for dataset in ("pledge_stat", "pledge_detail") if dataset in args.datasets
    ]
    if pledge_datasets:
        if universe is None:
            universe, _ = backfill_stock_universe(pro, RAW_DIR, policy=policy)
        symbols = universe["ts_code"].dropna().astype(str).tolist()
        for dataset in pledge_datasets:
            results[dataset] = backfill_pledge_by_symbol(
                pro,
                dataset,
                symbols,
                RAW_DIR,
                audit_dir,
                policy=policy,
                force=args.force,
                symbol_offset=args.symbol_offset,
                max_symbols=args.max_symbols,
                compact_output=not args.skip_compact,
            )
            print(json.dumps(results[dataset], ensure_ascii=False), flush=True)
    status = (
        "failed"
        if any(item.get("status") == "failed" for item in results.values())
        else (
            "partial"
            if any(item.get("status") in {"partial", "deferred"} for item in results.values())
            else "success"
        )
    )
    manifest = {
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": args.start,
        "end_date": args.end,
        "datasets": results,
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    manifest_path = publish_manifest(audit_dir, manifest)
    manifest["manifest_path"] = str(manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
