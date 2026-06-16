#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Incrementally refresh the production B1 feature cache without retraining.

Daily selector refresh needs fresh feature rows for the newest trading date, but
it should not wait for a full XGBoost retrain. This script rebuilds rows from an
incremental start date, merges them into the existing parquet cache, and keeps
the same deterministic train/test/oot split policy used by model training.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

from quant.features.variable_library import merge_daily_basic_features
from train_b1_tushare_models import assign_symbol_splits, build_dataset


def _parse_date(value: str) -> pd.Timestamp:
    if value.isdigit() and len(value) == 8:
        return pd.to_datetime(value, format="%Y%m%d")
    return pd.to_datetime(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally refresh B1 feature cache")
    parser.add_argument("--daily-dir", type=Path, default=PROJECT_ROOT / "data/raw/daily")
    parser.add_argument("--daily-basic-dir", type=Path, default=PROJECT_ROOT / "data/raw/daily_basic")
    parser.add_argument("--dataset-out", type=Path, default=PROJECT_ROOT / "data/features/b1/training_xgb_project_vars.parquet")
    parser.add_argument("--incremental-start-date", required=True)
    parser.add_argument("--oot-start", default="2025-01-01")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--executor", choices=["threads", "processes"], default="threads")
    parser.add_argument("--adaptive-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-workers", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=160)
    parser.add_argument("--worker-step", type=int, default=16)
    parser.add_argument("--load-target", type=float, default=0.80)
    parser.add_argument("--load-hard-limit", type=float, default=1.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_ts = _parse_date(args.incremental_start_date)
    start_str = start_ts.strftime("%Y%m%d")

    incremental = build_dataset(
        args.daily_dir,
        start_str,
        workers=args.workers,
        executor_type=args.executor,
        adaptive_workers=args.adaptive_workers,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        worker_step=args.worker_step,
        load_target=args.load_target,
        load_hard_limit=args.load_hard_limit,
    )
    incremental = merge_daily_basic_features(incremental, args.daily_basic_dir)

    if args.dataset_out.exists():
        existing = pd.read_parquet(args.dataset_out)
        existing["date"] = pd.to_datetime(existing["date"])
        kept = existing[existing["date"] < start_ts].copy()
        combined = pd.concat([kept, incremental], ignore_index=True, sort=False)
    else:
        combined = incremental

    combined["date"] = pd.to_datetime(combined["date"])
    combined = (
        combined.sort_values(["date", "symbol"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    combined = assign_symbol_splits(combined, args.oot_start, args.test_size, args.random_state)

    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.dataset_out, index=False)

    result = {
        "status": "success",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "incremental_start_date": start_ts.strftime("%Y-%m-%d"),
        "incremental_rows": int(len(incremental)),
        "total_rows": int(len(combined)),
        "date_min": combined["date"].min().strftime("%Y-%m-%d") if not combined.empty else None,
        "date_max": combined["date"].max().strftime("%Y-%m-%d") if not combined.empty else None,
        "output": str(args.dataset_out),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
