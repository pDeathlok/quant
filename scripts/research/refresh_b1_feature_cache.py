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
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.data.atomic_io import atomic_write_parquet
from quant.features.project_factor_layer import (
    LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION,
    resolve_project_factor_schema,
)
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
    parser.add_argument("--gate-cache", type=Path, default=None)
    parser.add_argument("--gate-manifest", type=Path, default=None)
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
    started = perf_counter()
    args = parse_args()
    factor_schema_version = resolve_project_factor_schema()
    start_ts = _parse_date(args.incremental_start_date)
    start_str = start_ts.strftime("%Y%m%d")

    gate_mode = "full_scan"
    gate_reason = None
    candidate_symbols: list[str] | None = None
    gate_source_symbol_count: int | None = None
    source_store = MarketDataStore(
        MarketDataStoreConfig.from_env(root=args.daily_dir.parent)
    )
    source_recent = source_store.read_market_range(
        args.daily_dir.name,
        start_date=start_str,
    )
    source_dates = pd.to_datetime(
        source_recent.get("date", source_recent.get("trade_date")),
        errors="coerce",
    )
    actual_source_latest = source_dates.max()
    if args.gate_cache is not None and args.gate_manifest is not None:
        try:
            gate_manifest = json.loads(
                args.gate_manifest.read_text(encoding="utf-8")
            )
            gate_processed_through = pd.Timestamp(
                gate_manifest["processed_through_date"]
            )
            if (
                gate_manifest.get("status") != "success"
                or pd.isna(actual_source_latest)
                or gate_processed_through.normalize()
                != actual_source_latest.normalize()
            ):
                raise ValueError(
                    "gate freshness mismatch: "
                    f"gate={gate_processed_through:%Y-%m-%d} "
                    f"source={actual_source_latest:%Y-%m-%d}"
                )
            gate = pd.read_parquet(args.gate_cache)
            gate["date"] = pd.to_datetime(gate["date"], errors="coerce")
            gate = gate[
                gate["date"].between(start_ts, actual_source_latest)
            ]
            candidate_symbols = sorted(
                gate["symbol"].dropna().astype(str).unique().tolist()
            )
            gate_source_symbol_count = int(
                gate_manifest.get("source_symbol_count") or 0
            )
            gate_mode = "signal_gate"
        except Exception as exc:
            gate_reason = str(exc)

    if candidate_symbols == []:
        incremental = pd.DataFrame()
        incremental.attrs["source_symbol_count"] = 0
        incremental.attrs["source_latest_trade_date"] = (
            actual_source_latest.strftime("%Y-%m-%d")
            if pd.notna(actual_source_latest)
            else None
        )
        incremental.attrs["symbol_error_count"] = 0
        incremental.attrs["symbol_error_rate"] = 0.0
        incremental.attrs["symbol_error_samples"] = []
    else:
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
            allow_empty=True,
            symbols=candidate_symbols,
        )
    coverage = {
        "source_symbol_count": int(
            gate_source_symbol_count
            if gate_source_symbol_count is not None
            else incremental.attrs.get("source_symbol_count", 0)
        ),
        "source_latest_trade_date": (
            actual_source_latest.strftime("%Y-%m-%d")
            if pd.notna(actual_source_latest)
            else incremental.attrs.get("source_latest_trade_date")
        ),
        "symbol_error_count": int(incremental.attrs.get("symbol_error_count", 0)),
        "symbol_error_rate": float(incremental.attrs.get("symbol_error_rate", 0.0)),
        "symbol_error_samples": list(incremental.attrs.get("symbol_error_samples", [])),
    }
    if not incremental.empty:
        incremental = merge_daily_basic_features(
            incremental,
            args.daily_basic_dir,
            min_match_rate=float(os.getenv("ROUTINE_DAILY_BASIC_MIN_MATCH_RATE", "0.98")),
        )

    if args.dataset_out.exists():
        existing = pd.read_parquet(args.dataset_out)
        existing["date"] = pd.to_datetime(existing["date"])
        if "factor_schema_version" not in existing.columns:
            if factor_schema_version != LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION:
                raise RuntimeError(
                    "Existing B1 feature cache has no factor schema metadata. "
                    "Rebuild it before using the current causal schema."
                )
            existing["factor_schema_version"] = factor_schema_version
        existing_schemas = set(
            existing["factor_schema_version"].dropna().astype(str).unique()
        )
        if existing_schemas - {factor_schema_version}:
            raise RuntimeError(
                "Existing B1 feature cache schema mismatch: "
                f"expected={factor_schema_version} actual={sorted(existing_schemas)}"
            )
        kept = existing[existing["date"] < start_ts].copy()
        combined = pd.concat([kept, incremental], ignore_index=True, sort=False)
    else:
        combined = incremental

    if combined.empty:
        raise RuntimeError(
            "B1 feature cache has no historical rows and the incremental window "
            "produced no signals"
        )

    combined["date"] = pd.to_datetime(combined["date"])
    if "factor_schema_version" not in combined.columns:
        combined["factor_schema_version"] = factor_schema_version
    combined["factor_schema_version"] = combined["factor_schema_version"].fillna(
        factor_schema_version
    )
    combined_schemas = set(
        combined["factor_schema_version"].dropna().astype(str).unique()
    )
    if combined_schemas != {factor_schema_version}:
        raise RuntimeError(
            "B1 feature cache would mix factor schemas: "
            f"expected={factor_schema_version} actual={sorted(combined_schemas)}"
        )
    combined = (
        combined.sort_values(["date", "symbol"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    combined = assign_symbol_splits(combined, args.oot_start, args.test_size, args.random_state)

    atomic_write_parquet(combined, args.dataset_out, index=False)

    result = {
        "status": "success",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "incremental_start_date": start_ts.strftime("%Y-%m-%d"),
        "incremental_rows": int(len(incremental)),
        "factor_schema_version": factor_schema_version,
        "gate_mode": gate_mode,
        "gate_fallback_reason": gate_reason,
        "gate_candidate_symbols": int(
            len(candidate_symbols)
            if candidate_symbols is not None
            else coverage["source_symbol_count"]
        ),
        **coverage,
        "daily_basic_match_rate": float(incremental["turnover_rate"].notna().mean())
        if "turnover_rate" in incremental.columns and len(incremental)
        else 0.0,
        "total_rows": int(len(combined)),
        "date_min": combined["date"].min().strftime("%Y-%m-%d") if not combined.empty else None,
        "date_max": combined["date"].max().strftime("%Y-%m-%d") if not combined.empty else None,
        "output": str(args.dataset_out),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
