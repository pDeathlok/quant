#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild both selector rule caches with one market scan and one factor pass."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

import analyze_b1_family_rule_backtest as family_rules
import analyze_z_skill_entry_exit_backtest as z_skills
from analyze_b1_xgb_entry_exit_grid import DEFAULT_DAILY_DIR
from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.data.atomic_io import atomic_write_parquet
from quant.features.daily_factor_layer import (
    DEFAULT_FACTOR_ROOT,
    attach_daily_base_factors,
    attach_daily_signal_factors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--force-refresh",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--incremental-start-date",
        default=None,
        help="只重建并替换该日期之后的缓存；不传且缓存有效时直接复用。",
    )
    parser.add_argument(
        "--factor-mode",
        choices=("stateful", "legacy"),
        default=os.getenv("ROUTINE_SIGNAL_FACTOR_MODE", "stateful"),
        help="stateful 持久化滚动状态；legacy 保留原全窗口重算用于对照。",
    )
    parser.add_argument(
        "--factor-root",
        type=Path,
        default=DEFAULT_FACTOR_ROOT,
    )
    return parser.parse_args()


def _parse_date(value: str) -> pd.Timestamp:
    if value.isdigit() and len(value) == 8:
        return pd.to_datetime(value, format="%Y%m%d")
    return pd.to_datetime(value)


def _load_cache(
    path: Path,
    expected_columns: set[str],
    *,
    force_refresh: bool,
) -> pd.DataFrame | None:
    if force_refresh or not path.exists():
        return None
    try:
        cached = pd.read_parquet(path)
    except Exception:
        return None
    if not {"symbol", "date", *expected_columns} <= set(cached.columns):
        return None
    cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
    return cached.dropna(subset=["symbol", "date"])


def _summary(frame: pd.DataFrame, signal_columns: list[str]) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "latest_date": None,
            "latest_rows": 0,
            "latest_hits": {},
        }
    dates = pd.to_datetime(frame["date"], errors="coerce")
    latest_date = dates.max()
    latest = frame[dates == latest_date]
    return {
        "rows": int(len(frame)),
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "latest_rows": int(len(latest)),
        "latest_hits": {
            column: int(latest[column].fillna(False).astype(bool).sum())
            for column in signal_columns
            if column in latest.columns
        },
    }


def _process_symbol(
    symbol: str,
    frame: pd.DataFrame,
    rebuild_start: str,
    factor_mode: str = "stateful",
    factor_root: Path = DEFAULT_FACTOR_ROOT,
) -> dict[str, Any]:
    """Compute shared factors once, then fan into both signal families."""

    try:
        normalized = family_rules.normalize_daily_frame(frame, symbol)
        if "name" in normalized.columns:
            names = normalized["name"].fillna("").astype(str)
            normalized = normalized[
                ~names.str.upper().str.contains("ST")
                & ~names.str.contains("退")
            ].copy()
        if len(normalized) < 160:
            return {
                "symbol": symbol,
                "family": None,
                "extended": None,
                "errors": [],
                "factor_cache_mode": "insufficient_history",
            }
        if factor_mode == "stateful":
            factored = attach_daily_signal_factors(
                normalized,
                symbol=symbol,
                factor_root=factor_root,
                persist_missing=True,
            )
            factor_cache_mode = factored.attrs.get(
                "signal_factor_cache_mode",
                "unknown",
            )
        else:
            factored = attach_daily_base_factors(
                normalized,
                symbol=symbol,
                compute_if_missing=True,
                persist_missing=False,
            )
            factor_cache_mode = "legacy_full"
    except Exception as exc:
        return {
            "symbol": symbol,
            "family": None,
            "extended": None,
            "errors": [f"shared_factors: {exc}"],
            "factor_cache_mode": "error",
        }

    errors: list[str] = []
    try:
        family = family_rules.process_frame(
            symbol,
            factored,
            factors_attached=True,
            raise_errors=True,
        )
    except Exception as exc:
        family = None
        errors.append(f"family: {exc}")
    try:
        extended = z_skills.process_frame(
            symbol,
            factored,
            rebuild_start,
            factors_attached=True,
            raise_errors=True,
        )
    except Exception as exc:
        extended = None
        errors.append(f"extended: {exc}")
    return {
        "symbol": symbol,
        "family": family,
        "extended": extended,
        "errors": errors,
        "factor_cache_mode": factor_cache_mode,
    }


def _merge_incremental_cache(
    cached: pd.DataFrame | None,
    frames: list[pd.DataFrame],
    rebuild_from: pd.Timestamp,
) -> pd.DataFrame:
    recent = (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else pd.DataFrame()
    )
    if not recent.empty:
        recent["date"] = pd.to_datetime(recent["date"], errors="coerce")
        recent = recent[recent["date"] >= rebuild_from].copy()
    old = (
        cached[pd.to_datetime(cached["date"], errors="coerce") < rebuild_from].copy()
        if cached is not None
        else pd.DataFrame()
    )
    combined = pd.concat([old, recent], ignore_index=True, sort=False)
    if combined.empty:
        raise RuntimeError("signal cache rebuild produced no rows")
    return (
        combined.dropna(subset=["symbol", "date"])
        .sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )


def main() -> None:
    started = perf_counter()
    args = parse_args()
    family_columns = {spec.name for spec in family_rules.build_signal_specs()}
    extended_columns = {spec.key for spec in z_skills.build_signal_specs()}
    family_cached = _load_cache(
        family_rules.SIGNAL_CACHE,
        family_columns,
        force_refresh=args.force_refresh,
    )
    extended_cached = _load_cache(
        z_skills.SIGNAL_CACHE,
        extended_columns,
        force_refresh=args.force_refresh,
    )
    if (
        args.incremental_start_date is None
        and not args.force_refresh
        and family_cached is not None
        and extended_cached is not None
    ):
        result = {
            "status": "success",
            "execution_mode": "cache_reuse",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "family": _summary(family_cached, sorted(family_columns)),
            "extended": _summary(extended_cached, sorted(extended_columns)),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    rebuild_from = _parse_date(
        args.incremental_start_date or args.start_date
    ).normalize()
    history_start = rebuild_from - pd.Timedelta(days=600)
    store = MarketDataStore(
        MarketDataStoreConfig(backend="parquet", root=args.daily_dir.parent)
    )
    market = store.read_market_range(
        args.daily_dir.name,
        start_date=history_start.strftime("%Y%m%d"),
    )
    if market.empty:
        raise RuntimeError(
            f"No canonical daily rows found for {history_start:%Y-%m-%d}+"
        )
    market_dates = pd.to_datetime(
        market.get("date", market.get("trade_date")),
        errors="coerce",
    )
    processed_through = market_dates.max()
    if pd.isna(processed_through):
        raise RuntimeError("Canonical daily rows have no valid trade date")
    tasks = [
        (str(symbol), group.reset_index(drop=True))
        for symbol, group in market.groupby("ts_code", sort=False)
    ]
    family_frames: list[pd.DataFrame] = []
    extended_frames: list[pd.DataFrame] = []
    symbol_errors: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                _process_symbol,
                symbol,
                frame,
                rebuild_from.strftime("%Y-%m-%d"),
                args.factor_mode,
                args.factor_root,
            )
            for symbol, frame in tasks
        ]
        factor_cache_modes: Counter[str] = Counter()
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            factor_cache_modes[result["factor_cache_mode"]] += 1
            if result["family"] is not None and not result["family"].empty:
                family_frames.append(result["family"])
            if result["extended"] is not None and not result["extended"].empty:
                extended_frames.append(result["extended"])
            if result["errors"]:
                symbol_errors.append(
                    {
                        "symbol": result["symbol"],
                        "errors": result["errors"],
                    }
                )
            if index % 500 == 0 or index == len(futures):
                print(
                    f"  combined signals: {index}/{len(futures)} symbols",
                    flush=True,
                )

    maximum_error_rate = float(
        os.getenv("ROUTINE_SIGNAL_MAX_SYMBOL_ERROR_RATE", "0.001")
    )
    error_rate = len(symbol_errors) / len(tasks) if tasks else 1.0
    if error_rate > maximum_error_rate:
        examples = symbol_errors[:10]
        raise RuntimeError(
            "signal cache symbol error rate exceeded gate: "
            f"{len(symbol_errors)}/{len(tasks)} ({error_rate:.2%}) "
            f"> {maximum_error_rate:.2%}; examples={examples}"
        )

    family = _merge_incremental_cache(
        family_cached,
        family_frames,
        rebuild_from,
    )
    extended = _merge_incremental_cache(
        extended_cached,
        extended_frames,
        rebuild_from,
    )
    atomic_write_parquet(family, family_rules.SIGNAL_CACHE, index=False)
    atomic_write_parquet(extended, z_skills.SIGNAL_CACHE, index=False)

    family_summary = _summary(family, sorted(family_columns))
    extended_summary = _summary(extended, sorted(extended_columns))
    result = {
        "status": "success",
        "execution_mode": "fused_single_market_scan",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "incremental_start_date": rebuild_from.strftime("%Y-%m-%d"),
        "processed_through_date": processed_through.strftime("%Y-%m-%d"),
        "symbols": len(tasks),
        "symbol_errors": len(symbol_errors),
        "symbol_error_rate": round(error_rate, 6),
        "symbol_error_examples": symbol_errors[:20],
        "factor_mode": args.factor_mode,
        "factor_cache_modes": dict(sorted(factor_cache_modes.items())),
        "factor_root": str(args.factor_root),
        "elapsed_seconds": perf_counter() - started,
        "family": family_summary,
        "extended": extended_summary,
        "family_cache": str(family_rules.SIGNAL_CACHE),
        "extended_cache": str(z_skills.SIGNAL_CACHE),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
