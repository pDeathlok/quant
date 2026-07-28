#!/usr/bin/env python
"""Compare legacy and stateful signal factors on canonical market data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

import analyze_b1_family_rule_backtest as family_rules
import analyze_z_skill_entry_exit_backtest as z_skills
from analyze_b1_xgb_entry_exit_grid import DEFAULT_DAILY_DIR
from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.features.daily_factor_layer import (
    SIGNAL_FACTOR_COLUMNS,
    attach_daily_base_factors,
    attach_daily_signal_factors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument(
        "--signal-date",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--factor-root",
        type=Path,
        default=PROJECT_ROOT / "data/validation/stateful_signal_factors",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def _signal_matrix(
    frame: pd.DataFrame | None,
    columns: list[str],
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns, dtype=bool)
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["symbol", "date"])
    out = out.set_index(["symbol", "date"])[columns]
    return out.fillna(False).astype(bool).sort_index()


def _signals_match(
    legacy: pd.DataFrame | None,
    stateful: pd.DataFrame | None,
    columns: list[str],
) -> bool:
    left = _signal_matrix(legacy, columns)
    right = _signal_matrix(stateful, columns)
    index = left.index.union(right.index)
    return left.reindex(index, fill_value=False).equals(
        right.reindex(index, fill_value=False)
    )


def _factor_difference(
    legacy: pd.DataFrame,
    stateful: pd.DataFrame,
) -> tuple[bool, float]:
    maximum = 0.0
    for column in SIGNAL_FACTOR_COLUMNS:
        left = legacy[column]
        right = stateful[column]
        if pd.api.types.is_bool_dtype(left) or pd.api.types.is_bool_dtype(right):
            if not left.fillna(False).astype(bool).equals(
                right.fillna(False).astype(bool)
            ):
                return False, np.inf
            continue
        left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
        right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
        if not np.allclose(
            left_values,
            right_values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        ):
            return False, float(
                np.nanmax(np.abs(left_values - right_values))
            )
        difference = np.abs(left_values - right_values)
        if np.isfinite(difference).any():
            maximum = max(maximum, float(np.nanmax(difference)))
    return True, maximum


def _normalize_symbol(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    normalized = family_rules.normalize_daily_frame(frame, symbol)
    if "name" in normalized.columns:
        names = normalized["name"].fillna("").astype(str)
        normalized = normalized[
            ~names.str.upper().str.contains("ST")
            & ~names.str.contains("退")
        ].copy()
    return normalized


def validate_symbol(
    symbol: str,
    frame: pd.DataFrame,
    signal_date: str,
    factor_root: Path,
) -> dict[str, Any]:
    normalized = _normalize_symbol(symbol, frame)
    if len(normalized) < 161:
        return {"symbol": symbol, "status": "skipped"}

    started = perf_counter()
    legacy = attach_daily_base_factors(
        normalized,
        symbol=symbol,
        compute_if_missing=True,
        persist_missing=False,
    )
    legacy_seconds = perf_counter() - started

    seed_started = perf_counter()
    attach_daily_signal_factors(
        normalized.iloc[:-1],
        symbol=symbol,
        factor_root=factor_root,
        persist_missing=True,
    )
    seed_seconds = perf_counter() - seed_started
    incremental_started = perf_counter()
    stateful = attach_daily_signal_factors(
        normalized,
        symbol=symbol,
        factor_root=factor_root,
        persist_missing=True,
    )
    incremental_seconds = perf_counter() - incremental_started

    factors_match, maximum_difference = _factor_difference(
        legacy,
        stateful,
    )
    family_columns = [spec.name for spec in family_rules.build_signal_specs()]
    z_columns = [spec.key for spec in z_skills.build_signal_specs()]
    legacy_family = family_rules.process_frame(
        symbol,
        legacy,
        factors_attached=True,
        raise_errors=True,
    )
    stateful_family = family_rules.process_frame(
        symbol,
        stateful,
        factors_attached=True,
        raise_errors=True,
    )
    legacy_z = z_skills.process_frame(
        symbol,
        legacy,
        signal_date,
        factors_attached=True,
        raise_errors=True,
    )
    stateful_z = z_skills.process_frame(
        symbol,
        stateful,
        signal_date,
        factors_attached=True,
        raise_errors=True,
    )
    family_match = _signals_match(
        legacy_family,
        stateful_family,
        family_columns,
    )
    z_match = _signals_match(legacy_z, stateful_z, z_columns)
    return {
        "symbol": symbol,
        "status": (
            "success"
            if factors_match and family_match and z_match
            else "mismatch"
        ),
        "factor_cache_mode": stateful.attrs.get(
            "signal_factor_cache_mode"
        ),
        "maximum_factor_difference": maximum_difference,
        "family_match": family_match,
        "z_match": z_match,
        "legacy_factor_seconds": legacy_seconds,
        "state_seed_seconds": seed_seconds,
        "state_increment_seconds": incremental_seconds,
    }


def main() -> None:
    args = parse_args()
    signal_date = pd.Timestamp(args.signal_date).normalize()
    history_start = signal_date - pd.Timedelta(days=600)
    store = MarketDataStore(
        MarketDataStoreConfig(
            backend="parquet",
            root=args.daily_dir.parent,
        )
    )
    market = store.read_market_range(
        args.daily_dir.name,
        start_date=history_start.strftime("%Y%m%d"),
        end_date=signal_date.strftime("%Y%m%d"),
    )
    if market.empty:
        raise RuntimeError("No canonical market rows available for validation")
    groups = [
        (str(symbol), group.reset_index(drop=True))
        for symbol, group in market.groupby("ts_code", sort=False)
    ]
    if args.limit and len(groups) > args.limit:
        rng = np.random.default_rng(args.seed)
        selected = sorted(
            rng.choice(len(groups), size=args.limit, replace=False).tolist()
        )
        groups = [groups[index] for index in selected]

    results: list[dict[str, Any]] = []
    started = perf_counter()
    signal_date_text = signal_date.strftime("%Y-%m-%d")
    if args.workers <= 1:
        for index, (symbol, frame) in enumerate(groups, start=1):
            results.append(
                validate_symbol(
                    symbol,
                    frame,
                    signal_date_text,
                    args.factor_root,
                )
            )
            if index % 25 == 0 or index == len(groups):
                print(
                    f"validated signals: {index}/{len(groups)}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    validate_symbol,
                    symbol,
                    frame,
                    signal_date_text,
                    args.factor_root,
                )
                for symbol, frame in groups
            ]
            for index, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                results.append(future.result())
                if index % 100 == 0 or index == len(futures):
                    print(
                        f"validated signals: {index}/{len(futures)}",
                        flush=True,
                    )

    checked = [item for item in results if item["status"] != "skipped"]
    mismatches = [item for item in checked if item["status"] != "success"]
    legacy_seconds = sum(
        float(item["legacy_factor_seconds"])
        for item in checked
    )
    incremental_seconds = sum(
        float(item["state_increment_seconds"])
        for item in checked
    )
    report = {
        "status": "success" if not mismatches and checked else "failed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "symbols_requested": len(groups),
        "symbols_checked": len(checked),
        "symbols_skipped": len(results) - len(checked),
        "workers": args.workers,
        "mismatches": mismatches[:20],
        "maximum_factor_difference": max(
            (
                float(item["maximum_factor_difference"])
                for item in checked
            ),
            default=None,
        ),
        "legacy_factor_seconds": legacy_seconds,
        "state_increment_seconds": incremental_seconds,
        "incremental_speedup": (
            legacy_seconds / incremental_seconds
            if incremental_seconds
            else None
        ),
        "elapsed_seconds": perf_counter() - started,
        "factor_root": str(args.factor_root),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(output, encoding="utf-8")
    print(output, flush=True)
    if report["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
