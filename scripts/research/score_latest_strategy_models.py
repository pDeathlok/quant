#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score the latest selector candidates with trained strategy models.

Daily selector refresh should not wait for a full model retrain. This script
reuses the latest trained XGBoost artifacts, builds today's Tushare-only factor
row for each rule-hit stock, and writes the model-scored candidate parquet that
the web selector reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

from analyze_b1_xgb_entry_exit_grid import DEFAULT_DAILY_DIR, DEFAULT_OUTPUT_DIR
from quant.data import list_partitioned_symbol_paths, read_partitioned_symbol_file
from quant.data.atomic_io import atomic_write_json
from quant.data.source_merge import normalize_tushare_daily
from quant.features.daily_factor_layer import attach_daily_base_factors
from quant.features.project_factor_layer import (
    LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION,
    calculate_project_market_factors,
    resolve_project_factor_schema,
)
from quant.features.variable_library import (
    PROJECT_FACTOR_COLUMNS,
    build_continuous_ohlc,
    build_latest_scale_ohlc,
)
from quant.ml.feature_coverage import (
    MODEL_FEATURE_HISTORY_YEARS,
    model_feature_history_start,
    validate_required_feature_coverage,
)
from train_z_skill_models_and_backtest import (
    AucGapEarlyStopping,
    LABELS,
    MODEL_DIR,
    PRIORITY_SIGNALS,
    add_predictions,
    load_models,
    write_latest_scored_candidates,
    _load_signal_cache,
)

# Older model artifacts were trained by executing the research script directly,
# so pickle recorded this callback class under __main__.
setattr(sys.modules["__main__"], "AucGapEarlyStopping", AucGapEarlyStopping)

DEFAULT_ACTIVE_FEATURE_CACHE = (
    PROJECT_ROOT / "data/features/b1/active_candidate_project_features.parquet"
)
DEFAULT_SCORE_MANIFEST_NAME = "latest_z_skill_model_scored_candidates_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score latest strategy candidates with trained models")
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=DEFAULT_ACTIVE_FEATURE_CACHE,
        help="Current-date B1/Z candidate project-factor cache.",
    )
    parser.add_argument("--target-date", default=None, help="YYYY-MM-DD. Default: latest date in signal caches.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument(
        "--executor",
        choices=("processes", "threads"),
        default=os.getenv("ROUTINE_MODEL_SCORE_EXECUTOR", "processes"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("ROUTINE_MODEL_SCORE_BATCH_SIZE", "8")),
    )
    parser.add_argument("--signals", nargs="*", default=PRIORITY_SIGNALS)
    return parser.parse_args()


def _process_symbol(
    args: tuple[str, pd.DataFrame, list[str], pd.Timestamp],
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    path_str, signal_rows, signals, target_date = args
    path = Path(path_str)
    try:
        history_start = model_feature_history_start(target_date)
        daily = read_partitioned_symbol_file(path, start_date=history_start, end_date=target_date)
        daily = normalize_tushare_daily(daily, path.stem)
        daily = daily.sort_values("date").reset_index(drop=True)
        daily = daily[(daily["date"] >= history_start) & (daily["date"] <= target_date)].reset_index(drop=True)
        if len(daily) < 130 or daily["date"].max() < target_date:
            return None, (
                f"{path.stem}: insufficient/current daily history "
                f"rows={len(daily)} latest={daily['date'].max() if not daily.empty else None}"
            ), None
        name = str(daily["name"].dropna().iloc[0]) if "name" in daily.columns and daily["name"].notna().any() else ""
        if "ST" in name.upper() or "退" in name:
            return None, None, f"{path.stem}: target signal belongs to ST/delisting stock"

        shared = attach_daily_base_factors(
            daily,
            symbol=path.stem,
            compute_if_missing=True,
            persist_missing=False,
        )
        factor_frame = calculate_project_market_factors(
            daily,
            symbol=path.stem,
            shared_factors=shared,
        )
        factors = factor_frame[
            [
                *[
                    column
                    for column in PROJECT_FACTOR_COLUMNS
                    if column in factor_frame.columns
                ],
                "factor_schema_version",
            ]
        ]
        factor_schema_version = resolve_project_factor_schema()
        price = (
            build_latest_scale_ohlc(daily)
            if factor_schema_version == LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
            else build_continuous_ohlc(daily)
        )
        close_pos = ((price["close"] - price["low"]) / (price["high"] - price["low"]).replace(0, np.nan)).rename("close_pos")
        result = pd.concat([daily, factors, close_pos], axis=1)
        result = result.loc[:, ~result.columns.duplicated(keep="last")]
        result["symbol"] = result["symbol"].fillna(result.get("ts_code", path.stem)).astype(str)

        row = result[result["date"] == target_date].copy()
        if row.empty:
            return None, f"{path.stem}: missing factor row for {target_date.date()}", None
        signal_rows = signal_rows.copy()
        signal_rows["date"] = pd.to_datetime(signal_rows["date"])
        merged = row.merge(signal_rows[["symbol", "date", *signals]], on=["symbol", "date"], how="inner")
        if merged.empty:
            return None, f"{path.stem}: signal/factor merge produced no row", None
        return merged, None, None
    except Exception as exc:
        return None, f"{path.stem}: {exc}", None


def _process_symbol_batch(
    batch: list[tuple[str, pd.DataFrame, list[str], pd.Timestamp]],
) -> list[tuple[pd.DataFrame | None, str | None, str | None]]:
    return [_process_symbol(args) for args in batch]


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    size = max(1, batch_size)
    return [
        items[offset : offset + size]
        for offset in range(0, len(items), size)
    ]


def _load_cached_candidate_rows(
    feature_cache_path: Path | None,
    target_rows: pd.DataFrame,
    signals: list[str],
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load exact-date candidate factors from the shared B1/Z sidecar.

    A stale or absent cache is an ordinary cache miss.  A cache claiming to
    cover the requested date is trusted only after its factor-column and
    schema contracts pass; malformed current data fails closed instead of
    silently mixing feature definitions in one scoring batch.
    """

    if feature_cache_path is None or not feature_cache_path.is_file():
        return pd.DataFrame()
    # The sidecar intentionally contains one active trade date, so a direct
    # read is cheap and avoids backend-specific timestamp filter coercion.
    cached = pd.read_parquet(feature_cache_path)
    required_keys = {"symbol", "date"}
    missing_keys = sorted(required_keys - set(cached.columns))
    if missing_keys:
        raise RuntimeError(
            "Active candidate feature cache is missing contract columns: "
            f"{missing_keys}; cache={feature_cache_path}"
        )
    cached = cached.copy()
    cached["date"] = pd.to_datetime(cached["date"], errors="raise")
    requested_symbols = set(target_rows["symbol"].dropna().astype(str))
    cached = cached[
        (cached["date"] == target_date)
        & cached["symbol"].astype(str).isin(requested_symbols)
    ].copy()
    if cached.empty:
        return cached
    if "factor_schema_version" not in cached.columns:
        raise RuntimeError(
            "Active candidate feature cache is missing contract columns: "
            f"['factor_schema_version']; cache={feature_cache_path}"
        )
    if cached.duplicated(["symbol", "date"]).any():
        duplicates = (
            cached.loc[cached.duplicated(["symbol", "date"], keep=False), "symbol"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            "Active candidate feature cache contains duplicate current rows: "
            f"{duplicates[:20]}"
        )
    missing_factors = sorted(set(PROJECT_FACTOR_COLUMNS) - set(cached.columns))
    if missing_factors:
        raise RuntimeError(
            "Active candidate feature cache does not satisfy the 147-factor contract: "
            f"missing={missing_factors[:20]} count={len(missing_factors)}"
        )
    expected_schema = resolve_project_factor_schema()
    schemas = set(cached["factor_schema_version"].dropna().astype(str).unique())
    if schemas != {expected_schema}:
        raise RuntimeError(
            "Active candidate feature cache schema mismatch: "
            f"expected={expected_schema} actual={sorted(schemas) or ['missing']}"
        )

    feature_columns = [column for column in cached.columns if column not in signals]
    signal_rows = target_rows[["symbol", "date", *signals]].copy()
    signal_rows["symbol"] = signal_rows["symbol"].astype(str)
    cached["symbol"] = cached["symbol"].astype(str)
    return cached[feature_columns].merge(
        signal_rows,
        on=["symbol", "date"],
        how="inner",
        validate="one_to_one",
    )


def build_latest_dataset(
    daily_dir: Path,
    signals: list[str],
    start_date: str,
    target_date: str | None,
    workers: int,
    *,
    executor_type: str = "threads",
    batch_size: int = 8,
    feature_cache_path: Path | None = None,
) -> pd.DataFrame:
    signal_df = _load_signal_cache(signals, start_date)
    signal_df["date"] = pd.to_datetime(signal_df["date"])
    target_ts = pd.Timestamp(target_date) if target_date else signal_df["date"].max()
    target_rows = signal_df[signal_df["date"] == target_ts].copy()
    if target_rows.empty:
        result = pd.DataFrame(columns=["symbol", "date", "name", *signals])
        result.attrs.update(
            excluded_candidates=[],
            executor_type=executor_type,
            batch_size=max(1, batch_size),
            feature_cache_path=(
                str(feature_cache_path) if feature_cache_path is not None else None
            ),
            feature_cache_rows=0,
            feature_cache_symbols=0,
            calculated_symbols=0,
            feature_elapsed_seconds=0.0,
            target_date=target_ts.strftime("%Y-%m-%d"),
            empty_candidate_set=True,
        )
        return result

    by_symbol = {symbol: group[["symbol", "date", *signals]].copy() for symbol, group in target_rows.groupby("symbol")}
    cached = _load_cached_candidate_rows(
        feature_cache_path,
        target_rows,
        signals,
        target_ts,
    )
    cached_symbols = set(cached["symbol"].astype(str)) if not cached.empty else set()
    missing_by_symbol = {
        symbol: rows
        for symbol, rows in by_symbol.items()
        if symbol not in cached_symbols
    }
    suffixes = (".SZ.parquet", ".SH.parquet", ".BJ.parquet")
    files = (
        [
            path
            for path in list_partitioned_symbol_paths(daily_dir)
            if path.name.endswith(suffixes) and path.stem in missing_by_symbol
        ]
        if missing_by_symbol
        else []
    )
    missing_files = sorted(set(missing_by_symbol) - {path.stem for path in files})
    if missing_files:
        raise RuntimeError(
            "Missing canonical daily files for model-scoring candidates: "
            f"{missing_files[:20]} (count={len(missing_files)})"
        )
    frames: list[pd.DataFrame] = [cached] if not cached.empty else []
    errors: list[str] = []
    excluded: list[str] = []
    started = perf_counter()
    tasks = [
        (str(path), missing_by_symbol[path.stem], signals, target_ts)
        for path in files
    ]
    executor_cls = (
        ProcessPoolExecutor
        if executor_type == "processes"
        else ThreadPoolExecutor
    )
    batches = _batched(tasks, batch_size)
    completed = 0
    with executor_cls(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(_process_symbol_batch, batch)
            for batch in batches
        ]
        for future in as_completed(futures):
            batch_results = future.result()
            completed += len(batch_results)
            for frame, error, exclusion in batch_results:
                if frame is not None and len(frame):
                    frames.append(frame)
                if error:
                    errors.append(error)
                if exclusion:
                    excluded.append(exclusion)
            if completed % 100 < len(batch_results) or completed == len(tasks):
                print(
                    f"  latest model scoring features: {completed}/{len(tasks)} files "
                    f"frames={len(frames)} excluded={len(excluded)}",
                    flush=True,
                )
    if errors:
        raise RuntimeError(
            "Latest model scoring did not cover every candidate symbol: "
            f"{sorted(errors)[:20]} (count={len(errors)}/{len(files)})"
        )
    if not frames:
        raise RuntimeError(f"No model-scoring feature rows produced for {target_ts.date()}")
    data = (
        pd.concat(frames, ignore_index=True)
        .replace([np.inf, -np.inf], np.nan)
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    data.attrs["excluded_candidates"] = sorted(excluded)
    data.attrs["executor_type"] = executor_type
    data.attrs["batch_size"] = max(1, batch_size)
    data.attrs["feature_cache_path"] = (
        str(feature_cache_path) if feature_cache_path is not None else None
    )
    data.attrs["feature_cache_rows"] = int(len(cached))
    data.attrs["feature_cache_symbols"] = int(len(cached_symbols))
    data.attrs["calculated_symbols"] = int(len(tasks))
    data.attrs["feature_elapsed_seconds"] = perf_counter() - started
    print(
        f"latest scoring dataset rows={len(data):,} date={target_ts.date()} "
        f"cache_symbols={len(cached_symbols)} calculated_symbols={len(tasks)} "
        f"executor={executor_type} batch_size={max(1, batch_size)} "
        f"elapsed={perf_counter() - started:.1f}s",
        flush=True,
    )
    return data


def ensure_model_features(data: pd.DataFrame, models: dict) -> pd.DataFrame:
    out = data.copy()
    required_features = list(
        dict.fromkeys(
            str(feature)
            for model in models.values()
            for feature in getattr(model, "feature_names_in_", [])
        )
    )
    if not required_features:
        raise RuntimeError("Loaded Z-skill models declare no required features")
    target_date = (
        pd.to_datetime(out["date"], errors="coerce").max()
        if "date" in out.columns
        else None
    )
    feature_coverage = validate_required_feature_coverage(
        out,
        required_features,
        target_date=target_date,
        context="Z-skill latest scoring",
    )
    for label_name in LABELS:
        col = f"pred_{label_name}"
        if col not in out.columns:
            out[col] = np.nan
    out.attrs["feature_coverage"] = feature_coverage
    return out


def _empty_feature_coverage(models: dict, target_date: str) -> dict[str, Any]:
    required_features = list(
        dict.fromkeys(
            str(feature)
            for model in models.values()
            for feature in getattr(model, "feature_names_in_", [])
        )
    )
    if not required_features:
        raise RuntimeError("Loaded Z-skill models declare no required features")
    return {
        "status": "valid",
        "target_date": target_date,
        "row_count": 0,
        "required_feature_count": len(required_features),
        "present_feature_count": 0,
        "covered_feature_count": 0,
        "missing_columns": [],
        "all_null_features": [],
        "partial_features": [],
        "minimum_coverage": 1.0,
        "mean_coverage": 1.0,
        "coverage": {},
        "non_null_counts": {},
        "empty_candidate_set": True,
    }


def main() -> None:
    started = perf_counter()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signals = list(dict.fromkeys(args.signals))
    print(f"scoring signals: {signals}", flush=True)
    models, _ = load_models(signals, args.model_dir, args.output_dir)
    data = build_latest_dataset(
        args.daily_dir,
        signals,
        args.start_date,
        args.target_date,
        args.workers,
        executor_type=args.executor,
        batch_size=args.batch_size,
        feature_cache_path=args.feature_cache,
    )
    excluded_candidates = list(data.attrs.get("excluded_candidates") or [])
    executor_type = str(data.attrs.get("executor_type") or args.executor)
    batch_size = int(data.attrs.get("batch_size") or args.batch_size)
    feature_cache_path = data.attrs.get("feature_cache_path")
    feature_cache_rows = int(data.attrs.get("feature_cache_rows") or 0)
    feature_cache_symbols = int(data.attrs.get("feature_cache_symbols") or 0)
    calculated_symbols = int(data.attrs.get("calculated_symbols") or 0)
    feature_elapsed_seconds = float(
        data.attrs.get("feature_elapsed_seconds") or 0.0
    )
    target_date_text = str(
        data.attrs.get("target_date")
        or pd.to_datetime(data["date"], errors="coerce").max().strftime("%Y-%m-%d")
    )
    if data.empty:
        feature_coverage = _empty_feature_coverage(models, target_date_text)
        predicted = data.copy()
    else:
        data = ensure_model_features(data, models)
        feature_coverage = dict(data.attrs["feature_coverage"])
        predicted = add_predictions(data, models, signals)
    playbook_path = args.output_dir / "latest_z_skill_model_operational_playbook.csv"
    if not playbook_path.exists():
        raise FileNotFoundError(f"Missing model playbook: {playbook_path}")
    playbooks = pd.read_csv(playbook_path)
    scored_path = write_latest_scored_candidates(predicted, signals, playbooks, args.output_dir)
    scored = pd.read_parquet(scored_path) if scored_path.exists() else pd.DataFrame()
    result = {
        "status": "success",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "target_date": target_date_text,
        "scored_signals": sorted(signals),
        "candidate_rows": int(len(data)),
        "excluded_candidate_count": len(excluded_candidates),
        "excluded_candidate_samples": excluded_candidates[:20],
        "executor_type": executor_type,
        "batch_size": batch_size,
        "workers": args.workers,
        "feature_cache": feature_cache_path,
        "feature_cache_rows": feature_cache_rows,
        "feature_cache_symbols": feature_cache_symbols,
        "calculated_symbols": calculated_symbols,
        "factor_schema_version": resolve_project_factor_schema(),
        "feature_history_years": MODEL_FEATURE_HISTORY_YEARS,
        "feature_coverage": feature_coverage,
        "feature_elapsed_seconds": round(feature_elapsed_seconds, 3),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "scored_rows": int(len(scored)),
        "model_pass_rows": int(scored["model_pass"].fillna(False).sum()) if "model_pass" in scored.columns else 0,
        "output": str(scored_path),
    }
    score_distribution = {
        column: {
            "p50": float(pd.to_numeric(predicted[column], errors="coerce").quantile(0.50)),
            "p90": float(pd.to_numeric(predicted[column], errors="coerce").quantile(0.90)),
            "max": float(pd.to_numeric(predicted[column], errors="coerce").max()),
        }
        for column in ("pred_up5", "pred_up8", "pred_down3")
        if column in predicted and predicted[column].notna().any()
    }
    result["score_distribution"] = score_distribution
    manifest_path = args.output_dir / DEFAULT_SCORE_MANIFEST_NAME
    atomic_write_json(result, manifest_path)
    result["manifest_path"] = str(manifest_path)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
