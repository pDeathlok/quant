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
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.data.source_merge import normalize_tushare_daily
from quant.features.daily_factor_layer import attach_daily_base_factors
from quant.features.canonical_factor_names import (
    assert_no_forbidden_factor_names,
    migrate_legacy_factor_columns,
)
from quant.features.project_factor_layer import (
    LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION,
    calculate_project_market_factors,
    resolve_project_factor_schema,
)
from quant.features.variable_library import (
    PROJECT_FACTOR_COLUMNS,
    merge_daily_basic_features,
)
from quant.ml.feature_coverage import (
    MODEL_FEATURE_HISTORY_YEARS,
    model_feature_history_start,
    validate_required_feature_coverage,
)
from quant.routine.paths import CONFIG_PATH
from quant.routine.strategies import load_strategy_release
from train_b1_tushare_models import assign_symbol_splits, build_dataset


DEFAULT_ADDITIONAL_GATE_CACHE = (
    PROJECT_ROOT / "data/features/z_skill_daily_candidates.parquet"
)
DEFAULT_FAMILY_GATE_CACHE = (
    PROJECT_ROOT / "data/features/b1/b1_family_rule_candidates.parquet"
)
DEFAULT_ACTIVE_FEATURE_CACHE = (
    PROJECT_ROOT / "data/features/b1/active_candidate_project_features.parquet"
)
DEFAULT_ACTIVE_FEATURE_MANIFEST = (
    PROJECT_ROOT / "data/features/b1/active_candidate_project_features_manifest.json"
)


def _parse_date(value: str) -> pd.Timestamp:
    if value.isdigit() and len(value) == 8:
        return pd.to_datetime(value, format="%Y%m%d")
    return pd.to_datetime(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _released_b1_required_features(
    config_path: Path = CONFIG_PATH,
) -> tuple[list[str], str]:
    """Read the required columns from the artifacts pinned for production."""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    release_payload = payload.get("release") or {}
    if release_payload.get("ranking_source") == "left_side_unified":
        if release_payload.get("legacy_model_lifecycle") != "rollback_only":
            raise RuntimeError(
                "Unified left B1 release must mark legacy models rollback_only"
            )
        if not DEFAULT_LEFT_SIDE_RANKING_CONFIG.enabled:
            raise RuntimeError("Unified left B1 release is not enabled")
        required = list(PROJECT_FACTOR_COLUMNS)
        assert_no_forbidden_factor_names(
            required,
            context="active unified-left project factors",
        )
        return required, DEFAULT_LEFT_SIDE_RANKING_CONFIG.release_id

    release = load_strategy_release(config_path)
    model_dir = Path(release.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    required: list[str] = []
    for model_name in release.model_names:
        model_path = model_dir / f"{model_name}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing released B1 model: {model_path}")
        model = joblib.load(model_path)
        model_features = list(getattr(model, "feature_names_in_", []))
        if not model_features:
            raise RuntimeError(
                f"Released B1 model declares no required features: {model_path}"
            )
        assert_no_forbidden_factor_names(
            model_features,
            context=f"released B1 model {model_name}",
        )
        required.extend(str(feature) for feature in model_features)
    return list(dict.fromkeys(required)), release.id


def _validate_latest_model_features(
    data: pd.DataFrame,
    required_features: list[str],
) -> dict[str, object]:
    if data.empty:
        raise RuntimeError("Cannot validate an empty B1 feature cache")
    latest_feature_date = pd.to_datetime(data["date"], errors="raise").max()
    latest_feature_rows = data[
        pd.to_datetime(data["date"], errors="raise") == latest_feature_date
    ]
    return validate_required_feature_coverage(
        latest_feature_rows,
        required_features,
        target_date=latest_feature_date,
        context="B1 feature-cache refresh",
    )


def _read_candidate_gate(
    path: Path,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    gate = pd.read_parquet(path, columns=["symbol", "date"])
    missing = sorted({"symbol", "date"} - set(gate.columns))
    if missing:
        raise RuntimeError(f"Candidate gate {path} is missing columns: {missing}")
    gate = gate[["symbol", "date"]].copy()
    gate["symbol"] = gate["symbol"].astype(str)
    gate["date"] = pd.to_datetime(gate["date"], errors="raise")
    return (
        gate[gate["date"].between(start_date, end_date)]
        .drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


def _gate_symbols_on_date(
    gate: pd.DataFrame,
    target_date: pd.Timestamp,
) -> set[str]:
    if gate.empty:
        return set()
    dates = pd.to_datetime(gate["date"], errors="raise")
    return set(gate.loc[dates == target_date, "symbol"].astype(str))


def _additional_only_symbols(
    b1_gate: pd.DataFrame,
    *additional_gates: pd.DataFrame,
    target_date: pd.Timestamp,
) -> list[str]:
    additional_symbols: set[str] = set()
    for gate in additional_gates:
        additional_symbols.update(_gate_symbols_on_date(gate, target_date))
    return sorted(
        additional_symbols
        - _gate_symbols_on_date(b1_gate, target_date)
    )


def _process_active_candidate_frame(
    args: tuple[str, pd.DataFrame, pd.Timestamp],
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Calculate one exact-date canonical-factor candidate row.

    Daily-basic columns are joined once after all B1 and Z rows are combined,
    so this worker calculates only the price/volume portion of the contract.
    """

    symbol, daily, target_date = args
    try:
        history_start = model_feature_history_start(target_date)
        daily = normalize_tushare_daily(daily, symbol)
        if "vol" in daily.columns and "volume" in daily.columns:
            daily = daily.drop(columns=["vol"])
        daily = daily.sort_values("date").reset_index(drop=True)
        daily = daily[
            daily["date"].between(history_start, target_date)
        ].reset_index(drop=True)
        if len(daily) < 130 or daily["date"].max() < target_date:
            return None, (
                f"{symbol}: insufficient/current daily history rows={len(daily)} "
                f"latest={daily['date'].max() if not daily.empty else None}"
            ), None
        name = (
            str(daily["name"].dropna().iloc[0])
            if "name" in daily.columns and daily["name"].notna().any()
            else ""
        )
        if "ST" in name.upper() or "退" in name:
            return None, None, f"{symbol}: target signal belongs to ST/delisting stock"

        shared = attach_daily_base_factors(
            daily,
            symbol=symbol,
            compute_if_missing=True,
            persist_missing=False,
        )
        factor_frame = calculate_project_market_factors(
            daily,
            symbol=symbol,
            shared_factors=shared,
        )
        factor_columns = [
            column for column in PROJECT_FACTOR_COLUMNS if column in factor_frame.columns
        ]
        factors = factor_frame[[*factor_columns, "factor_schema_version"]]
        result = pd.concat([daily, factors], axis=1)
        result = result.loc[:, ~result.columns.duplicated(keep="last")]
        if "symbol" not in result.columns:
            result["symbol"] = symbol
        else:
            result["symbol"] = result["symbol"].fillna(symbol).astype(str)
        row = result[result["date"] == target_date].copy()
        if row.empty:
            return None, f"{symbol}: missing factor row for {target_date.date()}", None
        keep_columns = [
            "ts_code",
            "trade_date",
            "date",
            "symbol",
            "name",
            "industry",
            "market",
            "factor_schema_version",
            *PROJECT_FACTOR_COLUMNS,
        ]
        present = list(dict.fromkeys(column for column in keep_columns if column in row))
        return row[present], None, None
    except Exception as exc:
        return None, f"{symbol}: {exc}", None


def _build_additional_candidate_features(
    daily_dir: Path,
    target_date: pd.Timestamp,
    symbols: list[str],
    *,
    workers: int,
    executor_type: str,
) -> pd.DataFrame:
    if not symbols:
        result = pd.DataFrame()
        result.attrs.update(
            source_symbol_count=0,
            symbol_error_count=0,
            symbol_error_samples=[],
            excluded_candidates=[],
        )
        return result

    history_start = model_feature_history_start(target_date)
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
    market = store.read_market_range(
        daily_dir.name,
        start_date=history_start.strftime("%Y%m%d"),
        symbols=symbols,
    )
    if market.empty:
        result = pd.DataFrame()
        result.attrs.update(
            source_symbol_count=len(symbols),
            symbol_error_count=len(symbols),
            symbol_error_samples=[f"missing canonical history: {symbol}" for symbol in symbols[:20]],
            excluded_candidates=[],
        )
        return result

    symbol_column = "ts_code" if "ts_code" in market.columns else "symbol"
    available = set(market[symbol_column].dropna().astype(str))
    errors = [
        f"missing canonical history: {symbol}"
        for symbol in sorted(set(symbols) - available)
    ]
    requested_symbols = set(symbols)
    tasks = [
        (str(symbol), group.reset_index(drop=True), target_date)
        for symbol, group in market.groupby(symbol_column, sort=True)
        if str(symbol) in requested_symbols
    ]
    frames: list[pd.DataFrame] = []
    exclusions: list[str] = []
    executor_cls = ThreadPoolExecutor if executor_type == "threads" else ProcessPoolExecutor
    with executor_cls(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_process_active_candidate_frame, task) for task in tasks]
        for future in as_completed(futures):
            frame, error, exclusion = future.result()
            if frame is not None and len(frame):
                frames.append(frame)
            if error:
                errors.append(error)
            if exclusion:
                exclusions.append(exclusion)
    result = (
        pd.concat(frames, ignore_index=True)
        .replace([np.inf, -np.inf], np.nan)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )
    result.attrs.update(
        source_symbol_count=len(symbols),
        symbol_error_count=len(errors),
        symbol_error_samples=sorted(errors)[:20],
        excluded_candidates=sorted(exclusions),
    )
    return result


def _assemble_active_candidate_cache(
    b1_features: pd.DataFrame,
    supplemental_features: pd.DataFrame,
    b1_gate: pd.DataFrame,
    z_gate: pd.DataFrame,
    family_gate: pd.DataFrame,
    *,
    target_date: pd.Timestamp,
    factor_schema_version: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a one-date sidecar without widening the production B1 cache."""

    b1_symbols = _gate_symbols_on_date(b1_gate, target_date)
    z_symbols = _gate_symbols_on_date(z_gate, target_date)
    family_symbols = _gate_symbols_on_date(family_gate, target_date)
    union_symbols = b1_symbols | z_symbols | family_symbols
    source = pd.DataFrame({"symbol": sorted(union_symbols)})
    source["date"] = target_date
    if not source.empty:
        source["candidate_source_b1"] = source["symbol"].isin(b1_symbols)
        source["candidate_source_z"] = source["symbol"].isin(z_symbols)
        source["candidate_source_family"] = source["symbol"].isin(
            family_symbols
        )
        source["candidate_sources"] = source.apply(
            lambda row: ",".join(
                label
                for label, column in (
                    ("b1", "candidate_source_b1"),
                    ("family", "candidate_source_family"),
                    ("z_skill", "candidate_source_z"),
                )
                if bool(row[column])
            ),
            axis=1,
        )

    feature_frames = []
    for frame in (b1_features, supplemental_features):
        if frame.empty:
            continue
        current = frame[
            pd.to_datetime(frame["date"], errors="raise") == target_date
        ].copy()
        if not current.empty:
            feature_frames.append(current)
    features = (
        pd.concat(feature_frames, ignore_index=True, sort=False)
        .sort_values(["date", "symbol"])
        .drop_duplicates(["symbol", "date"], keep="first")
        if feature_frames
        else pd.DataFrame()
    )
    if features.empty:
        output_columns = [
            "ts_code",
            "trade_date",
            "date",
            "symbol",
            "name",
            "industry",
            "market",
            "factor_schema_version",
            *PROJECT_FACTOR_COLUMNS,
            "candidate_source_b1",
            "candidate_source_family",
            "candidate_source_z",
            "candidate_sources",
        ]
        active = pd.DataFrame(columns=list(dict.fromkeys(output_columns)))
    else:
        missing_factors = sorted(set(PROJECT_FACTOR_COLUMNS) - set(features.columns))
        if missing_factors:
            raise RuntimeError(
                "Active candidate feature cache does not satisfy the canonical factor contract: "
                f"missing={missing_factors[:20]} count={len(missing_factors)}"
            )
        schemas = set(features["factor_schema_version"].dropna().astype(str).unique())
        if schemas != {factor_schema_version}:
            raise RuntimeError(
                "Active candidate feature cache schema mismatch: "
                f"expected={factor_schema_version} actual={sorted(schemas) or ['missing']}"
            )
        active = features.merge(
            source,
            on=["symbol", "date"],
            how="inner",
            validate="one_to_one",
        )
        output_columns = [
            "ts_code",
            "trade_date",
            "date",
            "symbol",
            "name",
            "industry",
            "market",
            "factor_schema_version",
            *PROJECT_FACTOR_COLUMNS,
            "candidate_source_b1",
            "candidate_source_family",
            "candidate_source_z",
            "candidate_sources",
        ]
        active = active[
            list(dict.fromkeys(column for column in output_columns if column in active))
        ].sort_values("symbol").reset_index(drop=True)

    produced_symbols = set(active["symbol"].astype(str)) if not active.empty else set()
    stats: dict[str, object] = {
        "target_date": target_date.strftime("%Y-%m-%d"),
        "b1_candidate_count": len(b1_symbols),
        "family_candidate_count": len(family_symbols),
        "z_candidate_count": len(z_symbols),
        "union_candidate_count": len(union_symbols),
        "overlap_candidate_count": sum(
            (
                int(symbol in b1_symbols)
                + int(symbol in family_symbols)
                + int(symbol in z_symbols)
            )
            >= 2
            for symbol in union_symbols
        ),
        "computed_candidate_count": len(produced_symbols),
        "missing_candidate_count": len(union_symbols - produced_symbols),
        "missing_candidate_symbols": sorted(union_symbols - produced_symbols),
        "missing_candidate_samples": sorted(union_symbols - produced_symbols)[:20],
    }
    return active, stats


def _validate_active_candidate_coverage(
    stats: dict[str, object],
    excluded_candidates: list[str],
) -> dict[str, object]:
    """Distinguish policy exclusions from unexplained feature-build misses."""

    missing = {str(value) for value in stats.get("missing_candidate_symbols") or []}
    excluded_by_policy = {
        str(value).split(":", 1)[0].strip()
        for value in excluded_candidates
        if str(value).strip()
    }
    explained = missing & excluded_by_policy
    unexplained = missing - explained
    if unexplained:
        raise RuntimeError(
            "Active candidate feature cache has unexplained missing symbols: "
            f"count={len(unexplained)} samples={sorted(unexplained)[:20]}"
        )
    union_count = int(stats.get("union_candidate_count") or 0)
    return {
        "candidate_coverage_status": "complete",
        "eligible_candidate_count": union_count - len(explained),
        "policy_excluded_candidate_count": len(explained),
        "policy_excluded_candidate_symbols": sorted(explained),
        "unexplained_missing_candidate_count": 0,
        "unexplained_missing_candidate_symbols": [],
    }


def _read_required_additional_gate(
    path: Path,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load the Z candidate sidecar proven by the unified signal manifest."""

    if not path.is_file():
        raise RuntimeError(
            "Unified signal checkpoint is incomplete: "
            f"missing {path}"
        )
    return _read_candidate_gate(
        path,
        start_date=start_date,
        end_date=end_date,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally refresh B1 feature cache")
    parser.add_argument("--daily-dir", type=Path, default=PROJECT_ROOT / "data/raw/daily")
    parser.add_argument("--daily-basic-dir", type=Path, default=PROJECT_ROOT / "data/raw/daily_basic")
    parser.add_argument("--dataset-out", type=Path, default=PROJECT_ROOT / "data/features/b1/training_xgb_project_vars.parquet")
    parser.add_argument("--gate-cache", type=Path, default=None)
    parser.add_argument("--gate-manifest", type=Path, default=None)
    parser.add_argument(
        "--additional-gate-cache",
        type=Path,
        default=DEFAULT_ADDITIONAL_GATE_CACHE,
        help="Additional exact-date candidate rows to include in the shared sidecar.",
    )
    parser.add_argument(
        "--family-gate-cache",
        type=Path,
        default=DEFAULT_FAMILY_GATE_CACHE,
        help="Exact-date family candidate rows consumed by shared feature users.",
    )
    parser.add_argument(
        "--active-feature-out",
        type=Path,
        default=DEFAULT_ACTIVE_FEATURE_CACHE,
    )
    parser.add_argument(
        "--active-feature-manifest",
        type=Path,
        default=DEFAULT_ACTIVE_FEATURE_MANIFEST,
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help=(
            "Publish only the exact-date active-candidate inference cache. "
            "The historical training cache is intentionally left unchanged."
        ),
    )
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
    b1_gate_rows = pd.DataFrame(columns=["symbol", "date"])
    additional_gate_rows = pd.DataFrame(columns=["symbol", "date"])
    family_gate_rows = pd.DataFrame(columns=["symbol", "date"])
    additional_gate_mode = "disabled"
    additional_gate_reason: str | None = None
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
            gate = _read_candidate_gate(
                args.gate_cache,
                start_date=start_ts,
                end_date=actual_source_latest,
            )
            b1_gate_rows = gate
            candidate_symbols = sorted(
                gate["symbol"].dropna().astype(str).unique().tolist()
            )
            gate_source_symbol_count = int(
                gate_manifest.get("source_symbol_count") or 0
            )
            gate_mode = "signal_gate"
        except Exception as exc:
            gate_reason = str(exc)

    if args.live_only and gate_mode != "signal_gate":
        raise RuntimeError(
            "Live feature refresh requires the exact-date unified signal gate: "
            f"{gate_reason or 'gate cache/manifest unavailable'}"
        )

    # The B1 gate manifest is the freshness checkpoint of the unified signal
    # refresh that also writes z_skill_daily_candidates.  Only consume the
    # additional gate when that checkpoint matches canonical market data.
    if gate_mode == "signal_gate" and args.additional_gate_cache is not None:
        try:
            additional_gate_rows = _read_required_additional_gate(
                args.additional_gate_cache,
                start_date=start_ts,
                end_date=actual_source_latest,
            )
            additional_gate_mode = "signal_gate"
        except RuntimeError:
            additional_gate_mode = "missing"
            additional_gate_reason = f"missing {args.additional_gate_cache}"
            raise
    if gate_mode == "signal_gate" and args.family_gate_cache is not None:
        family_gate_rows = _read_required_additional_gate(
            args.family_gate_cache,
            start_date=start_ts,
            end_date=actual_source_latest,
        )

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

    additional_symbols: list[str] = []
    if pd.notna(actual_source_latest) and (
        not additional_gate_rows.empty or not family_gate_rows.empty
    ):
        # B1 candidates have already paid for the canonical factor build in
        # build_dataset. Calculate the family/Z-only union exactly once here.
        additional_symbols = _additional_only_symbols(
            b1_gate_rows,
            additional_gate_rows,
            family_gate_rows,
            target_date=actual_source_latest,
        )
    if pd.notna(actual_source_latest):
        additional = _build_additional_candidate_features(
            args.daily_dir,
            actual_source_latest,
            additional_symbols,
            workers=args.workers,
            executor_type=args.executor,
        )
    else:
        additional = pd.DataFrame()
    additional_coverage = {
        "source_symbol_count": int(additional.attrs.get("source_symbol_count", 0)),
        "symbol_error_count": int(additional.attrs.get("symbol_error_count", 0)),
        "symbol_error_samples": list(additional.attrs.get("symbol_error_samples", [])),
        "excluded_candidates": list(additional.attrs.get("excluded_candidates", [])),
    }

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
    enrichment_batches: list[pd.DataFrame] = []
    if not incremental.empty:
        enrichment_batches.append(incremental.assign(_active_origin="b1"))
    if not additional.empty:
        enrichment_batches.append(additional.assign(_active_origin="z"))
    if enrichment_batches:
        enriched = merge_daily_basic_features(
            pd.concat(enrichment_batches, ignore_index=True, sort=False),
            args.daily_basic_dir,
            min_match_rate=float(os.getenv("ROUTINE_DAILY_BASIC_MIN_MATCH_RATE", "0.98")),
        )
        incremental = enriched[enriched["_active_origin"] == "b1"].drop(
            columns="_active_origin"
        ).reset_index(drop=True)
        additional = enriched[enriched["_active_origin"] == "z"].drop(
            columns="_active_origin"
        ).reset_index(drop=True)

    if args.dataset_out.exists() and not args.live_only:
        existing = migrate_legacy_factor_columns(
            pd.read_parquet(args.dataset_out),
            context=f"B1 feature-cache boundary {args.dataset_out}",
            copy=False,
        )
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

    if combined.empty and not args.live_only:
        raise RuntimeError(
            "B1 feature cache has no historical rows and the incremental window "
            "produced no signals"
        )

    if not combined.empty:
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
        if not args.live_only:
            combined = assign_symbol_splits(
                combined,
                args.oot_start,
                args.test_size,
                args.random_state,
            )

    required_features, release_id = _released_b1_required_features()

    if pd.isna(actual_source_latest):
        raise RuntimeError("Cannot publish active candidate features without a source trade date")
    active_b1_gate = b1_gate_rows
    if gate_mode != "signal_gate":
        active_b1_gate = (
            incremental.loc[
                pd.to_datetime(incremental["date"], errors="raise")
                == actual_source_latest,
                ["symbol", "date"],
            ].drop_duplicates()
            if not incremental.empty
            else pd.DataFrame(columns=["symbol", "date"])
        )
    active_features, active_stats = _assemble_active_candidate_cache(
        incremental,
        additional,
        active_b1_gate,
        additional_gate_rows,
        family_gate_rows,
        target_date=actual_source_latest,
        factor_schema_version=factor_schema_version,
    )
    candidate_coverage = _validate_active_candidate_coverage(
        active_stats,
        additional_coverage["excluded_candidates"],
    )
    active_b1_features = (
        active_features[
            active_features["candidate_source_b1"].fillna(False)
        ].copy()
        if "candidate_source_b1" in active_features.columns
        else active_features
    )
    if active_b1_features.empty:
        feature_coverage = {
            "status": "valid",
            "target_date": actual_source_latest.strftime("%Y-%m-%d"),
            "row_count": 0,
            "required_feature_count": len(required_features),
            "present_feature_count": len(
                set(required_features) & set(active_features.columns)
            ),
            "covered_feature_count": 0,
            "empty_candidate_set": True,
            "missing_columns": sorted(
                set(required_features) - set(active_features.columns)
            ),
            "all_null_features": [],
            "partial_features": [],
            "coverage": {},
            "non_null_counts": {},
        }
    else:
        feature_coverage = validate_required_feature_coverage(
            active_b1_features,
            required_features,
            target_date=actual_source_latest,
            context="B1 live active-candidate feature refresh",
        )
    updated_at = datetime.now().isoformat(timespec="seconds")
    active_manifest = {
        "status": "success",
        "updated_at": updated_at,
        **active_stats,
        **candidate_coverage,
        "factor_schema_version": factor_schema_version,
        "factor_count": len(PROJECT_FACTOR_COLUMNS),
        "additional_gate_mode": additional_gate_mode,
        "additional_gate_reason": additional_gate_reason,
        "additional_calculated_symbol_count": len(additional_symbols),
        "additional_feature_error_count": additional_coverage["symbol_error_count"],
        "additional_feature_error_samples": additional_coverage["symbol_error_samples"],
        "additional_excluded_candidates": additional_coverage["excluded_candidates"],
        "output": str(args.active_feature_out),
    }

    if not args.live_only:
        atomic_write_parquet(combined, args.dataset_out, index=False)
    atomic_write_parquet(active_features, args.active_feature_out, index=False)
    active_manifest["output_sha256"] = _sha256(args.active_feature_out)
    atomic_write_json(active_manifest, args.active_feature_manifest)

    result = {
        "status": "success",
        "updated_at": updated_at,
        "incremental_start_date": start_ts.strftime("%Y-%m-%d"),
        "incremental_rows": int(len(incremental)),
        "factor_schema_version": factor_schema_version,
        "release_id": release_id,
        "feature_history_years": MODEL_FEATURE_HISTORY_YEARS,
        "feature_coverage": feature_coverage,
        "gate_mode": gate_mode,
        "gate_fallback_reason": gate_reason,
        "additional_gate_mode": additional_gate_mode,
        "additional_gate_reason": additional_gate_reason,
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
        "history_cache_updated": not args.live_only,
        "active_candidate_features": active_stats,
        "active_feature_output": str(args.active_feature_out),
        "active_feature_manifest": str(args.active_feature_manifest),
        "date_min": combined["date"].min().strftime("%Y-%m-%d") if not combined.empty else None,
        "date_max": combined["date"].max().strftime("%Y-%m-%d") if not combined.empty else None,
        "output": str(args.active_feature_out if args.live_only else args.dataset_out),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
