#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild both selector rule caches with one market scan and one factor pass."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections import Counter
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
from quant.data.atomic_io import atomic_write_json
from quant.data.source_merge import normalize_tushare_daily
from quant.features.b1_gate import B1_GATE_METRIC_COLUMNS, calculate_b1_gate
from quant.features.daily_factor_layer import (
    DEFAULT_FACTOR_ROOT,
    attach_daily_base_factors,
    attach_daily_signal_factors,
)

B1_GATE_CACHE = PROJECT_ROOT / "data/features/b1/b1_gate_candidates.parquet"
B1_GATE_MANIFEST = PROJECT_ROOT / "data/features/b1/b1_gate_manifest.json"
CACHE_IDENTITY_SCHEMA_VERSION = 2
DEFAULT_BATCH_SIZE = 1
HASH_CHUNK_SIZE = 1024 * 1024

# A fast-path hit is valid only while every implementation/configuration input
# capable of changing the three published signal caches is byte-for-byte the
# same. Keep this deliberately narrower than a repository-wide hash so
# unrelated edits do not invalidate an expensive market scan.
CONTRACT_PATHS = (
    Path(__file__),
    PROJECT_ROOT / "scripts/research/analyze_b1_family_rule_backtest.py",
    PROJECT_ROOT / "scripts/research/analyze_z_skill_entry_exit_backtest.py",
    PROJECT_ROOT / "src/quant/data/market_data_store.py",
    PROJECT_ROOT / "src/quant/data/source_merge.py",
    PROJECT_ROOT / "src/quant/data/factors/__init__.py",
    PROJECT_ROOT / "src/quant/data/factors/alpha101.py",
    PROJECT_ROOT / "src/quant/data/factors/alpha191.py",
    PROJECT_ROOT / "src/quant/data/factors/base.py",
    PROJECT_ROOT / "src/quant/data/factors/data_adapter.py",
    PROJECT_ROOT / "src/quant/data/factors/momentum.py",
    PROJECT_ROOT / "src/quant/data/factors/technical.py",
    PROJECT_ROOT / "src/quant/features/b1_gate.py",
    PROJECT_ROOT / "src/quant/features/daily_factor_layer.py",
    PROJECT_ROOT / "src/quant/features/project_factor_layer.py",
    PROJECT_ROOT / "src/quant/features/variable_library.py",
    PROJECT_ROOT / "src/quant/strategies/custom/triple_volume_breakout.py",
    PROJECT_ROOT / "src/quant/strategies/custom/vegas_tunnel.py",
    PROJECT_ROOT / "configs/strategies/triple_volume_breakout.yaml",
)
SEMANTIC_ENV_KEYS = (
    "PROJECT_FACTOR_COMPATIBILITY_MODE",
    "ROUTINE_PRODUCTION_FACTOR_SCHEMA",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(
            os.getenv("ROUTINE_SIGNAL_PROCESS_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        ),
        help=(
            "每个进程任务包含的股票数；默认1保持已验证吞吐，"
            "可在目标机器基准后调大。"
        ),
    )
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
    parser.add_argument(
        "--family-cache",
        type=Path,
        default=family_rules.SIGNAL_CACHE,
    )
    parser.add_argument(
        "--extended-cache",
        type=Path,
        default=z_skills.SIGNAL_CACHE,
    )
    parser.add_argument("--b1-gate-cache", type=Path, default=B1_GATE_CACHE)
    parser.add_argument("--b1-gate-manifest", type=Path, default=B1_GATE_MANIFEST)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contract_fingerprint() -> str:
    records: list[dict[str, str]] = []
    for path in CONTRACT_PATHS:
        resolved = Path(path)
        if not resolved.is_file():
            records.append({"path": str(resolved), "sha256": "missing"})
            continue
        try:
            label = str(resolved.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            label = str(resolved.resolve())
        records.append({"path": label, "sha256": _sha256_file(resolved)})
    return _stable_json_fingerprint(records)


def _semantic_params_fingerprint(
    *,
    rebuild_from: pd.Timestamp,
    start_date: str,
    factor_mode: str,
) -> str:
    return _stable_json_fingerprint(
        {
            "identity_schema": CACHE_IDENTITY_SCHEMA_VERSION,
            "rebuild_from": rebuild_from.strftime("%Y-%m-%d"),
            "start_date": str(start_date),
            "factor_mode": factor_mode,
            "runtime": {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "environment": {
                key: os.getenv(key)
                for key in SEMANTIC_ENV_KEYS
            },
        }
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_identity(
    path: Path,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a content identity, reusing a prior digest for unchanged stats."""

    if not path.is_file():
        return None
    stat = path.stat()
    if (
        isinstance(previous, dict)
        and int(previous.get("size", -1)) == stat.st_size
        and int(previous.get("mtime_ns", -1)) == stat.st_mtime_ns
        and previous.get("sha256")
    ):
        digest = str(previous["sha256"])
    else:
        digest = _sha256_file(path)
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }


def _partitioned_source_identity(
    daily_dir: Path,
    *,
    history_start: pd.Timestamp,
    processed_through: pd.Timestamp,
    previous: dict[str, Any] | None = None,
    sql_url: str | None = None,
) -> dict[str, Any]:
    """Fingerprint exactly the parquet partitions read by this refresh.

    SQL rows do not currently expose a revision/high-watermark column, so a
    correctness-preserving fast path is deliberately disabled for SQL-backed
    reads. Parquet hashes are cached by size+mtime; on a normal day only the
    current month needs to be re-hashed.
    """

    if sql_url:
        return {
            "fingerprint": None,
            "reason": "sql_source_has_no_revision_fingerprint",
            "partitions": {},
        }
    partition_root = daily_dir.parent / f"{daily_dir.name}_partitioned"
    start_month = pd.Timestamp(history_start).strftime("%Y%m")
    end_month = pd.Timestamp(processed_through).strftime("%Y%m")
    previous_partitions = (
        previous.get("partitions", {})
        if isinstance(previous, dict)
        else {}
    )
    partitions: dict[str, dict[str, Any]] = {}
    for path in sorted(partition_root.glob("year_month=*/data.parquet")):
        month = path.parent.name.partition("=")[2]
        if month < start_month or month > end_month:
            continue
        key = str(path.relative_to(partition_root))
        identity = _file_identity(path, previous_partitions.get(key))
        if identity is not None:
            identity.pop("path", None)
            partitions[key] = identity
    if not partitions:
        return {
            "fingerprint": None,
            "reason": "partitioned_source_unavailable",
            "partitions": {},
        }
    fingerprint = _stable_json_fingerprint(
        {
            "history_start": pd.Timestamp(history_start).strftime("%Y-%m-%d"),
            "processed_through": pd.Timestamp(processed_through).strftime(
                "%Y-%m-%d"
            ),
            "partitions": {
                key: value["sha256"]
                for key, value in sorted(partitions.items())
            },
        }
    )
    return {
        "fingerprint": fingerprint,
        "reason": None,
        "partitions": partitions,
    }


def _market_frame_source_identity(
    market: pd.DataFrame,
    *,
    history_start: pd.Timestamp,
    processed_through: pd.Timestamp,
) -> dict[str, Any]:
    """Build a semantic identity for SQL reads without a revision column.

    Reading the projected 600-day frame is still much cheaper than evaluating
    every rolling rule for every symbol. Hashing normalized source values (not
    SQL result order) also catches late corrections that MAX(date)/COUNT would
    miss.
    """

    source_columns = [
        column
        for column in (
            "ts_code",
            "trade_date",
            "date",
            "name",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "pct_change",
            "vol",
            "volume",
            "amount",
            "adj_factor",
        )
        if column in market.columns
    ]
    source = market[source_columns].copy()
    if "trade_date" in source.columns:
        source["trade_date"] = (
            source["trade_date"].astype(str).str.replace("-", "", regex=False)
        )
    if "date" in source.columns:
        source["date"] = pd.to_datetime(
            source["date"],
            errors="coerce",
        ).astype("int64")
    for column in ("ts_code", "name"):
        if column in source.columns:
            source[column] = source[column].fillna("").astype(str)
    sort_columns = [
        column
        for column in ("trade_date", "date", "ts_code")
        if column in source.columns
    ]
    if sort_columns:
        source = source.sort_values(sort_columns).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(
        source,
        index=False,
        categorize=True,
    ).to_numpy(dtype="uint64", copy=False)
    digest = hashlib.sha256()
    digest.update("\0".join(source_columns).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    fingerprint = _stable_json_fingerprint(
        {
            "history_start": pd.Timestamp(history_start).strftime("%Y-%m-%d"),
            "processed_through": pd.Timestamp(processed_through).strftime(
                "%Y-%m-%d"
            ),
            "rows": int(len(source)),
            "columns": source_columns,
            "semantic_sha256": digest.hexdigest(),
        }
    )
    return {
        "fingerprint": fingerprint,
        "reason": None,
        "source_type": "sql_frame_semantic",
        "rows": int(len(source)),
        "partitions": {},
    }


def _output_identities(
    paths: dict[str, Path],
    previous: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]] | None:
    identities: dict[str, dict[str, Any]] = {}
    previous = previous if isinstance(previous, dict) else {}
    for key, path in paths.items():
        identity = _file_identity(path, previous.get(key))
        if identity is None:
            return None
        identities[key] = identity
    return identities


def _fast_path_result(
    *,
    manifest: dict[str, Any],
    source_identity: dict[str, Any],
    contract_fingerprint: str,
    params_fingerprint: str,
    output_paths: dict[str, Path],
) -> dict[str, Any] | None:
    """Return a zero-scan result only for an exact input/contract match."""

    identity = manifest.get("cache_identity")
    if not isinstance(identity, dict):
        return None
    if source_identity.get("fingerprint") is None:
        return None
    expected = {
        "schema_version": CACHE_IDENTITY_SCHEMA_VERSION,
        "source_fingerprint": source_identity.get("fingerprint"),
        "contract_fingerprint": contract_fingerprint,
        "params_fingerprint": params_fingerprint,
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        return None
    recorded_outputs = identity.get("outputs")
    if not isinstance(recorded_outputs, dict):
        return None
    current_outputs = _output_identities(output_paths, recorded_outputs)
    if current_outputs is None:
        return None
    if any(
        current_outputs[key].get("sha256")
        != recorded_outputs.get(key, {}).get("sha256")
        for key in output_paths
    ):
        return None
    family_summary = manifest.get("family")
    extended_summary = manifest.get("extended")
    if not isinstance(family_summary, dict) or not isinstance(
        extended_summary,
        dict,
    ):
        return None
    return {
        "status": "success",
        "execution_mode": "input_contract_cache_hit",
        "checkpoint_reused": True,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "incremental_start_date": manifest.get("incremental_start_date"),
        "processed_through_date": manifest.get("processed_through_date"),
        "symbols": int(manifest.get("source_symbol_count") or 0),
        "symbol_errors": 0,
        "symbol_error_rate": 0.0,
        "factor_mode": manifest.get("factor_mode"),
        "family": family_summary,
        "extended": extended_summary,
        "b1_gate_rows": int(manifest.get("candidate_rows") or 0),
        "b1_gate_symbols": int(manifest.get("candidate_symbols") or 0),
    }


@contextlib.contextmanager
def _publish_lock(manifest_path: Path):
    """Serialize all cache-set publication attempts for this manifest."""

    lock_path = manifest_path.with_suffix(f"{manifest_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _path_snapshot(path: Path) -> dict[str, Any] | None:
    identity = _file_identity(path)
    if identity is None:
        return None
    return {
        "size": identity["size"],
        "mtime_ns": identity["mtime_ns"],
        "sha256": identity["sha256"],
    }


def _publish_cache_set(
    frames: dict[str, pd.DataFrame],
    output_paths: dict[str, Path],
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    rewrite_keys: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Publish three parquet files plus commit-marker manifest transactionally.

    Filesystems cannot atomically rename four paths together. We therefore
    stage every artifact, replace the manifest last as the commit marker, and
    restore every pre-existing artifact if any replacement fails. Readers that
    validate the manifest identities never accept an intermediate mixed set.
    """

    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    stage_paths: dict[str, Path] = {}
    backup_paths: dict[str, Path] = {}
    targets = {**output_paths, "manifest": manifest_path}
    rewrite_keys = set(frames) if rewrite_keys is None else set(rewrite_keys)
    try:
        for key, frame in frames.items():
            target = output_paths[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            stage = target.with_name(f".{target.name}.{token}.stage")
            if key not in rewrite_keys and target.is_file():
                try:
                    os.link(target, stage)
                except OSError:
                    shutil.copy2(target, stage)
            else:
                frame.to_parquet(stage, index=False)
            stage_paths[key] = stage
        staged_identities = _output_identities(
            {
                key: stage_paths[key]
                for key in output_paths
            }
        )
        if staged_identities is None:
            raise RuntimeError("failed to fingerprint staged signal caches")
        manifest_identity = manifest.get("cache_identity")
        if not isinstance(manifest_identity, dict):
            raise RuntimeError("signal manifest missing cache identity")
        published_identities: dict[str, dict[str, Any]] = {}
        for key, target in output_paths.items():
            staged = staged_identities[key]
            published_identities[key] = {
                **staged,
                "path": str(target),
            }
        manifest_identity["outputs"] = published_identities
        manifest_stage = manifest_path.with_name(
            f".{manifest_path.name}.{token}.stage"
        )
        atomic_write_json(manifest, manifest_stage)
        stage_paths["manifest"] = manifest_stage

        for key, target in targets.items():
            if not target.is_file():
                continue
            backup = target.with_name(f".{target.name}.{token}.backup")
            try:
                os.link(target, backup)
            except OSError:
                shutil.copy2(target, backup)
            backup_paths[key] = backup

        replaced: list[str] = []
        try:
            for key in (*output_paths.keys(), "manifest"):
                os.replace(stage_paths[key], targets[key])
                replaced.append(key)
            return published_identities
        except Exception:
            for key in reversed(replaced):
                backup = backup_paths.get(key)
                if backup is not None and backup.is_file():
                    os.replace(backup, targets[key])
                else:
                    targets[key].unlink(missing_ok=True)
            raise
    finally:
        for path in (*stage_paths.values(), *backup_paths.values()):
            path.unlink(missing_ok=True)


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


def _build_b1_gate_rows(
    symbol: str,
    frame: pd.DataFrame,
    rebuild_start: str,
) -> pd.DataFrame:
    start = pd.Timestamp(rebuild_start)
    history_start = start - pd.Timedelta(days=450)
    daily = normalize_tushare_daily(frame, symbol)
    daily = (
        daily[pd.to_datetime(daily["date"], errors="coerce") >= history_start]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(daily) < 130:
        return pd.DataFrame(
            columns=["symbol", "date", *B1_GATE_METRIC_COLUMNS]
        )
    name = str(daily["name"].iloc[0]) if "name" in daily.columns else ""
    if "ST" in name.upper() or "退" in name:
        return pd.DataFrame(
            columns=["symbol", "date", *B1_GATE_METRIC_COLUMNS]
        )
    gate = calculate_b1_gate(daily)
    rows = daily.loc[
        gate["b1_gate"] & daily["date"].ge(start),
        ["symbol", "date"],
    ].copy()
    for column in B1_GATE_METRIC_COLUMNS:
        rows[column] = gate.loc[rows.index, column]
    return rows.reset_index(drop=True)


def _process_symbol(
    symbol: str,
    frame: pd.DataFrame,
    rebuild_start: str,
    factor_mode: str = "stateful",
    factor_root: Path = DEFAULT_FACTOR_ROOT,
) -> dict[str, Any]:
    """Compute shared factors once, then fan into both signal families."""

    try:
        gate_rows = _build_b1_gate_rows(
            symbol,
            frame,
            rebuild_start,
        )
        normalized = family_rules.normalize_daily_frame(frame, symbol)
        if "name" in normalized.columns:
            names = normalized["name"].fillna("").astype(str)
            normalized = normalized[
                ~names.str.upper().str.contains("ST")
                & ~names.str.contains("退")
            ].copy()
        if len(normalized) < 130:
            return {
                "symbol": symbol,
                "family": None,
                "extended": None,
                "b1_gate": gate_rows,
                "errors": [],
                "factor_cache_mode": "insufficient_history",
            }
        if len(normalized) < 160:
            return {
                "symbol": symbol,
                "family": None,
                "extended": None,
                "b1_gate": gate_rows.reset_index(drop=True),
                "errors": [],
                "factor_cache_mode": "insufficient_signal_history",
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
            "b1_gate": None,
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
    # The parent only publishes rows in the replacement interval. Filtering
    # here avoids serializing hundreds of thousands of historical candidate
    # rows back across process boundaries on every daily run.
    rebuild_from = pd.Timestamp(rebuild_start)
    if family is not None and not family.empty:
        family_dates = pd.to_datetime(family["date"], errors="coerce")
        family = family[family_dates >= rebuild_from].copy()
    if extended is not None and not extended.empty:
        extended_dates = pd.to_datetime(extended["date"], errors="coerce")
        extended = extended[extended_dates >= rebuild_from].copy()
    return {
        "symbol": symbol,
        "family": family,
        "extended": extended,
        "b1_gate": gate_rows.reset_index(drop=True),
        "errors": errors,
        "factor_cache_mode": factor_cache_mode,
    }


def _process_symbol_batch(
    tasks: list[tuple[str, pd.DataFrame]],
    rebuild_start: str,
    factor_mode: str,
    factor_root: Path,
) -> list[dict[str, Any]]:
    """Process several symbols per Future to amortize IPC/scheduler overhead."""

    return [
        _process_symbol(
            symbol,
            frame,
            rebuild_start,
            factor_mode,
            factor_root,
        )
        for symbol, frame in tasks
    ]


def _task_batches(
    tasks: list[tuple[str, pd.DataFrame]],
    batch_size: int,
) -> list[list[tuple[str, pd.DataFrame]]]:
    size = max(1, int(batch_size))
    return [tasks[index : index + size] for index in range(0, len(tasks), size)]


def _merge_incremental_cache(
    cached: pd.DataFrame | None,
    frames: list[pd.DataFrame],
    rebuild_from: pd.Timestamp,
    *,
    empty_columns: set[str] | None = None,
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
        # A completed market scan can legitimately produce no rule hits.  The
        # target-day manifest, rather than a candidate row, is the freshness
        # truth for that case.  Keep a schema-bearing parquet so downstream
        # readers can still load the empty candidate set deterministically.
        return pd.DataFrame(
            columns=["symbol", "date", *sorted(empty_columns or set())]
        )
    return (
        combined.dropna(subset=["symbol", "date"])
        .sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )


def _frames_exactly_equal(
    cached: pd.DataFrame | None,
    rebuilt: pd.DataFrame,
) -> bool:
    if cached is None or list(cached.columns) != list(rebuilt.columns):
        return False
    return cached.reset_index(drop=True).equals(rebuilt.reset_index(drop=True))


def _assert_source_unchanged(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    phase: str,
) -> None:
    before_fingerprint = before.get("fingerprint")
    after_fingerprint = after.get("fingerprint")
    if (
        before_fingerprint is None
        or after_fingerprint != before_fingerprint
    ):
        raise RuntimeError(f"canonical market source changed {phase}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = perf_counter()
    timings: dict[str, float] = {}
    family_columns = {spec.name for spec in family_rules.build_signal_specs()}
    extended_columns = {spec.key for spec in z_skills.build_signal_specs()}
    output_paths = {
        "family": Path(args.family_cache),
        "extended": Path(args.extended_cache),
        "b1_gate": Path(args.b1_gate_cache),
    }

    # Preserve the explicit no-date cache reuse mode. The routine path always
    # supplies an incremental date and uses the stronger identity gate below.
    if args.incremental_start_date is None and not args.force_refresh:
        family_cached = _load_cache(
            output_paths["family"],
            family_columns,
            force_refresh=False,
        )
        extended_cached = _load_cache(
            output_paths["extended"],
            extended_columns,
            force_refresh=False,
        )
        if family_cached is not None and extended_cached is not None:
            return {
                "status": "success",
                "execution_mode": "cache_reuse",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "family": _summary(family_cached, sorted(family_columns)),
                "extended": _summary(
                    extended_cached,
                    sorted(extended_columns),
                ),
                "elapsed_seconds": perf_counter() - started,
            }

    rebuild_from = _parse_date(
        args.incremental_start_date or args.start_date
    ).normalize()
    history_start = rebuild_from - pd.Timedelta(days=600)
    store = MarketDataStore(
        MarketDataStoreConfig.from_env(root=args.daily_dir.parent)
    )
    previous_manifest = _read_manifest(Path(args.b1_gate_manifest))
    previous_identity = previous_manifest.get("cache_identity", {})

    identity_started = perf_counter()
    contract_fingerprint = _contract_fingerprint()
    params_fingerprint = _semantic_params_fingerprint(
        rebuild_from=rebuild_from,
        start_date=args.start_date,
        factor_mode=args.factor_mode,
    )
    latest_before = store.latest_dataset_trade_date(args.daily_dir.name)
    source_identity = {
        "fingerprint": None,
        "reason": "source_latest_unavailable",
        "partitions": {},
    }
    if latest_before is not None and pd.notna(latest_before):
        source_identity = _partitioned_source_identity(
            args.daily_dir,
            history_start=history_start,
            processed_through=latest_before.normalize(),
            previous=(
                previous_identity.get("source")
                if isinstance(previous_identity, dict)
                else None
            ),
            sql_url=store.config.sql_url,
        )
    timings["identity_seconds"] = perf_counter() - identity_started

    if not args.force_refresh:
        fast_result = _fast_path_result(
            manifest=previous_manifest,
            source_identity=source_identity,
            contract_fingerprint=contract_fingerprint,
            params_fingerprint=params_fingerprint,
            output_paths=output_paths,
        )
        if fast_result is not None:
            fast_result.update(
                {
                    "family_cache": str(output_paths["family"]),
                    "extended_cache": str(output_paths["extended"]),
                    "b1_gate_cache": str(output_paths["b1_gate"]),
                    "b1_gate_manifest": str(args.b1_gate_manifest),
                    "timings": timings,
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            return fast_result

    market_read_started = perf_counter()
    market = store.read_market_range(
        args.daily_dir.name,
        start_date=history_start.strftime("%Y%m%d"),
    )
    timings["market_read_seconds"] = perf_counter() - market_read_started
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
    processed_through = processed_through.normalize()
    if store.config.sql_url:
        sql_identity_started = perf_counter()
        source_identity = _market_frame_source_identity(
            market,
            history_start=history_start,
            processed_through=processed_through,
        )
        timings["sql_source_identity_seconds"] = (
            perf_counter() - sql_identity_started
        )
        if not args.force_refresh:
            fast_result = _fast_path_result(
                manifest=previous_manifest,
                source_identity=source_identity,
                contract_fingerprint=contract_fingerprint,
                params_fingerprint=params_fingerprint,
                output_paths=output_paths,
            )
            if fast_result is not None:
                fast_result.update(
                    {
                        "family_cache": str(output_paths["family"]),
                        "extended_cache": str(output_paths["extended"]),
                        "b1_gate_cache": str(output_paths["b1_gate"]),
                        "b1_gate_manifest": str(args.b1_gate_manifest),
                        "timings": timings,
                        "elapsed_seconds": perf_counter() - started,
                    }
                )
                return fast_result
    elif latest_before is None or processed_through != latest_before.normalize():
        source_identity = _partitioned_source_identity(
            args.daily_dir,
            history_start=history_start,
            processed_through=processed_through,
            previous=(
                previous_identity.get("source")
                if isinstance(previous_identity, dict)
                else None
            ),
            sql_url=store.config.sql_url,
        )

    tasks = [
        (str(symbol), group.reset_index(drop=True))
        for symbol, group in market.groupby("ts_code", sort=False)
    ]
    source_symbol_count = len(tasks)
    batches = _task_batches(tasks, args.batch_size)
    worker_batch_count = len(batches)
    family_frames: list[pd.DataFrame] = []
    extended_frames: list[pd.DataFrame] = []
    b1_gate_frames: list[pd.DataFrame] = []
    symbol_errors: list[dict[str, Any]] = []
    factor_cache_modes: Counter[str] = Counter()

    compute_started = perf_counter()
    processed_symbols = 0
    next_progress = 500
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                _process_symbol_batch,
                batch,
                rebuild_from.strftime("%Y-%m-%d"),
                args.factor_mode,
                args.factor_root,
            )
            for batch in batches
        ]
        for future in as_completed(futures):
            for item in future.result():
                processed_symbols += 1
                factor_cache_modes[item["factor_cache_mode"]] += 1
                if item["family"] is not None and not item["family"].empty:
                    family_frames.append(item["family"])
                if item["extended"] is not None and not item["extended"].empty:
                    extended_frames.append(item["extended"])
                if item["b1_gate"] is not None and not item["b1_gate"].empty:
                    b1_gate_frames.append(item["b1_gate"])
                if item["errors"]:
                    symbol_errors.append(
                        {
                            "symbol": item["symbol"],
                            "errors": item["errors"],
                        }
                    )
            if processed_symbols >= next_progress or processed_symbols == len(tasks):
                print(
                    f"  combined signals: {processed_symbols}/{len(tasks)} symbols",
                    flush=True,
                )
                while next_progress <= processed_symbols:
                    next_progress += 500
    timings["compute_seconds"] = perf_counter() - compute_started

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

    # Keep the 125MB historical candidate caches out of memory while workers
    # are active. They are loaded only once, after the market/process payloads
    # can be released, and unchanged outputs are not rewritten.
    del market, market_dates, tasks, batches, futures
    cache_read_started = perf_counter()
    baseline_manifest_snapshot = _path_snapshot(Path(args.b1_gate_manifest))
    family_cached = _load_cache(
        output_paths["family"],
        family_columns,
        force_refresh=args.force_refresh,
    )
    extended_cached = _load_cache(
        output_paths["extended"],
        extended_columns,
        force_refresh=args.force_refresh,
    )
    family = _merge_incremental_cache(
        family_cached,
        family_frames,
        rebuild_from,
        empty_columns=family_columns,
    )
    extended = _merge_incremental_cache(
        extended_cached,
        extended_frames,
        rebuild_from,
        empty_columns=extended_columns,
    )
    b1_gate_cached = (
        pd.read_parquet(output_paths["b1_gate"])
        if output_paths["b1_gate"].exists() and not args.force_refresh
        else None
    )
    baseline_outputs = {
        key: _path_snapshot(path)
        for key, path in output_paths.items()
    }
    if b1_gate_cached is not None:
        b1_gate_cached["date"] = pd.to_datetime(
            b1_gate_cached["date"],
            errors="coerce",
        )
    recent_b1_gate = (
        pd.concat(b1_gate_frames, ignore_index=True, sort=False)
        if b1_gate_frames
        else pd.DataFrame(columns=["symbol", "date", *B1_GATE_METRIC_COLUMNS])
    )
    old_b1_gate = (
        b1_gate_cached[b1_gate_cached["date"] < rebuild_from].copy()
        if b1_gate_cached is not None
        else pd.DataFrame()
    )
    b1_gate = pd.concat(
        [old_b1_gate, recent_b1_gate],
        ignore_index=True,
        sort=False,
    )
    if not b1_gate.empty:
        b1_gate["date"] = pd.to_datetime(b1_gate["date"], errors="coerce")
        b1_gate = (
            b1_gate.dropna(subset=["symbol", "date"])
            .sort_values(["date", "symbol"])
            .drop_duplicates(["symbol", "date"], keep="last")
            .reset_index(drop=True)
        )
    timings["cache_read_merge_seconds"] = perf_counter() - cache_read_started

    rewritten = [
        key
        for key, cached, rebuilt in (
            ("family", family_cached, family),
            ("extended", extended_cached, extended),
            ("b1_gate", b1_gate_cached, b1_gate),
        )
        if not _frames_exactly_equal(cached, rebuilt)
    ]

    # Re-evaluate the source before taking the publish lock. A concurrent
    # source write must never overwrite the last-known-good cache set.
    if not store.config.sql_url:
        final_source_identity = _partitioned_source_identity(
            args.daily_dir,
            history_start=history_start,
            processed_through=processed_through,
            previous=source_identity,
            sql_url=None,
        )
        _assert_source_unchanged(
            source_identity,
            final_source_identity,
            phase="during signal refresh",
        )
        source_identity = final_source_identity

    target_date = processed_through

    def target_symbols(frame: pd.DataFrame) -> set[str]:
        if frame.empty or not {"symbol", "date"} <= set(frame.columns):
            return set()
        dates = pd.to_datetime(frame["date"], errors="coerce")
        return set(
            frame.loc[dates.eq(target_date), "symbol"].dropna().astype(str)
        )

    family_target_symbols = target_symbols(family)
    z_target_symbols = target_symbols(extended)
    b1_target_symbols = target_symbols(b1_gate)
    active_union_symbols = b1_target_symbols | z_target_symbols
    family_summary = _summary(family, sorted(family_columns))
    extended_summary = _summary(extended, sorted(extended_columns))
    signal_manifest = {
        "status": "success",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "incremental_start_date": rebuild_from.strftime("%Y-%m-%d"),
        "processed_through_date": target_date.strftime("%Y-%m-%d"),
        "source_symbol_count": source_symbol_count,
        "candidate_rows": int(len(recent_b1_gate)),
        "candidate_symbols": int(
            recent_b1_gate["symbol"].nunique()
            if not recent_b1_gate.empty
            else 0
        ),
        "family_target_candidate_count": len(family_target_symbols),
        "b1_candidate_count": len(b1_target_symbols),
        "z_candidate_count": len(z_target_symbols),
        "union_candidate_count": len(active_union_symbols),
        "overlap_candidate_count": len(b1_target_symbols & z_target_symbols),
        "empty_candidate_set": not active_union_symbols,
        "signal_scan_status": "complete",
        "factor_mode": args.factor_mode,
        "family": family_summary,
        "extended": extended_summary,
        "cache_identity": {
            "schema_version": CACHE_IDENTITY_SCHEMA_VERSION,
            "source_fingerprint": source_identity.get("fingerprint"),
            "contract_fingerprint": contract_fingerprint,
            "params_fingerprint": params_fingerprint,
            "source": source_identity,
            # Filled from staged bytes before the manifest commit marker is
            # atomically replaced.
            "outputs": {},
        },
    }
    manifest_path = Path(args.b1_gate_manifest)
    cache_write_started = perf_counter()
    checkpoint_reused = False
    with _publish_lock(manifest_path):
        # Another refresh may have completed while this one was computing.
        # Never merge this result onto one historical baseline and publish it
        # over a different baseline.
        if _path_snapshot(manifest_path) != baseline_manifest_snapshot or any(
            _path_snapshot(path) != baseline_outputs[key]
            for key, path in output_paths.items()
        ):
            raise RuntimeError(
                "signal cache baseline changed during refresh; retry on latest"
            )
        if not store.config.sql_url:
            locked_source_identity = _partitioned_source_identity(
                args.daily_dir,
                history_start=history_start,
                processed_through=processed_through,
                previous=source_identity,
                sql_url=None,
            )
            _assert_source_unchanged(
                source_identity,
                locked_source_identity,
                phase="before signal publish",
            )
        else:
            # SQL has no source revision column, so recompute the semantic
            # fingerprint while holding the publication lock. This closes the
            # gap between the final source check and the manifest commit.
            source_recheck_started = perf_counter()
            locked_market = store.read_market_range(
                args.daily_dir.name,
                start_date=history_start.strftime("%Y%m%d"),
            )
            if locked_market.empty:
                raise RuntimeError(
                    "canonical market source disappeared before signal publish"
                )
            locked_dates = pd.to_datetime(
                locked_market.get("date", locked_market.get("trade_date")),
                errors="coerce",
            )
            locked_source_identity = _market_frame_source_identity(
                locked_market,
                history_start=history_start,
                processed_through=locked_dates.max(),
            )
            timings["source_recheck_seconds"] = (
                perf_counter() - source_recheck_started
            )
            _assert_source_unchanged(
                source_identity,
                locked_source_identity,
                phase="before signal publish",
            )
            del locked_market, locked_dates
        locked_manifest = _read_manifest(manifest_path)
        locked_fast_result = _fast_path_result(
            manifest=locked_manifest,
            source_identity=source_identity,
            contract_fingerprint=contract_fingerprint,
            params_fingerprint=params_fingerprint,
            output_paths=output_paths,
        )
        if not rewritten and locked_fast_result is not None:
            output_identities = locked_manifest["cache_identity"]["outputs"]
            checkpoint_reused = True
        else:
            output_identities = _publish_cache_set(
                {
                    "family": family,
                    "extended": extended,
                    "b1_gate": b1_gate,
                },
                output_paths,
                signal_manifest,
                manifest_path,
                rewrite_keys=set(rewritten),
            )
    timings["cache_write_seconds"] = perf_counter() - cache_write_started

    return {
        "status": "success",
        "execution_mode": "fused_batched_market_scan",
        "checkpoint_reused": checkpoint_reused,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "incremental_start_date": rebuild_from.strftime("%Y-%m-%d"),
        "processed_through_date": processed_through.strftime("%Y-%m-%d"),
        "symbols": source_symbol_count,
        "symbol_errors": len(symbol_errors),
        "symbol_error_rate": round(error_rate, 6),
        "symbol_error_examples": symbol_errors[:20],
        "factor_mode": args.factor_mode,
        "factor_cache_modes": dict(sorted(factor_cache_modes.items())),
        "factor_root": str(args.factor_root),
        "workers": max(1, args.workers),
        "batch_size": max(1, args.batch_size),
        "worker_batches": worker_batch_count,
        "rewritten_outputs": rewritten,
        "timings": timings,
        "elapsed_seconds": perf_counter() - started,
        "family": family_summary,
        "extended": extended_summary,
        "family_cache": str(output_paths["family"]),
        "extended_cache": str(output_paths["extended"]),
        "b1_gate_cache": str(output_paths["b1_gate"]),
        "b1_gate_manifest": str(args.b1_gate_manifest),
        "b1_gate_rows": int(len(recent_b1_gate)),
        "b1_gate_symbols": int(
            recent_b1_gate["symbol"].nunique()
            if not recent_b1_gate.empty
            else 0
        ),
    }


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
