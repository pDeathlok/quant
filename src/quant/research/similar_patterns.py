"""Historical similar-pattern retrieval and distributional stock forecasts."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable, Iterator
from uuid import uuid4

import numpy as np
import pandas as pd

from quant.data import (
    MarketDataStore,
    MarketDataStoreConfig,
    list_partitioned_symbol_paths,
    read_partitioned_symbol_file,
)
from quant.data.factors.technical import KDJ
from quant.infrastructure.publication import ContextThreadPoolExecutor as ThreadPoolExecutor


_MATRIX_CACHE_SCHEMA_VERSION = 1
_MATRIX_CACHE_DIRNAME = "_matrix_cache_v1"
_MATRIX_CACHE_CHUNK_SYMBOLS = 96
_MATRIX_CACHE_CHUNK_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_VECTOR_CACHE_SOURCE_METADATA_FILENAME = "_refresh_metadata.json"
VECTOR_CACHE_PENDING_FILENAME = "_publication_pending.json"
_VECTOR_CACHE_COMMIT_FILENAME = "_publication_commit.json"
_VECTOR_SOURCE_COLUMNS = (
    "ts_code",
    "trade_date",
    "symbol",
    "date",
    "name",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "volume",
    "pct_chg",
    "pct_change",
)
_VECTOR_SOURCE_BATCH_SYMBOLS = 16
_VECTOR_INCREMENTAL_SCHEMA = 1
_PARTITION_CONTENT_DIGEST_CACHE: dict[tuple[str, int, int, int], str] = {}
_PARTITION_CONTENT_DIGEST_CACHE_MAX_ENTRIES = 512
_MATRIX_CACHE_FLOAT_FIELDS = (
    "close",
    "fwd_1d",
    "fwd_1d_volume_ratio",
    "fwd_20d",
    "fwd_60d",
    "max_runup_3d",
    "max_drawdown_3d",
    "max_drawdown_60d",
    "max_runup_60d",
)


@dataclass(frozen=True)
class SimilarPatternConfig:
    cache_schema_version: int = 5
    lookback_days: int = 60
    weekly_lookback: int = 26
    monthly_lookback: int = 12
    volume_price_interaction_days: int = 60
    min_history_days: int = 260
    forward_days: tuple[int, ...] = (1, 20, 60)
    max_candidates_per_symbol: int = 120
    candidate_step_days: int = 5
    candidate_start_date: str | None = None
    similarity_threshold: float | None = None
    take_profit_3d: float = 0.03
    stop_loss_3d: float = 0.03
    top_k: int = 80
    min_candidate_rows: int = 500
    signal_bearish_max: float = 45.0
    signal_bullish_min: float = 55.0
    min_effective_cases: int = 100
    max_effective_cases: int = 800
    max_events_per_date: int = 3
    similarity_weight_power: float = 2.0
    same_industry_weight: float = 1.25
    cross_industry_weight: float = 0.85
    same_regime_weight: float = 1.15
    regime_mismatch_weight: float = 0.75
    same_industry_regime_weight: float = 1.10
    industry_regime_mismatch_weight: float = 0.80
    recency_half_life_days: int = 1095
    transaction_cost: float = 0.001
    enable_risk_gate: bool = True


@dataclass(frozen=True)
class TargetSpec:
    symbol: str
    name: str
    target_date: pd.Timestamp


@dataclass(frozen=True)
class SimilarPatternResult:
    target: TargetSpec
    latest_snapshot: dict[str, float | str | None]
    similar_cases: pd.DataFrame
    forecast: pd.DataFrame
    status_probs: dict[str, float]
    t1_scenario_plan: pd.DataFrame | None = None
    sell_model_plan: pd.DataFrame | None = None
    sell_model_summary: dict[str, int | float | str | None] | None = None
    match_mode: str = "top_k"
    scan_summary: dict[str, int | float | str | None] | None = None


def normalize_daily_frame(frame: pd.DataFrame, symbol_hint: str | None = None) -> pd.DataFrame:
    """Normalize local Tushare/cache daily frames into ascending OHLCV rows."""
    if frame.empty:
        return frame
    return _continuous_ohlc_from_pct_change(_normalize_daily_source(frame, symbol_hint))


def _normalize_daily_source(frame: pd.DataFrame, symbol_hint: str | None = None) -> pd.DataFrame:
    """Canonical rows before the latest-close-dependent price adjustment."""
    if frame.empty:
        return frame
    out = frame.copy()
    if "trade_date" in out.columns:
        trade_dates = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        if "date" in out.columns:
            date_values = pd.to_datetime(out["date"], errors="coerce")
            out["date"] = date_values.fillna(trade_dates)
        else:
            out["date"] = trade_dates
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        raise ValueError("daily frame requires date or trade_date")

    if "volume" in out.columns and "vol" in out.columns:
        out["volume"] = out["volume"].fillna(out["vol"])
    elif "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    elif "volume" not in out.columns:
        out["volume"] = np.nan
    if "pct_change" not in out.columns:
        if "pct_chg" in out.columns:
            out["pct_change"] = out["pct_chg"]
        else:
            out["pct_change"] = out["close"].pct_change() * 100
    if "symbol" not in out.columns:
        out["symbol"] = symbol_hint or out.get("ts_code", pd.Series([""])).iloc[0]
    if "name" not in out.columns:
        out["name"] = ""

    cols = ["date", "symbol", "name", "open", "high", "low", "close", "volume", "amount", "pct_change"]
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    out = out[cols].dropna(subset=["date", "open", "high", "low", "close"]).copy()
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    out["symbol"] = out["symbol"].fillna(symbol_hint or "").astype(str)
    return out


def _continuous_ohlc_from_pct_change(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove ex-right jumps using Tushare's continuous pct_change series."""
    out = frame.copy()
    pct_change = pd.to_numeric(out["pct_change"], errors="coerce") / 100.0
    raw_close = pd.to_numeric(out["close"], errors="coerce")
    if len(out) < 2 or pct_change.notna().sum() < 2 or raw_close.iloc[-1] <= 0:
        return out
    growth = 1.0 + pct_change
    raw_growth = raw_close.pct_change() + 1.0
    growth = growth.where(growth.gt(0) & np.isfinite(growth), raw_growth)
    growth.iloc[0] = 1.0
    continuous = growth.fillna(1.0).cumprod()
    continuous = continuous / continuous.iloc[-1] * raw_close.iloc[-1]
    scale = continuous / raw_close.replace(0, np.nan)
    for column in ["open", "high", "low", "close"]:
        out[column] = pd.to_numeric(out[column], errors="coerce") * scale
    return out


def load_stock_basic(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ts_code", "name", "industry", "list_date"])
    basic = pd.read_parquet(path).copy()
    return basic[["ts_code", "name", "industry", "list_date"]].drop_duplicates("ts_code")


def load_daily_file(path: Path) -> pd.DataFrame:
    return normalize_daily_frame(read_partitioned_symbol_file(path), path.stem)


def vector_cache_key(config: SimilarPatternConfig) -> str:
    payload = {
        "cache_schema_version": config.cache_schema_version,
        "lookback_days": config.lookback_days,
        "weekly_lookback": config.weekly_lookback,
        "monthly_lookback": config.monthly_lookback,
        "volume_price_interaction_days": config.volume_price_interaction_days,
        "min_history_days": config.min_history_days,
        "forward_days": config.forward_days,
        "candidate_step_days": config.candidate_step_days,
        "candidate_start_date": config.candidate_start_date,
        "max_candidates_per_symbol": config.max_candidates_per_symbol,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def vector_cache_path(cache_dir: Path, symbol: str, config: SimilarPatternConfig) -> Path:
    safe_symbol = symbol.replace(".", "_")
    return cache_dir / vector_cache_key(config) / f"{safe_symbol}.npz"


class VectorCachePendingError(RuntimeError):
    """The library needs complete repair before any cache-backed read is safe."""


def assert_vector_cache_ready(config_dir: Path) -> None:
    """Fail closed on any pending marker; callers also hold the config read lock."""
    try:
        (config_dir / VECTOR_CACHE_PENDING_FILENAME).lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise VectorCachePendingError("vector publication state is unreadable") from exc
    raise VectorCachePendingError(f"vector publication pending; repair required: {config_dir}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_vector_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _clear_vector_pending(config_dir: Path) -> None:
    (config_dir / VECTOR_CACHE_PENDING_FILENAME).unlink()
    _fsync_directory(config_dir)


def _publication_symbols(config_dir: Path, config: SimilarPatternConfig) -> set[str]:
    """Recover the complete repair scope, including symbols deleted at the source."""
    symbols: set[str] = set()
    for filename in (VECTOR_CACHE_PENDING_FILENAME, _VECTOR_CACHE_COMMIT_FILENAME):
        path = config_dir / filename
        try:
            if path.is_symlink():
                raise ValueError("symlink publication state")
            payload = json.loads(path.read_text("utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            raise VectorCachePendingError(f"invalid publication state: {path}") from exc
        if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                or payload.get("config_key") != vector_cache_key(config)
                or not isinstance(payload.get("expected_symbols"), list)):
            raise VectorCachePendingError(f"invalid publication scope: {path}")
        for symbol in payload["expected_symbols"]:
            if (not isinstance(symbol, str) or not symbol or symbol in {".", ".."}
                    or "/" in symbol or "\\" in symbol):
                raise VectorCachePendingError(f"invalid publication symbol: {path}")
            symbols.add(symbol)
    for path in config_dir.glob("*.npz"):
        # This is the reversible A-share cache filename convention. Never read a
        # partial NPZ merely to discover which source must repair it.
        symbol = path.stem.replace("_", ".")
        if vector_cache_path(config_dir.parent, symbol, config).name != path.name:
            raise VectorCachePendingError(f"unrecognized vector cache filename: {path}")
        symbols.add(symbol)
    return symbols


@dataclass(frozen=True)
class _VectorCacheEntry:
    symbol: str
    path: Path
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class _CompiledVectorCache:
    generation_dir: Path
    manifest: dict[str, object]


def _vector_cache_inventory(
    daily_paths: list[Path],
    cache_dir: Path,
    config: SimilarPatternConfig,
) -> list[_VectorCacheEntry]:
    """Resolve the exact legacy cache inputs represented by a matrix generation."""
    assert_vector_cache_ready(cache_dir / vector_cache_key(config))
    entries: list[_VectorCacheEntry] = []
    for daily_path in daily_paths:
        symbol = daily_path.stem
        path = vector_cache_path(cache_dir, symbol, config)
        if not path.exists():
            continue
        stat = path.stat()
        entries.append(
            _VectorCacheEntry(
                symbol=symbol,
                path=path,
                size=int(stat.st_size),
                mtime_ns=int(stat.st_mtime_ns),
            )
        )
    assert_vector_cache_ready(cache_dir / vector_cache_key(config))
    return entries


def _vector_cache_inventory_fingerprint(
    entries: list[_VectorCacheEntry],
    config: SimilarPatternConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"matrix-schema:{_MATRIX_CACHE_SCHEMA_VERSION}".encode("ascii"))
    digest.update(f"vector-config:{vector_cache_key(config)}".encode("ascii"))
    cache_directories = sorted({entry.path.parent for entry in entries})
    for cache_directory in cache_directories:
        metadata_path = cache_directory / _VECTOR_CACHE_SOURCE_METADATA_FILENAME
        if metadata_path.exists():
            digest.update(metadata_path.read_bytes())
    for entry in entries:
        digest.update(entry.symbol.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry.path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry.size).encode("ascii"))
        digest.update(b":")
        digest.update(str(entry.mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@contextmanager
def _matrix_cache_lock(cache_root: Path, *, exclusive: bool) -> Iterator[None]:
    """Serialize generation replacement while allowing concurrent read scans."""
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / ".lock"
    with lock_path.open("a+b") as handle:
        import fcntl

        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(handle.fileno(), mode)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _compiled_manifest_is_valid(
    generation_dir: Path,
    *,
    inventory_fingerprint: str,
    config: SimilarPatternConfig,
) -> dict[str, object] | None:
    manifest_path = generation_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if manifest.get("schema_version") != _MATRIX_CACHE_SCHEMA_VERSION:
        return None
    if manifest.get("config_key") != vector_cache_key(config):
        return None
    if manifest.get("inventory_fingerprint") != inventory_fingerprint:
        return None
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return None
    for chunk in chunks:
        if not isinstance(chunk, dict):
            return None
        relative = chunk.get("path")
        if not isinstance(relative, str):
            return None
        chunk_dir = generation_dir / relative
        required_files = (
            "vectors.npy",
            "indices.npy",
            "dates.npy",
            "segments.json",
            *(f"{field}.npy" for field in _MATRIX_CACHE_FLOAT_FIELDS),
        )
        if any(not (chunk_dir / filename).exists() for filename in required_files):
            return None
    return manifest


def _save_compiled_chunk(
    chunk_dir: Path,
    entries: list[_VectorCacheEntry],
) -> tuple[dict[str, object], list[str]]:
    cached_stocks: list[dict[str, object]] = []
    vector_dim: int | None = None
    source_fingerprints: list[str] = []
    for entry in entries:
        cached = load_stock_vector_cache(entry.path)
        vectors = np.asarray(cached["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            continue
        if vector_dim is None:
            vector_dim = int(vectors.shape[1])
        elif int(vectors.shape[1]) != vector_dim:
            raise ValueError(
                f"vector dimension mismatch for {entry.symbol}: "
                f"expected={vector_dim} actual={vectors.shape[1]}"
            )
        row_count = int(vectors.shape[0])
        if len(cached["indices"]) != row_count:
            raise ValueError(f"cache row mismatch for {entry.symbol}")
        cached_stocks.append(cached)
        source_fingerprints.append(
            f"{entry.symbol}:{cached.get('source_fingerprint') or 'legacy'}"
        )
    if not cached_stocks or vector_dim is None:
        raise ValueError("compiled vector chunk has no usable rows")

    chunk_dir.mkdir(parents=True, exist_ok=False)
    segments: list[dict[str, object]] = []
    offset = 0
    for cached in cached_stocks:
        row_count = int(np.asarray(cached["vectors"]).shape[0])
        segments.append(
            {
                "symbol": str(cached["symbol"]),
                "name": str(cached["name"]),
                "industry": str(cached["industry"]),
                "start": offset,
                "stop": offset + row_count,
            }
        )
        offset += row_count

    np.save(
        chunk_dir / "vectors.npy",
        np.concatenate(
            [np.asarray(cached["vectors"], dtype=np.float32) for cached in cached_stocks],
            axis=0,
        ),
        allow_pickle=False,
    )
    np.save(
        chunk_dir / "indices.npy",
        np.concatenate(
            [np.asarray(cached["indices"], dtype=np.int32) for cached in cached_stocks]
        ),
        allow_pickle=False,
    )
    np.save(
        chunk_dir / "dates.npy",
        np.concatenate(
            [np.asarray(cached["dates"], dtype="U10") for cached in cached_stocks]
        ),
        allow_pickle=False,
    )
    for field in _MATRIX_CACHE_FLOAT_FIELDS:
        np.save(
            chunk_dir / f"{field}.npy",
            np.concatenate(
                [np.asarray(cached[field], dtype=np.float32) for cached in cached_stocks]
            ),
            allow_pickle=False,
        )
    (chunk_dir / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return (
        {
            "path": chunk_dir.name,
            "rows": offset,
            "symbols": len(segments),
            "vector_dim": vector_dim,
        },
        source_fingerprints,
    )


def _iter_vector_cache_chunks(
    entries: list[_VectorCacheEntry],
) -> Iterator[list[_VectorCacheEntry]]:
    chunk: list[_VectorCacheEntry] = []
    chunk_bytes = 0
    for entry in entries:
        exceeds_symbol_limit = len(chunk) >= _MATRIX_CACHE_CHUNK_SYMBOLS
        exceeds_byte_limit = (
            bool(chunk)
            and chunk_bytes + entry.size > _MATRIX_CACHE_CHUNK_MAX_SOURCE_BYTES
        )
        if exceeds_symbol_limit or exceeds_byte_limit:
            yield chunk
            chunk = []
            chunk_bytes = 0
        chunk.append(entry)
        chunk_bytes += entry.size
    if chunk:
        yield chunk


def _build_compiled_vector_cache(
    cache_root: Path,
    generation_dir: Path,
    entries: list[_VectorCacheEntry],
    config: SimilarPatternConfig,
    inventory_fingerprint: str,
) -> dict[str, object]:
    temp_dir = cache_root / f".{generation_dir.name}.building-{os.getpid()}-{uuid4().hex[:8]}"
    registry = _vector_artifact_registry(cache_root.parent.parent)
    registry.register(temp_dir, producer="similar_patterns", retention_class="temporary",
                      state="building", input_versions={"inventory": inventory_fingerprint},
                      ownership_boundary=True)
    temp_dir.mkdir(parents=True, exist_ok=False)
    chunks: list[dict[str, object]] = []
    source_fingerprints: list[str] = []
    vector_dim: int | None = None
    try:
        for chunk_number, chunk_entries in enumerate(_iter_vector_cache_chunks(entries)):
            chunk, chunk_sources = _save_compiled_chunk(
                temp_dir / f"chunk_{chunk_number:05d}",
                chunk_entries,
            )
            if vector_dim is None:
                vector_dim = int(chunk["vector_dim"])
            elif int(chunk["vector_dim"]) != vector_dim:
                raise ValueError("compiled vector chunks use inconsistent dimensions")
            chunks.append(chunk)
            source_fingerprints.extend(chunk_sources)
        if not chunks or vector_dim is None:
            raise ValueError("no usable legacy vector caches are available to compile")
        source_digest = hashlib.sha256("\n".join(source_fingerprints).encode("utf-8")).hexdigest()
        manifest: dict[str, object] = {
            "schema_version": _MATRIX_CACHE_SCHEMA_VERSION,
            "config_key": vector_cache_key(config),
            "inventory_fingerprint": inventory_fingerprint,
            "source_fingerprint_digest": source_digest,
            "inventory_entries": len(entries),
            "symbols": int(sum(int(chunk["symbols"]) for chunk in chunks)),
            "rows": int(sum(int(chunk["rows"]) for chunk in chunks)),
            "vector_dim": vector_dim,
            "chunks": chunks,
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if generation_dir.exists():
            shutil.rmtree(generation_dir)
        temp_dir.replace(generation_dir)
        registry.retire(temp_dir)
        return manifest
    except Exception:
        registry.retire(temp_dir)
        raise


def _ensure_compiled_vector_cache(
    entries: list[_VectorCacheEntry],
    cache_dir: Path,
    config: SimilarPatternConfig,
) -> _CompiledVectorCache:
    if not entries:
        raise ValueError("no legacy vector caches are available")
    config_dir = cache_dir.resolve() / vector_cache_key(config)
    registry = _vector_artifact_registry(cache_dir)
    cache_root = config_dir / _MATRIX_CACHE_DIRNAME
    with registry.lease([config_dir], owner="similar_patterns:compile", kind="build"), _matrix_cache_lock(config_dir, exclusive=False), _matrix_cache_lock(cache_root, exclusive=True):
        assert_vector_cache_ready(config_dir)
        entries = [_VectorCacheEntry(entry.symbol, entry.path, stat.st_size, stat.st_mtime_ns)
                   for entry in entries if entry.path.exists() for stat in [entry.path.stat()]]
        inventory_fingerprint = _vector_cache_inventory_fingerprint(entries, config)
        generation_dir = cache_root / inventory_fingerprint[:24]
        for artifact in registry.inventory([cache_root])["entries"]:
            abandoned = registry.root / artifact["path"]
            if (abandoned.parent == cache_root and ".building-" in abandoned.name
                    and artifact.get("producer") == "similar_patterns"
                    and artifact.get("ownership_boundary") and artifact.get("state") == "building"):
                registry.retire(abandoned)
        manifest = _compiled_manifest_is_valid(
            generation_dir,
            inventory_fingerprint=inventory_fingerprint,
            config=config,
        )
        if manifest is None:
            registry.register(generation_dir, producer="similar_patterns", retention_class="rebuildable",
                              state="building", input_versions={"inventory": inventory_fingerprint},
                              ownership_boundary=True)
            manifest = _build_compiled_vector_cache(
                cache_root,
                generation_dir,
                entries,
                config,
                inventory_fingerprint,
            )
        registry.register(generation_dir, producer="similar_patterns", retention_class="rebuildable",
                          state="committed", input_versions={"inventory": inventory_fingerprint},
                          ownership_boundary=True)
        consumer = f"similar_patterns:{vector_cache_key(config)}:compiled"
        registry.commit(consumer, generation_dir)
        # Reconcile all superseded owned outputs, including a commit interrupted
        # before retirement. The matrix writer lock excludes other compilers;
        # current/previous references and reader leases still prevent collection.
        for artifact in registry.inventory([cache_root])["entries"]:
            previous = registry.root / artifact["path"]
            if (previous.parent == cache_root and previous != generation_dir
                    and artifact.get("producer") == "similar_patterns"
                    and artifact.get("ownership_boundary")
                    and artifact.get("retention_class") in {"rebuildable", "temporary"}
                    and not artifact.get("protected")):
                registry.retire(previous)
    _collect_compiled_vector_caches(cache_dir, config)
    return _CompiledVectorCache(generation_dir=generation_dir, manifest=manifest)


def _collect_compiled_vector_caches(cache_dir: Path, config: SimilarPatternConfig) -> None:
    registry = _vector_artifact_registry(cache_dir)
    root = cache_dir.resolve() / vector_cache_key(config) / _MATRIX_CACHE_DIRNAME
    if root.exists():
        registry.collect_retired_children(root, dry_run=False)


def _semantic_daily_frame_fingerprint(frame: pd.DataFrame) -> str | None:
    """Hash the source values that can change a generated pattern vector."""
    if frame.empty:
        return None
    columns = [column for column in _VECTOR_SOURCE_COLUMNS if column in frame.columns]
    if not {"ts_code", "trade_date", "open", "high", "low", "close"}.issubset(columns):
        return None

    normalized = frame[columns].copy()
    normalized["ts_code"] = normalized["ts_code"].fillna("").astype(str)
    normalized["trade_date"] = pd.to_datetime(
        normalized["trade_date"].astype(str).str.replace("-", "", regex=False),
        format="%Y%m%d",
        errors="coerce",
    ).dt.strftime("%Y%m%d")
    if "name" in normalized.columns:
        normalized["name"] = normalized["name"].fillna("").astype(str)
    for column in columns:
        if column not in {"ts_code", "trade_date", "name"}:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")
    normalized = (
        normalized.dropna(subset=["trade_date"])
        .sort_values(["ts_code", "trade_date"], kind="mergesort")
        .drop_duplicates(["ts_code", "trade_date"], keep="last")
        .reset_index(drop=True)
    )
    if normalized.empty:
        return None

    digest = hashlib.sha256()
    digest.update("\0".join(columns).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(normalized, index=False, categorize=True)
        .to_numpy(dtype="uint64", copy=False)
        .tobytes()
    )
    latest = str(normalized["trade_date"].max())
    return f"{latest}:{len(normalized)}:{digest.hexdigest()}"


def _read_sql_vector_source(
    store: MarketDataStore, dataset: str, *, symbols: list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read only columns relevant to vector generation from the SQL canonical source."""
    if columns is None:
        columns = _sql_vector_columns(store, dataset)
    from quant.data.market_snapshot import current_market_snapshot

    snapshot = current_market_snapshot()
    if snapshot is not None:
        return snapshot.read(dataset, symbols=symbols, columns=columns)
    # Successful empty canonical reads are not permission to use an older mirror.
    return store._read_sql_range(dataset, symbols=symbols, columns=columns)


def _sql_vector_columns(store: MarketDataStore, dataset: str) -> list[str] | None:
    from quant.data.market_snapshot import current_market_snapshot

    snapshot = current_market_snapshot()
    if snapshot is not None:
        available = snapshot.manifest["datasets"][dataset]["columns"]
        return [column for column in _VECTOR_SOURCE_COLUMNS if column in available]
    try:
        from sqlalchemy import inspect

        available = {str(column["name"]) for column in inspect(store._engine()).get_columns(
            store._dataset_table_name(dataset)
        )}
        return [column for column in _VECTOR_SOURCE_COLUMNS if column in available]
    except Exception:
        # Unknown optional columns are not safe to project. The canonical reader
        # still fails closed on source errors, even with an unprojected batch.
        return None


def _partition_content_digest(path: Path) -> str:
    """Hash one parquet artifact, reusing unchanged partition digests in-process."""
    source_stat = path.stat()
    key = (
        str(path.resolve()),
        int(source_stat.st_size),
        int(source_stat.st_mtime_ns),
        int(source_stat.st_ctime_ns),
    )
    cached = _PARTITION_CONTENT_DIGEST_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if len(_PARTITION_CONTENT_DIGEST_CACHE) >= _PARTITION_CONTENT_DIGEST_CACHE_MAX_ENTRIES:
        _PARTITION_CONTENT_DIGEST_CACHE.clear()
    _PARTITION_CONTENT_DIGEST_CACHE[key] = value
    return value


def _parquet_daily_source_fingerprint(daily_dir: Path) -> str | None:
    partition_root = daily_dir.parent / f"{daily_dir.name}_partitioned"
    partition_paths = sorted(partition_root.glob("year_month=*/data.parquet"))
    if not partition_paths:
        return None
    digest = hashlib.sha256()
    for path in partition_paths:
        digest.update(path.parent.name.encode("utf-8"))
        digest.update(b":")
        digest.update(_partition_content_digest(path).encode("ascii"))
        digest.update(b"\n")
    return f"partitioned:{digest.hexdigest()}"


def partitioned_daily_source_fingerprint(daily_dir: Path) -> str | None:
    """Compatibility diagnostic; vector cache validity uses symbol identities instead."""
    config = MarketDataStoreConfig.from_env(root=daily_dir.parent)
    store = MarketDataStore(config)
    from quant.data.market_snapshot import current_market_snapshot

    if current_market_snapshot() is not None or config.backend in {"mysql", "sql"}:
        sql_frame = _read_sql_vector_source(store, daily_dir.name)
        sql_identity = _semantic_daily_frame_fingerprint(sql_frame)
        if sql_identity is not None:
            return f"sql-semantic:{sql_identity}"
        return None
    return _parquet_daily_source_fingerprint(daily_dir)


class VectorSourceChangedError(RuntimeError):
    """A staged vector build no longer represents a single source revision."""


def _source_rows_fingerprint(source: pd.DataFrame) -> str:
    columns = [column for column in source.columns if column != "amount"]
    digest = hashlib.sha256("\0".join(columns).encode("utf-8"))
    canonical = source[columns].copy()
    for column in ("open", "high", "low", "close", "volume", "pct_change"):
        canonical[column] = pd.to_numeric(canonical[column], errors="coerce").astype("float64")
    for column in ("symbol", "name"):
        canonical[column] = canonical[column].fillna("").astype(str)
    digest.update(pd.util.hash_pandas_object(canonical, index=False).to_numpy("uint64").tobytes())
    return digest.hexdigest()


class _VectorSource:
    """Bounded projected reads; identities never depend on another symbol's values."""

    def __init__(self, daily_dir: Path):
        from quant.data.market_snapshot import current_market_snapshot

        self.daily_dir = daily_dir
        self.store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
        self.pinned = current_market_snapshot() is not None
        self.sql = self.store.config.backend in {"mysql", "sql"}
        self.columns = _sql_vector_columns(self.store, daily_dir.name) if self.pinned or self.sql else None

    def token(self) -> object:
        if self.pinned:
            from quant.data.market_snapshot import current_market_snapshot

            return current_market_snapshot().manifest["fingerprint"]
        if self.sql:
            # The revision is a race guard, never a per-symbol cache key.
            return self.store.dataset_revision(self.daily_dir.name)
        partition_root = self.daily_dir.parent / f"{self.daily_dir.name}_partitioned"
        paths = sorted(partition_root.glob("year_month=*/data.parquet"))
        if not paths:
            paths = sorted(self.daily_dir.glob("*.parquet"))
        return tuple((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
                     for path in paths for stat in [path.stat()])

    def assert_token(self, expected: object) -> None:
        if self.token() != expected:
            raise VectorSourceChangedError("daily source revision changed during vector construction")

    def read(self, paths: list[Path]) -> dict[str, pd.DataFrame]:
        symbols = [path.stem for path in paths]
        if self.pinned or self.sql:
            frame = _read_sql_vector_source(self.store, self.daily_dir.name,
                                            symbols=symbols, columns=self.columns)
            groups = dict(tuple(frame.groupby("ts_code", sort=False))) if not frame.empty else {}
            return {symbol: groups.get(symbol, pd.DataFrame()) for symbol in symbols}
        import pyarrow.parquet as pq

        partitions = sorted((self.daily_dir.parent / f"{self.daily_dir.name}_partitioned").glob(
            "year_month=*/data.parquet"
        ))
        if not partitions:
            return {path.stem: pd.read_parquet(path, columns=[
                column for column in _VECTOR_SOURCE_COLUMNS if column in pq.read_schema(path).names
            ]) if path.exists() else pd.DataFrame() for path in paths}
        frames = []
        for path in partitions:
            columns = [column for column in _VECTOR_SOURCE_COLUMNS if column in pq.read_schema(path).names]
            frame = pd.read_parquet(path, columns=columns, filters=[("ts_code", "in", symbols)])
            if not frame.empty:
                frames.append(frame)
        # Preserve the store's date ordering and duplicate precedence before
        # normalization derives returns for sources without pct_change columns.
        combined = self.store._merge_frames(frames)
        groups = dict(tuple(combined.groupby("ts_code", sort=False))) if not combined.empty else {}
        return {symbol: groups.get(symbol, pd.DataFrame()) for symbol in symbols}


def save_stock_vector_cache(
    cache_path: Path,
    symbol: str,
    name: str,
    industry: str,
    daily: pd.DataFrame,
    indices: list[int],
    matrix: np.ndarray,
    config: SimilarPatternConfig,
    source_mtime_ns: int,
    source_size: int,
    source_fingerprint: str,
    *,
    source: pd.DataFrame | None = None,
    previous: dict[str, object] | None = None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dates = np.array([pd.Timestamp(daily.iloc[idx]["date"]).strftime("%Y-%m-%d") for idx in indices])
    close = daily["close"].to_numpy(dtype=float)
    old_positions = {int(idx): pos for pos, idx in enumerate(previous["indices"])} if previous else {}
    old_rows = int(previous["source_rows"]) if previous else 0
    horizons = {f"fwd_{days}d": days for days in config.forward_days}
    horizons.update({"fwd_1d_volume_ratio": 1, "max_runup_3d": 3, "max_drawdown_3d": 3,
                     "max_runup_60d": max(config.forward_days), "max_drawdown_60d": max(config.forward_days)})
    # Only new candidates and previously immature horizons need label evaluation.
    dirty = [idx for idx in indices if idx not in old_positions or idx + max(3, max(config.forward_days)) >= old_rows]
    updated = _forward_labels(daily, dirty, config)
    dirty_positions = {idx: pos for pos, idx in enumerate(dirty)}
    future = {}
    for field, horizon in horizons.items():
        future[field] = np.asarray([
            previous[field][old_positions[idx]]
            if idx in old_positions and idx + horizon < old_rows
            else updated[field][dirty_positions[idx]] for idx in indices
        ], dtype=np.float32)
    incremental = {}
    if source is not None:
        incremental = {
            "incremental_schema": np.array(_VECTOR_INCREMENTAL_SCHEMA),
            "source_rows": np.array(len(source)),
            "source_prefix_fingerprint": np.array(_source_rows_fingerprint(source)),
            "source_adjusted": np.array(_source_is_adjusted(source)),
        }
    # A reader must never observe a partially written npz, even for standalone builds.
    temp_path = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("wb") as handle:
            np.savez(
                handle,
                symbol=np.array(symbol), name=np.array(name), industry=np.array(industry),
                indices=np.array(indices, dtype=np.int32), dates=dates,
                close=np.array([close[idx] for idx in indices], dtype=np.float32),
                vectors=matrix.astype(np.float32),
                source_mtime_ns=np.array(source_mtime_ns, dtype=np.int64),
                source_size=np.array(source_size, dtype=np.int64),
                source_fingerprint=np.array(source_fingerprint),
                **incremental, **future,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(cache_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _source_is_adjusted(source: pd.DataFrame) -> bool:
    return (len(source) >= 2 and pd.to_numeric(source["pct_change"], errors="coerce").notna().sum() >= 2
            and pd.to_numeric(source["close"], errors="coerce").iloc[-1] > 0)


def _forward_labels(daily: pd.DataFrame, indices: list[int], config: SimilarPatternConfig) -> dict[str, np.ndarray]:
    close = daily["close"].to_numpy(dtype=float)
    high = daily["high"].to_numpy(dtype=float)
    low = daily["low"].to_numpy(dtype=float)
    volume = daily["volume"].replace(0, np.nan).to_numpy(dtype=float)
    volume_ma20 = pd.Series(volume).rolling(20).mean().to_numpy(dtype=float)
    future: dict[str, np.ndarray] = {}
    for days in config.forward_days:
        future[f"fwd_{days}d"] = np.array(
            [close[idx + days] / close[idx] - 1 if idx + days < len(close) else np.nan for idx in indices],
            dtype=np.float32,
        )
    max_forward = max(config.forward_days)
    future["fwd_1d_volume_ratio"] = np.array(
        [
            volume[idx + 1] / volume_ma20[idx]
            if idx + 1 < len(volume) and np.isfinite(volume[idx + 1]) and np.isfinite(volume_ma20[idx]) and volume_ma20[idx] > 0
            else np.nan
            for idx in indices
        ],
        dtype=np.float32,
    )
    future["max_runup_3d"] = np.array(
        [np.nanmax(high[idx + 1 : idx + 4]) / close[idx] - 1 if idx + 3 < len(high) else np.nan for idx in indices],
        dtype=np.float32,
    )
    future["max_drawdown_3d"] = np.array(
        [np.nanmin(low[idx + 1 : idx + 4]) / close[idx] - 1 if idx + 3 < len(low) else np.nan for idx in indices],
        dtype=np.float32,
    )
    future["max_drawdown_60d"] = np.array(
        [
            np.nanmin(close[idx + 1 : idx + max_forward + 1]) / close[idx] - 1
            if idx + max_forward < len(close)
            else np.nan
            for idx in indices
        ],
        dtype=np.float32,
    )
    future["max_runup_60d"] = np.array(
        [
            np.nanmax(close[idx + 1 : idx + max_forward + 1]) / close[idx] - 1
            if idx + max_forward < len(close)
            else np.nan
            for idx in indices
        ],
        dtype=np.float32,
    )
    return future


def load_stock_vector_cache(cache_path: Path) -> dict[str, object]:
    config_dir = cache_path.absolute().parent
    registry = _vector_artifact_registry(config_dir.parent)
    with registry.lease([config_dir], owner="similar_patterns:read", kind="read"), _matrix_cache_lock(config_dir, exclusive=False):
        assert_vector_cache_ready(config_dir)
        return _load_stock_vector_cache(cache_path)


def _load_stock_vector_cache(cache_path: Path) -> dict[str, object]:
    """Unchecked I/O for workers/repair already protected by the owning build lease."""
    with np.load(cache_path, allow_pickle=False) as data:
        cached = {
            "symbol": str(data["symbol"].item()),
            "name": str(data["name"].item()),
            "industry": str(data["industry"].item()),
            "indices": data["indices"],
            "dates": data["dates"],
            "close": data["close"],
            "vectors": data["vectors"],
            "fwd_1d": data["fwd_1d"],
            "fwd_1d_volume_ratio": data["fwd_1d_volume_ratio"],
            "fwd_20d": data["fwd_20d"],
            "fwd_60d": data["fwd_60d"],
            "max_runup_3d": data["max_runup_3d"],
            "max_drawdown_3d": data["max_drawdown_3d"],
            "max_drawdown_60d": data["max_drawdown_60d"],
            "max_runup_60d": data["max_runup_60d"],
            "source_mtime_ns": int(data["source_mtime_ns"].item())
            if "source_mtime_ns" in data.files
            else None,
            "source_size": int(data["source_size"].item())
            if "source_size" in data.files
            else None,
            "source_fingerprint": str(data["source_fingerprint"].item())
            if "source_fingerprint" in data.files
            else None,
        }
        for field in ("incremental_schema", "source_rows", "source_prefix_fingerprint", "source_adjusted"):
            if field in data.files:
                cached[field] = data[field].item()
        # Preserve additional configured forward horizons as well as the legacy fields.
        for field in data.files:
            if field.startswith("fwd_"):
                cached[field] = data[field]
        return cached


def _stock_cache_shape_valid(cached: dict[str, object], config: SimilarPatternConfig) -> bool:
    indices = np.asarray(cached["indices"])
    vectors = np.asarray(cached["vectors"])
    dimension = config.lookback_days * 4 + config.weekly_lookback + config.monthly_lookback + 12
    return (
        indices.ndim == 1 and len(indices) > 0 and np.all(np.diff(indices) > 0)
        and vectors.shape == (len(indices), dimension) and np.isfinite(vectors).all()
        and all(np.asarray(cached[field]).shape == indices.shape
                for field in ("dates", *_MATRIX_CACHE_FLOAT_FIELDS,
                              *(f"fwd_{days}d" for days in config.forward_days)))
    )


def build_stock_vector_cache(
    path: Path,
    info: dict[str, object],
    config: SimilarPatternConfig,
    cache_dir: Path,
    force: bool = False,
    source_fingerprint: str | None = None,
) -> dict[str, object]:
    """Build one symbol using the same staged, revision-checked path as batch builds.

    ``source_fingerprint`` remains accepted for caller compatibility, but a global
    fingerprint is deliberately not trusted as a symbol identity.
    """
    records = _build_vector_cache_files([path], {path.stem: info}, config, cache_dir, force, 1, None)
    return records.iloc[0].to_dict()


def _build_stock_vector_cache_worker(
    args: tuple[Path, dict[str, object], SimilarPatternConfig, Path, bool, pd.DataFrame, Path],
) -> dict[str, object]:
    path, info, config, cache_dir, force, raw, staging = args
    symbol = path.stem
    cache_path = vector_cache_path(cache_dir, symbol, config)
    started = perf_counter()
    record: dict[str, object] = {
        "symbol": symbol, "cache_path": str(cache_path), "vectors": 0,
        "vectors_built": 0, "vectors_reused": 0, "source_rows_read": len(raw),
        "vector_windows_evaluated": 0,
        "elapsed_sec": 0.0,
    }
    try:
        source = _normalize_daily_source(raw, symbol)
        daily = _continuous_ohlc_from_pct_change(source) if not source.empty else source
        if len(daily) < config.min_history_days + min(config.forward_days) + 1:
            return {**record, "status": "too_short"}
        industry = str(info.get("industry", ""))
        if info.get("name"):
            daily["name"] = daily["name"].replace("", np.nan).fillna(str(info["name"]))
        name = stock_name_from_basic_or_daily(info, daily)
        if is_excluded_stock(name):
            return {**record, "status": "excluded"}
        identity = "symbol-semantic:" + _source_rows_fingerprint(source)
        record["source_fingerprint"] = identity
        cached = None
        if cache_path.exists() and not force:
            try:
                cached = _load_stock_vector_cache(cache_path)
                if not _stock_cache_shape_valid(cached, config):
                    cached = None
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                cached = None
        if (cached is not None and cached.get("source_fingerprint") == identity
                and cached["name"] == name and cached["industry"] == industry):
            return {**record, "status": "cache_hit", "vectors": len(cached["indices"]),
                    "vectors_reused": len(cached["indices"]), "build_mode": "unchanged"}
        previous = None
        if cached is not None and cached.get("incremental_schema") == _VECTOR_INCREMENTAL_SCHEMA:
            old_rows = int(cached["source_rows"])
            if (0 < old_rows <= len(source)
                    and cached["source_prefix_fingerprint"] == _source_rows_fingerprint(source.iloc[:old_rows])
                    and cached["source_adjusted"] == _source_is_adjusted(source)):
                previous = cached
        weekly_close, monthly_close = resample_close_series(daily)
        old_positions = {int(idx): pos for pos, idx in enumerate(previous["indices"])} if previous else {}
        evaluated = candidate_end_indices(daily.iloc[:int(previous["source_rows"])], config) if previous else range(0)
        indices, vectors = [], []
        for idx in candidate_end_indices(daily, config):
            if idx in old_positions:
                vector = previous["vectors"][old_positions[idx]]
                record["vectors_reused"] += 1
            elif idx in evaluated:
                continue
            else:
                vector = build_pattern_vector(daily, idx, config, weekly_close, monthly_close)
                record["vector_windows_evaluated"] += 1
                if vector is not None:
                    record["vectors_built"] += 1
            if vector is not None:
                indices.append(idx)
                vectors.append(vector)
        if len(indices) == 0:
            return {**record, "status": "no_vectors"}
        matrix = np.vstack(vectors).astype(np.float32)
        save_stock_vector_cache(
            staging / cache_path.name,
            symbol,
            name,
            industry,
            daily,
            indices,
            matrix,
            config,
            source_mtime_ns=-1,
            source_size=-1,
            source_fingerprint=identity,
            source=source,
            previous=previous,
        )
        return {
            **record,
            "status": "built",
            "build_mode": "append" if previous else "full",
            "vectors": int(matrix.shape[0]),
            "elapsed_sec": round(perf_counter() - started, 4),
        }
    except Exception as exc:
        return {
            **record,
            "status": "error",
            "error": str(exc),
            "vectors": 0,
            "elapsed_sec": round(perf_counter() - started, 4),
        }


def _should_use_thread_pool_for_vector_cache() -> bool:
    """Avoid nested process pools when the current worker cannot safely spawn children."""
    try:
        return bool(mp.current_process().daemon)
    except Exception:
        return False


def build_vector_caches_parallel(
    daily_dir: Path,
    basic: pd.DataFrame,
    config: SimilarPatternConfig,
    cache_dir: Path,
    target_symbols: set[str] | None = None,
    max_symbols: int | None = None,
    workers: int = 1,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    source = _VectorSource(daily_dir)
    token = source.token()
    files = list_partitioned_symbol_paths(daily_dir)
    source.assert_token(token)
    if max_symbols is not None:
        files = files[:max_symbols]
    target_symbols = {symbol.upper() for symbol in (target_symbols or set())}
    files = [path for path in files if path.stem.upper() not in target_symbols]
    basic_map = basic.set_index("ts_code").to_dict("index") if not basic.empty else {}
    return _build_vector_cache_files(files, basic_map, config, cache_dir, force, workers, progress_callback,
                                     source_snapshot=(source, token))


def _vector_artifact_registry(cache_dir: Path):
    from quant.infrastructure.artifact_registry import ArtifactRegistry

    cache_dir = cache_dir.resolve()
    root = next((parent for parent in (cache_dir, *cache_dir.parents)
                 if (parent / "pyproject.toml").exists()), None)
    if root is None:
        # Standalone research trees need the same registry before and after the
        # cache itself is created; do not select the nearest *existing* ancestor.
        root = cache_dir.parent
        root.mkdir(parents=True, exist_ok=True)
    return ArtifactRegistry(root)


def _build_vector_cache_files(
    files: list[Path], basic_map: dict[str, dict[str, object]], config: SimilarPatternConfig,
    cache_dir: Path, force: bool, workers: int, progress_callback: Callable[[str], None] | None,
    *, source_snapshot: tuple[_VectorSource, object] | None = None,
) -> pd.DataFrame:
    if not files and source_snapshot is None:
        return pd.DataFrame()
    cache_dir = cache_dir.resolve()
    config_dir = cache_dir / vector_cache_key(config)
    registry = _vector_artifact_registry(cache_dir)
    source = source_snapshot[0] if source_snapshot else _VectorSource(files[0].parent)
    records: list[dict[str, object]] = []
    started = perf_counter()
    with registry.lease([config_dir], owner="similar_patterns:build", kind="build"), _matrix_cache_lock(config_dir, exclusive=True):
        try:
            assert_vector_cache_ready(config_dir)
            repairing = False
        except VectorCachePendingError:
            repairing = True
        expected_symbols = _publication_symbols(config_dir, config) | {path.stem for path in files}
        if repairing:
            # A partial retry cannot certify a library left half-published by a
            # dead process. Revalidate every existing and previously expected symbol.
            files = [source.daily_dir / f"{symbol}.parquet" for symbol in sorted(expected_symbols)]
        if not files:
            if repairing:
                raise VectorCachePendingError("pending vector publication has no recoverable scope")
            return pd.DataFrame()
        token = source_snapshot[1] if source_snapshot else source.token()
        source.assert_token(token)
        executor = None
        if workers > 1:
            if _should_use_thread_pool_for_vector_cache():
                executor = ThreadPoolExecutor(max_workers=workers)
            else:
                try:
                    # Source I/O stays in the parent. Never inherit its SQL pool.
                    executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
                except (AssertionError, BrokenPipeError, PermissionError):
                    executor = ThreadPoolExecutor(max_workers=workers)
        batch_size = min(_VECTOR_SOURCE_BATCH_SYMBOLS, max(1, workers) * 2)
        identities: dict[str, str] = {}
        with TemporaryDirectory(prefix=".vectors-building-", dir=config_dir) as temp, executor or nullcontext():
            staging = Path(temp)
            for start in range(0, len(files), batch_size):
                batch = files[start:start + batch_size]
                frames = source.read(batch)
                source.assert_token(token)
                for path in batch:
                    normalized = _normalize_daily_source(frames[path.stem], path.stem)
                    identities[path.stem] = _source_rows_fingerprint(normalized) if not normalized.empty else "empty"
                tasks = [(path, basic_map.get(path.stem, {}), config, cache_dir, force, frames[path.stem], staging)
                         for path in batch]
                if executor is None:
                    records.extend(_build_stock_vector_cache_worker(task) for task in tasks)
                else:
                    futures = [executor.submit(_build_stock_vector_cache_worker, task) for task in tasks]
                    records.extend(future.result() for future in as_completed(futures))
                if progress_callback:
                    progress_callback(f"vector cache {len(records)}/{len(files)} staged elapsed={perf_counter() - started:.1f}s")
            source.assert_token(token)
            if source.sql:
                # Some SQL writers predate the revision journal. Re-read bounded
                # projections, so those corrections cannot silently publish stale hits.
                for start in range(0, len(files), batch_size):
                    batch = files[start:start + batch_size]
                    frames = source.read(batch)
                    for path in batch:
                        normalized = _normalize_daily_source(frames[path.stem], path.stem)
                        identity = _source_rows_fingerprint(normalized) if not normalized.empty else "empty"
                        if identity != identities[path.stem]:
                            raise VectorSourceChangedError(f"daily source content changed for {path.stem}")
                source.assert_token(token)
            if any(record["status"] == "error" for record in records):
                for record in records:
                    if record["status"] == "built":
                        record["status"] = "aborted"
                return pd.DataFrame(records)
            pending = {
                "schema_version": 1, "config_key": vector_cache_key(config),
                "build_id": uuid4().hex, "phase": "repairing" if repairing else "publishing",
                "source_revision": str(token), "expected_symbols": sorted(expected_symbols),
                "affected_symbols": sorted(record["symbol"] for record in records if record["status"] != "cache_hit"),
                "symbol_sources": identities,
            }

            def commit() -> None:
                _validate_published_vector_records(records, identities, config)
                _durable_vector_json(config_dir / _VECTOR_CACHE_COMMIT_FILENAME, {
                    **pending, "phase": "committed",
                    "symbol_statuses": {record["symbol"]: record["status"] for record in records},
                })
                registry.register(config_dir, producer="similar_patterns", retention_class="rebuildable",
                                  state="committed", input_versions={"config": vector_cache_key(config),
                                  "source_revision": str(token),
                                  "symbol_sources": hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()})
                registry.commit(f"similar_patterns:{vector_cache_key(config)}:vectors", config_dir)

            _publish_vector_caches(records, staging, source, token, pending=pending,
                                   commit=commit, repairing=repairing)
    return pd.DataFrame(records)


def _validate_published_vector_records(
    records: list[dict[str, object]], identities: dict[str, str], config: SimilarPatternConfig,
) -> None:
    for record in records:
        path = Path(record["cache_path"])
        if record["status"] in {"built", "cache_hit"}:
            cached = _load_stock_vector_cache(path)
            if (cached["symbol"] != record["symbol"] or not _stock_cache_shape_valid(cached, config)
                    or cached["source_fingerprint"] != "symbol-semantic:" + identities[record["symbol"]]):
                raise VectorCachePendingError(f"published vector identity is incomplete: {path}")
        elif path.exists():
            raise VectorCachePendingError(f"ineligible vector remains published: {path}")


def _publish_vector_caches(
    records: list[dict[str, object]], staging: Path, source: _VectorSource, token: object,
    *, pending: dict[str, object], commit: Callable[[], None], repairing: bool,
) -> None:
    """A durable marker quarantines interrupted multi-file publication and repair."""
    config_dir = staging.parent
    marker = config_dir / VECTOR_CACHE_PENDING_FILENAME
    changed = [record for record in records if record["status"] != "cache_hit"]
    backups: dict[Path, Path | None] = {}
    commit_started = False
    source.assert_token(token)
    _durable_vector_json(marker, pending)
    try:
        for record in changed:
            destination = Path(record["cache_path"])
            backup = staging / f"{destination.name}.previous"
            if destination.exists():
                os.link(destination, backup)
                backups[destination] = backup
            else:
                backups[destination] = None
            if record["status"] == "built":
                (staging / destination.name).replace(destination)
            else:
                destination.unlink(missing_ok=True)
        _fsync_directory(config_dir)
        source.assert_token(token)
        commit_started = True
        commit()
        source.assert_token(token)
        _clear_vector_pending(config_dir)
    except BaseException:
        rollback_errors = []
        for destination, backup in backups.items():
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    backup.replace(destination)
            except OSError as exc:
                rollback_errors.append(exc)
        _fsync_directory(config_dir)
        # A failed repair rolls back to an already mixed library. Once a commit
        # starts, its metadata may also be partially durable; only repair can certify it.
        if not rollback_errors and not repairing and not commit_started:
            _clear_vector_pending(config_dir)
        elif not marker.exists():
            _durable_vector_json(marker, pending)
        raise


def latest_snapshot(daily: pd.DataFrame, idx: int) -> dict[str, float | str | None]:
    row = daily.iloc[idx]
    hist = daily.iloc[: idx + 1].copy()
    close = hist["close"]
    volume = hist["volume"].replace(0, np.nan)
    kdj_daily_j = KDJ().compute(hist)["J"].iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    high60 = close.rolling(60).max().iloc[-1]
    low60 = close.rolling(60).min().iloc[-1]
    vol20 = volume.rolling(20).mean().iloc[-1]
    return {
        "date": row["date"].strftime("%Y-%m-%d"),
        "close": _round(row["close"]),
        "ret_1d": _round(close.pct_change().iloc[-1] * 100),
        "ret_20d": _round(close.pct_change(20).iloc[-1] * 100),
        "ret_60d": _round(close.pct_change(60).iloc[-1] * 100),
        "drawdown_60d": _round((row["close"] / high60 - 1) * 100 if high60 else np.nan),
        "range_pos_60d": _round((row["close"] - low60) / (high60 - low60) if high60 != low60 else np.nan),
        "dist_ma20": _round((row["close"] / ma20 - 1) * 100 if ma20 else np.nan),
        "dist_ma60": _round((row["close"] / ma60 - 1) * 100 if ma60 else np.nan),
        "vol_ratio20": _round(row["volume"] / vol20 if vol20 else np.nan),
        "kdj_daily_j": _round(kdj_daily_j),
    }


def build_pattern_vector(
    daily: pd.DataFrame,
    end_idx: int,
    config: SimilarPatternConfig,
    weekly_close: pd.Series | None = None,
    monthly_close: pd.Series | None = None,
) -> np.ndarray | None:
    """Build a daily/weekly/monthly volume-price shape vector ending at end_idx."""
    if end_idx < config.min_history_days or end_idx >= len(daily):
        return None

    close_all = daily["close"].to_numpy(dtype=float)
    open_all = daily["open"].to_numpy(dtype=float)
    high_all = daily["high"].to_numpy(dtype=float)
    low_all = daily["low"].to_numpy(dtype=float)
    volume_all = daily["volume"].to_numpy(dtype=float)

    daily_close = close_all[end_idx - config.lookback_days : end_idx + 1]
    if len(daily_close) < config.lookback_days + 1 or np.any(daily_close[:-1] == 0):
        return None

    daily_ret = daily_close[1:] / daily_close[:-1] - 1
    close_path = daily_close[1:] / daily_close[0] - 1
    volume_tail = volume_all[end_idx - config.lookback_days + 1 : end_idx + 1].astype(float)
    volume_tail[volume_tail <= 0] = np.nan
    finite_volume = volume_tail[np.isfinite(volume_tail)]
    volume_fill = float(np.median(finite_volume)) if len(finite_volume) else 0.0
    volume_tail = np.nan_to_num(volume_tail, nan=volume_fill, posinf=volume_fill, neginf=0.0)
    vol_log = np.log1p(np.clip(volume_tail, 0, None))
    vol_z = _zscore(vol_log)
    volume_price_interaction = _zscore(daily_ret) * vol_z

    end_date = pd.Timestamp(daily["date"].iloc[end_idx])
    if weekly_close is None or monthly_close is None:
        weekly_close, monthly_close = resample_close_series(daily)
    weekly_hist = weekly_close[weekly_close.index <= end_date]
    monthly_hist = monthly_close[monthly_close.index <= end_date]
    if len(weekly_hist) < config.weekly_lookback + 1 or len(monthly_hist) < config.monthly_lookback + 1:
        return None
    weekly_ret = weekly_hist.pct_change().tail(config.weekly_lookback).to_numpy()
    monthly_ret = monthly_hist.pct_change().tail(config.monthly_lookback).to_numpy()

    price = close_all
    high = high_all
    low = low_all
    open_price = open_all
    if any(end_idx - window + 1 < 0 for window in (10, 20, 60, 120)):
        return None
    ma20 = np.nanmean(price[end_idx - 19 : end_idx + 1])
    ma60 = np.nanmean(price[end_idx - 59 : end_idx + 1])
    ma120 = np.nanmean(price[end_idx - 119 : end_idx + 1])
    max60 = np.nanmax(price[end_idx - 59 : end_idx + 1])
    min120 = np.nanmin(price[end_idx - 119 : end_idx + 1])
    max120 = np.nanmax(price[end_idx - 119 : end_idx + 1])
    ret_window = price[end_idx - 20 : end_idx + 1]
    ret_20_values = ret_window[1:] / ret_window[:-1] - 1
    range_20 = (high[end_idx - 19 : end_idx + 1] - low[end_idx - 19 : end_idx + 1]) / np.where(
        price[end_idx - 19 : end_idx + 1] == 0, np.nan, price[end_idx - 19 : end_idx + 1]
    )
    body_denominator = np.where(open_price[end_idx - 9 : end_idx + 1] == 0, np.nan, open_price[end_idx - 9 : end_idx + 1])
    body_10 = (price[end_idx - 9 : end_idx + 1] - open_price[end_idx - 9 : end_idx + 1]) / body_denominator
    range_denominator = max120 - min120
    candle = np.array(
        [
            price[end_idx] / price[end_idx - 5] - 1,
            price[end_idx] / price[end_idx - 20] - 1,
            price[end_idx] / price[end_idx - 60] - 1,
            price[end_idx] / price[end_idx - 120] - 1,
            price[end_idx] / ma20 - 1,
            price[end_idx] / ma60 - 1,
            price[end_idx] / ma120 - 1,
            price[end_idx] / max60 - 1,
            (price[end_idx] - min120) / range_denominator if range_denominator else np.nan,
            _nan_stat(ret_20_values, "std"),
            _nan_stat(range_20, "mean"),
            _nan_stat(body_10, "mean"),
        ],
        dtype=float,
    )

    vector = np.concatenate(
        [
            _zscore(daily_ret),
            _zscore(close_path),
            vol_z,
            _zscore(volume_price_interaction),
            _zscore(weekly_ret),
            _zscore(monthly_ret),
            _zscore(candle),
        ]
    )
    if not np.isfinite(vector).all():
        return None
    return vector.astype(np.float32)


def resample_close_series(daily: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    indexed = daily.set_index("date").sort_index()
    weekly_close = indexed["close"].resample("W-FRI").last().dropna()
    monthly_close = indexed["close"].resample("ME").last().dropna()
    return weekly_close, monthly_close


def make_candidate_row(
    daily: pd.DataFrame,
    end_idx: int,
    vector: np.ndarray,
    config: SimilarPatternConfig,
    industry: str = "",
) -> dict[str, object] | None:
    max_forward = max(config.forward_days)
    if end_idx + min(config.forward_days) >= len(daily):
        return None
    row = daily.iloc[end_idx]
    close = float(row["close"])
    if close <= 0:
        return None
    future = {}
    for days in config.forward_days:
        future[f"fwd_{days}d"] = (
            float(daily.iloc[end_idx + days]["close"] / close - 1)
            if end_idx + days < len(daily)
            else np.nan
        )
    volume = daily["volume"].replace(0, np.nan)
    vol_ma20 = volume.rolling(20).mean().iloc[end_idx]
    future["fwd_1d_volume_ratio"] = (
        float(volume.iloc[end_idx + 1] / vol_ma20)
        if np.isfinite(volume.iloc[end_idx + 1]) and np.isfinite(vol_ma20) and vol_ma20 > 0
        else np.nan
    )
    high_3d = daily.iloc[end_idx + 1 : end_idx + 4]["high"].astype(float)
    low_3d = daily.iloc[end_idx + 1 : end_idx + 4]["low"].astype(float)
    future["max_runup_3d"] = float(high_3d.max() / close - 1) if len(high_3d) == 3 else np.nan
    future["max_drawdown_3d"] = float(low_3d.min() / close - 1) if len(low_3d) == 3 else np.nan
    future_window = daily.iloc[end_idx + 1 : end_idx + max_forward + 1]["close"].astype(float)
    future["max_drawdown_60d"] = float(future_window.min() / close - 1) if len(future_window) == max_forward else np.nan
    future["max_runup_60d"] = float(future_window.max() / close - 1) if len(future_window) == max_forward else np.nan
    return {
        "symbol": str(row["symbol"]),
        "name": str(row.get("name", "")),
        "industry": industry,
        "date": pd.Timestamp(row["date"]),
        "close": close,
        "vector": vector,
        **future,
    }


def make_cached_candidate_row(
    cached: dict[str, object],
    position: int,
    distance: float,
    similarity: float,
) -> dict[str, object]:
    return {
        "symbol": cached["symbol"],
        "name": cached["name"],
        "industry": cached["industry"],
        "date": pd.Timestamp(str(cached["dates"][position])),
        "close": float(cached["close"][position]),
        "fwd_1d": float(cached["fwd_1d"][position]),
        "fwd_1d_volume_ratio": float(cached["fwd_1d_volume_ratio"][position]),
        "fwd_20d": float(cached["fwd_20d"][position]),
        "fwd_60d": float(cached["fwd_60d"][position]),
        "max_runup_3d": float(cached["max_runup_3d"][position]),
        "max_drawdown_3d": float(cached["max_drawdown_3d"][position]),
        "max_drawdown_60d": float(cached["max_drawdown_60d"][position]),
        "max_runup_60d": float(cached["max_runup_60d"][position]),
        "distance": distance,
        "similarity": similarity,
    }


def _make_compiled_candidate_row(
    arrays: dict[str, np.ndarray],
    segment: dict[str, object],
    position: int,
    distance: float,
    similarity: float,
) -> dict[str, object]:
    return {
        "symbol": str(segment["symbol"]),
        "name": str(segment["name"]),
        "industry": str(segment["industry"]),
        "date": pd.Timestamp(str(arrays["dates"][position])),
        "close": float(arrays["close"][position]),
        "fwd_1d": float(arrays["fwd_1d"][position]),
        "fwd_1d_volume_ratio": float(arrays["fwd_1d_volume_ratio"][position]),
        "fwd_20d": float(arrays["fwd_20d"][position]),
        "fwd_60d": float(arrays["fwd_60d"][position]),
        "max_runup_3d": float(arrays["max_runup_3d"][position]),
        "max_drawdown_3d": float(arrays["max_drawdown_3d"][position]),
        "max_drawdown_60d": float(arrays["max_drawdown_60d"][position]),
        "max_runup_60d": float(arrays["max_runup_60d"][position]),
        "distance": distance,
        "similarity": similarity,
    }


def _threshold_screen_mask(
    vectors: np.ndarray,
    target_matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Use one BLAS scan to conservatively reject definite non-matches.

    Every cell retained by the screen is recomputed with the legacy subtraction
    and norm expression.  The error guard is intentionally conservative so a
    float32 threshold-edge case cannot become a false negative.
    """
    if threshold <= 0:
        return np.ones((len(vectors), len(target_matrix)), dtype=bool)
    if threshold > 1:
        return np.zeros((len(vectors), len(target_matrix)), dtype=bool)
    rows = np.asarray(vectors, dtype=np.float32)
    targets = np.asarray(target_matrix, dtype=np.float32)
    row_norm_sq = np.einsum("ij,ij->i", rows, rows, optimize=False)
    target_norm_sq = np.einsum("ij,ij->i", targets, targets, optimize=False)
    dots = rows @ targets.T
    fast_distance_sq = row_norm_sq[:, None] + target_norm_sq[None, :] - np.float32(2.0) * dots
    np.maximum(fast_distance_sq, np.float32(0.0), out=fast_distance_sq)

    dimension = max(1, int(rows.shape[1]))
    eps = float(np.finfo(np.float32).eps)
    gamma = dimension * eps / max(1.0 - dimension * eps, eps)
    magnitude = (
        row_norm_sq[:, None]
        + target_norm_sq[None, :]
        + np.float32(2.0) * np.abs(dots)
    )
    # Covers the dot-product, subtraction/square/reduction and final formula
    # rounding paths.  Pattern vectors are finite float32 values by contract.
    guard_sq = np.maximum(np.float32(1e-4), np.float32(32.0 * gamma) * magnitude)
    cutoff_distance = np.float32(1.0 / threshold - 1.0)
    cutoff_sq = cutoff_distance * cutoff_distance
    return fast_distance_sq <= cutoff_sq + guard_sq


def _select_compiled_contiguous_positions(
    passing_positions: np.ndarray,
    similarities: np.ndarray,
    candidate_indices: np.ndarray,
    segment_offsets: np.ndarray,
    candidate_step_days: int,
) -> np.ndarray:
    """Vectorized equivalent of best-per-contiguous-run across many stocks."""
    if len(passing_positions) == 0:
        return np.empty(0, dtype=np.int64)
    positions = np.asarray(passing_positions, dtype=np.int64)
    values = np.asarray(similarities, dtype=np.float32)
    segment_ids = np.searchsorted(segment_offsets[1:], positions, side="right")
    indices = np.asarray(candidate_indices[positions], dtype=np.int64)
    boundaries = np.ones(len(positions), dtype=bool)
    if len(positions) > 1:
        boundaries[1:] = (segment_ids[1:] != segment_ids[:-1]) | (
            indices[1:] - indices[:-1] > max(1, candidate_step_days)
        )
    starts = np.flatnonzero(boundaries)
    group_ids = np.cumsum(boundaries, dtype=np.int64) - 1
    maxima = np.maximum.reduceat(values, starts)
    is_first_max_candidate = values == maxima[group_ids]
    selected = np.full(len(starts), len(positions), dtype=np.int64)
    np.minimum.at(
        selected,
        group_ids[is_first_max_candidate],
        np.flatnonzero(is_first_max_candidate),
    )
    return positions[selected]


def _scan_compiled_threshold_cache(
    compiled: _CompiledVectorCache,
    target_vectors: dict[str, np.ndarray],
    config: SimilarPatternConfig,
    target_symbol_set: set[str],
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, list[dict[str, object]]], int]:
    threshold = float(config.similarity_threshold or 0.0)
    target_symbols = list(target_vectors)
    target_matrix = np.vstack(
        [np.asarray(target_vectors[symbol], dtype=np.float32) for symbol in target_symbols]
    )
    expected_dim = int(compiled.manifest["vector_dim"])
    if target_matrix.shape[1] != expected_dim:
        raise ValueError(
            f"target vector dimension mismatch: expected={expected_dim} actual={target_matrix.shape[1]}"
        )
    matches: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in target_symbols}
    chunks = compiled.manifest["chunks"]
    if not isinstance(chunks, list):
        raise ValueError("compiled cache manifest has invalid chunks")
    scanned_candidates = 0
    started = perf_counter()
    config_dir = compiled.generation_dir.parent.parent
    registry = _vector_artifact_registry(config_dir.parent)
    with registry.lease([config_dir, compiled.generation_dir], owner="similar_patterns:read", kind="read"), _matrix_cache_lock(config_dir, exclusive=False), _matrix_cache_lock(compiled.generation_dir.parent, exclusive=False):
        assert_vector_cache_ready(config_dir)
        for chunk_number, chunk in enumerate(chunks, start=1):
            if not isinstance(chunk, dict) or not isinstance(chunk.get("path"), str):
                raise ValueError("compiled cache manifest has an invalid chunk")
            chunk_dir = compiled.generation_dir / str(chunk["path"])
            segments = json.loads((chunk_dir / "segments.json").read_text(encoding="utf-8"))
            if not isinstance(segments, list) or not segments:
                raise ValueError(f"compiled cache chunk has no segments: {chunk_dir}")
            arrays: dict[str, np.ndarray] = {
                "vectors": np.load(chunk_dir / "vectors.npy", mmap_mode="r", allow_pickle=False),
                "indices": np.load(chunk_dir / "indices.npy", mmap_mode="r", allow_pickle=False),
                "dates": np.load(chunk_dir / "dates.npy", mmap_mode="r", allow_pickle=False),
            }
            for field in _MATRIX_CACHE_FLOAT_FIELDS:
                arrays[field] = np.load(
                    chunk_dir / f"{field}.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
            vectors = arrays["vectors"]
            candidate_indices = arrays["indices"]
            if len(vectors) != len(candidate_indices):
                raise ValueError(f"compiled cache chunk row mismatch: {chunk_dir}")
            segment_offsets = np.array(
                [int(segment["start"]) for segment in segments] + [int(segments[-1]["stop"])],
                dtype=np.int64,
            )
            eligible = np.ones(len(vectors), dtype=bool)
            for segment in segments:
                if str(segment["symbol"]).upper() in target_symbol_set:
                    eligible[int(segment["start"]) : int(segment["stop"])] = False
            scanned_candidates += int(np.count_nonzero(eligible))

            screen = _threshold_screen_mask(vectors, target_matrix, threshold)
            for target_number, target_symbol in enumerate(target_symbols):
                screened_positions = np.flatnonzero(screen[:, target_number] & eligible)
                if len(screened_positions) == 0:
                    continue
                exact_vectors = np.asarray(vectors[screened_positions], dtype=np.float32)
                distances = np.linalg.norm(
                    exact_vectors - target_matrix[target_number],
                    axis=1,
                )
                similarities = np.float32(1.0) / (np.float32(1.0) + distances)
                passing = similarities >= threshold
                passing_positions = screened_positions[passing]
                passing_similarities = similarities[passing]
                passing_distances = distances[passing]
                if len(passing_positions) == 0:
                    continue
                selected_positions = _select_compiled_contiguous_positions(
                    passing_positions,
                    passing_similarities,
                    candidate_indices,
                    segment_offsets,
                    config.candidate_step_days,
                )
                passing_offsets = np.searchsorted(passing_positions, selected_positions)
                selected_segment_ids = np.searchsorted(
                    segment_offsets[1:],
                    selected_positions,
                    side="right",
                )
                for position, passing_offset, segment_id in zip(
                    selected_positions,
                    passing_offsets,
                    selected_segment_ids,
                ):
                    matches[target_symbol].append(
                        _make_compiled_candidate_row(
                            arrays,
                            segments[int(segment_id)],
                            int(position),
                            float(passing_distances[passing_offset]),
                            float(passing_similarities[passing_offset]),
                        )
                    )

            if chunk_number % 10 == 0 or chunk_number == len(chunks):
                counts = ", ".join(
                    f"{symbol}={len(rows):,}" for symbol, rows in matches.items()
                )
                message = (
                    f"matrix threshold scan {chunk_number}/{len(chunks)} chunks, "
                    f"candidates={scanned_candidates:,}, matches: {counts}, "
                    f"elapsed={perf_counter() - started:.1f}s"
                )
                print(f"  {message}", flush=True)
                if progress_callback:
                    progress_callback(message)
    return matches, scanned_candidates


def candidate_end_indices(daily: pd.DataFrame, config: SimilarPatternConfig) -> range:
    max_end = len(daily) - min(config.forward_days) - 1
    if max_end < config.min_history_days:
        return range(0)
    if config.candidate_start_date:
        start_date = pd.Timestamp(config.candidate_start_date)
        eligible = daily.index[daily["date"] >= start_date]
        first_date_idx = int(eligible[0]) if len(eligible) else max_end + 1
        start_end = max(config.min_history_days, first_date_idx)
    else:
        start_end = max(
            config.min_history_days,
            max_end - config.max_candidates_per_symbol * config.candidate_step_days,
        )
    if start_end > max_end:
        return range(0)
    return range(start_end, max_end + 1, max(1, config.candidate_step_days))


def build_stock_candidate_matrix(
    daily: pd.DataFrame,
    config: SimilarPatternConfig,
    weekly_close: pd.Series | None = None,
    monthly_close: pd.Series | None = None,
) -> tuple[list[int], np.ndarray]:
    """Build one stock's candidate vectors so target comparisons can be vectorized."""
    if weekly_close is None or monthly_close is None:
        weekly_close, monthly_close = resample_close_series(daily)
    indices: list[int] = []
    vectors: list[np.ndarray] = []
    for end_idx in candidate_end_indices(daily, config):
        vector = build_pattern_vector(daily, end_idx, config, weekly_close, monthly_close)
        if vector is None:
            continue
        indices.append(end_idx)
        vectors.append(vector)
    if not vectors:
        return [], np.empty((0, 0), dtype=np.float32)
    return indices, np.vstack(vectors).astype(np.float32)


def select_best_positions_from_contiguous_matches(
    candidate_indices: list[int],
    similarities: np.ndarray,
    threshold: float,
    candidate_step_days: int,
) -> list[int]:
    """Keep the best match in each contiguous threshold-passing run for one stock."""
    passing_positions = np.flatnonzero(similarities >= threshold)
    if len(passing_positions) == 0:
        return []

    max_gap = max(1, candidate_step_days)
    selected: list[int] = []
    run_positions: list[int] = [int(passing_positions[0])]
    previous_idx = candidate_indices[int(passing_positions[0])]

    for raw_position in passing_positions[1:]:
        position = int(raw_position)
        current_idx = candidate_indices[position]
        if current_idx - previous_idx <= max_gap:
            run_positions.append(position)
        else:
            best = max(run_positions, key=lambda pos: float(similarities[pos]))
            selected.append(best)
            run_positions = [position]
        previous_idx = current_idx

    best = max(run_positions, key=lambda pos: float(similarities[pos]))
    selected.append(best)
    return selected


def stock_name_from_basic_or_daily(info: dict[str, object], daily: pd.DataFrame) -> str:
    if info.get("name"):
        return str(info["name"])
    if "name" in daily.columns and daily["name"].notna().any():
        return str(daily["name"].dropna().iloc[-1])
    return ""


def is_excluded_stock(name: str) -> bool:
    return "ST" in name.upper() or "退" in name


def build_candidate_library(
    daily_dir: Path,
    basic: pd.DataFrame,
    config: SimilarPatternConfig,
    target_symbols: set[str],
    max_symbols: int | None = None,
) -> pd.DataFrame:
    """Build historical pattern candidates from local daily parquet files."""
    files = list_partitioned_symbol_paths(daily_dir)
    if max_symbols is not None:
        files = files[:max_symbols]
    basic_map = basic.set_index("ts_code").to_dict("index") if not basic.empty else {}
    rows: list[dict[str, object]] = []
    started = perf_counter()
    for n, path in enumerate(files, start=1):
        try:
            daily = load_daily_file(path)
        except Exception as exc:
            print(f"skip {path.name}: {exc}", flush=True)
            continue
        if len(daily) < config.min_history_days + max(config.forward_days) + 1:
            continue
        symbol = path.stem
        info = basic_map.get(symbol, {})
        industry = str(info.get("industry", ""))
        if info.get("name"):
            daily["name"] = daily["name"].replace("", np.nan).fillna(str(info["name"]))
        name = stock_name_from_basic_or_daily(info, daily)
        if is_excluded_stock(name):
            continue

        weekly_close, monthly_close = resample_close_series(daily)
        for end_idx in candidate_end_indices(daily, config):
            vector = build_pattern_vector(daily, end_idx, config, weekly_close, monthly_close)
            if vector is None:
                continue
            row = make_candidate_row(daily, end_idx, vector, config, industry)
            if row is not None:
                rows.append(row)

        if n % 500 == 0:
            elapsed = perf_counter() - started
            print(f"  library scan {n}/{len(files)} files, candidates={len(rows):,}, elapsed={elapsed:.1f}s", flush=True)

    library = pd.DataFrame(rows)
    if library.empty:
        return library
    target_symbols = {symbol.upper() for symbol in target_symbols}
    library = library[~library["symbol"].str.upper().isin(target_symbols)].reset_index(drop=True)
    return library


def analyze_target(
    symbol: str,
    daily: pd.DataFrame,
    library: pd.DataFrame,
    config: SimilarPatternConfig,
    basic: pd.DataFrame,
    target_date: str | None = None,
) -> SimilarPatternResult:
    if target_date:
        target_ts = pd.Timestamp(target_date)
        eligible = daily[daily["date"] <= target_ts]
        if eligible.empty:
            raise ValueError(f"{symbol} has no rows on or before {target_date}")
        end_idx = int(eligible.index[-1])
    else:
        end_idx = len(daily) - 1
        target_ts = pd.Timestamp(daily.iloc[end_idx]["date"])

    weekly_close, monthly_close = resample_close_series(daily)
    vector = build_pattern_vector(daily, end_idx, config, weekly_close, monthly_close)
    if vector is None:
        raise ValueError(f"{symbol} has insufficient history for target date {target_ts.date()}")
    if len(library) < config.min_candidate_rows:
        raise ValueError(f"candidate library too small: {len(library)} rows")

    matrix = np.vstack(library["vector"].to_numpy())
    distances = np.linalg.norm(matrix - vector, axis=1)
    similarity = 1 / (1 + distances)
    top_idx = np.argsort(distances)[: config.top_k]
    cases = library.iloc[top_idx].drop(columns=["vector"]).copy()
    cases["distance"] = distances[top_idx]
    cases["similarity"] = similarity[top_idx]
    cases = cases.sort_values(["distance", "date"]).reset_index(drop=True)
    cases["rank"] = np.arange(1, len(cases) + 1)

    forecast = summarize_forecast(cases)
    status_probs = summarize_status_probs(cases)
    basic_row = basic[basic["ts_code"] == symbol]
    name = str(basic_row["name"].iloc[0]) if not basic_row.empty else str(daily["name"].dropna().iloc[-1])
    target = TargetSpec(symbol=symbol, name=name, target_date=target_ts)
    return add_trade_plans(SimilarPatternResult(
        target=target,
        latest_snapshot=latest_snapshot(daily, end_idx),
        similar_cases=cases,
        forecast=forecast,
        status_probs=status_probs,
        match_mode="top_k",
        scan_summary={"library_rows": int(len(library)), "top_k": int(config.top_k)},
    ), config)


def prepare_target_context(
    symbol: str,
    daily: pd.DataFrame,
    config: SimilarPatternConfig,
    basic: pd.DataFrame,
    target_date: str | None = None,
) -> dict[str, object]:
    if target_date:
        target_ts = pd.Timestamp(target_date)
        eligible = daily[daily["date"] <= target_ts]
        if eligible.empty:
            raise ValueError(f"{symbol} has no rows on or before {target_date}")
        end_idx = int(eligible.index[-1])
    else:
        end_idx = len(daily) - 1
        target_ts = pd.Timestamp(daily.iloc[end_idx]["date"])
    weekly_close, monthly_close = resample_close_series(daily)
    vector = build_pattern_vector(daily, end_idx, config, weekly_close, monthly_close)
    if vector is None:
        raise ValueError(f"{symbol} has insufficient history for target date {target_ts.date()}")
    basic_row = basic[basic["ts_code"] == symbol]
    name = str(basic_row["name"].iloc[0]) if not basic_row.empty else str(daily["name"].dropna().iloc[-1])
    return {
        "symbol": symbol,
        "target": TargetSpec(symbol=symbol, name=name, target_date=target_ts),
        "snapshot": latest_snapshot(daily, end_idx),
        "vector": vector,
    }


def analyze_targets_by_threshold(
    daily_dir: Path,
    basic: pd.DataFrame,
    config: SimilarPatternConfig,
    target_symbols: list[str],
    target_date: str | None = None,
    max_symbols: int | None = None,
    vector_cache_dir: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, SimilarPatternResult]:
    access = nullcontext()
    if vector_cache_dir is not None:
        config_dir = vector_cache_dir.resolve() / vector_cache_key(config)
        registry = _vector_artifact_registry(vector_cache_dir)
        access = registry.lease([config_dir], owner="similar_patterns:read", kind="read")
    with access, _matrix_cache_lock(config_dir, exclusive=False) if vector_cache_dir is not None else nullcontext():
        if vector_cache_dir is not None:
            assert_vector_cache_ready(config_dir)
        result = _analyze_targets_by_threshold(
            daily_dir, basic, config, target_symbols, target_date, max_symbols,
            vector_cache_dir, progress_callback,
        )
    if vector_cache_dir is not None:
        _collect_compiled_vector_caches(vector_cache_dir, config)
    return result


def _analyze_targets_by_threshold(
    daily_dir: Path,
    basic: pd.DataFrame,
    config: SimilarPatternConfig,
    target_symbols: list[str],
    target_date: str | None = None,
    max_symbols: int | None = None,
    vector_cache_dir: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, SimilarPatternResult]:
    if vector_cache_dir is not None:
        assert_vector_cache_ready(vector_cache_dir / vector_cache_key(config))
    if config.similarity_threshold is None:
        raise ValueError("similarity_threshold is required for threshold mode")

    files = list_partitioned_symbol_paths(daily_dir)
    symbol_paths = {path.stem.upper(): path for path in files}
    target_contexts: dict[str, dict[str, object]] = {}
    for symbol in target_symbols:
        target_path = symbol_paths.get(symbol.upper())
        if target_path is None:
            print(f"missing target daily data: {symbol}", flush=True)
            continue
        daily = load_daily_file(target_path)
        target_contexts[symbol] = prepare_target_context(symbol, daily, config, basic, target_date)
    if not target_contexts:
        return {}

    target_vectors = {symbol: context["vector"] for symbol, context in target_contexts.items()}
    target_symbol_set = {symbol.upper() for symbol in target_contexts}
    target_matches: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in target_contexts}

    if max_symbols is not None:
        files = files[:max_symbols]
    basic_map = basic.set_index("ts_code").to_dict("index") if not basic.empty else {}
    scanned_candidates = 0
    started = perf_counter()
    cache_mode = "daily_files"
    compiled_manifest: dict[str, object] | None = None
    compiled_scan_complete = False
    matrix_cache_disabled = os.getenv("SIMILAR_PATTERN_DISABLE_MATRIX_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if vector_cache_dir is not None and not matrix_cache_disabled:
        try:
            inventory = _vector_cache_inventory(files, vector_cache_dir, config)
            compile_started = perf_counter()
            compiled = _ensure_compiled_vector_cache(inventory, vector_cache_dir, config)
            compiled_manifest = compiled.manifest
            compile_elapsed = perf_counter() - compile_started
            if compile_elapsed >= 0.5:
                message = (
                    f"matrix cache ready: chunks={len(compiled.manifest['chunks'])}, "
                    f"rows={int(compiled.manifest['rows']):,}, elapsed={compile_elapsed:.1f}s"
                )
                print(f"  {message}", flush=True)
                if progress_callback:
                    progress_callback(message)
            target_matches, scanned_candidates = _scan_compiled_threshold_cache(
                compiled,
                target_vectors,
                config,
                target_symbol_set,
                progress_callback,
            )
            cache_mode = "matrix_chunks"
            compiled_scan_complete = True
        except VectorCachePendingError:
            raise
        except Exception as exc:
            message = f"matrix cache unavailable; falling back to legacy scan: {exc}"
            print(f"  {message}", flush=True)
            if progress_callback:
                progress_callback(message)
            target_matches = {symbol: [] for symbol in target_contexts}
            scanned_candidates = 0

    if not compiled_scan_complete:
        cache_mode = "legacy_npz" if vector_cache_dir is not None else "daily_files"
        for n, path in enumerate(files, start=1):
            symbol = path.stem
            if symbol.upper() in target_symbol_set:
                continue
            cached: dict[str, object] | None = None
            daily: pd.DataFrame | None = None
            if vector_cache_dir is not None:
                cache_path = vector_cache_path(vector_cache_dir, symbol, config)
                if not cache_path.exists():
                    continue
                cached = _load_stock_vector_cache(cache_path)
                candidate_indices = [int(value) for value in cached["indices"]]
                candidate_matrix = cached["vectors"]
            else:
                try:
                    daily = load_daily_file(path)
                except Exception as exc:
                    print(f"skip {path.name}: {exc}", flush=True)
                    continue
                if len(daily) < config.min_history_days + max(config.forward_days) + 1:
                    continue
                info = basic_map.get(symbol, {})
                industry = str(info.get("industry", ""))
                if info.get("name"):
                    daily["name"] = daily["name"].replace("", np.nan).fillna(str(info["name"]))
                name = stock_name_from_basic_or_daily(info, daily)
                if is_excluded_stock(name):
                    continue

                weekly_close, monthly_close = resample_close_series(daily)
                candidate_indices, candidate_matrix = build_stock_candidate_matrix(
                    daily,
                    config,
                    weekly_close,
                    monthly_close,
                )
            if len(candidate_indices) == 0:
                continue
            scanned_candidates += len(candidate_indices)

            for target_symbol, target_vector in target_vectors.items():
                deltas = candidate_matrix - target_vector
                distances = np.linalg.norm(deltas, axis=1)
                similarities = 1 / (1 + distances)
                selected_positions = select_best_positions_from_contiguous_matches(
                    candidate_indices,
                    similarities,
                    config.similarity_threshold,
                    config.candidate_step_days,
                )
                for position in selected_positions:
                    if cached is not None:
                        row = make_cached_candidate_row(
                            cached,
                            position,
                            float(distances[position]),
                            float(similarities[position]),
                        )
                    else:
                        end_idx = candidate_indices[position]
                        row = make_candidate_row(daily, end_idx, candidate_matrix[position], config, industry)
                        if row is None:
                            continue
                        row.pop("vector", None)
                        row["distance"] = float(distances[position])
                        row["similarity"] = float(similarities[position])
                    target_matches[target_symbol].append(row)

            if n % 500 == 0:
                elapsed = perf_counter() - started
                counts = ", ".join(
                    f"{symbol}={len(rows):,}" for symbol, rows in target_matches.items()
                )
                message = (
                    f"threshold scan {n}/{len(files)} files, candidates={scanned_candidates:,}, "
                    f"matches: {counts}, elapsed={elapsed:.1f}s"
                )
                print(f"  {message}", flush=True)
                if progress_callback:
                    progress_callback(message)

    results: dict[str, SimilarPatternResult] = {}
    for symbol, context in target_contexts.items():
        cases = pd.DataFrame(target_matches[symbol])
        if cases.empty:
            cases = pd.DataFrame(
                columns=[
                    "symbol",
                    "name",
                    "industry",
                    "date",
                    "close",
                    "fwd_1d",
                    "fwd_20d",
                    "fwd_60d",
                    "max_drawdown_60d",
                    "max_runup_60d",
                    "distance",
                    "similarity",
                ]
            )
            forecast = summarize_forecast(cases)
            status_probs = {"上升": 0.0, "震荡": 0.0, "下跌": 0.0, "高波动": 0.0}
        else:
            cases = cases.sort_values(["similarity", "date"], ascending=[False, True]).reset_index(drop=True)
            cases["rank"] = np.arange(1, len(cases) + 1)
            forecast = summarize_forecast(cases)
            status_probs = summarize_status_probs(cases)
        results[symbol] = add_trade_plans(SimilarPatternResult(
            target=context["target"],
            latest_snapshot=context["snapshot"],
            similar_cases=cases,
            forecast=forecast,
            status_probs=status_probs,
            match_mode="threshold",
            scan_summary={
                "candidate_start_date": config.candidate_start_date,
                "candidate_step_days": int(config.candidate_step_days),
                "similarity_threshold": float(config.similarity_threshold),
                "scanned_candidates": int(scanned_candidates),
                "matched_cases": int(len(cases)),
                "contiguous_dedupe": "best_per_stock_run",
                "vector_cache": str(vector_cache_dir) if vector_cache_dir else None,
                "vector_cache_mode": cache_mode,
                "matrix_cache_fingerprint": (
                    compiled_manifest.get("inventory_fingerprint")
                    if compiled_manifest is not None and cache_mode == "matrix_chunks"
                    else None
                ),
                "max_symbols": max_symbols,
            },
        ), config)
    return results


def optimize_similar_cases(
    cases: pd.DataFrame,
    config: SimilarPatternConfig,
    *,
    target_date: pd.Timestamp,
    target_industry: str,
    target_market_regime: str,
    target_industry_regime: str = "",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Deduplicate correlated events and assign nonlinear forecast weights."""
    raw_cases = int(len(cases))
    if cases.empty:
        return cases.copy(), {
            "raw_cases": 0,
            "deduplicated_cases": 0,
            "effective_sample_size": 0.0,
            "sample_status": "insufficient",
        }

    out = cases.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["industry"] = out.get("industry", "").fillna("").astype(str)
    out["similarity"] = pd.to_numeric(out["similarity"], errors="coerce")
    out = out.dropna(subset=["date", "similarity"])
    out = out.sort_values(["similarity", "date"], ascending=[False, False])
    out = out.drop_duplicates(["date", "industry"], keep="first")
    out = (
        out.sort_values(["date", "similarity"], ascending=[False, False])
        .groupby("date", group_keys=False)
        .head(max(1, config.max_events_per_date))
    )

    threshold = float(config.similarity_threshold or 0.0)
    margins = (out["similarity"] - threshold).clip(lower=0.0001)
    max_margin = float(margins.max()) if not margins.empty else 0.0001
    base_weight = (margins / max(max_margin, 0.0001)).pow(config.similarity_weight_power)
    industry_factor = np.where(
        out["industry"].eq(str(target_industry)),
        config.same_industry_weight,
        config.cross_industry_weight,
    )
    if "market_regime" in out.columns and target_market_regime:
        regime_factor = np.where(
            out["market_regime"].fillna("neutral").astype(str).eq(str(target_market_regime)),
            config.same_regime_weight,
            config.regime_mismatch_weight,
        )
    else:
        regime_factor = np.ones(len(out), dtype=float)
    if "industry_regime" in out.columns and target_industry_regime:
        same_industry = out["industry"].eq(str(target_industry))
        industry_regime_factor = np.where(
            ~same_industry,
            1.0,
            np.where(
                out["industry_regime"].fillna("neutral").astype(str).eq(str(target_industry_regime)),
                config.same_industry_regime_weight,
                config.industry_regime_mismatch_weight,
            ),
        )
    else:
        industry_regime_factor = np.ones(len(out), dtype=float)
    ages = (pd.Timestamp(target_date) - out["date"]).dt.days.clip(lower=0)
    recency_factor = np.power(0.5, ages / max(1, config.recency_half_life_days))
    out["forecast_weight"] = (
        base_weight.to_numpy(dtype=float)
        * industry_factor
        * regime_factor
        * industry_regime_factor
        * recency_factor
    )
    out = out.sort_values(["forecast_weight", "similarity"], ascending=[False, False]).head(
        max(1, config.max_effective_cases)
    )
    weight = out["forecast_weight"].to_numpy(dtype=float)
    weight_sum = float(np.sum(weight))
    effective_sample_size = (
        float(weight_sum**2 / np.sum(np.square(weight)))
        if weight_sum > 0 and float(np.sum(np.square(weight))) > 0
        else 0.0
    )
    out = out.reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out, {
        "raw_cases": raw_cases,
        "deduplicated_cases": int(len(out)),
        "effective_sample_size": round(effective_sample_size, 2),
        "sample_status": "ready" if effective_sample_size >= config.min_effective_cases else "insufficient",
        "max_events_per_date": int(config.max_events_per_date),
        "max_effective_cases": int(config.max_effective_cases),
        "weight_power": float(config.similarity_weight_power),
    }


def build_probability_variant_cases(
    cases: pd.DataFrame,
    config: SimilarPatternConfig,
    *,
    target_date: pd.Timestamp,
    target_industry: str,
    target_market_regime: str,
    target_industry_regime: str = "",
) -> dict[str, pd.DataFrame]:
    """Build transferable single-condition variants used by global model selection."""
    if cases.empty:
        return {name: cases.copy() for name in ("event_dedupe", "nonlinear", "regime_industry", "recency")}

    normalized = cases.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["industry"] = normalized.get("industry", "").fillna("").astype(str)
    normalized["similarity"] = pd.to_numeric(normalized["similarity"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "similarity"])

    event_dedupe = normalized.sort_values(["similarity", "date"], ascending=[False, False]).drop_duplicates(
        ["date", "industry"], keep="first"
    )
    event_dedupe = (
        event_dedupe.sort_values(["date", "similarity"], ascending=[False, False])
        .groupby("date", group_keys=False)
        .head(max(1, config.max_events_per_date))
    )

    nonlinear = normalized.copy()
    threshold = float(config.similarity_threshold or 0.0)
    margin = (nonlinear["similarity"] - threshold).clip(lower=0.0001)
    nonlinear["forecast_weight"] = np.square(margin / max(float(margin.max()), 0.0001))

    regime_industry = normalized.copy()
    weight = regime_industry["similarity"].fillna(0.0)
    weight *= np.where(
        regime_industry["industry"].eq(str(target_industry)),
        config.same_industry_weight,
        config.cross_industry_weight,
    )
    if "market_regime" in regime_industry.columns and target_market_regime:
        weight *= np.where(
            regime_industry["market_regime"].fillna("neutral").astype(str).eq(str(target_market_regime)),
            config.same_regime_weight,
            config.regime_mismatch_weight,
        )
    if "industry_regime" in regime_industry.columns and target_industry_regime:
        same_industry = regime_industry["industry"].eq(str(target_industry))
        weight *= np.where(
            ~same_industry,
            1.0,
            np.where(
                regime_industry["industry_regime"].fillna("neutral").astype(str).eq(str(target_industry_regime)),
                config.same_industry_regime_weight,
                config.industry_regime_mismatch_weight,
            ),
        )
    regime_industry["forecast_weight"] = weight

    recency = normalized.copy()
    ages = (pd.Timestamp(target_date) - recency["date"]).dt.days.clip(lower=0)
    recency["forecast_weight"] = recency["similarity"].fillna(0.0) * np.power(
        0.5,
        ages / max(1, config.recency_half_life_days),
    )
    return {
        "event_dedupe": event_dedupe,
        "nonlinear": nonlinear,
        "regime_industry": regime_industry,
        "recency": recency,
    }


def classify_forecast_signal(
    up_probability: float | None,
    snapshot: dict[str, float | str | None],
    market_regime: str,
    config: SimilarPatternConfig,
) -> dict[str, object]:
    """Classify a forecast and veto weak bullish signals during breakdowns."""
    if up_probability is None or not np.isfinite(float(up_probability)):
        return {"signal": "observe", "risk_gate": "missing", "reasons": ["缺少有效概率"]}
    probability = float(up_probability)
    if probability >= config.signal_bullish_min:
        signal = "bullish"
    elif probability <= config.signal_bearish_max:
        signal = "bearish"
    else:
        signal = "observe"

    def value(key: str) -> float:
        raw = snapshot.get(key)
        try:
            parsed = float(raw) if raw is not None else np.nan
        except (TypeError, ValueError):
            return np.nan
        return parsed

    reasons: list[str] = []
    dist_ma20 = value("dist_ma20")
    dist_ma60 = value("dist_ma60")
    drawdown = value("drawdown_60d")
    volume_ratio = value("vol_ratio20")
    breakdown = (
        (np.isfinite(dist_ma20) and dist_ma20 <= -3.0)
        and (np.isfinite(dist_ma60) and dist_ma60 <= -5.0)
        and (
            (np.isfinite(drawdown) and drawdown <= -12.0)
            or (np.isfinite(volume_ratio) and volume_ratio >= 1.3)
        )
    )
    if breakdown:
        reasons.append("放量或深回撤破位")
    if market_regime == "risk_off":
        reasons.append("沪深300处于风险规避状态")
    blocked = config.enable_risk_gate and signal == "bullish" and (
        breakdown or (market_regime == "risk_off" and probability < 60.0)
    )
    return {
        "signal": "observe" if blocked else signal,
        "raw_signal": signal,
        "risk_gate": "blocked" if blocked else "passed",
        "reasons": reasons,
        "probability": round(probability, 2),
        "bearish_max": float(config.signal_bearish_max),
        "bullish_min": float(config.signal_bullish_min),
    }


def fit_probability_calibration(
    probabilities: list[float],
    outcomes: list[bool],
    *,
    min_samples: int = 20,
) -> dict[str, object]:
    """Fit a small monotonic reliability curve that is JSON serializable."""
    frame = pd.DataFrame({"probability": probabilities, "outcome": outcomes}).dropna()
    if len(frame) < min_samples or frame["outcome"].nunique() < 2:
        return {"status": "identity", "sample_count": int(len(frame)), "x": [0.0, 100.0], "y": [0.0, 100.0]}
    frame["bin"] = pd.qcut(frame["probability"], q=min(6, frame["probability"].nunique()), duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        x=("probability", "mean"),
        positives=("outcome", "sum"),
        count=("outcome", "size"),
    )
    grouped["y"] = (grouped["positives"] + 1.0) / (grouped["count"] + 2.0) * 100.0
    x_values = grouped["x"].astype(float).to_numpy()
    y_values = np.maximum.accumulate(grouped["y"].astype(float).to_numpy())
    x_values = np.r_[0.0, x_values, 100.0]
    y_values = np.r_[max(0.0, y_values[0] - 10.0), y_values, min(100.0, y_values[-1] + 10.0)]
    return {
        "status": "fitted",
        "sample_count": int(len(frame)),
        "x": [round(float(value), 4) for value in x_values],
        "y": [round(float(value), 4) for value in y_values],
    }


def apply_probability_calibration(
    probability: float | None,
    calibration: dict[str, object] | None,
) -> float | None:
    """Apply a serialized monotonic reliability curve."""
    if probability is None or not np.isfinite(float(probability)):
        return None
    if not calibration:
        return _round(float(probability))
    x_values = np.asarray(calibration.get("x") or [0.0, 100.0], dtype=float)
    y_values = np.asarray(calibration.get("y") or [0.0, 100.0], dtype=float)
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return _round(float(probability))
    return _round(float(np.interp(float(probability), x_values, y_values)))


def summarize_forecast(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col, label in [("fwd_1d", "next_1d"), ("fwd_20d", "next_1m"), ("fwd_60d", "next_3m")]:
        if cases.empty or col not in cases.columns or "similarity" not in cases.columns:
            rows.append(
                {
                    "horizon": label,
                    "sample_count": 0,
                    "up_probability": None,
                    "mean_return": None,
                    "p10": None,
                    "p25": None,
                    "median": None,
                    "p75": None,
                    "p90": None,
                }
            )
            continue
        values = cases[col].astype(float)
        weight_col = "forecast_weight" if "forecast_weight" in cases.columns else "similarity"
        weights = cases[weight_col].astype(float)
        if values.notna().sum() == 0 or weights.sum() <= 0:
            rows.append(
                {
                    "horizon": label,
                    "sample_count": 0,
                    "up_probability": None,
                    "mean_return": None,
                    "p10": None,
                    "p25": None,
                    "median": None,
                    "p75": None,
                    "p90": None,
                }
            )
            continue
        rows.append(
            {
                "horizon": label,
                "sample_count": int(values.notna().sum()),
                "up_probability": _round(float((weights[values > 0].sum() / weights.sum()) * 100)),
                "mean_return": _round(float(np.average(values, weights=weights)) * 100),
                "p10": _round(float(values.quantile(0.10)) * 100),
                "p25": _round(float(values.quantile(0.25)) * 100),
                "median": _round(float(values.quantile(0.50)) * 100),
                "p75": _round(float(values.quantile(0.75)) * 100),
                "p90": _round(float(values.quantile(0.90)) * 100),
            }
        )
    return pd.DataFrame(rows)


def summarize_status_probs(cases: pd.DataFrame) -> dict[str, float]:
    if cases.empty:
        return {"上升": 0.0, "震荡": 0.0, "下跌": 0.0, "高波动": 0.0}
    returns = cases["fwd_60d"].astype(float)
    drawdowns = cases["max_drawdown_60d"].astype(float)
    statuses = np.select(
        [
            returns >= 0.10,
            (returns > -0.05) & (returns < 0.10) & (drawdowns > -0.12),
            returns <= -0.10,
        ],
        ["上升", "震荡", "下跌"],
        default="高波动",
    )
    weights = cases["similarity"].astype(float).to_numpy()
    total = weights.sum()
    if total <= 0:
        return {"上升": 0.0, "震荡": 0.0, "下跌": 0.0, "高波动": 0.0}
    return {
        status: _round(float(weights[statuses == status].sum() / total * 100))
        for status in ["上升", "震荡", "下跌", "高波动"]
    }


def add_trade_plans(result: SimilarPatternResult, config: SimilarPatternConfig) -> SimilarPatternResult:
    scenario_plan = build_t1_scenario_plan(result.similar_cases, config.take_profit_3d, config.stop_loss_3d)
    model_plan, model_summary = build_sell_model_plan(
        result.similar_cases,
        config.take_profit_3d,
        config.stop_loss_3d,
    )
    return SimilarPatternResult(
        target=result.target,
        latest_snapshot=result.latest_snapshot,
        similar_cases=result.similar_cases,
        forecast=result.forecast,
        status_probs=result.status_probs,
        t1_scenario_plan=scenario_plan,
        sell_model_plan=model_plan,
        sell_model_summary=model_summary,
        match_mode=result.match_mode,
        scan_summary=result.scan_summary,
    )


def classify_t1_return(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    percent = value * 100
    if percent <= -3:
        return "大跌<=-3%"
    if percent <= -1:
        return "小跌-3~-1%"
    if percent < 1:
        return "震荡-1~1%"
    if percent < 3:
        return "小涨1~3%"
    return "大涨>=3%"


def classify_volume_ratio(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 0.8:
        return "缩量<0.8"
    if value <= 1.3:
        return "平量0.8~1.3"
    if value <= 2.0:
        return "放量1.3~2"
    return "巨量>2"


def action_from_probs(sample_count: int, up_prob: float, down_prob: float) -> str:
    if sample_count < 20:
        return "观察-样本不足"
    if down_prob >= 55 and down_prob > up_prob + 10:
        return "卖出/减仓"
    if up_prob >= 55 and down_prob <= 35:
        return "持有/可低吸"
    if down_prob >= 45:
        return "谨慎持有"
    return "持有观察"


def build_t1_scenario_plan(cases: pd.DataFrame, take_profit_3d: float, stop_loss_3d: float) -> pd.DataFrame:
    columns = [
        "t1_return_bucket",
        "t1_volume_bucket",
        "sample_count",
        "hit_up_3d_prob",
        "hit_down_3d_prob",
        "mean_fwd_20d",
        "median_fwd_20d",
        "action",
    ]
    required = {"fwd_1d", "fwd_1d_volume_ratio", "max_runup_3d", "max_drawdown_3d", "fwd_20d"}
    if cases.empty or not required.issubset(cases.columns):
        return pd.DataFrame(columns=columns)
    data = cases.copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
    if data.empty:
        return pd.DataFrame(columns=columns)
    data["t1_return_bucket"] = data["fwd_1d"].map(classify_t1_return)
    data["t1_volume_bucket"] = data["fwd_1d_volume_ratio"].map(classify_volume_ratio)
    data["hit_up_3d"] = data["max_runup_3d"] >= take_profit_3d
    data["hit_down_3d"] = data["max_drawdown_3d"] <= -stop_loss_3d
    rows: list[dict[str, object]] = []
    for (return_bucket, volume_bucket), group in data.groupby(["t1_return_bucket", "t1_volume_bucket"], sort=True):
        up_prob = float(group["hit_up_3d"].mean() * 100)
        down_prob = float(group["hit_down_3d"].mean() * 100)
        sample_count = int(len(group))
        rows.append(
            {
                "t1_return_bucket": return_bucket,
                "t1_volume_bucket": volume_bucket,
                "sample_count": sample_count,
                "hit_up_3d_prob": _round(up_prob),
                "hit_down_3d_prob": _round(down_prob),
                "mean_fwd_20d": _round(float(group["fwd_20d"].mean()) * 100),
                "median_fwd_20d": _round(float(group["fwd_20d"].median()) * 100),
                "action": action_from_probs(sample_count, up_prob, down_prob),
            }
        )
    return pd.DataFrame(rows).sort_values(["t1_return_bucket", "t1_volume_bucket"]).reset_index(drop=True)


def build_sell_model_plan(
    cases: pd.DataFrame,
    take_profit_3d: float,
    stop_loss_3d: float,
) -> tuple[pd.DataFrame, dict[str, int | float | str | None]]:
    columns = [
        "scenario",
        "t1_return",
        "t1_volume_ratio",
        "model_hit_up_3d_prob",
        "model_hit_down_3d_prob",
        "recommendation",
    ]
    required = {"fwd_1d", "fwd_1d_volume_ratio", "max_runup_3d", "max_drawdown_3d", "similarity", "distance"}
    if cases.empty or not required.issubset(cases.columns):
        return pd.DataFrame(columns=columns), {"status": "insufficient_data", "sample_count": 0}

    data = cases.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).copy()
    if len(data) < 80:
        return pd.DataFrame(columns=columns), {"status": "insufficient_data", "sample_count": int(len(data))}

    feature_cols = ["fwd_1d", "fwd_1d_volume_ratio", "similarity", "distance"]
    x = data[feature_cols].to_numpy(dtype=float)
    y_up = (data["max_runup_3d"].to_numpy(dtype=float) >= take_profit_3d).astype(int)
    y_down = (data["max_drawdown_3d"].to_numpy(dtype=float) <= -stop_loss_3d).astype(int)
    if len(np.unique(y_up)) < 2 or len(np.unique(y_down)) < 2:
        return pd.DataFrame(columns=columns), {"status": "single_class_label", "sample_count": int(len(data))}

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return pd.DataFrame(columns=columns), {"status": "sklearn_missing", "sample_count": int(len(data))}

    up_model = RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=20, random_state=42)
    down_model = RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=20, random_state=43)
    up_model.fit(x, y_up)
    down_model.fit(x, y_down)

    median_similarity = float(data["similarity"].median())
    median_distance = float(data["distance"].median())
    scenarios = [
        ("大跌缩量", -0.035, 0.7),
        ("大跌放量", -0.035, 1.6),
        ("小跌平量", -0.015, 1.0),
        ("震荡平量", 0.0, 1.0),
        ("小涨放量", 0.015, 1.6),
        ("大涨放量", 0.035, 1.8),
    ]
    rows = []
    for name, t1_ret, volume_ratio in scenarios:
        scenario_x = np.array([[t1_ret, volume_ratio, median_similarity, median_distance]], dtype=float)
        up_prob = float(up_model.predict_proba(scenario_x)[0, 1] * 100)
        down_prob = float(down_model.predict_proba(scenario_x)[0, 1] * 100)
        rows.append(
            {
                "scenario": name,
                "t1_return": _round(t1_ret * 100),
                "t1_volume_ratio": _round(volume_ratio),
                "model_hit_up_3d_prob": _round(up_prob),
                "model_hit_down_3d_prob": _round(down_prob),
                "recommendation": action_from_probs(len(data), up_prob, down_prob),
            }
        )
    summary = {
        "status": "trained",
        "sample_count": int(len(data)),
        "take_profit_3d": _round(take_profit_3d * 100),
        "stop_loss_3d": _round(stop_loss_3d * 100),
        "up_positive_rate": _round(float(y_up.mean() * 100)),
        "down_positive_rate": _round(float(y_down.mean() * 100)),
    }
    return pd.DataFrame(rows), summary


def write_result(result: SimilarPatternResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = result.target.symbol.replace(".", "_")
    cases_path = output_dir / f"{slug}_similar_cases.csv"
    forecast_path = output_dir / f"{slug}_forecast.csv"
    meta_path = output_dir / f"{slug}_summary.json"
    report_path = output_dir / f"{slug}_report.md"

    cases_out = result.similar_cases.copy()
    if "date" in cases_out.columns and not cases_out.empty:
        cases_out["date"] = pd.to_datetime(cases_out["date"]).dt.strftime("%Y-%m-%d")
    cases_out.to_csv(cases_path, index=False, encoding="utf-8-sig")
    result.forecast.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    if result.t1_scenario_plan is not None:
        result.t1_scenario_plan.to_csv(output_dir / f"{slug}_t1_scenario_plan.csv", index=False, encoding="utf-8-sig")
    if result.sell_model_plan is not None:
        result.sell_model_plan.to_csv(output_dir / f"{slug}_sell_model_plan.csv", index=False, encoding="utf-8-sig")
    meta = {
        "target": {
            "symbol": result.target.symbol,
            "name": result.target.name,
            "target_date": result.target.target_date.strftime("%Y-%m-%d"),
        },
        "latest_snapshot": result.latest_snapshot,
        "status_probs": result.status_probs,
        "match_mode": result.match_mode,
        "scan_summary": result.scan_summary,
        "sell_model_summary": result.sell_model_summary,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_markdown_report(result), encoding="utf-8")
    return {"cases": cases_path, "forecast": forecast_path, "summary": meta_path, "report": report_path}


def render_markdown_report(result: SimilarPatternResult) -> str:
    top_cases = result.similar_cases.head(20).copy()
    display_cols = [
        "rank",
        "symbol",
        "name",
        "industry",
        "date",
        "similarity",
        "fwd_1d",
        "fwd_20d",
        "fwd_60d",
        "max_drawdown_60d",
    ]
    if top_cases.empty:
        top_cases = pd.DataFrame(columns=display_cols)
    else:
        top_cases = top_cases[display_cols].copy()
        for col in ["fwd_1d", "fwd_20d", "fwd_60d", "max_drawdown_60d"]:
            top_cases[col] = (top_cases[col].astype(float) * 100).round(2)
        top_cases["similarity"] = top_cases["similarity"].round(4)
        top_cases["date"] = pd.to_datetime(top_cases["date"]).dt.strftime("%Y-%m-%d")

    lines = [
        f"# {result.target.name}（{result.target.symbol}）历史相似走势预测报告",
        "",
        f"- 目标日期：{result.target.target_date.strftime('%Y-%m-%d')}",
        "- 数据口径：本地 `data/raw/daily` 日线，按量价形态、量价交互、周/月趋势构建相似向量。",
        f"- 匹配模式：{result.match_mode}",
        "- 说明：该报告是研究原型输出，不构成投资建议；后续应切换统一前复权数据并做滚动回测。",
        "",
    ]
    if result.scan_summary:
        lines.extend(
            [
                "## 扫描设置",
                "",
                pd.DataFrame([result.scan_summary]).to_markdown(index=False),
                "",
            ]
        )
    lines.extend(
        [
        "## 当前走势画像",
        "",
        pd.DataFrame([result.latest_snapshot]).to_markdown(index=False),
        "",
        "## 相似案例后验预测",
        "",
        result.forecast.to_markdown(index=False),
        "",
        "## 未来三个月阶段概率",
        "",
        pd.DataFrame([result.status_probs]).to_markdown(index=False),
        "",
        "## T+1 量价情景操作计划",
        "",
        (result.t1_scenario_plan if result.t1_scenario_plan is not None else pd.DataFrame()).to_markdown(index=False),
        "",
        "## 3日卖出/持有模型建议",
        "",
        pd.DataFrame([result.sell_model_summary or {}]).to_markdown(index=False),
        "",
        (result.sell_model_plan if result.sell_model_plan is not None else pd.DataFrame()).to_markdown(index=False),
        "",
        "## Top 20 历史相似片段",
        "",
        top_cases.to_markdown(index=False),
        "",
        ]
    )
    return "\n".join(lines)


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros_like(arr, dtype=float)
    mean = float(finite.mean())
    std = float(finite.std())
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    out = (arr - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _nan_stat(values: np.ndarray, kind: str) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0
    if kind == "std":
        return float(finite.std())
    return float(finite.mean())


def _round(value: object, ndigits: int = 2) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, ndigits)
