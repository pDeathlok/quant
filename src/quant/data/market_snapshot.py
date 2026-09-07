"""Sealed canonical row exports for consistent reads across spawned workers."""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import shutil
import stat
import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import pandas as pd

MANIFEST_ENV = "QUANT_PINNED_MARKET_MANIFEST"
FINGERPRINT_ENV = "QUANT_PINNED_MARKET_FINGERPRINT"


class MarketSnapshotError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return _handle_digest(handle)


def _handle_digest(handle: BinaryIO) -> str:
    result = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def _identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _date_blocks(frame: pd.DataFrame) -> list[dict]:
    """Monthly row-content identities, independent of parquet/chunk encoding."""
    import pyarrow as pa

    frame = frame.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    columns = sorted(frame.columns)
    table = pa.Table.from_pandas(frame[columns], preserve_index=False).replace_schema_metadata(None)
    blocks = []
    for month, indices in frame.groupby(frame["trade_date"].str[:6], sort=True).indices.items():
        start, length = int(indices[0]), len(indices)
        # Compact slice buffers: IPC may otherwise include neighboring string
        # buffer bytes, making an unchanged month depend on a later append.
        block = table.take(pa.array(indices, type=pa.int64()))
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, block.schema) as writer:
            writer.write_table(block)
        blocks.append({
            "partition": month, "min_date": frame["trade_date"].iloc[start],
            "max_date": frame["trade_date"].iloc[start + length - 1], "rows": length,
            "sha256": hashlib.sha256(sink.getvalue()).hexdigest(),
        })
    return blocks


def _date(value) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float, bool)) or not str(value).strip():
            raise ValueError("invalid date")
        parsed = pd.Timestamp(value)
        if pd.isna(parsed):
            raise ValueError("missing date")
        return parsed.strftime("%Y%m%d")
    except (ValueError, TypeError, OverflowError) as exc:
        raise MarketSnapshotError("Invalid snapshot date") from exc


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


_DIGESTS: OrderedDict[tuple, str] = OrderedDict()
_DIGEST_LOCK = threading.Lock()
_DIGEST_CACHE_SIZE = 16384


@contextmanager
def _verified_file(path: Path, expected: str) -> Iterator[BinaryIO]:
    """Hash and decode one open inode; reject replacement or in-place writes."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("not a regular file")
            identity = _stat_identity(before)
            key = (str(path), identity)
            with _DIGEST_LOCK:
                digest = _DIGESTS.get(key)
            if digest is None:
                digest = _handle_digest(handle)
            if _stat_identity(os.fstat(handle.fileno())) != identity or digest != expected:
                raise ValueError("file changed")
            handle.seek(0)
            yield handle
            if (
                _stat_identity(os.fstat(handle.fileno())) != identity
                or _stat_identity(path.lstat()) != identity
            ):
                raise ValueError("file changed during read")
            with _DIGEST_LOCK:
                _DIGESTS[key] = digest
                _DIGESTS.move_to_end(key)
                while len(_DIGESTS) > _DIGEST_CACHE_SIZE:
                    _DIGESTS.popitem(last=False)
    except (OSError, ValueError) as exc:
        raise MarketSnapshotError("Pinned canonical chunk is unavailable or changed") from exc


def _read_verified_parquet(path: Path, expected: str, **kwargs) -> pd.DataFrame:
    if any(kwargs.get(key) is not None for key in ("filesystem", "storage_options")):
        raise MarketSnapshotError("Pinned reads cannot override the local filesystem")
    with _verified_file(path, expected) as handle:
        frame = pd.read_parquet(handle, **kwargs)
    return frame


@dataclass(frozen=True)
class SealedMarketSnapshot:
    manifest_path: Path
    root: Path
    fingerprint: str


_PIN: contextvars.ContextVar[SealedMarketSnapshot | None] = contextvars.ContextVar("market_snapshot", default=None)
_READERS: OrderedDict[tuple, SnapshotReader] = OrderedDict()
_READER_LOCK = threading.Lock()
_READER_CACHE_SIZE = 4


def _manifest_identity(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("manifest is not a regular file")
    return _stat_identity(value)


def _remember_reader(reader: SnapshotReader) -> None:
    key = (reader.manifest_path, reader.manifest["fingerprint"], reader._manifest_stat)
    _READERS[key] = reader
    _READERS.move_to_end(key)
    while len(_READERS) > _READER_CACHE_SIZE:
        _READERS.popitem(last=False)


def _cached_reader(manifest_path: Path, expected_fingerprint: str) -> SnapshotReader:
    path = manifest_path.absolute()
    try:
        before = _manifest_identity(path)
        key = (path, expected_fingerprint, before)
        # Serialize cache misses so concurrent stock workers parse only once.
        with _READER_LOCK:
            reader = _READERS.get(key)
            if reader is None:
                reader = SnapshotReader(path, expected_fingerprint=expected_fingerprint)
            if (
                reader.manifest["fingerprint"] != expected_fingerprint
                or reader._manifest_stat != before
                or _manifest_identity(path) != before
            ):
                raise ValueError("pinned manifest changed")
            _remember_reader(reader)
            return reader
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise MarketSnapshotError("Pinned canonical manifest is unavailable or invalid") from exc


def pinned_market_environment() -> dict[str, str]:
    pin = _PIN.get()
    if pin is not None:
        return {MANIFEST_ENV: str(pin.manifest_path), FINGERPRINT_ENV: pin.fingerprint}
    path, fingerprint = os.getenv(MANIFEST_ENV), os.getenv(FINGERPRINT_ENV)
    if path is None and fingerprint is None:
        return {}
    if not path or not fingerprint or not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
        raise MarketSnapshotError("Pinned market manifest and expected fingerprint are required together")
    return {MANIFEST_ENV: path, FINGERPRINT_ENV: fingerprint}


@contextmanager
def pinned_market_snapshot(manifest_path: Path) -> Iterator[SealedMarketSnapshot]:
    reader = SnapshotReader(manifest_path)
    with _READER_LOCK:
        _remember_reader(reader)
    pin = SealedMarketSnapshot(reader.manifest_path, reader.root, reader.manifest["fingerprint"])
    token = _PIN.set(pin)
    try:
        yield pin
    finally:
        _PIN.reset(token)


def current_market_snapshot() -> SnapshotReader | None:
    env = pinned_market_environment()
    return _cached_reader(Path(env[MANIFEST_ENV]), env[FINGERPRINT_ENV]) if env else None


def pinned_dataset_path(dataset: str) -> Path | None:
    """Redirect legacy per-date file consumers only to explicitly sealed files."""
    reader = current_market_snapshot()
    if reader is None:
        return None
    entry = reader.manifest["datasets"].get(dataset)
    if not entry or entry.get("source_kind") != "declared_file_capture":
        raise MarketSnapshotError(f"No sealed per-date file transport: {dataset}")
    expected = set()
    for chunk in entry["files"]:
        relative = Path(chunk["path"])
        path = reader.root / relative
        try:
            if relative.parent != Path(dataset) or path.is_symlink():
                raise ValueError("invalid sealed file path")
            reader._chunk_path(chunk)
            with _verified_file(path, chunk["sha256"]):
                pass
        except (OSError, ValueError) as exc:
            raise MarketSnapshotError("Sealed file source changed or is unavailable") from exc
        expected.add(path)
    directory = reader.root / dataset
    if directory.is_symlink() or not directory.is_dir() or set(directory.glob("*.parquet")) != expected:
        raise MarketSnapshotError("Sealed file source membership changed")
    return directory


def read_pinned_parquet(path: Path | str, **kwargs) -> pd.DataFrame:
    """Read a declared sealed file under a pin, or ordinary parquet without one.

    Legacy callers should use a file under ``pinned_dataset_path(dataset)``.
    A mutable/canonical path is never silently redirected or accepted in a pin.
    """
    reader = current_market_snapshot()
    if reader is None:
        return pd.read_parquet(path, **kwargs)
    path = Path(path).absolute()
    for entry in reader.manifest["datasets"].values():
        for chunk in entry["files"]:
            if reader.root / chunk["path"] == path:
                return _read_verified_parquet(reader._chunk_path(chunk), chunk["sha256"], **kwargs)
    raise MarketSnapshotError("Parquet path is not declared in the pinned market snapshot")


class SnapshotReader:
    def __init__(self, manifest_path: Path, *, expected_fingerprint: str | None = None):
        self.manifest_path = Path(manifest_path).absolute()
        self.root = self.manifest_path.parent
        try:
            fd = os.open(self.manifest_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as handle:
                before = os.fstat(handle.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("manifest is not a regular file")
                self._manifest_stat = _stat_identity(before)
                payload = handle.read()
                if _stat_identity(os.fstat(handle.fileno())) != self._manifest_stat:
                    raise ValueError("manifest changed during read")
            self.manifest = json.loads(payload)
            if not isinstance(self.manifest, dict):
                raise ValueError("manifest is not a mapping")
            content = {key: value for key, value in self.manifest.items() if key != "fingerprint"}
            if self.manifest.get("schema") != 1 or self.manifest.get("status") != "sealed":
                raise ValueError("unsealed source")
            if _identity(content) != self.manifest["fingerprint"]:
                raise ValueError("manifest identity mismatch")
            if expected_fingerprint is not None and self.manifest["fingerprint"] != expected_fingerprint:
                raise ValueError("pinned manifest was replaced")
            datasets = self.manifest["datasets"]
            if not isinstance(datasets, dict) or not datasets:
                raise ValueError("missing dataset declarations")
            for name, entry in datasets.items():
                if not re.fullmatch(r"[A-Za-z0-9_]+", name) or not isinstance(entry, dict):
                    raise ValueError("invalid dataset declaration")
                if not isinstance(entry.get("files"), list) or not isinstance(entry.get("columns"), list):
                    raise ValueError("invalid dataset schema")
                if not {"ts_code", "trade_date"} <= set(entry["columns"]):
                    raise ValueError("missing identity columns")
                if _date(entry.get("start_date")) != entry.get("start_date"):
                    raise ValueError("noncanonical coverage date")
            if _manifest_identity(self.manifest_path) != self._manifest_stat:
                raise ValueError("manifest changed during validation")
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise MarketSnapshotError("Pinned canonical manifest is unavailable or invalid") from exc

    def _chunk_path(self, chunk: dict) -> Path:
        try:
            relative = Path(chunk["path"])
            path = self.root / relative
            if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
                raise ValueError("invalid chunk path")
            path.resolve().relative_to(self.root.resolve())
            return path
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise MarketSnapshotError("Pinned canonical chunk path is invalid") from exc

    def available(self, dataset: str, symbols=None, columns=None) -> pd.DataFrame:
        if dataset not in self.manifest["datasets"]:
            raise MarketSnapshotError(f"Dataset was not sealed: {dataset}")
        entry = self.manifest["datasets"][dataset]
        return self.read(dataset, start_date=entry.get("start_date"),
                         symbols=symbols if symbols is not None else entry.get("symbols"), columns=columns)

    def _metadata_chunks(self, dataset: str, symbol: str | None = None) -> list[dict]:
        try:
            entry = self.manifest["datasets"][dataset]
        except KeyError as exc:
            raise MarketSnapshotError(f"Dataset was not sealed: {dataset}") from exc
        if symbol is not None and entry.get("symbols") is not None and symbol not in entry["symbols"]:
            raise MarketSnapshotError("Requested universe is outside the sealed coverage")
        chunks = [chunk for chunk in entry["files"] if symbol is None or symbol in chunk["symbols"]]
        for chunk in chunks:
            with _verified_file(self._chunk_path(chunk), chunk["sha256"]):
                pass
        return chunks

    def list_symbols(self, dataset: str) -> list[str]:
        return sorted({symbol for chunk in self._metadata_chunks(dataset) for symbol in chunk["symbols"]})

    def latest_date(self, dataset: str, symbol: str | None = None) -> pd.Timestamp | None:
        chunks = self._metadata_chunks(dataset, symbol)
        if not chunks:
            return None
        if symbol is None or all(chunk["symbols"] == [symbol] for chunk in chunks):
            return pd.Timestamp(max(chunk["max_date"] for chunk in chunks))
        # Mixed-symbol supplemental chunks need a projected key/date read.
        dates = self.available(dataset, symbols=[symbol], columns=["trade_date"])["trade_date"]
        return pd.to_datetime(dates, errors="raise").max() if not dates.empty else None

    def read(self, dataset: str, start_date=None, end_date=None, symbols=None, columns=None) -> pd.DataFrame:
        try:
            entry = self.manifest["datasets"][dataset]
        except KeyError as exc:
            raise MarketSnapshotError(f"Dataset was not sealed: {dataset}") from exc
        requested_start, requested_end = _date(start_date), _date(end_date)
        if requested_start and requested_end and requested_start > requested_end:
            raise MarketSnapshotError("Snapshot date range is reversed")
        covered_start = entry.get("start_date")
        if covered_start and (not requested_start or requested_start < covered_start):
            raise MarketSnapshotError("Requested history is outside the sealed coverage")
        universe = entry.get("symbols")
        if universe is not None and (symbols is None or not set(symbols) <= set(universe)):
            raise MarketSnapshotError("Requested universe is outside the sealed coverage")
        frames = []
        for chunk in entry["files"]:
            if symbols is not None and not set(symbols).intersection(chunk["symbols"]):
                continue
            if requested_start and chunk["max_date"] < requested_start:
                continue
            if requested_end and chunk["min_date"] > requested_end:
                continue
            path = self._chunk_path(chunk)
            wanted = list(dict.fromkeys([*columns, "ts_code", "trade_date"])) if columns is not None else None
            filters = [("ts_code", "in", list(symbols))] if symbols is not None else []
            if requested_start:
                filters.append(("trade_date", ">=", requested_start))
            if requested_end:
                filters.append(("trade_date", "<=", requested_end))
            frame = _read_verified_parquet(path, chunk["sha256"], columns=wanted, filters=filters or None)
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=entry["columns"])
        if not frame.empty:
            dates = pd.to_datetime(frame["trade_date"].astype(str), errors="raise").dt.strftime("%Y%m%d")
            if requested_start:
                frame = frame.loc[dates >= requested_start]
            if requested_end:
                frame = frame.loc[dates <= requested_end]
            if symbols is not None:
                frame = frame.loc[frame["ts_code"].isin(symbols)]
        if columns is not None:
            missing = set(columns) - set(frame.columns)
            if missing:
                raise MarketSnapshotError(f"Missing sealed columns: {sorted(missing)}")
            frame = frame[list(columns)]
        return frame.reset_index(drop=True)


def export_market_snapshot(store, directory: Path, *, datasets=("daily",), start_date=None, symbols=None,
                           supplemental_sources=None) -> SealedMarketSnapshot:
    """Export SQL tables under one repeatable read; publish only sealed bytes.

    A revision journal is not assumed transactionally coupled to raw writes.
    Identity therefore derives from the actual exported content, not its label.
    Call after source refresh and retain the directory until all readers exit.
    """
    from quant.data.atomic_io import atomic_write_json

    if current_market_snapshot() is not None:
        raise MarketSnapshotError("Cannot export mutable sources inside an existing pin")
    start_date = _date(start_date)
    datasets = tuple(sorted(set(datasets)))
    supplemental_sources = {name: Path(path) for name, path in (supplemental_sources or {}).items()}
    if not set(supplemental_sources) <= set(datasets):
        raise ValueError("Supplemental datasets must be declared")
    if not datasets or any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in datasets):
        raise ValueError("Invalid snapshot datasets")
    directory = Path(directory).absolute()
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(f".{directory.name}-{uuid.uuid4().hex}.building")
    staging.mkdir()
    manifest = {"schema": 1, "status": "sealed", "datasets": {}}

    def save(dataset, frame, filename=None, *, symbol=None):
        if not {"ts_code", "trade_date"} <= set(frame.columns):
            raise MarketSnapshotError(f"Canonical source schema missing: {dataset}")
        dates = pd.to_datetime(frame["trade_date"].astype(str), format="mixed", errors="raise").dt.strftime("%Y%m%d")
        if dates.isna().any() or frame["ts_code"].isna().any():
            raise MarketSnapshotError(f"Canonical source has missing row identities: {dataset}")
        frame = frame.copy()
        frame["trade_date"] = dates
        entry = manifest["datasets"].setdefault(dataset, {
            "columns": list(frame.columns), "files": [],
            "start_date": start_date,
            "symbols": sorted(symbols) if symbols is not None else None,
        })
        if frame.empty:
            return
        index = len(entry["files"])
        if symbol is not None:
            symbol_key = hashlib.sha256(str(symbol).encode()).hexdigest()
            relative = Path(f"{dataset}_partitioned") / f"symbol-{symbol_key}.parquet"
        else:
            relative = Path(dataset) / filename if filename else Path(f"{dataset}_partitioned") / f"part-{index:06d}.parquet"
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        chunk = {"path": str(relative), "sha256": _digest(path), "rows": len(frame),
                 "min_date": dates.min(), "max_date": dates.max(),
                 "symbols": sorted(frame["ts_code"].dropna().unique().tolist())}
        if symbol is not None:
            chunk.update(date_blocks_schema=1, date_blocks=_date_blocks(frame))
        entry["files"].append(chunk)

    try:
        if store.config.backend in {"sql", "mysql"}:
            from sqlalchemy import bindparam, inspect, text

            engine = store._engine()
            with engine.connect() as connection:
                dialect = connection.dialect.name
                if dialect in {"mysql", "mariadb"}:
                    connection = connection.execution_options(isolation_level="REPEATABLE READ")
                    connection.exec_driver_sql("START TRANSACTION WITH CONSISTENT SNAPSHOT")
                elif dialect == "sqlite":
                    connection.exec_driver_sql("BEGIN")
                else:
                    raise MarketSnapshotError(f"Unsupported snapshot isolation: {dialect}")
                for dataset in datasets:
                    if dataset in supplemental_sources:
                        continue
                    table = store._dataset_table_name(dataset)
                    if not inspect(connection).has_table(table):
                        raise MarketSnapshotError(f"Canonical table missing: {table}")
                    if dialect in {"mysql", "mariadb"}:
                        storage = connection.execute(text("SELECT ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:table"), {"table": table}).scalar_one()
                        if str(storage).lower() != "innodb":
                            raise MarketSnapshotError("Canonical snapshot requires transactional InnoDB tables")
                    filters, parameters = [], {}
                    if start_date:
                        filters.append("trade_date >= :start")
                        parameters["start"] = pd.Timestamp(start_date).date()
                    if symbols is not None:
                        filters.append("ts_code IN :symbols")
                        parameters["symbols"] = list(symbols)
                    ordering = "ts_code, trade_date" if dataset == "daily" else "trade_date, ts_code"
                    query = text(f"SELECT * FROM `{table}`" + (" WHERE " + " AND ".join(filters) if filters else "") + f" ORDER BY {ordering}")
                    if symbols is not None:
                        query = query.bindparams(bindparam("symbols", expanding=True))
                    pending = None
                    for frame in pd.read_sql_query(query, connection.execution_options(stream_results=True), params=parameters, chunksize=100000):
                        if dataset != "daily" or frame.empty:
                            save(dataset, frame)
                            continue
                        if frame["ts_code"].isna().any():
                            raise MarketSnapshotError("Canonical daily source has missing symbols")
                        # Only the final symbol can continue in the next SQL batch.
                        # Stable per-symbol files prevent unrelated append invalidation.
                        if pending is not None:
                            frame = pd.concat([pending, frame], ignore_index=True)
                        last_symbol = frame["ts_code"].iloc[-1]
                        tail = frame["ts_code"].eq(last_symbol)
                        for symbol, complete in frame.loc[~tail].groupby("ts_code", sort=False):
                            save(dataset, complete, symbol=symbol)
                        pending = frame.loc[tail].copy()
                    if pending is not None:
                        save(dataset, pending, symbol=pending["ts_code"].iloc[0])
                connection.rollback()
        else:
            # Explicit file sources must remain byte-identical through capture.
            file_datasets = [dataset for dataset in datasets if dataset not in supplemental_sources]
            paths = sorted({path for dataset in file_datasets for path in (store.config.root / f"{dataset}_partitioned").glob("**/*.parquet")})
            before = {path: _digest(path) for path in paths}
            for dataset in file_datasets:
                dataset_paths = sorted((store.config.root / f"{dataset}_partitioned").glob("**/*.parquet"))
                if not dataset_paths:
                    raise MarketSnapshotError(f"Canonical partitions missing: {dataset}")
                for path in dataset_paths:
                    frame = pd.read_parquet(path)
                    if start_date:
                        frame = frame.loc[pd.to_datetime(frame["trade_date"].astype(str)) >= pd.Timestamp(start_date)]
                    if symbols is not None:
                        frame = frame.loc[frame["ts_code"].isin(symbols)]
                    save(dataset, frame)
            after_paths = sorted({path for dataset in file_datasets for path in (store.config.root / f"{dataset}_partitioned").glob("**/*.parquet")})
            if paths != after_paths or any(_digest(path) != digest for path, digest in before.items()):
                raise MarketSnapshotError("Canonical files changed during export")
        for dataset, source in supplemental_sources.items():
            if not source.is_dir() or source.is_symlink():
                raise MarketSnapshotError(f"Declared file source missing or unsafe: {dataset}")
            paths = sorted(source.glob("*.parquet"))
            if any(path.is_symlink() for path in paths):
                raise MarketSnapshotError("Symlink supplemental source")
            before = {path: _digest(path) for path in paths}
            (staging / dataset).mkdir()
            if not paths:
                save(dataset, pd.DataFrame(columns=["ts_code", "trade_date"]))
            for path in paths:
                frame = pd.read_parquet(path)
                if start_date:
                    frame = frame.loc[pd.to_datetime(frame["trade_date"].astype(str)) >= pd.Timestamp(start_date)]
                if symbols is not None:
                    frame = frame.loc[frame["ts_code"].isin(symbols)]
                save(dataset, frame, filename=path.name)
            if paths != sorted(source.glob("*.parquet")) or any(_digest(path) != digest for path, digest in before.items()):
                raise MarketSnapshotError(f"Declared file source changed during export: {dataset}")
            manifest["datasets"][dataset]["source_kind"] = "declared_file_capture"
        if set(manifest["datasets"]) != set(datasets):
            raise MarketSnapshotError("Incomplete canonical export")
        manifest["fingerprint"] = _identity(manifest)
        atomic_write_json(manifest, staging / "manifest.json")
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return SealedMarketSnapshot(directory / "manifest.json", directory, manifest["fingerprint"])
