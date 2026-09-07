from __future__ import annotations

import fcntl
import filecmp
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from quant.infrastructure.artifact_registry import ArtifactRegistry


LONG_CACHE_PATTERNS = (
    "daily_returns_*.parquet",
    "daily_monthly_features_*.parquet",
)
LONG_CACHE_RETENTION_VERSIONS = 2
TUSHARE_SINGLE_SYMBOL_CACHE_RETENTION_DAYS = 7
ROOT_MARKET_REQUEST_CACHE_RETENTION_DAYS = 30
TUSHARE_LIVE_PROBE_RETENTION_DAYS = 2
ABANDONED_CACHE_RETENTION_DAYS = 2
FACTOR_SCHEMA_MIN_COVERAGE_RATIO = 0.98
TUSHARE_SINGLE_SYMBOL_CACHE_PATTERN = re.compile(
    r"^tushare_\d{6}\.(?:SH|SZ|BJ)_\d{8}_\d{8}_(?:None|qfq|hfq)\.parquet$"
)
ROOT_MARKET_REQUEST_CACHE_PATTERN = re.compile(
    r"^(?:(?:sh|sz|bj)\d{6}|tushare_\d{6}\.(?:SH|SZ|BJ))_"
    r"\d{8}_\d{8}_(?:None|qfq|hfq)\.parquet$"
)
ROOT_MARKET_REQUEST_CACHE_PROTECTED_PATTERN = re.compile(
    r"^(?:sz002594|tushare_002594\.SZ)_"
)
TUSHARE_DAILY_BASIC_CACHE_PATTERN = re.compile(
    r"^tushare_daily_basic_(\d{8})\.parquet$"
)
TUSHARE_CB_DAILY_CACHE_PATTERN = re.compile(
    r"^tushare_cb_daily_(\d{8})_all_all_all\.parquet$"
)
CONSOLIDATED_CB_DAILY_PATTERN = re.compile(
    r"^cb_daily_(\d{8})_(\d{8})\.parquet$"
)
SIMILAR_PATTERN_SMOKE_CACHE_DIRECTORIES = (
    "vector_cache_smoke",
    "vector_cache_model_smoke",
)
STRATEGY_SNAPSHOT_RETENTION_DAYS = 30
STRATEGY_SNAPSHOT_MAX_VERSIONS = 10
WORKSPACE_SNAPSHOT_RETENTION_DAYS = 14
WORKSPACE_SNAPSHOT_MAX_VERSIONS = 3
SOURCE_AUDIT_RETENTION_DAYS = 30
SOURCE_AUDIT_MAX_RUNS = 10
ROUTINE_RUN_RETENTION_DAYS = 14
ROUTINE_RUN_MAX_RUNS = 5
RUN_DIRECTORY_PATTERN = re.compile(r"^(\d{8}_\d{6})(?:_|$)")
B1_REPORT_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<timestamp>\d{8}_\d{6})(?P<suffix>\.[^.]+)$"
)
B1_REPORT_RETENTION_VERSIONS = 2
B1_REPORT_LARGE_SAMPLE_RETENTION_VERSIONS = 1
B1_REPORT_LARGE_SAMPLE_PREFIXES = {
    "z_skill_trade_samples",
    "z_skill_model_trade_samples",
}
B1_REPORT_LATEST_ALIASES = {
    ("z_skill_trade_samples", ".csv"): "latest_z_skill_trade_samples.csv",
    ("z_skill_model_trade_samples", ".csv"): "latest_z_skill_model_trade_samples.csv",
    (
        "b1_xgb_entry_exit_grid_non_overlap_summary",
        ".csv",
    ): "latest_non_overlap_summary.csv",
    ("b1_xgb_entry_exit_grid_summary", ".csv"): "latest_summary.csv",
}
SQL_SNAPSHOT_TABLES = (
    {
        "name": "selector_snapshots",
        "date_field": "signal_date",
        "group_fields": ("strategies_key", "include_extended"),
        "latest_values": {"LATEST"},
        "retention_days": STRATEGY_SNAPSHOT_RETENTION_DAYS,
        "max_versions": STRATEGY_SNAPSHOT_MAX_VERSIONS,
    },
    {
        "name": "long_stock_pool_snapshots",
        "date_field": "signal_date",
        "group_fields": ("variant",),
        "latest_values": {"latest"},
        "retention_days": STRATEGY_SNAPSHOT_RETENTION_DAYS,
        "max_versions": STRATEGY_SNAPSHOT_MAX_VERSIONS,
    },
    {
        "name": "web_workspace_snapshots",
        "date_field": "snapshot_date",
        "group_fields": ("workspace", "params_key"),
        "latest_values": {"latest"},
        "retention_days": WORKSPACE_SNAPSHOT_RETENTION_DAYS,
        "max_versions": WORKSPACE_SNAPSHOT_MAX_VERSIONS,
    },
)


@dataclass(frozen=True)
class _StoragePathSpec:
    name: str
    relative_path: str
    direct_files_only: bool = False


MANAGED_CACHE_STORAGE_PATHS = (
    _StoragePathSpec("root_market_requests", "data/cache", direct_files_only=True),
    _StoragePathSpec("source_merge_requests", "data/cache/source_merge"),
    _StoragePathSpec("daily_factor_layer", "data/features/daily_factor_layer"),
    _StoragePathSpec(
        "similar_pattern_vectors",
        "data/research/similar_patterns/vector_cache",
    ),
    _StoragePathSpec("long_strategy", "data/research/long_dividend_quality"),
    _StoragePathSpec(
        "convertible_bond_requests",
        "data/convertible_bond/tushare/tushare_cache",
    ),
    _StoragePathSpec("selector_snapshots", "data/selector_snapshots"),
    _StoragePathSpec("workspace_snapshots", "data/workspace_snapshots"),
    _StoragePathSpec("source_audit", "data/raw/source_audit"),
    _StoragePathSpec("routine_runs", "data/routine"),
    _StoragePathSpec(
        "b1_versioned_reports",
        "reports/b1/research/xgb_project_vars_strategy",
    ),
)


def _storage_for_path(path: Path, *, direct_files_only: bool) -> dict[str, int]:
    if not path.exists():
        return {"files": 0, "logical_bytes": 0, "allocated_bytes": 0}
    candidates = path.iterdir() if direct_files_only else path.rglob("*")
    files = 0
    logical_bytes = 0
    allocated_bytes = 0
    seen_inodes: set[tuple[int, int]] = set()
    for child in candidates:
        if not child.is_file() or child.is_symlink():
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        files += 1
        logical_bytes += int(stat.st_size)
        inode = (int(stat.st_dev), int(stat.st_ino))
        if inode in seen_inodes:
            continue
        seen_inodes.add(inode)
        allocated_bytes += int(getattr(stat, "st_blocks", 0)) * 512
    return {
        "files": files,
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
    }


def _managed_cache_storage(root: Path) -> dict[str, Any]:
    categories = {
        spec.name: _storage_for_path(
            root / spec.relative_path,
            direct_files_only=spec.direct_files_only,
        )
        for spec in MANAGED_CACHE_STORAGE_PATHS
    }
    return {
        "files": sum(item["files"] for item in categories.values()),
        "logical_bytes": sum(item["logical_bytes"] for item in categories.values()),
        "allocated_bytes": sum(item["allocated_bytes"] for item in categories.values()),
        "categories": categories,
    }


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate_identical_files(
    paths: Iterable[Path],
    *,
    error_prefix: str,
    errors: list[str],
) -> dict[str, int]:
    size_groups: dict[int, list[Path]] = {}
    for path in paths:
        try:
            if path.is_file() and not path.is_symlink():
                size_groups.setdefault(path.stat().st_size, []).append(path)
        except OSError as exc:
            errors.append(f"{error_prefix}:{path.name}:{exc}")

    deduplicated_files = 0
    deduplicated_bytes = 0
    for size, candidates in size_groups.items():
        if len(candidates) < 2:
            continue
        digest_groups: dict[str, list[Path]] = {}
        for path in candidates:
            try:
                digest_groups.setdefault(_file_sha256(path), []).append(path)
            except OSError as exc:
                errors.append(f"{error_prefix}:{path.name}:{exc}")
        for duplicates in digest_groups.values():
            if len(duplicates) < 2:
                continue
            canonical = duplicates[0]
            for path in duplicates[1:]:
                temp_path = path.with_name(f".{path.name}.dedup-{os.getpid()}")
                try:
                    canonical_stat = canonical.stat()
                    path_stat = path.stat()
                    if (canonical_stat.st_dev, canonical_stat.st_ino) == (
                        path_stat.st_dev,
                        path_stat.st_ino,
                    ):
                        continue
                    if not filecmp.cmp(canonical, path, shallow=False):
                        continue
                    os.link(canonical, temp_path)
                    os.replace(temp_path, path)
                    deduplicated_files += 1
                    if path_stat.st_nlink == 1:
                        deduplicated_bytes += size
                except OSError as exc:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    errors.append(f"{error_prefix}:{path.name}:{exc}")
    return {
        "deduplicated_files": deduplicated_files,
        "deduplicated_bytes": deduplicated_bytes,
    }


def _current_factor_schema_version() -> str:
    from quant.features.daily_factor_layer import SIGNAL_FACTOR_LAYER_VERSION

    return str(SIGNAL_FACTOR_LAYER_VERSION)


def _factor_schema_symbol_count(path: Path) -> int:
    if not path.is_dir() or path.is_symlink():
        return 0
    return sum(
        1
        for child in path.iterdir()
        if child.is_dir()
        and not child.is_symlink()
        and (child / "state.json").is_file()
        and any(child.glob("[0-9][0-9][0-9][0-9].parquet"))
    )


def _cleanup_factor_schema_caches(root: Path, errors: list[str]) -> dict[str, Any]:
    factor_root = root / "data/features/daily_factor_layer"
    current_version = _current_factor_schema_version()
    directories = sorted(
        (
            path
            for path in factor_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: path.name,
    ) if factor_root.exists() else []
    symbol_counts = {
        path.name: _factor_schema_symbol_count(path)
        for path in directories
    }
    current_count = symbol_counts.get(current_version, 0)
    previous_counts = [
        count
        for version, count in symbol_counts.items()
        if version != current_version and count > 0
    ]
    required_count = (
        max(1, math.ceil(max(previous_counts) * FACTOR_SCHEMA_MIN_COVERAGE_RATIO))
        if previous_counts
        else 1
    )
    current_ready = current_count >= required_count
    deleted_directories = 0
    reclaimed_bytes = 0
    if current_ready:
        for path in directories:
            if path.name == current_version:
                continue
            try:
                with ArtifactRegistry(root).deletion_guard(path, require_owned=False) as reason:
                    if reason:
                        continue
                    size = _directory_size(path)
                    shutil.rmtree(path)
                    deleted_directories += 1
                    reclaimed_bytes += size
            except OSError as exc:
                errors.append(f"daily_factor_schema:{path.name}:{exc}")
    return {
        "current_version": current_version,
        "current_symbol_count": current_count,
        "required_symbol_count": required_count,
        "coverage_threshold": FACTOR_SCHEMA_MIN_COVERAGE_RATIO,
        "current_ready": current_ready,
        "schema_symbol_counts": symbol_counts,
        "deleted_directories": deleted_directories,
        "reclaimed_bytes": reclaimed_bytes,
    }


def _cleanup_root_market_request_cache(
    root: Path,
    today: date,
    errors: list[str],
) -> dict[str, Any]:
    cache_dir = root / "data/cache"
    cutoff = today - timedelta(days=ROOT_MARKET_REQUEST_CACHE_RETENTION_DAYS)
    deleted_files = 0
    reclaimed_bytes = 0
    protected_files = 0
    for path in sorted(cache_dir.glob("*.parquet")):
        if ROOT_MARKET_REQUEST_CACHE_PATTERN.fullmatch(path.name) is None:
            continue
        if ROOT_MARKET_REQUEST_CACHE_PROTECTED_PATTERN.match(path.name):
            protected_files += 1
            continue
        try:
            if datetime.fromtimestamp(path.stat().st_mtime).date() >= cutoff:
                continue
            size = path.stat().st_size
            path.unlink()
            deleted_files += 1
            reclaimed_bytes += size
        except OSError as exc:
            errors.append(f"root_market_request:{path.name}:{exc}")
    return {
        "retention_days": ROOT_MARKET_REQUEST_CACHE_RETENTION_DAYS,
        "cutoff_date": cutoff.isoformat(),
        "protected_production_files": protected_files,
        "deleted_files": deleted_files,
        "reclaimed_bytes": reclaimed_bytes,
    }


def _cleanup_daily_basic_cache_directory(
    cache_dir: Path,
    raw_dir: Path,
    *,
    cutoff: date,
    error_prefix: str,
    errors: list[str],
) -> dict[str, int]:
    deleted_files = 0
    reclaimed_bytes = 0
    protected_files = 0
    for path in sorted(cache_dir.glob("tushare_daily_basic_*.parquet")):
        match = TUSHARE_DAILY_BASIC_CACHE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        trade_date_text = match.group(1)
        try:
            trade_date = datetime.strptime(trade_date_text, "%Y%m%d").date()
        except ValueError:
            continue
        if trade_date >= cutoff:
            continue
        raw_path = raw_dir / f"{trade_date_text}.parquet"
        try:
            if not raw_path.is_file() or raw_path.stat().st_size <= 0:
                protected_files += 1
                continue
            size = path.stat().st_size
            path.unlink()
            deleted_files += 1
            reclaimed_bytes += size
        except OSError as exc:
            errors.append(f"{error_prefix}:{path.name}:{exc}")
    return {
        "deleted_files": deleted_files,
        "protected_without_raw": protected_files,
        "reclaimed_bytes": reclaimed_bytes,
    }


def _cleanup_abandoned_cache_builds(
    root: Path,
    today: date,
    errors: list[str],
    *,
    protected_paths: Iterable[Path | str] = (),
) -> dict[str, Any]:
    cutoff = today - timedelta(days=ABANDONED_CACHE_RETENTION_DAYS)
    deleted_files = 0
    deleted_directories = 0
    reclaimed_bytes = 0
    kept_paths: dict[str, str] = {}
    registry = ArtifactRegistry(root)
    cache_roots = (
        root / "data/features/daily_factor_layer",
        root / "data/research/similar_patterns/vector_cache",
    )
    for cache_root in cache_roots:
        if not cache_root.exists():
            continue
        for path in sorted(cache_root.rglob("*"), reverse=True):
            name = path.name
            is_temp_file = path.is_file() and (
                name.endswith(".tmp")
                or ".tmp." in name
                or name.endswith(".tmp.parquet")
                or name.endswith(".partial")
            )
            is_build_directory = (
                path.is_dir()
                and not path.is_symlink()
                and name.startswith(".")
                and ".building-" in name
            )
            if not is_temp_file and not is_build_directory:
                continue
            try:
                with registry.deletion_guard(
                    path,
                    older_than=datetime.combine(cutoff, datetime.min.time()).timestamp(),
                    protected_paths=protected_paths,
                ) as reason:
                    if reason:
                        kept_paths[str(path.relative_to(root))] = reason
                        continue
                    size = path.stat().st_size if is_temp_file else _directory_size(path)
                    if is_temp_file:
                        path.unlink()
                        deleted_files += 1
                    else:
                        shutil.rmtree(path)
                        deleted_directories += 1
                    reclaimed_bytes += size
            except OSError as exc:
                errors.append(f"abandoned_cache_build:{path.name}:{exc}")
    return {
        "retention_days": ABANDONED_CACHE_RETENTION_DAYS,
        "cutoff_date": cutoff.isoformat(),
        "deleted_files": deleted_files,
        "deleted_directories": deleted_directories,
        "reclaimed_bytes": reclaimed_bytes,
        "kept_paths": kept_paths,
    }


def _parse_snapshot_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if len(text) == 8 and text.isdigit():
            return datetime.strptime(text, "%Y%m%d").date()
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _snapshot_keys_to_delete(
    records: Iterable[Mapping[str, Any]],
    *,
    date_field: str,
    group_fields: tuple[str, ...],
    cutoff: date,
    max_versions: int,
    latest_values: set[str] | None = None,
    deduplicate_dates: bool = True,
) -> list[str]:
    """Select expired snapshot keys while always protecting each group's newest date."""

    latest_values = latest_values or set()
    grouped: dict[tuple[str, ...], list[tuple[date, str, str]]] = {}
    for record in records:
        snapshot_key = str(record.get("snapshot_key") or "")
        raw_date = str(record.get(date_field) or "")
        if not snapshot_key or raw_date in latest_values:
            continue
        parsed_date = _parse_snapshot_date(raw_date)
        if parsed_date is None:
            continue
        group = tuple(str(record.get(field) or "") for field in group_fields)
        updated_at = str(record.get("updated_at") or record.get("generated_at") or "")
        grouped.setdefault(group, []).append((parsed_date, updated_at, snapshot_key))

    to_delete: list[str] = []
    for group_records in grouped.values():
        ordered = sorted(group_records, reverse=True)
        kept_dates: set[date] = set()
        for parsed_date, _, snapshot_key in ordered:
            is_newest = not kept_dates
            is_within_limit = len(kept_dates) < max_versions
            if parsed_date in kept_dates and deduplicate_dates:
                to_delete.append(snapshot_key)
                continue
            if parsed_date in kept_dates or is_newest or (parsed_date >= cutoff and is_within_limit):
                kept_dates.add(parsed_date)
                continue
            to_delete.append(snapshot_key)
    return to_delete


def _load_json_record(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _selector_snapshot_group(payload: Mapping[str, Any], path: Path) -> tuple[str, bool] | None:
    scope = payload.get("snapshot_scope")
    if not isinstance(scope, Mapping):
        scope = {}
    raw_strategies = scope.get("strategies")
    strategies = raw_strategies if isinstance(raw_strategies, list) else []
    strategy_key = ",".join(sorted(str(item).upper() for item in strategies)) or "ALL"
    if "include_extended" in scope:
        return strategy_key, bool(scope["include_extended"])
    if "include_extended" in payload:
        return strategy_key, bool(payload["include_extended"])

    schema_version = str(payload.get("selector_snapshot_schema_version") or "")
    signal_date = str(payload.get("signal_date") or "")
    if not schema_version or not signal_date:
        return None
    for include_extended in (False, True):
        raw = json.dumps(
            {
                "signal_date": signal_date,
                "strategies": strategy_key,
                "include_extended": include_extended,
                "schema_version": schema_version,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        if hashlib.sha1(raw.encode("utf-8")).hexdigest() == path.stem:
            return strategy_key, include_extended
    return None


def _delete_snapshot_files(
    records: list[dict[str, Any]],
    *,
    date_field: str,
    group_fields: tuple[str, ...],
    cutoff: date,
    max_versions: int,
    latest_values: set[str] | None,
    deduplicate_dates: bool = True,
    error_prefix: str,
    errors: list[str],
) -> tuple[int, int]:
    keys = set(
        _snapshot_keys_to_delete(
            records,
            date_field=date_field,
            group_fields=group_fields,
            cutoff=cutoff,
            max_versions=max_versions,
            latest_values=latest_values,
            deduplicate_dates=deduplicate_dates,
        )
    )
    deleted_files = 0
    reclaimed_bytes = 0
    for record in records:
        path = record.get("_path")
        if not isinstance(path, Path) or str(record.get("snapshot_key") or "") not in keys:
            continue
        try:
            size = path.stat().st_size
            path.unlink()
            deleted_files += 1
            reclaimed_bytes += size
        except OSError as exc:
            errors.append(f"{error_prefix}:{path.name}:{exc}")
    return deleted_files, reclaimed_bytes


def _cleanup_snapshot_files(root: Path, today: date, errors: list[str]) -> dict[str, Any]:
    strategy_cutoff = today - timedelta(days=STRATEGY_SNAPSHOT_RETENTION_DAYS)
    workspace_cutoff = today - timedelta(days=WORKSPACE_SNAPSHOT_RETENTION_DAYS)

    selector_records: list[dict[str, Any]] = []
    for path in sorted((root / "data/selector_snapshots").glob("*.json")):
        payload = _load_json_record(path)
        if payload is None:
            continue
        group = _selector_snapshot_group(payload, path)
        if group is None:
            continue
        strategy_key, include_extended = group
        selector_records.append(
            {
                "snapshot_key": path.stem,
                "signal_date": payload.get("signal_date"),
                "strategies_key": strategy_key,
                "include_extended": include_extended,
                "generated_at": payload.get("generated_at"),
                "_path": path,
            }
        )
    selector_deleted, selector_bytes = _delete_snapshot_files(
        selector_records,
        date_field="signal_date",
        group_fields=("strategies_key", "include_extended"),
        cutoff=strategy_cutoff,
        max_versions=STRATEGY_SNAPSHOT_MAX_VERSIONS,
        latest_values={"LATEST"},
        deduplicate_dates=False,
        error_prefix="selector_snapshots",
        errors=errors,
    )

    long_records: list[dict[str, Any]] = []
    for path in sorted((root / "data/long_stock_pool_snapshots").glob("*.json")):
        payload = _load_json_record(path)
        if payload is None:
            continue
        long_records.append(
            {
                "snapshot_key": path.stem,
                "signal_date": payload.get("signal_date"),
                "variant": payload.get("variant"),
                "generated_at": payload.get("generated_at"),
                "_path": path,
            }
        )
    long_deleted, long_bytes = _delete_snapshot_files(
        long_records,
        date_field="signal_date",
        group_fields=("variant",),
        cutoff=strategy_cutoff,
        max_versions=STRATEGY_SNAPSHOT_MAX_VERSIONS,
        latest_values={"latest"},
        deduplicate_dates=False,
        error_prefix="long_stock_pool_snapshots",
        errors=errors,
    )

    workspace_records: list[dict[str, Any]] = []
    workspace_root = root / "data/workspace_snapshots"
    for path in sorted(workspace_root.glob("*/*/*.json")):
        relative = path.relative_to(workspace_root)
        workspace_records.append(
            {
                "snapshot_key": str(relative),
                "snapshot_date": path.stem,
                "workspace": relative.parts[0],
                "params_key": relative.parts[1],
                "updated_at": path.stat().st_mtime_ns,
                "_path": path,
            }
        )
    workspace_deleted, workspace_bytes = _delete_snapshot_files(
        workspace_records,
        date_field="snapshot_date",
        group_fields=("workspace", "params_key"),
        cutoff=workspace_cutoff,
        max_versions=WORKSPACE_SNAPSHOT_MAX_VERSIONS,
        latest_values={"latest"},
        deduplicate_dates=False,
        error_prefix="workspace_snapshots",
        errors=errors,
    )
    for directory in sorted(workspace_root.glob("*/*"), reverse=True):
        try:
            directory.rmdir()
            directory.parent.rmdir()
        except OSError:
            continue

    return {
        "selector": {
            "retention_days": STRATEGY_SNAPSHOT_RETENTION_DAYS,
            "max_versions_per_group": STRATEGY_SNAPSHOT_MAX_VERSIONS,
            "deleted_files": selector_deleted,
            "reclaimed_bytes": selector_bytes,
        },
        "long_stock_pool": {
            "retention_days": STRATEGY_SNAPSHOT_RETENTION_DAYS,
            "max_versions_per_group": STRATEGY_SNAPSHOT_MAX_VERSIONS,
            "deleted_files": long_deleted,
            "reclaimed_bytes": long_bytes,
        },
        "workspace": {
            "retention_days": WORKSPACE_SNAPSHOT_RETENTION_DAYS,
            "max_versions_per_group": WORKSPACE_SNAPSHOT_MAX_VERSIONS,
            "deleted_files": workspace_deleted,
            "reclaimed_bytes": workspace_bytes,
        },
        "reclaimed_bytes": selector_bytes + long_bytes + workspace_bytes,
    }


def _cleanup_timestamped_directories(
    directory: Path,
    *,
    cutoff: date,
    max_runs: int,
    error_prefix: str,
    errors: list[str],
    group_by_suffix: bool = False,
) -> dict[str, int]:
    runs: list[tuple[datetime, Path, str]] = []
    if directory.exists():
        for path in directory.iterdir():
            match = RUN_DIRECTORY_PATTERN.match(path.name)
            if not path.is_dir() or not match:
                continue
            try:
                timestamp = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            suffix = path.name[len(match.group(1)) :].lstrip("_") or "default"
            runs.append((timestamp, path, suffix if group_by_suffix else "all"))
    runs.sort(reverse=True)
    deleted_directories = 0
    reclaimed_bytes = 0
    grouped: dict[str, list[tuple[datetime, Path]]] = {}
    for timestamp, path, group in runs:
        grouped.setdefault(group, []).append((timestamp, path))
    for group_runs in grouped.values():
        for index, (timestamp, path) in enumerate(group_runs):
            if index == 0 or (index < max_runs and timestamp.date() >= cutoff):
                continue
            try:
                size = _directory_size(path)
                shutil.rmtree(path)
                deleted_directories += 1
                reclaimed_bytes += size
            except OSError as exc:
                errors.append(f"{error_prefix}:{path.name}:{exc}")
    return {
        "deleted_directories": deleted_directories,
        "reclaimed_bytes": reclaimed_bytes,
    }


def _cleanup_cb_daily_request_cache(root: Path, errors: list[str]) -> dict[str, Any]:
    data_dir = root / "data/convertible_bond/tushare"
    consolidated_ranges: list[tuple[date, date]] = []
    for path in data_dir.glob("cb_daily_*.parquet"):
        match = CONSOLIDATED_CB_DAILY_PATTERN.fullmatch(path.name)
        if match is None or not path.is_file() or path.stat().st_size <= 0:
            continue
        try:
            consolidated_ranges.append(
                (
                    datetime.strptime(match.group(1), "%Y%m%d").date(),
                    datetime.strptime(match.group(2), "%Y%m%d").date(),
                )
            )
        except ValueError:
            continue

    deleted_files = 0
    reclaimed_bytes = 0
    protected_files = 0
    retained_paths: list[Path] = []
    cache_dir = data_dir / "tushare_cache"
    for path in sorted(cache_dir.glob("tushare_cb_daily_*.parquet")):
        match = TUSHARE_CB_DAILY_CACHE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        try:
            request_date = datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if not any(start <= request_date <= end for start, end in consolidated_ranges):
            protected_files += 1
            retained_paths.append(path)
            continue
        try:
            size = path.stat().st_size
            path.unlink()
            deleted_files += 1
            reclaimed_bytes += size
        except OSError as exc:
            errors.append(f"tushare_cb_daily:{path.name}:{exc}")
    deduplication = _deduplicate_identical_files(
        retained_paths,
        error_prefix="tushare_cb_daily_dedup",
        errors=errors,
    )
    return {
        "consolidated_ranges": [
            {"start": start.isoformat(), "end": end.isoformat()}
            for start, end in sorted(consolidated_ranges)
        ],
        "deleted_files": deleted_files,
        "protected_outside_consolidated_range": protected_files,
        **deduplication,
        "reclaimed_bytes": reclaimed_bytes + deduplication["deduplicated_bytes"],
    }


def _cleanup_b1_research_reports(root: Path, errors: list[str]) -> dict[str, Any]:
    from quant.data.atomic_io import atomic_link_or_copy

    report_dir = root / "reports/b1/research/xgb_project_vars_strategy"
    groups: dict[tuple[str, str], list[tuple[datetime, Path]]] = {}
    if report_dir.exists():
        for path in report_dir.iterdir():
            if not path.is_file():
                continue
            match = B1_REPORT_TIMESTAMP_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            try:
                timestamp = datetime.strptime(match.group("timestamp"), "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            key = (match.group("prefix"), match.group("suffix"))
            groups.setdefault(key, []).append((timestamp, path))

    deleted_files = 0
    reclaimed_bytes = 0
    linked_latest_files = 0
    deduplicated_bytes = 0
    kept_versions: dict[str, list[str]] = {}
    for key, records in sorted(groups.items()):
        records.sort(reverse=True)
        retention_versions = (
            B1_REPORT_LARGE_SAMPLE_RETENTION_VERSIONS
            if key[0] in B1_REPORT_LARGE_SAMPLE_PREFIXES
            else B1_REPORT_RETENTION_VERSIONS
        )
        kept = records[:retention_versions]
        kept_versions[f"{key[0]}{key[1]}"] = [path.name for _, path in kept]
        for _, path in records[retention_versions:]:
            try:
                stat = path.stat()
                path.unlink()
                deleted_files += 1
                if stat.st_nlink == 1:
                    reclaimed_bytes += stat.st_size
            except OSError as exc:
                errors.append(f"b1_research_report:{path.name}:{exc}")

        if not kept:
            continue
        latest_name = B1_REPORT_LATEST_ALIASES.get(key, f"latest_{key[0]}{key[1]}")
        latest_path = report_dir / latest_name
        newest_path = kept[0][1]
        if not latest_path.is_file():
            continue
        try:
            latest_stat = latest_path.stat()
            newest_stat = newest_path.stat()
            if (
                (latest_stat.st_dev, latest_stat.st_ino)
                == (newest_stat.st_dev, newest_stat.st_ino)
            ):
                continue
            if latest_stat.st_size != newest_stat.st_size:
                continue
            if not filecmp.cmp(latest_path, newest_path, shallow=False):
                continue
            atomic_link_or_copy(newest_path, latest_path)
            linked_latest_files += 1
            if latest_stat.st_nlink == 1:
                deduplicated_bytes += latest_stat.st_size
        except OSError as exc:
            errors.append(f"b1_research_report_link:{latest_name}:{exc}")

    return {
        "retention_versions": B1_REPORT_RETENTION_VERSIONS,
        "large_sample_retention_versions": B1_REPORT_LARGE_SAMPLE_RETENTION_VERSIONS,
        "groups": len(groups),
        "kept_versions": kept_versions,
        "deleted_files": deleted_files,
        "linked_latest_files": linked_latest_files,
        "reclaimed_bytes": reclaimed_bytes + deduplicated_bytes,
        "deleted_reclaimed_bytes": reclaimed_bytes,
        "deduplicated_bytes": deduplicated_bytes,
    }


def _cleanup_sql_snapshots(root: Path, today: date, errors: list[str]) -> dict[str, Any]:
    try:
        from sqlalchemy import bindparam, inspect, text

        from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
    except ImportError:
        return {"status": "skipped", "reason": "sqlalchemy unavailable", "deleted_rows": 0}

    store = MarketDataStore(MarketDataStoreConfig.from_env(root=root / "data"))
    if (
        store.config.backend not in {"mysql", "sql"}
        or not store.config.sql_url
    ):
        return {"status": "skipped", "reason": "sql backend disabled", "deleted_rows": 0}
    engine = None
    table_summaries: dict[str, Any] = {}
    total_deleted = 0
    try:
        try:
            engine = store._engine()
            if engine.dialect.name != "mysql":
                return {
                    "status": "skipped",
                    "reason": f"unsupported sql dialect: {engine.dialect.name}",
                    "deleted_rows": 0,
                }
            inspector = inspect(engine)
        except Exception as exc:
            errors.append(f"sql:connection:{exc}")
            return {"status": "failed", "deleted_rows": 0, "tables": {}}
        for definition in SQL_SNAPSHOT_TABLES:
            table_name = str(definition["name"])
            try:
                if not inspector.has_table(table_name):
                    table_summaries[table_name] = {"status": "skipped", "deleted_rows": 0}
                    continue
                fields = [
                    "snapshot_key",
                    str(definition["date_field"]),
                    *definition["group_fields"],
                    "updated_at",
                ]
                with engine.connect() as conn:
                    records = conn.execute(
                        text(f"SELECT {', '.join(fields)} FROM {table_name}")
                    ).mappings().all()
                cutoff = today - timedelta(days=int(definition["retention_days"]))
                keys = _snapshot_keys_to_delete(
                    records,
                    date_field=str(definition["date_field"]),
                    group_fields=tuple(definition["group_fields"]),
                    cutoff=cutoff,
                    max_versions=int(definition["max_versions"]),
                    latest_values=set(definition["latest_values"]),
                    deduplicate_dates=False,
                )
                deleted = 0
                statement = text(
                    f"DELETE FROM {table_name} WHERE snapshot_key IN :snapshot_keys"
                ).bindparams(bindparam("snapshot_keys", expanding=True))
                for offset in range(0, len(keys), 500):
                    with engine.begin() as conn:
                        result = conn.execute(statement, {"snapshot_keys": keys[offset : offset + 500]})
                    deleted += max(int(result.rowcount or 0), 0)
                total_deleted += deleted
                table_summaries[table_name] = {"status": "success", "deleted_rows": deleted}
            except Exception as exc:
                errors.append(f"sql:{table_name}:{exc}")
                table_summaries[table_name] = {"status": "failed", "deleted_rows": 0}
    finally:
        if engine is not None:
            engine.dispose()
    return {
        "status": "partial" if any(item["status"] == "failed" for item in table_summaries.values()) else "success",
        "deleted_rows": total_deleted,
        "tables": table_summaries,
    }


def activate_vector_config(project_root: Path, config_dir: Path) -> dict[str, Any]:
    """Rotate the production config only after successful vector publication.

    Call from services after pending-marker removal and validation, not from
    arbitrary research builds. Keep one prior config for rollback. Only older
    similar_patterns-owned trees and their exact producer cache-consumer names
    are retired; model/report references and live leases remain untouched.
    """
    from quant.routine.vector_refresh_policy import VECTOR_PUBLICATION_PENDING_FILENAME

    root = project_root.resolve()
    config_dir = config_dir.absolute()
    vector_root = root / "data/research/similar_patterns/vector_cache"
    if config_dir.parent != vector_root or not config_dir.is_dir() or config_dir.is_symlink():
        raise ValueError("expected a production vector configuration directory")
    try:
        (config_dir / VECTOR_PUBLICATION_PENDING_FILENAME).lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError("cannot activate a pending vector publication")
    registry = ArtifactRegistry(root)
    record = next(
        item for item in registry.inventory([config_dir])["entries"]
        if item["path"] == str(config_dir.relative_to(root))
    )
    if record.get("producer") != "similar_patterns" or record.get("state") != "committed":
        raise ValueError("activation requires a committed similar_patterns-owned configuration")
    consumer = "similar_patterns:active_config"
    displaced = registry.commit(consumer, config_dir)
    retired = {}
    for path in displaced:
        if path.parent != vector_root:
            retired[str(path)] = {"keep_reason": "unknown_config_location"}
            continue
        prefix = f"similar_patterns:{path.name}"
        retired[path.name] = registry.retire_owned_tree(
            path, producer="similar_patterns",
            consumers=(prefix + ":vectors", prefix + ":vectors:previous",
                       prefix + ":compiled", prefix + ":compiled:previous"),
            protected_consumers=(consumer, consumer + ":previous"),
        )
    return {
        "active_config": config_dir.name,
        "previous_configs": [path.name for path in registry.referenced_paths(consumer + ":previous")],
        "retired_configs": retired,
    }


def cleanup_vector_artifacts(
    project_root: Path, *, active_vector_paths: Iterable[Path | str] = (), dry_run: bool = True,
) -> dict[str, Any]:
    """Collect owned retired configs/independent generations, never select by mtime.

    Producers should durably pin config and committed generation references in
    ArtifactRegistry. The explicit paths also protect unmigrated callers; without
    ownership registration all existing/experimental caches are kept.
    """
    root = project_root.resolve()
    registry = ArtifactRegistry(root)
    protected = tuple(active_vector_paths)
    similar_root = root / "data/research/similar_patterns"
    vector_root = similar_root / "vector_cache"
    candidates = sorted(vector_root.iterdir()) if vector_root.is_dir() else []
    candidates += [similar_root / name for name in SIMILAR_PATTERN_SMOKE_CACHE_DIRECTORIES]
    result: dict[str, Any] = {
        "dry_run": dry_run, "kept_directory": None, "kept_directories": [],
        "kept_reasons": {}, "candidates": [], "deleted_directories": 0,
        "reclaimed_bytes": 0, "errors": [],
        "smoke": {"deleted_directories": 0, "reclaimed_bytes": 0},
        "compiled": {"deleted_directories": 0, "reclaimed_bytes": 0, "collections": {}},
    }
    for path in candidates:
        if not path.is_dir():
            continue
        try:
            with registry.deletion_guard(path, protected_paths=protected) as reason:
                if not reason:
                    result["candidates"].append(str(path.relative_to(root)))
                if reason or dry_run:
                    if path.parent == vector_root:
                        result["kept_directories"].append(path.name)
                    if reason:
                        result["kept_reasons"][str(path.relative_to(root))] = reason
                    continue
            # Recheck under the same registry protocol immediately before deletion.
            deleted = registry.delete_if_eligible(path, protected_paths=protected)
            if not deleted["deleted"]:
                result["kept_reasons"][str(path.relative_to(root))] = deleted["reason"]
                if path.parent == vector_root:
                    result["kept_directories"].append(path.name)
                continue
            bucket = result if path.parent == vector_root else result["smoke"]
            bucket["deleted_directories"] += 1
            bucket["reclaimed_bytes"] += deleted["reclaimed_bytes"]
        except OSError as exc:
            result["errors"].append(f"similar_patterns:{path.name}:{exc}")
    for config_dir in candidates:
        matrix_root = config_dir / "_matrix_cache_v1"
        if config_dir.parent != vector_root or not matrix_root.is_dir():
            continue
        try:
            collection = registry.collect_retired_children(
                matrix_root, dry_run=dry_run, protected_paths=protected,
            )
            result["compiled"]["collections"][config_dir.name] = collection
            result["compiled"]["deleted_directories"] += len(collection["deleted_paths"])
            result["compiled"]["reclaimed_bytes"] += collection["reclaimed_bytes"]
            result["reclaimed_bytes"] += collection["reclaimed_bytes"]
            result["errors"].extend(collection["errors"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["errors"].append(f"similar_patterns_compiled:{config_dir.name}:{exc}")
    if len(result["kept_directories"]) == 1:
        result["kept_directory"] = result["kept_directories"][0]
    return result


def cache_storage_inventory(
    project_root: Path, *, paths: Iterable[Path | str] | None = None,
    budgets: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Explicit read-only inventory; research/raw are measured, never auto-purged."""
    return ArtifactRegistry(project_root).inventory(
        paths if paths is not None else ("data", "reports"), budgets=budgets,
    )


def cleanup_publication_generations(
    project_root: Path, *, dry_run: bool = True, minimum_age_days: int = 2,
    completed_grace_days: int = 0, now: datetime | None = None,
) -> dict[str, Any]:
    """Preview/apply registered retired publication retention after writer release.

    Integration: register each generation before cloning (state='building'), hold
    a build lease, and call registry.commit('publication:current', generation_dir)
    after validated pointer publication while still holding writer.lock. Retire
    superseded generations or failed/abandoned stages explicitly. Read leases
    must cover pointer selection and full consumption (retry if selection raced
    collection). This hook never infers ownership/abandonment from directory age.

    Both preview and apply recheck writer exclusion and reachability. Unknown
    generations and current/previous references are kept even over budget.
    Unreferenced retired completed generations have no default grace period,
    retaining current + previous complete (plus any other references/readers).
    Aborted stages keep the two-day minimum; staging is never collected here.
    """
    if minimum_age_days < 0 or completed_grace_days < 0:
        raise ValueError("minimum publication age must be nonnegative")
    root = project_root.resolve()
    registry = ArtifactRegistry(root)
    directory = root / "data/publications"
    result: dict[str, Any] = {
        "status": "success", "dry_run": dry_run, "candidates": [], "kept": {},
        "deleted_directories": 0, "reclaimed_bytes": 0, "errors": [],
    }
    if not directory.exists():
        return result
    try:
        lock_path = directory / "writer.lock"
        if any(path.is_symlink() for path in (directory, lock_path, directory / "current.json")):
            raise ValueError("symlink publication metadata")
        # Never create or replace the writer's lock inode, even during dry-run.
        with lock_path.open("rb") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result["status"] = "skipped_writer_active"
                return result
            pointer = json.loads((directory / "current.json").read_text("utf-8"))
            generation = pointer["generation"]
            if not isinstance(generation, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", generation):
                raise ValueError("invalid publication generation")
            current = directory / generation
            if not (current / "tree").is_dir() or current.is_symlink():
                raise ValueError("current publication tree missing or unsafe")
            previous = registry.referenced_paths("publication:current:previous")
            if pointer.get("previous") is not None:
                previous_id = pointer["previous"]
                if not isinstance(previous_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", previous_id):
                    raise ValueError("invalid previous publication generation")
                previous = (*previous, directory / previous_id)
            if any(not (path / "tree").is_dir() or path.is_symlink() for path in previous):
                raise ValueError("previous publication tree missing or unsafe")
            protected = (current, *previous)
            for path in protected:
                state_path = path / "state.json"
                if state_path.is_symlink() or (path / "tree").is_symlink():
                    raise ValueError("symlink committed publication tree/state")
                state = json.loads(state_path.read_text("utf-8"))
                if not isinstance(state, dict) or state.get("status") != "validated":
                    raise ValueError("committed publication is not validated")
            current_time = now or datetime.now()
            result["inventory"] = registry.inventory([directory])
            for path in sorted(directory.iterdir()):
                if not path.is_dir() or path.is_symlink():
                    continue
                try:
                    state_path = path / "state.json"
                    if state_path.is_symlink():
                        raise ValueError("symlink publication state")
                    state = json.loads(state_path.read_text("utf-8"))
                except (OSError, ValueError):
                    state = {}
                if not isinstance(state, dict) or state.get("status") not in {"validated", "aborted"}:
                    result["kept"][path.name] = "staging_or_unknown_state"
                    continue
                if not previous and path != current:
                    # Until the writer migrates the previous pointer, no old
                    # complete tree is proven safe to retire (including baseline).
                    if state.get("status") != "aborted":
                        result["kept"][path.name] = "previous_generation_unknown"
                        continue
                grace_days = completed_grace_days if state["status"] == "validated" else minimum_age_days
                cutoff = current_time - timedelta(days=grace_days)
                with registry.deletion_guard(
                    path, older_than=cutoff.timestamp(), protected_paths=protected,
                ) as reason:
                    if reason:
                        result["kept"][path.name] = reason
                        continue
                    result["candidates"].append(str(path.relative_to(root)))
                    if dry_run:
                        continue
                # Writer lock stays held; delete_if_eligible rechecks leases that
                # may have started since inventory/preview under the registry lock.
                deletion = registry.delete_if_eligible(
                    path, protected_paths=protected, older_than=cutoff.timestamp(),
                )
                if deletion["deleted"]:
                    result["deleted_directories"] += 1
                    result["reclaimed_bytes"] += deletion["reclaimed_bytes"]
                else:
                    result["kept"][path.name] = deletion["reason"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["status"] = "unavailable"
        result["errors"].append(f"publication_retention:{exc}")
    return result


def prune_staged_publication_snapshots(
    staged_tree: Path, reference_date: date | None = None,
) -> dict[str, Any]:
    """Apply only JSON snapshot policies to a writer-owned unpublished copy.

    Call inside PublicationStore.begin's writer/build-lease context, before the
    strict audit and commit. Raises on an unsafe target; no SQL, source caches,
    raw inputs, or committed trees are touched. The caller should block commit
    on a partial result, just as with other staging/validation failures.
    """
    tree = staged_tree.absolute()
    if (
        tree.name != "tree" or len(tree.parents) < 4
        or tree.parents[1].name != "publications" or tree.parents[2].name != "data"
        or not re.fullmatch(r"[A-Za-z0-9_-]+", tree.parent.name)
    ):
        raise ValueError("expected a publication generation tree")
    root = tree.parents[3]
    generation_dir = tree.parent
    publication_dir = generation_dir.parent
    registry = ArtifactRegistry(root)
    with registry.build_mutation_guard(generation_dir):
        if not tree.is_dir() or tree.is_symlink():
            raise ValueError("staged tree missing or unsafe")
        # A separate nonblocking lock attempt verifies the publisher still owns
        # its stable writer lock. Never acquire a blocking lock in this order.
        lock_path = publication_dir / "writer.lock"
        if lock_path.is_symlink():
            raise ValueError("symlink publication writer lock")
        with lock_path.open("rb") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError("staged pruning requires an active publication writer")
        pointer_path = publication_dir / "current.json"
        state_path = generation_dir / "state.json"
        if pointer_path.is_symlink() or state_path.is_symlink():
            raise ValueError("symlink publication metadata")
        pointer = json.loads(pointer_path.read_text("utf-8"))
        state = json.loads(state_path.read_text("utf-8"))
        if not isinstance(pointer, dict) or not isinstance(state, dict):
            raise ValueError("invalid publication metadata")
        committed = pointer.get("committed_generations", [])
        if not isinstance(committed, list):
            raise ValueError("invalid committed generation inventory")
        if state.get("status") != "staging" or generation_dir.name in (
            pointer.get("generation"), pointer.get("previous"), *committed,
        ):
            raise ValueError("only unpublished staging trees can be pruned")
        # Existing snapshot policies follow glob paths; reject links before
        # invoking them so a malformed clone cannot redirect deletion outside it.
        snapshot_roots = [tree / relative for relative in (
            "data/selector_snapshots", "data/long_stock_pool_snapshots", "data/workspace_snapshots",
        )]
        for path in snapshot_roots:
            if (tree / "data").is_symlink() or path.is_symlink() or any(
                child.is_symlink() for child in path.rglob("*")
            ):
                raise ValueError("snapshot staging tree contains symlinks")
        errors: list[str] = []
        today = reference_date or date.today()
        snapshots = _cleanup_snapshot_files(tree, today, errors)
        return {
            "status": "partial" if errors else "success", "reference_date": today.isoformat(),
            "snapshots": snapshots, "reclaimed_bytes": snapshots["reclaimed_bytes"],
            "errors": errors,
        }


def cleanup_daily_caches(
    project_root: Path, reference_date: date | None = None, *,
    active_vector_paths: Iterable[Path | str] = (),
) -> dict[str, Any]:
    """Apply retention rules after a successful daily refresh."""

    root = project_root.resolve()
    today = reference_date or date.today()
    errors: list[str] = []
    active_vector_paths = tuple(active_vector_paths)
    storage_before = _managed_cache_storage(root)

    factor_schemas = _cleanup_factor_schema_caches(root, errors)
    root_market_requests = _cleanup_root_market_request_cache(root, today, errors)
    abandoned_cache_builds = _cleanup_abandoned_cache_builds(
        root, today, errors, protected_paths=active_vector_paths,
    )

    long_cache_dir = root / "data/research/long_dividend_quality"
    deleted_long_files = 0
    reclaimed_long_bytes = 0
    candidates: set[Path] = set()
    for pattern in LONG_CACHE_PATTERNS:
        candidates.update(long_cache_dir.glob(pattern))
    cache_groups: dict[str, list[Path]] = {}
    for path in candidates:
        cache_key = path.stem.removeprefix("daily_returns_").removeprefix(
            "daily_monthly_features_"
        )
        cache_groups.setdefault(cache_key, []).append(path)
    ordered_groups = sorted(
        cache_groups.items(),
        key=lambda item: (
            max(path.stat().st_mtime_ns for path in item[1]),
            item[0],
        ),
    )
    expired_groups = ordered_groups[:-LONG_CACHE_RETENTION_VERSIONS]
    for _, paths in expired_groups:
        for path in sorted(paths):
            try:
                size = path.stat().st_size
                path.unlink()
                deleted_long_files += 1
                reclaimed_long_bytes += size
            except OSError as exc:
                errors.append(f"long_strategy:{path.name}:{exc}")

    vector_summary = cleanup_vector_artifacts(
        root, active_vector_paths=active_vector_paths, dry_run=False,
    )
    reclaimed_vector_bytes = vector_summary["reclaimed_bytes"]
    reclaimed_smoke_bytes = vector_summary["smoke"]["reclaimed_bytes"]
    errors.extend(vector_summary["errors"])

    tushare_cache_dir = root / "data/cache/source_merge/tushare"
    tushare_cutoff = today - timedelta(days=TUSHARE_SINGLE_SYMBOL_CACHE_RETENTION_DAYS)
    deleted_tushare_files = 0
    reclaimed_tushare_bytes = 0
    for path in sorted(tushare_cache_dir.glob("*.parquet")):
        if not TUSHARE_SINGLE_SYMBOL_CACHE_PATTERN.fullmatch(path.name):
            continue
        try:
            modified_date = datetime.fromtimestamp(path.stat().st_mtime).date()
            if modified_date >= tushare_cutoff:
                continue
            size = path.stat().st_size
            path.unlink()
            deleted_tushare_files += 1
            reclaimed_tushare_bytes += size
        except OSError as exc:
            errors.append(f"tushare_single_symbol:{path.name}:{exc}")

    daily_basic_raw_dir = root / "data/raw/daily_basic"
    daily_basic_cache = _cleanup_daily_basic_cache_directory(
        tushare_cache_dir,
        daily_basic_raw_dir,
        cutoff=tushare_cutoff,
        error_prefix="tushare_daily_basic",
        errors=errors,
    )
    live_probe_cutoff = today - timedelta(days=TUSHARE_LIVE_PROBE_RETENTION_DAYS)
    live_probe_cache = _cleanup_daily_basic_cache_directory(
        root / "data/cache/source_merge/tushare_live_probe",
        daily_basic_raw_dir,
        cutoff=live_probe_cutoff,
        error_prefix="tushare_live_probe",
        errors=errors,
    )

    snapshots = _cleanup_snapshot_files(root, today, errors)
    source_audit = _cleanup_timestamped_directories(
        root / "data/raw/source_audit",
        cutoff=today - timedelta(days=SOURCE_AUDIT_RETENTION_DAYS),
        max_runs=SOURCE_AUDIT_MAX_RUNS,
        error_prefix="source_audit",
        errors=errors,
        group_by_suffix=True,
    )
    source_audit.update(
        {
            "retention_days": SOURCE_AUDIT_RETENTION_DAYS,
            "max_runs": SOURCE_AUDIT_MAX_RUNS,
        }
    )
    routine_runs = _cleanup_timestamped_directories(
        root / "data/routine",
        cutoff=today - timedelta(days=ROUTINE_RUN_RETENTION_DAYS),
        max_runs=ROUTINE_RUN_MAX_RUNS,
        error_prefix="routine_runs",
        errors=errors,
    )
    routine_runs.update(
        {
            "retention_days": ROUTINE_RUN_RETENTION_DAYS,
            "max_runs": ROUTINE_RUN_MAX_RUNS,
        }
    )
    convertible_bond_requests = _cleanup_cb_daily_request_cache(root, errors)
    b1_research_reports = _cleanup_b1_research_reports(root, errors)
    sql_snapshots = _cleanup_sql_snapshots(root, today, errors)

    reclaimed_bytes = (
        factor_schemas["reclaimed_bytes"]
        + root_market_requests["reclaimed_bytes"]
        + abandoned_cache_builds["reclaimed_bytes"]
        + reclaimed_long_bytes
        + reclaimed_vector_bytes
        + reclaimed_smoke_bytes
        + reclaimed_tushare_bytes
        + daily_basic_cache["reclaimed_bytes"]
        + live_probe_cache["reclaimed_bytes"]
        + snapshots["reclaimed_bytes"]
        + source_audit["reclaimed_bytes"]
        + routine_runs["reclaimed_bytes"]
        + convertible_bond_requests["reclaimed_bytes"]
        + b1_research_reports["reclaimed_bytes"]
    )
    storage_after = _managed_cache_storage(root)
    return {
        "status": "success" if not errors else "partial",
        "reference_date": today.isoformat(),
        "storage": {
            "before": storage_before,
            "after": storage_after,
            "logical_bytes_reduced": max(
                0,
                storage_before["logical_bytes"] - storage_after["logical_bytes"],
            ),
            "allocated_bytes_reduced": max(
                0,
                storage_before["allocated_bytes"] - storage_after["allocated_bytes"],
            ),
        },
        "daily_factor_schemas": factor_schemas,
        "root_market_request_cache": root_market_requests,
        "abandoned_cache_builds": abandoned_cache_builds,
        "long_strategy": {
            "retention_versions": LONG_CACHE_RETENTION_VERSIONS,
            "kept_versions": [key for key, _ in ordered_groups[-LONG_CACHE_RETENTION_VERSIONS:]],
            "deleted_files": deleted_long_files,
            "reclaimed_bytes": reclaimed_long_bytes,
        },
        "similar_patterns": vector_summary,
        "tushare_single_symbol": {
            "retention_days": TUSHARE_SINGLE_SYMBOL_CACHE_RETENTION_DAYS,
            "cutoff_date": tushare_cutoff.isoformat(),
            "deleted_files": deleted_tushare_files,
            "reclaimed_bytes": reclaimed_tushare_bytes,
        },
        "tushare_daily_basic": {
            "retention_days": TUSHARE_SINGLE_SYMBOL_CACHE_RETENTION_DAYS,
            "cutoff_date": tushare_cutoff.isoformat(),
            **daily_basic_cache,
        },
        "tushare_live_probe": {
            "retention_days": TUSHARE_LIVE_PROBE_RETENTION_DAYS,
            "cutoff_date": live_probe_cutoff.isoformat(),
            **live_probe_cache,
        },
        "snapshots": snapshots,
        "source_audit": source_audit,
        "routine_runs": routine_runs,
        "convertible_bond_request_cache": convertible_bond_requests,
        "b1_research_reports": b1_research_reports,
        "sql_snapshots": sql_snapshots,
        "reclaimed_bytes": reclaimed_bytes,
        "errors": errors,
    }


def run_cache_cleanup(
    project_root: Path,
    reference_date: date | None = None,
    cleanup_fn: Callable[[Path, date | None], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run cache retention without allowing maintenance errors to abort refreshes."""

    effective_date = reference_date or date.today()
    cleanup = cleanup_fn or cleanup_daily_caches
    try:
        return cleanup(project_root, effective_date)
    except Exception as exc:
        return {
            "status": "failed",
            "reference_date": effective_date.isoformat(),
            "reclaimed_bytes": 0,
            "errors": [str(exc)],
        }
