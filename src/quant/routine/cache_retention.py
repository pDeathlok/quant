from __future__ import annotations

import filecmp
import hashlib
import json
import re
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


LONG_CACHE_PATTERNS = (
    "daily_returns_*.parquet",
    "daily_monthly_features_*.parquet",
)
LONG_CACHE_RETENTION_VERSIONS = 2
TUSHARE_SINGLE_SYMBOL_CACHE_RETENTION_DAYS = 7
TUSHARE_SINGLE_SYMBOL_CACHE_PATTERN = re.compile(
    r"^tushare_\d{6}\.(?:SH|SZ|BJ)_\d{8}_\d{8}_(?:None|qfq|hfq)\.parquet$"
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


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


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
            continue
        try:
            size = path.stat().st_size
            path.unlink()
            deleted_files += 1
            reclaimed_bytes += size
        except OSError as exc:
            errors.append(f"tushare_cb_daily:{path.name}:{exc}")
    return {
        "consolidated_ranges": [
            {"start": start.isoformat(), "end": end.isoformat()}
            for start, end in sorted(consolidated_ranges)
        ],
        "deleted_files": deleted_files,
        "protected_outside_consolidated_range": protected_files,
        "reclaimed_bytes": reclaimed_bytes,
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


def cleanup_daily_caches(project_root: Path, reference_date: date | None = None) -> dict[str, Any]:
    """Apply retention rules after a successful daily refresh."""

    root = project_root.resolve()
    today = reference_date or date.today()
    errors: list[str] = []

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

    vector_cache_root = root / "data/research/similar_patterns/vector_cache"
    vector_directories = sorted(
        (
            path
            for path in vector_cache_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    ) if vector_cache_root.exists() else []
    kept_vector_directory = vector_directories[-1] if vector_directories else None
    deleted_vector_directories = 0
    reclaimed_vector_bytes = 0
    for path in vector_directories[:-1]:
        try:
            size = _directory_size(path)
            shutil.rmtree(path)
            deleted_vector_directories += 1
            reclaimed_vector_bytes += size
        except OSError as exc:
            errors.append(f"similar_patterns:{path.name}:{exc}")

    similar_pattern_root = root / "data/research/similar_patterns"
    deleted_smoke_directories = 0
    reclaimed_smoke_bytes = 0
    for directory_name in SIMILAR_PATTERN_SMOKE_CACHE_DIRECTORIES:
        path = similar_pattern_root / directory_name
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            size = _directory_size(path)
            shutil.rmtree(path)
            deleted_smoke_directories += 1
            reclaimed_smoke_bytes += size
        except OSError as exc:
            errors.append(f"similar_patterns_smoke:{path.name}:{exc}")

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
    deleted_daily_basic_files = 0
    reclaimed_daily_basic_bytes = 0
    protected_daily_basic_files = 0
    for path in sorted(tushare_cache_dir.glob("tushare_daily_basic_*.parquet")):
        match = TUSHARE_DAILY_BASIC_CACHE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        trade_date_text = match.group(1)
        try:
            trade_date = datetime.strptime(trade_date_text, "%Y%m%d").date()
        except ValueError:
            continue
        if trade_date >= tushare_cutoff:
            continue
        raw_path = daily_basic_raw_dir / f"{trade_date_text}.parquet"
        try:
            if not raw_path.is_file() or raw_path.stat().st_size <= 0:
                protected_daily_basic_files += 1
                continue
            size = path.stat().st_size
            path.unlink()
            deleted_daily_basic_files += 1
            reclaimed_daily_basic_bytes += size
        except OSError as exc:
            errors.append(f"tushare_daily_basic:{path.name}:{exc}")

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
        reclaimed_long_bytes
        + reclaimed_vector_bytes
        + reclaimed_smoke_bytes
        + reclaimed_tushare_bytes
        + reclaimed_daily_basic_bytes
        + snapshots["reclaimed_bytes"]
        + source_audit["reclaimed_bytes"]
        + routine_runs["reclaimed_bytes"]
        + convertible_bond_requests["reclaimed_bytes"]
        + b1_research_reports["reclaimed_bytes"]
    )
    return {
        "status": "success" if not errors else "partial",
        "reference_date": today.isoformat(),
        "long_strategy": {
            "retention_versions": LONG_CACHE_RETENTION_VERSIONS,
            "kept_versions": [key for key, _ in ordered_groups[-LONG_CACHE_RETENTION_VERSIONS:]],
            "deleted_files": deleted_long_files,
            "reclaimed_bytes": reclaimed_long_bytes,
        },
        "similar_patterns": {
            "kept_directory": kept_vector_directory.name if kept_vector_directory else None,
            "deleted_directories": deleted_vector_directories,
            "reclaimed_bytes": reclaimed_vector_bytes,
            "smoke": {
                "deleted_directories": deleted_smoke_directories,
                "reclaimed_bytes": reclaimed_smoke_bytes,
            },
        },
        "tushare_single_symbol": {
            "retention_days": TUSHARE_SINGLE_SYMBOL_CACHE_RETENTION_DAYS,
            "cutoff_date": tushare_cutoff.isoformat(),
            "deleted_files": deleted_tushare_files,
            "reclaimed_bytes": reclaimed_tushare_bytes,
        },
        "tushare_daily_basic": {
            "retention_days": TUSHARE_SINGLE_SYMBOL_CACHE_RETENTION_DAYS,
            "cutoff_date": tushare_cutoff.isoformat(),
            "deleted_files": deleted_daily_basic_files,
            "protected_without_raw": protected_daily_basic_files,
            "reclaimed_bytes": reclaimed_daily_basic_bytes,
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
