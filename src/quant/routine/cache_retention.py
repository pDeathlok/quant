from __future__ import annotations

import calendar
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any


LONG_CACHE_PATTERNS = (
    "daily_returns_*.parquet",
    "daily_monthly_features_*.parquet",
)
LONG_CACHE_RETENTION_MONTHS = 3


def _subtract_calendar_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 - months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def cleanup_daily_caches(project_root: Path, reference_date: date | None = None) -> dict[str, Any]:
    """Apply retention rules after a successful daily refresh.

    Tushare single-symbol request caches are intentionally outside the paths
    touched by this function and remain available for long-term reuse.
    """

    root = project_root.resolve()
    today = reference_date or date.today()
    cutoff = _subtract_calendar_months(today, LONG_CACHE_RETENTION_MONTHS)
    errors: list[str] = []

    long_cache_dir = root / "data/research/long_dividend_quality"
    deleted_long_files = 0
    reclaimed_long_bytes = 0
    candidates: set[Path] = set()
    for pattern in LONG_CACHE_PATTERNS:
        candidates.update(long_cache_dir.glob(pattern))
    for path in sorted(candidates):
        try:
            modified_date = datetime.fromtimestamp(path.stat().st_mtime).date()
            if modified_date >= cutoff:
                continue
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

    reclaimed_bytes = reclaimed_long_bytes + reclaimed_vector_bytes
    return {
        "status": "success" if not errors else "partial",
        "reference_date": today.isoformat(),
        "long_strategy": {
            "retention_months": LONG_CACHE_RETENTION_MONTHS,
            "cutoff_date": cutoff.isoformat(),
            "deleted_files": deleted_long_files,
            "reclaimed_bytes": reclaimed_long_bytes,
        },
        "similar_patterns": {
            "kept_directory": kept_vector_directory.name if kept_vector_directory else None,
            "deleted_directories": deleted_vector_directories,
            "reclaimed_bytes": reclaimed_vector_bytes,
        },
        "tushare_single_symbol": {
            "retention": "unchanged",
            "deleted_files": 0,
        },
        "reclaimed_bytes": reclaimed_bytes,
        "errors": errors,
    }
