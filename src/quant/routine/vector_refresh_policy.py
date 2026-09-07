"""Integrity repair is independent of the weekly historical-library schedule.

The services adapter supplies its active state directory, force/environment flag,
source date, and metadata. This module performs no refresh or metadata writes.
Semantic array/source-revision validation belongs to the vector reader; pass its
failure as ``integrity_error``. ZIP checks here detect malformed/truncated files,
not every possible array corruption.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

VECTOR_PUBLICATION_PENDING_FILENAME = "_publication_pending.json"


def _timestamp(value: Any, current: datetime) -> datetime | None:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None:
            return result.replace(tzinfo=current.tzinfo)
        if current.tzinfo is None:
            return result.astimezone().replace(tzinfo=None)
        return result.astimezone(current.tzinfo)
    except (ValueError, TypeError):
        return None


def vector_cache_refresh_decision(
    state_dir: Path, *, now: datetime | None = None, force: bool = False,
    source_trade_date: str | None = None, metadata: Mapping[str, Any] | None = None,
    minimum_refresh_age_days: float = 5, refresh_weekday: int = 4, refresh_hour: int = 15,
    integrity_error: str | None = None, repair_available: bool = True,
    mandatory_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Drop-in decision payload; mandatory-reference failures never become healthy skips.

    ``repair_available=False`` explicitly blocks an unusable library. Callers must
    treat status=unavailable as a failure, not continue reading the broken cache.
    Call under a read/build lease when retention can run concurrently.
    """
    if not 0 <= refresh_weekday <= 6 or not 0 <= refresh_hour <= 23:
        raise ValueError("invalid refresh schedule")
    if minimum_refresh_age_days < 0:
        raise ValueError("minimum refresh age must be nonnegative")
    current = now or datetime.now()
    repair_reason = integrity_error
    publication_pending = False
    try:
        (state_dir / VECTOR_PUBLICATION_PENDING_FILENAME).lstat()
    except FileNotFoundError:
        pass
    except OSError:
        # Inability to disprove a pending publication is not a healthy library.
        publication_pending = True
    else:
        publication_pending = True
    if publication_pending:
        repair_reason = "cache_publication_pending"
    if metadata is None:
        try:
            metadata = json.loads((state_dir / "_refresh_metadata.json").read_text("utf-8"))
        except FileNotFoundError:
            metadata = {}
        except (OSError, ValueError):
            metadata = {}
            repair_reason = repair_reason or "metadata_invalid"
    if not isinstance(metadata, Mapping):
        metadata = {}
        repair_reason = repair_reason or "metadata_invalid"
    metadata = dict(metadata)
    refreshed_at = _timestamp(metadata.get("refreshed_at"), current)
    cache_files: list[Path] = []
    try:
        cache_files = sorted(state_dir.glob("*.npz"))
        if not cache_files:
            repair_reason = repair_reason or "cache_missing"
        for field in ("errors", "cached_files"):
            if field not in metadata:
                continue
            value = int(str(metadata[field]))
            if value < 0:
                raise ValueError(f"invalid {field}")
            if field == "errors" and value:
                repair_reason = repair_reason or "previous_refresh_errors"
            if field == "cached_files" and value != len(cache_files):
                repair_reason = repair_reason or "cache_file_count_changed"
        for path in cache_files:
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                repair_reason = repair_reason or "empty_cache_file"
                break
            if not zipfile.is_zipfile(path):
                repair_reason = repair_reason or "cache_corrupt"
                break
        for path in mandatory_paths:
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                repair_reason = repair_reason or "mandatory_reference_missing"
                break
    except OSError:
        repair_reason = repair_reason or "cache_unreadable"
    except (ValueError, TypeError):
        repair_reason = repair_reason or "metadata_invalid"
    if metadata.get("config_key") is not None and metadata["config_key"] != state_dir.name:
        repair_reason = repair_reason or "config_key_mismatch"
    if refreshed_at is None:
        repair_reason = repair_reason or "refresh_time_missing"
    elif refreshed_at > current:
        repair_reason = repair_reason or "refresh_time_in_future"
    age_days = (
        max(0, (current - refreshed_at).total_seconds() / 86400) if refreshed_at else None
    )
    in_window = current.weekday() == refresh_weekday and current.hour >= refresh_hour
    if repair_reason and not repair_available:
        due, reason = False, "repair_unavailable"
    elif force:
        due, reason = True, "forced"
    elif repair_reason:
        due, reason = True, repair_reason
    elif not in_window:
        due, reason = False, "waiting_for_friday_close"
    elif str(source_trade_date) != current.date().isoformat():
        due, reason = False, "waiting_for_friday_trade_close"
    elif age_days is not None and age_days < minimum_refresh_age_days:
        due = False
        reason = (
            "friday_close_window_already_refreshed"
            if refreshed_at and refreshed_at.date() == current.date()
            else "minimum_refresh_age_not_reached"
        )
    else:
        due, reason = True, "friday_close_window"
    next_refresh = current.replace(hour=refresh_hour, minute=0, second=0, microsecond=0)
    next_refresh += timedelta(days=(refresh_weekday - current.weekday()) % 7)
    if next_refresh <= current:
        next_refresh += timedelta(days=7)
    return {
        "due": due, "reason": reason, "repair_required": bool(repair_reason),
        "repair_reason": repair_reason,
        "publication_pending": publication_pending,
        "status": "repair_required" if due and repair_reason else
        "unavailable" if repair_reason else "healthy",
        "cached_files": len(cache_files),
        "refreshed_at": refreshed_at.isoformat(timespec="seconds") if refreshed_at else None,
        "next_refresh_at": next_refresh.isoformat(timespec="seconds"),
        "refresh_age_days": age_days, "minimum_refresh_age_days": minimum_refresh_age_days,
        "metadata": metadata, "inferred_legacy": False,
    }
