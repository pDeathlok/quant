"""Validated, monotonic publication of point-in-time long-factor snapshots."""

from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from quant.data.atomic_io import atomic_link_or_copy, atomic_write_json, atomic_write_parquet
from quant.features.canonical_factor_names import assert_no_forbidden_factor_names
from quant.features.factor_registry import (
    LONG_PRODUCTION_FACTOR_COLUMNS,
    LONG_PRODUCTION_FACTOR_SCHEMA_VERSION,
)


@contextmanager
def _snapshot_lock(directory: Path, *, exclusive: bool) -> Iterator[None]:
    if exclusive:
        directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".publish.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_date(frame: pd.DataFrame) -> str:
    if frame.empty or "date" not in frame:
        raise RuntimeError("long_snapshot: empty or undated cross-section")
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.nunique() != 1:
        raise RuntimeError("long_snapshot: invalid or mixed cross-section dates")
    return dates.iloc[0].date().isoformat()


def validate_long_factor_frame(frame: pd.DataFrame, signal_date: str | None = None) -> str:
    assert_no_forbidden_factor_names(frame.columns, context="long_snapshot")
    required = {"date", "ts_code", "factor_schema_version", *LONG_PRODUCTION_FACTOR_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing or frame.columns.duplicated().any():
        raise RuntimeError(f"long_snapshot: incomplete or duplicate columns: {missing}")
    actual = _frame_date(frame)
    if signal_date is not None and actual != signal_date:
        raise RuntimeError(f"long_snapshot: date mismatch: expected={signal_date} actual={actual}")
    if frame["ts_code"].isna().any() or frame["ts_code"].duplicated().any():
        raise RuntimeError("long_snapshot: missing or duplicate symbols")
    if not frame["factor_schema_version"].eq(LONG_PRODUCTION_FACTOR_SCHEMA_VERSION).all():
        raise RuntimeError("long_snapshot: factor schema mismatch")
    return actual


def _manifest(frame: pd.DataFrame, directory: Path, signal_date: str, digest: str) -> dict[str, Any]:
    return {
        "status": "success",
        "factor_schema_version": LONG_PRODUCTION_FACTOR_SCHEMA_VERSION,
        "signal_date": signal_date,
        "rows": len(frame),
        "factor_count": len(LONG_PRODUCTION_FACTOR_COLUMNS),
        "expected_factor_count": len(LONG_PRODUCTION_FACTOR_COLUMNS),
        "missing_factors": [],
        "coverage_status": "complete",
        "source": "long_page_point_in_time_cross_section",
        "dated_path": str(directory / f"{signal_date.replace('-', '')}.parquet"),
        "latest_path": str(directory / "latest.parquet"),
        "data_sha256": digest,
        "publication_schema_version": "long-factor-snapshot-publication-v2",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def read_long_factor_snapshot(
    directory: Path,
    signal_date: str | None = None,
    *,
    latest: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Require a consistent latest pair; allow validated pre-manifest dated archives."""
    stem = "latest" if latest or signal_date is None else signal_date.replace("-", "")
    path = directory / f"{stem}.parquet"
    manifest_path = directory / f"{stem}.json"
    try:
        with _snapshot_lock(directory, exclusive=False):
            frame = pd.read_parquet(path)
            actual = validate_long_factor_frame(frame, signal_date)
            digest = _sha256(path)
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise RuntimeError("long_snapshot: invalid manifest object")
                expected = _manifest(frame, directory, actual, digest)
                for key in (
                    "status", "signal_date", "rows", "factor_count",
                    "factor_schema_version", "coverage_status", "data_sha256",
                ):
                    if manifest.get(key) != expected[key]:
                        raise RuntimeError(f"long_snapshot: manifest/data mismatch: {key}")
            elif stem == "latest":
                raise RuntimeError("long_snapshot: latest manifest missing")
            else:
                manifest = _manifest(frame, directory, actual, digest)
            return frame, {**manifest, "snapshot_path": str(path)}
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"long_snapshot: cannot validate {path}: {exc}") from exc


def publish_long_factor_snapshot(frame: pd.DataFrame, directory: Path) -> dict[str, Any]:
    """Publish history without ever moving latest backwards, including concurrent writers."""
    signal_date = validate_long_factor_frame(frame)
    dated_path = directory / f"{signal_date.replace('-', '')}.parquet"
    latest_path = directory / "latest.parquet"
    manifest_path = directory / "latest.json"
    with _snapshot_lock(directory, exclusive=True):
        current_dates: list[str] = []
        if latest_path.is_file():
            current_dates.append(_frame_date(pd.read_parquet(latest_path, columns=["date"])))
        if manifest_path.is_file():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            current_dates.append(pd.Timestamp(current["signal_date"]).date().isoformat())
        latest_date = max(current_dates, default=signal_date)
        publish_latest = signal_date >= latest_date
        atomic_write_parquet(frame, dated_path, index=False)
        manifest = _manifest(frame, directory, signal_date, _sha256(dated_path))
        manifest.update(latest_published=publish_latest, latest_signal_date=max(latest_date, signal_date))
        dated_manifest_path = dated_path.with_suffix(".json")
        atomic_write_json(manifest, dated_manifest_path)
        if publish_latest:
            atomic_link_or_copy(dated_path, latest_path)
            atomic_write_json(manifest, manifest_path)
        return {
            **manifest,
            "manifest_path": str(manifest_path if publish_latest else dated_manifest_path),
        }
