from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant.routine.vector_refresh_policy import (
    VECTOR_PUBLICATION_PENDING_FILENAME,
    vector_cache_refresh_decision,
)


def _cache(root: Path, *, refreshed_at: str = "2026-09-04T15:05:00") -> Path:
    directory = root / "config"
    directory.mkdir()
    with zipfile.ZipFile(directory / "000001_SZ.npz", "w") as archive:
        archive.writestr("vectors.npy", b"format checked here, semantic check belongs to reader")
    (directory / "_refresh_metadata.json").write_text(json.dumps({
        "refreshed_at": refreshed_at, "cached_files": 1, "errors": 0, "config_key": "config",
    }))
    return directory


@pytest.mark.parametrize("weekday", range(7))
def test_missing_cache_repairs_every_weekday_even_before_minimum_age(
    tmp_path: Path, weekday: int,
) -> None:
    current = datetime(2026, 9, 7, 10) + timedelta(days=weekday)
    decision = vector_cache_refresh_decision(
        tmp_path / "missing", now=current,
        metadata={"refreshed_at": (current - timedelta(hours=1)).isoformat()},
    )
    assert decision["due"] is True
    assert decision["repair_required"] is True
    assert decision["reason"] == "cache_missing"


@pytest.mark.parametrize("damage,reason", [
    ("empty", "empty_cache_file"), ("corrupt", "cache_corrupt"),
    ("count", "cache_file_count_changed"), ("errors", "previous_refresh_errors"),
    ("metadata", "metadata_invalid"), ("count_type", "metadata_invalid"),
    ("wrong_config", "config_key_mismatch"), ("time", "refresh_time_missing"),
])
def test_damage_bypasses_weekly_schedule(tmp_path: Path, damage: str, reason: str) -> None:
    directory = _cache(tmp_path)
    path = directory / "_refresh_metadata.json"
    metadata = json.loads(path.read_text())
    if damage in {"empty", "corrupt"}:
        (directory / "000001_SZ.npz").write_bytes(b"" if damage == "empty" else b"broken zip")
    elif damage == "count":
        metadata["cached_files"] = 2
    elif damage == "count_type":
        metadata["cached_files"] = "not a number"
    elif damage == "errors":
        metadata["errors"] = 1
    elif damage == "wrong_config":
        metadata["config_key"] = "experiment"
    elif damage == "time":
        metadata["refreshed_at"] = "not a timestamp"
    path.write_text("[" if damage == "metadata" else json.dumps(metadata))
    decision = vector_cache_refresh_decision(directory, now=datetime(2026, 9, 7, 10))
    assert decision["due"] is True
    assert decision["reason"] == reason


def test_mandatory_reference_and_semantic_corruption_block_when_repair_unavailable(
    tmp_path: Path,
) -> None:
    directory = _cache(tmp_path)
    decision = vector_cache_refresh_decision(
        directory, now=datetime(2026, 9, 7, 10), mandatory_paths=[directory / "missing.npy"],
        repair_available=False,
    )
    assert decision["due"] is False
    assert decision["status"] == "unavailable"
    assert decision["reason"] == "repair_unavailable"
    assert decision["repair_reason"] == "mandatory_reference_missing"
    semantic = vector_cache_refresh_decision(
        directory, now=datetime(2026, 9, 7, 10), integrity_error="invalid_array_shape",
    )
    assert semantic["due"] and semantic["reason"] == "invalid_array_shape"


@pytest.mark.parametrize("current,source,refreshed,reason,due", [
    (datetime(2026, 9, 7, 10), "2026-09-07", "2026-09-04T15:05:00", "waiting_for_friday_close", False),
    (datetime(2026, 9, 11, 14), "2026-09-11", "2026-09-04T15:05:00", "waiting_for_friday_close", False),
    (datetime(2026, 9, 11, 16), "2026-09-10", "2026-09-04T15:05:00", "waiting_for_friday_trade_close", False),
    (datetime(2026, 9, 11, 16), "2026-09-11", "2026-09-10T15:05:00", "minimum_refresh_age_not_reached", False),
    (datetime(2026, 9, 11, 16), "2026-09-11", "2026-09-11T15:05:00", "friday_close_window_already_refreshed", False),
    (datetime(2026, 9, 11, 16), "2026-09-11", "2026-09-04T15:05:00", "friday_close_window", True),
])
def test_healthy_library_preserves_weekly_policy(
    tmp_path: Path, current: datetime, source: str, refreshed: str, reason: str, due: bool,
) -> None:
    directory = _cache(tmp_path, refreshed_at=refreshed)
    decision = vector_cache_refresh_decision(directory, now=current, source_trade_date=source)
    assert decision["due"] is due
    assert decision["reason"] == reason
    assert decision["repair_required"] is False
    assert decision["status"] == "healthy"


def test_force_timezone_and_read_only_decision(tmp_path: Path) -> None:
    directory = _cache(tmp_path, refreshed_at="2026-09-04T07:05:00Z")
    metadata_path = directory / "_refresh_metadata.json"
    before = metadata_path.read_bytes()
    decision = vector_cache_refresh_decision(
        directory, now=datetime(2026, 9, 7, 10, tzinfo=timezone(timedelta(hours=8))), force=True,
    )
    assert decision["reason"] == "forced" and decision["due"] is True
    assert decision["next_refresh_at"] == "2026-09-11T15:00:00+08:00"
    assert metadata_path.read_bytes() == before


@pytest.mark.parametrize("weekday", range(7))
@pytest.mark.parametrize("marker_content", ["", "{", '{"schema_version":1,"phase":"publishing"}'])
def test_pending_publication_always_repairs_even_with_healthy_recent_metadata(
    tmp_path: Path, weekday: int, marker_content: str,
) -> None:
    current = datetime(2026, 9, 7, 10) + timedelta(days=weekday)
    directory = _cache(tmp_path, refreshed_at=(current - timedelta(hours=1)).isoformat())
    marker = directory / VECTOR_PUBLICATION_PENDING_FILENAME
    marker.write_text(marker_content)
    before = {path: path.read_bytes() for path in directory.iterdir()}
    decision = vector_cache_refresh_decision(directory, now=current)
    assert decision["publication_pending"] is True
    assert decision["repair_reason"] == "cache_publication_pending"
    assert decision["reason"] == "cache_publication_pending"
    assert decision["due"] is True
    assert decision["status"] == "repair_required"
    assert all(path.read_bytes() == content for path, content in before.items())


def test_pending_marker_missing_symlink_target_still_requires_repair(tmp_path: Path) -> None:
    directory = _cache(tmp_path)
    marker = directory / VECTOR_PUBLICATION_PENDING_FILENAME
    marker.symlink_to(directory / "missing")
    decision = vector_cache_refresh_decision(directory, now=datetime(2026, 9, 7, 10))
    assert decision["due"] and decision["publication_pending"]
    assert decision["repair_reason"] == "cache_publication_pending"


def test_unreadable_pending_state_is_not_healthy(tmp_path: Path, monkeypatch) -> None:
    directory = _cache(tmp_path)
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        if path.name == VECTOR_PUBLICATION_PENDING_FILENAME:
            raise PermissionError("marker unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    decision = vector_cache_refresh_decision(directory, now=datetime(2026, 9, 7, 10))
    assert decision["publication_pending"] and decision["due"]


def test_pending_repair_unavailable_blocks_even_force_and_clearing_restores_schedule(tmp_path: Path) -> None:
    directory = _cache(tmp_path)
    marker = directory / VECTOR_PUBLICATION_PENDING_FILENAME
    marker.write_text("pending")
    blocked = vector_cache_refresh_decision(
        directory, now=datetime(2026, 9, 7, 10), force=True, repair_available=False,
    )
    assert blocked["status"] == "unavailable"
    assert blocked["due"] is False and blocked["repair_required"] is True
    assert blocked["repair_reason"] == "cache_publication_pending"
    marker.unlink()
    healthy = vector_cache_refresh_decision(directory, now=datetime(2026, 9, 7, 10))
    assert healthy["publication_pending"] is False
    assert healthy["repair_required"] is False
    assert healthy["reason"] == "waiting_for_friday_close"
