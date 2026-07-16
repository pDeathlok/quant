from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

from quant.routine.cache_retention import cleanup_daily_caches


def _write_with_mtime(path: Path, modified_at: datetime, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    timestamp = modified_at.timestamp()
    os.utime(path, (timestamp, timestamp))


def test_cleanup_daily_caches_applies_requested_retention_rules(tmp_path: Path) -> None:
    long_cache = tmp_path / "data/research/long_dividend_quality"
    expired_return = long_cache / "daily_returns_expired.parquet"
    expired_features = long_cache / "daily_monthly_features_expired.parquet"
    boundary_return = long_cache / "daily_returns_boundary.parquet"
    current_features = long_cache / "daily_monthly_features_current.parquet"
    unrelated = long_cache / "research_manifest.json"
    _write_with_mtime(expired_return, datetime(2026, 4, 14, 23, 59), size=20)
    _write_with_mtime(expired_features, datetime(2026, 1, 1), size=30)
    _write_with_mtime(boundary_return, datetime(2026, 4, 15), size=40)
    _write_with_mtime(current_features, datetime(2026, 7, 15), size=50)
    _write_with_mtime(unrelated, datetime(2025, 1, 1), size=60)

    vector_root = tmp_path / "data/research/similar_patterns/vector_cache"
    old_vector = vector_root / "old_config"
    current_vector = vector_root / "current_config"
    _write_with_mtime(old_vector / "000001_SZ.npz", datetime(2026, 6, 1), size=70)
    _write_with_mtime(current_vector / "000001_SZ.npz", datetime(2026, 7, 15), size=80)
    os.utime(old_vector, (datetime(2026, 6, 1).timestamp(),) * 2)
    os.utime(current_vector, (datetime(2026, 7, 15).timestamp(),) * 2)

    tushare_cache = tmp_path / "data/cache/source_merge/tushare"
    tushare_file = tushare_cache / "tushare_000001.SZ_20260714_20260715_None.parquet"
    _write_with_mtime(tushare_file, datetime(2026, 7, 15), size=90)

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert not expired_return.exists()
    assert not expired_features.exists()
    assert boundary_return.exists()
    assert current_features.exists()
    assert unrelated.exists()
    assert not old_vector.exists()
    assert current_vector.exists()
    assert tushare_file.exists()
    assert summary["long_strategy"]["deleted_files"] == 2
    assert summary["similar_patterns"]["deleted_directories"] == 1
    assert summary["reclaimed_bytes"] == 20 + 30 + 70
    assert summary["errors"] == []


def test_cleanup_daily_caches_keeps_only_newest_vector_directory(tmp_path: Path) -> None:
    vector_root = tmp_path / "data/research/similar_patterns/vector_cache"
    for name, modified_at in [
        ("first", datetime(2026, 5, 1)),
        ("second", datetime(2026, 6, 1)),
        ("latest", datetime(2026, 7, 1)),
    ]:
        directory = vector_root / name
        _write_with_mtime(directory / "cache.npz", modified_at)
        os.utime(directory, (modified_at.timestamp(),) * 2)

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert sorted(path.name for path in vector_root.iterdir()) == ["latest"]
    assert summary["similar_patterns"]["kept_directory"] == "latest"
    assert summary["similar_patterns"]["deleted_directories"] == 2

