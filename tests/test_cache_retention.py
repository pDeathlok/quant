from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pytest

from quant.data.market_data_store import MarketDataStore
from quant.routine import cache_retention
from quant.routine.cache_retention import _snapshot_keys_to_delete, cleanup_daily_caches


@pytest.fixture(autouse=True)
def _disable_sql_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARKET_DATA_SQL_URL", raising=False)
    monkeypatch.setenv("MARKET_DATA_BACKEND", "file")


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
    tushare_boundary = tushare_cache / "tushare_000002.SZ_20260707_20260708_None.parquet"
    _write_with_mtime(tushare_boundary, datetime(2026, 7, 8), size=100)
    tushare_expired = tushare_cache / "tushare_000003.SZ_20260706_20260707_None.parquet"
    _write_with_mtime(tushare_expired, datetime(2026, 7, 7, 23, 59), size=110)
    expired_daily_basic_cache = tushare_cache / "tushare_daily_basic_20260707.parquet"
    expired_daily_basic_raw = tmp_path / "data/raw/daily_basic/20260707.parquet"
    recent_daily_basic_cache = tushare_cache / "tushare_daily_basic_20260714.parquet"
    orphan_daily_basic_cache = tushare_cache / "tushare_daily_basic_20260706.parquet"
    _write_with_mtime(expired_daily_basic_cache, datetime(2026, 7, 7), size=120)
    _write_with_mtime(expired_daily_basic_raw, datetime(2026, 7, 7), size=121)
    _write_with_mtime(recent_daily_basic_cache, datetime(2026, 7, 14), size=122)
    _write_with_mtime(orphan_daily_basic_cache, datetime(2026, 7, 6), size=123)

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert not expired_return.exists()
    assert not expired_features.exists()
    assert boundary_return.exists()
    assert current_features.exists()
    assert unrelated.exists()
    assert not old_vector.exists()
    assert current_vector.exists()
    assert tushare_file.exists()
    assert tushare_boundary.exists()
    assert not tushare_expired.exists()
    assert not expired_daily_basic_cache.exists()
    assert expired_daily_basic_raw.exists()
    assert recent_daily_basic_cache.exists()
    assert orphan_daily_basic_cache.exists()
    assert summary["long_strategy"]["retention_versions"] == 2
    assert summary["long_strategy"]["kept_versions"] == ["boundary", "current"]
    assert summary["long_strategy"]["deleted_files"] == 2
    assert summary["similar_patterns"]["deleted_directories"] == 1
    assert summary["tushare_single_symbol"] == {
        "retention_days": 7,
        "cutoff_date": "2026-07-08",
        "deleted_files": 1,
        "reclaimed_bytes": 110,
    }
    assert summary["tushare_daily_basic"] == {
        "retention_days": 7,
        "cutoff_date": "2026-07-08",
        "deleted_files": 1,
        "protected_without_raw": 1,
        "reclaimed_bytes": 120,
    }
    assert summary["reclaimed_bytes"] == 20 + 30 + 70 + 110 + 120
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
    similar_pattern_root = tmp_path / "data/research/similar_patterns"
    _write_with_mtime(similar_pattern_root / "vector_cache_smoke/cache.npz", datetime(2026, 7, 1), size=21)
    _write_with_mtime(
        similar_pattern_root / "vector_cache_model_smoke/cache.npz",
        datetime(2026, 7, 1),
        size=22,
    )

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert sorted(path.name for path in vector_root.iterdir()) == ["latest"]
    assert not (similar_pattern_root / "vector_cache_smoke").exists()
    assert not (similar_pattern_root / "vector_cache_model_smoke").exists()
    assert summary["similar_patterns"]["kept_directory"] == "latest"
    assert summary["similar_patterns"]["deleted_directories"] == 2
    assert summary["similar_patterns"]["smoke"] == {
        "deleted_directories": 2,
        "reclaimed_bytes": 43,
    }


def test_cleanup_daily_caches_keeps_two_complete_long_cache_versions(tmp_path: Path) -> None:
    long_cache = tmp_path / "data/research/long_dividend_quality"
    for index, name in enumerate(["old", "previous", "latest"], start=1):
        modified_at = datetime(2026, index, 1)
        _write_with_mtime(long_cache / f"daily_returns_{name}.parquet", modified_at)
        _write_with_mtime(long_cache / f"daily_monthly_features_{name}.parquet", modified_at)

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert sorted(path.name for path in long_cache.iterdir()) == [
        "daily_monthly_features_latest.parquet",
        "daily_monthly_features_previous.parquet",
        "daily_returns_latest.parquet",
        "daily_returns_previous.parquet",
    ]
    assert summary["long_strategy"]["kept_versions"] == ["previous", "latest"]
    assert summary["long_strategy"]["deleted_files"] == 2


def test_cleanup_daily_caches_removes_cb_requests_covered_by_consolidated_data(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data/convertible_bond/tushare"
    _write_with_mtime(
        data_dir / "cb_daily_20200101_20260714.parquet",
        datetime(2026, 7, 15),
    )
    covered = data_dir / "tushare_cache/tushare_cb_daily_20260714_all_all_all.parquet"
    outside = data_dir / "tushare_cache/tushare_cb_daily_20191231_all_all_all.parquet"
    _write_with_mtime(covered, datetime(2026, 7, 14), size=51)
    _write_with_mtime(outside, datetime(2026, 7, 14), size=52)

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert not covered.exists()
    assert outside.exists()
    assert summary["convertible_bond_request_cache"]["deleted_files"] == 1
    assert summary["convertible_bond_request_cache"][
        "protected_outside_consolidated_range"
    ] == 1
    assert summary["convertible_bond_request_cache"]["reclaimed_bytes"] == 51


def test_cleanup_daily_caches_caps_and_hardlinks_b1_research_reports(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports/b1/research/xgb_project_vars_strategy"
    oldest = report_dir / "z_skill_trade_samples_20260712_120000.csv"
    previous = report_dir / "z_skill_trade_samples_20260713_120000.csv"
    newest = report_dir / "z_skill_trade_samples_20260714_120000.csv"
    latest = report_dir / "latest_z_skill_trade_samples.csv"
    _write_with_mtime(oldest, datetime(2026, 7, 12), size=11)
    _write_with_mtime(previous, datetime(2026, 7, 13), size=12)
    _write_with_mtime(newest, datetime(2026, 7, 14), size=13)
    _write_with_mtime(latest, datetime(2026, 7, 14), size=13)

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert not oldest.exists()
    assert not previous.exists()
    assert newest.exists()
    assert latest.exists()
    assert (latest.stat().st_dev, latest.stat().st_ino) == (
        newest.stat().st_dev,
        newest.stat().st_ino,
    )
    reports = summary["b1_research_reports"]
    assert reports["deleted_files"] == 2
    assert reports["linked_latest_files"] == 1
    assert reports["reclaimed_bytes"] == 36


def test_cleanup_daily_caches_caps_snapshot_versions_and_keeps_newest_old_result(
    tmp_path: Path,
) -> None:
    selector_dir = tmp_path / "data/selector_snapshots"
    selector_dir.mkdir(parents=True)
    for day in range(1, 13):
        path = selector_dir / f"selector-{day}.json"
        path.write_text(
            '{"signal_date":"2026-07-%02d","generated_at":"2026-07-%02dT18:00:00",'
            '"snapshot_scope":{"strategies":["B1"],"include_extended":false}}'
            % (day, day),
            encoding="utf-8",
        )
    for day in range(1, 4):
        path = selector_dir / f"old-{day}.json"
        path.write_text(
            '{"signal_date":"2026-05-%02d","generated_at":"2026-05-%02dT18:00:00",'
            '"snapshot_scope":{"strategies":["B2"],"include_extended":false}}'
            % (day, day),
            encoding="utf-8",
        )

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    kept = {path.stem for path in selector_dir.glob("*.json")}
    assert {"selector-1", "selector-2"}.isdisjoint(kept)
    assert {f"selector-{day}" for day in range(3, 13)}.issubset(kept)
    assert "old-3" in kept
    assert {"old-1", "old-2"}.isdisjoint(kept)
    assert summary["snapshots"]["selector"]["deleted_files"] == 4


def test_cleanup_daily_caches_limits_workspace_audit_and_routine_history(tmp_path: Path) -> None:
    workspace = tmp_path / "data/workspace_snapshots/cb/params"
    workspace.mkdir(parents=True)
    for day in range(10, 15):
        _write_with_mtime(workspace / f"2026-07-{day}.json", datetime(2026, 7, day))
    _write_with_mtime(workspace / "latest.json", datetime(2026, 7, 14))

    audit_root = tmp_path / "data/raw/source_audit"
    for day in range(1, 13):
        _write_with_mtime(audit_root / f"202607{day:02d}_120000" / "manifest.json", datetime(2026, 7, day))

    routine_root = tmp_path / "data/routine"
    for day in range(8, 15):
        _write_with_mtime(routine_root / f"202607{day:02d}_180000" / "plan.json", datetime(2026, 7, day))
    _write_with_mtime(routine_root / "latest_refresh_status.json", datetime(2026, 1, 1))

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert sorted(path.stem for path in workspace.glob("*.json")) == [
        "2026-07-12",
        "2026-07-13",
        "2026-07-14",
        "latest",
    ]
    assert len(list(audit_root.iterdir())) == 10
    assert len([path for path in routine_root.iterdir() if path.is_dir()]) == 5
    assert (routine_root / "latest_refresh_status.json").exists()
    assert summary["snapshots"]["workspace"]["deleted_files"] == 2
    assert summary["source_audit"]["deleted_directories"] == 2
    assert summary["routine_runs"]["deleted_directories"] == 2


def test_snapshot_keys_to_delete_deduplicates_dates_and_protects_latest() -> None:
    records = [
        {"snapshot_key": "latest", "signal_date": "LATEST", "group": "A"},
        {"snapshot_key": "new-schema", "signal_date": "2026-07-14", "group": "A", "updated_at": "2"},
        {"snapshot_key": "old-schema", "signal_date": "2026-07-14", "group": "A", "updated_at": "1"},
        {"snapshot_key": "expired", "signal_date": "2026-01-01", "group": "A"},
        {"snapshot_key": "invalid", "signal_date": "unknown", "group": "A"},
    ]

    deleted = _snapshot_keys_to_delete(
        records,
        date_field="signal_date",
        group_fields=("group",),
        cutoff=date(2026, 7, 1),
        max_versions=3,
        latest_values={"LATEST"},
    )

    assert deleted == ["old-schema", "expired"]


def test_cleanup_daily_caches_preserves_legacy_selector_without_complete_scope(
    tmp_path: Path,
) -> None:
    selector_dir = tmp_path / "data/selector_snapshots"
    selector_dir.mkdir(parents=True)
    legacy = selector_dir / "unknown-key.json"
    legacy.write_text(
        '{"signal_date":"2020-01-01","snapshot_scope":{"strategies":["B1"]}}',
        encoding="utf-8",
    )

    cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert legacy.exists()


def test_cleanup_sql_snapshots_returns_failure_when_engine_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MARKET_DATA_BACKEND", "mysql")
    monkeypatch.setenv("MARKET_DATA_SQL_URL", "mysql+pymysql://test")
    monkeypatch.setattr(
        MarketDataStore,
        "_engine",
        lambda self: (_ for _ in ()).throw(RuntimeError("database offline")),
    )
    errors: list[str] = []

    summary = cache_retention._cleanup_sql_snapshots(tmp_path, date(2026, 7, 15), errors)

    assert summary["status"] == "failed"
    assert errors == ["sql:connection:database offline"]
