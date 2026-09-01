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


def _write_factor_symbol(schema_dir: Path, symbol: str, *, size: int = 16) -> None:
    symbol_dir = schema_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    (symbol_dir / "state.json").write_text("{}", encoding="utf-8")
    (symbol_dir / "2026.parquet").write_bytes(b"x" * size)


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


def test_cleanup_daily_caches_removes_replaced_factor_schema_only_after_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    factor_root = tmp_path / "data/features/daily_factor_layer"
    for index in range(4):
        _write_factor_symbol(factor_root / "signal-v1", f"old-{index}", size=20)
        _write_factor_symbol(factor_root / "signal-v2", f"new-{index}", size=30)
    monkeypatch.setattr(cache_retention, "_current_factor_schema_version", lambda: "signal-v2")

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert not (factor_root / "signal-v1").exists()
    assert (factor_root / "signal-v2").exists()
    factor_summary = summary["daily_factor_schemas"]
    assert factor_summary["current_ready"] is True
    assert factor_summary["deleted_directories"] == 1
    assert factor_summary["reclaimed_bytes"] == 4 * (20 + 2)
    assert summary["storage"]["logical_bytes_reduced"] >= factor_summary["reclaimed_bytes"]


def test_cleanup_daily_caches_preserves_previous_factor_schema_during_migration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    factor_root = tmp_path / "data/features/daily_factor_layer"
    for index in range(10):
        _write_factor_symbol(factor_root / "signal-v1", f"old-{index}")
    for index in range(5):
        _write_factor_symbol(factor_root / "signal-v2", f"new-{index}")
    monkeypatch.setattr(cache_retention, "_current_factor_schema_version", lambda: "signal-v2")

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert (factor_root / "signal-v1").exists()
    assert (factor_root / "signal-v2").exists()
    assert summary["daily_factor_schemas"]["current_ready"] is False
    assert summary["daily_factor_schemas"]["deleted_directories"] == 0


def test_cleanup_daily_caches_expires_rebuildable_root_and_probe_requests(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "data/cache"
    expired = cache_root / "sh600000_20100101_20260520_qfq.parquet"
    protected = cache_root / "sz002594_20100101_20260520_qfq.parquet"
    recent = cache_root / "sh600519_20100101_20260714_qfq.parquet"
    unrelated = cache_root / "research_input.parquet"
    _write_with_mtime(expired, datetime(2026, 6, 1), size=41)
    _write_with_mtime(protected, datetime(2026, 1, 1), size=42)
    _write_with_mtime(recent, datetime(2026, 7, 14), size=43)
    _write_with_mtime(unrelated, datetime(2025, 1, 1), size=44)

    probe = cache_root / "source_merge/tushare_live_probe/tushare_daily_basic_20260712.parquet"
    raw = tmp_path / "data/raw/daily_basic/20260712.parquet"
    _write_with_mtime(probe, datetime(2026, 7, 12), size=45)
    _write_with_mtime(raw, datetime(2026, 7, 12), size=46)

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert not expired.exists()
    assert protected.exists()
    assert recent.exists()
    assert unrelated.exists()
    assert not probe.exists()
    assert summary["root_market_request_cache"]["deleted_files"] == 1
    assert summary["root_market_request_cache"]["protected_production_files"] == 1
    assert summary["tushare_live_probe"]["deleted_files"] == 1


def test_cleanup_daily_caches_removes_only_old_abandoned_build_outputs(
    tmp_path: Path,
) -> None:
    factor_root = tmp_path / "data/features/daily_factor_layer"
    old_temp = factor_root / "current/000001.SZ/.state.tmp.parquet"
    recent_temp = factor_root / "current/000002.SZ/.state.tmp.parquet"
    old_build = (
        tmp_path
        / "data/research/similar_patterns/vector_cache/config/_matrix_cache_v1"
        / ".fingerprint.building-123-abcd"
    )
    _write_with_mtime(old_temp, datetime(2026, 7, 10), size=51)
    _write_with_mtime(recent_temp, datetime(2026, 7, 14), size=52)
    _write_with_mtime(old_build / "vectors.npy", datetime(2026, 7, 10), size=53)
    os.utime(old_build, (datetime(2026, 7, 10).timestamp(),) * 2)

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    assert not old_temp.exists()
    assert recent_temp.exists()
    assert not old_build.exists()
    abandoned = summary["abandoned_cache_builds"]
    assert abandoned["deleted_files"] == 1
    assert abandoned["deleted_directories"] == 1
    assert abandoned["reclaimed_bytes"] == 51 + 53


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


def test_cleanup_daily_caches_hardlinks_identical_protected_cb_requests(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "data/convertible_bond/tushare/tushare_cache"
    first = cache_dir / "tushare_cb_daily_20170101_all_all_all.parquet"
    second = cache_dir / "tushare_cb_daily_20170102_all_all_all.parquet"
    different = cache_dir / "tushare_cb_daily_20170103_all_all_all.parquet"
    _write_with_mtime(first, datetime(2026, 7, 10), size=61)
    _write_with_mtime(second, datetime(2026, 7, 11), size=61)
    different.parent.mkdir(parents=True, exist_ok=True)
    different.write_bytes(b"y" * 61)

    summary = cleanup_daily_caches(tmp_path, reference_date=date(2026, 7, 15))

    first_stat = first.stat()
    second_stat = second.stat()
    assert (first_stat.st_dev, first_stat.st_ino) == (
        second_stat.st_dev,
        second_stat.st_ino,
    )
    assert different.stat().st_ino != first_stat.st_ino
    cb_cache = summary["convertible_bond_request_cache"]
    assert cb_cache["protected_outside_consolidated_range"] == 3
    assert cb_cache["deduplicated_files"] == 1
    assert cb_cache["deduplicated_bytes"] == 61


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
