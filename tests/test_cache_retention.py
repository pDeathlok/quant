from __future__ import annotations

import fcntl
import json
import os
from datetime import date, datetime
from pathlib import Path

import pytest

from quant.data.market_data_store import MarketDataStore
from quant.infrastructure.artifact_registry import ArtifactRegistry
from quant.infrastructure.publication import PublicationStore
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


def _register_retired(root: Path, path: Path, *, last_access: float = 1) -> ArtifactRegistry:
    registry = ArtifactRegistry(root)
    registry.register(
        path, producer="test", input_versions={"source": "v1"},
        retention_class="rebuildable", state="retired", last_access=last_access,
    )
    return registry


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
    _register_retired(tmp_path, old_vector)

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


def test_cleanup_daily_caches_keeps_unowned_vectors_including_smoke(tmp_path: Path) -> None:
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

    assert sorted(path.name for path in vector_root.iterdir()) == ["first", "latest", "second"]
    assert (similar_pattern_root / "vector_cache_smoke").exists()
    assert (similar_pattern_root / "vector_cache_model_smoke").exists()
    assert summary["similar_patterns"]["kept_directory"] is None
    assert summary["similar_patterns"]["deleted_directories"] == 0
    assert summary["similar_patterns"]["smoke"] == {
        "deleted_directories": 0,
        "reclaimed_bytes": 0,
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
    _register_retired(tmp_path, old_temp, last_access=datetime(2026, 7, 10).timestamp())
    _register_retired(tmp_path, recent_temp, last_access=datetime(2026, 7, 14).timestamp())
    _register_retired(tmp_path, old_build, last_access=datetime(2026, 7, 10).timestamp())

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


def test_vectors_pin_production_previous_and_interrupted_build_not_newest_mtime(
    tmp_path: Path,
) -> None:
    vector_root = tmp_path / "data/research/similar_patterns/vector_cache"
    paths = {name: vector_root / name for name in ("production", "previous", "experiment", "retired")}
    for path in paths.values():
        _write_with_mtime(path / "cache.npz", datetime(2026, 1, 1))
        _register_retired(tmp_path, path)
    registry = ArtifactRegistry(tmp_path)
    registry.set_references("vectors:active", [paths["production"]])
    registry.set_references("vectors:previous", [paths["previous"]])
    registry.register(paths["experiment"], producer="test", input_versions={}, state="building")
    os.utime(paths["experiment"], (datetime(2026, 9, 6).timestamp(),) * 2)

    preview = cache_retention.cleanup_vector_artifacts(tmp_path)
    assert preview["candidates"] == [str(paths["retired"].relative_to(tmp_path))]
    assert paths["retired"].exists()
    result = cleanup_daily_caches(tmp_path, reference_date=date(2026, 9, 6))
    assert result["similar_patterns"]["deleted_directories"] == 1
    assert all(paths[name].exists() for name in ("production", "previous", "experiment"))
    assert not paths["retired"].exists()


@pytest.mark.parametrize("kind", ["read", "build"])
def test_vector_leases_prevent_parent_and_abandoned_build_cleanup(tmp_path: Path, kind: str) -> None:
    config = tmp_path / "data/research/similar_patterns/vector_cache/config"
    abandoned = config / "_matrix_cache_v1/.generation.building-123"
    temporary = abandoned / ".vectors.tmp.npy"
    _write_with_mtime(temporary, datetime(2020, 1, 1))
    for path in (config, abandoned, temporary):
        _register_retired(tmp_path, path)
    with ArtifactRegistry(tmp_path).lease([config], owner="vector-worker", kind=kind):
        result = cleanup_daily_caches(tmp_path, reference_date=date(2026, 9, 6))
        assert temporary.exists()
        assert result["similar_patterns"]["deleted_directories"] == 0
        assert result["abandoned_cache_builds"]["deleted_directories"] == 0


def test_unowned_old_build_and_explicit_active_config_are_kept(tmp_path: Path) -> None:
    config = tmp_path / "data/research/similar_patterns/vector_cache/config"
    temporary = config / "_matrix_cache_v1/.generation.building-123/vectors.npy"
    _write_with_mtime(temporary, datetime(2020, 1, 1))
    os.utime(temporary.parent, (1, 1))
    result = cleanup_daily_caches(tmp_path, reference_date=date(2026, 9, 6))
    assert temporary.exists()
    assert result["abandoned_cache_builds"]["deleted_directories"] == 0
    _register_retired(tmp_path, config)
    result = cleanup_daily_caches(
        tmp_path, reference_date=date(2026, 9, 6), active_vector_paths=[config],
    )
    assert config.exists()
    assert result["similar_patterns"]["deleted_directories"] == 0


def test_factor_schema_cleanup_respects_live_reader(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data/features/daily_factor_layer"
    for name in ("old", "current"):
        _write_factor_symbol(root / name, "000001.SZ")
    monkeypatch.setattr(cache_retention, "_current_factor_schema_version", lambda: "current")
    with ArtifactRegistry(tmp_path).lease([root / "old"], owner="factor-reader"):
        result = cleanup_daily_caches(tmp_path, reference_date=date(2026, 9, 6))
        assert result["daily_factor_schemas"]["deleted_directories"] == 0
        assert (root / "old").exists()


def _publication_generations(root: Path) -> tuple[Path, ArtifactRegistry]:
    directory = root / "data/publications"
    registry = ArtifactRegistry(root)
    for name in ("baseline-old", "previous", "current", "staging", "failed", "unknown"):
        path = directory / name
        _write_with_mtime(path / "tree/outputs/snapshot.json", datetime(2026, 1, 1))
        (path / "state.json").write_text(json.dumps({
            "status": "staging" if name == "staging" else "aborted" if name == "failed" else "validated",
        }))
        if name != "unknown":
            _register_retired(root, path)
    registry.commit("publication:current", directory / "baseline-old")
    registry.commit("publication:current", directory / "previous")
    registry.commit("publication:current", directory / "current")
    for name in ("baseline-old", "previous", "current"):
        registry.retire(directory / name)
    registry.register(directory / "staging", producer="test", input_versions={}, state="building")
    (directory / "current.json").write_text(json.dumps({"generation": "current"}))
    (directory / "writer.lock").touch()
    return directory, registry


def test_publication_retention_preview_and_apply_keep_previous_staging_readers(tmp_path: Path) -> None:
    directory, registry = _publication_generations(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    preview = cache_retention.cleanup_publication_generations(tmp_path)
    assert preview["status"] == "success" and preview["dry_run"] is True
    assert {Path(path).name for path in preview["candidates"]} == {"baseline-old", "failed"}
    assert all(path.read_bytes() == contents for path, contents in before.items())
    with registry.lease([directory / "baseline-old"], owner="old-reader"):
        applied = cache_retention.cleanup_publication_generations(tmp_path, dry_run=False)
        assert applied["deleted_directories"] == 1
        assert (directory / "baseline-old/tree/outputs/snapshot.json").exists()
    assert not (directory / "failed").exists()
    for name in ("current", "previous", "staging", "unknown"):
        assert (directory / name / "tree/outputs/snapshot.json").exists()


def test_publication_retention_cannot_enter_active_writer_or_delete_recent_stage(tmp_path: Path) -> None:
    directory, registry = _publication_generations(tmp_path)
    _register_retired(tmp_path, directory / "failed", last_access=datetime.now().timestamp())
    with (directory / "writer.lock").open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        result = cache_retention.cleanup_publication_generations(tmp_path, dry_run=False)
        assert result["status"] == "skipped_writer_active"
        assert (directory / "baseline-old").exists()
    result = cache_retention.cleanup_publication_generations(tmp_path, dry_run=False)
    assert result["kept"]["failed"] == "not_expired"
    assert (directory / "failed/tree/outputs/snapshot.json").exists()


def test_publication_unknown_previous_and_corrupt_pointer_fail_closed(tmp_path: Path) -> None:
    directory, registry = _publication_generations(tmp_path)
    registry.set_references("publication:current:previous", [])
    result = cache_retention.cleanup_publication_generations(tmp_path, dry_run=False)
    assert result["kept"]["previous"] == "previous_generation_unknown"
    assert (directory / "baseline-old").exists()
    (directory / "current.json").write_text("{")
    result = cache_retention.cleanup_publication_generations(tmp_path, dry_run=False)
    assert result["status"] == "unavailable"
    assert result["deleted_directories"] == 0


def test_storage_inventory_budgets_do_not_delete_research_raw_or_reports(tmp_path: Path) -> None:
    for name in ("data/research/experiment", "data/raw/history", "reports/evidence"):
        _write_with_mtime(tmp_path / name, datetime(2020, 1, 1))
    before = sorted(tmp_path.rglob("*"))
    result = cache_retention.cache_storage_inventory(tmp_path, budgets={"data": 0, "reports": 0})
    assert result["dry_run"]
    assert result["totals"]["logical_bytes"] == 48
    assert result["budgets"]["data"]["over_budget_bytes"] > 0
    assert sorted(tmp_path.rglob("*")) == before


def test_publication_pointer_previous_is_pinned_without_registry_reference(tmp_path: Path) -> None:
    directory, registry = _publication_generations(tmp_path)
    registry.set_references("publication:current:previous", [])
    (directory / "current.json").write_text(json.dumps({"generation": "current", "previous": "previous"}))
    result = cache_retention.cleanup_publication_generations(tmp_path, dry_run=False)
    assert result["status"] == "success"
    assert result["deleted_directories"] == 2
    assert (directory / "previous/tree/outputs/snapshot.json").exists()
    assert (directory / "current/tree/outputs/snapshot.json").exists()


def test_publication_staging_state_overrides_retirement_and_old_access(tmp_path: Path) -> None:
    directory, registry = _publication_generations(tmp_path)
    registry.retire(directory / "staging")
    result = cache_retention.cleanup_publication_generations(tmp_path, dry_run=False)
    assert result["kept"]["staging"] == "staging_or_unknown_state"
    assert (directory / "staging/tree/outputs/snapshot.json").exists()


def test_publication_missing_previous_fails_closed(tmp_path: Path) -> None:
    directory, _ = _publication_generations(tmp_path)
    (directory / "current.json").write_text('{"generation":"current","previous":"missing"}')
    result = cache_retention.cleanup_publication_generations(tmp_path, dry_run=False)
    assert result["status"] == "unavailable"
    assert result["deleted_directories"] == 0
    assert (directory / "baseline-old").exists()


def test_completed_publications_cap_to_current_previous_without_two_day_accumulation(tmp_path: Path) -> None:
    directory, _ = _publication_generations(tmp_path)
    _register_retired(tmp_path, directory / "baseline-old", last_access=datetime.now().timestamp())
    result = cache_retention.cleanup_publication_generations(tmp_path, dry_run=False)
    assert not (directory / "baseline-old").exists()
    assert (directory / "current").exists() and (directory / "previous").exists()
    assert result["deleted_directories"] == 2


def test_staged_snapshot_pruning_is_file_only_and_does_not_mutate_current(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "data/workspace_snapshots/example/params"
    for day in range(1, 13):
        _write_with_mtime(workspace / f"2026-09-{day:02d}.json", datetime(2026, 9, day))
    _write_with_mtime(workspace / "latest.json", datetime(2026, 9, 12))
    raw = tmp_path / "data/raw/important.json"
    _write_with_mtime(raw, datetime(2020, 1, 1))
    store = PublicationStore(tmp_path, managed=("data/workspace_snapshots",))

    def forbidden(*args, **kwargs):
        raise AssertionError("staged snapshot pruning must not run other cleanup")

    monkeypatch.setattr(cache_retention, "_cleanup_sql_snapshots", forbidden)
    monkeypatch.setattr(cache_retention, "cleanup_daily_caches", forbidden)
    with store.begin("new-generation") as view:
        current = store.view()
        committed_workspace = current.resolve(workspace)
        staged_workspace = view.resolve(workspace)
        assert len(list(committed_workspace.glob("*.json"))) == 13
        result = cache_retention.prune_staged_publication_snapshots(
            view.directory / view.generation / "tree", date(2026, 9, 12),
        )
        assert result["status"] == "success"
        assert result["snapshots"]["workspace"]["deleted_files"] == 9
        assert len(list(staged_workspace.glob("*.json"))) == 4
        assert len(list(committed_workspace.glob("*.json"))) == 13
        assert len(list(workspace.glob("*.json"))) == 13
        assert raw.exists()
        with pytest.raises(ValueError):
            cache_retention.prune_staged_publication_snapshots(current.directory / current.generation / "tree")


@pytest.mark.parametrize("unsafe", ["no_lease", "no_writer", "committed", "symlink"])
def test_staged_snapshot_pruning_rejects_unowned_or_published_tree(tmp_path: Path, unsafe: str) -> None:
    directory, registry = _publication_generations(tmp_path)
    stage = directory / "staging"
    tree = stage / "tree"
    if unsafe == "committed":
        (directory / "current.json").write_text(json.dumps({
            "generation": "current", "committed_generations": ["staging"],
        }))
    if unsafe == "symlink":
        (tree / "data").mkdir()
        (tree / "data/workspace_snapshots").symlink_to(directory / "current/tree", target_is_directory=True)
    if unsafe == "no_lease":
        with pytest.raises(ValueError, match="live build lease"):
            cache_retention.prune_staged_publication_snapshots(tree)
    else:
        with registry.lease([stage], owner="publisher", kind="build"):
            with (directory / "writer.lock").open("rb") as lock:
                if unsafe != "no_writer":
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                with pytest.raises(ValueError):
                    cache_retention.prune_staged_publication_snapshots(tree)
    assert (directory / "current/tree/outputs/snapshot.json").exists()


def test_daily_vector_retention_retries_independent_compiled_generation_gc(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    config = tmp_path / "data/research/similar_patterns/vector_cache/production"
    config.mkdir(parents=True)
    registry.register(config, producer="vectors", input_versions={}, retention_class="rebuildable", state="committed")
    registry.set_references("active-config", [config])
    for name in ("retired", "previous", "current"):
        path = config / "_matrix_cache_v1" / name
        _write_with_mtime(path / "vectors.npy", datetime(2026, 9, 6))
        registry.register(
            path, producer="vectors", input_versions={}, retention_class="rebuildable",
            state="committed", ownership_boundary=True,
        )
        registry.commit("compiled", path)
        for previous in registry.referenced_paths("compiled:previous"):
            registry.retire(previous)
    preview = cache_retention.cleanup_vector_artifacts(tmp_path)
    assert len(preview["compiled"]["collections"]["production"]["candidates"]) == 1
    assert (config / "_matrix_cache_v1/retired").exists()
    with registry.lease([config], owner="reader"):
        kept = cache_retention.cleanup_vector_artifacts(tmp_path, dry_run=False)
        assert kept["compiled"]["deleted_directories"] == 0
    applied = cache_retention.cleanup_vector_artifacts(tmp_path, dry_run=False)
    assert applied["compiled"]["deleted_directories"] == 1
    assert applied["reclaimed_bytes"] == 16
    assert not (config / "_matrix_cache_v1/retired").exists()
    assert (config / "_matrix_cache_v1/previous").exists()
    assert (config / "_matrix_cache_v1/current").exists()


@pytest.mark.parametrize("pin_config", [False, True])
def test_explicit_active_paths_protect_compiled_children(tmp_path: Path, pin_config: bool) -> None:
    registry = ArtifactRegistry(tmp_path)
    config = tmp_path / "data/research/similar_patterns/vector_cache/production"
    generation = config / "_matrix_cache_v1/retired"
    generation.mkdir(parents=True)
    registry.register(config, producer="vectors", input_versions={},
                      retention_class="rebuildable", state="committed")
    registry.register(generation, producer="vectors", input_versions={},
                      retention_class="rebuildable", state="retired", ownership_boundary=True)
    protected = config if pin_config else generation
    for dry_run in (True, False):
        result = cache_retention.cleanup_vector_artifacts(
            tmp_path, active_vector_paths=[protected], dry_run=dry_run,
        )
        collection = result["compiled"]["collections"]["production"]
        assert collection["candidates"] == []
        assert collection["kept"][str(generation.relative_to(tmp_path))] == "explicit_reference"
        assert generation.exists()


def test_activation_rotates_configs_but_keeps_production_previous_and_leases(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    root = tmp_path / "data/research/similar_patterns/vector_cache"
    configs = [root / name for name in ("old", "previous", "production", "experiment")]
    for config in configs:
        config.mkdir(parents=True)
        registry.register(config, producer="similar_patterns", input_versions={},
                          retention_class="rebuildable", state="committed")
        registry.commit(f"similar_patterns:{config.name}:vectors", config)
    for config in configs[:2]:
        cache_retention.activate_vector_config(tmp_path, config)
    with registry.lease([configs[0]], owner="reader"):
        result = cache_retention.activate_vector_config(tmp_path, configs[2])
        assert result["active_config"] == "production"
        assert result["previous_configs"] == ["previous"]
        cache_retention.cleanup_vector_artifacts(tmp_path, dry_run=False)
        assert all(config.exists() for config in configs)
    cache_retention.cleanup_vector_artifacts(tmp_path, dry_run=False)
    assert not configs[0].exists()
    assert all(config.exists() for config in configs[1:])
    marker = configs[3] / "_publication_pending.json"
    marker.write_text("{")
    with pytest.raises(ValueError, match="pending"):
        cache_retention.activate_vector_config(tmp_path, configs[3])
    assert registry.referenced_paths("similar_patterns:active_config") == (configs[2],)
