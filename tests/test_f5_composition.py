from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import contextmanager

import pytest
import pandas as pd

from quant.routine.composed_operations import execute_composed_operation, validate_composed_closure
from quant.routine.operation_contracts import InputSnapshot, ResourceClaim
from quant.routine.operation_identity import core_input_snapshots, dataset_manifest_changes
from quant.routine.production_dag import production_dag_mode, require_pinned_market, sealed_core_market_inputs
from quant.routine.resource_scheduler import ResourceBudget, ResourceScheduler, current_resource_grant


def test_default_core_and_explicit_rollback(monkeypatch):
    monkeypatch.delenv("ROUTINE_DAG_EXECUTOR", raising=False)
    assert production_dag_mode("all") == production_dag_mode("short") == "core"
    assert production_dag_mode("byd") == "legacy"
    monkeypatch.setenv("ROUTINE_DAG_EXECUTOR", "shadow")
    assert production_dag_mode("all") == "shadow"
    monkeypatch.setenv("ROUTINE_DAG_EXECUTOR", "enabled")
    with pytest.raises(ValueError, match="Full identity-audited"):
        production_dag_mode("all")


def test_sealed_inputs_do_not_hash_mutable_mirrors(tmp_path):
    sealed = InputSnapshot("content", payload={"sealed_datasets": ("daily", "daily_basic", "top_list")})
    nodes = ("data.market_daily", "data.daily_basic", "data.top_list", "source.market_daily_parquet")
    snapshots = core_input_snapshots(tmp_path, nodes, canonical_market=sealed)
    assert all(value is sealed for value in snapshots.values())
    incomplete = InputSnapshot("content", payload={"sealed_datasets": ("daily",)})
    with pytest.raises(ValueError, match="does not cover daily_basic"):
        core_input_snapshots(tmp_path, nodes, canonical_market=incomplete)


def test_sealed_export_lifetime_and_failure_cleanup(monkeypatch, tmp_path):
    import json
    from quant.routine.checkpoint_store import canonical_fingerprint

    events = []
    content = {"datasets": {"daily": {}, "daily_basic": {}, "top_list": {}}}
    fingerprint = canonical_fingerprint(content)
    content["fingerprint"] = fingerprint

    def export(store, directory, *, datasets, supplemental_sources):
        assert tuple(datasets) == ("daily", "daily_basic", "top_list")
        assert set(supplemental_sources) == {"daily_basic", "top_list"}
        directory.mkdir()
        events.append(directory)
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps(content))
        return SimpleNamespace(manifest_path=manifest, root=directory, fingerprint=fingerprint,
                               metrics={"canonical_export_seconds": 1.25})

    @contextmanager
    def pin(manifest):
        assert manifest.is_file()
        events.append("entered")
        try:
            yield SimpleNamespace(manifest_path=manifest, fingerprint=fingerprint)
        finally:
            assert manifest.is_file()
            events.append("exited")

    monkeypatch.setitem(sys.modules, "quant.data.market_snapshot", SimpleNamespace(
        export_market_snapshot=export, pinned_market_snapshot=pin,
    ))
    with pytest.raises(RuntimeError, match="consumer failed"):
        with sealed_core_market_inputs(tmp_path) as snapshot:
            assert require_pinned_market() is snapshot
            assert snapshot.changes.full_rebuild
            assert snapshot.payload["capture_metrics"] == {"canonical_export_seconds": 1.25}
            raise RuntimeError("consumer failed")
    assert events[1:] == ["entered", "exited"]
    assert not events[0].exists()
    with pytest.raises(ValueError, match="pinned_market_inputs"):
        require_pinned_market()


def test_callback_executes_on_every_run_and_preserves_payload(tmp_path):
    calls = []
    payload = {"signal_date": "2026-09-04", "stocks": []}

    def callback():
        calls.append(current_resource_grant().granted_workers)
        return payload

    for _ in range(2):
        result = execute_composed_operation(
            "build_selector_payload", callback, target_trade_date="2026-09-04",
            scope="short", project_root=tmp_path,
        )
        assert result is payload
    assert calls == [1, 1]
    assert not (tmp_path / "data/cache/routine_operations").exists()


@pytest.mark.parametrize("payload", [None, {"status": "failed"}, {"status": "shadow_only"}])
def test_callback_never_converts_missing_or_failed_result_to_success(payload, tmp_path):
    with pytest.raises(RuntimeError):
        execute_composed_operation(
            "build_selector_payload", lambda: payload, target_trade_date="2026-09-04",
            scope="short", project_root=tmp_path,
        )


def test_callback_subdivides_parent_grant_and_releases_on_failure(tmp_path):
    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=1, io_slots=1, memory_mb=512, db_connections=1))
    claim = ResourceClaim(io_slots=1, memory_mb=512, db_connections=1)
    with scheduler.activate(), scheduler.reserve(claim) as grant, grant.activate():
        with pytest.raises(RuntimeError):
            execute_composed_operation(
                "build_selector_payload", lambda: {"status": "failed"},
                target_trade_date="2026-09-04", scope="short", project_root=tmp_path,
            )
    with scheduler.reserve(claim):
        pass


def test_active_closure_rejects_unregistered_operations():
    from dataclasses import replace
    from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY, DependencyRegistry

    validate_composed_closure("all")
    validate_composed_closure("short")
    original = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    changed = DependencyRegistry([
        replace(node, operation_id="new_unmigrated_owner") if node.node_id == "product.similar" else node
        for node in original.nodes.values()
    ], original.scope_roots)
    with pytest.raises(ValueError, match="new_unmigrated_owner"):
        validate_composed_closure("all", dependencies=changed)


def test_complete_manifest_diff_tracks_append_correction_deletion():
    old = {"columns": ["ts_code", "trade_date"], "files": [
        {"path": "daily/first", "sha256": "old", "min_date": "20200102", "symbols": ["A"]},
    ]}
    assert dataset_manifest_changes(old, old).partitions == ()
    assert not dataset_manifest_changes(old, old).full_rebuild
    appended = {**old, "files": [*old["files"],
        {"path": "daily/second", "sha256": "new", "min_date": "20260907", "symbols": ["B"]},
    ]}
    change = dataset_manifest_changes(old, appended)
    assert change.partitions == ("20260907",)
    assert change.keys == ("B",)
    assert not change.full_rebuild
    corrected = {**old, "files": [{**old["files"][0], "sha256": "corrected"}]}
    assert dataset_manifest_changes(old, corrected).partitions == ("20200102",)
    assert dataset_manifest_changes(old, {**old, "files": []}).keys == ("A",)
    assert dataset_manifest_changes(None, old).full_rebuild
    assert dataset_manifest_changes(old, {**old, "columns": ["changed"]}).full_rebuild


def test_stable_symbol_chunks_do_not_dirty_unmodified_symbols():
    first = {"path": "daily/A/part-0", "sha256": "a", "min_date": "20200101", "symbols": ["A"]}
    second = {"path": "daily/B/part-0", "sha256": "b", "min_date": "20200101", "symbols": ["B"]}
    old = {"columns": ["ts_code", "trade_date"], "files": [first, second]}
    new = {**old, "files": [first, second, {
        "path": "daily/A/part-1", "sha256": "a-new", "min_date": "20260907", "symbols": ["A"],
    }]}
    changed = dataset_manifest_changes(old, new)
    assert changed.keys == ("A",)
    assert changed.partitions == ("20260907",)
    assert not changed.full_rebuild


@pytest.fixture
def sealed_real_sources(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
    from quant.data.market_snapshot import export_market_snapshot

    raw = tmp_path / "data/raw"
    basic = raw / "daily_basic"
    top = raw / "top_list"
    basic.mkdir(parents=True)
    top.mkdir()
    dates = pd.bdate_range("2026-08-03", periods=25)
    for number, date in enumerate(dates):
        pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [date.strftime("%Y%m%d")],
                      "turnover_rate": [float(number + 1)], "total_mv": [100.], "circ_mv": [50.]}).to_parquet(
            basic / f"{date:%Y%m%d}.parquet", index=False,
        )
    pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260904"],
                  "net_amount": [1.], "amount": [10.], "net_rate": [.1], "pct_change": [2.]}).to_parquet(
        top / "tushare_top_list_20260904.parquet", index=False,
    )
    url = f"sqlite:///{tmp_path / 'market.sqlite'}"
    engine = create_engine(url)
    with engine.begin() as connection:
        pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260904"], "close": [10.]}).to_sql(
            "market_daily", connection, index=False,
        )
    monkeypatch.setenv("MARKET_DATA_BACKEND", "sql")
    monkeypatch.setenv("MARKET_DATA_SQL_URL", url)
    store = MarketDataStore(MarketDataStoreConfig(backend="sql", sql_url=url, root=raw))
    sealed = export_market_snapshot(store, tmp_path / "sealed", datasets=("daily", "daily_basic", "top_list"),
                                    supplemental_sources={"daily_basic": basic, "top_list": top})
    yield sealed, basic, top, engine
    engine.dispose()


def test_actual_daily_basic_and_top_list_consumers_ignore_mutable_sources(sealed_real_sources):
    from quant.data.market_snapshot import pinned_market_snapshot
    from quant.features.variable_library import load_daily_basic_features
    from quant.features.market_sentiment import read_top_list_features

    sealed, basic, top, _ = sealed_real_sources
    expected_basic = load_daily_basic_features(basic)
    expected_top = read_top_list_features(top)
    for directory in (basic, top):
        for path in directory.glob("*.parquet"):
            path.write_bytes(b"invalid-current-source")
    with pinned_market_snapshot(sealed.manifest_path):
        pd.testing.assert_frame_equal(load_daily_basic_features(basic), expected_basic)
        pd.testing.assert_frame_equal(read_top_list_features(top), expected_top)


def test_native_child_receives_pin_and_sealed_file_args(sealed_real_sources, monkeypatch, tmp_path):
    from quant.data.market_snapshot import pinned_market_snapshot
    from quant.routine.operation_adapters import refresh_chan_model_scores
    from quant.routine.operation_contracts import OperationContext

    sealed, _, _, _ = sealed_real_sources
    recorded = []

    def run(command, **kwargs):
        recorded.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=0, stdout='{"status":"success","end":"2026-09-04"}', stderr="")

    monkeypatch.setattr("quant.routine.operation_adapters.subprocess.run", run)
    with pinned_market_snapshot(sealed.manifest_path):
        result = refresh_chan_model_scores(OperationContext(
            target_trade_date="2026-09-04", scope="short", granted_workers=1,
            upstream_results={}, identity_required=True, project_root=tmp_path,
        ))
    assert result.status == "success"
    command, env = recorded[0]
    assert env["QUANT_PINNED_MARKET_MANIFEST"] == str(sealed.manifest_path)
    assert env["DAILY_FACTOR_ROOT"] == str(sealed.root.parent / "derived_factors")
    assert command[command.index("--daily-basic-dir") + 1] == str(sealed.root / "daily_basic")
    assert command[command.index("--top-list-dir") + 1] == str(sealed.root / "top_list")


def test_published_source_manifest_preserves_dataset_specific_incremental_journals(sealed_real_sources, tmp_path):
    from quant.infrastructure.publication import PublicationStore
    from quant.routine.production_dag import save_core_source_manifest

    _, basic, _, _ = sealed_real_sources
    publication = PublicationStore(tmp_path, ("data/cache/routine_source_manifests",))
    with publication.begin("first") as view:
        with sealed_core_market_inputs(tmp_path) as first:
            save_core_source_manifest(tmp_path, first)
        publication.commit(view, {"status": "success", "freshness_audit": {"status": "success"}})
    path = sorted(basic.glob("*.parquet"))[0]
    frame = pd.read_parquet(path)
    frame["turnover_rate"] = 99.
    frame.to_parquet(path, index=False)
    with publication.begin("unpublished"):
        with sealed_core_market_inputs(tmp_path) as corrected:
            before = first.payload["dataset_fingerprints"]
            after = corrected.payload["dataset_fingerprints"]
            assert before["daily"] == after["daily"]
            assert before["daily_basic"] != after["daily_basic"]
            changes = corrected.payload["dataset_changes"]
            assert changes["daily"]["partitions"] == ()
            assert changes["daily_basic"]["partitions"] == (path.stem,)
            assert not changes["daily_basic"]["full_rebuild"]
            save_core_source_manifest(tmp_path, corrected)
    # An uncommitted attempt must not erase a repair journal for the next run.
    with publication.begin("retry"):
        with sealed_core_market_inputs(tmp_path) as retry:
            assert retry.payload["dataset_changes"]["daily_basic"]["partitions"] == (path.stem,)


def _replace_with_valid_manifest(path):
    import json
    from quant.routine.checkpoint_store import canonical_fingerprint

    manifest = json.loads(path.read_text())
    manifest.pop("fingerprint")
    manifest["replacement_marker"] = "different-valid-identity"
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("replace_before_read", [True, False])
def test_acquisition_rejects_valid_manifest_replacement(sealed_real_sources, tmp_path, monkeypatch, replace_before_read):
    from quant.data import market_snapshot

    original_export = market_snapshot.export_market_snapshot
    original_pin = market_snapshot.pinned_market_snapshot

    def export(*args, **kwargs):
        sealed = original_export(*args, **kwargs)
        if replace_before_read:
            _replace_with_valid_manifest(sealed.manifest_path)
        return sealed

    @contextmanager
    def pin(path):
        if not replace_before_read:
            _replace_with_valid_manifest(path)
        with original_pin(path) as acquired:
            yield acquired

    monkeypatch.setattr(market_snapshot, "export_market_snapshot", export)
    monkeypatch.setattr(market_snapshot, "pinned_market_snapshot", pin)
    with pytest.raises(ValueError, match="descriptor|changed after export"):
        with sealed_core_market_inputs(tmp_path):
            pytest.fail("A replaced source identity must not reach consumers")


def test_cached_core_still_validates_manifest_before_baseline_save(sealed_real_sources, tmp_path):
    from quant.data.market_snapshot import MarketSnapshotError
    from quant.infrastructure.publication import PublicationStore, publication_path
    from quant.routine.production_dag import CORE_SOURCE_MANIFEST, save_core_source_manifest

    publication = PublicationStore(tmp_path, ("data/cache/routine_source_manifests",))
    with publication.begin("cached-attempt"):
        with sealed_core_market_inputs(tmp_path) as snapshot:
            # No market reader is invoked when all heavy handlers hit cache.
            _replace_with_valid_manifest(Path(snapshot.payload["manifest_path"]))
            with pytest.raises(MarketSnapshotError, match="invalid"):
                save_core_source_manifest(tmp_path, snapshot)
            assert not publication_path(tmp_path / CORE_SOURCE_MANIFEST).exists()


def test_baseline_uses_active_reader_not_payload_path(sealed_real_sources, tmp_path):
    from dataclasses import replace
    from quant.infrastructure.publication import PublicationStore, publication_path
    from quant.routine.production_dag import CORE_SOURCE_MANIFEST, save_core_source_manifest

    publication = PublicationStore(tmp_path, ("data/cache/routine_source_manifests",))
    with publication.begin("verified-reader"):
        with sealed_core_market_inputs(tmp_path) as snapshot:
            with pytest.raises(ValueError, match="does not match"):
                save_core_source_manifest(tmp_path, replace(snapshot, fingerprint="wrong"))
            save_core_source_manifest(tmp_path, replace(snapshot, payload={"manifest_path": "/must-not-read.json"}))
            assert publication_path(tmp_path / CORE_SOURCE_MANIFEST).exists()


def test_strict_native_pipeline_reads_sealed_sql_and_supplements_in_real_child(sealed_real_sources, monkeypatch):
    from quant.data.market_snapshot import pinned_market_snapshot
    from quant.routine.operation_adapters import _strict_python
    from quant.routine.operation_contracts import OperationContext
    from quant.routine.paths import PROJECT_ROOT

    sealed, _, _, _ = sealed_real_sources
    monkeypatch.setenv("MARKET_DATA_SQL_URL", "invalid-driver://must-not-open")
    code = """
import json
from pathlib import Path
import pandas as pd
from quant.data import MarketDataStore
from quant.features.variable_library import load_daily_basic_features
from quant.features.market_sentiment import read_top_list_features
from train_chan_daily_models import read_daily_basic_features
from refresh_chan_model_live_scores import _read_daily_basic_turnover_history
from quant.research.similar_patterns import _VectorSource
basic = Path('/nonexistent-current/daily_basic')
assert MarketDataStore().read_frame('daily', '000001.SZ')['close'].iloc[0] == 10
assert _VectorSource(Path('/nonexistent-current/daily')).read([Path('000001.SZ.parquet')])['000001.SZ']['close'].iloc[0] == 10
assert len(load_daily_basic_features(basic)) == 25
assert len(read_top_list_features(Path('/nonexistent-current/top_list'))) == 1
assert len(read_daily_basic_features(basic, pd.Series(['20260904']))) == 1
assert len(_read_daily_basic_turnover_history(basic, '20260803', '20260904')) == 25
print(json.dumps({'status': 'success', 'end': '2026-09-04'}))
"""
    with pinned_market_snapshot(sealed.manifest_path):
        result = _strict_python(OperationContext(
            target_trade_date="2026-09-04", scope="short", granted_workers=1,
            upstream_results={}, identity_required=True, project_root=PROJECT_ROOT,
        ), ["-c", code], ("probe",), date_field="end")
    assert result.status == "success", result.error


@pytest.mark.parametrize("backend", ["sql", "file"])
def test_late_vector_and_thread_reads_keep_sealed_sql_after_correction(sealed_real_sources, monkeypatch, backend):
    from sqlalchemy import text
    from quant.data.market_snapshot import pinned_market_snapshot
    from quant.data import MarketDataStore
    from quant.research.similar_patterns import _VectorSource
    from quant.webapp.services import ThreadPoolExecutor

    sealed, basic, _, engine = sealed_real_sources
    with engine.begin() as connection:
        connection.execute(text("UPDATE market_daily SET close = 99"))
    monkeypatch.setenv("MARKET_DATA_BACKEND", backend)

    def forbidden(*args, **kwargs):
        raise AssertionError("private SQL bypass")

    monkeypatch.setattr(MarketDataStore, "_read_sql_range", forbidden)

    def consume():
        source = _VectorSource(basic.parent / "daily")
        assert source.token() == sealed.fingerprint
        return source.read([Path("000001.SZ.parquet")])["000001.SZ"]["close"].iloc[0]

    with pinned_market_snapshot(sealed.manifest_path):
        with ThreadPoolExecutor(max_workers=3) as executor:
            assert [future.result() for future in [executor.submit(consume) for _ in range(3)]] == [10.] * 3


def test_similar_child_receives_factory_environment_and_is_joined_on_error(sealed_real_sources, monkeypatch):
    import queue
    from quant.data.market_snapshot import pinned_market_snapshot, current_market_snapshot
    from quant.webapp import services

    sealed, _, _, _ = sealed_real_sources
    events = []
    captured = {}

    class Process:
        pid = 12345

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.alive = True

        def start(self):
            pass

        def is_alive(self):
            return self.alive

        def terminate(self):
            assert current_market_snapshot().manifest["fingerprint"] == sealed.fingerprint
            events.append("terminate")
            self.alive = False

        def join(self, timeout=None):
            events.append("join")

    def fail_queue(_):
        raise RuntimeError("queue failure")

    monkeypatch.setattr(services.mp, "get_context", lambda _: SimpleNamespace(Queue=queue.Queue, Process=Process))
    monkeypatch.setattr(services, "_drain_similar_pattern_worker_queue", fail_queue)
    monkeypatch.setattr(services, "_register_active_worker", lambda *args: None)
    monkeypatch.setattr(services, "_clear_active_worker", lambda *args: events.append("clear"))
    with pinned_market_snapshot(sealed.manifest_path):
        with pytest.raises(RuntimeError, match="queue failure"):
            services._run_similar_pattern_analysis_isolated()
    assert events == ["terminate", "join", "join", "clear"]
    environment = captured["args"][3]
    assert environment["QUANT_PINNED_MARKET_MANIFEST"] == str(sealed.manifest_path)
    assert environment["QUANT_PINNED_MARKET_FINGERPRINT"] == sealed.fingerprint

    def consume(**kwargs):
        assert current_market_snapshot().read("daily")["close"].iloc[0] == 10.
        return {"status": "success"}

    monkeypatch.setattr(services, "refresh_similar_pattern_analysis", consume)
    result_queue = queue.Queue()
    services._similar_patterns_worker(result_queue, market_environment=environment)
    assert result_queue.get_nowait()["ok"]
    assert current_market_snapshot() is None


def test_late_supplement_readers_ignore_live_files_and_reject_corruption(sealed_real_sources, monkeypatch):
    from quant.data.market_snapshot import MarketSnapshotError, pinned_market_snapshot
    from quant.research import long_dividend_quality as long
    from quant.routine.convertible_bond_allotment import _latest_daily_basic_frame
    from quant.webapp.services import _selector_turnover_feature_rows

    sealed, basic, _, _ = sealed_real_sources
    monkeypatch.setattr(long, "DAILY_BASIC_FEATURE_COLUMNS", ("ts_code", "trade_date", "turnover_rate"))
    for path in basic.glob("*.parquet"):
        path.write_bytes(b"mutable corruption")
    with pinned_market_snapshot(sealed.manifest_path):
        latest = _latest_daily_basic_frame(basic)[0]
        assert latest["turnover_rate"].iloc[0] == 25.
        assert _selector_turnover_feature_rows(["000001.SZ"], "2026-09-04", basic)
        path = sealed.root / "daily_basic" / "20260904.parquet"
        assert long._read_daily_basic_period_end(path)["turnover_rate"].iloc[0] == 25.
        path.chmod(0o600)
        path.write_bytes(b"sealed corruption")
        for consume in (lambda: _latest_daily_basic_frame(basic),
                        lambda: _selector_turnover_feature_rows(["000001.SZ"], "2026-09-04", basic),
                        lambda: long._read_daily_basic_period_end(path)):
            with pytest.raises(MarketSnapshotError):
                consume()


def test_early_market_consumers_never_swallow_pinned_errors(sealed_real_sources, monkeypatch):
    from quant.data import MarketDataStore
    from quant.data.market_snapshot import MarketSnapshotError, pinned_market_snapshot
    from quant.routine.convertible_bond_grid_plan import _underlying_stock_daily
    from quant.routine.convertible_bond_allotment import _attach_stock_market_snapshots
    from quant.application.workspaces.byd import load_byd_daily_frame

    sealed, basic, _, _ = sealed_real_sources

    def fail(*args, **kwargs):
        raise MarketSnapshotError("fixture invalid sealed source")

    monkeypatch.setattr(MarketDataStore, "read_market_range", fail)
    monkeypatch.setattr(MarketDataStore, "read_frame", fail)
    with pinned_market_snapshot(sealed.manifest_path):
        for consume in (lambda: _underlying_stock_daily(pd.DataFrame({"stk_code": ["000001.SZ"]}), "20260904"),
                        lambda: _attach_stock_market_snapshots([{"stock_code": "000001.SZ"}], daily_dir=basic.parent / "daily"),
                        lambda: load_byd_daily_frame(daily_dir=basic.parent / "daily", expected_trade_date="20260904")):
            with pytest.raises(MarketSnapshotError, match="fixture invalid"):
                consume()


def test_long_caches_reuse_across_export_roots_and_repair_only_changed_period(tmp_path, monkeypatch):
    import json
    from sqlalchemy import create_engine, text
    from quant.data import MarketDataStore, MarketDataStoreConfig
    from quant.data.market_snapshot import export_market_snapshot, pinned_market_snapshot
    from quant.research import long_dividend_quality as long

    raw = tmp_path / "raw"
    basic = raw / "daily_basic"
    basic.mkdir(parents=True)
    for date in ("20260831", "20260904"):
        record = {column: [1.] for column in long.DAILY_BASIC_FEATURE_COLUMNS}
        record.update(ts_code=["000001.SZ"], trade_date=[date])
        pd.DataFrame(record).to_parquet(basic / f"{date}.parquet", index=False)
    url = f"sqlite:///{tmp_path / 'long.sqlite'}"
    engine = create_engine(url)
    dates = pd.bdate_range(end="2026-09-04", periods=160)
    daily = pd.DataFrame({"ts_code": "000001.SZ", "trade_date": dates.strftime("%Y%m%d"),
                          "open": 10., "high": 11., "low": 9., "close": 10., "pct_chg": 0.})
    with engine.begin() as connection:
        daily.to_sql("market_daily", connection, index=False)
    monkeypatch.setenv("MARKET_DATA_BACKEND", "sql")
    monkeypatch.setenv("MARKET_DATA_SQL_URL", url)
    monkeypatch.setattr(long, "RESEARCH_CACHE_DIR", tmp_path / "cache")
    store = MarketDataStore(MarketDataStoreConfig(backend="sql", sql_url=url, root=raw))
    reads = []
    read = MarketDataStore.read_market_range

    def count_read(self, *args, **kwargs):
        reads.append(args)
        return read(self, *args, **kwargs)

    monkeypatch.setattr(MarketDataStore, "read_market_range", count_read)
    results = []
    try:
        for iteration in range(3):
            if iteration == 2:
                path = basic / "20260904.parquet"
                corrected = pd.read_parquet(path)
                corrected["dv_ttm"] = 2.
                corrected.to_parquet(path, index=False)
                with engine.begin() as connection:
                    connection.execute(text("UPDATE market_daily SET close=12 WHERE trade_date='20260904'"))
            sealed = export_market_snapshot(store, tmp_path / f"export-{iteration}",
                                            datasets=("daily", "daily_basic"), supplemental_sources={"daily_basic": basic})
            # Neither temporary export roots nor mutable canonical paths are cache identity.
            monkeypatch.setattr(long, "DAILY_DIR", tmp_path / f"unused-{iteration}" / "daily")
            with pinned_market_snapshot(sealed.manifest_path):
                sampled, coverage = long.load_daily_basic_monthly(pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-04"))
                features, _ = long.load_daily_monthly_features(pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-04"),
                                                              pd.DataFrame(), include_daily_returns=False)
                results.append((sampled, coverage, features))
            assert len(reads) == (2 if iteration == 2 else 1)
    finally:
        engine.dispose()
    assert results[0][1]["refreshed_periods"] == 2
    assert results[1][1]["cache_hit_periods"] == 2
    assert results[1][1]["refreshed_periods"] == 0
    pd.testing.assert_frame_equal(results[0][0], results[1][0])
    pd.testing.assert_frame_equal(results[0][2], results[1][2])
    assert results[2][1]["cache_hit_periods"] == 1
    assert results[2][1]["refreshed_periods"] == 1
    assert results[2][2].iloc[-1]["close"] == 12.
    manifest = json.loads((tmp_path / "cache/daily_basic_monthly_period_ends_v1.manifest.json").read_text())
    assert manifest["source_dir"] == "daily_basic:sealed"
    assert set(manifest["periods"]["2026-09"]["source"]) == {"path", "sha256"}


def test_heavy_handler_skips_unchanged_and_appends_next_day_without_full_rebuild(tmp_path):
    from quant.application.daily_dependencies import DependencyRegistry, Layer
    from quant.routine.checkpoint_store import CheckpointStore
    from quant.routine.dag_executor import DailyDagExecutor
    from quant.routine.operation_contracts import CacheMode, CachePolicy, NodeChanges
    from quant.routine.operation_adapters import result_from_payload
    from quant.routine.operation_registry import OperationRegistry
    from tests.test_daily_dag_executor import _node, _operation

    node = _node("product.signal", "signal", Layer.PRODUCT)
    definition = _operation("signal", (node.node_id,), input_ids=("data.market_daily",),
                            cache=CachePolicy(CacheMode.APPEND_STATE, "v1", ("output.txt",)))
    calls = []

    def heavy(context):
        calls.append((context.target_trade_date, context.full_rebuild))
        (tmp_path / "output.txt").write_text(context.target_trade_date)
        return result_from_payload({"status": "success", "node_changes": {
            node.node_id: NodeChanges(partitions=context.dirty_partitions, full_rebuild=context.full_rebuild),
        }}, (node.node_id,), context)

    executor = DailyDagExecutor(
        DependencyRegistry((node,), {"short": (node.node_id,)}), OperationRegistry((definition,)),
        project_root=tmp_path, checkpoint_store=CheckpointStore(tmp_path, tmp_path / "checkpoints"),
        handlers={"signal": heavy},
    )
    first = {"data.market_daily": InputSnapshot("first", changes=NodeChanges(full_rebuild=True))}
    for _ in range(2):
        result = executor.execute(target_trade_date="2026-09-04", scope="short", input_snapshots=first)
        assert result["status"] == "success"
    appended = {"data.market_daily": InputSnapshot("append", changes=NodeChanges(("20260907",), ("A",)))}
    assert executor.execute(target_trade_date="2026-09-07", scope="short", input_snapshots=appended)["status"] == "success"
    assert calls == [("2026-09-04", True), ("2026-09-07", False)]
