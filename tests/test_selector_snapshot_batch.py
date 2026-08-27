import json
from types import SimpleNamespace

import pytest

from quant.webapp import services


class _FakeBegin:
    def __init__(self, engine: "_FakeEngine") -> None:
        self.engine = engine

    def __enter__(self) -> "_FakeConnection":
        self.engine.begin_count += 1
        return _FakeConnection(self.engine)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


class _FakeConnection:
    def __init__(self, engine: "_FakeEngine") -> None:
        self.engine = engine

    def execute(self, statement, parameters=None):
        sql = " ".join(str(statement).split())
        self.engine.events.append((sql, parameters))
        if self.engine.fail_insert and sql.startswith("INSERT INTO"):
            raise RuntimeError("simulated batch insert failure")
        return SimpleNamespace()


class _FakeEngine:
    def __init__(self, events: list[tuple[str, object]], *, fail_insert: bool) -> None:
        self.events = events
        self.fail_insert = fail_insert
        self.begin_count = 0
        self.dispose_count = 0

    def begin(self) -> _FakeBegin:
        return _FakeBegin(self)

    def dispose(self) -> None:
        self.dispose_count += 1


def _install_snapshot_dependencies(
    monkeypatch,
    *,
    sql_url: str | None,
    fail_insert: bool = False,
) -> tuple[list[_FakeEngine], list[tuple[str, object]], list[str]]:
    engines: list[_FakeEngine] = []
    events: list[tuple[str, object]] = []
    cache_clears: list[str] = []

    class FakeConfig:
        @classmethod
        def from_env(cls, *args, **kwargs):
            return cls()

    class FakeStore:
        def __init__(self, config) -> None:
            self.config = SimpleNamespace(sql_url=sql_url)

        def _engine(self) -> _FakeEngine:
            engine = _FakeEngine(events, fail_insert=fail_insert)
            engines.append(engine)
            return engine

    monkeypatch.setattr(services, "MarketDataStoreConfig", FakeConfig)
    monkeypatch.setattr(services, "MarketDataStore", FakeStore)
    monkeypatch.setattr(
        services,
        "_selector_snapshot_dates_cached",
        SimpleNamespace(cache_clear=lambda: cache_clears.append("clear")),
    )
    monkeypatch.setattr(services, "_SELECTOR_SNAPSHOT_SCHEMA_READY_URLS", set())
    return engines, events, cache_clears


def _selector_payload() -> dict:
    return {
        "signal_date": "2026-08-12",
        "generated_at": "2026-08-12T18:30:00",
        "available_strategies": [{"key": "B2"}, {"key": "B1"}],
        "stocks": [
            {"symbol": symbol, "model_score_available": True, "feature_quality": {"status": "complete"}}
            for symbol in ("000001.SZ", "000002.SZ")
        ],
    }


def _install_filtered_payload_stub(monkeypatch) -> None:
    def filtered(payload: dict, strategies: list[str]) -> dict:
        return {
            **payload,
            "stocks": [{
                "symbol": f"{strategies[0]}-candidate", "model_score_available": True,
                "feature_quality": {"status": "complete"},
            }],
        }

    monkeypatch.setattr(services, "_filtered_selector_payload", filtered)


def test_strategy_pool_snapshots_batch_files_sql_and_cache_invalidation(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SELECTOR_SNAPSHOT_DIR", tmp_path / "snapshots")
    _install_filtered_payload_stub(monkeypatch)
    engines, events, cache_clears = _install_snapshot_dependencies(
        monkeypatch,
        sql_url="mysql+pymysql://batch-test",
    )

    written = services._write_strategy_pool_snapshots(
        _selector_payload(),
        include_extended=True,
    )

    assert written == {"ALL": 2, "B1": 1, "B2": 1}
    snapshot_paths = sorted((tmp_path / "snapshots").glob("*.json"))
    assert len(snapshot_paths) == 3
    stored_scopes = {
        tuple(json.loads(path.read_text(encoding="utf-8"))["snapshot_scope"]["strategies"])
        for path in snapshot_paths
    }
    assert stored_scopes == {("ALL",), ("B1",), ("B2",)}
    assert all(path.suffix == ".json" for path in (tmp_path / "snapshots").iterdir())
    assert cache_clears == ["clear"]

    inserts = [(sql, params) for sql, params in events if sql.startswith("INSERT INTO")]
    creates = [sql for sql, _ in events if sql.startswith("CREATE TABLE")]
    alters = [sql for sql, _ in events if sql.startswith("ALTER TABLE")]
    assert len(inserts) == 1
    assert isinstance(inserts[0][1], list)
    assert len(inserts[0][1]) == 3
    assert len(creates) == 1
    assert len(alters) == 2
    assert len(engines) == 1
    assert engines[0].begin_count == 2  # one-time schema setup + one batch UPSERT
    assert engines[0].dispose_count == 1

    services._write_strategy_pool_snapshots(_selector_payload(), include_extended=True)

    creates = [sql for sql, _ in events if sql.startswith("CREATE TABLE")]
    inserts = [(sql, params) for sql, params in events if sql.startswith("INSERT INTO")]
    assert len(creates) == 1
    assert len(inserts) == 2
    assert len(engines) == 2
    assert engines[1].begin_count == 1
    assert engines[1].dispose_count == 1
    assert cache_clears == ["clear", "clear"]


def test_strategy_pool_snapshot_file_failure_clears_cache_once_and_skips_sql(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SELECTOR_SNAPSHOT_DIR", tmp_path / "snapshots")
    _install_filtered_payload_stub(monkeypatch)
    engines, _, cache_clears = _install_snapshot_dependencies(
        monkeypatch,
        sql_url="mysql+pymysql://file-failure-test",
    )
    real_atomic_write_text = services.atomic_write_text
    write_count = 0

    def fail_second_write(text: str, target):
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise OSError("simulated filesystem failure")
        return real_atomic_write_text(text, target)

    monkeypatch.setattr(services, "atomic_write_text", fail_second_write)

    with pytest.raises(OSError, match="simulated filesystem failure"):
        services._write_strategy_pool_snapshots(_selector_payload(), include_extended=True)

    assert len(list((tmp_path / "snapshots").glob("*.json"))) == 1
    assert cache_clears == ["clear"]
    assert engines == []


def test_strategy_pool_snapshot_sql_failure_keeps_atomic_files_and_invalidates_schema_cache(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SELECTOR_SNAPSHOT_DIR", tmp_path / "snapshots")
    _install_filtered_payload_stub(monkeypatch)
    engines, events, cache_clears = _install_snapshot_dependencies(
        monkeypatch,
        sql_url="mysql+pymysql://sql-failure-test",
        fail_insert=True,
    )

    with pytest.raises(RuntimeError, match="MySQL snapshot publication failed"):
        services._write_strategy_pool_snapshots(
            _selector_payload(),
            include_extended=True,
        )

    assert len(list((tmp_path / "snapshots").glob("*.json"))) == 3
    assert cache_clears == ["clear"]
    assert len([sql for sql, _ in events if sql.startswith("INSERT INTO")]) == 1
    assert engines[0].dispose_count == 1
    assert services._SELECTOR_SNAPSHOT_SCHEMA_READY_URLS == set()
