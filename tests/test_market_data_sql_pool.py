from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import event, text

from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig


@pytest.fixture(autouse=True)
def isolated_engines(monkeypatch: pytest.MonkeyPatch):
    for name in ("CONNECT", "READ", "WRITE"):
        monkeypatch.setenv(f"MARKET_DATA_SQL_{name}_TIMEOUT", "10")
    module = sys.modules.get("quant.data.sql_pool")
    if module is not None:
        module.dispose_sql_engines()
    yield
    module = sys.modules.get("quant.data.sql_pool")
    if module is not None:
        module.dispose_sql_engines()


def _store(tmp_path: Path, *, sql_url: str | None = None) -> MarketDataStore:
    return MarketDataStore(
        MarketDataStoreConfig(backend="sql", root=tmp_path, sql_url=sql_url or f"sqlite:///{tmp_path / 'market.db'}")
    )


def test_engine_is_reused_across_store_instances_and_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(sqlalchemy, "create_engine", factory)

    with ThreadPoolExecutor(max_workers=8) as executor:
        engines = list(executor.map(lambda _: _store(tmp_path)._engine(), range(40)))

    assert all(engine is engines[0] for engine in engines)
    factory.assert_called_once()


def test_engine_cache_includes_url_and_mysql_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(sqlalchemy, "create_engine", factory)
    first = _store(tmp_path, sql_url="mysql+pymysql://localhost/one")._engine()
    second = _store(tmp_path, sql_url="mysql+pymysql://localhost/two")._engine()
    monkeypatch.setenv("MARKET_DATA_SQL_READ_TIMEOUT", "20")
    third_store = _store(tmp_path, sql_url="mysql+pymysql://localhost/one")
    third = third_store._engine()

    assert len({id(first), id(second), id(third)}) == 3
    assert third_store._engine() is third
    assert factory.call_count == 3
    assert factory.call_args.kwargs["connect_args"] == {
        "connect_timeout": 10, "read_timeout": 20, "write_timeout": 10
    }
    assert factory.call_args.kwargs["pool_pre_ping"] is True
    assert factory.call_args.kwargs["pool_recycle"] == 300


def test_queries_and_writes_return_connections_without_disposing_engine(tmp_path: Path) -> None:
    store = _store(tmp_path)
    engine = store._engine()
    connected = MagicMock()
    disposed = MagicMock()
    event.listen(engine, "connect", connected)
    event.listen(engine, "engine_disposed", disposed)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE market_daily (ts_code TEXT, trade_date TEXT, close REAL)"))
        connection.execute(text("INSERT INTO market_daily VALUES ('000001.SZ', '2026-09-04', 42.0)"))
        connection.execute(text("CREATE TABLE routine_dataset_revisions (dataset_id TEXT, revision INTEGER)"))
        connection.execute(text("INSERT INTO routine_dataset_revisions VALUES ('market_daily', 7)"))

    for _ in range(3):
        current = _store(tmp_path)
        assert current.read_frame("daily", "000001.SZ")["close"].tolist() == [42.0]
        assert len(current.read_market_range()) == 1
        assert current.list_symbols() == ["000001.SZ"]
        assert current.latest_trade_date("daily", "000001.SZ") == pd.Timestamp("2026-09-04")
        assert current.latest_dataset_trade_date("daily") == pd.Timestamp("2026-09-04")
        assert current.dataset_revision("daily") == 7
        current.write_frame(pd.DataFrame({"value": [42]}), "example", "key")
    with pytest.raises(RuntimeError):
        store.read_market_range("missing_table")
    assert len(store.read_market_range()) == 1

    assert connected.call_count == 1
    disposed.assert_not_called()
    assert engine.pool.checkedout() == 0


@pytest.mark.parametrize("failure", [False, True])
def test_batch_sql_writer_preserves_shared_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    store = _store(tmp_path)
    engine = MagicMock()
    monkeypatch.setattr(store, "_engine", lambda: engine)
    monkeypatch.setattr(pd.DataFrame, "to_sql", MagicMock())
    monkeypatch.setattr(sqlalchemy, "inspect", MagicMock())
    monkeypatch.setattr(sqlalchemy, "Table", MagicMock())
    connection = engine.begin.return_value.__enter__.return_value
    if failure:
        connection.execute.side_effect = RuntimeError("write failed")
    frame = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260904"], "close": [42.0]})

    if failure:
        with pytest.raises(RuntimeError, match="write failed"):
            store._write_sql_batch(frame, "daily", "trade_date", replace_partitions=True)
    else:
        assert store._write_sql_batch(frame, "daily", "trade_date", replace_partitions=True) == 1

    engine.dispose.assert_not_called()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_fork_gets_a_new_pool_without_touching_parent_connections(tmp_path: Path) -> None:
    store = _store(tmp_path)
    engine = store._engine()
    parent_pid = os.getpid()
    disposed = []
    event.listen(engine, "engine_disposed", lambda current: disposed.append(os.getpid()))
    with engine.connect() as connection:
        connection.info["owner_pid"] = parent_pid
        assert connection.execute(text("SELECT 1")).scalar() == 1
    # A fork can happen while another parent thread owns the registry lock.
    pool_module = sys.modules.get("quant.data.sql_pool")
    lock = pool_module._engine_lock if pool_module is not None else None
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def child() -> None:
        receiver.close()
        try:
            child_engine = store._engine()
            with child_engine.connect() as connection:
                sender.send((
                    child_engine is not engine,
                    connection.info.get("owner_pid") != parent_pid,
                    connection.execute(text("SELECT 1")).scalar(),
                ))
        except Exception as exc:
            sender.send(type(exc).__name__)
        finally:
            sender.close()

    process = context.Process(target=child)
    if lock is not None:
        lock.acquire()
    try:
        process.start()
    finally:
        if lock is not None:
            lock.release()
    sender.close()
    try:
        assert receiver.poll(10), "forked reader deadlocked"
        assert receiver.recv() == (True, True, 1)
        process.join(10)
        assert process.exitcode == 0
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(10)

    assert store._engine() is engine
    assert disposed == []
    with engine.connect() as connection:
        assert connection.info["owner_pid"] == parent_pid
        assert connection.execute(text("SELECT 1")).scalar() == 1
