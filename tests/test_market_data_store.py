from __future__ import annotations

import traceback
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from quant.data import market_data_store as market
from quant.data.dataset_revision_store import DatasetRevisionStore, PartitionRevisionInput


READS = (
    "read_frame",
    "read_market_range",
    "list_symbols",
    "latest_trade_date",
    "latest_dataset_trade_date",
    "dataset_revision",
    "symbol_file",
    "symbol_file_range",
    "symbol_paths",
)
SECRET_URL = "mysql+pymysql://test_user:test_password@invalid.test/market"


def _read(store: market.MarketDataStore, operation: str):
    if operation == "read_frame":
        return store.read_frame("daily", "000001.SZ")
    if operation == "read_market_range":
        return store.read_market_range("daily", start_date="20260903", end_date="20260904")
    if operation == "latest_trade_date":
        return store.latest_trade_date("daily", "000001.SZ")
    path = store.config.root / "daily" / "000001.SZ.parquet"
    if operation == "symbol_file":
        return market.read_partitioned_symbol_file(path)
    if operation == "symbol_file_range":
        return market.read_partitioned_symbol_file(path, start_date="2026-09-03")
    if operation == "symbol_paths":
        return market.list_partitioned_symbol_paths(path.parent)
    return getattr(store, operation)("daily")


@pytest.fixture
def mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    store = market.MarketDataStore(market.MarketDataStoreConfig(backend="parquet", root=tmp_path))
    frame = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": ["20260904"], "close": [1.0]}
    )
    store.write_frame(frame, "daily", "000001.SZ")
    store.write_market_batch(frame)
    monkeypatch.setenv("MARKET_DATA_ROOT", str(tmp_path))
    return tmp_path


def _store(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str = "mysql",
    mirror_parquet: bool = True,
    sql_url: str | None = SECRET_URL,
) -> market.MarketDataStore:
    monkeypatch.setenv("MARKET_DATA_ROOT", str(root))
    monkeypatch.setenv("MARKET_DATA_BACKEND", backend)
    monkeypatch.setenv("MARKET_DATA_MIRROR_PARQUET", str(int(mirror_parquet)))
    if sql_url is None:
        monkeypatch.delenv("MARKET_DATA_SQL_URL", raising=False)
    else:
        monkeypatch.setenv("MARKET_DATA_SQL_URL", sql_url)
    return market.MarketDataStore(market.MarketDataStoreConfig.from_env(root))


def _assert_unavailable(store: market.MarketDataStore, operation: str) -> None:
    with pytest.raises(market.MarketDataUnavailableError) as caught:
        _read(store, operation)
    rendered = "".join(traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__))
    assert "test_password" not in rendered
    assert "invalid.test" not in rendered
    assert SECRET_URL not in rendered


@pytest.mark.parametrize("operation", READS)
@pytest.mark.parametrize("mirror_parquet", [True, False])
@pytest.mark.parametrize("failure_stage", ["engine", "connect", "query"])
def test_sql_failures_are_typed_and_never_read_mirrors(
    mirror: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    mirror_parquet: bool,
    failure_stage: str,
) -> None:
    store = _store(mirror, monkeypatch, mirror_parquet=mirror_parquet)
    failure = RuntimeError(f"could not read {SECRET_URL}")
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    engine_factory = MagicMock(return_value=engine)
    if failure_stage == "engine":
        engine_factory.side_effect = failure
    elif failure_stage == "connect":
        engine.connect.side_effect = failure
    else:
        connection.execute.side_effect = failure
    monkeypatch.setattr(market.MarketDataStore, "_engine", engine_factory)
    monkeypatch.setattr(DatasetRevisionStore, "_engine", engine_factory)
    monkeypatch.setattr(pd, "read_sql_query", MagicMock(side_effect=failure))
    parquet_read = MagicMock(side_effect=AssertionError("SQL must not read parquet"))
    monkeypatch.setattr(pd, "read_parquet", parquet_read)

    _assert_unavailable(store, operation)

    parquet_read.assert_not_called()
    engine.dispose.assert_not_called()


@pytest.mark.parametrize("operation", READS)
@pytest.mark.parametrize("backend", ["mysql", "sql"])
@pytest.mark.parametrize("mirror_parquet", [True, False])
def test_missing_sql_configuration_is_unavailable_not_a_file_backend(
    mirror: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    backend: str,
    mirror_parquet: bool,
) -> None:
    store = _store(mirror, monkeypatch, backend=backend, mirror_parquet=mirror_parquet, sql_url=None)
    engine_factory = MagicMock(side_effect=AssertionError("SQL URL is missing"))
    monkeypatch.setattr(market.MarketDataStore, "_engine", engine_factory)
    monkeypatch.setattr(DatasetRevisionStore, "_engine", engine_factory)
    parquet_read = MagicMock(side_effect=AssertionError("SQL must not read parquet"))
    monkeypatch.setattr(pd, "read_parquet", parquet_read)

    _assert_unavailable(store, operation)

    parquet_read.assert_not_called()
    engine_factory.assert_not_called()


@pytest.fixture
def sql_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'canonical.db'}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE market_daily (ts_code TEXT, trade_date TEXT, close REAL)"))
    revision_store = DatasetRevisionStore(sql_url=str(engine.url))
    revision_store.get("market_daily")
    monkeypatch.setattr(market.MarketDataStore, "_engine", lambda self: engine)
    monkeypatch.setattr(DatasetRevisionStore, "_engine", lambda self: engine)
    yield engine
    engine.dispose()


@pytest.mark.parametrize("operation", READS)
@pytest.mark.parametrize("mirror_parquet", [True, False])
def test_successfully_empty_sql_results_stay_empty(
    mirror: Path,
    monkeypatch: pytest.MonkeyPatch,
    sql_engine,
    operation: str,
    mirror_parquet: bool,
) -> None:
    store = _store(mirror, monkeypatch, mirror_parquet=mirror_parquet)
    parquet_read = MagicMock(side_effect=AssertionError("SQL must not read parquet"))
    monkeypatch.setattr(pd, "read_parquet", parquet_read)

    result = _read(store, operation)

    if isinstance(result, pd.DataFrame):
        assert result.empty
        assert list(result.columns) == ["ts_code", "trade_date", "close"]
    else:
        assert result == ([] if operation in {"list_symbols", "symbol_paths"} else None)
    parquet_read.assert_not_called()


@pytest.mark.parametrize("mirror_date", ["20260903", "20260904"])
def test_canonical_sql_wins_over_stale_date_and_same_date_stale_revision(
    mirror: Path, monkeypatch: pytest.MonkeyPatch, sql_engine, mirror_date: str
) -> None:
    file_store = market.MarketDataStore(market.MarketDataStoreConfig(backend="parquet", root=mirror))
    stale = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [mirror_date], "close": [1.0]})
    file_store.write_market_batch(stale)
    with sql_engine.begin() as connection:
        connection.execute(text("INSERT INTO market_daily VALUES ('000001.SZ', '2026-09-04', 42.0)"))
    revision = DatasetRevisionStore(sql_url=str(sql_engine.url)).commit(
        "market_daily", {"20260904": PartitionRevisionInput(1, "a" * 64)}, watermark="20260904"
    )
    store = _store(mirror, monkeypatch)

    assert store.read_frame("daily", "000001.SZ")["close"].tolist() == [42.0]
    assert store.read_market_range()["trade_date"].tolist() == ["20260904"]
    assert store.list_symbols() == ["000001.SZ"]
    assert store.latest_trade_date("daily", "000001.SZ") == pd.Timestamp("2026-09-04")
    assert store.latest_dataset_trade_date("daily") == pd.Timestamp("2026-09-04")
    assert store.dataset_revision() == revision.revision


@pytest.mark.parametrize("operation", READS)
def test_missing_sql_tables_are_unavailable(
    mirror: Path, monkeypatch: pytest.MonkeyPatch, sql_engine, operation: str
) -> None:
    with sql_engine.begin() as connection:
        connection.execute(text("DROP TABLE market_daily"))
        connection.execute(text("DROP TABLE routine_dataset_revisions"))
    store = _store(mirror, monkeypatch)

    _assert_unavailable(store, operation)


@pytest.mark.parametrize("operation", READS)
@pytest.mark.parametrize("mirror_parquet", [True, False])
@pytest.mark.parametrize("backend", ["parquet", "file"])
def test_explicit_file_backend_preserves_reads_and_metadata(
    mirror: Path, monkeypatch: pytest.MonkeyPatch, operation: str, mirror_parquet: bool, backend: str
) -> None:
    store = _store(mirror, monkeypatch, backend=backend, mirror_parquet=mirror_parquet)
    engine_factory = MagicMock(side_effect=AssertionError("file backend must not use SQL"))
    monkeypatch.setattr(market.MarketDataStore, "_engine", engine_factory)
    monkeypatch.setattr(DatasetRevisionStore, "_engine", engine_factory)

    result = _read(store, operation)

    if isinstance(result, pd.DataFrame):
        assert result["close"].tolist() == [1.0]
    elif operation == "list_symbols":
        assert result == ["000001.SZ"]
    elif operation == "symbol_paths":
        assert result == [mirror / "daily" / "000001.SZ.parquet"]
    elif operation == "dataset_revision":
        assert result == 1
    else:
        assert result == pd.Timestamp("2026-09-04")
    engine_factory.assert_not_called()


def test_explicit_file_backend_keeps_legacy_symbol_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, monkeypatch, backend="parquet")
    frame = pd.DataFrame({"trade_date": ["20260904"], "close": [1.0]})
    store.write_frame(frame, "daily", "000001.SZ")

    assert _read(store, "symbol_paths") == [tmp_path / "daily" / "000001.SZ.parquet"]
    pd.testing.assert_frame_equal(_read(store, "symbol_file"), frame)


@pytest.mark.parametrize("operation", ["read_frame", "latest_trade_date", "latest_dataset_trade_date"])
def test_invalid_sql_dates_are_unavailable_not_empty_metadata(
    mirror: Path, monkeypatch: pytest.MonkeyPatch, sql_engine, operation: str
) -> None:
    with sql_engine.begin() as connection:
        connection.execute(text("INSERT INTO market_daily VALUES ('000001.SZ', 'invalid-date', 42.0)"))
    store = _store(mirror, monkeypatch)

    _assert_unavailable(store, operation)


@pytest.mark.parametrize("operation", ["latest_trade_date", "latest_dataset_trade_date"])
def test_sql_latest_dates_accept_datetime_values(
    mirror: Path, monkeypatch: pytest.MonkeyPatch, sql_engine, operation: str
) -> None:
    with sql_engine.begin() as connection:
        connection.execute(text("INSERT INTO market_daily VALUES ('000001.SZ', '2026-09-04 00:00:00', 42.0)"))
    store = _store(mirror, monkeypatch)

    assert _read(store, operation) == pd.Timestamp("2026-09-04")


@pytest.mark.parametrize("missing_table", [True, False])
def test_batch_write_can_initialize_or_extend_sql_schema_without_using_mirror(
    mirror: Path, monkeypatch: pytest.MonkeyPatch, sql_engine, missing_table: bool
) -> None:
    if missing_table:
        with sql_engine.begin() as connection:
            connection.execute(text("DROP TABLE market_daily"))
    store = _store(mirror, monkeypatch, sql_url=str(sql_engine.url))
    incoming = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": ["20260904"], "close": [42.0], "new_column": [1.0]}
    )
    sql_write = MagicMock(return_value=1)
    monkeypatch.setattr(store, "_write_sql_batch", sql_write)
    parquet_read = MagicMock(side_effect=AssertionError("must not compare against a mirror"))
    monkeypatch.setattr(pd, "read_parquet", parquet_read)
    monkeypatch.setattr(store, "_write_partitioned_parquet", MagicMock(return_value=1))

    result = store.write_market_batch(incoming)

    assert result["sql_rows"] == 1
    assert result["changed_keys"] == ["000001.SZ"]
    sql_write.assert_called_once()
    parquet_read.assert_not_called()


def test_batch_write_source_outage_does_not_publish_files_or_revision(
    mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(mirror, monkeypatch)
    monkeypatch.setattr(store, "_engine", MagicMock(side_effect=RuntimeError(SECRET_URL)))
    sql_write = MagicMock()
    parquet_write = MagicMock()
    revision_store = MagicMock()
    monkeypatch.setattr(store, "_write_sql_batch", sql_write)
    monkeypatch.setattr(store, "_write_partitioned_parquet", parquet_write)
    monkeypatch.setattr(store, "_revision_store", revision_store)

    with pytest.raises(market.MarketDataUnavailableError):
        store.write_market_batch(
            pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260904"], "close": [42.0]})
        )

    sql_write.assert_not_called()
    parquet_write.assert_not_called()
    revision_store.assert_not_called()


def test_sql_filtered_empty_result_keeps_projection_and_ignores_mirror(
    mirror: Path, monkeypatch: pytest.MonkeyPatch, sql_engine
) -> None:
    with sql_engine.begin() as connection:
        connection.execute(text("INSERT INTO market_daily VALUES ('000001.SZ', '20260904', 42.0)"))
    store = _store(mirror, monkeypatch)

    result = store.read_market_range(
        start_date="20260904", end_date="20260904", symbols=["000002.SZ"], columns=["close"]
    )

    assert result.empty
    assert list(result.columns) == ["close"]
    assert store.latest_trade_date("daily", "000002.SZ") is None


@pytest.mark.parametrize("operation", READS)
def test_missing_sql_dependency_is_typed(
    mirror: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = _store(mirror, monkeypatch)
    monkeypatch.setattr(
        market.MarketDataStore, "_engine", MagicMock(side_effect=ImportError("missing driver"))
    )

    _assert_unavailable(store, operation)


@pytest.mark.parametrize("operation", READS)
def test_sql_connection_cleanup_failure_is_not_reported_as_success(
    mirror: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = _store(mirror, monkeypatch)
    engine = MagicMock()
    connection_scope = engine.connect.return_value
    connection_scope.__exit__.side_effect = RuntimeError(SECRET_URL)
    result = connection_scope.__enter__.return_value.execute.return_value
    result.scalar_one.return_value = None
    result.scalar_one_or_none.return_value = None
    monkeypatch.setattr(market.MarketDataStore, "_engine", MagicMock(return_value=engine))
    monkeypatch.setattr(
        pd, "read_sql_query", MagicMock(return_value=pd.DataFrame(columns=["ts_code", "trade_date", "close"]))
    )

    _assert_unavailable(store, operation)

    engine.dispose.assert_not_called()
