import json
import os
import subprocess
import sys

import pandas as pd
import pytest
from sqlalchemy import create_engine

from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
from quant.data.market_snapshot import (
    MarketSnapshotError, SnapshotReader, export_market_snapshot,
    pinned_market_environment, pinned_market_snapshot,
    pinned_dataset_path,
)


@pytest.fixture
def canonical(tmp_path):
    url = f"sqlite:///{tmp_path / 'canonical.sqlite'}"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        for dataset in ("daily", "daily_basic"):
            pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"],
                          "trade_date": ["2026-09-04", "2026-09-04"],
                          "close": [10.0, 11.0]}).to_sql(f"market_{dataset}", connection, index=False)
    yield MarketDataStore(MarketDataStoreConfig(backend="sql", sql_url=url, root=tmp_path / "raw")), engine
    engine.dispose()


def test_export_holds_one_sql_snapshot_across_tables(canonical, tmp_path, monkeypatch):
    store, engine = canonical
    original = pd.read_sql_query
    changed = []

    def chunks(*args, **kwargs):
        kwargs["chunksize"] = 1
        for frame in original(*args, **kwargs):
            yield frame
            if not changed:
                with engine.begin() as writer:
                    writer.exec_driver_sql("UPDATE market_daily SET close=99")
                    writer.exec_driver_sql("UPDATE market_daily_basic SET close=99")
                changed.append(True)

    monkeypatch.setattr(pd, "read_sql_query", chunks)
    snapshot = export_market_snapshot(store, tmp_path / "sealed", datasets=("daily", "daily_basic"))
    assert changed
    with pinned_market_snapshot(snapshot.manifest_path):
        assert store.read_market_range("daily")["close"].tolist() == [10, 11]
        assert store.read_market_range("daily_basic")["close"].tolist() == [10, 11]
        assert store.latest_dataset_trade_date("daily") == pd.Timestamp("2026-09-04")
        assert store.list_symbols("daily") == ["000001.SZ", "000002.SZ"]
        with pytest.raises(MarketSnapshotError, match="not sealed"):
            store.read_frame("top_list", "000001.SZ")
        with pytest.raises(MarketSnapshotError, match="writes are forbidden"):
            store.write_frame(pd.DataFrame(), "daily", "000001.SZ")


def test_spawned_reader_never_reopens_mutable_sql(canonical, tmp_path):
    store, engine = canonical
    snapshot = export_market_snapshot(store, tmp_path / "sealed")
    with engine.begin() as writer:
        writer.exec_driver_sql("UPDATE market_daily SET close=99")
    with pinned_market_snapshot(snapshot.manifest_path):
        env = {**os.environ, **pinned_market_environment(),
               "MARKET_DATA_BACKEND": "sql", "MARKET_DATA_SQL_URL": "invalid-driver://unavailable"}
        process = subprocess.run([sys.executable, "-c", "from quant.data.market_data_store import MarketDataStore; print(MarketDataStore().read_frame('daily','000001.SZ')['close'].iloc[0])"],
                                 env=env, capture_output=True, text=True, check=True)
    assert process.stdout.strip() == "10.0"


def test_corrupt_or_missing_sealed_data_never_falls_back(canonical, tmp_path):
    store, _ = canonical
    snapshot = export_market_snapshot(store, tmp_path / "sealed")
    with pinned_market_snapshot(snapshot.manifest_path):
        assert store.read_frame("daily", "000001.SZ")["close"].iloc[0] == 10
        entry = json.loads(snapshot.manifest_path.read_text())["datasets"]["daily"]
        chunk = snapshot.root / next(item["path"] for item in entry["files"] if "000001.SZ" in item["symbols"])
        chunk.write_bytes(b"invalid")
        with pytest.raises(MarketSnapshotError, match="changed"):
            store.read_frame("daily", "000001.SZ")
    manifest = json.loads(snapshot.manifest_path.read_text())
    manifest["status"] = "building"
    snapshot.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(MarketSnapshotError):
        SnapshotReader(snapshot.manifest_path)


def test_export_failure_does_not_leave_a_sealed_generation(canonical, tmp_path):
    store, _ = canonical
    with pytest.raises(MarketSnapshotError, match="missing"):
        export_market_snapshot(store, tmp_path / "sealed", datasets=("daily", "missing"))
    assert not (tmp_path / "sealed").exists()
    assert not list(tmp_path.glob("*.building"))


def test_partial_universe_cannot_be_used_as_full_market(canonical, tmp_path):
    store, _ = canonical
    snapshot = export_market_snapshot(store, tmp_path / "sealed", symbols=["000001.SZ"])
    with pinned_market_snapshot(snapshot.manifest_path):
        with pytest.raises(MarketSnapshotError, match="universe"):
            store.read_market_range("daily")
        assert len(store.read_frame("daily", "000001.SZ")) == 1


def test_supplemental_file_source_is_explicit_and_sealed(canonical, tmp_path):
    store, _ = canonical
    source = tmp_path / "daily_basic"
    source.mkdir()
    pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260904"],
                  "turnover_rate": [1.25]}).to_parquet(source / "daily_basic_20260904.parquet")
    sealed = export_market_snapshot(store, tmp_path / "sealed", datasets=("daily", "daily_basic"),
                                    supplemental_sources={"daily_basic": source})
    # The canonical SQL daily_basic table is deliberately different.
    with pinned_market_snapshot(sealed.manifest_path):
        directory = pinned_dataset_path("daily_basic")
        assert directory != source
        assert store.read_frame("daily_basic", "000001.SZ")["turnover_rate"].iloc[0] == 1.25
        (directory / "unexpected.parquet").write_bytes(b"invalid")
        with pytest.raises(MarketSnapshotError, match="membership"):
            pinned_dataset_path("daily_basic")


def test_empty_supplemental_source_preserves_declared_empty_result(canonical, tmp_path):
    store, _ = canonical
    source = tmp_path / "top_list"
    source.mkdir()
    sealed = export_market_snapshot(store, tmp_path / "sealed", datasets=("daily", "top_list"),
                                    supplemental_sources={"top_list": source})
    with pinned_market_snapshot(sealed.manifest_path):
        assert store.read_market_range("top_list").empty
        assert pinned_dataset_path("top_list").is_dir()


@pytest.mark.parametrize("backend", ["sql", "mysql", "parquet"])
@pytest.mark.parametrize("pin_transport", ["context", "environment"])
def test_explicit_config_and_environment_cannot_bypass_pin(
    canonical, tmp_path, monkeypatch, backend, pin_transport,
):
    from contextlib import nullcontext
    from quant.data.market_snapshot import FINGERPRINT_ENV, MANIFEST_ENV

    source, engine = canonical
    sealed = export_market_snapshot(source, tmp_path / "sealed")
    with engine.begin() as writer:
        writer.exec_driver_sql("UPDATE market_daily SET close=99")
    monkeypatch.setenv("MARKET_DATA_BACKEND", "parquet")
    monkeypatch.setenv("MARKET_DATA_ROOT", str(tmp_path / "unsealed"))
    explicit = MarketDataStore(MarketDataStoreConfig(
        backend=backend, root=tmp_path / "unsealed", sql_url=source.config.sql_url,
    ))
    if pin_transport == "environment":
        monkeypatch.setenv(MANIFEST_ENV, str(sealed.manifest_path))
        monkeypatch.setenv(FINGERPRINT_ENV, sealed.fingerprint)
    context = pinned_market_snapshot(sealed.manifest_path) if pin_transport == "context" else nullcontext()
    with context:
        for reader in (explicit, MarketDataStore()):
            assert reader.read_frame("daily", "000001.SZ")["close"].iloc[0] == 10
            assert reader.read_market_range("daily")["close"].tolist() == [10, 11]


def test_chunk_replacement_between_verification_and_read_fails_closed(
    canonical, tmp_path, monkeypatch,
):
    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    entry = json.loads(sealed.manifest_path.read_text())["datasets"]["daily"]
    chunk = sealed.root / next(item["path"] for item in entry["files"] if "000001.SZ" in item["symbols"])
    original_read = pd.read_parquet
    replacement = original_read(chunk)
    replacement["close"] = 99.0
    replacement_path = tmp_path / "replacement.parquet"
    replacement.to_parquet(replacement_path, index=False)
    replaced = []

    def replace_before_open(path, *args, **kwargs):
        if (path == chunk or hasattr(path, "fileno")) and not replaced:
            replacement_path.replace(chunk)
            replaced.append(True)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", replace_before_open)
    with pinned_market_snapshot(sealed.manifest_path):
        try:
            actual = store.read_frame("daily", "000001.SZ")
        except MarketSnapshotError:
            return
        assert replaced
        assert actual["close"].tolist() == [10.0], "unverified replacement rows escaped the pin"


def test_pin_rejects_replacement_with_another_valid_manifest(canonical, tmp_path):
    import shutil

    store, engine = canonical
    first = export_market_snapshot(store, tmp_path / "first")
    with engine.begin() as writer:
        writer.exec_driver_sql("UPDATE market_daily SET close=99")
    second = export_market_snapshot(store, tmp_path / "second")
    assert first.fingerprint != second.fingerprint
    with pinned_market_snapshot(first.manifest_path) as pin:
        for path in second.root.glob("**/*.parquet"):
            shutil.copyfile(path, first.root / path.relative_to(second.root))
        shutil.copyfile(second.manifest_path, first.manifest_path)
        try:
            actual = store.read_frame("daily", "000001.SZ")
        except MarketSnapshotError:
            return
        assert pin.fingerprint == first.fingerprint
        assert actual["close"].tolist() == [10.0], "active pin silently adopted a different valid snapshot"


@pytest.mark.parametrize("start_date", ["2026-9-4", "2026/09/04"])
def test_export_normalizes_accepted_date_strings_in_coverage(canonical, tmp_path, start_date):
    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed", start_date=start_date)
    reader = SnapshotReader(sealed.manifest_path)
    assert reader.manifest["datasets"]["daily"]["start_date"] == "20260904"
    assert len(reader.read("daily", start_date="2026-09-04")) == 2


@pytest.mark.parametrize("bound", ["start_date", "end_date"])
def test_invalid_read_dates_fail_instead_of_becoming_empty_success(canonical, tmp_path, bound):
    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    with pinned_market_snapshot(sealed.manifest_path):
        with pytest.raises((MarketSnapshotError, ValueError)):
            store.read_market_range("daily", **{bound: "not-a-date"})


def test_empty_pinned_sql_never_lists_legacy_files(canonical, tmp_path, monkeypatch):
    from quant.data.market_data_store import list_partitioned_symbol_paths

    store, engine = canonical
    with engine.begin() as writer:
        writer.exec_driver_sql("DELETE FROM market_daily")
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    legacy = tmp_path / "legacy/daily"
    legacy.mkdir(parents=True)
    pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260904"], "close": [99.0]}).to_parquet(
        legacy / "000001.SZ.parquet", index=False,
    )
    monkeypatch.setenv("MARKET_DATA_BACKEND", "parquet")
    monkeypatch.setenv("MARKET_DATA_ROOT", str(legacy.parent))
    with pinned_market_snapshot(sealed.manifest_path):
        assert store.read_market_range("daily").empty
        assert store.list_symbols("daily") == []
        assert store.latest_dataset_trade_date("daily") is None
        assert list_partitioned_symbol_paths(legacy) == []


def test_metadata_queries_do_not_materialize_all_market_columns(canonical, tmp_path, monkeypatch):
    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    original_read = pd.read_parquet
    projections = []

    def capture_projection(*args, **kwargs):
        projections.append(kwargs.get("columns"))
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", capture_projection)
    with pinned_market_snapshot(sealed.manifest_path):
        assert store.list_symbols("daily") == ["000001.SZ", "000002.SZ"]
        assert store.latest_dataset_trade_date("daily") == pd.Timestamp("2026-09-04")
        assert store.latest_trade_date("daily", "000001.SZ") == pd.Timestamp("2026-09-04")
    assert all(columns is not None and set(columns) <= {"ts_code", "trade_date"} for columns in projections)


@pytest.mark.parametrize("missing", ["manifest", "fingerprint", "wrong_fingerprint"])
def test_environment_pin_requires_matching_expected_fingerprint(canonical, tmp_path, monkeypatch, missing):
    from quant.data.market_snapshot import FINGERPRINT_ENV, MANIFEST_ENV

    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    monkeypatch.setenv(MANIFEST_ENV, str(sealed.manifest_path))
    monkeypatch.setenv(FINGERPRINT_ENV, sealed.fingerprint)
    if missing == "wrong_fingerprint":
        monkeypatch.setenv(FINGERPRINT_ENV, "0" * 64)
    else:
        monkeypatch.delenv(MANIFEST_ENV if missing == "manifest" else FINGERPRINT_ENV)
    with pytest.raises(MarketSnapshotError):
        store.read_frame("daily", "000001.SZ")


def test_context_pin_exports_identity_and_overrides_conflicting_environment(canonical, tmp_path, monkeypatch):
    from quant.data.market_snapshot import FINGERPRINT_ENV, MANIFEST_ENV

    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    monkeypatch.setenv(MANIFEST_ENV, str(tmp_path / "missing.json"))
    monkeypatch.setenv(FINGERPRINT_ENV, "0" * 64)
    with pinned_market_snapshot(sealed.manifest_path):
        assert pinned_market_environment() == {MANIFEST_ENV: str(sealed.manifest_path), FINGERPRINT_ENV: sealed.fingerprint}
        assert store.read_frame("daily", "000001.SZ")["close"].iloc[0] == 10
    with pytest.raises(MarketSnapshotError):
        store.read_frame("daily", "000001.SZ")


@pytest.mark.parametrize("pinned", [False, True])
def test_legacy_parquet_helper_forwards_projection_and_filters(canonical, tmp_path, pinned):
    from contextlib import nullcontext
    from quant.data.market_snapshot import read_pinned_parquet

    store, _ = canonical
    source = tmp_path / "daily_basic"
    source.mkdir()
    path = source / "daily_basic_20260904.parquet"
    pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"], "trade_date": ["20260904"] * 2,
                  "turnover_rate": [1.25, 2.5]}).to_parquet(path, index=False)
    sealed = export_market_snapshot(store, tmp_path / "sealed", datasets=("daily", "daily_basic"),
                                    supplemental_sources={"daily_basic": source})
    context = pinned_market_snapshot(sealed.manifest_path) if pinned else nullcontext()
    with context:
        actual_path = pinned_dataset_path("daily_basic") / path.name if pinned else path
        frame = read_pinned_parquet(actual_path, columns=["turnover_rate"], filters=[("ts_code", "==", "000002.SZ")])
        assert frame.to_dict("list") == {"turnover_rate": [2.5]}
        if pinned:
            with pytest.raises(MarketSnapshotError, match="not declared"):
                read_pinned_parquet(path)
            actual_path.write_bytes(b"broken")
            with pytest.raises(MarketSnapshotError, match="changed"):
                read_pinned_parquet(actual_path)


def test_inplace_mutation_during_pinned_parquet_decode_fails_closed(canonical, tmp_path, monkeypatch):
    from quant.data.market_snapshot import read_pinned_parquet

    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    chunk = sealed.root / json.loads(sealed.manifest_path.read_text())["datasets"]["daily"]["files"][0]["path"]
    original_read = pd.read_parquet

    def mutate_open_inode(handle, **kwargs):
        frame = original_read(handle, **kwargs)
        with chunk.open("r+b") as writer:
            writer.seek(4)
            writer.write(b"tampered")
            writer.flush()
            os.fsync(writer.fileno())
        return frame

    monkeypatch.setattr(pd, "read_parquet", mutate_open_inode)
    with pinned_market_snapshot(sealed.manifest_path):
        with pytest.raises(MarketSnapshotError, match="changed"):
            read_pinned_parquet(chunk)


@pytest.mark.parametrize("date", ["", "NaT", "2026-02-30", 20260904])
def test_export_rejects_invalid_date_before_creating_generation(canonical, tmp_path, date):
    store, _ = canonical
    with pytest.raises(MarketSnapshotError, match="date"):
        export_market_snapshot(store, tmp_path / "sealed", start_date=date)
    assert not (tmp_path / "sealed").exists()


def test_reversed_reader_date_bounds_fail_closed(canonical, tmp_path):
    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    with pinned_market_snapshot(sealed.manifest_path):
        with pytest.raises(MarketSnapshotError, match="reversed"):
            store.read_market_range("daily", start_date="20260905", end_date="20260904")


def test_daily_symbol_partitions_are_stable_across_sql_batches_and_append(canonical, tmp_path, monkeypatch):
    store, engine = canonical
    rows = pd.DataFrame({
        "ts_code": ["000001.SZ"] * 5 + ["000002.SZ"] * 3 + ["000003.SZ"] * 2,
        "trade_date": [f"2026-09-{day:02d}" for day in [1, 2, 3, 4, 5, 1, 2, 3, 1, 2]],
        "close": [float(value) for value in range(10)],
    })
    with engine.begin() as writer:
        rows.to_sql("market_daily", writer, if_exists="replace", index=False)
    original_query = pd.read_sql_query
    batch_size = [2]

    def small_batches(*args, **kwargs):
        kwargs["chunksize"] = batch_size[0]
        return original_query(*args, **kwargs)

    monkeypatch.setattr(pd, "read_sql_query", small_batches)
    first = export_market_snapshot(store, tmp_path / "first")
    batch_size[0] = 3
    second = export_market_snapshot(store, tmp_path / "second")
    assert first.fingerprint == second.fingerprint
    first_files = json.loads(first.manifest_path.read_text())["datasets"]["daily"]["files"]
    assert len(first_files) == 3
    assert all(len(chunk["symbols"]) == 1 for chunk in first_files)
    with pinned_market_snapshot(first.manifest_path):
        for symbol, expected in rows.groupby("ts_code", sort=False):
            assert store.read_frame("daily", symbol)["close"].tolist() == expected["close"].tolist()
    with engine.begin() as writer:
        writer.exec_driver_sql("INSERT INTO market_daily VALUES ('000001.SZ', '2026-09-06', 100.0)")
    third = export_market_snapshot(store, tmp_path / "third")
    third_files = json.loads(third.manifest_path.read_text())["datasets"]["daily"]["files"]
    before = {chunk["symbols"][0]: chunk for chunk in first_files}
    after = {chunk["symbols"][0]: chunk for chunk in third_files}
    assert before["000001.SZ"]["path"] == after["000001.SZ"]["path"]
    assert before["000001.SZ"]["sha256"] != after["000001.SZ"]["sha256"]
    assert before["000002.SZ"] == after["000002.SZ"]
    assert before["000003.SZ"] == after["000003.SZ"]


def test_single_symbol_read_opens_only_its_partition(canonical, tmp_path, monkeypatch):
    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    original_read = pd.read_parquet
    opened = []

    def record_open(handle, **kwargs):
        opened.append(os.fstat(handle.fileno()).st_ino)
        return original_read(handle, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", record_open)
    with pinned_market_snapshot(sealed.manifest_path):
        assert store.read_frame("daily", "000001.SZ")["close"].tolist() == [10.0]
    assert len(opened) == 1
    entry = json.loads(sealed.manifest_path.read_text())["datasets"]["daily"]
    expected = next(chunk for chunk in entry["files"] if chunk["symbols"] == ["000001.SZ"])
    assert opened == [(sealed.root / expected["path"]).stat().st_ino]


def test_metadata_uses_manifest_without_parquet_decode(canonical, tmp_path, monkeypatch):
    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("metadata must not decode parquet"))
    with pinned_market_snapshot(sealed.manifest_path):
        assert store.list_symbols() == ["000001.SZ", "000002.SZ"]
        assert store.latest_dataset_trade_date("daily") == pd.Timestamp("20260904")
        assert store.latest_trade_date("daily", "000001.SZ") == pd.Timestamp("20260904")


def test_daily_append_changes_only_current_month_content_block(canonical, tmp_path):
    from quant.routine.operation_identity import dataset_manifest_changes

    store, engine = canonical
    with engine.begin() as writer:
        pd.DataFrame({
            "ts_code": ["000001.SZ"] * 3,
            "trade_date": ["2025-01-02", "2026-08-31", "2026-09-01"],
            "close": [10.0, 11.0, 12.0],
        }).to_sql("market_daily", writer, if_exists="replace", index=False)
    previous = export_market_snapshot(store, tmp_path / "previous")
    old_chunk = json.loads(previous.manifest_path.read_text())["datasets"]["daily"]["files"][0]
    with engine.begin() as writer:
        writer.exec_driver_sql("INSERT INTO market_daily VALUES ('000001.SZ', '2026-09-02', 13.0)")
    current = export_market_snapshot(store, tmp_path / "current")
    new_chunk = json.loads(current.manifest_path.read_text())["datasets"]["daily"]["files"][0]
    assert old_chunk["path"] == new_chunk["path"]
    assert old_chunk["sha256"] != new_chunk["sha256"]
    assert old_chunk["date_blocks_schema"] == new_chunk["date_blocks_schema"] == 1
    old_blocks = {block["partition"]: block for block in old_chunk["date_blocks"]}
    new_blocks = {block["partition"]: block for block in new_chunk["date_blocks"]}
    changed = [block for key in old_blocks.keys() | new_blocks.keys() if old_blocks.get(key) != new_blocks.get(key)
               for block in (old_blocks.get(key), new_blocks.get(key)) if block is not None]
    assert {block["partition"] for block in changed} == {"202609"}
    assert min(block["min_date"] for block in changed) == "20260901"
    assert min(block["min_date"] for block in changed) != old_chunk["min_date"]
    assert len(list(current.root.glob("**/*.parquet"))) == 1
    journal = dataset_manifest_changes(
        json.loads(previous.manifest_path.read_text())["datasets"]["daily"],
        json.loads(current.manifest_path.read_text())["datasets"]["daily"],
    )
    assert not journal.full_rebuild
    assert journal.partitions == ("20260901",)
    assert journal.keys == ("000001.SZ",)


@pytest.mark.parametrize("mutation", ["correction", "deletion"])
def test_date_blocks_capture_historical_changes_without_rounding(canonical, tmp_path, mutation):
    store, engine = canonical
    with engine.begin() as writer:
        pd.DataFrame({
            "ts_code": ["000001.SZ"] * 2,
            "trade_date": ["2025-01-02", "2026-09-01"], "close": [1.0, 2.0],
        }).to_sql("market_daily", writer, if_exists="replace", index=False)
    old = export_market_snapshot(store, tmp_path / "old")
    with engine.begin() as writer:
        if mutation == "correction":
            writer.exec_driver_sql("UPDATE market_daily SET close=1.000000000000001 WHERE trade_date='2025-01-02'")
        else:
            writer.exec_driver_sql("DELETE FROM market_daily WHERE trade_date='2025-01-02'")
    new = export_market_snapshot(store, tmp_path / "new")
    def blocks(snapshot):
        chunk = json.loads(snapshot.manifest_path.read_text())["datasets"]["daily"]["files"][0]
        return {block["partition"]: block for block in chunk["date_blocks"]}
    before, after = blocks(old), blocks(new)
    assert before["202609"] == after["202609"]
    assert before["202501"] != after.get("202501")


def test_arrow_block_hash_preserves_decimal_precision_and_ignores_pandas_index():
    from decimal import Decimal
    from quant.data.market_snapshot import _date_blocks

    frame = pd.DataFrame({
        "ts_code": ["000001.SZ"], "trade_date": ["20260901"],
        "value": [Decimal("1.000000000000000001")],
    })
    before = _date_blocks(frame)
    reordered = frame[list(reversed(frame.columns))].copy()
    reordered.index = pd.Index([42], name="source_row")
    assert _date_blocks(reordered) == before
    frame.loc[0, "value"] = Decimal("1.000000000000000002")
    assert _date_blocks(frame)[0]["sha256"] != before[0]["sha256"]


def test_digest_cache_does_not_thrash_above_512_symbol_files(tmp_path, monkeypatch):
    from quant.data import market_snapshot

    assert market_snapshot._DIGEST_CACHE_SIZE >= 8192
    paths = []
    for index in range(600):
        path = tmp_path / str(index)
        path.write_bytes(b"immutable fixture")
        paths.append(path)
    expected = market_snapshot._digest(paths[0])
    original_digest = market_snapshot._handle_digest
    hashed = []

    def record_digest(handle):
        hashed.append(handle.name)
        return original_digest(handle)

    monkeypatch.setattr(market_snapshot, "_handle_digest", record_digest)
    for _ in range(2):
        for path in paths:
            with market_snapshot._verified_file(path, expected):
                pass
    assert len(hashed) == len(paths)
    assert len(market_snapshot._DIGESTS) <= market_snapshot._DIGEST_CACHE_SIZE


def test_repeated_pinned_reads_parse_manifest_only_once(canonical, tmp_path, monkeypatch):
    from quant.data import market_snapshot

    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    original_loads = json.loads
    parsed = []

    def count_manifest(value, *args, **kwargs):
        result = original_loads(value, *args, **kwargs)
        if isinstance(result, dict) and result.get("status") == "sealed":
            parsed.append(result["fingerprint"])
        return result

    monkeypatch.setattr(market_snapshot.json, "loads", count_manifest)
    with pinned_market_snapshot(sealed.manifest_path):
        first_reader = market_snapshot.current_market_snapshot()
        for _ in range(12):
            assert store.read_frame("daily", "000001.SZ")["close"].tolist() == [10.0]
            assert market_snapshot.current_market_snapshot() is first_reader
    assert parsed == [sealed.fingerprint]


def test_reader_cache_miss_is_single_parse_across_threads(canonical, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from quant.data import market_snapshot

    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    monkeypatch.setenv(market_snapshot.MANIFEST_ENV, str(sealed.manifest_path))
    monkeypatch.setenv(market_snapshot.FINGERPRINT_ENV, sealed.fingerprint)
    original_loads = json.loads
    parsed = []

    def count_manifest(value, *args, **kwargs):
        result = original_loads(value, *args, **kwargs)
        if isinstance(result, dict) and result.get("status") == "sealed":
            parsed.append(result["fingerprint"])
        return result

    monkeypatch.setattr(market_snapshot.json, "loads", count_manifest)
    with ThreadPoolExecutor(max_workers=4) as executor:
        readers = list(executor.map(lambda _: market_snapshot.current_market_snapshot(), range(12)))
    assert all(reader is readers[0] for reader in readers)
    assert parsed == [sealed.fingerprint]


@pytest.mark.parametrize("replacement", ["valid_other_snapshot", "invalid_json", "symlink"])
def test_cached_reader_rejects_manifest_replacement(canonical, tmp_path, replacement):
    from quant.data import market_snapshot

    store, engine = canonical
    first = export_market_snapshot(store, tmp_path / "first")
    with engine.begin() as writer:
        writer.exec_driver_sql("UPDATE market_daily SET close=99")
    second = export_market_snapshot(store, tmp_path / "second")
    replacement_path = tmp_path / "replacement.json"
    if replacement == "valid_other_snapshot":
        replacement_path.write_bytes(second.manifest_path.read_bytes())
    elif replacement == "invalid_json":
        replacement_path.write_bytes(b"invalid JSON")
    else:
        replacement_path.symlink_to(first.manifest_path)
    with pinned_market_snapshot(first.manifest_path):
        assert market_snapshot.current_market_snapshot().manifest["fingerprint"] == first.fingerprint
        replacement_path.replace(first.manifest_path)
        with pytest.raises(MarketSnapshotError):
            market_snapshot.current_market_snapshot()


def test_cached_reader_rechecks_expected_fingerprint(canonical, tmp_path, monkeypatch):
    from quant.data import market_snapshot

    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    monkeypatch.setenv(market_snapshot.MANIFEST_ENV, str(sealed.manifest_path))
    monkeypatch.setenv(market_snapshot.FINGERPRINT_ENV, sealed.fingerprint)
    reader = market_snapshot.current_market_snapshot()
    monkeypatch.setenv(market_snapshot.FINGERPRINT_ENV, "0" * 64)
    with pytest.raises(MarketSnapshotError):
        market_snapshot.current_market_snapshot()
    monkeypatch.setenv(market_snapshot.FINGERPRINT_ENV, sealed.fingerprint)
    assert market_snapshot.current_market_snapshot() is reader


def test_reader_cache_pin_switch_and_lru_bound(canonical, tmp_path, monkeypatch):
    from collections import OrderedDict
    from quant.data import market_snapshot

    store, engine = canonical
    first = export_market_snapshot(store, tmp_path / "first")
    with engine.begin() as writer:
        writer.exec_driver_sql("UPDATE market_daily SET close=99")
    second = export_market_snapshot(store, tmp_path / "second")
    third = export_market_snapshot(store, tmp_path / "third")
    monkeypatch.setattr(market_snapshot, "_READERS", OrderedDict())
    monkeypatch.setattr(market_snapshot, "_READER_CACHE_SIZE", 2)
    with pinned_market_snapshot(first.manifest_path):
        first_reader = market_snapshot.current_market_snapshot()
        with pinned_market_snapshot(second.manifest_path):
            assert store.read_frame("daily", "000001.SZ")["close"].tolist() == [99.0]
            assert market_snapshot.current_market_snapshot() is not first_reader
        assert market_snapshot.current_market_snapshot() is first_reader
        assert store.read_frame("daily", "000001.SZ")["close"].tolist() == [10.0]
        with pinned_market_snapshot(third.manifest_path):
            assert market_snapshot.current_market_snapshot().manifest_path == third.manifest_path
        assert len(market_snapshot._READERS) == 2
    assert market_snapshot.current_market_snapshot() is None
    assert {key[0] for key in market_snapshot._READERS} == {first.manifest_path, third.manifest_path}


def test_manifest_replacement_during_parse_is_not_cached(canonical, tmp_path, monkeypatch):
    from quant.data import market_snapshot

    store, _ = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    original_loads = json.loads
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(sealed.manifest_path.read_bytes())

    def replace_after_parse(value, *args, **kwargs):
        result = original_loads(value, *args, **kwargs)
        if isinstance(result, dict) and result.get("status") == "sealed":
            replacement.replace(sealed.manifest_path)
        return result

    monkeypatch.setattr(market_snapshot.json, "loads", replace_after_parse)
    monkeypatch.setenv(market_snapshot.MANIFEST_ENV, str(sealed.manifest_path))
    monkeypatch.setenv(market_snapshot.FINGERPRINT_ENV, sealed.fingerprint)
    with pytest.raises(MarketSnapshotError):
        market_snapshot.current_market_snapshot()
    assert all(key[0] != sealed.manifest_path for key in market_snapshot._READERS)


def test_spawned_child_caches_parse_and_rejects_wrong_expected_identity(canonical, tmp_path):
    store, engine = canonical
    sealed = export_market_snapshot(store, tmp_path / "sealed")
    with engine.begin() as writer:
        writer.exec_driver_sql("UPDATE market_daily SET close=99")
    code = """
import json, os
from quant.data import MarketDataStore
from quant.data import market_snapshot
original = json.loads
parsed = []
def count(value, *args, **kwargs):
    result = original(value, *args, **kwargs)
    if isinstance(result, dict) and result.get('status') == 'sealed':
        parsed.append(result['fingerprint'])
    return result
market_snapshot.json.loads = count
store = MarketDataStore()
for _ in range(8):
    assert store.read_frame('daily', '000001.SZ')['close'].iloc[0] == 10
assert len(parsed) == 1
os.environ[market_snapshot.FINGERPRINT_ENV] = '0' * 64
try:
    store.read_frame('daily', '000001.SZ')
except market_snapshot.MarketSnapshotError:
    print('cached-once-and-failed-closed')
else:
    raise AssertionError('cached reader ignored expected identity')
"""
    with pinned_market_snapshot(sealed.manifest_path):
        env = {**os.environ, **pinned_market_environment(), "MARKET_DATA_SQL_URL": "invalid-driver://must-not-open"}
        process = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert process.stdout.strip() == "cached-once-and-failed-closed"
