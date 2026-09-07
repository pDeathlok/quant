from copy import deepcopy

import pandas as pd
import pytest

from quant.routine.operation_identity import dataset_manifest_changes


def block(period, first, last, rows=1, digest="a", symbols=None):
    return {"partition": period, "min_date": first, "max_date": last,
            "rows": rows, "sha256": digest * 64, "symbols": symbols or ["A"]}


def dataset(blocks, digest="a"):
    return {"columns": ["ts_code", "trade_date", "close"], "files": [{
        "path": "daily_partitioned/symbol-A.parquet", "sha256": digest * 64,
        "rows": sum(item["rows"] for item in blocks), "symbols": ["A"],
        "min_date": min(item["min_date"] for item in blocks),
        "max_date": max(item["max_date"] for item in blocks), "date_blocks_schema": 1, "date_blocks": blocks,
    }]}


def test_append_to_same_symbol_file_only_dirties_current_month():
    history = block("202001", "20200102", "20200131", rows=20)
    september = block("202609", "20260901", "20260904", rows=4, digest="b")
    before = dataset([history, september])
    after = dataset([history, {**september, "max_date": "20260907", "rows": 5, "sha256": "c" * 64}], digest="c")
    changes = dataset_manifest_changes(before, after)
    assert changes.partitions == ("20260901",)
    assert changes.keys == ("A",)
    assert not changes.full_rebuild
    assert not dataset_manifest_changes(after, after).partitions


def test_new_month_correction_and_deleted_month_have_bounded_journals():
    history = block("202001", "20200102", "20200131", rows=20)
    recent = block("202608", "20260803", "20260831", rows=21, digest="b")
    before = dataset([history, recent])
    appended = dataset([history, recent, block("202609", "20260901", "20260901", digest="c")], digest="c")
    assert dataset_manifest_changes(before, appended).partitions == ("20260901",)
    corrected = dataset([history, {**recent, "sha256": "d" * 64}], digest="d")
    assert dataset_manifest_changes(before, corrected).partitions == ("20260803",)
    deleted = dataset([history], digest="e")
    assert dataset_manifest_changes(before, deleted).partitions == ("20260803",)


@pytest.mark.parametrize("mutation", ["missing", "short_rows", "duplicate_period", "wrong_bounds", "bad_hash", "wrong_symbols"])
def test_incomplete_block_evidence_falls_back_to_complete_chunk(mutation):
    history = block("202001", "20200102", "20200131", rows=20)
    recent = block("202609", "20260901", "20260904", rows=4, digest="b")
    before = dataset([history, recent])
    after = deepcopy(before)
    chunk = after["files"][0]
    chunk["sha256"] = "c" * 64
    if mutation == "missing":
        chunk.pop("date_blocks")
    elif mutation == "short_rows":
        chunk["date_blocks"][0]["rows"] = 1
    elif mutation == "duplicate_period":
        chunk["date_blocks"].append(history)
    elif mutation == "wrong_bounds":
        chunk["date_blocks"][0]["min_date"] = "20200103"
    elif mutation == "bad_hash":
        chunk["date_blocks"][0]["sha256"] = "unknown"
    else:
        chunk["date_blocks"][0]["symbols"] = ["other"]
    assert dataset_manifest_changes(before, after).partitions == ("20200102",)


def test_block_diff_does_not_read_source_files(monkeypatch):
    history = block("202001", "20200102", "20200102")
    before = dataset([history])
    after = dataset([history, block("202609", "20260907", "20260907", digest="b")], digest="b")

    def forbidden(*args, **kwargs):
        raise AssertionError("Journal must compare metadata, not reread rows")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    assert dataset_manifest_changes(before, after).partitions == ("20260907",)


def test_real_sql_symbol_append_keeps_historical_months_and_other_symbols_clean(tmp_path):
    import json
    from sqlalchemy import create_engine
    from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
    from quant.data.market_snapshot import export_market_snapshot

    url = f"sqlite:///{tmp_path / 'source.sqlite'}"
    engine = create_engine(url)
    store = MarketDataStore(MarketDataStoreConfig(backend="sql", sql_url=url, root=tmp_path / "raw"))
    try:
        with engine.begin() as connection:
            pd.DataFrame({"ts_code": ["A", "A", "B"], "trade_date": ["20200102", "20260904", "20200102"],
                          "close": [10., 11., 20.]}).to_sql("market_daily", connection, index=False)

        def exported(name):
            sealed = export_market_snapshot(store, tmp_path / name)
            return json.loads(sealed.manifest_path.read_text())["datasets"]["daily"]

        before = exported("before")
        with engine.begin() as connection:
            connection.exec_driver_sql("INSERT INTO market_daily VALUES ('A', '20260907', 12)")
        after = exported("after")
        changes = dataset_manifest_changes(before, after)
        assert changes.partitions == ("20260904",)
        assert changes.keys == ("A",)
        assert not changes.full_rebuild
        assert len(before["files"]) == len(after["files"]) == 2
        with engine.begin() as connection:
            connection.exec_driver_sql("UPDATE market_daily SET close=99 WHERE ts_code='A' AND trade_date='20200102'")
        corrected = exported("corrected")
        assert dataset_manifest_changes(after, corrected).partitions == ("20200102",)
        assert dataset_manifest_changes(after, corrected).keys == ("A",)
        with engine.begin() as connection:
            connection.exec_driver_sql("DELETE FROM market_daily WHERE ts_code='A' AND trade_date='20200102'")
        deleted = exported("deleted")
        assert dataset_manifest_changes(corrected, deleted).partitions == ("20200102",)
        assert dataset_manifest_changes(corrected, deleted).keys == ("A",)
    finally:
        engine.dispose()
