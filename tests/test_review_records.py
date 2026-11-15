import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant.webapp import review_records as store
from quant.webapp.api import router


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "REVIEW_RECORDS_PATH", tmp_path / "review_records.json")
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_snapshot_and_note_survive_reload_without_overwriting_entry(client):
    original = {
        "symbol": "000001.sz", "name": "平安银行", "source": "短线策略",
        "signal_date": "2026-11-11", "information": "命中 B1，模型分 86.5",
        "snapshot": {"row": {"model_score_normalized": 86.5, "matched_groups": ["B1"]}},
    }
    response = client.post("/api/review-records", json=original)
    assert response.status_code == 200
    entry = response.json()["record"]
    assert entry["symbol"] == "000001.SZ"
    assert entry["xueqiu_url"] == "https://xueqiu.com/S/SZ000001"
    assert entry["created_at"].endswith("+08:00")
    original["snapshot"]["row"]["model_score_normalized"] = 12
    path = f'/api/review-records/{entry["id"]}/note'
    assert client.put(path, json={"note": "观察量能\n等待确认"}).status_code == 200
    reloaded = client.get("/api/review-records").json()["records"][0]
    assert reloaded["note"] == "观察量能\n等待确认"
    for field in ("snapshot", "information", "signal_date", "source", "created_at"):
        assert reloaded[field] == entry[field]
    assert client.put(path, json={"note": "", "snapshot": {}}).status_code == 422
    assert client.put(path, json={"note": ""}).status_code == 200
    assert json.loads(store.REVIEW_RECORDS_PATH.read_text())["records"][0]["note"] == ""


def test_separate_entries_for_same_stock_and_delete_only_target(client):
    first = client.post("/api/review-records", json={"symbol": "920001", "signal_date": "2026-11-10"}).json()["record"]
    second = client.post("/api/review-records", json={"symbol": "920001", "signal_date": "2026-11-11"}).json()["record"]
    assert first["id"] != second["id"]
    assert first["xueqiu_url"] == "https://xueqiu.com/S/BJ920001"
    assert len(client.get("/api/review-records").json()["records"]) == 2
    assert client.delete(f'/api/review-records/{first["id"]}').status_code == 200
    assert client.get("/api/review-records").json()["records"] == [second]
    assert client.delete(f'/api/review-records/{first["id"]}').status_code == 404
    assert client.put("/api/review-records/missing/note", json={"note": "x"}).status_code == 404


def test_validation_and_corrupt_storage_do_not_destroy_history(client):
    assert client.post("/api/review-records", json={"symbol": "bad"}).status_code == 400
    assert client.post("/api/review-records", json={"symbol": "600000", "signal_date": "2026-02-30"}).status_code == 422
    store.REVIEW_RECORDS_PATH.write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError):
        store.add_review_record({"symbol": "600000"})
    assert store.REVIEW_RECORDS_PATH.read_text() == "broken"


def test_concurrent_adds_and_note_updates_preserve_all_records(client):
    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(lambda i: store.add_review_record({"symbol": "600000", "note": str(i)})["record"], range(16)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda row: store.update_review_note(row["id"], "已复盘"), entries))
    records = store.get_review_records()["records"]
    assert len(records) == 16
    assert {row["id"] for row in records} == {row["id"] for row in entries}
    assert all(row["note"] == "已复盘" for row in records)
