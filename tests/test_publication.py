import json

import pytest

from quant.infrastructure.publication import (
    ContextThreadPoolExecutor,
    PublicationError,
    PublicationStore,
    assert_publication_writable,
    publication_context,
    publication_path,
)


def audit(status="success"):
    return {"status": status, "freshness_audit": {"status": status, "failures": []}}


def test_failed_generation_never_changes_visible_outputs(tmp_path):
    output = tmp_path / "snapshots/one.json"
    output.parent.mkdir()
    output.write_text('{"date": "old"}')
    store = PublicationStore(tmp_path, ("snapshots",))
    with store.begin("failed") as stage:
        publication_path(output).write_text('{"date": "new"}')
        assert json.loads(store.view().resolve(output).read_text())["date"] == "old"
        with pytest.raises(PublicationError, match="strict freshness"):
            store.commit(stage, audit("failed"))
    assert json.loads(store.view().resolve(output).read_text())["date"] == "old"
    assert json.loads(output.read_text())["date"] == "old"


def test_publication_switches_all_outputs_and_pins_existing_reader(tmp_path):
    store = PublicationStore(tmp_path, ("snapshots",))
    with store.begin("first") as stage:
        a = publication_path(tmp_path / "snapshots/a.json")
        a.parent.mkdir(parents=True)
        a.write_text("old-a")
        (a.parent / "b.json").write_text("old-b")
        store.commit(stage, audit())
    pinned = store.view()
    with store.begin("second") as stage:
        publication_path(tmp_path / "snapshots/a.json").write_text("new-a")
        publication_path(tmp_path / "snapshots/b.json").write_text("new-b")
        store.commit(stage, audit())
    for name in ("a", "b"):
        path = tmp_path / f"snapshots/{name}.json"
        assert pinned.resolve(path).read_text() == f"old-{name}"
        assert store.view().resolve(path).read_text() == f"new-{name}"


def test_worker_context_is_propagated_but_readers_cannot_materialize(tmp_path):
    store = PublicationStore(tmp_path, ("snapshots",))
    with store.begin("run") as stage:
        with ContextThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(publication_path, tmp_path / "snapshots/a.json").result()
        assert result == stage.resolve(tmp_path / "snapshots/a.json")
    with publication_context(store.view()):
        with pytest.raises(PublicationError, match="committed result"):
            assert_publication_writable()


def test_corrupt_pointer_does_not_fall_back_to_live_legacy_data(tmp_path):
    store = PublicationStore(tmp_path, ("snapshots",))
    store.directory.mkdir(parents=True)
    (store.directory / "current.json").write_text('{"generation":"missing"}')
    with pytest.raises(PublicationError, match="unavailable"):
        store.view()


def test_concurrent_publisher_is_rejected(tmp_path):
    store = PublicationStore(tmp_path, ("snapshots",))
    with store.begin("one"):
        with pytest.raises(PublicationError, match="already running"):
            with store.begin("two"):
                pass


def test_committed_generation_rejects_late_writes(tmp_path):
    from quant.data.atomic_io import atomic_write_json

    store = PublicationStore(tmp_path, ("snapshots",))
    with store.begin("one") as stage:
        atomic_write_json({"date": "old"}, tmp_path / "snapshots/a.json")
        store.commit(stage, audit())
        with pytest.raises(PublicationError, match="immutable"):
            atomic_write_json({"date": "new"}, tmp_path / "snapshots/a.json")


def test_missing_writer_state_prevents_materialization(tmp_path):
    store = PublicationStore(tmp_path, ("snapshots",))
    with store.begin("one") as stage:
        (stage.directory / stage.generation / "state.json").unlink()
        with pytest.raises(PublicationError, match="state is unavailable"):
            assert_publication_writable()


def test_reusing_generation_id_does_not_mutate_committed_state(tmp_path):
    store = PublicationStore(tmp_path, ("snapshots",))
    with store.begin("one") as stage:
        store.commit(stage, audit())
    before = (store.directory / "one/state.json").read_bytes()
    with pytest.raises(PublicationError, match="already exists"):
        with store.begin("one"):
            pass
    assert store.view().generation == "one"
    assert (store.directory / "one/state.json").read_bytes() == before


def test_workspace_sql_failure_leaves_committed_payload_unchanged(tmp_path):
    from types import SimpleNamespace

    from quant.infrastructure.workspace_snapshots import WorkspaceSnapshotRepository

    class FailedStore:
        config = SimpleNamespace(sql_url="configured")

        def _engine(self):
            raise RuntimeError("simulated SQL outage")

    repository = WorkspaceSnapshotRepository(
        tmp_path / "snapshots", "workspace_snapshots",
        store_factory=FailedStore,
    )
    repository.write("one", "2026-09-04", {"signal_date": "2026-09-04", "value": "old"}, write_sql=False)
    store = PublicationStore(tmp_path, ("snapshots",))
    with store.begin("one"):
        with pytest.raises(RuntimeError, match="SQL persistence failed"):
            repository.write("one", "2026-09-04", {"signal_date": "2026-09-04", "value": "new"})
    with publication_context(store.view()):
        assert repository.read("one", "2026-09-04")["value"] == "old"
        assert repository.read("one", "2026-09-07") is None


def test_web_request_is_pinned_while_another_generation_commits(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from quant.webapp import services
    from quant.webapp.publication_middleware import PublicationMiddleware

    store = PublicationStore(tmp_path, ("snapshots",))
    path = tmp_path / "snapshots/a.json"
    with store.begin("old") as stage:
        target = publication_path(path)
        target.parent.mkdir(parents=True)
        target.write_text("old")
        store.commit(stage, audit())
    monkeypatch.setattr(services, "_publication_store", lambda: store)
    app = FastAPI()
    app.add_middleware(PublicationMiddleware)

    @app.get("/read")
    def read():
        first = publication_path(path).read_text()
        with store.begin("new") as stage:
            publication_path(path).write_text("new")
            store.commit(stage, audit())
        second = publication_path(path).read_text()
        return {"first": first, "second": second}

    response = TestClient(app).get("/read")
    assert response.json() == {"first": "old", "second": "old"}
    assert response.headers["x-quant-generation"] == "old"
    assert store.view().generation == "new"


def test_explicit_generation_cannot_select_staging_or_aborted_outputs(tmp_path):
    store = PublicationStore(tmp_path, ("snapshots",))
    with store.begin("unfinished"):
        with pytest.raises(PublicationError, match="unavailable"):
            store.view("unfinished")
    with pytest.raises(PublicationError, match="unavailable"):
        store.view("unfinished")


@pytest.mark.parametrize("managed", [("../outside",), ("/outside",), ("snapshots", "snapshots/file.json")])
def test_publication_rejects_unsafe_output_contracts(tmp_path, managed):
    with pytest.raises(PublicationError):
        PublicationStore(tmp_path, managed)


def test_corrupt_current_state_is_not_served(tmp_path):
    store = PublicationStore(tmp_path, ("snapshots",))
    with store.begin("run") as stage:
        store.commit(stage, audit())
    (store.directory / "run/state.json").write_text('{"status":"aborted"}')
    with pytest.raises(PublicationError, match="unavailable"):
        store.view()


def test_canonical_profile_outage_cannot_read_parquet_mirror(tmp_path, monkeypatch):
    from quant.data.market_data_store import MarketDataUnavailableError
    from quant.webapp import services

    def unavailable(*args, **kwargs):
        raise MarketDataUnavailableError("source unavailable")

    monkeypatch.setattr(services, "DAILY_DIR", tmp_path)
    monkeypatch.setattr(services.MarketDataStore, "read_frame", unavailable)
    monkeypatch.setattr(services.pd, "read_parquet", lambda *a, **kw: pytest.fail("mirror fallback"))
    services._daily_profile_at_or_before.cache_clear()
    with pytest.raises(MarketDataUnavailableError):
        services._daily_profile_at_or_before("000001.SZ", "2026-09-04")


def test_isolated_vector_worker_uses_parent_worker_limit(monkeypatch):
    import os
    from queue import Queue
    from quant.webapp import services

    monkeypatch.setenv("SIMILAR_PATTERN_CACHE_WORKERS", "20")
    captured = {}

    def refresh(**kwargs):
        captured["workers"] = os.environ["SIMILAR_PATTERN_CACHE_WORKERS"]
        captured["total"] = os.environ["ROUTINE_TOTAL_WORKERS"]
        return {"status": "success"}

    monkeypatch.setattr(services, "refresh_similar_pattern_analysis", refresh)
    results = Queue()
    services._similar_patterns_worker(results, worker_limit=2)
    assert captured == {"workers": "2", "total": "2"}
    assert results.get_nowait()["ok"] is True
    assert os.environ["SIMILAR_PATTERN_CACHE_WORKERS"] == "20"
