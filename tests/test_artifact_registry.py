from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import threading
from pathlib import Path

import pytest

from quant.infrastructure.artifact_registry import ArtifactRegistry, publication_read_lease
from quant.infrastructure.publication import PublicationStore


def _artifact(root: Path, name: str, **kwargs) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"artifact")
    ArtifactRegistry(root).register(
        path, producer="test", input_versions={"source": "revision-1"},
        retention_class=kwargs.pop("retention_class", "rebuildable"),
        state=kwargs.pop("state", "retired"), **kwargs,
    )
    return path


def _process_lease(root: str, path: str, connection, crash: bool) -> None:
    with ArtifactRegistry(Path(root)).lease([path], owner="worker", kind="build"):
        connection.send("ready")
        connection.recv()
        if crash:
            os._exit(0)


def test_unknown_ownership_and_unregistered_read_only_inventory(tmp_path: Path) -> None:
    path = tmp_path / "data/research/input.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"evidence")
    registry = ArtifactRegistry(tmp_path)

    assert registry.delete_if_eligible(path)["reason"] == "unknown_ownership"
    before = sorted(tmp_path.rglob("*"))
    report = registry.inventory(["data", "data/research"], budgets={"data": 0})

    assert report["dry_run"] is True
    assert report["totals"]["logical_bytes"] == 8
    assert report["budgets"]["data"]["over_budget_bytes"] == path.stat().st_blocks * 512
    assert all(not item["deletion_candidate"] for item in report["entries"])
    assert sorted(tmp_path.rglob("*")) == before
    assert not registry.directory.exists()


def test_protected_report_reaches_model_and_raw_inputs_transitively(tmp_path: Path) -> None:
    source = _artifact(tmp_path, "data/raw/history", retention_class="raw")
    model = _artifact(tmp_path, "data/research/model", dependencies=[source])
    report = _artifact(tmp_path, "reports/active", dependencies=[model])
    registry = ArtifactRegistry(tmp_path)
    registry.set_references("production-report", [report])

    for path in (report, model, source):
        assert registry.delete_if_eligible(path)["deleted"] is False
    registry.set_references("production-report", [])
    assert registry.delete_if_eligible(report)["deleted"] is True
    assert registry.delete_if_eligible(model)["deleted"] is True
    assert registry.delete_if_eligible(source)["deleted"] is False


def test_commits_keep_current_previous_and_do_not_promote_building(tmp_path: Path) -> None:
    first = _artifact(tmp_path, "vectors/first")
    second = _artifact(tmp_path, "vectors/second")
    third = _artifact(tmp_path, "vectors/third")
    interrupted = _artifact(tmp_path, "vectors/interrupted", state="building")
    registry = ArtifactRegistry(tmp_path)
    for path in (first, second, third):
        registry.commit("vectors", path)
        registry.retire(path)
    registry.commit("vectors", third)
    assert registry.referenced_paths("vectors:previous") == (second,)
    assert registry.delete_if_eligible(first)["deleted"] is True
    for path in (second, third, interrupted):
        assert registry.delete_if_eligible(path)["deleted"] is False


@pytest.mark.parametrize("kind", ["read", "build"])
def test_lease_protects_parent_child_and_dependency_without_ttl(tmp_path: Path, kind: str) -> None:
    source = _artifact(tmp_path, "source/data")
    path = _artifact(tmp_path, "vectors/config/file", dependencies=[source], last_access=1)
    registry = ArtifactRegistry(tmp_path)
    with registry.lease([path], owner="worker", kind=kind):
        os.utime(path, (1, 1))
        assert registry.delete_if_eligible(path)["deleted"] is False
        assert registry.delete_if_eligible(path.parent)["deleted"] is False
        assert registry.delete_if_eligible(source)["deleted"] is False
    assert registry.delete_if_eligible(path)["deleted"] is True


@pytest.mark.parametrize("crash", [False, True])
def test_cross_process_leases_and_crash_release(tmp_path: Path, crash: bool) -> None:
    path = _artifact(tmp_path, "vectors/config", last_access=1)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_process_lease, args=(str(tmp_path), str(path), child, crash))
    process.start()
    try:
        assert parent.poll(15)
        assert parent.recv() == "ready"
        assert ArtifactRegistry(tmp_path).delete_if_eligible(path)["deleted"] is False
        parent.send("release")
        process.join(15)
        assert process.exitcode == 0
        assert ArtifactRegistry(tmp_path).delete_if_eligible(path)["deleted"] is True
        state = json.loads(ArtifactRegistry(tmp_path).state_path.read_text())
        assert state["leases"] == {}
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        child.close()


def test_acquisition_and_deletion_are_serialized(tmp_path: Path) -> None:
    path = _artifact(tmp_path, "vectors/file")
    registry = ArtifactRegistry(tmp_path)
    attempted = threading.Event()
    entered = threading.Event()
    errors: list[Exception] = []

    def reader() -> None:
        attempted.set()
        try:
            with registry.lease([path], owner="reader"):
                entered.set()
        except Exception as exc:
            errors.append(exc)

    with registry.deletion_guard(path) as reason:
        assert reason is None
        worker = threading.Thread(target=reader)
        worker.start()
        assert attempted.wait(5)
        assert not entered.wait(0.05)
        path.unlink()
    worker.join(5)
    assert not worker.is_alive()
    assert not entered.is_set()
    assert len(errors) == 1 and isinstance(errors[0], FileNotFoundError)


def test_build_lease_can_precede_directory_and_register_inside_lease(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    path = tmp_path / "new-build"
    with registry.lease([path], owner="builder", kind="build"):
        _artifact(tmp_path, "new-build")
        registry.commit("current", path)
        assert registry.delete_if_eligible(path)["deleted"] is False
    assert path.exists()


@pytest.mark.parametrize("payload", ["{", "[]", '{"schema_version": 9}',
                                      '{"schema_version":1,"artifacts":{},"references":[]}'])
def test_bad_registry_fails_closed(tmp_path: Path, payload: str) -> None:
    path = _artifact(tmp_path, "retired")
    registry = ArtifactRegistry(tmp_path)
    registry.state_path.write_text(payload, encoding="utf-8")
    assert registry.delete_if_eligible(path)["reason"].startswith("registry_unavailable")
    assert registry.inventory([path])["errors"]
    assert path.exists()


def test_missing_lease_file_is_conservative_keep(tmp_path: Path) -> None:
    path = _artifact(tmp_path, "retired")
    registry = ArtifactRegistry(tmp_path)
    state = json.loads(registry.state_path.read_text())
    state["leases"]["a" * 32] = {"paths": ["retired"], "kind": "build", "owner": "lost"}
    registry.state_path.write_text(json.dumps(state))
    assert registry.delete_if_eligible(path)["reason"] == "live_lease"


def test_path_escape_symlinks_and_ownership_reassignment_are_rejected(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    path = _artifact(tmp_path, "owned")
    (tmp_path / "alias").symlink_to(path)
    for unsafe in (tmp_path, tmp_path / "../outside", tmp_path / "alias"):
        with pytest.raises(ValueError):
            registry.register(unsafe, producer="test", input_versions={})
    with pytest.raises(ValueError, match="another producer"):
        registry.register(path, producer="someone-else", input_versions={})
    assert registry.delete_if_eligible(tmp_path / "alias")["deleted"] is False
    assert path.exists()


def test_inventory_deduplicates_hardlinks_and_overlapping_scopes(tmp_path: Path) -> None:
    first = _artifact(tmp_path, "files/first")
    second = tmp_path / "files/second"
    os.link(first, second)
    registry = ArtifactRegistry(tmp_path)
    before = registry.state_path.read_bytes()
    report = registry.inventory(["files", first, second], budgets={"files": 0})
    assert report["totals"] == {
        "files": 2, "logical_bytes": 16, "allocated_bytes": first.stat().st_blocks * 512,
    }
    assert registry.state_path.read_bytes() == before
    assert first.exists() and second.exists()


def test_missing_reference_report_and_explicit_last_access_not_mtime(tmp_path: Path) -> None:
    path = _artifact(tmp_path, "retired", last_access=100)
    registry = ArtifactRegistry(tmp_path)
    registry.set_references("missing", ["not-built"])
    os.utime(path, (1, 1))
    with registry.deletion_guard(path, older_than=50) as reason:
        assert reason == "not_expired"
    assert registry.inventory([path])["missing_references"] == ["not-built"]


def _publication(root: Path) -> tuple[PublicationStore, Path]:
    generation = root / "data/publications/generation"
    (generation / "tree").mkdir(parents=True)
    (generation / "state.json").write_text('{"status":"validated"}')
    (generation.parent / "current.json").write_text('{"generation":"generation"}')
    ArtifactRegistry(root).register(
        generation, producer="publisher", input_versions={}, retention_class="rebuildable",
        state="retired", last_access=1,
    )
    return PublicationStore(root, managed=()), generation


def test_publication_read_lease_does_not_wait_for_writer(tmp_path: Path) -> None:
    store, generation = _publication(tmp_path)
    with (store.directory / "writer.lock").open("wb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with publication_read_lease(tmp_path, store.view) as view:
            assert view.generation == "generation"
            assert ArtifactRegistry(tmp_path).delete_if_eligible(generation)["deleted"] is False
    assert ArtifactRegistry(tmp_path).delete_if_eligible(generation)["deleted"] is True


def test_publication_view_selection_is_atomic_with_lease_acquisition(tmp_path: Path) -> None:
    store, generation = _publication(tmp_path)
    attempted = threading.Event()
    finished = threading.Event()
    results = []

    def deleter() -> None:
        attempted.set()
        results.append(ArtifactRegistry(tmp_path).delete_if_eligible(generation))
        finished.set()

    worker = threading.Thread(target=deleter)

    def select():
        view = store.view()
        worker.start()
        assert attempted.wait(5)
        assert not finished.wait(0.05)
        return view

    with publication_read_lease(tmp_path, select) as view:
        assert finished.wait(5)
        assert view.generation == "generation"
        assert results[0]["deleted"] is False
    worker.join(5)
    assert not worker.is_alive()


def test_legacy_publication_read_lease_does_not_create_registry(tmp_path: Path) -> None:
    store = PublicationStore(tmp_path, managed=())
    with publication_read_lease(tmp_path, store.view) as view:
        assert view.generation is None
    assert not ArtifactRegistry(tmp_path).directory.exists()


def test_completed_long_lease_updates_last_access_at_release(tmp_path: Path, monkeypatch) -> None:
    path = _artifact(tmp_path, "retired", last_access=1)
    registry = ArtifactRegistry(tmp_path)
    monkeypatch.setattr("quant.infrastructure.artifact_registry.time.time", lambda: 100)
    with registry.lease([path], owner="long-build", kind="build"):
        monkeypatch.setattr("quant.infrastructure.artifact_registry.time.time", lambda: 10000)
    with registry.deletion_guard(path, older_than=500) as reason:
        assert reason == "not_expired"


def _compiled_generations(root: Path) -> tuple[ArtifactRegistry, Path, list[Path]]:
    registry = ArtifactRegistry(root)
    config = root / "vectors/config"
    config.mkdir(parents=True)
    registry.register(
        config, producer="vectors", input_versions={}, retention_class="rebuildable", state="committed",
    )
    registry.set_references("active-vector-config", [config])
    generations = []
    for index in range(5):
        generation = config / "_matrix_cache_v1" / f"generation-{index}"
        generation.mkdir(parents=True)
        (generation / "vectors.npy").write_bytes(bytes([index]) * 32)
        registry.register(
            generation, producer="vectors", input_versions={"revision": str(index)},
            retention_class="rebuildable", state="committed", ownership_boundary=True,
            dependencies=[config],
        )
        registry.commit("compiled", generation)
        for previous in registry.referenced_paths("compiled:previous"):
            registry.retire(previous)
        generations.append(generation)
    return registry, config, generations


def test_independent_compiled_generations_bound_growth_under_active_config(tmp_path: Path) -> None:
    registry, config, generations = _compiled_generations(tmp_path)
    cache_root = config / "_matrix_cache_v1"
    unknown = cache_root / "experimental-unknown"
    unknown.mkdir()
    (unknown / "vectors.npy").write_bytes(b"unknown")
    before = registry.state_path.read_bytes()
    preview = registry.collect_retired_children(cache_root)
    assert {Path(key).name for key in preview["candidates"]} == {
        "generation-0", "generation-1", "generation-2",
    }
    assert all(path.exists() for path in generations)
    assert registry.state_path.read_bytes() == before
    applied = registry.collect_retired_children(cache_root, dry_run=False)
    assert len(applied["deleted_paths"]) == 3
    assert applied["reclaimed_bytes"] == 96
    assert all(path.exists() for path in generations[-2:])
    assert config.exists() and unknown.exists()
    assert registry.delete_if_eligible(config)["deleted"] is False


@pytest.mark.parametrize("kind", ["read", "build"])
def test_config_wide_lease_overrides_all_independent_boundaries(tmp_path: Path, kind: str) -> None:
    registry, config, generations = _compiled_generations(tmp_path)
    with registry.lease([config], owner="reader-or-builder", kind=kind):
        result = registry.collect_retired_children(config / "_matrix_cache_v1", dry_run=False)
        assert result["deleted_paths"] == []
        assert all(path.exists() for path in generations)
    result = registry.collect_retired_children(config / "_matrix_cache_v1", dry_run=False)
    assert len(result["deleted_paths"]) == 3


def test_generation_specific_lease_and_report_dependency_survive_collection(tmp_path: Path) -> None:
    registry, config, generations = _compiled_generations(tmp_path)
    report = _artifact(tmp_path, "reports/published", dependencies=[generations[1]])
    registry.set_references("report", [report])
    with registry.lease([generations[0]], owner="generation-reader"):
        result = registry.collect_retired_children(config / "_matrix_cache_v1", dry_run=False)
        assert result["deleted_paths"] == [str(generations[2].relative_to(tmp_path))]
        assert all(path.exists() for path in (generations[0], generations[1], *generations[-2:]))


@pytest.mark.parametrize("restriction", ["protected", "raw", "evidence", "building", "unknown", "different_owner"])
def test_boundaries_cannot_override_hard_or_unknown_ancestor_protection(tmp_path: Path, restriction: str) -> None:
    registry, config, generations = _compiled_generations(tmp_path)
    if restriction == "unknown":
        registry.set_references("unknown-owner", ["vectors"])
    elif restriction == "different_owner":
        registry.register(
            "vectors", producer="someone-else", input_versions={}, retention_class="rebuildable",
            state="committed",
        )
    else:
        registry.register(
            config, producer="vectors", input_versions={},
            retention_class=restriction if restriction in {"raw", "evidence"} else "rebuildable",
            state="building" if restriction == "building" else "committed",
            protected=restriction == "protected",
        )
    result = registry.collect_retired_children(config / "_matrix_cache_v1", dry_run=False)
    assert result["deleted_paths"] == []
    assert all(path.exists() for path in generations)


def test_unmarked_and_unretired_generations_are_conservatively_kept(tmp_path: Path) -> None:
    registry, config, generations = _compiled_generations(tmp_path)
    registry.register(
        generations[0], producer="vectors", input_versions={}, retention_class="rebuildable",
        state="retired", ownership_boundary=False,
    )
    registry.register(
        generations[1], producer="vectors", input_versions={}, retention_class="rebuildable",
        state="building", ownership_boundary=True,
    )
    result = registry.collect_retired_children(config / "_matrix_cache_v1", dry_run=False)
    assert result["deleted_paths"] == [str(generations[2].relative_to(tmp_path))]
    assert generations[0].exists() and generations[1].exists()


def test_explicit_config_dependency_can_pin_independent_generation(tmp_path: Path) -> None:
    registry, config, generations = _compiled_generations(tmp_path)
    registry.register(
        config, producer="vectors", input_versions={}, retention_class="rebuildable",
        state="committed", dependencies=[generations[0]],
    )
    result = registry.collect_retired_children(config / "_matrix_cache_v1", dry_run=False)
    assert generations[0].exists()
    assert len(result["deleted_paths"]) == 2


def test_parent_deletion_cannot_remove_a_pinned_independent_descendant(tmp_path: Path) -> None:
    registry, config, generations = _compiled_generations(tmp_path)
    registry.retire(config)
    registry.set_references("active-vector-config", [])
    assert registry.delete_if_eligible(config)["deleted"] is False
    assert all(path.exists() for path in generations)
