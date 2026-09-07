from __future__ import annotations

from quant.application.daily_dependencies import DependencyRegistry, Layer
from quant.infrastructure.publication import (
    PublicationStore, PublicationView, current_publication,
    publication_context, publication_path,
)
from quant.routine.checkpoint_store import CheckpointStore
from quant.routine.dag_executor import DailyDagExecutor
from quant.routine.operation_adapters import result_from_payload
from quant.routine.operation_contracts import CacheMode, CachePolicy, NodeChanges
from quant.routine.operation_registry import OperationRegistry
from tests.test_daily_dag_executor import _node, _operation


def test_dag_checkpoint_and_adapter_hash_staged_output_but_canonical_contract(tmp_path):
    live = tmp_path / "web/output.txt"
    live.parent.mkdir()
    live.write_text("old-live")
    contract = tmp_path / "config.json"
    contract.write_text('{"version": 1}')
    node = _node("product.test", "test", Layer.PRODUCT)
    operation = _operation("test", (node.node_id,), cache=CachePolicy(
        CacheMode.EXACT_DATE, "code-model-schema-v1", ("web/output.txt",),
        contract_paths=("config.json",),
    ))
    store = CheckpointStore(tmp_path, tmp_path / "checkpoints")
    publication = PublicationStore(tmp_path, ("web", "config.json"))
    calls = []

    def build(context):
        calls.append(current_publication().generation)
        publication_path(live).write_text("new-staged")
        return result_from_payload({
            "status": "success", "node_changes": {node.node_id: NodeChanges(("20260904",), ("A",))},
        }, (node.node_id,), context)

    executor = DailyDagExecutor(
        DependencyRegistry((node,), {"demo": (node.node_id,)}), OperationRegistry((operation,)),
        project_root=tmp_path, checkpoint_store=store, handlers={"test": build},
    )
    with publication.begin("test-stage"):
        first = executor.execute(target_trade_date="2026-09-04", scope="demo")
        assert first["status"] == "success"
        assert live.read_text() == "old-live"
        assert publication_path(live).read_text() == "new-staged"
        # Staged configuration must not affect canonical code/model identity.
        publication_path(contract).write_text('{"version": "staged-ignore"}')
        second = executor.execute(target_trade_date="2026-09-04", scope="demo")
        assert second["status"] == "success"
        assert calls == ["test-stage"]
        assert second["operations"]["test"].node_changes == {node.node_id: NodeChanges()}
        publication_path(live).write_text("tampered-stage")
        third = executor.execute(target_trade_date="2026-09-04", scope="demo")
        assert third["status"] == "success"
        assert calls == ["test-stage", "test-stage"]
        contract.write_text('{"version": 2}')
        assert executor.execute(target_trade_date="2026-09-04", scope="demo")["status"] == "success"
        assert len(calls) == 3
    assert live.read_text() == "old-live"


def test_directory_membership_and_malformed_checkpoint_fail_closed(tmp_path):
    directory = tmp_path / "outputs"
    directory.mkdir()
    (directory / "one").write_text("one")
    node = _node("product.test", "test", Layer.PRODUCT)
    operation = _operation("test", (node.node_id,), cache=CachePolicy(CacheMode.EXACT_DATE, "v1", ("outputs",)))
    calls = []

    def build(context):
        calls.append(1)
        return result_from_payload({"status": "success", "changed_keys": ()}, (node.node_id,), context)

    store = CheckpointStore(tmp_path, tmp_path / "checkpoints")
    executor = DailyDagExecutor(
        DependencyRegistry((node,), {"demo": (node.node_id,)}), OperationRegistry((operation,)),
        project_root=tmp_path, checkpoint_store=store, handlers={"test": build},
    )
    args = {"target_trade_date": "2026-09-04", "scope": "demo"}
    assert executor.execute(**args)["status"] == "success"
    assert executor.execute(**args)["status"] == "success"
    assert len(calls) == 1
    (directory / "two").write_text("two")
    assert executor.execute(**args)["status"] == "success"
    assert len(calls) == 2
    checkpoint = next((tmp_path / "checkpoints").glob("*.json"))
    checkpoint.write_text("[]")
    assert executor.execute(**args)["status"] == "success"
    assert len(calls) == 3


def test_checkpoint_does_not_replay_another_generations_staging_paths(tmp_path):
    node = _node("product.test", "test", Layer.PRODUCT)
    operation = _operation("test", (node.node_id,), cache=CachePolicy(CacheMode.EXACT_DATE, "v1", ("output",)))
    store = CheckpointStore(tmp_path, tmp_path / "checkpoints")
    calls = []

    def build(context):
        calls.append(current_publication().generation)
        return result_from_payload({
            "status": "success", "output": str(publication_path(tmp_path / "output")),
            "changed_keys": (),
        }, (node.node_id,), context)

    executor = DailyDagExecutor(
        DependencyRegistry((node,), {"demo": (node.node_id,)}), OperationRegistry((operation,)),
        project_root=tmp_path, checkpoint_store=store, handlers={"test": build},
    )
    for generation in ("one", "two"):
        view = PublicationView(tmp_path, tmp_path / "generations", generation, ("output",), True)
        output = view.resolve(tmp_path / "output")
        output.parent.mkdir(parents=True)
        output.write_text("identical-cloned-content")
        with publication_context(view):
            report = executor.execute(target_trade_date="2026-09-04", scope="demo")
            assert report["status"] == "success"
            assert report["node_results"][node.node_id]["output"] == str(output)
    assert calls == ["one", "two"]
