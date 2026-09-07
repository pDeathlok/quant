from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time

import pytest

from quant.application.daily_dependencies import (
    Cadence,
    DependencyEdge,
    DependencyNode,
    DependencyRegistry,
    FreshnessMode,
    FreshnessPolicy,
    IncrementalPolicy,
    Layer,
    Lifecycle,
)
from quant.routine.checkpoint_store import CheckpointStore
from quant.routine.dag_executor import DailyDagExecutor, ResourceBudget
from quant.routine.operation_contracts import (
    CacheMode,
    CachePolicy,
    ExecutionMode,
    InputSnapshot,
    NodeChanges,
    OperationContext,
    OperationDefinition,
    OperationResult,
    ResourceClaim,
)
from quant.routine.operation_registry import OperationRegistry


def _node(
    node_id: str,
    operation_id: str,
    layer: Layer,
    *,
    inputs: tuple[DependencyEdge, ...] = (),
) -> DependencyNode:
    return DependencyNode(
        node_id=node_id,
        layer=layer,
        owner="tests",
        lifecycle=Lifecycle.PRODUCTION,
        cadence=Cadence.TRADE_DAILY,
        inputs=inputs,
        freshness=FreshnessPolicy(FreshnessMode.EXACT_TRADE_DATE),
        incremental=IncrementalPolicy(partition_key="trade_date"),
        operation_id=operation_id,
    )


def _operation(
    operation_id: str,
    produces: tuple[str, ...],
    *,
    cpu_slots: int = 1,
    cache: CachePolicy | None = None,
    input_ids: tuple[str, ...] | None = (),
) -> OperationDefinition:
    return OperationDefinition(
        operation_id=operation_id,
        entrypoint="tests.test_daily_dag_executor:_unused",
        produces=produces,
        execution_mode=ExecutionMode.THREAD,
        resources=ResourceClaim(
            cpu_slots=cpu_slots,
            requested_workers=cpu_slots,
            max_workers=cpu_slots,
        ),
        cache=cache or CachePolicy(CacheMode.NONE, "1"),
        input_ids=input_ids,
    )


def _success(node_id: str, **kwargs) -> OperationResult:
    return OperationResult(
        status="success",
        node_results={node_id: {"status": "success", **kwargs}},
    )


def test_independent_operations_run_in_parallel(tmp_path: Path) -> None:
    left = _node("product.left", "build_left", Layer.PRODUCT)
    right = _node("product.right", "build_right", Layer.PRODUCT)
    dependencies = DependencyRegistry(
        (left, right),
        {"demo": (left.node_id, right.node_id)},
    )
    operations = OperationRegistry(
        (
            _operation("build_left", (left.node_id,)),
            _operation("build_right", (right.node_id,)),
        )
    )
    barrier = threading.Barrier(2, timeout=2)
    thread_ids: set[int] = set()

    def handler(node_id: str):
        def run(_context: OperationContext) -> OperationResult:
            thread_ids.add(threading.get_ident())
            barrier.wait()
            return _success(node_id)

        return run

    result = DailyDagExecutor(
        dependencies,
        operations,
        project_root=tmp_path,
        budget=ResourceBudget(cpu_slots=2),
        handlers={
            "build_left": handler(left.node_id),
            "build_right": handler(right.node_id),
        },
    ).execute(target_trade_date="2026-08-31", scope="demo")

    assert result["status"] == "success"
    assert len(thread_ids) == 2
    assert result["resource_usage"]["max_cpu_slots"] == 2


def test_resource_budget_prevents_oversubscription(tmp_path: Path) -> None:
    first = _node("product.first", "first", Layer.PRODUCT)
    second = _node("product.second", "second", Layer.PRODUCT)
    dependencies = DependencyRegistry(
        (first, second),
        {"demo": (first.node_id, second.node_id)},
    )
    operations = OperationRegistry(
        (
            _operation("first", (first.node_id,), cpu_slots=2),
            _operation("second", (second.node_id,), cpu_slots=2),
        )
    )
    active = 0
    max_active = 0
    lock = threading.Lock()

    def handler(node_id: str):
        def run(_context: OperationContext) -> OperationResult:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return _success(node_id)

        return run

    result = DailyDagExecutor(
        dependencies,
        operations,
        project_root=tmp_path,
        budget=ResourceBudget(cpu_slots=2),
        handlers={
            "first": handler(first.node_id),
            "second": handler(second.node_id),
        },
    ).execute(target_trade_date="2026-08-31", scope="demo")

    assert result["status"] == "success"
    assert max_active == 1
    assert result["resource_usage"]["max_cpu_slots"] == 2


def test_one_operation_materializes_multiple_nodes_once(tmp_path: Path) -> None:
    core = _node("product.core", "build_selector", Layer.PRODUCT)
    extended = _node("product.extended", "build_selector", Layer.PRODUCT)
    dependencies = DependencyRegistry(
        (core, extended),
        {"demo": (core.node_id, extended.node_id)},
    )
    operations = OperationRegistry(
        (_operation("build_selector", (core.node_id, extended.node_id)),)
    )
    calls = 0

    def build(_context: OperationContext) -> OperationResult:
        nonlocal calls
        calls += 1
        return OperationResult(
            status="success",
            node_results={
                core.node_id: {"status": "success", "stocks": 10},
                extended.node_id: {"status": "success", "stocks": 20},
            },
        )

    result = DailyDagExecutor(
        dependencies,
        operations,
        project_root=tmp_path,
        handlers={"build_selector": build},
    ).execute(target_trade_date="2026-08-31", scope="demo")

    assert result["status"] == "success"
    assert calls == 1
    assert set(result["node_results"]) == {core.node_id, extended.node_id}


def test_changes_propagate_to_downstream_context(tmp_path: Path) -> None:
    source = _node("data.market", "refresh_market", Layer.DATA_SOURCE)
    feature = _node(
        "product.signal",
        "build_signal",
        Layer.PRODUCT,
        inputs=(DependencyEdge(source.node_id),),
    )
    dependencies = DependencyRegistry(
        (source, feature),
        {"demo": (feature.node_id,)},
    )
    operations = OperationRegistry(
        (
            _operation("refresh_market", (source.node_id,)),
            _operation("build_signal", (feature.node_id,)),
        )
    )
    observed: dict[str, tuple[str, ...]] = {}

    def refresh(_context: OperationContext) -> OperationResult:
        return OperationResult(
            status="success",
            node_results={source.node_id: {"status": "success"}},
            changed_partitions=("20260830",),
            changed_keys=("000001.SZ",),
            dataset_revisions={source.node_id: 4},
        )

    def build(context: OperationContext) -> OperationResult:
        observed["partitions"] = context.dirty_partitions
        observed["keys"] = context.dirty_keys
        assert context.upstream_revisions == {source.node_id: 4}
        return _success(feature.node_id)

    result = DailyDagExecutor(
        dependencies,
        operations,
        project_root=tmp_path,
        handlers={"refresh_market": refresh, "build_signal": build},
    ).execute(target_trade_date="2026-08-31", scope="demo")

    assert result["status"] == "success"
    assert observed == {
        "partitions": ("20260830",),
        "keys": ("000001.SZ",),
    }


def test_initial_changeset_is_injected_into_selected_root_operation(
    tmp_path: Path,
) -> None:
    feature = _node("feature.project", "build_project", Layer.PRODUCT)
    dependencies = DependencyRegistry((feature,), {"demo": (feature.node_id,)})
    operations = OperationRegistry(
        (_operation("build_project", (feature.node_id,)),)
    )
    observed: dict[str, tuple[str, ...]] = {}

    def build(context: OperationContext) -> OperationResult:
        observed["partitions"] = context.dirty_partitions
        observed["keys"] = context.dirty_keys
        return _success(feature.node_id)

    result = DailyDagExecutor(
        dependencies,
        operations,
        project_root=tmp_path,
        handlers={"build_project": build},
    ).execute(
        target_trade_date="2026-08-31",
        scope="demo",
        dirty_partitions=("20260829", "20260830"),
        dirty_keys=("000001.SZ",),
    )

    assert result["status"] == "success"
    assert observed == {
        "partitions": ("20260829", "20260830"),
        "keys": ("000001.SZ",),
    }


def test_valid_checkpoint_skips_handler(tmp_path: Path) -> None:
    feature = _node("product.cached", "build_cached", Layer.PRODUCT)
    dependencies = DependencyRegistry((feature,), {"demo": (feature.node_id,)})
    operation = _operation(
        "build_cached",
        (feature.node_id,),
        cache=CachePolicy(
            CacheMode.EXACT_DATE,
            "1",
            output_paths=("data/cached.txt",),
        ),
    )
    operations = OperationRegistry((operation,))
    calls = 0

    def build(_context: OperationContext) -> OperationResult:
        nonlocal calls
        calls += 1
        output = tmp_path / "data/cached.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("current", encoding="utf-8")
        return OperationResult(
            status="success", node_results={feature.node_id: {"status": "success"}},
            output_fingerprints={feature.node_id: "content-current"},
            node_changes={feature.node_id: NodeChanges(partitions=("20260831",))},
        )

    store = CheckpointStore(tmp_path, tmp_path / "checkpoints")
    executor = DailyDagExecutor(
        dependencies,
        operations,
        project_root=tmp_path,
        checkpoint_store=store,
        handlers={"build_cached": build},
    )
    first = executor.execute(target_trade_date="2026-08-31", scope="demo")
    second = executor.execute(target_trade_date="2026-08-31", scope="demo")

    assert first["status"] == second["status"] == "success"
    assert calls == 1
    assert second["operations"]["build_cached"].metrics["checkpoint_reused"] is True
    assert second["operations"]["build_cached"].changed_partitions == ()
    assert second["operations"]["build_cached"].node_changes == {feature.node_id: NodeChanges()}


def test_failed_operation_cancels_only_its_descendants(tmp_path: Path) -> None:
    source = _node("data.source", "source", Layer.DATA_SOURCE)
    child = _node(
        "product.child",
        "child",
        Layer.PRODUCT,
        inputs=(DependencyEdge(source.node_id),),
    )
    independent = _node("product.independent", "independent", Layer.PRODUCT)
    dependencies = DependencyRegistry(
        (source, child, independent),
        {"demo": (child.node_id, independent.node_id)},
    )
    operations = OperationRegistry(
        (
            _operation("source", (source.node_id,)),
            _operation("child", (child.node_id,)),
            _operation("independent", (independent.node_id,)),
        )
    )

    result = DailyDagExecutor(
        dependencies,
        operations,
        project_root=tmp_path,
        handlers={
            "source": lambda _context: OperationResult(
                status="failed",
                node_results={},
                error="source unavailable",
            ),
            "child": lambda _context: _success(child.node_id),
            "independent": lambda _context: _success(independent.node_id),
        },
    ).execute(target_trade_date="2026-08-31", scope="demo")

    assert result["status"] == "failed"
    assert result["failed_operations"] == ["source"]
    assert result["cancelled_operations"] == ["child"]
    assert independent.node_id in result["node_results"]


def test_corrected_input_invalidates_only_declared_dependents(tmp_path: Path) -> None:
    a = _node("data.a", "source", Layer.DATA_SOURCE)
    b = _node("data.b", "source", Layer.DATA_SOURCE)
    left = _node("product.left", "left", Layer.PRODUCT, inputs=(DependencyEdge(a.node_id),))
    right = _node("product.right", "right", Layer.PRODUCT, inputs=(DependencyEdge(b.node_id),))
    leaf = _node("product.leaf", "leaf", Layer.PRODUCT, inputs=(DependencyEdge(left.node_id),))
    dependencies = DependencyRegistry((a, b, left, right, leaf), {"demo": (leaf.node_id, right.node_id)})
    definitions = [_operation("source", (a.node_id, b.node_id))]
    counts = {node.operation_id: 0 for node in (left, right, leaf)}
    observed = {}
    handlers = {}
    for node in (left, right, leaf):
        definitions.append(_operation(
            node.operation_id, (node.node_id,),
            cache=CachePolicy(CacheMode.EXACT_DATE, "code-model-schema-v1", (f"{node.operation_id}.txt",)),
        ))

        def build(context, node=node):
            counts[node.operation_id] += 1
            observed[node.operation_id] = context
            fingerprint = ":".join(context.upstream_fingerprints.values())
            (tmp_path / f"{node.operation_id}.txt").write_text(fingerprint)
            return OperationResult(
                status="success", node_results={node.node_id: {"status": "success"}},
                output_fingerprints={node.node_id: fingerprint},
                node_changes={node.node_id: NodeChanges(context.dirty_partitions, context.dirty_keys)},
            )

        handlers[node.operation_id] = build
    executor = DailyDagExecutor(
        dependencies, OperationRegistry(definitions), project_root=tmp_path,
        checkpoint_store=CheckpointStore(tmp_path, tmp_path / "checkpoints"), handlers=handlers,
    )
    arguments = dict(target_trade_date="2026-09-04", scope="demo", node_ids=(left.node_id, right.node_id, leaf.node_id))
    snapshots = {a.node_id: InputSnapshot("canonical:a:1"), b.node_id: InputSnapshot("canonical:b:1")}
    assert executor.execute(**arguments, input_snapshots=snapshots)["status"] == "success"
    assert executor.execute(**arguments, input_snapshots=snapshots)["status"] == "success"
    assert counts == {"left": 1, "right": 1, "leaf": 1}

    snapshots[a.node_id] = InputSnapshot("canonical:a:2", changes=NodeChanges(("20260830",), ("000001.SZ",)))
    corrected = executor.execute(**arguments, input_snapshots=snapshots)
    assert corrected["status"] == "success"
    assert counts == {"left": 2, "right": 1, "leaf": 2}
    assert observed["left"].upstream_fingerprints == {a.node_id: "canonical:a:2"}
    assert observed["left"].full_rebuild is False
    assert observed["leaf"].dirty_partitions == ("20260830",)
    assert observed["leaf"].dirty_keys == ("000001.SZ",)
    assert corrected["operations"]["left"].input_fingerprints == {a.node_id: "canonical:a:2"}
    # Replaying the same correction journal is not another data revision.
    assert executor.execute(**arguments, input_snapshots=snapshots)["status"] == "success"
    assert counts == {"left": 2, "right": 1, "leaf": 2}
    snapshots[a.node_id] = InputSnapshot("canonical:a:3")
    assert executor.execute(**arguments, input_snapshots=snapshots)["status"] == "success"
    assert counts == {"left": 3, "right": 1, "leaf": 3}
    assert observed["left"].full_rebuild is True


def test_multi_output_producer_filters_lineage_and_changes_per_input_node(tmp_path: Path) -> None:
    a = _node("data.a", "source", Layer.DATA_SOURCE)
    b = _node("data.b", "source", Layer.DATA_SOURCE)
    left = _node("product.left", "left", Layer.PRODUCT, inputs=(DependencyEdge(a.node_id),))
    right = _node("product.right", "right", Layer.PRODUCT, inputs=(DependencyEdge(b.node_id),))
    deps = DependencyRegistry((a, b, left, right), {"demo": (left.node_id, right.node_id)})
    registry = OperationRegistry((
        _operation("source", (a.node_id, b.node_id)),
        _operation("left", (left.node_id,)), _operation("right", (right.node_id,)),
    ))
    seen = {}

    def consume(context, node):
        seen[node] = context
        return _success(node)

    result = DailyDagExecutor(deps, registry, project_root=tmp_path, handlers={
        "source": lambda context: OperationResult(
            status="success", node_results={a.node_id: {}, b.node_id: {}},
            output_fingerprints={a.node_id: "a2", b.node_id: "b1"},
            dataset_revisions={a.node_id: 2, b.node_id: 1},
            changed_partitions=("old",), changed_keys=("A",),
            node_changes={a.node_id: NodeChanges(("old",), ("A",)), b.node_id: NodeChanges()},
        ),
        "left": lambda context: consume(context, left.node_id),
        "right": lambda context: consume(context, right.node_id),
    }).execute(target_trade_date="2026-09-04", scope="demo", dirty_keys=iter(("root-only",)))
    assert result["status"] == "success"
    assert seen[left.node_id].dirty_keys == ("A",)
    assert seen[right.node_id].dirty_keys == ()
    assert seen[right.node_id].dirty_partitions == ()
    assert seen[right.node_id].upstream_fingerprints == {b.node_id: "b1"}
    assert seen[right.node_id].upstream_revisions == {b.node_id: 1}
    assert set(seen[right.node_id].upstream_results) == {b.node_id}


@pytest.mark.parametrize("input_ids, snapshots, match", [
    (None, {}, "incomplete input/contract identity"),
    (("source:canonical",), {}, "missing external input identities"),
])
def test_incomplete_identity_rejected_before_execution(tmp_path, input_ids, snapshots, match):
    node = _node("product.test", "test", Layer.PRODUCT)
    executor = DailyDagExecutor(
        DependencyRegistry((node,), {"demo": (node.node_id,)}),
        OperationRegistry((_operation("test", (node.node_id,), input_ids=input_ids),)),
        project_root=tmp_path, checkpoint_store=CheckpointStore(tmp_path, tmp_path / "checkpoints"),
        handlers={"test": lambda context: pytest.fail("must not execute unaudited work")},
    )
    with pytest.raises(ValueError, match=match):
        executor.execute(target_trade_date="2026-09-04", scope="demo", input_snapshots=snapshots)
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("missing", ["fingerprint", "changes"])
def test_incomplete_output_identity_never_saves_checkpoint(tmp_path, missing):
    node = _node("product.test", "test", Layer.PRODUCT)
    (tmp_path / "output").write_text("exists-but-not-verified")
    operation = _operation("test", (node.node_id,), cache=CachePolicy(CacheMode.EXACT_DATE, "1", ("output",)))
    result = DailyDagExecutor(
        DependencyRegistry((node,), {"demo": (node.node_id,)}), OperationRegistry((operation,)),
        project_root=tmp_path, checkpoint_store=CheckpointStore(tmp_path, tmp_path / "checkpoints"),
        handlers={"test": lambda context: OperationResult(
            status="success", node_results={node.node_id: {}},
            output_fingerprints={} if missing == "fingerprint" else {node.node_id: "content"},
            node_changes={} if missing == "changes" else {node.node_id: NodeChanges()},
        )},
    ).execute(target_trade_date="2026-09-04", scope="demo")
    assert result["status"] == "failed"
    assert result["operations"]["test"].error_category == "contract"
    assert "incomplete output identity" in result["operations"]["test"].error
    assert not (tmp_path / "checkpoints").exists()


def test_graph_inputs_cannot_be_omitted_by_a_binding(tmp_path):
    source = _node("data.source", "source", Layer.DATA_SOURCE)
    child = _node("product.child", "child", Layer.PRODUCT, inputs=(DependencyEdge(source.node_id),))
    deps = DependencyRegistry((source, child), {"demo": (child.node_id,)})
    ops = OperationRegistry((_operation("source", (source.node_id,)), _operation("child", (child.node_id,))))
    executor = DailyDagExecutor(deps, ops, project_root=tmp_path, require_identity=True,
                                handlers={"child": lambda context: pytest.fail("missing graph input")})
    with pytest.raises(ValueError, match="missing external input identities"):
        executor.execute(target_trade_date="2026-09-04", scope="demo", node_ids=(child.node_id,))


def test_checkpoint_hashes_runtime_parameters_code_and_state(tmp_path):
    (tmp_path / "contract.py").write_text("version = 1")
    operation = _operation("test", ("product.test",), cache=CachePolicy(
        CacheMode.APPEND_STATE, "v1", ("output",), state_path="state", contract_paths=("contract.py",),
    ))
    store = CheckpointStore(tmp_path, tmp_path / "checkpoints")
    context = OperationContext("2026-09-04", "demo", 1, {}, parameters={"model": "a"})
    before, _ = store.build_identity(operation, context)
    after, _ = store.build_identity(operation, replace(context, parameters={"model": "b"}))
    assert before != after
    (tmp_path / "contract.py").write_text("version = 2")
    assert before != store.build_identity(operation, context)[0]
    for name in ("output", "state"):
        (tmp_path / name).write_text("current")
    identity, payload = store.build_identity(operation, context)
    result = OperationResult(
        status="success", node_results={"product.test": {}},
        output_fingerprints={"product.test": "current"}, node_changes={"product.test": NodeChanges()},
        contract_identity=identity,
    )
    store.save(operation, identity, payload, result)
    assert store.load(operation, identity) is not None
    (tmp_path / "state").write_text("corrupt-state")
    assert store.load(operation, identity) is None


def test_contract_and_integrity_changes_request_full_rebuild(tmp_path):
    node = _node("product.test", "test", Layer.PRODUCT)
    operation = _operation("test", (node.node_id,), cache=CachePolicy(CacheMode.EXACT_DATE, "v1", ("output",)))
    store = CheckpointStore(tmp_path, tmp_path / "checkpoints")
    context = OperationContext("2026-09-04", "demo", 1, {})
    identity, payload = store.build_identity(operation, context)
    (tmp_path / "output").write_text("current")
    store.save(operation, identity, payload, OperationResult(
        status="success", node_results={node.node_id: {}},
        output_fingerprints={node.node_id: "current"}, node_changes={node.node_id: NodeChanges()},
        contract_identity=identity,
    ))
    changed_operation = replace(operation, cache=replace(operation.cache, contract_version="v2"))
    _, changed_payload = store.build_identity(changed_operation, context)
    assert store.requires_full_rebuild(changed_operation, context, changed_payload)
    assert store.requires_full_rebuild(operation, context, payload)


def test_atomic_operation_cycle_is_rejected_before_work(tmp_path):
    first = _node("data.first", "shared", Layer.DATA_SOURCE)
    middle = _node("product.middle", "middle", Layer.PRODUCT, inputs=(DependencyEdge(first.node_id),))
    last = _node("product.last", "shared", Layer.PRODUCT, inputs=(DependencyEdge(middle.node_id),))
    deps = DependencyRegistry((first, middle, last), {"demo": (last.node_id,)})
    registry = OperationRegistry((
        _operation("shared", (first.node_id, last.node_id)), _operation("middle", (middle.node_id,)),
    ))
    with pytest.raises(ValueError, match="cyclic atomic operation"):
        DailyDagExecutor(deps, registry, project_root=tmp_path, handlers={
            "shared": lambda context: pytest.fail("cyclic graph"),
            "middle": lambda context: pytest.fail("cyclic graph"),
        }).execute(target_trade_date="2026-09-04", scope="demo")


def test_source_operation_cannot_claim_parameter_only_identity(tmp_path):
    source = _node("data.source", "source", Layer.DATA_SOURCE)
    product = _node("product.test", "test", Layer.PRODUCT, inputs=(DependencyEdge(source.node_id),))
    executor = DailyDagExecutor(
        DependencyRegistry((source, product), {"demo": (product.node_id,)}),
        OperationRegistry((_operation("source", (source.node_id,)), _operation("test", (product.node_id,)))),
        project_root=tmp_path, require_identity=True,
        handlers={
            "source": lambda context: pytest.fail("source revision must be declared"),
            "test": lambda context: pytest.fail("source revision must be declared"),
        },
    )
    with pytest.raises(ValueError, match="pinned source identity"):
        executor.execute(target_trade_date="2026-09-04", scope="demo")
