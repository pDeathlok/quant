from __future__ import annotations

from datetime import date
from pathlib import Path
import threading
import time

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
        return _success(feature.node_id)

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
