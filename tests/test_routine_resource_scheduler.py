from __future__ import annotations

from contextvars import ContextVar
import threading
import time

import pytest

from quant.application.daily_dependencies import DependencyRegistry, Layer
from quant.infrastructure.publication import ContextThreadPoolExecutor
from quant.routine.dag_executor import DailyDagExecutor
from quant.routine.operation_contracts import ResourceClaim
from quant.routine.operation_registry import OperationRegistry
from quant.routine.resource_scheduler import (
    ResourceBudget,
    ResourceScheduler,
    current_resource_grant,
    current_resource_scheduler,
)
from tests.test_daily_dag_executor import _node, _operation, _success


def test_external_and_dag_child_workers_share_resource_envelopes(tmp_path):
    scheduler = ResourceScheduler(ResourceBudget(
        cpu_slots=2, io_slots=2, memory_mb=512, db_connections=2,
        rate_limit_concurrency={"api": 2},
    ))
    node = _node("product.test", "test", Layer.PRODUCT)
    from dataclasses import replace

    operation = replace(_operation("test", (node.node_id,), cpu_slots=2), resources=ResourceClaim(
        cpu_slots=2, io_slots=2, memory_mb=512, requested_workers=2, max_workers=2,
        db_connections=2, rate_limit_group="api", api_slots=2,
    ))
    started = threading.Event()
    active = 0
    peak = 0
    lock = threading.Lock()
    request = ContextVar("test-request", default="missing")
    request.set("publication-request")

    def child(item, grant):
        nonlocal active, peak
        assert current_resource_grant() is grant
        assert request.get() == "publication-request"
        assert grant.granted_workers == 1
        with lock:
            active += 1
            peak = max(active, peak)
        time.sleep(0.01)
        with lock:
            active -= 1
        return item * 2

    def handler(context):
        started.set()
        assert current_resource_scheduler() is scheduler
        assert current_resource_grant() is context.resource_grant
        assert context.resource_grant.map(
            child, range(8), claim=ResourceClaim(
                memory_mb=128, io_slots=1, db_connections=1, rate_limit_group="api",
            ), max_pending=2,
        ) == list(range(0, 16, 2))
        return _success(node.node_id)

    def execute():
        return DailyDagExecutor(
            DependencyRegistry((node,), {"demo": (node.node_id,)}),
            OperationRegistry((operation,)), project_root=tmp_path, handlers={"test": handler},
        ).execute(target_trade_date="2026-09-04", scope="demo")

    held = threading.Event()
    release = threading.Event()

    def external_work():
        with scheduler.reserve(ResourceClaim(memory_mb=128)):
            held.set()
            assert release.wait(5)

    with scheduler.activate(), ContextThreadPoolExecutor(max_workers=2) as pool:
        external = pool.submit(external_work)
        assert held.wait(5)
        future = pool.submit(execute)
        assert not started.wait(0.05)
        release.set()
        external.result(timeout=5)
        report = future.result(timeout=5)
    assert report["status"] == "success"
    assert peak == 2
    assert scheduler.usage()["max_cpu_slots"] == 2
    assert scheduler.usage()["max_db_connections"] == 2
    assert scheduler.usage()["max_memory_mb"] == 512
    assert scheduler.usage()["max_api_concurrency"] == {"api": 2}
    with scheduler.reserve(operation.resources):
        pass  # All parent and child grants were returned.


def test_child_claims_cannot_expand_or_change_parent_budget():
    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=4))
    with scheduler.reserve(ResourceClaim(cpu_slots=4, requested_workers=2, max_workers=2)) as grant:
        with pytest.raises(ValueError, match="CPU slots"):
            with grant.reserve(ResourceClaim(cpu_slots=3)):
                pass
        with pytest.raises(ValueError, match="db_connections"):
            with grant.reserve(ResourceClaim(db_connections=1)):
                pass
        with pytest.raises(ValueError, match="API group"):
            with grant.reserve(ResourceClaim(rate_limit_group="undeclared")):
                pass
        with pytest.raises(ValueError, match="max_pending"):
            grant.map(lambda item, child: item, range(2), claim=ResourceClaim(), max_pending=100)
    with pytest.raises(RuntimeError, match="closed"):
        with grant.reserve(ResourceClaim()):
            pass


def test_child_queue_is_bounded_and_failure_releases_grants():
    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=2))
    consumed = 0
    finished = 0
    lock = threading.Lock()

    def items():
        nonlocal consumed
        for item in range(8):
            with lock:
                assert consumed - finished < 2
                consumed += 1
            yield item

    def child(item, grant):
        nonlocal finished
        time.sleep(0.005)
        with lock:
            finished += 1
        if item == 3:
            raise RuntimeError("child failed")
        return item

    claim = ResourceClaim(cpu_slots=2, requested_workers=2, max_workers=2)
    with scheduler.reserve(claim) as grant:
        with pytest.raises(RuntimeError, match="child failed"):
            grant.map(child, items(), claim=ResourceClaim(memory_mb=64), max_pending=2)
    assert consumed < 8
    with scheduler.reserve(claim):
        pass


def test_nested_children_subdivide_instead_of_reacquiring_global_slots():
    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=2))
    with scheduler.reserve(ResourceClaim(cpu_slots=2, requested_workers=2, max_workers=2)) as parent:
        results = parent.map(
            lambda item, child: child.map(
                lambda value, grandchild: (value, grandchild.granted_workers),
                (item, item + 1), claim=ResourceClaim(memory_mb=32),
            ),
            (0, 2), claim=ResourceClaim(memory_mb=64),
        )
    assert results == [[(0, 1), (1, 1)], [(2, 1), (3, 1)]]
    assert scheduler.usage()["max_cpu_slots"] == 2


def test_nested_dag_uses_parent_child_scheduler_without_global_deadlock(tmp_path):
    node = _node("product.nested", "nested", Layer.PRODUCT)
    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=1))
    with scheduler.reserve(ResourceClaim()) as parent:
        with parent.child_scheduler() as children:
            with pytest.raises(ValueError, match="API group"):
                children.try_acquire(ResourceClaim(rate_limit_group="unreserved"))
            report = DailyDagExecutor(
                DependencyRegistry((node,), {"demo": (node.node_id,)}),
                OperationRegistry((_operation("nested", (node.node_id,)),)),
                project_root=tmp_path, scheduler=children,
                handlers={"nested": lambda context: _success(node.node_id)},
            ).execute(target_trade_date="2026-09-04", scope="demo")
    assert report["status"] == "success"
    assert scheduler.usage()["max_cpu_slots"] == 1


def test_nested_dag_automatically_subdivides_current_grant(tmp_path):
    from quant.routine.resource_scheduler import run_with_resources

    node = _node("product.nested", "nested", Layer.PRODUCT)
    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=1))

    def nested():
        return DailyDagExecutor(
            DependencyRegistry((node,), {"demo": (node.node_id,)}),
            OperationRegistry((_operation("nested", (node.node_id,)),)),
            project_root=tmp_path,
            handlers={"nested": lambda context: _success(node.node_id)},
        ).execute(target_trade_date="2026-09-04", scope="demo")

    with scheduler.activate():
        report = run_with_resources(ResourceClaim(), nested)
    assert report["status"] == "success"
    assert scheduler.usage()["max_cpu_slots"] == 1


def test_pool_hook_subdivides_parent_and_releases_on_failure():
    from quant.routine.resource_scheduler import run_with_resources

    scheduler = ResourceScheduler(ResourceBudget(cpu_slots=1))

    def fail():
        assert current_resource_grant().granted_workers == 1
        raise RuntimeError("pool failed")

    with scheduler.activate(), scheduler.reserve(ResourceClaim()) as parent:
        with pytest.raises(RuntimeError, match="pool failed"):
            run_with_resources(ResourceClaim(memory_mb=64), fail)
        assert current_resource_grant() is parent
    with scheduler.reserve(ResourceClaim()):
        pass
