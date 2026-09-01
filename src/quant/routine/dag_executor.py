"""Resource-bounded executor for the declarative daily refresh DAG."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
import importlib
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable, Mapping

from quant.application.daily_dependencies import DependencyRegistry
from quant.routine.checkpoint_store import CheckpointStore
from quant.routine.operation_contracts import (
    OperationContext,
    OperationDefinition,
    OperationResult,
    ResourceClaim,
)
from quant.routine.operation_registry import OperationRegistry


OperationHandler = Callable[[OperationContext], OperationResult]


@dataclass(frozen=True)
class ResourceBudget:
    cpu_slots: int
    io_slots: int = 4
    memory_mb: int = 4096
    rate_limit_concurrency: Mapping[str, int] | None = None

    @classmethod
    def from_environment(cls) -> "ResourceBudget":
        cpu = max(1, int(os.getenv("ROUTINE_TOTAL_WORKERS", str(os.cpu_count() or 1))))
        return cls(
            cpu_slots=cpu,
            io_slots=max(1, int(os.getenv("ROUTINE_IO_SLOTS", "4"))),
            memory_mb=max(256, int(os.getenv("ROUTINE_MEMORY_BUDGET_MB", "4096"))),
            rate_limit_concurrency={"tushare": 1, "akshare": 2},
        )


class _ResourcePool:
    def __init__(self, budget: ResourceBudget) -> None:
        self.budget = budget
        self._cpu = 0
        self._io = 0
        self._memory = 0
        self._groups: dict[str, int] = {}
        self._lock = threading.Lock()
        self.max_cpu = 0
        self.max_io = 0
        self.max_memory = 0

    def validate(self, definition: OperationDefinition) -> None:
        claim = definition.resources
        if claim.cpu_slots > self.budget.cpu_slots:
            raise ValueError(
                f"operation {definition.operation_id} requests {claim.cpu_slots} CPU "
                f"slots but budget is {self.budget.cpu_slots}"
            )
        if claim.io_slots > self.budget.io_slots:
            raise ValueError(
                f"operation {definition.operation_id} requests {claim.io_slots} IO "
                f"slots but budget is {self.budget.io_slots}"
            )
        if claim.memory_mb > self.budget.memory_mb:
            raise ValueError(
                f"operation {definition.operation_id} requests {claim.memory_mb} MB "
                f"but budget is {self.budget.memory_mb} MB"
            )

    def try_acquire(self, claim: ResourceClaim) -> bool:
        with self._lock:
            group_limit = None
            group_used = 0
            if claim.rate_limit_group:
                limits = self.budget.rate_limit_concurrency or {}
                group_limit = max(1, int(limits.get(claim.rate_limit_group, 1)))
                group_used = self._groups.get(claim.rate_limit_group, 0)
            if (
                self._cpu + claim.cpu_slots > self.budget.cpu_slots
                or self._io + claim.io_slots > self.budget.io_slots
                or self._memory + claim.memory_mb > self.budget.memory_mb
                or (group_limit is not None and group_used >= group_limit)
            ):
                return False
            self._cpu += claim.cpu_slots
            self._io += claim.io_slots
            self._memory += claim.memory_mb
            if claim.rate_limit_group:
                self._groups[claim.rate_limit_group] = group_used + 1
            self.max_cpu = max(self.max_cpu, self._cpu)
            self.max_io = max(self.max_io, self._io)
            self.max_memory = max(self.max_memory, self._memory)
            return True

    def release(self, claim: ResourceClaim) -> None:
        with self._lock:
            self._cpu -= claim.cpu_slots
            self._io -= claim.io_slots
            self._memory -= claim.memory_mb
            if claim.rate_limit_group:
                current = self._groups.get(claim.rate_limit_group, 0) - 1
                if current > 0:
                    self._groups[claim.rate_limit_group] = current
                else:
                    self._groups.pop(claim.rate_limit_group, None)


def _load_handler(entrypoint: str) -> OperationHandler:
    module_name, function_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    handler = getattr(module, function_name)
    if not callable(handler):
        raise TypeError(f"operation entrypoint is not callable: {entrypoint}")
    return handler


class DailyDagExecutor:
    def __init__(
        self,
        dependencies: DependencyRegistry,
        operations: OperationRegistry,
        *,
        project_root: Path,
        checkpoint_store: CheckpointStore | None = None,
        budget: ResourceBudget | None = None,
        handlers: Mapping[str, OperationHandler] | None = None,
        progress_callback: Callable[[str, str, Mapping[str, Any]], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.dependencies = dependencies
        self.operations = operations
        self.project_root = project_root.resolve()
        self.checkpoint_store = checkpoint_store
        self.budget = budget or ResourceBudget.from_environment()
        self.handlers = dict(handlers or {})
        self.progress_callback = progress_callback
        self.sleep_fn = sleep_fn

    def _emit(self, operation_id: str, status: str, **details: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(operation_id, status, details)

    def _selected_nodes(
        self,
        scope: str,
        node_ids: Iterable[str] | None,
    ) -> tuple[str, ...]:
        if node_ids is None:
            return self.dependencies.required_node_ids(scope)
        selected = set(node_ids)
        unknown = sorted(selected - set(self.dependencies.nodes))
        if unknown:
            raise KeyError(f"unknown daily dependency nodes: {unknown}")
        return self.dependencies.topological_order(selected)

    def _operation_dependencies(
        self,
        selected_nodes: Iterable[str],
    ) -> dict[str, set[str]]:
        selected = set(selected_nodes)
        required = {
            self.dependencies.nodes[node_id].operation_id: set()
            for node_id in selected
        }
        for node_id in selected:
            node = self.dependencies.nodes[node_id]
            operation_id = node.operation_id
            for edge in node.inputs:
                if edge.upstream not in selected:
                    continue
                upstream_operation = self.dependencies.nodes[edge.upstream].operation_id
                if upstream_operation != operation_id:
                    required[operation_id].add(upstream_operation)
        return required

    def _context_for(
        self,
        definition: OperationDefinition,
        operation_dependencies: Mapping[str, set[str]],
        completed: Mapping[str, OperationResult],
        *,
        target_trade_date: str,
        scope: str,
        initial_dirty_partitions: Iterable[str] = (),
        initial_dirty_keys: Iterable[str] = (),
    ) -> OperationContext:
        upstream_results: dict[str, Mapping[str, Any]] = {}
        upstream_revisions: dict[str, int] = {}
        upstream_fingerprints: dict[str, str] = {}
        dirty_partitions: set[str] = set(initial_dirty_partitions)
        dirty_keys: set[str] = set(initial_dirty_keys)
        for operation_id in operation_dependencies[definition.operation_id]:
            result = completed[operation_id]
            upstream_results.update(result.node_results)
            upstream_revisions.update(result.dataset_revisions)
            upstream_fingerprints.update(result.output_fingerprints)
            dirty_partitions.update(result.changed_partitions)
            dirty_keys.update(result.changed_keys)
        claim = definition.resources
        granted_workers = min(
            claim.requested_workers,
            claim.max_workers,
            claim.cpu_slots,
        )
        return OperationContext(
            target_trade_date=target_trade_date,
            scope=scope,
            granted_workers=max(1, granted_workers),
            upstream_results=upstream_results,
            upstream_revisions=upstream_revisions,
            upstream_fingerprints=upstream_fingerprints,
            dirty_partitions=tuple(sorted(dirty_partitions)),
            dirty_keys=tuple(sorted(dirty_keys)),
            parameters=definition.parameters,
        )

    def _run_operation(
        self,
        definition: OperationDefinition,
        context: OperationContext,
    ) -> OperationResult:
        identity = ""
        identity_payload: Mapping[str, Any] = {}
        if self.checkpoint_store is not None:
            identity, identity_payload = self.checkpoint_store.build_identity(
                definition,
                context,
            )
            cached = self.checkpoint_store.load(definition, identity)
            if cached is not None:
                return replace(
                    cached,
                    metrics={**cached.metrics, "checkpoint_reused": True},
                )
        handler = self.handlers.get(definition.operation_id)
        if handler is None:
            handler = _load_handler(definition.entrypoint)
        last_result: OperationResult | None = None
        last_error: BaseException | None = None
        for attempt in range(1, definition.retry.attempts + 1):
            try:
                result = handler(context)
                if not isinstance(result, OperationResult):
                    raise TypeError(
                        f"operation {definition.operation_id} returned "
                        f"{type(result).__name__}, expected OperationResult"
                    )
                last_result = result
                retryable = (
                    result.status == "failed"
                    and attempt < definition.retry.attempts
                    and (
                        not definition.retry.retryable_categories
                        or result.error_category
                        in definition.retry.retryable_categories
                    )
                )
                if not retryable:
                    break
            except BaseException as exc:
                last_error = exc
                if attempt >= definition.retry.attempts:
                    break
            self.sleep_fn(definition.retry.interval_seconds)
        if last_error is not None and (
            last_result is None or last_result.status != "success"
        ):
            result = OperationResult(
                status="failed",
                node_results={},
                error_category="exception",
                error=str(last_error),
                metrics={"attempts": definition.retry.attempts},
            )
        elif last_result is not None:
            result = replace(
                last_result,
                metrics={
                    **last_result.metrics,
                    "checkpoint_reused": False,
                },
            )
        else:
            result = OperationResult(
                status="failed",
                node_results={},
                error_category="empty_result",
                error=f"operation {definition.operation_id} produced no result",
            )
        if result.status == "success":
            missing_nodes = sorted(
                set(definition.produces) - set(result.node_results)
            )
            if missing_nodes:
                return OperationResult(
                    status="failed",
                    node_results=result.node_results,
                    error_category="contract",
                    error=(
                        f"operation {definition.operation_id} omitted node results "
                        f"for {missing_nodes}"
                    ),
                    metrics=result.metrics,
                )
            if self.checkpoint_store is not None:
                self.checkpoint_store.save(
                    definition,
                    identity,
                    identity_payload,
                    result,
                )
        return result

    def plan(
        self,
        *,
        target_trade_date: str,
        scope: str,
        node_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Validate and serialize the executable graph without running handlers."""

        selected_nodes = self._selected_nodes(scope, node_ids)
        self.operations.validate_against_dependencies(
            self.dependencies,
            node_ids=selected_nodes,
        )
        definitions = self.operations.required_operations(
            self.dependencies,
            selected_nodes,
        )
        operation_dependencies = self._operation_dependencies(selected_nodes)
        resources = _ResourcePool(self.budget)
        for definition in definitions:
            resources.validate(definition)
        selected = set(selected_nodes)
        return {
            "status": "success",
            "mode": "shadow",
            "target_trade_date": target_trade_date,
            "scope": scope,
            "node_count": len(selected_nodes),
            "operation_count": len(definitions),
            "collapsed_node_count": len(selected_nodes) - len(definitions),
            "resource_budget": {
                "cpu_slots": self.budget.cpu_slots,
                "io_slots": self.budget.io_slots,
                "memory_mb": self.budget.memory_mb,
            },
            "operations": [
                {
                    "operation_id": definition.operation_id,
                    "produces": [
                        node_id
                        for node_id in definition.produces
                        if node_id in selected
                    ],
                    "depends_on": sorted(
                        operation_dependencies.get(definition.operation_id, set())
                    ),
                    "execution_mode": definition.execution_mode.value,
                    "cache_mode": definition.cache.mode.value,
                    "contract_version": definition.cache.contract_version,
                    "resources": {
                        "cpu_slots": definition.resources.cpu_slots,
                        "io_slots": definition.resources.io_slots,
                        "memory_mb": definition.resources.memory_mb,
                        "requested_workers": (
                            definition.resources.requested_workers
                        ),
                        "max_workers": definition.resources.max_workers,
                        "rate_limit_group": (
                            definition.resources.rate_limit_group
                        ),
                    },
                }
                for definition in definitions
            ],
        }

    def execute(
        self,
        *,
        target_trade_date: str,
        scope: str,
        node_ids: Iterable[str] | None = None,
        dirty_partitions: Iterable[str] = (),
        dirty_keys: Iterable[str] = (),
    ) -> dict[str, Any]:
        selected_nodes = self._selected_nodes(scope, node_ids)
        self.operations.validate_against_dependencies(
            self.dependencies,
            node_ids=selected_nodes,
        )
        definitions = {
            definition.operation_id: definition
            for definition in self.operations.required_operations(
                self.dependencies,
                selected_nodes,
            )
            if definition.enabled
        }
        operation_dependencies = self._operation_dependencies(selected_nodes)
        operation_dependencies = {
            operation_id: dependencies & set(definitions)
            for operation_id, dependencies in operation_dependencies.items()
            if operation_id in definitions
        }
        resources = _ResourcePool(self.budget)
        for definition in definitions.values():
            resources.validate(definition)

        pending = set(definitions)
        running: dict[Future[OperationResult], str] = {}
        completed: dict[str, OperationResult] = {}
        started_at = time.monotonic()
        with ThreadPoolExecutor(
            max_workers=max(1, len(definitions)),
            thread_name_prefix="quant-daily-dag",
        ) as executor:
            while pending or running:
                blocked = [
                    operation_id
                    for operation_id in pending
                    if any(
                        dependency in completed
                        and completed[dependency].status != "success"
                        for dependency in operation_dependencies[operation_id]
                    )
                ]
                for operation_id in blocked:
                    pending.remove(operation_id)
                    completed[operation_id] = OperationResult(
                        status="cancelled",
                        node_results={},
                        error_category="upstream_failed",
                        error="one or more upstream operations failed",
                    )
                    self._emit(operation_id, "cancelled")

                ready = sorted(
                    (
                        operation_id
                        for operation_id in pending
                        if operation_dependencies[operation_id] <= set(completed)
                    ),
                    key=lambda operation_id: (
                        -definitions[operation_id].resources.cpu_slots,
                        operation_id,
                    ),
                )
                scheduled = False
                for operation_id in ready:
                    definition = definitions[operation_id]
                    if not resources.try_acquire(definition.resources):
                        continue
                    context = self._context_for(
                        definition,
                        operation_dependencies,
                        completed,
                        target_trade_date=target_trade_date,
                        scope=scope,
                        initial_dirty_partitions=dirty_partitions,
                        initial_dirty_keys=dirty_keys,
                    )
                    pending.remove(operation_id)
                    future = executor.submit(
                        self._run_operation,
                        definition,
                        context,
                    )
                    running[future] = operation_id
                    self._emit(
                        operation_id,
                        "running",
                        granted_workers=context.granted_workers,
                    )
                    scheduled = True

                if not running:
                    if pending:
                        raise RuntimeError(
                            "daily DAG cannot schedule remaining operations: "
                            + ", ".join(sorted(pending))
                        )
                    break
                if scheduled and ready:
                    continue
                done, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
                for future in done:
                    operation_id = running.pop(future)
                    definition = definitions[operation_id]
                    resources.release(definition.resources)
                    try:
                        result = future.result()
                    except BaseException as exc:
                        result = OperationResult(
                            status="failed",
                            node_results={},
                            error_category="executor",
                            error=str(exc),
                        )
                    completed[operation_id] = result
                    self._emit(
                        operation_id,
                        result.status,
                        error=result.error,
                    )

        node_results: dict[str, Mapping[str, Any]] = {}
        for result in completed.values():
            node_results.update(result.node_results)
        failed = sorted(
            operation_id
            for operation_id, result in completed.items()
            if result.status == "failed"
        )
        cancelled = sorted(
            operation_id
            for operation_id, result in completed.items()
            if result.status == "cancelled"
        )
        return {
            "status": "success" if not failed and not cancelled else "failed",
            "target_trade_date": target_trade_date,
            "scope": scope,
            "operations": completed,
            "node_results": node_results,
            "failed_operations": failed,
            "cancelled_operations": cancelled,
            "resource_usage": {
                "max_cpu_slots": resources.max_cpu,
                "max_io_slots": resources.max_io,
                "max_memory_mb": resources.max_memory,
                "budget_cpu_slots": self.budget.cpu_slots,
                "budget_io_slots": self.budget.io_slots,
                "budget_memory_mb": self.budget.memory_mb,
            },
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        }


__all__ = ["DailyDagExecutor", "OperationHandler", "ResourceBudget"]
