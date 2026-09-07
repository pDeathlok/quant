"""Resource-bounded executor for the declarative daily refresh DAG."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import replace
import importlib
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

from quant.application.daily_dependencies import DependencyRegistry, Layer
from quant.infrastructure.publication import ContextThreadPoolExecutor
from quant.routine.checkpoint_store import CheckpointStore, validate_result_identity
from quant.routine.operation_contracts import (
    CacheMode,
    InputSnapshot,
    NodeChanges,
    OperationContext,
    OperationDefinition,
    OperationHandler,
    OperationResult,
)
from quant.routine.operation_registry import OperationRegistry
from quant.routine.resource_scheduler import (
    ResourceBudget,
    ResourceGrant,
    ResourceScheduler,
    current_resource_grant,
    current_resource_scheduler,
)


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
        scheduler: ResourceScheduler | None = None,
        require_identity: bool = False,
        progress_callback: Callable[[str, str, Mapping[str, Any]], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.dependencies = dependencies
        self.operations = operations
        self.project_root = project_root.resolve()
        self.checkpoint_store = checkpoint_store
        parent = current_resource_grant()
        self.scheduler = scheduler or (parent.scheduler if parent else None) or current_resource_scheduler() or ResourceScheduler(
            budget or ResourceBudget.from_environment()
        )
        if budget is not None and budget != self.scheduler.budget:
            raise ValueError("executor budget must match shared scheduler budget")
        self.budget = self.scheduler.budget
        self.require_identity = require_identity or checkpoint_store is not None
        self.handlers = {**operations.handlers, **(handlers or {})}
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
            selected = set(self.dependencies.required_node_ids(scope))
            pending = list(selected)
            while pending:
                node_id = pending.pop()
                definition = self.operations.definitions[self.dependencies.nodes[node_id].operation_id]
                inputs = set(definition.input_ids or ()) | {
                    edge.upstream for produced in definition.produces
                    for edge in self.dependencies.nodes[produced].inputs
                }
                additions = (inputs & set(self.dependencies.nodes)) - selected
                selected.update(additions)
                pending.extend(additions)
            return self.dependencies.topological_order(selected)
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
        for operation_id in required:
            for input_id in self._input_ids(self.operations.definitions[operation_id]):
                upstream_operation = self.operations.operation_by_node.get(input_id)
                if upstream_operation in required and upstream_operation != operation_id:
                    required[operation_id].add(upstream_operation)
        unresolved = {operation: set(inputs) for operation, inputs in required.items()}
        while unresolved:
            ready = {operation for operation, inputs in unresolved.items() if not inputs}
            if not ready:
                raise ValueError(f"cyclic atomic operation dependencies: {sorted(unresolved)}")
            unresolved = {
                operation: inputs - ready
                for operation, inputs in unresolved.items() if operation not in ready
            }
        return required

    def _input_ids(self, definition: OperationDefinition) -> tuple[str, ...]:
        # A multi-output operation executes atomically even for a partial stage.
        graph_inputs = {
            edge.upstream
            for node_id in definition.produces
            for edge in self.dependencies.nodes[node_id].inputs
            if edge.upstream not in definition.produces
        }
        inputs = tuple(sorted(graph_inputs | set(definition.input_ids or ())))
        if self.require_identity and not inputs and any(
            self.dependencies.nodes[node].layer == Layer.DATA_SOURCE
            for node in definition.produces
        ):
            raise ValueError(
                f"operation {definition.operation_id} requires a pinned source identity; "
                "a data-source operation cannot be parameter-only"
            )
        return inputs

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
        input_snapshots: Mapping[str, InputSnapshot] | None = None,
        grant: ResourceGrant | None = None,
    ) -> OperationContext:
        upstream_results: dict[str, Mapping[str, Any]] = {}
        upstream_revisions: dict[str, int] = {}
        upstream_fingerprints: dict[str, str] = {}
        root = not operation_dependencies[definition.operation_id]
        dirty_partitions: set[str] = set(initial_dirty_partitions if root else ())
        dirty_keys: set[str] = set(initial_dirty_keys if root else ())
        changes: dict[str, NodeChanges] = {}
        for node_id in self._input_ids(definition):
            producer = self.operations.operation_by_node.get(node_id)
            result = completed.get(producer)
            if result is not None:
                if node_id in result.node_results:
                    upstream_results[node_id] = result.node_results[node_id]
                if node_id in result.dataset_revisions:
                    upstream_revisions[node_id] = result.dataset_revisions[node_id]
                if node_id in result.output_fingerprints:
                    upstream_fingerprints[node_id] = result.output_fingerprints[node_id]
                changes[node_id] = result.node_changes.get(node_id, NodeChanges(
                    partitions=result.changed_partitions, keys=result.changed_keys,
                ))
            elif node_id in (input_snapshots or {}):
                snapshot = input_snapshots[node_id]
                upstream_results[node_id] = snapshot.payload
                upstream_fingerprints[node_id] = snapshot.fingerprint
                if snapshot.revision is not None:
                    upstream_revisions[node_id] = snapshot.revision
                changes[node_id] = snapshot.changes
        for change in changes.values():
            dirty_partitions.update(change.partitions)
            dirty_keys.update(change.keys)
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
            required_input_ids=self._input_ids(definition),
            input_changes=changes,
            full_rebuild=any(change.full_rebuild for change in changes.values()),
            resource_grant=grant,
            identity_required=self.require_identity,
            project_root=self.project_root,
            output_paths={
                node: self.dependencies.nodes[node].outputs or definition.cache.output_paths
                for node in definition.produces
            },
        )

    def _run_operation(
        self,
        definition: OperationDefinition,
        context: OperationContext,
    ) -> OperationResult:
        identity = ""
        identity_payload: Mapping[str, Any] = {}
        identity_store = self.checkpoint_store or (
            CheckpointStore(self.project_root, self.project_root / ".unused-checkpoints")
            if self.require_identity else None
        )
        if identity_store is not None:
            identity, identity_payload = identity_store.build_identity(
                definition,
                context,
            )
            cached = self.checkpoint_store.load(definition, identity) if self.checkpoint_store else None
            if cached is not None:
                return replace(
                    cached,
                    metrics={**cached.metrics, "checkpoint_reused": True},
                )
            if self.checkpoint_store is not None and definition.cache.mode != CacheMode.NONE:
                context = replace(context, full_rebuild=(
                    context.full_rebuild or self.checkpoint_store.requires_full_rebuild(
                        definition, context, identity_payload,
                    )
                ))
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
            # Runtime-selected model/configuration bytes must remain the ones
            # used to identify this run, including when no checkpoint is saved.
            if identity_store is not None:
                final_identity, _ = identity_store.build_identity(definition, context)
                if final_identity != identity:
                    return OperationResult(
                        status="failed", node_results={}, error_category="contract",
                        error=f"operation {definition.operation_id} contract changed during execution",
                        metrics=result.metrics,
                    )
            missing_nodes = sorted(
                set(definition.produces) - set(result.node_results)
            )
            invalid_nodes = sorted(
                node for node, payload in result.node_results.items()
                if payload.get("status") in {"failed", "cancelled", "shadow", "shadow_only"}
            )
            if missing_nodes or invalid_nodes or set(result.node_results) - set(definition.produces):
                return OperationResult(
                    status="failed",
                    node_results=result.node_results,
                    error_category="contract",
                    error=(
                        f"operation {definition.operation_id} omitted node results "
                        f"for {missing_nodes}; invalid nodes {invalid_nodes}"
                    ),
                    metrics=result.metrics,
                )
            result = replace(
                result,
                input_fingerprints=dict(context.upstream_fingerprints),
                input_revisions=dict(context.upstream_revisions),
                contract_identity=identity or None,
            )
            if self.require_identity:
                validate_result_identity(definition, result)
            if self.checkpoint_store is not None and definition.cache.mode != CacheMode.NONE:
                self.checkpoint_store.save(
                    definition,
                    identity,
                    identity_payload,
                    result,
                )
        return result

    def _run_with_grant(
        self, definition: OperationDefinition, context: OperationContext,
    ) -> OperationResult:
        grant = context.resource_grant
        assert grant is not None
        try:
            with grant.activate():
                return self._run_operation(definition, context)
        except ValueError as exc:
            return OperationResult(
                status="failed", node_results={}, error_category="contract", error=str(exc),
            )
        finally:
            grant.close()

    def plan(
        self,
        *,
        target_trade_date: str,
        scope: str,
        node_ids: Iterable[str] | None = None,
        production: bool = False,
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
        if production:
            self.operations.validate_executable(
                definitions, self.handlers, require_identity=self.require_identity,
            )
        operation_dependencies = self._operation_dependencies(selected_nodes)
        resources = self.scheduler
        for definition in definitions:
            resources.validate(definition.resources, definition.operation_id)
        selected = set(selected_nodes)
        return {
            "status": "success",
            "mode": "production" if production else "shadow",
            "target_trade_date": target_trade_date,
            "scope": scope,
            "node_count": len(selected_nodes),
            "operation_count": len(definitions),
            "collapsed_node_count": len(selected_nodes) - len(definitions),
            "external_input_ids": sorted({
                input_id for definition in definitions for input_id in self._input_ids(definition)
            } - {node for definition in definitions for node in definition.produces}),
            "resource_budget": {
                "cpu_slots": self.budget.cpu_slots,
                "io_slots": self.budget.io_slots,
                "memory_mb": self.budget.memory_mb,
            },
            "operations": [
                {
                    "operation_id": definition.operation_id,
                    "executable": definition.production_ready or definition.operation_id in self.handlers,
                    "identity_declared": definition.input_ids is not None,
                    "input_ids": list(self._input_ids(definition)),
                    "all_produced_nodes": list(definition.produces),
                    "output_paths": list(definition.cache.output_paths),
                    "contract_paths": list(definition.cache.contract_paths),
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
                        "db_connections": definition.resources.db_connections,
                        "api_slots": definition.resources.api_slots,
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
        input_snapshots: Mapping[str, InputSnapshot] | None = None,
    ) -> dict[str, Any]:
        parent = current_resource_grant()
        if parent is not None and self.scheduler is parent.scheduler:
            # The coordinator already owns these slots. Reacquiring them from
            # the global scheduler would deadlock at a saturated budget.
            with parent.child_scheduler() as children:
                return DailyDagExecutor(
                    self.dependencies, self.operations, project_root=self.project_root,
                    checkpoint_store=self.checkpoint_store, scheduler=children,
                    handlers=self.handlers, require_identity=self.require_identity,
                    progress_callback=self.progress_callback, sleep_fn=self.sleep_fn,
                ).execute(
                    target_trade_date=target_trade_date, scope=scope, node_ids=node_ids,
                    dirty_partitions=dirty_partitions, dirty_keys=dirty_keys,
                    input_snapshots=input_snapshots,
                )
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
        }
        self.operations.validate_executable(
            definitions.values(), self.handlers, require_identity=self.require_identity,
        )
        operation_dependencies = self._operation_dependencies(selected_nodes)
        operation_dependencies = {
            operation_id: dependencies & set(definitions)
            for operation_id, dependencies in operation_dependencies.items()
            if operation_id in definitions
        }
        resources = self.scheduler
        for definition in definitions.values():
            resources.validate(definition.resources, definition.operation_id)
        snapshots = dict(input_snapshots or {})
        if any(not isinstance(value, InputSnapshot) for value in snapshots.values()):
            raise ValueError("input_snapshots must contain InputSnapshot values")
        produced = {node for definition in definitions.values() for node in definition.produces}
        if produced & set(snapshots):
            raise ValueError("external input snapshots overlap selected operation outputs")
        if self.require_identity:
            missing = {
                node for definition in definitions.values() for node in self._input_ids(definition)
                if node not in produced and node not in snapshots
            }
            if missing:
                raise ValueError(f"missing external input identities: {sorted(missing)}")
        # Iterables may be one-shot; do not consume dirty hints once per worker.
        dirty_partitions = tuple(dirty_partitions)
        dirty_keys = tuple(dirty_keys)

        pending = set(definitions)
        running: dict[Future[OperationResult], str] = {}
        grants: dict[str, ResourceGrant] = {}
        completed: dict[str, OperationResult] = {}
        started_at = time.monotonic()
        with resources.activate(), ContextThreadPoolExecutor(
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
                    context = self._context_for(
                        definition,
                        operation_dependencies,
                        completed,
                        target_trade_date=target_trade_date,
                        scope=scope,
                        initial_dirty_partitions=dirty_partitions,
                        initial_dirty_keys=dirty_keys,
                        input_snapshots=snapshots,
                    )
                    grant = resources.try_acquire(definition.resources)
                    if grant is None:
                        continue
                    context = replace(context, resource_grant=grant)
                    pending.remove(operation_id)
                    try:
                        future = executor.submit(self._run_with_grant, definition, context)
                    except BaseException:
                        grant.close()
                        raise
                    running[future] = operation_id
                    grants[operation_id] = grant
                    self._emit(
                        operation_id,
                        "running",
                        granted_workers=context.granted_workers,
                    )
                    scheduled = True

                if not running:
                    if pending:
                        if ready:
                            resources.wait_for_release()
                            continue
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
                    grants.pop(operation_id).close()
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
            "resource_usage": resources.usage(),
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        }


__all__ = ["DailyDagExecutor", "OperationHandler", "ResourceBudget", "ResourceScheduler"]
