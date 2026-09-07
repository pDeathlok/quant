"""Executable, non-cacheable adapters for composition-owned workspace callbacks."""

from __future__ import annotations

from typing import Any, Callable
from pathlib import Path
from dataclasses import replace

from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY, DependencyRegistry
from quant.routine.default_operations import DEFAULT_DAILY_OPERATION_REGISTRY, COMPOSED_OPERATION_GROUPS
from quant.routine.operation_contracts import (
    CacheMode, CachePolicy, NodeChanges, OperationBinding, OperationContext,
    OperationResult, ResourceClaim,
)
from quant.routine.dag_executor import DailyDagExecutor
from quant.routine.operation_registry import OperationRegistry
from quant.routine.resource_scheduler import current_resource_grant


COMPOSITION_CALLBACK_IDS = frozenset({
    "refresh_data", "refresh_daily_basic", "refresh_reference_inputs",
    "generate_daily_plan", "build_selector_payload", "write_strategy_pool_snapshots",
    "refresh_long_factor_snapshot", "refresh_long_stock_pool_variants",
    "generate_chan_model_strategy", "build_convertible_bond_grid_workspace",
    "build_convertible_bond_allotment_workspace", "build_byd_daily_workspace",
    "refresh_similar_pattern_analysis",
})


def validate_composed_closure(
    scope: str, *, dependencies: DependencyRegistry = DEFAULT_DAILY_DEPENDENCY_REGISTRY,
) -> None:
    """Fail before source writes when an active node lacks an executable owner."""
    owners = {
        member for operation in COMPOSITION_CALLBACK_IDS
        for member in COMPOSED_OPERATION_GROUPS.get(operation, (operation,))
    }
    owners.update(
        key for key, definition in DEFAULT_DAILY_OPERATION_REGISTRY.definitions.items()
        if definition.production_ready
    )
    active = {dependencies.nodes[node].operation_id for node in dependencies.required_node_ids(scope)}
    missing = active - owners
    if missing:
        raise ValueError(f"Active operations have no executable composition owner: {sorted(missing)}")


def execute_composed_operation(
    operation_id: str, callback: Callable[..., Any], *args: Any,
    target_trade_date: str, scope: str, project_root: Path,
    **kwargs: Any,
) -> Any:
    """Invoke the real callback exactly once, never manufacture a cache identity.

    The existing callback and postflight own domain validation. Its returned
    failure status or exception must fail execution. These adapters intentionally
    do not assert that hidden inputs of the callback have been identity-audited.
    """
    if operation_id not in COMPOSITION_CALLBACK_IDS:
        raise ValueError(f"Callback is not registered for production: {operation_id}")
    members = COMPOSED_OPERATION_GROUPS.get(operation_id, (operation_id,))
    active = {
        DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes[node].operation_id
        for node in DEFAULT_DAILY_DEPENDENCY_REGISTRY.required_node_ids(scope)
    }
    members = tuple(member for member in members if member in active or member == operation_id)
    if not members:
        raise ValueError(f"No active members for composition group {operation_id}")
    produced = tuple(
        node for member in members
        for node in DEFAULT_DAILY_OPERATION_REGISTRY.definitions[member].produces
    )
    dependencies = DependencyRegistry([
        replace(node, operation_id=operation_id) if node.operation_id in members else node
        for node in DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes.values()
    ], DEFAULT_DAILY_DEPENDENCY_REGISTRY.scope_roots)
    definition = replace(
        DEFAULT_DAILY_OPERATION_REGISTRY.definitions[members[0]],
        operation_id=operation_id, produces=produced,
    )
    values: list[Any] = []

    def invoke(context: OperationContext) -> OperationResult:
        value = callback(*args, **kwargs)
        if isinstance(value, dict) and value.get("status") in {
            "failed", "cancelled", "error", "shadow", "shadow_only",
        }:
            raise RuntimeError(f"{operation_id}: {value.get('error') or value.get('stderr_tail') or value}")
        if value is None:
            raise RuntimeError(f"{operation_id} returned no result")
        values.append(value)
        payload = value if isinstance(value, dict) else {"result": value}
        return OperationResult(
            status="success", node_results={node: payload for node in definition.produces},
            node_changes={node: NodeChanges(full_rebuild=True) for node in definition.produces},
            metrics={"identity_audited": False, "central_reuse": False, "atomic_group_members": members},
        )

    parent = current_resource_grant()
    binding = OperationBinding(
        handler=invoke,
        input_ids=tuple(sorted({
            edge.upstream for node in definition.produces
            for edge in DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes[node].inputs
            if edge.upstream not in definition.produces
        })),
        cache=CachePolicy(CacheMode.NONE, "composition-callback-v1"),
        resources=parent.claim if parent else replace(definition.resources, db_connections=max(1, definition.resources.db_connections)),
        parameters={"identity_audited": False},
    )
    registry = OperationRegistry([
        item for key, item in DEFAULT_DAILY_OPERATION_REGISTRY.definitions.items()
        if key not in members
    ] + [definition], bindings={operation_id: binding})
    registry.validate_against_dependencies(dependencies)
    execution = DailyDagExecutor(
        dependencies, registry, project_root=project_root, require_identity=False,
    ).execute(target_trade_date=target_trade_date, scope=scope, node_ids=produced)
    if execution["status"] != "success":
        errors = [result.error for result in execution["operations"].values() if result.error]
        raise RuntimeError(f"{operation_id} failed: {'; '.join(errors)}")
    if len(values) != 1:
        raise RuntimeError(f"{operation_id} did not execute exactly once")
    return values[0]


__all__ = ["execute_composed_operation", "validate_composed_closure", "COMPOSITION_CALLBACK_IDS"]
