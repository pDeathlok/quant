"""Production operation registry derived from the daily dependency graph."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Iterable

from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
    DependencyNode,
    DependencyRegistry,
    Layer,
)
from quant.routine.operation_contracts import (
    CacheMode,
    CachePolicy,
    ExecutionMode,
    OperationDefinition,
    ResourceClaim,
    RetryPolicy,
)
from quant.routine.operation_registry import OperationRegistry


_RESOURCE_PROFILES: dict[str, ResourceClaim] = {
    "refresh_data": ResourceClaim(1, 1, 768, "tushare", 1, 1),
    "refresh_daily_basic": ResourceClaim(1, 1, 768, "tushare", 1, 1),
    "refresh_index_000300": ResourceClaim(1, 1, 512, "tushare", 1, 1),
    "refresh_stock_basic": ResourceClaim(1, 1, 512, "tushare", 1, 1),
    "refresh_financial_pit": ResourceClaim(1, 1, 768, "tushare", 1, 1),
    "refresh_analyst_forecasts": ResourceClaim(1, 1, 512, "tushare", 1, 1),
    "refresh_top_list": ResourceClaim(1, 1, 512, "tushare", 1, 1),
    "refresh_strategy_signal_cache": ResourceClaim(8, 1, 2560, None, 8, 8),
    "refresh_active_project_features": ResourceClaim(6, 1, 2560, None, 6, 6),
    "run_right_side_unified": ResourceClaim(6, 1, 2048, None, 6, 6),
    "run_left_side_unified": ResourceClaim(2, 1, 1024, None, 2, 2),
    "refresh_chan_model_scores": ResourceClaim(4, 1, 2048, None, 4, 4),
    "build_selector_payload": ResourceClaim(1, 1, 1536, None, 1, 1),
    "refresh_long_factor_snapshot": ResourceClaim(4, 1, 2048, None, 4, 4),
    "refresh_long_stock_pool_variants": ResourceClaim(4, 1, 2048, None, 4, 4),
    "refresh_similar_reference_vectors_when_due": ResourceClaim(
        4, 1, 2048, None, 4, 4
    ),
    "score_similar_patterns": ResourceClaim(4, 1, 2048, None, 4, 4),
    "refresh_similar_pattern_analysis": ResourceClaim(4, 1, 2048, None, 4, 4),
}

_APPEND_STATE_OPERATIONS = {
    "refresh_strategy_signal_cache",
    "refresh_chan_model_scores",
    "refresh_similar_reference_vectors_when_due",
}

_ENTRYPOINTS = {
    "refresh_strategy_signal_cache": (
        "quant.routine.operation_adapters:refresh_strategy_signals"
    ),
    "refresh_active_project_features": (
        "quant.routine.operation_adapters:refresh_active_project_features"
    ),
    "run_right_side_unified": (
        "quant.routine.operation_adapters:run_right_side_unified"
    ),
    "run_left_side_unified": (
        "quant.routine.operation_adapters:run_left_side_unified"
    ),
    "refresh_chan_model_scores": (
        "quant.routine.operation_adapters:refresh_chan_model_scores"
    ),
}


def _default_claim(operation_id: str) -> ResourceClaim:
    return _RESOURCE_PROFILES.get(
        operation_id,
        ResourceClaim(cpu_slots=1, io_slots=1, memory_mb=512),
    )


def _cache_mode(operation_id: str, nodes: Iterable[DependencyNode]) -> CacheMode:
    materialized = tuple(nodes)
    if not any(node.outputs for node in materialized):
        return CacheMode.NONE
    if operation_id in _APPEND_STATE_OPERATIONS:
        return CacheMode.APPEND_STATE
    if any(node.layer == Layer.DATA_SOURCE for node in materialized):
        return CacheMode.PARTITION_REPLACE
    return CacheMode.EXACT_DATE


def _contract_version(operation_id: str, nodes: Iterable[DependencyNode]) -> str:
    payload = {
        "operation_id": operation_id,
        "nodes": [
            {
                "node_id": node.node_id,
                "contract_version": node.contract_version,
                "contract_sources": list(node.contract_sources),
                "outputs": list(node.outputs),
            }
            for node in sorted(nodes, key=lambda item: item.node_id)
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def build_default_operation_registry(
    dependencies: DependencyRegistry = DEFAULT_DAILY_DEPENDENCY_REGISTRY,
) -> OperationRegistry:
    grouped: dict[str, list[DependencyNode]] = defaultdict(list)
    for node in dependencies.nodes.values():
        grouped[node.operation_id].append(node)

    definitions: list[OperationDefinition] = []
    for operation_id, nodes in sorted(grouped.items()):
        outputs = tuple(
            dict.fromkeys(
                output
                for node in sorted(nodes, key=lambda item: item.ui_order)
                for output in node.outputs
            )
        )
        contract_paths = tuple(
            dict.fromkeys(
                path
                for node in nodes
                for path in node.contract_sources
            )
        )
        mode = _cache_mode(operation_id, nodes)
        definitions.append(
            OperationDefinition(
                operation_id=operation_id,
                entrypoint=_ENTRYPOINTS.get(
                    operation_id,
                    "quant.routine.operation_adapters:shadow_only",
                ),
                produces=tuple(
                    node.node_id
                    for node in sorted(nodes, key=lambda item: item.ui_order)
                ),
                execution_mode=ExecutionMode.THREAD,
                resources=_default_claim(operation_id),
                cache=CachePolicy(
                    mode=mode,
                    contract_version=_contract_version(operation_id, nodes),
                    output_paths=outputs,
                    contract_paths=contract_paths,
                ),
                retry=RetryPolicy(),
                parameters={
                    "migration_mode": (
                        "enabled"
                        if operation_id in _ENTRYPOINTS
                        else "shadow_only"
                    )
                },
            )
        )
    registry = OperationRegistry(definitions)
    registry.validate_against_dependencies(dependencies)
    return registry


DEFAULT_DAILY_OPERATION_REGISTRY = build_default_operation_registry()


__all__ = [
    "DEFAULT_DAILY_OPERATION_REGISTRY",
    "build_default_operation_registry",
]
