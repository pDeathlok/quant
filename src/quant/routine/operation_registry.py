"""Validated registry of executable daily refresh operations."""

from __future__ import annotations

from collections.abc import Iterable

from quant.application.daily_dependencies import DependencyRegistry
from quant.routine.operation_contracts import OperationDefinition


class OperationRegistry:
    def __init__(self, definitions: Iterable[OperationDefinition]) -> None:
        materialized = tuple(definitions)
        self.definitions = {
            definition.operation_id: definition for definition in materialized
        }
        if len(self.definitions) != len(materialized):
            raise ValueError("operation registry contains duplicate operation ids")
        produced: dict[str, str] = {}
        for definition in materialized:
            for node_id in definition.produces:
                previous = produced.get(node_id)
                if previous is not None:
                    raise ValueError(
                        f"node {node_id} is produced by both {previous} and "
                        f"{definition.operation_id}"
                    )
                produced[node_id] = definition.operation_id
        self.operation_by_node = produced

    def validate_against_dependencies(
        self,
        dependencies: DependencyRegistry,
        *,
        node_ids: Iterable[str] | None = None,
    ) -> None:
        selected = set(node_ids or dependencies.nodes)
        errors: list[str] = []
        for node_id in sorted(selected):
            node = dependencies.nodes[node_id]
            operation = self.definitions.get(node.operation_id)
            if operation is None:
                errors.append(
                    f"node {node_id} references unregistered operation {node.operation_id}"
                )
                continue
            if node_id not in operation.produces:
                errors.append(
                    f"operation {node.operation_id} does not declare node {node_id}"
                )
        for operation in self.definitions.values():
            unknown = sorted(set(operation.produces) - set(dependencies.nodes))
            if unknown:
                errors.append(
                    f"operation {operation.operation_id} produces unknown nodes {unknown}"
                )
            mismatched = sorted(
                node_id
                for node_id in operation.produces
                if node_id in dependencies.nodes
                and dependencies.nodes[node_id].operation_id
                != operation.operation_id
            )
            if mismatched:
                errors.append(
                    f"operation {operation.operation_id} mismatches dependency nodes "
                    f"{mismatched}"
                )
        if errors:
            raise ValueError("invalid operation registry: " + "; ".join(errors))

    def required_operations(
        self,
        dependencies: DependencyRegistry,
        node_ids: Iterable[str],
    ) -> tuple[OperationDefinition, ...]:
        selected = set(node_ids)
        ordered_ids: list[str] = []
        for node_id in dependencies.topological_order(selected):
            operation_id = dependencies.nodes[node_id].operation_id
            if operation_id not in ordered_ids:
                ordered_ids.append(operation_id)
        return tuple(self.definitions[operation_id] for operation_id in ordered_ids)


__all__ = ["OperationRegistry"]
