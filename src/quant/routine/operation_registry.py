"""Validated registry of executable daily refresh operations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
import importlib

from quant.application.daily_dependencies import DependencyRegistry
from quant.routine.operation_contracts import OperationBinding, OperationDefinition, OperationHandler


class OperationRegistry:
    def __init__(
        self,
        definitions: Iterable[OperationDefinition],
        *,
        bindings: Mapping[str, OperationBinding] | None = None,
    ) -> None:
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
        self.handlers: dict[str, OperationHandler] = {}
        for operation_id, binding in (bindings or {}).items():
            if operation_id not in self.definitions:
                raise ValueError(f"binding references unknown operation {operation_id}")
            original = self.definitions[operation_id]
            self.definitions[operation_id] = replace(
                original,
                input_ids=binding.input_ids,
                cache=replace(
                    binding.cache,
                    contract_paths=tuple(dict.fromkeys((*original.cache.contract_paths, *binding.cache.contract_paths))),
                    optional_contract_paths=tuple(dict.fromkeys((*original.cache.optional_contract_paths, *binding.cache.optional_contract_paths))),
                    environment_keys=tuple(dict.fromkeys((*original.cache.environment_keys, *binding.cache.environment_keys))),
                    track_python_imports=original.cache.track_python_imports or binding.cache.track_python_imports,
                ),
                resources=binding.resources or original.resources,
                parameters=dict(binding.parameters),
                production_ready=True,
            )
            self.handlers[operation_id] = binding.handler

    def validate_against_dependencies(
        self,
        dependencies: DependencyRegistry,
        *,
        node_ids: Iterable[str] | None = None,
    ) -> None:
        selected = set(dependencies.nodes if node_ids is None else node_ids)
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

    def validate_executable(
        self,
        definitions: Iterable[OperationDefinition],
        handlers: Mapping[str, OperationHandler],
        *,
        require_identity: bool = False,
    ) -> None:
        errors = []
        for definition in definitions:
            operation_id = definition.operation_id
            if not definition.enabled:
                errors.append(f"{operation_id} is disabled")
            handler = handlers.get(operation_id)
            if handler is None and (
                not definition.production_ready
                or definition.entrypoint.endswith(":shadow_only")
            ):
                errors.append(f"{operation_id} is shadow-only; supply an OperationBinding")
                continue
            if handler is None:
                try:
                    module, function = definition.entrypoint.split(":", 1)
                    handler = getattr(importlib.import_module(module), function)
                except (ImportError, AttributeError) as exc:
                    errors.append(f"{operation_id} cannot load handler: {exc}")
            if not callable(handler):
                errors.append(f"{operation_id} handler is not callable")
            if require_identity and definition.input_ids is None:
                errors.append(f"{operation_id} has incomplete input/contract identity; supply an audited binding")
        if errors:
            raise ValueError("invalid executable operations: " + "; ".join(errors))

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
