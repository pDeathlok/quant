from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
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
from quant.routine.dag_executor import DailyDagExecutor, ResourceBudget
from quant.routine.default_operations import DEFAULT_DAILY_OPERATION_REGISTRY
from quant.routine.operation_contracts import (
    CacheMode,
    CachePolicy,
    ExecutionMode,
    OperationDefinition,
    OperationResult,
    ResourceClaim,
)
from quant.routine.operation_registry import OperationRegistry
from quant.routine.production_dag import production_output_node_ids


@pytest.fixture
def graph():
    def node(node_id, operation_id, layer, *inputs):
        return DependencyNode(
            node_id=node_id,
            operation_id=operation_id,
            layer=layer,
            owner="tests",
            lifecycle=Lifecycle.PRODUCTION,
            cadence=Cadence.TRADE_DAILY,
            inputs=tuple(DependencyEdge(value) for value in inputs),
            freshness=FreshnessPolicy(FreshnessMode.EXACT_TRADE_DATE),
            incremental=IncrementalPolicy(),
        )

    nodes = (
        node("data.market", "source_callback", Layer.DATA_SOURCE),
        node("feature.shared", "build_shared", Layer.FEATURE, "data.market"),
        node("feature.future", "new_model", Layer.FEATURE, "feature.shared"),
        node("score.future", "new_model", Layer.MODEL_SCORE, "feature.future"),
        node("product.adapter", "new_model", Layer.PRODUCT, "score.future"),
        node("product.page", "page_callback", Layer.PRODUCT, "product.adapter"),
        node("product.unrelated", "other_model", Layer.PRODUCT),
    )
    dependencies = DependencyRegistry(
        reversed(nodes),
        {"page": ("product.page",), "other": ("product.unrelated",)},
    )
    operations = OperationRegistry(
        OperationDefinition(
            operation_id=operation_id,
            entrypoint="quant.routine.operation_adapters:shadow_only",
            produces=tuple(node.node_id for node in nodes if node.operation_id == operation_id),
            execution_mode=ExecutionMode.THREAD,
            resources=ResourceClaim(),
            cache=CachePolicy(CacheMode.NONE, "test-v1"),
            input_ids=(),
            production_ready=not operation_id.endswith("callback"),
        )
        for operation_id in sorted({node.operation_id for node in nodes})
    )
    return dependencies, operations


def test_new_registered_ready_model_is_selected_in_topological_order(graph):
    dependencies, operations = graph
    selected = production_output_node_ids("page", dependencies=dependencies, operations=operations)
    assert selected == (
        "feature.shared", "feature.future", "score.future", "product.adapter",
    )
    assert isinstance(selected, tuple)
    assert len(selected) == len(set(selected))


def test_completed_operation_excludes_every_owned_node_and_accepts_generator(graph):
    dependencies, operations = graph
    selected = production_output_node_ids(
        "page", completed_operation_ids=iter(("new_model",)),
        dependencies=dependencies, operations=operations,
    )
    assert selected == ("feature.shared",)


def test_unrequired_ready_model_and_nonready_callbacks_are_not_selected(graph):
    dependencies, operations = graph
    selected = production_output_node_ids("page", dependencies=dependencies, operations=operations)
    assert set(selected).isdisjoint({"data.market", "product.page", "product.unrelated"})


@pytest.mark.parametrize("missing_owner", ["new_model", "source_callback", "page_callback"])
def test_missing_required_owner_is_rejected_before_completed_or_ready_filtering(graph, missing_owner):
    dependencies, operations = graph
    incomplete = OperationRegistry(
        value for key, value in operations.definitions.items() if key != missing_owner
    )
    with pytest.raises(ValueError, match="unregistered operation " + missing_owner):
        production_output_node_ids(
            "page", completed_operation_ids=(missing_owner,),
            dependencies=dependencies, operations=incomplete,
        )


def test_missing_required_node_in_owner_outputs_is_rejected(graph):
    dependencies, operations = graph
    incomplete = OperationRegistry(
        replace(value, produces=("feature.future", "score.future")) if key == "new_model" else value
        for key, value in operations.definitions.items()
    )
    with pytest.raises(ValueError, match="does not declare node product.adapter"):
        production_output_node_ids("page", dependencies=dependencies, operations=incomplete)


def test_missing_unrequired_owner_does_not_block_selected_scope(graph):
    dependencies, operations = graph
    partial = OperationRegistry(
        value for key, value in operations.definitions.items() if key != "other_model"
    )
    assert production_output_node_ids("page", dependencies=dependencies, operations=partial)


def test_selected_atomic_output_runs_once_without_repeating_feature_stage(graph, tmp_path):
    dependencies, operations = graph
    selected = production_output_node_ids(
        "page", completed_operation_ids=("build_shared",),
        dependencies=dependencies, operations=operations,
    )
    calls = Counter()

    def handler(operation_id):
        def run(_context):
            calls[operation_id] += 1
            return OperationResult(
                status="success",
                node_results={
                    node: {"status": "success"}
                    for node in operations.definitions[operation_id].produces
                },
            )
        return run

    report = DailyDagExecutor(
        dependencies, operations, project_root=tmp_path,
        budget=ResourceBudget(cpu_slots=1),
        handlers={key: handler(key) for key in operations.definitions},
    ).execute(target_trade_date="2026-11-13", scope="page", node_ids=selected)
    assert report["status"] == "success"
    assert calls == {"new_model": 1}
    assert set(report["node_results"]) == set(selected)


def test_empty_output_selection_never_expands_to_full_graph(graph, tmp_path):
    dependencies, operations = graph
    selected = production_output_node_ids(
        "page", completed_operation_ids=("build_shared", "new_model"),
        dependencies=dependencies, operations=operations,
    )
    assert selected == ()
    calls = []

    def handler(_context):
        calls.append(True)
        return OperationResult(status="failed", node_results={}, error="unexpected execution")

    report = DailyDagExecutor(
        dependencies, operations, project_root=tmp_path,
        budget=ResourceBudget(cpu_slots=1),
        handlers={key: handler for key in operations.definitions},
    ).execute(target_trade_date="2026-11-13", scope="page", node_ids=selected)
    assert calls == []
    assert report["status"] == "success"
    assert report["operations"] == {}


@pytest.mark.parametrize("scope", ["all", "short"])
def test_default_selection_covers_all_required_ready_operations(scope):
    dependencies = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    operations = DEFAULT_DAILY_OPERATION_REGISTRY
    completed = {"refresh_strategy_signal_cache", "refresh_active_project_features"}
    selected = production_output_node_ids(scope, completed_operation_ids=completed)
    required = dependencies.required_node_ids(scope)
    expected = tuple(
        node for node in required
        if operations.definitions[dependencies.nodes[node].operation_id].production_ready
        and dependencies.nodes[node].operation_id not in completed
    )
    assert selected == expected
    if scope == "short":
        assert not {"feature.chan_live", "score.chan"}.intersection(selected)
    else:
        assert {"feature.chan_live", "score.chan"}.issubset(selected)


def test_unknown_scope_fails_closed(graph):
    dependencies, operations = graph
    with pytest.raises(ValueError, match="unknown daily dependency scope"):
        production_output_node_ids("unknown", dependencies=dependencies, operations=operations)
