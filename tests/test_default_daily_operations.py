from __future__ import annotations

from dataclasses import replace

import pytest

from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
)
from quant.routine.dag_executor import ResourceBudget
from quant.routine.default_operations import DEFAULT_DAILY_OPERATION_REGISTRY
from quant.routine.operation_adapters import (
    refresh_active_project_features, refresh_chan_model_scores,
    refresh_strategy_signals, result_from_payload,
)
from quant.routine.operation_contracts import (
    CacheMode, CachePolicy, InputSnapshot, NodeChanges, OperationBinding,
    OperationContext, OperationResult, ResourceClaim,
)
from quant.routine.production_dag import (
    build_daily_dag_plan, execute_daily_operations, snapshots_from_results,
    validate_daily_operation_closure,
)


def test_default_operation_registry_covers_dependency_graph() -> None:
    DEFAULT_DAILY_OPERATION_REGISTRY.validate_against_dependencies(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY
    )


def test_short_plan_collapses_shared_model_and_selector_operations(tmp_path) -> None:
    plan = build_daily_dag_plan(
        "2026-08-31",
        "short",
        project_root=tmp_path,
        budget=ResourceBudget(cpu_slots=10, io_slots=4, memory_mb=8192),
    )
    operations = {
        item["operation_id"]: item for item in plan["operations"]
    }

    assert plan["status"] == "success"
    assert plan["collapsed_node_count"] >= 4
    assert set(operations["run_left_side_unified"]["produces"]) == {
        "feature.left_side_unified",
        "score.left_side_unified",
        "product.left_side_unified_adapter",
    }
    assert set(operations["run_right_side_unified"]["produces"]) == {
        "feature.right_side_unified",
        "score.right_side_unified",
        "product.right_side_unified_adapter",
    }
    assert set(operations["build_selector_payload"]["produces"]) == {
        "score.selector",
        "product.selector_core",
        "product.selector_extended",
    }
    assert "refresh_active_project_features" in operations[
        "run_left_side_unified"
    ]["depends_on"]
    assert operations["refresh_strategy_signal_cache"]["cache_mode"] == (
        "append_state"
    )
    assert operations["refresh_strategy_signal_cache"]["resources"][
        "cpu_slots"
    ] == 8


def test_operation_plan_rejects_resource_profile_above_host_budget(
    tmp_path,
) -> None:
    try:
        build_daily_dag_plan(
            "2026-08-31",
            "short",
            project_root=tmp_path,
            budget=ResourceBudget(cpu_slots=4, io_slots=4, memory_mb=8192),
        )
    except ValueError as exc:
        assert "refresh_strategy_signal_cache requests 8 CPU" in str(exc)
    else:
        raise AssertionError("oversubscribed production profile must fail closed")


def test_unbound_web_product_fails_before_any_execution(tmp_path):
    with pytest.raises(ValueError, match="build_selector_payload is shadow-only"):
        execute_daily_operations(
            "2026-09-04", "short", ("product.selector_core",), project_root=tmp_path,
        )


def test_web_callable_binding_executes_all_owned_nodes_and_carries_stage_inputs(tmp_path):
    definition = DEFAULT_DAILY_OPERATION_REGISTRY.definitions["build_selector_payload"]
    for relative in definition.cache.contract_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test contract")
    required = {
        edge.upstream for node_id in definition.produces
        for edge in DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes[node_id].inputs
        if edge.upstream not in definition.produces
    }
    observed = []

    def web_owned(context):
        observed.append(context)
        return OperationResult(
            status="success", node_results={node: {"status": "success"} for node in definition.produces},
            output_fingerprints={node: f"{node}:content:1" for node in definition.produces},
            node_changes={node: NodeChanges() for node in definition.produces},
        )

    binding = OperationBinding(
        handler=web_owned, input_ids=tuple(sorted(required)),
        cache=CachePolicy(CacheMode.NONE, "selector-code-schema-model-v1"),
        resources=ResourceClaim(), parameters={"variant": "both"},
    )
    report = execute_daily_operations(
        "2026-09-04", "short", ("product.selector_core",), project_root=tmp_path,
        bindings={"build_selector_payload": binding}, require_identity=True,
        input_snapshots={node: InputSnapshot(f"{node}:canonical:1") for node in required},
    )
    assert report["status"] == "success"
    assert len(observed) == 1
    assert observed[0].parameters == {"variant": "both"}
    assert set(report["node_results"]) == set(definition.produces)
    snapshots = snapshots_from_results(report["operations"].values())
    assert set(snapshots) == set(definition.produces)
    assert all(value.fingerprint.endswith(":content:1") for value in snapshots.values())


def test_existing_manual_cache_payload_is_preserved_without_central_identity(tmp_path):
    payload = {"status": "success", "checkpoint_reused": True, "refresh_reason": "same-input"}
    context = OperationContext("2026-09-04", "short", 2, {}, dirty_keys=("A",))
    result = result_from_payload(payload, ("product.test",), context)
    assert result.node_results["product.test"]["checkpoint_reused"] is True
    assert result.node_results["product.test"]["refresh_reason"] == "same-input"
    assert result.changed_keys == ("A",)
    assert result.node_changes["product.test"].full_rebuild is True
    assert result.output_fingerprints == {}
    with pytest.raises(ValueError, match="fingerprint"):
        snapshots_from_results((result,))


@pytest.mark.parametrize("adapter", [
    refresh_active_project_features, refresh_chan_model_scores, refresh_strategy_signals,
])
def test_strict_native_adapter_passes_full_rebuild_and_worker_budget(adapter, monkeypatch, tmp_path):
    import json
    import subprocess

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "status": "success", "processed_through_date": "2026-09-04",
            "source_latest_trade_date": "2026-09-04", "end": "2026-09-04",
        }), "")

    monkeypatch.setattr("quant.routine.operation_adapters.subprocess.run", run)
    context = OperationContext(
        "2026-09-04", "short", 2, {}, full_rebuild=True, identity_required=True, project_root=tmp_path,
    )
    assert adapter(context).status == "success"
    command, kwargs = calls[0]
    assert "19900101" in command
    assert "2" in command
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert kwargs["cwd"] == tmp_path
    if adapter is refresh_strategy_signals:
        assert "quant.research.strategy_signal_cache" in command
        assert "--force-refresh" in command
        assert command[command.index("--factor-mode") + 1] == "legacy"
    if adapter is refresh_chan_model_scores:
        assert "--rebuild-candidates" in command
        assert command[command.index("--top-list-dir") + 1] == str(tmp_path / "data/raw/top_list")


def test_disabled_and_noncallable_operations_are_rejected():
    definition = DEFAULT_DAILY_OPERATION_REGISTRY.definitions["refresh_chan_model_scores"]
    with pytest.raises(ValueError, match="disabled"):
        DEFAULT_DAILY_OPERATION_REGISTRY.validate_executable((replace(definition, enabled=False),), {})
    with pytest.raises(ValueError, match="not callable"):
        DEFAULT_DAILY_OPERATION_REGISTRY.validate_executable((definition,), {definition.operation_id: 42})


def test_plan_exposes_unmigrated_and_unaudited_operations(tmp_path):
    plan = build_daily_dag_plan("2026-09-04", "short", project_root=tmp_path,
                                budget=ResourceBudget(cpu_slots=10, memory_mb=8192))
    operations = {item["operation_id"]: item for item in plan["operations"]}
    assert operations["build_selector_payload"]["executable"] is False
    assert operations["refresh_strategy_signal_cache"]["executable"] is True
    assert operations["refresh_strategy_signal_cache"]["identity_declared"] is True


def test_active_production_closure_manifest_rejects_missing_bindings(tmp_path):
    with pytest.raises(ValueError, match="shadow-only"):
        validate_daily_operation_closure("2026-09-04", "short", project_root=tmp_path)


def test_active_production_closure_manifest_accepts_complete_callable_bindings(tmp_path):
    nodes = DEFAULT_DAILY_DEPENDENCY_REGISTRY.required_node_ids("short")
    definitions = DEFAULT_DAILY_OPERATION_REGISTRY.required_operations(DEFAULT_DAILY_DEPENDENCY_REGISTRY, nodes)
    bindings = {
        definition.operation_id: OperationBinding(
            handler=lambda context: pytest.fail("manifest must not execute handlers"),
            input_ids=tuple(sorted({
                edge.upstream for node in definition.produces
                for edge in DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes[node].inputs
                if edge.upstream not in definition.produces
            })),
            cache=CachePolicy(CacheMode.NONE, "audited-code-model-schema-v1"),
            resources=ResourceClaim(),
        )
        for definition in definitions
    }
    for definition in definitions:
        if not bindings[definition.operation_id].input_ids:
            bindings[definition.operation_id] = replace(
                bindings[definition.operation_id], input_ids=(f"source:{definition.operation_id}",),
            )
    manifest = validate_daily_operation_closure(
        "2026-09-04", "short", bindings=bindings, project_root=tmp_path,
    )
    assert manifest["mode"] == "production"
    assert manifest["operation_count"] == len(definitions)
    assert all(item["executable"] and item["identity_declared"] for item in manifest["operations"])
    bindings.pop("build_selector_payload")
    with pytest.raises(ValueError, match="build_selector_payload is shadow-only"):
        validate_daily_operation_closure("2026-09-04", "short", bindings=bindings, project_root=tmp_path)
