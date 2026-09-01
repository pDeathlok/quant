from __future__ import annotations

from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
)
from quant.routine.dag_executor import ResourceBudget
from quant.routine.default_operations import DEFAULT_DAILY_OPERATION_REGISTRY
from quant.routine.production_dag import build_daily_dag_plan


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
