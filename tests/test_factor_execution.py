from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from quant.features.factor_execution import (
    FACTOR_CALCULATORS,
    allocate_worker_budget,
    bounded_executor_results,
    build_factor_execution_plan,
    validate_factor_execution_registry,
)


def test_factor_calculator_registry_covers_and_orders_daily_factors() -> None:
    validate_factor_execution_registry()
    plan = build_factor_execution_plan(["rs_b1_support_ok"])
    ids = [item.calculator_id for item in plan]

    assert ids.index("project_daily") < ids.index("right_side_rule")
    assert all(item.produces for item in FACTOR_CALCULATORS)


def test_worker_budget_never_oversubscribes_a_concurrent_phase() -> None:
    allocation = allocate_worker_budget(
        {"model_score": 4, "chan": 4, "right_side": 6},
        total_budget=8,
    )

    assert sum(allocation.values()) == 8
    assert allocation["model_score"] <= 4
    assert allocation["chan"] <= 4
    assert allocation["right_side"] <= 6


def test_bounded_executor_processes_every_task() -> None:
    def add(left: int, right: int) -> int:
        return left + right

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            bounded_executor_results(
                executor,
                add,
                ((value, 1) for value in range(20)),
                max_pending=3,
            )
        )

    assert sorted(results) == list(range(1, 21))
