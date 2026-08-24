"""Factor calculator DAG, bounded execution, and shared worker budgeting."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import asdict, dataclass
import os
from typing import Any, Callable, Iterable, Iterator, Mapping

from quant.features.factor_governance import load_factor_governance_config
from quant.features.factor_registry import FACTOR_REGISTRY, FactorDefinition


@dataclass(frozen=True)
class FactorCalculatorDefinition:
    calculator_id: str
    entrypoint: str
    produces: tuple[str, ...]
    dependencies: tuple[str, ...]
    factor_dependencies: tuple[str, ...]
    partition_axis: str
    executor: str
    default_workers: int
    max_workers: int
    max_pending_multiplier: int
    materialization: str
    routine_node: str
    enabled: bool = True


_BASE_CALCULATORS: dict[str, dict[str, Any]] = {
    "project_daily": {
        "entrypoint": "quant.features.project_factor_layer.calculate_project_factor_frame",
        "dependencies": (), "factor_dependencies": (), "partition_axis": "symbol",
        "executor": "processes", "default_workers": 8, "max_workers": 8,
        "max_pending_multiplier": 2, "materialization": "daily_snapshot",
        "routine_node": "feature.project_daily",
    },
    "project_daily_candidate": {
        "entrypoint": "quant.features.project_factor_layer.calculate_legacy_market_factors",
        "dependencies": (), "factor_dependencies": (), "partition_axis": "symbol",
        "executor": "processes", "default_workers": 4, "max_workers": 8,
        "max_pending_multiplier": 2, "materialization": "on_demand", "routine_node": "",
    },
    "right_side_rule": {
        "entrypoint": "quant.research.right_side_unified_features.compute_right_side_rule_features",
        "dependencies": ("project_daily",),
        "factor_dependencies": ("pct_chg", "amplitude_1", "volume_relative_5d", "volume_relative_20d", "kdj_d_j"),
        "partition_axis": "symbol", "executor": "processes", "default_workers": 6,
        "max_workers": 6, "max_pending_multiplier": 2, "materialization": "daily_snapshot",
        "routine_node": "feature.right_side_unified",
    },
    "right_side_identity": {
        "entrypoint": "canonical_right_side_signal_cache", "dependencies": ("right_side_rule",),
        "factor_dependencies": (), "partition_axis": "symbol", "executor": "serial",
        "default_workers": 1, "max_workers": 1, "max_pending_multiplier": 1,
        "materialization": "signal_cache", "routine_node": "score.right_side_unified",
    },
    "left_side_rule": {
        "entrypoint": "quant.research.left_side_unified_features.compute_left_side_rule_features",
        "dependencies": ("project_daily",), "factor_dependencies": ("pct_chg", "kdj_d_j", "bbi"),
        "partition_axis": "symbol", "executor": "processes", "default_workers": 6,
        "max_workers": 8, "max_pending_multiplier": 2, "materialization": "daily_snapshot",
        "routine_node": "feature.left_side_unified",
    },
    "left_side_identity": {
        "entrypoint": "canonical_left_side_signal_cache", "dependencies": ("left_side_rule",),
        "factor_dependencies": (), "partition_axis": "symbol", "executor": "serial",
        "default_workers": 1, "max_workers": 1, "max_pending_multiplier": 1,
        "materialization": "signal_cache", "routine_node": "score.left_side_unified",
    },
    "selector_live": {
        "entrypoint": "quant.webapp.services._selector_live_feature_rows", "dependencies": (),
        "factor_dependencies": (), "partition_axis": "cross_section", "executor": "serial",
        "default_workers": 1, "max_workers": 1, "max_pending_multiplier": 1,
        "materialization": "consumer_local", "routine_node": "score.selector",
    },
    "chan_live": {
        "entrypoint": "scripts.research.refresh_chan_model_live_scores", "dependencies": (),
        "factor_dependencies": (), "partition_axis": "symbol", "executor": "processes",
        "default_workers": 4, "max_workers": 4, "max_pending_multiplier": 2,
        "materialization": "daily_snapshot", "routine_node": "feature.chan_live",
    },
    "long_snapshot": {
        "entrypoint": "quant.webapp.services._tea_master_live_scores", "dependencies": (),
        "factor_dependencies": (), "partition_axis": "cross_section", "executor": "serial",
        "default_workers": 1, "max_workers": 1, "max_pending_multiplier": 1,
        "materialization": "weekly_snapshot", "routine_node": "feature.long_snapshot",
    },
    "long_research": {
        "entrypoint": "quant.features.long_weekly_factors.build_long_weekly_factor_frame",
        "dependencies": (), "factor_dependencies": (), "partition_axis": "cross_section",
        "executor": "serial", "default_workers": 1, "max_workers": 1,
        "max_pending_multiplier": 1, "materialization": "on_demand", "routine_node": "",
    },
    "long_external": {
        "entrypoint": "quant.features.long_external_factors.build_weekly_external_factor_cache",
        "dependencies": (), "factor_dependencies": (), "partition_axis": "dataset",
        "executor": "threads", "default_workers": 4, "max_workers": 6,
        "max_pending_multiplier": 2, "materialization": "on_demand", "routine_node": "",
    },
}


def build_factor_calculator_registry(
    factors: tuple[FactorDefinition, ...] = FACTOR_REGISTRY,
) -> tuple[FactorCalculatorDefinition, ...]:
    overrides = load_factor_governance_config()["calculators"]
    unknown = sorted(set(overrides) - set(_BASE_CALCULATORS))
    if unknown:
        raise ValueError(f"unknown factor calculator overrides: {unknown}")
    produced: dict[str, list[str]] = {name: [] for name in _BASE_CALCULATORS}
    for factor in factors:
        if factor.calculator_id not in produced:
            raise ValueError(f"factor {factor.name} has unknown calculator_id {factor.calculator_id}")
        produced[factor.calculator_id].append(factor.name)
    calculators: list[FactorCalculatorDefinition] = []
    for calculator_id, base in _BASE_CALCULATORS.items():
        settings = {**base, **overrides.get(calculator_id, {})}
        calculators.append(
            FactorCalculatorDefinition(
                calculator_id=calculator_id,
                entrypoint=str(settings["entrypoint"]),
                produces=tuple(produced[calculator_id]),
                dependencies=tuple(settings["dependencies"]),
                factor_dependencies=tuple(settings["factor_dependencies"]),
                partition_axis=str(settings["partition_axis"]),
                executor=str(settings["executor"]),
                default_workers=max(1, int(settings["default_workers"])),
                max_workers=max(1, int(settings["max_workers"])),
                max_pending_multiplier=max(1, int(settings["max_pending_multiplier"])),
                materialization=str(settings["materialization"]),
                routine_node=str(settings["routine_node"]),
                enabled=bool(settings.get("enabled", True)),
            )
        )
    return tuple(calculators)


FACTOR_CALCULATORS = build_factor_calculator_registry()


def validate_factor_execution_registry() -> None:
    calculators = {item.calculator_id: item for item in FACTOR_CALCULATORS}
    factors = {item.name: item for item in FACTOR_REGISTRY}
    for calculator in FACTOR_CALCULATORS:
        missing_calculators = sorted(set(calculator.dependencies) - set(calculators))
        if missing_calculators:
            raise ValueError(f"calculator {calculator.calculator_id} has missing dependencies: {missing_calculators}")
        missing_factors = sorted(set(calculator.factor_dependencies) - set(factors))
        if missing_factors:
            raise ValueError(f"calculator {calculator.calculator_id} has missing factor dependencies: {missing_factors}")
        if calculator.executor not in {"serial", "threads", "processes"}:
            raise ValueError(f"calculator {calculator.calculator_id} has invalid executor {calculator.executor}")
    for factor in FACTOR_REGISTRY:
        calculator = calculators[factor.calculator_id]
        if factor.refresh_cadence == "trade_daily" and not calculator.enabled:
            raise ValueError(f"daily factor {factor.name} belongs to disabled calculator {factor.calculator_id}")
        if factor.refresh_cadence == "trade_daily" and not calculator.routine_node:
            raise ValueError(f"daily factor {factor.name} has no routine DAG node via {factor.calculator_id}")
    build_factor_execution_plan()


def build_factor_execution_plan(
    required_factors: Iterable[str] | None = None,
) -> tuple[FactorCalculatorDefinition, ...]:
    calculators = {item.calculator_id: item for item in FACTOR_CALCULATORS}
    factors = {item.name: item for item in FACTOR_REGISTRY}
    selected = tuple(
        item.name for item in FACTOR_REGISTRY if item.refresh_cadence == "trade_daily"
    ) if required_factors is None else tuple(dict.fromkeys(str(name) for name in required_factors))
    missing = sorted(set(selected) - set(factors))
    if missing:
        raise KeyError(f"unregistered factors requested: {missing}")
    roots = {factors[name].calculator_id for name in selected}
    ordered: list[FactorCalculatorDefinition] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(calculator_id: str) -> None:
        if calculator_id in visited:
            return
        if calculator_id in visiting:
            raise ValueError(f"factor calculator dependency cycle at {calculator_id}")
        visiting.add(calculator_id)
        calculator = calculators[calculator_id]
        if not calculator.enabled:
            raise ValueError(f"required factor calculator is disabled: {calculator_id}")
        for dependency in calculator.dependencies:
            visit(dependency)
        visiting.remove(calculator_id)
        visited.add(calculator_id)
        ordered.append(calculator)

    for calculator_id in sorted(roots):
        visit(calculator_id)
    return tuple(ordered)


def factor_execution_plan_payload() -> dict[str, Any]:
    plan = build_factor_execution_plan()
    return {
        "calculators": [asdict(item) for item in plan],
        "calculator_count": len(plan),
        "factor_count": sum(item.refresh_cadence == "trade_daily" for item in FACTOR_REGISTRY),
    }


def calculator_execution_settings(calculator_id: str) -> FactorCalculatorDefinition:
    for calculator in FACTOR_CALCULATORS:
        if calculator.calculator_id == calculator_id:
            return calculator
    raise KeyError(f"unknown factor calculator: {calculator_id}")


def configured_worker_budget(default: int | None = None) -> int:
    configured = int(load_factor_governance_config()["execution"].get("global_cpu_worker_budget") or 0)
    fallback = int(default or os.cpu_count() or 4)
    return max(1, int(os.getenv("ROUTINE_CPU_WORKER_BUDGET", configured or fallback)))


def allocate_worker_budget(
    requested: Mapping[str, int],
    *,
    total_budget: int | None = None,
) -> dict[str, int]:
    """Allocate a deterministic shared budget without oversubscribing a phase."""

    caps = {str(name): max(1, int(value)) for name, value in requested.items()}
    if not caps:
        return {}
    budget = max(len(caps), int(total_budget or configured_worker_budget()))
    allocation = {name: 1 for name in sorted(caps)}
    remaining = budget - len(allocation)
    while remaining > 0:
        progressed = False
        for name in sorted(caps):
            if allocation[name] >= caps[name]:
                continue
            allocation[name] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return allocation


def bounded_executor_results(
    executor: Any,
    function: Callable[..., Any],
    tasks: Iterable[tuple[Any, ...]],
    *,
    max_pending: int,
) -> Iterator[Any]:
    """Yield completed results while keeping only a bounded number of futures."""

    iterator = iter(tasks)
    pending: set[Any] = set()

    def submit_one() -> bool:
        try:
            arguments = next(iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(function, *arguments))
        return True

    for _ in range(max(1, int(max_pending))):
        if not submit_one():
            break
    while pending:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            pending.remove(future)
            yield future.result()
            submit_one()
