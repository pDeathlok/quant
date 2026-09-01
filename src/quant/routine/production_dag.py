"""Shadow planning facade for the production daily refresh DAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
)
from quant.routine.dag_executor import DailyDagExecutor, ResourceBudget
from quant.routine.default_operations import DEFAULT_DAILY_OPERATION_REGISTRY
from quant.routine.paths import PROJECT_ROOT


def build_daily_dag_plan(
    target_trade_date: str,
    scope: str,
    *,
    project_root: Path = PROJECT_ROOT,
    budget: ResourceBudget | None = None,
) -> dict[str, Any]:
    executor = DailyDagExecutor(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        DEFAULT_DAILY_OPERATION_REGISTRY,
        project_root=project_root,
        budget=budget,
    )
    return executor.plan(
        target_trade_date=target_trade_date,
        scope=scope,
    )


def execute_daily_operations(
    target_trade_date: str,
    scope: str,
    node_ids: Iterable[str],
    *,
    project_root: Path = PROJECT_ROOT,
    budget: ResourceBudget | None = None,
    dirty_partitions: Iterable[str] = (),
    dirty_keys: Iterable[str] = (),
    progress_callback: (
        Callable[[str, str, Mapping[str, Any]], None] | None
    ) = None,
) -> dict[str, Any]:
    executor = DailyDagExecutor(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        DEFAULT_DAILY_OPERATION_REGISTRY,
        project_root=project_root,
        budget=budget,
        progress_callback=progress_callback,
    )
    return executor.execute(
        target_trade_date=target_trade_date,
        scope=scope,
        node_ids=node_ids,
        dirty_partitions=dirty_partitions,
        dirty_keys=dirty_keys,
    )


__all__ = ["build_daily_dag_plan", "execute_daily_operations"]
