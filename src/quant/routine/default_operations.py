"""Production operation registry derived from the daily dependency graph."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from typing import Iterable, Mapping

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
    OperationBinding,
    OperationDefinition,
    ResourceClaim,
    RetryPolicy,
)
from quant.routine.operation_registry import OperationRegistry


CORE_OPERATION_INPUTS = {
    "refresh_strategy_signal_cache": ("data.market_daily", "source.market_daily_parquet"),
    "refresh_active_project_features": (
        "data.market_daily", "source.market_daily_parquet", "data.daily_basic", "feature.strategy_signals",
    ),
    "run_right_side_unified": (
        "data.market_daily", "source.market_daily_parquet",
        "feature.strategy_signals", "feature.project_daily",
    ),
    "run_left_side_unified": (
        "data.market_daily", "source.market_daily_parquet", "data.daily_basic",
        "feature.strategy_signals", "feature.project_daily",
    ),
    "refresh_chan_model_scores": (
        "data.market_daily", "source.market_daily_parquet", "data.daily_basic", "data.top_list",
    ),
}

# These are atomic ownership groups of existing executable routines, not
# independently cacheable nodes. The web composition supplies the real callback.
COMPOSED_OPERATION_GROUPS = {
    "refresh_reference_inputs": (
        "refresh_index_000300", "refresh_stock_basic", "refresh_financial_pit",
        "refresh_analyst_forecasts", "refresh_top_list", "refresh_market_regime_snapshot",
        "refresh_tradability_for_research", "refresh_long_research_external",
    ),
    "build_byd_daily_workspace": (
        "refresh_byd_intraday_when_due", "build_byd_daily_features",
        "fit_or_reuse_byd_runtime_model", "build_byd_daily_workspace",
    ),
    "build_convertible_bond_grid_workspace": (
        "refresh_convertible_bond_daily", "refresh_convertible_bond_reference",
        "build_convertible_bond_grid_features", "build_convertible_bond_grid_workspace",
    ),
    "build_convertible_bond_allotment_workspace": (
        "refresh_convertible_bond_allotment_events", "build_convertible_bond_allotment_workspace",
    ),
    "refresh_similar_pattern_analysis": (
        "read_similar_watchlist", "refresh_similar_reference_vectors_when_due",
        "build_similar_target_context", "score_similar_patterns", "refresh_similar_pattern_analysis",
    ),
    "build_selector_payload": ("build_selector_live_features", "build_selector_payload"),
}

_IMPLEMENTATION_PATHS = {
    "refresh_strategy_signal_cache": (
        "src/quant/research/strategy_signal_cache.py", "src/quant/research/b1_family_rules.py",
        "src/quant/research/z_skill_rules.py", "src/quant/research/rule_windows.py",
        "src/quant/research/rule_backtest_support.py",
    ),
    "refresh_active_project_features": (
        "scripts/research/refresh_b1_feature_cache.py", "scripts/research/train_b1_tushare_models.py",
        "configs/strategies/b1_selected.yaml",
    ),
    "run_right_side_unified": (
        "src/quant/routine/right_side_unified_shadow.py", "src/quant/routine/project_feature_cache.py",
        "src/quant/research/right_side_unified.py",
    ),
    "run_left_side_unified": (
        "src/quant/routine/project_feature_cache.py",
        "src/quant/features/candlestick_context.py",
    ),
    "refresh_chan_model_scores": (
        "scripts/research/refresh_chan_model_live_scores.py", "scripts/research/backtest_chan_daily.py",
        "scripts/research/train_chan_daily_models.py", "src/quant/features/market_sentiment.py",
    ),
    "refresh_long_factor_snapshot": (
        "src/quant/research/long_dividend_quality.py", "src/quant/research/tea_master_long.py",
        "src/quant/research/rule_backtest_support.py",
    ),
    "refresh_long_stock_pool_variants": (
        "src/quant/research/long_dividend_quality.py", "src/quant/research/tea_master_long.py",
        "src/quant/research/rule_backtest_support.py",
    ),
}

_CHAN_OUTPUTS = (
    "reports/chan_daily/chan_daily_candidates.parquet",
    "reports/chan_daily/model_filter/chan_model_scored_candidates.parquet",
    "reports/chan_daily/model_filter/live_refresh_manifest.json",
)


def _core_model_contracts(operation_id: str) -> tuple[str, ...]:
    from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
    from quant.application.selector_ranking import DEFAULT_SELECTOR_RANKING_CONFIG
    from quant.routine.paths import PROJECT_ROOT

    if operation_id == "run_right_side_unified":
        paths = DEFAULT_SELECTOR_RANKING_CONFIG.paths
        return tuple(path.relative_to(PROJECT_ROOT).as_posix() for path in (
            paths.artifact, paths.artifact_manifest, paths.promotion_approval,
        ))
    if operation_id == "run_left_side_unified":
        paths = DEFAULT_LEFT_SIDE_RANKING_CONFIG.paths
        return tuple(path.relative_to(PROJECT_ROOT).as_posix() for path in (
            paths.artifact, paths.artifact_manifest, paths.ranking_decision,
        ))
    if operation_id == "refresh_chan_model_scores":
        return tuple(f"models/research/chan_daily/{target}.joblib" for target in (
            "target_win10", "target_big10", "target_good",
        )) + ("reports/chan_daily/model_filter/chan_model_dataset.parquet",)
    return ()


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
    claim = _RESOURCE_PROFILES.get(
        operation_id,
        ResourceClaim(cpu_slots=1, io_slots=1, memory_mb=512),
    )
    return replace(claim, db_connections=1) if operation_id in CORE_OPERATION_INPUTS else claim


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
    *,
    bindings: Mapping[str, OperationBinding] | None = None,
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
        contract_paths = tuple(dict.fromkeys((
            *contract_paths, *_IMPLEMENTATION_PATHS.get(operation_id, ()),
            *_core_model_contracts(operation_id),
            *(("src/quant/routine/operation_adapters.py", "src/quant/routine/default_operations.py")
              if operation_id in CORE_OPERATION_INPUTS else ()),
        )))
        if operation_id == "refresh_chan_model_scores":
            outputs = _CHAN_OUTPUTS
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
                    optional_contract_paths=(
                        (".env",)
                        if operation_id in CORE_OPERATION_INPUTS else ()
                    ),
                    environment_keys=(
                        ("ROUTINE_PRODUCTION_FACTOR_SCHEMA", "PROJECT_FACTOR_COMPATIBILITY_MODE",
                         "ROUTINE_SIGNAL_FACTOR_MODE", "DAILY_FACTOR_ROOT",
                         "ROUTINE_DAILY_BASIC_MIN_MATCH_RATE")
                        if operation_id in CORE_OPERATION_INPUTS else ()
                    ),
                    track_python_imports=operation_id in CORE_OPERATION_INPUTS,
                ),
                retry=RetryPolicy(),
                production_ready=operation_id in _ENTRYPOINTS,
                input_ids=CORE_OPERATION_INPUTS.get(operation_id),
                parameters={
                    "migration_mode": (
                        "enabled"
                        if operation_id in _ENTRYPOINTS
                        else "shadow_only"
                    )
                },
            )
        )
    registry = OperationRegistry(definitions, bindings=bindings)
    registry.validate_against_dependencies(dependencies)
    return registry


DEFAULT_DAILY_OPERATION_REGISTRY = build_default_operation_registry()


__all__ = [
    "DEFAULT_DAILY_OPERATION_REGISTRY",
    "CORE_OPERATION_INPUTS",
    "COMPOSED_OPERATION_GROUPS",
    "build_default_operation_registry",
]
