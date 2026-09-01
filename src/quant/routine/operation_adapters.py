"""Stable adapters between the DAG contract and existing production routines."""

from __future__ import annotations

from quant.routine.operation_contracts import OperationContext, OperationResult


def _result(
    payload: dict,
    node_ids: tuple[str, ...],
    context: OperationContext,
) -> OperationResult:
    payload = {**payload, "granted_workers": context.granted_workers}
    if payload.get("status") != "success":
        return OperationResult(
            status="failed",
            node_results={},
            error_category="operation",
            error=str(
                payload.get("stderr_tail")
                or payload.get("error")
                or payload.get("reason")
                or "operation failed"
            ),
        )
    return OperationResult(
        status="success",
        node_results={node_id: payload for node_id in node_ids},
        metrics={"granted_workers": context.granted_workers},
    )


def refresh_strategy_signals(context: OperationContext) -> OperationResult:
    from quant.routine.pipeline import refresh_strategy_signal_cache

    payload = refresh_strategy_signal_cache(workers=context.granted_workers)
    return _result(payload, ("feature.strategy_signals",), context)


def refresh_active_project_features(
    context: OperationContext,
) -> OperationResult:
    from quant.routine.pipeline import build_features

    start_date = min(context.dirty_partitions) if context.dirty_partitions else None
    payload = build_features(
        incremental_start_date=start_date,
        workers=context.granted_workers,
    )
    return _result(payload, ("feature.project_daily",), context)


def run_right_side_unified(context: OperationContext) -> OperationResult:
    from quant.routine.right_side_unified_production import (
        run_right_side_unified_production,
    )

    payload = run_right_side_unified_production(
        context.target_trade_date,
        factor_workers=context.granted_workers,
    )
    return _result(
        payload,
        (
            "feature.right_side_unified",
            "score.right_side_unified",
            "product.right_side_unified_adapter",
        ),
        context,
    )


def run_left_side_unified(context: OperationContext) -> OperationResult:
    from quant.routine.left_side_unified_production import run_left_side_production

    payload = run_left_side_production(
        context.target_trade_date,
        factor_workers=context.granted_workers,
    )
    return _result(
        payload,
        (
            "feature.left_side_unified",
            "score.left_side_unified",
            "product.left_side_unified_adapter",
        ),
        context,
    )


def refresh_chan_model_scores(context: OperationContext) -> OperationResult:
    from quant.routine.pipeline import refresh_chan_model_scores as refresh

    payload = refresh(progress_callback=None, workers=context.granted_workers)
    return _result(payload, ("feature.chan_live", "score.chan"), context)


def shadow_only(context: OperationContext) -> OperationResult:
    raise RuntimeError(
        "operation has no cutover adapter; run with ROUTINE_DAG_EXECUTOR=shadow"
    )


__all__ = [
    "refresh_active_project_features",
    "refresh_chan_model_scores",
    "refresh_strategy_signals",
    "run_left_side_unified",
    "run_right_side_unified",
    "shadow_only",
]
