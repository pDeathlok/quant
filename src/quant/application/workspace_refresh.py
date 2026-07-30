from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


WorkspacePayload = dict[str, Any]
WorkspaceOperation = Callable[..., WorkspacePayload]


@dataclass(frozen=True)
class WorkspaceRefreshOperations:
    """Workspace use cases required by the daily refresh coordinator."""

    latest_signal_date: Callable[[], str | None]
    refresh_chan: WorkspaceOperation
    refresh_long: WorkspaceOperation
    refresh_convertible_bonds: WorkspaceOperation
    refresh_allotments: WorkspaceOperation
    refresh_byd: WorkspaceOperation
    refresh_similar_patterns: Callable[[], WorkspacePayload]


def refresh_daily_workspaces(
    operations: WorkspaceRefreshOperations,
    *,
    max_workers: int | None = None,
) -> dict[str, WorkspacePayload]:
    """Refresh independent workspaces with bounded, failure-isolated concurrency."""

    signal_date = operations.latest_signal_date()
    trade_date = str(signal_date).replace("-", "") if signal_date else None

    def refresh_chan() -> WorkspacePayload:
        payload = operations.refresh_chan(
            top_n=20,
            refresh=True,
            signal_date=signal_date,
        )
        return {
            "status": "success",
            "signal_date": payload.get("signal_date"),
            "candidates": len(payload.get("candidates") or []),
        }

    def refresh_long() -> WorkspacePayload:
        variants: list[WorkspacePayload] = []
        for variant in ("tea", "tea_safe", "v44"):
            payload = operations.refresh_long(
                variant=variant,
                signal_date=signal_date,
                refresh=True,
            )
            variants.append(
                {
                    "variant": variant,
                    "signal_date": payload.get("signal_date"),
                    "stocks": len(payload.get("stocks") or []),
                }
            )
        return {"status": "success", "variants": variants}

    def refresh_convertible_bonds() -> WorkspacePayload:
        payload = operations.refresh_convertible_bonds(
            trade_date=trade_date,
            limit=18,
            refresh=bool(trade_date),
        )
        return {
            "status": "success",
            "trade_date": payload.get("trade_date") or signal_date,
            "candidates": len(payload.get("candidates") or payload.get("items") or []),
        }

    def refresh_allotments() -> WorkspacePayload:
        payload = operations.refresh_allotments(
            refresh=True,
            expected_trade_date=signal_date,
            validate_quality=True,
        )
        return {
            "status": "success",
            "generated_at": payload.get("generated_at"),
            "records": len(payload.get("records") or []),
            "quality": payload.get("quality"),
        }

    def refresh_byd() -> WorkspacePayload:
        payload = operations.refresh_byd(refresh=True)
        planned_t = payload.get("planned_t") or {}
        return {
            "status": "success",
            "signal_date": planned_t.get("signal_date"),
            "alerts": len(payload.get("alerts") or []),
        }

    def refresh_similar_patterns() -> WorkspacePayload:
        payload = operations.refresh_similar_patterns()
        return {
            "status": "success",
            "generated_at": payload.get("generated_at"),
            "targets": len(payload.get("results") or []),
        }

    jobs: dict[str, Callable[[], WorkspacePayload]] = {
        "chan_model_strategy": refresh_chan,
        "long_stock_pool": refresh_long,
        "convertible_bond_plan": refresh_convertible_bonds,
        "convertible_bond_allotments": refresh_allotments,
        "byd_daily_plan": refresh_byd,
        "similar_patterns": refresh_similar_patterns,
    }
    configured_workers = (
        max_workers
        if max_workers is not None
        else int(os.getenv("ROUTINE_WEB_WORKSPACE_WORKERS", "6"))
    )
    worker_count = max(1, min(configured_workers, len(jobs)))
    results: dict[str, WorkspacePayload] = {}
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="quant-workspace",
    ) as executor:
        futures = {executor.submit(job): name for name, job in jobs.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = {"status": "failed", "error": str(exc)}
    return results
