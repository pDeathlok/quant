from __future__ import annotations

from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from quant.webapp.services import (
    get_b1_plan,
    get_dashboard,
    get_latest_refresh_status,
    get_research_index,
    get_stock_selector_payload,
    refresh_b1_plan,
    refresh_dashboard,
    start_latest_refresh,
)


router = APIRouter()
SELECTOR_REPLAY_MIN_DATE = date(2026, 6, 1)


class RefreshPlanRequest(BaseModel):
    signal_date: str | None = None


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "quant-webapp"}


@router.get("/b1/plan")
def b1_plan(refresh: bool = False, signal_date: str | None = None) -> dict[str, Any]:
    try:
        return get_b1_plan(refresh=refresh, signal_date=signal_date)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B1 计划生成失败: {exc}") from exc


@router.post("/b1/plan/refresh")
def refresh_plan(body: RefreshPlanRequest | None = None) -> dict[str, Any]:
    try:
        payload = body or RefreshPlanRequest()
        return refresh_b1_plan(signal_date=payload.signal_date)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B1 计划刷新失败: {exc}") from exc


@router.get("/b1/strategies")
def b1_strategies() -> dict[str, Any]:
    plan = get_b1_plan()
    return {
        "generated_at": plan.get("generated_at"),
        "signal_date": plan.get("signal_date"),
        "strategy_pool": plan.get("strategy_pool", []),
    }


@router.get("/b1/signals")
def b1_signals(
    strategy_id: str | None = None,
    mode: Literal["unique", "all"] = Query(default="unique"),
) -> dict[str, Any]:
    plan = get_b1_plan()
    rows = plan.get("unique_symbols" if mode == "unique" else "plan_rows", [])
    if strategy_id:
        rows = [row for row in rows if row.get("strategy_id") == strategy_id]
    return {
        "generated_at": plan.get("generated_at"),
        "signal_date": plan.get("signal_date"),
        "mode": mode,
        "strategy_id": strategy_id,
        "rows": rows,
    }


@router.get("/b1/history")
def b1_history(refresh: bool = False) -> dict[str, Any]:
    try:
        return refresh_dashboard() if refresh else get_dashboard()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B1 历史复盘读取失败: {exc}") from exc


@router.get("/research/b1")
def b1_research(limit: int = Query(default=200, ge=1, le=2000)) -> dict[str, Any]:
    try:
        return get_research_index(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B1 研究结果读取失败: {exc}") from exc


@router.get("/selector/stocks")
def stock_selector(
    strategies: str | None = None,
    signal_date: str | None = None,
    include_z_skill: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    try:
        if signal_date and date.fromisoformat(signal_date) < SELECTOR_REPLAY_MIN_DATE:
            raise HTTPException(status_code=400, detail="复盘查询暂从 2026-06-01 开始")
        selected = [item.strip() for item in strategies.split(",")] if strategies else None
        return get_stock_selector_payload(
            strategies=selected,
            signal_date=signal_date,
            include_z_skill=include_z_skill,
            use_cache=not refresh,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"日期格式错误: {signal_date}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"选股器生成失败: {exc}") from exc


@router.post("/selector/refresh-latest")
def selector_refresh_latest() -> dict[str, Any]:
    try:
        return start_latest_refresh()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"最新数据刷新启动失败: {exc}") from exc


@router.get("/selector/refresh-latest/status")
def selector_refresh_latest_status() -> dict[str, Any]:
    return get_latest_refresh_status()
