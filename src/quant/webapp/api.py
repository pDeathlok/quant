from __future__ import annotations

from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from quant.webapp.services import (
    add_similar_pattern_watch_symbol,
    get_byd_daily_strategy,
    get_b1_plan,
    get_chan_model_strategy_plan,
    get_convertible_bond_allotments,
    get_convertible_bond_grid_plan,
    get_dashboard,
    get_latest_refresh_status,
    get_long_stock_pool,
    get_selector_calendar,
    get_research_index,
    get_similar_pattern_analysis,
    get_similar_pattern_watchlist,
    get_stock_selector_payload,
    refresh_similar_pattern_analysis,
    refresh_b1_plan,
    refresh_chan_model_strategy_plan,
    refresh_dashboard,
    remove_similar_pattern_watch_symbol,
    save_similar_pattern_watch_note,
    start_latest_refresh,
)


router = APIRouter()
SELECTOR_REPLAY_MIN_DATE = date(2020, 1, 1)


class RefreshPlanRequest(BaseModel):
    signal_date: str | None = None


class RefreshLatestRequest(BaseModel):
    scope: str = "all"


class SimilarPatternWatchRequest(BaseModel):
    symbol: str


class SimilarPatternWatchNoteRequest(BaseModel):
    content: str = ""


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "quant-webapp"}


def _build_byd_daily_plan(
    shares: int = Query(default=10000, ge=0, le=200000),
    cost: float = Query(default=110.6061, gt=0),
    refresh: bool = False,
) -> dict[str, Any]:
    try:
        return get_byd_daily_strategy(
            shares=shares,
            cost=cost,
            refresh=refresh,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BYD 日线计划生成失败: {exc}") from exc


@router.get("/byd/daily-plan")
def byd_daily_plan(
    shares: int = Query(default=10000, ge=0, le=200000),
    cost: float = Query(default=110.6061, gt=0),
    refresh: bool = False,
) -> dict[str, Any]:
    return _build_byd_daily_plan(
        shares=shares,
        cost=cost,
        refresh=refresh,
    )



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


@router.get("/chan/strategy-plan")
def chan_strategy_plan(
    top_n: int = Query(default=20, ge=1, le=100),
    signal_date: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    try:
        if signal_date:
            date.fromisoformat(signal_date)
        return get_chan_model_strategy_plan(top_n=top_n, refresh=refresh, signal_date=signal_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"缠论策略参数错误: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"缠论策略计划生成失败: {exc}") from exc


@router.post("/chan/strategy-plan/refresh")
def chan_strategy_plan_refresh(
    top_n: int = Query(default=20, ge=1, le=100),
    signal_date: str | None = None,
) -> dict[str, Any]:
    try:
        if signal_date:
            date.fromisoformat(signal_date)
        return refresh_chan_model_strategy_plan(top_n=top_n, signal_date=signal_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"缠论策略参数错误: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"缠论策略计划刷新失败: {exc}") from exc


@router.get("/long/stock-pool")
def long_stock_pool(
    variant: str = Query(default="tea"),
    signal_date: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    try:
        if signal_date:
            date.fromisoformat(signal_date)
        return get_long_stock_pool(variant=variant, signal_date=signal_date, refresh=refresh)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"长线策略参数错误: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"长线股票池生成失败: {exc}") from exc


@router.get("/convertible-bonds/plan")
def convertible_bonds_plan(
    trade_date: str | None = None,
    limit: int = Query(default=18, ge=1, le=50),
    refresh: bool = False,
) -> dict[str, Any]:
    try:
        return get_convertible_bond_grid_plan(trade_date=trade_date, limit=limit, refresh=refresh)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"可转债计划参数错误: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"可转债计划生成失败: {exc}") from exc


@router.get("/convertible-bonds/allotments")
def convertible_bond_allotments(
    limit: int = Query(default=80, ge=1, le=300),
    include_listed_days: int = Query(default=90, ge=0, le=2000),
    refresh: bool = False,
    stage_scope: Literal["pipeline", "all"] = Query(default="pipeline"),
) -> dict[str, Any]:
    try:
        return get_convertible_bond_allotments(
            limit=limit,
            include_listed_days=include_listed_days,
            refresh=refresh,
            stage_scope=stage_scope,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"配债股参数错误: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"配债股数据生成失败: {exc}") from exc


@router.get("/selector/stocks")
def stock_selector(
    strategies: str | None = None,
    signal_date: str | None = None,
    include_extended: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    try:
        if signal_date and date.fromisoformat(signal_date) < SELECTOR_REPLAY_MIN_DATE:
            raise HTTPException(status_code=400, detail="复盘查询暂从 2026-06-01 开始")
        selected = [item.strip() for item in strategies.split(",")] if strategies else None
        return get_stock_selector_payload(
            strategies=selected,
            signal_date=signal_date,
            include_extended=include_extended,
            use_cache=not refresh,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"日期格式错误: {signal_date}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"选股器生成失败: {exc}") from exc


@router.get("/selector/calendar")
def selector_calendar(
    start: str = "2026-06-01",
    end: str | None = None,
) -> dict[str, Any]:
    try:
        if date.fromisoformat(start) < SELECTOR_REPLAY_MIN_DATE:
            start = SELECTOR_REPLAY_MIN_DATE.isoformat()
        return get_selector_calendar(start=start, end=end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="日期格式错误") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"交易日历读取失败: {exc}") from exc


@router.post("/selector/refresh-latest")
def selector_refresh_latest(body: RefreshLatestRequest | None = None) -> dict[str, Any]:
    try:
        payload = body or RefreshLatestRequest()
        return start_latest_refresh(scope=payload.scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"最新数据刷新启动失败: {exc}") from exc


@router.get("/selector/refresh-latest/status")
def selector_refresh_latest_status() -> dict[str, Any]:
    return get_latest_refresh_status()


@router.get("/similar-patterns/watchlist")
def similar_pattern_watchlist() -> dict[str, Any]:
    try:
        return get_similar_pattern_watchlist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"相似走势自选池读取失败: {exc}") from exc


@router.post("/similar-patterns/watchlist")
def add_similar_pattern_watchlist_symbol(body: SimilarPatternWatchRequest) -> dict[str, Any]:
    try:
        return add_similar_pattern_watch_symbol(body.symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"相似走势自选池保存失败: {exc}") from exc


@router.delete("/similar-patterns/watchlist/{symbol}")
def delete_similar_pattern_watchlist_symbol(symbol: str) -> dict[str, Any]:
    try:
        return remove_similar_pattern_watch_symbol(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"相似走势自选池删除失败: {exc}") from exc


@router.put("/similar-patterns/watchlist/{symbol}/note")
def update_similar_pattern_watchlist_note(
    symbol: str,
    body: SimilarPatternWatchNoteRequest,
) -> dict[str, Any]:
    try:
        return save_similar_pattern_watch_note(symbol, body.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"自选股笔记保存失败: {exc}") from exc


@router.get("/similar-patterns/analysis")
def similar_pattern_analysis(refresh: bool = False) -> dict[str, Any]:
    try:
        return get_similar_pattern_analysis(refresh=refresh)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"相似走势分析失败: {exc}") from exc


@router.post("/similar-patterns/analysis/refresh")
def refresh_similar_pattern_analysis_api() -> dict[str, Any]:
    try:
        return refresh_similar_pattern_analysis()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"相似走势分析刷新失败: {exc}") from exc
