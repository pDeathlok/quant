from __future__ import annotations

from typing import Any

from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY


REFRESH_SCOPE_LABELS = {
    "all": "全部工作区",
    "short": "短线策略",
    "chan": "缠论策略",
    "long": "长线策略",
    "cb": "可转债策略",
    "cbAllotment": "配债股",
    "byd": "多状态右侧",
    "similar": "自选池",
    "rightSideShadow": "右侧统一模型影子",
    "rightSideRankingCandidate": "右侧统一模型生产候选验收",
    "leftSideRanking": "左侧统一模型生产验收",
}

# Dependency steps come from the registry. The live futures preview runs
# after publication and has its own source and freshness checks.
REFRESH_SCOPE_STEPS = {
    scope: list(DEFAULT_DAILY_DEPENDENCY_REGISTRY.progress_steps(scope))
    for scope in DEFAULT_DAILY_DEPENDENCY_REGISTRY.scope_roots
}
REFRESH_SCOPE_STEPS["all"].append("index_futures")

REFRESH_STEP_DEFINITIONS = {
    "refresh_data": {"label": "拉取 Tushare 最新日线数据", "percent": 10},
    "feature_cache": {"label": "增量刷新统一日频因子层", "percent": 35},
    "daily_plan": {"label": "生成最新策略每日计划", "percent": 45},
    "signal_cache": {"label": "重建全市场策略规则信号", "percent": 56},
    "model_score": {"label": "计算当日策略模型分", "percent": 70},
    "selector_core": {"label": "计算短线核心股票池", "percent": 78},
    "selector_extended": {"label": "计算短线全策略股票池", "percent": 88},
    "chan_model_strategy": {"label": "生成缠论模型策略候选", "percent": 90},
    "long_stock_pool": {"label": "刷新长线因子截面、页面股票池与带血筹每日计划", "percent": 92},
    "convertible_bond_plan": {"label": "刷新可转债策略计划", "percent": 94},
    "convertible_bond_allotment": {"label": "刷新配债股数据", "percent": 96},
    "byd_daily_plan": {"label": "刷新多状态右侧次日计划", "percent": 97},
    "similar_patterns": {"label": "刷新自选池相似走势", "percent": 97},
    "snapshot": {"label": "写入策略股票池快照", "percent": 98},
    "index_futures": {"label": "刷新股指开仓预案", "percent": 99},
    "right_side_shadow_features": {"label": "构建右侧影子 canonical 因子", "percent": 80},
    "right_side_shadow_score": {"label": "计算右侧统一模型影子排序", "percent": 90},
    "right_side_shadow_product": {"label": "发布右侧影子候选", "percent": 98},
    "right_side_unified_features": {"label": "构建右侧生产 canonical 因子", "percent": 72},
    "right_side_unified_score": {"label": "计算右侧生产统一排序", "percent": 76},
    "right_side_unified_adapter": {"label": "校验右侧 selector 排序适配", "percent": 78},
    "left_side_unified_features": {"label": "构建左侧生产统一因子", "percent": 73},
    "left_side_unified_score": {"label": "计算左侧生产统一排序", "percent": 77},
    "left_side_unified_adapter": {"label": "校验左侧 selector 排序适配", "percent": 79},
    "application_portfolio": {"label": "刷新组合策略排名", "percent": 80},
}


def refresh_step_definition(key: str) -> dict[str, Any]:
    # Presentation metadata must not prevent a registered step from running.
    return REFRESH_STEP_DEFINITIONS.get(key, {"label": key, "percent": 0})


def normalize_refresh_scope(scope: str | None = None) -> str:
    """Normalize a public refresh scope and reject unsupported values."""

    if not scope:
        return "all"
    value = str(scope).strip()
    aliases = {
        "transition": "byd",
        "cb-allotment": "cbAllotment",
        "allotment": "cbAllotment",
        "convertible_bond": "cb",
        "convertible-bond": "cb",
    }
    normalized = aliases.get(value, value)
    if normalized not in REFRESH_SCOPE_STEPS:
        raise ValueError(f"未知刷新范围: {scope}")
    return normalized


def build_progress_steps(scope: str | None = None) -> list[dict[str, Any]]:
    """Build fresh mutable progress records for one refresh scope."""

    normalized = normalize_refresh_scope(scope)
    return [
        {
            "key": key,
            "label": refresh_step_definition(key)["label"],
            "status": "pending",
            "percent": refresh_step_definition(key)["percent"],
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": None,
            "checkpoint_reused": False,
        }
        for key in REFRESH_SCOPE_STEPS[normalized]
    ]
