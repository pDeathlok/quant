from __future__ import annotations

from typing import Any


REFRESH_SCOPE_LABELS = {
    "all": "全部工作区",
    "short": "短线策略",
    "chan": "缠论策略",
    "long": "长线策略",
    "cb": "可转债策略",
    "cbAllotment": "配债股",
    "byd": "BYD 做T",
    "similar": "自选池",
}

REFRESH_SCOPE_STEPS = {
    "all": [
        "refresh_data",
        "feature_cache",
        "daily_plan",
        "signal_cache",
        "model_score",
        "selector_core",
        "selector_extended",
        "chan_model_strategy",
        "long_stock_pool",
        "convertible_bond_plan",
        "convertible_bond_allotment",
        "byd_daily_plan",
        "similar_patterns",
        "snapshot",
    ],
    "short": [
        "refresh_data",
        "feature_cache",
        "daily_plan",
        "signal_cache",
        "model_score",
        "selector_core",
        "selector_extended",
        "snapshot",
    ],
    "chan": ["refresh_data", "chan_model_strategy"],
    "long": ["refresh_data", "long_stock_pool"],
    "cb": ["refresh_data", "convertible_bond_plan"],
    "cbAllotment": ["refresh_data", "convertible_bond_allotment"],
    "byd": ["refresh_data", "byd_daily_plan"],
    "similar": ["similar_patterns"],
}

REFRESH_STEP_DEFINITIONS = {
    "refresh_data": {"label": "拉取 Tushare 最新日线数据", "percent": 10},
    "feature_cache": {"label": "增量刷新统一日频因子层", "percent": 35},
    "daily_plan": {"label": "生成最新策略每日计划", "percent": 45},
    "signal_cache": {"label": "重建全市场策略规则信号", "percent": 56},
    "model_score": {"label": "计算当日策略模型分", "percent": 70},
    "selector_core": {"label": "计算短线核心股票池", "percent": 78},
    "selector_extended": {"label": "计算短线全策略股票池", "percent": 88},
    "chan_model_strategy": {"label": "生成缠论模型策略候选", "percent": 90},
    "long_stock_pool": {"label": "刷新长线因子截面与页面股票池", "percent": 92},
    "convertible_bond_plan": {"label": "刷新可转债策略计划", "percent": 94},
    "convertible_bond_allotment": {"label": "刷新配债股数据", "percent": 96},
    "byd_daily_plan": {"label": "刷新 BYD 做T日线计划", "percent": 97},
    "similar_patterns": {"label": "刷新自选池相似走势", "percent": 97},
    "snapshot": {"label": "写入策略股票池快照", "percent": 98},
}


def normalize_refresh_scope(scope: str | None = None) -> str:
    """Normalize a public refresh scope and reject unsupported values."""

    if not scope:
        return "all"
    value = str(scope).strip()
    aliases = {
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
            "label": REFRESH_STEP_DEFINITIONS[key]["label"],
            "status": "pending",
            "percent": REFRESH_STEP_DEFINITIONS[key]["percent"],
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": None,
            "checkpoint_reused": False,
        }
        for key in REFRESH_SCOPE_STEPS[normalized]
    ]
