from __future__ import annotations

import json
import re
import ast
import hashlib
import importlib
import importlib.util
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant.features.factor_execution import (
    allocate_worker_budget,
    calculator_execution_settings,
    configured_worker_budget,
)
import yaml

from quant.application.refresh_contracts import (
    REFRESH_SCOPE_LABELS,
    REFRESH_STEP_DEFINITIONS,
    build_progress_steps as _progress_steps,
    normalize_refresh_scope as _normalize_refresh_scope,
)
from quant.application.selector_ranking import (
    DEFAULT_SELECTOR_RANKING_CONFIG,
    SelectorRankingSource,
    apply_selector_ranking_source,
)
from quant.application.left_side_ranking import (
    DEFAULT_LEFT_SIDE_RANKING_CONFIG,
    load_left_side_ranking_candidates,
)
from quant.application.blood_chip_long_plan import (
    BLOOD_CHIP_LONG_SCHEMA_VERSION,
    build_blood_chip_daily_iteration,
    build_blood_chip_long_plan,
)
from quant.application.workspaces.byd import (
    BydWorkspaceDependencies,
    build_byd_daily_strategy,
)
from quant.application.workspaces.convertible_bonds import (
    DEFAULT_CONVERTIBLE_BOND_GRID_LIMIT,
    DEFAULT_ALLOTMENT_INCLUDE_LISTED_DAYS,
    DEFAULT_ALLOTMENT_LIMIT,
    DEFAULT_ALLOTMENT_STAGE_SCOPE,
    ConvertibleBondAllotmentDependencies,
    ConvertibleBondGridDependencies,
    build_convertible_bond_allotment_workspace,
    build_convertible_bond_grid_workspace,
    evaluate_convertible_bond_allotment_quality,
)
from quant.data.atomic_io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from quant.data.source_merge import normalize_ts_code
from quant.data.tushare_fetcher import TushareDataFetcher
from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
from quant.features.factor_registry import (
    FACTOR_REGISTRY,
    LONG_PRODUCTION_FACTOR_COLUMNS,
    LONG_PRODUCTION_FACTOR_SCHEMA_VERSION,
)
from quant.features.project_factor_layer import (
    PROJECT_FACTOR_SCHEMA_VERSION,
    calculate_project_factor_frame,
)
from quant.features.selector_buy_hold_factor_contract import (
    SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION,
    SELECTOR_BUY_HOLD_MANIFEST_SCHEMA_VERSION,
    SELECTOR_BUY_HOLD_RELEASE_ID,
    validate_selector_buy_hold_artifact,
)
from quant.features.variable_library import (
    PROJECT_FACTOR_COLUMNS,
    load_daily_basic_features,
)
from quant.infrastructure.workspace_snapshots import (
    WorkspaceSnapshotRepository,
    canonical_snapshot_date,
    workspace_params_key,
)
from quant.routine.b1_daily_plan import DAILY_PLAN_PATH, FEATURE_PATH, build_daily_plan, write_daily_plan
from quant.routine.cache_retention import run_cache_cleanup
from quant.routine.convertible_bond_allotment import build_convertible_bond_allotment_payload
from quant.routine.convertible_bond_grid_plan import build_convertible_bond_grid_plan, refresh_convertible_bond_daily
from quant.routine.dashboard import DASHBOARD_PATH, build_dashboard_payload, write_dashboard_json
from quant.routine.paths import DAILY_DIR, PROJECT_ROOT, WEB_DATA_DIR
from quant.research.similar_patterns import (
    SimilarPatternConfig,
    SimilarPatternResult,
    apply_probability_calibration,
    analyze_targets_by_threshold,
    build_probability_variant_cases,
    build_vector_caches_parallel,
    classify_forecast_signal,
    load_stock_basic,
    optimize_similar_cases,
    summarize_forecast,
    vector_cache_key,
)
from quant.research.similar_patterns_validation import build_industry_regime, load_market_regime
from quant.research.blood_chip import load_benchmark, load_canonical_daily
from quant.research.short_side_groups import GROUP_SIDE as SHORT_GROUP_SIDE
from quant.strategies.custom.b1_family import add_b1_family_signals
from quant.strategies.custom.triple_volume_breakout import add_triple_volume_strategy_pool_signals
from quant.strategies.custom.z_skill_patterns import (
    EXTENDED_STRATEGIES,
    build_extended_daily_signals,
    _stock_basic_map,
)
from quant.strategies.custom.chan_model import (
    add_chan_model_strategy_columns,
    select_chan_model_candidates,
    summarize_chan_model_strategy,
)


REPORT_DIR = PROJECT_ROOT / "reports/b1/research/xgb_project_vars_strategy"
FAMILY_SIGNAL_CACHE = PROJECT_ROOT / "data/features/b1/b1_family_rule_candidates.parquet"
EXTENDED_CANDIDATE_CACHE = PROJECT_ROOT / "data/features/z_skill_daily_candidates.parquet"
EXTENDED_SIGNAL_CACHE = PROJECT_ROOT / "data/features/z_skill_daily_signals.json"
EXTENDED_PLAYBOOK = REPORT_DIR / "latest_z_skill_operational_playbook.csv"
EXTENDED_MODEL_PLAYBOOK = REPORT_DIR / "latest_z_skill_model_operational_playbook.csv"
EXTENDED_MODEL_SCORED = REPORT_DIR / "latest_z_skill_model_scored_candidates.parquet"
EXTENDED_MODEL_SUMMARY = REPORT_DIR / "latest_z_skill_model_entry_exit_backtest.csv"
VEGAS_TUNNEL_REPORT_DIR = PROJECT_ROOT / "reports/vegas_tunnel"
SELECTOR_HISTORY_SIGNAL_SAMPLES = PROJECT_ROOT / "data/research/selector_history_full/selector_signal_history_samples.parquet"
SELECTOR_BUY_HOLD_SCORE_CALIBRATION = PROJECT_ROOT / "config/selector_buy_hold_score_calibration.json"
SELECTOR_BUY_HOLD_MODEL_DIR = (
    PROJECT_ROOT / "models/production/selector_buy_hold_registry_v3"
)
SELECTOR_BUY_HOLD_ROLLBACK_MODEL_DIR = (
    PROJECT_ROOT / "models/production/selector_buy_hold"
)
SELECTOR_SCORE_PROBABILITY_BANDS = PROJECT_ROOT / "config/selector_score_probability_bands.json"
SELECTOR_MODEL_HISTORY = (
    PROJECT_ROOT
    / "data/research/selector_buy_hold_registry_v2/selector_buy_hold_registry_dataset.parquet"
)
SELECTOR_DAILY_BASIC_DIR = PROJECT_ROOT / "data/raw/daily_basic"
SELECTOR_MARKET_FEATURE_COLUMNS = (
    "selector_market_mean_1d",
    "selector_market_median_1d",
    "selector_market_dispersion_1d",
    "selector_market_up_ratio_1d",
    "selector_market_up5_ratio_1d",
    "selector_market_down5_ratio_1d",
    "selector_market_mean_5d",
    "selector_market_mean_20d",
    "selector_market_up_ratio_5d",
    "selector_market_up_ratio_20d",
)
LONG_MARKET_REGIME_FEATURE_COLUMNS = (
    "market_regime",
    "index_ma_120_slope_20d",
    "index_return_20d",
    "index_return_60d",
    "index_return_120d",
    "index_drawdown_60d",
    "index_overheat",
)
FAMILY_RULE_PATTERN = "b1_family_rule_backtest_*.csv"
FUSION_PATTERN = "b1_model_zettaranc_fusion_*.csv"
MODEL_FILTERED_SIGNALS = {
    "B2",
    "BREATHING",
    "NANA",
    "YIDONG_DILIAN",
    "KEY_K",
    "GOLDEN_BOWL",
    "DUICHEN_VA",
    "ZAIHOU",
    "YUEYUE",
    "VIOLENCE_K",
}
MODEL_SIGNAL_LABELS = {
    "B2": "B2 模型分",
    "BREATHING": "呼吸结构",
    "NANA": "娜娜图形",
    "YIDONG_DILIAN": "异动地量",
    "KEY_K": "关键K",
    "GOLDEN_BOWL": "黄金碗",
    "DUICHEN_VA": "对称VA",
    "ZAIHOU": "灾后重建",
    "YUEYUE": "跃跃欲试",
    "VIOLENCE_K": "暴力K",
}
_LONG_RESEARCH_MODULE_LOCK = threading.Lock()
_TEA_MASTER_MODULE_LOCK = threading.Lock()
_LONG_LIVE_DATA_LOCK = threading.Lock()
_TEA_MASTER_LIVE_SCORE_LOCK = threading.Lock()
_SELECTOR_SNAPSHOT_SCHEMA_LOCK = threading.Lock()
_SELECTOR_SNAPSHOT_SCHEMA_READY_URLS: set[str] = set()
STRATEGY_GROUPS = [
    {"key": "B1", "label": "B1", "status": "超跌反弹，模型+规则", "members": ["B1"]},
    {"key": "B2", "label": "B2", "status": "B1后/独立右侧确认", "members": ["B2"]},
    {"key": "B3", "label": "B3", "status": "B2后分歧转一致", "members": ["B3"]},
    {"key": "SB1", "label": "SB1", "status": "横盘下破洗盘", "members": ["SB1"]},
    {"key": "SUPER_B1", "label": "超级B1", "status": "放量下杀后企稳", "members": ["SUPER_B1"]},
    {"key": "STRONG_K", "label": "强K/突破", "status": "关键K、暴力K", "members": ["KEY_K", "VIOLENCE_K"]},
    {"key": "DOUBLE_YANG", "label": "双阳结构", "status": "平行重炮、双枪", "members": ["PINGHANG", "DOUBLE_GUN"]},
    {"key": "LOW_PULLBACK", "label": "缩量回调低吸", "status": "异动地量、娜娜、对称VA", "members": ["YIDONG_DILIAN", "NANA", "DUICHEN_VA"]},
    {"key": "SUPPORT_PULLBACK", "label": "支撑回踩", "status": "黄金碗、灾后重建", "members": ["GOLDEN_BOWL", "ZAIHOU"]},
    {"key": "RHYTHM_PLATFORM", "label": "节奏/平台", "status": "呼吸结构、跃跃欲试", "members": ["BREATHING", "YUEYUE"]},
    {"key": "CHANGAN", "label": "长安战法", "status": "日线三日确认", "members": ["CHANGAN"]},
    {"key": "KENGQI", "label": "坑里起好货", "status": "日线填坑观察", "members": ["KENGQI"]},
    {"key": "VEGAS", "label": "维加斯隧道", "status": "趋势回踩右侧确认", "members": ["VEGAS"]},
    {"key": "TRIPLE_VOLUME_BREAKOUT", "label": "三倍量突破", "status": "缩量盘整后右侧突破", "members": ["TRIPLE_VOLUME_BREAKOUT"]},
]
STRATEGY_GROUP_MEMBERS = {
    item["key"]: set(item["members"])
    for item in STRATEGY_GROUPS
}
STRATEGY_MEMBER_TO_GROUP = {
    member: item["key"]
    for item in STRATEGY_GROUPS
    for member in item["members"]
}
STRATEGY_GROUP_LABELS = {item["key"]: item["label"] for item in STRATEGY_GROUPS}
FAMILY_SIGNAL_COLUMNS = {
    "B1_RULE": ("signal_b1", "B1", "B1 规则观察", "J<=-10、缩量回调、未连续四阴、接近 BBI/MA60 支撑"),
    "B2_ANY": ("b2_any_pchg4_vol15", "B2", "B2 独立右侧确认", "涨幅>=4%、5日量比>=1.5、收盘位置>=75%、J<80"),
    "B2_OVERSOLD": ("b2_oversold_pchg3_vol12", "B2", "B2 超跌后右侧确认", "近5日 J<0 后，涨幅>=3%、5日量比>=1.2、收盘位置>=70%"),
    "B2_BBI": ("b2_bbi_reclaim_vol12", "B2", "B2 BBI 收复确认", "昨日不高于 BBI，今日收复 BBI、涨幅>=3%、5日量比>=1.2"),
    "B2_FROM_B1": ("b2_pchg4_vol15", "B2", "B2 B1 后确认", "B1 后 3 日内涨幅>=4%、5日量比>=1.5、上影较小、J<55"),
    "B3_BROAD_SMALL": ("b3_broad_small_pos", "B3", "B3 宽口径小阳延续", "宽口径 B2 后三日内小阳，涨幅0%-2%、振幅<7%、收盘在中位以上"),
    "B3_BROAD_PULLBACK": ("b3_broad_calm_pullback", "B3", "B3 宽口径缩量分歧", "宽口径 B2 后三日内-1%到2%、振幅<7%、5日量比<=1.3"),
    "B3_SMALL": ("b3_small_pos_amp7", "B3", "B3 标准小阳延续", "标准 B2 后三日内小阳，涨幅0%-2%、振幅<7%"),
    "SB1": ("sb1_range10_vol12", "SB1", "SB1 洗盘观察", "前三日横盘区间<10%、放量阴线下破、J<0"),
    "SUPER_B1": ("super_washout_j10_closepos40", "SUPER_B1", "超级 B1", "近三日放量下杀，随后 J<-10 且收盘位置>=40%"),
    "VEGAS_TUNNEL": ("signal_vegas_tunnel", "VEGAS", "维加斯隧道", "EMA144/169 隧道上行，EMA10>EMA20>隧道上沿，近8日回踩2.5%范围后收阳放量站上EMA10"),
    "TRIPLE_VOLUME_BREAKOUT": ("signal_tvb_merged", "TRIPLE_VOLUME_BREAKOUT", "三倍量缩量盘整突破", "2.5倍量扩展候选池，3倍量命中提升为保守主策略；突破日前平均缩量，MA5>MA10>MA20且MA20上行"),
}
DEFAULT_ACTION_LEVELS = {"可小仓实操", "谨慎实操"}
DEFAULT_FAMILIES = {
    "B1",
    "B2",
    "B3",
    "SB1",
    "SUPER_B1",
    "STRONG_K",
    "DOUBLE_YANG",
    "LOW_PULLBACK",
    "SUPPORT_PULLBACK",
    "RHYTHM_PLATFORM",
    "VEGAS",
    "TRIPLE_VOLUME_BREAKOUT",
    *MODEL_FILTERED_SIGNALS,
}
DEFAULT_SELECTOR_LIMIT = 50
DEFAULT_FAMILY_CAP = 12
SELECTOR_SNAPSHOT_SCHEMA_VERSION = "buy_hold_score_v9_exact_date_feature_contract"
SELECTOR_SNAPSHOT_DIR = PROJECT_ROOT / "data/selector_snapshots"
SELECTOR_SNAPSHOT_TABLE = "selector_snapshots"
LONG_STOCK_POOL_SCHEMA_VERSION = "qfq_price_snapshot_v12_price_bands_structure_evidence"
LONG_STOCK_POOL_SNAPSHOT_DIR = PROJECT_ROOT / "data/long_stock_pool_snapshots"
LONG_STOCK_POOL_SNAPSHOT_TABLE = "long_stock_pool_snapshots"
BLOOD_CHIP_LONG_SNAPSHOT_DIR = PROJECT_ROOT / "data/blood_chip_long_snapshots"
BLOOD_CHIP_DAILY_DIR = PROJECT_ROOT / "data/raw/daily_partitioned"
BLOOD_CHIP_BENCHMARK_PATH = PROJECT_ROOT / "data/raw/index_000300.SH.parquet"
LONG_PRICE_SCORE_BAND_CONFIG_PATH = PROJECT_ROOT / "config/long_price_score_bands.json"
LONG_FACTOR_SNAPSHOT_SCHEMA_VERSION = LONG_PRODUCTION_FACTOR_SCHEMA_VERSION
LONG_FACTOR_SNAPSHOT_DIR = PROJECT_ROOT / "data/features/long"
LONG_FACTOR_REQUIRED_COLUMNS = (
    "date",
    "ts_code",
    "good_stock_score",
    "profitability_score",
    "fundamental_growth_score",
    "balance_sheet_score",
    "business_stability_score",
    "historical_value_score",
    "roe",
    "pe_ttm",
    "pb",
    "pr_pe",
    "pr_pb",
    "roe_hist_percentile",
    "pe_hist_percentile",
    "pb_hist_percentile",
    "pr_pe_hist_percentile",
    "pr_pb_hist_percentile",
    "valuation_history_points",
)
WEB_WORKSPACE_SNAPSHOT_SCHEMA_VERSION = "workspace_payload_v1"
WEB_WORKSPACE_SNAPSHOT_TABLE = "web_workspace_snapshots"
WEB_WORKSPACE_SNAPSHOT_DIR = PROJECT_ROOT / "data/workspace_snapshots"
CONVERTIBLE_BOND_GRID_PLAN_PATH = PROJECT_ROOT / "data/web/convertible_bond_grid_plan.json"
REFRESH_STATUS_PATH = PROJECT_ROOT / "data/routine/latest_refresh_status.json"
REFRESH_MANIFEST_ROOT = PROJECT_ROOT / "data/routine"
REFRESH_RUNNING_STALE_SECONDS = 6 * 60 * 60
REFRESH_NO_PROGRESS_STALE_SECONDS = 30 * 60
SIMILAR_PATTERNS_TIMEOUT_SECONDS = 45 * 60
LONG_STATE_DIR = PROJECT_ROOT / "data/long_strategy_state"
LONG_VARIANTS = {
    "tea": "core14_soft_plus",
    "tea_safe": "core14_soft_spread",
    "v44": "v44_tea_master_defensive_neutral",
    "v43": "v43_tea_master_core_only",
    "v34": "v34_pit_universe_guarded_sleeve",
    "v35": "v35_pit_universe_riskon_recovery_sleeve",
    "v33": "v33_bull_boost_defensive_bear_sleeve",
    "v31": "v31_bull_bear_exposure_sleeve",
}
LONG_VARIANT_LABELS = {
    "tea": "长线好股票 · 好价格",
    "tea_safe": "茶大长线稳健网格",
    "v44": "防守中性长期组合",
    "v43": "核心质量长期组合",
    "v34": "PIT 成长防守长期组合",
    "v35": "PIT 趋势恢复长期组合",
    "v33": "趋势增强长期组合",
    "v31": "牛熊防守长期组合",
}
TEA_LONG_VARIANTS = {
    "tea": "core14_soft_plus",
    "tea_safe": "core14_soft_spread",
}


def _selected_yaml_variant(path: Path, strategy_id: str) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    strategies = payload.get("strategies") or []
    matches = [
        item
        for item in strategies
        if str(item.get("id")) == strategy_id and bool(item.get("enabled", True))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path} expected one enabled strategy {strategy_id}, found {len(matches)}"
        )
    variant = str((matches[0].get("backtest") or {}).get("variant") or "")
    if not variant:
        raise ValueError(f"{path} strategy {strategy_id} has no backtest.variant")
    return variant


def _reload_production_strategy_configs() -> None:
    global add_triple_volume_strategy_pool_signals

    LONG_VARIANTS["v44"] = _selected_yaml_variant(
        PROJECT_ROOT / "configs/strategies/long_dividend_quality.yaml",
        "l1_market_regime_quality",
    )
    tea_variant = _selected_yaml_variant(
        PROJECT_ROOT / "configs/strategies/tea_master_long.yaml",
        "tea_master_regime_grid",
    )
    LONG_VARIANTS["tea"] = tea_variant
    TEA_LONG_VARIANTS["tea"] = tea_variant
    module_name = "quant.strategies.custom.triple_volume_breakout"
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    add_triple_volume_strategy_pool_signals = (
        module.add_triple_volume_strategy_pool_signals
    )
LONG_VALUATION_WINDOW_MONTHS = 84
LONG_VALUATION_MINIMUM_MONTHS = 24
LONG_LIVE_DAILY_BASIC_LOOKBACK_MONTHS = 96
_REFRESH_LOCK = threading.Lock()
_REFRESH_CONTEXT = threading.local()
_REFRESH_ACTIVE_PROCS: dict[str, mp.Process] = {}
_SIMILAR_PATTERN_REFRESH_GUARD = threading.Lock()
_SIMILAR_PATTERN_REFRESH_INFLIGHT: tuple[
    tuple[tuple[str, ...], bool],
    Future[dict[str, Any]],
] | None = None
_REFRESH_STATUS: dict[str, Any] = {
    "status": "idle",
    "run_id": None,
    "trade_date": None,
    "attempt": 0,
    "resumed_from": None,
    "started_at": None,
    "finished_at": None,
    "updated_at": None,
    "message": "尚未启动刷新任务",
    "percent": 0,
    "current_step": None,
    "steps": [],
    "result": None,
    "error": None,
}
REFRESH_CHILD_PROCESS_MARKERS = (
    "scripts/research/refresh_b1_feature_cache.py",
    "scripts/research/rebuild_strategy_signal_cache.py",
)
SIMILAR_PATTERN_STATE_DIR = PROJECT_ROOT / "data/research/similar_patterns"
SIMILAR_PATTERN_WATCHLIST_PATH = SIMILAR_PATTERN_STATE_DIR / "watchlist.json"
OPERATION_PLANS_PATH = WEB_DATA_DIR / "operation_plans.json"
SIMILAR_PATTERN_ANALYSIS_PATH = SIMILAR_PATTERN_STATE_DIR / "web_watchlist_analysis.json"
SIMILAR_PATTERN_VECTOR_CACHE_DIR = SIMILAR_PATTERN_STATE_DIR / "vector_cache"
SIMILAR_PATTERN_VALIDATION_PATH = PROJECT_ROOT / "reports/similar_patterns/validation_2025/calibration.json"
SIMILAR_PATTERN_CALIBRATION_MAX_FLAT_SPAN = 15.0
SIMILAR_PATTERN_VECTOR_CACHE_REFRESH_DAYS = 7
SIMILAR_PATTERN_VECTOR_CACHE_MIN_REFRESH_AGE_DAYS = 5
SIMILAR_PATTERN_VECTOR_CACHE_REFRESH_WEEKDAY = 4  # Friday
SIMILAR_PATTERN_VECTOR_CACHE_REFRESH_HOUR = 15
SIMILAR_PATTERN_VECTOR_CACHE_METADATA = "_refresh_metadata.json"
CONVERTIBLE_BOND_ALLOTMENT_DAILY_PATH = PROJECT_ROOT / "data/routine/convertible_bond_allotments_latest.json"
SIMILAR_PATTERN_DEFAULT_WATCHLIST = ["002594.SZ", "002788.SZ"]
WATCHLIST_ALERT_INDICATORS = {
    "ret_20d",
    "drawdown_60d",
    "vol_ratio20",
    "dist_ma20",
    "dist_ma60",
    "kdj_daily_j",
    "opportunity_score",
    "holding_score",
}
WATCHLIST_ALERT_OPERATORS = {"gt", "eq", "lt"}
WATCHLIST_ALERT_MAX_REMINDERS = 20
WATCHLIST_ALERT_MAX_CONDITIONS = 20
WATCHLIST_ALERT_NOTE_MAX_LENGTH = 1_000
SIMILAR_PATTERN_CONFIG = SimilarPatternConfig(
    candidate_step_days=5,
    candidate_start_date="2018-01-01",
    similarity_threshold=0.055,
    take_profit_3d=0.03,
    stop_loss_3d=0.03,
)
SIMILARITY_SCORE_CONTRAST = 15.0


def _similarity_score_ceiling(config: SimilarPatternConfig = SIMILAR_PATTERN_CONFIG) -> float:
    """Return the fixed global upper bound of the forecast-weight formula."""
    return float(
        max(config.same_industry_weight, config.cross_industry_weight)
        * max(1.0, config.same_regime_weight, config.regime_mismatch_weight)
        * max(
            1.0,
            config.same_industry_regime_weight,
            config.industry_regime_mismatch_weight,
        )
    )


def _similarity_score_config() -> dict[str, float]:
    return {
        "similarity_score_ceiling": _similarity_score_ceiling(),
        "similarity_score_contrast": SIMILARITY_SCORE_CONTRAST,
    }


def _similar_pattern_vector_cache_state_dir() -> Path:
    return SIMILAR_PATTERN_VECTOR_CACHE_DIR / vector_cache_key(SIMILAR_PATTERN_CONFIG)


def _read_similar_pattern_vector_cache_metadata() -> dict[str, Any]:
    path = _similar_pattern_vector_cache_state_dir() / SIMILAR_PATTERN_VECTOR_CACHE_METADATA
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _next_similar_pattern_vector_cache_refresh_at(current: datetime) -> datetime:
    candidate = current.replace(
        hour=SIMILAR_PATTERN_VECTOR_CACHE_REFRESH_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )
    days_until_refresh = (SIMILAR_PATTERN_VECTOR_CACHE_REFRESH_WEEKDAY - current.weekday()) % 7
    candidate = candidate + timedelta(days=days_until_refresh)
    if candidate <= current:
        candidate = candidate + timedelta(days=SIMILAR_PATTERN_VECTOR_CACHE_REFRESH_DAYS)
    return candidate


def _similar_pattern_vector_cache_in_refresh_window(current: datetime) -> bool:
    return (
        current.weekday() == SIMILAR_PATTERN_VECTOR_CACHE_REFRESH_WEEKDAY
        and current.hour >= SIMILAR_PATTERN_VECTOR_CACHE_REFRESH_HOUR
    )


def _similar_pattern_vector_cache_refreshed_this_window(
    refreshed_at: pd.Timestamp,
    current: datetime,
) -> bool:
    if pd.isna(refreshed_at) or not _similar_pattern_vector_cache_in_refresh_window(current):
        return False
    return refreshed_at.to_pydatetime().date() == current.date()


def _similar_pattern_source_date_is_current(source_trade_date: str | None, current: datetime) -> bool:
    if not source_trade_date:
        return False
    parsed = pd.to_datetime(source_trade_date, errors="coerce")
    return pd.notna(parsed) and parsed.date() == current.date()


def _latest_similar_pattern_target_date(symbols: list[str]) -> str | None:
    """Read only watchlist files; target vectors themselves are still built live later."""
    latest: pd.Timestamp | None = None
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=DAILY_DIR.parent))
    for symbol in symbols:
        try:
            candidate = store.latest_trade_date(DAILY_DIR.name, symbol)
        except Exception:
            candidate = None
        if pd.notna(candidate) and (latest is None or candidate > latest):
            latest = pd.Timestamp(candidate)
    return latest.strftime("%Y-%m-%d") if latest is not None else None


def _similar_pattern_vector_cache_refresh_decision(
    *,
    force: bool = False,
    now: datetime | None = None,
    source_trade_date: str | None = None,
) -> dict[str, Any]:
    state_dir = _similar_pattern_vector_cache_state_dir()
    cache_files = list(state_dir.glob("*.npz")) if state_dir.exists() else []
    metadata = _read_similar_pattern_vector_cache_metadata()
    current = now or datetime.now()
    force_from_env = os.getenv("SIMILAR_PATTERN_FORCE_VECTOR_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    refreshed_at = pd.to_datetime(metadata.get("refreshed_at"), errors="coerce")

    refresh_age_days = (
        max(0.0, (current - refreshed_at.to_pydatetime()).total_seconds() / 86_400)
        if pd.notna(refreshed_at)
        else None
    )
    minimum_age_reached = (
        refresh_age_days is None
        or refresh_age_days >= SIMILAR_PATTERN_VECTOR_CACHE_MIN_REFRESH_AGE_DAYS
    )
    repair_reason: str | None = None
    if not cache_files:
        repair_reason = "cache_missing"
    elif int(metadata.get("errors") or 0) > 0:
        repair_reason = "previous_refresh_errors"
    elif metadata.get("cached_files") is not None and int(metadata["cached_files"]) != len(cache_files):
        repair_reason = "cache_file_count_changed"
    elif any(path.stat().st_size == 0 for path in cache_files):
        repair_reason = "empty_cache_file"
    elif pd.isna(refreshed_at):
        repair_reason = "refresh_time_missing"

    if force or force_from_env:
        due, reason = True, "forced"
    elif not _similar_pattern_vector_cache_in_refresh_window(current):
        due, reason = False, "waiting_for_friday_close"
    elif not _similar_pattern_source_date_is_current(source_trade_date, current):
        due, reason = False, "waiting_for_friday_trade_close"
    elif not minimum_age_reached:
        if _similar_pattern_vector_cache_refreshed_this_window(refreshed_at, current):
            due, reason = False, "friday_close_window_already_refreshed"
        else:
            due, reason = False, "minimum_refresh_age_not_reached"
    else:
        due, reason = True, repair_reason or "friday_close_window"

    next_refresh_at = _next_similar_pattern_vector_cache_refresh_at(current).isoformat(
        timespec="seconds"
    )
    return {
        "due": due,
        "reason": reason,
        "cached_files": len(cache_files),
        "refreshed_at": refreshed_at.to_pydatetime().isoformat(timespec="seconds")
        if pd.notna(refreshed_at)
        else None,
        "next_refresh_at": next_refresh_at,
        "refresh_age_days": refresh_age_days,
        "minimum_refresh_age_days": SIMILAR_PATTERN_VECTOR_CACHE_MIN_REFRESH_AGE_DAYS,
        "metadata": metadata,
        "inferred_legacy": False,
    }


def _write_similar_pattern_vector_cache_metadata(
    *,
    refreshed_at: datetime,
    source_trade_date: str | None,
    cache_audit: pd.DataFrame,
) -> dict[str, Any]:
    state_dir = _similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "config_key": vector_cache_key(SIMILAR_PATTERN_CONFIG),
        "refreshed_at": refreshed_at.isoformat(timespec="seconds"),
        "source_trade_date": source_trade_date,
        "refresh_interval_days": SIMILAR_PATTERN_VECTOR_CACHE_REFRESH_DAYS,
        "minimum_refresh_age_days": SIMILAR_PATTERN_VECTOR_CACHE_MIN_REFRESH_AGE_DAYS,
        "cached_files": len(list(state_dir.glob("*.npz"))),
        "rebuilt": int(cache_audit["status"].eq("built").sum()) if not cache_audit.empty else 0,
        "reused": int(cache_audit["status"].eq("cache_hit").sum()) if not cache_audit.empty else 0,
        "errors": int(cache_audit["status"].eq("error").sum()) if not cache_audit.empty else 0,
    }
    path = state_dir / SIMILAR_PATTERN_VECTOR_CACHE_METADATA
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return payload


CHAN_MODEL_SCORED_PATH = PROJECT_ROOT / "reports/chan_daily/model_filter/chan_model_scored_candidates.parquet"
CHAN_MODEL_STRATEGY_DIR = PROJECT_ROOT / "reports/chan_daily/model_strategy"
CHAN_MODEL_STRATEGY_SCORED_PATH = CHAN_MODEL_STRATEGY_DIR / "chan_model_strategy_scored.parquet"
CHAN_MODEL_LATEST_CANDIDATES_PATH = CHAN_MODEL_STRATEGY_DIR / "chan_model_latest_candidates.csv"
CHAN_MODEL_SUMMARY_PATH = CHAN_MODEL_STRATEGY_DIR / "chan_model_strategy_summary.csv"


def _persist_refresh_status_unlocked() -> None:
    if (
        _REFRESH_STATUS.get("status") in {"success", "failed"}
        and _REFRESH_STATUS.get("run_id")
        and not _REFRESH_STATUS.get("manifest_path")
    ):
        try:
            _write_terminal_refresh_manifest_unlocked()
        except Exception as exc:
            _REFRESH_STATUS["manifest_error"] = str(exc)
    atomic_write_json(_REFRESH_STATUS, REFRESH_STATUS_PATH)


def _write_terminal_refresh_manifest_unlocked() -> Path | None:
    if _REFRESH_STATUS.get("status") not in {"success", "failed"}:
        return None
    run_id = str(_REFRESH_STATUS.get("run_id") or "").strip()
    if not run_id:
        return None
    started_at = _parse_refresh_timestamp(_REFRESH_STATUS.get("started_at")) or datetime.now()
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip("-") or "run"
    run_dir = REFRESH_MANIFEST_ROOT / f"{started_at.strftime('%Y%m%d_%H%M%S')}_{safe_run_id}"
    manifest_path = run_dir / "manifest.json"
    _REFRESH_STATUS["manifest_path"] = str(manifest_path)
    if manifest_path.exists():
        return manifest_path
    payload = {
        "schema_version": 1,
        "kind": "web_daily_refresh",
        **_REFRESH_STATUS,
    }
    atomic_write_json(payload, manifest_path)
    return manifest_path


def _load_persisted_refresh_status() -> dict[str, Any] | None:
    try:
        if not REFRESH_STATUS_PATH.exists():
            return None
        payload = json.loads(REFRESH_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return _ensure_refresh_scope(payload)


def _ensure_refresh_scope(status: dict[str, Any]) -> dict[str, Any]:
    scoped = dict(status)
    try:
        scope = _normalize_refresh_scope(scoped.get("scope"))
    except ValueError:
        scope = "all"
    scoped["scope"] = scope
    scoped["scope_label"] = REFRESH_SCOPE_LABELS[scope]
    return scoped


def _refresh_attempt_number(
    resume_status: dict[str, Any] | None,
) -> int:
    if not resume_status:
        return 1
    try:
        return max(1, int(resume_status.get("attempt") or 1)) + 1
    except (TypeError, ValueError):
        return 2


def _refresh_resume_source(
    resume_status: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not resume_status:
        return None
    return {
        "run_id": resume_status.get("run_id"),
        "attempt": max(1, _refresh_attempt_number(resume_status) - 1),
        "started_at": resume_status.get("started_at"),
        "finished_at": resume_status.get("finished_at"),
        "elapsed_seconds": resume_status.get("elapsed_seconds"),
        "manifest_path": resume_status.get("manifest_path"),
    }


def _parse_refresh_timestamp(value: Any) -> datetime | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    parsed = ts.to_pydatetime()
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _last_refresh_progress_at(status: dict[str, Any]) -> datetime | None:
    return (
        _parse_refresh_timestamp(status.get("updated_at"))
        or _parse_refresh_timestamp(status.get("finished_at"))
        or _parse_refresh_timestamp(status.get("started_at"))
    )


def _step_status_map(status: dict[str, Any] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for step in (status or {}).get("steps") or []:
        key = str(step.get("key") or "")
        if key:
            mapping[key] = str(step.get("status") or "pending")
    return mapping


def _tail_resume_ready(status: dict[str, Any] | None, scope: str) -> bool:
    if not status or scope not in {"all", "short"}:
        return False
    dependency_audit = (
        ((status.get("result") or {}).get("dependency_postflight") or {}).get(
            "freshness_audit"
        )
        or {}
    )
    if dependency_audit.get("status") == "failed":
        return False
    step_map = _step_status_map(status)
    return step_map.get("selector_extended") == "success" and step_map.get("snapshot") != "success"


def _source_expected_trade_date(results: dict[str, Any] | None) -> str | None:
    refresh_data = (results or {}).get("refresh_data") or {}
    expected = str(refresh_data.get("expected_trade_date") or "").replace("-", "")
    if len(expected) == 8 and expected.isdigit():
        return expected
    return None


def _local_market_trade_date() -> str | None:
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=DAILY_DIR.parent))
    latest = store.latest_dataset_trade_date(DAILY_DIR.name)
    return latest.strftime("%Y%m%d") if latest is not None else None


def _input_resume_ready(status: dict[str, Any] | None, scope: str) -> bool:
    """Reuse same-day source refreshes after a downstream stage fails."""

    if not status or status.get("status") != "failed":
        return False
    started_at = _parse_refresh_timestamp(status.get("started_at"))
    if started_at is None or started_at.date() != datetime.now().date():
        return False
    results = status.get("result") or {}
    refresh_data = results.get("refresh_data") or {}
    if refresh_data.get("status") != "success":
        return False
    expected_trade_date = _source_expected_trade_date(results)
    if expected_trade_date is None or _local_market_trade_date() != expected_trade_date:
        return False
    if scope in {"all", "short", "chan", "long", "cbAllotment"} and (
        results.get("refresh_daily_basic") or {}
    ).get("status") != "success":
        return False
    if scope in {"all", "short", "chan", "long", "cbAllotment"}:
        basic_date = str(
            (results.get("refresh_daily_basic") or {}).get("latest_trade_date")
            or ""
        ).replace("-", "")
        if basic_date != expected_trade_date:
            return False
    reference = results.get("refresh_reference_inputs") or {}
    if reference.get("status") != "success":
        return False
    if scope in {"all", "long"}:
        analyst = (reference.get("steps") or {}).get("analyst_forecast_snapshot") or {}
        # A degraded analyst refresh explicitly means a last-known-good
        # snapshot is available and the reference stage had no critical error.
        # Treat it as a reusable input checkpoint; otherwise an unrelated
        # downstream retry needlessly repeats every market-data request.
        if analyst.get("status") not in {"success", "skipped", "degraded"}:
            return False
    return True


def _refresh_resume_ready(status: dict[str, Any] | None, scope: str) -> bool:
    return _tail_resume_ready(status, scope) or _input_resume_ready(status, scope)


def _completed_checkpoint_ready(
    status: dict[str, Any] | None,
    scope: str,
) -> bool:
    """Return whether a committed same-date run can seed idempotent reuse.

    Source polling still runs before any reuse decision.  This checkpoint only
    supplies prior result metadata; the dependency preflight decides whether
    downstream artifacts remain reusable after the polls complete.
    """

    if not status or status.get("status") != "success":
        return False
    try:
        if _normalize_refresh_scope(status.get("scope")) != _normalize_refresh_scope(scope):
            return False
    except ValueError:
        return False
    results = status.get("result") or {}
    expected_trade_date = _source_expected_trade_date(results)
    if expected_trade_date is None or _local_market_trade_date() != expected_trade_date:
        return False
    postflight = results.get("dependency_postflight") or {}
    return bool(
        postflight.get("status") == "success"
        and postflight.get("baseline_committed") is True
        and str(postflight.get("target_trade_date") or "").replace("-", "")
        == expected_trade_date
    )


def _retry_preflight_identity_stable(
    previous: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> bool:
    """Prove that a failed run's completed checkpoints remain reusable.

    A normal next-trading-day preflight marks every exact-date downstream node
    dirty relative to yesterday's committed baseline.  On a same-day retry we
    compare against the failed attempt's own preflight identity instead: code,
    model and data-source identities must all remain unchanged.  Output nodes
    are checked separately by their exact-date artifact gates.
    """

    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    if not (
        previous.get("identity_complete") is True
        and current.get("identity_complete") is True
        and previous.get("scope") == current.get("scope")
        and previous.get("target_trade_date") == current.get("target_trade_date")
    ):
        return False
    for field in ("node_contract_hashes", "model_contract_hashes"):
        old = previous.get(field)
        new = current.get(field)
        if not isinstance(old, dict) or not isinstance(new, dict) or old != new:
            return False
    old_states = previous.get("node_state_fingerprints")
    new_states = current.get("node_state_fingerprints")
    if not isinstance(old_states, dict) or not isinstance(new_states, dict):
        return False
    source_nodes = {
        str(node_id)
        for node_id in set(old_states) | set(new_states)
        if str(node_id).startswith("data.")
    }
    return bool(source_nodes) and all(
        old_states.get(node_id)
        and old_states.get(node_id) == new_states.get(node_id)
        for node_id in source_nodes
    )


def _new_refresh_run_id(scope: str) -> str:
    return f"{scope}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _owned_refresh_context_active_unlocked() -> bool:
    context_run_id = getattr(_REFRESH_CONTEXT, "run_id", None)
    if not context_run_id:
        return True
    if _REFRESH_STATUS.get("run_id") != context_run_id:
        return False
    return _REFRESH_STATUS.get("status") not in {"success", "failed"}


def _refresh_child_process_pids() -> list[int]:
    pids: list[int] = []
    seen: set[int] = set()
    for marker in REFRESH_CHILD_PROCESS_MARKERS:
        try:
            result = subprocess.run(
                ["pgrep", "-fl", marker],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            pid_text, _, command = line.partition(" ")
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            if pid == os.getpid() or pid in seen:
                continue
            if not any(child_marker in command for child_marker in REFRESH_CHILD_PROCESS_MARKERS):
                continue
            seen.add(pid)
            pids.append(pid)
    return pids


def _terminate_refresh_child_processes() -> int:
    terminated = 0
    for pid in _refresh_child_process_pids():
        signaled = False
        try:
            os.killpg(pid, signal.SIGTERM)
            signaled = True
        except ProcessLookupError:
            try:
                os.kill(pid, signal.SIGTERM)
                signaled = True
            except ProcessLookupError:
                pass
        except PermissionError:
            try:
                os.kill(pid, signal.SIGTERM)
                signaled = True
            except (ProcessLookupError, PermissionError):
                pass
        if signaled:
            terminated += 1
    return terminated


def _terminate_active_worker_unlocked(status: dict[str, Any] | None = None) -> bool:
    worker_key = str((status or _REFRESH_STATUS).get("active_worker_key") or "")
    proc = _REFRESH_ACTIVE_PROCS.pop(worker_key, None) if worker_key else None
    terminated = False
    if proc is not None and proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        terminated = True
    if _terminate_refresh_child_processes():
        terminated = True
    _REFRESH_STATUS.pop("active_worker_key", None)
    _REFRESH_STATUS.pop("active_worker_pid", None)
    return terminated


def _register_active_worker(worker_key: str, proc: mp.Process) -> None:
    with _REFRESH_LOCK:
        _REFRESH_ACTIVE_PROCS[worker_key] = proc
        _REFRESH_STATUS["active_worker_key"] = worker_key
        _REFRESH_STATUS["active_worker_pid"] = proc.pid
        _REFRESH_STATUS["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _persist_refresh_status_unlocked()


def _clear_active_worker(worker_key: str) -> None:
    with _REFRESH_LOCK:
        _REFRESH_ACTIVE_PROCS.pop(worker_key, None)
        if _REFRESH_STATUS.get("active_worker_key") == worker_key:
            _REFRESH_STATUS.pop("active_worker_key", None)
            _REFRESH_STATUS.pop("active_worker_pid", None)
            _REFRESH_STATUS["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _persist_refresh_status_unlocked()


def _is_refresh_status_stale(status: dict[str, Any]) -> bool:
    if status.get("status") not in {"running", "queued"}:
        return False
    started = _parse_refresh_timestamp(status.get("started_at"))
    if started is None:
        return True
    last_progress = _last_refresh_progress_at(status) or started
    now = datetime.now()
    return (
        (now - started).total_seconds() > REFRESH_RUNNING_STALE_SECONDS
        or (now - last_progress).total_seconds() > REFRESH_NO_PROGRESS_STALE_SECONDS
    )


def _expire_stale_refresh_status_unlocked(status: dict[str, Any]) -> dict[str, Any]:
    if not _is_refresh_status_stale(status):
        return status
    killed = _terminate_active_worker_unlocked(status)
    expired = dict(status)
    expired.update(
        {
            "status": "failed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "message": "上一次刷新任务长时间无进展，请重新触发更新",
            "error": (
                f"刷新任务超过 {REFRESH_RUNNING_STALE_SECONDS // 3600} 小时仍未完成，"
                if _last_refresh_progress_at(status) and _parse_refresh_timestamp(status.get('started_at'))
                and (datetime.now() - (_parse_refresh_timestamp(status.get('started_at')) or datetime.now())).total_seconds() > REFRESH_RUNNING_STALE_SECONDS
                else f"刷新任务超过 {REFRESH_NO_PROGRESS_STALE_SECONDS // 60} 分钟无进展，"
            ) + ("已终止卡住的后台节点并标记为过期。" if killed else "已标记为过期。"),
        }
    )
    steps = []
    for step in expired.get("steps") or []:
        step_copy = dict(step)
        if step_copy.get("status") == "running":
            step_copy["status"] = "failed"
        steps.append(step_copy)
    expired["steps"] = steps
    _REFRESH_STATUS.update(expired)
    _persist_refresh_status_unlocked()
    return expired


def _expire_interrupted_refresh_status_unlocked(status: dict[str, Any]) -> dict[str, Any]:
    if status.get("status") not in {"running", "queued"}:
        return status
    _terminate_active_worker_unlocked(status)
    interrupted = dict(status)
    interrupted.update(
        {
            "status": "failed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "message": "上一次刷新任务已随服务重启中断，请重新触发更新",
            "error": "检测到服务重启后仍残留 running/queued 状态；后台线程已不存在，已自动解锁一键更新按钮。",
        }
    )
    steps = []
    for step in interrupted.get("steps") or []:
        step_copy = dict(step)
        if step_copy.get("status") == "running":
            step_copy["status"] = "failed"
        steps.append(step_copy)
    interrupted["steps"] = steps
    _REFRESH_STATUS.update(interrupted)
    _persist_refresh_status_unlocked()
    return interrupted


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _read_daily_payload_cache(path: Path) -> dict[str, Any] | None:
    try:
        return read_json_file(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_daily_payload_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _is_daily_payload_current(payload: dict[str, Any], today: date | None = None) -> bool:
    generated_at = pd.to_datetime(payload.get("generated_at"), errors="coerce")
    if pd.isna(generated_at):
        return False
    return generated_at.date() == (today or date.today())


def _load_project_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_byd_daily_strategy(
    shares: int = 10000,
    cost: float = 110.6061,
    refresh: bool = False,
) -> dict[str, Any]:
    """Compatibility facade for the application-layer BYD workspace."""

    def is_snapshot_current(payload: dict[str, Any]) -> bool:
        expected = str(_local_market_trade_date() or "").replace("-", "")
        planned = payload.get("planned_t") or payload.get("daily_t_plan") or {}
        actual = str(planned.get("signal_date") or "").replace("-", "")
        return not expected or actual == expected

    return build_byd_daily_strategy(
        shares=shares,
        cost=cost,
        refresh=refresh,
        dependencies=BydWorkspaceDependencies(
            read_snapshot=_read_workspace_snapshot,
            write_snapshot=_write_workspace_snapshot,
            is_snapshot_current=is_snapshot_current,
        ),
    )


@lru_cache(maxsize=1)
def _long_research_module():
    path = PROJECT_ROOT / "scripts/research/backtest_long_dividend_quality.py"
    module_name = "quant_long_dividend_quality_research"
    with _LONG_RESEARCH_MODULE_LOCK:
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, "_quant_services_import_complete", False):
            return module
        if module is not None:
            sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载长线策略研究脚本: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if sys.modules.get(module_name) is module:
                sys.modules.pop(module_name, None)
            raise
        module._quant_services_import_complete = True
        return module


@lru_cache(maxsize=1)
def _tea_master_research_module():
    path = PROJECT_ROOT / "scripts/research/backtest_tea_master_long.py"
    module_name = "quant_tea_master_long_research"
    with _TEA_MASTER_MODULE_LOCK:
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, "_quant_services_import_complete", False):
            return module
        if module is not None:
            sys.modules.pop(module_name, None)
        script_dir = str(path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载茶大长线策略脚本: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if sys.modules.get(module_name) is module:
                sys.modules.pop(module_name, None)
            raise
        module._quant_services_import_complete = True
        return module


@lru_cache(maxsize=2)
def _load_live_long_base_full_cached(
    signal_date: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the historical reference path used when no live checkpoint exists."""

    module = _long_research_module()
    requested_end = pd.to_datetime(signal_date) if signal_date else None
    start = module.parse_date("20130101")
    stock_basic = module.load_stock_basic()
    daily_basic, coverage = module.load_daily_basic_monthly(start, requested_end)
    if daily_basic.empty:
        raise RuntimeError("缺少 daily_basic，无法生成长线股票池")
    features, _ = module.load_daily_monthly_features(
        start,
        requested_end,
        stock_basic,
        candidate_symbols=None,
        use_cache=False,
        include_daily_returns=False,
    )
    return features, daily_basic, stock_basic, coverage


@lru_cache(maxsize=2)
def _load_live_long_base_cached(
    signal_date: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build live inputs with enough monthly history for valuation percentiles."""

    module = _long_research_module()
    requested_end = pd.to_datetime(signal_date) if signal_date else pd.Timestamp.now().normalize()
    basic_start = requested_end - pd.DateOffset(months=LONG_LIVE_DAILY_BASIC_LOOKBACK_MONTHS)
    stock_basic = module.load_stock_basic()
    daily_basic, coverage = module.load_daily_basic_monthly(basic_start, requested_end)
    if daily_basic.empty:
        raise RuntimeError("缺少 daily_basic，无法生成长线股票池")
    rebalance_dates = sorted(pd.to_datetime(daily_basic["date"].dropna().unique()))
    if not rebalance_dates:
        raise RuntimeError("daily_basic 没有可用调仓截面")
    price_start = pd.Timestamp(rebalance_dates[0])
    latest_rebalance = pd.Timestamp(rebalance_dates[-1])
    candidate_symbols: set[str] | None = None
    if hasattr(module, "filter_daily_basic_point_in_time"):
        live_universe_config = type(
            "LiveLongUniverseConfig",
            (),
            {
                "variant": "tea",
                "prefilter_min_dv_ttm": 0.0,
                "prefilter_min_total_mv": 800000.0,
                "prefilter_min_circ_mv": 500000.0,
            },
        )()
        latest_basic = daily_basic[daily_basic["date"] == latest_rebalance].copy()
        latest_basic = module.filter_daily_basic_point_in_time(
            latest_basic,
            live_universe_config,
        )
        candidate_symbols = set(latest_basic["ts_code"].dropna().astype(str))
    history_features = pd.DataFrame()
    module_root = getattr(module, "PROJECT_ROOT", None)
    if module_root is not None:
        full_cache_key = hashlib.sha1(
            "20130101|none|qfq_ohlc_price_v1|".encode("utf-8")
        ).hexdigest()[:16]
        full_cache_path = (
            Path(module_root)
            / "data/research/long_dividend_quality"
            / f"daily_monthly_features_{full_cache_key}.parquet"
        )
        if full_cache_path.exists():
            history_features = pd.read_parquet(full_cache_path)
            history_features["date"] = pd.to_datetime(history_features["date"])
            history_features = history_features[
                (history_features["date"] >= price_start)
                & (history_features["date"] <= requested_end)
            ].copy()
            if candidate_symbols is not None:
                history_features = history_features[
                    history_features["ts_code"].astype(str).isin(candidate_symbols)
                ].copy()

    cached_last_date = (
        pd.to_datetime(history_features["date"].max())
        if not history_features.empty
        else pd.NaT
    )
    if pd.isna(cached_last_date) or cached_last_date < latest_rebalance:
        # Historical percentiles come from the reusable month-end cache. Only
        # compute the missing latest section from daily bars (the loader keeps
        # its own 450-day technical warm-up), instead of rescanning six years.
        recent_start = latest_rebalance if not history_features.empty else price_start
        recent_features, _ = module.load_daily_monthly_features(
            recent_start,
            requested_end,
            stock_basic,
            candidate_symbols=candidate_symbols,
            use_cache=True,
            include_daily_returns=False,
        )
        features = pd.concat([history_features, recent_features], ignore_index=True, sort=False)
    else:
        features = history_features
    feature_sort_columns = [
        column for column in ["date", "ts_code", "trade_date"] if column in features.columns
    ]
    features = (
        features.sort_values(feature_sort_columns)
        .drop_duplicates(["date", "ts_code"], keep="last")
        .reset_index(drop=True)
    )
    daily_basic = daily_basic[daily_basic["date"].isin(rebalance_dates)].copy()
    coverage = dict(coverage)
    coverage.update(
        {
            "live_windowed": True,
            "price_history_days": 450,
            "daily_basic_lookback_months": LONG_LIVE_DAILY_BASIC_LOOKBACK_MONTHS,
            "live_rebalance_dates": [pd.Timestamp(item).date().isoformat() for item in rebalance_dates],
            "live_feature_rows": int(len(features)),
            "live_daily_basic_rows": int(len(daily_basic)),
            "live_candidate_symbols": (
                len(candidate_symbols) if candidate_symbols is not None else None
            ),
            "live_history_cache_last_date": (
                pd.Timestamp(cached_last_date).date().isoformat()
                if not pd.isna(cached_last_date)
                else None
            ),
            "live_recent_feature_start": (
                recent_start.date().isoformat()
                if "recent_start" in locals()
                else None
            ),
        }
    )
    return features, daily_basic, stock_basic, coverage


def _load_live_long_base(
    signal_date: str | None,
    *,
    full_history: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    normalized_date = pd.to_datetime(signal_date).date().isoformat() if signal_date else None
    with _LONG_LIVE_DATA_LOCK:
        loader = _load_live_long_base_full_cached if full_history else _load_live_long_base_cached
        features, daily_basic, stock_basic, coverage = loader(normalized_date)
    return features, daily_basic, stock_basic, dict(coverage)


def _require_exact_regime_row(
    frame: pd.DataFrame,
    target_date: str | pd.Timestamp,
    *,
    context: str,
    required_columns: list[str] | tuple[str, ...],
) -> pd.Series:
    """Return a complete regime row for the exact decision date.

    Current inference must never substitute a previous observation or a
    synthetic neutral value when a market/industry context feature is stale.
    Historical rows may still retain their documented neutral semantics.
    """

    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        raise RuntimeError(f"{context} has an invalid decision date: {target_date!r}")
    target = pd.Timestamp(target).normalize()
    if frame.empty or "date" not in frame.columns:
        raise RuntimeError(
            f"{context} has no exact decision date observation for "
            f"{target.date().isoformat()}"
        )
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    exact = normalized.loc[normalized["date"].eq(target)]
    if exact.empty:
        latest = normalized["date"].dropna().max()
        latest_text = latest.date().isoformat() if pd.notna(latest) else "none"
        raise RuntimeError(
            f"{context} has no exact decision date observation for "
            f"{target.date().isoformat()}; latest={latest_text}"
        )
    missing_columns = [column for column in required_columns if column not in exact.columns]
    if missing_columns:
        raise RuntimeError(f"{context} missing required columns: {missing_columns}")
    row = exact.iloc[-1]
    missing_values = [column for column in required_columns if pd.isna(row[column])]
    if missing_values:
        raise RuntimeError(
            f"{context} missing required values on {target.date().isoformat()}: "
            f"{missing_values}"
        )
    return row


def _prepare_tea_master_live_data(
    module: Any,
    signal_date: str | None,
    *,
    full_history: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    daily_features, daily_basic, stock_basic, coverage = _load_live_long_base(
        signal_date,
        full_history=full_history,
    )
    executable_start_text = max("20130101", coverage.get("first_trade_date") or "20130101")
    executable_start = module.parse_date(executable_start_text)
    daily_features = daily_features[daily_features["date"] >= executable_start].copy()
    daily_basic = daily_basic[daily_basic["date"] >= executable_start].copy()
    before_rows = len(daily_basic)
    live_config = type(
        "LiveTeaConfig",
        (),
        {
            "variant": "tea",
            "prefilter_min_dv_ttm": 0.0,
            "prefilter_min_total_mv": 800000.0,
            "prefilter_min_circ_mv": 500000.0,
        },
    )()
    daily_basic = module.filter_daily_basic_point_in_time(daily_basic, live_config)
    coverage["point_in_time_universe"] = True
    coverage["point_in_time_universe_rows_before"] = int(before_rows)
    coverage["point_in_time_universe_rows_after"] = int(len(daily_basic))
    merged = daily_features.merge(
        daily_basic.drop(columns=["trade_date"]),
        on=["date", "ts_code"],
        how="inner",
    )
    merged = module.load_financial_asof(merged)
    merged = module.add_empty_analyst_forecast_columns(merged)
    decision_date = pd.to_datetime(merged["date"], errors="coerce").max()
    market_regime = module.load_market_regime(merged["date"].min(), decision_date)
    regime_row = _require_exact_regime_row(
        market_regime,
        decision_date,
        context="tea live market regime",
        required_columns=LONG_MARKET_REGIME_FEATURE_COLUMNS,
    )
    coverage["market_regime_feature_date"] = pd.Timestamp(regime_row["date"]).date().isoformat()
    merged = merged.merge(market_regime, on="date", how="left")
    merged["market_regime"] = merged["market_regime"].fillna("neutral")
    return merged, stock_basic, coverage


@lru_cache(maxsize=2)
def _tea_master_live_scores_cached(
    signal_date: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the shared Tea/Tea-safe score frame once per decision date."""

    module = _tea_master_research_module()
    merged, _, coverage = _prepare_tea_master_live_data(module, signal_date)
    scored = module.build_tea_scores(
        merged,
        valuation_window_months=LONG_VALUATION_WINDOW_MONTHS,
        valuation_minimum_months=LONG_VALUATION_MINIMUM_MONTHS,
    )
    return scored, dict(coverage)


def _tea_master_live_scores(
    signal_date: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized_date = (
        pd.to_datetime(signal_date).date().isoformat() if signal_date else None
    )
    # functools.lru_cache is coherent but does not deduplicate concurrent
    # misses. Tea and Tea-safe start together, so serialize only the first
    # expensive build and let the second caller take the completed cache hit.
    with _TEA_MASTER_LIVE_SCORE_LOCK:
        scored, coverage = _tea_master_live_scores_cached(normalized_date)
    return scored, dict(coverage)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return numeric


def _safe_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).lower() in {"1", "true", "yes", "y"}


def _long_state_path(variant: str) -> Path:
    return LONG_STATE_DIR / f"{variant}_latest_states.json"


def _load_long_previous_states(variant: str) -> dict[str, dict[str, Any]]:
    path = _long_state_path(variant)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("stocks", []) if isinstance(payload, dict) else []
    return {
        str(row.get("ts_code")): row
        for row in rows
        if isinstance(row, dict) and row.get("ts_code")
    }


def _persist_long_states(variant: str, payload: dict[str, Any]) -> None:
    LONG_STATE_DIR.mkdir(parents=True, exist_ok=True)
    _long_state_path(variant).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _long_snapshot_key(variant: str, signal_date: str | None) -> tuple[str, str]:
    date_key = str(signal_date or "latest")
    raw = f"{LONG_STOCK_POOL_SCHEMA_VERSION}|{variant}|{date_key}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24], date_key


def _long_snapshot_path(snapshot_key: str) -> Path:
    return LONG_STOCK_POOL_SNAPSHOT_DIR / f"{snapshot_key}.json"


def _read_long_stock_pool_snapshot(
    variant: str,
    signal_date: str | None,
    allow_sql: bool = True,
) -> dict[str, Any] | None:
    snapshot_key, _ = _long_snapshot_key(variant, signal_date)
    requested = _canonical_workspace_snapshot_date(signal_date) if signal_date else None
    local_candidates: list[tuple[str, Path]] = []
    exact_path = _long_snapshot_path(snapshot_key)
    if exact_path.exists():
        local_candidates.append((requested or "", exact_path))
    if LONG_STOCK_POOL_SNAPSHOT_DIR.exists():
        for path in LONG_STOCK_POOL_SNAPSHOT_DIR.glob("*.json"):
            if path == exact_path:
                continue
            try:
                candidate_payload = read_json_file(path)
            except Exception:
                continue
            if candidate_payload.get("variant") != variant:
                continue
            candidate_date = _canonical_workspace_snapshot_date(candidate_payload.get("signal_date"))
            if requested and candidate_date > requested:
                continue
            local_candidates.append((candidate_date, path))
    local_candidates.sort(key=lambda item: item[0], reverse=True)
    for candidate_date, path in local_candidates:
        try:
            payload = read_json_file(path)
        except Exception:
            continue
        if payload.get("schema_version") != LONG_STOCK_POOL_SCHEMA_VERSION:
            continue
        cached_date = _canonical_workspace_snapshot_date(payload.get("signal_date") or candidate_date)
        payload["cache"] = {
            "hit": True,
            "backend": "filesystem",
            "snapshot_key": path.stem,
            "snapshot_date": cached_date,
            "requested_date": requested,
            "stale": bool(requested and cached_date != requested),
        }
        return payload
    if not allow_sql:
        return None
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    if store.config.sql_url:
        try:
            from sqlalchemy import text

            with store._engine().begin() as conn:
                row = conn.execute(
                    text(f"SELECT payload_json FROM {LONG_STOCK_POOL_SNAPSHOT_TABLE} WHERE snapshot_key = :snapshot_key"),
                    {"snapshot_key": snapshot_key},
                ).mappings().first()
            if row and row.get("payload_json"):
                payload = json.loads(row["payload_json"])
                payload["cache"] = {"hit": True, "backend": "mysql", "snapshot_key": snapshot_key}
                return payload
        except Exception:
            pass

    return None


def _long_stock_pool_snapshot_dates(variant: str) -> set[str]:
    dates: set[str] = set()
    if LONG_STOCK_POOL_SNAPSHOT_DIR.exists():
        for path in LONG_STOCK_POOL_SNAPSHOT_DIR.glob("*.json"):
            try:
                payload = read_json_file(path)
            except Exception:
                continue
            if payload.get("variant") == variant and payload.get("signal_date"):
                dates.add(str(payload["signal_date"]))

    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    if store.config.sql_url:
        try:
            from sqlalchemy import text

            with store._engine().begin() as conn:
                rows = conn.execute(
                    text(
                        f"""
                        SELECT signal_date
                        FROM {LONG_STOCK_POOL_SNAPSHOT_TABLE}
                        WHERE variant = :variant
                        """
                    ),
                    {"variant": variant},
                ).mappings().all()
            for row in rows:
                if row.get("signal_date"):
                    dates.add(str(row["signal_date"]))
        except Exception:
            pass
    return dates


def _write_long_stock_pool_snapshot(
    payload: dict[str, Any],
    variant: str,
    signal_date: str | None,
    write_sql: bool = True,
) -> None:
    snapshot_key, date_key = _long_snapshot_key(variant, signal_date)
    payload_to_store = dict(payload)
    payload_to_store["schema_version"] = LONG_STOCK_POOL_SCHEMA_VERSION
    payload_to_store["cache"] = {"hit": False, "backend": "generated", "snapshot_key": snapshot_key}
    payload_json = json.dumps(payload_to_store, ensure_ascii=False, default=str)
    LONG_STOCK_POOL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = _long_snapshot_path(snapshot_key)
    temporary_path = snapshot_path.with_suffix(".json.tmp")
    temporary_path.write_text(payload_json, encoding="utf-8")
    temporary_path.replace(snapshot_path)
    if not write_sql:
        return
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    if store.config.sql_url:
        try:
            from sqlalchemy import text

            with store._engine().begin() as conn:
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS {LONG_STOCK_POOL_SNAPSHOT_TABLE} (
                            snapshot_key VARCHAR(64) PRIMARY KEY,
                            variant VARCHAR(32) NOT NULL,
                            signal_date VARCHAR(16) NOT NULL,
                            generated_at VARCHAR(32) NOT NULL,
                            stock_count INT NOT NULL,
                            payload_json LONGTEXT NOT NULL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        f"""
                        INSERT INTO {LONG_STOCK_POOL_SNAPSHOT_TABLE}
                            (snapshot_key, variant, signal_date, generated_at, stock_count, payload_json)
                        VALUES
                            (:snapshot_key, :variant, :signal_date, :generated_at, :stock_count, :payload_json)
                        ON DUPLICATE KEY UPDATE
                            generated_at = VALUES(generated_at),
                            stock_count = VALUES(stock_count),
                            payload_json = VALUES(payload_json)
                        """
                    ),
                    {
                        "snapshot_key": snapshot_key,
                        "variant": variant,
                        "signal_date": date_key,
                        "generated_at": str(payload.get("generated_at") or datetime.now().isoformat(timespec="seconds")),
                        "stock_count": len(payload.get("stocks") or []),
                        "payload_json": payload_json,
                    },
                )
        except Exception:
            return


def _workspace_params_key(params: dict[str, Any] | None = None) -> str:
    return workspace_params_key(params)


def _canonical_workspace_snapshot_date(value: Any) -> str:
    return canonical_snapshot_date(value)


def _workspace_snapshot_repository() -> WorkspaceSnapshotRepository:
    return WorkspaceSnapshotRepository(
        directory=WEB_WORKSPACE_SNAPSHOT_DIR,
        table_name=WEB_WORKSPACE_SNAPSHOT_TABLE,
        schema_version=WEB_WORKSPACE_SNAPSHOT_SCHEMA_VERSION,
        store_factory=lambda: MarketDataStore(
            MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data")
        ),
    )


def _workspace_snapshot_key(workspace: str, snapshot_date: str, params_key: str) -> str:
    return _workspace_snapshot_repository().snapshot_key(workspace, snapshot_date, params_key)


def _workspace_snapshot_file_path(workspace: str, params_key: str, snapshot_date: str) -> Path:
    return _workspace_snapshot_repository().file_path(workspace, params_key, snapshot_date)


def _read_filesystem_workspace_snapshot(
    workspace: str,
    snapshot_date: str | None,
    params_key: str,
) -> dict[str, Any] | None:
    return _workspace_snapshot_repository().read_filesystem(
        workspace,
        snapshot_date,
        params_key,
    )


def _read_workspace_snapshot(
    workspace: str,
    snapshot_date: str | None = None,
    params: dict[str, Any] | None = None,
    allow_sql: bool = True,
) -> dict[str, Any] | None:
    return _workspace_snapshot_repository().read(
        workspace,
        snapshot_date=snapshot_date,
        params=params,
        allow_sql=allow_sql,
    )


def _write_workspace_snapshot(
    workspace: str,
    snapshot_date: str | None,
    payload: dict[str, Any],
    params: dict[str, Any] | None = None,
    write_sql: bool = True,
) -> None:
    _workspace_snapshot_repository().write(
        workspace,
        snapshot_date,
        payload,
        params=params,
        write_sql=write_sql,
    )


def _write_filesystem_workspace_snapshot(
    workspace: str,
    params_key: str,
    snapshot_date: str,
    payload_json: str,
) -> None:
    _workspace_snapshot_repository().write_filesystem(
        workspace,
        params_key,
        snapshot_date,
        payload_json,
    )


@lru_cache(maxsize=4)
def _chan_model_strategy_dates() -> set[str]:
    source = CHAN_MODEL_STRATEGY_SCORED_PATH if CHAN_MODEL_STRATEGY_SCORED_PATH.exists() else CHAN_MODEL_SCORED_PATH
    if not source.exists():
        return set()
    try:
        frame = pd.read_parquet(source)
        if "chan_model_signal" not in frame.columns:
            frame = add_chan_model_strategy_columns(frame)
        selected = frame[frame["chan_model_signal"].eq(1)].copy()
        if selected.empty:
            return set()
        dates = pd.to_datetime(selected["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return set(dates.dropna().tolist())
    except Exception:
        return set()


def _workspace_snapshot_dates(workspace: str, params: dict[str, Any] | None = None) -> set[str]:
    return _workspace_snapshot_repository().dates(workspace, params=params)


def _long_entry_tranche(row: pd.Series, target_weight: float, regime: str) -> tuple[float, str]:
    close = _safe_float(row.get("close"))
    target_price = _safe_float(row.get("target_price"))
    sleeve = str(row.get("sleeve", "core") or "core")
    long_score = _safe_float(row.get("long_score"), 0.0) or 0.0
    growth_score = _safe_float(row.get("growth_rank_score", row.get("growth_score")), 0.0) or 0.0

    if sleeve == "growth":
        tranche = 0.45 if regime == "risk_on" else 0.0
    elif regime == "risk_on":
        tranche = 0.60
    elif regime == "risk_off":
        tranche = 0.35
    else:
        tranche = 0.50

    price_note = "按趋势分批建仓"
    if close and target_price and target_price > 0:
        ratio = close / target_price
        if ratio <= 0.97:
            tranche += 0.30
            price_note = "价格低于目标区，首批可更积极"
        elif ratio <= 1.00:
            tranche += 0.20
            price_note = "价格接近目标区，按计划分批"
        elif ratio >= (1.06 if sleeve == "growth" else 1.08):
            tranche = min(tranche, 0.25 if sleeve == "growth" else 0.30)
            price_note = "价格偏高，只给观察/小仓"

    if sleeve == "growth" and growth_score >= 85:
        tranche += 0.10
    elif sleeve != "growth" and long_score >= 82:
        tranche += 0.10

    tranche = float(np.clip(tranche, 0.20, 1.0))
    return round(target_weight * tranche, 4), price_note


def _long_price_levels(module: Any, row: pd.Series, regime: str) -> dict[str, Any]:
    target_price = _safe_float(row.get("target_price"))
    close = _safe_float(row.get("close"))
    ma20 = _safe_float(row.get("ma_20"))
    ma60 = _safe_float(row.get("ma_60"))
    ma120 = _safe_float(row.get("ma_120"))
    sleeve = str(row.get("sleeve", "core") or "core")

    levels: dict[str, Any] = {
        "current_price": round(close, 2) if close is not None else None,
        "entry_target_price": round(target_price, 2) if target_price is not None else None,
        "entry_aggressive_price": round(target_price * 0.97, 2) if target_price is not None else None,
        "entry_small_position_price": round(target_price * (1.06 if sleeve == "growth" else 1.08), 2) if target_price is not None else None,
        "reduce_ma60_price": round(ma60, 2) if ma60 is not None else None,
        "exit_ma120_price": round(ma120 * (0.94 if regime == "risk_on" else 0.98), 2) if ma120 is not None else None,
    }
    if ma20 is None or ma60 is None or ma120 is None:
        levels.update(
            {
                "t_buy_min_price": None,
                "t_buy_max_price": None,
                "t_sell_trigger_price": None,
                "t_buy_text": "均线不足",
                "t_sell_text": "均线不足",
            }
        )
        return levels

    params = module.style_grid_parameters(row.to_dict(), row, regime)
    t_buy_min = ma120 * 0.98
    t_buy_max = max(ma20 * float(params["buy_ma20"]), ma60 * float(params["buy_ma60"]))
    t_sell_trigger = min(ma20 * float(params["sell_ma20"]), ma60 * float(params["sell_ma60"]))
    t_buy_text = f"{t_buy_min:.2f}-{t_buy_max:.2f}" if t_buy_min <= t_buy_max else "无有效低吸区"
    levels.update(
        {
            "t_buy_min_price": round(t_buy_min, 2),
            "t_buy_max_price": round(t_buy_max, 2),
            "t_sell_trigger_price": round(t_sell_trigger, 2),
            "t_buy_text": t_buy_text,
            "t_sell_text": f">= {t_sell_trigger:.2f}",
        }
    )
    return levels


def _long_t_action(module: Any, row: pd.Series, regime: str) -> tuple[str, str, str]:
    close = _safe_float(row.get("close"))
    ma20 = _safe_float(row.get("ma_20"))
    ma60 = _safe_float(row.get("ma_60"))
    ma120 = _safe_float(row.get("ma_120"))
    if close is None or ma20 is None or ma60 is None or ma120 is None:
        return "HOLD", "数据不足", "缺少均线数据，暂不做T"
    params = module.style_grid_parameters(row.to_dict(), row, regime)
    sell_signal = close > ma20 * float(params["sell_ma20"]) or close > ma60 * float(params["sell_ma60"])
    buy_signal = (
        regime != "risk_off"
        and close >= ma120 * 0.98
        and (close <= ma20 * float(params["buy_ma20"]) or close <= ma60 * float(params["buy_ma60"]))
    )
    if str(row.get("sleeve", "core")) == "growth":
        buy_signal = buy_signal and (_safe_float(row.get("trend_score"), 0.0) or 0.0) >= 60
    if sell_signal:
        return "T_SELL", str(params["profile"]), "价格偏离短中期均线，允许高抛回到核心仓"
    if buy_signal:
        return "T_BUY", str(params["profile"]), "价格回到低吸区，允许补回机动仓"
    return "HOLD", str(params["profile"]), "未触发网格/做T阈值"


def _long_exit_reason(row: pd.Series, regime: str) -> tuple[bool, str]:
    long_score = _safe_float(row.get("long_score"), 0.0) or 0.0
    quality_score = _safe_float(row.get("quality_score"), 0.0) or 0.0
    risk_score = _safe_float(row.get("risk_score"), 0.0) or 0.0
    close = _safe_float(row.get("close"))
    ma120 = _safe_float(row.get("ma_120"))
    if long_score < 58:
        return True, "长期评分跌破 58，基本面/趋势综合失效"
    if close is not None and ma120 is not None and close < ma120 * (0.94 if regime == "risk_on" else 0.98):
        return True, "价格跌破 MA120 风控线"
    if regime == "risk_off" and quality_score < 55:
        return True, "熊市中质量评分不足"
    if regime == "risk_off" and risk_score < 35:
        return True, "熊市中波动/下行风险过高"
    return False, ""


def _long_reduce_reason(row: pd.Series, regime: str) -> tuple[bool, str]:
    long_score = _safe_float(row.get("long_score"), 0.0) or 0.0
    close = _safe_float(row.get("close"))
    ma60 = _safe_float(row.get("ma_60"))
    pullback = _safe_bool(row.get("market_pullback_warning"))
    negative = _safe_bool(row.get("analyst_negative_warning"))
    if long_score < 68:
        return True, "长期评分回落，进入降仓观察"
    if close is not None and ma60 is not None and close < ma60:
        return True, "价格跌破 MA60，趋势转弱"
    if regime == "risk_off" or pullback:
        return True, "市场风险升高，降低风险暴露"
    if negative:
        return True, "券商预测出现负面修正警告"
    return False, ""


def _long_status_row(
    module: Any,
    row: pd.Series,
    *,
    variant: str,
    selected: bool,
    target_weight: float,
    previous: dict[str, Any] | None,
    rank: int | None,
) -> dict[str, Any]:
    ts_code = str(row.get("ts_code"))
    regime = str(row.get("market_regime", "neutral") or "neutral")
    prev_state = str((previous or {}).get("state") or "WATCH")
    long_score = _safe_float(row.get("long_score"), 0.0) or 0.0
    trend_score = _safe_float(row.get("trend_score"), 0.0) or 0.0
    ma120_slope = _safe_float(row.get("ma_120_slope_20d"), 0.0) or 0.0
    close = _safe_float(row.get("close"))
    ma120 = _safe_float(row.get("ma_120"))
    target_price = _safe_float(row.get("target_price"))
    target_upside = None
    if close and target_price:
        target_upside = target_price / close - 1.0

    exit_signal, exit_reason = _long_exit_reason(row, regime)
    reduce_signal, reduce_reason = _long_reduce_reason(row, regime)
    t_action, t_profile, t_reason = _long_t_action(module, row, regime)
    price_levels = _long_price_levels(module, row, regime)
    entry_weight, entry_note = _long_entry_tranche(row, target_weight, regime)

    if exit_signal:
        state = "EXIT"
        action = "清仓"
        reason = exit_reason
        target_weight = 0.0
        entry_weight = 0.0
    elif selected:
        if prev_state in {"WATCH", "EXIT", "COOLDOWN"} and entry_weight < target_weight * 0.85:
            state = "BUILDING"
            action = "分批建仓"
            reason = entry_note
        elif reduce_signal:
            state = "REDUCE"
            action = "降仓观察"
            reason = reduce_reason
        elif t_action in {"T_SELL", "T_BUY"}:
            state = "T_ACTIVE"
            action = "高抛" if t_action == "T_SELL" else "低吸"
            reason = t_reason
        else:
            state = "CORE"
            action = "核心持有"
            reason = "核心条件仍成立，按目标仓位持有"
    else:
        watch_condition = (
            long_score >= 72
            and close is not None
            and ma120 is not None
            and close > ma120
            and ma120_slope >= -0.005
        )
        if reduce_signal and prev_state in {"CORE", "T_ACTIVE", "BUILDING", "REDUCE"}:
            state = "REDUCE"
            action = "降仓观察"
            reason = reduce_reason
        elif watch_condition and long_score >= 80 and trend_score >= 65:
            state = "BUILDING"
            action = "候选建仓"
            reason = "长期评分和趋势满足建仓观察条件，等待更合适价格或席位"
        elif watch_condition:
            state = "WATCH"
            action = "观察"
            reason = "进入观察池，等待评分/趋势/价格进一步确认"
        else:
            state = "WATCH"
            action = "弱观察"
            reason = "未进入目标组合，仅保留低优先级观察"

    return {
        "ts_code": ts_code,
        "name": row.get("name"),
        "industry": row.get("industry"),
        "variant": variant,
        "rank": rank,
        "state": state,
        "previous_state": prev_state,
        "action": action,
        "reason": reason,
        "sleeve": row.get("sleeve", "core"),
        "t_action": t_action,
        "t_profile": t_profile,
        "t_reason": t_reason,
        "market_regime": regime,
        "target_weight": round(float(target_weight), 4),
        "first_tranche_weight": round(float(entry_weight), 4),
        "close": close,
        "target_price": target_price,
        "target_upside": round(target_upside, 4) if target_upside is not None else None,
        "price_levels": price_levels,
        "long_score": round(long_score, 2),
        "growth_score": round(_safe_float(row.get("growth_score"), 0.0) or 0.0, 2),
        "forecast_core_rank_score": round(_safe_float(row.get("forecast_core_rank_score"), long_score) or long_score, 2),
        "quality_score": round(_safe_float(row.get("quality_score"), 0.0) or 0.0, 2),
        "value_score": round(_safe_float(row.get("value_score"), 0.0) or 0.0, 2),
        "trend_score": round(trend_score, 2),
        "risk_score": round(_safe_float(row.get("risk_score"), 0.0) or 0.0, 2),
        "dividend_score": round(_safe_float(row.get("dividend_score"), 0.0) or 0.0, 2),
        "dv_ttm": round(_safe_float(row.get("dv_ttm"), 0.0) or 0.0, 2),
        "pe_ttm": round(_safe_float(row.get("pe_ttm"), 0.0) or 0.0, 2),
        "pb": round(_safe_float(row.get("pb"), 0.0) or 0.0, 2),
        "analyst_report_count_180d": int(_safe_float(row.get("analyst_report_count_180d"), 0.0) or 0),
        "analyst_org_count_180d": int(_safe_float(row.get("analyst_org_count_180d"), 0.0) or 0),
        "analyst_institution_count_180d": int(_safe_float(row.get("analyst_institution_count_180d"), 0.0) or 0),
        "analyst_research_report_count_180d": int(_safe_float(row.get("analyst_research_report_count_180d"), 0.0) or 0),
        "analyst_consensus_report_count_180d": int(_safe_float(row.get("analyst_consensus_report_count_180d"), 0.0) or 0),
        "analyst_forward_years_180d": int(_safe_float(row.get("analyst_forward_years_180d"), 0.0) or 0),
        "analyst_forward_growth_score": round(_safe_float(row.get("analyst_forward_growth_score"), 0.0) or 0.0, 2),
        "analyst_target_upside_180d": round(_safe_float(row.get("analyst_target_upside_180d"), 0.0) or 0.0, 4),
        "analyst_negative_warning": _safe_bool(row.get("analyst_negative_warning")),
        "close_above_ma120": bool(close is not None and ma120 is not None and close > ma120),
        "ma_120_slope_20d": round(ma120_slope, 4),
    }


def _tea_entry_tranche(row: pd.Series, target_weight: float, regime: str) -> tuple[float, str]:
    close = _safe_float(row.get("close"))
    target_price = _safe_float(row.get("target_price"))
    tranche = 0.62 if regime == "risk_on" else (0.45 if regime == "risk_off" else 0.52)
    note = "按茶大长线规则分批建仓"
    if close and target_price:
        ratio = close / target_price
        if ratio <= 0.98:
            tranche += 0.20
            note = "价格在目标区下沿，首批可更积极"
        elif ratio >= 1.08:
            tranche = min(tranche, 0.30)
            note = "价格高于目标区，只给观察/小仓"
    return round(float(target_weight) * float(np.clip(tranche, 0.20, 1.0)), 4), note


def _tea_price_levels(row: pd.Series) -> dict[str, Any]:
    close = _safe_float(row.get("close"))
    target_price = _safe_float(row.get("target_price"))
    ma20 = _safe_float(row.get("ma_20"))
    ma60 = _safe_float(row.get("ma_60"))
    ma120 = _safe_float(row.get("ma_120"))
    t_buy_min = ma120 * 0.98 if ma120 is not None else None
    t_buy_max = max(ma20 * 1.015, ma60 * 1.035) if ma20 is not None and ma60 is not None else None
    t_sell_trigger = min(ma20 * 1.12, ma60 * 1.18) if ma20 is not None and ma60 is not None else None
    return {
        "current_price": round(close, 2) if close is not None else None,
        "entry_target_price": round(target_price, 2) if target_price is not None else None,
        "entry_aggressive_price": round(target_price * 0.98, 2) if target_price is not None else None,
        "entry_small_position_price": round(target_price * 1.08, 2) if target_price is not None else None,
        "reduce_ma60_price": round(ma60, 2) if ma60 is not None else None,
        "exit_ma120_price": round(ma120 * 0.96, 2) if ma120 is not None else None,
        "t_buy_min_price": round(t_buy_min, 2) if t_buy_min is not None else None,
        "t_buy_max_price": round(t_buy_max, 2) if t_buy_max is not None else None,
        "t_sell_trigger_price": round(t_sell_trigger, 2) if t_sell_trigger is not None else None,
        "t_buy_text": f"{t_buy_min:.2f}-{t_buy_max:.2f}" if t_buy_min is not None and t_buy_max is not None and t_buy_min <= t_buy_max else "无有效低吸区",
        "t_sell_text": f">= {t_sell_trigger:.2f}" if t_sell_trigger is not None else "均线不足",
    }


def _attach_analyst_forecast_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    module = _long_research_module()
    out = frame.drop(columns=[col for col in frame.columns if col.startswith("analyst_")], errors="ignore")
    try:
        enriched = module.load_analyst_forecast_asof(out)
    except Exception:
        return module.add_empty_analyst_forecast_columns(out)

    scored: list[pd.DataFrame] = []
    growth_columns = [
        "analyst_forward_eps_growth_180d",
        "analyst_forward_revenue_growth_180d",
        "analyst_forward_net_profit_growth_180d",
    ]
    for _, group in enriched.groupby("date", sort=False):
        group = group.copy()
        has_growth_metrics = group[growth_columns].notna().any(axis=1)
        growth_score = (
            module.percentile_score(group["analyst_forward_eps_growth_180d"].clip(-0.5, 1.5), True) * 0.35
            + module.percentile_score(group["analyst_forward_revenue_growth_180d"].clip(-0.5, 1.5), True) * 0.30
            + module.percentile_score(group["analyst_forward_net_profit_growth_180d"].clip(-0.8, 2.0), True) * 0.25
            + module.percentile_score(group["analyst_report_count_180d"], True) * 0.10
        )
        group["analyst_forward_growth_score"] = growth_score.where(has_growth_metrics)
        scored.append(group)
    return pd.concat(scored, ignore_index=True) if scored else enriched


def _publish_long_factor_snapshot(frame: pd.DataFrame, signal_ts: pd.Timestamp) -> dict[str, Any]:
    """Publish the exact point-in-time factor cross-section used by the long page."""

    if frame.empty:
        raise RuntimeError("长线因子截面为空，拒绝发布页面股票池")
    required_columns = tuple(
        dict.fromkeys((*LONG_FACTOR_REQUIRED_COLUMNS, *LONG_PRODUCTION_FACTOR_COLUMNS))
    )
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"长线因子截面缺少必需字段: {', '.join(missing)}")

    signal_date = pd.to_datetime(signal_ts).date().isoformat()
    snapshot = frame.copy()
    snapshot["date"] = pd.to_datetime(snapshot["date"], errors="coerce")
    snapshot = snapshot[snapshot["date"].dt.date.astype(str) == signal_date].copy()
    if snapshot.empty:
        raise RuntimeError(f"长线因子截面未覆盖目标日期: {signal_date}")
    if snapshot["ts_code"].astype(str).duplicated().any():
        raise RuntimeError(f"长线因子截面存在重复股票: {signal_date}")

    identity_columns = ["date", "ts_code", "name", "industry", "close"]
    factor_columns = list(LONG_PRODUCTION_FACTOR_COLUMNS)
    columns = list(dict.fromkeys([*identity_columns, *factor_columns]))
    snapshot = snapshot[[column for column in columns if column in snapshot.columns]].copy()
    snapshot["factor_schema_version"] = LONG_FACTOR_SNAPSHOT_SCHEMA_VERSION
    snapshot = snapshot.sort_values("ts_code").reset_index(drop=True)

    dated_path = LONG_FACTOR_SNAPSHOT_DIR / f"{signal_date.replace('-', '')}.parquet"
    latest_path = LONG_FACTOR_SNAPSHOT_DIR / "latest.parquet"
    manifest_path = LONG_FACTOR_SNAPSHOT_DIR / "latest.json"
    atomic_write_parquet(snapshot, dated_path, index=False)
    atomic_write_parquet(snapshot, latest_path, index=False)
    manifest = {
        "status": "success",
        "factor_schema_version": LONG_FACTOR_SNAPSHOT_SCHEMA_VERSION,
        "signal_date": signal_date,
        "rows": int(len(snapshot)),
        "factor_count": int(len(factor_columns)),
        "expected_factor_count": int(len(LONG_PRODUCTION_FACTOR_COLUMNS)),
        "missing_factors": [],
        "coverage_status": "complete",
        "source": "long_page_point_in_time_cross_section",
        "dated_path": str(dated_path),
        "latest_path": str(latest_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(manifest, manifest_path)
    return {**manifest, "manifest_path": str(manifest_path)}


def _long_good_stock_assessment(row: pd.Series) -> dict[str, Any]:
    """Evaluate business quality without using valuation or price trend."""

    name = str(row.get("name") or "").strip()
    industry = str(row.get("industry") or "")
    eligible_name = not bool(re.search(r"ST|退", name, re.IGNORECASE))
    is_financial = bool(re.search(r"银行|证券|保险|多元金融|金融服务", industry))
    score = _safe_float(row.get("good_stock_score"), 0.0) or 0.0
    profitability = _safe_float(row.get("profitability_score"), 0.0) or 0.0
    growth = _safe_float(row.get("fundamental_growth_score"), 0.0) or 0.0
    balance = _safe_float(row.get("balance_sheet_score"), 0.0) or 0.0
    stability = _safe_float(row.get("business_stability_score"), 0.0) or 0.0
    coverage = _safe_float(row.get("good_stock_data_coverage"), 0.0) or 0.0
    listing_years = _safe_float(row.get("listing_years"), 0.0) or 0.0
    roe = _safe_float(row.get("roe"))
    roe_percentile = _safe_float(row.get("roe_hist_percentile"))
    roe_history_points = int(_safe_float(row.get("roe_history_points"), 0.0) or 0)
    margin = _safe_float(row.get("netprofit_margin"))
    revenue_growth = _safe_float(row.get("or_yoy"))
    eps_growth = _safe_float(row.get("basic_eps_yoy"))
    debt = _safe_float(row.get("debt_to_assets"))

    profitable = roe is not None and roe > 0 and (
        is_financial or (margin is not None and margin > 0)
    )
    balance_ok = is_financial or (debt is not None and debt <= 88)
    severe_deterioration = (
        revenue_growth is not None
        and eps_growth is not None
        and revenue_growth <= -25
        and eps_growth <= -40
    )
    is_good_stock = bool(
        score >= 60
        and profitability >= 50
        and stability >= 30
        and coverage >= 0.60
        and listing_years >= 2
        and profitable
        and balance_ok
        and not severe_deterioration
        and eligible_name
    )

    strengths: list[str] = []
    if profitability >= 70:
        strengths.append("盈利能力较强")
    elif profitability >= 55:
        strengths.append("盈利能力达标")
    if growth >= 65:
        strengths.append("成长质量较好")
    if balance >= 65:
        strengths.append("财务安全性较好")
    if stability >= 65:
        strengths.append("经营稳定性较好")
    if not strengths:
        strengths.append("综合基本面达到观察标准")

    return {
        "is_good_stock": is_good_stock,
        "good_stock_score": round(score, 2),
        "profitability_score": round(profitability, 2),
        "fundamental_growth_score": round(growth, 2),
        "balance_sheet_score": round(balance, 2),
        "business_stability_score": round(stability, 2),
        "good_stock_data_coverage": round(coverage, 2),
        "good_stock_reason": "、".join(strengths),
        "is_financial_industry": is_financial,
        "roe_hist_percentile": round(roe_percentile, 2) if roe_percentile is not None else None,
        "roe_history_points": roe_history_points,
    }


def _long_good_price_assessment(row: pd.Series) -> dict[str, Any]:
    """Evaluate the validation-selected, cross-date historical price score."""

    close = _safe_float(row.get("close"))
    pe = _safe_float(row.get("pe_ttm"))
    pb = _safe_float(row.get("pb"))
    pr = _safe_float(row.get("pr"))
    pr_from_pe = _safe_float(row.get("pr_pe"))
    pr_from_pb = _safe_float(row.get("pr_pb"))
    pr_formula_gap = _safe_float(row.get("pr_formula_gap"))
    pe_percentile = _safe_float(row.get("pe_hist_percentile"))
    pb_percentile = _safe_float(row.get("pb_hist_percentile"))
    pr_percentile = _safe_float(row.get("pr_hist_percentile"))
    pr_pe_percentile = _safe_float(row.get("pr_pe_hist_percentile"))
    pr_pb_percentile = _safe_float(row.get("pr_pb_hist_percentile"))
    valuation_profile = str(row.get("valuation_profile") or "balanced")
    pr_pe_weight = _safe_float(row.get("pr_pe_weight"), 0.50) or 0.50
    pr_pb_weight = _safe_float(row.get("pr_pb_weight"), 0.50) or 0.50
    history_points = int(_safe_float(row.get("valuation_history_points"), 0.0) or 0)
    historical_value_score = _safe_float(row.get("historical_value_score"))
    ma120 = _safe_float(row.get("ma_120"))
    ma120_slope = _safe_float(row.get("ma_120_slope_20d"))
    complete_percentiles = bool(
        historical_value_score is not None
        and all(value is not None for value in [pe_percentile, pb_percentile, pr_percentile])
    )
    has_history = bool(history_points >= 24 and complete_percentiles)
    trend_floor_price = ma120 * 0.90 if ma120 is not None else None
    trend_price_guard_passed = bool(
        close is not None
        and trend_floor_price is not None
        and close >= trend_floor_price
    )
    trend_slope_guard_passed = bool(
        ma120_slope is not None and ma120_slope >= -0.06
    )
    trend_guard = bool(trend_price_guard_passed and trend_slope_guard_passed)
    is_good_price = bool(
        has_history
        and historical_value_score >= 60
        and trend_guard
    )
    near_good_price = bool(
        not is_good_price
        and has_history
        and historical_value_score >= 55
        and trend_guard
    )

    if is_good_price:
        price_state = "GOOD_PRICE"
        price_reason = f"历史归一化价格分 {historical_value_score:.1f}，通过研究规则"
    elif history_points < 24:
        price_state = "WAIT_HISTORY"
        price_reason = f"估值历史仅 {history_points} 个月，未达到 24 个月最低要求"
    elif not complete_percentiles:
        price_state = "WAIT_HISTORY"
        price_reason = "当前 PE、PB 或双 PR 口径不完整，暂不判定好价格"
    elif not trend_guard:
        price_state = "WAIT_STABILITY"
        structure_issues: list[str] = []
        if close is None:
            structure_issues.append("缺少当前价格")
        elif trend_floor_price is None:
            structure_issues.append("缺少MA120，无法计算90%价格门槛")
        elif not trend_price_guard_passed:
            structure_issues.append(
                f"收盘 {close:.2f} 低于MA120的90%门槛 {trend_floor_price:.2f}"
            )
        if ma120_slope is None:
            structure_issues.append("缺少MA120近20日斜率")
        elif not trend_slope_guard_passed:
            structure_issues.append(
                f"MA120近20日斜率 {ma120_slope:.2%}，低于 -6.00% 门槛"
            )
        price_reason = "；".join(structure_issues) or "长期价格结构保护尚未通过"
    elif near_good_price:
        price_state = "NEAR_GOOD_PRICE"
        price_reason = f"历史归一化价格分 {historical_value_score:.1f}，接近推荐标准"
    else:
        price_state = "WAIT_PRICE"
        price_reason = f"历史归一化价格分 {historical_value_score:.1f}，当前估值分位仍需等待"

    return {
        "is_good_price": is_good_price,
        "price_score": round(historical_value_score, 2) if historical_value_score is not None else None,
        # Compatibility alias for older snapshots and downstream research.
        "good_price_score": round(historical_value_score, 2) if historical_value_score is not None else None,
        "price_score_normalization": "per_stock_trailing_history_percentile",
        "price_score_cross_date_comparable": True,
        "price_score_history_frequency": "month_end",
        "price_score_history_window_months": LONG_VALUATION_WINDOW_MONTHS,
        "price_score_min_history_points": LONG_VALUATION_MINIMUM_MONTHS,
        "price_state": price_state,
        "price_state_reason": price_reason,
        "good_price_rule": "composite_60_guard",
        "good_price_evidence": "RESEARCH_PENDING_OOS_CONFIRMATION",
        "valuation_history_points": history_points,
        "pe_hist_percentile": round(pe_percentile, 2) if pe_percentile is not None else None,
        "pb_hist_percentile": round(pb_percentile, 2) if pb_percentile is not None else None,
        "pr_hist_percentile": round(pr_percentile, 2) if pr_percentile is not None else None,
        "pr": round(pr, 4) if pr is not None else None,
        "pr_from_pe": round(pr_from_pe, 4) if pr_from_pe is not None else None,
        "pr_from_pb": round(pr_from_pb, 4) if pr_from_pb is not None else None,
        "pr_formula_gap": round(pr_formula_gap, 4) if pr_formula_gap is not None else None,
        "pr_pe_hist_percentile": round(pr_pe_percentile, 2) if pr_pe_percentile is not None else None,
        "pr_pb_hist_percentile": round(pr_pb_percentile, 2) if pr_pb_percentile is not None else None,
        "valuation_profile": valuation_profile,
        "pr_pe_weight": round(pr_pe_weight, 2),
        "pr_pb_weight": round(pr_pb_weight, 2),
        "pe_ttm": round(pe, 2) if pe is not None else None,
        "pb": round(pb, 2) if pb is not None else None,
        "trend_guard_passed": trend_guard,
        "trend_price_guard_passed": trend_price_guard_passed,
        "trend_slope_guard_passed": trend_slope_guard_passed,
        "ma120_slope_20d": round(ma120_slope, 6) if ma120_slope is not None else None,
        "price_levels": {
            "current_price": round(close, 2) if close is not None else None,
            "trend_reference_price": round(ma120, 2) if ma120 is not None else None,
            "trend_floor_price": (
                round(trend_floor_price, 2) if trend_floor_price is not None else None
            ),
        },
    }


@lru_cache(maxsize=1)
def _long_price_score_backtest_payload() -> dict[str, Any]:
    """Load the versioned, reproducible price-score band calibration."""

    try:
        payload = json.loads(LONG_PRICE_SCORE_BAND_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {
            "available": False,
            "bands": [],
            "conclusion": "价格分分档回测暂不可用",
        }
    bands = payload.get("bands") if isinstance(payload, dict) else None
    if not isinstance(bands, list) or not bands:
        return {
            "available": False,
            "bands": [],
            "conclusion": "价格分分档回测暂不可用",
        }
    return {**payload, "available": True}


def _analyst_forecast_years(row: pd.Series) -> list[dict[str, Any]]:
    forecasts: list[dict[str, Any]] = []
    signal_date = pd.to_datetime(row.get("date"), errors="coerce")
    signal_year = None if pd.isna(signal_date) else int(signal_date.year)
    for horizon in range(3):
        prefix = f"analyst_forward_y{horizon}"
        year = _safe_float(row.get(f"{prefix}_year"))
        eps_mean = _safe_float(row.get(f"{prefix}_eps_mean_180d"))
        eps_std = _safe_float(row.get(f"{prefix}_eps_std_180d"))
        price_mean = _safe_float(row.get(f"{prefix}_price_mean_180d"))
        price_std = _safe_float(row.get(f"{prefix}_price_std_180d"))
        forecasts.append(
            {
                "horizon": horizon,
                "forecast_year": (
                    int(year)
                    if year is not None
                    else signal_year + horizon
                    if signal_year is not None
                    else None
                ),
                "eps_mean": round(eps_mean, 4) if eps_mean is not None else None,
                "eps_std": round(eps_std, 4) if eps_std is not None else None,
                "eps_estimate_count": int(
                    _safe_float(row.get(f"{prefix}_eps_estimate_count_180d"), 0.0) or 0
                ),
                "price_mean": round(price_mean, 2) if price_mean is not None else None,
                "price_std": round(price_std, 2) if price_std is not None else None,
                "price_estimate_count": int(
                    _safe_float(row.get(f"{prefix}_price_estimate_count_180d"), 0.0) or 0
                ),
                "price_basis": "eps_x_forecast_pe",
            }
        )
    return forecasts


def _tea_good_stock_price_row(
    row: pd.Series,
    *,
    variant: str,
    rank: int | None = None,
) -> dict[str, Any]:
    stock = _long_good_stock_assessment(row)
    price = _long_good_price_assessment(row)
    recommended = bool(stock["is_good_stock"] and price["is_good_price"])
    level = "RECOMMENDED" if recommended else "WATCH"
    price_summary = {
        "GOOD_PRICE": "价格分达标",
        "NEAR_GOOD_PRICE": "价格分接近推荐线",
        "WAIT_PRICE": "价格分未达推荐线",
        "WAIT_HISTORY": "估值历史不足",
        "WAIT_STABILITY": price["price_state_reason"],
    }.get(str(price["price_state"]), "价格条件待确认")
    display_reason = f"{stock['good_stock_reason']}；{price_summary}"
    close = _safe_float(row.get("close"))
    analyst_growth = _safe_float(row.get("analyst_forward_growth_score"))
    analyst_upside = _safe_float(row.get("analyst_target_upside_180d"))
    analyst_eps_3y_mean = _safe_float(row.get("analyst_forward_eps_3y_mean_180d"))
    analyst_eps_3y_variance = _safe_float(row.get("analyst_forward_eps_3y_variance_180d"))
    return {
        "ts_code": str(row.get("ts_code")),
        "name": row.get("name"),
        "industry": row.get("industry"),
        "variant": variant,
        "rank": rank,
        "state": level,
        "action": "推荐" if recommended else "观察",
        "recommendation_level": level,
        "recommendation_since": None,
        "recommendation_days": None,
        "display_reason": display_reason,
        "close": close,
        "target_price": _safe_float(row.get("target_price")),
        "long_score": stock["good_stock_score"],
        "growth_score": stock["fundamental_growth_score"],
        "quality_score": round(_safe_float(row.get("quality_score"), 0.0) or 0.0, 2),
        "value_score": round(_safe_float(row.get("value_score"), 0.0) or 0.0, 2),
        "trend_score": round(_safe_float(row.get("trend_score"), 0.0) or 0.0, 2),
        "risk_score": round(_safe_float(row.get("risk_score"), 0.0) or 0.0, 2),
        "roe": _safe_float(row.get("roe")),
        "netprofit_margin": _safe_float(row.get("netprofit_margin")),
        "or_yoy": _safe_float(row.get("or_yoy")),
        "basic_eps_yoy": _safe_float(row.get("basic_eps_yoy")),
        "debt_to_assets": _safe_float(row.get("debt_to_assets")),
        "pe_ttm": round(_safe_float(row.get("pe_ttm"), 0.0) or 0.0, 2),
        "pb": round(_safe_float(row.get("pb"), 0.0) or 0.0, 2),
        "dv_ttm": round(_safe_float(row.get("dv_ttm"), 0.0) or 0.0, 2),
        "market_regime": str(row.get("market_regime") or "neutral"),
        "analyst_report_count_180d": int(_safe_float(row.get("analyst_report_count_180d"), 0.0) or 0),
        "analyst_org_count_180d": int(_safe_float(row.get("analyst_org_count_180d"), 0.0) or 0),
        "analyst_institution_count_180d": int(_safe_float(row.get("analyst_institution_count_180d"), 0.0) or 0),
        "analyst_research_report_count_180d": int(_safe_float(row.get("analyst_research_report_count_180d"), 0.0) or 0),
        "analyst_consensus_report_count_180d": int(_safe_float(row.get("analyst_consensus_report_count_180d"), 0.0) or 0),
        "analyst_forward_years_180d": int(_safe_float(row.get("analyst_forward_years_180d"), 0.0) or 0),
        "analyst_forward_growth_score": round(analyst_growth, 2) if analyst_growth is not None else None,
        "analyst_target_upside_180d": round(analyst_upside, 4) if analyst_upside is not None else None,
        "analyst_forward_eps_3y_mean_180d": (
            round(analyst_eps_3y_mean, 4) if analyst_eps_3y_mean is not None else None
        ),
        "analyst_forward_eps_3y_variance_180d": (
            round(analyst_eps_3y_variance, 6)
            if analyst_eps_3y_variance is not None
            else None
        ),
        "analyst_forward_eps_3y_years_180d": int(
            _safe_float(row.get("analyst_forward_eps_3y_years_180d"), 0.0) or 0
        ),
        "analyst_forward_eps_3y_estimate_count_180d": int(
            _safe_float(row.get("analyst_forward_eps_3y_estimate_count_180d"), 0.0) or 0
        ),
        "analyst_forecast_3y": _analyst_forecast_years(row),
        **stock,
        **price,
    }


def _upgrade_cached_tea_analyst_display(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    stocks = payload.get("stocks") or []
    if payload.get("schema_version") == LONG_STOCK_POOL_SCHEMA_VERSION and all(
        item.get("recommendation_level")
        and item.get("display_reason")
        and "price_score" in item
        and len(item.get("analyst_forecast_3y") or []) == 3
        for item in stocks
    ):
        return payload, False
    signal_date = payload.get("signal_date")
    if not stocks or not signal_date:
        return payload, False

    display_frame = pd.DataFrame(
        {
            "date": pd.to_datetime([signal_date] * len(stocks)),
            "ts_code": [str(item.get("ts_code")) for item in stocks],
            "close": [item.get("close") for item in stocks],
        }
    )
    enriched = _attach_analyst_forecast_for_display(display_frame).set_index("ts_code", drop=False)
    analyst_columns = [
        "analyst_report_count_180d",
        "analyst_org_count_180d",
        "analyst_institution_count_180d",
        "analyst_research_report_count_180d",
        "analyst_consensus_report_count_180d",
        "analyst_forward_years_180d",
        "analyst_forward_growth_score",
        "analyst_target_upside_180d",
        "analyst_forward_eps_3y_mean_180d",
        "analyst_forward_eps_3y_variance_180d",
        "analyst_forward_eps_3y_years_180d",
        "analyst_forward_eps_3y_estimate_count_180d",
    ]
    for horizon in range(3):
        prefix = f"analyst_forward_y{horizon}"
        analyst_columns.extend(
            [
                f"{prefix}_year",
                f"{prefix}_eps_mean_180d",
                f"{prefix}_eps_std_180d",
                f"{prefix}_eps_estimate_count_180d",
                f"{prefix}_price_mean_180d",
                f"{prefix}_price_std_180d",
                f"{prefix}_price_estimate_count_180d",
            ]
        )
    upgraded = dict(payload)
    upgraded_stocks: list[dict[str, Any]] = []
    for item in stocks:
        upgraded_item = dict(item)
        code = str(item.get("ts_code"))
        if code in enriched.index:
            row = enriched.loc[code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            for column in analyst_columns:
                value = _safe_float(row.get(column))
                if column in {
                    "analyst_report_count_180d",
                    "analyst_org_count_180d",
                    "analyst_institution_count_180d",
                    "analyst_research_report_count_180d",
                    "analyst_consensus_report_count_180d",
                    "analyst_forward_years_180d",
                    "analyst_forward_eps_3y_years_180d",
                    "analyst_forward_eps_3y_estimate_count_180d",
                    *{
                        f"analyst_forward_y{horizon}_{metric}_estimate_count_180d"
                        for horizon in range(3)
                        for metric in ("eps", "price")
                    },
                    *{f"analyst_forward_y{horizon}_year" for horizon in range(3)},
                }:
                    upgraded_item[column] = int(value or 0)
                else:
                    precision = (
                        6
                        if column == "analyst_forward_eps_3y_variance_180d"
                        else 4
                        if column in {
                            "analyst_target_upside_180d",
                            "analyst_forward_eps_3y_mean_180d",
                        }
                        else 2
                    )
                    upgraded_item[column] = round(value, precision) if value is not None else None
            upgraded_item["analyst_forecast_3y"] = _analyst_forecast_years(row)
        if upgraded_item.get("price_score") is None:
            upgraded_item["price_score"] = upgraded_item.get("good_price_score")
        upgraded_item.setdefault(
            "price_score_normalization", "per_stock_trailing_history_percentile"
        )
        upgraded_item.setdefault("price_score_cross_date_comparable", True)
        upgraded_item.setdefault("price_score_history_frequency", "month_end")
        upgraded_item.setdefault(
            "price_score_history_window_months", LONG_VALUATION_WINDOW_MONTHS
        )
        upgraded_item.setdefault(
            "price_score_min_history_points", LONG_VALUATION_MINIMUM_MONTHS
        )
        existing_since = pd.to_datetime(upgraded_item.get("recommendation_since"), errors="coerce")
        _decorate_long_recommendation_display(
            upgraded_item,
            selected=upgraded_item.get("rank") is not None,
            signal_ts=pd.Timestamp(signal_date),
            recommendation_since=None if pd.isna(existing_since) else pd.Timestamp(existing_since),
        )
        upgraded_stocks.append(upgraded_item)
    upgraded["stocks"] = upgraded_stocks
    upgraded["state_counts"] = {
        str(key): int(value)
        for key, value in pd.Series(
            [item["recommendation_level"] for item in upgraded_stocks]
        ).value_counts().to_dict().items()
    }
    upgraded["schema_version"] = LONG_STOCK_POOL_SCHEMA_VERSION
    return upgraded, True


def _tea_t_action(row: pd.Series, regime: str) -> tuple[str, str, str]:
    close = _safe_float(row.get("close"))
    ma20 = _safe_float(row.get("ma_20"))
    ma60 = _safe_float(row.get("ma_60"))
    ma120 = _safe_float(row.get("ma_120"))
    if close is None or ma20 is None or ma60 is None or ma120 is None:
        return "HOLD", "tea_grid", "缺少均线数据，暂不做T"
    sell_signal = close > ma20 * 1.12 or close > ma60 * 1.18
    buy_signal = regime != "risk_off" and close >= ma120 * 0.98 and (close <= ma20 * 1.015 or close <= ma60 * 1.035)
    if sell_signal:
        return "T_SELL", "tea_grid", "价格偏离短中期均线，允许高抛回到核心仓"
    if buy_signal:
        return "T_BUY", "tea_grid", "价格回到低吸区，允许补回机动仓"
    return "HOLD", "tea_grid", "未触发茶大网格阈值"


def _tea_exit_reason(row: pd.Series, regime: str) -> tuple[bool, str]:
    tea_score = _safe_float(row.get("tea_score"), 0.0) or 0.0
    trend_score = _safe_float(row.get("trend_score"), 0.0) or 0.0
    risk_score = _safe_float(row.get("risk_score"), 0.0) or 0.0
    close = _safe_float(row.get("close"))
    ma120 = _safe_float(row.get("ma_120"))
    ma120_slope = _safe_float(row.get("ma_120_slope_20d"), 0.0) or 0.0
    if tea_score < 55:
        return True, "茶大综合评分跌破清仓线"
    if close is not None and ma120 is not None and close < ma120 * (0.94 if regime == "risk_on" else 0.98):
        return True, "价格跌破年线防守区"
    if ma120_slope < -0.04:
        return True, "年线斜率明显转弱"
    if regime == "risk_off" and risk_score < 35:
        return True, "弱市且风险评分过低"
    if regime != "risk_on" and (trend_score < 45 or (close is not None and ma120 is not None and close < ma120 * 1.01)):
        return True, "非强势市场中趋势不再满足持仓条件"
    return False, ""


def _tea_reduce_reason(row: pd.Series, regime: str) -> tuple[bool, str]:
    tea_score = _safe_float(row.get("tea_score"), 0.0) or 0.0
    trend_score = _safe_float(row.get("trend_score"), 0.0) or 0.0
    risk_score = _safe_float(row.get("risk_score"), 0.0) or 0.0
    close = _safe_float(row.get("close"))
    ma60 = _safe_float(row.get("ma_60"))
    if tea_score < 68:
        return True, "茶大评分回落，降仓观察"
    if trend_score < 55:
        return True, "趋势评分走弱，降低组合优先级"
    if close is not None and ma60 is not None and close < ma60:
        return True, "价格跌破 MA60，先降仓控制风险"
    if regime == "risk_off" and risk_score < 55:
        return True, "弱市风险评分不足，降低风险暴露"
    return False, ""


def _tea_status_row(
    row: pd.Series,
    *,
    variant: str,
    selected: bool,
    target_weight: float,
    previous_state: str | None,
    rank: int | None,
) -> dict[str, Any]:
    regime = str(row.get("market_regime", "neutral") or "neutral")
    tea_score = _safe_float(row.get("tea_score"), 0.0) or 0.0
    trend_score = _safe_float(row.get("trend_score"), 0.0) or 0.0
    close = _safe_float(row.get("close"))
    target_price = _safe_float(row.get("target_price"))
    target_upside = target_price / close - 1.0 if close and target_price else None
    first_weight, entry_note = _tea_entry_tranche(row, target_weight, regime)
    t_action, t_profile, t_reason = _tea_t_action(row, regime)
    exit_signal, exit_reason = _tea_exit_reason(row, regime)
    reduce_signal, reduce_reason = _tea_reduce_reason(row, regime)
    prev_state = str(previous_state or "WATCH")

    if exit_signal:
        state = "EXIT"
        action = "清仓"
        reason = exit_reason
        target_weight = 0.0
        first_weight = 0.0
    elif selected:
        if prev_state in {"WATCH", "EXIT", "COOLDOWN"} and first_weight < target_weight * 0.85:
            state = "BUILDING"
            action = "分批建仓"
            reason = entry_note
        elif reduce_signal:
            state = "REDUCE"
            action = "降仓观察"
            reason = reduce_reason
        elif t_action in {"T_SELL", "T_BUY"}:
            state = "T_ACTIVE"
            action = "高抛" if t_action == "T_SELL" else "低吸"
            reason = t_reason
        else:
            state = "CORE"
            action = "核心持有"
            reason = "茶大核心条件延续，按目标仓位持有"
    elif reduce_signal or prev_state in {"CORE", "T_ACTIVE", "BUILDING", "REDUCE"}:
        state = "REDUCE"
        action = "降仓观察"
        reason = reduce_reason or "上一期目标股已移出本期组合，先降仓并观察"
        target_weight = 0.0
        first_weight = 0.0
    else:
        state = "WATCH"
        action = "观察"
        reason = "进入茶大高分观察池，等待趋势/价格/席位确认"
        target_weight = 0.0
        first_weight = 0.0
    return {
        "ts_code": str(row.get("ts_code")),
        "name": row.get("name"),
        "industry": row.get("industry"),
        "variant": variant,
        "rank": rank,
        "state": state,
        "previous_state": prev_state,
        "action": action,
        "reason": reason,
        "sleeve": row.get("sleeve", "core"),
        "t_action": t_action,
        "t_profile": t_profile,
        "t_reason": t_reason,
        "market_regime": regime,
        "target_weight": round(float(target_weight), 4),
        "first_tranche_weight": first_weight,
        "close": close,
        "target_price": target_price,
        "target_upside": round(target_upside, 4) if target_upside is not None else None,
        "price_levels": _tea_price_levels(row),
        "long_score": round(tea_score, 2),
        "growth_score": round(_safe_float(row.get("satellite_score"), 0.0) or 0.0, 2),
        "forecast_core_rank_score": round(tea_score, 2),
        "quality_score": round(_safe_float(row.get("quality_score"), 0.0) or 0.0, 2),
        "value_score": round(_safe_float(row.get("value_score"), 0.0) or 0.0, 2),
        "trend_score": round(trend_score, 2),
        "risk_score": round(_safe_float(row.get("risk_score"), 0.0) or 0.0, 2),
        "dividend_score": round(_safe_float(row.get("value_score"), 0.0) or 0.0, 2),
        "dv_ttm": round(_safe_float(row.get("dv_ttm"), 0.0) or 0.0, 2),
        "pe_ttm": round(_safe_float(row.get("pe_ttm"), 0.0) or 0.0, 2),
        "pb": round(_safe_float(row.get("pb"), 0.0) or 0.0, 2),
        "analyst_report_count_180d": int(_safe_float(row.get("analyst_report_count_180d"), 0.0) or 0),
        "analyst_org_count_180d": int(_safe_float(row.get("analyst_org_count_180d"), 0.0) or 0),
        "analyst_institution_count_180d": int(_safe_float(row.get("analyst_institution_count_180d"), 0.0) or 0),
        "analyst_research_report_count_180d": int(_safe_float(row.get("analyst_research_report_count_180d"), 0.0) or 0),
        "analyst_consensus_report_count_180d": int(_safe_float(row.get("analyst_consensus_report_count_180d"), 0.0) or 0),
        "analyst_forward_years_180d": int(_safe_float(row.get("analyst_forward_years_180d"), 0.0) or 0),
        "analyst_forward_growth_score": (
            round(growth_score, 2)
            if (growth_score := _safe_float(row.get("analyst_forward_growth_score"))) is not None
            else None
        ),
        "analyst_target_upside_180d": (
            round(target_upside_score, 4)
            if (target_upside_score := _safe_float(row.get("analyst_target_upside_180d"))) is not None
            else None
        ),
        "analyst_negative_warning": _safe_bool(row.get("analyst_negative_warning")),
        "close_above_ma120": bool(close is not None and _safe_float(row.get("ma_120")) is not None and close > float(row.get("ma_120"))),
        "ma_120_slope_20d": round(_safe_float(row.get("ma_120_slope_20d"), 0.0) or 0.0, 4),
    }


def _tea_previous_target_checkpoint(variant_key: str, signal_ts: pd.Timestamp) -> set[str] | None:
    """Recover the prior rebalance target set from the latest persisted live snapshot."""

    snapshot = _read_long_stock_pool_snapshot(
        variant_key,
        signal_ts.date().isoformat(),
        allow_sql=False,
    )
    if not snapshot or not snapshot.get("signal_date"):
        return None
    snapshot_ts = pd.to_datetime(snapshot["signal_date"], errors="coerce")
    if pd.isna(snapshot_ts) or snapshot_ts > signal_ts:
        return None
    rows = snapshot.get("stocks") or []
    if snapshot_ts.to_period("M") == signal_ts.to_period("M"):
        return {
            str(row.get("ts_code"))
            for row in rows
            if row.get("ts_code") and str(row.get("previous_state") or "") == "CORE"
        }
    return {
        str(row.get("ts_code"))
        for row in rows
        if row.get("ts_code") and row.get("rank") is not None
    }


def _continuous_recommendation_starts(
    targets: pd.DataFrame,
    signal_ts: pd.Timestamp,
) -> dict[str, pd.Timestamp]:
    """Find the first signal date in each latest uninterrupted recommendation streak."""

    if targets.empty or "rebalance_date" not in targets.columns or "ts_code" not in targets.columns:
        return {}
    eligible = targets[pd.to_datetime(targets["rebalance_date"], errors="coerce") <= signal_ts].copy()
    if eligible.empty:
        return {}
    eligible["rebalance_date"] = pd.to_datetime(eligible["rebalance_date"], errors="coerce")
    dates = sorted(eligible["rebalance_date"].dropna().unique())
    if not dates:
        return {}
    latest_date = pd.Timestamp(dates[-1])
    active = set(eligible.loc[eligible["rebalance_date"] == latest_date, "ts_code"].astype(str))
    starts = {code: latest_date for code in active}
    for candidate_date in reversed(dates[:-1]):
        candidate_ts = pd.Timestamp(candidate_date)
        codes = set(eligible.loc[eligible["rebalance_date"] == candidate_ts, "ts_code"].astype(str))
        active &= codes
        if not active:
            break
        for code in active:
            starts[code] = candidate_ts
    return starts


def _previous_recommendation_starts(
    variant_key: str,
    signal_ts: pd.Timestamp,
) -> dict[str, pd.Timestamp]:
    snapshot = _read_long_stock_pool_snapshot(
        variant_key,
        signal_ts.date().isoformat(),
        allow_sql=False,
    )
    if not snapshot or not snapshot.get("signal_date"):
        return {}
    snapshot_ts = pd.to_datetime(snapshot.get("signal_date"), errors="coerce")
    if pd.isna(snapshot_ts) or snapshot_ts > signal_ts:
        return {}
    starts: dict[str, pd.Timestamp] = {}
    for row in snapshot.get("stocks") or []:
        code = row.get("ts_code")
        if not code or row.get("rank") is None:
            continue
        started = pd.to_datetime(row.get("recommendation_since"), errors="coerce")
        starts[str(code)] = snapshot_ts if pd.isna(started) else pd.Timestamp(started)
    return starts


def _decorate_long_recommendation_display(
    item: dict[str, Any],
    *,
    selected: bool,
    signal_ts: pd.Timestamp,
    recommendation_since: pd.Timestamp | None = None,
) -> dict[str, Any]:
    state = str(item.get("state") or "WATCH")
    if selected and state in {"CORE", "BUILDING", "T_ACTIVE"}:
        level = "RECOMMENDED"
    elif state == "EXIT":
        level = "AVOID"
    elif state == "REDUCE":
        level = "CAUTION"
    else:
        level = "WATCH"

    item["recommendation_level"] = level
    if level == "RECOMMENDED":
        start = pd.Timestamp(recommendation_since) if recommendation_since is not None else signal_ts
        start = min(start.normalize(), signal_ts.normalize())
        item["recommendation_since"] = start.date().isoformat()
        item["recommendation_days"] = max(1, int((signal_ts.normalize() - start).days) + 1)
    else:
        item["recommendation_since"] = None
        item["recommendation_days"] = None

    original_reason = str(item.get("reason") or "")
    if level == "RECOMMENDED":
        if state == "BUILDING":
            display_reason = "本期进入推荐池，价格与风险条件支持分批执行"
        elif state == "T_ACTIVE" and str(item.get("t_action")) == "T_BUY":
            display_reason = "价格进入策略低吸区，推荐条件仍然成立"
        elif state == "T_ACTIVE" and str(item.get("t_action")) == "T_SELL":
            display_reason = "价格短期偏高，但中长期推荐条件仍然成立"
        else:
            display_reason = "核心筛选条件延续，维持推荐"
    elif level == "CAUTION":
        display_reason = (
            original_reason
            .replace("降仓观察", "转为谨慎观察")
            .replace("降仓", "降低推荐优先级")
            .replace("降低组合优先级", "降低推荐优先级")
        ) or "风险条件转弱，降低推荐优先级"
    elif level == "AVOID":
        display_reason = original_reason or "当前趋势条件失效，暂时回避"
    else:
        display_reason = original_reason or "尚未满足本期推荐条件"
    item["display_reason"] = display_reason

    levels = item.get("price_levels") or {}
    close = _safe_float(item.get("close"))
    aggressive = _safe_float(levels.get("entry_aggressive_price"))
    entry = _safe_float(levels.get("entry_target_price"))
    small_position = _safe_float(levels.get("entry_small_position_price"))
    if level == "AVOID":
        price_state = "TREND_INVALID"
        price_reason = str(item.get("display_reason") or "趋势条件未恢复")
    elif level == "CAUTION":
        price_state = "RISK_RISING"
        price_reason = str(item.get("display_reason") or "风险条件正在转弱")
    elif level == "WATCH":
        price_state = "WAIT_SIGNAL"
        price_reason = "价格不是唯一条件，等待评分与趋势重新满足推荐门槛"
    elif state == "T_ACTIVE" and str(item.get("t_action")) == "T_BUY":
        price_state = "BUY_ZONE"
        price_reason = "价格进入策略低吸区"
    elif state == "T_ACTIVE" and str(item.get("t_action")) == "T_SELL":
        price_state = "WAIT_PULLBACK"
        price_reason = "价格偏离短中期均线，等待回落"
    elif close is not None and aggressive is not None and close <= aggressive:
        price_state = "AGGRESSIVE"
        price_reason = f"当前价不高于积极参考线 {aggressive:.2f}"
    elif close is not None and entry is not None and close <= entry:
        price_state = "BUY_ZONE"
        price_reason = f"当前价进入建议建仓区，不高于 {entry:.2f}"
    elif close is not None and small_position is not None and close <= small_position:
        price_state = "SCALE_IN"
        price_reason = f"高于理想建仓线 {entry:.2f}，仅适合分批"
    else:
        price_state = "WAIT_PULLBACK"
        price_reason = (
            f"当前价格偏高，等待回落至 {entry:.2f} 附近"
            if entry is not None
            else "当前缺少有效建仓参考线"
        )
    item["price_state"] = price_state
    item["price_state_reason"] = price_reason
    return item


@lru_cache(maxsize=8)
def _build_tea_master_stock_pool_cached(variant_key: str, signal_date: str | None) -> dict[str, Any]:
    module = _tea_master_research_module()
    config_name = TEA_LONG_VARIANTS.get(variant_key, variant_key)
    config = next((item for item in module.CONFIGS if item.name == config_name), None)
    if config is None:
        raise ValueError(f"未知茶大长线策略版本: {variant_key}")
    scored, coverage = _tea_master_live_scores(signal_date)
    eligible_scored = scored.copy()
    if signal_date:
        eligible_scored = eligible_scored[eligible_scored["date"] <= pd.to_datetime(signal_date)].copy()
    if eligible_scored.empty:
        raise RuntimeError("所选日期之前没有茶大长线评分截面")
    signal_ts = pd.to_datetime(eligible_scored["date"].max())
    scored_date = eligible_scored[pd.to_datetime(eligible_scored["date"]) == signal_ts].copy()
    if scored_date.empty:
        scored_date = eligible_scored.sort_values("date").drop_duplicates("ts_code", keep="last").copy()
    scored_date = _attach_analyst_forecast_for_display(scored_date)
    factor_snapshot = (
        _publish_long_factor_snapshot(scored_date, signal_ts)
        if variant_key == "tea"
        else None
    )

    rows = [
        _tea_good_stock_price_row(row, variant=variant_key)
        for _, row in scored_date.iterrows()
    ]
    all_good_stock_rows = [row for row in rows if row["is_good_stock"]]
    recommended_rows = sorted(
        [row for row in all_good_stock_rows if row["recommendation_level"] == "RECOMMENDED"],
        key=lambda item: (-(item.get("price_score") or 0), -(item.get("good_stock_score") or 0)),
    )
    watch_rows = sorted(
        [row for row in all_good_stock_rows if row["recommendation_level"] == "WATCH"],
        key=lambda item: (-(item.get("good_stock_score") or 0), -(item.get("price_score") or 0)),
    )
    # Keep both decision states visible without turning the page into a full
    # universe dump. Counts below always describe the complete good-stock pool.
    rows = recommended_rows[:40] + watch_rows[:40]
    for index, item in enumerate(rows, start=1):
        item["rank"] = index

    regime_values = scored_date["market_regime"].dropna().astype(str)
    regime = str(regime_values.iloc[0]) if not regime_values.empty else "neutral"
    state_counts = {
        "RECOMMENDED": len(recommended_rows),
        "WATCH": len(watch_rows),
    }
    analyst_covered = sum(
        1 for row in all_good_stock_rows if row.get("analyst_report_count_180d", 0) > 0
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": signal_ts.date().isoformat(),
        "variant": variant_key,
        "variant_id": config_name,
        "variant_name": LONG_VARIANT_LABELS.get(variant_key, variant_key),
        "market_regime": regime,
        "coverage": coverage,
        "factor_snapshot": factor_snapshot,
        "state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "quality_price_summary": {
            "good_stock_count": len(all_good_stock_rows),
            "recommended_count": len(recommended_rows),
            "watch_count": len(watch_rows),
            "analyst_covered_count": analyst_covered,
            "displayed_count": len(rows),
        },
        "price_score_backtest": _long_price_score_backtest_payload(),
        "stocks": rows,
        "notes": [
            "长线页面只负责选好股票与合适的建仓价格，持仓管理由自选池承担。",
            "好股票由盈利能力、成长质量、财务安全和经营稳定性决定，价格与趋势不参与好股票资格。",
            "好股票统一进入观察池；价格分达到60且通过长期价格结构保护时升级为推荐。",
            "PR-PE和PR-PB同时保留，并按利润驱动、资本驱动或均衡型行业调整权重。",
            "价格分为0—100分，使用每只股票最近7年月末历史的PE/PB/双PR分位归一化，不使用单日横截面排名，因此跨日含义一致；至少24个有效点。",
            "券商预测严格按 report_date <= signal_date 展示，目前不参与好股票或好价格评分。",
            "未来三年研报统计按当年E、次年E、后年E分别展示EPS及EPS×预测PE隐含股价的均值、样本标准差；严格按信号日可见数据计算。",
        ],
    }


@lru_cache(maxsize=8)
def _build_long_stock_pool_cached(variant_key: str, signal_date: str | None, persist_state: bool) -> dict[str, Any]:
    module = _long_research_module()
    variant = LONG_VARIANTS.get(variant_key, variant_key)
    if variant not in LONG_VARIANTS.values():
        raise ValueError(f"未知长线策略版本: {variant_key}")

    requested_end = pd.to_datetime(signal_date) if signal_date else None
    config = module.BacktestConfig(variant=variant, start="20130101", end=None)
    features, daily_basic, stock_basic, coverage = _load_live_long_base(signal_date)
    if variant in getattr(module, "PIT_UNIVERSE_VARIANTS", set()):
        candidate_symbols = None
        coverage["point_in_time_universe"] = True
    else:
        candidate_symbols = module.select_candidate_symbols_from_daily_basic(daily_basic, stock_basic, config)
        coverage["point_in_time_universe"] = False
        features = features[features["ts_code"].astype(str).isin(candidate_symbols)].copy()
    if requested_end is not None:
        features = features[features["date"] <= requested_end].copy()
        daily_basic = daily_basic[daily_basic["date"] <= requested_end].copy()
    common_dates = sorted(set(features["date"].dropna()) & set(daily_basic["date"].dropna()))
    if not common_dates:
        raise RuntimeError("日线特征与 daily_basic 没有共同信号日期")
    signal_ts = common_dates[-1]
    latest_features = features[features["date"] == signal_ts].copy()
    latest_basic = daily_basic[daily_basic["date"] == signal_ts].copy()
    latest_features = latest_features.sort_values(
        ["date", "ts_code", "trade_date"]
    ).drop_duplicates(["date", "ts_code"], keep="last")
    latest_basic = latest_basic.sort_values(
        ["date", "ts_code", "trade_date"]
    ).drop_duplicates(["date", "ts_code"], keep="last")
    if variant in getattr(module, "PIT_UNIVERSE_VARIANTS", set()):
        latest_basic = module.filter_daily_basic_point_in_time(latest_basic, config)
    merged = latest_features.merge(
        latest_basic.drop(columns=["trade_date"]),
        on=["date", "ts_code"],
        how="inner",
    )
    if len(merged) < 50:
        raise RuntimeError(
            "最新信号日长线截面不完整: "
            f"date={pd.Timestamp(signal_ts).date().isoformat()} rows={len(merged)}"
        )
    merged = module.load_financial_asof(merged)
    if variant in module.GROWTH_VARIANTS:
        merged = module.load_analyst_forecast_asof(merged)
    else:
        merged = module.add_empty_analyst_forecast_columns(merged)
    market_regime = module.load_market_regime(signal_ts, signal_ts)
    regime_row = _require_exact_regime_row(
        market_regime,
        signal_ts,
        context=f"{variant_key} live market regime",
        required_columns=LONG_MARKET_REGIME_FEATURE_COLUMNS,
    )
    coverage["market_regime_feature_date"] = pd.Timestamp(regime_row["date"]).date().isoformat()
    merged = merged.merge(market_regime, on="date", how="left")
    merged["cashflow_quality"] = np.nan
    scored = module.build_scores(merged, config)
    targets = module.make_monthly_targets(scored, config)

    selected_codes = set(targets["ts_code"].astype(str)) if not targets.empty else set()
    target_map = targets.set_index("ts_code").to_dict(orient="index") if not targets.empty else {}
    previous_states = _load_long_previous_states(variant_key)

    rows: list[dict[str, Any]] = []
    scored_by_symbol = scored.set_index("ts_code", drop=False)
    selected_order = {code: i + 1 for i, code in enumerate(targets["ts_code"].astype(str).tolist())} if not targets.empty else {}
    for code in selected_codes:
        if code not in scored_by_symbol.index:
            continue
        row = scored_by_symbol.loc[code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        target_info = target_map.get(code, {})
        enriched = row.copy()
        for key, value in target_info.items():
            enriched[key] = value
        rows.append(
            _long_status_row(
                module,
                enriched,
                variant=variant_key,
                selected=True,
                target_weight=float(target_info.get("target_weight", 0.0) or 0.0),
                previous=previous_states.get(code),
                rank=selected_order.get(code),
            )
        )

    candidate = scored[~scored["ts_code"].astype(str).isin(selected_codes)].copy()
    candidate["watch_rank_score"] = candidate.get("forecast_core_rank_score", candidate["long_score"]).fillna(candidate["long_score"])
    watch_filter = (
        (candidate["long_score"] >= 72)
        & (candidate["close"] > candidate["ma_120"])
        & (candidate["ma_120_slope_20d"] >= -0.005)
    )
    watch_rows = candidate[watch_filter].sort_values(
        ["watch_rank_score", "long_score", "trend_score", "quality_score"],
        ascending=False,
    ).head(30)
    reduce_filter = (
        (candidate["long_score"].between(58, 72, inclusive="left"))
        | (candidate["close"] < candidate["ma_60"])
        | candidate.get("analyst_negative_warning", False).fillna(False)
    )
    reduce_rows = candidate[reduce_filter].sort_values(
        ["long_score", "risk_score"],
        ascending=[True, True],
    ).head(15)
    appended: set[str] = set(selected_codes)
    for _, row in pd.concat([watch_rows, reduce_rows], ignore_index=True).iterrows():
        code = str(row.get("ts_code"))
        if code in appended:
            continue
        appended.add(code)
        rows.append(
            _long_status_row(
                module,
                row,
                variant=variant_key,
                selected=False,
                target_weight=0.0,
                previous=previous_states.get(code),
                rank=None,
            )
        )

    for item in rows:
        code = str(item.get("ts_code"))
        previous = previous_states.get(code) or {}
        previous_since = pd.to_datetime(previous.get("recommendation_since"), errors="coerce")
        _decorate_long_recommendation_display(
            item,
            selected=item.get("rank") is not None,
            signal_ts=pd.Timestamp(signal_ts),
            recommendation_since=None if pd.isna(previous_since) else pd.Timestamp(previous_since),
        )

    state_order = {"CORE": 0, "T_ACTIVE": 1, "BUILDING": 2, "REDUCE": 3, "WATCH": 4, "EXIT": 5, "COOLDOWN": 6}
    rows = sorted(
        rows,
        key=lambda item: (
            state_order.get(str(item.get("state")), 99),
            item.get("rank") or 999,
            -(item.get("long_score") or 0),
        ),
    )
    regime = str(scored["market_regime"].dropna().iloc[0]) if "market_regime" in scored.columns and not scored["market_regime"].dropna().empty else "neutral"
    state_counts = pd.Series([row["recommendation_level"] for row in rows]).value_counts().to_dict()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": pd.Timestamp(signal_ts).date().isoformat(),
        "variant": variant_key,
        "variant_id": variant,
        "variant_name": LONG_VARIANT_LABELS.get(variant_key, variant_key),
        "market_regime": regime,
        "coverage": coverage,
        "state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "stocks": rows,
        "notes": [
            "长线股票池使用最新可用月度信号截面生成，券商预测严格按 report_date <= signal_date 合并。",
            "推荐程度与价格状态分开表达；推荐角标为连续入选的自然日数，不代表真实账户持仓。",
            "策略目标和参考首批仅用于组合研究，不生成降仓或清仓等账户指令。",
        ],
    }
    if persist_state:
        _persist_long_states(variant_key, payload)
    return payload


def get_long_stock_pool(
    variant: str = "tea",
    signal_date: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    variant_key = variant if variant in LONG_VARIANTS else next(
        (key for key, value in LONG_VARIANTS.items() if value == variant),
        variant,
    )
    if not refresh:
        cached = _read_long_stock_pool_snapshot(variant_key, signal_date, allow_sql=False)
        if cached is not None:
            if variant_key in TEA_LONG_VARIANTS:
                cached, upgraded = _upgrade_cached_tea_analyst_display(cached)
                if upgraded:
                    _write_long_stock_pool_snapshot(cached, variant_key, signal_date, write_sql=False)
                    cached["cache"] = {
                        "hit": True,
                        "backend": "filesystem",
                        "migrated": True,
                        "schema_version": LONG_STOCK_POOL_SCHEMA_VERSION,
                    }
            return cached
    if variant_key in TEA_LONG_VARIANTS:
        if refresh:
            _build_tea_master_stock_pool_cached.cache_clear()
            _tea_master_live_scores_cached.cache_clear()
        payload = _build_tea_master_stock_pool_cached(variant_key, signal_date)
        _write_long_stock_pool_snapshot(payload, variant_key, signal_date, write_sql=refresh)
        payload["cache"] = {"hit": False, "backend": "generated"}
        return payload
    if refresh:
        _build_long_stock_pool_cached.cache_clear()
    payload = _build_long_stock_pool_cached(variant_key, signal_date, True)
    _write_long_stock_pool_snapshot(payload, variant_key, signal_date, write_sql=refresh)
    payload["cache"] = {"hit": False, "backend": "generated"}
    return payload


def _read_blood_chip_long_snapshot(
    signal_date: str | None = None,
    *,
    strictly_before: bool = False,
) -> dict[str, Any] | None:
    requested = (
        pd.Timestamp(signal_date).normalize().date().isoformat()
        if signal_date
        else None
    )
    candidates: list[tuple[str, Path]] = []
    if BLOOD_CHIP_LONG_SNAPSHOT_DIR.exists():
        for path in BLOOD_CHIP_LONG_SNAPSHOT_DIR.glob("*.json"):
            try:
                payload = read_json_file(path)
            except Exception:
                continue
            if payload.get("schema_version") != BLOOD_CHIP_LONG_SCHEMA_VERSION:
                continue
            candidate_date = str(payload.get("signal_date") or "")
            if not candidate_date:
                continue
            if requested and (
                candidate_date > requested
                or (strictly_before and candidate_date >= requested)
            ):
                continue
            candidates.append((candidate_date, path))
    if not candidates:
        return None
    candidate_date, path = max(candidates, key=lambda item: item[0])
    payload = read_json_file(path)
    payload["cache"] = {
        "hit": True,
        "backend": "filesystem",
        "snapshot_key": path.stem,
        "snapshot_date": candidate_date,
        "requested_date": requested,
        "stale": bool(requested and candidate_date != requested),
    }
    return payload


def _write_blood_chip_long_snapshot(payload: dict[str, Any]) -> Path:
    signal_date = pd.Timestamp(payload["signal_date"]).strftime("%Y%m%d")
    target = BLOOD_CHIP_LONG_SNAPSHOT_DIR / f"{signal_date}.json"
    stored = dict(payload)
    stored["schema_version"] = BLOOD_CHIP_LONG_SCHEMA_VERSION
    stored["cache"] = {
        "hit": False,
        "backend": "generated",
        "snapshot_key": target.stem,
    }
    return atomic_write_json(stored, target)


def _resolve_blood_chip_signal_date(signal_date: str | None) -> str:
    if signal_date:
        return pd.Timestamp(signal_date).normalize().date().isoformat()
    local_date = _local_market_trade_date()
    if local_date:
        return pd.Timestamp(local_date).date().isoformat()
    if not BLOOD_CHIP_BENCHMARK_PATH.exists():
        raise FileNotFoundError(BLOOD_CHIP_BENCHMARK_PATH)
    benchmark_dates = pd.read_parquet(
        BLOOD_CHIP_BENCHMARK_PATH,
        columns=["trade_date"],
    )["trade_date"]
    latest = pd.to_datetime(benchmark_dates.astype(str), errors="coerce").max()
    if pd.isna(latest):
        raise ValueError("沪深300基准没有可用交易日")
    return pd.Timestamp(latest).date().isoformat()


def _build_blood_chip_long_plan_live(signal_date: str | None = None) -> dict[str, Any]:
    requested_date = _resolve_blood_chip_signal_date(signal_date)
    requested_ts = pd.Timestamp(requested_date)
    start_date = (requested_ts - pd.Timedelta(days=620)).strftime("%Y%m%d")
    requested_end = requested_ts.strftime("%Y%m%d")
    daily = load_canonical_daily(
        BLOOD_CHIP_DAILY_DIR,
        start_date,
        requested_end,
    )
    actual_end = pd.to_datetime(
        daily["trade_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    ).max()
    if pd.isna(actual_end):
        raise ValueError("规范日线没有可用交易日")
    actual_date = pd.Timestamp(actual_end).date().isoformat()
    benchmark = load_benchmark(
        BLOOD_CHIP_BENCHMARK_PATH,
        start_date,
        pd.Timestamp(actual_end).strftime("%Y%m%d"),
    )
    stock_basic_path = PROJECT_ROOT / "data/raw/stock_basic.parquet"
    stock_basic = (
        pd.read_parquet(stock_basic_path)
        if stock_basic_path.exists()
        else pd.DataFrame()
    )
    payload = build_blood_chip_long_plan(
        daily,
        benchmark,
        signal_date=actual_date,
        stock_basic=stock_basic,
    )
    previous = _read_blood_chip_long_snapshot(actual_date, strictly_before=True)
    payload["daily_iteration"] = build_blood_chip_daily_iteration(payload, previous)
    return payload


def get_blood_chip_long_plan(
    signal_date: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Read or regenerate the daily blood-chip medium/long strategy snapshot."""

    if not refresh:
        cached = _read_blood_chip_long_snapshot(signal_date)
        if cached is not None:
            return cached
    payload = _build_blood_chip_long_plan_live(signal_date)
    _write_blood_chip_long_snapshot(payload)
    payload["cache"] = {
        "hit": False,
        "backend": "generated",
        "snapshot_key": pd.Timestamp(payload["signal_date"]).strftime("%Y%m%d"),
    }
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _frame_records(frame: pd.DataFrame | None, limit: int | None = None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    out = frame.head(limit).copy() if limit else frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return [_json_ready(record) for record in out.replace([np.inf, -np.inf], np.nan).to_dict("records")]


def _canonical_watch_symbol(raw_symbol: str) -> str:
    symbol = str(raw_symbol or "").strip().upper()
    if not symbol:
        return ""
    symbol = symbol.replace("_", ".")
    return normalize_ts_code(symbol)


def _normalize_watch_symbol(raw_symbol: str) -> str:
    symbol = _canonical_watch_symbol(raw_symbol)
    if not symbol:
        raise ValueError("股票代码不能为空")
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=DAILY_DIR.parent))
    if store.latest_trade_date(DAILY_DIR.name, symbol) is None:
        raise ValueError(f"本地日线数据不存在: {symbol}")
    return symbol


@lru_cache(maxsize=1)
def _stock_basic_for_similar_patterns() -> pd.DataFrame:
    return load_stock_basic(PROJECT_ROOT / "data/raw/stock_basic.parquet")


def _stock_profile_from_basic(symbol: str, basic: pd.DataFrame | None = None) -> dict[str, str]:
    frame = basic if basic is not None else _stock_basic_for_similar_patterns()
    if frame is not None and not frame.empty:
        row = frame[frame["ts_code"] == symbol]
        if not row.empty:
            return {
                "symbol": symbol,
                "name": str(row["name"].iloc[0]),
                "industry": str(row["industry"].iloc[0]),
            }
    return {"symbol": symbol, "name": symbol, "industry": ""}


def _normalize_watch_alert_conditions(
    raw_conditions: Any,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(raw_conditions, list):
        if strict:
            raise ValueError("提醒条件必须是列表")
        return []
    if len(raw_conditions) > WATCHLIST_ALERT_MAX_CONDITIONS:
        if strict:
            raise ValueError(f"每个提醒最多设置 {WATCHLIST_ALERT_MAX_CONDITIONS} 条条件")
        raw_conditions = raw_conditions[:WATCHLIST_ALERT_MAX_CONDITIONS]

    normalized: list[dict[str, Any]] = []
    for index, raw_condition in enumerate(raw_conditions):
        if not isinstance(raw_condition, dict):
            if strict:
                raise ValueError("提醒条件格式无效")
            continue
        kind = str(raw_condition.get("kind") or "")
        if kind not in {"price", "indicator"}:
            if strict:
                raise ValueError("提醒条件类型必须是价格或指标")
            continue
        operator = str(raw_condition.get("operator") or "")
        if operator not in WATCHLIST_ALERT_OPERATORS:
            if strict:
                raise ValueError("提醒条件运算符无效")
            continue
        conjunction = str(raw_condition.get("conjunction") or "and").lower()
        if conjunction not in {"and", "or"}:
            if strict:
                raise ValueError("条件关系必须是 and 或 or")
            conjunction = "and"
        try:
            value = float(raw_condition.get("value"))
        except (TypeError, ValueError):
            if strict:
                raise ValueError("提醒阈值必须是数字")
            continue
        if not np.isfinite(value) or (kind == "price" and value <= 0):
            if strict:
                raise ValueError("价格阈值必须大于 0" if kind == "price" else "指标阈值必须是有限数字")
            continue
        condition_id = str(raw_condition.get("id") or f"condition-{index + 1}").strip()[:64]
        condition: dict[str, Any] = {
            "id": condition_id or f"condition-{index + 1}",
            "conjunction": "and" if index == 0 else conjunction,
            "kind": kind,
            "operator": operator,
            "value": value,
        }
        if kind == "indicator":
            indicator = str(raw_condition.get("indicator") or "")
            if indicator not in WATCHLIST_ALERT_INDICATORS:
                if strict:
                    raise ValueError("不支持该指标提醒")
                continue
            condition["indicator"] = indicator
        normalized.append(condition)
    return normalized


def _normalize_watch_alert_reminders(
    raw_reminders: Any,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(raw_reminders, list):
        if strict:
            raise ValueError("提醒列表格式无效")
        return []
    if len(raw_reminders) > WATCHLIST_ALERT_MAX_REMINDERS:
        if strict:
            raise ValueError(f"每只股票最多设置 {WATCHLIST_ALERT_MAX_REMINDERS} 个提醒")
        raw_reminders = raw_reminders[:WATCHLIST_ALERT_MAX_REMINDERS]

    normalized: list[dict[str, Any]] = []
    for index, raw_reminder in enumerate(raw_reminders):
        if not isinstance(raw_reminder, dict):
            if strict:
                raise ValueError("提醒格式无效")
            continue
        note = str(raw_reminder.get("note") or "").strip()
        if len(note) > WATCHLIST_ALERT_NOTE_MAX_LENGTH:
            if strict:
                raise ValueError(f"提醒备注不能超过 {WATCHLIST_ALERT_NOTE_MAX_LENGTH} 个字符")
            note = note[:WATCHLIST_ALERT_NOTE_MAX_LENGTH]
        conditions = _normalize_watch_alert_conditions(
            raw_reminder.get("conditions", []),
            strict=strict,
        )
        if not conditions:
            if strict:
                raise ValueError("每个提醒至少需要一个条件")
            continue
        reminder_id = str(raw_reminder.get("id") or f"reminder-{index + 1}").strip()[:64]
        normalized.append(
            {
                "id": reminder_id or f"reminder-{index + 1}",
                "note": note,
                "conditions": conditions,
            }
        )
    return normalized


def _normalize_watch_alert_config(raw_config: Any, *, strict: bool = False) -> dict[str, Any]:
    if not isinstance(raw_config, dict):
        if strict:
            raise ValueError("提醒设置格式无效")
        raw_config = {}
    reminders = _normalize_watch_alert_reminders(
        raw_config.get("reminders", []),
        strict=strict,
    )
    updated_at = str(raw_config.get("updated_at") or "")
    return {
        "enabled": bool(raw_config.get("enabled", True)) and bool(reminders),
        "reminders": reminders,
        "updated_at": updated_at,
    }


OPERATION_PLAN_HORIZONS = {"tomorrow", "long_term"}
OPERATION_PLAN_STATUSES = {"planned", "done", "cancelled"}
OPERATION_PLAN_CHECKLIST_MAX_ITEMS = 100
OPERATION_PLAN_CHECKLIST_TEXT_MAX_LENGTH = 500


def _normalize_operation_plan_checklist(
    raw_checklist: Any,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    if raw_checklist is None:
        return []
    if not isinstance(raw_checklist, list):
        if strict:
            raise ValueError("计划清单格式无效")
        return []
    if len(raw_checklist) > OPERATION_PLAN_CHECKLIST_MAX_ITEMS:
        if strict:
            raise ValueError(f"每个计划最多添加 {OPERATION_PLAN_CHECKLIST_MAX_ITEMS} 条清单")
        raw_checklist = raw_checklist[:OPERATION_PLAN_CHECKLIST_MAX_ITEMS]

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(raw_checklist):
        if not isinstance(raw_item, dict):
            if strict:
                raise ValueError("计划清单项格式无效")
            continue
        text = str(raw_item.get("text") or "").strip()
        if not text:
            if strict:
                raise ValueError("计划清单内容不能为空")
            continue
        if len(text) > OPERATION_PLAN_CHECKLIST_TEXT_MAX_LENGTH:
            if strict:
                raise ValueError(
                    f"计划清单单项不能超过 {OPERATION_PLAN_CHECKLIST_TEXT_MAX_LENGTH} 个字符"
                )
            text = text[:OPERATION_PLAN_CHECKLIST_TEXT_MAX_LENGTH]
        item_id = str(raw_item.get("id") or f"check-{index + 1}").strip()[:64]
        if not item_id or item_id in seen_ids:
            if strict:
                raise ValueError("计划清单项标识无效或重复")
            item_id = f"check-{index + 1}"
            while item_id in seen_ids:
                item_id = f"{item_id}-next"
        seen_ids.add(item_id)
        normalized.append(
            {
                "id": item_id,
                "text": text,
                "completed": bool(raw_item.get("completed", False)),
            }
        )
    return normalized


def _read_operation_plans() -> list[dict[str, Any]]:
    if not OPERATION_PLANS_PATH.exists():
        return []
    try:
        payload = json.loads(OPERATION_PLANS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("plans", []) if isinstance(payload, dict) else []
    plans: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        normalized = dict(row)
        normalized["checklist"] = _normalize_operation_plan_checklist(
            row.get("checklist", []),
        )
        plans.append(normalized)
    return plans


def _write_operation_plans(plans: list[dict[str, Any]]) -> None:
    atomic_write_json(
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "plans": plans,
        },
        OPERATION_PLANS_PATH,
    )


def get_operation_plans() -> dict[str, Any]:
    plans = _read_operation_plans()
    plans.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("updated_at") or ""),
        ),
        reverse=True,
    )
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "plans": plans,
    }


def save_operation_plan(data: dict[str, Any], plan_id: str | None = None) -> dict[str, Any]:
    title = str(data.get("title") or "").strip()
    content = str(data.get("content") or "").strip()
    horizon = str(data.get("horizon") or "tomorrow")
    status = str(data.get("status") or "planned")
    target_date = str(data.get("target_date") or "").strip()
    symbol = str(data.get("symbol") or "").strip().upper()
    checklist = _normalize_operation_plan_checklist(
        data.get("checklist", []),
        strict=True,
    )
    if not title:
        raise ValueError("计划标题不能为空")
    if len(title) > 120 or len(content) > 20_000 or len(symbol) > 30:
        raise ValueError("计划内容超过长度限制")
    if horizon not in OPERATION_PLAN_HORIZONS:
        raise ValueError("计划周期无效")
    if status not in OPERATION_PLAN_STATUSES:
        raise ValueError("计划状态无效")
    if target_date:
        try:
            date.fromisoformat(target_date)
        except ValueError as exc:
            raise ValueError("目标日期格式无效") from exc

    plans = _read_operation_plans()
    now = datetime.now().isoformat(timespec="seconds")
    if plan_id:
        existing = next((item for item in plans if item.get("id") == plan_id), None)
        if existing is None:
            raise ValueError("操作计划不存在")
        created_at = existing.get("created_at") or now
    else:
        plan_id = uuid.uuid4().hex
        created_at = now
    saved = {
        "id": plan_id,
        "horizon": horizon,
        "title": title,
        "symbol": symbol,
        "target_date": target_date,
        "content": content,
        "checklist": checklist,
        "status": status,
        "created_at": created_at,
        "updated_at": now,
    }
    plans = [saved if item.get("id") == plan_id else item for item in plans]
    if not any(item.get("id") == plan_id for item in plans):
        plans.append(saved)
    _write_operation_plans(plans)
    return get_operation_plans()


def remove_operation_plan(plan_id: str) -> dict[str, Any]:
    plans = _read_operation_plans()
    remaining = [item for item in plans if item.get("id") != plan_id]
    if len(remaining) == len(plans):
        raise ValueError("操作计划不存在")
    _write_operation_plans(remaining)
    return get_operation_plans()


def _read_similar_pattern_watchlist_state() -> dict[str, Any]:
    default = {
        "symbols": list(SIMILAR_PATTERN_DEFAULT_WATCHLIST),
        "notes": {},
        "pinned": [],
        "alerts": {},
    }
    if not SIMILAR_PATTERN_WATCHLIST_PATH.exists():
        return default
    try:
        payload = json.loads(SIMILAR_PATTERN_WATCHLIST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return default

    raw_symbols = payload.get("symbols", payload) if isinstance(payload, dict) else payload
    symbols: list[str] = []
    for item in raw_symbols if isinstance(raw_symbols, list) else []:
        symbol = _canonical_watch_symbol(str(item))
        if not re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", symbol):
            continue
        if symbol not in symbols:
            symbols.append(symbol)

    raw_notes = payload.get("notes", {}) if isinstance(payload, dict) else {}
    notes: dict[str, dict[str, str]] = {}
    if isinstance(raw_notes, dict):
        for raw_symbol, raw_note in raw_notes.items():
            symbol = _canonical_watch_symbol(str(raw_symbol))
            if symbol not in symbols:
                continue
            if isinstance(raw_note, dict):
                content = str(raw_note.get("content") or "")
                updated_at = str(raw_note.get("updated_at") or "")
            else:
                content = str(raw_note or "")
                updated_at = ""
            if content:
                notes[symbol] = {"content": content, "updated_at": updated_at}
    symbols = symbols or list(SIMILAR_PATTERN_DEFAULT_WATCHLIST)
    raw_pinned = payload.get("pinned", []) if isinstance(payload, dict) else []
    pinned_values = raw_pinned if isinstance(raw_pinned, list) else []
    pinned = [
        symbol
        for symbol in symbols
        if symbol in {_canonical_watch_symbol(str(item)) for item in pinned_values}
    ]
    raw_alerts = payload.get("alerts", {}) if isinstance(payload, dict) else {}
    alerts = {}
    if isinstance(raw_alerts, dict):
        for symbol in symbols:
            alert_config = _normalize_watch_alert_config(raw_alerts.get(symbol, {}))
            if alert_config["reminders"]:
                alerts[symbol] = alert_config
    ordered_symbols = pinned + [symbol for symbol in symbols if symbol not in pinned]
    return {"symbols": ordered_symbols, "notes": notes, "pinned": pinned, "alerts": alerts}


def _read_similar_pattern_watchlist_symbols() -> list[str]:
    return list(_read_similar_pattern_watchlist_state()["symbols"])


def _write_similar_pattern_watchlist_symbols(symbols: list[str]) -> None:
    state = _read_similar_pattern_watchlist_state()
    state["symbols"] = symbols
    _write_similar_pattern_watchlist_state(state)


def _write_similar_pattern_watchlist_state(state: dict[str, Any]) -> None:
    SIMILAR_PATTERN_WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    symbols = list(dict.fromkeys(state.get("symbols", [])))
    pinned_set = set(state.get("pinned", []))
    alerts = {}
    for symbol in symbols:
        alert_config = _normalize_watch_alert_config(state.get("alerts", {}).get(symbol, {}))
        if alert_config["reminders"]:
            alerts[symbol] = alert_config
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": symbols,
        "pinned": [symbol for symbol in symbols if symbol in pinned_set],
        "notes": {
            symbol: state["notes"][symbol]
            for symbol in symbols
            if symbol in state["notes"]
        },
        "alerts": alerts,
    }
    SIMILAR_PATTERN_WATCHLIST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _similar_pattern_watchlist_profiles(
    basic: pd.DataFrame | None = None,
    *,
    include_scores: bool = True,
) -> list[dict[str, Any]]:
    state = _read_similar_pattern_watchlist_state()
    scores = _watchlist_buy_hold_scores(tuple(state["symbols"])) if include_scores else {}
    profiles = []
    for symbol in state["symbols"]:
        profile = _stock_profile_from_basic(symbol, basic)
        note = state["notes"].get(symbol, {})
        profile["note"] = note.get("content", "")
        profile["note_updated_at"] = note.get("updated_at", "")
        profile["pinned"] = symbol in state["pinned"]
        alert_config = _normalize_watch_alert_config(state["alerts"].get(symbol, {}))
        profile["alerts"] = alert_config
        profile["alert_count"] = len(alert_config["reminders"])
        profile["alert_condition_count"] = sum(
            len(reminder["conditions"])
            for reminder in alert_config["reminders"]
        )
        profile.update(scores.get(symbol, {}))
        profiles.append(profile)
    return profiles


def get_similar_pattern_watchlist(*, include_scores: bool = True) -> dict[str, Any]:
    basic = _stock_basic_for_similar_patterns()
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "stocks": _similar_pattern_watchlist_profiles(
            basic,
            include_scores=include_scores,
        ),
    }


def add_similar_pattern_watch_symbol(symbol: str, note: str = "") -> dict[str, Any]:
    normalized = _normalize_watch_symbol(symbol)
    state = _read_similar_pattern_watchlist_state()
    if normalized in state["symbols"]:
        return get_similar_pattern_watchlist(include_scores=False)
    state["symbols"].append(normalized)
    cleaned_note = str(note or "").strip()
    if cleaned_note:
        if len(cleaned_note) > 20_000:
            raise ValueError("笔记不能超过 20000 个字符")
        state["notes"][normalized] = {
            "content": cleaned_note,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    _write_similar_pattern_watchlist_state(state)
    return get_similar_pattern_watchlist(include_scores=False)


def remove_similar_pattern_watch_symbol(symbol: str) -> dict[str, Any]:
    state = _read_similar_pattern_watchlist_state()
    normalized = _canonical_watch_symbol(symbol)
    if normalized not in state["symbols"]:
        raise ValueError(f"股票不在自选池中: {normalized}")
    state["symbols"] = [item for item in state["symbols"] if item != normalized]
    state["pinned"] = [item for item in state["pinned"] if item != normalized]
    state["notes"].pop(normalized, None)
    state["alerts"].pop(normalized, None)
    _write_similar_pattern_watchlist_state(state)
    return get_similar_pattern_watchlist(include_scores=False)


def reorder_similar_pattern_watchlist(symbols: list[str]) -> dict[str, Any]:
    state = _read_similar_pattern_watchlist_state()
    normalized = [_canonical_watch_symbol(symbol) for symbol in symbols]
    if len(normalized) != len(set(normalized)):
        raise ValueError("自选池排序不能包含重复股票")
    if set(normalized) != set(state["symbols"]):
        raise ValueError("自选池排序必须包含当前全部股票")
    pinned_set = set(state["pinned"])
    state["symbols"] = [symbol for symbol in normalized if symbol in pinned_set] + [
        symbol for symbol in normalized if symbol not in pinned_set
    ]
    _write_similar_pattern_watchlist_state(state)
    return get_similar_pattern_watchlist(include_scores=False)


def set_similar_pattern_watch_pin(symbol: str, pinned: bool) -> dict[str, Any]:
    normalized = _canonical_watch_symbol(symbol)
    state = _read_similar_pattern_watchlist_state()
    if normalized not in state["symbols"]:
        raise ValueError(f"股票不在自选池中: {normalized}")
    symbols = [item for item in state["symbols"] if item != normalized]
    current_pinned = [item for item in state["pinned"] if item != normalized]
    if pinned:
        state["pinned"] = [normalized, *current_pinned]
        state["symbols"] = [normalized, *symbols]
    else:
        state["pinned"] = current_pinned
        state["symbols"] = [*symbols[: len(current_pinned)], normalized, *symbols[len(current_pinned) :]]
    _write_similar_pattern_watchlist_state(state)
    return get_similar_pattern_watchlist(include_scores=False)


def save_similar_pattern_watch_note(symbol: str, content: str) -> dict[str, Any]:
    normalized = _canonical_watch_symbol(symbol)
    state = _read_similar_pattern_watchlist_state()
    if normalized not in state["symbols"]:
        raise ValueError(f"股票不在自选池中: {normalized}")
    cleaned = str(content or "").strip()
    if len(cleaned) > 20_000:
        raise ValueError("笔记不能超过 20000 个字符")
    if cleaned:
        state["notes"][normalized] = {
            "content": cleaned,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    else:
        state["notes"].pop(normalized, None)
    _write_similar_pattern_watchlist_state(state)
    return get_similar_pattern_watchlist(include_scores=False)


def save_similar_pattern_watch_alerts(symbol: str, config: dict[str, Any]) -> dict[str, Any]:
    normalized = _canonical_watch_symbol(symbol)
    state = _read_similar_pattern_watchlist_state()
    if normalized not in state["symbols"]:
        raise ValueError(f"股票不在自选池中: {normalized}")
    cleaned = _normalize_watch_alert_config(config, strict=True)
    if cleaned["reminders"]:
        cleaned["updated_at"] = datetime.now().isoformat(timespec="seconds")
        state["alerts"][normalized] = cleaned
    else:
        state["alerts"].pop(normalized, None)
    _write_similar_pattern_watchlist_state(state)
    return get_similar_pattern_watchlist(include_scores=False)


def _read_similar_pattern_validation() -> dict[str, Any]:
    if not SIMILAR_PATTERN_VALIDATION_PATH.exists():
        return {}
    try:
        return json.loads(SIMILAR_PATTERN_VALIDATION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _similar_pattern_validated_symbols(validation: dict[str, Any]) -> set[str]:
    targets = {
        str(symbol).upper()
        for symbol in validation.get("targets") or []
        if str(symbol).strip()
    }
    if targets:
        return targets
    targets.update(
        str(item.get("symbol") or "").upper()
        for item in validation.get("summary") or []
        if isinstance(item, dict) and item.get("symbol")
    )
    model_selection = validation.get("model_selection") or {}
    for horizon_policy in model_selection.values():
        selected = (horizon_policy or {}).get("selected") or {}
        targets.update(str(symbol).upper() for symbol in (selected.get("by_symbol") or {}))
    return targets


def _calibration_max_flat_span(calibration: dict[str, Any] | None) -> float:
    if not calibration:
        return 0.0
    x_values = np.asarray(calibration.get("x") or [], dtype=float)
    y_values = np.asarray(calibration.get("y") or [], dtype=float)
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    max_span = 0.0
    start = 0
    for index in range(1, len(y_values) + 1):
        if index < len(y_values) and np.isclose(y_values[index], y_values[index - 1], atol=1e-6):
            continue
        if index - start >= 2:
            max_span = max(max_span, float(x_values[index - 1] - x_values[start]))
        start = index
    return round(max_span, 4)


def _similar_pattern_probability_policy(
    validation: dict[str, Any],
    symbol: str,
    horizon: str,
) -> dict[str, Any]:
    model_selection = validation.get("model_selection") or {}
    selected = dict(((model_selection.get(horizon) or {}).get("selected") or {}))
    source = str(selected.get("source") or "optimized")
    validated_symbols = _similar_pattern_validated_symbols(validation)
    normalized_symbol = str(symbol).upper()
    reason = "validated_policy"
    applicable = not validated_symbols or normalized_symbol in validated_symbols
    if not applicable:
        source = "optimized"
        reason = "symbol_out_of_validation_scope"
    elif source == "calibrated":
        calibration = (validation.get("calibrations") or {}).get(horizon)
        flat_span = _calibration_max_flat_span(calibration)
        if flat_span > SIMILAR_PATTERN_CALIBRATION_MAX_FLAT_SPAN:
            source = "optimized"
            reason = "calibration_information_collapse"
        selected["calibration_max_flat_span"] = flat_span
    selected.update(
        {
            "source": source,
            "applicable": applicable,
            "reason": reason,
            "validated_symbols": sorted(validated_symbols),
        }
    )
    selected.setdefault("bearish_max", SIMILAR_PATTERN_CONFIG.signal_bearish_max)
    selected.setdefault("bullish_min", SIMILAR_PATTERN_CONFIG.signal_bullish_min)
    selected.setdefault("enable_risk_gate", SIMILAR_PATTERN_CONFIG.enable_risk_gate)
    if reason != "validated_policy":
        selected.update(
            {
                "bearish_max": SIMILAR_PATTERN_CONFIG.signal_bearish_max,
                "bullish_min": SIMILAR_PATTERN_CONFIG.signal_bullish_min,
                "enable_risk_gate": SIMILAR_PATTERN_CONFIG.enable_risk_gate,
            }
        )
    return selected


def _upgrade_cached_similar_pattern_probability_policy(payload: dict[str, Any]) -> dict[str, Any]:
    validation = _read_similar_pattern_validation()
    for result in payload.get("results") or []:
        symbol = str((result.get("target") or {}).get("symbol") or "").upper()
        raw_by_horizon = {
            str(row.get("horizon")): row.get("up_probability")
            for row in result.get("forecast") or []
        }
        decisions_by_horizon = {
            str(row.get("horizon")): row
            for row in result.get("decisions") or []
        }
        for row in result.get("optimized_forecast") or []:
            horizon = str(row.get("horizon") or "")
            policy = _similar_pattern_probability_policy(validation, symbol, horizon)
            source = str(policy.get("source") or "optimized")
            if source in {"optimized", "full_weighting"}:
                selected_probability = row.get("up_probability")
            elif source == "raw_baseline":
                selected_probability = raw_by_horizon.get(horizon)
            elif source == "calibrated":
                selected_probability = row.get("calibrated_up_probability")
            elif row.get("probability_source") == source:
                selected_probability = row.get("selected_up_probability")
            else:
                source = "optimized"
                selected_probability = row.get("up_probability")
                policy["reason"] = "cached_variant_unavailable"
            row["selected_up_probability"] = selected_probability
            row["probability_source"] = source
            row["probability_policy_reason"] = policy.get("reason")
            decision_config = replace(
                SIMILAR_PATTERN_CONFIG,
                signal_bearish_max=float(policy["bearish_max"]),
                signal_bullish_min=float(policy["bullish_min"]),
                enable_risk_gate=bool(policy["enable_risk_gate"]),
            )
            decision = classify_forecast_signal(
                selected_probability,
                result.get("latest_snapshot") or {},
                str(result.get("market_regime") or "neutral"),
                decision_config,
            )
            decision.update(
                {
                    "horizon": horizon,
                    "probability_source": source,
                    "probability_policy_reason": policy.get("reason"),
                }
            )
            decisions_by_horizon[horizon] = decision
        result["decisions"] = list(decisions_by_horizon.values())
    return payload


def _similar_pattern_result_payload(
    result: SimilarPatternResult,
    *,
    profile: dict[str, Any] | None = None,
    market_regime: pd.DataFrame | None = None,
    industry_regime: pd.DataFrame | None = None,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = profile or {}
    validation = validation or {}
    optimized_cases = result.similar_cases.copy()
    market_regime = market_regime if market_regime is not None else pd.DataFrame()
    market_row = _require_exact_regime_row(
        market_regime,
        result.target.target_date,
        context="similar-pattern market regime",
        required_columns=["market_regime"],
    )
    target_market_regime = str(market_row["market_regime"])
    market_regime_feature_date = pd.Timestamp(market_row["date"]).date().isoformat()
    normalized_market_regime = market_regime.copy()
    normalized_market_regime["date"] = pd.to_datetime(
        normalized_market_regime["date"], errors="coerce"
    ).dt.normalize()
    market_map = normalized_market_regime.set_index("date")["market_regime"]
    optimized_cases["date"] = pd.to_datetime(optimized_cases["date"], errors="coerce").dt.normalize()
    optimized_cases["market_regime"] = optimized_cases["date"].map(market_map).fillna("neutral")

    target_industry_regime = "neutral"
    industry_regime_feature_date: str | None = None
    target_industry = str(profile.get("industry") or "")
    if target_industry:
        industry_regime = industry_regime if industry_regime is not None else pd.DataFrame()
        industry_row = _require_exact_regime_row(
            industry_regime,
            result.target.target_date,
            context=f"similar-pattern industry regime ({target_industry})",
            required_columns=["industry_regime"],
        )
        target_industry_regime = str(industry_row["industry_regime"])
        industry_regime_feature_date = pd.Timestamp(industry_row["date"]).date().isoformat()
        normalized_industry_regime = industry_regime.copy()
        normalized_industry_regime["date"] = pd.to_datetime(
            normalized_industry_regime["date"], errors="coerce"
        ).dt.normalize()
        industry_map = normalized_industry_regime.set_index("date")["industry_regime"]
        optimized_cases["industry_regime"] = np.where(
            optimized_cases["industry"].fillna("").astype(str).eq(target_industry),
            optimized_cases["date"].map(industry_map).fillna("neutral"),
            "cross_industry",
        )
    context_cases = optimized_cases.copy()
    optimized_cases, optimization_summary = optimize_similar_cases(
        optimized_cases,
        SIMILAR_PATTERN_CONFIG,
        target_date=result.target.target_date,
        target_industry=str(profile.get("industry") or ""),
        target_market_regime=target_market_regime,
        target_industry_regime=target_industry_regime,
    )
    optimized_forecast = summarize_forecast(optimized_cases)
    variant_cases = build_probability_variant_cases(
        context_cases,
        SIMILAR_PATTERN_CONFIG,
        target_date=result.target.target_date,
        target_industry=str(profile.get("industry") or ""),
        target_market_regime=target_market_regime,
        target_industry_regime=target_industry_regime,
    )
    variant_forecasts = {
        name: summarize_forecast(cases).set_index("horizon")
        for name, cases in variant_cases.items()
    }
    calibration_payload = validation.get("calibrations") or {}
    calibration_map = (
        calibration_payload
        if any(horizon in calibration_payload for horizon in ["next_1d", "next_1m", "next_3m"])
        else calibration_payload.get(result.target.symbol, {})
    )
    model_selection_payload = validation.get("model_selection") or {}
    model_selection = (
        model_selection_payload
        if any(horizon in model_selection_payload for horizon in ["next_1d", "next_1m", "next_3m"])
        else model_selection_payload.get(result.target.symbol, {})
    )
    raw_forecast_map = result.forecast.set_index("horizon") if not result.forecast.empty else pd.DataFrame()
    decisions: list[dict[str, Any]] = []
    for row_index, row in optimized_forecast.iterrows():
        horizon = str(row["horizon"])
        calibrated = apply_probability_calibration(row.get("up_probability"), calibration_map.get(horizon))
        optimized_forecast.at[row_index, "calibrated_up_probability"] = calibrated
        selected_policy = _similar_pattern_probability_policy(
            validation,
            result.target.symbol,
            horizon,
        )
        selected_source = str(selected_policy.get("source") or "raw_baseline")
        if selected_source == "raw_baseline" and horizon in raw_forecast_map.index:
            selected_probability = raw_forecast_map.loc[horizon, "up_probability"]
        elif selected_source in variant_forecasts and horizon in variant_forecasts[selected_source].index:
            selected_probability = variant_forecasts[selected_source].loc[horizon, "up_probability"]
        elif selected_source in {"optimized", "full_weighting"}:
            selected_probability = row.get("up_probability")
        elif selected_source == "calibrated":
            selected_probability = calibrated
        else:
            selected_probability = row.get("up_probability")
        optimized_forecast.at[row_index, "selected_up_probability"] = selected_probability
        optimized_forecast.at[row_index, "probability_source"] = selected_source
        optimized_forecast.at[row_index, "probability_policy_reason"] = selected_policy.get("reason")
        decision_config = replace(
            SIMILAR_PATTERN_CONFIG,
            signal_bearish_max=float(selected_policy.get("bearish_max", SIMILAR_PATTERN_CONFIG.signal_bearish_max)),
            signal_bullish_min=float(selected_policy.get("bullish_min", SIMILAR_PATTERN_CONFIG.signal_bullish_min)),
            enable_risk_gate=bool(selected_policy.get("enable_risk_gate", False)),
        )
        decision = classify_forecast_signal(
            selected_probability,
            result.latest_snapshot,
            target_market_regime,
            decision_config,
        )
        decision["horizon"] = horizon
        decision["probability_source"] = selected_source
        decision["probability_policy_reason"] = selected_policy.get("reason")
        decisions.append(decision)

    top_cases = optimized_cases.copy()
    if not top_cases.empty:
        for col in ["fwd_1d", "fwd_20d", "fwd_60d", "max_drawdown_60d"]:
            if col in top_cases.columns:
                top_cases[col] = (pd.to_numeric(top_cases[col], errors="coerce") * 100).round(2)
        if "similarity" in top_cases.columns:
            raw_similarity = pd.to_numeric(top_cases["similarity"], errors="coerce")
            forecast_weight = pd.to_numeric(
                top_cases.get("forecast_weight", pd.Series(0.0, index=top_cases.index)),
                errors="coerce",
            )
            normalized_weight = (forecast_weight / _similarity_score_ceiling()).clip(lower=0, upper=1)
            top_cases["similarity_score"] = (
                np.log1p(SIMILARITY_SCORE_CONTRAST * normalized_weight)
                / np.log1p(SIMILARITY_SCORE_CONTRAST)
                * 100
            ).round(1)
            top_cases["similarity"] = raw_similarity.round(4)
    return {
        "target": {
            "symbol": result.target.symbol,
            "name": result.target.name,
            "target_date": result.target.target_date.strftime("%Y-%m-%d"),
        },
        "latest_snapshot": _json_ready(result.latest_snapshot),
        "forecast": _frame_records(result.forecast),
        "optimized_forecast": _frame_records(optimized_forecast),
        "decisions": _json_ready(decisions),
        "optimization_summary": _json_ready(optimization_summary),
        "market_regime": target_market_regime,
        "industry_regime": target_industry_regime,
        "feature_freshness": {
            "target_date": result.target.target_date.strftime("%Y-%m-%d"),
            "market_regime_date": market_regime_feature_date,
            "industry_regime_date": industry_regime_feature_date,
            "exact_date_contract": True,
        },
        "validation_summary": [
            item for item in (validation.get("summary") or []) if item.get("symbol") == result.target.symbol
        ],
        "global_policy": _json_ready(model_selection),
        "status_probs": _json_ready(result.status_probs),
        "t1_scenario_plan": _frame_records(result.t1_scenario_plan),
        "sell_model_summary": _json_ready(result.sell_model_summary or {}),
        "sell_model_plan": _frame_records(result.sell_model_plan),
        "top_cases": _frame_records(top_cases, limit=10),
        "scan_summary": _json_ready(result.scan_summary or {}),
    }


def _read_similar_pattern_analysis_cache() -> dict[str, Any] | None:
    if not SIMILAR_PATTERN_ANALYSIS_PATH.exists():
        return None
    try:
        return json.loads(SIMILAR_PATTERN_ANALYSIS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def refresh_similar_pattern_watchlist_scores() -> dict[str, Any]:
    """Refresh lightweight buy/hold scores without rebuilding pattern analysis."""

    profiles = _similar_pattern_watchlist_profiles(
        _stock_basic_for_similar_patterns(),
        include_scores=True,
    )
    missing = [
        str(item.get("symbol") or "")
        for item in profiles
        if _safe_float(item.get("opportunity_score")) is None
        or _safe_float(item.get("holding_score")) is None
    ]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise RuntimeError(
            "自选池买入分或持有分缺失，拒绝覆盖评分缓存: "
            f"missing={len(missing)}/{len(profiles)} ({preview}{suffix})"
        )

    cached = _read_similar_pattern_analysis_cache() or {}
    current_symbols = {
        str(item.get("symbol") or "").upper()
        for item in profiles
        if item.get("symbol")
    }
    preserved_results = [
        item
        for item in cached.get("results") or []
        if str(item.get("target", {}).get("symbol") or "").upper()
        in current_symbols
    ]
    refreshed_at = datetime.now().isoformat(timespec="seconds")
    score_dates = sorted(
        {
            str(item.get("score_date"))
            for item in profiles
            if item.get("score_date")
        }
    )
    payload = {
        **cached,
        "generated_at": cached.get("generated_at") or refreshed_at,
        "watchlist_scores_generated_at": refreshed_at,
        "watchlist_score_dates": score_dates,
        "watchlist": profiles,
        "results": preserved_results,
        "config": cached.get("config") or _similarity_score_config(),
        "global_policy": cached.get("global_policy") or {},
    }
    atomic_write_json(payload, SIMILAR_PATTERN_ANALYSIS_PATH)
    return {
        "status": "success",
        "watchlist_count": len(profiles),
        "scored_count": len(profiles) - len(missing),
        "score_dates": score_dates,
        "analysis_results_preserved": len(preserved_results),
    }


def _collect_watchlist_strategy_hits(symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Collect current cross-workspace hits for watchlist symbols from existing strategy payloads."""
    targets = {str(symbol).upper() for symbol in symbols}
    hits: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in targets}

    def append_hit(symbol: str, key: str, label: str, detail: str, signal_date: Any = None) -> None:
        normalized = _canonical_watch_symbol(symbol)
        if normalized not in targets:
            return
        candidate = {
            "strategy_key": key,
            "strategy_label": label,
            "detail": detail,
            "signal_date": str(signal_date or ""),
        }
        if not any(item["strategy_key"] == key for item in hits[normalized]):
            hits[normalized].append(candidate)

    try:
        latest_signal_date = _latest_candidate_signal_date()
        short_payload = _read_selector_snapshot(
            latest_signal_date,
            None,
            True,
        )
        if short_payload is None:
            short_payload = get_stock_selector_payload(
                signal_date=latest_signal_date,
                include_extended=True,
                use_cache=True,
            )
        # Keep cross-workspace badges aligned with the short-strategy surface:
        # raw snapshot rows that fail the OOT quality gate or fall outside the
        # displayed shortlist must not be presented as visible strategy hits.
        short_payload = _display_selector_payload(short_payload)
        for item in short_payload.get("stocks") or []:
            families = " / ".join(str(value) for value in (item.get("matched_families") or [])[:4])
            append_hit(
                str(item.get("symbol") or ""),
                "short",
                "短线",
                families or f"命中 {int(item.get('matched_count') or 0)} 个策略组",
                short_payload.get("signal_date"),
            )
    except Exception:
        pass

    try:
        chan_payload = get_chan_model_strategy_plan(top_n=20)
        for item in chan_payload.get("candidates") or []:
            append_hit(
                str(item.get("symbol") or ""),
                "chan",
                "缠论",
                str(item.get("rule_name") or item.get("signal_name") or "缠论候选"),
                chan_payload.get("signal_date"),
            )
    except Exception:
        pass

    try:
        long_payload = get_long_stock_pool(variant="tea")
        for item in long_payload.get("stocks") or []:
            append_hit(
                str(item.get("ts_code") or ""),
                "long",
                "长线",
                f"{item.get('state') or '-'} · {item.get('action') or item.get('t_action') or '-'}",
                long_payload.get("signal_date"),
            )
    except Exception:
        pass

    return hits


def _attach_watchlist_strategy_hits(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach strategy badges without changing the similar-pattern forecast itself."""
    symbols = [
        str(item.get("target", {}).get("symbol") or "").upper()
        for item in payload.get("results") or []
        if item.get("target", {}).get("symbol")
    ]
    strategy_hits = _collect_watchlist_strategy_hits(symbols)
    for item in payload.get("results") or []:
        symbol = str(item.get("target", {}).get("symbol") or "").upper()
        item["strategy_hits"] = strategy_hits.get(symbol, [])
        item["strategy_hit_count"] = len(item["strategy_hits"])
    return payload


def _refresh_similar_pattern_analysis_once(
    symbols: list[str],
    progress_callback=None,
    *,
    force_vector_cache: bool = False,
) -> dict[str, Any]:
    basic = _stock_basic_for_similar_patterns()
    source_trade_date = _latest_similar_pattern_target_date(symbols)
    cache_schedule = _similar_pattern_vector_cache_refresh_decision(
        force=force_vector_cache,
        source_trade_date=source_trade_date,
    )
    if cache_schedule["due"]:
        cache_audit = build_vector_caches_parallel(
            DAILY_DIR,
            basic,
            SIMILAR_PATTERN_CONFIG,
            SIMILAR_PATTERN_VECTOR_CACHE_DIR,
            target_symbols=set(symbols),
            workers=max(1, int(os.getenv("SIMILAR_PATTERN_CACHE_WORKERS", "4"))),
            progress_callback=progress_callback,
        )
        cache_errors = cache_audit[cache_audit["status"].eq("error")]
        if not cache_errors.empty:
            examples = ", ".join(
                f"{row.get('symbol', '?')}: {row.get('error', 'unknown error')}"
                for row in cache_errors.head(3).to_dict("records")
            )
            raise RuntimeError(
                f"相似走势周级向量缓存构建失败: errors={len(cache_errors)}; {examples}"
            )
        cache_metadata = _write_similar_pattern_vector_cache_metadata(
            refreshed_at=datetime.now(),
            source_trade_date=source_trade_date,
            cache_audit=cache_audit,
        )
        cache_refreshed = True
    else:
        cache_audit = pd.DataFrame(columns=["status"])
        cache_metadata = dict(cache_schedule["metadata"])
        cache_refreshed = False
    results = analyze_targets_by_threshold(
        DAILY_DIR,
        basic,
        SIMILAR_PATTERN_CONFIG,
        target_symbols=symbols,
        vector_cache_dir=SIMILAR_PATTERN_VECTOR_CACHE_DIR,
        progress_callback=progress_callback,
    )
    missing_targets = [symbol for symbol in symbols if symbol not in results]
    if missing_targets:
        preview = ", ".join(missing_targets[:5])
        suffix = "..." if len(missing_targets) > 5 else ""
        raise RuntimeError(
            f"相似走势未生成全部自选股目标结果: missing={len(missing_targets)}/{len(symbols)} "
            f"({preview}{suffix})"
        )
    target_dates = {
        results[symbol].target.target_date.strftime("%Y-%m-%d")
        for symbol in symbols
        if symbol in results
    }
    if not source_trade_date or (symbols and target_dates != {source_trade_date}):
        raise RuntimeError(
            "相似走势目标特征日期未统一更新到最新交易日: "
            f"expected={source_trade_date} actual={sorted(target_dates)}"
        )
    validation = _read_similar_pattern_validation()
    global_model_selection = validation.get("model_selection") or {}
    next_day_policy = (global_model_selection.get("next_1d", {}).get("selected") or {})
    market_regime = load_market_regime(PROJECT_ROOT / "data/raw/index_000300.SH.parquet")
    profiles = {symbol: _stock_profile_from_basic(symbol, basic) for symbol in symbols}
    target_industries = sorted(
        {
            str(profile.get("industry") or "")
            for profile in profiles.values()
            if profile.get("industry")
        }
    )
    industry_regimes = {
        industry: build_industry_regime(
            DAILY_DIR,
            basic,
            industry,
        )
        for industry in target_industries
    }
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_date": source_trade_date,
        "watchlist": _similar_pattern_watchlist_profiles(basic),
        "config": {
            "candidate_start_date": SIMILAR_PATTERN_CONFIG.candidate_start_date,
            "candidate_step_days": SIMILAR_PATTERN_CONFIG.candidate_step_days,
            "similarity_threshold": SIMILAR_PATTERN_CONFIG.similarity_threshold,
            "take_profit_3d": SIMILAR_PATTERN_CONFIG.take_profit_3d,
            "stop_loss_3d": SIMILAR_PATTERN_CONFIG.stop_loss_3d,
            "signal_bearish_max": next_day_policy.get(
                "bearish_max", SIMILAR_PATTERN_CONFIG.signal_bearish_max
            ),
            "signal_bullish_min": next_day_policy.get(
                "bullish_min", SIMILAR_PATTERN_CONFIG.signal_bullish_min
            ),
            "max_effective_cases": SIMILAR_PATTERN_CONFIG.max_effective_cases,
            "max_events_per_date": SIMILAR_PATTERN_CONFIG.max_events_per_date,
            **_similarity_score_config(),
        },
        "global_policy": _json_ready(global_model_selection),
        "results": [
            _similar_pattern_result_payload(
                results[symbol],
                profile=profiles[symbol],
                market_regime=market_regime,
                industry_regime=industry_regimes.get(str(profiles[symbol].get("industry") or "")),
                validation=validation,
            )
            for symbol in symbols
            if symbol in results
        ],
        "cache": {
            "hit": False,
            "backend": "generated",
            "rebuilt": int(cache_audit["status"].eq("built").sum()) if not cache_audit.empty else 0,
            "reused": (
                int(cache_audit["status"].eq("cache_hit").sum())
                if not cache_audit.empty
                else int(cache_schedule["cached_files"])
            ),
            "errors": int(cache_audit["status"].eq("error").sum()) if not cache_audit.empty else 0,
            "reference_library_policy": "weekly",
            "reference_library_refreshed": cache_refreshed,
            "reference_library_reason": cache_schedule["reason"],
            "reference_library_refreshed_at": cache_metadata.get("refreshed_at"),
            "reference_library_refresh_age_days": cache_schedule.get("refresh_age_days"),
            "reference_library_minimum_refresh_age_days": cache_schedule[
                "minimum_refresh_age_days"
            ],
            "reference_library_next_refresh_at": (
                _next_similar_pattern_vector_cache_refresh_at(
                    datetime.fromisoformat(str(cache_metadata["refreshed_at"]))
                ).isoformat(timespec="seconds")
                if cache_refreshed and cache_metadata.get("refreshed_at")
                else cache_schedule.get("next_refresh_at")
            ),
            "reference_library_source_trade_date": cache_metadata.get("source_trade_date"),
            "target_vectors": "live_from_latest_daily_data",
        },
    }
    payload = _attach_watchlist_strategy_hits(payload)
    SIMILAR_PATTERN_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIMILAR_PATTERN_ANALYSIS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def refresh_similar_pattern_analysis(
    progress_callback=None,
    *,
    force_vector_cache: bool = False,
) -> dict[str, Any]:
    """Run one analysis per watchlist revision and share it with concurrent callers."""
    global _SIMILAR_PATTERN_REFRESH_INFLIGHT

    requested_symbols = tuple(_read_similar_pattern_watchlist_symbols())
    requested_key = (requested_symbols, bool(force_vector_cache))

    while True:
        with _SIMILAR_PATTERN_REFRESH_GUARD:
            active = _SIMILAR_PATTERN_REFRESH_INFLIGHT
            if active is None:
                future: Future[dict[str, Any]] = Future()
                _SIMILAR_PATTERN_REFRESH_INFLIGHT = (requested_key, future)
                owner = True
                active_key = requested_key
            else:
                active_key, future = active
                owner = False

        if owner:
            try:
                payload = _refresh_similar_pattern_analysis_once(
                    list(requested_symbols),
                    progress_callback=progress_callback,
                    force_vector_cache=force_vector_cache,
                )
            except BaseException as exc:
                future.set_exception(exc)
                raise
            else:
                future.set_result(payload)
                return payload
            finally:
                with _SIMILAR_PATTERN_REFRESH_GUARD:
                    if (
                        _SIMILAR_PATTERN_REFRESH_INFLIGHT is not None
                        and _SIMILAR_PATTERN_REFRESH_INFLIGHT[1] is future
                    ):
                        _SIMILAR_PATTERN_REFRESH_INFLIGHT = None

        if active_key == requested_key:
            payload = future.result()
            return {
                **payload,
                "cache": {
                    **(payload.get("cache") or {}),
                    "coalesced": True,
                },
            }

        # A different watchlist revision is being calculated. Let it finish,
        # then re-read the latest revision and start or join its refresh.
        try:
            future.result()
        except Exception:
            pass
        requested_symbols = tuple(_read_similar_pattern_watchlist_symbols())
        requested_key = (requested_symbols, bool(force_vector_cache))


def _similar_patterns_worker(result_queue: mp.Queue) -> None:
    def emit_progress(message: str) -> None:
        result_queue.put({"type": "progress", "message": message})

    try:
        payload = refresh_similar_pattern_analysis(progress_callback=emit_progress)
    except Exception as exc:
        result_queue.put(
            {
                "type": "result",
                "ok": False,
                "error": f"{exc}\n{traceback.format_exc(limit=5)}",
            }
        )
        return
    result_queue.put({"type": "result", "ok": True, "payload": payload})


def _drain_similar_pattern_worker_queue(result_queue: mp.Queue) -> dict[str, Any] | None:
    final_result: dict[str, Any] | None = None
    while True:
        try:
            message = result_queue.get_nowait()
        except queue.Empty:
            return final_result
        if message.get("type") == "progress":
            _set_refresh_progress(
                step_key="similar_patterns",
                message=f"相似走势决策台：{message.get('message') or '仍在计算'}",
                percent=97,
                complete_previous=False,
            )
            continue
        final_result = dict(message)


def _run_similar_pattern_analysis_isolated(timeout_seconds: int = SIMILAR_PATTERNS_TIMEOUT_SECONDS) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_similar_patterns_worker, args=(result_queue,), daemon=False)
    proc.start()
    _register_active_worker("similar_patterns", proc)
    try:
        deadline = monotonic() + timeout_seconds
        result: dict[str, Any] | None = None
        while proc.is_alive():
            result = _drain_similar_pattern_worker_queue(result_queue) or result
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            proc.join(timeout=min(5.0, remaining))

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            raise TimeoutError(
                f"相似走势决策台刷新超过 {timeout_seconds // 60} 分钟无结果，已终止卡住的子进程"
            )
        result = _drain_similar_pattern_worker_queue(result_queue) or result
        if result is None:
            raise RuntimeError("相似走势决策台刷新未返回结果")
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "相似走势决策台刷新失败"))
        return dict(result.get("payload") or {})
    finally:
        _clear_active_worker("similar_patterns")


def get_similar_pattern_analysis(refresh: bool = False) -> dict[str, Any]:
    if not refresh:
        cached = _read_similar_pattern_analysis_cache()
        cached_symbols = [
            str(item.get("target", {}).get("symbol") or "").upper()
            for item in (cached or {}).get("results", [])
        ]
        watchlist_symbols = _read_similar_pattern_watchlist_symbols()
        if cached is not None:
            cached = _upgrade_cached_similar_pattern_probability_policy(cached)
            cached_watchlist = {
                str(item.get("symbol") or "").upper(): item
                for item in cached.get("watchlist") or []
                if item.get("symbol")
            }
            result_by_symbol = {
                str(item.get("target", {}).get("symbol") or "").upper(): item
                for item in cached.get("results", [])
                if item.get("target", {}).get("symbol")
            }
            cached["results"] = [
                result_by_symbol[symbol]
                for symbol in watchlist_symbols
                if symbol in result_by_symbol
            ]
            cached = _attach_watchlist_strategy_hits(cached)
            persisted_profiles = _similar_pattern_watchlist_profiles(
                _stock_basic_for_similar_patterns(),
                include_scores=False,
            )
            score_fields = (
                "opportunity_score",
                "holding_score",
                "buy_score",
                "hold_score",
                "score_date",
                "score_target",
                "score_feature_source",
                "feature_quality",
                "model_score_available",
            )
            cached["watchlist"] = [
                {
                    **profile,
                    **{
                        field: cached_watchlist[profile["symbol"]][field]
                        for field in score_fields
                        if field in cached_watchlist.get(profile["symbol"], {})
                    },
                }
                for profile in persisted_profiles
            ]
            cached.setdefault("config", {}).update(_similarity_score_config())
            cached["cache"] = {
                **(cached.get("cache") or {}),
                "hit": True,
                "backend": "json",
                "stale": not _is_daily_payload_current(cached),
                "watchlist_changed": cached_symbols != watchlist_symbols,
            }
            return cached
        watchlist = get_similar_pattern_watchlist(include_scores=False)
        return {
            "generated_at": watchlist.get("updated_at"),
            "watchlist": watchlist.get("stocks") or [],
            "results": [],
            "config": _similarity_score_config(),
            "global_policy": {},
            "cache": {
                "hit": False,
                "backend": "none",
                "stale": True,
                "missing": True,
                "watchlist_changed": True,
            },
        }
    return refresh_similar_pattern_analysis()


def _signal_group_key(signal: dict[str, Any]) -> str:
    family = str(signal.get("strategy_family") or signal.get("strategy_key") or "").upper()
    return str(signal.get("strategy_group") or STRATEGY_MEMBER_TO_GROUP.get(family, family))


def _signal_group_label(signal: dict[str, Any]) -> str:
    group = _signal_group_key(signal)
    return STRATEGY_GROUP_LABELS.get(group, group)


def _strategy_filter_members(strategies: list[str] | None) -> set[str]:
    selected = {item.upper() for item in strategies or [] if item}
    members: set[str] = set()
    for item in selected:
        members.update(STRATEGY_GROUP_MEMBERS.get(item, {item}))
    return members


def _strategy_filter_groups(strategies: list[str] | None) -> set[str]:
    selected = {item.upper() for item in strategies or [] if item}
    groups: set[str] = set()
    for item in selected:
        groups.add(item if item in STRATEGY_GROUP_MEMBERS else STRATEGY_MEMBER_TO_GROUP.get(item, item))
    return groups


def _signal_matches_filter(signal: dict[str, Any], selected_members: set[str], selected_groups: set[str]) -> bool:
    strategy_key = str(signal.get("strategy_key") or "").upper()
    family = str(signal.get("strategy_family") or strategy_key).upper()
    group = _signal_group_key(signal).upper()
    return strategy_key in selected_members or family in selected_members or group in selected_groups


def _enrich_signal_group(signal: dict[str, Any]) -> dict[str, Any]:
    out = dict(signal)
    group = _signal_group_key(out)
    out["strategy_group"] = group
    out["strategy_group_label"] = STRATEGY_GROUP_LABELS.get(group, group)
    return out


def _selector_snapshot_key(
    signal_date: str | None,
    strategies: list[str] | None,
    include_extended: bool,
) -> tuple[str, str, str]:
    strategy_key = ",".join(sorted({str(item).upper() for item in strategies or [] if item})) or "ALL"
    date_key = signal_date or "LATEST"
    raw = json.dumps(
        {
            "signal_date": date_key,
            "strategies": strategy_key,
            "include_extended": include_extended,
            "schema_version": SELECTOR_SNAPSHOT_SCHEMA_VERSION,
            "ranking_source": DEFAULT_SELECTOR_RANKING_CONFIG.source.value,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest(), date_key, strategy_key


def _selector_snapshot_path(snapshot_key: str) -> Path:
    return SELECTOR_SNAPSHOT_DIR / f"{snapshot_key}.json"


def _selector_snapshot_matches_current_ranking_source(
    payload: Mapping[str, Any],
) -> bool:
    """Reject cache fallback across a selector ranking-source cutover."""

    return str(payload.get("ranking_source") or "") == (
        DEFAULT_SELECTOR_RANKING_CONFIG.source.value
    )


@lru_cache(maxsize=32)
def _selector_snapshot_dates_cached(
    strategies: tuple[str, ...],
    include_extended: bool,
) -> tuple[str, ...]:
    _, _, strategy_key = _selector_snapshot_key(None, strategies, include_extended)
    dates: set[str] = set()
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    if store.config.sql_url:
        try:
            from sqlalchemy import text

            with store._engine().begin() as conn:
                rows = conn.execute(
                    text(
                        f"""
                        SELECT DISTINCT signal_date
                        FROM {SELECTOR_SNAPSHOT_TABLE}
                        WHERE signal_date <> 'LATEST'
                          AND strategies_key = :strategies_key
                          AND include_extended = :include_extended
                        """
                    ),
                    {"strategies_key": strategy_key, "include_extended": include_extended},
                ).mappings().all()
            for row in rows:
                value = row.get("signal_date")
                parsed = pd.to_datetime(value, errors="coerce")
                if pd.notna(parsed) and parsed.weekday() < 5:
                    dates.add(parsed.strftime("%Y-%m-%d"))
        except Exception:
            pass

    if SELECTOR_SNAPSHOT_DIR.exists():
        for path in SELECTOR_SNAPSHOT_DIR.glob("*.json"):
            try:
                payload = read_json_file(path)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            candidate_date = str(payload.get("signal_date") or "")
            if not candidate_date or candidate_date == "LATEST":
                continue
            snapshot_key, date_key, key = _selector_snapshot_key(candidate_date, strategies, include_extended)
            if key != strategy_key or snapshot_key != path.stem:
                continue
            parsed = pd.to_datetime(date_key, errors="coerce")
            if pd.notna(parsed) and parsed.weekday() < 5:
                dates.add(parsed.strftime("%Y-%m-%d"))
    return tuple(sorted(dates))


def _selector_snapshot_dates(strategies: list[str] | None, include_extended: bool) -> list[str]:
    normalized = tuple(sorted({str(item).upper() for item in strategies or [] if item}))
    return list(_selector_snapshot_dates_cached(normalized, include_extended))


def _latest_selector_snapshot_date(strategies: list[str] | None, include_extended: bool) -> str | None:
    dates = _selector_snapshot_dates(strategies, include_extended)
    return dates[-1] if dates else None


def _resolve_selector_signal_date(
    signal_date: str | None,
    strategies: list[str] | None,
    include_extended: bool,
) -> str | None:
    snapshot_dates = _selector_snapshot_dates(strategies, include_extended)
    if not signal_date:
        return snapshot_dates[-1] if snapshot_dates else _latest_candidate_signal_date()
    target = pd.to_datetime(signal_date, errors="raise")
    target_iso = target.strftime("%Y-%m-%d")
    if not snapshot_dates:
        return target_iso
    if target.weekday() < 5 and (
        target_iso in snapshot_dates
        or target_iso == _latest_candidate_signal_date()
    ):
        return target_iso
    previous = [item for item in snapshot_dates if item <= target_iso]
    return previous[-1] if previous else target_iso


@lru_cache(maxsize=1)
def _latest_candidate_signal_date() -> str | None:
    """Latest date available from strategy candidate/model caches."""

    latest_dates: list[pd.Timestamp] = []
    parquet_paths = [FAMILY_SIGNAL_CACHE, PROJECT_ROOT / "data/features/z_skill_daily_candidates.parquet", EXTENDED_MODEL_SCORED]
    for path in parquet_paths:
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path, columns=["date"])
        except Exception:
            continue
        if frame.empty:
            continue
        latest = pd.to_datetime(frame["date"], errors="coerce").max()
        if pd.notna(latest):
            latest_dates.append(latest)
    if latest_dates:
        return max(latest_dates).strftime("%Y-%m-%d")
    return None


def _read_selector_snapshot(
    signal_date: str | None,
    strategies: list[str] | None,
    include_extended: bool,
) -> dict[str, Any] | None:
    snapshot_key, _, _ = _selector_snapshot_key(signal_date, strategies, include_extended)
    path = _selector_snapshot_path(snapshot_key)
    if path.exists():
        try:
            payload = read_json_file(path)
            if not _selector_snapshot_matches_current_ranking_source(payload):
                return None
            payload["cache"] = {"hit": True, "backend": "filesystem", "snapshot_key": snapshot_key}
            return payload
        except Exception:
            pass
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    if store.config.sql_url:
        try:
            from sqlalchemy import text

            with store._engine().begin() as conn:
                row = conn.execute(
                    text(f"SELECT payload_json FROM {SELECTOR_SNAPSHOT_TABLE} WHERE snapshot_key = :snapshot_key"),
                    {"snapshot_key": snapshot_key},
                ).mappings().first()
                if not row and signal_date:
                    _, _, strategy_key = _selector_snapshot_key(signal_date, strategies, include_extended)
                    row = conn.execute(
                        text(
                            f"""
                            SELECT snapshot_key, payload_json
                            FROM {SELECTOR_SNAPSHOT_TABLE}
                            WHERE signal_date = :signal_date
                              AND strategies_key = :strategies_key
                              AND include_extended = :include_extended
                            ORDER BY generated_at DESC
                            LIMIT 1
                            """
                        ),
                        {
                            "signal_date": signal_date,
                            "strategies_key": strategy_key,
                            "include_extended": include_extended,
                        },
                    ).mappings().first()
            if row and row.get("payload_json"):
                payload = json.loads(row["payload_json"])
                if not signal_date or str(payload.get("signal_date") or "") == str(signal_date):
                    if not _selector_snapshot_matches_current_ranking_source(payload):
                        return None
                    stored_key = str(row.get("snapshot_key") or snapshot_key)
                    if stored_key != snapshot_key and payload.get("selector_snapshot_schema_version") != SELECTOR_SNAPSHOT_SCHEMA_VERSION:
                        return None
                    payload["cache"] = {
                        "hit": True,
                        "backend": "mysql",
                        "snapshot_key": stored_key,
                        "key_fallback": stored_key != snapshot_key,
                    }
                    return payload
        except Exception:
            pass

    return None


def _prepare_selector_snapshot_write(
    payload: dict[str, Any],
    strategies: list[str] | None,
    include_extended: bool,
) -> dict[str, Any]:
    signal_date = str(payload.get("signal_date") or "")
    snapshot_key, date_key, strategy_key = _selector_snapshot_key(
        signal_date,
        strategies,
        include_extended,
    )
    payload_to_store = dict(payload)
    payload_to_store["selector_snapshot_schema_version"] = SELECTOR_SNAPSHOT_SCHEMA_VERSION
    payload_to_store["snapshot_scope"] = {
        "strategies": sorted({str(item).upper() for item in strategies or [] if item}) or ["ALL"],
        "include_extended": bool(include_extended),
    }
    payload_to_store["cache"] = {"hit": False, "backend": "generated", "snapshot_key": snapshot_key}
    payload_json = json.dumps(payload_to_store, ensure_ascii=False, default=str)
    return {
        "snapshot_path": _selector_snapshot_path(snapshot_key),
        "payload_json": payload_json,
        "sql_values": {
            "snapshot_key": snapshot_key,
            "signal_date": date_key,
            "strategies_key": strategy_key,
            "include_extended": include_extended,
            "generated_at": str(
                payload.get("generated_at")
                or datetime.now().isoformat(timespec="seconds")
            ),
            "stock_count": len(payload.get("stocks") or []),
            "payload_json": payload_json,
        },
    }


def _selector_snapshot_schema_cache_key(sql_url: str) -> str:
    return f"{sql_url}|{SELECTOR_SNAPSHOT_TABLE}"


def _ensure_selector_snapshot_schema(engine: Any, sql_url: str) -> None:
    """Initialize the selector snapshot schema once per process and SQL target."""

    cache_key = _selector_snapshot_schema_cache_key(sql_url)
    if cache_key in _SELECTOR_SNAPSHOT_SCHEMA_READY_URLS:
        return
    with _SELECTOR_SNAPSHOT_SCHEMA_LOCK:
        if cache_key in _SELECTOR_SNAPSHOT_SCHEMA_READY_URLS:
            return
        from sqlalchemy import text

        with engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {SELECTOR_SNAPSHOT_TABLE} (
                        snapshot_key VARCHAR(64) PRIMARY KEY,
                        signal_date VARCHAR(16) NOT NULL,
                        strategies_key VARCHAR(512) NOT NULL,
                        include_extended BOOLEAN NOT NULL,
                        generated_at VARCHAR(32) NOT NULL,
                        stock_count INT NOT NULL,
                        payload_json LONGTEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    )
                    """
                )
            )
            try:
                conn.execute(
                    text(
                        f"ALTER TABLE {SELECTOR_SNAPSHOT_TABLE} "
                        "ADD COLUMN include_extended BOOLEAN NOT NULL DEFAULT 0"
                    )
                )
            except Exception:
                pass
            try:
                legacy_flag_column = "include_" + "z" "_skill"
                conn.execute(
                    text(
                        f"ALTER TABLE {SELECTOR_SNAPSHOT_TABLE} "
                        f"MODIFY COLUMN {legacy_flag_column} BOOLEAN NOT NULL DEFAULT 0"
                    )
                )
            except Exception:
                pass
        _SELECTOR_SNAPSHOT_SCHEMA_READY_URLS.add(cache_key)


def _write_selector_snapshot_batch(
    snapshots: list[tuple[dict[str, Any], list[str] | None, bool]],
) -> None:
    if not snapshots:
        return
    prepared = [
        _prepare_selector_snapshot_write(payload, strategies, include_extended)
        for payload, strategies, include_extended in snapshots
    ]

    SELECTOR_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    published_files = 0
    try:
        for item in prepared:
            atomic_write_text(item["payload_json"], item["snapshot_path"])
            published_files += 1
    finally:
        if published_files:
            _selector_snapshot_dates_cached.cache_clear()

    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    sql_url = store.config.sql_url
    if not sql_url:
        return
    engine = None
    schema_cache_key = _selector_snapshot_schema_cache_key(str(sql_url))
    try:
        from sqlalchemy import text

        engine = store._engine()
        _ensure_selector_snapshot_schema(engine, str(sql_url))
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {SELECTOR_SNAPSHOT_TABLE}
                        (snapshot_key, signal_date, strategies_key, include_extended,
                         generated_at, stock_count, payload_json)
                    VALUES
                        (:snapshot_key, :signal_date, :strategies_key, :include_extended,
                         :generated_at, :stock_count, :payload_json)
                    ON DUPLICATE KEY UPDATE
                        generated_at = VALUES(generated_at),
                        stock_count = VALUES(stock_count),
                        payload_json = VALUES(payload_json)
                    """
                ),
                [item["sql_values"] for item in prepared],
            )
    except Exception:
        with _SELECTOR_SNAPSHOT_SCHEMA_LOCK:
            _SELECTOR_SNAPSHOT_SCHEMA_READY_URLS.discard(schema_cache_key)
        return
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass


def _write_selector_snapshot(
    payload: dict[str, Any],
    strategies: list[str] | None,
    include_extended: bool,
) -> None:
    _write_selector_snapshot_batch([(payload, strategies, include_extended)])


def _strategy_keys_from_payload(payload: dict[str, Any]) -> list[str]:
    keys = []
    for item in payload.get("available_strategies") or []:
        key = str(item.get("key") or "").upper()
        if key:
            keys.append(key)
    return sorted(set(keys))


def _filtered_selector_payload(payload: dict[str, Any], strategies: list[str]) -> dict[str, Any]:
    selected = {item.upper() for item in strategies if item}
    selected_members = _strategy_filter_members(strategies)
    selected_groups = _strategy_filter_groups(strategies)
    rows = []
    for stock in payload.get("stocks") or []:
        signals = [
            signal
            for signal in stock.get("signals") or []
            if _signal_matches_filter(signal, selected_members, selected_groups)
        ]
        signals = _dedupe_signals_by_operation(signals)
        if not signals:
            continue
        row = {key: value for key, value in stock.items() if key != "signals"}
        row["signals"] = signals
        built = build_selector_stock_row(row, signals, payload.get("signal_date"))
        if built is not None:
            rows.append(built)
    rows = sorted(
        rows,
        key=lambda item: (item["selector_score"], item["matched_count"], item["best_profit_factor"]),
        reverse=True,
    )
    rows = _apply_historical_score_normalization(rows)
    rows = apply_selector_ranking_source(
        rows,
        str(payload.get("signal_date") or "") or None,
    )
    rows = sorted(
        rows,
        key=lambda item: (item["selector_score"], item["matched_count"], item["best_profit_factor"]),
        reverse=True,
    )
    filtered = dict(payload)
    filtered["stocks"] = rows
    filtered["snapshot_scope"] = {"strategies": sorted(selected)}
    filtered["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return filtered


def _write_strategy_pool_snapshots(payload: dict[str, Any], include_extended: bool) -> dict[str, Any]:
    strategy_keys = _strategy_keys_from_payload(payload)
    extended_keys = {str(item.get("key") or "").upper() for item in EXTENDED_STRATEGIES}
    snapshots: list[tuple[dict[str, Any], list[str] | None, bool]] = [
        (payload, None, include_extended)
    ]
    written = {"ALL": len(payload.get("stocks") or [])}
    for strategy_key in strategy_keys:
        filtered = _filtered_selector_payload(payload, [strategy_key])
        members = STRATEGY_GROUP_MEMBERS.get(strategy_key, {strategy_key})
        snapshots.append(
            (filtered, [strategy_key], bool(members & extended_keys))
        )
        written[strategy_key] = len(filtered.get("stocks") or [])
    _write_selector_snapshot_batch(snapshots)
    return written


def _run_post_snapshot_cache_cleanup(results: dict[str, Any]) -> dict[str, Any]:
    """Enforce retention after the refresh writes its newest snapshot version."""

    summary = run_cache_cleanup(PROJECT_ROOT)
    results["cache_cleanup_after_snapshot"] = summary
    sql_status = str((summary.get("sql_snapshots") or {}).get("status") or "")
    if summary.get("status") != "success" or sql_status in {"failed", "partial"}:
        errors = summary.get("errors") or []
        detail = "; ".join(str(error) for error in errors[:3]) or f"sql_status={sql_status or 'unknown'}"
        raise RuntimeError(f"快照写入后缓存保留清理失败: {detail}")
    return summary


def _display_selector_payload(payload: dict[str, Any], limit: int = DEFAULT_SELECTOR_LIMIT) -> dict[str, Any]:
    rows = [dict(row) for row in payload.get("stocks") or []]
    complete_total = int(payload.get("total_stock_count") or len(rows))
    display_rows = [row for row in rows if _row_display_quality_gate(row)]
    if any(
        row.get("buy_score_source") != "historical_return_model"
        or row.get("hold_score_source") != "historical_return_model"
        for row in display_rows
    ):
        rescored_rows = _apply_historical_score_normalization(
            [dict(row) for row in display_rows]
        )
        display_rows = [
            rescored
            if rescored.get("buy_score_source") == "historical_return_model"
            and rescored.get("hold_score_source") == "historical_return_model"
            else original
            for original, rescored in zip(display_rows, rescored_rows)
        ]
    display_rows = apply_selector_ranking_source(
        display_rows,
        str(payload.get("signal_date") or "") or None,
    )
    display_rows = sorted(
        display_rows,
        key=lambda item: (item["selector_score"], item["matched_count"], item["best_profit_factor"]),
        reverse=True,
    )
    if not (payload.get("snapshot_scope") or {}).get("strategies"):
        display_rows = _diversify_default_rows(display_rows, len(display_rows))
    _apply_selector_score_presentation(display_rows)
    total = len(display_rows)
    display = display_rows[:limit] if limit > 0 else display_rows
    out = dict(payload)
    out["stocks"] = display
    out["total_stock_count"] = total
    out["complete_stock_count"] = complete_total
    out["display_limit"] = limit
    out["is_truncated"] = total > len(display)
    out["score_presentation"] = {
        "schema_version": "selector-score-presentation-v1",
        "range": [0.0, 100.0],
        "direction": "higher_is_better",
        "rank_scope": "current_actionable_candidate_pool",
        "rank_method": "competition_rank_descending",
        "fields": {
            "model": "model_score_normalized",
            "buy": "buy_score_normalized",
            "hold": "hold_score_normalized",
        },
        "targets": {
            "model": "T+5 好路径排序（冲高且控制回撤）",
            "buy": "T+5 最大冲高收益",
            "hold": "T+5 收盘收益",
        },
    }
    out["score_probability_bands"] = _selector_score_probability_bands()
    return out


def _score_in_display_range(value: Any) -> float | None:
    score = _safe_float(value, None)
    if score is None or not 0.0 <= score <= 100.0:
        return None
    return round(float(score), 1)


def _competition_ranks(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    rank_field: str,
) -> None:
    values = [
        float(row[score_field])
        for row in rows
        if row.get(score_field) is not None
    ]
    ordered = sorted(values, reverse=True)
    rank_by_score: dict[float, int] = {}
    for index, score in enumerate(ordered, start=1):
        rank_by_score.setdefault(score, index)
    count = len(ordered)
    count_field = f"{rank_field}_count"
    for row in rows:
        score = row.get(score_field)
        row[rank_field] = rank_by_score.get(float(score)) if score is not None else None
        row[count_field] = count


def _apply_selector_score_presentation(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose one 0-100, higher-is-better display contract for all scores."""

    model_source_labels = {
        SelectorRankingSource.RIGHT_SIDE_UNIFIED.value: "右侧统一模型",
        "left_side_unified": "左侧统一模型",
    }
    for row in rows:
        ranking_source = str(row.get("ranking_source") or "")
        model_score = None
        if ranking_source in model_source_labels:
            model_score = _score_in_display_range(
                row.get("ranking_score_normalized", row.get("ranking_score_percent"))
            )
        row["model_score_normalized"] = model_score
        row["buy_score_normalized"] = _score_in_display_range(
            row.get("opportunity_score")
        )
        row["hold_score_normalized"] = _score_in_display_range(
            row.get("holding_score")
        )
        row["model_score_source_label"] = model_source_labels.get(
            ranking_source, "统一模型未覆盖"
        )
        row["score_normalization_schema_version"] = (
            "selector-score-presentation-v1"
        )

    for prefix in ("model", "buy", "hold"):
        _competition_ranks(
            rows,
            score_field=f"{prefix}_score_normalized",
            rank_field=f"{prefix}_score_rank",
        )
    return rows


def _score_probability_source_path(value: Any) -> Path:
    path = (PROJECT_ROOT / str(value or "")).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("score probability source escapes project root") from exc
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _selector_score_probability_bands() -> dict[str, Any]:
    """Load OOT band frequencies only when they match active artifacts."""

    unavailable = {
        "available": False,
        "schema_version": "selector-score-probability-bands-v1",
        "reason": "分档概率校准不可用或已过期",
        "calibrations": [],
    }
    try:
        payload = json.loads(
            SELECTOR_SCORE_PROBABILITY_BANDS.read_text(encoding="utf-8")
        )
        if payload.get("schema_version") != (
            "selector-score-probability-bands-v1"
        ):
            return unavailable
        calibrations = payload.get("calibrations")
        if not isinstance(calibrations, list) or not calibrations:
            return unavailable
        source_hashes: dict[Path, str] = {}
        for calibration in calibrations:
            if not isinstance(calibration, dict):
                return unavailable
            source = calibration.get("source")
            bands = calibration.get("bands")
            if not isinstance(source, dict) or not isinstance(bands, list):
                return unavailable
            for path_field, hash_field in (
                ("artifact_path", "artifact_sha256"),
                ("sample_path", "sample_sha256"),
            ):
                source_path = _score_probability_source_path(
                    source.get(path_field)
                )
                if not source_path.is_file():
                    return unavailable
                digest = source_hashes.get(source_path)
                if digest is None:
                    digest = _file_sha256(source_path)
                    source_hashes[source_path] = digest
                if digest != str(source.get(hash_field) or ""):
                    return unavailable
            for band in bands:
                probability = _safe_float((band or {}).get("probability_pct"), None)
                sample_count = int((band or {}).get("sample_count") or 0)
                if probability is None or not 0.0 <= probability <= 100.0:
                    return unavailable
                if sample_count <= 0:
                    return unavailable
        return {**payload, "available": True}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return unavailable


def get_selector_calendar(start: str = "2026-06-01", end: str | None = None) -> dict[str, Any]:
    start_date = date.fromisoformat(start)
    snapshot_dates = set(_selector_snapshot_dates(None, True))
    long_snapshot_dates = _long_stock_pool_snapshot_dates("tea")
    chan_strategy_dates = _chan_model_strategy_dates()
    chan_snapshot_dates = _workspace_snapshot_dates("chan_model_strategy", params={"top_n": 20})
    latest_snapshot = max(snapshot_dates) if snapshot_dates else None
    latest_long_snapshot = max(long_snapshot_dates) if long_snapshot_dates else None
    latest_chan_signal = max(chan_strategy_dates) if chan_strategy_dates else None
    latest_chan_snapshot = max(chan_snapshot_dates) if chan_snapshot_dates else None
    if end:
        end_date = date.fromisoformat(end)
    elif latest_snapshot:
        end_date = max(date.today(), date.fromisoformat(latest_snapshot) + timedelta(days=7))
    else:
        end_date = date.today()
    if end_date < start_date:
        end_date = start_date
    days: list[dict[str, Any]] = []
    cursor = start_date
    while cursor <= end_date:
        iso = cursor.isoformat()
        is_closed = cursor.weekday() >= 5
        previous = sorted(item for item in snapshot_dates if item <= iso)
        effective = previous[-1] if previous else None
        if is_closed:
            status = "closed"
            label = "休市"
        elif iso in snapshot_dates:
            status = "ready"
            label = "已生成"
        else:
            status = "open_missing_data"
            label = "缺数据"
        days.append(
            {
                "date": iso,
                "status": status,
                "label": label,
                "is_open": not is_closed,
                "has_selector_snapshot": iso in snapshot_dates,
                "has_long_stock_pool_snapshot": iso in long_snapshot_dates,
                "has_chan_model_strategy": iso in chan_strategy_dates,
                "has_chan_model_strategy_snapshot": iso in chan_snapshot_dates,
                "effective_signal_date": effective,
                "disabled": is_closed,
            }
        )
        cursor += timedelta(days=1)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "latest_signal_date": latest_snapshot,
        "latest_long_signal_date": latest_long_snapshot,
        "latest_chan_signal_date": latest_chan_signal,
        "latest_chan_snapshot_date": latest_chan_snapshot,
        "days": days,
    }


def get_b1_plan(refresh: bool = False, signal_date: str | None = None) -> dict[str, Any]:
    if refresh or signal_date or not DAILY_PLAN_PATH.exists():
        payload = build_daily_plan(signal_date=signal_date)
        return payload
    return read_json_file(DAILY_PLAN_PATH)


def refresh_b1_plan(signal_date: str | None = None) -> dict[str, Any]:
    output_path = write_daily_plan(signal_date=signal_date)
    return read_json_file(output_path)


def get_dashboard() -> dict[str, Any]:
    if DASHBOARD_PATH.exists():
        return read_json_file(DASHBOARD_PATH)
    try:
        return build_dashboard_payload()
    except FileNotFoundError:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "latest_signal_date": None,
            "strategies": [],
            "all_periods": [],
            "latest_signals": [],
            "recent_daily": [],
            "notes": ["暂未找到历史复盘文件，请先运行例行 dashboard 任务。"],
        }


def _convertible_bond_grid_payload_is_usable(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    quality = payload.get("data_quality") or {}
    if quality:
        coverage = _safe_float(quality.get("premium_coverage"), 0.0) or 0.0
        minimum = _safe_float(quality.get("minimum_premium_coverage"), 0.90) or 0.90
        if quality.get("status") != "success" or quality.get("stale") or coverage < minimum:
            return False
    candidates = [item for item in payload.get("candidates") or [] if isinstance(item, dict)]
    if len(candidates) < 3:
        return True
    premiums = [
        value
        for item in candidates
        if (value := _safe_float(item.get("premium_rate"), None)) is not None
    ]
    if len(premiums) / len(candidates) < 0.80:
        return False
    return not (max(premiums) - min(premiums) < 1e-6 and abs(premiums[0]) < 1e-6)


def get_convertible_bond_grid_plan(
    trade_date: str | None = None,
    limit: int = DEFAULT_CONVERTIBLE_BOND_GRID_LIMIT,
    refresh: bool = False,
) -> dict[str, Any]:
    def read_snapshot(*args, **kwargs) -> dict[str, Any] | None:
        payload = _read_workspace_snapshot(*args, **kwargs)
        return payload if _convertible_bond_grid_payload_is_usable(payload) else None

    def read_legacy_snapshot() -> dict[str, Any] | None:
        try:
            payload = json.loads(
                CONVERTIBLE_BOND_GRID_PLAN_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            return None
        return payload if _convertible_bond_grid_payload_is_usable(payload) else None

    params = {"limit": int(limit)}
    dependencies = ConvertibleBondGridDependencies(
        read_snapshot=read_snapshot,
        read_legacy_snapshot=read_legacy_snapshot,
        promote_legacy_snapshot=lambda snapshot_date, payload: (
            _write_filesystem_workspace_snapshot(
                "convertible_bond_grid_plan",
                _workspace_params_key(params),
                snapshot_date,
                json.dumps(payload, ensure_ascii=False, default=str),
            )
        ),
        write_snapshot=_write_workspace_snapshot,
        refresh_daily=refresh_convertible_bond_daily,
        build_plan=build_convertible_bond_grid_plan,
    )
    return build_convertible_bond_grid_workspace(
        trade_date=trade_date,
        limit=limit,
        refresh=refresh,
        dependencies=dependencies,
    )


def _convertible_bond_allotment_min_coverage() -> float:
    try:
        configured = float(os.getenv("ROUTINE_ALLOTMENT_MIN_COVERAGE", "0.90"))
    except (TypeError, ValueError):
        configured = 0.90
    return min(1.0, max(0.0, configured))


def _convertible_bond_allotment_quality(
    payload: dict[str, Any],
    *,
    expected_trade_date: str | None = None,
) -> dict[str, Any]:
    return evaluate_convertible_bond_allotment_quality(
        payload,
        expected_trade_date=expected_trade_date,
        minimum_coverage=_convertible_bond_allotment_min_coverage(),
    )


def get_convertible_bond_allotments(
    limit: int = DEFAULT_ALLOTMENT_LIMIT,
    include_listed_days: int = DEFAULT_ALLOTMENT_INCLUDE_LISTED_DAYS,
    refresh: bool = False,
    stage_scope: str = DEFAULT_ALLOTMENT_STAGE_SCOPE,
    expected_trade_date: str | None = None,
    validate_quality: bool = False,
) -> dict[str, Any]:
    dependencies = ConvertibleBondAllotmentDependencies(
        read_snapshot=_read_workspace_snapshot,
        read_daily_cache=lambda: _read_daily_payload_cache(
            CONVERTIBLE_BOND_ALLOTMENT_DAILY_PATH
        ),
        write_daily_cache=lambda payload: _write_daily_payload_cache(
            CONVERTIBLE_BOND_ALLOTMENT_DAILY_PATH,
            payload,
        ),
        write_snapshot=_write_workspace_snapshot,
        build_payload=build_convertible_bond_allotment_payload,
        is_daily_current=_is_daily_payload_current,
    )
    return build_convertible_bond_allotment_workspace(
        limit=limit,
        include_listed_days=include_listed_days,
        refresh=refresh,
        stage_scope=stage_scope,
        expected_trade_date=expected_trade_date,
        validate_quality=validate_quality,
        dependencies=dependencies,
        minimum_coverage=_convertible_bond_allotment_min_coverage(),
    )


def refresh_dashboard() -> dict[str, Any]:
    output_path = write_dashboard_json()
    return read_json_file(output_path)


def latest_report_file(pattern: str) -> Path | None:
    candidates = sorted(REPORT_DIR.glob(pattern))
    return candidates[-1] if candidates else None


def latest_vegas_report_file(pattern: str) -> Path | None:
    candidates = sorted(VEGAS_TUNNEL_REPORT_DIR.glob(pattern))
    return candidates[-1] if candidates else None


def dataframe_records(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    return json.loads(df.head(limit).to_json(orient="records", force_ascii=False))


def get_research_index(limit: int = 200) -> dict[str, Any]:
    family_path = latest_report_file(FAMILY_RULE_PATTERN)
    fusion_path = latest_report_file(FUSION_PATTERN)
    family_rows = dataframe_records(family_path, limit=limit) if family_path else []
    fusion_rows = dataframe_records(fusion_path, limit=limit) if fusion_path else []
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_dir": str(REPORT_DIR),
        "web_data_dir": str(WEB_DATA_DIR),
        "family_rule_report": str(family_path) if family_path else None,
        "fusion_report": str(fusion_path) if fusion_path else None,
        "family_rule_rows": family_rows,
        "fusion_rows": fusion_rows,
    }


def _numeric(value: Any, default: float | None = None) -> float | None:
    if value is None or pd.isna(value):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _chan_candidate_record(row: pd.Series) -> dict[str, Any]:
    return {
        "date": str(pd.to_datetime(row.get("date")).date()) if pd.notna(row.get("date")) else "",
        "symbol": str(row.get("symbol") or ""),
        "name": _clean_text(row.get("name")),
        "rule_id": _clean_text(row.get("chan_model_rule_id")),
        "rule_name": _clean_text(row.get("chan_model_rule_name")),
        "rule_description": _clean_text(row.get("chan_model_description")),
        "rank_score": _numeric(row.get("chan_model_rank_score")),
        "signal_name": _clean_text(row.get("chan_signal_name")),
        "chan_score": _numeric(row.get("chan_score")),
        "pred_good": _numeric(row.get("pred_target_good")),
        "pred_big10": _numeric(row.get("pred_target_big10")),
        "pred_win10": _numeric(row.get("pred_target_win10")),
        "entry_gap_pct": _numeric(row.get("entry_gap_pct")),
        "position_pct": _numeric(row.get("chan_model_position_pct")),
        "close": _numeric(row.get("close")),
        "entry_open": _numeric(row.get("entry_open")),
        "center_low": _numeric(row.get("chan_center_low")),
        "center_high": _numeric(row.get("chan_center_high")),
        "center_width": _numeric(row.get("chan_center_width")),
        "stroke_amplitude": _numeric(row.get("chan_stroke_amplitude")),
        "buy_plan": _clean_text(row.get("chan_model_buy_plan")),
        "sell_plan": _clean_text(row.get("chan_model_sell_plan")),
        "structure_note": _clean_text(row.get("chan_structure_note")),
    }


def _chan_summary_records(summary: pd.DataFrame) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    return [
        {
            "rule_id": str(row.get("rule_id") or ""),
            "split": str(row.get("split") or ""),
            "rows": int(row.get("rows") or 0),
            "avg_return_10d": _numeric(row.get("avg_return_10d")),
            "median_return_10d": _numeric(row.get("median_return_10d")),
            "win_rate_10d": _numeric(row.get("win_rate_10d")),
            "big_win_rate_10d": _numeric(row.get("big_win_rate_10d")),
            "profit_factor_10d": _numeric(row.get("profit_factor_10d")),
        }
        for _, row in summary.iterrows()
    ]


def _build_chan_model_strategy_payload(top_n: int = 20, signal_date: str | None = None) -> dict[str, Any]:
    if not CHAN_MODEL_SCORED_PATH.exists():
        raise FileNotFoundError(f"缺少缠论模型评分文件: {CHAN_MODEL_SCORED_PATH}")
    scored = pd.read_parquet(CHAN_MODEL_SCORED_PATH)
    strategy_frame = add_chan_model_strategy_columns(scored)
    if signal_date is None:
        atomic_write_parquet(strategy_frame, CHAN_MODEL_STRATEGY_SCORED_PATH, index=False)
        atomic_write_csv(
            strategy_frame,
            CHAN_MODEL_STRATEGY_DIR / "chan_model_strategy_scored.csv",
            index=False,
        )

    candidates = select_chan_model_candidates(strategy_frame, trade_date=signal_date, top_n=top_n)
    if signal_date is None:
        atomic_write_csv(candidates, CHAN_MODEL_LATEST_CANDIDATES_PATH, index=False)
    summary = summarize_chan_model_strategy(strategy_frame)
    if signal_date is None:
        atomic_write_csv(summary, CHAN_MODEL_SUMMARY_PATH, index=False)

    signal_date = (
        str(pd.to_datetime(candidates["date"].iloc[0]).date())
        if not candidates.empty
        else signal_date
    )
    records = [_chan_candidate_record(row) for _, row in candidates.iterrows()]
    summary_records = _chan_summary_records(summary)
    oot_primary = next(
        (item for item in summary_records if item["split"] == "oot" and item["rule_id"] == "chan_model_primary"),
        None,
    )
    oot_expanded = next(
        (item for item in summary_records if item["split"] == "oot" and item["rule_id"] == "chan_model_expanded"),
        None,
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": signal_date,
        "top_n": int(top_n),
        "candidate_count": len(records),
        "primary_count": sum(1 for item in records if item["rule_id"] == "chan_model_primary"),
        "expanded_count": sum(1 for item in records if item["rule_id"] == "chan_model_expanded"),
        "candidates": records,
        "summary": summary_records,
        "oot_primary": oot_primary,
        "oot_expanded": oot_expanded,
        "execution_notes": [
            "T日收盘确认缠论三买与模型分，T+1 开盘执行。",
            "主策略优先，扩容策略仅在候选不足时补充。",
            "T+1 高开不超过 3% 可执行；3%-6% 降仓观察；超过 6% 放弃。",
            "优先持有 5-10 个交易日；跌破信号日低点或最近中枢下沿退出。",
        ],
        "files": {
            "scored": str(CHAN_MODEL_STRATEGY_SCORED_PATH),
            "candidates": str(CHAN_MODEL_LATEST_CANDIDATES_PATH),
            "summary": str(CHAN_MODEL_SUMMARY_PATH),
        },
    }
    return payload


def get_chan_model_strategy_plan(
    top_n: int = 20,
    refresh: bool = False,
    signal_date: str | None = None,
) -> dict[str, Any]:
    params = {"top_n": int(top_n)}
    if not refresh:
        cached = _read_workspace_snapshot(
            "chan_model_strategy",
            snapshot_date=signal_date,
            params=params,
            allow_sql=False,
        )
        if cached is not None:
            return cached
    payload = _build_chan_model_strategy_payload(top_n=top_n, signal_date=signal_date)
    if signal_date and str(payload.get("signal_date") or "") != str(signal_date):
        raise RuntimeError(
            f"缠论结果日期未更新到最新交易日: expected={signal_date} actual={payload.get('signal_date')}"
        )
    _write_workspace_snapshot(
        "chan_model_strategy",
        payload.get("signal_date"),
        payload,
        params=params,
        write_sql=refresh,
    )
    return payload


def refresh_chan_model_strategy_plan(top_n: int = 20, signal_date: str | None = None) -> dict[str, Any]:
    return get_chan_model_strategy_plan(top_n=top_n, refresh=True, signal_date=signal_date)


def _fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}%"


def _fmt_rate(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def _fmt_price(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _metrics_payload(row: pd.Series | dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    getter = row.get if isinstance(row, dict) else row.get
    fields = [
        "trades",
        "avg_return_pct",
        "win_rate",
        "max_drawdown_pct",
        "profit_factor",
        "stop_rate",
        "take_profit_rate",
        "expiry_rate",
    ]
    out: dict[str, Any] = {}
    for field in fields:
        value = getter(field, None)
        if value is None or pd.isna(value):
            out[field] = None
        elif field == "trades":
            out[field] = int(value)
        else:
            out[field] = float(value)
    return out


def _metrics_text(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "暂无正式回测指标"
    return (
        f"OOT {metrics.get('trades') or 0} 笔，"
        f"均值 {_fmt_pct(metrics.get('avg_return_pct'))}，"
        f"胜率 {_fmt_rate(metrics.get('win_rate'))}，"
        f"最大回撤 {_fmt_pct(metrics.get('max_drawdown_pct'))}，"
        f"PF {float(metrics.get('profit_factor') or 0):.2f}（前复权口径）"
    )


def _entry_rule_text(rule: str) -> str:
    if not rule:
        return ""
    parts: list[str] = []
    up5 = re.search(r"up5_ge_(\d+\.\d+)", rule)
    up8 = re.search(r"up8_ge_(\d+\.\d+)", rule)
    down3 = re.search(r"down3_le_(\d+\.\d+)", rule)
    if up5:
        parts.append(f"模型判断 5 日内冲高 5% 的概率 >= {float(up5.group(1)):.0%}")
    if up8:
        parts.append(f"模型判断 5 日内冲高 8% 的概率 >= {float(up8.group(1)):.0%}")
    if down3:
        parts.append(f"模型判断跌破风险概率 <= {float(down3.group(1)):.0%}")
    return "；".join(parts) if parts else rule


def _split_signal_and_open_filters(text: str) -> tuple[str, str]:
    if not text:
        return "", "T+1 开盘观察"
    parts = [part.strip() for part in re.split(r"[，,；;]", text) if part.strip()]
    signal_parts = [
        part for part in parts
        if "信号日" in part or ("收盘位置" in part and "T+1" not in part)
    ]
    open_parts = [part for part in parts if part not in signal_parts]
    return "；".join(signal_parts), "；".join(open_parts) if open_parts else text


def _buy_plan_text(
    open_filter: str,
    *,
    prefix: str = "",
    action_level: str = "",
    intraday_approx: bool = False,
) -> str:
    signal_filter, execution_filter = _split_signal_and_open_filters(open_filter)
    parts: list[str] = []
    if prefix:
        parts.append(prefix.rstrip("；"))
    if signal_filter:
        parts.append(f"信号确认条件（T日收盘后已确认）：{signal_filter}")
    parts.append(f"T+1 开盘执行条件：{execution_filter}")
    if action_level:
        parts.append(f"实操分层：{action_level}")
    suffix = "当前页面只给出日线观察名单，正式买点需要分钟级确认。" if intraday_approx else "不满足开盘条件则空仓观察。"
    return "；".join(parts) + f"。{suffix}"


def _exit_rule_text(rule: str) -> str:
    if not rule:
        return "按策略卖出规则执行"
    expiry = re.fullmatch(r"expiry_T(\d+)_close", rule)
    if expiry:
        return f"不设固定止盈止损，买入后最多持有到 T+{expiry.group(1)}，到期按收盘价卖出"

    legacy_fixed = re.fullmatch(r"fixed_tp(\d+)_sl(\d+)_T(\d+)", rule)
    if legacy_fixed:
        tp_raw, sl_raw, hold = legacy_fixed.groups()
        tp = float(tp_raw)
        sl = float(sl_raw) / 10
        return (
            f"固定止盈止损：买入后若盈利达到 {tp:.1f}% 则止盈卖出；"
            f"若亏损达到 {sl:.1f}% 则盘中触发止损卖出；"
            f"若一直未触发止盈/止损，最多持有到 T+{hold}，到期按收盘价卖出"
        )

    fixed = re.fullmatch(r"fixed_tp(\d+(?:\.\d+)?)%_sl(\d+(?:\.\d+)?)%_(intraday|close)_T(\d+)", rule)
    if fixed:
        tp, sl, trigger, hold = fixed.groups()
        trigger_text = "盘中触发" if trigger == "intraday" else "收盘确认"
        return (
            f"固定止盈止损：买入后若盈利达到 {tp}% 则止盈卖出；"
            f"若亏损达到 {sl}% 则{trigger_text}止损卖出；"
            f"若一直未触发止盈/止损，最多持有到 T+{hold}，到期按收盘价卖出"
        )

    trailing = re.fullmatch(
        r"trail_target(\d+(?:\.\d+)?)%_dd(\d+(?:\.\d+)?)%_sl(\d+(?:\.\d+)?)%_(intraday|close)_T(\d+)",
        rule,
    )
    if trailing:
        target, drawdown, sl, trigger, hold = trailing.groups()
        trigger_text = "盘中触发" if trigger == "intraday" else "收盘确认"
        return (
            f"目标后回撤卖出：买入后若先盈利达到 {target}%，开始跟踪最高点；"
            f"之后若从高点回撤 {drawdown}% 则卖出；"
            f"同时设置 {sl}% {trigger_text}止损；"
            f"若未触发上述条件，最多持有到 T+{hold}，到期按收盘价卖出"
        )

    legacy_trailing = re.fullmatch(r"trail_target(\d+)_dd(\d+)_sl(\d+)_T(\d+)", rule)
    if legacy_trailing:
        target_raw, drawdown_raw, sl_raw, hold = legacy_trailing.groups()
        target = float(target_raw)
        drawdown = float(drawdown_raw)
        sl = float(sl_raw) / 10
        return (
            f"目标后回撤卖出：买入后若先盈利达到 {target:.1f}%，开始跟踪最高点；"
            f"之后若从高点回撤 {drawdown:.1f}% 则卖出；"
            f"同时设置 {sl:.1f}% 盘中触发止损；"
            f"若未触发上述条件，最多持有到 T+{hold}，到期按收盘价卖出"
        )

    return rule


@lru_cache(maxsize=1)
def _extended_playbooks() -> dict[str, pd.Series]:
    if not EXTENDED_PLAYBOOK.exists():
        return {}
    df = pd.read_csv(EXTENDED_PLAYBOOK)
    if "signal" not in df.columns:
        return {}
    return {str(row["signal"]): row for _, row in df.iterrows()}


@lru_cache(maxsize=1)
def _extended_model_playbooks() -> dict[str, pd.Series]:
    if not EXTENDED_MODEL_PLAYBOOK.exists():
        return {}
    df = pd.read_csv(EXTENDED_MODEL_PLAYBOOK)
    if "signal" not in df.columns:
        return {}
    return {str(row["signal"]): row for _, row in df.iterrows()}


@lru_cache(maxsize=1)
def _extended_model_risk_managed_playbooks() -> dict[str, pd.Series]:
    if not EXTENDED_MODEL_SUMMARY.exists():
        return {}
    df = pd.read_csv(EXTENDED_MODEL_SUMMARY)
    oot = df[df["split"] == "oot"].copy()
    if oot.empty:
        return {}
    oot = oot[~oot["exit_rule"].astype(str).str.startswith("expiry")].copy()
    if oot.empty:
        return {}
    oot = oot[
        (oot["trades"].fillna(0) >= 8)
        & (oot["avg_return_pct"].fillna(0) >= 0)
        & (oot["profit_factor"].fillna(0) >= 1.2)
        & (oot["max_drawdown_pct"].fillna(-100) >= -35)
    ].copy()
    if oot.empty:
        return {}
    oot["selection_score"] = (
        oot["avg_return_pct"].fillna(0) * 0.45
        + np.minimum(oot["profit_factor"].fillna(0), 4) * 1.3
        + oot["win_rate"].fillna(0) * 2.5
        + oot["max_drawdown_pct"].fillna(-100) / 20
        + np.minimum(oot["trades"].fillna(0), 500) / 500
    )
    best = oot.sort_values(["signal", "selection_score"], ascending=[True, False]).groupby("signal").head(1)
    return {str(row["signal"]): row for _, row in best.iterrows()}


def _model_playbook_for(signal_key: str) -> pd.Series | None:
    playbook = _extended_model_playbooks().get(signal_key)
    if playbook is not None and str(playbook.get("exit_rule", "")).startswith("expiry"):
        risk_managed = _extended_model_risk_managed_playbooks().get(signal_key)
        if risk_managed is not None:
            adjusted = risk_managed.copy()
            for field in ["action_level", "signal_description"]:
                if field in playbook.index and field not in adjusted.index:
                    adjusted[field] = playbook.get(field)
                elif field in playbook.index and pd.isna(adjusted.get(field)):
                    adjusted[field] = playbook.get(field)
            return adjusted
    return playbook


@lru_cache(maxsize=8)
def _model_scored_candidates_for_date(signal_date: str | None = None) -> dict[tuple[str, str], pd.Series]:
    # Once both unified rankers are active, per-strategy Z-skill model scores
    # are rollback-only artifacts.  Signal discovery still comes from the
    # canonical rule caches, while ordering comes exclusively from the two
    # unified score contracts.
    if (
        DEFAULT_SELECTOR_RANKING_CONFIG.source
        == SelectorRankingSource.RIGHT_SIDE_UNIFIED
        and DEFAULT_LEFT_SIDE_RANKING_CONFIG.enabled
    ):
        return {}
    if not EXTENDED_MODEL_SCORED.exists():
        return {}
    df = pd.read_parquet(EXTENDED_MODEL_SCORED)
    if df.empty or not {"symbol", "signal", "model_pass"} <= set(df.columns):
        return {}
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if signal_date:
            target_date = pd.to_datetime(signal_date)
            df = df[df["date"] == target_date].copy()
        else:
            df = df[df["date"] == df["date"].max()].copy()
    result = {
        (str(row["symbol"]), str(row["signal"])): row
        for _, row in df[df["model_pass"].fillna(False).astype(bool)].iterrows()
    }
    if (
        DEFAULT_SELECTOR_RANKING_CONFIG.source
        == SelectorRankingSource.RIGHT_SIDE_UNIFIED
    ):
        preserved = set(DEFAULT_SELECTOR_RANKING_CONFIG.preserved_legacy_signals)
        result = {
            key: value
            for key, value in result.items()
            if key[1].upper() in preserved
        }
    return result


def _model_scored_candidates() -> dict[tuple[str, str], pd.Series]:
    return _model_scored_candidates_for_date(None)


def _model_score_reason(row: pd.Series | None) -> str:
    if row is None:
        return ""
    parts = []
    for label, col in [("up5", "pred_up5"), ("up8", "pred_up8"), ("down3", "pred_down3")]:
        value = row.get(col)
        if pd.notna(value):
            parts.append(f"{label}={float(value):.3f}")
    return "，".join(parts)


def _extended_signal_payload(signal: dict[str, Any], model_score: pd.Series | None = None) -> dict[str, Any]:
    strategy_key = str(signal.get("strategy_key"))
    model_playbook = _model_playbook_for(strategy_key) if model_score is not None else None
    playbook = model_playbook if model_playbook is not None else _extended_playbooks().get(strategy_key)
    if playbook is None:
        return _enrich_signal_group(signal)
    metrics = _metrics_payload(playbook)
    action_level = str(playbook.get("action_level") or "观察")
    entry_rule = str(playbook.get("entry_rule") or "")
    open_filter = str(playbook.get("open_filter_description") or signal.get("buy_plan") or "T+1 开盘观察")
    exit_rule = str(playbook.get("exit_rule") or signal.get("sell_plan") or "按策略卖出")
    source = "模型版" if model_playbook is not None else "规则版"
    enriched = dict(signal)
    enriched["metrics"] = metrics
    enriched["action_level"] = action_level
    enriched["playbook_source"] = source
    enriched["metrics_text"] = _metrics_text(metrics)
    threshold_text = f"模型买入条件：{_entry_rule_text(entry_rule)}" if entry_rule and model_playbook is not None else ""
    enriched["buy_plan"] = _buy_plan_text(open_filter, prefix=threshold_text, action_level=action_level)
    enriched["sell_plan"] = f"{_exit_rule_text(exit_rule)}。依据：该策略{source}买卖组合回测 playbook。"
    enriched["logic"] = f"{signal.get('logic')}（已完成该策略{source}买卖组合回测，当前结论：{action_level}）"
    model_reason = _model_score_reason(model_score)
    if model_reason:
        enriched["reason"] = f"{signal.get('reason')}；模型分 {model_reason}"
        enriched["model_score"] = {
            "pred_up5": float(model_score.get("pred_up5")) if pd.notna(model_score.get("pred_up5")) else None,
            "pred_up8": float(model_score.get("pred_up8")) if pd.notna(model_score.get("pred_up8")) else None,
            "pred_down3": float(model_score.get("pred_down3")) if pd.notna(model_score.get("pred_down3")) else None,
        }
    if action_level == "可小仓实操":
        enriched["strength_score"] = float(enriched.get("strength_score") or 0) + 1.5
    elif action_level in {"谨慎观察", "谨慎实操", "模型观察"}:
        enriched["strength_score"] = float(enriched.get("strength_score") or 0) + 0.4
    return _enrich_signal_group(enriched)


def _model_filtered_signal_payload(signal_key: str, model_score: pd.Series) -> dict[str, Any]:
    playbook = _model_playbook_for(signal_key)
    metrics = _metrics_payload(playbook) if playbook is not None else None
    action_level = str(playbook.get("action_level") or "模型观察") if playbook is not None else "模型观察"
    entry_rule = str(playbook.get("entry_rule") or "") if playbook is not None else ""
    open_filter = str(playbook.get("open_filter_description") or "T+1 开盘观察") if playbook is not None else "T+1 开盘观察"
    exit_rule = str(playbook.get("exit_rule") or "按策略卖出") if playbook is not None else "按策略卖出"
    model_reason = _model_score_reason(model_score)
    label = MODEL_SIGNAL_LABELS.get(signal_key, signal_key)
    return _enrich_signal_group({
        "strategy_key": signal_key,
        "strategy_family": signal_key,
        "strategy_name": label,
        "timeframe": "日线级，收盘确认，T+1 开盘观察",
        "logic": f"{label} 规则候选，并通过该策略独立 XGBoost 模型分过滤。",
        "reason": f"模型分 {model_reason}",
        "buy_plan": _buy_plan_text(
            open_filter,
            prefix=f"模型买入条件：{_entry_rule_text(entry_rule)}",
            action_level=action_level,
        ),
        "sell_plan": f"{_exit_rule_text(exit_rule)}。依据：该策略模型版买卖组合回测 playbook。",
        "metrics": metrics,
        "metrics_text": _metrics_text(metrics),
        "action_level": action_level,
        "playbook_source": "模型版",
        "model_score": {
            "pred_up5": float(model_score.get("pred_up5")) if pd.notna(model_score.get("pred_up5")) else None,
            "pred_up8": float(model_score.get("pred_up8")) if pd.notna(model_score.get("pred_up8")) else None,
            "pred_down3": float(model_score.get("pred_down3")) if pd.notna(model_score.get("pred_down3")) else None,
        },
        "strength_score": max(float(model_score.get("pred_up5") or 0) - float(model_score.get("pred_down3") or 0), 0) * 3 + 1.2,
    })


@lru_cache(maxsize=1)
def _vegas_tunnel_best_metrics() -> pd.Series | None:
    path = latest_vegas_report_file("vegas_tunnel_param_grid_summary_*.csv")
    if path is None:
        path = latest_vegas_report_file("vegas_tunnel_summary_*.csv")
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    part = df[df["period"].astype(str) == "all"].copy() if "period" in df.columns else df.copy()
    if part.empty:
        part = df.copy()
    if "trades" in part.columns:
        liquid = part[pd.to_numeric(part["trades"], errors="coerce").fillna(0) >= 20].copy()
        if not liquid.empty:
            part = liquid
    if "rank_score" not in part.columns:
        part["rank_score"] = (
            part["daily_sharpe"].fillna(0)
            + 0.20 * part["avg_return_pct"].fillna(0)
            + 0.02 * part["max_drawdown_pct"].fillna(0)
            + part["profit_factor"].fillna(0) * 0.30
        )
    return part.sort_values(["rank_score", "daily_sharpe", "avg_return_pct"], ascending=False).iloc[0]


def _vegas_tunnel_signal_payload(latest_row: pd.Series) -> dict[str, Any]:
    best = _vegas_tunnel_best_metrics()
    metrics = _metrics_payload(best) if best is not None else {
        "trades": 42,
        "avg_return_pct": 1.52,
        "win_rate": 0.4524,
        "max_drawdown_pct": -31.18,
        "profit_factor": 1.7853,
    }
    exit_rule = str(best.get("exit_rule") if best is not None else "fixed_tp16%_sl8%_T9")
    param_text = ""
    if best is not None and "fast_span" in best.index:
        param_text = (
            f"参数：EMA{int(best['fast_span'])}/EMA{int(best['momentum_span'])}，"
            f"隧道EMA{int(best['tunnel_short_span'])}/{int(best['tunnel_long_span'])}，"
            f"回踩{float(best['near_tunnel_pct']):.1%}，窗口{int(best['pullback_window'])}日。"
        )
    distance = _safe_float(latest_row.get("vegas_tunnel_distance"), 0.0) or 0.0
    slope = _safe_float(latest_row.get("vegas_tunnel_slope_20d"), 0.0) or 0.0
    volume_strength = _safe_float(latest_row.get("vegas_volume_strength"), 0.0) or 0.0
    score = _safe_float(latest_row.get("vegas_candidate_score"), 0.0) or 0.0
    return _enrich_signal_group({
        "strategy_key": "VEGAS_TUNNEL",
        "strategy_family": "VEGAS",
        "strategy_name": "维加斯隧道",
        "timeframe": "日线级，收盘确认，T+1 开盘观察",
        "logic": f"EMA144/169 形成上行隧道，EMA10>EMA20>隧道上沿；近8日回踩或贴近隧道2.5%范围后，当日收阳放量重新站上 EMA10。{param_text}",
        "reason": (
            f"距隧道上沿={distance:.2%}，"
            f"隧道20日斜率={slope:.2%}，"
            f"量能强度={volume_strength:.2%}，"
            f"候选分={score:.3f}"
        ),
        "buy_plan": "T+1 开盘观察；高开过大不追，优先选择开盘不显著脱离确认日收盘且不跌破确认日低点的标的。",
        "sell_plan": f"{_exit_rule_text(exit_rule)}。依据：维加斯隧道策略多卖出计划回测。",
        "metrics": metrics,
        "metrics_text": _metrics_text(metrics),
        "action_level": "谨慎实操",
        "playbook_source": "规则版",
        "strength_score": 1.0 + min(max(score, 0.0), 1.0),
    })


def _stock_basic_profile(symbol: str) -> dict[str, str]:
    basic = _stock_basic_map().get(symbol, {})
    return {
        "name": _clean_text(basic.get("name")),
        "industry": _clean_text(basic.get("industry")),
    }


def _fill_stock_profile(stock: dict[str, Any], signal_date: str | None = None) -> dict[str, Any]:
    profile = _stock_basic_profile(str(stock.get("symbol") or ""))
    if not _clean_text(stock.get("name")) and profile["name"]:
        stock["name"] = profile["name"]
    if not _clean_text(stock.get("industry")) and profile["industry"]:
        stock["industry"] = profile["industry"]
    try:
        close = float(stock.get("close") or 0)
    except (TypeError, ValueError):
        close = 0
    if close <= 0:
        daily_profile = _daily_profile_at_or_before(str(stock.get("symbol") or ""), signal_date)
        if daily_profile.get("close"):
            stock["close"] = daily_profile["close"]
        if not stock.get("date") and daily_profile.get("date"):
            stock["date"] = daily_profile["date"]
    return stock


@lru_cache(maxsize=16384)
def _daily_profile_at_or_before(symbol: str, signal_date: str | None = None) -> dict[str, Any]:
    if not symbol:
        return {}
    try:
        store = MarketDataStore(MarketDataStoreConfig.from_env(root=DAILY_DIR.parent))
        cols = store.read_frame(DAILY_DIR.name, symbol)
        wanted = [col for col in ["date", "trade_date", "close"] if col in cols.columns]
        cols = cols[wanted].copy() if wanted else cols
    except Exception:
        path = DAILY_DIR / f"{symbol}.parquet"
        if not path.exists():
            return {}
        try:
            cols = pd.read_parquet(path, columns=["date", "trade_date", "close"])
        except Exception:
            try:
                cols = pd.read_parquet(path)
            except Exception:
                return {}
    if cols.empty or "close" not in cols.columns:
        return {}
    parsed_date = pd.to_datetime(cols["date"], errors="coerce") if "date" in cols.columns else pd.Series(pd.NaT, index=cols.index)
    if "trade_date" in cols.columns:
        trade_date = pd.to_datetime(cols["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        parsed_date = parsed_date.fillna(trade_date)
    cols["date"] = parsed_date
    cols = cols.dropna(subset=["close"]).sort_values("date")
    cols = cols[pd.to_numeric(cols["close"], errors="coerce") > 0]
    if signal_date:
        target_date = pd.to_datetime(signal_date)
        cols = cols[cols["date"] <= target_date].copy()
    if cols.empty:
        return {}
    row = cols.iloc[-1]
    return {
        "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row.get("date")) else None,
        "close": float(row["close"]),
    }


def _latest_daily_profile(symbol: str) -> dict[str, Any]:
    return _daily_profile_at_or_before(symbol, None)


def _signal_quality_gate(signal: dict[str, Any]) -> bool:
    """Keep the default selector focused on signals with usable OOT evidence."""
    family = str(signal.get("strategy_family") or "")
    group = _signal_group_key(signal)
    if family == "B1":
        return True

    if family in MODEL_FILTERED_SIGNALS and family != "B2" and signal.get("playbook_source") != "模型版":
        return False

    metrics = signal.get("metrics") or {}
    trades = float(metrics.get("trades") or 0)
    avg_return = float(metrics.get("avg_return_pct") or 0)
    win_rate = float(metrics.get("win_rate") or 0)
    max_drawdown = float(metrics.get("max_drawdown_pct") or 0)
    profit_factor = float(metrics.get("profit_factor") or 0)
    action_level = str(signal.get("action_level") or "")

    if family in {"B2", "B3", "SB1", "SUPER_B1"}:
        return (
            trades >= 8
            and avg_return >= 1.0
            and profit_factor >= 1.8
            and max_drawdown >= -15
            and win_rate >= 0.33
        )

    if family == "VEGAS":
        return (
            trades >= 20
            and avg_return >= 1.0
            and profit_factor >= 1.5
            and max_drawdown >= -35
            and win_rate >= 0.35
        )

    if family == "TRIPLE_VOLUME_BREAKOUT":
        return (
            trades >= 15
            and avg_return >= 1.2
            and profit_factor >= 2.5
            and max_drawdown >= -18
            and win_rate >= 0.55
        )

    if group in DEFAULT_FAMILIES and action_level in DEFAULT_ACTION_LEVELS:
        return (
            trades >= 80
            and avg_return >= 0.8
            and profit_factor >= 1.5
            and max_drawdown >= -25
        )

    return False


def _signal_model_edge_pct(signal: dict[str, Any]) -> float | None:
    model_score = signal.get("model_score") or {}
    try:
        pred_up5 = float(model_score.get("pred_up5") or 0)
        pred_up8 = float(model_score.get("pred_up8") or 0)
        pred_up10 = float(model_score.get("pred_up10") or 0)
        pred_down3 = float(model_score.get("pred_down3") or 0)
        reason = str(signal.get("reason") or "")
        if pred_up10 <= 0:
            match = re.search(r"up10=([-+]?\d+(?:\.\d+)?)", reason)
            pred_up10 = float(match.group(1)) if match else 0
        if pred_down3 <= 0:
            match = re.search(r"down3=([-+]?\d+(?:\.\d+)?)", reason)
            pred_down3 = float(match.group(1)) if match else 0
    except (TypeError, ValueError):
        return None
    if pred_up5 <= 0 and pred_up8 <= 0 and pred_up10 <= 0 and pred_down3 <= 0:
        return None
    return pred_up5 * 5.0 + pred_up8 * 8.0 + pred_up10 * 10.0 - pred_down3 * 3.0


def _signal_dynamic_strength(signal: dict[str, Any]) -> float:
    """Per-candidate shape/model strength used to break ties inside a strategy."""
    strength = _safe_float(signal.get("strength_score"), 0.0) or 0.0
    model_edge = _signal_model_edge_pct(signal)
    if model_edge is not None:
        strength += max(min(model_edge, 10.0), -10.0) / 10.0

    reason = str(signal.get("reason") or "")
    j_match = re.search(r"J=([-+]?\d+(?:\.\d+)?)", reason)
    if j_match:
        try:
            j_value = float(j_match.group(1))
            if j_value < 0:
                strength += min(abs(j_value), 40.0) / 40.0
            elif j_value < 35:
                strength += (35.0 - j_value) / 70.0
        except ValueError:
            pass
    return float(np.clip(strength, -2.0, 4.0))


@lru_cache(maxsize=1)
def _selector_buy_hold_score_artifact() -> dict[str, Any]:
    if not SELECTOR_BUY_HOLD_SCORE_CALIBRATION.exists():
        return {}
    try:
        payload = json.loads(SELECTOR_BUY_HOLD_SCORE_CALIBRATION.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if payload.get("schema_version") != "selector_buy_hold_score_calibration_v1":
        return {}
    return payload


def _score_mode_config(mode: str) -> dict[str, Any]:
    payload = _selector_buy_hold_score_artifact()
    modes = payload.get("modes") if isinstance(payload, dict) else {}
    config = modes.get(mode) if isinstance(modes, dict) else None
    return config if isinstance(config, dict) else {}


def _fallback_score_weights(mode: str) -> dict[str, float]:
    if mode == "hold":
        return {
            "avg_weight": 1.05,
            "model_weight": 0.18,
            "drawdown_penalty": 0.055,
            "pf_weight": 0.90,
            "win_weight": 2.20,
            "group_weight": 0.0,
            "sample_scale": 240.0,
        }
    return {
        "avg_weight": 0.95,
        "model_weight": 0.45,
        "drawdown_penalty": 0.035,
        "pf_weight": 0.80,
        "win_weight": 2.00,
        "group_weight": 0.0,
        "sample_scale": 240.0,
    }


def _score_mode_weights(mode: str) -> dict[str, float]:
    config = _score_mode_config(mode)
    weights = config.get("weights") if isinstance(config, dict) else None
    defaults = _fallback_score_weights(mode)
    if not isinstance(weights, dict):
        return defaults
    return {
        "avg_weight": float(weights.get("avg_weight", defaults["avg_weight"])),
        "model_weight": float(weights.get("model_weight", defaults["model_weight"])),
        "drawdown_penalty": float(weights.get("drawdown_penalty", defaults["drawdown_penalty"])),
        "pf_weight": float(weights.get("pf_weight", defaults["pf_weight"])),
        "win_weight": float(weights.get("win_weight", defaults["win_weight"])),
        "group_weight": float(weights.get("group_weight", defaults["group_weight"])),
        "sample_scale": float(weights.get("sample_scale", defaults["sample_scale"]) or defaults["sample_scale"]),
    }


def _score_group_edge(mode: str, group: str) -> float:
    config = _score_mode_config(mode)
    edges = config.get("group_edges") if isinstance(config, dict) else None
    if not isinstance(edges, dict):
        return 0.0
    return _safe_float(edges.get(group), 0.0) or 0.0


def _score_mode_resonance_weight(mode: str) -> float:
    config = _score_mode_config(mode)
    weights = config.get("weights") if isinstance(config, dict) else None
    if isinstance(weights, dict):
        value = _safe_float(weights.get("resonance_weight"), None)
        if value is not None:
            return value
    return 0.1 if mode == "hold" else 0.2


def _signal_raw_score(signal: dict[str, Any], *, mode: str) -> float:
    metrics = signal.get("metrics") or {}
    weights = _score_mode_weights(mode)
    trades = _safe_float(metrics.get("trades"), 0.0) or 0.0
    avg_return = float(np.clip(_safe_float(metrics.get("avg_return_pct"), 0.0) or 0.0, -10.0, 10.0))
    win_rate = _safe_float(metrics.get("win_rate"), 0.0) or 0.0
    max_drawdown = abs(_safe_float(metrics.get("max_drawdown_pct"), 0.0) or 0.0)
    profit_factor = _safe_float(metrics.get("profit_factor"), 0.0) or 0.0
    reliability = min(1.0, np.sqrt(max(trades, 0.0) / max(weights["sample_scale"], 1.0)))
    model_edge = float(np.clip(_signal_model_edge_pct(signal) or 0.0, -10.0, 10.0))
    pf_bonus = min(max(profit_factor - 1.0, 0.0), 4.0)
    win_bonus = max(win_rate - 0.35, 0.0)
    group_edge = _score_group_edge(mode, _signal_group_key(signal))
    return (
        avg_return * reliability * weights["avg_weight"]
        + model_edge * weights["model_weight"]
        + pf_bonus * weights["pf_weight"]
        + win_bonus * weights["win_weight"]
        + group_edge * weights["group_weight"]
        - min(max_drawdown, 50.0) * weights["drawdown_penalty"]
    )


def _sample_raw_scores(frame: pd.DataFrame, *, mode: str) -> pd.Series:
    weights = _score_mode_weights(mode)
    trades = pd.to_numeric(frame.get("metrics_trades"), errors="coerce").fillna(0)
    avg_return = pd.to_numeric(frame.get("metrics_avg_return_pct"), errors="coerce").fillna(0).clip(-10, 10)
    win_rate = pd.to_numeric(frame.get("metrics_win_rate"), errors="coerce").fillna(0)
    drawdown = pd.to_numeric(frame.get("metrics_max_drawdown_pct"), errors="coerce").fillna(0).abs().clip(upper=50)
    pf = pd.to_numeric(frame.get("metrics_profit_factor"), errors="coerce").fillna(0).clip(upper=5)
    pred_up5 = pd.to_numeric(frame.get("pred_up5"), errors="coerce").fillna(0)
    pred_up8 = pd.to_numeric(frame.get("pred_up8"), errors="coerce").fillna(0)
    pred_up10 = pd.to_numeric(frame.get("pred_up10"), errors="coerce").fillna(0)
    pred_down3 = pd.to_numeric(frame.get("pred_down3"), errors="coerce").fillna(0)
    groups = frame.get("strategy_group")
    if groups is None:
        group_edge = pd.Series(0.0, index=frame.index)
    else:
        group_edge = groups.astype(str).map(lambda group: _score_group_edge(mode, group)).fillna(0.0)
    reliability = np.minimum(1.0, np.sqrt(np.maximum(trades, 0) / max(weights["sample_scale"], 1.0)))
    model_edge = (pred_up5 * 5.0 + pred_up8 * 8.0 + pred_up10 * 10.0 - pred_down3 * 3.0).clip(-10, 10)
    pf_bonus = np.maximum(pf - 1.0, 0).clip(upper=4)
    win_bonus = np.maximum(win_rate - 0.35, 0)
    return (
        avg_return * reliability * weights["avg_weight"]
        + model_edge * weights["model_weight"]
        + pf_bonus * weights["pf_weight"]
        + win_bonus * weights["win_weight"]
        + group_edge * weights["group_weight"]
        - drawdown * weights["drawdown_penalty"]
    )


@lru_cache(maxsize=1)
def _selector_score_calibration() -> dict[str, Any]:
    if not SELECTOR_HISTORY_SIGNAL_SAMPLES.exists():
        return {}
    try:
        frame = pd.read_parquet(SELECTOR_HISTORY_SIGNAL_SAMPLES)
    except Exception:
        return {}
    if frame.empty or "strategy_group" not in frame.columns:
        return {}
    calibration: dict[str, Any] = {"global": {}, "group": {}}
    for mode in ["buy", "hold"]:
        scored = frame[["strategy_group"]].copy()
        scored["raw"] = _sample_raw_scores(frame, mode=mode)
        raw = pd.to_numeric(scored["raw"], errors="coerce").dropna().sort_values().to_numpy()
        if len(raw):
            calibration["global"][mode] = raw
        groups: dict[str, np.ndarray] = {}
        for group, part in scored.groupby("strategy_group"):
            values = pd.to_numeric(part["raw"], errors="coerce").dropna().sort_values().to_numpy()
            if len(values) >= 20:
                groups[str(group)] = values
        calibration["group"][mode] = groups
    return calibration


def _percentile_rank(value: float, distribution: np.ndarray | None) -> float | None:
    if distribution is None or len(distribution) == 0:
        return None
    return float(np.searchsorted(distribution, value, side="right") / len(distribution))


def _normalized_signal_score(signal: dict[str, Any], *, mode: str) -> float:
    raw = _signal_raw_score(signal, mode=mode)
    calibration = _selector_score_calibration()
    group = _signal_group_key(signal)
    global_pct = _percentile_rank(raw, (calibration.get("global") or {}).get(mode))
    group_pct = _percentile_rank(raw, ((calibration.get("group") or {}).get(mode) or {}).get(group))
    if global_pct is None and group_pct is None:
        return float(np.clip(50.0 + raw * 8.0, 0.0, 100.0))
    if global_pct is None:
        blended = group_pct
    elif group_pct is None:
        blended = global_pct
    else:
        blended = 0.55 * group_pct + 0.45 * global_pct
    return float(np.clip((blended or 0.0) * 100.0, 0.0, 100.0))


def _signal_selector_score(signal: dict[str, Any]) -> float:
    """Rank by conservative expected return, then lightly reward risk quality."""
    metrics = signal.get("metrics") or {}
    trades = float(metrics.get("trades") or 0)
    avg_return = float(metrics.get("avg_return_pct") or 0)
    max_drawdown = abs(float(metrics.get("max_drawdown_pct") or 0))
    profit_factor = float(metrics.get("profit_factor") or 0)
    reliability = min(1.0, np.sqrt(max(trades, 0) / 240))
    conservative_avg = max(min(avg_return, 10.0), -10.0) * reliability
    model_edge = _signal_model_edge_pct(signal)
    expected_return = conservative_avg * 0.85
    if model_edge is not None:
        expected_return += max(min(model_edge, 10.0), -10.0) * 0.20
    elif str(signal.get("playbook_source") or "") != "模型版" and trades < 80:
        expected_return *= 0.65
    risk_penalty = min(max_drawdown, 40.0) * 0.05
    pf_bonus = min(max(profit_factor - 1.0, 0.0), 3.0) * 0.15
    return expected_return - risk_penalty + pf_bonus


def _calibrated_signal_score(signal: dict[str, Any], *, model_weight: float) -> float:
    """Historical-sample calibrated score used for stock-level display fields."""
    mode = "buy" if model_weight >= 0.3 else "hold"
    return _normalized_signal_score(signal, mode=mode)


def _aggregate_signal_scores(signal_scores: list[float], group_count: int, *, resonance_weight: float = 0.2) -> float:
    if not signal_scores:
        return 0.0
    ordered = sorted(signal_scores, reverse=True)
    best = ordered[0]
    resonance = 0.08 * sum(max(score - 50.0, 0.0) for score in ordered[1:3])
    group_bonus = 2.5 * resonance_weight * np.log1p(group_count)
    return float(np.clip(best + resonance + group_bonus, 0.0, 100.0))


def _aggregate_raw_scores(signal_scores: list[float], group_count: int, *, resonance_weight: float = 0.2) -> float:
    if not signal_scores:
        return 0.0
    ordered = sorted(signal_scores, reverse=True)
    best = ordered[0]
    resonance = 0.08 * sum(max(score, 0.0) for score in ordered[1:3])
    group_bonus = resonance_weight * np.log1p(group_count)
    return float(best + resonance + group_bonus)


@lru_cache(maxsize=1)
def _selector_buy_hold_models() -> dict[str, dict[str, Any]]:
    try:
        import joblib
    except ImportError:
        return {}
    manifest_path = SELECTOR_BUY_HOLD_MODEL_DIR / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        manifest.get("status") != "success"
        or manifest.get("schema_version")
        != SELECTOR_BUY_HOLD_MANIFEST_SCHEMA_VERSION
        or manifest.get("release_id") != SELECTOR_BUY_HOLD_RELEASE_ID
    ):
        return {}
    artifacts: dict[str, dict[str, Any]] = {}
    for mode in ("buy", "hold"):
        path = SELECTOR_BUY_HOLD_MODEL_DIR / f"{mode}.joblib"
        if not path.exists():
            return {}
        try:
            artifact = joblib.load(path)
            features = validate_selector_buy_hold_artifact(artifact)
        except Exception:
            return {}
        model_manifest = (manifest.get("models") or {}).get(mode) or {}
        if (
            model_manifest.get("sha256") != _file_sha256(path)
            or artifact.get("release_id") != SELECTOR_BUY_HOLD_RELEASE_ID
            or tuple(manifest.get("model_input_columns") or ()) != features
        ):
            return {}
        artifacts[mode] = artifact
    if tuple(artifacts["buy"]["features"]) != tuple(artifacts["hold"]["features"]):
        return {}
    return artifacts


@lru_cache(maxsize=32)
def _selector_model_feature_rows(signal_date: str) -> dict[str, dict[str, Any]]:
    if not SELECTOR_MODEL_HISTORY.exists():
        return {}
    try:
        frame = pd.read_parquet(
            SELECTOR_MODEL_HISTORY,
            filters=[("date", "==", pd.Timestamp(signal_date))],
        )
    except Exception:
        return {}
    if frame.empty or "symbol" not in frame.columns:
        return {}
    return {
        str(row["symbol"]): row.to_dict()
        for _, row in frame.sort_values("date").drop_duplicates("symbol", keep="last").iterrows()
    }


@lru_cache(maxsize=32)
def _selector_model_score_date(signal_date: str | None) -> str | None:
    """Use the requested market date; never substitute an older feature date."""
    if not signal_date:
        return None
    try:
        value = pd.Timestamp(signal_date)
    except (TypeError, ValueError):
        return None
    if pd.isna(value):
        return None
    return value.strftime("%Y-%m-%d")


def _selector_market_feature_values_from_daily(
    daily: pd.DataFrame,
    signal_date: str,
) -> dict[str, Any]:
    """Calculate the model's market cross-section on the exact signal date."""
    if daily.empty or "pct_chg" not in daily.columns:
        return {}
    date_values = (
        daily["date"]
        if "date" in daily.columns
        else daily.get("trade_date")
    )
    if date_values is None:
        return {}
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(date_values, errors="coerce"),
            "pct_chg": pd.to_numeric(daily["pct_chg"], errors="coerce"),
        }
    ).dropna(subset=["date", "pct_chg"])
    if frame.empty:
        return {}
    frame["date"] = frame["date"].dt.normalize()
    frame["up"] = (frame["pct_chg"] > 0).astype(float)
    frame["up5"] = (frame["pct_chg"] >= 5).astype(float)
    frame["down5"] = (frame["pct_chg"] <= -5).astype(float)
    market = (
        frame.groupby("date", as_index=True)
        .agg(
            selector_market_mean_1d=("pct_chg", "mean"),
            selector_market_median_1d=("pct_chg", "median"),
            selector_market_dispersion_1d=("pct_chg", "std"),
            selector_market_up_ratio_1d=("up", "mean"),
            selector_market_up5_ratio_1d=("up5", "mean"),
            selector_market_down5_ratio_1d=("down5", "mean"),
        )
        .sort_index()
    )
    for source, window in (
        ("selector_market_mean_1d", 5),
        ("selector_market_mean_1d", 20),
        ("selector_market_up_ratio_1d", 5),
        ("selector_market_up_ratio_1d", 20),
    ):
        target = source.removesuffix("_1d") + f"_{window}d"
        market[target] = market[source].rolling(window, min_periods=window).mean()
    target_date = pd.Timestamp(signal_date).normalize()
    if target_date not in market.index:
        return {}
    row = market.loc[target_date]
    return {
        column: row.get(column, np.nan)
        for column in SELECTOR_MARKET_FEATURE_COLUMNS
    }


def _selector_turnover_feature_rows(
    symbols: list[str],
    signal_date: str,
    daily_basic_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Build exact-date turnover ratios from current and trailing daily_basic."""
    if not symbols:
        return {}
    root = daily_basic_dir or SELECTOR_DAILY_BASIC_DIR
    target = pd.Timestamp(signal_date).normalize()
    dated_paths: list[tuple[pd.Timestamp, Path]] = []
    for path in root.glob("*.parquet"):
        if not (path.stem.isdigit() and len(path.stem) == 8):
            continue
        path_date = pd.to_datetime(path.stem, format="%Y%m%d", errors="coerce")
        if pd.notna(path_date) and path_date <= target:
            dated_paths.append((path_date, path))
    frames: list[pd.DataFrame] = []
    for _, path in sorted(dated_paths)[-30:]:
        try:
            frame = pd.read_parquet(
                path,
                columns=["ts_code", "trade_date", "turnover_rate"],
            )
        except Exception:
            continue
        frames.append(frame)
    if not frames:
        return {}
    turnover = pd.concat(frames, ignore_index=True)
    turnover["ts_code"] = turnover["ts_code"].astype(str)
    turnover = turnover[turnover["ts_code"].isin(symbols)].copy()
    turnover["date"] = pd.to_datetime(
        turnover["trade_date"].astype(str).str.replace("-", "", regex=False),
        format="%Y%m%d",
        errors="coerce",
    )
    turnover["turnover_rate"] = pd.to_numeric(
        turnover["turnover_rate"],
        errors="coerce",
    )
    result: dict[str, dict[str, float]] = {}
    for symbol, part in turnover.groupby("ts_code", sort=False):
        part = part.sort_values("date").drop_duplicates("date", keep="last")
        current = part[part["date"].eq(target)]
        if current.empty or pd.isna(current.iloc[-1]["turnover_rate"]):
            continue
        current_value = float(current.iloc[-1]["turnover_rate"])
        history = part[part["date"] <= target]["turnover_rate"]
        values: dict[str, float] = {}
        for window in (5, 20):
            mean = history.tail(window).mean()
            values[f"selector_turnover_relative_{window}d"] = (
                current_value / float(mean)
                if pd.notna(mean) and float(mean) != 0
                else np.nan
            )
        result[str(symbol)] = values
    return result


def _selector_watchlist_feature_row_from_daily(
    symbol: str,
    signal_date: str,
    daily: pd.DataFrame,
) -> dict[str, Any]:
    required = {"date", "open", "high", "low", "close", "pre_close", "pct_chg", "volume"}
    if daily.empty or not required.issubset(daily.columns):
        return {}
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily[daily["date"] <= pd.Timestamp(signal_date)].sort_values("date").tail(80).copy()
    if daily.empty:
        return {}
    numeric_columns = required - {"date"}
    for column in numeric_columns:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    returns = daily["pct_chg"] / 100.0
    log_returns = np.log1p(returns.clip(lower=-0.999999))
    latest = daily.iloc[-1]
    if pd.Timestamp(latest["date"]).normalize() != pd.Timestamp(signal_date).normalize():
        return {}
    values: dict[str, Any] = {
        "symbol": symbol,
        "date": latest["date"],
        "matched_groups": [],
        "matched_count": 0,
        "best_profit_factor": np.nan,
        "best_avg_return_pct": np.nan,
        "selector_return_1d": latest["pct_chg"],
        "selector_amplitude_1d": (latest["high"] - latest["low"]) / latest["pre_close"] * 100.0,
        "selector_close_pos": (latest["close"] - latest["low"]) / (latest["high"] - latest["low"]),
        "selector_gap_1d": (latest["open"] / latest["pre_close"] - 1.0) * 100.0,
        "selector_high_1d": (latest["high"] / latest["pre_close"] - 1.0) * 100.0,
        "selector_low_1d": (latest["low"] / latest["pre_close"] - 1.0) * 100.0,
    }
    for window in (3, 5, 10, 20, 60):
        values[f"selector_return_{window}d"] = (np.exp(log_returns.tail(window).sum()) - 1.0) * 100.0
    for window in (5, 10, 20, 60):
        values[f"selector_volatility_{window}d"] = daily["pct_chg"].tail(window).std()
        values[f"selector_volume_relative_{window}d"] = latest["volume"] / daily["volume"].tail(window).mean()
    for window in (5, 10, 20):
        values[f"selector_positive_ratio_{window}d"] = (daily["pct_chg"].tail(window) > 0).mean()
    return values


@lru_cache(maxsize=1)
def _selector_active_model_features() -> tuple[str, ...]:
    artifacts = _selector_buy_hold_models()
    if not artifacts:
        return ()
    return tuple(str(value) for value in artifacts["buy"]["features"])


@lru_cache(maxsize=32)
def _selector_production_snapshot_rows(
    signal_date: str,
) -> dict[str, dict[str, Any]]:
    """Load exact-date upstream factors owned by released daily calculators."""

    features = _selector_active_model_features()
    if not features:
        return {}
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}
    source_specs = (
        (
            PROJECT_ROOT / "data/features/right_side_unified/latest_features.parquet",
            frozenset({"project_daily", "right_side_rule"}),
        ),
        (
            PROJECT_ROOT / "data/features/left_side_unified/latest_features.parquet",
            frozenset({"project_daily", "left_side_rule"}),
        ),
        (
            PROJECT_ROOT / "data/features/long/latest.parquet",
            frozenset({"long_snapshot"}),
        ),
    )
    target = pd.Timestamp(signal_date).normalize()
    rows: dict[str, dict[str, Any]] = {}
    source_names: dict[str, set[str]] = {}
    for path, calculators in source_specs:
        if not path.is_file():
            continue
        try:
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(path).schema.names)
        except Exception:
            continue
        symbol_column = "symbol" if "symbol" in available else "ts_code"
        owned = [
            feature
            for feature in features
            if feature in available
            and definitions[feature].calculator_id in calculators
        ]
        if not owned or "date" not in available or symbol_column not in available:
            continue
        try:
            frame = pd.read_parquet(
                path,
                columns=[symbol_column, "date", *owned],
                filters=[("date", "==", target)],
            )
        except Exception:
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame = frame.loc[frame["date"].eq(target)].drop_duplicates(
            symbol_column, keep="last"
        )
        for record in frame.to_dict("records"):
            symbol = normalize_ts_code(str(record[symbol_column]))
            destination = rows.setdefault(symbol, {})
            for feature in owned:
                incoming = record.get(feature)
                existing = destination.get(feature)
                if pd.notna(existing) and pd.notna(incoming):
                    left = np.float32(existing)
                    right = np.float32(incoming)
                    if not bool(left == right):
                        raise RuntimeError(
                            "selector exact-date canonical factor mismatch: "
                            f"symbol={symbol} date={signal_date} factor={feature} "
                            f"source={path}"
                        )
                if feature not in destination or pd.isna(existing):
                    destination[feature] = incoming
            source_names.setdefault(symbol, set()).add(
                str(path.relative_to(PROJECT_ROOT))
            )
    for symbol, values in rows.items():
        values["_selector_snapshot_sources"] = sorted(source_names[symbol])
    return rows


def _selector_project_feature_rows(
    symbols: list[str],
    signal_date: str,
    store: MarketDataStore,
) -> dict[str, dict[str, Any]]:
    """Materialize canonical project factors for watchlist-only symbols."""

    if not symbols:
        return {}
    target = pd.Timestamp(signal_date).normalize()
    daily = store.read_market_range(
        DAILY_DIR.name,
        start_date=(target - pd.Timedelta(days=2600)).strftime("%Y-%m-%d"),
        end_date=target.strftime("%Y-%m-%d"),
        symbols=symbols,
        columns=[
            "ts_code",
            "symbol",
            "trade_date",
            "date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "volume",
            "amount",
        ],
    )
    if daily.empty:
        return {}
    target_keys = pd.DataFrame(
        {
            "ts_code": symbols,
            "trade_date": target.strftime("%Y%m%d"),
        }
    )
    daily_basic = load_daily_basic_features(
        SELECTOR_DAILY_BASIC_DIR,
        target_keys=target_keys,
    )
    result: dict[str, dict[str, Any]] = {}
    symbol_column = "symbol" if "symbol" in daily.columns else "ts_code"
    for raw_symbol, frame in daily.groupby(symbol_column, sort=False):
        symbol = normalize_ts_code(str(raw_symbol))
        symbol_basic = (
            daily_basic[daily_basic["ts_code"].astype(str).eq(symbol)]
            if not daily_basic.empty and "ts_code" in daily_basic
            else pd.DataFrame()
        )
        factors = calculate_project_factor_frame(
            frame,
            symbol,
            daily_basic_features=symbol_basic,
            factor_schema_version=PROJECT_FACTOR_SCHEMA_VERSION,
        )
        if factors.empty:
            continue
        exact = factors.loc[
            pd.to_datetime(factors["date"], errors="coerce").dt.normalize().eq(target)
        ]
        if not exact.empty:
            result[symbol] = exact.iloc[-1].to_dict()
    return result


def _selector_live_feature_rows(
    symbols: list[str],
    signal_date: str,
) -> dict[str, dict[str, Any]]:
    """Build complete live selector factors from exact-date local inputs."""
    if not symbols:
        return {}
    canonical_symbols = list(
        dict.fromkeys(normalize_ts_code(str(symbol)) for symbol in symbols)
    )
    features = _selector_active_model_features()
    if not features:
        return {}
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=DAILY_DIR.parent))
    target = pd.Timestamp(signal_date).normalize()
    snapshot_rows = _selector_production_snapshot_rows(signal_date)
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}
    project_features = tuple(
        feature
        for feature in features
        if definitions[feature].calculator_id == "project_daily"
    )
    project_fallback_symbols = [
        symbol
        for symbol in canonical_symbols
        if not set(project_features).issubset(snapshot_rows.get(symbol, {}))
    ]
    project_rows = _selector_project_feature_rows(
        project_fallback_symbols,
        signal_date,
        store,
    )
    market = store.read_market_range(
        DAILY_DIR.name,
        start_date=(target - pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
        end_date=target.strftime("%Y-%m-%d"),
        columns=["ts_code", "trade_date", "date", "pct_chg"],
    )
    market_values = _selector_market_feature_values_from_daily(market, signal_date)
    daily = store.read_market_range(
        DAILY_DIR.name,
        start_date=(target - pd.Timedelta(days=130)).strftime("%Y-%m-%d"),
        end_date=target.strftime("%Y-%m-%d"),
        symbols=canonical_symbols,
        columns=[
            "ts_code",
            "symbol",
            "trade_date",
            "date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "pct_chg",
            "volume",
        ],
    )
    if daily.empty:
        return {}
    turnover_rows = _selector_turnover_feature_rows(
        canonical_symbols,
        signal_date,
    )
    symbol_column = "symbol" if "symbol" in daily.columns else "ts_code"
    result: dict[str, dict[str, Any]] = {}
    for symbol, frame in daily.groupby(symbol_column, sort=False):
        canonical_symbol = str(symbol)
        feature = _selector_watchlist_feature_row_from_daily(
            canonical_symbol,
            signal_date,
            frame.copy(),
        )
        if not feature:
            continue
        feature.update(market_values)
        feature.update(turnover_rows.get(canonical_symbol, {}))
        snapshot = snapshot_rows.get(canonical_symbol, {})
        feature.update(
            {
                key: value
                for key, value in snapshot.items()
                if not key.startswith("_")
            }
        )
        feature.update(project_rows.get(canonical_symbol, {}))
        feature["selector_excess_return_1d"] = (
            feature.get("selector_return_1d", np.nan)
            - feature.get("selector_market_mean_1d", np.nan)
        )
        for model_feature in features:
            feature.setdefault(model_feature, np.nan)
        sources = ["live_daily", "daily_basic", "market_cross_section"]
        sources.extend(snapshot.get("_selector_snapshot_sources", []))
        if canonical_symbol in project_rows:
            sources.append("project_daily_on_demand")
        feature["_score_feature_source"] = "+".join(dict.fromkeys(sources))
        feature["_score_feature_date"] = signal_date
        result[canonical_symbol] = feature
    return result


@lru_cache(maxsize=4096)
def _selector_watchlist_feature_row(symbol: str, signal_date: str) -> dict[str, Any]:
    """Build the return-model factors for a stock outside the daily candidate list."""
    candidate = _selector_model_feature_rows(signal_date).get(symbol)
    if candidate is not None:
        return dict(candidate)
    try:
        return _selector_live_feature_rows([symbol], signal_date).get(symbol, {})
    except Exception:
        return {}


def _historical_percentile_scores(
    predictions: np.ndarray,
    reference: np.ndarray,
    normalization_width: float = 2.0,
) -> np.ndarray:
    values = np.asarray(reference, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.full(len(predictions), 50.0)
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = max(float((q75 - q25) / 1.349), 1e-6)
    z_score = (np.asarray(predictions, dtype=float) - median) / scale
    width = max(float(normalization_width), 0.1)
    return np.clip(50.0 + 100.0 / np.pi * np.arctan(z_score / width), 0.0, 100.0)


def _matched_group_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        try:
            if bool(pd.isna(value)):
                return []
        except (TypeError, ValueError):
            pass
        items = [value]
    return [str(item) for item in items if item is not None and str(item)]


def _selector_feature_rows_for_score_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    symbols_by_date: dict[str, set[str]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "")
        signal_date = str(row.get("date") or "")
        if symbol and signal_date:
            symbols_by_date.setdefault(signal_date, set()).add(symbol)
    feature_rows: dict[str, dict[str, Any]] = {}
    for signal_date, symbols in symbols_by_date.items():
        historical = _selector_model_feature_rows(signal_date)
        for symbol in symbols:
            if symbol in historical:
                feature_rows[symbol] = {
                    **historical[symbol],
                    "_score_feature_source": "model_history",
                    "_score_feature_date": signal_date,
                }
        missing_symbols = sorted(symbol for symbol in symbols if symbol not in feature_rows)
        if not missing_symbols:
            continue
        try:
            live_rows = _selector_live_feature_rows(missing_symbols, signal_date)
        except Exception:
            live_rows = {}
        feature_rows.update(live_rows)
    return feature_rows


def _apply_return_model_scores(
    rows: list[dict[str, Any]],
    feature_rows_by_symbol: dict[str, dict[str, Any]] | None = None,
) -> None:
    artifacts = _selector_buy_hold_models()
    if not artifacts:
        for row in rows:
            row["feature_quality"] = {
                "status": "failed",
                "error": "selector_return_models_unavailable",
                "source": "unavailable",
                "date": str(row.get("date") or "") or None,
            }
        return
    if feature_rows_by_symbol is None:
        feature_rows_by_symbol = _selector_feature_rows_for_score_rows(rows)
    for mode, artifact in artifacts.items():
        features = [str(column) for column in artifact.get("features") or []]
        if not features:
            continue
        indexes: list[int] = []
        feature_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            signal_date = str(row.get("date") or "")
            symbol = str(row.get("symbol") or "")
            source = feature_rows_by_symbol.get(symbol)
            if source is None:
                row["feature_quality"] = {
                    "status": "failed",
                    "error": "missing_exact_date_feature_row",
                    "source": "unavailable",
                    "date": signal_date or None,
                }
                continue
            source_name = str(source.get("_score_feature_source") or "model_history")
            source_date = pd.to_datetime(
                source.get("_score_feature_date", source.get("date")),
                errors="coerce",
            )
            expected_date = pd.to_datetime(signal_date, errors="coerce")
            if (
                pd.isna(source_date)
                or pd.isna(expected_date)
                or source_date.normalize() != expected_date.normalize()
            ):
                row["feature_quality"] = {
                    "status": "failed",
                    "error": "feature_date_mismatch",
                    "source": source_name,
                    "date": (
                        source_date.strftime("%Y-%m-%d")
                        if pd.notna(source_date)
                        else None
                    ),
                    "expected_date": signal_date or None,
                }
                continue
            groups = set(_matched_group_values(row.get("matched_groups")))
            values = dict(source)
            values["matched_count"] = len(groups)
            values["best_profit_factor"] = row.get("best_profit_factor")
            values["best_avg_return_pct"] = row.get("best_avg_return_pct")
            for feature in features:
                if feature.startswith("group__"):
                    values[feature] = float(feature.removeprefix("group__") in groups)
            indexes.append(index)
            feature_rows.append(values)
        if not feature_rows:
            continue
        raw_frame = pd.DataFrame(feature_rows)
        missing_columns = [feature for feature in features if feature not in raw_frame.columns]
        frame = (
            raw_frame.reindex(columns=features)
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
        )
        native_missing = (
            artifact.get("preprocessing") == "xgboost_native_nan_float32_v1"
        )
        all_nan_features = (
            []
            if native_missing
            else [feature for feature in features if frame[feature].isna().all()]
        )
        if missing_columns or all_nan_features:
            error = (
                "incomplete_model_features: "
                f"missing_columns={missing_columns}; "
                f"all_nan_features={all_nan_features}"
            )
            for index in indexes:
                source = feature_rows_by_symbol.get(str(rows[index].get("symbol") or ""), {})
                source_name = str(source.get("_score_feature_source") or "model_history")
                rows[index]["feature_quality"] = {
                    "status": "failed",
                    "error": error,
                    "source": source_name,
                    "date": str(rows[index].get("date") or "") or None,
                    "required_feature_count": len(features),
                    "missing_columns": missing_columns,
                    "all_nan_features": all_nan_features,
                }
                rows[index]["score_feature_source"] = source_name
                rows[index]["score_date"] = str(rows[index].get("date") or "") or None
            continue
        # The released XGBoost contract intentionally uses native missing-value
        # routing: rule factors from the other strategy side are structurally
        # absent, not silently imputed.  Still require the shared project,
        # selector and long layers to provide a meaningful minimum footprint.
        minimum_non_null = (
            max(1, int(np.ceil(len(features) * 0.45)))
            if native_missing
            else len(features)
        )
        non_null_counts = frame.notna().sum(axis=1)
        invalid_positions = (
            non_null_counts.lt(minimum_non_null)
            if native_missing
            else frame.isna().any(axis=1)
        )
        if invalid_positions.any():
            valid_positions: list[int] = []
            for position, invalid in enumerate(invalid_positions.tolist()):
                if not invalid:
                    valid_positions.append(position)
                    continue
                index = indexes[position]
                source = feature_rows_by_symbol.get(
                    str(rows[index].get("symbol") or ""),
                    {},
                )
                source_name = str(
                    source.get("_score_feature_source") or "model_history"
                )
                rows[index]["feature_quality"] = {
                    "status": "failed",
                    "error": (
                        "insufficient_non_null_model_features"
                        if native_missing
                        else "incomplete_row_features"
                    ),
                    "source": source_name,
                    "date": str(rows[index].get("date") or "") or None,
                    "required_feature_count": len(features),
                    "available_feature_count": int(non_null_counts.iloc[position]),
                    "minimum_available_feature_count": minimum_non_null,
                    "missing_features": frame.columns[
                        frame.iloc[position].isna()
                    ].tolist(),
                }
                rows[index]["score_feature_source"] = source_name
                rows[index]["score_date"] = str(rows[index].get("date") or "") or None
            if not valid_positions:
                continue
            indexes = [indexes[position] for position in valid_positions]
            frame = frame.iloc[valid_positions].reset_index(drop=True)
        try:
            imputer = artifact.get("imputer")
            transformed = imputer.transform(frame) if imputer is not None else frame
            if "models" in artifact:
                component_scores: dict[str, np.ndarray] = {}
                for component, model in artifact["models"].items():
                    predictions = model.predict(transformed)
                    component_scores[component] = _historical_percentile_scores(
                        predictions,
                        artifact["score_references"][component],
                        (artifact.get("normalization_widths") or {}).get(component, 2.0),
                    )
                buy_weight = float(artifact.get("buy_weight", 0.0))
                scores = buy_weight * component_scores["buy"] + (1.0 - buy_weight) * component_scores["hold"]
            else:
                predictions = artifact["model"].predict(transformed)
                scores = _historical_percentile_scores(
                    predictions,
                    artifact["score_reference"],
                    artifact.get("normalization_width", 2.0),
                )
        except Exception as exc:
            for index in indexes:
                source = feature_rows_by_symbol.get(str(rows[index].get("symbol") or ""), {})
                source_name = str(source.get("_score_feature_source") or "model_history")
                rows[index]["feature_quality"] = {
                    "status": "failed",
                    "error": f"model_scoring_failed: {type(exc).__name__}: {exc}",
                    "source": source_name,
                    "date": str(rows[index].get("date") or "") or None,
                    "required_feature_count": len(features),
                }
            continue
        historical_name = "historical_buy_score" if mode == "buy" else "historical_hold_score"
        for position, (index, score) in enumerate(zip(indexes, scores)):
            rows[index][historical_name] = round(float(np.clip(score, 0.0, 100.0)), 1)
            rows[index][f"{mode}_score_source"] = "historical_return_model"
            source = feature_rows_by_symbol.get(str(rows[index].get("symbol") or ""), {})
            score_date = pd.to_datetime(
                source.get("_score_feature_date", source.get("date")),
                errors="coerce",
            )
            rows[index]["score_date"] = (
                score_date.strftime("%Y-%m-%d")
                if pd.notna(score_date)
                else str(rows[index].get("date") or "") or None
            )
            source_name = str(source.get("_score_feature_source") or "model_history")
            rows[index]["score_feature_source"] = source_name
            rows[index]["feature_quality"] = {
                "status": "complete",
                "error": None,
                "source": source_name,
                "date": rows[index]["score_date"],
                "required_feature_count": len(features),
                "available_feature_count": int(frame.iloc[position].notna().sum()),
                "minimum_available_feature_count": minimum_non_null,
            }


def _apply_historical_score_normalization(
    rows: list[dict[str, Any]],
    feature_rows_by_symbol: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Expose fixed-history scores without a date-local cross-sectional adjustment.

    Historical buy and hold scores are calibrated against their respective
    fixed historical distributions. Keeping the displayed score equal to that
    calibrated value makes the same signal comparable across selection dates
    and independent of the other candidates present on a given day.
    """
    if not rows:
        return rows
    _apply_return_model_scores(rows, feature_rows_by_symbol)
    for row in rows:
        has_model_scores = (
            row.get("buy_score_source") == "historical_return_model"
            and row.get("hold_score_source") == "historical_return_model"
        )
        if not has_model_scores:
            row["model_score_available"] = False
            row["score_target"] = "historical_strategy_quality_fallback"
            continue
        historical_buy_score = _safe_float(row.get("historical_buy_score"), 50.0)
        historical_hold_score = _safe_float(row.get("historical_hold_score"), 50.0)
        buy_score = float(np.clip(50.0 if historical_buy_score is None else historical_buy_score, 0.0, 100.0))
        hold_score = float(np.clip(50.0 if historical_hold_score is None else historical_hold_score, 0.0, 100.0))
        row["opportunity_score"] = round(buy_score, 1)
        row["holding_score"] = round(hold_score, 1)
        row["selector_score"] = row["opportunity_score"]
        row["score_target"] = "historical_return_model_score"
        row["model_score_available"] = True
        row.update(_score_interpretation(row["opportunity_score"], row["holding_score"]))
    return rows


def _watchlist_buy_hold_scores(symbols: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Score every watchlist stock with the selector's fixed-history models."""
    if not symbols:
        return {}
    market_date = _latest_similar_pattern_target_date(list(symbols))
    score_date = _selector_model_score_date(market_date)
    if not score_date:
        return {}
    candidate_rows = _selector_model_feature_rows(score_date)
    feature_rows = {
        symbol: {
            **candidate_rows[symbol],
            "_score_feature_source": "model_history",
            "_score_feature_date": score_date,
        }
        for symbol in symbols
        if symbol in candidate_rows
    }
    missing_symbols = [symbol for symbol in symbols if symbol not in feature_rows]
    if missing_symbols:
        try:
            feature_rows.update(
                _selector_live_feature_rows(missing_symbols, score_date)
            )
        except Exception:
            pass
    rows = [
        {
            "symbol": symbol,
            "date": score_date,
            "matched_groups": _matched_group_values(
                feature_rows.get(symbol, {}).get("matched_groups"),
            ),
            "best_profit_factor": feature_rows.get(symbol, {}).get(
                "best_profit_factor"
            ),
            "best_avg_return_pct": feature_rows.get(symbol, {}).get(
                "best_avg_return_pct"
            ),
        }
        for symbol in symbols
    ]
    _apply_historical_score_normalization(rows, feature_rows)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        score = {
            "score_date": score_date,
            "score_target": row["score_target"],
            "score_feature_source": row.get("score_feature_source"),
            "feature_quality": row.get("feature_quality"),
            "model_score_available": bool(row.get("model_score_available")),
        }
        if row.get("model_score_available"):
            score.update(
                opportunity_score=row["opportunity_score"],
                holding_score=row["holding_score"],
                buy_score=row["opportunity_score"],
                hold_score=row["holding_score"],
            )
        result[row["symbol"]] = score
    return result


def _score_interpretation(opportunity_score: float, holding_score: float) -> dict[str, str]:
    """Coarse score bands from the full-history calibration pass.

    Keep these deliberately coarse: the OOT test supports ranking by broad
    buckets better than treating the raw decimal as a precise probability.
    """
    if opportunity_score >= 85:
        band = "极端尾部"
        percentile = "历史参考约 Top 0.02%"
        usage = "买入条件质量很强，仍需按开盘条件和止损执行"
    elif opportunity_score >= 70:
        band = "极高"
        percentile = "历史参考约 Top 1.4%"
        usage = "优先观察，等待开盘条件确认"
    elif opportunity_score >= 60:
        band = "高"
        percentile = "历史参考约 Top 8.6%"
        usage = "可观察，不适合追高"
    elif opportunity_score >= 50:
        band = "中上"
        percentile = "高于历史参考中位数"
        usage = "只适合结合策略细节筛选"
    elif opportunity_score >= 45:
        band = "中性偏低"
        percentile = "历史参考中段偏下"
        usage = "只适合结合策略细节筛选"
    else:
        band = "低"
        percentile = "低于主要观察区"
        usage = "不建议仅因分数参与"

    if opportunity_score >= 70 and holding_score < 45:
        risk_note = "买入分高但持有分弱，偏短线机会，不宜按 T+5 持有逻辑理解"
    elif holding_score >= 70:
        risk_note = "买入与持有评分相对一致"
    else:
        risk_note = "持有分偏弱，需重视回撤和卖出纪律"

    return {
        "score_band": band,
        "score_percentile_label": percentile,
        "score_usage_hint": usage,
        "score_risk_note": risk_note,
    }


def _row_display_quality_gate(row: dict[str, Any]) -> bool:
    return any(_signal_quality_gate(signal) for signal in row.get("signals") or [])


def _signal_operation_key(signal: dict[str, Any]) -> str:
    family = _signal_group_key(signal)
    operation_key = str(signal.get("operation_key") or "")
    return f"{family}::{operation_key}" if operation_key else family


def _dedupe_signals_by_operation(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the best variant for the same strategy family and buy operation."""
    best_by_operation: dict[str, dict[str, Any]] = {}
    for signal in signals:
        key = _signal_operation_key(signal)
        current = best_by_operation.get(key)
        if current is None:
            best_by_operation[key] = signal
            continue
        current_key = (
            _signal_selector_score(current),
            (current.get("metrics") or {}).get("profit_factor") or 0,
            (current.get("metrics") or {}).get("avg_return_pct") or -999,
        )
        candidate_key = (
            _signal_selector_score(signal),
            (signal.get("metrics") or {}).get("profit_factor") or 0,
            (signal.get("metrics") or {}).get("avg_return_pct") or -999,
        )
        if candidate_key > current_key:
            best_by_operation[key] = signal
    return list(best_by_operation.values())


def build_selector_stock_row(
    stock: dict[str, Any],
    signals: list[dict[str, Any]],
    signal_date: str | None = None,
) -> dict[str, Any] | None:
    """Build the page-equivalent stock-level selector row from signal payloads."""
    signals = _dedupe_signals_by_operation(signals)
    if not signals:
        return None
    _fill_stock_profile(stock, signal_date)
    groups = sorted({_signal_group_key(signal) for signal in signals})
    group_labels = [STRATEGY_GROUP_LABELS.get(group, group) for group in groups]
    best_pf = max((signal.get("metrics") or {}).get("profit_factor") or 0 for signal in signals)
    best_avg = max((signal.get("metrics") or {}).get("avg_return_pct") or -999 for signal in signals)
    ordered_signals = sorted(
        signals,
        key=lambda item: (_signal_selector_score(item), (item.get("metrics") or {}).get("profit_factor") or 0),
        reverse=True,
    )
    signal_scores = [_signal_selector_score(signal) for signal in ordered_signals]
    legacy_score = (signal_scores[0] if signal_scores else 0) + 0.08 * sum(max(score, 0) for score in signal_scores[1:3]) + 0.15 * np.log1p(len(groups))
    historical_buy_score = _aggregate_signal_scores(
        [_calibrated_signal_score(signal, model_weight=0.4) for signal in ordered_signals],
        len(groups),
        resonance_weight=_score_mode_resonance_weight("buy"),
    )
    historical_hold_score = _aggregate_signal_scores(
        [_calibrated_signal_score(signal, model_weight=0.2) for signal in ordered_signals],
        len(groups),
        resonance_weight=_score_mode_resonance_weight("hold"),
    )
    buy_raw_score = _aggregate_raw_scores(
        [_signal_raw_score(signal, mode="buy") for signal in ordered_signals],
        len(groups),
        resonance_weight=_score_mode_resonance_weight("buy"),
    )
    hold_raw_score = _aggregate_raw_scores(
        [_signal_raw_score(signal, mode="hold") for signal in ordered_signals],
        len(groups),
        resonance_weight=_score_mode_resonance_weight("hold"),
    )
    score_info = _score_interpretation(float(historical_buy_score), float(historical_hold_score))
    return {
        **{key: value for key, value in stock.items() if key != "signals"},
        "matched_count": len(groups),
        "matched_families": group_labels,
        "matched_groups": groups,
        "matched_group_sides": {
            group: SHORT_GROUP_SIDE.get(group, "unknown") for group in groups
        },
        "matched_strategy_names": [signal.get("strategy_name") for signal in ordered_signals],
        "best_profit_factor": best_pf,
        "best_avg_return_pct": best_avg if best_avg > -999 else None,
        "selector_score": round(float(historical_buy_score), 1),
        "opportunity_score": round(float(historical_buy_score), 1),
        "holding_score": round(float(historical_hold_score), 1),
        "historical_buy_score": round(float(historical_buy_score), 1),
        "historical_hold_score": round(float(historical_hold_score), 1),
        "buy_raw_score": round(float(buy_raw_score), 4),
        "hold_raw_score": round(float(hold_raw_score), 4),
        "legacy_selector_score": round(float(legacy_score), 2),
        "score_target": "historical_strategy_quality_fallback",
        "model_score_available": False,
        **score_info,
        "rank_reason": f"按 {ordered_signals[0].get('strategy_name')} 领衔，叠加 {len(groups)} 个策略组共振；当前买入分按未来 5 日冲高目标做历史校准",
        "signals": ordered_signals,
    }


def _primary_family(row: dict[str, Any]) -> str:
    signals = row.get("signals") or []
    if signals:
        return _signal_group_key(signals[0])
    families = _matched_group_values(row.get("matched_groups"))
    if not families:
        families = _matched_group_values(row.get("matched_families"))
    return str(families[0]) if families else ""


def _diversify_default_rows(rows: list[dict[str, Any]], limit: int = DEFAULT_SELECTOR_LIMIT) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    seen: set[str] = set()
    for row in rows:
        family = _primary_family(row)
        if family_counts.get(family, 0) >= DEFAULT_FAMILY_CAP:
            continue
        selected.append(row)
        seen.add(str(row.get("symbol") or ""))
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= limit:
            return selected
    for row in rows:
        symbol = str(row.get("symbol") or "")
        if symbol in seen:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


@lru_cache(maxsize=1)
def _family_best_metrics() -> dict[str, pd.Series]:
    path = latest_report_file("latest_b1_family_rule_backtest.csv") or latest_report_file(FAMILY_RULE_PATTERN)
    if not path:
        return {}
    df = pd.read_csv(path)
    oot = df[df["split"] == "oot"].copy()
    if oot.empty:
        return {}
    oot["score"] = (
        oot["profit_factor"].fillna(0)
        + 0.2 * oot["avg_return_pct"].fillna(0)
        + 0.02 * oot["max_drawdown_pct"].fillna(0)
        + np.minimum(oot["trades"].fillna(0), 100) / 500
    )
    best = oot.sort_values(["family", "score"], ascending=[True, False]).groupby("family").head(1)
    return {str(row["family"]): row for _, row in best.iterrows()}


@lru_cache(maxsize=1)
def _family_best_metrics_by_signal() -> dict[str, pd.Series]:
    path = latest_report_file("latest_b1_family_rule_backtest.csv") or latest_report_file(FAMILY_RULE_PATTERN)
    if not path:
        return {}
    df = pd.read_csv(path)
    oot = df[df["split"] == "oot"].copy()
    if oot.empty:
        return {}
    oot["score"] = (
        oot["profit_factor"].fillna(0)
        + 0.2 * oot["avg_return_pct"].fillna(0)
        + 0.02 * oot["max_drawdown_pct"].fillna(0)
        + np.minimum(oot["trades"].fillna(0), 100) / 500
    )
    best = oot.sort_values(["signal", "score"], ascending=[True, False]).groupby("signal").head(1)
    return {str(row["signal"]): row for _, row in best.iterrows()}


@lru_cache(maxsize=1)
def _family_risk_managed_metrics_by_signal() -> dict[str, pd.Series]:
    path = latest_report_file("latest_b1_family_rule_backtest.csv") or latest_report_file(FAMILY_RULE_PATTERN)
    if not path:
        return {}
    df = pd.read_csv(path)
    oot = df[df["split"] == "oot"].copy()
    if oot.empty:
        return {}
    oot = oot[~oot["exit_rule"].astype(str).str.startswith("expiry")].copy()
    if oot.empty:
        return {}
    oot = oot[
        (oot["profit_factor"].fillna(0) >= 1.4)
        & (oot["avg_return_pct"].fillna(0) >= 0.5)
        & (oot["max_drawdown_pct"].fillna(-100) >= -13)
    ].copy()
    if oot.empty:
        return {}
    oot["score"] = (
        oot["profit_factor"].fillna(0)
        + 0.25 * oot["avg_return_pct"].fillna(0)
        + 0.06 * oot["max_drawdown_pct"].fillna(0)
        + np.minimum(oot["trades"].fillna(0), 100) / 500
    )
    best = oot.sort_values(["signal", "score"], ascending=[True, False]).groupby("signal").head(1)
    return {str(row["signal"]): row for _, row in best.iterrows()}


def _b1_model_signal(row: dict[str, Any]) -> dict[str, Any]:
    metrics = _metrics_payload(
        {
            "trades": row.get("oot_trades"),
            "avg_return_pct": row.get("oot_avg_return_pct"),
            "win_rate": row.get("oot_win_rate"),
            "max_drawdown_pct": row.get("oot_max_drawdown_pct"),
            "profit_factor": row.get("oot_profit_factor"),
        }
    )
    price_range = (
        f"{float(row['buy_min_price']):.2f} - {float(row['buy_max_price']):.2f}"
        if row.get("buy_min_price") is not None and row.get("buy_max_price") is not None
        else f">= {float(row['buy_min_price']):.2f}" if row.get("buy_min_price") is not None else "按策略开盘条件观察"
    )
    unified_rank = row.get("ranking_score_normalized")
    pred_up10 = float(row.get("pred_up10_es") or 0)
    pred_down3 = float(row.get("pred_down3_es") or 0)
    uses_unified_rank = unified_rank is not None
    reason = (
        f"左侧统一排序百分位={float(unified_rank):.2f}"
        if uses_unified_rank
        else f"up10={pred_up10:.3f}，down3={pred_down3:.3f}，J={float(row.get('kdj_d_j') or 0):.2f}"
    )
    return _enrich_signal_group({
        "strategy_key": "B1_MODEL",
        "strategy_family": "B1",
        "strategy_name": row.get("strategy_name") or "B1 模型 Top20",
        "operation_key": str(row.get("buy_filter") or row.get("open_gap_text") or "B1"),
        "timeframe": "日线级，收盘后生成名单，T+1 开盘观察",
        "logic": f"{row.get('entry_rule')}；{row.get('buy_filter')}",
        "reason": reason,
        "buy_plan": f"{row.get('open_gap_text')}；参考买入价 {price_range}。不满足开盘条件则空仓观察。",
        "sell_plan": row.get("sell_summary") or "按策略卖出规则执行",
        "metrics": metrics,
        "metrics_text": _metrics_text(metrics),
        "strength_score": (
            float(unified_rank) / 25.0
            if uses_unified_rank
            else max(pred_up10 - pred_down3, 0) * 3
        ),
    })


def _family_signal_payload(strategy_key: str, latest_row: pd.Series, model_score: pd.Series | None = None) -> dict[str, Any]:
    signal_column, family, name, logic = FAMILY_SIGNAL_COLUMNS[strategy_key]
    if family == "VEGAS":
        return _vegas_tunnel_signal_payload(latest_row)
    if family == "TRIPLE_VOLUME_BREAKOUT":
        return _triple_volume_breakout_signal_payload(latest_row)
    is_intraday_approx = family in {"SB1", "SUPER_B1"}
    model_playbook = _model_playbook_for(family) if family in MODEL_FILTERED_SIGNALS and model_score is not None else None
    best = model_playbook if model_playbook is not None else _family_best_metrics_by_signal().get(signal_column)
    if best is None:
        best = _family_best_metrics().get(family)
    if family == "B3" and best is not None and str(best.get("exit_rule", "")).startswith("expiry"):
        risk_managed = _family_risk_managed_metrics_by_signal().get(signal_column)
        if risk_managed is not None:
            best = risk_managed
    metrics = _metrics_payload(best) if best is not None else None
    if best is not None:
        buy = str(best.get("open_filter_description", "T+1 开盘观察"))
        sell = _exit_rule_text(str(best.get("exit_rule", "按该策略最优回测退出规则")))
        description = str(best.get("signal_description", logic))
    else:
        buy = "T+1 开盘观察；当前只作为规则信号，不自动交易"
        sell = "暂无正式卖出组合，先按观察信号处理"
        description = logic
    pct_chg = float(latest_row.get("pct_chg") or 0)
    if family == "B2":
        strength_score = min(max(pct_chg, 0), 10) / 10
    elif family == "B3":
        strength_score = max(0, 1 - abs(pct_chg - 1) / 2)
    else:
        strength_score = 0.2 if family in {"SB1", "SUPER_B1"} else 0
    model_reason = _model_score_reason(model_score)
    model_entry = ""
    action_level = str(best.get("action_level") or "") if best is not None and hasattr(best, "get") else ""
    if model_playbook is not None:
        model_entry = f"模型买入条件：{_entry_rule_text(str(model_playbook.get('entry_rule') or ''))}；"
        if model_reason:
            strength_score += max(float(model_score.get("pred_up5") or 0) - float(model_score.get("pred_down3") or 0), 0) * 2
    elif family in MODEL_FILTERED_SIGNALS:
        model_entry = "模型分未覆盖或未通过；当前先展示规则候选；"
    return _enrich_signal_group({
        "strategy_key": strategy_key,
        "strategy_family": family,
        "strategy_name": name,
        "timeframe": "盘中战法的日线近似观察" if is_intraday_approx else "日线级，收盘确认，T+1 开盘观察",
        "logic": description,
        "reason": (
            f"J={float(latest_row.get('kdj_d_j') or 0):.2f}，"
            f"涨跌幅={_fmt_pct(pct_chg)}，"
            f"收盘={_fmt_price(latest_row.get('close'))}"
            + (f"；模型分 {model_reason}" if model_reason else "")
        ),
        "buy_plan": _buy_plan_text(
            buy,
            prefix=model_entry,
            action_level=action_level,
            intraday_approx=is_intraday_approx,
        ),
        "sell_plan": sell,
        "metrics": metrics,
        "metrics_text": _metrics_text(metrics),
        "action_level": action_level,
        "playbook_source": "模型版" if model_playbook is not None else "规则版",
        "strength_score": strength_score,
    })


def _triple_volume_breakout_signal_payload(row: pd.Series) -> dict[str, Any]:
    metrics = _parse_tvb_metrics(row.get("tvb_metrics"))
    tier = str(row.get("tvb_tier") or "expanded")
    variant_name = str(row.get("tvb_variant_name") or "三倍量缩量盘整突破")
    volume_multiple = _safe_float(row.get("tvb_volume_multiple"), 0.0) or 0.0
    score = _safe_float(row.get("tvb_score"), 78.0) or 78.0
    action_level = "可小仓实操" if tier == "conservative" else "谨慎实操"
    position = "小仓到标准短线仓" if tier == "conservative" else "观察到小仓"
    days_since = _safe_float(row.get("days_since_triple_volume"), None)
    consolidation = _safe_float(row.get("consolidation_range"), None)
    breakout_pct = _safe_float(row.get("breakout_pct"), None)
    volume_dryness = _safe_float(row.get("volume_dryness"), None)
    details = []
    if days_since is not None:
        details.append(f"三倍量后第 {days_since:.0f} 天")
    if consolidation is not None:
        details.append(f"盘整振幅 {((consolidation - 1) * 100):.1f}%")
    if breakout_pct is not None:
        details.append(f"突破幅度 {_fmt_pct(breakout_pct * 100)}")
    if volume_dryness is not None:
        details.append(f"缩量强度 {_fmt_pct(volume_dryness * 100)}")
    reason = "，".join(details) if details else "命中三倍量缩量盘整突破合并策略"
    metrics_text = _metrics_text(metrics)
    return _enrich_signal_group({
        "strategy_key": "TRIPLE_VOLUME_BREAKOUT",
        "strategy_family": "TRIPLE_VOLUME_BREAKOUT",
        "strategy_name": variant_name,
        "operation_key": f"TVB_{tier}_{volume_multiple:g}x",
        "timeframe": "日线级，收盘确认，T+1 开盘观察",
        "logic": str(row.get("tvb_description") or "缩量盘整后右侧突破"),
        "reason": reason,
        "buy_plan": f"{row.get('tvb_buy_plan') or 'T+1 开盘观察'}；建议仓位：{position}。",
        "sell_plan": str(row.get("tvb_sell_plan") or "固定止盈5%，硬止损2.5%，最长T+9。"),
        "metrics": metrics,
        "metrics_text": metrics_text,
        "action_level": action_level,
        "playbook_source": "保守主策略" if tier == "conservative" else "扩展候选池",
        "strength_score": (score - 70.0) / 10.0,
    })


def _parse_tvb_metrics(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {
            "trades": 35,
            "avg_return_pct": 1.845164,
            "win_rate": 0.628571,
            "max_drawdown_pct": -14.194666,
            "profit_factor": 3.014726,
        }
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {
        "trades": 35,
        "avg_return_pct": 1.845164,
        "win_rate": 0.628571,
        "max_drawdown_pct": -14.194666,
        "profit_factor": 3.014726,
    }


@lru_cache(maxsize=16)
def _family_signals_for_date(signal_date: str | None = None) -> dict[str, list[dict[str, Any]]]:
    scored = _model_scored_candidates_for_date(signal_date)
    if FAMILY_SIGNAL_CACHE.exists():
        cached = pd.read_parquet(FAMILY_SIGNAL_CACHE)
        cached["date"] = pd.to_datetime(cached["date"])
        if signal_date:
            selected_date = pd.to_datetime(signal_date)
        else:
            selected_date = cached["date"].max()
        latest_rows = cached[cached["date"] == selected_date].copy()
        signal_rows: dict[str, list[dict[str, Any]]] = {}
        for _, row in latest_rows.iterrows():
            symbol = str(row.get("symbol"))
            for key, (column, _, _, _) in FAMILY_SIGNAL_COLUMNS.items():
                if column in row.index and bool(row.get(column, False)):
                    family = FAMILY_SIGNAL_COLUMNS[key][1]
                    model_score = scored.get((symbol, family))
                    signal_rows.setdefault(symbol, []).append(_family_signal_payload(key, row, model_score=model_score))
        return signal_rows

    features = pd.read_parquet(FEATURE_PATH)
    features["date"] = pd.to_datetime(features["date"])
    if signal_date:
        latest_date = pd.to_datetime(signal_date)
    else:
        latest_date = features["date"].max()
    signal_rows: dict[str, list[dict[str, Any]]] = {}
    for symbol, group in features.groupby("symbol", sort=False):
        try:
            enriched = add_b1_family_signals(group)
            triple_volume = add_triple_volume_strategy_pool_signals(group)
            for col in triple_volume.columns:
                if col not in enriched.columns:
                    enriched[col] = triple_volume[col]
        except Exception:
            continue
        latest = enriched[enriched["date"] == latest_date]
        if latest.empty:
            continue
        row = latest.iloc[-1]
        for key, (column, _, _, _) in FAMILY_SIGNAL_COLUMNS.items():
            if int(row.get(column, 0) or 0) == 1:
                family = FAMILY_SIGNAL_COLUMNS[key][1]
                model_score = scored.get((str(symbol), family))
                signal_rows.setdefault(str(symbol), []).append(_family_signal_payload(key, row, model_score=model_score))
    return signal_rows


def _left_side_ranked_signal(strategy_group: str) -> dict[str, Any]:
    label = STRATEGY_GROUP_LABELS.get(strategy_group, strategy_group)
    return _enrich_signal_group(
        {
            "strategy_key": strategy_group,
            "strategy_family": strategy_group,
            "strategy_group": strategy_group,
            "strategy_name": label,
            "operation_key": f"{strategy_group}_UNIFIED_RANK",
            "timeframe": "日线级，收盘后生成候选，T+1 开盘观察",
            "logic": f"命中{label}原始策略候选，由左侧统一模型进行横截面排序。",
            "reason": "已进入当日左侧统一排序候选池。",
            "buy_plan": "T+1 开盘观察，按统一排序与策略既有风控执行。",
            "sell_plan": "按该策略既有退出规则执行。",
            "metrics": None,
            "metrics_text": "",
            "action_level": "观察",
            "playbook_source": "左侧统一排序",
            "strength_score": 0.0,
        }
    )


def _materialize_left_side_ranked_candidates(
    stocks: dict[str, dict[str, Any]],
    candidates: pd.DataFrame,
    latest_profiles: pd.DataFrame,
    signal_date: str | None,
) -> None:
    family_profiles = _family_profiles_for_date(signal_date)
    for _, candidate in candidates.iterrows():
        symbol = str(candidate.get("symbol") or "")
        if not symbol:
            raise RuntimeError("left-side ranked candidate has no symbol")
        if symbol not in stocks:
            family_profile = family_profiles.get(symbol) or {}
            market_profile = (
                latest_profiles.loc[symbol]
                if symbol in latest_profiles.index
                else {}
            )
            stocks[symbol] = {
                "symbol": symbol,
                "name": family_profile.get("name")
                or (
                    market_profile.get("name", "")
                    if hasattr(market_profile, "get")
                    else ""
                ),
                "date": family_profile.get("date")
                or (
                    market_profile.get("date").strftime("%Y-%m-%d")
                    if hasattr(market_profile, "get")
                    and pd.notna(market_profile.get("date"))
                    else signal_date
                ),
                "close": family_profile.get("close")
                if family_profile
                else (
                    float(market_profile.get("close"))
                    if hasattr(market_profile, "get")
                    and pd.notna(market_profile.get("close"))
                    else None
                ),
                "industry": family_profile.get("industry")
                or (
                    market_profile.get("industry", "")
                    if hasattr(market_profile, "get")
                    else ""
                ),
                "signals": [],
            }
            _fill_stock_profile(stocks[symbol], signal_date)

        existing_groups = {
            _signal_group_key(signal)
            for signal in stocks[symbol].get("signals") or []
        }
        for strategy_group in DEFAULT_LEFT_SIDE_RANKING_CONFIG.strategy_keys:
            if bool(candidate.get(strategy_group, False)) and strategy_group not in existing_groups:
                stocks[symbol]["signals"].append(
                    _left_side_ranked_signal(strategy_group)
                )
                existing_groups.add(strategy_group)


def _latest_family_signals() -> dict[str, list[dict[str, Any]]]:
    return _family_signals_for_date(None)


@lru_cache(maxsize=16)
def _family_profiles_for_date(signal_date: str | None = None) -> dict[str, dict[str, Any]]:
    if not FAMILY_SIGNAL_CACHE.exists():
        return {}
    cached = pd.read_parquet(FAMILY_SIGNAL_CACHE)
    cached["date"] = pd.to_datetime(cached["date"])
    if signal_date:
        selected_date = pd.to_datetime(signal_date)
    else:
        selected_date = cached["date"].max()
    latest_rows = cached[cached["date"] == selected_date].copy()
    profiles: dict[str, dict[str, Any]] = {}
    for _, row in latest_rows.iterrows():
        symbol = str(row.get("symbol"))
        profiles[symbol] = {
            "symbol": symbol,
            "name": row.get("name") or "",
            "date": row.get("date").strftime("%Y-%m-%d") if pd.notna(row.get("date")) else None,
            "close": float(row.get("close")) if pd.notna(row.get("close")) else None,
            "industry": row.get("industry") or "",
        }
    return profiles


def _latest_family_profiles() -> dict[str, dict[str, Any]]:
    return _family_profiles_for_date(None)


@lru_cache(maxsize=4)
def _latest_extended_signals(signal_date: str | None) -> dict[str, dict[str, Any]]:
    if EXTENDED_SIGNAL_CACHE.exists():
        try:
            cached = read_json_file(EXTENDED_SIGNAL_CACHE)
            if cached.get("signal_date") == signal_date and isinstance(cached.get("signals"), dict):
                return cached["signals"]
        except Exception:
            pass
    cached_signals = _extended_signals_from_candidate_cache(signal_date)
    if cached_signals:
        return cached_signals
    signals = build_extended_daily_signals(DAILY_DIR, signal_date=signal_date, max_workers=32)
    atomic_write_json(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "signal_date": signal_date,
            "signals": signals,
        },
        EXTENDED_SIGNAL_CACHE,
    )
    return signals


def _extended_signals_from_candidate_cache(signal_date: str | None) -> dict[str, dict[str, Any]]:
    """Build selector-ready extended signals from the daily z-skill parquet cache."""

    if not EXTENDED_CANDIDATE_CACHE.exists():
        return {}
    strategy_meta = {str(item["key"]).upper(): item for item in EXTENDED_STRATEGIES}
    signal_cols = list(strategy_meta)
    try:
        cached = pd.read_parquet(EXTENDED_CANDIDATE_CACHE)
    except Exception:
        return {}
    signal_cols = [col for col in signal_cols if col in cached.columns]
    if not signal_cols:
        return {}
    if cached.empty or not {"date", "symbol"}.issubset(cached.columns):
        return {}
    cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
    cached = cached.dropna(subset=["date", "symbol"])
    if cached.empty:
        return {}
    if signal_date:
        target = pd.to_datetime(signal_date, errors="coerce")
        if pd.isna(target):
            return {}
        selected_date = target
    else:
        selected_date = cached["date"].max()
    latest = cached[cached["date"] == selected_date].copy()
    if latest.empty:
        return {}

    basic = _stock_basic_map()
    signals_by_symbol: dict[str, dict[str, Any]] = {}
    for _, row in latest.iterrows():
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        row_signals: list[dict[str, Any]] = []
        for key in signal_cols:
            value = row.get(key, False)
            if pd.isna(value) or not bool(value):
                continue
            meta = strategy_meta[key]
            label = str(meta.get("label") or key)
            status = str(meta.get("status") or "日线观察")
            row_signals.append(
                {
                    "strategy_key": key,
                    "strategy_family": key,
                    "strategy_name": label,
                    "timeframe": "日线级，收盘确认，T+1 开盘观察",
                    "logic": f"{label} 缓存命中，来自全市场扩展策略规则信号。",
                    "reason": status,
                    "buy_plan": "T+1 开盘观察，按策略 playbook 与模型分共同过滤。",
                    "sell_plan": "按策略 playbook 风控退出。",
                    "strength_score": 1.0,
                }
            )
        if not row_signals:
            continue
        profile = basic.get(symbol, {})
        signals_by_symbol[symbol] = {
            "symbol": symbol,
            "name": _clean_text(row.get("name")) or _clean_text(profile.get("name")),
            "date": selected_date.strftime("%Y-%m-%d"),
            "close": float(row.get("close")) if pd.notna(row.get("close")) else None,
            "industry": _clean_text(profile.get("industry")),
            "signals": row_signals,
        }
    return signals_by_symbol


def _clear_selector_caches() -> None:
    _reload_production_strategy_configs()
    for func in [
        _extended_playbooks,
        _extended_model_playbooks,
        _extended_model_risk_managed_playbooks,
        _model_scored_candidates_for_date,
        _daily_profile_at_or_before,
        _family_best_metrics,
        _family_best_metrics_by_signal,
        _family_risk_managed_metrics_by_signal,
        _vegas_tunnel_best_metrics,
        _family_signals_for_date,
        _family_profiles_for_date,
        _latest_extended_signals,
        _load_live_long_base_cached,
        _load_live_long_base_full_cached,
        _tea_master_live_scores_cached,
        _selector_buy_hold_score_artifact,
        _selector_score_calibration,
        _selector_buy_hold_models,
        _selector_model_feature_rows,
        _selector_model_score_date,
        _selector_snapshot_dates_cached,
        _latest_candidate_signal_date,
        _stock_basic_for_similar_patterns,
        _stock_basic_map,
        _long_research_module,
        _tea_master_research_module,
    ]:
        try:
            func.cache_clear()
        except AttributeError:
            pass
    for module_name in (
        "quant_long_dividend_quality_research",
        "quant_tea_master_long_research",
    ):
        sys.modules.pop(module_name, None)


def _mark_refresh_step_started(step: dict[str, Any], now: datetime) -> None:
    step.setdefault("started_at", None)
    step.setdefault("finished_at", None)
    step.setdefault("elapsed_seconds", None)
    step.setdefault("checkpoint_reused", False)
    if step["started_at"] is None:
        step["started_at"] = now.isoformat(timespec="seconds")


def _mark_refresh_step_finished(step: dict[str, Any], now: datetime) -> None:
    _mark_refresh_step_started(step, now)
    step["finished_at"] = now.isoformat(timespec="seconds")
    started_at = _parse_refresh_timestamp(step["started_at"])
    step["elapsed_seconds"] = round(
        max(0.0, (now - started_at).total_seconds()) if started_at else 0.0,
        3,
    )


def _mark_refresh_step_checkpoint_reused(
    step: dict[str, Any],
    now: datetime,
) -> None:
    timestamp = now.isoformat(timespec="seconds")
    step["status"] = "success"
    step["started_at"] = timestamp
    step["finished_at"] = timestamp
    step["elapsed_seconds"] = 0.0
    step["checkpoint_reused"] = True


def _set_refresh_progress(
    *,
    status: str = "running",
    step_key: str | None = None,
    step_status: str | None = None,
    message: str,
    percent: int | None = None,
    result: Any = None,
    error: str | None = None,
    complete_previous: bool = True,
    checkpoint_reused: bool = False,
) -> None:
    with _REFRESH_LOCK:
        if not _owned_refresh_context_active_unlocked():
            return
        now = datetime.now()
        steps = list(_REFRESH_STATUS.get("steps") or _progress_steps())
        for step in steps:
            step.setdefault("started_at", None)
            step.setdefault("finished_at", None)
            step.setdefault("elapsed_seconds", None)
            step.setdefault("checkpoint_reused", False)
        if step_key:
            seen_current = False
            for step in steps:
                if step["key"] == step_key:
                    next_step_status = step_status or ("running" if status == "running" else status)
                    if checkpoint_reused:
                        _mark_refresh_step_checkpoint_reused(step, now)
                    else:
                        step["checkpoint_reused"] = False
                        step["status"] = next_step_status
                        _mark_refresh_step_started(step, now)
                        if next_step_status in {"success", "failed"}:
                            _mark_refresh_step_finished(step, now)
                    seen_current = True
                elif complete_previous and not seen_current and step["status"] in {"pending", "running"}:
                    step["status"] = "success"
                    _mark_refresh_step_finished(step, now)
            if step_status == "success" and not checkpoint_reused:
                for step in steps:
                    if step["key"] == step_key:
                        step["status"] = "success"
                        _mark_refresh_step_finished(step, now)
        next_percent = percent if percent is not None else _REFRESH_STATUS.get("percent", 0)
        if status == "running":
            next_percent = max(int(_REFRESH_STATUS.get("percent", 0) or 0), int(next_percent or 0))
        _REFRESH_STATUS.update(
            {
                "status": status,
                "message": message,
                "percent": next_percent,
                "current_step": step_key,
                "steps": steps,
                "result": result,
                "error": error,
                "updated_at": now.isoformat(timespec="seconds"),
            }
        )
        if status in {"success", "failed"}:
            _REFRESH_STATUS["finished_at"] = now.isoformat(timespec="seconds")
            started_at = _parse_refresh_timestamp(_REFRESH_STATUS.get("started_at"))
            _REFRESH_STATUS["elapsed_seconds"] = round(
                max(0.0, (now - started_at).total_seconds()) if started_at else 0.0,
                3,
            )
            try:
                _write_terminal_refresh_manifest_unlocked()
            except Exception as exc:
                _REFRESH_STATUS["manifest_error"] = str(exc)
        _persist_refresh_status_unlocked()


def _refresh_long_stock_pool_variant(variant: str, signal_date: str | None) -> dict[str, Any]:
    variant_key = variant if variant in LONG_VARIANTS else next(
        (key for key, value in LONG_VARIANTS.items() if value == variant),
        variant,
    )
    if variant_key in TEA_LONG_VARIANTS:
        payload = _build_tea_master_stock_pool_cached(variant_key, signal_date)
    else:
        payload = _build_long_stock_pool_cached(variant_key, signal_date, True)
    _write_long_stock_pool_snapshot(payload, variant_key, signal_date)
    result = {
        "variant": variant_key,
        "signal_date": payload.get("signal_date"),
        "stocks": len(payload.get("stocks") or []),
    }
    if payload.get("factor_snapshot"):
        result["factor_snapshot"] = payload["factor_snapshot"]
    return result


def _refresh_long_stock_pool_variants(variants: list[str], signal_date: str | None) -> list[dict[str, Any]]:
    if not variants:
        return []
    with ThreadPoolExecutor(max_workers=min(3, len(variants))) as executor:
        results = list(executor.map(lambda variant: _refresh_long_stock_pool_variant(variant, signal_date), variants))
    ordered = sorted(results, key=lambda item: variants.index(item["variant"]))
    if signal_date:
        stale = [item for item in ordered if str(item.get("signal_date") or "") != str(signal_date)]
        if stale:
            details = ", ".join(f"{item['variant']}={item.get('signal_date')}" for item in stale)
            raise RuntimeError(f"长线结果日期未更新到最新交易日: expected={signal_date}; {details}")
    return ordered


def _refresh_long_workspace(
    variants: list[str],
    signal_date: str | None,
) -> dict[str, Any]:
    """Refresh both long-horizon sub-strategies under one progress step."""

    variant_results = _refresh_long_stock_pool_variants(variants, signal_date)
    blood_chip = get_blood_chip_long_plan(signal_date=signal_date, refresh=True)
    if signal_date and str(blood_chip.get("signal_date") or "") != str(signal_date):
        raise RuntimeError(
            "带血筹结果日期未更新到最新交易日: "
            f"expected={signal_date} actual={blood_chip.get('signal_date')}"
        )
    return {
        "variants": variant_results,
        "blood_chip": {
            "signal_date": blood_chip.get("signal_date"),
            "candidates": len(blood_chip.get("candidates") or []),
            "simulated_positions": len(blood_chip.get("simulated_positions") or []),
        },
    }


def _ensure_selector_long_factor_snapshot(
    signal_date: str,
    *,
    force_refresh: bool,
) -> dict[str, Any]:
    """Publish the exact-date long-factor slice before selector v3 scoring.

    The selector consumes a subset of the governed long-page PIT factors.  A
    short-only routine therefore needs the factor slice even though it does
    not build or publish the long-page strategy pools.
    """

    active_features = _selector_active_model_features()
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}
    needs_long_snapshot = any(
        definitions[feature].calculator_id == "long_snapshot"
        for feature in active_features
    )
    if not needs_long_snapshot:
        return {
            "status": "skipped",
            "reason": "active_selector_model_has_no_long_snapshot_features",
            "signal_date": signal_date,
            "checkpoint_reused": True,
        }

    manifest_path = LONG_FACTOR_SNAPSHOT_DIR / "latest.json"
    latest_path = LONG_FACTOR_SNAPSHOT_DIR / "latest.parquet"
    current: dict[str, Any] = {}
    try:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = {}
    current_is_valid = bool(
        latest_path.is_file()
        and current.get("status") == "success"
        and current.get("signal_date") == signal_date
        and current.get("coverage_status") == "complete"
        and current.get("factor_schema_version")
        == LONG_FACTOR_SNAPSHOT_SCHEMA_VERSION
        and int(current.get("factor_count") or 0)
        == len(LONG_PRODUCTION_FACTOR_COLUMNS)
    )
    if current_is_valid and not force_refresh:
        return {**current, "checkpoint_reused": True}

    # Clear once after shared inputs are refreshed.  The later all-scope long
    # workspace reuses this same tea payload rather than recalculating it.
    _build_tea_master_stock_pool_cached.cache_clear()
    payload = _build_tea_master_stock_pool_cached("tea", signal_date)
    snapshot = dict(payload.get("factor_snapshot") or {})
    if (
        snapshot.get("status") != "success"
        or snapshot.get("signal_date") != signal_date
        or snapshot.get("coverage_status") != "complete"
        or int(snapshot.get("factor_count") or 0)
        != len(LONG_PRODUCTION_FACTOR_COLUMNS)
    ):
        raise RuntimeError(
            "selector 长线因子截面发布失败: "
            f"expected={signal_date} actual={snapshot}"
        )
    return {**snapshot, "checkpoint_reused": False}


def _resume_tail_refresh_from_cached_selector(scope: str, resume_status: dict[str, Any] | None = None) -> dict[str, Any]:
    refresh_scope = _normalize_refresh_scope(scope)
    step_map = _step_status_map(resume_status)
    results: dict[str, Any] = dict((resume_status or {}).get("result") or {})
    full_payload = get_stock_selector_payload(include_extended=True, use_cache=True, full_snapshot=True)
    signal_date = full_payload.get("signal_date")

    if not signal_date:
        raise RuntimeError("断点续跑失败：未找到可复用的短线快照 signal_date")
    expected_trade_date = _source_expected_trade_date(results)
    if (
        expected_trade_date is None
        or _local_market_trade_date() != expected_trade_date
    ):
        raise RuntimeError("断点续跑快照已过期: source checkpoint is missing or stale")
    expected_signal_date = pd.to_datetime(
        expected_trade_date,
        format="%Y%m%d",
    ).date().isoformat()
    if str(signal_date) != expected_signal_date:
        raise RuntimeError(
            f"断点续跑快照已过期: expected={expected_signal_date} actual={signal_date}"
        )

    if refresh_scope == "short":
        _set_refresh_progress(step_key="snapshot", message="检测到断点，正在补写短线策略股票池快照", percent=98)
        written_pools = _write_strategy_pool_snapshots(full_payload, include_extended=True)
        results["snapshot"] = {
            "status": "success",
            "storage": "mysql" if MarketDataStore(MarketDataStoreConfig.from_env()).config.sql_url else "json",
            "strategy_pools": written_pools,
        }
        _run_post_snapshot_cache_cleanup(results)
        return results

    _build_tea_master_stock_pool_cached.cache_clear()
    _build_long_stock_pool_cached.cache_clear()
    long_variants = ["tea", "tea_safe", "v44"]
    trade_date = str(signal_date).replace("-", "") if signal_date else None

    pending_steps = [
        (
            "chan_model_strategy",
            lambda: get_chan_model_strategy_plan(20, True, signal_date),
            "缠论模型策略候选生成失败",
        ),
        ("long_stock_pool", lambda: _refresh_long_workspace(long_variants, signal_date), "长线策略与带血筹计划生成失败"),
        (
            "convertible_bond_plan",
            lambda: get_convertible_bond_grid_plan(
                trade_date,
                DEFAULT_CONVERTIBLE_BOND_GRID_LIMIT,
                bool(trade_date),
            ),
            "可转债策略计划刷新失败",
        ),
        (
            "convertible_bond_allotment",
            lambda: get_convertible_bond_allotments(
                refresh=True,
                stage_scope=DEFAULT_ALLOTMENT_STAGE_SCOPE,
                expected_trade_date=signal_date,
                validate_quality=True,
            ),
            "配债股数据刷新失败",
        ),
        ("byd_daily_plan", lambda: get_byd_daily_strategy(refresh=True), "BYD 做T日线计划刷新失败"),
        ("similar_patterns", lambda: _run_similar_pattern_analysis_isolated(), "相似走势决策台刷新失败"),
    ]
    pending_steps = [item for item in pending_steps if step_map.get(item[0]) != "success"]

    for step_key, _, _ in pending_steps:
        _set_refresh_progress(
            step_key=step_key,
            message=f"检测到断点，正在续跑 {REFRESH_STEP_DEFINITIONS[step_key]['label']}",
            percent=REFRESH_STEP_DEFINITIONS[step_key]["percent"],
            complete_previous=False,
        )

    if pending_steps:
        with ThreadPoolExecutor(max_workers=min(6, len(pending_steps))) as executor:
            futures = {
                executor.submit(func): (step_key, failure_message)
                for step_key, func, failure_message in pending_steps
            }
            for future in as_completed(futures):
                step_key, failure_message = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:
                    _set_refresh_progress(
                        status="failed",
                        step_key=step_key,
                        step_status="failed",
                        message="刷新任务失败",
                        error=f"{failure_message}: {exc}\n{traceback.format_exc(limit=5)}",
                        complete_previous=False,
                    )
                    raise
                if step_key == "long_stock_pool":
                    results[step_key] = {"status": "success", **payload}
                elif step_key == "chan_model_strategy":
                    results[step_key] = {
                        "status": "success",
                        "signal_date": payload.get("signal_date"),
                        "candidates": len(payload.get("candidates") or []),
                        "primary_count": payload.get("primary_count"),
                        "expanded_count": payload.get("expanded_count"),
                    }
                elif step_key == "convertible_bond_plan":
                    results[step_key] = {
                        "status": "success",
                        "trade_date": payload.get("trade_date") or signal_date,
                        "candidates": len(payload.get("candidates") or payload.get("items") or []),
                        "data_refresh": payload.get("data_refresh"),
                    }
                elif step_key == "convertible_bond_allotment":
                    results[step_key] = {
                        "status": "success",
                        "asof": payload.get("asof"),
                        "event_polled_through": payload.get(
                            "event_polled_through"
                        ),
                        "records": len(payload.get("records") or []),
                        "quality": payload.get("quality"),
                    }
                elif step_key == "similar_patterns":
                    results[step_key] = {
                        "status": "success",
                        "generated_at": payload.get("generated_at"),
                        "target_date": payload.get("target_date"),
                        "reference_library_refreshed_at": (
                            payload.get("cache") or {}
                        ).get("reference_library_refreshed_at"),
                        "targets": len(payload.get("results") or []),
                    }
                else:
                    planned_t = payload.get("planned_t") or {}
                    results[step_key] = {
                        "status": "success",
                        "signal_date": planned_t.get("signal_date"),
                        "alerts": len(payload.get("alerts") or []),
                    }
                _set_refresh_progress(
                    step_key=step_key,
                    step_status="success",
                    message=f"{failure_message.removesuffix('失败')}完成",
                    percent=REFRESH_STEP_DEFINITIONS[step_key]["percent"],
                    complete_previous=False,
                )

    _set_refresh_progress(step_key="snapshot", message="正在写入策略股票池快照", percent=98)
    written_pools = _write_strategy_pool_snapshots(full_payload, include_extended=True)
    results["snapshot"] = {
        "status": "success",
        "storage": "mysql" if MarketDataStore(MarketDataStoreConfig.from_env()).config.sql_url else "json",
        "strategy_pools": written_pools,
    }
    if "long_stock_pool" in results:
        results["snapshot"]["long_stock_pools"] = results["long_stock_pool"].get("variants")
    _run_post_snapshot_cache_cleanup(results)
    return results


def _run_latest_refresh_job(
    scope: str = "all",
    resume_status: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> None:
    from quant.routine.pipeline import (
        build_features,
        generate_dashboard,
        generate_daily_plan,
        refresh_chan_model_scores,
        refresh_data,
        refresh_daily_basic_data,
        refresh_factor_registry_snapshot,
        refresh_reference_inputs,
        publish_daily_dependency_contract,
        resolve_daily_dependency_source_options,
        refresh_strategy_signal_cache,
        score_latest_models,
    )

    refresh_scope = _normalize_refresh_scope(scope)
    refresh_label = REFRESH_SCOPE_LABELS[refresh_scope]
    refresh_run_id = run_id or _new_refresh_run_id(refresh_scope)
    _REFRESH_CONTEXT.run_id = refresh_run_id
    resume_tail = _tail_resume_ready(resume_status, refresh_scope)
    resume_inputs = not resume_tail and _input_resume_ready(resume_status, refresh_scope)
    reuse_completed = _completed_checkpoint_ready(resume_status, refresh_scope)
    early_workspace_executor: ThreadPoolExecutor | None = None
    early_workspace_futures: dict[Any, tuple[str, str]] = {}
    early_workspace_results_lock = threading.Lock()
    workspace_failure_step: str | None = None

    def shutdown_early_workspaces(*, cancel_pending: bool) -> None:
        nonlocal early_workspace_executor
        if early_workspace_executor is None:
            return
        if cancel_pending:
            for pending_future in early_workspace_futures:
                if not pending_future.done():
                    pending_future.cancel()
        early_workspace_executor.shutdown(
            wait=True,
            cancel_futures=cancel_pending,
        )
        early_workspace_executor = None

    def publish_dependency_gate(
        results: dict[str, Any],
        target_date: str,
        *,
        phase: str,
        strict_freshness: bool,
    ) -> dict[str, Any]:
        result = publish_daily_dependency_contract(
            target_date,
            refresh_scope,
            results,
            phase=phase,
            strict_freshness=strict_freshness,
        )
        results[f"dependency_{phase}"] = result
        unresolved = result.get("refresh_node_ids") or []
        if strict_freshness and (
            result.get("status") != "success" or unresolved
        ):
            failures = (result.get("freshness_audit") or {}).get("failures") or []
            details = ", ".join(
                f"{item.get('node_id')}={item.get('actual')}"
                for item in failures
            )
            if unresolved:
                details = "; ".join(
                    part
                    for part in (
                        details,
                        "unresolved=" + ",".join(str(value) for value in unresolved),
                    )
                    if part
                )
            raise RuntimeError(
                "每日依赖合同新鲜度门禁失败: " + (details or "unknown failure")
            )
        return result

    resume_contract_preview: dict[str, Any] = {}
    resume_preview_requires_source_refresh = False
    inherited_refresh_nodes = set(
        (
            ((resume_status or {}).get("result") or {}).get(
                "dependency_preflight"
            )
            or {}
        ).get("refresh_node_ids")
        or []
    ) if not reuse_completed else set()
    if resume_tail or resume_inputs:
        prior_results = dict((resume_status or {}).get("result") or {})
        prior_trade_date = _source_expected_trade_date(prior_results)
        if prior_trade_date is None:
            resume_tail = False
            resume_inputs = False
        else:
            resume_contract_preview = publish_daily_dependency_contract(
                pd.to_datetime(prior_trade_date, format="%Y%m%d").date().isoformat(),
                refresh_scope,
                prior_results,
                phase="resume_preflight",
                strict_freshness=False,
            )
            preview_nodes = resume_contract_preview.get("refresh_nodes") or []
            resume_preview_requires_source_refresh = any(
                item.get("layer") == "data_source"
                for item in preview_nodes
            )
            inherited_refresh_nodes.update(
                str(item.get("node_id"))
                for item in preview_nodes
                if item.get("node_id")
            )
            if resume_tail:
                completed_steps = {
                    key
                    for key, value in _step_status_map(resume_status).items()
                    if value == "success"
                }
                tail_conflicts = [
                    item
                    for item in preview_nodes
                    if not item.get("ui_step")
                    or str(item.get("ui_step")) in completed_steps
                ]
                if tail_conflicts:
                    resume_tail = False
                    resume_inputs = _input_resume_ready(
                        resume_status,
                        refresh_scope,
                    )
            if resume_inputs and resume_preview_requires_source_refresh:
                resume_inputs = False

    try:
        with _REFRESH_LOCK:
            current_run_id = _REFRESH_STATUS.get("run_id")
            if current_run_id not in {None, refresh_run_id}:
                return
            if current_run_id == refresh_run_id and _REFRESH_STATUS.get("status") in {"success", "failed"}:
                return
            attempt_started_at = (
                _REFRESH_STATUS.get("started_at")
                if current_run_id == refresh_run_id
                else None
            ) or datetime.now().isoformat(timespec="seconds")
            attempt_steps = list(
                _REFRESH_STATUS.get("steps")
                if current_run_id == refresh_run_id
                else []
            ) or _progress_steps(refresh_scope)
            _REFRESH_STATUS.update(
                {
                    "status": "running",
                    "run_id": refresh_run_id,
                    "trade_date": _source_expected_trade_date(
                        (resume_status or {}).get("result")
                    ),
                    "attempt": (
                        _REFRESH_STATUS.get("attempt")
                        if current_run_id == refresh_run_id
                        else None
                    )
                    or (1 if reuse_completed else _refresh_attempt_number(resume_status)),
                    "resumed_from": _REFRESH_STATUS.get("resumed_from")
                    if current_run_id == refresh_run_id
                    else (
                        None
                        if reuse_completed
                        else _refresh_resume_source(resume_status)
                    ),
                    "started_at": attempt_started_at,
                    "finished_at": None,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "message": (
                        f"{refresh_label}检测到断点，正在续跑尾段任务"
                        if resume_tail
                        else f"{refresh_label}检测到同日数据检查点，正在续跑计算阶段"
                        if resume_inputs
                        else f"{refresh_label}正在轮询输入并校验同日成功基线"
                        if reuse_completed
                        else f"{refresh_label}刷新任务已启动"
                    ),
                    "percent": 90 if resume_tail else 35 if resume_inputs else 1,
                    "current_step": (
                        "similar_patterns"
                        if (refresh_scope == "similar" and resume_inputs) or resume_tail
                        else "feature_cache"
                        if resume_inputs
                        else "refresh_data"
                    ),
                    "steps": attempt_steps,
                    "scope": refresh_scope,
                    "scope_label": refresh_label,
                    "result": None,
                    "error": None,
                    "manifest_path": None,
                    "manifest_error": None,
                }
            )
            if resume_tail:
                reused_keys = {
                    str(step.get("key") or "")
                    for step in (resume_status or {}).get("steps") or []
                    if step.get("status") == "success"
                }
                now = datetime.now()
                for step in _REFRESH_STATUS["steps"]:
                    if step.get("key") in reused_keys:
                        _mark_refresh_step_checkpoint_reused(step, now)
            _persist_refresh_status_unlocked()
        cache_cleanup = run_cache_cleanup(PROJECT_ROOT)
        source_options = resolve_daily_dependency_source_options(refresh_scope)
        source_option_result = {"status": "success", **source_options}
        results: dict[str, Any] = (
            dict((resume_status or {}).get("result") or {})
            if (resume_inputs or reuse_completed)
            else {}
        )
        results["cache_cleanup"] = cache_cleanup
        results["dependency_source_options"] = source_option_result
        if resume_tail:
            try:
                results = _resume_tail_refresh_from_cached_selector(refresh_scope, resume_status)
            except RuntimeError as exc:
                if not str(exc).startswith("断点续跑快照已过期:"):
                    raise
                # A stale selector snapshot is not a terminal failure when the
                # same-day source inputs are still valid. Fall back to the
                # compute-stage checkpoint and regenerate the selector before
                # retrying downstream workspaces.
                resume_tail = False
                resume_inputs = _input_resume_ready(resume_status, refresh_scope)
                if resume_preview_requires_source_refresh:
                    resume_inputs = False
                if not resume_inputs:
                    # Source checkpoints without an independent expected trade
                    # date are not safe to recover from downstream artifacts.
                    # Restart the source stages instead.
                    results = {}
                else:
                    results = dict((resume_status or {}).get("result") or {})
                results["tail_resume_fallback"] = {
                    "status": "success",
                    "reason": str(exc),
                    "action": "recompute_from_same_day_inputs",
                }
                results["cache_cleanup"] = cache_cleanup
                results["dependency_source_options"] = source_option_result
            else:
                results["cache_cleanup"] = cache_cleanup
                results["dependency_source_options"] = source_option_result
                expected_tail_trade_date = _source_expected_trade_date(results)
                if expected_tail_trade_date is None:
                    raise RuntimeError("断点续跑缺少目标交易日，无法执行每日依赖门禁")
                publish_dependency_gate(
                    results,
                    pd.to_datetime(
                        expected_tail_trade_date,
                        format="%Y%m%d",
                    ).date().isoformat(),
                    phase="postflight",
                    strict_freshness=True,
                )
                _set_refresh_progress(
                    status="success",
                    step_key="snapshot",
                    message="刷新任务完成，所有工作区数据与策略结果已生成",
                    percent=100,
                    result=results,
                )
                with _REFRESH_LOCK:
                    for step in _REFRESH_STATUS["steps"]:
                        step["status"] = "success"
                    _persist_refresh_status_unlocked()
                return

        if not resume_inputs:
            _set_refresh_progress(step_key="refresh_data", message=f"{refresh_label}：正在共享拉取 Tushare 最新日线数据", percent=10)
            results["refresh_data"] = refresh_data(
                dry_run=False,
                progress_callback=lambda percent, message: _set_refresh_progress(
                    step_key="refresh_data",
                    message=message,
                    percent=percent,
                ),
            )
            if results["refresh_data"].get("status") == "failed":
                raise RuntimeError(results["refresh_data"].get("stderr_tail") or "Tushare 数据刷新失败")

            if source_options["include_daily_basic"]:
                results["refresh_daily_basic"] = refresh_daily_basic_data(
                    dry_run=False,
                    progress_callback=lambda percent, message: _set_refresh_progress(
                        step_key="refresh_data",
                        message=message,
                        percent=percent,
                        complete_previous=False,
                    ),
                )
                if results["refresh_daily_basic"].get("status") == "failed":
                    raise RuntimeError("Tushare daily_basic 刷新失败")

            results["refresh_reference_inputs"] = refresh_reference_inputs(
                dry_run=False,
                include_financials=source_options["include_financials"],
                include_analyst=source_options["include_analyst"],
                include_stock_basic=source_options["include_stock_basic"],
                include_index=source_options["include_index"],
                include_market_regime=source_options["include_market_regime"],
                include_tradability=source_options["include_tradability"],
                long_factor_datasets=source_options["long_factor_datasets"],
            )
            if results["refresh_reference_inputs"].get("status") not in {"success", "skipped"}:
                details = results["refresh_reference_inputs"].get("critical_errors") or []
                raise RuntimeError(f"参考数据刷新失败: {details}")

        results["factor_registry"] = refresh_factor_registry_snapshot()
        if results["factor_registry"].get("status") != "success":
            raise RuntimeError(
                "统一因子注册表刷新失败: "
                + str(results["factor_registry"].get("error") or "unknown error")
            )

        _set_refresh_progress(
            step_key="refresh_data",
            step_status="success",
            message=(
                "已复用同日 Tushare 数据检查点"
                if resume_inputs
                else "Tushare 最新日线数据已拉取完成"
            ),
            percent=35,
            complete_previous=False,
            checkpoint_reused=resume_inputs,
        )
        _clear_selector_caches()
        expected_trade_date = _source_expected_trade_date(results)
        local_trade_date = _local_market_trade_date()
        if expected_trade_date is None or local_trade_date != expected_trade_date:
            raise RuntimeError(
                "行情新鲜度门禁失败: "
                f"expected={expected_trade_date} actual={local_trade_date}"
            )
        expected_signal_date = pd.to_datetime(
            expected_trade_date,
            format="%Y%m%d",
        ).date().isoformat()
        with _REFRESH_LOCK:
            if _owned_refresh_context_active_unlocked():
                _REFRESH_STATUS["trade_date"] = expected_trade_date
                _REFRESH_STATUS["updated_at"] = datetime.now().isoformat(
                    timespec="seconds"
                )
                _persist_refresh_status_unlocked()
        dependency_preflight = publish_dependency_gate(
            results,
            expected_signal_date,
            phase="preflight",
            strict_freshness=False,
        )
        planned_refresh_nodes = set(
            dependency_preflight.get("refresh_node_ids") or []
        )
        daily_basic_repair_dates = sorted(
            str(value)
            for value in (
                (results.get("refresh_daily_basic") or {}).get(
                    "downstream_refresh_dates"
                )
                or []
            )
            if str(value)
        )
        daily_basic_repair_start = (
            daily_basic_repair_dates[0] if daily_basic_repair_dates else None
        )
        dependency_preflight["daily_basic_repair_dates"] = (
            daily_basic_repair_dates
        )
        prior_attempt_preflight = (
            ((resume_status or {}).get("result") or {}).get(
                "dependency_preflight"
            )
            or {}
        )
        retry_identity_stable = bool(
            (resume_inputs or resume_tail)
            and _retry_preflight_identity_stable(
                prior_attempt_preflight,
                dependency_preflight,
            )
        )
        if retry_identity_stable:
            planned_refresh_nodes.difference_update(
                str(node_id)
                for node_id in prior_attempt_preflight.get(
                    "refresh_node_ids"
                )
                or []
            )
        else:
            planned_refresh_nodes.update(inherited_refresh_nodes)
        dependency_preflight["retry_identity_stable"] = retry_identity_stable
        dependency_preflight["refresh_node_ids"] = sorted(planned_refresh_nodes)
        planned_refresh_ui_steps = {
            str(item.get("ui_step"))
            for contract_payload in (
                dependency_preflight,
                resume_contract_preview,
                (
                    ((resume_status or {}).get("result") or {}).get(
                        "dependency_preflight"
                    )
                    or {}
                ),
            )
            for item in (contract_payload.get("refresh_nodes") or [])
            if item.get("ui_step")
        }
        if resume_contract_preview:
            dependency_preflight["resume_contract_preview"] = {
                "refresh_node_ids": resume_contract_preview.get(
                    "refresh_node_ids"
                )
                or [],
                "changed_model_nodes": resume_contract_preview.get(
                    "changed_model_nodes"
                )
                or [],
                "changed_contract_nodes": resume_contract_preview.get(
                    "changed_contract_nodes"
                )
                or [],
            }

        if reuse_completed and not planned_refresh_nodes:
            # Source/event polls and contract compilation above are always
            # executed.  If their semantic fingerprints did not change, all
            # downstream exact-date products are already proven current by
            # the committed baseline and can be reused atomically.
            with _REFRESH_LOCK:
                now = datetime.now()
                for step in _REFRESH_STATUS["steps"]:
                    if step.get("key") != "refresh_data":
                        _mark_refresh_step_checkpoint_reused(step, now)
                _persist_refresh_status_unlocked()
            publish_dependency_gate(
                results,
                expected_signal_date,
                phase="postflight",
                strict_freshness=True,
            )
            _set_refresh_progress(
                status="success",
                step_key="snapshot",
                message="输入与合同未变化，已复用同日全部下游产物",
                percent=100,
                result=results,
                complete_previous=False,
                checkpoint_reused=True,
            )
            with _REFRESH_LOCK:
                for step in _REFRESH_STATUS["steps"]:
                    step["status"] = "success"
                _persist_refresh_status_unlocked()
            return

        # These workspaces only need the refreshed shared inputs. Start their
        # network-heavy work before the CPU-bound short-selector branch and
        # join the same futures at the normal downstream barrier.
        if refresh_scope == "all":
            early_signal_date = expected_signal_date
            early_trade_date = early_signal_date.replace("-", "")
            early_jobs = [
                (
                    "convertible_bond_plan",
                    lambda: get_convertible_bond_grid_plan(
                        early_trade_date,
                        DEFAULT_CONVERTIBLE_BOND_GRID_LIMIT,
                        True,
                    ),
                    "可转债策略计划刷新失败",
                    "可转债策略计划已提前刷新完成",
                ),
                (
                    "convertible_bond_allotment",
                    lambda: get_convertible_bond_allotments(
                        refresh=True,
                        stage_scope=DEFAULT_ALLOTMENT_STAGE_SCOPE,
                        expected_trade_date=early_signal_date,
                        validate_quality=True,
                    ),
                    "配债股数据刷新失败",
                    "配债股数据已提前刷新完成",
                ),
                (
                    "byd_daily_plan",
                    lambda: get_byd_daily_strategy(refresh=True),
                    "BYD 做T日线计划刷新失败",
                    "BYD 做T日线计划已提前刷新完成",
                ),
            ]

            def run_early_workspace(
                step_key: str,
                operation,
                success_message: str,
            ) -> Any:
                payload = operation()
                if step_key == "convertible_bond_plan":
                    actual_date = str(payload.get("trade_date") or "").replace("-", "")
                    if actual_date != early_trade_date:
                        raise RuntimeError(
                            "可转债计划日期未更新到目标交易日: "
                            f"expected={early_trade_date} actual={actual_date}"
                        )
                elif step_key == "byd_daily_plan":
                    actual_date = str(
                        (payload.get("planned_t") or {}).get("signal_date") or ""
                    ).replace("-", "")
                    if actual_date != early_trade_date:
                        raise RuntimeError(
                            "BYD 日线计划日期未更新到目标交易日: "
                            f"expected={early_trade_date} actual={actual_date}"
                        )
                with early_workspace_results_lock:
                    if step_key == "convertible_bond_plan":
                        results[step_key] = {
                            "status": "success",
                            "trade_date": payload.get("trade_date") or early_signal_date,
                            "candidates": len(payload.get("candidates") or payload.get("items") or []),
                            "data_refresh": payload.get("data_refresh"),
                        }
                    elif step_key == "convertible_bond_allotment":
                        results[step_key] = {
                            "status": "success",
                            "asof": payload.get("asof"),
                            "event_polled_through": payload.get(
                                "event_polled_through"
                            ),
                            "records": len(payload.get("records") or []),
                            "quality": payload.get("quality"),
                        }
                    else:
                        planned_t = payload.get("planned_t") or {}
                        results[step_key] = {
                            "status": "success",
                            "signal_date": planned_t.get("signal_date"),
                            "alerts": len(payload.get("alerts") or []),
                        }
                _set_refresh_progress(
                    step_key=step_key,
                    step_status="success",
                    message=success_message,
                    percent=40,
                    complete_previous=False,
                )
                return payload

            pending_early_jobs = [
                item
                for item in early_jobs
                if (
                    (results.get(item[0]) or {}).get("status") != "success"
                    or item[0] in planned_refresh_ui_steps
                )
            ]
            if pending_early_jobs:
                early_workspace_executor = ThreadPoolExecutor(
                    max_workers=len(pending_early_jobs),
                    thread_name_prefix="quant-early-workspace",
                )
                for step_key, operation, failure_message, success_message in pending_early_jobs:
                    _set_refresh_progress(
                        step_key=step_key,
                        message=f"共享数据已就绪，后台提前执行 {REFRESH_STEP_DEFINITIONS[step_key]['label']}",
                        percent=35,
                        complete_previous=False,
                    )
                    future = early_workspace_executor.submit(
                        run_early_workspace,
                        step_key,
                        operation,
                        success_message,
                    )
                    early_workspace_futures[future] = (step_key, failure_message)

        if refresh_scope not in {"all", "short"}:
            signal_date = expected_signal_date
            trade_date = str(signal_date).replace("-", "") if signal_date else None
            if refresh_scope == "chan":
                _set_refresh_progress(step_key="chan_model_strategy", message="正在生成缠论模型策略候选", percent=92)
                results["refresh_chan_model_scores"] = refresh_chan_model_scores(
                    progress_callback=lambda percent, message: _set_refresh_progress(
                        step_key="chan_model_strategy",
                        message=message,
                        percent=percent,
                        complete_previous=False,
                    )
                )
                if results["refresh_chan_model_scores"].get("status") == "failed":
                    raise RuntimeError(
                        results["refresh_chan_model_scores"].get("stderr_tail") or "缠论实时评分刷新失败"
                    )
                payload = get_chan_model_strategy_plan(top_n=20, refresh=True, signal_date=signal_date)
                if str(payload.get("signal_date") or "") != signal_date:
                    raise RuntimeError(
                        "缠论结果日期未更新到目标交易日: "
                        f"expected={signal_date} actual={payload.get('signal_date')}"
                    )
                results["chan_model_strategy"] = {
                    "status": "success",
                    "signal_date": payload.get("signal_date"),
                    "candidates": len(payload.get("candidates") or []),
                    "primary_count": payload.get("primary_count"),
                    "expanded_count": payload.get("expanded_count"),
                }
                _set_refresh_progress(
                    step_key="chan_model_strategy",
                    step_status="success",
                    message="缠论模型策略候选生成完成",
                    percent=98,
                    complete_previous=False,
                )
            elif refresh_scope == "long":
                _build_tea_master_stock_pool_cached.cache_clear()
                _build_long_stock_pool_cached.cache_clear()
                _set_refresh_progress(step_key="long_stock_pool", message="正在计算长线策略与带血筹每日计划", percent=92)
                long_workspace = _refresh_long_workspace(["tea", "tea_safe", "v44"], signal_date)
                results["long_stock_pool"] = {"status": "success", **long_workspace}
                _set_refresh_progress(
                    step_key="long_stock_pool",
                    step_status="success",
                    message="长线策略与带血筹每日计划生成完成",
                    percent=98,
                    complete_previous=False,
                )
            elif refresh_scope == "cb":
                _set_refresh_progress(step_key="convertible_bond_plan", message="正在刷新可转债策略计划", percent=92)
                payload = get_convertible_bond_grid_plan(
                    trade_date,
                    DEFAULT_CONVERTIBLE_BOND_GRID_LIMIT,
                    bool(trade_date),
                )
                actual_date = str(payload.get("trade_date") or "").replace("-", "")
                if actual_date != trade_date:
                    raise RuntimeError(
                        "可转债计划日期未更新到目标交易日: "
                        f"expected={trade_date} actual={actual_date}"
                    )
                results["convertible_bond_plan"] = {
                    "status": "success",
                    "trade_date": payload.get("trade_date") or signal_date,
                    "candidates": len(payload.get("candidates") or payload.get("items") or []),
                    "data_refresh": payload.get("data_refresh"),
                }
                _set_refresh_progress(
                    step_key="convertible_bond_plan",
                    step_status="success",
                    message="可转债策略计划刷新完成",
                    percent=98,
                    complete_previous=False,
                )
            elif refresh_scope == "cbAllotment":
                _set_refresh_progress(step_key="convertible_bond_allotment", message="正在刷新配债股数据", percent=92)
                payload = get_convertible_bond_allotments(
                    refresh=True,
                    stage_scope=DEFAULT_ALLOTMENT_STAGE_SCOPE,
                    expected_trade_date=signal_date,
                    validate_quality=True,
                )
                results["convertible_bond_allotment"] = {
                    "status": "success",
                    "asof": payload.get("asof"),
                    "event_polled_through": payload.get(
                        "event_polled_through"
                    ),
                    "records": len(payload.get("records") or []),
                    "quality": payload.get("quality"),
                }
                _set_refresh_progress(
                    step_key="convertible_bond_allotment",
                    step_status="success",
                    message="配债股数据刷新完成",
                    percent=98,
                    complete_previous=False,
                )
            elif refresh_scope == "byd":
                _set_refresh_progress(step_key="byd_daily_plan", message="正在刷新 BYD 做T日线计划", percent=92)
                payload = get_byd_daily_strategy(refresh=True)
                planned_t = payload.get("planned_t") or {}
                actual_date = str(planned_t.get("signal_date") or "").replace("-", "")
                if actual_date != trade_date:
                    raise RuntimeError(
                        "BYD 日线计划日期未更新到目标交易日: "
                        f"expected={trade_date} actual={actual_date}"
                    )
                results["byd_daily_plan"] = {
                    "status": "success",
                    "signal_date": planned_t.get("signal_date"),
                    "alerts": len(payload.get("alerts") or []),
                }
                _set_refresh_progress(
                    step_key="byd_daily_plan",
                    step_status="success",
                    message="BYD 做T日线计划刷新完成",
                    percent=98,
                    complete_previous=False,
                )
            elif refresh_scope == "similar":
                _set_refresh_progress(
                    step_key="similar_patterns",
                    message="正在刷新最新交易日的自选池相似走势",
                    percent=92,
                )
                payload = _run_similar_pattern_analysis_isolated()
                if str(payload.get("target_date") or "") != signal_date:
                    raise RuntimeError(
                        "相似走势目标日期未更新到目标交易日: "
                        f"expected={signal_date} actual={payload.get('target_date')}"
                    )
                results["similar_patterns"] = {
                    "status": "success",
                    "generated_at": payload.get("generated_at"),
                    "target_date": payload.get("target_date"),
                    "reference_library_refreshed_at": (
                        payload.get("cache") or {}
                    ).get("reference_library_refreshed_at"),
                    "targets": len(payload.get("results") or []),
                }
                _set_refresh_progress(
                    step_key="similar_patterns",
                    step_status="success",
                    message="自选池相似走势刷新完成",
                    percent=98,
                    complete_previous=False,
                )
            publish_dependency_gate(
                results,
                signal_date,
                phase="postflight",
                strict_freshness=True,
            )
            _set_refresh_progress(
                status="success",
                step_key=_REFRESH_STATUS.get("current_step"),
                message=f"{refresh_label}刷新完成，共享数据与本页策略结果已生成",
                percent=100,
                result=results,
            )
            with _REFRESH_LOCK:
                for step in _REFRESH_STATUS["steps"]:
                    step["status"] = "success"
                _persist_refresh_status_unlocked()
            return

        expected_incremental_date = pd.to_datetime(
            expected_trade_date,
            format="%Y%m%d",
        ).normalize()

        def artifact_current(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                dates = pd.read_parquet(path, columns=["date"])["date"]
                latest = pd.to_datetime(dates, errors="coerce").max()
                return pd.notna(latest) and latest.normalize() == expected_incremental_date
            except Exception:
                return False

        def b1_gate_current() -> bool:
            path = PROJECT_ROOT / "data/features/b1/b1_gate_manifest.json"
            if not path.exists():
                return False
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                processed = pd.to_datetime(
                    payload.get("processed_through_date"),
                    errors="coerce",
                )
                return (
                    payload.get("status") == "success"
                    and pd.notna(processed)
                    and processed.normalize() == expected_incremental_date
                )
            except (OSError, json.JSONDecodeError, TypeError):
                return False

        def active_feature_sidecar_current() -> bool:
            """Require both the exact-date sidecar and its complete coverage proof."""

            parquet_path = (
                PROJECT_ROOT
                / "data/features/b1/active_candidate_project_features.parquet"
            )
            manifest_path = (
                PROJECT_ROOT
                / "data/features/b1/active_candidate_project_features_manifest.json"
            )
            if not parquet_path.is_file() or not manifest_path.is_file():
                return False
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                target = pd.to_datetime(payload.get("target_date"), errors="coerce")
                union_count = int(payload.get("union_candidate_count") or 0)
                parquet_current = (
                    union_count == 0 or artifact_current(parquet_path)
                )
                return (
                    payload.get("status") == "success"
                    and payload.get("candidate_coverage_status") == "complete"
                    and int(payload.get("factor_count") or 0)
                    == len(PROJECT_FACTOR_COLUMNS)
                    and payload.get("factor_schema_version")
                    == PROJECT_FACTOR_SCHEMA_VERSION
                    and pd.notna(target)
                    and target.normalize() == expected_incremental_date
                    and parquet_current
                )
            except (OSError, json.JSONDecodeError, TypeError):
                return False

        reusable_checkpoints = resume_inputs or reuse_completed
        feature_ready = (
            reusable_checkpoints
            and daily_basic_repair_start is None
            and "feature.project_daily" not in planned_refresh_nodes
            and (results.get("feature_cache") or {}).get("status") == "success"
            and active_feature_sidecar_current()
        )
        signal_paths = (
            PROJECT_ROOT / "data/features/b1/b1_gate_candidates.parquet",
            PROJECT_ROOT / "data/features/b1/b1_family_rule_candidates.parquet",
            PROJECT_ROOT / "data/features/z_skill_daily_candidates.parquet",
        )
        signal_ready = (
            reusable_checkpoints
            and "feature.strategy_signals" not in planned_refresh_nodes
            and (
                (
                    feature_ready
                    and all(artifact_current(path) for path in signal_paths[1:])
                )
                or (
                    b1_gate_current()
                    and all(path.is_file() for path in signal_paths)
                )
            )
        )
        if feature_ready:
            results["feature_cache"]["checkpoint_reused"] = True
            _set_refresh_progress(
                step_key="feature_cache",
                step_status="success",
                message="已复用同日 B1 特征缓存检查点",
                percent=45,
                complete_previous=False,
                checkpoint_reused=True,
            )
        if signal_ready:
            results["signal_cache"] = {
                "status": "success",
                "checkpoint_reused": True,
                "date": expected_incremental_date.date().isoformat(),
            }
            _set_refresh_progress(
                step_key="signal_cache",
                step_status="success",
                message="已复用同日全市场规则信号检查点",
                percent=68,
                complete_previous=False,
                checkpoint_reused=True,
            )

        # The signal pass owns the exact B1 gate for the new date. Run it first
        # so feature refresh computes the expensive 159-column frame only for
        # gate hits, while preserving a full-scan fallback for stale manifests.
        if not signal_ready:
            _set_refresh_progress(
                step_key="signal_cache",
                message="正在增量构建全市场规则信号与 B1 门控",
                percent=46,
                complete_previous=False,
            )
            results["signal_cache"] = refresh_strategy_signal_cache(
                progress_callback=lambda percent, message: _set_refresh_progress(
                    step_key="signal_cache",
                    message=message,
                    percent=max(46, min(68, percent)),
                    complete_previous=False,
                ),
            )
            if results["signal_cache"].get("status") == "failed":
                raise RuntimeError(
                    results["signal_cache"].get("stderr_tail")
                    or "策略规则信号重建失败"
                )
            _set_refresh_progress(
                step_key="signal_cache",
                step_status="success",
                message="策略规则信号与 B1 门控构建完成",
                percent=68,
                complete_previous=False,
            )

        if not feature_ready:
            _set_refresh_progress(
                step_key="feature_cache",
                message="正在按 B1 门控增量构建特征缓存",
                percent=35,
                complete_previous=False,
            )
            results["feature_cache"] = build_features(
                incremental_start_date=daily_basic_repair_start,
                progress_callback=lambda percent, message: _set_refresh_progress(
                    step_key="feature_cache",
                    message=message,
                    percent=percent,
                    complete_previous=False,
                ),
            )
            if results["feature_cache"].get("status") == "failed":
                raise RuntimeError(
                    results["feature_cache"].get("stderr_tail")
                    or "B1 特征缓存刷新失败"
                )
            _set_refresh_progress(
                step_key="feature_cache",
                step_status="success",
                message="B1 特征缓存刷新完成",
                percent=45,
                complete_previous=False,
            )

        # Generate independent outputs together after both upstream caches are
        # complete. Model and Chan scoring are capped at four workers each.
        # The right-side build waits for the short-lived model score to release
        # its workers, then overlaps its six workers with Chan's four so the
        # sustained CPU budget still fits the 10-core production host.
        _set_refresh_progress(
            step_key="daily_plan",
            message="正在并行生成每日计划、Dashboard、模型分与缠论评分",
            percent=50,
            complete_previous=False,
        )
        model_score_ready = (
            reusable_checkpoints
            and not (
                planned_refresh_nodes
                & {"score.z_skill"}
            )
            and (results.get("model_score") or {}).get("status") == "success"
        )
        _set_refresh_progress(
            step_key="model_score",
            message="已复用同日策略模型分检查点" if model_score_ready else "正在计算当日策略模型分",
            percent=70,
            complete_previous=False,
        )
        if model_score_ready:
            results["model_score"]["checkpoint_reused"] = True
            _set_refresh_progress(
                step_key="model_score",
                step_status="success",
                message="已复用同日策略模型分检查点",
                percent=70,
                complete_previous=False,
                checkpoint_reused=True,
            )
        _set_refresh_progress(
            step_key="chan_model_strategy",
            message="正在并行刷新缠论实时评分",
            percent=72,
            complete_previous=False,
        )
        worker_budget = configured_worker_budget()
        model_score_request = max(
            1, int(os.getenv("ROUTINE_MODEL_SCORE_WORKERS", "4"))
        )
        chan_execution = calculator_execution_settings("chan_live")
        chan_score_request = min(
            chan_execution.max_workers,
            max(
                1,
                int(
                    os.getenv(
                        "ROUTINE_CHAN_WORKERS",
                        str(chan_execution.default_workers),
                    )
                ),
            ),
        )
        right_side_enabled = (
            DEFAULT_SELECTOR_RANKING_CONFIG.source
            == SelectorRankingSource.RIGHT_SIDE_UNIFIED
        )
        left_side_enabled = DEFAULT_LEFT_SIDE_RANKING_CONFIG.enabled
        legacy_strategy_model_score_required = not (
            right_side_enabled and left_side_enabled
        )
        right_side_execution = calculator_execution_settings("right_side_rule")
        right_side_request = min(
            right_side_execution.max_workers,
            max(
                1,
                int(
                    os.getenv(
                        "ROUTINE_RIGHT_SIDE_WORKERS",
                        str(right_side_execution.default_workers),
                    )
                ),
            ),
        )
        model_phase = allocate_worker_budget(
            {"model_score": model_score_request, "chan": chan_score_request},
            total_budget=worker_budget,
        )
        right_side_phase = allocate_worker_budget(
            {"right_side": right_side_request, "chan": chan_score_request},
            total_budget=worker_budget,
        )
        chan_score_workers = min(model_phase["chan"], right_side_phase["chan"])
        model_score_workers = min(
            model_score_request,
            max(1, worker_budget - chan_score_workers),
        )
        right_side_workers = min(
            right_side_request,
            max(1, worker_budget - chan_score_workers),
        )
        model_score_gate = threading.Event()
        model_score_failures: list[BaseException] = []
        if model_score_ready or not legacy_strategy_model_score_required:
            model_score_gate.set()
        if not legacy_strategy_model_score_required:
            results["model_score"] = {
                "status": "retired",
                "reason": "two_unified_short_rankers_active",
                "active_consumers": 0,
            }

        def score_models_then_release_workers() -> dict[str, Any]:
            try:
                payload = score_latest_models(workers=model_score_workers)
                if payload.get("status") == "failed":
                    model_score_failures.append(
                        RuntimeError(
                            payload.get("stderr_tail")
                            or "当日策略模型分计算失败"
                        )
                    )
                return payload
            except BaseException as exc:
                model_score_failures.append(exc)
                raise
            finally:
                model_score_gate.set()

        def build_right_side_after_model_score() -> dict[str, Any]:
            model_score_gate.wait()
            if model_score_failures:
                return {
                    "status": "cancelled",
                    "reason": "策略模型分失败，取消右侧统一因子构建",
                }
            from quant.routine.right_side_unified_production import (
                run_right_side_unified_production,
            )

            return run_right_side_unified_production(
                expected_signal_date,
                factor_workers=right_side_workers,
            )

        def build_left_side_after_signal_cache() -> dict[str, Any]:
            from quant.routine.left_side_unified_production import (
                run_left_side_production,
            )

            return run_left_side_production(expected_signal_date)

        if right_side_enabled:
            _set_refresh_progress(
                step_key="right_side_unified_features",
                message="等待模型评分释放资源后，并行构建右侧统一因子",
                percent=72,
                complete_previous=False,
            )
        if left_side_enabled:
            _set_refresh_progress(
                step_key="left_side_unified_features",
                message="正在构建左侧统一因子与排序分",
                percent=72,
                complete_previous=False,
            )

        with ThreadPoolExecutor(
            max_workers=4 + int(right_side_enabled) + int(left_side_enabled),
            thread_name_prefix="quant-daily-output",
        ) as executor:
            output_futures = {
                executor.submit(
                    refresh_chan_model_scores,
                    progress_callback=lambda percent, message: _set_refresh_progress(
                        step_key="chan_model_strategy",
                        message=message,
                        percent=percent,
                        complete_previous=False,
                    ),
                    workers=chan_score_workers,
                ): ("refresh_chan_model_scores", "缠论实时评分刷新失败"),
            }
            if not left_side_enabled:
                output_futures[executor.submit(generate_daily_plan)] = (
                    "generate_daily_plan",
                    "最新策略每日计划生成失败",
                )
                output_futures[
                    executor.submit(generate_dashboard, allow_incompatible=True)
                ] = ("generate_dashboard", "B1 Dashboard 生成失败")
            else:
                results["generate_dashboard"] = {
                    "status": "retired",
                    "reason": "left_side_unified_replaces_legacy_b1_model_dashboard",
                    "active_consumers": 0,
                }
            if not model_score_ready and legacy_strategy_model_score_required:
                output_futures[
                    executor.submit(score_models_then_release_workers)
                ] = ("model_score", "当日策略模型分计算失败")
            if right_side_enabled:
                output_futures[
                    executor.submit(build_right_side_after_model_score)
                ] = ("right_side_unified", "右侧统一生产排序失败")
            if left_side_enabled:
                output_futures[
                    executor.submit(build_left_side_after_signal_cache)
                ] = ("left_side_unified", "左侧统一生产排序失败")

            completed_daily_outputs: set[str] = set()
            for future in as_completed(output_futures):
                result_key, failure_message = output_futures[future]
                payload = future.result()
                if payload.get("status") == "failed":
                    raise RuntimeError(
                        payload.get("stderr_tail")
                        or payload.get("error")
                        or failure_message
                    )
                if result_key == "right_side_unified":
                    if payload.get("status") == "cancelled":
                        continue
                    results["right_side_unified_features"] = {
                        "status": "success",
                        "target_date": payload.get("target_date"),
                        "factor_workers": right_side_workers,
                    }
                    results["right_side_unified_scores"] = payload
                    results["right_side_unified_adapter"] = {
                        **(payload.get("selector_adapter") or {}),
                        "status": "success",
                    }
                    for step_key, message, percent in (
                        (
                            "right_side_unified_features",
                            "右侧生产因子构建完成",
                            72,
                        ),
                        (
                            "right_side_unified_score",
                            "右侧统一排序分计算完成",
                            76,
                        ),
                        (
                            "right_side_unified_adapter",
                            "右侧 selector 排序适配校验完成",
                            78,
                        ),
                    ):
                        _set_refresh_progress(
                            step_key=step_key,
                            step_status="success",
                            message=message,
                            percent=percent,
                            complete_previous=False,
                        )
                    continue
                if result_key == "left_side_unified":
                    results["left_side_unified_features"] = {
                        "status": "success",
                        "target_date": payload.get("target_date"),
                        "checkpoint_reused": payload.get("checkpoint_reused", False),
                    }
                    results["left_side_unified_scores"] = payload
                    results["left_side_unified_adapter"] = {
                        **(payload.get("adapter") or {}),
                        "status": "success",
                    }
                    plan_payload = generate_daily_plan()
                    results["generate_daily_plan"] = {
                        **plan_payload,
                        "status": "success",
                        "ranking_source": "left_side_unified",
                    }
                    for step_key, message, percent in (
                        ("left_side_unified_features", "左侧生产因子构建完成", 73),
                        ("left_side_unified_score", "左侧统一排序分计算完成", 77),
                        ("left_side_unified_adapter", "左侧 selector 排序适配校验完成", 79),
                    ):
                        _set_refresh_progress(
                            step_key=step_key,
                            step_status="success",
                            message=message,
                            percent=percent,
                            complete_previous=False,
                        )
                    _set_refresh_progress(
                        step_key="daily_plan",
                        step_status="success",
                        message="左侧统一排序每日计划已生成；旧B1模型看板已退役",
                        percent=80,
                        complete_previous=False,
                    )
                    continue

                results[result_key] = payload
                if result_key == "model_score":
                    _set_refresh_progress(
                        step_key="model_score",
                        step_status="success",
                        message="当日策略模型分计算完成",
                        percent=70,
                        complete_previous=False,
                    )
                elif result_key == "refresh_chan_model_scores":
                    _set_refresh_progress(
                        step_key="chan_model_strategy",
                        step_status="success",
                        message="缠论实时评分刷新完成",
                        percent=72,
                        complete_previous=False,
                    )
                elif result_key in {
                    "generate_daily_plan",
                    "generate_dashboard",
                }:
                    completed_daily_outputs.add(result_key)
                    if completed_daily_outputs == {
                        "generate_daily_plan",
                        "generate_dashboard",
                    }:
                        _set_refresh_progress(
                            step_key="daily_plan",
                            step_status="success",
                            message=(
                                "最新策略每日计划已生成；正式 B1 历史看板因模型兼容门禁保留上一有效版本"
                                if results["generate_dashboard"].get("status")
                                == "skipped"
                                else "最新策略每日计划与 Dashboard 已生成"
                            ),
                            percent=72,
                            complete_previous=False,
                        )
        _set_refresh_progress(
            step_key="selector_core",
            message="正在准备选股模型所需的当日长线因子截面",
            percent=79,
            complete_previous=False,
        )
        selector_long_snapshot = _ensure_selector_long_factor_snapshot(
            expected_signal_date,
            force_refresh="feature.long_snapshot" in planned_refresh_nodes,
        )
        results["selector_long_factor_snapshot"] = selector_long_snapshot
        selector_long_snapshot_rebuilt = not bool(
            selector_long_snapshot.get("checkpoint_reused")
        )
        _clear_selector_caches()

        _set_refresh_progress(step_key="selector_core", message="正在计算核心策略股票池", percent=80)
        core_payload = get_stock_selector_payload(use_cache=False)
        if str(core_payload.get("signal_date") or "") != expected_signal_date:
            raise RuntimeError(
                "核心策略结果日期未更新到最新交易日: "
                f"expected={expected_signal_date} actual={core_payload.get('signal_date')}"
            )
        results["selector_core"] = {
            "status": "success",
            "signal_date": core_payload.get("signal_date"),
            "stocks": len(core_payload.get("stocks") or []),
        }

        _set_refresh_progress(step_key="selector_extended", message="正在计算全市场扩展策略信号", percent=90)
        full_payload = get_stock_selector_payload(
            signal_date=core_payload.get("signal_date"),
            include_extended=True,
            use_cache=False,
            full_snapshot=True,
        )
        if str(full_payload.get("signal_date") or "") != expected_signal_date:
            raise RuntimeError(
                "扩展策略结果日期未更新到最新交易日: "
                f"expected={expected_signal_date} actual={full_payload.get('signal_date')}"
            )
        results["selector_extended"] = {
            "status": "success",
            "signal_date": full_payload.get("signal_date"),
            "stocks": len(full_payload.get("stocks") or []),
        }
        _set_refresh_progress(
            step_key="selector_extended",
            step_status="success",
            message="全市场扩展策略信号已计算完成",
            percent=90,
            complete_previous=False,
        )

        if refresh_scope == "short":
            _set_refresh_progress(step_key="snapshot", message="正在写入短线策略股票池快照", percent=98)
            written_pools = _write_strategy_pool_snapshots(full_payload, include_extended=True)
            results["snapshot"] = {
                "status": "success",
                "storage": "mysql" if MarketDataStore(MarketDataStoreConfig.from_env()).config.sql_url else "json",
                "strategy_pools": written_pools,
            }
            _run_post_snapshot_cache_cleanup(results)
            publish_dependency_gate(
                results,
                expected_signal_date,
                phase="postflight",
                strict_freshness=True,
            )
            _set_refresh_progress(
                status="success",
                step_key="snapshot",
                message="短线策略刷新完成，共享数据与本页策略结果已生成",
                percent=100,
                result=results,
            )
            with _REFRESH_LOCK:
                for step in _REFRESH_STATUS["steps"]:
                    step["status"] = "success"
                _persist_refresh_status_unlocked()
            return

        if not selector_long_snapshot_rebuilt:
            _build_tea_master_stock_pool_cached.cache_clear()
        _build_long_stock_pool_cached.cache_clear()
        long_variants = ["tea", "tea_safe", "v44"]
        signal_date = full_payload.get("signal_date")
        _set_refresh_progress(
            step_key="chan_model_strategy",
            message="正在并行计算长线、缠论与相似走势，并汇合提前启动的工作区",
            percent=92,
            complete_previous=False,
        )
        _set_refresh_progress(
            step_key="long_stock_pool",
            message="正在并行计算长线、缠论与相似走势，并汇合提前启动的工作区",
            percent=92,
            complete_previous=False,
        )
        _set_refresh_progress(
            step_key="similar_patterns",
            message="正在并行计算长线、缠论与相似走势，并汇合提前启动的工作区",
            percent=92,
            complete_previous=False,
        )
        with ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="quant-late-workspace",
        ) as executor:
            futures = {
                executor.submit(get_chan_model_strategy_plan, 20, True, signal_date): (
                    "chan_model_strategy",
                    "缠论模型策略候选生成失败",
                ),
                executor.submit(_refresh_long_workspace, long_variants, signal_date): (
                    "long_stock_pool",
                    "长线策略与带血筹计划生成失败",
                ),
                executor.submit(_run_similar_pattern_analysis_isolated): (
                    "similar_patterns",
                    "相似走势决策台刷新失败",
                ),
            }
            futures.update(early_workspace_futures)
            for future in as_completed(futures):
                result_key, failure_message = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:
                    workspace_failure_step = result_key
                    raise RuntimeError(f"{failure_message}: {exc}") from exc
                if result_key == "long_stock_pool":
                    results[result_key] = {"status": "success", **payload}
                elif result_key == "chan_model_strategy":
                    if str(payload.get("signal_date") or "") != expected_signal_date:
                        raise RuntimeError(
                            "缠论结果日期未更新到目标交易日: "
                            f"expected={expected_signal_date} "
                            f"actual={payload.get('signal_date')}"
                        )
                    results[result_key] = {
                        "status": "success",
                        "signal_date": payload.get("signal_date"),
                        "candidates": len(payload.get("candidates") or []),
                        "primary_count": payload.get("primary_count"),
                        "expanded_count": payload.get("expanded_count"),
                    }
                elif result_key == "convertible_bond_plan":
                    results[result_key] = {
                        "status": "success",
                        "trade_date": payload.get("trade_date") or signal_date,
                        "candidates": len(payload.get("candidates") or payload.get("items") or []),
                        "data_refresh": payload.get("data_refresh"),
                    }
                elif result_key == "convertible_bond_allotment":
                    results[result_key] = {
                        "status": "success",
                        "asof": payload.get("asof"),
                        "event_polled_through": payload.get(
                            "event_polled_through"
                        ),
                        "records": len(payload.get("records") or []),
                    }
                elif result_key == "similar_patterns":
                    if str(payload.get("target_date") or "") != expected_signal_date:
                        raise RuntimeError(
                            "相似走势目标日期未更新到目标交易日: "
                            f"expected={expected_signal_date} "
                            f"actual={payload.get('target_date')}"
                        )
                    results[result_key] = {
                        "status": "success",
                        "generated_at": payload.get("generated_at"),
                        "target_date": payload.get("target_date"),
                        "reference_library_refreshed_at": (
                            payload.get("cache") or {}
                        ).get("reference_library_refreshed_at"),
                        "targets": len(payload.get("results") or []),
                    }
                else:
                    planned_t = payload.get("planned_t") or {}
                    results[result_key] = {
                        "status": "success",
                        "signal_date": planned_t.get("signal_date"),
                        "alerts": len(payload.get("alerts") or []),
                    }
                if future not in early_workspace_futures:
                    _set_refresh_progress(
                        step_key=result_key,
                        step_status="success",
                        message=f"{failure_message.removesuffix('失败')}完成",
                        percent={
                            "chan_model_strategy": 93,
                            "long_stock_pool": 94,
                            "similar_patterns": 97,
                        }[result_key],
                        complete_previous=False,
                    )

        shutdown_early_workspaces(cancel_pending=False)

        _set_refresh_progress(step_key="snapshot", message="正在写入策略股票池快照", percent=98)
        written_pools = _write_strategy_pool_snapshots(full_payload, include_extended=True)
        results["snapshot"] = {
            "status": "success",
            "storage": "mysql" if MarketDataStore(MarketDataStoreConfig.from_env()).config.sql_url else "json",
            "strategy_pools": written_pools,
            "long_stock_pools": results["long_stock_pool"]["variants"],
        }
        _run_post_snapshot_cache_cleanup(results)
        publish_dependency_gate(
            results,
            expected_signal_date,
            phase="postflight",
            strict_freshness=True,
        )

        _set_refresh_progress(
            status="success",
            step_key="snapshot",
            message="刷新任务完成，所有工作区数据与策略结果已生成",
            percent=100,
            result=results,
        )
        with _REFRESH_LOCK:
            for step in _REFRESH_STATUS["steps"]:
                step["status"] = "success"
    except Exception as exc:
        # Let running early work finish (and merge any successful checkpoint
        # payloads) before persisting the terminal failure state. This avoids
        # background threads mutating results after the run has ended.
        shutdown_early_workspaces(cancel_pending=True)
        _set_refresh_progress(
            status="failed",
            step_key=workspace_failure_step or _REFRESH_STATUS.get("current_step"),
            message="刷新任务失败",
            result=results if "results" in locals() else None,
            error=f"{exc}\n{traceback.format_exc(limit=5)}",
        )
    finally:
        shutdown_early_workspaces(cancel_pending=True)
        if getattr(_REFRESH_CONTEXT, "run_id", None) == refresh_run_id:
            delattr(_REFRESH_CONTEXT, "run_id")


def start_latest_refresh(scope: str = "all") -> dict[str, Any]:
    refresh_scope = _normalize_refresh_scope(scope)
    refresh_label = REFRESH_SCOPE_LABELS[refresh_scope]
    resume_status: dict[str, Any] | None = None
    run_id = _new_refresh_run_id(refresh_scope)
    with _REFRESH_LOCK:
        current_status = _ensure_refresh_scope(dict(_REFRESH_STATUS))
        if current_status.get("status") == "running":
            if _is_refresh_status_stale(current_status):
                current_status = _expire_stale_refresh_status_unlocked(current_status)
                if _refresh_resume_ready(current_status, refresh_scope):
                    resume_status = dict(current_status)
            else:
                return dict(current_status)
        elif current_status.get("status") == "failed":
            if _refresh_resume_ready(current_status, refresh_scope):
                resume_status = dict(current_status)
        elif current_status.get("status") == "success":
            if _completed_checkpoint_ready(current_status, refresh_scope):
                resume_status = dict(current_status)
        elif current_status.get("status") == "idle":
            persisted = _load_persisted_refresh_status()
            if persisted and persisted.get("status") != "idle":
                persisted = _ensure_refresh_scope(persisted)
                if persisted.get("status") in {"running", "queued"}:
                    persisted = _expire_interrupted_refresh_status_unlocked(persisted)
                if _refresh_resume_ready(persisted, refresh_scope):
                    resume_status = dict(persisted)
                elif _completed_checkpoint_ready(persisted, refresh_scope):
                    resume_status = dict(persisted)
        reuse_completed = _completed_checkpoint_ready(
            resume_status,
            refresh_scope,
        )
        thread = threading.Thread(
            target=_run_latest_refresh_job,
            args=(refresh_scope, resume_status, run_id),
            name=f"quant-{refresh_scope}-refresh",
            daemon=True,
        )
        attempt_started_at = datetime.now().isoformat(timespec="seconds")
        _REFRESH_STATUS.update(
            {
                "status": "queued",
                "run_id": run_id,
                "trade_date": _source_expected_trade_date(
                    (resume_status or {}).get("result")
                ),
                "attempt": 1 if reuse_completed else _refresh_attempt_number(resume_status),
                "resumed_from": (
                    None if reuse_completed else _refresh_resume_source(resume_status)
                ),
                "started_at": attempt_started_at,
                "finished_at": None,
                "updated_at": attempt_started_at,
                "message": (
                    f"{refresh_label}检测到同日成功基线，已进入增量校验队列"
                    if reuse_completed
                    else f"{refresh_label}刷新任务已进入后台队列"
                    if not resume_status
                    else f"{refresh_label}检测到断点，已进入自动续跑队列"
                ),
                "percent": 0 if not resume_status else int(resume_status.get("percent") or 35),
                "current_step": None if not resume_status else str(resume_status.get("current_step") or "feature_cache"),
                "steps": _progress_steps(refresh_scope),
                "scope": refresh_scope,
                "scope_label": refresh_label,
                "result": None,
                "error": None,
                "manifest_path": None,
                "manifest_error": None,
            }
        )
        _persist_refresh_status_unlocked()
        thread.start()
        return dict(_REFRESH_STATUS)


def get_latest_refresh_status() -> dict[str, Any]:
    with _REFRESH_LOCK:
        if _REFRESH_STATUS.get("status") == "idle":
            persisted = _load_persisted_refresh_status()
            if persisted and persisted.get("status") != "idle":
                if persisted.get("status") in {"running", "queued"}:
                    return dict(_expire_interrupted_refresh_status_unlocked(persisted))
                return dict(_expire_stale_refresh_status_unlocked(persisted))
        if _is_refresh_status_stale(_REFRESH_STATUS):
            return dict(_expire_stale_refresh_status_unlocked(_REFRESH_STATUS))
        return _ensure_refresh_scope(_REFRESH_STATUS)


def _selected_extended_keys(strategies: list[str] | None) -> set[str]:
    selected = _strategy_filter_members(strategies)
    known = {str(item.get("key", "")).upper() for item in EXTENDED_STRATEGIES}
    return selected & known


def get_stock_selector_payload(
    strategies: list[str] | None = None,
    signal_date: str | None = None,
    include_extended: bool = False,
    use_cache: bool = True,
    full_snapshot: bool = False,
) -> dict[str, Any]:
    extended_filter = _selected_extended_keys(strategies)
    effective_include_extended = include_extended or bool(extended_filter)
    if not signal_date and not use_cache:
        signal_date = _latest_candidate_signal_date()
    elif signal_date and not use_cache:
        parsed_signal_date = pd.to_datetime(signal_date, errors="raise")
        if parsed_signal_date.weekday() >= 5:
            signal_date = _resolve_selector_signal_date(signal_date, strategies, effective_include_extended)
    else:
        signal_date = _resolve_selector_signal_date(signal_date, strategies, effective_include_extended)
    if use_cache:
        cached = _read_selector_snapshot(signal_date, strategies, effective_include_extended)
        if cached is not None:
            return cached if full_snapshot else _display_selector_payload(cached)

    plan = get_b1_plan(signal_date=signal_date)
    effective_signal_date = signal_date or plan.get("signal_date")
    stocks: dict[str, dict[str, Any]] = {}
    for row in plan.get("plan_rows", []):
        symbol = str(row.get("symbol"))
        if symbol not in stocks:
            stocks[symbol] = {
                "symbol": symbol,
                "name": row.get("name") or "",
                "date": row.get("date") or plan.get("signal_date"),
                "close": row.get("close"),
                "industry": row.get("industry") or "",
                "signals": [],
            }
            _fill_stock_profile(stocks[symbol], effective_signal_date)
        stocks[symbol]["signals"].append(_b1_model_signal(row))

    schema_cols = set(pd.read_parquet(FEATURE_PATH).columns)
    profile_cols = [col for col in ["date", "symbol", "name", "industry", "close"] if col in schema_cols]
    features = pd.read_parquet(FEATURE_PATH, columns=profile_cols)
    features["date"] = pd.to_datetime(features["date"])
    if effective_signal_date:
        target = pd.to_datetime(effective_signal_date)
        available = features[features["date"] <= target]["date"]
        feature_date = available.max() if not available.empty else features["date"].max()
    else:
        feature_date = features["date"].max()
    latest = features[features["date"] == feature_date].drop_duplicates("symbol").set_index("symbol")
    for symbol, signals in _family_signals_for_date(effective_signal_date).items():
        if symbol not in stocks:
            profile = _family_profiles_for_date(effective_signal_date).get(symbol)
            row = latest.loc[symbol] if symbol in latest.index else {}
            stocks[symbol] = {
                "symbol": symbol,
                "name": (profile or {}).get("name") or (row.get("name", "") if hasattr(row, "get") else ""),
                "date": (profile or {}).get("date") or (row.get("date").strftime("%Y-%m-%d") if hasattr(row, "get") and pd.notna(row.get("date")) else plan.get("signal_date")),
                "close": (profile or {}).get("close") if profile else (float(row.get("close")) if hasattr(row, "get") and pd.notna(row.get("close")) else None),
                "industry": (profile or {}).get("industry") or (row.get("industry", "") if hasattr(row, "get") else ""),
                "signals": [],
            }
            _fill_stock_profile(stocks[symbol], effective_signal_date)
        stocks[symbol]["signals"].extend(signals)

    model_scored = _model_scored_candidates_for_date(effective_signal_date)
    if effective_include_extended:
        for symbol, payload in _latest_extended_signals(effective_signal_date).items():
            stock_name = _clean_text(payload.get("name"))
            basic_profile = _stock_basic_map().get(symbol, {})
            if not stock_name:
                stock_name = _clean_text(basic_profile.get("name"))
            if "ST" in stock_name.upper() or "退" in stock_name:
                continue
            if symbol not in stocks:
                stocks[symbol] = {
                    "symbol": symbol,
                    "name": stock_name,
                    "date": payload.get("date") or plan.get("signal_date"),
                    "close": payload.get("close"),
                    "industry": _clean_text(payload.get("industry")) or _clean_text(basic_profile.get("industry")),
                    "signals": [],
                }
                _fill_stock_profile(stocks[symbol], effective_signal_date)
            else:
                stock = stocks[symbol]
                if not _clean_text(stock.get("industry")):
                    stock["industry"] = _clean_text(payload.get("industry")) or _clean_text(basic_profile.get("industry"))
                if not _clean_text(stock.get("name")):
                    stock["name"] = stock_name
                _fill_stock_profile(stock, effective_signal_date)
            enriched_signals = []
            for signal in payload.get("signals") or []:
                strategy_key = str(signal.get("strategy_key"))
                if extended_filter and strategy_key.upper() not in extended_filter:
                    continue
                model_score = model_scored.get((symbol, strategy_key))
                enriched_signals.append(_extended_signal_payload(signal, model_score=model_score))
            stocks[symbol]["signals"].extend(enriched_signals)

    for (symbol, signal_key), model_score in model_scored.items():
        if signal_key not in MODEL_FILTERED_SIGNALS:
            continue
        existing = stocks.get(symbol, {}).get("signals", [])
        if any(signal.get("strategy_family") == signal_key and signal.get("playbook_source") == "模型版" for signal in existing):
            continue
        row = latest.loc[symbol] if symbol in latest.index else {}
        if symbol not in stocks:
            basic_profile = _stock_basic_profile(symbol)
            stocks[symbol] = {
                "symbol": symbol,
                "name": (row.get("name", "") if hasattr(row, "get") else "") or basic_profile["name"],
                "date": row.get("date").strftime("%Y-%m-%d") if hasattr(row, "get") and pd.notna(row.get("date")) else plan.get("signal_date"),
                "close": float(row.get("close")) if hasattr(row, "get") and pd.notna(row.get("close")) else None,
                "industry": (row.get("industry", "") if hasattr(row, "get") else "") or basic_profile["industry"],
                "signals": [],
            }
            _fill_stock_profile(stocks[symbol], effective_signal_date)
        stocks[symbol]["signals"].append(_model_filtered_signal_payload(signal_key, model_score))

    if DEFAULT_LEFT_SIDE_RANKING_CONFIG.enabled:
        left_candidates, _ = load_left_side_ranking_candidates(
            effective_signal_date,
            config=DEFAULT_LEFT_SIDE_RANKING_CONFIG,
        )
        _materialize_left_side_ranked_candidates(
            stocks,
            left_candidates,
            latest,
            effective_signal_date,
        )

    selected = {item.upper() for item in strategies or [] if item}
    selected_members = _strategy_filter_members(strategies)
    selected_groups = _strategy_filter_groups(strategies)
    rows = []
    for stock in stocks.values():
        if selected:
            signals = [
                signal
                for signal in stock["signals"]
                if _signal_matches_filter(signal, selected_members, selected_groups)
            ]
        else:
            signals = stock["signals"]
            if not full_snapshot:
                signals = [signal for signal in signals if _signal_quality_gate(signal)]
        if not signals:
            continue
        row = build_selector_stock_row(stock, signals, effective_signal_date)
        if row is not None:
            rows.append(row)
    rows = sorted(
        rows,
        key=lambda item: (item["selector_score"], item["matched_count"], item["best_profit_factor"]),
        reverse=True,
    )
    rows = _apply_historical_score_normalization(rows)
    rows = apply_selector_ranking_source(
        rows,
        effective_signal_date,
        require_all_ranked_candidates=bool(
            full_snapshot and effective_include_extended
        ),
    )
    rows = sorted(
        rows,
        key=lambda item: (item["selector_score"], item["matched_count"], item["best_profit_factor"]),
        reverse=True,
    )
    if not selected:
        rows = _diversify_default_rows(rows, len(rows) or DEFAULT_SELECTOR_LIMIT)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": effective_signal_date,
        "execution_date": plan.get("execution_date"),
        "ranking_source": DEFAULT_SELECTOR_RANKING_CONFIG.source.value,
        "available_strategies": [
            {
                "key": item["key"],
                "label": item["label"],
                "status": item["status"],
                "members": item["members"],
                "side": SHORT_GROUP_SIDE.get(item["key"], "unknown"),
            }
            for item in STRATEGY_GROUPS
        ],
        "stocks": rows,
        "total_stock_count": len(rows),
        "display_limit": None if full_snapshot else DEFAULT_SELECTOR_LIMIT,
        "is_truncated": False,
        "notes": [
            "选股器按合并后的策略组聚合命中；同一策略组下相同买入操作只保留综合效果最优的一版，不同买入操作会同时展示，命中按策略组去重计算。",
            "2026-06-14 已统一短线和长线为前复权价格口径；模型训练、标签、策略回测和页面价格展示使用同一套价格体系。",
            "B1 已合并模型分和规则信号；B2/B3 当前使用全市场规则候选缓存，不再受 B1 模型候选池限制。",
            "买入分使用同一模型版本下跨日期可比的 0-100 固定历史参考标尺；它是未来 5 日最大冲高收益预测的鲁棒映射，不是胜率，也不是直接经验分位。",
            "持有分使用同一模型版本下跨日期可比的固定历史参考标尺，以未来 5 日收盘收益为主、冲高能力为辅助；分数用于历史排序，不等同于收益承诺。",
            f"前端默认先过滤为可操作候选，再展示保守预期收益分 Top{DEFAULT_SELECTOR_LIMIT}；MySQL 快照仍保存完整规则全集，便于后续复盘和重新训练。",
            "SB1 和超级B1 本质偏盘中/尾盘战法，正式交易前仍需要分钟级数据确认买点。",
            "异动地量、黄金碗等策略已完成模型版买点评估；当前选股器对所有策略使用同一套筛选、排序、快照和复盘口径。",
        ],
    }
    if full_snapshot:
        _write_selector_snapshot(payload, strategies, effective_include_extended)
    out = payload if full_snapshot else _display_selector_payload(payload)
    out["cache"] = {"hit": False, "backend": "generated"}
    return out
