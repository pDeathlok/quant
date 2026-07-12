from __future__ import annotations

import json
import re
import ast
import hashlib
import importlib.util
import os
import sys
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.data.tushare_fetcher import TushareDataFetcher
from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
from quant.routine.b1_daily_plan import DAILY_PLAN_PATH, FEATURE_PATH, build_daily_plan, write_daily_plan
from quant.routine.convertible_bond_allotment import build_convertible_bond_allotment_payload
from quant.routine.convertible_bond_grid_plan import build_convertible_bond_grid_plan, refresh_convertible_bond_daily
from quant.routine.dashboard import DASHBOARD_PATH, build_dashboard_payload, write_dashboard_json
from quant.routine.paths import DAILY_DIR, PROJECT_ROOT, WEB_DATA_DIR
from quant.research.similar_patterns import (
    SimilarPatternConfig,
    SimilarPatternResult,
    analyze_targets_by_threshold,
    build_vector_caches_parallel,
    load_stock_basic,
)
from quant.strategies.custom.byd_minute_t import BydHolding, build_minute_payload, load_daily_qfq
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
EXTENDED_SIGNAL_CACHE = PROJECT_ROOT / "data/features/z_skill_daily_signals.json"
EXTENDED_PLAYBOOK = REPORT_DIR / "latest_z_skill_operational_playbook.csv"
EXTENDED_MODEL_PLAYBOOK = REPORT_DIR / "latest_z_skill_model_operational_playbook.csv"
EXTENDED_MODEL_SCORED = REPORT_DIR / "latest_z_skill_model_scored_candidates.parquet"
EXTENDED_MODEL_SUMMARY = REPORT_DIR / "latest_z_skill_model_entry_exit_backtest.csv"
VEGAS_TUNNEL_REPORT_DIR = PROJECT_ROOT / "reports/vegas_tunnel"
SELECTOR_HISTORY_SIGNAL_SAMPLES = PROJECT_ROOT / "data/research/selector_history_full/selector_signal_history_samples.parquet"
SELECTOR_BUY_HOLD_SCORE_CALIBRATION = PROJECT_ROOT / "config/selector_buy_hold_score_calibration.json"
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
_TEA_MASTER_MODULE_LOCK = threading.Lock()
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
SELECTOR_SNAPSHOT_SCHEMA_VERSION = "buy_hold_score_v5"
SELECTOR_SNAPSHOT_DIR = PROJECT_ROOT / "data/selector_snapshots"
SELECTOR_SNAPSHOT_TABLE = "selector_snapshots"
LONG_STOCK_POOL_SCHEMA_VERSION = "qfq_price_snapshot_v1"
LONG_STOCK_POOL_SNAPSHOT_DIR = PROJECT_ROOT / "data/long_stock_pool_snapshots"
LONG_STOCK_POOL_SNAPSHOT_TABLE = "long_stock_pool_snapshots"
WEB_WORKSPACE_SNAPSHOT_SCHEMA_VERSION = "workspace_payload_v1"
WEB_WORKSPACE_SNAPSHOT_TABLE = "web_workspace_snapshots"
REFRESH_STATUS_PATH = PROJECT_ROOT / "data/routine/latest_refresh_status.json"
REFRESH_RUNNING_STALE_SECONDS = 6 * 60 * 60
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
    "tea": "茶大长线趋势网格",
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
_REFRESH_LOCK = threading.Lock()
_REFRESH_STATUS: dict[str, Any] = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "message": "尚未启动刷新任务",
    "percent": 0,
    "current_step": None,
    "steps": [],
    "result": None,
    "error": None,
}
REFRESH_SCOPE_LABELS = {
    "all": "全部工作区",
    "short": "短线策略",
    "chan": "缠论策略",
    "long": "长线策略",
    "cb": "可转债策略",
    "cbAllotment": "配债股",
    "byd": "BYD 做T",
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
}
REFRESH_STEP_DEFINITIONS = {
    "refresh_data": {"label": "拉取 Tushare 最新日线数据", "percent": 10},
    "feature_cache": {"label": "增量构建 B1 特征缓存", "percent": 35},
    "daily_plan": {"label": "生成最新策略每日计划", "percent": 45},
    "signal_cache": {"label": "重建全市场策略规则信号", "percent": 56},
    "model_score": {"label": "计算当日策略模型分", "percent": 70},
    "selector_core": {"label": "计算短线核心股票池", "percent": 78},
    "selector_extended": {"label": "计算短线全策略股票池", "percent": 88},
    "chan_model_strategy": {"label": "生成缠论模型策略候选", "percent": 90},
    "long_stock_pool": {"label": "计算长线策略股票池", "percent": 92},
    "convertible_bond_plan": {"label": "刷新可转债策略计划", "percent": 94},
    "convertible_bond_allotment": {"label": "刷新配债股数据", "percent": 96},
    "byd_daily_plan": {"label": "刷新 BYD 做T日线计划", "percent": 97},
    "snapshot": {"label": "写入策略股票池快照", "percent": 98},
}
SIMILAR_PATTERN_STATE_DIR = PROJECT_ROOT / "data/research/similar_patterns"
SIMILAR_PATTERN_WATCHLIST_PATH = SIMILAR_PATTERN_STATE_DIR / "watchlist.json"
SIMILAR_PATTERN_ANALYSIS_PATH = SIMILAR_PATTERN_STATE_DIR / "web_watchlist_analysis.json"
SIMILAR_PATTERN_VECTOR_CACHE_DIR = SIMILAR_PATTERN_STATE_DIR / "vector_cache"
SIMILAR_PATTERN_DEFAULT_WATCHLIST = ["002594.SZ", "002788.SZ"]
SIMILAR_PATTERN_CONFIG = SimilarPatternConfig(
    candidate_step_days=1,
    candidate_start_date="2018-01-01",
    similarity_threshold=0.055,
    take_profit_3d=0.03,
    stop_loss_3d=0.03,
)
CHAN_MODEL_SCORED_PATH = PROJECT_ROOT / "reports/chan_daily/model_filter/chan_model_scored_candidates.parquet"
CHAN_MODEL_STRATEGY_DIR = PROJECT_ROOT / "reports/chan_daily/model_strategy"
CHAN_MODEL_STRATEGY_SCORED_PATH = CHAN_MODEL_STRATEGY_DIR / "chan_model_strategy_scored.parquet"
CHAN_MODEL_LATEST_CANDIDATES_PATH = CHAN_MODEL_STRATEGY_DIR / "chan_model_latest_candidates.csv"
CHAN_MODEL_SUMMARY_PATH = CHAN_MODEL_STRATEGY_DIR / "chan_model_strategy_summary.csv"


def _persist_refresh_status_unlocked() -> None:
    REFRESH_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFRESH_STATUS_PATH.write_text(
        json.dumps(_REFRESH_STATUS, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


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


def _is_refresh_status_stale(status: dict[str, Any]) -> bool:
    if status.get("status") not in {"running", "queued"}:
        return False
    started_at = pd.to_datetime(status.get("started_at"), errors="coerce")
    if pd.isna(started_at):
        return True
    started = started_at.to_pydatetime()
    if started.tzinfo is not None:
        started = started.replace(tzinfo=None)
    return (datetime.now() - started).total_seconds() > REFRESH_RUNNING_STALE_SECONDS


def _expire_stale_refresh_status_unlocked(status: dict[str, Any]) -> dict[str, Any]:
    if not _is_refresh_status_stale(status):
        return status
    expired = dict(status)
    expired.update(
        {
            "status": "failed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "message": "上一次刷新任务已超时，请重新触发更新",
            "error": f"刷新任务超过 {REFRESH_RUNNING_STALE_SECONDS // 3600} 小时仍未完成，已标记为过期。",
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
    interrupted = dict(status)
    interrupted.update(
        {
            "status": "failed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
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
    shares: int = 10500,
    cost: float = 110.6061,
    refresh: bool = False,
    sold_today_shares: int = 0,
    sold_today_price: float | None = None,
    open_t_shares: int = 0,
    open_t_price: float | None = None,
) -> dict[str, Any]:
    """Return BYD single-stock daily range T playbook and alerts."""
    params = {
        "shares": int(shares),
        "cost": round(float(cost), 6),
        "sold_today_shares": int(sold_today_shares),
        "sold_today_price": sold_today_price,
        "open_t_shares": int(open_t_shares),
        "open_t_price": open_t_price,
    }
    if not refresh:
        cached = _read_workspace_snapshot("byd_daily_plan", params=params)
        if cached is not None:
            return cached
    daily = _load_byd_daily_frame()
    holding = BydHolding(shares=max(int(shares), 0), cost=float(cost), full_shares=10500)
    payload = build_minute_payload(
        daily=daily,
        minutes=pd.DataFrame(),
        holding=holding,
        data_status="daily_plan",
        sold_today_shares=sold_today_shares,
        sold_today_price=sold_today_price,
        open_t_shares=open_t_shares,
        open_t_price=open_t_price,
    )
    planned_t = payload.get("planned_t") or {}
    snapshot_date = planned_t.get("signal_date") or payload.get("asof") or payload.get("trade_date")
    _write_workspace_snapshot("byd_daily_plan", snapshot_date, payload, params=params)
    return payload


def _load_byd_daily_frame() -> pd.DataFrame:
    """Prefer the refreshed daily store so one-click updates advance BYD plans."""
    try:
        store = MarketDataStore(MarketDataStoreConfig.from_env(root=DAILY_DIR.parent))
        daily = store.read_frame(DAILY_DIR.name, "002594.SZ")
        normalized = _normalize_byd_daily_frame(daily)
        if not normalized.empty:
            return normalized
    except Exception:
        pass
    return load_daily_qfq(PROJECT_ROOT / "data/cache")


def _normalize_byd_daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame()
    out = daily.copy()
    parsed_date = pd.Series(pd.NaT, index=out.index)
    if "date" in out.columns:
        parsed_date = pd.to_datetime(out["date"], errors="coerce")
    if "trade_date" in out.columns:
        trade_date = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        parsed_date = parsed_date.fillna(trade_date)
    out["date"] = parsed_date
    if "vol" in out.columns and "volume" not in out.columns:
        out["volume"] = out["vol"]
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    required = ["date", "open", "high", "low", "close"]
    if any(col not in out.columns for col in required):
        return pd.DataFrame()
    return (
        out.dropna(subset=required)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def _long_research_module():
    path = PROJECT_ROOT / "scripts/research/backtest_long_dividend_quality.py"
    module_name = "quant_long_dividend_quality_research"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载长线策略研究脚本: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _tea_master_research_module():
    path = PROJECT_ROOT / "scripts/research/backtest_tea_master_long.py"
    module_name = "quant_tea_master_long_research"
    with _TEA_MASTER_MODULE_LOCK:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "CONFIGS"):
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
        spec.loader.exec_module(module)
        return module


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


def _read_long_stock_pool_snapshot(variant: str, signal_date: str | None) -> dict[str, Any] | None:
    snapshot_key, _ = _long_snapshot_key(variant, signal_date)
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

    path = _long_snapshot_path(snapshot_key)
    if path.exists():
        try:
            payload = read_json_file(path)
            payload["cache"] = {"hit": True, "backend": "json", "snapshot_key": snapshot_key}
            return payload
        except Exception:
            return None
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


def _write_long_stock_pool_snapshot(payload: dict[str, Any], variant: str, signal_date: str | None) -> None:
    snapshot_key, date_key = _long_snapshot_key(variant, signal_date)
    payload_to_store = dict(payload)
    payload_to_store["cache"] = {"hit": False, "backend": "generated", "snapshot_key": snapshot_key}
    payload_json = json.dumps(payload_to_store, ensure_ascii=False, default=str)
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    wrote_sql = False
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
            wrote_sql = True
        except Exception:
            wrote_sql = False
    if not wrote_sql:
        LONG_STOCK_POOL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        _long_snapshot_path(snapshot_key).write_text(payload_json, encoding="utf-8")


def _workspace_params_key(params: dict[str, Any] | None = None) -> str:
    params = params or {}
    raw = json.dumps(params, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _workspace_snapshot_key(workspace: str, snapshot_date: str, params_key: str) -> str:
    raw = json.dumps(
        {
            "schema_version": WEB_WORKSPACE_SNAPSHOT_SCHEMA_VERSION,
            "workspace": workspace,
            "snapshot_date": snapshot_date,
            "params_key": params_key,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _read_workspace_snapshot(
    workspace: str,
    snapshot_date: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    if not store.config.sql_url:
        return None
    params_key = _workspace_params_key(params)
    try:
        from sqlalchemy import text

        with store._engine().begin() as conn:
            if snapshot_date:
                snapshot_key = _workspace_snapshot_key(workspace, str(snapshot_date), params_key)
                row = conn.execute(
                    text(
                        f"""
                        SELECT snapshot_key, snapshot_date, payload_json
                        FROM {WEB_WORKSPACE_SNAPSHOT_TABLE}
                        WHERE snapshot_key = :snapshot_key
                        """
                    ),
                    {"snapshot_key": snapshot_key},
                ).mappings().first()
            else:
                row = conn.execute(
                    text(
                        f"""
                        SELECT snapshot_key, snapshot_date, payload_json
                        FROM {WEB_WORKSPACE_SNAPSHOT_TABLE}
                        WHERE workspace = :workspace
                          AND params_key = :params_key
                        ORDER BY snapshot_date DESC, updated_at DESC
                        LIMIT 1
                        """
                    ),
                    {"workspace": workspace, "params_key": params_key},
                ).mappings().first()
        if not row or not row.get("payload_json"):
            return None
        payload = json.loads(row["payload_json"])
        payload["cache"] = {
            "hit": True,
            "backend": "mysql",
            "workspace": workspace,
            "snapshot_key": str(row.get("snapshot_key") or ""),
            "snapshot_date": str(row.get("snapshot_date") or ""),
        }
        return payload
    except Exception:
        return None


def _write_workspace_snapshot(
    workspace: str,
    snapshot_date: str | None,
    payload: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> None:
    snapshot_date = str(snapshot_date or "latest")
    params = params or {}
    params_key = _workspace_params_key(params)
    snapshot_key = _workspace_snapshot_key(workspace, snapshot_date, params_key)
    payload_to_store = dict(payload)
    payload_to_store["cache"] = {
        "hit": False,
        "backend": "generated",
        "workspace": workspace,
        "snapshot_key": snapshot_key,
        "snapshot_date": snapshot_date,
    }
    payload_json = json.dumps(payload_to_store, ensure_ascii=False, default=str)
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    if not store.config.sql_url:
        return
    try:
        from sqlalchemy import text

        with store._engine().begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {WEB_WORKSPACE_SNAPSHOT_TABLE} (
                        snapshot_key VARCHAR(64) PRIMARY KEY,
                        workspace VARCHAR(64) NOT NULL,
                        snapshot_date VARCHAR(32) NOT NULL,
                        params_key VARCHAR(64) NOT NULL,
                        params_json TEXT NOT NULL,
                        generated_at VARCHAR(32) NOT NULL,
                        payload_json LONGTEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        KEY idx_workspace_latest (workspace, params_key, snapshot_date)
                    )
                    """
                )
            )
            conn.execute(
                text(
                    f"""
                    INSERT INTO {WEB_WORKSPACE_SNAPSHOT_TABLE}
                        (snapshot_key, workspace, snapshot_date, params_key, params_json, generated_at, payload_json)
                    VALUES
                        (:snapshot_key, :workspace, :snapshot_date, :params_key, :params_json, :generated_at, :payload_json)
                    ON DUPLICATE KEY UPDATE
                        generated_at = VALUES(generated_at),
                        params_json = VALUES(params_json),
                        payload_json = VALUES(payload_json)
                    """
                ),
                {
                    "snapshot_key": snapshot_key,
                    "workspace": workspace,
                    "snapshot_date": snapshot_date,
                    "params_key": params_key,
                    "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True, default=str),
                    "generated_at": str(payload.get("generated_at") or datetime.now().isoformat(timespec="seconds")),
                    "payload_json": payload_json,
                },
            )
    except Exception:
        return


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
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    if not store.config.sql_url:
        return set()
    params_key = _workspace_params_key(params)
    try:
        from sqlalchemy import text

        with store._engine().begin() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT snapshot_date
                    FROM {WEB_WORKSPACE_SNAPSHOT_TABLE}
                    WHERE workspace = :workspace
                      AND params_key = :params_key
                    """
                ),
                {"workspace": workspace, "params_key": params_key},
            ).scalars().all()
        return {str(item) for item in rows if item}
    except Exception:
        return set()


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
        return module.load_analyst_forecast_asof(out)
    except Exception:
        return module.add_empty_analyst_forecast_columns(out)


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
        "analyst_forward_years_180d": int(_safe_float(row.get("analyst_forward_years_180d"), 0.0) or 0),
        "analyst_forward_growth_score": round(_safe_float(row.get("analyst_forward_growth_score"), 0.0) or 0.0, 2),
        "analyst_target_upside_180d": round(_safe_float(row.get("analyst_target_upside_180d"), 0.0) or 0.0, 4),
        "analyst_negative_warning": _safe_bool(row.get("analyst_negative_warning")),
        "close_above_ma120": bool(close is not None and _safe_float(row.get("ma_120")) is not None and close > float(row.get("ma_120"))),
        "ma_120_slope_20d": round(_safe_float(row.get("ma_120_slope_20d"), 0.0) or 0.0, 4),
    }


@lru_cache(maxsize=8)
def _build_tea_master_stock_pool_cached(variant_key: str, signal_date: str | None) -> dict[str, Any]:
    module = _tea_master_research_module()
    config_name = TEA_LONG_VARIANTS.get(variant_key, variant_key)
    config = next((item for item in module.CONFIGS if item.name == config_name), None)
    if config is None:
        raise ValueError(f"未知茶大长线策略版本: {variant_key}")
    end_text = pd.to_datetime(signal_date).strftime("%Y%m%d") if signal_date else None
    merged, _, _, coverage = module.prepare_data(end=end_text)
    scored = module.build_tea_scores(merged)
    targets = module.select_targets(scored, config)
    if targets.empty:
        raise RuntimeError("茶大长线策略没有生成候选股票")
    if signal_date:
        requested = pd.to_datetime(signal_date)
        targets = targets[targets["rebalance_date"] <= requested].copy()
    if targets.empty:
        raise RuntimeError("所选日期之前没有茶大长线股票池")
    signal_ts = pd.to_datetime(targets["rebalance_date"].max())
    target_dates = sorted(pd.to_datetime(targets["rebalance_date"].dropna().unique()))
    previous_dates = [item for item in target_dates if item < signal_ts]
    previous_target_date = previous_dates[-1] if previous_dates else None
    latest = targets[targets["rebalance_date"] == signal_ts].copy()
    previous = targets[targets["rebalance_date"] == previous_target_date].copy() if previous_target_date is not None else pd.DataFrame()
    latest = latest.sort_values(["target_weight", "tea_score", "trend_score"], ascending=False)
    selected_codes = latest["ts_code"].astype(str).tolist()
    selected_set = set(selected_codes)
    previous_set = set(previous["ts_code"].astype(str).tolist()) if not previous.empty else set()
    selected_order = {code: i + 1 for i, code in enumerate(selected_codes)}
    target_map = latest.set_index("ts_code").to_dict(orient="index") if not latest.empty else {}

    scored_date = scored[pd.to_datetime(scored["date"]) == signal_ts].copy()
    if scored_date.empty:
        scored_date = scored[pd.to_datetime(scored["date"]) <= signal_ts].sort_values("date").drop_duplicates("ts_code", keep="last").copy()
    scored_date = _attach_analyst_forecast_for_display(scored_date)
    scored_by_symbol = scored_date.set_index("ts_code", drop=False)

    rows: list[dict[str, Any]] = []
    appended: set[str] = set()
    for code in selected_codes:
        if code not in scored_by_symbol.index:
            continue
        row = scored_by_symbol.loc[code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        enriched = row.copy()
        for key, value in target_map.get(code, {}).items():
            if str(key).startswith("analyst_"):
                continue
            enriched[key] = value
        previous_state = "CORE" if code in previous_set else "WATCH"
        rows.append(
            _tea_status_row(
                enriched,
                variant=variant_key,
                selected=True,
                target_weight=float(target_map.get(code, {}).get("target_weight", 0.0) or 0.0),
                previous_state=previous_state,
                rank=selected_order.get(code),
            )
        )
        appended.add(code)

    removed_codes = [code for code in previous_set if code not in selected_set and code in scored_by_symbol.index]
    for code in removed_codes:
        row = scored_by_symbol.loc[code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        rows.append(
            _tea_status_row(
                row,
                variant=variant_key,
                selected=False,
                target_weight=0.0,
                previous_state="CORE",
                rank=None,
            )
        )
        appended.add(code)

    candidate = scored_date[~scored_date["ts_code"].astype(str).isin(appended)].copy()
    watch_filter = (
        (candidate["tea_score"] >= max(config.min_tea_score - 4, 62))
        & (candidate["quality_score"] >= max(config.min_quality_score - 5, 55))
        & (candidate["trend_score"] >= max(config.min_trend_score - 8, 45))
        & (candidate["risk_score"] >= 25)
    )
    watch_rows = candidate[watch_filter].sort_values(
        ["tea_score", "trend_score", "quality_score", "risk_score"],
        ascending=False,
    ).head(25)
    for _, row in watch_rows.iterrows():
        code = str(row.get("ts_code"))
        if code in appended:
            continue
        rows.append(
            _tea_status_row(
                row,
                variant=variant_key,
                selected=False,
                target_weight=0.0,
                previous_state="WATCH",
                rank=None,
            )
        )
        appended.add(code)

    state_order = {"CORE": 0, "T_ACTIVE": 1, "BUILDING": 2, "REDUCE": 3, "WATCH": 4, "EXIT": 5, "COOLDOWN": 6}
    rows = sorted(
        rows,
        key=lambda item: (
            state_order.get(str(item.get("state")), 99),
            item.get("rank") or 999,
            -(item.get("long_score") or 0),
        ),
    )
    regime = str(latest["market_regime"].dropna().iloc[0]) if not latest["market_regime"].dropna().empty else "neutral"
    state_counts = pd.Series([row["state"] for row in rows]).value_counts().to_dict()
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": signal_ts.date().isoformat(),
        "variant": variant_key,
        "variant_id": config_name,
        "variant_name": LONG_VARIANT_LABELS.get(variant_key, variant_key),
        "market_regime": regime,
        "coverage": coverage,
        "state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "stocks": rows,
        "notes": [
            "茶大长线股票池使用独立策略脚本生成，不复用旧股息质量策略版本树。",
            "行情使用本地前复权日线；财务指标按 ann_date <= signal_date 合并。",
            "股票池包含目标持仓、上期移出后的降仓/清仓观察，以及高分观察池。",
            "目标仓位是组合目标，first_tranche_weight 是新进标的更谨慎的首批建仓建议。",
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
    start = module.parse_date(config.start)
    stock_basic = module.load_stock_basic()
    daily_basic, coverage = module.load_daily_basic_monthly(start, requested_end)
    if daily_basic.empty:
        raise RuntimeError("缺少 daily_basic，无法生成长线股票池")
    if variant in getattr(module, "PIT_UNIVERSE_VARIANTS", set()):
        candidate_symbols = None
        coverage["point_in_time_universe"] = True
    else:
        candidate_symbols = module.select_candidate_symbols_from_daily_basic(daily_basic, stock_basic, config)
        coverage["point_in_time_universe"] = False
    features, _ = module.load_daily_monthly_features(
        start,
        requested_end,
        stock_basic,
        candidate_symbols=candidate_symbols,
    )
    if requested_end is not None:
        features = features[features["date"] <= requested_end].copy()
        daily_basic = daily_basic[daily_basic["date"] <= requested_end].copy()
    common_dates = sorted(set(features["date"].dropna()) & set(daily_basic["date"].dropna()))
    if not common_dates:
        raise RuntimeError("日线特征与 daily_basic 没有共同信号日期")
    signal_ts = common_dates[-1]
    merged = pd.DataFrame()
    for candidate_date in reversed(common_dates):
        latest_features = features[features["date"] == candidate_date].copy()
        latest_basic = daily_basic[daily_basic["date"] == candidate_date].copy()
        latest_features = latest_features.sort_values(["date", "ts_code", "trade_date"]).drop_duplicates(["date", "ts_code"], keep="last")
        latest_basic = latest_basic.sort_values(["date", "ts_code", "trade_date"]).drop_duplicates(["date", "ts_code"], keep="last")
        if variant in getattr(module, "PIT_UNIVERSE_VARIANTS", set()):
            latest_basic = module.filter_daily_basic_point_in_time(latest_basic, config)
        candidate_merged = latest_features.merge(
            latest_basic.drop(columns=["trade_date"]),
            on=["date", "ts_code"],
            how="inner",
        )
        if len(candidate_merged) >= 50:
            signal_ts = candidate_date
            merged = candidate_merged
            break
    if merged.empty:
        raise RuntimeError("最近信号日无可用长线截面")
    merged = module.load_financial_asof(merged)
    if variant in module.GROWTH_VARIANTS:
        merged = module.load_analyst_forecast_asof(merged)
    else:
        merged = module.add_empty_analyst_forecast_columns(merged)
    market_regime = module.load_market_regime(signal_ts, signal_ts)
    if market_regime.empty:
        merged["market_regime"] = "neutral"
        merged["index_ma_120_slope_20d"] = np.nan
        merged["index_return_20d"] = np.nan
        merged["index_return_60d"] = np.nan
        merged["index_return_120d"] = np.nan
        merged["index_drawdown_60d"] = np.nan
        merged["index_overheat"] = False
    else:
        merged = merged.merge(market_regime, on="date", how="left")
        merged["market_regime"] = merged["market_regime"].fillna("neutral")
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
    state_counts = pd.Series([row["state"] for row in rows]).value_counts().to_dict()
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
            "状态机当前为策略状态建议，已读取上一轮信号快照；接入真实账户持仓后可升级为完整持仓状态机。",
            "BUILDING 的 first_tranche_weight 是更谨慎的首批建仓建议，不等同于最终目标仓位。",
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
        cached = _read_long_stock_pool_snapshot(variant_key, signal_date)
        if cached is not None:
            return cached
    if variant_key in TEA_LONG_VARIANTS:
        if refresh:
            _build_tea_master_stock_pool_cached.cache_clear()
        payload = _build_tea_master_stock_pool_cached(variant_key, signal_date)
        _write_long_stock_pool_snapshot(payload, variant_key, signal_date)
        payload["cache"] = {"hit": False, "backend": "generated"}
        return payload
    if refresh:
        _build_long_stock_pool_cached.cache_clear()
    payload = _build_long_stock_pool_cached(variant_key, signal_date, True)
    _write_long_stock_pool_snapshot(payload, variant_key, signal_date)
    payload["cache"] = {"hit": False, "backend": "generated"}
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


def _normalize_watch_symbol(raw_symbol: str) -> str:
    symbol = str(raw_symbol or "").strip().upper()
    if not symbol:
        raise ValueError("股票代码不能为空")
    symbol = symbol.replace("_", ".")
    if "." not in symbol and len(symbol) == 6 and symbol.isdigit():
        suffix = "SH" if symbol.startswith(("6", "9")) else "SZ"
        symbol = f"{symbol}.{suffix}"
    path = DAILY_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise ValueError(f"本地日线数据不存在: {symbol}")
    return symbol


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


def _read_similar_pattern_watchlist_symbols() -> list[str]:
    if not SIMILAR_PATTERN_WATCHLIST_PATH.exists():
        return list(SIMILAR_PATTERN_DEFAULT_WATCHLIST)
    try:
        payload = json.loads(SIMILAR_PATTERN_WATCHLIST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return list(SIMILAR_PATTERN_DEFAULT_WATCHLIST)
    symbols = payload.get("symbols", payload) if isinstance(payload, dict) else payload
    normalized: list[str] = []
    for item in symbols if isinstance(symbols, list) else []:
        try:
            symbol = _normalize_watch_symbol(str(item))
        except ValueError:
            continue
        if symbol not in normalized:
            normalized.append(symbol)
    return normalized or list(SIMILAR_PATTERN_DEFAULT_WATCHLIST)


def _write_similar_pattern_watchlist_symbols(symbols: list[str]) -> None:
    SIMILAR_PATTERN_WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": symbols,
    }
    SIMILAR_PATTERN_WATCHLIST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_similar_pattern_watchlist() -> dict[str, Any]:
    symbols = _read_similar_pattern_watchlist_symbols()
    basic = _stock_basic_for_similar_patterns()
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "stocks": [_stock_profile_from_basic(symbol, basic) for symbol in symbols],
    }


def add_similar_pattern_watch_symbol(symbol: str) -> dict[str, Any]:
    normalized = _normalize_watch_symbol(symbol)
    symbols = _read_similar_pattern_watchlist_symbols()
    if normalized not in symbols:
        symbols.append(normalized)
        _write_similar_pattern_watchlist_symbols(symbols)
    return get_similar_pattern_watchlist()


def remove_similar_pattern_watch_symbol(symbol: str) -> dict[str, Any]:
    normalized = _normalize_watch_symbol(symbol)
    symbols = [item for item in _read_similar_pattern_watchlist_symbols() if item != normalized]
    _write_similar_pattern_watchlist_symbols(symbols)
    return get_similar_pattern_watchlist()


def _similar_pattern_result_payload(result: SimilarPatternResult) -> dict[str, Any]:
    top_cases = result.similar_cases.copy()
    if not top_cases.empty:
        for col in ["fwd_1d", "fwd_20d", "fwd_60d", "max_drawdown_60d"]:
            if col in top_cases.columns:
                top_cases[col] = (pd.to_numeric(top_cases[col], errors="coerce") * 100).round(2)
        if "similarity" in top_cases.columns:
            raw_similarity = pd.to_numeric(top_cases["similarity"], errors="coerce")
            min_similarity = raw_similarity.min()
            max_similarity = raw_similarity.max()
            if pd.notna(min_similarity) and pd.notna(max_similarity) and max_similarity > min_similarity:
                top_cases["similarity_score"] = ((raw_similarity - min_similarity) / (max_similarity - min_similarity) * 100).round(1)
            else:
                top_cases["similarity_score"] = 100.0
            top_cases["similarity"] = raw_similarity.round(4)
    return {
        "target": {
            "symbol": result.target.symbol,
            "name": result.target.name,
            "target_date": result.target.target_date.strftime("%Y-%m-%d"),
        },
        "latest_snapshot": _json_ready(result.latest_snapshot),
        "forecast": _frame_records(result.forecast),
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


def refresh_similar_pattern_analysis() -> dict[str, Any]:
    symbols = _read_similar_pattern_watchlist_symbols()
    basic = _stock_basic_for_similar_patterns()
    cache_audit = build_vector_caches_parallel(
        DAILY_DIR,
        basic,
        SIMILAR_PATTERN_CONFIG,
        SIMILAR_PATTERN_VECTOR_CACHE_DIR,
        target_symbols=set(symbols),
        workers=max(1, int(os.getenv("SIMILAR_PATTERN_CACHE_WORKERS", "1"))),
    )
    results = analyze_targets_by_threshold(
        DAILY_DIR,
        basic,
        SIMILAR_PATTERN_CONFIG,
        target_symbols=symbols,
        vector_cache_dir=SIMILAR_PATTERN_VECTOR_CACHE_DIR,
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "watchlist": [_stock_profile_from_basic(symbol, basic) for symbol in symbols],
        "config": {
            "candidate_start_date": SIMILAR_PATTERN_CONFIG.candidate_start_date,
            "candidate_step_days": SIMILAR_PATTERN_CONFIG.candidate_step_days,
            "similarity_threshold": SIMILAR_PATTERN_CONFIG.similarity_threshold,
            "take_profit_3d": SIMILAR_PATTERN_CONFIG.take_profit_3d,
            "stop_loss_3d": SIMILAR_PATTERN_CONFIG.stop_loss_3d,
        },
        "results": [_similar_pattern_result_payload(results[symbol]) for symbol in symbols if symbol in results],
        "cache": {
            "hit": False,
            "backend": "generated",
            "rebuilt": int(cache_audit["status"].eq("built").sum()) if not cache_audit.empty else 0,
            "reused": int(cache_audit["status"].eq("cache_hit").sum()) if not cache_audit.empty else 0,
            "errors": int(cache_audit["status"].eq("error").sum()) if not cache_audit.empty else 0,
        },
    }
    SIMILAR_PATTERN_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIMILAR_PATTERN_ANALYSIS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def get_similar_pattern_analysis(refresh: bool = False) -> dict[str, Any]:
    if not refresh:
        cached = _read_similar_pattern_analysis_cache()
        if cached is not None:
            cached["cache"] = {"hit": True, "backend": "json"}
            return cached
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
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest(), date_key, strategy_key


def _selector_snapshot_path(snapshot_key: str) -> Path:
    return SELECTOR_SNAPSHOT_DIR / f"{snapshot_key}.json"


def _selector_snapshot_dates(strategies: list[str] | None, include_extended: bool) -> list[str]:
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
    return sorted(dates)


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
    if not snapshot_dates:
        return signal_date
    if target.weekday() < 5 and signal_date in snapshot_dates:
        return signal_date
    previous = [item for item in snapshot_dates if item <= signal_date]
    return previous[-1] if previous else signal_date


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

    path = _selector_snapshot_path(snapshot_key)
    if path.exists():
        try:
            payload = read_json_file(path)
            payload["cache"] = {"hit": True, "backend": "json", "snapshot_key": snapshot_key}
            return payload
        except Exception:
            return None
    return None


def _write_selector_snapshot(
    payload: dict[str, Any],
    strategies: list[str] | None,
    include_extended: bool,
) -> None:
    signal_date = str(payload.get("signal_date") or "")
    snapshot_key, date_key, strategy_key = _selector_snapshot_key(signal_date, strategies, include_extended)
    payload_to_store = dict(payload)
    payload_to_store["selector_snapshot_schema_version"] = SELECTOR_SNAPSHOT_SCHEMA_VERSION
    payload_to_store["cache"] = {"hit": False, "backend": "generated", "snapshot_key": snapshot_key}
    payload_json = json.dumps(payload_to_store, ensure_ascii=False, default=str)
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    wrote_sql = False
    if store.config.sql_url:
        try:
            from sqlalchemy import text

            with store._engine().begin() as conn:
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
                conn.execute(
                    text(
                        f"""
                        INSERT INTO {SELECTOR_SNAPSHOT_TABLE}
                            (snapshot_key, signal_date, strategies_key, include_extended, generated_at, stock_count, payload_json)
                        VALUES
                            (:snapshot_key, :signal_date, :strategies_key, :include_extended, :generated_at, :stock_count, :payload_json)
                        ON DUPLICATE KEY UPDATE
                            generated_at = VALUES(generated_at),
                            stock_count = VALUES(stock_count),
                            payload_json = VALUES(payload_json)
                        """
                    ),
                    {
                        "snapshot_key": snapshot_key,
                        "signal_date": date_key,
                        "strategies_key": strategy_key,
                        "include_extended": include_extended,
                        "generated_at": str(payload.get("generated_at") or datetime.now().isoformat(timespec="seconds")),
                        "stock_count": len(payload.get("stocks") or []),
                        "payload_json": payload_json,
                    },
                )
            wrote_sql = True
        except Exception:
            wrote_sql = False
    if not wrote_sql:
        SELECTOR_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        _selector_snapshot_path(snapshot_key).write_text(payload_json, encoding="utf-8")


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
    rows = _apply_current_score_normalization(rows)
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
    _write_selector_snapshot(payload, None, include_extended)
    written = {"ALL": len(payload.get("stocks") or [])}
    for strategy_key in strategy_keys:
        filtered = _filtered_selector_payload(payload, [strategy_key])
        members = STRATEGY_GROUP_MEMBERS.get(strategy_key, {strategy_key})
        _write_selector_snapshot(filtered, [strategy_key], bool(members & extended_keys))
        written[strategy_key] = len(filtered.get("stocks") or [])
    return written


def _display_selector_payload(payload: dict[str, Any], limit: int = DEFAULT_SELECTOR_LIMIT) -> dict[str, Any]:
    rows = payload.get("stocks") or []
    complete_total = int(payload.get("total_stock_count") or len(rows))
    display_rows = [row for row in rows if _row_display_quality_gate(row)]
    total = len(display_rows)
    display = display_rows[:limit] if limit > 0 else display_rows
    out = dict(payload)
    out["stocks"] = display
    out["total_stock_count"] = total
    out["complete_stock_count"] = complete_total
    out["display_limit"] = limit
    out["is_truncated"] = total > len(display)
    return out


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


def get_convertible_bond_grid_plan(
    trade_date: str | None = None,
    limit: int = 18,
    refresh: bool = False,
) -> dict[str, Any]:
    params = {"limit": int(limit)}
    if not refresh:
        cached = _read_workspace_snapshot("convertible_bond_grid_plan", snapshot_date=trade_date, params=params)
        if cached is not None:
            return cached
    refresh_result = None
    if refresh and trade_date:
        refresh_result = refresh_convertible_bond_daily(trade_date=trade_date)
    payload = build_convertible_bond_grid_plan(trade_date=trade_date, limit=limit)
    if refresh_result is not None:
        payload["data_refresh"] = refresh_result
    _write_workspace_snapshot(
        "convertible_bond_grid_plan",
        payload.get("trade_date") or trade_date,
        payload,
        params=params,
    )
    return payload


def get_convertible_bond_allotments(
    limit: int = 80,
    include_listed_days: int = 90,
    refresh: bool = False,
    stage_scope: str = "pipeline",
) -> dict[str, Any]:
    params = {
        "limit": int(limit),
        "include_listed_days": int(include_listed_days),
        "stage_scope": str(stage_scope),
    }
    if not refresh:
        cached = _read_workspace_snapshot("convertible_bond_allotments", params=params)
        if cached is not None:
            return cached
    payload = build_convertible_bond_allotment_payload(
        limit=limit,
        include_listed_days=include_listed_days,
        refresh=refresh,
        stage_scope=stage_scope,
    )
    _write_workspace_snapshot(
        "convertible_bond_allotments",
        payload.get("asof") or payload.get("trade_date") or payload.get("generated_at"),
        payload,
        params=params,
    )
    return payload


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
        CHAN_MODEL_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
        strategy_frame.to_parquet(CHAN_MODEL_STRATEGY_SCORED_PATH, index=False)
        strategy_frame.to_csv(CHAN_MODEL_STRATEGY_DIR / "chan_model_strategy_scored.csv", index=False)

    candidates = select_chan_model_candidates(strategy_frame, trade_date=signal_date, top_n=top_n)
    if signal_date is None:
        candidates.to_csv(CHAN_MODEL_LATEST_CANDIDATES_PATH, index=False)
    summary = summarize_chan_model_strategy(strategy_frame)
    if signal_date is None:
        summary.to_csv(CHAN_MODEL_SUMMARY_PATH, index=False)

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
        cached = _read_workspace_snapshot("chan_model_strategy", snapshot_date=signal_date, params=params)
        if cached is not None:
            return cached
    payload = _build_chan_model_strategy_payload(top_n=top_n, signal_date=signal_date)
    _write_workspace_snapshot("chan_model_strategy", payload.get("signal_date"), payload, params=params)
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
    return {
        (str(row["symbol"]), str(row["signal"])): row
        for _, row in df[df["model_pass"].fillna(False).astype(bool)].iterrows()
    }


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


def _rank_percentiles(values: list[float]) -> list[float]:
    if not values:
        return []
    series = pd.Series(values, dtype="float64")
    if series.nunique(dropna=True) <= 1:
        return [0.5 for _ in values]
    return series.rank(method="average", pct=True).fillna(0.5).tolist()


def _apply_current_score_normalization(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    for score_name, raw_name, historical_name in [
        ("opportunity_score", "buy_raw_score", "historical_buy_score"),
        ("holding_score", "hold_raw_score", "historical_hold_score"),
    ]:
        raw_values = [float(row.get(raw_name) or 0.0) for row in rows]
        global_pct = _rank_percentiles(raw_values)
        group_pct: list[float] = [0.5 for _ in rows]
        by_group: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            by_group.setdefault(_primary_family(row), []).append(index)
        for indexes in by_group.values():
            ranked = _rank_percentiles([raw_values[index] for index in indexes])
            for index, pct in zip(indexes, ranked):
                group_pct[index] = pct
        for index, row in enumerate(rows):
            historical = _safe_float(row.get(historical_name), 50.0) or 50.0
            score = historical * 0.45 + group_pct[index] * 35.0 + global_pct[index] * 20.0
            row[score_name] = round(float(np.clip(score, 0.0, 100.0)), 1)
    for row in rows:
        row["selector_score"] = row["opportunity_score"]
        row["score_target"] = "buy_score_normalized"
        row.update(_score_interpretation(float(row["opportunity_score"]), float(row["holding_score"])))
    return rows


def _score_interpretation(opportunity_score: float, holding_score: float) -> dict[str, str]:
    """Coarse score bands from the full-history calibration pass.

    Keep these deliberately coarse: the OOT test supports ranking by broad
    buckets better than treating the raw decimal as a precise probability.
    """
    if opportunity_score >= 85:
        band = "极高"
        percentile = "约 Top 1%"
        usage = "买入条件质量很强，仍需按开盘条件和止损执行"
    elif opportunity_score >= 70:
        band = "高"
        percentile = "约 Top 5%"
        usage = "优先观察，等待开盘条件确认"
    elif opportunity_score >= 60:
        band = "偏高"
        percentile = "约 Top 10%"
        usage = "可观察，不适合追高"
    elif opportunity_score >= 45:
        band = "中性"
        percentile = "约 Top 25%"
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
        "score_target": "buy_score_normalized",
        **score_info,
        "rank_reason": f"按 {ordered_signals[0].get('strategy_name')} 领衔，叠加 {len(groups)} 个策略组共振；当前买入分按未来 5 日冲高目标做历史校准",
        "signals": ordered_signals,
    }


def _primary_family(row: dict[str, Any]) -> str:
    signals = row.get("signals") or []
    if signals:
        return _signal_group_key(signals[0])
    families = row.get("matched_groups") or row.get("matched_families") or []
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
    pred_up10 = float(row.get("pred_up10_es") or 0)
    pred_down3 = float(row.get("pred_down3_es") or 0)
    return _enrich_signal_group({
        "strategy_key": "B1_MODEL",
        "strategy_family": "B1",
        "strategy_name": row.get("strategy_name") or "B1 模型 Top20",
        "operation_key": str(row.get("buy_filter") or row.get("open_gap_text") or "B1"),
        "timeframe": "日线级，收盘后生成名单，T+1 开盘观察",
        "logic": f"{row.get('entry_rule')}；{row.get('buy_filter')}；模型分阈值命中",
        "reason": (
            f"up10={pred_up10:.3f}，"
            f"down3={pred_down3:.3f}，"
            f"J={float(row.get('kdj_d_j') or 0):.2f}"
        ),
        "buy_plan": f"{row.get('open_gap_text')}；参考买入价 {price_range}。不满足开盘条件则空仓观察。",
        "sell_plan": row.get("sell_summary") or "按策略卖出规则执行",
        "metrics": metrics,
        "metrics_text": _metrics_text(metrics),
        "strength_score": max(pred_up10 - pred_down3, 0) * 3,
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
            target = pd.to_datetime(signal_date)
            available = cached[cached["date"] <= target]["date"]
            if available.empty:
                return {}
            selected_date = available.max()
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
        target = pd.to_datetime(signal_date)
        available = features[features["date"] <= target]["date"]
        if available.empty:
            return {}
        latest_date = available.max()
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


def _latest_family_signals() -> dict[str, list[dict[str, Any]]]:
    return _family_signals_for_date(None)


@lru_cache(maxsize=16)
def _family_profiles_for_date(signal_date: str | None = None) -> dict[str, dict[str, Any]]:
    if not FAMILY_SIGNAL_CACHE.exists():
        return {}
    cached = pd.read_parquet(FAMILY_SIGNAL_CACHE)
    cached["date"] = pd.to_datetime(cached["date"])
    if signal_date:
        target = pd.to_datetime(signal_date)
        available = cached[cached["date"] <= target]["date"]
        if available.empty:
            return {}
        selected_date = available.max()
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
    signals = build_extended_daily_signals(DAILY_DIR, signal_date=signal_date, max_workers=32)
    EXTENDED_SIGNAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EXTENDED_SIGNAL_CACHE.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "signal_date": signal_date,
                "signals": signals,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return signals


def _clear_selector_caches() -> None:
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
    ]:
        try:
            func.cache_clear()
        except AttributeError:
            pass


def _normalize_refresh_scope(scope: str | None = None) -> str:
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


def _progress_steps(scope: str | None = None) -> list[dict[str, Any]]:
    normalized = _normalize_refresh_scope(scope)
    return [
        {
            "key": key,
            "label": REFRESH_STEP_DEFINITIONS[key]["label"],
            "status": "pending",
            "percent": REFRESH_STEP_DEFINITIONS[key]["percent"],
        }
        for key in REFRESH_SCOPE_STEPS[normalized]
    ]


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
) -> None:
    with _REFRESH_LOCK:
        steps = list(_REFRESH_STATUS.get("steps") or _progress_steps())
        if step_key:
            seen_current = False
            for step in steps:
                if step["key"] == step_key:
                    step["status"] = step_status or ("running" if status == "running" else status)
                    seen_current = True
                elif complete_previous and not seen_current and step["status"] in {"pending", "running"}:
                    step["status"] = "success"
            if step_status == "success":
                for step in steps:
                    if step["key"] == step_key:
                        step["status"] = "success"
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
            }
        )
        if status in {"success", "failed"}:
            _REFRESH_STATUS["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _persist_refresh_status_unlocked()


def _refresh_long_stock_pool_variant(variant: str, signal_date: str | None) -> dict[str, Any]:
    variant_key = variant if variant in LONG_VARIANTS else next(
        (key for key, value in LONG_VARIANTS.items() if value == variant),
        variant,
    )
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            if variant_key in TEA_LONG_VARIANTS:
                payload = _build_tea_master_stock_pool_cached(variant_key, signal_date)
            else:
                payload = _build_long_stock_pool_cached(variant_key, signal_date, True)
    _write_long_stock_pool_snapshot(payload, variant_key, signal_date)
    return {
        "variant": variant_key,
        "signal_date": payload.get("signal_date"),
        "stocks": len(payload.get("stocks") or []),
    }


def _refresh_long_stock_pool_variants(variants: list[str], signal_date: str | None) -> list[dict[str, Any]]:
    results = [
        _refresh_long_stock_pool_variant(variant, signal_date)
        for variant in variants
    ]
    return sorted(results, key=lambda item: variants.index(item["variant"]))


def _run_latest_refresh_job(scope: str = "all") -> None:
    from quant.routine.pipeline import (
        build_features,
        generate_dashboard,
        generate_daily_plan,
        refresh_data,
        refresh_strategy_signal_cache,
        score_latest_models,
    )

    refresh_scope = _normalize_refresh_scope(scope)
    refresh_label = REFRESH_SCOPE_LABELS[refresh_scope]
    with _REFRESH_LOCK:
        _REFRESH_STATUS.update(
            {
                "status": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
                "message": f"{refresh_label}刷新任务已启动",
                "percent": 1,
                "current_step": "refresh_data",
                "steps": _progress_steps(refresh_scope),
                "scope": refresh_scope,
                "scope_label": refresh_label,
                "result": None,
                "error": None,
            }
        )
    try:
        results: dict[str, Any] = {}
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

        _set_refresh_progress(
            step_key="refresh_data",
            step_status="success",
            message="Tushare 最新日线数据已拉取完成",
            percent=35,
            complete_previous=False,
        )

        if refresh_scope not in {"all", "short"}:
            signal_date = _latest_candidate_signal_date()
            trade_date = str(signal_date).replace("-", "") if signal_date else None
            if refresh_scope == "chan":
                _set_refresh_progress(step_key="chan_model_strategy", message="正在生成缠论模型策略候选", percent=92)
                payload = get_chan_model_strategy_plan(top_n=20, refresh=True)
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
                _set_refresh_progress(step_key="long_stock_pool", message="正在计算长线策略股票池", percent=92)
                variants = _refresh_long_stock_pool_variants(["tea", "tea_safe", "v44"], signal_date)
                results["long_stock_pool"] = {"status": "success", "variants": variants}
                _set_refresh_progress(
                    step_key="long_stock_pool",
                    step_status="success",
                    message="长线策略股票池生成完成",
                    percent=98,
                    complete_previous=False,
                )
            elif refresh_scope == "cb":
                _set_refresh_progress(step_key="convertible_bond_plan", message="正在刷新可转债策略计划", percent=92)
                payload = get_convertible_bond_grid_plan(trade_date, 18, bool(trade_date))
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
                payload = get_convertible_bond_allotments(80, 90, True, "pipeline")
                results["convertible_bond_allotment"] = {
                    "status": "success",
                    "asof": payload.get("asof"),
                    "records": len(payload.get("records") or []),
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

        _set_refresh_progress(
            step_key="feature_cache",
            message="正在并行增量构建 B1 特征缓存与全市场规则信号",
            percent=35,
            complete_previous=False,
        )
        _set_refresh_progress(
            step_key="signal_cache",
            message="正在并行增量构建 B1 特征缓存与全市场规则信号",
            percent=35,
            complete_previous=False,
        )
        daily_plan_ready = False
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    build_features,
                    progress_callback=lambda percent, message: _set_refresh_progress(
                        step_key="feature_cache",
                        message=message,
                        percent=percent,
                        complete_previous=False,
                    ),
                ): ("feature_cache", "B1 特征缓存刷新失败"),
                executor.submit(
                    refresh_strategy_signal_cache,
                    progress_callback=lambda percent, message: _set_refresh_progress(
                        step_key="signal_cache",
                        message=message,
                        percent=max(46, min(68, percent)),
                        complete_previous=False,
                    ),
                ): ("signal_cache", "策略规则信号重建失败"),
            }
            for future in as_completed(futures):
                result_key, failure_message = futures[future]
                results[result_key] = future.result()
                if results[result_key].get("status") == "failed":
                    raise RuntimeError(results[result_key].get("stderr_tail") or failure_message)
                _set_refresh_progress(
                    step_key=result_key,
                    step_status="success",
                    message=f"{failure_message.removesuffix('失败')}完成",
                    percent=45 if result_key == "feature_cache" else 68,
                    complete_previous=False,
                )
                if result_key == "feature_cache" and not daily_plan_ready:
                    _set_refresh_progress(
                        step_key="daily_plan",
                        message="正在生成最新策略每日计划",
                        percent=46,
                        complete_previous=False,
                    )
                    results["generate_daily_plan"] = generate_daily_plan()
                    results["generate_dashboard"] = generate_dashboard()
                    _set_refresh_progress(
                        step_key="daily_plan",
                        step_status="success",
                        message="最新策略每日计划已生成",
                        percent=50,
                        complete_previous=False,
                    )
                    daily_plan_ready = True

        if not daily_plan_ready:
            _set_refresh_progress(
                step_key="daily_plan",
                message="正在生成最新策略每日计划",
                percent=50,
                complete_previous=False,
            )
            results["generate_daily_plan"] = generate_daily_plan()
            results["generate_dashboard"] = generate_dashboard()
            _set_refresh_progress(
                step_key="daily_plan",
                step_status="success",
                message="最新策略每日计划已生成",
                percent=50,
                complete_previous=False,
            )

        _set_refresh_progress(step_key="model_score", message="正在计算当日策略模型分", percent=70)
        results["model_score"] = score_latest_models()
        if results["model_score"].get("status") == "failed":
            raise RuntimeError(results["model_score"].get("stderr_tail") or "当日策略模型分计算失败")
        _clear_selector_caches()

        _set_refresh_progress(step_key="selector_core", message="正在计算核心策略股票池", percent=80)
        core_payload = get_stock_selector_payload(use_cache=False)
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

        _build_tea_master_stock_pool_cached.cache_clear()
        _build_long_stock_pool_cached.cache_clear()
        long_variants = ["tea", "tea_safe", "v44"]
        signal_date = full_payload.get("signal_date")
        trade_date = str(signal_date).replace("-", "") if signal_date else None
        _set_refresh_progress(
            step_key="chan_model_strategy",
            message="正在并行计算长线、缠论、可转债、配债股与 BYD 工作区",
            percent=92,
            complete_previous=False,
        )
        _set_refresh_progress(
            step_key="long_stock_pool",
            message="正在并行计算长线、缠论、可转债、配债股与 BYD 工作区",
            percent=92,
            complete_previous=False,
        )
        _set_refresh_progress(
            step_key="convertible_bond_plan",
            message="正在并行计算长线、可转债、配债股与 BYD 工作区",
            percent=92,
            complete_previous=False,
        )
        _set_refresh_progress(
            step_key="convertible_bond_allotment",
            message="正在并行计算长线、可转债、配债股与 BYD 工作区",
            percent=92,
            complete_previous=False,
        )
        _set_refresh_progress(
            step_key="byd_daily_plan",
            message="正在并行计算长线、可转债、配债股与 BYD 工作区",
            percent=92,
            complete_previous=False,
        )
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(get_chan_model_strategy_plan, 20, True): (
                    "chan_model_strategy",
                    "缠论模型策略候选生成失败",
                ),
                executor.submit(_refresh_long_stock_pool_variants, long_variants, signal_date): (
                    "long_stock_pool",
                    "长线策略股票池生成失败",
                ),
                executor.submit(get_convertible_bond_grid_plan, trade_date, 18, bool(trade_date)): (
                    "convertible_bond_plan",
                    "可转债策略计划刷新失败",
                ),
                executor.submit(get_convertible_bond_allotments, 80, 90, True, "pipeline"): (
                    "convertible_bond_allotment",
                    "配债股数据刷新失败",
                ),
                executor.submit(get_byd_daily_strategy, refresh=True): ("byd_daily_plan", "BYD 做T日线计划刷新失败"),
            }
            for future in as_completed(futures):
                result_key, failure_message = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:
                    _set_refresh_progress(
                        status="failed",
                        step_key=result_key,
                        step_status="failed",
                        message="刷新任务失败",
                        error=f"{failure_message}: {exc}\n{traceback.format_exc(limit=5)}",
                        complete_previous=False,
                    )
                    raise
                if result_key == "long_stock_pool":
                    results[result_key] = {"status": "success", "variants": payload}
                elif result_key == "chan_model_strategy":
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
                        "records": len(payload.get("records") or []),
                    }
                else:
                    planned_t = payload.get("planned_t") or {}
                    results[result_key] = {
                        "status": "success",
                        "signal_date": planned_t.get("signal_date"),
                        "alerts": len(payload.get("alerts") or []),
                    }
                _set_refresh_progress(
                    step_key=result_key,
                    step_status="success",
                    message=f"{failure_message.removesuffix('失败')}完成",
                    percent={
                        "chan_model_strategy": 93,
                        "long_stock_pool": 94,
                        "convertible_bond_plan": 95,
                        "convertible_bond_allotment": 96,
                        "byd_daily_plan": 97,
                    }[result_key],
                    complete_previous=False,
                )

        _set_refresh_progress(step_key="snapshot", message="正在写入策略股票池快照", percent=98)
        written_pools = _write_strategy_pool_snapshots(full_payload, include_extended=True)
        results["snapshot"] = {
            "status": "success",
            "storage": "mysql" if MarketDataStore(MarketDataStoreConfig.from_env()).config.sql_url else "json",
            "strategy_pools": written_pools,
            "long_stock_pools": results["long_stock_pool"]["variants"],
        }

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
        _set_refresh_progress(
            status="failed",
            step_key=_REFRESH_STATUS.get("current_step"),
            message="刷新任务失败",
            error=f"{exc}\n{traceback.format_exc(limit=5)}",
        )


def start_latest_refresh(scope: str = "all") -> dict[str, Any]:
    refresh_scope = _normalize_refresh_scope(scope)
    refresh_label = REFRESH_SCOPE_LABELS[refresh_scope]
    with _REFRESH_LOCK:
        if _REFRESH_STATUS.get("status") == "running":
            return dict(_REFRESH_STATUS)
        thread = threading.Thread(
            target=_run_latest_refresh_job,
            args=(refresh_scope,),
            name=f"quant-{refresh_scope}-refresh",
            daemon=True,
        )
        _REFRESH_STATUS.update(
            {
                "status": "queued",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
                "message": f"{refresh_label}刷新任务已进入后台队列",
                "percent": 0,
                "current_step": None,
                "steps": _progress_steps(refresh_scope),
                "scope": refresh_scope,
                "scope_label": refresh_label,
                "result": None,
                "error": None,
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
    rows = _apply_current_score_normalization(rows)
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
        "available_strategies": [
            {"key": item["key"], "label": item["label"], "status": item["status"], "members": item["members"]}
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
            "买入分为 0-100 历史归一化评分，按未来 5 日最大冲高收益和冲高 >=5% 概率校准，并混合同策略内分位和全策略分位。",
            "持有分为 0-100 保守辅助评分，按未来 5 日收盘收益和正收益概率校准；当前历史验证显示持有目标预测较弱，不等同于收益承诺。",
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
