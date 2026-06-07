from __future__ import annotations

import json
import re
import threading
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
from quant.routine.b1_daily_plan import DAILY_PLAN_PATH, FEATURE_PATH, build_daily_plan, write_daily_plan
from quant.routine.dashboard import DASHBOARD_PATH, build_dashboard_payload, write_dashboard_json
from quant.routine.paths import DAILY_DIR, PROJECT_ROOT, WEB_DATA_DIR
from quant.strategies.custom.b1_family import add_b1_family_signals
from quant.strategies.custom.z_skill_patterns import Z_SKILL_STRATEGIES, build_z_skill_daily_signals, _stock_basic_map


REPORT_DIR = PROJECT_ROOT / "reports/b1/research/xgb_project_vars_strategy"
FAMILY_SIGNAL_CACHE = PROJECT_ROOT / "data/features/b1/b1_family_rule_candidates.parquet"
Z_SKILL_SIGNAL_CACHE = PROJECT_ROOT / "data/features/z_skill_daily_signals.json"
Z_SKILL_PLAYBOOK = REPORT_DIR / "latest_z_skill_operational_playbook.csv"
Z_SKILL_MODEL_PLAYBOOK = REPORT_DIR / "latest_z_skill_model_operational_playbook.csv"
Z_SKILL_MODEL_SCORED = REPORT_DIR / "latest_z_skill_model_scored_candidates.parquet"
Z_SKILL_MODEL_SUMMARY = REPORT_DIR / "latest_z_skill_model_entry_exit_backtest.csv"
FAMILY_RULE_PATTERN = "b1_family_rule_backtest_*.csv"
FUSION_PATTERN = "b1_model_zettaranc_fusion_*.csv"
MODEL_FILTERED_SIGNALS = {"B2", "BREATHING", "NANA", "YIDONG_DILIAN", "KEY_K", "GOLDEN_BOWL"}
MODEL_SIGNAL_LABELS = {
    "B2": "B2 模型分",
    "BREATHING": "呼吸结构",
    "NANA": "娜娜图形",
    "YIDONG_DILIAN": "异动地量",
    "KEY_K": "关键K",
    "GOLDEN_BOWL": "黄金碗",
}
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
}
DEFAULT_ACTION_LEVELS = {"可小仓实操", "谨慎实操"}
DEFAULT_FAMILIES = {"B1", "B2", "B3", "SB1", "SUPER_B1", "YIDONG_DILIAN", "GOLDEN_BOWL", "KENGQI", "PINGHANG"}
DEFAULT_SELECTOR_LIMIT = 50
_REFRESH_LOCK = threading.Lock()
_REFRESH_STATUS: dict[str, Any] = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "message": "尚未启动刷新任务",
    "result": None,
    "error": None,
}


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


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


def refresh_dashboard() -> dict[str, Any]:
    output_path = write_dashboard_json()
    return read_json_file(output_path)


def latest_report_file(pattern: str) -> Path | None:
    candidates = sorted(REPORT_DIR.glob(pattern))
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
        f"PF {float(metrics.get('profit_factor') or 0):.2f}"
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
def _z_skill_playbooks() -> dict[str, pd.Series]:
    if not Z_SKILL_PLAYBOOK.exists():
        return {}
    df = pd.read_csv(Z_SKILL_PLAYBOOK)
    if "signal" not in df.columns:
        return {}
    return {str(row["signal"]): row for _, row in df.iterrows()}


@lru_cache(maxsize=1)
def _z_skill_model_playbooks() -> dict[str, pd.Series]:
    if not Z_SKILL_MODEL_PLAYBOOK.exists():
        return {}
    df = pd.read_csv(Z_SKILL_MODEL_PLAYBOOK)
    if "signal" not in df.columns:
        return {}
    return {str(row["signal"]): row for _, row in df.iterrows()}


@lru_cache(maxsize=1)
def _z_skill_model_risk_managed_playbooks() -> dict[str, pd.Series]:
    if not Z_SKILL_MODEL_SUMMARY.exists():
        return {}
    df = pd.read_csv(Z_SKILL_MODEL_SUMMARY)
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
    playbook = _z_skill_model_playbooks().get(signal_key)
    if playbook is not None and str(playbook.get("exit_rule", "")).startswith("expiry"):
        risk_managed = _z_skill_model_risk_managed_playbooks().get(signal_key)
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
    if not Z_SKILL_MODEL_SCORED.exists():
        return {}
    df = pd.read_parquet(Z_SKILL_MODEL_SCORED)
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


def _z_skill_signal_payload(signal: dict[str, Any], model_score: pd.Series | None = None) -> dict[str, Any]:
    strategy_key = str(signal.get("strategy_key"))
    model_playbook = _model_playbook_for(strategy_key)
    playbook = model_playbook if model_playbook is not None else _z_skill_playbooks().get(strategy_key)
    if playbook is None:
        return signal
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
    threshold_text = f"模型买入条件：{_entry_rule_text(entry_rule)}；" if entry_rule and model_playbook is not None else ""
    enriched["buy_plan"] = f"{threshold_text}开盘执行条件：{open_filter}；实操分层：{action_level}。符合信号后 T+1 开盘执行，不满足则空仓。"
    enriched["sell_plan"] = f"{_exit_rule_text(exit_rule)}。依据：z-skill {source}买卖组合回测 playbook。"
    enriched["logic"] = f"{signal.get('logic')}（已完成 z-skill {source}买卖组合回测，当前结论：{action_level}）"
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
    return enriched


def _model_filtered_signal_payload(signal_key: str, model_score: pd.Series) -> dict[str, Any]:
    playbook = _model_playbook_for(signal_key)
    metrics = _metrics_payload(playbook) if playbook is not None else None
    action_level = str(playbook.get("action_level") or "模型观察") if playbook is not None else "模型观察"
    entry_rule = str(playbook.get("entry_rule") or "") if playbook is not None else ""
    open_filter = str(playbook.get("open_filter_description") or "T+1 开盘观察") if playbook is not None else "T+1 开盘观察"
    exit_rule = str(playbook.get("exit_rule") or "按策略卖出") if playbook is not None else "按策略卖出"
    model_reason = _model_score_reason(model_score)
    label = MODEL_SIGNAL_LABELS.get(signal_key, signal_key)
    return {
        "strategy_key": signal_key,
        "strategy_family": signal_key,
        "strategy_name": label,
        "timeframe": "日线级，收盘确认，T+1 开盘观察",
        "logic": f"{label} 规则候选，并通过该策略独立 XGBoost 模型分过滤。",
        "reason": f"模型分 {model_reason}",
        "buy_plan": f"模型买入条件：{_entry_rule_text(entry_rule)}；开盘执行条件：{open_filter}；实操分层：{action_level}。符合信号后 T+1 开盘执行，不满足则空仓。",
        "sell_plan": f"{_exit_rule_text(exit_rule)}。依据：z-skill 模型版买卖组合回测 playbook。",
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
    }


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
    if family == "B1":
        return True

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

    if family in DEFAULT_FAMILIES and action_level in DEFAULT_ACTION_LEVELS:
        return (
            trades >= 80
            and avg_return >= 0.8
            and profit_factor >= 1.5
            and max_drawdown >= -25
        )

    return False


def _signal_selector_score(signal: dict[str, Any]) -> float:
    """Rank a signal with OOT performance, sample reliability and current strength."""
    metrics = signal.get("metrics") or {}
    trades = float(metrics.get("trades") or 0)
    avg_return = float(metrics.get("avg_return_pct") or 0)
    win_rate = float(metrics.get("win_rate") or 0)
    max_drawdown = abs(float(metrics.get("max_drawdown_pct") or 0))
    profit_factor = float(metrics.get("profit_factor") or 0)
    reliability = min(1.0, np.sqrt(max(trades, 0) / 80))
    capped_pf = min(profit_factor, 4.0)
    capped_avg = max(min(avg_return, 8.0), -8.0)
    drawdown_penalty = min(max_drawdown, 40.0) / 20
    historical_score = capped_avg * 0.35 + capped_pf * 1.2 + win_rate * 2.5 - drawdown_penalty
    strength_score = float(signal.get("strength_score") or 0)
    return historical_score * (0.45 + 0.55 * reliability) + strength_score


def _signal_operation_key(signal: dict[str, Any]) -> str:
    family = str(signal.get("strategy_family") or signal.get("strategy_key") or "")
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
    return {
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
    }


def _family_signal_payload(strategy_key: str, latest_row: pd.Series, model_score: pd.Series | None = None) -> dict[str, Any]:
    signal_column, family, name, logic = FAMILY_SIGNAL_COLUMNS[strategy_key]
    is_intraday_approx = family in {"SB1", "SUPER_B1"}
    model_playbook = _model_playbook_for(family) if family in MODEL_FILTERED_SIGNALS else None
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
    return {
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
        "buy_plan": f"{model_entry}开盘执行条件：{buy}。{('当前页面只给出日线观察名单，正式买点需要分钟级确认。' if is_intraday_approx else '不满足开盘条件则空仓观察。')}",
        "sell_plan": sell,
        "metrics": metrics,
        "metrics_text": _metrics_text(metrics),
        "action_level": action_level,
        "playbook_source": "模型版" if model_playbook is not None else "规则版",
        "strength_score": strength_score,
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
                    if family in MODEL_FILTERED_SIGNALS and model_score is None:
                        continue
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
                if family in MODEL_FILTERED_SIGNALS and model_score is None:
                    continue
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
def _latest_z_skill_signals(signal_date: str | None) -> dict[str, dict[str, Any]]:
    if Z_SKILL_SIGNAL_CACHE.exists():
        try:
            cached = read_json_file(Z_SKILL_SIGNAL_CACHE)
            if cached.get("signal_date") == signal_date and isinstance(cached.get("signals"), dict):
                return cached["signals"]
        except Exception:
            pass
    signals = build_z_skill_daily_signals(DAILY_DIR, signal_date=signal_date, max_workers=32)
    Z_SKILL_SIGNAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    Z_SKILL_SIGNAL_CACHE.write_text(
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
        _z_skill_playbooks,
        _z_skill_model_playbooks,
        _z_skill_model_risk_managed_playbooks,
        _model_scored_candidates_for_date,
        _daily_profile_at_or_before,
        _family_best_metrics,
        _family_best_metrics_by_signal,
        _family_risk_managed_metrics_by_signal,
        _family_signals_for_date,
        _family_profiles_for_date,
        _latest_z_skill_signals,
    ]:
        try:
            func.cache_clear()
        except AttributeError:
            pass


def _run_latest_refresh_job() -> None:
    from quant.routine.pipeline import run_daily_pipeline

    with _REFRESH_LOCK:
        _REFRESH_STATUS.update(
            {
                "status": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
                "message": "正在拉取 Tushare 最新日线数据，并重算每日股票池",
                "result": None,
                "error": None,
            }
        )
    try:
        result = run_daily_pipeline(skip_data=False, skip_backtest=True)
        _clear_selector_caches()
        with _REFRESH_LOCK:
            _REFRESH_STATUS.update(
                {
                    "status": "success",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "message": "刷新任务完成，页面可重新加载最新股票池",
                    "result": result,
                    "error": None,
                }
            )
    except Exception as exc:
        with _REFRESH_LOCK:
            _REFRESH_STATUS.update(
                {
                    "status": "failed",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "message": "刷新任务失败",
                    "result": None,
                    "error": f"{exc}\n{traceback.format_exc(limit=5)}",
                }
            )


def start_latest_refresh() -> dict[str, Any]:
    with _REFRESH_LOCK:
        if _REFRESH_STATUS.get("status") == "running":
            return dict(_REFRESH_STATUS)
        thread = threading.Thread(target=_run_latest_refresh_job, name="quant-latest-refresh", daemon=True)
        _REFRESH_STATUS.update(
            {
                "status": "queued",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
                "message": "刷新任务已进入后台队列",
                "result": None,
                "error": None,
            }
        )
        thread.start()
        return dict(_REFRESH_STATUS)


def get_latest_refresh_status() -> dict[str, Any]:
    with _REFRESH_LOCK:
        return dict(_REFRESH_STATUS)


def _selected_z_skill_keys(strategies: list[str] | None) -> set[str]:
    selected = {item.upper() for item in strategies or [] if item}
    known = {str(item.get("key", "")).upper() for item in Z_SKILL_STRATEGIES}
    return selected & known


def get_stock_selector_payload(
    strategies: list[str] | None = None,
    signal_date: str | None = None,
    include_z_skill: bool = False,
) -> dict[str, Any]:
    plan = get_b1_plan(signal_date=signal_date)
    effective_signal_date = plan.get("signal_date") or signal_date
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
    z_skill_filter = _selected_z_skill_keys(strategies)
    should_load_z_skill = include_z_skill or bool(z_skill_filter)
    if should_load_z_skill:
        for symbol, payload in _latest_z_skill_signals(effective_signal_date).items():
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
                if z_skill_filter and strategy_key.upper() not in z_skill_filter:
                    continue
                model_score = model_scored.get((symbol, strategy_key))
                if strategy_key in MODEL_FILTERED_SIGNALS and model_score is None:
                    continue
                enriched_signals.append(_z_skill_signal_payload(signal, model_score=model_score))
            stocks[symbol]["signals"].extend(enriched_signals)

    for (symbol, signal_key), model_score in model_scored.items():
        if signal_key not in MODEL_FILTERED_SIGNALS:
            continue
        existing = stocks.get(symbol, {}).get("signals", [])
        if any(signal.get("strategy_family") == signal_key for signal in existing):
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
    rows = []
    for stock in stocks.values():
        if selected:
            signals = [
                signal
                for signal in stock["signals"]
                if signal["strategy_key"].upper() in selected or signal["strategy_family"].upper() in selected
            ]
        else:
            signals = stock["signals"]
            signals = [signal for signal in signals if _signal_quality_gate(signal)]
        if not signals:
            continue
        signals = _dedupe_signals_by_operation(signals)
        if not signals:
            continue
        _fill_stock_profile(stock, effective_signal_date)
        families = sorted({signal["strategy_family"] for signal in signals})
        best_pf = max((signal.get("metrics") or {}).get("profit_factor") or 0 for signal in signals)
        best_avg = max((signal.get("metrics") or {}).get("avg_return_pct") or -999 for signal in signals)
        ordered_signals = sorted(
            signals,
            key=lambda item: (_signal_selector_score(item), (item.get("metrics") or {}).get("profit_factor") or 0),
            reverse=True,
        )
        signal_scores = [_signal_selector_score(signal) for signal in ordered_signals]
        selector_score = (
            (signal_scores[0] if signal_scores else 0)
            + 0.12 * sum(signal_scores[:3])
            + 0.35 * np.log1p(len(ordered_signals))
        )
        rows.append(
            {
                **{key: value for key, value in stock.items() if key != "signals"},
                "matched_count": len(families),
                "matched_families": families,
                "matched_strategy_names": [signal["strategy_name"] for signal in signals],
                "best_profit_factor": best_pf,
                "best_avg_return_pct": best_avg if best_avg > -999 else None,
                "selector_score": round(float(selector_score), 2),
                "rank_reason": f"按 {ordered_signals[0]['strategy_name']} 领衔，叠加 {len(families)} 个策略家族共振",
                "signals": ordered_signals,
            }
        )
    rows = sorted(
        rows,
        key=lambda item: (item["selector_score"], item["matched_count"], item["best_profit_factor"]),
        reverse=True,
    )
    if not selected:
        rows = rows[:DEFAULT_SELECTOR_LIMIT]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": plan.get("signal_date"),
        "execution_date": plan.get("execution_date"),
        "available_strategies": [
            {"key": "B1", "label": "B1", "status": "模型+规则"},
            {"key": "B2", "label": "B2", "status": "日线规则已接入"},
            {"key": "B3", "label": "B3", "status": "日线规则已接入"},
            {"key": "SB1", "label": "SB1", "status": "盘中战法日线近似"},
            {"key": "SUPER_B1", "label": "超级 B1", "status": "盘中战法日线近似"},
            *Z_SKILL_STRATEGIES,
        ],
        "stocks": rows,
        "notes": [
            "选股器按股票聚合命中策略家族；同一策略家族下相同买入操作只保留综合效果最优的一版，不同买入操作会同时展示，命中仍按策略家族去重计算。",
            "B1 已合并模型分和规则信号；B2/B3 当前使用全市场规则候选缓存，不再受 B1 模型候选池限制。",
            "历史均值是该股票命中策略在 OOT 回测中的平均单笔收益；PF 是总盈利除以总亏损，越高说明盈亏结构越好。",
            "股票池默认按综合分排序：综合考虑历史均值、胜率、PF、最大回撤、样本量可靠性、当前信号强度和多策略共振。",
            f"默认首页只展示实操候选 Top{DEFAULT_SELECTOR_LIMIT}；点击左侧具体策略时展示该策略完整候选，便于继续观察和复盘。",
            "为保证首屏速度，默认首页先加载 B1/B2/B3/模型候选；z-skill 高频战法按策略筛选时再生成/读取。",
            "SB1 和超级B1 本质偏盘中/尾盘战法，正式交易前仍需要分钟级数据确认买点。",
            "z-skill 高频战法已完成模型版买点评估；异动地量、黄金碗当前标记为可小仓实操，呼吸结构谨慎实操，关键K和灾后重建先模型观察。",
        ],
    }
