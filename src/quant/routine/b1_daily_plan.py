from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from quant.routine.paths import PROJECT_ROOT, ROUTINE_DIR, WEB_DATA_DIR


FEATURE_PATH = PROJECT_ROOT / "data/features/b1/training_xgb_project_vars.parquet"
MODEL_DIR = PROJECT_ROOT / "models/research/b1_xgb_project_vars"
FUSION_DIR = PROJECT_ROOT / "reports/b1/research/xgb_project_vars_strategy"
DAILY_PLAN_PATH = WEB_DATA_DIR / "b1_daily_plan.json"
MODEL_NAMES = ["up5_es", "up8_es", "up10_es", "down2_es", "down3_es"]


@dataclass(frozen=True)
class EntryThresholds:
    min_up5: float | None = None
    min_up8: float | None = None
    min_up10: float | None = None
    max_down2: float | None = None
    max_down3: float | None = None


@dataclass(frozen=True)
class OpenGapRule:
    min_gap_pct: float | None = None
    max_gap_pct: float | None = None
    require_j10: bool = False


def latest_fusion_summary_path() -> Path:
    candidates = sorted(FUSION_DIR.glob("b1_model_zettaranc_fusion_*.csv"))
    if not candidates:
        raise FileNotFoundError("未找到 b1_model_zettaranc_fusion_*.csv，请先运行融合回测")
    return candidates[-1]


def parse_entry_rule(name: str) -> EntryThresholds:
    def value(pattern: str) -> float | None:
        match = re.search(pattern, name)
        return float(match.group(1)) if match else None

    return EntryThresholds(
        min_up5=value(r"up5_ge_(\d+\.\d+)"),
        min_up8=value(r"up8_ge_(\d+\.\d+)"),
        min_up10=value(r"up10_ge_(\d+\.\d+)"),
        max_down2=value(r"down2_le_(\d+\.\d+)"),
        max_down3=value(r"down3_le_(\d+\.\d+)"),
    )


def parse_open_gap_rule(buy_filter: str) -> OpenGapRule:
    if buy_filter == "model_only":
        return OpenGapRule()
    require_j10 = "j10" in buy_filter
    if buy_filter == "gap_0_to_2_j10":
        return OpenGapRule(0.0, 2.0, require_j10)
    if buy_filter == "gap_le_1_j10":
        return OpenGapRule(0.0, 1.0, require_j10)
    if buy_filter in {"gap_j10", "gap_j0", "gap_up"}:
        return OpenGapRule(0.0, None, require_j10)
    return OpenGapRule(0.0, None, require_j10)


def open_gap_text(rule: OpenGapRule) -> str:
    if rule.min_gap_pct is None and rule.max_gap_pct is None:
        return "不限制 T+1 开盘涨跌幅"
    if rule.min_gap_pct is not None and rule.max_gap_pct is not None:
        return f"{rule.min_gap_pct:.0f}% <= T+1 开盘涨幅 <= {rule.max_gap_pct:.0f}%"
    if rule.min_gap_pct is not None:
        return f"T+1 开盘涨幅 >= {rule.min_gap_pct:.0f}%"
    return f"T+1 开盘涨幅 <= {rule.max_gap_pct:.0f}%"


def sell_plan(exit_mode: str) -> dict[str, Any]:
    fixed = re.match(r"fixed_tp(\d+)_sl(\d+)_T(\d+)", exit_mode)
    if fixed:
        tp, sl, hold = fixed.groups()
        stop_pct = int(sl) / 10 if len(sl) == 2 else int(sl)
        return {
            "type": "fixed",
            "summary": f"止盈 {tp}%；盘中硬止损 {stop_pct:g}%；最长持有到 T+{hold}",
            "take_profit_pct": float(tp),
            "stop_loss_pct": float(stop_pct),
            "max_hold_days": int(hold),
            "details": "盘中跌破止损价立即按可成交价退出；若 T+2 或之后跳空低开低于止损价，按实际开盘价计算退出。",
        }
    if exit_mode == "structure_time":
        return {
            "type": "structure_time",
            "summary": "结构止损；3天不涨退出；最长持有到 T+7",
            "details": "该版本历史胜率高，但日线结构退出存在跳空滞后风险，当前只作为观察策略。",
        }
    if exit_mode == "structure_stop":
        return {
            "type": "structure_stop",
            "summary": "结构位收盘跌破退出；最长持有到 T+7",
            "details": "结构位参考信号日低点与 T+1 低点；该版本不替代盘中硬止损。",
        }
    if exit_mode == "sell_score":
        return {
            "type": "sell_score",
            "summary": "持股评分走弱退出；最长持有到 T+7",
            "details": "评分退出用于观察卖飞与破位风险，当前不作为主策略。",
        }
    return {"type": exit_mode, "summary": exit_mode, "details": ""}


def load_fusion_top20(path: Path | None = None) -> pd.DataFrame:
    source = path or latest_fusion_summary_path()
    summary = pd.read_csv(source)
    oot = summary[(summary["split"] == "oot") & (summary["trades"] >= 50)].copy()
    scores = _stable_score(summary)
    oot["selection_score"] = oot["combo"].map(scores)
    top = oot.sort_values(["selection_score", "profit_factor", "avg_return_pct"], ascending=[False, False, False]).head(20)
    top = top.reset_index(drop=True)
    top["priority"] = np.arange(1, len(top) + 1)
    top["source_path"] = str(source)
    return top


def _stable_score(summary: pd.DataFrame) -> pd.Series:
    pivot = summary.pivot_table(
        index="combo",
        columns="split",
        values=["avg_return_pct", "profit_factor", "max_drawdown_pct", "trades"],
        aggfunc="first",
    )
    rows = []
    for combo in pivot.index:
        try:
            oot_pf = pivot.loc[combo, ("profit_factor", "oot")]
            oot_avg = pivot.loc[combo, ("avg_return_pct", "oot")]
            oot_dd = pivot.loc[combo, ("max_drawdown_pct", "oot")]
            test_avg = pivot.loc[combo, ("avg_return_pct", "test")]
            train_avg = pivot.loc[combo, ("avg_return_pct", "train")]
            oot_trades = pivot.loc[combo, ("trades", "oot")]
        except KeyError:
            continue
        if pd.isna(oot_pf) or pd.isna(oot_avg) or pd.isna(oot_dd) or pd.isna(oot_trades):
            continue
        gap_penalty = abs(float(train_avg or 0) - float(test_avg or 0)) if pd.notna(test_avg) and pd.notna(train_avg) else 0
        trade_penalty = 0 if oot_trades >= 100 else (100 - oot_trades) / 100
        rows.append(
            {
                "combo": combo,
                "selection_score": float(oot_pf) + 0.25 * float(oot_avg) + 0.02 * float(oot_dd) - gap_penalty - trade_penalty,
            }
        )
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).set_index("combo")["selection_score"]


def predict_models(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    for model_name in MODEL_NAMES:
        model_path = MODEL_DIR / f"{model_name}.joblib"
        model = joblib.load(model_path)
        feature_cols = list(model.feature_names_in_)
        missing = [col for col in feature_cols if col not in out.columns]
        if missing:
            raise ValueError(f"{model_path} 缺少特征列: {missing[:20]}")
        X = out[feature_cols].replace([np.inf, -np.inf], np.nan)
        out[f"pred_{model_name}"] = model.predict_proba(X)[:, 1]
    return out.dropna(subset=[f"pred_{name}" for name in MODEL_NAMES]).copy()


def apply_entry_thresholds(df: pd.DataFrame, thresholds: EntryThresholds) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if thresholds.min_up5 is not None:
        mask &= df["pred_up5_es"] >= thresholds.min_up5
    if thresholds.min_up8 is not None:
        mask &= df["pred_up8_es"] >= thresholds.min_up8
    if thresholds.min_up10 is not None:
        mask &= df["pred_up10_es"] >= thresholds.min_up10
    if thresholds.max_down2 is not None:
        mask &= df["pred_down2_es"] <= thresholds.max_down2
    if thresholds.max_down3 is not None:
        mask &= df["pred_down3_es"] <= thresholds.max_down3
    return mask


def apply_post_close_filter(df: pd.DataFrame, gap_rule: OpenGapRule) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if gap_rule.require_j10:
        mask &= df["kdj_d_j"] <= -10
    return mask


def build_strategy_pool(top20: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in top20.iterrows():
        gap_rule = parse_open_gap_rule(str(row["buy_filter"]))
        thresholds = parse_entry_rule(str(row["entry_rule"]))
        rows.append(
            {
                "priority": int(row["priority"]),
                "strategy_id": str(row["combo"]),
                "name": f"Top{int(row['priority']):02d} {row['buy_filter']} {row['exit_mode']}",
                "entry_rule": str(row["entry_rule"]),
                "buy_filter": str(row["buy_filter"]),
                "buy_filter_desc": str(row["buy_filter_desc"]),
                "exit_mode": str(row["exit_mode"]),
                "open_gap_rule": {
                    "min_gap_pct": gap_rule.min_gap_pct,
                    "max_gap_pct": gap_rule.max_gap_pct,
                    "text": open_gap_text(gap_rule),
                },
                "entry_thresholds": thresholds.__dict__,
                "sell_plan": sell_plan(str(row["exit_mode"])),
                "metrics": {
                    "trades": int(row["trades"]),
                    "avg_return_pct": float(row["avg_return_pct"]),
                    "win_rate": float(row["win_rate"]),
                    "max_drawdown_pct": float(row["max_drawdown_pct"]),
                    "profit_factor": float(row["profit_factor"]),
                    "min_return_pct": float(row["min_return_pct"]),
                    "stop_rate": float(row["stop_rate"]),
                    "take_profit_rate": float(row["take_profit_rate"]),
                    "expiry_rate": float(row["expiry_rate"]),
                },
            }
        )
    return rows


def build_daily_plan(signal_date: str | None = None, max_rows: int = 500) -> dict[str, Any]:
    top20 = load_fusion_top20()
    candidates = pd.read_parquet(FEATURE_PATH)
    candidates["date"] = pd.to_datetime(candidates["date"])
    if signal_date:
        requested_date = pd.to_datetime(signal_date)
        available_dates = candidates.loc[candidates["date"] <= requested_date, "date"]
        if available_dates.empty:
            target_date = candidates["date"].max()
        else:
            target_date = available_dates.max()
    else:
        target_date = candidates["date"].max()
    latest = candidates[candidates["date"] == target_date].copy()
    if "name" in latest.columns:
        name = latest["name"].fillna("").astype(str)
        latest = latest[~name.str.upper().str.contains("ST") & ~name.str.contains("退")].copy()
    latest = predict_models(latest)

    plan_rows = []
    for _, strategy in top20.iterrows():
        thresholds = parse_entry_rule(str(strategy["entry_rule"]))
        gap_rule = parse_open_gap_rule(str(strategy["buy_filter"]))
        mask = apply_entry_thresholds(latest, thresholds) & apply_post_close_filter(latest, gap_rule)
        matched = latest[mask].copy()
        if matched.empty:
            continue
        matched["priority"] = int(strategy["priority"])
        matched["strategy_id"] = str(strategy["combo"])
        matched["strategy_name"] = f"Top{int(strategy['priority']):02d} {strategy['buy_filter']} {strategy['exit_mode']}"
        matched["entry_rule"] = str(strategy["entry_rule"])
        matched["buy_filter"] = str(strategy["buy_filter"])
        matched["exit_mode"] = str(strategy["exit_mode"])
        matched["open_gap_text"] = open_gap_text(gap_rule)
        matched["buy_min_price"] = (
            matched["close"] * (1 + gap_rule.min_gap_pct / 100)
            if gap_rule.min_gap_pct is not None
            else np.nan
        )
        matched["buy_max_price"] = (
            matched["close"] * (1 + gap_rule.max_gap_pct / 100)
            if gap_rule.max_gap_pct is not None
            else np.nan
        )
        matched["oot_trades"] = int(strategy["trades"])
        matched["oot_avg_return_pct"] = float(strategy["avg_return_pct"])
        matched["oot_win_rate"] = float(strategy["win_rate"])
        matched["oot_max_drawdown_pct"] = float(strategy["max_drawdown_pct"])
        matched["oot_profit_factor"] = float(strategy["profit_factor"])
        matched["sell_summary"] = sell_plan(str(strategy["exit_mode"]))["summary"]
        plan_rows.append(matched)

    if plan_rows:
        plan_df = pd.concat(plan_rows, ignore_index=True)
        plan_df = plan_df.sort_values(
            ["priority", "pred_up10_es", "pred_down3_es", "symbol"],
            ascending=[True, False, True, True],
        ).head(max_rows)
    else:
        plan_df = pd.DataFrame()

    unique = pd.DataFrame()
    if not plan_df.empty:
        unique = (
            plan_df.sort_values(["symbol", "priority", "pred_up10_es"], ascending=[True, True, False])
            .drop_duplicates("symbol")
            .sort_values(["priority", "pred_up10_es", "pred_down3_es"], ascending=[True, False, True])
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": target_date.strftime("%Y-%m-%d"),
        "execution_date": "下一个交易日",
        "source": {
            "feature_path": str(FEATURE_PATH),
            "fusion_summary_path": str(top20["source_path"].iloc[0]) if not top20.empty else "",
        },
        "strategy_pool": build_strategy_pool(top20),
        "plan_rows": _records(plan_df),
        "unique_symbols": _records(unique),
        "notes": [
            "收盘后只生成观察名单；是否买入取决于次日开盘价是否落入策略要求区间。",
            "同一股票若命中多个策略，unique_symbols 默认展示优先级最高的一条。",
            "结构类退出当前只作为观察策略，主策略仍优先使用盘中硬止损控制尾部风险。",
        ],
    }
    return payload


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    normalized = df.copy()
    keep_cols = [
        "date",
        "symbol",
        "name",
        "industry",
        "close",
        "priority",
        "strategy_id",
        "strategy_name",
        "entry_rule",
        "buy_filter",
        "exit_mode",
        "open_gap_text",
        "buy_min_price",
        "buy_max_price",
        "pred_up5_es",
        "pred_up8_es",
        "pred_up10_es",
        "pred_down2_es",
        "pred_down3_es",
        "kdj_d_j",
        "oot_trades",
        "oot_avg_return_pct",
        "oot_win_rate",
        "oot_max_drawdown_pct",
        "oot_profit_factor",
        "sell_summary",
    ]
    present = [col for col in keep_cols if col in normalized.columns]
    normalized = normalized[present].copy()
    for col in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[col]):
            normalized[col] = normalized[col].dt.strftime("%Y-%m-%d")
    normalized = normalized.replace([np.inf, -np.inf], np.nan)
    return json.loads(normalized.to_json(orient="records", force_ascii=False))


def write_daily_plan(output_path: Path = DAILY_PLAN_PATH, signal_date: str | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_daily_plan(signal_date=signal_date)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_dir = ROUTINE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "b1_daily_plan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
