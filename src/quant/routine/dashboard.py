from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.routine.paths import REPORTS_DIR, WEB_DATA_DIR


SUMMARY_PATH = REPORTS_DIR / "summary.csv"
TRADES_PATH = REPORTS_DIR / "trades.csv"
DASHBOARD_PATH = WEB_DATA_DIR / "dashboard.json"
MODEL_COMPATIBILITY_PATH = REPORTS_DIR / "model_compatibility_audit.json"


COMBO_TO_STRATEGY = {
    "stable_up8_055_down3_055_trail4_dd2_sl15_T7": "b1_stable",
    "aggressive_up8_065_down3_050_trail5_dd2_sl15_T9": "b1_aggressive",
    "baseline_up8_055_trail5_dd2_sl2_T9": "b1_baseline",
}

COMBO_TO_NAME = {
    "stable_up8_055_down3_055_trail4_dd2_sl15_T7": "B1 稳健版",
    "aggressive_up8_065_down3_050_trail5_dd2_sl15_T9": "B1 进攻版",
    "baseline_up8_055_trail5_dd2_sl2_T9": "B1 旧基准",
}


def _records(df: pd.DataFrame) -> list[dict]:
    normalized = df.copy()
    for col in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[col]):
            normalized[col] = normalized[col].dt.strftime("%Y-%m-%d")
    return json.loads(normalized.to_json(orient="records", force_ascii=False))


def build_dashboard_payload(summary_path: Path = SUMMARY_PATH, trades_path: Path = TRADES_PATH) -> dict:
    if summary_path == SUMMARY_PATH and MODEL_COMPATIBILITY_PATH.exists():
        compatibility = json.loads(MODEL_COMPATIBILITY_PATH.read_text(encoding="utf-8"))
        if compatibility.get("status") != "valid":
            raise RuntimeError(
                "B1 dashboard publication blocked by model compatibility audit: "
                + str(compatibility.get("reason") or compatibility.get("status"))
            )
    summary = pd.read_csv(summary_path)
    trades = pd.read_csv(trades_path, parse_dates=["date"])
    summary["strategy_id"] = summary["combo"].map(COMBO_TO_STRATEGY).fillna(summary["combo"])
    summary["strategy_name"] = summary["combo"].map(COMBO_TO_NAME).fillna(summary["combo"])
    trades["strategy_id"] = trades["combo"].map(COMBO_TO_STRATEGY).fillna(trades["combo"])
    trades["strategy_name"] = trades["combo"].map(COMBO_TO_NAME).fillna(trades["combo"])

    latest_date = trades["date"].max()
    latest_trades = trades[trades["date"] == latest_date].copy()
    latest_signals = latest_trades.sort_values(
        ["strategy_id", "pred_up8_es", "pred_down3_es"],
        ascending=[True, False, True],
    ).head(200)

    daily = (
        trades.groupby(["date", "strategy_id", "strategy_name"], as_index=False)
        .agg(
            trades=("symbol", "count"),
            avg_return_pct=("return_pct", "mean"),
            win_rate=("return_pct", lambda s: float((s > 0).mean())),
            stop_rate=("exit_type", lambda s: float((s == "stop_loss").mean())),
        )
        .sort_values(["date", "strategy_id"])
    )

    recent_daily = daily[daily["date"] >= latest_date - pd.Timedelta(days=90)].copy()
    oot_summary = summary[summary["period"] == "oot_2025plus"].copy()
    all_summary = summary.copy()

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_signal_date": latest_date.strftime("%Y-%m-%d"),
        "strategies": _records(oot_summary),
        "all_periods": _records(all_summary),
        "latest_signals": _records(latest_signals),
        "recent_daily": _records(recent_daily),
    }


def write_dashboard_json(output_path: Path = DASHBOARD_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_payload()
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
