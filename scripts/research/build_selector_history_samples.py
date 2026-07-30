#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build page-equivalent historical selector samples.

This is the data foundation for selector-score calibration. It replays the
same stock-level aggregation used by the web selector, then joins forward
return labels so 2025 can be used for weight tuning and 2026 for OOT checks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.features.variable_library import build_continuous_ohlc
from quant.data import read_partitioned_symbol_file
from quant.routine.paths import DAILY_DIR
from quant.webapp.services import (
    EXTENDED_STRATEGIES,
    FAMILY_SIGNAL_CACHE,
    FAMILY_SIGNAL_COLUMNS,
    MODEL_FILTERED_SIGNALS,
    PROJECT_ROOT as SERVICE_PROJECT_ROOT,
    build_selector_stock_row,
    get_b1_plan,
    _b1_model_signal,
    _extended_signal_payload,
    _family_signal_payload,
    _model_filtered_signal_payload,
    _model_scored_candidates_for_date,
    _signal_group_key,
    _signal_selector_score,
    _stock_basic_profile,
)


EXTENDED_CANDIDATE_CACHE = PROJECT_ROOT / "data/features/z_skill_daily_candidates.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/research/selector_history"
DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2026-12-31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build historical all-strategy selector samples.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--family-cache", type=Path, default=FAMILY_SIGNAL_CACHE)
    parser.add_argument("--extended-cache", type=Path, default=EXTENDED_CANDIDATE_CACHE)
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--max-dates", type=int, default=None, help="Optional smoke-test cap on replay dates.")
    parser.add_argument("--no-b1-plan", action="store_true", help="Skip historical B1 model daily-plan rows.")
    parser.add_argument("--csv", action="store_true", help="Also write CSV copies next to parquet outputs.")
    return parser.parse_args()


def _date_filter(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out[(out["date"] >= pd.Timestamp(start_date)) & (out["date"] <= pd.Timestamp(end_date))].copy()


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _stock_row(symbol: str, source: pd.Series | dict[str, Any], signal_date: str) -> dict[str, Any]:
    get = source.get
    profile = _stock_basic_profile(symbol)
    return {
        "symbol": symbol,
        "name": _clean_text(get("name", "")) or profile["name"],
        "date": signal_date,
        "close": float(get("close")) if pd.notna(get("close", np.nan)) else None,
        "industry": _clean_text(get("industry", "")) or profile["industry"],
        "signals": [],
    }


def _minimal_extended_signal(signal_key: str) -> dict[str, Any]:
    meta = next((item for item in EXTENDED_STRATEGIES if item["key"] == signal_key), None)
    label = meta["label"] if meta else signal_key
    status = meta["status"] if meta else "日线策略信号"
    return {
        "strategy_key": signal_key,
        "strategy_family": signal_key,
        "strategy_name": label,
        "timeframe": "日线级，收盘确认，T+1 开盘观察",
        "logic": f"{label} 规则命中（历史候选样本按页面同款 playbook 补充指标）。",
        "reason": status,
        "buy_plan": "T+1 开盘按该策略 playbook 条件观察。",
        "sell_plan": "按该策略 playbook 卖出规则执行。",
        "metrics": None,
        "metrics_text": "暂无正式回测指标",
        "strength_score": 0.0,
    }


def load_family_signals(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    cols = None
    df = pd.read_parquet(path, columns=cols)
    if df.empty:
        return df
    return _date_filter(df, start_date, end_date)


def load_extended_signals(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    return _date_filter(df, start_date, end_date)


def replay_date(
    signal_date: pd.Timestamp,
    family_df: pd.DataFrame,
    extended_df: pd.DataFrame,
    include_b1_plan: bool,
) -> list[dict[str, Any]]:
    date_text = signal_date.strftime("%Y-%m-%d")
    stocks: dict[str, dict[str, Any]] = {}

    if include_b1_plan:
        try:
            plan = get_b1_plan(signal_date=date_text)
            for plan_row in plan.get("plan_rows", []):
                symbol = str(plan_row.get("symbol"))
                stock = stocks.setdefault(symbol, _stock_row(symbol, plan_row, date_text))
                stock["signals"].append(_b1_model_signal(plan_row))
        except Exception as exc:
            print(f"  skip B1 plan {date_text}: {exc}", flush=True)

    day_family = family_df[family_df["date"] == signal_date]
    for _, row in day_family.iterrows():
        symbol = str(row.get("symbol"))
        stock = stocks.setdefault(symbol, _stock_row(symbol, row, date_text))
        for key, (column, _, _, _) in FAMILY_SIGNAL_COLUMNS.items():
            if column in row.index and bool(row.get(column, False)):
                stock["signals"].append(_family_signal_payload(key, row))

    day_extended = extended_df[extended_df["date"] == signal_date]
    for _, row in day_extended.iterrows():
        symbol = str(row.get("symbol"))
        stock = stocks.setdefault(symbol, _stock_row(symbol, row, date_text))
        for meta in EXTENDED_STRATEGIES:
            key = meta["key"]
            if key in row.index and bool(row.get(key, False)):
                stock["signals"].append(_extended_signal_payload(_minimal_extended_signal(key)))

    model_scored = _model_scored_candidates_for_date(date_text)
    for (symbol, signal_key), model_score in model_scored.items():
        if signal_key not in MODEL_FILTERED_SIGNALS:
            continue
        existing = stocks.get(symbol, {}).get("signals", [])
        if any(signal.get("strategy_family") == signal_key and signal.get("playbook_source") == "模型版" for signal in existing):
            continue
        stock = stocks.setdefault(symbol, _stock_row(symbol, {"date": date_text}, date_text))
        stock["signals"].append(_model_filtered_signal_payload(signal_key, model_score))

    rows = []
    for stock in stocks.values():
        row = build_selector_stock_row(stock, stock.get("signals") or [], date_text)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda item: (item["selector_score"], item["matched_count"], item["best_profit_factor"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _forward_labels_for_symbol(daily_dir: Path, symbol: str, dates: pd.Series) -> pd.DataFrame:
    path = daily_dir / f"{symbol}.parquet"
    daily = read_partitioned_symbol_file(path)
    if daily.empty:
        return pd.DataFrame({"symbol": symbol, "date": dates})
    if "trade_date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    if "vol" in daily.columns and "volume" not in daily.columns:
        daily["volume"] = daily["vol"]
    daily = daily.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if daily.empty:
        return pd.DataFrame({"symbol": symbol, "date": dates})
    price = build_continuous_ohlc(daily)
    daily[["open", "high", "low", "close"]] = price[["open", "high", "low", "close"]]
    label = daily[["date", "open", "high", "low", "close"]].copy()
    base = label["close"].replace(0, np.nan)
    for day in [1, 3, 5]:
        label[f"future_return_t{day}_pct"] = (label["close"].shift(-day) / base - 1) * 100
    future_high = pd.concat([label["high"].shift(-day) for day in range(1, 6)], axis=1).max(axis=1)
    future_low = pd.concat([label["low"].shift(-day) for day in range(1, 6)], axis=1).min(axis=1)
    label["future_max_high_t5_pct"] = (future_high / base - 1) * 100
    label["future_max_drawdown_t5_pct"] = (future_low / base - 1) * 100
    label["symbol"] = symbol
    wanted = pd.DataFrame({"symbol": symbol, "date": pd.to_datetime(dates)})
    label_cols = [
        "symbol",
        "date",
        "future_return_t1_pct",
        "future_return_t3_pct",
        "future_return_t5_pct",
        "future_max_high_t5_pct",
        "future_max_drawdown_t5_pct",
    ]
    return wanted.merge(label[label_cols], on=["symbol", "date"], how="left")


def add_forward_labels(samples: pd.DataFrame, daily_dir: Path) -> pd.DataFrame:
    if samples.empty:
        return samples
    frames = []
    for n, (symbol, part) in enumerate(samples.groupby("symbol"), start=1):
        frames.append(_forward_labels_for_symbol(daily_dir, str(symbol), part["date"]))
        if n % 500 == 0:
            print(f"  forward labels: {n}/{samples['symbol'].nunique()} symbols", flush=True)
    labels = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return samples.merge(labels, on=["symbol", "date"], how="left")


def _model_value(signal: dict[str, Any], field: str) -> float | None:
    model_score = signal.get("model_score") or {}
    value = model_score.get(field)
    if value is not None and not pd.isna(value):
        return float(value)
    reason = str(signal.get("reason") or "")
    short = field.replace("pred_", "")
    match = re.search(rf"{re.escape(short)}=([-+]?\d+(?:\.\d+)?)", reason)
    return float(match.group(1)) if match else None


def flatten_stock_rows(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_rows = []
    signal_rows = []
    for row in rows:
        signals = row.get("signals") or []
        stock_rows.append(
            {
                **{key: value for key, value in row.items() if key != "signals"},
                "signals_json": json.dumps(signals, ensure_ascii=False, default=str),
            }
        )
        for order, signal in enumerate(signals, start=1):
            metrics = signal.get("metrics") or {}
            signal_rows.append(
                {
                    "date": row.get("date"),
                    "symbol": row.get("symbol"),
                    "name": row.get("name"),
                    "industry": row.get("industry"),
                    "stock_rank": row.get("rank"),
                    "stock_selector_score": row.get("selector_score"),
                    "signal_order": order,
                    "signal_selector_score": round(float(_signal_selector_score(signal)), 4),
                    "strategy_key": signal.get("strategy_key"),
                    "strategy_family": signal.get("strategy_family"),
                    "strategy_group": _signal_group_key(signal),
                    "strategy_name": signal.get("strategy_name"),
                    "playbook_source": signal.get("playbook_source"),
                    "action_level": signal.get("action_level"),
                    "buy_plan": signal.get("buy_plan"),
                    "sell_plan": signal.get("sell_plan"),
                    "logic": signal.get("logic"),
                    "reason": signal.get("reason"),
                    "metrics_trades": metrics.get("trades"),
                    "metrics_avg_return_pct": metrics.get("avg_return_pct"),
                    "metrics_win_rate": metrics.get("win_rate"),
                    "metrics_max_drawdown_pct": metrics.get("max_drawdown_pct"),
                    "metrics_profit_factor": metrics.get("profit_factor"),
                    "pred_up5": _model_value(signal, "pred_up5"),
                    "pred_up8": _model_value(signal, "pred_up8"),
                    "pred_up10": _model_value(signal, "pred_up10"),
                    "pred_down3": _model_value(signal, "pred_down3"),
                }
            )
    return pd.DataFrame(stock_rows), pd.DataFrame(signal_rows)


def write_outputs(stock_samples: pd.DataFrame, signal_samples: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stock_path = args.output_dir / "selector_stock_history_samples.parquet"
    signal_path = args.output_dir / "selector_signal_history_samples.parquet"
    stock_samples.to_parquet(stock_path, index=False)
    signal_samples.to_parquet(signal_path, index=False)
    csv_paths = {}
    if args.csv:
        stock_csv = stock_path.with_suffix(".csv")
        signal_csv = signal_path.with_suffix(".csv")
        stock_samples.to_csv(stock_csv, index=False)
        signal_samples.to_csv(signal_csv, index=False)
        csv_paths = {"stock_csv": str(stock_csv), "signal_csv": str(signal_csv)}

    summary = {
        "status": "success",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "stock_rows": int(len(stock_samples)),
        "signal_rows": int(len(signal_samples)),
        "dates": int(stock_samples["date"].nunique()) if not stock_samples.empty else 0,
        "symbols": int(stock_samples["symbol"].nunique()) if not stock_samples.empty else 0,
        "stock_parquet": str(stock_path),
        "signal_parquet": str(signal_path),
        **csv_paths,
    }
    summary_path = args.output_dir / "selector_history_samples_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**summary, "summary_json": str(summary_path)}


def main() -> None:
    args = parse_args()
    if SERVICE_PROJECT_ROOT != PROJECT_ROOT:
        raise RuntimeError(f"Unexpected project root mismatch: {SERVICE_PROJECT_ROOT} != {PROJECT_ROOT}")
    family = load_family_signals(args.family_cache, args.start_date, args.end_date)
    extended = load_extended_signals(args.extended_cache, args.start_date, args.end_date)
    dates = sorted(set(family["date"].dropna().unique()).union(set(extended["date"].dropna().unique())))
    if args.max_dates:
        dates = dates[: args.max_dates]
    if not dates:
        raise RuntimeError("No historical strategy dates found in family/extended caches")

    all_rows: list[dict[str, Any]] = []
    for n, signal_date in enumerate(dates, start=1):
        day_rows = replay_date(pd.Timestamp(signal_date), family, extended, include_b1_plan=not args.no_b1_plan)
        all_rows.extend(day_rows)
        if n % 20 == 0 or n == len(dates):
            print(f"  replayed selector dates: {n}/{len(dates)} rows={len(all_rows):,}", flush=True)

    stock_samples, signal_samples = flatten_stock_rows(all_rows)
    stock_samples["date"] = pd.to_datetime(stock_samples["date"])
    signal_samples["date"] = pd.to_datetime(signal_samples["date"])
    stock_samples = add_forward_labels(stock_samples, args.daily_dir)
    stock_samples = stock_samples.sort_values(["date", "rank", "symbol"]).reset_index(drop=True)
    signal_samples = signal_samples.sort_values(["date", "stock_rank", "symbol", "signal_order"]).reset_index(drop=True)
    result = write_outputs(stock_samples, signal_samples, args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
