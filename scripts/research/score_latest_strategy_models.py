#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score the latest selector candidates with trained strategy models.

Daily selector refresh should not wait for a full model retrain. This script
reuses the latest trained XGBoost artifacts, builds today's Tushare-only factor
row for each rule-hit stock, and writes the model-scored candidate parquet that
the web selector reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

import build_training_data_parallel as btd
from analyze_b1_xgb_entry_exit_grid import DEFAULT_DAILY_DIR, DEFAULT_OUTPUT_DIR
from quant.data.source_merge import normalize_tushare_daily
from quant.features.variable_library import build_continuous_ohlc, calculate_project_extra_features
from train_z_skill_models_and_backtest import (
    AucGapEarlyStopping,
    LABELS,
    MODEL_DIR,
    PRIORITY_SIGNALS,
    add_predictions,
    load_models,
    write_latest_scored_candidates,
    _load_signal_cache,
)

# Older model artifacts were trained by executing the research script directly,
# so pickle recorded this callback class under __main__.
setattr(sys.modules["__main__"], "AucGapEarlyStopping", AucGapEarlyStopping)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score latest strategy candidates with trained models")
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--target-date", default=None, help="YYYY-MM-DD. Default: latest date in signal caches.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--signals", nargs="*", default=PRIORITY_SIGNALS)
    return parser.parse_args()


def _process_symbol(args: tuple[str, pd.DataFrame, list[str], pd.Timestamp]) -> pd.DataFrame | None:
    path_str, signal_rows, signals, target_date = args
    path = Path(path_str)
    try:
        daily = pd.read_parquet(path)
        daily = normalize_tushare_daily(daily, path.stem)
        daily = daily.sort_values("date").reset_index(drop=True)
        history_start = target_date - pd.Timedelta(days=450)
        daily = daily[(daily["date"] >= history_start) & (daily["date"] <= target_date)].reset_index(drop=True)
        if len(daily) < 130 or daily["date"].max() < target_date:
            return None
        name = str(daily["name"].dropna().iloc[0]) if "name" in daily.columns and daily["name"].notna().any() else ""
        if "ST" in name.upper() or "退" in name:
            return None

        factors = pd.concat([btd.calculate_factors_single_stock(daily), calculate_project_extra_features(daily)], axis=1)
        factors = factors.loc[:, ~factors.columns.duplicated(keep="last")]
        price = build_continuous_ohlc(daily)
        close_pos = ((price["close"] - price["low"]) / (price["high"] - price["low"]).replace(0, np.nan)).rename("close_pos")
        result = pd.concat([daily, factors, close_pos], axis=1)
        result = result.loc[:, ~result.columns.duplicated(keep="last")]
        result["symbol"] = result["symbol"].fillna(result.get("ts_code", path.stem)).astype(str)

        row = result[result["date"] == target_date].copy()
        if row.empty:
            return None
        signal_rows = signal_rows.copy()
        signal_rows["date"] = pd.to_datetime(signal_rows["date"])
        merged = row.merge(signal_rows[["symbol", "date", *signals]], on=["symbol", "date"], how="inner")
        return merged if not merged.empty else None
    except Exception as exc:
        print(f"skip {path.name}: {exc}", flush=True)
        return None


def build_latest_dataset(daily_dir: Path, signals: list[str], start_date: str, target_date: str | None, workers: int) -> pd.DataFrame:
    signal_df = _load_signal_cache(signals, start_date)
    signal_df["date"] = pd.to_datetime(signal_df["date"])
    target_ts = pd.Timestamp(target_date) if target_date else signal_df["date"].max()
    target_rows = signal_df[signal_df["date"] == target_ts].copy()
    if target_rows.empty:
        return pd.DataFrame(columns=["symbol", "date", *signals])

    by_symbol = {symbol: group[["symbol", "date", *signals]].copy() for symbol, group in target_rows.groupby("symbol")}
    suffixes = (".SZ.parquet", ".SH.parquet", ".BJ.parquet")
    files = [path for path in sorted(daily_dir.glob("*.parquet")) if path.name.endswith(suffixes) and path.stem in by_symbol]
    frames: list[pd.DataFrame] = []
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_process_symbol, (str(path), by_symbol[path.stem], signals, target_ts)) for path in files]
        for n, future in enumerate(as_completed(futures), start=1):
            frame = future.result()
            if frame is not None and len(frame):
                frames.append(frame)
            if n % 500 == 0 or n == len(futures):
                print(f"  latest model scoring features: {n}/{len(futures)} files frames={len(frames)}", flush=True)
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", *signals])
    data = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    print(f"latest scoring dataset rows={len(data):,} date={target_ts.date()} elapsed={perf_counter() - started:.1f}s", flush=True)
    return data


def ensure_model_features(data: pd.DataFrame, models: dict) -> pd.DataFrame:
    out = data.copy()
    for model in models.values():
        for col in getattr(model, "feature_names_in_", []):
            if col not in out.columns:
                out[col] = np.nan
    for label_name in LABELS:
        col = f"pred_{label_name}"
        if col not in out.columns:
            out[col] = np.nan
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signals = list(dict.fromkeys(args.signals))
    print(f"scoring signals: {signals}", flush=True)
    models, _ = load_models(signals, args.model_dir, args.output_dir)
    data = build_latest_dataset(args.daily_dir, signals, args.start_date, args.target_date, args.workers)
    data = ensure_model_features(data, models)
    predicted = add_predictions(data, models, signals) if not data.empty else data
    playbook_path = args.output_dir / "latest_z_skill_model_operational_playbook.csv"
    if not playbook_path.exists():
        raise FileNotFoundError(f"Missing model playbook: {playbook_path}")
    playbooks = pd.read_csv(playbook_path)
    scored_path = write_latest_scored_candidates(predicted, signals, playbooks, args.output_dir)
    scored = pd.read_parquet(scored_path) if scored_path.exists() else pd.DataFrame()
    result = {
        "status": "success",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "target_date": None if data.empty else pd.to_datetime(data["date"]).max().strftime("%Y-%m-%d"),
        "candidate_rows": int(len(data)),
        "scored_rows": int(len(scored)),
        "model_pass_rows": int(scored["model_pass"].fillna(False).sum()) if "model_pass" in scored.columns else 0,
        "output": str(scored_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
