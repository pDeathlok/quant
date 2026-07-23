#!/usr/bin/env python
"""Refresh live Chan model scores and backfill web snapshots.

The training dataset only contains rows with enough future bars to build labels.
For the web desk we also need recent signal dates, so this script reuses the
saved Chan models to score unlabeled recent candidates and appends them to the
model score file consumed by the web service.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts/research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts/research"))

from backtest_chan_daily import build_candidates, read_daily_file
from train_chan_daily_models import (
    BASE_FEATURES,
    add_candidate_cross_section_ranks,
    add_stock_features,
    read_daily_basic_features,
)
from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from quant.features.market_sentiment import (
    build_limit_proxy_features,
    normalize_ts_code,
    read_top_list_features,
)
from quant.strategies.custom.chan_model import (
    add_chan_model_strategy_columns,
    select_chan_model_candidates,
    summarize_chan_model_strategy,
)


DEFAULT_DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports/chan_daily"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models/research/chan_daily"
DEFAULT_TOP_LIST_DIR = PROJECT_ROOT / "data/raw/moneyflow"
DEFAULT_DAILY_BASIC_DIR = PROJECT_ROOT / "data/raw/daily_basic"
DEFAULT_SCORED_PATH = DEFAULT_REPORT_DIR / "model_filter/chan_model_scored_candidates.parquet"
DEFAULT_REFRESH_MANIFEST_PATH = DEFAULT_REPORT_DIR / "model_filter/live_refresh_manifest.json"
DEFAULT_OUTPUT_DIR = DEFAULT_REPORT_DIR / "model_strategy"


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _business_dates(start: str, end: str) -> list[str]:
    return [item.strftime("%Y-%m-%d") for item in pd.bdate_range(start=start, end=end)]


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _load_models(model_dir: Path) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for target in ["target_win10", "target_big10", "target_good"]:
        path = model_dir / f"{target}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"缺少缠论模型文件: {path}")
        models[target] = joblib.load(path)
    return models


def _resolve_daily_path(daily_dir: Path, symbol: str) -> Path | None:
    value = str(symbol)
    direct = daily_dir / f"{value}.parquet"
    if direct.exists():
        return direct
    digits = value.split(".", 1)[0].zfill(6)
    legacy = daily_dir / f"{digits}.parquet"
    if legacy.exists():
        return legacy
    matches = sorted(daily_dir.glob(f"{digits}.*.parquet"))
    if matches:
        return matches[0]
    suffix = "SH" if digits.startswith(("6", "9")) else "BJ" if digits.startswith(("4", "8")) else "SZ"
    canonical = daily_dir / f"{digits}.{suffix}.parquet"
    partition_root = daily_dir.parent / f"{daily_dir.name}_partitioned"
    return canonical if partition_root.exists() else None


def _build_recent_feature_dataset(
    candidates: pd.DataFrame,
    daily_dir: Path,
    daily_basic_dir: Path,
    top_list_dir: Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    signal = candidates[candidates["signal_chan_daily_long"].eq(1)].copy()
    signal["date"] = pd.to_datetime(signal["date"], errors="coerce")
    signal = signal[signal["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    signal["symbol"] = signal["symbol"].map(normalize_ts_code)
    if signal.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for symbol, group in signal.groupby("symbol"):
        path = _resolve_daily_path(daily_dir, str(symbol))
        if path is None:
            continue
        daily = add_stock_features(read_daily_file(path))
        daily["symbol"] = daily["symbol"].map(normalize_ts_code)
        daily["entry_open"] = daily["open"].shift(-1)
        daily["entry_gap_pct"] = (daily["entry_open"] / daily["close"] - 1) * 100
        keep = [
            "symbol",
            "ts_code",
            "date",
            "ret_1d",
            "ret_3d",
            "ret_5d",
            "ret_10d",
            "ret_20d",
            "close_pos_20",
            "ma5_dist",
            "ma10_dist",
            "ma20_dist",
            "ma60_dist",
            "ma20_slope_5d",
            "volume_rel5",
            "volume_rel20",
            "volume_z20",
            "turnover_rate",
            "turnover_rate_ma20",
            "turnover_rate_rel20",
            "volatility_20d",
            "amount_rel20",
            "entry_open",
            "entry_gap_pct",
        ]
        merged = group.merge(daily[[col for col in keep if col in daily.columns]], on=["symbol", "date"], how="left")
        frames.append(merged)
    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    market = build_limit_proxy_features(daily_dir, start=start)
    if not market.empty:
        data = data.merge(market, on="date", how="left")

    top = read_top_list_features(top_list_dir, start=start)
    if not top.empty:
        data["ts_code"] = data["symbol"].map(normalize_ts_code)
        data = data.merge(top, on=["ts_code", "date"], how="left")

    daily_basic = read_daily_basic_features(daily_basic_dir, data["date"])
    if not daily_basic.empty:
        data["ts_code"] = data["symbol"].map(normalize_ts_code)
        data = data.merge(daily_basic, on=["ts_code", "date"], how="left")

    for col in ["top_list_count", "top_net_amount_ratio", "top_net_rate"]:
        if col not in data.columns:
            data[col] = np.nan
    data["top_list_count"] = data["top_list_count"].fillna(0)
    data = add_candidate_cross_section_ranks(data)
    data["split"] = "live"
    for col in ["hold_5d_close", "hold_10d_close", "hold_20d_close", "target_win10", "target_big10", "target_good"]:
        if col not in data.columns:
            data[col] = np.nan
    return data.replace([np.inf, -np.inf], np.nan)


def _add_predictions(data: pd.DataFrame, models: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = data.copy()
    all_features = set()
    for bundle in models.values():
        all_features.update(bundle["features"])
    for feature in sorted(all_features):
        if feature not in out.columns:
            out[feature] = np.nan
    for target, bundle in models.items():
        x = bundle["imputer"].transform(out[bundle["features"]])
        out[f"pred_{target}"] = bundle["model"].predict_proba(x)[:, 1]
    return out


def _write_strategy_outputs(scored: pd.DataFrame, output_dir: Path, top_n: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_frame = add_chan_model_strategy_columns(scored)
    atomic_write_parquet(
        strategy_frame,
        output_dir / "chan_model_strategy_scored.parquet",
        index=False,
    )
    atomic_write_csv(strategy_frame, output_dir / "chan_model_strategy_scored.csv", index=False)
    candidates = select_chan_model_candidates(strategy_frame, top_n=top_n)
    atomic_write_csv(candidates, output_dir / "chan_model_latest_candidates.csv", index=False)
    summary = summarize_chan_model_strategy(strategy_frame)
    atomic_write_csv(summary, output_dir / "chan_model_strategy_summary.csv", index=False)
    return {
        "latest_signal_date": pd.to_datetime(candidates["date"].iloc[0]).strftime("%Y-%m-%d") if not candidates.empty else None,
        "latest_candidates": int(len(candidates)),
        "strategy_signal_dates": int(strategy_frame[strategy_frame["chan_model_signal"].eq(1)]["date"].nunique()),
    }


def refresh_live_scores(args: argparse.Namespace) -> dict[str, Any]:
    _load_env(PROJECT_ROOT / ".env")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scored_path.parent.mkdir(parents=True, exist_ok=True)

    if args.rebuild_candidates:
        candidate_path = args.report_dir / "chan_daily_candidates.parquet"
        incremental_start = args.start if candidate_path.exists() else args.candidate_start_date
        fresh_candidates = build_candidates(args.daily_dir, incremental_start, args.max_workers)
        historical_candidates = pd.read_parquet(candidate_path) if candidate_path.exists() else pd.DataFrame()
        if not historical_candidates.empty:
            historical_candidates["date"] = pd.to_datetime(historical_candidates["date"], errors="coerce")
            historical_candidates = historical_candidates[
                ~historical_candidates["date"].between(pd.Timestamp(args.start), pd.Timestamp(args.end))
            ].copy()
        candidates = pd.concat([historical_candidates, fresh_candidates], ignore_index=True, sort=False)
        candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce")
        candidates = candidates.sort_values(["date", "symbol"]).drop_duplicates(
            ["date", "symbol"],
            keep="last",
        )
        args.report_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(candidates, candidate_path, index=False)
        atomic_write_csv(candidates, candidate_path.with_suffix(".csv"), index=False)
    else:
        candidates = pd.read_parquet(args.report_dir / "chan_daily_candidates.parquet")

    live = _build_recent_feature_dataset(
        candidates=candidates,
        daily_dir=args.daily_dir,
        daily_basic_dir=args.daily_basic_dir,
        top_list_dir=args.top_list_dir,
        start=args.start,
        end=args.end,
    )
    models = _load_models(args.model_dir)
    live_scored = _add_predictions(live, models) if not live.empty else live

    historical = pd.read_parquet(args.scored_path) if args.scored_path.exists() else pd.DataFrame()
    if not historical.empty:
        historical["date"] = pd.to_datetime(historical["date"], errors="coerce")
        start_ts = pd.Timestamp(args.start)
        end_ts = pd.Timestamp(args.end)
        historical = historical[~historical["date"].between(start_ts, end_ts)].copy()
    combined = pd.concat([historical, live_scored], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    atomic_write_parquet(combined, args.scored_path, index=False)
    atomic_write_csv(combined, args.scored_path.with_suffix(".csv"), index=False)

    strategy_meta = _write_strategy_outputs(combined, args.output_dir, args.top_n)

    snapshot_results: list[dict[str, Any]] = []
    if args.backfill_snapshots:
        from quant.webapp.services import get_chan_model_strategy_plan

        for signal_date in _business_dates(args.start, args.end):
            payload = get_chan_model_strategy_plan(top_n=args.top_n, refresh=True, signal_date=signal_date)
            snapshot_results.append(
                {
                    "signal_date": signal_date,
                    "candidates": int(len(payload.get("candidates") or [])),
                    "primary_count": int(payload.get("primary_count") or 0),
                    "expanded_count": int(payload.get("expanded_count") or 0),
                }
            )

    result = {
        "status": "success",
        "start": args.start,
        "end": args.end,
        "live_rows": int(len(live_scored)),
        "live_dates": sorted(pd.to_datetime(live_scored["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique().tolist()) if not live_scored.empty else [],
        "combined_rows": int(len(combined)),
        "combined_max_date": combined["date"].max().strftime("%Y-%m-%d") if not combined.empty else None,
        "strategy": strategy_meta,
        "snapshots": snapshot_results,
    }
    manifest_path = args.scored_path.parent / DEFAULT_REFRESH_MANIFEST_PATH.name
    atomic_write_json(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "processed_through": args.end,
            "daily_dir": str(args.daily_dir),
            "daily_basic_dir": str(args.daily_basic_dir),
            "live_rows": result["live_rows"],
            "combined_max_date": result["combined_max_date"],
        },
        manifest_path,
    )
    result["manifest_path"] = str(manifest_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--top-list-dir", type=Path, default=DEFAULT_TOP_LIST_DIR)
    parser.add_argument("--daily-basic-dir", type=Path, default=DEFAULT_DAILY_BASIC_DIR)
    parser.add_argument("--scored-path", type=Path, default=DEFAULT_SCORED_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-start-date", default="2015-01-01")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--rebuild-candidates", action="store_true")
    parser.add_argument("--skip-backfill-snapshots", action="store_true")
    args = parser.parse_args()
    args.backfill_snapshots = not args.skip_backfill_snapshots
    return args


if __name__ == "__main__":
    result = refresh_live_scores(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
