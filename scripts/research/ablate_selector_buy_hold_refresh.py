#!/usr/bin/env python
"""Ablate selector buy/hold retraining cadence and history length."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json
from quant.features.canonical_factor_names import assert_no_forbidden_factor_names
from quant.features.selector_buy_hold_factor_contract import (
    SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256,
    validate_selector_buy_hold_artifact,
)
from quant.research.selector_buy_hold_refresh_ablation import (
    choose_maintenance_aware_candidate,
    run_walk_forward,
)


DATASET = (
    PROJECT_ROOT
    / "data/research/selector_buy_hold_registry_v2/selector_buy_hold_registry_dataset.parquet"
)
DATASET_MANIFEST = DATASET.parent / "dataset_manifest.json"
MODEL_DIR = PROJECT_ROOT / "models/production/selector_buy_hold_registry_v2"
REPORT_DIR = PROJECT_ROOT / "reports/research/selector_buy_hold_refresh_ablation_v1"
REPORT_PATH = REPORT_DIR / "report.json"
SUMMARY_PATH = REPORT_DIR / "summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-start", default="2024-01-01")
    parser.add_argument("--development-end", default="2025-12-31")
    parser.add_argument("--evaluation-end", default="2026-08-14")
    parser.add_argument("--window-months", default="12,24,36,60,expanding")
    parser.add_argument("--frequency-months", default="static,1,3,6,12")
    parser.add_argument("--window-stage-frequency", type=int, default=3)
    parser.add_argument("--daily-train-cap", type=int, default=128)
    parser.add_argument("--trees", type=int, default=90)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--minimum-training-rows", type=int, default=20_000)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def parse_month_values(value: str, *, allow_static: bool) -> tuple[int | None, ...]:
    output: list[int | None] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if item in {"expanding", "static"}:
            if item == "static" and not allow_static:
                raise ValueError("static is only valid for refresh frequency")
            output.append(None)
        else:
            parsed = int(item)
            if parsed <= 0:
                raise ValueError("month values must be positive")
            output.append(parsed)
    return tuple(dict.fromkeys(output))


def model_factory(*, trees: int, n_jobs: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=trees,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=40,
        subsample=0.75,
        colsample_bytree=0.50,
        reg_lambda=10.0,
        reg_alpha=0.05,
        max_bin=64,
        tree_method="hist",
        random_state=20260827,
        n_jobs=n_jobs,
    )


def _summary_row(stage: str, row: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "stage": stage,
        "frequency_months": row.get("frequency_months") or "static",
        "window_months": row.get("window_months") or "expanding",
        "fit_count": row["fit_count"],
        "average_training_rows": row["average_training_rows"],
    }
    for sample in ("development", "verification", "all"):
        metrics = row["metrics"].get(sample) or {}
        output[f"{sample}_mean_daily_spearman"] = metrics.get(
            "mean_daily_spearman"
        )
        for mode in ("buy", "hold"):
            output[f"{sample}_{mode}_daily_spearman"] = (
                metrics.get(mode) or {}
            ).get("daily_spearman")
            output[f"{sample}_{mode}_decile_spread"] = (
                metrics.get(mode) or {}
            ).get("decile_spread")
    return output


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    try:
        report_dir.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("refresh ablation report directory escapes project root") from exc
    report_path = report_dir / "report.json"
    summary_path = report_dir / "summary.csv"
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("factor_contract_sha256") != SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256:
        raise RuntimeError("refresh ablation dataset factor contract drifted")
    artifacts = {
        mode: joblib.load(MODEL_DIR / f"{mode}.joblib") for mode in ("buy", "hold")
    }
    for artifact in artifacts.values():
        validate_selector_buy_hold_artifact(artifact)
    features = tuple(str(value) for value in artifacts["buy"]["features"])
    if features != tuple(str(value) for value in artifacts["hold"]["features"]):
        raise RuntimeError("buy/hold production feature contracts disagree")
    assert_no_forbidden_factor_names(features, context="refresh ablation model inputs")
    hold_buy_weight = float(artifacts["hold"].get("buy_weight", 0.0))
    columns = [
        "symbol",
        "date",
        "label_end_date",
        "future_max_high_t5_pct",
        "future_return_t5_pct",
        *features,
    ]
    data = pd.read_parquet(DATASET, columns=columns)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["label_end_date"] = pd.to_datetime(data["label_end_date"], errors="raise")
    data = data.sort_values(["date", "symbol"]).reset_index(drop=True)
    hashes = pd.util.hash_pandas_object(data["symbol"], index=False).to_numpy()
    data["_sample_hash"] = hashes
    data["_sample_rank"] = data.groupby("date")["_sample_hash"].rank(
        method="first"
    )
    data["training_sample"] = data["_sample_rank"].le(args.daily_train_cap)
    data = data.drop(columns=["_sample_hash", "_sample_rank"])
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors="coerce").astype(np.float32)

    evaluation_start = pd.Timestamp(args.evaluation_start)
    development_end = pd.Timestamp(args.development_end)
    evaluation_end = min(pd.Timestamp(args.evaluation_end), data["date"].max())
    windows = parse_month_values(args.window_months, allow_static=False)
    frequencies = parse_month_values(args.frequency_months, allow_static=True)
    factory = lambda: model_factory(trees=args.trees, n_jobs=args.n_jobs)
    report_dir.mkdir(parents=True, exist_ok=True)
    window_rows: list[dict[str, Any]] = []
    for window in windows:
        print(f"window stage: window={window or 'expanding'}", flush=True)
        result = run_walk_forward(
            data,
            features=features,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            development_end=development_end,
            frequency_months=args.window_stage_frequency,
            window_months=window,
            hold_buy_weight=hold_buy_weight,
            model_factory=factory,
            minimum_training_rows=args.minimum_training_rows,
        )
        window_rows.append(result)
        atomic_write_json({"status": "running", "window_stage": window_rows}, report_path)
    chosen_window = choose_maintenance_aware_candidate(
        window_rows,
        complexity_key=lambda row: -float(row.get("window_months") or 10_000),
    )
    selected_window = chosen_window.get("window_months")

    frequency_rows: list[dict[str, Any]] = []
    for frequency in frequencies:
        print(f"frequency stage: frequency={frequency or 'static'}", flush=True)
        result = run_walk_forward(
            data,
            features=features,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            development_end=development_end,
            frequency_months=frequency,
            window_months=selected_window,
            hold_buy_weight=hold_buy_weight,
            model_factory=factory,
            minimum_training_rows=args.minimum_training_rows,
        )
        frequency_rows.append(result)
        atomic_write_json(
            {
                "status": "running",
                "window_stage": window_rows,
                "selected_window_months": selected_window,
                "frequency_stage": frequency_rows,
            },
            report_path,
        )
    chosen_frequency = choose_maintenance_aware_candidate(
        frequency_rows,
        complexity_key=lambda row: -float(row.get("frequency_months") or 10_000),
    )
    selected_frequency = chosen_frequency.get("frequency_months")
    summary = [
        *(_summary_row("window", row) for row in window_rows),
        *(_summary_row("frequency", row) for row in frequency_rows),
    ]
    atomic_write_csv(pd.DataFrame(summary), summary_path, index=False)
    payload = {
        "status": "success",
        "schema_version": "selector-buy-hold-refresh-ablation-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_sha256": manifest.get("output_sha256"),
        "factor_contract_sha256": SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256,
        "model_input_count": len(features),
        "model_input_columns": list(features),
        "forbidden_alias_intersection": [],
        "evaluation": {
            "start": evaluation_start.date().isoformat(),
            "development_end": development_end.date().isoformat(),
            "end": evaluation_end.date().isoformat(),
            "daily_train_cap": args.daily_train_cap,
            "trees": args.trees,
            "hold_buy_weight": hold_buy_weight,
            "selection_tolerance_daily_spearman": 0.005,
        },
        "window_stage": window_rows,
        "selected_window_months": selected_window,
        "frequency_stage": frequency_rows,
        "selected_frequency_months": selected_frequency,
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
    }
    atomic_write_json(payload, report_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
