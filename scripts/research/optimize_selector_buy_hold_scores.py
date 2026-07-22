#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the idempotent selector score rebuild/train/champion pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAILY_GLOB = PROJECT_ROOT / "data/raw/daily_partitioned/year_month=*/data.parquet"
HISTORY = PROJECT_ROOT / "data/research/selector_model_history_2020.parquet"
REPORT_DIR = PROJECT_ROOT / "reports/selector_buy_hold_model_multitask"
MODEL_DIR = PROJECT_ROOT / "models/candidates/selector_buy_hold_multitask"
PIPELINE_REPORT = REPORT_DIR / "pipeline_run.json"


@dataclass(frozen=True)
class RollingSplits:
    train_start: str
    train_end: str
    valid_end: str
    test_end: str
    market_end: str
    label_horizon: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously optimize buy/hold historical scores.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--valid-end", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--valid-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--promote", action="store_true")
    return parser.parse_args()


def market_dates() -> list[Any]:
    values = (
        pl.scan_parquet(str(DAILY_GLOB), hive_partitioning=True)
        .select(pl.col("date").unique())
        .sort("date")
        .collect()["date"]
        .to_list()
    )
    if not values:
        raise RuntimeError("Cannot determine latest market date")
    return values


def _iso_date(value: Any) -> str:
    return value.date().isoformat() if hasattr(value, "date") else str(value)[:10]


def resolve_rolling_splits(args: argparse.Namespace, dates: list[Any]) -> RollingSplits:
    explicit = (args.train_start, args.train_end, args.valid_end, args.test_end)
    if any(explicit) and not all(explicit):
        raise ValueError("Specify all four split dates, or omit all of them for automatic rolling splits")
    market_end = _iso_date(dates[-1])
    if all(explicit):
        return RollingSplits(*explicit, market_end=market_end, label_horizon=args.label_horizon)
    required = args.train_days + args.valid_days + args.test_days + args.label_horizon
    if len(dates) < required:
        raise ValueError(f"Need at least {required} market sessions, found {len(dates)}")
    mature_dates = dates[: -args.label_horizon] if args.label_horizon else dates
    test_dates = mature_dates[-args.test_days :]
    valid_dates = mature_dates[-(args.test_days + args.valid_days) : -args.test_days]
    train_dates = mature_dates[
        -(args.test_days + args.valid_days + args.train_days) : -(args.test_days + args.valid_days)
    ]
    return RollingSplits(
        train_start=_iso_date(train_dates[0]),
        train_end=_iso_date(train_dates[-1]),
        valid_end=_iso_date(valid_dates[-1]),
        test_end=_iso_date(test_dates[-1]),
        market_end=market_end,
        label_horizon=args.label_horizon,
    )


def validate_history(path: Path, splits: RollingSplits) -> dict[str, Any]:
    frame = pl.scan_parquet(path)
    schema = frame.collect_schema().names()
    required = {"symbol", "date", "future_return_t5_pct", "future_max_high_t5_pct"}
    missing = sorted(required.difference(schema))
    if missing:
        raise RuntimeError(f"History is missing required columns: {missing}")
    selector_columns = [column for column in schema if column.startswith("selector_")]
    test_end = pl.lit(splits.test_end).str.to_date()
    summary = frame.select(
        pl.len().alias("rows"),
        pl.col("date").n_unique().alias("dates"),
        pl.col("symbol").n_unique().alias("symbols"),
        pl.col("date").min().alias("date_min"),
        pl.col("date").max().alias("date_max"),
        pl.struct(["symbol", "date"]).n_unique().alias("unique_keys"),
        pl.col("future_return_t5_pct").filter(pl.col("date") <= test_end).is_not_null().mean().alias("hold_label_coverage"),
        pl.col("future_max_high_t5_pct").filter(pl.col("date") <= test_end).is_not_null().mean().alias("buy_label_coverage"),
        pl.any_horizontal([pl.col(column).is_not_null() for column in selector_columns])
        .mean()
        .alias("factor_row_coverage"),
    ).collect().row(0, named=True)
    result = {
        key: (_iso_date(value) if key in {"date_min", "date_max"} else value)
        for key, value in summary.items()
    }
    problems = []
    if result["rows"] != result["unique_keys"]:
        problems.append("duplicate_symbol_date_keys")
    if result["date_max"] != splits.market_end:
        problems.append("history_not_fresh")
    if min(result["buy_label_coverage"], result["hold_label_coverage"]) < 0.99:
        problems.append("mature_label_coverage_below_99pct")
    if not selector_columns or result["factor_row_coverage"] < 0.99:
        problems.append("factor_coverage_below_99pct")
    result["selector_factor_columns"] = len(selector_columns)
    result["passing"] = not problems
    result["problems"] = problems
    if problems:
        raise RuntimeError(f"History quality validation failed: {', '.join(problems)}")
    return result


def run(command: list[str]) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    args = parse_args()
    dates = market_dates()
    splits = resolve_rolling_splits(args, dates)
    end_date = args.end_date or splits.market_end
    run(
        [
            sys.executable,
            "scripts/research/build_selector_model_history.py",
            "--start-date",
            args.start_date,
            "--end-date",
            end_date,
            "--output",
            str(HISTORY),
        ]
    )
    quality = validate_history(HISTORY, splits)
    train_command = [
        sys.executable,
        "scripts/research/train_selector_buy_hold_models.py",
        "--history",
        str(HISTORY),
        "--factor-data",
        str(HISTORY),
        "--output-dir",
        str(REPORT_DIR),
        "--model-dir",
        str(MODEL_DIR),
        "--train-start",
        splits.train_start,
        "--train-end",
        splits.train_end,
        "--valid-end",
        splits.valid_end,
        "--test-end",
        splits.test_end,
        "--n-jobs",
        str(args.n_jobs),
    ]
    if args.promote:
        train_command.append("--promote")
    run(train_command)
    payload = {
        "status": "success",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "history_end_date": end_date,
        "splits": asdict(splits),
        "history_quality": quality,
        "report": str(REPORT_DIR / "selector_buy_hold_model_report.json"),
        "promotion_requested": args.promote,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
