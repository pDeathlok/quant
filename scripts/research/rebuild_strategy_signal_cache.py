#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild daily rule-signal caches for the stock selector.

This script refreshes signal candidates only. It intentionally does not run the
entry/exit grid, so the web "latest data" button can update today's selector
pool without waiting for a full research backtest.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

from analyze_b1_family_rule_backtest import build_signal_candidates as build_family_signals
from analyze_z_skill_entry_exit_backtest import (
    SIGNAL_CACHE as EXTENDED_SIGNAL_CACHE,
    build_signal_candidates as build_extended_signals,
)
from analyze_b1_xgb_entry_exit_grid import DEFAULT_DAILY_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild selector rule-signal caches")
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--force-refresh", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _summary(df: pd.DataFrame, signal_cols: list[str]) -> dict:
    if df.empty:
        return {"rows": 0, "latest_date": None, "latest_rows": 0, "latest_hits": {}}
    dates = pd.to_datetime(df["date"])
    latest_date = dates.max()
    latest = df[dates == latest_date]
    return {
        "rows": int(len(df)),
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "latest_rows": int(len(latest)),
        "latest_hits": {col: int(latest[col].fillna(False).astype(bool).sum()) for col in signal_cols if col in latest.columns},
    }


def main() -> None:
    args = parse_args()
    family = build_family_signals(force_refresh=args.force_refresh, workers=args.workers)
    extended = build_extended_signals(args.daily_dir, args.start_date, args.force_refresh, args.workers)
    family_cols = [col for col in family.columns if col.startswith(("b1_", "b2_", "b3_", "sb1_", "super_b1"))]
    extended_cols = [
        "CHANGAN",
        "PINGHANG",
        "DOUBLE_GUN",
        "YIDONG_DILIAN",
        "NANA",
        "GOLDEN_BOWL",
        "BREATHING",
        "KENGQI",
        "DUICHEN_VA",
        "ZAIHOU",
        "YUEYUE",
        "KEY_K",
        "VIOLENCE_K",
    ]
    result = {
        "status": "success",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "family": _summary(family, family_cols),
        "extended": _summary(extended, extended_cols),
        "extended_cache": str(EXTENDED_SIGNAL_CACHE),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
