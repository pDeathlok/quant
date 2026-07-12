#!/usr/bin/env python
"""Build strategy-ready candidate lists from Chan model scores."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.strategies.custom.chan_model import (
    add_chan_model_strategy_columns,
    select_chan_model_candidates,
    summarize_chan_model_strategy,
)


DEFAULT_SCORED_PATH = PROJECT_ROOT / "reports/chan_daily/model_filter/chan_model_scored_candidates.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/chan_daily/model_strategy"


def build_strategy_outputs(
    scored_path: Path,
    output_dir: Path,
    trade_date: str | None,
    top_n: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = pd.read_parquet(scored_path)
    strategy_frame = add_chan_model_strategy_columns(scored)
    strategy_frame.to_parquet(output_dir / "chan_model_strategy_scored.parquet", index=False)
    strategy_frame.to_csv(output_dir / "chan_model_strategy_scored.csv", index=False)

    candidates = select_chan_model_candidates(strategy_frame, trade_date=trade_date, top_n=top_n)
    candidates.to_csv(output_dir / "chan_model_latest_candidates.csv", index=False)
    summary = summarize_chan_model_strategy(strategy_frame)
    summary.to_csv(output_dir / "chan_model_strategy_summary.csv", index=False)

    resolved_trade_date = (
        pd.to_datetime(candidates["date"].iloc[0]).strftime("%Y-%m-%d")
        if not candidates.empty
        else None
    )
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scored_path": str(scored_path),
        "trade_date": resolved_trade_date,
        "top_n": int(top_n),
        "candidate_rows": int(len(candidates)),
        "summary": summary.to_dict("records"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-path", type=Path, default=DEFAULT_SCORED_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_strategy_outputs(args.scored_path, args.output_dir, args.trade_date, args.top_n)
    print(json.dumps(result, ensure_ascii=False, indent=2))
