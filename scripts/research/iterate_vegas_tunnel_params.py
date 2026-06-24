#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Iterate Vegas tunnel parameter sets and exit plans."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

from backtest_vegas_tunnel import add_future_prices, build_exit_rules, evaluate, read_daily_file
from quant.strategies.custom.vegas_tunnel import add_vegas_tunnel_signals


DEFAULT_DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/vegas_tunnel"


@dataclass(frozen=True)
class VegasParamSet:
    param_id: str
    fast_span: int
    momentum_span: int
    tunnel_short_span: int
    tunnel_long_span: int
    near_tunnel_pct: float
    pullback_window: int
    max_tunnel_distance: float = 0.18


def default_param_grid() -> list[VegasParamSet]:
    rows: list[VegasParamSet] = []
    for fast_span, momentum_span in [(10, 20), (12, 24), (20, 40)]:
        for tunnel_short_span, tunnel_long_span in [(120, 144), (144, 169), (169, 200)]:
            for near_tunnel_pct in [0.025, 0.035]:
                for pullback_window in [8, 12]:
                    param_id = (
                        f"f{fast_span}_m{momentum_span}_"
                        f"t{tunnel_short_span}_{tunnel_long_span}_"
                        f"near{near_tunnel_pct:.1%}_pb{pullback_window}_max18"
                    )
                    rows.append(
                        VegasParamSet(
                            param_id=param_id,
                            fast_span=fast_span,
                            momentum_span=momentum_span,
                            tunnel_short_span=tunnel_short_span,
                            tunnel_long_span=tunnel_long_span,
                            near_tunnel_pct=near_tunnel_pct,
                            pullback_window=pullback_window,
                        )
                    )
    return rows


def _signals_for_params(daily: pd.DataFrame, params: VegasParamSet, start_ts: pd.Timestamp) -> pd.DataFrame | None:
    signal_frame = add_vegas_tunnel_signals(
        daily,
        fast_span=params.fast_span,
        momentum_span=params.momentum_span,
        tunnel_short_span=params.tunnel_short_span,
        tunnel_long_span=params.tunnel_long_span,
        near_tunnel_pct=params.near_tunnel_pct,
        pullback_window=params.pullback_window,
        max_tunnel_distance=params.max_tunnel_distance,
        min_history=max(params.tunnel_long_span + 20, 180),
    )
    signals = signal_frame[
        (signal_frame["date"] >= start_ts)
        & (signal_frame["signal_vegas_tunnel"] == 1)
    ].copy()
    if signals.empty:
        return None
    keep = [
        "date",
        "symbol",
        "name",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema12",
        "ema24",
        "ema144",
        "ema169",
        "vegas_tunnel_upper",
        "vegas_tunnel_lower",
        "vegas_tunnel_mid",
        "vegas_tunnel_slope_20d",
        "vegas_tunnel_distance",
        "vegas_fast_spread",
        "vegas_volume_strength",
        "vegas_candidate_score",
    ]
    present = [col for col in keep if col in signals.columns]
    out = signals[present].copy()
    for key, value in asdict(params).items():
        out[key] = value
    return out


def build_param_candidates(
    daily_dir: Path,
    params_grid: Iterable[VegasParamSet],
    start_date: str,
    max_workers: int,
) -> pd.DataFrame:
    files = sorted(daily_dir.glob("*.parquet"))
    params_list = list(params_grid)
    start_ts = pd.to_datetime(start_date)
    frames: list[pd.DataFrame] = []

    def process(path: Path) -> list[pd.DataFrame]:
        try:
            daily = read_daily_file(path)
            min_needed = max(param.tunnel_long_span for param in params_list) + 40
            if len(daily) < min_needed:
                return []
            result = []
            for params in params_list:
                signals = _signals_for_params(daily, params, start_ts)
                if signals is not None and not signals.empty:
                    result.append(signals)
            return result
        except Exception as exc:
            print(f"  skip {path.name}: {exc}", flush=True)
            return []

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [executor.submit(process, path) for path in files]
        for n, future in enumerate(as_completed(futures), start=1):
            frames.extend(future.result())
            if n % 500 == 0 or n == len(files):
                print(f"  vegas grid signals: {n}/{len(files)} files", flush=True)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["param_id", "date", "symbol"]).reset_index(drop=True)


def evaluate_param_grid(candidates: pd.DataFrame, daily_dir: Path) -> pd.DataFrame:
    max_hold_days = max(rule.hold_days for rule in build_exit_rules())
    enriched = add_future_prices(candidates, daily_dir, max_hold_days)
    rows: list[pd.DataFrame] = []
    param_cols = [
        "param_id",
        "fast_span",
        "momentum_span",
        "tunnel_short_span",
        "tunnel_long_span",
        "near_tunnel_pct",
        "pullback_window",
        "max_tunnel_distance",
    ]
    for param_id, part in enriched.groupby("param_id", sort=False):
        summary = evaluate(part.copy(), build_exit_rules())
        if summary.empty:
            continue
        first = part.iloc[0]
        for col in param_cols:
            summary[col] = first[col]
        summary["raw_candidates"] = int(len(part))
        daily_count = part.groupby("date").size()
        summary["avg_daily_candidates"] = float(daily_count.mean()) if not daily_count.empty else 0.0
        summary["max_daily_candidates"] = int(daily_count.max()) if not daily_count.empty else 0
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["rank_score"] = (
        out["daily_sharpe"].fillna(0)
        + 0.20 * out["avg_return_pct"].fillna(0)
        + 0.25 * out["profit_factor"].fillna(0)
        + 0.015 * out["max_drawdown_pct"].fillna(0)
        + np.minimum(out["trades"].fillna(0), 80) / 250
    )
    return out.sort_values(["period", "rank_score"], ascending=[True, False]).reset_index(drop=True)


def write_report(summary: pd.DataFrame, output_dir: Path, timestamp: str, meta: dict) -> Path:
    path = output_dir / f"vegas_tunnel_param_grid_report_{timestamp}.md"
    key_cols = [
        "period",
        "param_id",
        "exit_rule",
        "trades",
        "avg_return_pct",
        "win_rate",
        "daily_sharpe",
        "max_drawdown_pct",
        "profit_factor",
        "raw_candidates",
        "avg_daily_candidates",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# 维加斯隧道参数迭代回测\n\n")
        f.write("口径：T+0 选股，T+1 开盘买入，T+2 起检查卖出；复用维加斯隧道多卖出计划。\n\n")
        f.write("## 参数空间\n\n")
        f.write("```json\n")
        f.write(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
        f.write("\n```\n\n")
        for period in ["2024", "2025plus", "all"]:
            part = summary[summary["period"] == period].copy()
            if part.empty:
                continue
            f.write(f"## {period} Top 20 by rank_score\n\n")
            top = part.sort_values(["rank_score", "daily_sharpe", "avg_return_pct"], ascending=False).head(20)
            f.write(top[key_cols].to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterate Vegas tunnel parameter sets")
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--candidate-cache", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_grid = default_param_grid()
    meta = {
        "start_date": args.start_date,
        "param_sets": [asdict(item) for item in params_grid],
    }

    if args.candidate_cache and args.candidate_cache.exists():
        print(f"Loading candidate cache: {args.candidate_cache}")
        candidates = pd.read_parquet(args.candidate_cache)
        candidates["date"] = pd.to_datetime(candidates["date"])
    else:
        print(f"Building {len(params_grid)} Vegas parameter sets from {args.daily_dir}")
        candidates = build_param_candidates(args.daily_dir, params_grid, args.start_date, args.max_workers)
        if args.candidate_cache:
            args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
            candidates.to_parquet(args.candidate_cache, index=False)
            print(f"Candidate cache written: {args.candidate_cache}")

    if candidates.empty:
        raise RuntimeError("No Vegas parameter candidates found")
    print(f"Raw param candidates: {len(candidates):,}")
    summary = evaluate_param_grid(candidates, args.daily_dir)
    summary_path = args.output_dir / f"vegas_tunnel_param_grid_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    report_path = write_report(summary, args.output_dir, timestamp, meta)
    print(f"Summary CSV: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
