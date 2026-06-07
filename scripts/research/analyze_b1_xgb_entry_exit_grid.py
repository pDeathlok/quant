#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Explore B1 entry/exit rules using XGBoost research models.

The entry side uses direct probability thresholds only. There is no manually
weighted entry score in this research path.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from analyze_b1_entry_exit_grid import ExitRule, add_future_prices, simulate_exit, summarize_returns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data/features/b1/training_xgb_project_vars.parquet"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models/research/b1_xgb_project_vars"
DEFAULT_DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/b1/research/xgb_project_vars_strategy"

MODEL_NAMES = ["up5_es", "up8_es", "up10_es", "down2_es", "down3_es"]


@dataclass(frozen=True)
class ThresholdEntryRule:
    name: str
    min_up5: float | None = None
    min_up8: float | None = None
    min_up10: float | None = None
    max_down2: float | None = None
    max_down3: float | None = None


def predict_xgb_models(candidates: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    out = candidates.copy()
    for model_name in MODEL_NAMES:
        model_path = model_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        model = joblib.load(model_path)
        feature_cols = list(model.feature_names_in_)
        missing = [col for col in feature_cols if col not in out.columns]
        if missing:
            raise ValueError(f"{model_path} missing feature columns: {missing[:20]}")
        X = out[feature_cols].replace([np.inf, -np.inf], np.nan)
        out[f"pred_{model_name}"] = model.predict_proba(X)[:, 1]
    return out.dropna(subset=[f"pred_{name}" for name in MODEL_NAMES]).copy()


def build_entry_rules() -> list[ThresholdEntryRule]:
    rules: list[ThresholdEntryRule] = [ThresholdEntryRule("B1_all")]

    for up8 in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for down3 in [0.40, 0.45, 0.50, 0.55, 0.60]:
            rules.append(ThresholdEntryRule(f"up8_ge_{up8:.2f}_down3_le_{down3:.2f}", min_up8=up8, max_down3=down3))

    for up5 in [0.50, 0.55, 0.60, 0.65]:
        for up8 in [0.50, 0.55, 0.60, 0.65]:
            for down2 in [0.45, 0.50, 0.55, 0.60]:
                rules.append(
                    ThresholdEntryRule(
                        f"up5_ge_{up5:.2f}_up8_ge_{up8:.2f}_down2_le_{down2:.2f}",
                        min_up5=up5,
                        min_up8=up8,
                        max_down2=down2,
                    )
                )

    for up8 in [0.50, 0.55, 0.60, 0.65]:
        for up10 in [0.20, 0.25, 0.30, 0.35, 0.40]:
            for down3 in [0.45, 0.50, 0.55, 0.60]:
                rules.append(
                    ThresholdEntryRule(
                        f"up8_ge_{up8:.2f}_up10_ge_{up10:.2f}_down3_le_{down3:.2f}",
                        min_up8=up8,
                        min_up10=up10,
                        max_down3=down3,
                    )
                )

    for up10 in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for down3 in [0.40, 0.45, 0.50, 0.55, 0.60]:
            rules.append(ThresholdEntryRule(f"up10_ge_{up10:.2f}_down3_le_{down3:.2f}", min_up10=up10, max_down3=down3))

    deduped: dict[str, ThresholdEntryRule] = {}
    for rule in rules:
        deduped[rule.name] = rule
    return list(deduped.values())


def build_exit_rules() -> list[ExitRule]:
    rules: list[ExitRule] = []
    hold_map = {"T5": 4, "T7": 6, "T9": 8, "T12": 11}
    for label, hold_days in hold_map.items():
        rules.append(ExitRule(f"expiry_{label}_close", "expiry", hold_days))

    for label, hold_days in hold_map.items():
        for tp in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
            for sl in [0.01, 0.015, 0.02, 0.025]:
                rules.append(ExitRule(f"fixed_tp{tp:.1%}_sl{sl:.1%}_{label}", "fixed", hold_days, tp, sl))

    for label, hold_days in hold_map.items():
        for target in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
            for trail in [0.015, 0.02, 0.03, 0.05]:
                for sl in [0.01, 0.015, 0.02, 0.025]:
                    rules.append(
                        ExitRule(
                            f"trail_target{target:.1%}_dd{trail:.1%}_sl{sl:.1%}_{label}",
                            "trailing",
                            hold_days,
                            target,
                            sl,
                            trail,
                        )
                    )
    return rules


def apply_entry_rule(df: pd.DataFrame, rule: ThresholdEntryRule) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if rule.min_up5 is not None:
        mask &= df["pred_up5_es"] >= rule.min_up5
    if rule.min_up8 is not None:
        mask &= df["pred_up8_es"] >= rule.min_up8
    if rule.min_up10 is not None:
        mask &= df["pred_up10_es"] >= rule.min_up10
    if rule.max_down2 is not None:
        mask &= df["pred_down2_es"] <= rule.max_down2
    if rule.max_down3 is not None:
        mask &= df["pred_down3_es"] <= rule.max_down3
    return mask


def drop_overlapping_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Keep the first signal per stock until that trade exits."""
    if trades.empty:
        return trades
    required = {"symbol", "date", "exit_date"}
    if not required <= set(trades.columns):
        missing = sorted(required - set(trades.columns))
        raise ValueError(f"Cannot enforce non-overlap without columns: {missing}")

    ordered = trades.sort_values(["symbol", "date", "exit_date"]).reset_index()
    entry_ns = pd.to_datetime(ordered["date"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    exit_ns = pd.to_datetime(ordered["exit_date"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    keep_mask = np.zeros(len(ordered), dtype=bool)

    symbols = ordered["symbol"].to_numpy()
    starts = np.r_[0, np.flatnonzero(symbols[1:] != symbols[:-1]) + 1]
    ends = np.r_[starts[1:], len(ordered)]
    nat = np.iinfo("int64").min
    for start, end in zip(starts, ends):
        current_exit = nat
        for pos in range(start, end):
            if exit_ns[pos] == nat:
                continue
            if entry_ns[pos] > current_exit:
                keep_mask[pos] = True
                current_exit = exit_ns[pos]

    kept_index = ordered.loc[keep_mask, "index"].to_numpy()
    return trades.loc[kept_index].sort_values(["date", "symbol"]).reset_index(drop=True)


def evaluate_grid(
    candidates: pd.DataFrame,
    entry_rules: Iterable[ThresholdEntryRule],
    exit_rules: Iterable[ExitRule],
    min_trades: int,
    position_policy: str,
    grid_workers: int,
) -> pd.DataFrame:
    rows = []
    exit_rules_list = list(exit_rules)

    def evaluate_one_exit(entry_df: pd.DataFrame, entry_rule: ThresholdEntryRule, exit_rule: ExitRule) -> tuple[ExitRule, int, int, list[dict]]:
        trades = simulate_exit(entry_df, exit_rule)
        if trades.empty:
            return exit_rule, 0, 0, []
        trades = trades.merge(entry_df[["date", "symbol", "split"]], on=["date", "symbol"], how="left")
        raw_trades = len(trades)
        if position_policy == "non_overlap":
            trades = drop_overlapping_trades(trades)
        skipped_overlaps = raw_trades - len(trades)

        exit_rows = []
        for split in ["train", "test", "oot"]:
            split_trades = trades[trades["split"] == split] if "split" in trades.columns else pd.DataFrame()
            metrics = summarize_returns(split_trades)
            if not metrics:
                continue
            exit_rows.append(
                {
                    "split": split,
                    "entry_rule": entry_rule.name,
                    "exit_rule": exit_rule.name,
                    "exit_kind": exit_rule.kind,
                    "hold_days": exit_rule.hold_days,
                    "take_profit": exit_rule.take_profit,
                    "stop_loss": exit_rule.stop_loss,
                    "trail_drawdown": exit_rule.trail_drawdown,
                    "stop_trigger": exit_rule.stop_trigger,
                    "min_up5": entry_rule.min_up5,
                    "min_up8": entry_rule.min_up8,
                    "min_up10": entry_rule.min_up10,
                    "max_down2": entry_rule.max_down2,
                    "max_down3": entry_rule.max_down3,
                    "position_policy": position_policy,
                    "raw_trades": raw_trades,
                    "skipped_overlaps": skipped_overlaps,
                    **metrics,
                }
            )
        return exit_rule, raw_trades, len(trades), exit_rows

    for i, entry_rule in enumerate(entry_rules, start=1):
        entry_df = candidates[apply_entry_rule(candidates, entry_rule)].copy()
        if len(entry_df) < min_trades:
            continue
        print(f"entry {i}: {entry_rule.name} candidates={len(entry_df):,}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, grid_workers)) as executor:
            futures = [
                executor.submit(evaluate_one_exit, entry_df, entry_rule, exit_rule)
                for exit_rule in exit_rules_list
            ]
            for exit_idx, future in enumerate(as_completed(futures), start=1):
                exit_rule, raw_trades, kept_trades, exit_rows = future.result()
                rows.extend(exit_rows)
                if exit_idx % 100 == 0 or exit_idx == len(exit_rules_list):
                    print(
                        f"  {entry_rule.name}: exits {exit_idx}/{len(exit_rules_list)} "
                        f"latest={exit_rule.name} raw={raw_trades:,} kept={kept_trades:,}",
                        flush=True,
                    )
    return pd.DataFrame(rows)


def attach_split_to_trades(candidates: pd.DataFrame, daily_dir: Path, max_hold_days: int) -> pd.DataFrame:
    enriched = add_future_prices(candidates, daily_dir, max_hold_days=max_hold_days)
    return enriched.dropna(subset=["entry_open"]).copy()


def write_report(summary: pd.DataFrame, output_dir: Path, timestamp: str, min_oot_trades: int, position_policy: str) -> Path:
    report_path = output_dir / f"b1_xgb_entry_exit_grid_{position_policy}_report_{timestamp}.md"
    key_cols = [
        "entry_rule",
        "exit_rule",
        "trades",
        "avg_return_pct",
        "median_return_pct",
        "win_rate",
        "daily_sharpe",
        "max_drawdown_pct",
        "profit_factor",
        "stop_trigger",
        "stop_rate",
        "trailing_stop_rate",
        "expiry_rate",
    ]
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# B1 XGBoost 买入/卖出阈值探索\n\n")
        f.write("买入侧只使用模型概率阈值组合，不使用手工加权 entry_score。\n\n")
        f.write(f"持仓约束：`{position_policy}`。\n\n")
        for split in ["test", "oot", "train"]:
            part = summary[(summary["split"] == split) & (summary["trades"] >= min_oot_trades)].copy()
            if part.empty:
                continue
            f.write(f"## {split} Top 30 by max_drawdown + profit_factor\n\n")
            ranked = part.sort_values(["max_drawdown_pct", "profit_factor", "avg_return_pct"], ascending=[False, False, False]).head(30)
            f.write(ranked[key_cols].to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
            f.write(f"## {split} Top 30 by avg_return_pct\n\n")
            ranked = part.sort_values(["avg_return_pct", "profit_factor", "max_drawdown_pct"], ascending=[False, False, False]).head(30)
            f.write(ranked[key_cols].to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 XGBoost threshold entry/exit grid exploration")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-entry-candidates", type=int, default=50)
    parser.add_argument("--min-report-trades", type=int, default=50)
    parser.add_argument("--position-policy", choices=["all_signals", "non_overlap"], default="all_signals")
    parser.add_argument("--grid-workers", type=int, default=0, help="0 means auto: min(64, max(8, cpu_count*4))")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"loading dataset: {args.dataset}", flush=True)
    candidates = pd.read_parquet(args.dataset)
    candidates["date"] = pd.to_datetime(candidates["date"])
    candidates = predict_xgb_models(candidates, args.model_dir)

    max_hold_days = max(rule.hold_days for rule in build_exit_rules())
    print("adding future prices", flush=True)
    candidates = attach_split_to_trades(candidates, args.daily_dir, max_hold_days)

    grid_workers = args.grid_workers or min(64, max(8, (os.cpu_count() or 4) * 4))
    print(f"evaluating threshold grid with grid_workers={grid_workers}", flush=True)
    summary = evaluate_grid(
        candidates,
        build_entry_rules(),
        build_exit_rules(),
        args.min_entry_candidates,
        args.position_policy,
        grid_workers,
    )
    summary_path = args.output_dir / f"b1_xgb_entry_exit_grid_{args.position_policy}_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    report_path = write_report(summary, args.output_dir, timestamp, args.min_report_trades, args.position_policy)

    latest_summary = args.output_dir / f"latest_{args.position_policy}_summary.csv"
    latest_report = args.output_dir / f"latest_{args.position_policy}_report.md"
    summary.to_csv(latest_summary, index=False)
    latest_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"summary: {summary_path}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
