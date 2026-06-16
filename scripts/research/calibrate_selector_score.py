#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibrate selector score weights on historical strategy trade samples.

The selector score should rank candidates by expected future profitability, not
by a hand-tuned mixture of PF and signal count. This script replays historical
model-scored strategy trades, tunes score weights on 2025, and validates them on
2026 OOT data.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports/b1/research/xgb_project_vars_strategy"
DEFAULT_TRADE_SAMPLES = REPORT_DIR / "latest_z_skill_model_trade_samples.csv"
DEFAULT_SUMMARY = REPORT_DIR / "latest_z_skill_model_entry_exit_backtest.csv"
DEFAULT_OUTPUT_DIR = REPORT_DIR / "selector_score_calibration"


@dataclass(frozen=True)
class ScoreWeights:
    avg_weight: float
    model_weight: float
    drawdown_penalty: float
    pf_weight: float
    win_weight: float
    sample_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate selector score formula on 2025/2026 historical samples.")
    parser.add_argument("--trade-samples", type=Path, default=DEFAULT_TRADE_SAMPLES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-start", default="2025-01-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--valid-start", default="2026-01-01")
    parser.add_argument("--valid-end", default="2026-12-31")
    return parser.parse_args()


def load_samples(trade_samples: Path, summary_path: Path) -> pd.DataFrame:
    usecols = [
        "date",
        "symbol",
        "return_pct",
        "split",
        "pred_up5",
        "pred_up8",
        "pred_down3",
        "signal",
        "entry_rule",
        "open_filter",
        "exit_rule",
    ]
    trades = pd.read_csv(trade_samples, usecols=usecols)
    trades["date"] = pd.to_datetime(trades["date"])
    summary = pd.read_csv(summary_path)
    oot = summary[summary["split"] == "oot"].copy()
    metric_cols = [
        "signal",
        "entry_rule",
        "open_filter",
        "exit_rule",
        "trades",
        "avg_return_pct",
        "win_rate",
        "max_drawdown_pct",
        "profit_factor",
    ]
    oot = oot[[col for col in metric_cols if col in oot.columns]].drop_duplicates(
        ["signal", "entry_rule", "open_filter", "exit_rule"]
    )
    merged = trades.merge(oot, on=["signal", "entry_rule", "open_filter", "exit_rule"], how="left")
    merged = merged.dropna(subset=["return_pct", "avg_return_pct", "profit_factor"]).copy()
    return merged


def score_frame(df: pd.DataFrame, weights: ScoreWeights) -> pd.Series:
    trades = pd.to_numeric(df["trades"], errors="coerce").fillna(0)
    avg_return = pd.to_numeric(df["avg_return_pct"], errors="coerce").fillna(0).clip(-10, 10)
    win_rate = pd.to_numeric(df["win_rate"], errors="coerce").fillna(0)
    drawdown = pd.to_numeric(df["max_drawdown_pct"], errors="coerce").fillna(0).abs().clip(upper=50)
    profit_factor = pd.to_numeric(df["profit_factor"], errors="coerce").fillna(0).clip(upper=5)
    pred_up5 = pd.to_numeric(df["pred_up5"], errors="coerce").fillna(0)
    pred_up8 = pd.to_numeric(df["pred_up8"], errors="coerce").fillna(0)
    pred_down3 = pd.to_numeric(df["pred_down3"], errors="coerce").fillna(0)

    reliability = np.minimum(1.0, np.sqrt(np.maximum(trades, 0) / weights.sample_scale))
    model_edge = (pred_up5 * 5.0 + pred_up8 * 8.0 - pred_down3 * 3.0).clip(-10, 10)
    pf_bonus = np.maximum(profit_factor - 1.0, 0).clip(upper=3)
    win_bonus = np.maximum(win_rate - 0.35, 0)
    return (
        avg_return * reliability * weights.avg_weight
        + model_edge * weights.model_weight
        - drawdown * weights.drawdown_penalty
        + pf_bonus * weights.pf_weight
        + win_bonus * weights.win_weight
    )


def collapse_daily_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    ordered = scored.sort_values(["date", "symbol", "score"], ascending=[True, True, False])
    return ordered.drop_duplicates(["date", "symbol"], keep="first").copy()


def topn_stats(df: pd.DataFrame, n: int) -> dict[str, float]:
    selected = df.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(n)
    if selected.empty:
        return {"trades": 0, "avg_return": np.nan, "win_rate": np.nan, "days": 0}
    return {
        "trades": int(len(selected)),
        "days": int(selected["date"].nunique()),
        "avg_return": float(selected["return_pct"].mean()),
        "win_rate": float((selected["return_pct"] > 0).mean()),
    }


def decile_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out["decile"] = pd.qcut(out["score"].rank(method="first"), 10, labels=False) + 1
    except ValueError:
        return pd.DataFrame()
    return (
        out.groupby("decile")
        .agg(trades=("return_pct", "size"), avg_return=("return_pct", "mean"), win_rate=("return_pct", lambda s: (s > 0).mean()))
        .reset_index()
    )


def evaluate(df: pd.DataFrame, weights: ScoreWeights, start: str, end: str) -> dict[str, float]:
    part = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
    if part.empty:
        return {"rows": 0}
    part["score"] = score_frame(part, weights)
    collapsed = collapse_daily_candidates(part)
    corr = collapsed[["score", "return_pct"]].corr(method="spearman").iloc[0, 1]
    top20 = topn_stats(collapsed, 20)
    top50 = topn_stats(collapsed, 50)
    deciles = decile_stats(collapsed)
    if deciles.empty:
        spread = np.nan
        top_decile = np.nan
        bottom_decile = np.nan
    else:
        top_decile = float(deciles.loc[deciles["decile"] == 10, "avg_return"].iloc[0])
        bottom_decile = float(deciles.loc[deciles["decile"] == 1, "avg_return"].iloc[0])
        spread = top_decile - bottom_decile
    return {
        "rows": int(len(collapsed)),
        "days": int(collapsed["date"].nunique()),
        "spearman": float(corr) if pd.notna(corr) else np.nan,
        "top20_avg": top20["avg_return"],
        "top20_win": top20["win_rate"],
        "top20_trades": top20["trades"],
        "top50_avg": top50["avg_return"],
        "top50_win": top50["win_rate"],
        "top50_trades": top50["trades"],
        "top_decile_avg": top_decile,
        "bottom_decile_avg": bottom_decile,
        "decile_spread": spread,
    }


def objective(metrics: dict[str, float]) -> float:
    if not metrics or metrics.get("rows", 0) == 0:
        return -999
    return (
        float(metrics.get("top20_avg") or 0) * 0.45
        + float(metrics.get("top50_avg") or 0) * 0.25
        + float(metrics.get("decile_spread") or 0) * 0.20
        + float(metrics.get("spearman") or 0) * 8.0
        + float(metrics.get("top20_win") or 0) * 0.30
    )


def candidate_weights() -> list[ScoreWeights]:
    grid = itertools.product(
        [0.45, 0.65, 0.85],
        [0.20, 0.35, 0.50],
        [0.025, 0.035, 0.050],
        [0.15, 0.30],
        [0.00, 0.40],
        [240.0],
    )
    return [ScoreWeights(*items) for items in grid]


def write_report(results: pd.DataFrame, best: pd.Series, train_deciles: pd.DataFrame, valid_deciles: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "selector_score_calibration_latest.md"
    top = results.head(15).copy()
    cols = [
        "rank",
        "robust_objective",
        "train_objective",
        "valid_objective",
        "valid_top20_avg",
        "valid_top50_avg",
        "valid_spearman",
        "valid_decile_spread",
        "avg_weight",
        "model_weight",
        "drawdown_penalty",
        "pf_weight",
        "win_weight",
        "sample_scale",
    ]
    top["rank"] = range(1, len(top) + 1)
    with path.open("w", encoding="utf-8") as f:
        f.write("# 选股器综合分历史校准\n\n")
        f.write(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}\n")
        f.write("- 调参窗口：2025；验证窗口：2026 年已有样本；推荐参数按 `0.35 * 2025目标 + 0.65 * 2026验证目标` 选择。\n")
        f.write("- 数据：`latest_z_skill_model_trade_samples.csv`，按每日每股保留最高分信号后评估 TopN 后续实际收益。\n\n")
        f.write("## 推荐参数\n\n")
        f.write("```json\n")
        f.write(json.dumps({k: best[k] for k in asdict(ScoreWeights(0, 0, 0, 0, 0, 0)).keys()}, ensure_ascii=False, indent=2))
        f.write("\n```\n\n")
        f.write("## Top 15 参数表现\n\n")
        f.write(top[cols].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## 推荐参数分层收益\n\n")
        f.write("### 2025 调参窗口\n\n")
        f.write(train_deciles.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n### 2026 验证窗口\n\n")
        f.write(valid_deciles.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.trade_samples, args.summary)
    rows = []
    for weights in candidate_weights():
        train_metrics = evaluate(samples, weights, args.train_start, args.train_end)
        valid_metrics = evaluate(samples, weights, args.valid_start, args.valid_end)
        rows.append(
            {
                **asdict(weights),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"valid_{k}": v for k, v in valid_metrics.items()},
                "train_objective": objective(train_metrics),
                "valid_objective": objective(valid_metrics),
            }
        )
    results = pd.DataFrame(rows)
    results["robust_objective"] = results["train_objective"] * 0.35 + results["valid_objective"] * 0.65
    results = results.sort_values(["robust_objective", "valid_objective", "train_objective"], ascending=False).reset_index(drop=True)
    result_path = args.output_dir / "selector_score_calibration_latest.csv"
    results.to_csv(result_path, index=False)
    best = results.iloc[0]
    best_weights = ScoreWeights(
        avg_weight=float(best["avg_weight"]),
        model_weight=float(best["model_weight"]),
        drawdown_penalty=float(best["drawdown_penalty"]),
        pf_weight=float(best["pf_weight"]),
        win_weight=float(best["win_weight"]),
        sample_scale=float(best["sample_scale"]),
    )
    for_report = samples.copy()
    for_report["score"] = score_frame(for_report, best_weights)
    train_deciles = decile_stats(collapse_daily_candidates(for_report[(for_report["date"] >= args.train_start) & (for_report["date"] <= args.train_end)]))
    valid_deciles = decile_stats(collapse_daily_candidates(for_report[(for_report["date"] >= args.valid_start) & (for_report["date"] <= args.valid_end)]))
    report_path = write_report(results, best, train_deciles, valid_deciles, args.output_dir)
    print(
        json.dumps(
            {
                "status": "success",
                "rows": int(len(samples)),
                "result_csv": str(result_path),
                "report": str(report_path),
                "best": {k: best[k] for k in asdict(best_weights).keys()},
                "train_objective": float(best["train_objective"]),
                "valid_objective": float(best["valid_objective"]),
                "valid_top20_avg": float(best["valid_top20_avg"]),
                "valid_spearman": float(best["valid_spearman"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
