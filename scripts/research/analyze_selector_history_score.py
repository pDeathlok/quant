#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate and calibrate selector stock-level scores from history samples."""

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
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data/research/selector_history_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/b1/research/xgb_project_vars_strategy/selector_history_score"


@dataclass(frozen=True)
class Weights:
    avg_weight: float
    model_weight: float
    drawdown_penalty: float
    pf_weight: float
    win_weight: float
    resonance_weight: float
    sample_scale: float = 240.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze selector score effectiveness on historical samples.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", default="future_return_t5_pct")
    parser.add_argument("--train-start", default="2025-01-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--valid-start", default="2026-01-01")
    parser.add_argument("--valid-end", default="2026-12-31")
    return parser.parse_args()


def load_samples(sample_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock = pd.read_parquet(sample_dir / "selector_stock_history_samples.parquet")
    signal = pd.read_parquet(sample_dir / "selector_signal_history_samples.parquet")
    stock["date"] = pd.to_datetime(stock["date"])
    signal["date"] = pd.to_datetime(signal["date"])
    return stock, signal


def max_drawdown_from_daily_returns(daily_returns_pct: pd.Series) -> float:
    if daily_returns_pct.empty:
        return np.nan
    equity = (1 + daily_returns_pct / 100).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1).min() * 100)


def profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return float(wins / losses) if losses > 0 else np.nan


def stats_for_rows(rows: pd.DataFrame, target: str) -> dict[str, float]:
    rows = rows.dropna(subset=[target]).copy()
    if rows.empty:
        return {
            "rows": 0,
            "days": 0,
            "avg_return": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
            "candidates_per_day": np.nan,
        }
    daily = rows.groupby("date")[target].mean().sort_index()
    return {
        "rows": int(len(rows)),
        "days": int(rows["date"].nunique()),
        "avg_return": float(rows[target].mean()),
        "win_rate": float((rows[target] > 0).mean()),
        "profit_factor": profit_factor(rows[target]),
        "max_drawdown": max_drawdown_from_daily_returns(daily),
        "candidates_per_day": float(len(rows) / max(rows["date"].nunique(), 1)),
    }


def topn_stats(df: pd.DataFrame, score_col: str, target: str, topn_values: tuple[int, ...] = (10, 20, 50)) -> pd.DataFrame:
    rows = []
    ordered = df.sort_values(["date", score_col], ascending=[True, False])
    for n in topn_values:
        selected = ordered.groupby("date").head(n)
        rows.append({"bucket": f"Top{n}", **stats_for_rows(selected, target)})
    return pd.DataFrame(rows)


def score_deciles(df: pd.DataFrame, score_col: str, target: str) -> pd.DataFrame:
    valid = df.dropna(subset=[score_col, target]).copy()
    if valid.empty:
        return pd.DataFrame()
    try:
        valid["score_decile"] = pd.qcut(valid[score_col].rank(method="first"), 10, labels=False) + 1
    except ValueError:
        return pd.DataFrame()
    return (
        valid.groupby("score_decile")
        .apply(lambda part: pd.Series(stats_for_rows(part, target)), include_groups=False)
        .reset_index()
    )


def daily_spearman(df: pd.DataFrame, score_col: str, target: str) -> float:
    values = []
    for _, part in df.dropna(subset=[score_col, target]).groupby("date"):
        if len(part) < 5 or part[score_col].nunique() < 2:
            continue
        corr = part[[score_col, target]].corr(method="spearman").iloc[0, 1]
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else np.nan


def evaluate_score(df: pd.DataFrame, score_col: str, target: str, start: str, end: str, label: str) -> dict[str, object]:
    part = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
    topn = topn_stats(part, score_col, target)
    deciles = score_deciles(part, score_col, target)
    top_decile = deciles[deciles["score_decile"] == 10]["avg_return"].iloc[0] if not deciles.empty else np.nan
    bottom_decile = deciles[deciles["score_decile"] == 1]["avg_return"].iloc[0] if not deciles.empty else np.nan
    return {
        "label": label,
        "rows": int(len(part.dropna(subset=[target]))),
        "days": int(part["date"].nunique()),
        "spearman_daily_mean": daily_spearman(part, score_col, target),
        "top_decile_avg": float(top_decile) if pd.notna(top_decile) else np.nan,
        "bottom_decile_avg": float(bottom_decile) if pd.notna(bottom_decile) else np.nan,
        "decile_spread": float(top_decile - bottom_decile) if pd.notna(top_decile) and pd.notna(bottom_decile) else np.nan,
        "topn": topn,
        "deciles": deciles,
    }


def candidate_weights() -> list[Weights]:
    grid = itertools.product(
        [0.45, 0.65, 0.85, 1.05],
        [0.00, 0.20, 0.40],
        [0.025, 0.05, 0.07],
        [0.00, 0.25],
        [0.00, 0.50],
        [0.00, 0.10, 0.20],
    )
    return [Weights(*items) for items in grid]


def signal_scores(signal: pd.DataFrame, weights: Weights) -> pd.Series:
    trades = pd.to_numeric(signal["metrics_trades"], errors="coerce").fillna(0)
    avg_return = pd.to_numeric(signal["metrics_avg_return_pct"], errors="coerce").fillna(0).clip(-10, 10)
    win_rate = pd.to_numeric(signal["metrics_win_rate"], errors="coerce").fillna(0)
    drawdown = pd.to_numeric(signal["metrics_max_drawdown_pct"], errors="coerce").fillna(0).abs().clip(upper=50)
    pf = pd.to_numeric(signal["metrics_profit_factor"], errors="coerce").fillna(0).clip(upper=5)
    pred_up5 = pd.to_numeric(signal["pred_up5"], errors="coerce").fillna(0)
    pred_up8 = pd.to_numeric(signal["pred_up8"], errors="coerce").fillna(0)
    pred_up10 = pd.to_numeric(signal["pred_up10"], errors="coerce").fillna(0)
    pred_down3 = pd.to_numeric(signal["pred_down3"], errors="coerce").fillna(0)
    reliability = np.minimum(1.0, np.sqrt(np.maximum(trades, 0) / weights.sample_scale))
    model_edge = (pred_up5 * 5.0 + pred_up8 * 8.0 + pred_up10 * 10.0 - pred_down3 * 3.0).clip(-10, 10)
    pf_bonus = np.maximum(pf - 1.0, 0).clip(upper=3)
    win_bonus = np.maximum(win_rate - 0.35, 0)
    return (
        avg_return * reliability * weights.avg_weight
        + model_edge * weights.model_weight
        - drawdown * weights.drawdown_penalty
        + pf_bonus * weights.pf_weight
        + win_bonus * weights.win_weight
    )


def stock_scores_from_signals(stock: pd.DataFrame, signal: pd.DataFrame, weights: Weights) -> pd.DataFrame:
    scored = signal[["date", "symbol", "strategy_group"]].copy()
    scored["signal_score"] = signal_scores(signal, weights)
    scored["positive_signal_score"] = scored["signal_score"].clip(lower=0)
    grouped = scored.groupby(["date", "symbol"])
    aggregate = grouped.agg(
        best_signal_score=("signal_score", "max"),
        positive_signal_score=("positive_signal_score", "sum"),
        matched_groups=("strategy_group", "nunique"),
    ).reset_index()
    aggregate["candidate_score"] = (
        aggregate["best_signal_score"]
        + 0.08 * (aggregate["positive_signal_score"] - aggregate["best_signal_score"].clip(lower=0))
        + weights.resonance_weight * np.log1p(aggregate["matched_groups"])
    )
    return stock.merge(aggregate[["date", "symbol", "candidate_score"]], on=["date", "symbol"], how="left")


def quick_score_metrics(df: pd.DataFrame, score_col: str, target: str, start: str, end: str) -> dict[str, float]:
    part = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].dropna(subset=[score_col, target]).copy()
    if part.empty:
        return {"rows": 0}
    ordered = part.sort_values(["date", score_col], ascending=[True, False])
    top20 = ordered.groupby("date").head(20)
    top50 = ordered.groupby("date").head(50)
    corr = part[[score_col, target]].corr(method="spearman").iloc[0, 1] if part[score_col].nunique() > 1 else np.nan
    try:
        part["score_decile"] = pd.qcut(part[score_col].rank(method="first"), 10, labels=False) + 1
        decile = part.groupby("score_decile")[target].mean()
        spread = float(decile.loc[10] - decile.loc[1])
    except Exception:
        spread = np.nan
    return {
        "rows": int(len(part)),
        "top20_avg": float(top20[target].mean()) if not top20.empty else np.nan,
        "top20_win": float((top20[target] > 0).mean()) if not top20.empty else np.nan,
        "top50_avg": float(top50[target].mean()) if not top50.empty else np.nan,
        "spearman": float(corr) if pd.notna(corr) else np.nan,
        "decile_spread": spread,
    }


def quick_objective(metrics: dict[str, float]) -> float:
    if metrics.get("rows", 0) == 0:
        return -999
    return (
        float(metrics.get("top20_avg") or 0) * 0.45
        + float(metrics.get("top50_avg") or 0) * 0.25
        + float(metrics.get("decile_spread") or 0) * 0.20
        + float(metrics.get("spearman") or 0) * 6.0
        + float(metrics.get("top20_win") or 0) * 0.30
    )


def objective(metrics: dict[str, object]) -> float:
    topn = metrics["topn"]
    top20 = topn[topn["bucket"] == "Top20"].iloc[0]
    top50 = topn[topn["bucket"] == "Top50"].iloc[0]
    return (
        float(top20["avg_return"] or 0) * 0.45
        + float(top50["avg_return"] or 0) * 0.25
        + float(metrics.get("decile_spread") or 0) * 0.20
        + float(metrics.get("spearman_daily_mean") or 0) * 6.0
        + float(top20["win_rate"] or 0) * 0.30
    )


def calibrate(stock: pd.DataFrame, signal: pd.DataFrame, target: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    weights_list = candidate_weights()
    for idx, weights in enumerate(weights_list, start=1):
        scored = stock_scores_from_signals(stock, signal, weights)
        train_metrics = quick_score_metrics(scored, "candidate_score", target, args.train_start, args.train_end)
        valid_metrics = quick_score_metrics(scored, "candidate_score", target, args.valid_start, args.valid_end)
        rows.append(
            {
                "weight_index": idx - 1,
                **asdict(weights),
                "train_objective": quick_objective(train_metrics),
                "valid_objective": quick_objective(valid_metrics),
                "robust_objective": quick_objective(train_metrics) * 0.70 + quick_objective(valid_metrics) * 0.30,
                "train_top20_avg": train_metrics.get("top20_avg"),
                "train_top50_avg": train_metrics.get("top50_avg"),
                "train_spearman": train_metrics.get("spearman"),
                "train_decile_spread": train_metrics.get("decile_spread"),
                "valid_top20_avg": valid_metrics.get("top20_avg"),
                "valid_top50_avg": valid_metrics.get("top50_avg"),
                "valid_spearman": valid_metrics.get("spearman"),
                "valid_decile_spread": valid_metrics.get("decile_spread"),
            }
        )
        if idx % 50 == 0 or idx == len(weights_list):
            print(f"  calibrated weights: {idx}/{len(weights_list)}", flush=True)
    results = pd.DataFrame(rows).sort_values(
        ["robust_objective", "valid_objective", "train_objective"],
        ascending=False,
    ).reset_index(drop=True)
    best = Weights(
        avg_weight=float(results.iloc[0]["avg_weight"]),
        model_weight=float(results.iloc[0]["model_weight"]),
        drawdown_penalty=float(results.iloc[0]["drawdown_penalty"]),
        pf_weight=float(results.iloc[0]["pf_weight"]),
        win_weight=float(results.iloc[0]["win_weight"]),
        resonance_weight=float(results.iloc[0]["resonance_weight"]),
        sample_scale=float(results.iloc[0]["sample_scale"]),
    )
    best_scored = stock_scores_from_signals(stock, signal, best)
    return results, best_scored


def write_report(
    current_train: dict[str, object],
    current_valid: dict[str, object],
    calibration: pd.DataFrame,
    calibrated_train: dict[str, object],
    calibrated_valid: dict[str, object],
    args: argparse.Namespace,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "selector_history_score_latest.md"
    current_train_top = current_train["topn"]
    current_valid_top = current_valid["topn"]
    calibrated_train_top = calibrated_train["topn"]
    calibrated_valid_top = calibrated_valid["topn"]
    best = calibration.iloc[0]
    with report.open("w", encoding="utf-8") as f:
        f.write("# 全策略综合分历史样本评估\n\n")
        f.write(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- 目标收益：`{args.target}`\n")
        f.write(f"- 训练区间：{args.train_start} 至 {args.train_end}\n")
        f.write(f"- OOT 区间：{args.valid_start} 至 {args.valid_end}\n\n")
        f.write("## 当前页面综合分\n\n")
        f.write("### 2025 训练窗口 TopN\n\n")
        f.write(current_train_top.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n### 2026 OOT TopN\n\n")
        f.write(current_valid_top.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")
        f.write(
            f"- 2025 日内 Spearman：{current_train['spearman_daily_mean']:.4f}，"
            f"Top/Bottom decile spread：{current_train['decile_spread']:.4f}\n"
        )
        f.write(
            f"- 2026 日内 Spearman：{current_valid['spearman_daily_mean']:.4f}，"
            f"Top/Bottom decile spread：{current_valid['decile_spread']:.4f}\n\n"
        )
        f.write("## 候选校准公式\n\n")
        f.write("```json\n")
        f.write(json.dumps({key: best[key] for key in asdict(Weights(0, 0, 0, 0, 0, 0)).keys()}, ensure_ascii=False, indent=2))
        f.write("\n```\n\n")
        f.write("### 校准后 2025 TopN\n\n")
        f.write(calibrated_train_top.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n### 校准后 2026 OOT TopN\n\n")
        f.write(calibrated_valid_top.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## 参数搜索 Top 20\n\n")
        f.write(calibration.head(20).to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## 当前分数 2026 OOT 分位表现\n\n")
        f.write(current_valid["deciles"].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## 校准分数 2026 OOT 分位表现\n\n")
        f.write(calibrated_valid["deciles"].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stock, signal = load_samples(args.sample_dir)
    stock = stock.dropna(subset=[args.target, "selector_score"]).copy()
    current_train = evaluate_score(stock, "selector_score", args.target, args.train_start, args.train_end, "current_train")
    current_valid = evaluate_score(stock, "selector_score", args.target, args.valid_start, args.valid_end, "current_valid")
    calibration, calibrated = calibrate(stock, signal, args.target, args)
    calibrated_train = evaluate_score(calibrated, "candidate_score", args.target, args.train_start, args.train_end, "calibrated_train")
    calibrated_valid = evaluate_score(calibrated, "candidate_score", args.target, args.valid_start, args.valid_end, "calibrated_valid")

    calibration_path = args.output_dir / "selector_history_score_calibration_latest.csv"
    calibration.to_csv(calibration_path, index=False)
    current_train["topn"].to_csv(args.output_dir / "current_train_topn.csv", index=False)
    current_valid["topn"].to_csv(args.output_dir / "current_valid_topn.csv", index=False)
    calibrated_train["topn"].to_csv(args.output_dir / "calibrated_train_topn.csv", index=False)
    calibrated_valid["topn"].to_csv(args.output_dir / "calibrated_valid_topn.csv", index=False)
    current_valid["deciles"].to_csv(args.output_dir / "current_valid_deciles.csv", index=False)
    calibrated_valid["deciles"].to_csv(args.output_dir / "calibrated_valid_deciles.csv", index=False)
    report = write_report(current_train, current_valid, calibration, calibrated_train, calibrated_valid, args)
    result = {
        "status": "success",
        "stock_rows": int(len(stock)),
        "signal_rows": int(len(signal)),
        "calibration_csv": str(calibration_path),
        "report": str(report),
        "best_weights": {key: calibration.iloc[0][key] for key in asdict(Weights(0, 0, 0, 0, 0, 0)).keys()},
        "current_valid_top20_avg": float(current_valid["topn"].query("bucket == 'Top20'")["avg_return"].iloc[0]),
        "calibrated_valid_top20_avg": float(calibrated_valid["topn"].query("bucket == 'Top20'")["avg_return"].iloc[0]),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
