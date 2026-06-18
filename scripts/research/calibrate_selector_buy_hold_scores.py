#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibrate web selector buy/hold scores against historical outcomes."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data/research/selector_history_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/b1/research/xgb_project_vars_strategy/selector_buy_hold_score"
DEFAULT_ARTIFACT = DEFAULT_OUTPUT_DIR / "selector_buy_hold_score_calibration_latest.json"
DEFAULT_CONFIG_ARTIFACT = PROJECT_ROOT / "config/selector_buy_hold_score_calibration.json"


@dataclass(frozen=True)
class ScoreWeights:
    avg_weight: float
    model_weight: float
    drawdown_penalty: float
    pf_weight: float
    win_weight: float
    group_weight: float
    resonance_weight: float
    sample_scale: float = 240.0


TARGETS = {
    "buy": {
        "target": "future_max_high_t5_pct",
        "hit_threshold": 5.0,
        "hit_label": "future_max_high_t5_pct >= 5%",
    },
    "hold": {
        "target": "future_return_t5_pct",
        "hit_threshold": 0.0,
        "hit_label": "future_return_t5_pct > 0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate selector buy/hold scores on historical samples.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-artifact", type=Path, default=DEFAULT_CONFIG_ARTIFACT)
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
    label_cols = ["date", "symbol", "future_max_high_t5_pct", "future_return_t5_pct"]
    signal = signal.merge(stock[label_cols], on=["date", "symbol"], how="inner")
    return stock, signal


def group_edges(signal: pd.DataFrame, target: str, train_start: str, train_end: str) -> dict[str, float]:
    train = signal[(signal["date"] >= pd.Timestamp(train_start)) & (signal["date"] <= pd.Timestamp(train_end))]
    train = train.dropna(subset=[target, "strategy_group"]).copy()
    if train.empty:
        return {}
    global_mean = float(train[target].mean())
    global_std = float(train[target].std(ddof=0)) or 1.0
    grouped = train.groupby("strategy_group").agg(rows=(target, "size"), avg=(target, "mean"))
    grouped["edge"] = ((grouped["avg"] - global_mean) / global_std).clip(-1.5, 1.5)
    grouped.loc[grouped["rows"] < 80, "edge"] = grouped.loc[grouped["rows"] < 80, "edge"] * 0.35
    return {str(group): float(row["edge"]) for group, row in grouped.iterrows()}


def raw_score(frame: pd.DataFrame, weights: ScoreWeights, edges: dict[str, float]) -> pd.Series:
    trades = pd.to_numeric(frame["metrics_trades"], errors="coerce").fillna(0)
    avg_return = pd.to_numeric(frame["metrics_avg_return_pct"], errors="coerce").fillna(0).clip(-10, 10)
    win_rate = pd.to_numeric(frame["metrics_win_rate"], errors="coerce").fillna(0)
    drawdown = pd.to_numeric(frame["metrics_max_drawdown_pct"], errors="coerce").fillna(0).abs().clip(upper=50)
    pf = pd.to_numeric(frame["metrics_profit_factor"], errors="coerce").fillna(0).clip(upper=5)
    pred_up5 = pd.to_numeric(frame["pred_up5"], errors="coerce").fillna(0)
    pred_up8 = pd.to_numeric(frame["pred_up8"], errors="coerce").fillna(0)
    pred_up10 = pd.to_numeric(frame["pred_up10"], errors="coerce").fillna(0)
    pred_down3 = pd.to_numeric(frame["pred_down3"], errors="coerce").fillna(0)
    group_edge = frame["strategy_group"].astype(str).map(edges).fillna(0)

    reliability = np.minimum(1.0, np.sqrt(np.maximum(trades, 0) / weights.sample_scale))
    model_edge = (pred_up5 * 5.0 + pred_up8 * 8.0 + pred_up10 * 10.0 - pred_down3 * 3.0).clip(-10, 10)
    pf_bonus = np.maximum(pf - 1.0, 0).clip(upper=4)
    win_bonus = np.maximum(win_rate - 0.35, 0)
    return (
        avg_return * reliability * weights.avg_weight
        + model_edge * weights.model_weight
        + pf_bonus * weights.pf_weight
        + win_bonus * weights.win_weight
        + group_edge * weights.group_weight
        - drawdown * weights.drawdown_penalty
    )


def aggregate_signal_scores(signal: pd.DataFrame, weights: ScoreWeights, edges: dict[str, float]) -> pd.DataFrame:
    scored = signal[["date", "symbol", "strategy_group"]].copy()
    scored["signal_score"] = raw_score(signal, weights, edges)
    scored = scored.sort_values(["date", "symbol", "signal_score"], ascending=[True, True, False])
    scored["signal_rank"] = scored.groupby(["date", "symbol"]).cumcount()
    top = scored[scored["signal_rank"] == 0][["date", "symbol", "signal_score"]].rename(
        columns={"signal_score": "best_score"}
    )
    extra = scored[(scored["signal_rank"] > 0) & (scored["signal_rank"] <= 2)].copy()
    extra["extra_positive"] = extra["signal_score"].clip(lower=0)
    extra_sum = extra.groupby(["date", "symbol"], sort=False)["extra_positive"].sum().rename("extra_positive").reset_index()
    group_count = scored.groupby(["date", "symbol"], sort=False)["strategy_group"].nunique().rename("matched_groups").reset_index()
    aggregate = top.merge(extra_sum, on=["date", "symbol"], how="left").merge(group_count, on=["date", "symbol"], how="left")
    aggregate["extra_positive"] = aggregate["extra_positive"].fillna(0)
    aggregate["candidate_score"] = (
        aggregate["best_score"]
        + 0.08 * aggregate["extra_positive"]
        + weights.resonance_weight * np.log1p(aggregate["matched_groups"])
    )
    return aggregate[["date", "symbol", "candidate_score", "matched_groups"]]


def stats_for_rows(rows: pd.DataFrame, target: str, hit_threshold: float) -> dict[str, float]:
    rows = rows.dropna(subset=[target])
    if rows.empty:
        return {"rows": 0, "days": 0}
    hit = rows[target] > hit_threshold
    return {
        "rows": int(len(rows)),
        "days": int(rows["date"].nunique()),
        "avg_return": float(rows[target].mean()),
        "hit_rate": float(hit.mean()),
    }


def evaluate(scored: pd.DataFrame, target: str, hit_threshold: float, start: str, end: str) -> dict[str, Any]:
    part = scored[(scored["date"] >= pd.Timestamp(start)) & (scored["date"] <= pd.Timestamp(end))]
    part = part.dropna(subset=["candidate_score", target]).copy()
    if part.empty:
        return {"rows": 0}
    ordered = part.sort_values(["date", "candidate_score"], ascending=[True, False])
    top10 = ordered.groupby("date").head(10)
    top20 = ordered.groupby("date").head(20)
    top50 = ordered.groupby("date").head(50)
    corr = part[["candidate_score", target]].corr(method="spearman").iloc[0, 1]
    try:
        part["score_decile"] = pd.qcut(part["candidate_score"].rank(method="first"), 10, labels=False) + 1
        part["hit"] = part[target] > hit_threshold
        deciles = part.groupby("score_decile").agg(
            rows=(target, "size"),
            days=("date", "nunique"),
            avg_return=(target, "mean"),
            hit_rate=("hit", "mean"),
        ).reset_index()
        spread = float(deciles.loc[deciles["score_decile"] == 10, "avg_return"].iloc[0] - deciles.loc[deciles["score_decile"] == 1, "avg_return"].iloc[0])
        hit_spread = float(deciles.loc[deciles["score_decile"] == 10, "hit_rate"].iloc[0] - deciles.loc[deciles["score_decile"] == 1, "hit_rate"].iloc[0])
    except Exception:
        deciles = pd.DataFrame()
        spread = np.nan
        hit_spread = np.nan
    return {
        "rows": int(len(part)),
        "days": int(part["date"].nunique()),
        "spearman": float(corr) if pd.notna(corr) else np.nan,
        "top10": stats_for_rows(top10, target, hit_threshold),
        "top20": stats_for_rows(top20, target, hit_threshold),
        "top50": stats_for_rows(top50, target, hit_threshold),
        "decile_spread": spread,
        "hit_spread": hit_spread,
        "deciles": deciles,
    }


def objective(metrics: dict[str, Any]) -> float:
    if not metrics or metrics.get("rows", 0) == 0:
        return -999.0
    top20 = metrics["top20"]
    top50 = metrics["top50"]
    return (
        float(top20.get("avg_return") or 0) * 0.40
        + float(top50.get("avg_return") or 0) * 0.20
        + float(metrics.get("decile_spread") or 0) * 0.20
        + float(metrics.get("spearman") or 0) * 6.0
        + float(top20.get("hit_rate") or 0) * 0.50
        + float(metrics.get("hit_spread") or 0) * 1.20
    )


def candidate_weights(mode: str) -> list[ScoreWeights]:
    if mode == "buy":
        grid = itertools.product(
            [1.05, 1.25],
            [0.0, 0.25, 0.45],
            [0.025, 0.045],
            [0.0, 0.2],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.1, 0.2],
        )
    else:
        grid = itertools.product(
            [1.05, 1.25],
            [0.0, 0.15],
            [0.045, 0.065],
            [0.0, 0.2],
            [0.0, 0.8],
            [0.0, 1.0],
            [0.1, 0.2],
        )
    return [ScoreWeights(*items) for items in grid]


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "deciles"}


def calibrate_mode(stock: pd.DataFrame, signal: pd.DataFrame, mode: str, args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    target = TARGETS[mode]["target"]
    hit_threshold = TARGETS[mode]["hit_threshold"]
    edges = group_edges(signal, target, args.train_start, args.train_end)
    labels = stock[["date", "symbol", target]].copy()
    rows = []
    best_payload: dict[str, Any] | None = None
    best_deciles = pd.DataFrame()
    for weights in candidate_weights(mode):
        scored = aggregate_signal_scores(signal, weights, edges).merge(labels, on=["date", "symbol"], how="left")
        train_metrics = evaluate(scored, target, hit_threshold, args.train_start, args.train_end)
        valid_metrics = evaluate(scored, target, hit_threshold, args.valid_start, args.valid_end)
        train_objective = objective(train_metrics)
        valid_objective = objective(valid_metrics)
        robust_objective = train_objective * 0.35 + valid_objective * 0.65
        row = {
            **asdict(weights),
            "train_objective": train_objective,
            "valid_objective": valid_objective,
            "robust_objective": robust_objective,
            "train_top20_avg": (train_metrics.get("top20") or {}).get("avg_return"),
            "train_top20_hit": (train_metrics.get("top20") or {}).get("hit_rate"),
            "valid_top20_avg": (valid_metrics.get("top20") or {}).get("avg_return"),
            "valid_top20_hit": (valid_metrics.get("top20") or {}).get("hit_rate"),
            "valid_spearman": valid_metrics.get("spearman"),
            "valid_decile_spread": valid_metrics.get("decile_spread"),
            "valid_hit_spread": valid_metrics.get("hit_spread"),
        }
        rows.append(row)
        if best_payload is None or robust_objective > best_payload["robust_objective"]:
            best_payload = {
                **row,
                "weights": asdict(weights),
                "group_edges": edges,
                "train_metrics": compact_metrics(train_metrics),
                "valid_metrics": compact_metrics(valid_metrics),
            }
            best_deciles = valid_metrics.get("deciles", pd.DataFrame())
    assert best_payload is not None
    results = pd.DataFrame(rows).sort_values(
        ["robust_objective", "valid_objective", "train_objective"],
        ascending=False,
    )
    return best_payload, results, best_deciles


def write_markdown(payload: dict[str, Any], tables: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    path = output_dir / "selector_buy_hold_score_calibration_latest.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# 短线策略买入分 / 持有分历史校准\n\n")
        f.write(f"- 生成时间：{payload['generated_at']}\n")
        f.write(f"- 训练窗口：{payload['train_window'][0]} 至 {payload['train_window'][1]}\n")
        f.write(f"- OOT 验证窗口：{payload['valid_window'][0]} 至 {payload['valid_window'][1]}\n")
        f.write("- 买入分目标：未来 5 日最大冲高收益与冲高 >= 5% 概率。\n")
        f.write("- 持有分目标：未来 5 日收盘收益与正收益概率。\n\n")
        for mode, title in [("buy", "买入分"), ("hold", "持有分")]:
            mode_payload = payload["modes"][mode]
            f.write(f"## {title}\n\n")
            f.write("### 推荐参数\n\n```json\n")
            f.write(json.dumps(mode_payload["weights"], ensure_ascii=False, indent=2))
            f.write("\n```\n\n")
            f.write("### OOT TopN\n\n")
            f.write(pd.DataFrame(
                [
                    {"bucket": key, **value}
                    for key, value in mode_payload["valid_metrics"].items()
                    if key in {"top10", "top20", "top50"}
                ]
            ).to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n### OOT 分位\n\n")
            f.write(tables[f"{mode}_deciles"].to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n### 参数搜索 Top 15\n\n")
            f.write(tables[f"{mode}_results"].head(15).to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stock, signal = load_samples(args.sample_dir)
    payload: dict[str, Any] = {
        "schema_version": "selector_buy_hold_score_calibration_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_dir": str(args.sample_dir),
        "train_window": [args.train_start, args.train_end],
        "valid_window": [args.valid_start, args.valid_end],
        "targets": TARGETS,
        "modes": {},
    }
    tables: dict[str, pd.DataFrame] = {}
    for mode in ["buy", "hold"]:
        best, results, deciles = calibrate_mode(stock, signal, mode, args)
        payload["modes"][mode] = best
        tables[f"{mode}_results"] = results
        tables[f"{mode}_deciles"] = deciles
        results.to_csv(args.output_dir / f"{mode}_calibration_results.csv", index=False)
        deciles.to_csv(args.output_dir / f"{mode}_valid_deciles.csv", index=False)
    artifact = args.output_dir / "selector_buy_hold_score_calibration_latest.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    config_payload = dict(payload)
    config_payload.pop("sample_dir", None)
    args.config_artifact.parent.mkdir(parents=True, exist_ok=True)
    args.config_artifact.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    report = write_markdown(payload, tables, args.output_dir)
    print(json.dumps({"status": "success", "artifact": str(artifact), "report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
