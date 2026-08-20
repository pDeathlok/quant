#!/usr/bin/env python
"""Compare daily-only and point-in-time enriched B1 research variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


METRICS = (
    "trades",
    "avg_return_pct",
    "win_rate",
    "profit_factor",
    "daily_sharpe",
    "daily_return_lcb95_pct",
    "max_drawdown_pct",
)


def _variant_columns(frame: pd.DataFrame, keys: list[str], variant: str) -> pd.DataFrame:
    columns = [*keys, *[column for column in METRICS if column in frame]]
    return frame[columns].rename(
        columns={column: f"{variant}_{column}" for column in METRICS if column in frame}
    )


def compare_selected(baseline_dir: Path, enriched_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(baseline_dir / "selected_by_test.csv")
    enriched = pd.read_csv(enriched_dir / "selected_by_test.csv")
    baseline_oot = baseline[baseline["split"] == "oot"]
    enriched_oot = enriched[enriched["split"] == "oot"]
    oot = _variant_columns(baseline_oot, ["exit_rule"], "base").merge(
        _variant_columns(enriched_oot, ["exit_rule"], "enriched"),
        on="exit_rule",
        how="outer",
        validate="one_to_one",
    )

    baseline_years = pd.read_csv(baseline_dir / "selected_periods.csv")
    enriched_years = pd.read_csv(enriched_dir / "selected_periods.csv")
    baseline_years = baseline_years[baseline_years["period"].isin(["oot_2025", "oot_2026"])]
    enriched_years = enriched_years[enriched_years["period"].isin(["oot_2025", "oot_2026"])]
    years = _variant_columns(baseline_years, ["exit_rule", "period"], "base").merge(
        _variant_columns(enriched_years, ["exit_rule", "period"], "enriched"),
        on=["exit_rule", "period"],
        how="outer",
        validate="one_to_one",
    )
    return oot, years


def compare_model_quality(baseline_dir: Path, enriched_dir: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_dir / "model_quality.csv").rename(
        columns={column: f"daily_{column}" for column in ("auc", "pr_auc", "brier_score", "log_loss")}
    )
    enriched = pd.read_csv(enriched_dir / "model_quality.csv").rename(
        columns={column: f"enriched_{column}" for column in ("auc", "pr_auc", "brier_score", "log_loss")}
    )
    merged = baseline.merge(enriched, on=["model", "split"], suffixes=("_daily", "_enriched"), validate="one_to_one")
    for metric in ("auc", "pr_auc", "brier_score", "log_loss"):
        merged[f"delta_{metric}"] = merged[f"enriched_{metric}"] - merged[f"daily_{metric}"]
    return merged


def enriched_top_features(training_report: Path) -> pd.DataFrame:
    payload = json.loads(training_report.read_text(encoding="utf-8"))
    base = set(PROJECT_FACTOR_COLUMNS)
    rows = []
    for model_name, report in payload.items():
        if model_name == "dataset":
            continue
        for rank, item in enumerate(report.get("top_features", []), start=1):
            if item["feature"] not in base:
                rows.append(
                    {
                        "model": model_name,
                        "rank": rank,
                        "feature": item["feature"],
                        "importance": item["importance"],
                    }
                )
    return pd.DataFrame(rows)


def production_oot_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[frame["period"] == "oot_2025plus"][
        ["combo", "trades", "avg_return_pct", "win_rate", "profit_factor", "daily_sharpe", "max_drawdown_pct"]
    ]


def compare_activity(baseline_dir: Path, enriched_dir: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_dir / "stable_activity_sensitivity.csv")
    enriched = pd.read_csv(enriched_dir / "stable_activity_sensitivity.csv")
    baseline = baseline[baseline["period"] == "oot"]
    enriched = enriched[enriched["period"] == "oot"]
    columns = [
        "max_down3", "latest_signal_count", "empty_day_rate", "avg_signals_per_day",
        "trades", "avg_return_pct", "profit_factor", "daily_sharpe", "max_drawdown_pct",
    ]
    return baseline[columns].rename(
        columns={column: f"base_{column}" for column in columns if column != "max_down3"}
    ).merge(
        enriched[columns].rename(
            columns={column: f"enriched_{column}" for column in columns if column != "max_down3"}
        ),
        on="max_down3",
        how="outer",
        validate="one_to_one",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--enriched-eval", type=Path, required=True)
    parser.add_argument("--baseline-training-report", type=Path, required=True)
    parser.add_argument("--enriched-training-report", type=Path, required=True)
    parser.add_argument("--production-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_quality = compare_model_quality(args.baseline_eval, args.enriched_eval)
    selected_oot, selected_years = compare_selected(args.baseline_eval, args.enriched_eval)
    top_features = enriched_top_features(args.enriched_training_report)
    production = production_oot_table(args.production_summary)
    activity = compare_activity(args.baseline_eval, args.enriched_eval)
    enriched_current = pd.read_csv(args.enriched_eval / "current_threshold_periods.csv")
    enriched_selected = pd.read_csv(args.enriched_eval / "selected_by_test.csv")
    stable_oot = enriched_current[
        enriched_current["entry_rule"].eq("current__b1_stable")
        & enriched_current["period"].eq("oot")
    ].iloc[0]
    aggressive_oot = enriched_current[
        enriched_current["entry_rule"].eq("current__b1_aggressive")
        & enriched_current["period"].eq("oot")
    ].iloc[0]
    selected_aggressive_oot = enriched_selected[
        enriched_selected["exit_rule"].str.contains("b1_aggressive")
        & enriched_selected["split"].eq("oot")
    ].iloc[0]
    activity_current = activity[activity["max_down3"].eq(0.40)].iloc[0]
    activity_relaxed = activity[activity["max_down3"].eq(0.50)].iloc[0]
    enriched_training = json.loads(args.enriched_training_report.read_text(encoding="utf-8"))
    enrichment = enriched_training["dataset"].get("weekly_enrichment", {})

    model_quality.to_csv(args.output_dir / "model_quality_comparison.csv", index=False)
    selected_oot.to_csv(args.output_dir / "selected_oot_comparison.csv", index=False)
    selected_years.to_csv(args.output_dir / "selected_year_comparison.csv", index=False)
    top_features.to_csv(args.output_dir / "enriched_top_features.csv", index=False)
    activity.to_csv(args.output_dir / "stable_activity_comparison.csv", index=False)

    oot_quality = model_quality[model_quality["split"] == "oot"]
    lines = [
        "# B1 新因子重训练对照",
        "",
        "两组使用同一批 B1 原始门槛候选、相同标签、相同股票级 train/test 切分和 2025+ OOT；增强版只增加历史时点可得的周频、财务、分析师及外部因子。",
        "",
        "## 结论",
        "",
        f"- **增强版稳健策略可作为下一阶段候选**：沿用原稳健阈值的 OOT 为 {int(stable_oot['trades']):,} 笔，单笔 {stable_oot['avg_return_pct']:.4f}%，PF {stable_oot['profit_factor']:.4f}，最大回撤 {stable_oot['max_drawdown_pct']:.2f}%。",
        f"- **不建议放宽风险阈值来强行出股**：down3=0.40 在最新日为 {int(activity_current['enriched_latest_signal_count'])} 只；放到 0.50 才有 {int(activity_relaxed['enriched_latest_signal_count'])} 只，但整体 OOT 单笔收益从 {activity_current['enriched_avg_return_pct']:.4f}% 降到 {activity_relaxed['enriched_avg_return_pct']:.4f}%。",
        f"- **原进攻阈值暂不合格**：增强模型直接迁移后的 OOT 单笔 {aggressive_oot['avg_return_pct']:.4f}%，PF {aggressive_oot['profit_factor']:.4f}。test 期重校准后的进攻候选 OOT 为 {int(selected_aggressive_oot['trades'])} 笔、单笔 {selected_aggressive_oot['avg_return_pct']:.4f}%、PF {selected_aggressive_oot['profit_factor']:.4f}，样本仍偏少，应先观察而非发布。",
        "- 当前结果为 research 候选，未覆盖生产模型和策略配置；交易回测尚未计手续费、滑点与容量约束。",
        "",
        "## 数据与准入",
        "",
        f"- 长线周频源覆盖 B1 候选比例：{enrichment.get('matched_rate', float('nan')):.2%}",
        f"- 候选增强因子：{enrichment.get('candidate_feature_count', 0)}；录取（含可用性标记）：{enrichment.get('admitted_feature_count', 0)}",
        f"- 未来日期匹配行数：{enrichment.get('future_row_count', 'unknown')}",
        "",
        "## 分类质量变化",
        "",
        oot_quality[["model", "daily_auc", "enriched_auc", "delta_auc", "daily_brier_score", "enriched_brier_score"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        f"OOT AUC 平均变化：{oot_quality['delta_auc'].mean():+.4f}。",
        "",
        "## 各自仅在 test 期选择阈值后的 OOT 对照",
        "",
        selected_oot.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 分年稳定性",
        "",
        selected_years.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 稳健版风险上限放宽的活跃度—绩效代价（OOT）",
        "",
        activity.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 增强因子进入各模型 Top 40 的情况",
        "",
        (top_features.to_markdown(index=False, floatfmt=".6f") if not top_features.empty else "没有增强因子进入 Top 40。"),
        "",
        "## 当前生产模型的已发布 OOT 参照",
        "",
        production.to_markdown(index=False, floatfmt=".4f"),
        "",
        "> 生产参照使用旧因子公式和旧概率阈值，只用于量级参考；不能把阈值数值直接迁移到新版模型。",
        "",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"B1 variant comparison written: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
