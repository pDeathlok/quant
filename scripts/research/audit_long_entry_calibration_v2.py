#!/usr/bin/env python
"""Leave-one-validation-year-out audit of V2 model/config selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "research"))

from train_long_entry_price_models_v2 import guardrail_mask
from quant.research.long_entry_v2 import month_end_week_mask, select_industry_capped, summarize_selection


REPORT_DIR = PROJECT_ROOT / "reports/long_entry_model_v2"


def config_key(row: pd.Series | dict) -> tuple:
    return (
        row["candidate"],
        float(row["risk_weight"]),
        row["guardrail"],
        int(row["industry_cap"]),
    )


def run() -> None:
    predictions = pd.read_parquet(REPORT_DIR / "walk_forward_predictions.parquet")
    grid = pd.read_csv(REPORT_DIR / "model_calibration_grid.csv")
    validation = predictions[~predictions["reused_test"]].copy()
    validation = validation[month_end_week_mask(validation)]
    rows: list[dict] = []
    for config in grid.itertuples(index=False):
        candidate = validation[validation["candidate"].eq(config.candidate)].copy()
        candidate["audit_score"] = (
            candidate["opportunity_pct"] * (1.0 - float(config.risk_weight))
            + candidate["downside_safe_pct"] * float(config.risk_weight)
        )
        eligible = candidate[guardrail_mask(candidate, str(config.guardrail))]
        for fold, baseline in candidate.groupby("fold"):
            labelled = baseline.dropna(subset=["return_52w"])
            fold_eligible = eligible[eligible["fold"].eq(fold)].dropna(subset=["return_52w"])
            selected = select_industry_capped(
                fold_eligible,
                score_column="audit_score",
                top_n=20,
                max_per_industry=int(config.industry_cap),
            )
            rows.append(
                {
                    "candidate": config.candidate,
                    "risk_weight": float(config.risk_weight),
                    "guardrail": config.guardrail,
                    "industry_cap": int(config.industry_cap),
                    "fold": fold,
                    **summarize_selection(selected, labelled),
                }
            )
    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(REPORT_DIR / "calibration_config_fold_metrics.csv", index=False)

    nested_rows: list[dict] = []
    folds = sorted(fold_metrics["fold"].unique())
    keys = ["candidate", "risk_weight", "guardrail", "industry_cap"]
    for held_out in folds:
        inner = fold_metrics[fold_metrics["fold"].ne(held_out)]
        ranking = (
            inner.groupby(keys, as_index=False)
            .agg(
                positive_inner_folds=("return_delta", lambda values: int((values > 0).sum())),
                worst_inner_delta=("return_delta", "min"),
                mean_inner_delta=("return_delta", "mean"),
                mean_inner_mae=("mean_mae_26w", "mean"),
            )
            .sort_values(
                ["positive_inner_folds", "worst_inner_delta", "mean_inner_delta", "mean_inner_mae"],
                ascending=[False, False, False, False],
            )
        )
        chosen = ranking.iloc[0]
        mask = pd.Series(True, index=fold_metrics.index)
        for key in keys:
            mask &= fold_metrics[key].eq(chosen[key])
        held = fold_metrics[mask & fold_metrics["fold"].eq(held_out)].iloc[0]
        nested_rows.append(
            {
                "held_out_fold": held_out,
                **{key: chosen[key] for key in keys},
                "held_out_return_delta": held["return_delta"],
                "held_out_return_52w": held["mean_return_52w"],
                "held_out_excess_52w": held["mean_excess_52w"],
                "held_out_mae_26w": held["mean_mae_26w"],
            }
        )
    nested = pd.DataFrame(nested_rows)
    nested.to_csv(REPORT_DIR / "nested_calibration_audit.csv", index=False)
    summary = {
        "positive_held_out_folds": int(nested["held_out_return_delta"].gt(0).sum()),
        "mean_held_out_delta": float(nested["held_out_return_delta"].mean()),
        "worst_held_out_delta": float(nested["held_out_return_delta"].min()),
        "chosen_config_stability": nested[keys].value_counts().reset_index(name="folds").to_dict("records"),
    }
    (REPORT_DIR / "nested_calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# V2 校准稳健性审计",
        "",
        "每次留出一个完整验证年，只用另外三个验证年选择模型、风险权重、价格护栏和行业上限，再评价被留出的年份。",
        "",
        nested.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"- 留出年正增量：{summary['positive_held_out_folds']}/{len(nested)}。",
        f"- 平均留出增量：{summary['mean_held_out_delta']:.2%}；最差留出增量：{summary['worst_held_out_delta']:.2%}。",
        "- 这仍是历史验证内的二层审计，不等于新的独立样本外；用途是识别配置选择是否只靠单一年份。",
    ]
    (REPORT_DIR / "calibration_robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
