#!/usr/bin/env python
"""Audit fitted shallow rules and summarize held-out evidence without retuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from quant.research.sequence_strategy_validation import event_metrics, purged_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output
    results = json.loads((output / "results.json").read_text())
    audit = []
    explicit_rules = []
    for side in ["left", "right"]:
        frame = pd.read_parquet(output / f"{side}_scored_events.parquet")
        splits = purged_split(frame)
        for variant in ["distilled_rules", "feedback_rules"]:
            model = joblib.load(output / f"{side}_{variant}.joblib")
            tree, features, leaves = model["tree"], model["features"], model["leaves"]
            stack = [(0, [])]
            while stack:
                node, conditions = stack.pop()
                left = int(tree.tree_.children_left[node])
                if left < 0:
                    if node in leaves:
                        explicit_rules.append(
                            {
                                "side": side,
                                "variant": variant,
                                "leaf": node,
                                "all_conditions": conditions,
                            }
                        )
                    continue
                feature = features[tree.tree_.feature[node]]
                threshold = float(tree.tree_.threshold[node])
                stack.append(
                    (
                        left,
                        conditions
                        + [{"feature": feature, "operator": "<=", "threshold": threshold}],
                    )
                )
                stack.append(
                    (
                        int(tree.tree_.children_right[node]),
                        conditions
                        + [{"feature": feature, "operator": ">", "threshold": threshold}],
                    )
                )
            if variant == "distilled_rules":
                fidelity = {
                    "train_teacher_r2": float(
                        tree.score(splits["train"][features], splits["train"].baseline_score)
                    ),
                    "selection_teacher_r2": float(
                        tree.score(
                            splits["selection"][features], splits["selection"].baseline_score
                        )
                    ),
                }
            else:
                fidelity = {}
            for split, events in splits.items():
                nodes = tree.apply(events[features])
                for leaf in leaves:
                    part = events[nodes == leaf]
                    audit.append(
                        {
                            "side": side,
                            "variant": variant,
                            "split": split,
                            "leaf": leaf,
                            **event_metrics(part),
                            **fidelity,
                        }
                    )
    pd.DataFrame(audit).to_csv(output / "rule_stability.csv", index=False)
    (output / "explicit_rules.json").write_text(
        json.dumps(explicit_rules, indent=2, ensure_ascii=False) + "\n"
    )
    # A candidate needs both usable trade evidence and return improvement before promotion.
    decisions = []
    for row in results:
        if row["variant"] == "baseline":
            continue
        gain = row["incremental_gain"]
        # A sample this small is a research gate, never an automatic production release.
        evidence = bool(
            row["trades"] >= 100
            and (row["win_rate"] or 0) >= 0.60
            and (row["profit_factor"] or 0) > 1
            and row["total_return"] > 0
            and gain["positive_gain_supported"]
        )
        decisions.append(
            {
                "side": row["side"],
                "variant": row["variant"],
                "positive_return_gain_supported": gain["positive_gain_supported"],
                "high_success_research_gate": evidence,
                "production_promoted": False,
                "reason": "Sampled research only; requires independent untouched confirmation and production feature parity",
            }
        )
    (output / "admission_decision.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n"
    )
    coverage = json.loads((output / "coverage.json").read_text())
    text = [
        "# 实验审计补充",
        "",
        f"抽样股票{len(coverage['sampled_symbols'])}只，可用行情{coverage['available_histories']}只，合格候选事件{coverage['eligible_rows']}条。",
        f"历史右侧标签对照{coverage['label_comparison_rows']}条，平均绝对差{coverage['label_mean_absolute_difference']:.8f}。",
        "",
        "浅树蒸馏以历史原模型的样本外分数为目标，属于近似提炼；并非把原XGBoost的每个分支原样搬出。",
        "第二轮浅树加入2025上半年的输赢反馈，使用实际盈利标签优化；2025下半年只做规则选择。",
        "简单规则的逐叶稳定性见 rule_stability.csv；输赢case与特征均值见 cases.csv 和 feedback_case_summary.csv。",
        "",
        "胜率由已平仓交易计算，尚未退出的仓位单列，组合总收益包含剩余仓位盯市。",
        "允许空仓版本若验证期没有足够证据，测试期也保持现金；全现金没有胜率。",
        "资金账户使用20个等额上限仓位、同股不重复持仓、涨停不买、跌停延期卖出、双边成本合计20bps。",
        "未计100股整手、排队成交概率和冲击成本，因此不能直接作为实盘收益承诺。",
        "",
        "各版本测试期结果已查看，不得继续据此调参并宣称同一测试期仍然独立。",
        "当前报告不替换生产模型；抽样股票池不等于真实全市场每日Top20。",
    ]
    (output / "audit.md").write_text("\n".join(text) + "\n")
    print(json.dumps(decisions, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
