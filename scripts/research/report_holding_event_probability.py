#!/usr/bin/env python
"""Summarize the current 15-event holdings experiment without selecting new policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from quant.research.holding_event_probability import (  # noqa: E402
    TARGETS,
    event_labels,
    threshold_metrics,
)

FAMILIES = {
    "high_up": "盘中上涨",
    "open_up": "开盘上涨",
    "close_up": "收盘上涨",
    "open_down": "开盘下跌",
    "close_down": "收盘下跌",
}


def name(target):
    return FAMILIES[target[:-1]] + "≥" + target[-1] + "%"


def percent(value):
    return f"{value:.2%}" if pd.notna(value) else "—"


def supported(row):
    return (
        row["signals"] >= 50
        and row["signal_dates"] >= 20
        and row["precision"] >= 0.8
        and row["precision_low90"] >= 0.7
        and row["lift"] >= 1.2
    )


def write_report(output, source):
    summary = json.loads((output / "run_summary.json").read_text())
    policy = json.loads((output / "locked_policy.json").read_text())
    if set(policy["policies"]) != set(TARGETS):
        raise ValueError("This report requires the current 15-event schema")
    confirmation = pd.read_csv(output / "confirmation_metrics.csv")
    locked = summary["locked_threshold_confirmation"]
    cross = [
        target
        for target in TARGETS
        if (rows := [r for r in locked if r["target"] == target])
        and len(rows) >= 2
        and all(supported(r) for r in rows)
    ]
    current = summary["confirmation_supported_targets"]
    lines = [
        "# 持仓事件模型：开盘、盘中上涨与收盘",
        "",
        "当前标签为开盘涨跌、收盘涨跌、盘中上涨，分别按1%/2%/3%建模；不包含盘中下跌。",
        "",
        f"本地数据截至{summary['latest_data_date']}，最新覆盖{summary['latest_rows']}只股票。幅度均相对T日收盘，使用用户授权的收盘与全天量价代理。2026是已用于前序研究的确认期，不是全新盲测。",
        "",
        f"按验证期选出的特征组：`{summary['feature_group']}`。当前模型阶段合格事件："
        + ("、".join(map(name, current)) or "无")
        + "。",
        "两种模型阶段均通过预设标准的事件："
        + ("、".join(map(name, cross)) or "无")
        + "。两阶段确认日期有重叠，不是两次独立盲测。",
        "",
        "合格标准：命中率≥80%，基础发生率提升≥1.2倍，至少50条且20个日期，五交易日区块重采样90%区间下沿≥70%。区间未作多重搜索调整。无合格项时不强制生成信号。",
        "",
        "## 当日特征的增量",
        "",
        "|特征组|验证合格事件数|验证平均Brier skill|2026平均Brier skill|2026平均AUC|2026平均ECE|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["group_ranking"]:
        group = row["feature_group"]
        c = confirmation[confirmation.feature_group.eq(group)]
        lines.append(
            f"|{group}|{row['supported_targets']}|{percent(row['mean_brier_skill'])}|{percent((1 - c.brier / c.base_rate_brier).mean())}|{c.roc_auc.mean():.3f}|{percent(c.ece.mean())}|"
        )
    lines += [
        "",
        "历史基线、当日基础、当日衍生三组初轮分别161/193/211个特征；池身份在各组共用。Brier skill的基础发生率使用评估期实际频率，仅作事后参考。",
        "",
        "## 固定阈值的验证与确认",
        "",
        "|事件|阶段|阈值|信号数|日期数|命中率|90%区间|覆盖率|基础发生率|提升|",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    rows = [dict(p, target=t, phase="2025H2验证") for t, p in policy["policies"].items() if p]
    for row in rows + locked:
        lines.append(
            f"|{name(row['target'])}|{row['phase']}|{percent(row['threshold'])}|{int(row['signals'])}|{int(row['signal_dates'])}|{percent(row['precision'])}|{percent(row['precision_low90'])}–{percent(row['precision_high90'])}|{percent(row['coverage'])}|{percent(row['base_rate'])}|{row['lift']:.2f}|"
        )
    if not rows:
        lines += [
            "",
            "验证期没有事件通过预设标准，因此没有锁定交易候选阈值，也未扩展训练制造信号。",
        ]
    lines += [
        "",
        "未通过验证筛选："
        + "、".join(name(t) for t, p in policy["policies"].items() if p is None)
        + "。",
        "",
        "## 每个事件的概率质量：2026确认",
        "",
        "|事件|历史基线AUC|当日基础AUC|当日衍生AUC|历史基线Brier|当日基础Brier|当日衍生Brier|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        records = [
            confirmation[confirmation.feature_group.eq(g) & confirmation.target.eq(target)].iloc[0]
            for g in ("lag1", "t0_base", "t0_extended")
        ]
        lines.append(
            "|"
            + name(target)
            + "|"
            + "|".join(f"{r.roc_auc:.3f}" for r in records)
            + "|"
            + "|".join(f"{r.brier:.4f}" for r in records)
            + "|"
        )
    lines += [
        "",
        "## 固定80%概率门槛：2026诊断",
        "",
        "下表统一使用80%门槛展示所选特征组的高概率质量，不依据这些确认结果重新选择阈值。",
        "",
        "|事件|信号数|日期数|平均预测概率|实际命中率|90%区间|覆盖率|基础发生率|",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    curves = pd.read_csv(output / "confirmation_thresholds.csv")
    for row in curves[
        curves.feature_group.eq(summary["feature_group"]) & curves.threshold.eq(0.8)
    ].to_dict("records"):
        lines.append(
            f"|{name(row['target'])}|{row['signals']}|{row['signal_dates']}|"
            f"{percent(row['mean_probability'])}|{percent(row['precision'])}|"
            f"{percent(row['precision_low90'])}–{percent(row['precision_high90'])}|"
            f"{percent(row['coverage'])}|{percent(row['base_rate'])}|"
        )
    refit_curves_path = output / "refit_thresholds.csv"
    if refit_curves_path.exists():
        refit_curves = pd.read_csv(refit_curves_path)
        lines += [
            "",
            "## 扩展模型2026下半年：上涨事件诊断",
            "",
            "统一展示预测概率≥80%的上涨事件。它们未在早期验证阶段锁定，"
            "因此这里只报告近期探索结果，不据此改变正式候选规则。"
            "当前无锁定信号，不等于近期上涨模型没有有价值的结果。",
            "",
            "|事件|信号数|日期数|实际命中率|90%区间|覆盖率|基础发生率|提升|",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
        for row in refit_curves[
            refit_curves.target.str.contains("_up") & refit_curves.threshold.eq(0.8)
        ].to_dict("records"):
            lines.append(
                f"|{name(row['target'])}|{row['signals']}|{row['signal_dates']}|"
                f"{percent(row['precision'])}|{percent(row['precision_low90'])}–"
                f"{percent(row['precision_high90'])}|{percent(row['coverage'])}|"
                f"{percent(row['base_rate'])}|{row['lift']:.2f}|"
            )
        lines += [
            "",
            "开盘上涨各档不足50条样本；收盘上涨未达到80%概率门槛的"
            "情况不应写成0%命中率。盘中上涨≥1%的近期表现值得独立前推验证，"
            "不能把已观察到的这段时间再称为新规则的盲测。",
        ]
    # Diagnose fixed policies by pool; never select a new threshold or pool here.
    if rows:
        raw = pd.read_parquet(
            source,
            columns=[
                "symbol",
                "date",
                "pool_short",
                "pool_good",
                "pool_long",
                "y_open",
                "y_high",
                "y_close",
                "tail_return",
            ],
        )
        raw = pd.concat([raw, event_labels(raw).add_prefix("label_")], axis=1)
        slices = []
        context_slices = []
        for phase, path in [
            ("frozen_2026", output / f"{summary['feature_group']}_probabilities.parquet"),
            ("refit_2026H2", output / "refit_probabilities.parquet"),
        ]:
            if not path.exists():
                continue
            predictions = pd.read_parquet(path)
            predictions = predictions[predictions.date.ge("2026-01-01")]
            joined = predictions.merge(raw, on=["symbol", "date"], validate="one_to_one")
            for target, rule in policy["policies"].items():
                if rule is None:
                    continue
                for context, mask in [
                    ("T日跌幅≥4.8%", joined.tail_return.le(-0.048)),
                    ("T日涨跌幅>-4.8%", joined.tail_return.gt(-0.048)),
                ]:
                    sub = joined.loc[mask]
                    metrics = threshold_metrics(
                        sub[["date", target, f"label_{target}"]].rename(
                            columns={target: "probability", f"label_{target}": "outcome"}
                        ),
                        [rule["threshold"]],
                    )
                    context_slices += [
                        dict(r, target=target, phase=phase, context=context)
                        for r in metrics.to_dict("records")
                    ]
                for pool in ("union", "short", "good", "long"):
                    sub = joined if pool == "union" else joined[joined[f"pool_{pool}"].eq(True)]
                    metrics = threshold_metrics(
                        sub[["date", target, f"label_{target}"]].rename(
                            columns={target: "probability", f"label_{target}": "outcome"}
                        ),
                        [rule["threshold"]],
                    )
                    slices += [
                        dict(r, target=target, phase=phase, pool=pool)
                        for r in metrics.to_dict("records")
                    ]
        pd.DataFrame(slices).to_csv(output / "pool_confirmation.csv", index=False)
        pd.DataFrame(context_slices).to_csv(output / "signal_context.csv", index=False)
        lines += [
            "",
            "## 当前阶段分池诊断",
            "",
            "|事件|池|信号数|日期数|命中率|90%区间|提升|",
            "|---|---|---:|---:|---:|---|---:|",
        ]
        phase = (
            "refit_2026H2" if (output / "refit_probabilities.parquet").exists() else "frozen_2026"
        )
        for row in slices:
            if row["phase"] != phase:
                continue
            lines.append(
                f"|{name(row['target'])}|{row['pool']}|{row['signals']}|{row['signal_dates']}|{percent(row['precision'])}|{percent(row['precision_low90'])}–{percent(row['precision_high90'])}|{row['lift']:.2f}|"
            )
        lines += ["", "各池重叠，样本不能相加。分池结果仅作诊断，不依据确认期重新筛池。"]
        lines += [
            "",
            "## 当日涨跌背景诊断",
            "",
            "4.8%仅为大跌分组边界，不是准确跌停判定，也不是新增选股条件。",
            "",
            "|事件|阶段|T日背景|信号数|日期数|命中率|90%区间|",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for row in context_slices:
            lines.append(
                f"|{name(row['target'])}|{row['phase']}|{row['context']}|{row['signals']}|"
                f"{row['signal_dates']}|{percent(row['precision'])}|"
                f"{percent(row['precision_low90'])}–{percent(row['precision_high90'])}|"
            )
    lines += [
        "",
        "## 输出与解释",
        "",
        f"最新模型阶段：`{summary['latest_model_phase']}`。",
        "",
        "|事件|最新研究候选数|",
        "|---|---:|",
    ]
    lines += [f"|{name(t)}|{summary['latest_signals'][t]}|" for t in TARGETS]
    lines += [
        "",
        "研究候选表示当前阶段及阈值达标，不等于跨期稳定或可成交。开盘跌和盘中涨可同时发生；开盘/收盘同一时点的涨跌互斥。盘中涨幅不能当成可实现收益。T日低点与下影仍是合法输入特征，不再预测次日低点下跌。",
        "",
        f"方法见 [模型说明]({ROOT}/docs/strategies/holding_event_probability.md)。原始指标、锁定阈值、可靠性分箱、模型和逐股票日概率位于本目录。",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(output / "summary.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/research/holding_event_probability_v2"
    )
    parser.add_argument(
        "--source", type=Path, default=ROOT / "reports/research/tail_overnight/dataset.parquet"
    )
    args = parser.parse_args()
    write_report(args.output, args.source)
