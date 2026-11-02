#!/usr/bin/env python
"""Matched 20-session controls and evidence audit; no new parameter selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from run_kronos_rolling import save_json
from run_weekly_direction_research import backtest, period_part

from quant.research.sequence_strategy_validation import event_metrics, paired_block_gain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    events = pd.read_parquet(out / "events.parquet")
    prices = pd.read_parquet(out / "execution_bars.parquet")
    calendar = pd.DatetimeIndex(pd.to_datetime(json.loads((out / "calendar.json").read_text())))
    records = []
    for side in ["left", "right"]:
        model = pd.read_parquet(out / f"{side}_model_predictions.parquet")
        threshold = json.loads((out / f"{side}_model_selection_frozen.json").read_text())["chosen"][
            "threshold"
        ]
        frame = events[events.side.eq(side) & events.horizon.eq(20)]
        for period in ["selection", "test"]:
            block, active = period_part(frame, calendar, period)
            scored, _ = period_part(model, calendar, period)
            variants = {
                "baseline20": (block, "baseline_percentile"),
                "weekly_macd_boll20": (block[block.macd_boll], "baseline_percentile"),
                "weekly_model_rank20": (scored, "model_probability"),
                "weekly_model_threshold20": (
                    scored[scored.model_probability.ge(threshold)],
                    "model_probability",
                ),
            }
            baseline = None
            for name, (sample, score) in variants.items():
                curve, trades, metrics = backtest(sample, prices, active, score)
                if name == "weekly_macd_boll20":
                    baseline = curve
                record = {
                    "side": side,
                    "period": period,
                    "variant": name,
                    **metrics,
                    "events": event_metrics(sample),
                }
                if name in ["weekly_model_rank20", "weekly_model_threshold20"]:
                    record["gain_vs_same_week_gate"] = paired_block_gain(
                        baseline, curve, repetitions=1000
                    )
                records.append(record)
                curve.to_csv(out / f"{side}_{period}_control_{name}_equity.csv", index=False)
                trades.to_parquet(
                    out / f"{side}_{period}_control_{name}_trades.parquet", index=False
                )
    save_json(out / "matched_controls.json", records)
    choices = json.loads((out / "selection_decision.json").read_text())
    search = json.loads((out / "rule_search.json").read_text())
    evidence = []
    for side in ["left", "right"]:
        choice = next(x for x in choices if x["side"] == side and x["kind"] == "rule")
        rule = next(
            x
            for x in search
            if x["side"] == side and all(x[k] == choice[k] for k in ["week", "entry", "horizon"])
        )
        evidence.append(rule)
    save_json(out / "selected_rule_period_evidence.json", evidence)
    lines = [
        "# 同持有期对照与准入审计",
        "",
        "以下均固定20日持有，模型阈值沿用2025H2冻结选择，没有另选参数。",
        "",
        "|侧|版本|2025H2收益|2026收益|2026胜率|2026回撤|2026敞口|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in records if r["period"] == "test"]:
        prior = next(
            r
            for r in records
            if r["side"] == row["side"]
            and r["variant"] == row["variant"]
            and r["period"] == "selection"
        )
        win = f"{row['win_rate']:.1%}" if row["win_rate"] is not None else "—"
        lines.append(
            f"|{row['side']}|{row['variant']}|{prior['total_return']:.2%}|{row['total_return']:.2%}|{win}|{row['max_drawdown']:.2%}|{row['mean_exposure']:.1%}|"
        )
    lines += ["", "选中简单规则的跨期事件统计："]
    for row in evidence:
        lines.append(
            f"- {row['side']} {row['week']} / {row['entry']} / {row['horizon']}日，跨期准入={row['supported']}。"
        )
        for period in ["development_2024", "development_2025H1", "selection"]:
            r = row[period]
            win = f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "—"
            lines.append(
                f"  - {period}: 事件{r['trades']}，胜率{win}，均值{r['average_net_return']:.2%}，PF={r['profit_factor']:.3f}。"
            )
    lines += [
        "",
        "模型排序版和阈值版的差异同时涉及敞口与交易筛选，不能把少交易产生的减损直接当成alpha。",
        "所有规则与模型都未自动上线；验证选择带有多重比较，新增股票同一时间区间的复核不等于全新的时间外确认。",
    ]
    (out / "matched_controls.md").write_text("\n".join(lines) + "\n")
    print((out / "matched_controls.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
