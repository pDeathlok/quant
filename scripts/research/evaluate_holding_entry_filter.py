#!/usr/bin/env python
"""Re-evaluate existing event forecasts under the user's T-day return filter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from quant.research.holding_event_probability import event_labels, threshold_metrics  # noqa: E402

TARGETS = ("high_up1", "open_up1", "open_up2", "open_up3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/research/holding_event_probability_v2"
    )
    parser.add_argument(
        "--source", type=Path, default=ROOT / "reports/research/tail_overnight/dataset.parquet"
    )
    parser.add_argument("--max-abs-return", type=float, default=0.095)
    args = parser.parse_args()
    if not 0 < args.max_abs_return < 1:
        raise ValueError("max-abs-return must be between zero and one")
    raw = pd.read_parquet(
        args.source, columns=["symbol", "date", "tail_return", "y_open", "y_high", "y_close"]
    )
    raw = pd.concat([raw, event_labels(raw).add_prefix("label_")], axis=1)
    predictions = pd.read_parquet(args.output / "refit_probabilities.parquet")
    joined = predictions.merge(raw, on=["symbol", "date"], validate="one_to_one")
    # Do not intersect entry_executable: its legacy 4.8% proxy would silently
    # undo the user-authorized replacement with 9.5%.
    joined["entry_return_eligible"] = joined.tail_return.abs().lt(args.max_abs_return)
    rows = []
    selections = []
    for target in TARGETS:
        for gate in ("unfiltered", "user_return_filter"):
            subset = joined if gate == "unfiltered" else joined[joined.entry_return_eligible]
            observed = subset[["date", target, f"label_{target}"]].rename(
                columns={target: "probability", f"label_{target}": "outcome"}
            )
            stats = threshold_metrics(observed, [0.8])
            rows += [
                dict(r, target=target, gate=gate, max_abs_return=args.max_abs_return)
                for r in stats.to_dict("records")
            ]
        selected = joined[joined.entry_return_eligible & joined[target].ge(0.8)].copy()
        selected["target"] = target
        selected["probability"] = selected[target]
        selected["outcome"] = selected[f"label_{target}"]
        selections.append(
            selected[["symbol", "date", "target", "tail_return", "probability", "outcome"]]
        )
    results = pd.DataFrame(rows)
    results.to_csv(args.output / "entry_filter_diagnostic.csv", index=False)
    pd.concat(selections).to_csv(
        args.output / "entry_filtered_research_candidates.csv", index=False
    )
    joined[joined.date.eq(joined.date.max()) & joined.entry_return_eligible].to_csv(
        args.output / "latest_entry_filtered_probabilities.csv", index=False
    )
    lines = [
        "# T日收盘涨跌幅过滤复核",
        "",
        f"按用户指定，保留 −{args.max_abs_return:.1%} < T日收盘涨跌幅 < +{args.max_abs_return:.1%} 的股票，边界也剔除。涨跌幅缺失不准入。不叠加原来的4.8%买入代理，也不改变已拟合模型及80%概率门槛。",
        "",
        "这是对已有2026下半年预测的过滤复核，未重新训练，不是新增独立前推测试。涨跌幅阈值是用户指定的研究口径，不等于真实涨跌停识别或成交保证。",
        "",
        "|事件|过滤前样本数|过滤前命中率|过滤后样本数|日期数|过滤后命中率|90%区间|过滤后覆盖率|",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    names = dict(
        high_up1="盘中涨≥1%", open_up1="开盘涨≥1%", open_up2="开盘涨≥2%", open_up3="开盘涨≥3%"
    )
    for target in TARGETS:
        before = results[(results.target == target) & (results.gate == "unfiltered")].iloc[0]
        after = results[(results.target == target) & (results.gate == "user_return_filter")].iloc[0]
        lines.append(
            f"|{names[target]}|{before.signals}|{before.precision:.2%}|{after.signals}|{after.signal_dates}|{after.precision:.2%}|{after.precision_low90:.2%}–{after.precision_high90:.2%}|{after.coverage:.2%}|"
        )
    lines += [
        "",
        "覆盖率分母为过滤后有有效标签的股票日。命中率为预测概率≥80%的样本中实际发生事件的比例；事件幅度相对T日收盘价。开盘上涨各档相互嵌套，样本不可相加。",
        "",
        "最新日的label为空，不能计入历史命中率；候选CSV保留该缺失以便区分研究预测与已观察结果。",
    ]
    (args.output / "entry_filter_summary.md").write_text("\n".join(lines) + "\n")
    print(
        results[
            [
                "target",
                "gate",
                "signals",
                "signal_dates",
                "precision",
                "precision_low90",
                "precision_high90",
                "coverage",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
