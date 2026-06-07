#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze whether B1 should buy on T+1 gap-up or gap-down opens.

The signal is generated on T+0. Execution is still T+1 open, but this pass adds
filters based on the T+1 open gap versus the T+0 close before simulating exits.
All metrics use the same non-overlap holding rule as the latest B1 research.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_b1_entry_exit_grid import ExitRule, add_future_prices, simulate_exit, summarize_returns
from analyze_b1_xgb_entry_exit_grid import (
    DEFAULT_DAILY_DIR,
    DEFAULT_DATASET,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    ThresholdEntryRule,
    apply_entry_rule,
    drop_overlapping_trades,
    predict_xgb_models,
)


@dataclass(frozen=True)
class StrategyCase:
    name: str
    role: str
    entry_rule: ThresholdEntryRule
    exit_rule: ExitRule


@dataclass(frozen=True)
class GapFilter:
    name: str
    description: str
    min_gap_pct: float | None = None
    max_gap_pct: float | None = None


def build_cases() -> list[StrategyCase]:
    return [
        StrategyCase(
            "new_robust_tp10_sl15_t7",
            "新稳健观察版",
            ThresholdEntryRule("up10_ge_0.25_down3_le_0.40", min_up10=0.25, max_down3=0.40),
            ExitRule(
                "fixed_tp10.0%_sl1.5%_intraday_T7",
                "fixed",
                hold_days=6,
                take_profit=0.10,
                stop_loss=0.015,
                stop_trigger="intraday",
            ),
        ),
        StrategyCase(
            "new_robust_tp10_sl15_t5",
            "新稳健观察版备选",
            ThresholdEntryRule("up10_ge_0.25_down3_le_0.40", min_up10=0.25, max_down3=0.40),
            ExitRule(
                "fixed_tp10.0%_sl1.5%_intraday_T5",
                "fixed",
                hold_days=4,
                take_profit=0.10,
                stop_loss=0.015,
                stop_trigger="intraday",
            ),
        ),
        StrategyCase(
            "low_dd_tp3_sl15_t5",
            "低回撤观察版",
            ThresholdEntryRule("up10_ge_0.40_down3_le_0.40", min_up10=0.40, max_down3=0.40),
            ExitRule(
                "fixed_tp3.0%_sl1.5%_intraday_T5",
                "fixed",
                hold_days=4,
                take_profit=0.03,
                stop_loss=0.015,
                stop_trigger="intraday",
            ),
        ),
        StrategyCase(
            "aggressive_expiry_t7",
            "进攻观察版",
            ThresholdEntryRule("up8_ge_0.70_down3_le_0.45", min_up8=0.70, max_down3=0.45),
            ExitRule("expiry_T7_close_no_stop", "expiry", hold_days=6),
        ),
        StrategyCase(
            "aggressive_expiry_t9",
            "进攻观察版",
            ThresholdEntryRule("up8_ge_0.70_down3_le_0.45", min_up8=0.70, max_down3=0.45),
            ExitRule("expiry_T9_close_no_stop", "expiry", hold_days=8),
        ),
    ]


def build_gap_filters() -> list[GapFilter]:
    filters = [
        GapFilter("all", "不限制 T+1 高低开"),
        GapFilter("gap_down", "只买 T+1 低开", max_gap_pct=0.0),
        GapFilter("gap_up_flat", "只买 T+1 平开或高开", min_gap_pct=0.0),
    ]

    bins = [
        ("gap_le_-3", "低开 <= -3%", None, -3.0),
        ("gap_-3_to_-2", "低开 -3% 到 -2%", -3.0, -2.0),
        ("gap_-2_to_-1", "低开 -2% 到 -1%", -2.0, -1.0),
        ("gap_-1_to_0", "低开 -1% 到 0%", -1.0, 0.0),
        ("gap_0_to_1", "高开 0% 到 1%", 0.0, 1.0),
        ("gap_1_to_2", "高开 1% 到 2%", 1.0, 2.0),
        ("gap_2_to_3", "高开 2% 到 3%", 2.0, 3.0),
        ("gap_ge_3", "高开 >= 3%", 3.0, None),
    ]
    filters.extend(GapFilter(*item) for item in bins)

    seen = {item.name for item in filters}
    for upper in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
        name = f"gap_le_{upper:g}"
        if name in seen:
            continue
        filters.append(GapFilter(name, f"T+1 开盘涨跌幅 <= {upper:g}%", max_gap_pct=upper))
        seen.add(filters[-1].name)
    for lower in [-3.0, -2.0, -1.0, 0.0]:
        for upper in [0.0, 1.0, 2.0]:
            if lower < upper:
                name = f"gap_{lower:g}_to_{upper:g}"
                if name in seen:
                    continue
                filters.append(
                    GapFilter(
                        name,
                        f"{lower:g}% <= T+1 开盘涨跌幅 <= {upper:g}%",
                        min_gap_pct=lower,
                        max_gap_pct=upper,
                    )
                )
                seen.add(name)
    return filters


def apply_gap_filter(df: pd.DataFrame, gap_filter: GapFilter) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if gap_filter.min_gap_pct is not None:
        mask &= df["t1_open_gap_pct"] >= gap_filter.min_gap_pct
    if gap_filter.max_gap_pct is not None:
        mask &= df["t1_open_gap_pct"] <= gap_filter.max_gap_pct
    return df[mask].copy()


def evaluate_case_filter(candidates: pd.DataFrame, case: StrategyCase, gap_filter: GapFilter) -> list[dict]:
    entry_df = candidates[apply_entry_rule(candidates, case.entry_rule)].copy()
    entry_df = apply_gap_filter(entry_df, gap_filter)
    if entry_df.empty:
        return []

    trades = simulate_exit(entry_df, case.exit_rule)
    if trades.empty:
        return []
    trades = trades.merge(
        entry_df[["date", "symbol", "split", "close", "entry_open", "t1_open_gap_pct"]],
        on=["date", "symbol"],
        how="left",
    )
    raw_trades = len(trades)
    trades = drop_overlapping_trades(trades)
    skipped_overlaps = raw_trades - len(trades)

    rows = []
    for split in ["train", "test", "oot"]:
        split_trades = trades[trades["split"] == split].copy()
        metrics = summarize_returns(split_trades)
        if not metrics:
            continue
        rows.append(
            {
                "case": case.name,
                "role": case.role,
                "entry_rule": case.entry_rule.name,
                "exit_rule": case.exit_rule.name,
                "gap_filter": gap_filter.name,
                "gap_description": gap_filter.description,
                "min_gap_pct": gap_filter.min_gap_pct,
                "max_gap_pct": gap_filter.max_gap_pct,
                "split": split,
                "raw_trades": raw_trades,
                "skipped_overlaps": skipped_overlaps,
                "overlap_skip_rate": skipped_overlaps / raw_trades if raw_trades else np.nan,
                "avg_gap_pct": float(split_trades["t1_open_gap_pct"].mean()) if not split_trades.empty else np.nan,
                "median_gap_pct": float(split_trades["t1_open_gap_pct"].median()) if not split_trades.empty else np.nan,
                "min_return_pct": float(split_trades["return_pct"].min()) if not split_trades.empty else np.nan,
                "max_return_pct": float(split_trades["return_pct"].max()) if not split_trades.empty else np.nan,
                **metrics,
            }
        )
    return rows


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}%"


def fmt_rate(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def fmt_float(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def markdown_table(rows: list[dict], headers: list[str]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(header, "")) for header in headers) + "|")
    return "\n".join(lines)


def format_rows(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, row in df.iterrows():
        out.append(
            {
                "策略": row["case"],
                "类型": row["role"],
                "高低开过滤": row["gap_description"],
                "交易数": fmt_num(row["trades"]),
                "平均高低开": fmt_pct(row["avg_gap_pct"]),
                "平均收益": fmt_pct(row["avg_return_pct"]),
                "胜率": fmt_rate(row["win_rate"]),
                "最大回撤": fmt_pct(row["max_drawdown_pct"]),
                "PF": fmt_float(row["profit_factor"]),
                "止损率": fmt_rate(row["stop_rate"]),
                "最差单笔": fmt_pct(row["min_return_pct"]),
            }
        )
    return out


def write_report(summary: pd.DataFrame, output_dir: Path, timestamp: str) -> Path:
    oot = summary[summary["split"] == "oot"].copy()
    path = output_dir / f"b1_t1_gap_entry_analysis_{timestamp}.md"

    main_filters = oot[oot["gap_filter"].isin(["all", "gap_down", "gap_up_flat"])].copy()
    bin_filters = oot[oot["gap_filter"].isin(["gap_le_-3", "gap_-3_to_-2", "gap_-2_to_-1", "gap_-1_to_0", "gap_0_to_1", "gap_1_to_2", "gap_2_to_3", "gap_ge_3"])].copy()

    best_by_case = []
    for case, part in oot.groupby("case"):
        baseline = part[part["gap_filter"] == "all"]
        min_trades = max(30, int(float(baseline["trades"].iloc[0]) * 0.25)) if not baseline.empty else 30
        eligible = part[part["trades"] >= min_trades].copy()
        if eligible.empty:
            continue
        best = eligible.sort_values(["profit_factor", "avg_return_pct", "max_drawdown_pct"], ascending=[False, False, False]).iloc[0]
        best_by_case.append(best)
    best_by_case_df = pd.DataFrame(best_by_case)

    with path.open("w", encoding="utf-8") as f:
        f.write("# B1 T+1 高低开买入过滤分析\n\n")
        f.write("本轮分析的问题是：B1 信号在 T 日出现后，T+1 是低开买入更好，还是高开买入更好，以及应该使用什么高低开比例过滤。\n\n")
        f.write("计算口径：\n\n")
        f.write("- `T+1 开盘涨跌幅 = T+1 开盘价 / T 日收盘价 - 1`。\n")
        f.write("- 仍然按 T+1 开盘买入，卖出逻辑沿用上一轮修正后的跳空感知成交逻辑。\n")
        f.write("- 所有结果使用 `non_overlap`：同一只股票持仓未结束前，不重复买入。\n")
        f.write("- 本文重点看 OOT。\n\n")

        f.write("## 1. 低开 vs 高开\n\n")
        f.write(markdown_table(format_rows(main_filters.sort_values(["case", "gap_filter"])), ["策略", "类型", "高低开过滤", "交易数", "平均高低开", "平均收益", "胜率", "最大回撤", "PF", "止损率", "最差单笔"]))
        f.write("\n\n")

        f.write("## 2. 分区间表现\n\n")
        f.write(markdown_table(format_rows(bin_filters.sort_values(["case", "min_gap_pct", "max_gap_pct"])), ["策略", "类型", "高低开过滤", "交易数", "平均高低开", "平均收益", "胜率", "最大回撤", "PF", "止损率", "最差单笔"]))
        f.write("\n\n")

        f.write("## 3. 每个策略的较优过滤\n\n")
        f.write("这里要求过滤后至少保留基准交易数的 25%，且不少于 30 笔，避免只因为样本太少而看起来很好。\n\n")
        f.write(markdown_table(format_rows(best_by_case_df.sort_values(["role", "case"])), ["策略", "类型", "高低开过滤", "交易数", "平均高低开", "平均收益", "胜率", "最大回撤", "PF", "止损率", "最差单笔"]))
        f.write("\n\n")

        f.write("## 4. 初步结论\n\n")
        f.write("- B1 不适合简单地只看“高开更强”或“低开更便宜”；不同卖出结构下，最优高低开区间不同。\n")
        f.write("- 对新稳健观察版，应重点看 `T+1 开盘涨跌幅 <= 0%` 或 `-2% 到 0%` 一类过滤是否提高 PF，同时确认交易数是否还能接受。\n")
        f.write("- 对进攻观察版，高低开过滤很容易让样本变得更少，因此只适合做观察，不适合据此直接确定上线规则。\n")
        f.write("- 如果过滤后收益改善但最大回撤没有改善，说明高低开只能改善入场价格，不能替代流动性、波动和市场环境过滤。\n")
    return path


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading dataset: {DEFAULT_DATASET}", flush=True)
    candidates = pd.read_parquet(DEFAULT_DATASET)
    candidates["date"] = pd.to_datetime(candidates["date"])
    candidates = predict_xgb_models(candidates, DEFAULT_MODEL_DIR)

    cases = build_cases()
    max_hold_days = max(case.exit_rule.hold_days for case in cases)
    print("adding future prices", flush=True)
    candidates = add_future_prices(candidates, DEFAULT_DAILY_DIR, max_hold_days=max_hold_days)
    candidates = candidates.dropna(subset=["entry_open", "close"]).copy()
    candidates["t1_open_gap_pct"] = (candidates["entry_open"] / candidates["close"] - 1.0) * 100.0
    candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=["t1_open_gap_pct"])

    rows = []
    filters = build_gap_filters()
    total = len(cases) * len(filters)
    done = 0
    for case in cases:
        for gap_filter in filters:
            rows.extend(evaluate_case_filter(candidates, case, gap_filter))
            done += 1
            if done % 25 == 0 or done == total:
                print(f"evaluated {done}/{total}", flush=True)

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"b1_t1_gap_entry_analysis_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    latest_summary = output_dir / "latest_t1_gap_entry_analysis.csv"
    summary.to_csv(latest_summary, index=False)

    report_path = write_report(summary, output_dir, timestamp)
    latest_report = output_dir / "latest_t1_gap_entry_analysis.md"
    latest_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"summary: {summary_path}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
