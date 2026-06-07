#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Focused B1 stop-trigger and aggressive-exit comparison.

This research pass keeps selected B1 XGBoost entry ideas fixed and compares
gap-aware intraday hard stops with close-confirmed hard stops. It also tests
adding hard stops to the high-win expiry aggressive setup.
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
class FocusCase:
    group: str
    entry_rule: ThresholdEntryRule
    exit_rule: ExitRule
    note: str


def build_focus_cases() -> list[FocusCase]:
    stable = ThresholdEntryRule(
        "up5_ge_0.65_up8_ge_0.65_down2_le_0.60",
        min_up5=0.65,
        min_up8=0.65,
        max_down2=0.60,
    )
    robust_fixed = ThresholdEntryRule(
        "up10_ge_0.25_down3_le_0.40",
        min_up10=0.25,
        max_down3=0.40,
    )
    robust_fixed_alt = ThresholdEntryRule(
        "up10_ge_0.30_down3_le_0.40",
        min_up10=0.30,
        max_down3=0.40,
    )
    low_drawdown_fixed = ThresholdEntryRule(
        "up10_ge_0.40_down3_le_0.40",
        min_up10=0.40,
        max_down3=0.40,
    )
    high_win = ThresholdEntryRule(
        "up8_ge_0.70_down3_le_0.45",
        min_up8=0.70,
        max_down3=0.45,
    )
    high_win_looser_risk = ThresholdEntryRule(
        "up8_ge_0.70_down3_le_0.50",
        min_up8=0.70,
        max_down3=0.50,
    )
    high_win_strict_risk = ThresholdEntryRule(
        "up8_ge_0.70_down3_le_0.40",
        min_up8=0.70,
        max_down3=0.40,
    )

    cases: list[FocusCase] = []

    for entry_rule in [robust_fixed, robust_fixed_alt]:
        for hold_label, hold_days in [("T5", 4), ("T7", 6)]:
            for stop_loss in [0.01, 0.015]:
                for trigger in ["intraday", "close"]:
                    cases.append(
                        FocusCase(
                            "new_robust_fixed_stop_compare",
                            entry_rule,
                            ExitRule(
                                f"fixed_tp10.0%_sl{stop_loss:.1%}_{trigger}_{hold_label}",
                                "fixed",
                                hold_days,
                                take_profit=0.10,
                                stop_loss=stop_loss,
                                stop_trigger=trigger,
                            ),
                            "修正后完整网格筛出的新稳健候选：固定止盈 10%",
                        )
                    )

    for hold_label, hold_days in [("T5", 4), ("T7", 6), ("T9", 8), ("T12", 11)]:
        for trigger in ["intraday", "close"]:
            cases.append(
                FocusCase(
                    "low_drawdown_fixed_stop_compare",
                    low_drawdown_fixed,
                    ExitRule(
                        f"fixed_tp3.0%_sl1.5%_{trigger}_{hold_label}",
                        "fixed",
                        hold_days,
                        take_profit=0.03,
                        stop_loss=0.015,
                        stop_trigger=trigger,
                    ),
                    "修正后完整网格筛出的低回撤候选：固定止盈 3%",
                )
            )

    for target in [0.03, 0.04]:
        for trigger in ["intraday", "close"]:
            cases.append(
                FocusCase(
                    "stable_stop_trigger_compare",
                    stable,
                    ExitRule(
                        f"trail_target{target:.1%}_dd1.5%_sl1.0%_{trigger}_T5",
                        "trailing",
                        hold_days=4,
                        take_profit=target,
                        stop_loss=0.01,
                        trail_drawdown=0.015,
                        stop_trigger=trigger,
                    ),
                    "主策略硬止损触发方式对比",
                )
            )

    high_win_entries = [high_win, high_win_strict_risk, high_win_looser_risk]
    for entry_rule in high_win_entries:
        for hold_label, hold_days in [("T5", 4), ("T7", 6), ("T9", 8), ("T12", 11)]:
            cases.append(
                FocusCase(
                    "aggressive_expiry_baseline",
                    entry_rule,
                    ExitRule(f"expiry_{hold_label}_close_no_stop", "expiry", hold_days),
                    "高胜率进攻：到期卖出，无硬止损",
                )
            )
            for stop_loss in [0.01, 0.015, 0.02, 0.03]:
                for trigger in ["intraday", "close"]:
                    cases.append(
                        FocusCase(
                            "aggressive_expiry_with_stop",
                            entry_rule,
                            ExitRule(
                                f"expiry_{hold_label}_sl{stop_loss:.1%}_{trigger}",
                                "expiry",
                                hold_days,
                                stop_loss=stop_loss,
                                stop_trigger=trigger,
                            ),
                            "高胜率进攻：到期卖出叠加硬止损",
                        )
                    )

        for target in [0.06, 0.08, 0.10]:
            for trail in [0.015, 0.02, 0.03]:
                for stop_loss in [0.01, 0.015, 0.02]:
                    for trigger in ["intraday", "close"]:
                        cases.append(
                            FocusCase(
                                "aggressive_trailing_with_stop",
                                entry_rule,
                                ExitRule(
                                    f"trail_target{target:.1%}_dd{trail:.1%}_sl{stop_loss:.1%}_{trigger}_T12",
                                    "trailing",
                                    hold_days=11,
                                    take_profit=target,
                                    stop_loss=stop_loss,
                                    trail_drawdown=trail,
                                    stop_trigger=trigger,
                                ),
                                "高胜率进攻：大目标回撤止盈叠加硬止损",
                            )
                        )
    return cases


def evaluate_case(candidates: pd.DataFrame, case: FocusCase) -> list[dict]:
    entry_df = candidates[apply_entry_rule(candidates, case.entry_rule)].copy()
    trades = simulate_exit(entry_df, case.exit_rule)
    if trades.empty:
        return []

    trades = trades.merge(entry_df[["date", "symbol", "split"]], on=["date", "symbol"], how="left")
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
                "group": case.group,
                "note": case.note,
                "split": split,
                "entry_rule": case.entry_rule.name,
                "exit_rule": case.exit_rule.name,
                "exit_kind": case.exit_rule.kind,
                "hold_days": case.exit_rule.hold_days,
                "take_profit": case.exit_rule.take_profit,
                "stop_loss": case.exit_rule.stop_loss,
                "stop_trigger": case.exit_rule.stop_trigger,
                "trail_drawdown": case.exit_rule.trail_drawdown,
                "raw_trades": raw_trades,
                "skipped_overlaps": skipped_overlaps,
                "overlap_skip_rate": skipped_overlaps / raw_trades if raw_trades else np.nan,
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
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "分组": row["group"],
                "买入规则": row["entry_rule"],
                "卖出规则": row["exit_rule"],
                "交易数": fmt_num(row["trades"]),
                "均值": fmt_pct(row["avg_return_pct"]),
                "胜率": fmt_rate(row["win_rate"]),
                "最大回撤": fmt_pct(row["max_drawdown_pct"]),
                "PF": fmt_float(row["profit_factor"]),
                "止损率": fmt_rate(row["stop_rate"]),
                "最差单笔": fmt_pct(row["min_return_pct"]),
                "全样本跳过": fmt_rate(row["overlap_skip_rate"]),
            }
        )
    return rows


def write_report(summary: pd.DataFrame, output_dir: Path, timestamp: str) -> Path:
    oot = summary[summary["split"] == "oot"].copy()
    stable = oot[oot["group"] == "stable_stop_trigger_compare"].sort_values(
        ["entry_rule", "take_profit", "stop_trigger"]
    )
    robust_fixed = oot[oot["group"] == "new_robust_fixed_stop_compare"].sort_values(
        ["entry_rule", "hold_days", "stop_loss", "stop_trigger"]
    )
    low_dd_fixed = oot[oot["group"] == "low_drawdown_fixed_stop_compare"].sort_values(
        ["entry_rule", "hold_days", "stop_trigger"]
    )

    baseline = oot[
        (oot["group"] == "aggressive_expiry_baseline")
        & (oot["entry_rule"] == "up8_ge_0.70_down3_le_0.45")
        & (oot["exit_rule"].isin(["expiry_T7_close_no_stop", "expiry_T9_close_no_stop", "expiry_T12_close_no_stop"]))
    ].sort_values(["avg_return_pct"], ascending=False)

    candidates = oot[
        (oot["group"].isin(["aggressive_expiry_with_stop", "aggressive_trailing_with_stop"]))
        & (oot["trades"] >= 30)
        & (oot["avg_return_pct"] > 1.0)
        & (oot["profit_factor"] > 1.5)
    ].copy()
    conservative_aggressive = candidates[
        (candidates["max_drawdown_pct"] > -20)
        & (candidates["win_rate"] >= 0.45)
    ].sort_values(["win_rate", "max_drawdown_pct", "avg_return_pct"], ascending=[False, False, False])
    pf_aggressive = candidates[
        candidates["max_drawdown_pct"] > -12
    ].sort_values(["profit_factor", "avg_return_pct", "max_drawdown_pct"], ascending=[False, False, False])
    high_win_kept = candidates[
        candidates["win_rate"] >= 0.52
    ].sort_values(["max_drawdown_pct", "avg_return_pct", "profit_factor"], ascending=[False, False, False])

    path = output_dir / f"b1_stop_close_aggressive_optimization_{timestamp}.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# B1 止损触发方式与高胜率进攻优化\n\n")
        f.write("本轮只做聚焦补充回测，不替换上一轮完整网格结果。所有组合均使用 `non_overlap` 口径：同一股票持仓未结束前不重复买入。\n\n")
        f.write("## 1. 口径说明\n\n")
        f.write("- `intraday`：盘中最低价触及止损线即卖出；如果当日开盘已经低于止损线，则按开盘价成交，收益按真实开盘跌幅计算；否则按止损价成交。\n")
        f.write("- `close`：收盘价跌破止损线才卖出，收益按当日收盘价真实计算，因此单笔亏损可能超过止损阈值。\n")
        f.write("- 回撤止盈同样使用真实可成交假设：达到目标看 `high`，跌破回撤线看 `low`；若开盘已低于回撤止盈线，则按开盘价卖出。\n")
        f.write("- 到期卖出如果不带 `sl`，没有硬止损；如果带 `sl`，会先检查硬止损，再到期收盘卖出。\n\n")

        f.write("## 2. 旧主策略：盘中止损 vs 收盘止损\n\n")
        f.write(markdown_table(format_rows(stable), ["分组", "买入规则", "卖出规则", "交易数", "均值", "胜率", "最大回撤", "PF", "止损率", "最差单笔", "全样本跳过"]))
        f.write("\n\n")

        f.write("## 3. 新稳健候选：固定止盈 10% 对照\n\n")
        f.write(markdown_table(format_rows(robust_fixed), ["分组", "买入规则", "卖出规则", "交易数", "均值", "胜率", "最大回撤", "PF", "止损率", "最差单笔", "全样本跳过"]))
        f.write("\n\n")

        f.write("## 4. 低回撤候选：固定止盈 3% 对照\n\n")
        f.write(markdown_table(format_rows(low_dd_fixed), ["分组", "买入规则", "卖出规则", "交易数", "均值", "胜率", "最大回撤", "PF", "止损率", "最差单笔", "全样本跳过"]))
        f.write("\n\n")

        f.write("## 5. 高胜率进攻原始组合\n\n")
        f.write(markdown_table(format_rows(baseline), ["分组", "买入规则", "卖出规则", "交易数", "均值", "胜率", "最大回撤", "PF", "止损率", "最差单笔", "全样本跳过"]))
        f.write("\n\n")

        f.write("## 6. 进攻优化候选：尽量维持胜率并压回撤\n\n")
        f.write(markdown_table(format_rows(conservative_aggressive.head(20)), ["分组", "买入规则", "卖出规则", "交易数", "均值", "胜率", "最大回撤", "PF", "止损率", "最差单笔", "全样本跳过"]))
        f.write("\n\n")

        f.write("## 7. 进攻优化候选：低回撤高 PF\n\n")
        f.write(markdown_table(format_rows(pf_aggressive.head(20)), ["分组", "买入规则", "卖出规则", "交易数", "均值", "胜率", "最大回撤", "PF", "止损率", "最差单笔", "全样本跳过"]))
        f.write("\n\n")

        f.write("## 8. 仍保持高胜率的候选\n\n")
        f.write(markdown_table(format_rows(high_win_kept.head(20)), ["分组", "买入规则", "卖出规则", "交易数", "均值", "胜率", "最大回撤", "PF", "止损率", "最差单笔", "全样本跳过"]))
        f.write("\n\n")

        f.write("## 9. 初步结论\n\n")
        if not stable.empty:
            best_stable = stable.sort_values(["profit_factor", "max_drawdown_pct"], ascending=[False, False]).iloc[0]
            f.write(
                f"- 主策略中，当前最优仍是 `{best_stable['exit_rule']}`，OOT 平均收益 {fmt_pct(best_stable['avg_return_pct'])}，"
                f"胜率 {fmt_rate(best_stable['win_rate'])}，最大回撤 {fmt_pct(best_stable['max_drawdown_pct'])}，PF {fmt_float(best_stable['profit_factor'])}。\n"
            )
        if not conservative_aggressive.empty:
            best_cons = conservative_aggressive.iloc[0]
            f.write(
                f"- 如果优先维持进攻版胜率，可观察 `{best_cons['entry_rule']} + {best_cons['exit_rule']}`，"
                f"OOT 胜率 {fmt_rate(best_cons['win_rate'])}，平均收益 {fmt_pct(best_cons['avg_return_pct'])}，"
                f"最大回撤 {fmt_pct(best_cons['max_drawdown_pct'])}。\n"
            )
        if not pf_aggressive.empty:
            best_pf = pf_aggressive.iloc[0]
            f.write(
                f"- 如果优先控制回撤和 PF，可观察 `{best_pf['entry_rule']} + {best_pf['exit_rule']}`，"
                f"OOT 平均收益 {fmt_pct(best_pf['avg_return_pct'])}，最大回撤 {fmt_pct(best_pf['max_drawdown_pct'])}，"
                f"PF {fmt_float(best_pf['profit_factor'])}，但胜率只有 {fmt_rate(best_pf['win_rate'])}。\n"
            )
        f.write("- 收盘止损更贴近“日线收盘确认”执行，但亏损不会被严格锁定在止损阈值，极端日可能超过阈值。\n")
    return path


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"loading dataset: {DEFAULT_DATASET}", flush=True)
    candidates = pd.read_parquet(DEFAULT_DATASET)
    candidates["date"] = pd.to_datetime(candidates["date"])
    candidates = predict_xgb_models(candidates, DEFAULT_MODEL_DIR)

    cases = build_focus_cases()
    max_hold_days = max(case.exit_rule.hold_days for case in cases)
    print("adding future prices", flush=True)
    candidates = add_future_prices(candidates, DEFAULT_DAILY_DIR, max_hold_days=max_hold_days)
    candidates = candidates.dropna(subset=["entry_open"]).copy()

    rows = []
    for index, case in enumerate(cases, start=1):
        rows.extend(evaluate_case(candidates, case))
        if index % 50 == 0 or index == len(cases):
            print(f"evaluated {index}/{len(cases)} cases", flush=True)

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"b1_stop_close_aggressive_optimization_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    latest_summary = output_dir / "latest_stop_close_aggressive_optimization.csv"
    summary.to_csv(latest_summary, index=False)
    report_path = write_report(summary, output_dir, timestamp)
    latest_report = output_dir / "latest_stop_close_aggressive_optimization.md"
    latest_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"summary: {summary_path}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
