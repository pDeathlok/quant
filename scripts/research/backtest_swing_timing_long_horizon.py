#!/usr/bin/env python3
"""Backtest frozen A-share swing exits from 40 sessions to one trading year."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_swing_timing as base
from quant.backtest import AShareExecutionConfig
from quant.data.atomic_io import atomic_write_json
from quant.research.manifest import write_research_manifest
from quant.research.swing_timing import (
    SwingExitRule,
    evaluate_periods,
    select_rule_without_holdout,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/swing_timing_long_horizon_v1"
LONG_PERIODS = [
    ("development_2014_2019", "2014-01-01", "2019-12-31"),
    ("selection_2020_2023", "2020-01-01", "2023-12-31"),
    ("reused_2024", "2024-01-01", "2024-12-31"),
    ("latest_complete_2025", "2025-01-01", "2025-12-31"),
    ("full_complete_sample", "2014-01-01", "2026-12-31"),
]


def frozen_long_exit_rules() -> list[SwingExitRule]:
    """Reference plus 3/6/12-month exits frozen before this run."""

    return [
        SwingExitRule(
            name="reference_tp6_sl6_h40",
            take_profit=0.06,
            stop_loss=0.06,
            hold_days=40,
        ),
        SwingExitRule(
            name="quarter_tp10_sl7_h60",
            take_profit=0.10,
            stop_loss=0.07,
            hold_days=60,
        ),
        SwingExitRule(
            name="half_year_tp15_sl8_h120",
            take_profit=0.15,
            stop_loss=0.08,
            hold_days=120,
        ),
        SwingExitRule(
            name="one_year_tp20_sl10_h244",
            take_profit=0.20,
            stop_loss=0.10,
            hold_days=244,
        ),
        SwingExitRule(
            name="one_year_tp25_sl12_h244",
            take_profit=0.25,
            stop_loss=0.12,
            hold_days=244,
        ),
    ]


def rebuild_period_summary(all_trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for rule_id, trades in all_trades.groupby("rule_id", sort=False):
        period = evaluate_periods(trades, LONG_PERIODS)
        period["rule_id"] = rule_id
        period["entry_rule"] = str(trades["entry_rule"].iloc[0])
        period["exit_rule"] = str(trades["exit_rule"].iloc[0])
        rows.append(period)
    return pd.concat(rows, ignore_index=True)


def diagnostic_acceptance(row: pd.Series) -> dict[str, bool]:
    """High-win, low-frequency gate for the complete 2024 diagnostic year."""

    return {
        "trades_at_least_50": bool(row["trades"] >= 50),
        "win_rate_at_least_60pct": bool(row["win_rate"] >= 0.60),
        "wilson_lower_at_least_50pct": bool(
            row["win_rate_wilson_lower_95"] >= 0.50
        ),
        "profit_factor_at_least_1_30": bool(row["profit_factor"] >= 1.30),
        "average_net_return_positive": bool(row["avg_net_return"] > 0),
        "event_drawdown_no_worse_than_20pct": bool(
            row["event_equity_max_drawdown"] >= -0.20
        ),
    }


def _formatted_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in [
        "win_rate",
        "win_rate_wilson_lower_95",
        "avg_net_return",
        "event_equity_max_drawdown",
        "positive_year_share",
    ]:
        result[column] = result[column].map(base._fmt_pct)
    result["profit_factor"] = result["profit_factor"].map(base._fmt_num)
    for column in ["average_holding_sessions", "median_holding_sessions"]:
        result[column] = result[column].map(
            lambda value: base._fmt_num(value, digits=1)
        )
    return result


def build_long_report(
    *,
    selected: pd.Series,
    passed_selection_constraints: bool,
    selected_periods: pd.DataFrame,
    selection_top: pd.DataFrame,
    acceptance: dict[str, bool],
    audit: dict[str, int],
    gate_counts: dict[str, int],
    exit_rules: list[SwingExitRule],
    signal_start: pd.Timestamp,
    complete_signal_end: pd.Timestamp,
) -> str:
    chosen_id = str(selected["rule_id"])
    period_table = base._markdown_table(
        _formatted_metrics(selected_periods),
        [
            "period",
            "trades",
            "win_rate",
            "win_rate_wilson_lower_95",
            "avg_net_return",
            "profit_factor",
            "average_holding_sessions",
            "median_holding_sessions",
            "event_equity_max_drawdown",
            "positive_year_share",
        ],
    )
    top = selection_top.copy()
    top["eligible"] = (
        top["trades"].ge(50)
        & top["avg_net_return"].gt(0)
        & top["profit_factor"].ge(1.10)
        & top["positive_year_share"].ge(0.50)
    ).map({True: "YES", False: "NO"})
    top_table = base._markdown_table(
        _formatted_metrics(top),
        [
            "rule_id",
            "eligible",
            "trades",
            "win_rate",
            "win_rate_wilson_lower_95",
            "avg_net_return",
            "profit_factor",
            "average_holding_sessions",
        ],
    )
    passed = all(acceptance.values())
    acceptance_lines = "\n".join(
        f"- {'PASS' if value else 'FAIL'} — `{key}`"
        for key, value in acceptance.items()
    )
    exit_lines = "\n".join(
        f"- `{rule.name}`：止盈 {rule.take_profit:.0%}、止损 {rule.stop_loss:.0%}、最长 {rule.hold_days} 个交易日。"
        for rule in exit_rules
    )
    conclusion = (
        "通过长周期诊断门槛，可以进入纸面跟踪，但仍不代表未来收益保证。"
        if passed
        else "未通过长周期高胜率门槛，不能因为延长持有期就直接升级为生产信号。"
    )
    return f"""# 个股波段择时：最长一年持有期回测

## 结论

只使用 2020–2023 选择区，选出的长周期规则是 `{chosen_id}`。

{conclusion}

## 选中规则分期结果

{period_table}

由于一年持有需要 T+250 左右的完整价格路径，最新完整信号只到 {complete_signal_end:%Y-%m-%d}。`reused_2024` 是完整诊断年；`latest_complete_2025` 只是拥有完整一年路径的2025年早期信号，不能代表完整2025年。

## 2024 完整诊断年门槛

{acceptance_lines}

选择区基础资格（交易数 ≥50、平均净收益 >0、利润因子 ≥1.10、活跃年份至少一半为正）{('通过' if passed_selection_constraints else '未通过；当前仅为失败组合中的相对最佳者')}。

## 有限迭代排名（只看 2020–2023）

{top_table}

本轮固定为 4 组入场 × {len(exit_rules)} 组退出，共 {4 * len(exit_rules)} 次试验；没有根据2024–2025结果追加参数。

## 持有期方案

{exit_lines}

一年按244个交易日估算。买入仍为信号后下一交易日开盘，A股T+1；最长持有期从买入日之后的可卖交易日计算，最后保留5个交易日的跌停延迟退出缓冲。

## 日期边界

- 原始日线从2010年开始，但点时好公司样本最早为 {signal_start:%Y-%m-%d}。
- 没有把今天知道的公司质量回填到2010–2014，因此信号历史起点不做虚假延伸。
- 最长路径为 T+{audit['maximum_path_day']}；{audit['complete_max_horizon_rows']:,} 个候选拥有完整路径，{audit['incomplete_or_unmatched_rows']:,} 个候选因路径不完整被排除。
- 入场过滤计数：`{json.dumps(gate_counts, ensure_ascii=False, sort_keys=True)}`。

## 风险说明

- 拉长持有期会减少时间退出，但不会自动提高胜率；止损通常仍会提前结束交易。
- 回撤是按入场日等权聚合的事件回撤，不是建模资金占用、同时持仓数后的组合净值。
- 历史涨跌停和停牌仍使用保守代理，进入实盘前需要补齐正式点时可交易性数据。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weekly-path",
        type=Path,
        default=base.DEFAULT_WEEKLY_PATH,
    )
    parser.add_argument("--daily-dir", type=Path, default=base.DEFAULT_DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entry_rules = base.frozen_entry_rules()
    exit_rules = frozen_long_exit_rules()
    execution = AShareExecutionConfig(slippage=0.0005)

    print("Loading point-in-time weekly signals...")
    signals = base.load_signals(args.weekly_path.resolve())
    print("Attaching complete paths through one trading year...")
    enriched, audit = base.prepare_price_paths(
        signals,
        entry_rules,
        exit_rules,
        args.daily_dir.resolve(),
    )
    print(f"Running {len(entry_rules) * len(exit_rules)} frozen combinations...")
    all_trades, _, yearly_summary, gate_counts = base.run_grid(
        enriched,
        entry_rules,
        exit_rules,
        execution,
    )
    period_summary = rebuild_period_summary(all_trades)
    selected, passed_selection_constraints = select_rule_without_holdout(
        period_summary,
        minimum_trades=50,
    )
    selected_id = str(selected["rule_id"])
    selected_trades = all_trades.loc[all_trades["rule_id"].eq(selected_id)].copy()
    selected_periods = period_summary.loc[
        period_summary["rule_id"].eq(selected_id)
    ].copy()
    diagnostic = selected_periods.loc[
        selected_periods["period"].eq("reused_2024")
    ].iloc[0]
    acceptance = diagnostic_acceptance(diagnostic)
    selection_top = (
        period_summary.loc[
            period_summary["period"].eq("selection_2020_2023")
        ]
        .sort_values(
            ["win_rate_wilson_lower_95", "profit_factor", "avg_net_return"],
            ascending=False,
            na_position="last",
        )
        .head(12)
    )

    exit_metadata = pd.DataFrame(
        [
            {
                "exit_rule": rule.name,
                "hold_days": rule.hold_days,
                "take_profit": rule.take_profit,
                "stop_loss": rule.stop_loss,
            }
            for rule in exit_rules
        ]
    )
    horizon_summary = period_summary.merge(exit_metadata, on="exit_rule", how="left")
    period_summary.to_csv(output_dir / "period_summary.csv", index=False)
    yearly_summary.to_csv(output_dir / "yearly_summary.csv", index=False)
    horizon_summary.to_csv(output_dir / "horizon_summary.csv", index=False)
    selected_trades.to_parquet(output_dir / "selected_trades.parquet", index=False)
    atomic_write_json(
        {
            "schema_version": "swing-timing-long-grid/v1",
            "selection_period": "2020-01-01/2023-12-31",
            "complete_diagnostic_period": "2024-01-01/2024-12-31",
            "selected_rule_id": selected_id,
            "passed_selection_constraints": passed_selection_constraints,
            "diagnostic_acceptance": acceptance,
            "entry_rules": [rule.to_dict() for rule in entry_rules],
            "exit_rules": [rule.to_dict() for rule in exit_rules],
            "execution": execution.to_metadata(),
        },
        output_dir / "frozen_grid.json",
    )
    report = build_long_report(
        selected=selected,
        passed_selection_constraints=passed_selection_constraints,
        selected_periods=selected_periods,
        selection_top=selection_top,
        acceptance=acceptance,
        audit=audit,
        gate_counts=gate_counts,
        exit_rules=exit_rules,
        signal_start=signals["date"].min(),
        complete_signal_end=enriched["date"].max(),
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    write_research_manifest(
        output_dir,
        strategy_name="a_share_swing_timing_up_to_one_year_v1",
        parameters={
            "entry_rules": [asdict(rule) for rule in entry_rules],
            "exit_rules": [asdict(rule) for rule in exit_rules],
            "execution": execution.to_metadata(),
            "selection_minimum_trades": 50,
            "acceptance": acceptance,
        },
        data_paths=[args.weekly_path.resolve()],
        start_date=signals["date"].min().strftime("%Y%m%d"),
        end_date=enriched["date"].max().strftime("%Y%m%d"),
        random_seed=20260809,
        project_root=PROJECT_ROOT,
        code_paths=[
            Path(__file__).resolve(),
            PROJECT_ROOT / "scripts/research/backtest_swing_timing.py",
            PROJECT_ROOT / "src/quant/research/swing_timing.py",
            PROJECT_ROOT / "src/quant/features/variable_library.py",
        ],
        extra={
            "daily_store": base.daily_partition_fingerprint(
                args.daily_dir.resolve()
            ),
            "audit": audit,
            "gate_counts": gate_counts,
            "raw_daily_starts_2010_but_pit_signals_start_2014": True,
            "selection_is_not_fresh_holdout": True,
        },
    )
    print(f"Selected: {selected_id}")
    print(f"Selection constraints passed: {passed_selection_constraints}")
    print(f"2024 diagnostic acceptance passed: {all(acceptance.values())}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
