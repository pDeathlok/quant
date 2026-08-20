#!/usr/bin/env python3
"""Run the frozen v1 low-frequency A-share swing-timing experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtest import AShareExecutionConfig
from quant.data.atomic_io import atomic_write_json
from quant.research.manifest import write_research_manifest
from quant.research.swing_timing import (
    SwingEntryRule,
    SwingExitRule,
    attach_future_price_paths,
    apply_entry_execution_gates,
    apply_same_symbol_cooldown,
    entry_signal_mask,
    evaluate_periods,
    select_rule_without_holdout,
    simulate_swing_exits,
    summarize_swing_trades,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEEKLY_PATH = PROJECT_ROOT / "data/features/long_entry/weekly_training_v2.parquet"
DEFAULT_DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/swing_timing_v1"
SIGNAL_COLUMNS = [
    "date",
    "trade_date",
    "ts_code",
    "name",
    "industry",
    "close",
    "is_good_stock",
    "good_stock_score",
    "historical_value_score_5y",
    "close_to_ma20",
    "close_to_ma60",
    "close_to_ma120",
    "ma_120_slope_20d",
    "return_120d",
    "volatility_60d_cross_section_pct",
    "market_return_13w",
    "market_return_26w",
    "market_drawdown_52w",
    "market_volatility_13w",
]
PERIODS = [
    ("development_2014_2019", "2014-01-01", "2019-12-31"),
    ("selection_2020_2023", "2020-01-01", "2023-12-31"),
    ("reused_2024_2025", "2024-01-01", "2025-12-31"),
    ("latest_2026", "2026-01-01", "2026-12-31"),
    ("full_sample", "2014-01-01", "2026-12-31"),
]


def frozen_entry_rules() -> list[SwingEntryRule]:
    """Four hypothesis-driven variants frozen before exact path backtesting."""

    return [
        SwingEntryRule(name="core_reclaim", min_good_stock_score=60),
        SwingEntryRule(
            name="low_vol_reclaim",
            min_good_stock_score=65,
            max_volatility_percentile=0.50,
        ),
        SwingEntryRule(
            name="quality_value_reclaim",
            min_good_stock_score=68,
            min_historical_value_score=35,
            max_volatility_percentile=0.60,
        ),
        SwingEntryRule(
            name="strict_quality_reclaim",
            min_good_stock_score=70,
            min_historical_value_score=45,
            max_volatility_percentile=0.50,
            max_return_120d=0.30,
            max_close_to_ma20=0.03,
            max_entry_gap=0.01,
        ),
    ]


def frozen_exit_rules() -> list[SwingExitRule]:
    """Four exits, including one deliberately high-hit-rate/poor-payoff check."""

    return [
        SwingExitRule(
            name="balanced_tp8_sl6_h40",
            take_profit=0.08,
            stop_loss=0.06,
            hold_days=40,
        ),
        SwingExitRule(
            name="symmetric_tp6_sl6_h40",
            take_profit=0.06,
            stop_loss=0.06,
            hold_days=40,
        ),
        SwingExitRule(
            name="short_tp6_sl5_h20",
            take_profit=0.06,
            stop_loss=0.05,
            hold_days=20,
        ),
        SwingExitRule(
            name="hit_rate_check_tp5_sl7_h40",
            take_profit=0.05,
            stop_loss=0.07,
            hold_days=40,
        ),
    ]


def daily_partition_fingerprint(daily_dir: Path) -> dict[str, object]:
    paths = sorted((daily_dir.parent / f"{daily_dir.name}_partitioned").glob("year_month=*/data.parquet"))
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        stat = path.stat()
        total_bytes += stat.st_size
        digest.update(
            f"{path.parent.name}/{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode()
        )
    return {
        "method": "sha256(path,size,mtime_ns); content hashes omitted for partitioned 11GB store",
        "partitions": len(paths),
        "bytes": total_bytes,
        "fingerprint": digest.hexdigest(),
        "first_partition": str(paths[0].relative_to(PROJECT_ROOT)) if paths else None,
        "last_partition": str(paths[-1].relative_to(PROJECT_ROOT)) if paths else None,
    }


def _fmt_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.2%}"


def _fmt_num(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    headers = [column for column in columns if column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame[headers].iterrows():
        lines.append(
            "| "
            + " | ".join(str(row[column]).replace("|", "\\|") for column in headers)
            + " |"
        )
    return "\n".join(lines)


def load_signals(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=SIGNAL_COLUMNS)
    missing = sorted(set(SIGNAL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"weekly signal dataset missing columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["symbol"] = frame["ts_code"]
    frame["signal_id"] = (
        frame["ts_code"] + "|" + frame["date"].dt.strftime("%Y%m%d")
    )
    return frame.sort_values(["date", "ts_code"]).reset_index(drop=True)


def prepare_price_paths(
    signals: pd.DataFrame,
    entry_rules: list[SwingEntryRule],
    exit_rules: list[SwingExitRule],
    daily_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    masks = [entry_signal_mask(signals, rule) for rule in entry_rules]
    union_mask = pd.concat(masks, axis=1).any(axis=1)
    candidates = signals.loc[union_mask].copy()
    maximum_path_day = max(
        rule.hold_days + 1 + rule.exit_grace_days for rule in exit_rules
    )
    enriched = attach_future_price_paths(
        candidates,
        daily_dir,
        maximum_path_day=maximum_path_day,
    )
    complete_column = f"close_t{maximum_path_day}"
    complete = enriched.loc[enriched[complete_column].notna()].copy()
    audit = {
        "weekly_rows": int(len(signals)),
        "union_signal_rows": int(len(candidates)),
        "price_matched_rows": int(len(enriched)),
        "complete_max_horizon_rows": int(len(complete)),
        "incomplete_or_unmatched_rows": int(len(candidates) - len(complete)),
        "maximum_path_day": maximum_path_day,
    }
    return complete, audit


def run_grid(
    enriched: pd.DataFrame,
    entry_rules: list[SwingEntryRule],
    exit_rules: list[SwingExitRule],
    execution: AShareExecutionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    all_period_rows: list[pd.DataFrame] = []
    all_year_rows: list[dict[str, object]] = []
    selected_columns = [
        "rule_id",
        "entry_rule",
        "exit_rule",
        "date",
        "entry_date",
        "exit_date",
        "ts_code",
        "name",
        "industry",
        "entry_gap",
        "entry_fill",
        "exit_fill",
        "exit_day",
        "exit_type",
        "gross_return",
        "net_return",
        "fees",
        "shares",
        "good_stock_score",
        "historical_value_score_5y",
        "close_to_ma20",
        "close_to_ma60",
        "close_to_ma120",
        "ma_120_slope_20d",
        "return_120d",
        "volatility_60d_cross_section_pct",
        "market_return_13w",
        "market_return_26w",
        "market_drawdown_52w",
        "market_volatility_13w",
    ]
    carry_columns = [
        column
        for column in selected_columns
        if column
        not in {
            "rule_id",
            "entry_rule",
            "exit_rule",
            "exit_date",
            "entry_fill",
            "exit_fill",
            "exit_day",
            "exit_type",
            "gross_return",
            "net_return",
            "fees",
            "shares",
        }
    ]
    all_trade_frames: list[pd.DataFrame] = []
    gate_counts: dict[str, int] = {}

    for entry_rule in entry_rules:
        signal_rows = enriched.loc[entry_signal_mask(enriched, entry_rule)].copy()
        executable = apply_entry_execution_gates(signal_rows, entry_rule)
        executable["entry_date"] = pd.to_datetime(executable["date_t1"])
        executable = apply_same_symbol_cooldown(
            executable,
            entry_rule.cooldown_calendar_days,
        )
        gate_counts[f"{entry_rule.name}:signal"] = int(len(signal_rows))
        gate_counts[f"{entry_rule.name}:executable_after_cooldown"] = int(
            len(executable)
        )
        for exit_rule in exit_rules:
            rule_id = f"{entry_rule.name}__{exit_rule.name}"
            trades = simulate_swing_exits(
                executable,
                exit_rule,
                execution=execution,
                result_columns=carry_columns,
            )
            if trades.empty:
                continue
            trades["rule_id"] = rule_id
            trades["entry_rule"] = entry_rule.name
            trades["exit_rule"] = exit_rule.name
            all_trade_frames.append(
                trades[[column for column in selected_columns if column in trades.columns]].copy()
            )

            period = evaluate_periods(trades, PERIODS)
            period["rule_id"] = rule_id
            period["entry_rule"] = entry_rule.name
            period["exit_rule"] = exit_rule.name
            all_period_rows.append(period)

            years = pd.to_datetime(trades["entry_date"]).dt.year
            for year, sample in trades.groupby(years, sort=True):
                all_year_rows.append(
                    {
                        "rule_id": rule_id,
                        "entry_rule": entry_rule.name,
                        "exit_rule": exit_rule.name,
                        "year": int(year),
                        **summarize_swing_trades(sample),
                    }
                )

    if not all_trade_frames or not all_period_rows:
        raise RuntimeError("frozen grid generated no executable trades")
    return (
        pd.concat(all_trade_frames, ignore_index=True),
        pd.concat(all_period_rows, ignore_index=True),
        pd.DataFrame(all_year_rows),
        gate_counts,
    )


def holdout_acceptance(row: pd.Series) -> dict[str, bool]:
    return {
        "trades_at_least_80": bool(row["trades"] >= 80),
        "win_rate_at_least_60pct": bool(row["win_rate"] >= 0.60),
        "wilson_lower_at_least_55pct": bool(
            row["win_rate_wilson_lower_95"] >= 0.55
        ),
        "profit_factor_at_least_1_30": bool(row["profit_factor"] >= 1.30),
        "average_net_return_positive": bool(row["avg_net_return"] > 0),
        "event_drawdown_no_worse_than_15pct": bool(
            row["event_equity_max_drawdown"] >= -0.15
        ),
        "both_reused_years_positive": bool(row["positive_year_share"] >= 1.0),
    }


def build_report(
    *,
    selected: pd.Series,
    passed_selection_constraints: bool,
    selected_periods: pd.DataFrame,
    selection_top: pd.DataFrame,
    acceptance: dict[str, bool],
    audit: dict[str, int],
    gate_counts: dict[str, int],
    entry_rules: list[SwingEntryRule],
    exit_rules: list[SwingExitRule],
    execution: AShareExecutionConfig,
) -> str:
    chosen_id = str(selected["rule_id"])
    chosen_periods = selected_periods.copy()
    for column in [
        "win_rate",
        "win_rate_wilson_lower_95",
        "avg_net_return",
        "event_equity_max_drawdown",
        "positive_year_share",
    ]:
        chosen_periods[column] = chosen_periods[column].map(_fmt_pct)
    chosen_periods["profit_factor"] = chosen_periods["profit_factor"].map(_fmt_num)
    top = selection_top.copy()
    top["eligible"] = (
        top["trades"].ge(80)
        & top["avg_net_return"].gt(0)
        & top["profit_factor"].ge(1.10)
        & top["positive_year_share"].ge(0.50)
    ).map({True: "YES", False: "NO"})
    for column in ["win_rate", "win_rate_wilson_lower_95", "avg_net_return"]:
        top[column] = top[column].map(_fmt_pct)
    top["profit_factor"] = top["profit_factor"].map(_fmt_num)
    passed = all(acceptance.values())
    conclusion = (
        "通过预注册门槛，可进入纸面跟踪；仍不能直接视为实盘保证。"
        if passed
        else "未通过预注册门槛，不能发布为高胜率生产信号；应继续研究或保持空仓。"
    )
    acceptance_lines = "\n".join(
        f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in acceptance.items()
    )
    period_table = _markdown_table(
        chosen_periods,
        [
            "period",
            "trades",
            "win_rate",
            "win_rate_wilson_lower_95",
            "avg_net_return",
            "profit_factor",
            "event_equity_max_drawdown",
            "positive_year_share",
        ],
    )
    top_table = _markdown_table(
        top,
        [
            "rule_id",
            "eligible",
            "trades",
            "win_rate",
            "win_rate_wilson_lower_95",
            "avg_net_return",
            "profit_factor",
        ],
    )
    return f"""# 个股波段择时 v1：冻结规则回测

## 结论

按 2020–2023 选择区、在不读取 2024–2026 指标的选择函数中，选出的规则是 `{chosen_id}`。

{conclusion}

这里的“胜率”是扣除双边滑点、佣金、过户费和卖出印花税后的单笔正收益比例。胜率不是唯一目标；低止盈、宽止损虽然可能提高命中率，但若期望值和利润因子不合格会被拒绝。

## 选中规则分期结果

{period_table}

`development_2014_2019` 用于观察跨周期稳定性；`selection_2020_2023` 是唯一允许用于规则选择的区间；`reused_2024_2025` 是复用诊断区；`latest_2026` 仅含具备完整最长持有路径的样本。由于此前项目已多次观察 2024–2026，这两段不能被称为全新盲测。

## 验收门槛（2024–2025 复用诊断区）

{acceptance_lines}

选择区的基础可用性约束（交易数 ≥80、平均净收益 >0、利润因子 ≥1.10、至少一半年份为正）{('通过' if passed_selection_constraints else '未通过；当前规则只是失败组合中的相对最佳者')}。

## 有限迭代排名（只看 2020–2023）

{top_table}

总试验数固定为 {len(entry_rules) * len(exit_rules)}：4 组入场 × 4 组退出。没有根据 2024–2026 结果追加参数。

## 研究设计

- 标的：周频点时好公司样本，不使用未来才知道的公司质量信息。
- 大势：13周与26周市场收益为正、52周回撤高于 -18%、13周波动率低于 30%。
- 趋势：价格高于 MA60/MA120、MA120 斜率为正、120日涨幅在冻结区间内。
- 买点：周末收盘重新站在 MA20 上方且距离不超过 3%–4%；下一交易日开盘跳空必须在 -3% 至 +1%/+1.5% 内。
- 执行：信号收盘后才生成订单，T+1 开盘买入，买入日不得卖出；同日止盈止损同时触发时按止损优先，跳空穿止损按开盘成交。
- 可交易性：一字涨停不买；一字跌停不卖并最多延迟 5 个交易日。历史正式涨跌停/停牌快照不完整，这是代理规则而非完整交易所撮合。
- 成本：名义单笔 10 万元；双边滑点 {execution.slippage:.2%}，佣金 {execution.commission_rate:.2%}（单边最低 5 元），过户费 {execution.transfer_fee_rate:.3%}，卖出印花税 {execution.stamp_tax_rate:.2%}。
- 冷却：同一股票两个入场信号至少间隔 56 个自然日；不强制每天选 Top-N，无合格信号时空仓。
- 回撤：报告的是按入场日等权聚合事件收益的研究型回撤，不是含仓位上限和资金占用的完整组合净值回撤。

## 数据审计

- 周频点时样本：{audit['weekly_rows']:,} 行。
- 冻结规则并集：{audit['union_signal_rows']:,} 行。
- 匹配到 T+1 日线路径：{audit['price_matched_rows']:,} 行。
- 具备最长路径：{audit['complete_max_horizon_rows']:,} 行；不完整或未匹配 {audit['incomplete_or_unmatched_rows']:,} 行。
- 最长所需路径：T+{audit['maximum_path_day']}（含跌停延迟退出缓冲）。
- 各入场门过滤计数：`{json.dumps(gate_counts, ensure_ascii=False, sort_keys=True)}`。

## 不能从本报告推出的结论

- 不能把复用区结果当作真正未见样本，也不能保证未来仍有相同胜率。
- 没有建模组合同时持仓数、资金容量、冲击成本和排队成交，不能据此直接给出仓位规模。
- 历史可交易性使用保守代理；若进入下一阶段，应补齐逐日涨跌停、停牌、ST 与上市状态点时快照，再做封存盲测和纸面跟踪。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly-path", type=Path, default=DEFAULT_WEEKLY_PATH)
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entry_rules = frozen_entry_rules()
    exit_rules = frozen_exit_rules()
    execution = AShareExecutionConfig(slippage=0.0005)

    print("Loading point-in-time weekly signals...")
    signals = load_signals(args.weekly_path.resolve())
    print("Attaching causal T+1 and future daily price paths...")
    enriched, audit = prepare_price_paths(
        signals,
        entry_rules,
        exit_rules,
        args.daily_dir.resolve(),
    )
    print(f"Running {len(entry_rules) * len(exit_rules)} frozen combinations...")
    all_trades, period_summary, yearly_summary, gate_counts = run_grid(
        enriched,
        entry_rules,
        exit_rules,
        execution,
    )
    selected, passed_selection_constraints = select_rule_without_holdout(
        period_summary
    )
    selected_id = str(selected["rule_id"])
    selected_trades = all_trades.loc[all_trades["rule_id"].eq(selected_id)].copy()
    selected_periods = period_summary.loc[
        period_summary["rule_id"].eq(selected_id)
    ].copy()
    reused = selected_periods.loc[
        selected_periods["period"].eq("reused_2024_2025")
    ].iloc[0]
    acceptance = holdout_acceptance(reused)

    selection_top = (
        period_summary.loc[
            period_summary["period"].eq("selection_2020_2023")
        ]
        .sort_values(
            ["win_rate_wilson_lower_95", "profit_factor", "avg_net_return"],
            ascending=False,
            na_position="last",
        )
        .head(10)
    )
    period_summary.to_csv(output_dir / "period_summary.csv", index=False)
    yearly_summary.to_csv(output_dir / "yearly_summary.csv", index=False)
    selected_trades.to_parquet(output_dir / "selected_trades.parquet", index=False)
    atomic_write_json(
        {
            "schema_version": "swing-timing-grid/v1",
            "selection_period": "2020-01-01/2023-12-31",
            "reused_diagnostic_period": "2024-01-01/2025-12-31",
            "selected_rule_id": selected_id,
            "passed_selection_constraints": passed_selection_constraints,
            "holdout_acceptance": acceptance,
            "entry_rules": [rule.to_dict() for rule in entry_rules],
            "exit_rules": [rule.to_dict() for rule in exit_rules],
            "execution": execution.to_metadata(),
        },
        output_dir / "frozen_grid.json",
    )
    report = build_report(
        selected=selected,
        passed_selection_constraints=passed_selection_constraints,
        selected_periods=selected_periods,
        selection_top=selection_top,
        acceptance=acceptance,
        audit=audit,
        gate_counts=gate_counts,
        entry_rules=entry_rules,
        exit_rules=exit_rules,
        execution=execution,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    daily_fingerprint = daily_partition_fingerprint(args.daily_dir.resolve())
    write_research_manifest(
        output_dir,
        strategy_name="a_share_low_frequency_swing_timing_v1",
        parameters={
            "entry_rules": [asdict(rule) for rule in entry_rules],
            "exit_rules": [asdict(rule) for rule in exit_rules],
            "execution": execution.to_metadata(),
            "selection_function": "wilson_lower_then_profit_factor_without_holdout",
            "acceptance": acceptance,
        },
        data_paths=[args.weekly_path.resolve()],
        start_date=signals["date"].min().strftime("%Y%m%d"),
        end_date=signals["date"].max().strftime("%Y%m%d"),
        random_seed=20260809,
        project_root=PROJECT_ROOT,
        code_paths=[
            Path(__file__).resolve(),
            PROJECT_ROOT / "src/quant/research/swing_timing.py",
            PROJECT_ROOT / "src/quant/features/variable_library.py",
        ],
        extra={
            "daily_store": daily_fingerprint,
            "audit": audit,
            "gate_counts": gate_counts,
            "selection_is_not_fresh_holdout": True,
            "historical_tradability": "one-price limit proxy; formal PIT snapshots incomplete",
        },
    )
    print(f"Selected: {selected_id}")
    print(f"Selection constraints passed: {passed_selection_constraints}")
    print(f"Reused acceptance passed: {all(acceptance.values())}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
