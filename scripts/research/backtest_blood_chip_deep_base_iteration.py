#!/usr/bin/env python3
"""Mechanism iteration for mature deep bases and retest-reclaim additions."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest_blood_chip_deep_base import (
    PERIODS,
    _decision,
    _json_value,
    _representative_cases,
    _settlement_end,
    _yearly_metrics,
)
from quant.research.blood_chip import load_benchmark
from quant.research.blood_chip_deep_base import (
    DeepBaseExecutionConfig,
    DeepBaseSignalConfig,
    generate_deep_base_signals,
    run_deep_base_backtest,
    summarize_deep_base_result,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate deep-base entries with mature drawdowns and reclaim adds."
    )
    parser.add_argument(
        "--feature-cache",
        default="data/research/blood_chip_deep_base/features.parquet",
    )
    parser.add_argument(
        "--benchmark",
        default="data/raw/index_000300.SH.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/research/blood_chip_deep_base_iteration",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    import json

    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _percent(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2%}"


def _number(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2f}"


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    decision: dict[str, object],
    signal_counts: dict[str, int],
) -> None:
    display = metrics.copy()
    for column in (
        "win_rate",
        "win_rate_wilson_lower_95",
        "median_net_return",
        "capital_weighted_trade_return",
        "total_return",
        "annualized_return",
        "benchmark_total_return",
        "maximum_drawdown",
        "average_deployed_fraction",
        "third_stage_rate",
        "hard_stop_rate",
        "structural_break_rate",
    ):
        display[column] = display[column].map(_percent)
    display["capital_profit_factor"] = display["capital_profit_factor"].map(_number)
    columns = [
        "period",
        "policy",
        "trades",
        "win_rate",
        "win_rate_wilson_lower_95",
        "median_net_return",
        "capital_weighted_trade_return",
        "capital_profit_factor",
        "total_return",
        "annualized_return",
        "benchmark_total_return",
        "maximum_drawdown",
        "average_deployed_fraction",
        "third_stage_rate",
        "hard_stop_rate",
        "structural_break_rate",
    ]
    selected = decision.get("selected_policy")
    conclusion = (
        f"`{selected}` 通过预注册门槛，只建议进入冻结前向模拟。"
        if selected
        else "二次机制迭代仍没有候选通过高胜率门槛；保留研究，不替换线上策略。"
    )
    path.write_text(
        f"""# 深跌筑底型带血筹：成熟底与二次探底确认

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

{conclusion}

- 回溯综合排序第一：`{decision.get('best_research_policy')}`。
- 信号固定为前高跌幅至少65%、峰值距今至少750日、深跌持续至少120日。
- 开发/验证选择不读取2021年以后诊断结果。

## 为什么迭代

首轮只完成第二段的交易在开发、验证期胜率分别只有4.8%和5.3%，说明“仍在区间下半部就加仓”实际放大了继续破位。新执行先等待区间45%以下的二次探底，再要求重新站上中轴且20日收益转正才加30%；第三段要求站到区间70%以上，仍不追出原区间。

## 信号数量

- development_2013_2016：{signal_counts['development_2013_2016']}
- validation_2017_2020：{signal_counts['validation_2017_2020']}
- seen_diagnostic_2021_2024：{signal_counts['seen_diagnostic_2021_2024']}

## 结果

{display[columns].to_markdown(index=False)}

## 决策

门槛仍为每个开发/验证期至少40笔、胜率≥55%、中位净收益>0、资金PF≥1.50、最大回撤≥-35%。当前：`{decision.get('deployment_decision')}`。

2021年以后仅为已见诊断，不是盲测。本报告仅供研究与教育用途，不构成投资建议。
""",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading features: {args.feature_cache}", flush=True)
    features = pd.read_parquet(args.feature_cache)
    benchmark = load_benchmark(Path(args.benchmark), "20100104", "20260821")
    calendar = pd.DatetimeIndex(
        sorted(pd.to_datetime(benchmark["trade_date"].astype(str)).dropna().unique())
    )
    signal_config = DeepBaseSignalConfig(
        minimum_drawdown_from_peak=0.65,
        minimum_peak_age_sessions=750,
        minimum_deep_drawdown_sessions=120,
    )
    signals = generate_deep_base_signals(features, signal_config)
    entry_dates = pd.to_datetime(signals["entry_date"], errors="coerce")
    signal_counts = {
        period: int(entry_dates.between(start, end, inclusive="both").sum())
        for period, (start, end) in PERIODS.items()
    }
    print(f"mature dd65 signals: {len(signals):,}", flush=True)

    metric_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    for hold_sessions in (250, 500):
        for structural_exit_enabled in (True, False):
            exit_key = "structural" if structural_exit_enabled else "hard_only"
            policy = f"dd65_mature_reclaim_hold{hold_sessions}_{exit_key}"
            execution = DeepBaseExecutionConfig(
                stage_policy="retest_reclaim",
                structural_exit_enabled=structural_exit_enabled,
                maximum_holding_sessions=hold_sessions,
            )
            for period, (entry_start, entry_end) in PERIODS.items():
                settlement_end = _settlement_end(
                    calendar,
                    entry_end,
                    hold_sessions,
                    execution.maximum_missing_market_sessions,
                )
                panel = features.loc[
                    features["date"].between(
                        pd.Timestamp(entry_start), settlement_end, inclusive="both"
                    )
                ].copy()
                print(f"running {policy} {period}", flush=True)
                result = run_deep_base_backtest(
                    panel,
                    signals,
                    execution,
                    entry_start,
                    entry_end,
                )
                summary = summarize_deep_base_result(result, benchmark)
                metric_rows.append(
                    {
                        "policy": policy,
                        "period": period,
                        "entry_start": entry_start,
                        "entry_end": entry_end,
                        "maximum_holding_sessions": hold_sessions,
                        "structural_exit_enabled": structural_exit_enabled,
                        **summary,
                    }
                )
                if not result.trades.empty:
                    trades = result.trades.copy()
                    trades["policy"] = policy
                    trades["period"] = period
                    trade_frames.append(trades)

    metrics = pd.DataFrame(metric_rows)
    trades = pd.concat(trade_frames, ignore_index=True, sort=False)
    yearly = _yearly_metrics(trades)
    cases = _representative_cases(trades)
    decision = _decision(metrics)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    trades.to_parquet(output_dir / "trades.parquet", index=False, compression="zstd")
    yearly.to_csv(output_dir / "yearly_metrics.csv", index=False)
    cases.to_csv(output_dir / "representative_cases.csv", index=False)
    _write_json(output_dir / "decision.json", decision)
    _write_json(
        output_dir / "metrics.json",
        {
            "signal_config": signal_config.to_dict(),
            "signal_counts": signal_counts,
            "metrics": metrics.to_dict(orient="records"),
        },
    )
    _write_report(output_dir / "report.md", metrics, decision, signal_counts)
    print(f"decision: {decision['deployment_decision']}", flush=True)


if __name__ == "__main__":
    main()
