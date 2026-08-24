#!/usr/bin/env python3
"""Backtest deep-drawdown blood-chip bases with long staged holding."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.research.blood_chip import load_benchmark, load_canonical_daily
from quant.research.blood_chip_deep_base import (
    DeepBaseExecutionConfig,
    DeepBaseSignalConfig,
    build_deep_base_features,
    generate_deep_base_signals,
    run_deep_base_backtest,
    summarize_deep_base_result,
)


PERIODS = {
    "development_2013_2016": ("2013-01-04", "2016-12-30"),
    "validation_2017_2020": ("2017-01-03", "2020-12-31"),
    "seen_diagnostic_2021_2024": ("2021-01-04", "2024-07-30"),
}
DRAW_DOWN_THRESHOLDS = (0.50, 0.65, 0.80)
HOLDING_SESSIONS = (250, 500)
SELECTION_PERIODS = ("development_2013_2016", "validation_2017_2020")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest deep-drawdown exhausted-selling price bases."
    )
    parser.add_argument(
        "--daily-root",
        default="data/raw/daily_partitioned",
        help="Canonical monthly daily-bar partition root.",
    )
    parser.add_argument(
        "--benchmark",
        default="data/raw/index_000300.SH.parquet",
        help="CSI 300 benchmark parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/research/blood_chip_deep_base",
        help="Research output directory.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/research/blood_chip_deep_base",
        help="Feature cache directory.",
    )
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="Rebuild the causal feature cache from canonical daily bars.",
    )
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_or_build_features(
    daily_root: Path,
    cache_dir: Path,
    *,
    build_cache: bool,
) -> pd.DataFrame:
    cache_path = cache_dir / "features.parquet"
    if cache_path.exists() and not build_cache:
        print(f"loading feature cache: {cache_path}", flush=True)
        return pd.read_parquet(cache_path)
    print("loading canonical daily bars 2010-01-04..2026-08-21", flush=True)
    daily = load_canonical_daily(daily_root, "20100104", "20260821")
    print(f"building deep-base features for {len(daily):,} rows", flush=True)
    features = build_deep_base_features(daily)
    cache_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cache_path, index=False, compression="zstd")
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(features)),
        "symbols": int(features["ts_code"].nunique()),
        "first_date": pd.Timestamp(features["date"].min()).date().isoformat(),
        "last_date": pd.Timestamp(features["date"].max()).date().isoformat(),
        "price_contract": "causal_continuous_forward_only",
    }
    _write_json(cache_dir / "features.metadata.json", metadata)
    print(f"wrote feature cache: {cache_path}", flush=True)
    return features


def _settlement_end(
    calendar: pd.DatetimeIndex,
    entry_end: str,
    hold_sessions: int,
    missing_sessions: int,
) -> pd.Timestamp:
    end = pd.Timestamp(entry_end)
    position = int(calendar.searchsorted(end, side="right") - 1)
    settlement_position = min(
        position + hold_sessions + missing_sessions + 5,
        len(calendar) - 1,
    )
    return pd.Timestamp(calendar[settlement_position])


def _yearly_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["entry_year"] = pd.to_datetime(frame["entry_date"]).dt.year
    frame["pnl"] = (
        pd.to_numeric(frame["exit_value"], errors="coerce")
        - pd.to_numeric(frame["fees"], errors="coerce")
        - pd.to_numeric(frame["entry_value"], errors="coerce")
    )
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["policy", "period", "entry_year"], observed=True, sort=True
    ):
        returns = pd.to_numeric(group["net_return"], errors="coerce")
        losses = float(-group.loc[group["pnl"] <= 0, "pnl"].sum())
        rows.append(
            {
                "policy": keys[0],
                "period": keys[1],
                "entry_year": int(keys[2]),
                "trades": int(len(group)),
                "win_rate": float(returns.gt(0).mean()),
                "average_net_return": float(returns.mean()),
                "median_net_return": float(returns.median()),
                "capital_profit_factor": (
                    float(group.loc[group["pnl"] > 0, "pnl"].sum() / losses)
                    if losses > 0
                    else np.nan
                ),
                "hard_stop_rate": float(
                    group["exit_reason"].eq("hard_stop").mean()
                ),
                "structural_break_rate": float(
                    group["exit_reason"].eq("structural_break").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _representative_cases(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for keys, group in trades.groupby(["policy", "period"], observed=True, sort=False):
        ordered = group.sort_values("net_return")
        selections = [
            ("largest_losses", ordered.head(3)),
            ("largest_winners", ordered.tail(3).sort_values("net_return", ascending=False)),
        ]
        for label, selected in selections:
            case = selected.copy()
            case.insert(0, "case_type", label)
            case.insert(0, "case_period", keys[1])
            case.insert(0, "case_policy", keys[0])
            rows.append(case)
    return pd.concat(rows, ignore_index=True, sort=False)


def _decision(metrics: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, dict[str, bool]] = {}
    scores: list[tuple[tuple[float, float, float, int], str]] = []
    for policy, group in metrics.groupby("policy", observed=True, sort=False):
        selected = group.set_index("period").reindex(SELECTION_PERIODS)
        complete = not selected.isna().all(axis=1).any()
        policy_checks = {
            "periods_complete": bool(complete),
            "minimum_40_trades_each": bool(
                complete and selected["trades"].ge(40).all()
            ),
            "win_rate_at_least_55pct_each": bool(
                complete and selected["win_rate"].ge(0.55).all()
            ),
            "positive_median_each": bool(
                complete and selected["median_net_return"].gt(0.0).all()
            ),
            "capital_pf_at_least_1_50_each": bool(
                complete and selected["capital_profit_factor"].ge(1.50).all()
            ),
            "maximum_drawdown_not_below_minus_35pct_each": bool(
                complete and selected["maximum_drawdown"].ge(-0.35).all()
            ),
        }
        policy_checks["all_passed"] = all(policy_checks.values())
        checks[str(policy)] = policy_checks
        if complete:
            annualized = selected["annualized_return"].clip(lower=-0.999999)
            annualized_geometric = float(
                np.prod(1.0 + annualized.to_numpy(dtype=float)) ** 0.5 - 1.0
            )
            score = (
                float(selected["win_rate"].min()),
                float(selected["capital_profit_factor"].min()),
                annualized_geometric,
                int(selected["trades"].sum()),
            )
            scores.append((score, str(policy)))
    qualified = [policy for _, policy in scores if checks[policy]["all_passed"]]
    qualified_scores = [(score, policy) for score, policy in scores if policy in qualified]
    selected_policy = max(qualified_scores)[1] if qualified_scores else None
    best_research_policy = max(scores)[1] if scores else None
    return {
        "selected_on_development_and_validation_only": True,
        "selection_periods": list(SELECTION_PERIODS),
        "seen_diagnostic_excluded_from_selection": True,
        "qualification_thresholds": {
            "minimum_trades_each_period": 40,
            "minimum_win_rate_each_period": 0.55,
            "minimum_median_net_return_each_period": 0.0,
            "minimum_capital_profit_factor_each_period": 1.50,
            "minimum_maximum_drawdown_each_period": -0.35,
        },
        "checks": checks,
        "qualified_policies": qualified,
        "selected_policy": selected_policy,
        "best_research_policy": best_research_policy,
        "deployment_decision": (
            "enter_forward_shadow"
            if selected_policy is not None
            else "research_only_keep_existing_online_strategy"
        ),
    }


def _percent(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2%}"


def _number(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2f}"


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    decision: dict[str, Any],
    signal_counts: dict[str, dict[str, int]],
) -> None:
    table = metrics.copy()
    table["trades"] = table["trades"].astype(int)
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
    ):
        table[column] = table[column].map(_percent)
    table["capital_profit_factor"] = table["capital_profit_factor"].map(_number)
    display_columns = [
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
    ]
    selected = decision.get("selected_policy")
    best = decision.get("best_research_policy")
    if selected:
        conclusion = f"`{selected}` 通过预注册高胜率门槛，建议只进入冻结前向模拟。"
    else:
        conclusion = (
            "没有候选同时通过预注册的高胜率、正中位数、资金PF和回撤门槛；"
            "保留研究，不替换现有线上策略。"
        )
    signal_lines = []
    for threshold, counts in signal_counts.items():
        detail = "，".join(f"{period}={count}" for period, count in counts.items())
        signal_lines.append(f"- 前高跌幅至少 {threshold}：{detail}")
    content = f"""# 深跌筑底型带血筹长持回测

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

{conclusion}

- 开发和验证阶段的综合排序第一为 `{best}`；这只是回溯研究最优，不等于可上线。
- 2021 年以后是已见诊断区间，不是独立盲测。
- 只有 `selected_policy` 非空时才允许进入冻结前向模拟，本研究从不直接替换线上策略。

## 策略定义

- 从截至信号日前一日可见的历史前高至少下跌 50%/65%/80%。
- 深跌状态持续至少 40 个交易日，峰值距今至少 120 个交易日。
- 60 日价格区间宽度不超过 25%，60 日收益位于 -12% 至 +12%。
- 最近 20 日下跌成交占比与波动均相对前期收缩，且当前价格位于区间 15%—65%。
- 20% 首仓；区间下半部存活确认后加 30%；区间中上部且 20 日收益转正后加 50%。
- 不要求建满；结构破位或区间下沿下方 10% 硬止损；不止盈，比较 250/500 日持有。
- 连续 60 个市场交易日无行情按全额损失核销，避免退市样本按最后价格冻结。

## 信号数量

{chr(10).join(signal_lines)}

## 组合结果

{table[display_columns].to_markdown(index=False)}

## 决策门槛

每个候选必须在 2013—2016 和 2017—2020 两个入场期分别满足：至少40笔、胜率≥55%、中位净收益>0、资金PF≥1.50、最大回撤≥-35%。当前决策：`{decision['deployment_decision']}`。

## 解释边界

- “卖压耗尽”由成交占比、波动收缩和价格横盘代理，不能识别真实卖方身份。
- 本地因果前高始于 2010 年；更早历史峰值不可见，部分长期下跌公司的真实跌幅可能被低估。
- 长持回测对退市、长期停牌和极端跳空高度敏感；核销规则是保守近似，不等同真实可成交价格。
- 组合结果包含现金与容量约束，单笔结果不能替代组合路径。

本报告仅供研究与教育用途，不构成个性化投资建议、收益承诺或交易指令。
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = _load_or_build_features(
        Path(args.daily_root),
        cache_dir,
        build_cache=bool(args.build_cache),
    )
    benchmark = load_benchmark(Path(args.benchmark), "20100104", "20260821")
    calendar = pd.DatetimeIndex(
        sorted(pd.to_datetime(benchmark["trade_date"].astype(str)).dropna().unique())
    )

    signals_by_threshold: dict[float, pd.DataFrame] = {}
    signal_counts: dict[str, dict[str, int]] = {}
    for threshold in DRAW_DOWN_THRESHOLDS:
        print(f"generating signals for drawdown >= {threshold:.0%}", flush=True)
        config = DeepBaseSignalConfig(minimum_drawdown_from_peak=threshold)
        signals = generate_deep_base_signals(features, config)
        signals_by_threshold[threshold] = signals
        counts: dict[str, int] = {}
        for period, (start, end) in PERIODS.items():
            entry_dates = pd.to_datetime(signals["entry_date"], errors="coerce")
            counts[period] = int(entry_dates.between(start, end, inclusive="both").sum())
        signal_counts[f"{threshold:.0%}"] = counts
        print(f"signals: {len(signals):,}", flush=True)

    metric_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    for threshold in DRAW_DOWN_THRESHOLDS:
        signals = signals_by_threshold[threshold]
        for hold_sessions in HOLDING_SESSIONS:
            policy = f"dd{int(threshold * 100)}_hold{hold_sessions}"
            execution = DeepBaseExecutionConfig(
                maximum_holding_sessions=hold_sessions
            )
            for period, (entry_start, entry_end) in PERIODS.items():
                settlement_end = _settlement_end(
                    calendar,
                    entry_end,
                    hold_sessions,
                    execution.maximum_missing_market_sessions,
                )
                panel_start = pd.Timestamp(entry_start)
                panel = features.loc[
                    features["date"].between(
                        panel_start,
                        settlement_end,
                        inclusive="both",
                    )
                ].copy()
                print(
                    f"running {policy} {period} through {settlement_end.date()}",
                    flush=True,
                )
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
                        "minimum_drawdown_from_peak": threshold,
                        "maximum_holding_sessions": hold_sessions,
                        "period": period,
                        "entry_start": entry_start,
                        "entry_end": entry_end,
                        "settlement_end": settlement_end,
                        **summary,
                    }
                )
                if not result.trades.empty:
                    trades = result.trades.copy()
                    trades["policy"] = policy
                    trades["period"] = period
                    trade_frames.append(trades)

    metrics = pd.DataFrame(metric_rows)
    trades = (
        pd.concat(trade_frames, ignore_index=True, sort=False)
        if trade_frames
        else pd.DataFrame()
    )
    yearly = _yearly_metrics(trades)
    cases = _representative_cases(trades)
    decision = _decision(metrics)

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    _write_json(
        output_dir / "metrics.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "periods": PERIODS,
            "signal_counts": signal_counts,
            "metrics": metrics.to_dict(orient="records"),
        },
    )
    trades.to_parquet(output_dir / "trades.parquet", index=False, compression="zstd")
    yearly.to_csv(output_dir / "yearly_metrics.csv", index=False)
    cases.to_csv(output_dir / "representative_cases.csv", index=False)
    _write_json(output_dir / "decision.json", decision)
    _write_report(output_dir / "report.md", metrics, decision, signal_counts)
    print(f"decision: {decision['deployment_decision']}", flush=True)
    print(f"report: {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
