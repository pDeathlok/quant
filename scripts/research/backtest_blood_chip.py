#!/usr/bin/env python3
"""Build, iterate and freeze the price-volume blood-chip backtest."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.research.blood_chip import (
    BloodChipBacktestConfig,
    BloodChipBacktestResult,
    BloodChipSignalConfig,
    analyze_blood_chip_cases,
    build_blood_chip_features,
    generate_blood_chip_signals,
    load_benchmark,
    load_canonical_daily,
    run_blood_chip_backtest,
    summarize_blood_chip_result,
)


DEVELOPMENT_PERIOD = ("development_2014_2019", "2014-01-01", "2019-12-31")
ITERATION_PERIOD = ("iteration_2020_2022", "2020-01-01", "2022-12-30")
HOLDOUT_PERIOD = ("holdout_2023_2026", "2023-01-03", "2026-02-06")

CACHE_COLUMNS = [
    "ts_code",
    "trade_date",
    "date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_chg",
    "vol",
    "amount",
    "adjustment_factor",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "market_return_1d",
    "market_return_60d",
    "market_beta_60d",
    "residual_return_1d",
    "residual_return_3d",
    "residual_return_5d",
    "history_days",
    "prior_amount_median_20d",
    "amount_ratio_5d",
    "down_impact_3d",
    "down_impact_5d",
    "impact_ratio_5d",
    "drawdown_20d",
    "return_120d",
    "volatility_60d",
    "clv_3d",
    "residual_5d_percentile",
    "amount_ratio_5d_percentile",
    "impact_ratio_5d_percentile",
    "drawdown_20d_percentile",
    "shock_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest suspected A-share fire sales with causal price-volume signals."
    )
    parser.add_argument("--daily-root", default="data/raw/daily_partitioned")
    parser.add_argument("--benchmark", default="data/raw/index_000300.SH.parquet")
    parser.add_argument("--output-dir", default="reports/research/blood_chip")
    parser.add_argument("--cache-dir", default="data/research/blood_chip")
    parser.add_argument("--start-date", default="20130104")
    parser.add_argument("--end-date", default="20260807")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--frozen-holdout", action="store_true")
    args = parser.parse_args()
    if args.development_only and args.frozen_holdout:
        parser.error("--development-only and --frozen-holdout are mutually exclusive")
    return args


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_or_build_features(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / "features.parquet"
    benchmark = load_benchmark(args.benchmark, args.start_date, args.end_date)
    if feature_path.exists() and not args.build_cache:
        print(f"loading cached features: {feature_path}", flush=True)
        return pd.read_parquet(feature_path), benchmark
    print("loading canonical daily partitions", flush=True)
    daily = load_canonical_daily(args.daily_root, args.start_date, args.end_date)
    print(f"building causal features for {len(daily):,} rows", flush=True)
    features = build_blood_chip_features(daily, benchmark)
    del daily
    gc.collect()
    features = features[[column for column in CACHE_COLUMNS if column in features]].copy()
    print(f"writing feature cache: {feature_path}", flush=True)
    features.to_parquet(feature_path, index=False, compression="zstd")
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_sha": _git_sha(),
        "start_date": str(features["trade_date"].min()),
        "end_date": str(features["trade_date"].max()),
        "rows": len(features),
        "symbols": int(features["ts_code"].nunique()),
        "price_basis": "causal_continuous_from_raw_ohlc_and_pre_close",
        "amount_unit": "thousand_cny",
        "source": str(Path(args.daily_root).resolve()),
        "benchmark": str(Path(args.benchmark).resolve()),
    }
    _write_json(cache_dir / "features.metadata.json", metadata)
    return features, benchmark


def _candidate_signal_configs() -> dict[str, BloodChipSignalConfig]:
    baseline = BloodChipSignalConfig()
    return {
        "baseline": baseline,
        "avoid_high_volatility": replace(
            baseline,
            maximum_volatility_60d=0.60,
        ),
        "avoid_extreme_spiral": replace(
            baseline,
            maximum_volatility_60d=0.60,
            minimum_market_return_60d=-0.20,
        ),
        "avoid_overextended": replace(
            baseline,
            maximum_return_120d=0.75,
            maximum_volatility_60d=0.60,
            minimum_market_return_60d=-0.20,
        ),
        "conservative_quality": replace(
            baseline,
            maximum_return_120d=0.50,
            maximum_volatility_60d=0.55,
            minimum_market_return_60d=-0.15,
        ),
    }


def _positive_year_share(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    years = pd.to_datetime(trades["entry_date"]).dt.year
    annual = pd.to_numeric(trades["net_return"], errors="coerce").groupby(years).mean()
    return float((annual > 0).mean()) if not annual.empty else np.nan


def _run_period(
    features: pd.DataFrame,
    signals: pd.DataFrame,
    benchmark: pd.DataFrame,
    backtest_config: BloodChipBacktestConfig,
    period: tuple[str, str, str],
) -> tuple[BloodChipBacktestResult, dict[str, Any]]:
    name, start, end = period
    result = run_blood_chip_backtest(
        features,
        signals,
        backtest_config,
        entry_start=start,
        entry_end=end,
    )
    metrics = summarize_blood_chip_result(result, benchmark)
    metrics.update(
        {
            "period": name,
            "entry_start": start,
            "entry_end": end,
            "positive_year_share": _positive_year_share(result.trades),
        }
    )
    return result, metrics


def _select_without_holdout(
    metric_rows: list[dict[str, Any]],
) -> tuple[str, bool, str]:
    table = pd.DataFrame(metric_rows)
    baseline_iteration = table.loc[
        table["rule"].eq("baseline") & table["period"].eq(ITERATION_PERIOD[0])
    ]
    if baseline_iteration.empty:
        return "baseline", False, "baseline_iteration_missing"
    baseline_pf = float(baseline_iteration.iloc[0]["profit_factor"])
    baseline_drawdown = float(baseline_iteration.iloc[0]["maximum_drawdown"])
    eligible_names: list[str] = []
    for rule, group in table.groupby("rule", sort=False):
        indexed = group.set_index("period")
        if DEVELOPMENT_PERIOD[0] not in indexed.index or ITERATION_PERIOD[0] not in indexed.index:
            continue
        development = indexed.loc[DEVELOPMENT_PERIOD[0]]
        iteration = indexed.loc[ITERATION_PERIOD[0]]
        if (
            int(development["trades"]) >= 100
            and int(iteration["trades"]) >= 100
            and float(development["average_net_return"]) > 0
            and float(iteration["average_net_return"]) > 0
            and float(iteration["profit_factor"]) >= baseline_pf
            and float(iteration["maximum_drawdown"]) >= baseline_drawdown
            and float(development["positive_year_share"]) >= 0.50
            and float(iteration["positive_year_share"]) >= 0.50
        ):
            eligible_names.append(str(rule))
    if not eligible_names:
        return "baseline", False, "no_candidate_passed_predeclared_constraints"
    ranking = table.loc[
        table["rule"].isin(eligible_names) & table["period"].eq(ITERATION_PERIOD[0])
    ].sort_values(
        [
            "win_rate_wilson_lower_95",
            "profit_factor",
            "average_net_return",
            "trades",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return str(ranking.iloc[0]["rule"]), True, "selected_on_development_and_iteration_only"


def _representative_cases(trades: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if trades.empty:
        return {}
    columns = [
        "ts_code",
        "signal_date",
        "entry_date",
        "exit_date",
        "exit_reason",
        "reentry_number",
        "net_return",
        "return_120d",
        "market_return_60d",
        "shock_score",
        "absorption_score",
        "impact_decay",
        "rebound_from_event_low",
    ]
    available = [column for column in columns if column in trades]
    winners = trades.nlargest(min(5, len(trades)), "net_return")[available]
    losers = trades.nsmallest(min(5, len(trades)), "net_return")[available]
    reentries = trades.loc[trades["reentry_number"].gt(0)]
    successful = reentries.loc[reentries["net_return"].gt(0)].nlargest(5, "net_return")
    failed = reentries.loc[reentries["net_return"].le(0)].nsmallest(5, "net_return")
    return {
        "largest_winners": winners.to_dict("records"),
        "largest_losses": losers.to_dict("records"),
        "successful_reentries": successful[available].to_dict("records"),
        "failed_reentries": failed[available].to_dict("records"),
    }


def _metric_markdown(metrics: list[dict[str, Any]]) -> str:
    if not metrics:
        return "无可用指标。"
    table = pd.DataFrame(metrics).copy()
    columns = [
        "rule",
        "period",
        "trades",
        "win_rate",
        "average_net_return",
        "profit_factor",
        "total_return",
        "annualized_return",
        "maximum_drawdown",
        "reentry_trades",
        "successful_reentries",
    ]
    table = table[[column for column in columns if column in table]]
    for column in (
        "win_rate",
        "average_net_return",
        "total_return",
        "annualized_return",
        "maximum_drawdown",
    ):
        if column in table:
            table[column] = table[column].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.2%}"
            )
    if "profit_factor" in table:
        table["profit_factor"] = table["profit_factor"].map(
            lambda value: "—" if pd.isna(value) else f"{float(value):.2f}"
        )
    return table.to_markdown(index=False)


def _write_report(
    output_dir: Path,
    metrics: list[dict[str, Any]],
    frozen: dict[str, Any] | None,
    cases: dict[str, Any] | None,
) -> None:
    lines = [
        "# 带血筹量价长周期回测",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat()}",
        "",
        "## 结论表",
        "",
        _metric_markdown(metrics),
        "",
        "## 固定口径",
        "",
        "- 信号日收盘计算，下一可交易日开盘成交；A 股 T+1。",
        "- 因果连续价格处理除权，不用未来复权因子回写历史。",
        "- 默认 10 个仓位、10% 硬止损、最长持有 120 个交易日。",
        "- 止损后只要出现新的冲击事件和新的吸收确认即可重入。",
        "- 2023 年后的盲测不参与参数选择。",
        "",
    ]
    if frozen is not None:
        signal_config = frozen.get("signal_config", {})
        lines.extend(
            [
                "## 冻结配置",
                "",
                f"- 规则：`{frozen.get('selected_rule')}`",
                f"- 是否完全未看盲测选择：`{frozen.get('selected_without_holdout')}`",
                f"- 选择说明：{frozen.get('selection_reason')}",
                "- 二轮质量过滤："
                f"120 日涨幅 ≤ {signal_config.get('maximum_return_120d')}，"
                f"60 日年化波动 ≤ {signal_config.get('maximum_volatility_60d')}，"
                f"沪深 300 的 60 日收益 ≥ {signal_config.get('minimum_market_return_60d')}。",
                "",
            ]
        )
        lines.extend(
            [
                "## Case 驱动的迭代",
                "",
                "- 基准规则的亏损样本并非吸收不足，反而常有更高的波动、反弹和吸收分；这更像泡沫破裂后的短暂反抽。",
                "- 因此第二轮没有继续抬高吸收阈值，而是加入前趋势、个股波动和市场极端状态过滤。",
                "- 上述逻辑只用入场时已经可见的数据，并且先在 2014—2022 年选择和冻结，再运行 2023 年后的盲测。",
                "- 盲测打开后不再据此调整冻结参数；任何后续改动都必须标记为新假设并寻找新的样本验证。",
                "",
            ]
        )
    if cases:
        lines.extend(["## 代表案例", ""])
        for title, rows in cases.items():
            lines.append(f"### {title}")
            lines.append("")
            if rows:
                lines.append(pd.DataFrame(rows).to_markdown(index=False))
            else:
                lines.append("无。")
            lines.append("")
    lines.extend(
        [
            "## 局限",
            "",
            "量价只能识别疑似不计价格卖压，不能证明卖方身份；日线成交量不能区分主动买卖。回测未使用公告或财务事实排除知情卖出，因此结果只能作为研究池筛选证据。",
            "",
            "本报告仅供研究与教育用途，不构成个性化投资建议、收益承诺或交易指令。",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_development(
    features: pd.DataFrame,
    benchmark: pd.DataFrame,
    output_dir: Path,
) -> None:
    backtest_config = BloodChipBacktestConfig()
    metric_rows: list[dict[str, Any]] = []
    baseline_trades: list[pd.DataFrame] = []
    baseline_equity: list[pd.DataFrame] = []
    candidate_configs = _candidate_signal_configs()
    for rule_name, signal_config in candidate_configs.items():
        print(f"generating signals: {rule_name}", flush=True)
        signals = generate_blood_chip_signals(features, signal_config)
        print(f"{rule_name}: {len(signals):,} signals", flush=True)
        for period in (DEVELOPMENT_PERIOD, ITERATION_PERIOD):
            result, metrics = _run_period(
                features,
                signals,
                benchmark,
                backtest_config,
                period,
            )
            metrics["rule"] = rule_name
            metric_rows.append(metrics)
            if rule_name == "baseline":
                trades = result.trades.copy()
                trades["period"] = period[0]
                baseline_trades.append(trades)
                equity = result.equity_curve.copy()
                equity["period"] = period[0]
                baseline_equity.append(equity)
        del signals
        gc.collect()
    selected_rule, passed, reason = _select_without_holdout(metric_rows)
    frozen = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_sha": _git_sha(),
        "selected_rule": selected_rule,
        "selected_without_holdout": True,
        "selection_constraints_passed": passed,
        "selection_reason": reason,
        "signal_config": candidate_configs[selected_rule].to_dict(),
        "backtest_config": backtest_config.to_dict(),
        "development_period": DEVELOPMENT_PERIOD,
        "iteration_period": ITERATION_PERIOD,
        "holdout_period": HOLDOUT_PERIOD,
    }
    baseline_trade_frame = (
        pd.concat(baseline_trades, ignore_index=True, sort=False)
        if baseline_trades
        else pd.DataFrame()
    )
    baseline_equity_frame = (
        pd.concat(baseline_equity, ignore_index=True, sort=False)
        if baseline_equity
        else pd.DataFrame()
    )
    baseline_trade_frame.to_parquet(
        output_dir / "baseline_trades.parquet", index=False, compression="zstd"
    )
    baseline_equity_frame.to_parquet(
        output_dir / "baseline_equity.parquet", index=False, compression="zstd"
    )
    iteration_trades = baseline_trade_frame.loc[
        baseline_trade_frame.get("period", pd.Series(dtype=str)).eq(ITERATION_PERIOD[0])
    ].copy()
    analyze_blood_chip_cases(iteration_trades).to_csv(
        output_dir / "baseline_cases.csv", index=False
    )
    baseline_metrics = [row for row in metric_rows if row["rule"] == "baseline"]
    _write_json(output_dir / "baseline_metrics.json", {"metrics": baseline_metrics})
    _write_json(output_dir / "candidate_metrics.json", {"metrics": metric_rows})
    _write_json(output_dir / "frozen_config.json", frozen)
    cases = _representative_cases(iteration_trades)
    _write_json(output_dir / "baseline_representative_cases.json", cases)
    _write_report(output_dir, metric_rows, frozen, cases)
    print(f"frozen rule: {selected_rule} ({reason})", flush=True)


def run_holdout(
    features: pd.DataFrame,
    benchmark: pd.DataFrame,
    output_dir: Path,
) -> None:
    frozen_path = output_dir / "frozen_config.json"
    if not frozen_path.exists():
        raise FileNotFoundError(
            f"missing {frozen_path}; run --development-only before frozen holdout"
        )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if not frozen.get("selected_without_holdout"):
        raise RuntimeError("frozen configuration is not marked as selected without holdout")
    signal_config = BloodChipSignalConfig(**frozen["signal_config"])
    backtest_config = BloodChipBacktestConfig(**frozen["backtest_config"])
    print(f"running frozen holdout: {frozen['selected_rule']}", flush=True)
    signals = generate_blood_chip_signals(features, signal_config)
    result, metrics = _run_period(
        features,
        signals,
        benchmark,
        backtest_config,
        HOLDOUT_PERIOD,
    )
    metrics["rule"] = frozen["selected_rule"]
    result.trades.to_parquet(
        output_dir / "holdout_trades.parquet", index=False, compression="zstd"
    )
    result.equity_curve.to_parquet(
        output_dir / "holdout_equity.parquet", index=False, compression="zstd"
    )
    result.rejected_entries.to_parquet(
        output_dir / "holdout_rejected_entries.parquet", index=False, compression="zstd"
    )
    analyze_blood_chip_cases(result.trades).to_csv(
        output_dir / "holdout_cases.csv", index=False
    )
    if not result.trades.empty:
        yearly = result.trades.copy()
        yearly["entry_year"] = pd.to_datetime(yearly["entry_date"]).dt.year
        yearly_summary = yearly.groupby("entry_year", observed=True).agg(
            trades=("net_return", "size"),
            win_rate=("net_return", lambda values: float((values > 0).mean())),
            average_net_return=("net_return", "mean"),
            median_net_return=("net_return", "median"),
            stop_rate=("exit_reason", lambda values: float((values == "stop_loss").mean())),
        )
        yearly_summary.reset_index().to_csv(
            output_dir / "holdout_yearly_trades.csv", index=False
        )
    cases = _representative_cases(result.trades)
    _write_json(output_dir / "holdout_metrics.json", {"metrics": metrics})
    _write_json(output_dir / "holdout_representative_cases.json", cases)
    existing_metrics: list[dict[str, Any]] = []
    candidate_path = output_dir / "candidate_metrics.json"
    if candidate_path.exists():
        existing_metrics = json.loads(candidate_path.read_text(encoding="utf-8")).get(
            "metrics", []
        )
    _write_report(output_dir, [*existing_metrics, metrics], frozen, cases)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features, benchmark = _load_or_build_features(args)
    if args.frozen_holdout:
        run_holdout(features, benchmark, output_dir)
    else:
        run_development(features, benchmark, output_dir)


if __name__ == "__main__":
    main()
