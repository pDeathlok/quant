#!/usr/bin/env python3
"""Backtest point-in-time survival and regime overlays on a frozen price base."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_blood_chip_deep_base import (
    PERIODS,
    SELECTION_PERIODS,
    _json_value,
    _settlement_end,
    _yearly_metrics,
)
from quant.features.long_quality_factors import build_annual_quality_events
from quant.research.blood_chip import load_benchmark
from quant.research.blood_chip_deep_base import (
    DeepBaseExecutionConfig,
    DeepBaseSignalConfig,
    generate_deep_base_signals,
    run_deep_base_backtest,
    summarize_deep_base_result,
)
from quant.research.blood_chip_multidimensional import (
    INDUSTRY_DIAGNOSTIC_POLICIES,
    POLICY_GATES,
    SELECTION_ELIGIBLE_POLICIES,
    MultidimensionalGateConfig,
    add_current_industry_repair_features,
    add_market_repair_features,
    apply_multidimensional_gates,
    merge_capital_pressure_asof,
    merge_daily_basic_on_signal_date,
    merge_financial_survival_asof,
    signals_for_policy,
)


POLICIES = tuple(POLICY_GATES)
SOURCE_COLUMNS = {
    "fina_indicator": [
        "ts_code",
        "end_date",
        "ann_date",
        "roe",
        "roa",
        "netprofit_margin",
        "grossprofit_margin",
        "debt_to_assets",
        "current_ratio",
        "quick_ratio",
        "ar_turn",
        "inv_turn",
        "assets_turn",
        "or_yoy",
        "basic_eps_yoy",
    ],
    "income": [
        "ts_code",
        "ann_date",
        "end_date",
        "report_type",
        "revenue",
        "n_income_attr_p",
    ],
    "cashflow": [
        "ts_code",
        "ann_date",
        "end_date",
        "report_type",
        "n_cashflow_act",
        "c_pay_acq_const_fiolta",
    ],
    "balancesheet": [
        "ts_code",
        "ann_date",
        "end_date",
        "report_type",
        "total_assets",
        "total_liab",
        "total_hldr_eqy_exc_min_int",
        "money_cap",
        "inventories",
        "intan_assets",
        "goodwill",
    ],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest multidimensional survival overlays for deep-base blood chips."
    )
    parser.add_argument(
        "--feature-cache",
        default="data/research/blood_chip_deep_base/features.parquet",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--daily-basic-root", default="data/raw/daily_basic")
    parser.add_argument("--benchmark", default="data/raw/index_000300.SH.parquet")
    parser.add_argument(
        "--output-dir",
        default="reports/research/blood_chip_multidimensional",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_available_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    selected = [column for column in columns if column in available]
    return pd.read_parquet(path, columns=selected)


def _load_signal_daily_basic(
    signals: pd.DataFrame,
    root: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    signal_dates = pd.to_datetime(signals["signal_date"], errors="coerce")
    symbols_by_date = {
        pd.Timestamp(date): set(group["ts_code"].astype(str))
        for date, group in signals.assign(_date=signal_dates).groupby("_date", sort=True)
    }
    columns = [
        "ts_code",
        "trade_date",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ]
    frames: list[pd.DataFrame] = []
    missing_files = 0
    for index, (date, symbols) in enumerate(symbols_by_date.items(), start=1):
        path = root / f"{date:%Y%m%d}.parquet"
        if not path.exists():
            missing_files += 1
            continue
        frame = _read_available_columns(path, columns)
        frame = frame.loc[frame["ts_code"].astype(str).isin(symbols)].copy()
        if not frame.empty:
            frames.append(frame)
        if index % 250 == 0:
            print(f"loaded daily_basic signal dates: {index}/{len(symbols_by_date)}", flush=True)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined, {
        "requested_signal_dates": len(symbols_by_date),
        "missing_signal_date_files": missing_files,
        "matched_rows": int(len(combined)),
    }


def _build_enriched_signals(
    features: pd.DataFrame,
    benchmark: pd.DataFrame,
    raw_dir: Path,
    daily_basic_root: Path,
    config: MultidimensionalGateConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal_config = DeepBaseSignalConfig(
        minimum_drawdown_from_peak=0.65,
        minimum_peak_age_sessions=750,
        minimum_deep_drawdown_sessions=120,
    )
    signals = generate_deep_base_signals(features, signal_config)
    # The last eligible entry cohort ends in 2024; later price rows are retained
    # only for settlement and industry cross-sectional state.
    signals = signals.loc[
        pd.to_datetime(signals["entry_date"], errors="coerce").le(pd.Timestamp("2024-07-30"))
    ].copy()
    print(f"fixed mature dd65 signals: {len(signals):,}", flush=True)

    financial_sources = {
        name: _read_available_columns(raw_dir / f"{name}.parquet", columns)
        for name, columns in SOURCE_COLUMNS.items()
    }
    annual_events = build_annual_quality_events(
        financial_sources["fina_indicator"],
        financial_sources["income"],
        financial_sources["cashflow"],
        financial_sources["balancesheet"],
    )
    print(f"annual point-in-time events: {len(annual_events):,}", flush=True)
    enriched = merge_financial_survival_asof(signals, annual_events)

    daily_basic, basic_manifest = _load_signal_daily_basic(signals, daily_basic_root)
    enriched = merge_daily_basic_on_signal_date(enriched, daily_basic)
    print(f"daily_basic matched signals: {int(enriched['daily_basic_coverage'].sum()):,}", flush=True)

    pledge = pd.read_parquet(
        raw_dir / "pledge_stat.parquet",
        columns=["ts_code", "end_date", "pledge_ratio"],
    )
    holder = pd.read_parquet(
        raw_dir / "holder_trade.parquet",
        columns=["ts_code", "ann_date", "in_de", "change_ratio"],
    )
    enriched = merge_capital_pressure_asof(enriched, pledge, holder)
    enriched = add_market_repair_features(enriched, benchmark)
    stock_basic = pd.read_parquet(raw_dir / "stock_basic.parquet", columns=["ts_code", "industry"])
    enriched = add_current_industry_repair_features(
        enriched,
        features[["ts_code", "date", "return_20d"]],
        stock_basic,
    )
    enriched = apply_multidimensional_gates(enriched, config)
    return enriched, {
        "signal_config": signal_config.to_dict(),
        "gate_config": config.to_dict(),
        "annual_events": int(len(annual_events)),
        "daily_basic": basic_manifest,
        "industry_mapping": "current_stock_basic_mapping_diagnostic_only",
    }


def _coverage_rows(enriched: pd.DataFrame) -> pd.DataFrame:
    entry_dates = pd.to_datetime(enriched["entry_date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    flags = [
        "financial_coverage",
        "daily_basic_coverage",
        "pledge_observed",
        "holder_activity_observed",
        "market_coverage",
        "survival_gate",
        "value_scale_gate",
        "capital_pressure_gate",
        "market_repair_gate",
        "current_industry_repair_gate",
        "auditable_combined_gate",
    ]
    for period, (start, end) in PERIODS.items():
        group = enriched.loc[entry_dates.between(start, end, inclusive="both")]
        total = len(group)
        for flag in flags:
            passed = int(group[flag].fillna(False).astype(bool).sum()) if flag in group else 0
            rows.append(
                {
                    "period": period,
                    "dimension": flag,
                    "base_signals": total,
                    "observed_or_passed": passed,
                    "rate": passed / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _signal_counts(enriched: pd.DataFrame, config: MultidimensionalGateConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        selected = signals_for_policy(enriched, policy, config)
        dates = pd.to_datetime(selected["entry_date"], errors="coerce")
        for period, (start, end) in PERIODS.items():
            count = int(dates.between(start, end, inclusive="both").sum())
            base_dates = pd.to_datetime(enriched["entry_date"], errors="coerce")
            base_count = int(base_dates.between(start, end, inclusive="both").sum())
            rows.append(
                {
                    "policy": policy,
                    "period": period,
                    "signals": count,
                    "base_signals": base_count,
                    "retention_rate": count / base_count if base_count else np.nan,
                    "selection_eligible_policy": policy in SELECTION_ELIGIBLE_POLICIES,
                    "current_industry_mapping_bias": policy in INDUSTRY_DIAGNOSTIC_POLICIES,
                }
            )
    return pd.DataFrame(rows)


def _decision(metrics: pd.DataFrame) -> dict[str, Any]:
    eligible = metrics.loc[metrics["policy"].isin(SELECTION_ELIGIBLE_POLICIES)].copy()
    checks: dict[str, dict[str, bool]] = {}
    scores: list[tuple[tuple[float, float, float, int], str]] = []
    for policy, group in eligible.groupby("policy", observed=True, sort=False):
        selected = group.set_index("period").reindex(SELECTION_PERIODS)
        complete = not selected.isna().all(axis=1).any()
        policy_checks = {
            "periods_complete": bool(complete),
            "minimum_40_trades_each": bool(complete and selected["trades"].ge(40).all()),
            "win_rate_at_least_55pct_each": bool(complete and selected["win_rate"].ge(0.55).all()),
            "positive_median_each": bool(complete and selected["median_net_return"].gt(0.0).all()),
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
            geometric = float(np.prod(1.0 + annualized.to_numpy(dtype=float)) ** 0.5 - 1.0)
            scores.append(
                (
                    (
                        float(selected["win_rate"].min()),
                        float(selected["capital_profit_factor"].min()),
                        geometric,
                        int(selected["trades"].sum()),
                    ),
                    str(policy),
                )
            )
    qualified = [policy for _, policy in scores if checks[policy]["all_passed"]]
    selected_scores = [(score, policy) for score, policy in scores if policy in qualified]
    return {
        "selected_on_development_and_validation_only": True,
        "selection_periods": list(SELECTION_PERIODS),
        "seen_diagnostic_excluded_from_selection": True,
        "price_only_is_reference_not_upgrade_candidate": True,
        "industry_policies_excluded_for_current_mapping_bias": sorted(
            INDUSTRY_DIAGNOSTIC_POLICIES
        ),
        "selection_eligible_policies": sorted(SELECTION_ELIGIBLE_POLICIES),
        "qualification_thresholds": {
            "minimum_trades_each_period": 40,
            "minimum_win_rate_each_period": 0.55,
            "minimum_median_net_return_each_period": 0.0,
            "minimum_capital_profit_factor_each_period": 1.50,
            "minimum_maximum_drawdown_each_period": -0.35,
        },
        "checks": checks,
        "qualified_policies": qualified,
        "selected_policy": max(selected_scores)[1] if selected_scores else None,
        "best_research_policy": max(scores)[1] if scores else None,
        "deployment_decision": (
            "enter_forward_shadow"
            if selected_scores
            else "research_only_keep_existing_online_strategy"
        ),
    }


def _add_failure_diagnostics(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    out["loss_rate"] = 1.0 - pd.to_numeric(out["win_rate"], errors="coerce")
    baseline = out.loc[out["policy"].eq("price_only"), ["period", "loss_rate"]].rename(
        columns={"loss_rate": "price_only_loss_rate"}
    )
    out = out.merge(baseline, on="period", how="left", validate="many_to_one")
    out["failure_rate_reduction_vs_price_only"] = (
        out["price_only_loss_rate"] - out["loss_rate"]
    )
    return out


def _percent(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2%}"


def _number(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2f}"


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    counts: pd.DataFrame,
    yearly: pd.DataFrame,
    decision: dict[str, Any],
    manifest: dict[str, Any],
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
        "failure_rate_reduction_vs_price_only",
    ):
        display[column] = display[column].map(_percent)
    display["capital_profit_factor"] = display["capital_profit_factor"].map(_number)
    columns = [
        "period",
        "policy",
        "signals",
        "trades",
        "win_rate",
        "median_net_return",
        "capital_profit_factor",
        "total_return",
        "annualized_return",
        "maximum_drawdown",
        "failure_rate_reduction_vs_price_only",
    ]
    coverage_pivot = coverage.pivot(index="dimension", columns="period", values="rate").reset_index()
    for column in coverage_pivot.columns[1:]:
        coverage_pivot[column] = coverage_pivot[column].map(_percent)
    count_display = counts.copy()
    count_display["retention_rate"] = count_display["retention_rate"].map(_percent)
    stability = yearly.loc[
        yearly["policy"].isin(["price_only", "survival_value", "survival_market"]),
        [
            "policy",
            "period",
            "entry_year",
            "trades",
            "win_rate",
            "median_net_return",
            "capital_profit_factor",
        ],
    ].copy()
    stability["win_rate"] = stability["win_rate"].map(_percent)
    stability["median_net_return"] = stability["median_net_return"].map(_percent)
    stability["capital_profit_factor"] = stability["capital_profit_factor"].map(_number)
    eligible_validation = metrics.loc[
        metrics["period"].eq("validation_2017_2020")
        & metrics["policy"].isin(SELECTION_ELIGIBLE_POLICIES)
    ].sort_values(["win_rate", "median_net_return"], ascending=False)
    best_validation = eligible_validation.iloc[0] if not eligible_validation.empty else None
    selected = decision.get("selected_policy")
    if selected:
        conclusion = f"`{selected}` 通过冻结门槛，只建议进入前向影子观察。"
    else:
        conclusion = "没有可审计多维候选同时通过冻结门槛；保留研究，不替换线上策略。"
    path.write_text(
        f"""# 带血筹多维生存过滤与周期修复回测

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

{conclusion}

- 固定价格层：前高回撤≥65%、峰值距今≥750日、深跌≥120日、60日筑底、二次探底收复后加仓、250日持有、只保留硬止损。
- 开发/验证综合排序第一：`{decision.get('best_research_policy')}`；综合排序不能替代硬门槛。
- 2021年以后只是已见诊断，不是盲测。
- 当前行业映射策略永不参与升级选择。
- 验证期最接近门槛的是 `{best_validation['policy'] if best_validation is not None else '—'}`：胜率 {_percent(best_validation['win_rate']) if best_validation is not None else '—'}、中位收益 {_percent(best_validation['median_net_return']) if best_validation is not None else '—'}，仍未达到55%与正中位数。
- 10个计划槽位和250日持有使开发期实际完成交易最多只有24笔，低于冻结的40笔统计门槛；这不是放宽门槛的理由。

## 维度覆盖与通过率

{coverage_pivot.to_markdown(index=False)}

`pledge_observed` 低不代表无质押；只是本地质押表在该信号日前是否有记录。无记录允许通过风险门，但不会得到排序加分。财务缺失不能通过生存门。

## 信号保留

{count_display[['period', 'policy', 'signals', 'base_signals', 'retention_rate', 'selection_eligible_policy', 'current_industry_mapping_bias']].to_markdown(index=False)}

## 组合回测

{display[columns].to_markdown(index=False)}

## 年度稳定性

{stability.to_markdown(index=False)}

过滤改善主要集中在2019—2021；2018与2023仍出现成片失败，说明“静态入场时健康 + 宽松市场门”不能覆盖持有期间的系统性和行业性恶化。

## 决策门槛

每个候选必须在2013—2016与2017—2020分别达到：完成交易≥40、胜率≥55%、中位净收益>0、资金PF≥1.50、最大回撤≥-35%。当前决策：`{decision['deployment_decision']}`。

## 点时与偏差边界

- 年度财务只使用各报告期本地首次披露值，最晚源公告日不晚于信号日；不把后续更正回填到早期信号。
- `daily_basic` 使用信号日收盘截面；沪深300只使用信号日及以前价格。
- 质押统计同日记录从下一交易日开始使用；股东增减持只累计信号日前180天已公告事件。
- 行业由当前 `stock_basic.industry` 回贴历史，存在幸存、迁移和重分类偏差，因此只做诊断。来源标记：`{manifest['industry_mapping']}`。
- 本地财务、质押和股东数据的历史抓取完整性仍可能随时期变化；覆盖率下降时，样本变化不能全部解释为经济规律。
- 本研究仅供研究与教育用途，不构成投资建议、收益承诺或交易指令。
""",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading price feature cache: {args.feature_cache}", flush=True)
    features = pd.read_parquet(args.feature_cache)
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    benchmark = load_benchmark(Path(args.benchmark), "20100104", "20260821")
    benchmark["trade_date"] = pd.to_datetime(
        benchmark["trade_date"].astype(str), errors="coerce"
    )
    calendar = pd.DatetimeIndex(sorted(benchmark["trade_date"].dropna().unique()))
    config = MultidimensionalGateConfig()
    enriched, manifest = _build_enriched_signals(
        features,
        benchmark,
        Path(args.raw_dir),
        Path(args.daily_basic_root),
        config,
    )
    coverage = _coverage_rows(enriched)
    counts = _signal_counts(enriched, config)
    enriched.to_parquet(
        output_dir / "signal_diagnostics.parquet", index=False, compression="zstd"
    )
    coverage.to_csv(output_dir / "coverage.csv", index=False)

    execution = DeepBaseExecutionConfig(
        stage_policy="retest_reclaim",
        structural_exit_enabled=False,
        maximum_holding_sessions=250,
    )
    diagnostic_columns = [
        column
        for column in enriched.columns
        if column
        not in {
            "entry_date",
            "entry_open",
            "signal_close",
            "signal_score",
        }
    ]
    diagnostics = enriched[diagnostic_columns].drop_duplicates(
        ["ts_code", "base_event_id", "signal_date"], keep="last"
    )
    metric_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    count_lookup = counts.set_index(["policy", "period"])["signals"]
    feature_dates = pd.to_datetime(features["date"], errors="coerce")
    feature_symbols = features["ts_code"].astype(str)
    for policy in POLICIES:
        policy_signals = signals_for_policy(enriched, policy, config)
        for period, (entry_start, entry_end) in PERIODS.items():
            settlement_end = _settlement_end(
                calendar,
                entry_end,
                execution.maximum_holding_sessions,
                execution.maximum_missing_market_sessions,
            )
            policy_entry_dates = pd.to_datetime(
                policy_signals["entry_date"], errors="coerce"
            )
            period_symbols = set(
                policy_signals.loc[
                    policy_entry_dates.between(entry_start, entry_end, inclusive="both"),
                    "ts_code",
                ].astype(str)
            )
            panel = features.loc[
                feature_dates.between(entry_start, settlement_end, inclusive="both")
                & feature_symbols.isin(period_symbols)
            ].copy()
            print(f"running {policy} {period}", flush=True)
            result = run_deep_base_backtest(
                panel,
                policy_signals,
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
                    "signals": int(count_lookup.get((policy, period), 0)),
                    "selection_eligible_policy": policy in SELECTION_ELIGIBLE_POLICIES,
                    "current_industry_mapping_bias": policy in INDUSTRY_DIAGNOSTIC_POLICIES,
                    **summary,
                }
            )
            if not result.trades.empty:
                trades = result.trades.copy()
                trades["policy"] = policy
                trades["period"] = period
                trades = trades.merge(
                    diagnostics,
                    on=["ts_code", "base_event_id", "signal_date"],
                    how="left",
                    validate="many_to_one",
                    suffixes=("", "_diagnostic"),
                )
                trade_frames.append(trades)

    metrics = _add_failure_diagnostics(pd.DataFrame(metric_rows))
    trades = pd.concat(trade_frames, ignore_index=True, sort=False) if trade_frames else pd.DataFrame()
    yearly = _yearly_metrics(trades)
    decision = _decision(metrics)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    trades.to_parquet(output_dir / "trades.parquet", index=False, compression="zstd")
    yearly.to_csv(output_dir / "yearly_metrics.csv", index=False)
    _write_json(output_dir / "decision.json", decision)
    _write_json(
        output_dir / "metrics.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "periods": PERIODS,
            "manifest": manifest,
            "metrics": metrics.to_dict(orient="records"),
        },
    )
    _write_report(
        output_dir / "report.md",
        metrics,
        coverage,
        counts,
        yearly,
        decision,
        manifest,
    )
    print(f"decision: {decision['deployment_decision']}", flush=True)
    print(f"report: {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
