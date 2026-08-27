#!/usr/bin/env python3
"""Validate a rare, strict monthly blood-chip structure by panic cohort."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.features.long_quality_factors import build_annual_quality_events
from quant.research.blood_chip_deep_base import build_deep_base_features
from quant.research.blood_chip_multidimensional import merge_financial_survival_asof
from quant.research.monthly_low_zone_profit_lock import (
    ProfitLockPortfolioConfig,
    simulate_profit_lock_portfolio,
)
from quant.research.monthly_low_zone_strict import (
    HISTORICAL_STRICT_PERIODS,
    PRIMARY_STRICT_STRUCTURE,
    STRICT_PERIODS,
    STRICT_STRUCTURE_SPECS,
    StrictLowZoneConfig,
    add_strict_gate_columns,
    decide_strict_structure,
    materialize_strict_structures,
    summarize_strict_events,
)


FINANCIAL_SOURCE_COLUMNS = {
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("reports/research/monthly_low_zone_profit_lock/events.parquet"),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("data/research/blood_chip_deep_base/features.parquet"),
    )
    parser.add_argument(
        "--supplemental-root",
        type=Path,
        default=Path("data/research/low9_kdj_rebound/supplemental_daily"),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(
            "data/research/low9_kdj_rebound/index_000001.SH_20100101_20260731.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/research/monthly_low_zone_strict"),
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
        if np.isposinf(value):
            return "infinity"
        if np.isneginf(value):
            return "negative_infinity"
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


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


def _load_annual_events(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources = {
        name: _read_available_columns(raw_dir / f"{name}.parquet", columns)
        for name, columns in FINANCIAL_SOURCE_COLUMNS.items()
    }
    events = build_annual_quality_events(
        sources["fina_indicator"],
        sources["income"],
        sources["cashflow"],
        sources["balancesheet"],
    )
    return events, {
        "source_rows": {name: int(len(frame)) for name, frame in sources.items()},
        "annual_quality_events": int(len(events)),
        "annual_quality_symbols": int(events["ts_code"].nunique()),
    }


def _attach_point_in_time_financials(
    events: pd.DataFrame,
    annual_events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal_columns = [
        "signal_id",
        "anchor_id",
        "ts_code",
        "rule",
        "signal_date",
    ]
    signals = events[signal_columns].drop_duplicates("signal_id").copy()
    enriched = merge_financial_survival_asof(signals, annual_events)
    financial_columns = [
        column
        for column in enriched.columns
        if column not in set(signal_columns) - {"signal_id"}
    ]
    out = events.merge(
        enriched[financial_columns],
        on="signal_id",
        how="left",
        validate="many_to_one",
    )
    coverage = enriched.get("financial_coverage", pd.Series(False, index=enriched.index))
    causal = pd.to_datetime(
        enriched.get("annual_quality_available_at"), errors="coerce"
    ).le(pd.to_datetime(enriched["signal_date"], errors="coerce"))
    return out, {
        "unique_signals": int(len(signals)),
        "financial_coverage_signals": int(coverage.fillna(False).sum()),
        "causal_financial_signals": int(causal.fillna(False).sum()),
    }


def _load_benchmark(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    source = frame["trade_date"] if "trade_date" in frame else frame["date"]
    compact = pd.to_datetime(
        source.astype("string").str.replace(r"\.0$", "", regex=True).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    fallback = pd.to_datetime(source, errors="coerce")
    frame["date"] = compact.fillna(fallback).dt.normalize()
    return frame.loc[frame["date"].le(cutoff)].sort_values("date").copy()


def _load_portfolio_prices(
    feature_cache: Path,
    supplemental_root: Path,
    symbols: set[str],
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not symbols:
        return pd.DataFrame(columns=["ts_code", "date", "adjusted_close"]), {
            "symbols": 0,
            "rows": 0,
        }
    columns = ["ts_code", "date", "adjusted_close"]
    print(f"loading primary price paths for {len(symbols)} strict symbols", flush=True)
    primary = pd.read_parquet(
        feature_cache,
        columns=columns,
        filters=[("ts_code", "in", sorted(symbols))],
    )
    primary["date"] = pd.to_datetime(primary["date"], errors="coerce").dt.normalize()
    primary = primary.loc[primary["date"].le(cutoff)].copy()
    raw_columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "vol",
        "amount",
    ]
    supplemental_parts: list[pd.DataFrame] = []
    for symbol in sorted(symbols):
        path = supplemental_root / f"{symbol}.parquet"
        if path.exists():
            supplemental_parts.append(pd.read_parquet(path, columns=raw_columns))
    if supplemental_parts:
        raw = pd.concat(supplemental_parts, ignore_index=True, sort=False)
        supplemental = build_deep_base_features(raw)[columns]
        supplemental = supplemental.loc[supplemental["date"].le(cutoff)]
    else:
        supplemental = pd.DataFrame(columns=columns)
    combined = pd.concat([primary, supplemental], ignore_index=True, sort=False)
    combined = (
        combined.sort_values(["ts_code", "date"])
        .drop_duplicates(["ts_code", "date"], keep="last")
        .reset_index(drop=True)
    )
    return combined, {
        "requested_symbols": int(len(symbols)),
        "primary_rows": int(len(primary)),
        "supplemental_rows": int(len(supplemental)),
        "combined_rows": int(len(combined)),
        "combined_symbols": int(combined["ts_code"].nunique()),
    }


def _portfolio_metrics(curve: pd.DataFrame) -> dict[str, float]:
    if curve.empty:
        return {
            "total_return": np.nan,
            "cagr": np.nan,
            "maximum_drawdown": np.nan,
            "worst_rolling_24m_return": np.nan,
            "mean_invested_fraction": np.nan,
        }
    nav = pd.to_numeric(curve["nav"], errors="coerce").dropna()
    dates = pd.to_datetime(curve.loc[nav.index, "date"], errors="coerce")
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    years = max(float((dates.iloc[-1] - dates.iloc[0]).days / 365.25), 1.0 / 365.25)
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)
    drawdown = nav / nav.cummax() - 1.0
    rolling = nav / nav.shift(504) - 1.0
    worst_rolling = float(rolling.dropna().min()) if rolling.notna().any() else total_return
    invested = pd.to_numeric(curve.get("invested_fraction"), errors="coerce")
    return {
        "total_return": total_return,
        "cagr": cagr,
        "maximum_drawdown": float(drawdown.min()),
        "worst_rolling_24m_return": worst_rolling,
        "mean_invested_fraction": float(invested.mean()),
    }


def _run_primary_portfolio(
    daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    primary_events: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    portfolio_events = primary_events.copy()
    portfolio_events["source_rule"] = portfolio_events["rule"]
    portfolio_events["rule"] = "anchor_direct"
    curve_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    config = ProfitLockPortfolioConfig(
        maximum_anchors=20,
        target_anchor_fraction=0.10,
        probe_budget_fraction=0.25,
    )
    for period, (start, end) in STRICT_PERIODS.items():
        effective_end = min(end, cutoff)
        print(f"strict portfolio: {period}", flush=True)
        curve, trades, audit = simulate_profit_lock_portfolio(
            daily,
            portfolio_events,
            calendar,
            start_date=start,
            end_date=effective_end,
            target_return=0.15,
            add_rule=None,
            config=config,
        )
        curve["period"] = period
        trades["period"] = period
        curve_parts.append(curve)
        trade_parts.append(trades)
        metrics.append({"period": period, **_portfolio_metrics(curve), **audit})
        audits.append({"period": period, **audit})
    return (
        pd.concat(curve_parts, ignore_index=True, sort=False),
        pd.concat(trade_parts, ignore_index=True, sort=False),
        pd.DataFrame(metrics),
        audits,
    )


def _case_catalog(
    gated: pd.DataFrame,
    structured: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = structured.loc[
        structured["structure"].eq(PRIMARY_STRICT_STRUCTURE)
        & structured["horizon"].eq(504)
        & np.isclose(structured["target_return"], 0.15)
    ].copy()
    primary["anchor_month"] = pd.PeriodIndex(
        primary["month_period"], freq="M"
    ).to_timestamp(how="end").normalize()
    complete = primary.loc[
        primary["entry_status"].eq("accepted")
        & primary["outcome_completed"].fillna(False)
        & primary["baseline_outcome_completed"].fillna(False)
    ].copy()
    primary_cohorts = cohorts.loc[
        cohorts["structure"].eq(PRIMARY_STRICT_STRUCTURE)
        & cohorts["horizon"].eq(504)
        & np.isclose(cohorts["target_return"], 0.15)
    ].copy()
    cohort_cases = pd.concat(
        [
            primary_cohorts.loc[primary_cohorts["cohort_return"].le(0.0)].assign(
                category="all_nonpositive_cohorts"
            ),
            primary_cohorts.nlargest(3, "cohort_return").assign(
                category="top_three_cohorts"
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    event_parts: list[pd.DataFrame] = []
    negative_months = set(
        primary_cohorts.loc[
            primary_cohorts["cohort_return"].le(0.0), "anchor_month"
        ]
    )
    if negative_months:
        event_parts.append(
            complete.loc[complete["anchor_month"].isin(negative_months)]
            .sort_values(["anchor_month", "net_return"])
            .groupby("anchor_month", as_index=False, observed=True)
            .head(5)
            .assign(category="worst_events_in_nonpositive_cohort")
        )
    strict_price = gated.loc[
        gated["rule"].eq("no_new_low_20")
        & gated["gate_systemic_deep"]
        & gated["gate_dd70"]
        & gated["horizon"].eq(504)
        & np.isclose(gated["target_return"], 0.15)
        & gated["entry_status"].eq("accepted")
        & gated["outcome_completed"].fillna(False)
        & gated["baseline_outcome_completed"].fillna(False)
    ].copy()
    removed_survival = strict_price.loc[~strict_price["gate_survival80"]].copy()
    if not removed_survival.empty:
        removed_survival["category"] = np.select(
            [
                removed_survival["net_return"].gt(0.0),
                removed_survival["net_return"].le(-0.50),
            ],
            ["survival_removed_winner", "survival_removed_tail_loss"],
            default="survival_removed_other_loss",
        )
        event_parts.append(removed_survival)
    unresolved = primary.loc[
        primary["anchor_month"].ge(pd.Timestamp("2025-01-01"))
        & ~(
            primary["outcome_completed"].fillna(False)
            & primary["baseline_outcome_completed"].fillna(False)
        )
    ].copy()
    if not unresolved.empty:
        unresolved["category"] = "time_out_unresolved"
        event_parts.append(unresolved)
    event_cases = (
        pd.concat(event_parts, ignore_index=True, sort=False)
        if event_parts
        else primary.head(0).assign(category=pd.Series(dtype="string"))
    )
    keep = [
        column
        for column in (
            "category",
            "anchor_id",
            "ts_code",
            "month_period",
            "signal_date",
            "entry_date",
            "exit_date",
            "exit_reason",
            "net_return",
            "baseline_net_return",
            "target_hit",
            "holding_sessions",
            "drawdown_from_prior_peak",
            "breadth_positive_share_20d",
            "breadth_median_return_20d",
            "profit_positive_share_5y",
            "cfo_positive_share_5y",
            "financial_age_days",
        )
        if column in event_cases
    ]
    return event_cases[keep], cohort_cases


def _format_percent(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = out[column].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.2%}"
            )
    return out


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    cohorts: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    case_events: pd.DataFrame,
    cohort_cases: pd.DataFrame,
    decision: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    primary_metrics = metrics.loc[
        metrics["structure"].eq(PRIMARY_STRICT_STRUCTURE)
        & metrics["horizon"].eq(504)
        & np.isclose(metrics["target_return"], 0.15)
    ].copy()
    sensitivity = metrics.loc[
        metrics["period"].eq("historical_2013_2024")
        & metrics["horizon"].eq(504)
        & np.isclose(metrics["target_return"], 0.15)
    ].copy()
    primary_display = _format_percent(
        primary_metrics,
        [
            "target_return",
            "event_win_rate",
            "event_win_wilson_low",
            "median_event_return",
            "mean_event_return",
            "positive_cohort_share",
            "positive_cohort_wilson_low",
            "cohort_equal_mean_return",
            "cohort_bootstrap_ci95_low",
            "cohort_bootstrap_ci95_high",
            "worst_cohort_return",
            "leave_one_cohort_out_min_mean",
        ],
    )
    sensitivity_display = _format_percent(
        sensitivity,
        [
            "event_win_rate",
            "median_event_return",
            "positive_cohort_share",
            "cohort_equal_mean_return",
            "cohort_bootstrap_ci95_low",
            "worst_cohort_return",
            "leave_one_cohort_out_min_mean",
        ],
    )
    portfolio_display = _format_percent(
        portfolio_metrics,
        [
            "total_return",
            "cagr",
            "maximum_drawdown",
            "worst_rolling_24m_return",
            "mean_invested_fraction",
        ],
    )
    cohort_display = _format_percent(
        cohort_cases,
        ["cohort_return", "worst_event_return"],
    )
    case_summary = (
        case_events.groupby("category", as_index=False).size()
        if not case_events.empty
        else pd.DataFrame()
    )
    conclusion = {
        "historically_robust_research_candidate": (
            "主结构通过冻结的历史批次、尾部、邻域与组合检查，但历史已暴露，"
            "仍只能作为观察或极小试验仓候选。"
        ),
        "promising_but_independent_sample_insufficient": (
            "主结构的收益质量有吸引力，但独立恐慌月份不足，不能把股票级高胜率"
            "解释为高确定性。"
        ),
        "strict_structure_failed_robustness": (
            "主结构未通过冻结的严格稳健性检查，不应进入实盘。"
        ),
    }[decision["status"]]
    path.write_text(
        f"""# 月线带血筹严格确定性验证

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

{conclusion}

`deployment_eligible` 固定为 `{str(decision['deployment_eligible']).lower()}`。2013—2024 已在规则提出前被查看；需要至少两个冻结规则后新增、互不相邻且已完成的系统性恐慌锚月，才能重新讨论可部署性。

## 冻结主结构

市场上涨家数占比不高于 20% 且横截面 20 日中位收益不高于 -8%；个股从前高至少跌 70%；点时年报连续盈利和经营现金流为正比例均不低于 80%；随后连续 20 个交易日不创新低才入场。每锚净值 2.5%，最多 20 锚，504 日上限，15% 止盈，往返成本 20bp。

## 主结构分段结果

{primary_display.to_markdown(index=False) if not primary_display.empty else '_主结构没有完成事件_'}

## 已登记消融与阈值邻域

{sensitivity_display.to_markdown(index=False) if not sensitivity_display.empty else '_没有可比较结构_'}

## 日度盯市组合

{portfolio_display.to_markdown(index=False) if not portfolio_display.empty else '_主结构没有可模拟事件_'}

## 最差与最好批次

{cohort_display.to_markdown(index=False) if not cohort_display.empty else '_没有完成批次_'}

## Case 审计

{case_summary.to_markdown(index=False) if not case_summary.empty else '_没有 case_'}

## 冻结判定

```json
{json.dumps(_json_value(decision), ensure_ascii=False, indent=2)}
```

## 数据审计

```json
{json.dumps(_json_value(metadata), ensure_ascii=False, indent=2)}
```

本研究仅供研究与教育用途，不构成投资建议、收益承诺或交易指令。
""",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    cutoff = pd.Timestamp("2026-07-31")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = StrictLowZoneConfig()
    print(f"loading profit-lock events: {args.events}", flush=True)
    events = pd.read_parquet(args.events)
    events = events.loc[
        events["horizon"].eq(504)
        & events["rule"].isin({spec.rule for spec in STRICT_STRUCTURE_SPECS})
    ].copy()
    print("building point-in-time annual quality states", flush=True)
    annual_events, financial_metadata = _load_annual_events(args.raw_dir)
    events, merge_metadata = _attach_point_in_time_financials(events, annual_events)
    gated = add_strict_gate_columns(events, config)
    structured = materialize_strict_structures(gated)
    print("summarizing stock events and independent panic months", flush=True)
    metrics, cohorts = summarize_strict_events(structured, config)
    case_events, cohort_cases = _case_catalog(gated, structured, cohorts)
    primary_events = structured.loc[
        structured["structure"].eq(PRIMARY_STRICT_STRUCTURE)
        & structured["horizon"].eq(504)
        & np.isclose(structured["target_return"], 0.15)
    ].copy()
    if primary_events.empty:
        portfolio_curves = pd.DataFrame()
        portfolio_trades = pd.DataFrame()
        portfolio_metrics = pd.DataFrame()
        portfolio_audits: list[dict[str, Any]] = []
        price_metadata: dict[str, Any] = {"requested_symbols": 0, "combined_rows": 0}
    else:
        benchmark = _load_benchmark(args.benchmark, cutoff)
        calendar = pd.DatetimeIndex(sorted(benchmark["date"].dropna().unique()))
        daily, price_metadata = _load_portfolio_prices(
            args.feature_cache,
            args.supplemental_root,
            set(primary_events["ts_code"].astype(str)),
            cutoff,
        )
        (
            portfolio_curves,
            portfolio_trades,
            portfolio_metrics,
            portfolio_audits,
        ) = _run_primary_portfolio(daily, calendar, primary_events, cutoff)
    decision = decide_strict_structure(
        metrics,
        config,
        portfolio_metrics=portfolio_metrics,
    )
    metadata = {
        "analysis_cutoff": cutoff,
        "exposure_warning": (
            "2013-2024 were exposed before this strict rule; results are research-only"
        ),
        "source_events": str(args.events),
        "source_event_rows": int(len(events)),
        "structured_event_rows": int(len(structured)),
        "primary_event_rows": int(len(primary_events)),
        "config": config.to_dict(),
        "structure_specs": [asdict(spec) for spec in STRICT_STRUCTURE_SPECS],
        "financial": financial_metadata,
        "financial_merge": merge_metadata,
        "portfolio_prices": price_metadata,
        "portfolio_audits": portfolio_audits,
        "historical_periods": HISTORICAL_STRICT_PERIODS,
    }
    gated.to_parquet(
        args.output_dir / "gated_events.parquet", index=False, compression="zstd"
    )
    structured.to_parquet(
        args.output_dir / "structured_events.parquet", index=False, compression="zstd"
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    cohorts.to_csv(args.output_dir / "cohort_metrics.csv", index=False)
    case_events.to_parquet(
        args.output_dir / "case_events.parquet", index=False, compression="zstd"
    )
    cohort_cases.to_csv(args.output_dir / "cohort_cases.csv", index=False)
    if not portfolio_curves.empty:
        portfolio_curves.to_parquet(
            args.output_dir / "portfolio_curves.parquet",
            index=False,
            compression="zstd",
        )
        portfolio_trades.to_parquet(
            args.output_dir / "portfolio_trades.parquet",
            index=False,
            compression="zstd",
        )
        portfolio_metrics.to_csv(args.output_dir / "portfolio_metrics.csv", index=False)
    _write_json(args.output_dir / "decision.json", decision)
    _write_json(args.output_dir / "metadata.json", metadata)
    _write_report(
        args.output_dir / "report.md",
        metrics,
        cohorts,
        portfolio_metrics,
        case_events,
        cohort_cases,
        decision,
        metadata,
    )
    print(f"strict status: {decision['status']}", flush=True)
    print(f"report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
