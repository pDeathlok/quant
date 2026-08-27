#!/usr/bin/env python3
"""Backtest completed monthly low-9 and monthly/weekly KDJ low-zone events."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.research import blood_chip as blood_chip_core
from quant.research.monthly_low_zone import (
    EVENT_PERIODS,
    SIGNAL_RULES,
    MonthlyLowZoneConfig,
    build_monthly_weekly_features,
    evaluate_monthly_low_zone_events,
    generate_monthly_low_zone_signals,
    summarize_monthly_low_zone_events,
)


SELECTION_PERIODS = ("development_2013_2016", "validation_2017_2020")
SELECTION_HORIZON = 252
DAILY_COLUMNS = [
    "ts_code",
    "trade_date",
    "date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "amount",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--stock-basic-history",
        type=Path,
        default=Path("data/raw/stock_basic_history.parquet"),
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
        default=Path("reports/research/monthly_low_zone"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/research/monthly_low_zone"),
    )
    parser.add_argument("--build-cache", action="store_true")
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
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_feature_daily(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    selected = [column for column in DAILY_COLUMNS if column in available]
    required = {
        "ts_code",
        "date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "amount",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
    }
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"feature cache missing columns: {missing}")
    frame = pd.read_parquet(path, columns=selected)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.loc[frame["date"].le(cutoff)].copy()
    if "trade_date" not in frame:
        frame["trade_date"] = frame["date"].dt.strftime("%Y%m%d")
    return frame[DAILY_COLUMNS]


def _load_supplemental_daily(root: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    paths = sorted(root.glob("*.parquet"))
    for index, path in enumerate(paths, start=1):
        frame = pd.read_parquet(
            path,
            columns=[
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
            ],
        )
        frames.append(frame)
        if index % 100 == 0:
            print(f"loaded supplemental delisted histories: {index}/{len(paths)}", flush=True)
    if not frames:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    prepared = blood_chip_core._prepare_daily(raw)
    prepared = prepared.loc[prepared["date"].le(cutoff)].copy()
    continuous = blood_chip_core._add_causal_continuous_prices(prepared)
    return continuous[DAILY_COLUMNS]


def _load_combined_daily(
    feature_cache: Path,
    supplemental_root: Path,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"loading causal feature daily: {feature_cache}", flush=True)
    primary = _load_feature_daily(feature_cache, cutoff)
    supplemental = _load_supplemental_daily(supplemental_root, cutoff)
    primary_symbols = set(primary["ts_code"].astype(str).unique())
    overlap = int(supplemental["ts_code"].astype(str).isin(primary_symbols).sum())
    combined = pd.concat([primary, supplemental], ignore_index=True, sort=False)
    combined = (
        combined.sort_values(["ts_code", "date"])
        .drop_duplicates(["ts_code", "date"], keep="last")
        .reset_index(drop=True)
    )
    return combined, {
        "primary_rows": int(len(primary)),
        "primary_symbols": int(primary["ts_code"].nunique()),
        "supplemental_rows": int(len(supplemental)),
        "supplemental_symbols": int(supplemental["ts_code"].nunique()),
        "supplemental_overlap_rows": overlap,
        "combined_rows": int(len(combined)),
        "combined_symbols": int(combined["ts_code"].nunique()),
    }


def _load_benchmark(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    values = frame["trade_date"] if "trade_date" in frame else frame["date"]
    parsed = pd.to_datetime(values, errors="coerce")
    compact = pd.to_datetime(
        values.astype("string").str.replace(r"\.0$", "", regex=True).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    frame["date"] = compact.fillna(parsed)
    return frame.loc[frame["date"].le(cutoff)].sort_values("date").copy()


def _load_or_build_multitimeframe_cache(
    daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    cache_dir: Path,
    *,
    build_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    monthly_path = cache_dir / "monthly_features.parquet"
    weekly_path = cache_dir / "weekly_features.parquet"
    if monthly_path.exists() and weekly_path.exists() and not build_cache:
        print("loading monthly/weekly feature cache", flush=True)
        return pd.read_parquet(monthly_path), pd.read_parquet(weekly_path), True
    print(f"building completed monthly/weekly bars for {len(daily):,} daily rows", flush=True)
    monthly, weekly = build_monthly_weekly_features(daily, calendar)
    cache_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(monthly_path, index=False, compression="zstd")
    weekly.to_parquet(weekly_path, index=False, compression="zstd")
    return monthly, weekly, False


def _decision(metrics: pd.DataFrame) -> dict[str, Any]:
    primary = metrics.loc[metrics["horizon"].eq(SELECTION_HORIZON)].copy()
    checks: dict[str, dict[str, bool]] = {}
    scores: list[tuple[tuple[float, float, float, int], str]] = []
    for rule in SIGNAL_RULES:
        selected = primary.loc[primary["rule"].eq(rule)].set_index("period").reindex(
            SELECTION_PERIODS
        )
        complete = not selected.isna().all(axis=1).any()
        rule_checks = {
            "periods_complete": bool(complete),
            "minimum_100_events_each": bool(
                complete and selected["completed_events"].ge(100).all()
            ),
            "win_rate_at_least_60pct_each": bool(
                complete and selected["win_rate"].ge(0.60).all()
            ),
            "positive_median_each": bool(
                complete and selected["median_net_return"].gt(0.0).all()
            ),
            "profit_factor_at_least_1_50_each": bool(
                complete and selected["profit_factor"].ge(1.50).all()
            ),
            "excess_win_rate_at_least_50pct_each": bool(
                complete and selected["excess_win_rate"].ge(0.50).all()
            ),
            "positive_signal_date_cluster_ci95_lower_each": bool(
                complete and selected["date_cluster_ci95_low"].gt(0.0).all()
            ),
        }
        rule_checks["all_passed"] = all(rule_checks.values())
        checks[rule] = rule_checks
        if complete:
            score = (
                float(selected["win_rate"].min()),
                float(selected["profit_factor"].min()),
                float(selected["median_net_return"].min()),
                int(selected["completed_events"].sum()),
            )
            scores.append((score, rule))
    qualified = [rule for _, rule in scores if checks[rule]["all_passed"]]
    qualified_scores = [(score, rule) for score, rule in scores if rule in qualified]
    usable_scores = [
        (score, rule)
        for score, rule in scores
        if checks[rule]["minimum_100_events_each"]
    ]
    small_sample_scores = [
        (score, rule)
        for score, rule in scores
        if not checks[rule]["minimum_100_events_each"]
    ]
    selected_rule = max(qualified_scores)[1] if qualified_scores else None
    return {
        "selected_on_development_and_validation_only": True,
        "selection_periods": list(SELECTION_PERIODS),
        "selection_horizon_sessions": SELECTION_HORIZON,
        "seen_diagnostic_excluded_from_selection": True,
        "qualification_thresholds": {
            "minimum_completed_events_each_period": 100,
            "minimum_win_rate_each_period": 0.60,
            "minimum_median_net_return_each_period": 0.0,
            "minimum_profit_factor_each_period": 1.50,
            "minimum_excess_win_rate_each_period": 0.50,
            "minimum_signal_date_cluster_ci95_lower_each_period": 0.0,
        },
        "checks": checks,
        "qualified_rules": qualified,
        "selected_rule": selected_rule,
        "statistically_usable_rules": [rule for _, rule in usable_scores],
        "best_research_rule": max(usable_scores)[1] if usable_scores else None,
        "best_small_sample_diagnostic_rule": (
            max(small_sample_scores)[1] if small_sample_scores else None
        ),
        "staged_portfolio_decision": (
            "eligible_for_separate_staged_portfolio_research"
            if selected_rule is not None
            else "signal_layer_failed_do_not_build_staged_portfolio"
        ),
    }


def _yearly_metrics(events: pd.DataFrame) -> pd.DataFrame:
    completed = events.loc[
        events["entry_status"].eq("accepted")
        & events["outcome_completed"].fillna(False)
    ].copy()
    completed["signal_year"] = pd.to_datetime(completed["signal_date"]).dt.year
    rows: list[dict[str, Any]] = []
    for keys, group in completed.groupby(
        ["rule", "horizon", "signal_year"], observed=True, sort=True
    ):
        returns = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        losses = float(-returns.loc[returns <= 0.0].sum())
        rows.append(
            {
                "rule": keys[0],
                "horizon": int(keys[1]),
                "signal_year": int(keys[2]),
                "events": int(len(returns)),
                "win_rate": float(returns.gt(0.0).mean()),
                "median_net_return": float(returns.median()),
                "profit_factor": (
                    float(returns.loc[returns > 0.0].sum() / losses)
                    if losses > 0.0
                    else np.nan
                ),
                "writeoff_rate": float(
                    group["exit_reason"].eq("missing_bar_writeoff").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _percent(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2%}"


def _number(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "—" if pd.isna(number) else f"{float(number):.2f}"


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    decision: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    primary = metrics.loc[metrics["horizon"].eq(SELECTION_HORIZON)].copy()
    display = primary[
        [
            "period",
            "rule",
            "signals",
            "completed_events",
            "signal_dates",
            "win_rate",
            "win_rate_wilson_lower_95",
            "median_net_return",
            "profit_factor",
            "mean_excess_net_return",
            "excess_win_rate",
            "mean_mae",
            "writeoff_rate",
            "date_equal_mean_net_return",
            "date_cluster_ci95_low",
            "date_cluster_ci95_high",
        ]
    ].copy()
    for column in (
        "win_rate",
        "win_rate_wilson_lower_95",
        "median_net_return",
        "mean_excess_net_return",
        "excess_win_rate",
        "mean_mae",
        "writeoff_rate",
        "date_equal_mean_net_return",
        "date_cluster_ci95_low",
        "date_cluster_ci95_high",
    ):
        display[column] = display[column].map(_percent)
    display["profit_factor"] = display["profit_factor"].map(_number)
    usable_rules = decision.get("statistically_usable_rules", [])
    horizon = metrics.loc[
        metrics["rule"].isin(usable_rules),
        [
            "period",
            "rule",
            "horizon",
            "completed_events",
            "win_rate",
            "median_net_return",
            "profit_factor",
            "excess_win_rate",
        ],
    ].copy()
    for column in ("win_rate", "median_net_return", "excess_win_rate"):
        horizon[column] = horizon[column].map(_percent)
    horizon["profit_factor"] = horizon["profit_factor"].map(_number)
    best = decision.get("best_research_rule")
    small_sample_best = decision.get("best_small_sample_diagnostic_rule")
    stability = yearly.loc[
        yearly["rule"].eq(best)
        & yearly["horizon"].eq(SELECTION_HORIZON)
        & yearly["signal_year"].between(2013, 2024)
    ].copy()
    if not stability.empty:
        stability["win_rate"] = stability["win_rate"].map(_percent)
        stability["median_net_return"] = stability["median_net_return"].map(_percent)
        stability["profit_factor"] = stability["profit_factor"].map(_number)
        stability["writeoff_rate"] = stability["writeoff_rate"].map(_percent)
    selected = decision.get("selected_rule")
    conclusion = (
        f"`{selected}` 通过信号层冻结门槛，可以进入独立分批组合研究。"
        if selected
        else "没有规则同时通过开发期和验证期的高胜率门槛；暂不构建分批组合。"
    )
    path.write_text(
        f"""# 月线低9与月周KDJ低位区间回测

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

{conclusion}

- 开发/验证综合排序第一：`{best}`；排序不能替代60%胜率与100事件硬门槛。
- 小样本诊断中排名第一的是 `{small_sample_best}`，但它未达到每期100事件，不参与策略选择。
- 所有规则先要求从此前完成月线前高回撤至少50%，信号在月末收盘后确认并于下一可交易日开盘进入。
- 2021年以后仅为已见诊断，不是盲测。
- 同一月末的大批信号按一个相关簇处理；日期等权收益95%置信下界也必须为正。
- 当前执行决定：`{decision['staged_portfolio_decision']}`。

## 12个月主检验

{display.to_markdown(index=False)}

## 达到最低样本量规则的6/12/24个月分期对照

{horizon.to_markdown(index=False)}

## 综合排序第一的年度稳定性

{stability.to_markdown(index=False) if not stability.empty else '_无已完成事件_'}

## 口径

- 月线低9：连续9根完成月线 `close[t] < close[t-4]`，仅计数等于9的月份触发。
- KDJ：9期RSV，K/D按1/3递推，J=3K-2D；周线只在W-FRI标签不晚于月末时可见。
- 七条规则固定比较月线低9、月J≤-10/-20、月周J≤-10及周J从≤-10上穿的修复形态。
- 退市股补充历史纳入研究；目标日停牌则在复牌首日开盘退出，连续60个市场交易日无行情则按-100%核销。
- 收益扣20bp往返成本；未走完目标期限的事件不进入对应统计。

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
    benchmark = _load_benchmark(args.benchmark, cutoff)
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(benchmark["date"]).dropna().unique()))
    daily, daily_metadata = _load_combined_daily(
        args.feature_cache,
        args.supplemental_root,
        cutoff,
    )
    monthly, weekly, cache_hit = _load_or_build_multitimeframe_cache(
        daily,
        calendar,
        args.cache_dir,
        build_cache=bool(args.build_cache),
    )
    config = MonthlyLowZoneConfig()
    print("generating seven frozen monthly low-zone rules", flush=True)
    signals = generate_monthly_low_zone_signals(monthly, weekly, config)
    print(f"signals: {len(signals):,}", flush=True)
    events = evaluate_monthly_low_zone_events(
        daily,
        signals,
        benchmark,
        calendar,
        config,
    )
    metrics = summarize_monthly_low_zone_events(events)
    yearly = _yearly_metrics(events)
    decision = _decision(metrics)
    metadata = {
        "analysis_cutoff": cutoff,
        "price_contract": "causal_forward_only_continuous_ohlc",
        "completed_month_contract": "market_last_session_of_calendar_month",
        "completed_week_contract": "W-FRI label must be <= month signal date",
        "config": config.to_dict(),
        "daily": daily_metadata,
        "monthly_rows": int(len(monthly)),
        "weekly_rows": int(len(weekly)),
        "signals": int(len(signals)),
        "unique_signal_symbol_dates": int(
            signals[["ts_code", "signal_date"]].drop_duplicates().shape[0]
        ),
        "event_rows": int(len(events)),
        "supplemental_files": len(list(args.supplemental_root.glob("*.parquet"))),
        "cache_hit": cache_hit,
        "stock_basic_history": str(args.stock_basic_history),
        "benchmark": str(args.benchmark),
        "selection_periods": EVENT_PERIODS,
    }
    signals.to_parquet(args.output_dir / "signals.parquet", index=False, compression="zstd")
    events.to_parquet(args.output_dir / "events.parquet", index=False, compression="zstd")
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    yearly.to_csv(args.output_dir / "yearly_metrics.csv", index=False)
    _write_json(args.output_dir / "decision.json", decision)
    _write_json(args.output_dir / "metadata.json", metadata)
    _write_report(
        args.output_dir / "report.md",
        metrics,
        yearly,
        decision,
        metadata,
    )
    print(f"selected rule: {decision['selected_rule']}", flush=True)
    print(f"report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
