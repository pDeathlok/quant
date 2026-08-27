#!/usr/bin/env python3
"""Validate causal profit locks and partial builds after monthly low-9 anchors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.research.blood_chip_deep_base import build_deep_base_features
from quant.research.monthly_low_zone_profit_lock import (
    PROFIT_LOCK_PERIODS,
    ProfitLockConfig,
    ProfitLockPortfolioConfig,
    assemble_staged_anchor_events,
    evaluate_profit_lock_events,
    simulate_profit_lock_portfolio,
    summarize_profit_lock_events,
)
from quant.research.monthly_low_zone_confirmation import (
    MonthlyConfirmationConfig,
    build_market_breadth_features,
)


RESEARCH_RULES = ("anchor_direct", "no_new_low_20", "range_mid_reclaim")
HORIZONS = (252, 504)
TARGETS = (0.10, 0.15, 0.20)
PANIC_GATES = (
    "panic_pos20",
    "panic_pos30",
    "panic_median_m8",
    "panic_pos30_median_m5",
)
PRIMARY_SELECTION_PERIODS = (
    "development_2013_2016",
    "exposed_validation_2017_2020",
)
PORTFOLIO_PERIODS = {
    "development_2013_2016": (pd.Timestamp("2013-01-01"), pd.Timestamp("2016-12-31")),
    "exposed_validation_2017_2020": (
        pd.Timestamp("2017-01-01"),
        pd.Timestamp("2020-12-31"),
    ),
    "seen_diagnostic_2021_2024": (
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
    "time_out_2025_to_cutoff": (
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2026-07-31"),
    ),
}


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
        "--signals",
        type=Path,
        default=Path("reports/research/monthly_low_zone_confirmation_breadth/signals.parquet"),
    )
    parser.add_argument(
        "--baseline-events",
        type=Path,
        default=Path("reports/research/monthly_low_zone_confirmation_breadth/events.parquet"),
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
        default=Path("reports/research/monthly_low_zone_profit_lock"),
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
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_benchmark(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    source = frame["trade_date"] if "trade_date" in frame else frame["date"]
    parsed = pd.to_datetime(source, errors="coerce")
    compact = pd.to_datetime(
        source.astype("string").str.replace(r"\.0$", "", regex=True).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    frame["date"] = compact.fillna(parsed).dt.normalize()
    return frame.loc[frame["date"].le(cutoff)].sort_values("date").copy()


def _load_daily_prices(
    feature_cache: Path,
    supplemental_root: Path,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "ts_code",
        "date",
        "open",
        "high",
        "low",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
        "return_20d",
        "prior_amount_median_20d",
    ]
    print(f"loading primary execution prices: {feature_cache}", flush=True)
    primary = pd.read_parquet(feature_cache, columns=columns)
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
    paths = sorted(supplemental_root.glob("*.parquet"))
    raw_parts: list[pd.DataFrame] = []
    for index, path in enumerate(paths, start=1):
        raw_parts.append(pd.read_parquet(path, columns=raw_columns))
        if index % 100 == 0:
            print(f"loaded supplemental histories: {index}/{len(paths)}", flush=True)
    if raw_parts:
        supplemental_raw = pd.concat(raw_parts, ignore_index=True, sort=False)
        print(
            f"building causal prices for supplemental rows: {len(supplemental_raw):,}",
            flush=True,
        )
        supplemental = build_deep_base_features(supplemental_raw)
        supplemental = supplemental[columns]
        supplemental = supplemental.loc[supplemental["date"].le(cutoff)].copy()
    else:
        supplemental = pd.DataFrame(columns=columns)
    primary_symbols = set(primary["ts_code"].astype(str).unique())
    overlap_rows = int(
        supplemental["ts_code"].astype(str).isin(primary_symbols).sum()
    )
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
        "supplemental_overlap_rows": overlap_rows,
        "combined_rows": int(len(combined)),
        "combined_symbols": int(combined["ts_code"].nunique()),
    }


def _attach_systemic_panic_state(
    events: pd.DataFrame,
    breadth: pd.DataFrame,
) -> pd.DataFrame:
    anchor_dates = (
        events.loc[events["rule"].eq("anchor_direct"), ["anchor_id", "signal_date"]]
        .drop_duplicates("anchor_id")
        .copy()
    )
    anchor_dates["signal_date"] = pd.to_datetime(
        anchor_dates["signal_date"], errors="coerce"
    ).dt.normalize()
    state = anchor_dates.merge(
        breadth,
        left_on="signal_date",
        right_on="date",
        how="left",
        validate="many_to_one",
    ).drop(columns="date")
    state["panic_pos20"] = state["breadth_positive_share_20d"].le(0.20)
    state["panic_pos30"] = state["breadth_positive_share_20d"].le(0.30)
    state["panic_median_m8"] = state["breadth_median_return_20d"].le(-0.08)
    state["panic_pos30_median_m5"] = state["panic_pos30"] & state[
        "breadth_median_return_20d"
    ].le(-0.05)
    return events.merge(state.drop(columns="signal_date"), on="anchor_id", how="left")


def _panic_event_metrics(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.loc[
        events["rule"].eq("anchor_direct")
        & events["horizon"].eq(504)
        & events["baseline_outcome_completed"].fillna(False)
        & events["outcome_completed"].fillna(False)
    ].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for period, (start, end) in PROFIT_LOCK_PERIODS.items():
        scoped = frame.loc[frame["signal_date"].between(start, end, inclusive="both")]
        for gate in PANIC_GATES:
            gated = scoped.loc[scoped[gate].fillna(False)]
            for target, group in gated.groupby("target_return", observed=True, sort=True):
                returns = pd.to_numeric(group["net_return"], errors="coerce").dropna()
                losses = float(-returns.loc[returns <= 0.0].sum())
                cohorts = group.groupby("signal_date", observed=True)[
                    "net_return"
                ].mean()
                standard_error = (
                    float(cohorts.std(ddof=1) / np.sqrt(len(cohorts)))
                    if len(cohorts) > 1
                    else np.nan
                )
                rows.append(
                    {
                        "period": period,
                        "panic_gate": gate,
                        "target_return": float(target),
                        "events": int(len(returns)),
                        "signal_dates": int(len(cohorts)),
                        "positive_signal_dates": int(cohorts.gt(0.0).sum()),
                        "win_rate": float(returns.gt(0.0).mean())
                        if len(returns)
                        else np.nan,
                        "median_net_return": float(returns.median())
                        if len(returns)
                        else np.nan,
                        "mean_net_return": float(returns.mean())
                        if len(returns)
                        else np.nan,
                        "profit_factor": (
                            float(returns.loc[returns > 0.0].sum() / losses)
                            if losses > 0.0
                            else np.nan
                        ),
                        "date_equal_mean_net_return": float(cohorts.mean())
                        if len(cohorts)
                        else np.nan,
                        "date_cluster_ci95_low": (
                            float(cohorts.mean() - 1.96 * standard_error)
                            if np.isfinite(standard_error)
                            else np.nan
                        ),
                        "worst_signal_date_return": float(cohorts.min())
                        if len(cohorts)
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _summarize_staged(staged: pd.DataFrame) -> pd.DataFrame:
    frame = staged.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for period, (start, end) in PROFIT_LOCK_PERIODS.items():
        scoped = frame.loc[frame["signal_date"].between(start, end, inclusive="both")]
        for keys, group in scoped.groupby(
            ["policy", "horizon", "target_return"], observed=True, sort=True
        ):
            complete = group.loc[group["outcome_completed"].fillna(False)]
            budget_returns = pd.to_numeric(
                complete["budget_return"], errors="coerce"
            ).dropna()
            committed_returns = pd.to_numeric(
                complete["committed_return"], errors="coerce"
            ).dropna()
            losses = float(-budget_returns.loc[budget_returns <= 0.0].sum())
            date_returns = complete.groupby("signal_date", observed=True)[
                "budget_return"
            ].mean()
            rows.append(
                {
                    "period": period,
                    "policy": keys[0],
                    "horizon": int(keys[1]),
                    "target_return": float(keys[2]),
                    "completed_anchors": int(len(complete)),
                    "signal_dates": int(complete["signal_date"].nunique()),
                    "win_rate": float(budget_returns.gt(0.0).mean())
                    if len(budget_returns)
                    else np.nan,
                    "median_budget_return": float(budget_returns.median())
                    if len(budget_returns)
                    else np.nan,
                    "mean_budget_return": float(budget_returns.mean())
                    if len(budget_returns)
                    else np.nan,
                    "median_committed_return": float(committed_returns.median())
                    if len(committed_returns)
                    else np.nan,
                    "profit_factor": (
                        float(budget_returns.loc[budget_returns > 0.0].sum() / losses)
                        if losses > 0.0
                        else np.nan
                    ),
                    "mean_build_fraction": float(complete["build_fraction"].mean())
                    if len(complete)
                    else np.nan,
                    "add_rate": float(
                        complete["add_status"].eq("added_before_probe_exit").mean()
                    )
                    if len(complete)
                    else np.nan,
                    "date_equal_mean_budget_return": float(date_returns.mean())
                    if len(date_returns)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _case_catalog(events: pd.DataFrame, staged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    direct = events.loc[events["rule"].eq("anchor_direct")]
    for _, event in direct.iterrows():
        categories: list[str] = []
        if event["target_hit"] and event["baseline_net_return"] <= 0.0:
            categories.append("target_salvaged_baseline_loser")
        if (
            event["target_hit"]
            and event["baseline_exit_reason"] == "missing_bar_writeoff"
        ):
            categories.append("target_before_later_writeoff")
        if event["target_hit"] and event["holding_sessions"] > 252:
            categories.append("slow_target_after_252")
        if not event["target_hit"] and event["net_return"] <= -0.50:
            categories.append("never_rebounded_tail_loss")
        for category in categories:
            rows.append(
                {
                    "category": category,
                    "anchor_id": event["anchor_id"],
                    "ts_code": event["ts_code"],
                    "signal_date": event["signal_date"],
                    "horizon": event["horizon"],
                    "target_return": event["target_return"],
                    "exit_date": event["exit_date"],
                    "holding_sessions": event["holding_sessions"],
                    "net_return": event["net_return"],
                    "baseline_net_return": event["baseline_net_return"],
                    "baseline_exit_reason": event["baseline_exit_reason"],
                    "policy": pd.NA,
                }
            )
    for _, event in staged.iterrows():
        category = {
            "confirmation_after_probe_exit": "confirmation_after_probe_exit",
            "added_before_probe_exit": "confirmation_added_before_exit",
        }.get(event["add_status"])
        if category is None:
            continue
        rows.append(
            {
                "category": category,
                "anchor_id": event["anchor_id"],
                "ts_code": event["ts_code"],
                "signal_date": event["signal_date"],
                "horizon": event["horizon"],
                "target_return": event["target_return"],
                "exit_date": event["exit_date"],
                "holding_sessions": np.nan,
                "net_return": event["budget_return"],
                "baseline_net_return": event["probe_net_return"],
                "baseline_exit_reason": event["probe_exit_reason"],
                "policy": event["policy"],
            }
        )
    return pd.DataFrame(rows)


def _decision(metrics: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    candidates: list[tuple[tuple[float, float, float], str]] = []
    direct = metrics.loc[metrics["rule"].eq("anchor_direct")]
    for (horizon, target), group in direct.groupby(
        ["horizon", "target_return"], observed=True, sort=True
    ):
        key = f"h{int(horizon)}_tp{int(round(float(target) * 100))}"
        selected = group.set_index("period").reindex(PRIMARY_SELECTION_PERIODS)
        complete = not selected.isna().all(axis=1).any()
        rule_checks = {
            "periods_present": bool(complete),
            "minimum_samples": bool(
                complete
                and selected.loc["development_2013_2016", "completed_events"] >= 100
                and selected.loc[
                    "exposed_validation_2017_2020", "completed_events"
                ]
                >= 100
            ),
            "win_rate_at_least_75pct": bool(
                complete and selected["win_rate"].ge(0.75).all()
            ),
            "positive_median": bool(
                complete and selected["median_net_return"].gt(0.0).all()
            ),
            "profit_factor_at_least_1_50": bool(
                complete and selected["profit_factor"].ge(1.50).all()
            ),
            "positive_date_equal_mean": bool(
                complete
                and selected["date_equal_mean_net_return"].gt(0.0).all()
            ),
        }
        rule_checks["event_gate_passed"] = all(rule_checks.values())
        checks[key] = rule_checks
        if rule_checks["event_gate_passed"]:
            score = (
                float(selected["win_rate"].min()),
                float(selected["profit_factor"].min()),
                float(selected["mean_net_return"].mean()),
            )
            candidates.append((score, key))
    selected = max(candidates)[1] if candidates else None
    return {
        "exposure_warning": (
            "2017-2020 and 2021-2024 were inspected before this iteration; "
            "event passes are research candidates, not untouched validation"
        ),
        "checks": checks,
        "event_candidate": selected,
        "status": (
            "event_candidate_requires_portfolio_validation"
            if selected
            else "no_reasonable_profit_lock_event"
        ),
        "selected_structure": None,
    }


def _portfolio_metrics(curve: pd.DataFrame) -> dict[str, float]:
    if curve.empty:
        return {
            "total_return": np.nan,
            "cagr": np.nan,
            "maximum_drawdown": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_zero_rate": np.nan,
            "positive_rolling_12m_share": np.nan,
            "mean_invested_fraction": np.nan,
            "maximum_active_anchors": np.nan,
        }
    frame = curve.copy().sort_values("date")
    nav = pd.to_numeric(frame["nav"], errors="coerce")
    daily_return = nav.pct_change(fill_method=None).dropna()
    elapsed_days = max(
        (pd.Timestamp(frame["date"].iloc[-1]) - pd.Timestamp(frame["date"].iloc[0])).days,
        1,
    )
    years = elapsed_days / 365.25
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)
    drawdown = nav / nav.cummax() - 1.0
    volatility = float(daily_return.std(ddof=1) * np.sqrt(252.0))
    sharpe = (
        float(daily_return.mean() / daily_return.std(ddof=1) * np.sqrt(252.0))
        if len(daily_return) > 1 and daily_return.std(ddof=1) > 0.0
        else np.nan
    )
    rolling = nav / nav.shift(252) - 1.0
    rolling = rolling.dropna()
    return {
        "total_return": total_return,
        "cagr": cagr,
        "maximum_drawdown": float(drawdown.min()),
        "annualized_volatility": volatility,
        "sharpe_zero_rate": sharpe,
        "positive_rolling_12m_share": float(rolling.gt(0.0).mean())
        if len(rolling)
        else np.nan,
        "mean_invested_fraction": float(frame["invested_fraction"].mean()),
        "maximum_active_anchors": float(frame["active_anchors"].max()),
    }


def _portfolio_decision(
    portfolio_metrics: pd.DataFrame,
    event_decision: dict[str, Any],
    panic_metrics: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, dict[str, bool]] = {}
    candidates: list[tuple[tuple[float, float], str]] = []
    research_leads: list[tuple[tuple[float, float], str]] = []
    selection_periods = tuple(PORTFOLIO_PERIODS)[:3]
    for (
        panic_gate,
        capacity,
        policy,
        lock_fraction,
        target,
    ), group in portfolio_metrics.groupby(
        [
            "panic_gate",
            "maximum_anchors",
            "policy",
            "profit_lock_fraction",
            "target_return",
        ],
        observed=True,
        sort=True,
    ):
        key = (
            f"{panic_gate}_anchors{int(capacity)}_{policy}_"
            f"lock{int(round(float(lock_fraction) * 100))}_"
            f"tp{int(round(float(target) * 100))}"
        )
        selected = group.set_index("period").reindex(selection_periods)
        complete = not selected.isna().all(axis=1).any()
        operational_checks = {
            "periods_present": bool(complete),
            "positive_cagr_each": bool(complete and selected["cagr"].gt(0.0).all()),
            "maximum_drawdown_no_worse_than_25pct": bool(
                complete and selected["maximum_drawdown"].ge(-0.25).all()
            ),
            "positive_rolling_12m_share_at_least_60pct": bool(
                complete
                and selected["positive_rolling_12m_share"].ge(0.60).all()
            ),
        }
        if panic_gate == "none":
            sample_ok = True
        else:
            samples = panic_metrics.loc[
                panic_metrics["panic_gate"].eq(panic_gate)
                & np.isclose(panic_metrics["target_return"], float(target))
            ].set_index("period").reindex(selection_periods)
            sample_ok = bool(
                not samples.isna().all(axis=1).any()
                and samples["signal_dates"].ge(24).all()
            )
        rule_checks = {
            **operational_checks,
            "minimum_24_independent_panic_dates_each": sample_ok,
        }
        rule_checks["operational_gate_passed"] = all(operational_checks.values())
        rule_checks["portfolio_gate_passed"] = all(rule_checks.values())
        checks[key] = rule_checks
        if rule_checks["operational_gate_passed"]:
            score = (
                float(selected["cagr"].min()),
                float(selected["cagr"].mean()),
            )
            research_leads.append((score, key))
        if rule_checks["portfolio_gate_passed"]:
            score = (
                float(selected["cagr"].min()),
                float(selected["cagr"].mean()),
            )
            candidates.append((score, key))
    selected_structure = max(candidates)[1] if candidates else None
    result = dict(event_decision)
    result.update(
        {
            "portfolio_checks": checks,
            "best_operational_research_lead": (
                max(research_leads)[1] if research_leads else None
            ),
            "selected_structure": selected_structure,
            "status": (
                "reasonable_research_structure_found"
                if selected_structure
                else "promising_structure_but_independent_panic_sample_insufficient"
                if research_leads
                else "event_candidate_failed_portfolio_validation"
            ),
        }
    )
    return result


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
    staged_metrics: pd.DataFrame,
    cases: pd.DataFrame,
    decision: dict[str, Any],
    metadata: dict[str, Any],
    portfolio_metrics: pd.DataFrame | None = None,
    panic_metrics: pd.DataFrame | None = None,
) -> None:
    direct = metrics.loc[metrics["rule"].eq("anchor_direct")].copy()
    direct = _format_percent(
        direct,
        [
            "target_return",
            "win_rate",
            "median_net_return",
            "mean_net_return",
            "target_hit_rate",
            "tail_loss_rate",
            "date_equal_mean_net_return",
            "date_cluster_ci95_low",
            "date_cluster_ci95_high",
        ],
    )
    staged_display = _format_percent(
        staged_metrics,
        [
            "target_return",
            "win_rate",
            "median_budget_return",
            "mean_budget_return",
            "median_committed_return",
            "mean_build_fraction",
            "add_rate",
            "date_equal_mean_budget_return",
        ],
    )
    case_summary = (
        cases.groupby(["category", "horizon", "target_return"], as_index=False)
        .agg(cases=("anchor_id", "count"))
        .sort_values(["category", "horizon", "target_return"])
        if not cases.empty
        else pd.DataFrame()
    )
    if decision.get("selected_structure"):
        conclusion = f"研究结构 `{decision['selected_structure']}` 通过冻结的事件与组合门槛。"
    elif decision.get("best_operational_research_lead"):
        conclusion = (
            f"最强研究线索 `{decision['best_operational_research_lead']}` 通过收益和回撤门槛，"
            "但独立恐慌月份不足，不能升级为已验证结构。"
        )
    elif decision["event_candidate"]:
        conclusion = "事件层通过，但没有结构通过冻结的日度组合门槛。"
    else:
        conclusion = "没有止盈结构通过冻结事件门槛。"
    portfolio_display = (
        _format_percent(
            portfolio_metrics,
            [
                "target_return",
                "profit_lock_fraction",
                "total_return",
                "cagr",
                "maximum_drawdown",
                "annualized_volatility",
                "positive_rolling_12m_share",
                "mean_invested_fraction",
            ],
        )
        if portfolio_metrics is not None
        else pd.DataFrame()
    )
    panic_display = (
        _format_percent(
            panic_metrics,
            [
                "target_return",
                "win_rate",
                "median_net_return",
                "mean_net_return",
                "date_equal_mean_net_return",
                "date_cluster_ci95_low",
                "worst_signal_date_return",
            ],
        )
        if panic_metrics is not None
        else pd.DataFrame()
    )
    path.write_text(
        f"""# 月线低9等待型止盈与试仓状态机

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

{conclusion}

2017—2020 与 2021—2024 已在提出本轮规则前被查看，因此本报告只形成研究候选；2025 单独列为有限时间外诊断。

## 直接试仓事件

{direct.to_markdown(index=False)}

## 确认前仍持有才加仓

{staged_display.to_markdown(index=False)}

## 日度盯市组合

{portfolio_display.to_markdown(index=False) if not portfolio_display.empty else '_事件层未通过，未构建组合_'}

## 系统性恐慌事件稳健性

{panic_display.to_markdown(index=False) if not panic_display.empty else '_未运行恐慌分层_'}

## Case 摘要

{case_summary.to_markdown(index=False) if not case_summary.empty else '_无案例_'}

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
    calendar = pd.DatetimeIndex(sorted(benchmark["date"].dropna().unique()))
    daily, daily_metadata = _load_daily_prices(
        args.feature_cache, args.supplemental_root, cutoff
    )
    print("building same-date liquid-universe panic breadth", flush=True)
    breadth = build_market_breadth_features(daily, MonthlyConfirmationConfig())
    signals = pd.read_parquet(args.signals)
    signal_lookup = signals.loc[
        signals["rule"].isin(RESEARCH_RULES),
        ["signal_id", "anchor_id", "rule", "signal_date"],
    ].drop_duplicates("signal_id")
    baseline = pd.read_parquet(args.baseline_events)
    baseline = baseline.loc[
        baseline["rule"].isin(RESEARCH_RULES) & baseline["horizon"].isin(HORIZONS)
    ].copy()
    baseline = baseline.merge(
        signal_lookup[["signal_id", "anchor_id"]],
        on="signal_id",
        how="left",
        validate="many_to_one",
    )
    if baseline["anchor_id"].isna().any():
        raise ValueError("baseline events missing anchor_id after signal merge")
    baseline["anchor_id"] = baseline["anchor_id"].astype(np.int64)
    event_parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        print(f"evaluating causal profit locks: horizon={horizon}", flush=True)
        event_parts.append(
            evaluate_profit_lock_events(
                daily,
                baseline,
                calendar,
                ProfitLockConfig(
                    horizon_sessions=horizon,
                    target_returns=TARGETS,
                ),
                benchmark,
                resolve_observed_targets=True,
            )
        )
    events = pd.concat(event_parts, ignore_index=True, sort=False)
    events = _attach_systemic_panic_state(events, breadth)
    metrics = summarize_profit_lock_events(events)
    panic_metrics = _panic_event_metrics(events)
    staged_parts: list[pd.DataFrame] = []
    for add_rule in ("no_new_low_20", "range_mid_reclaim"):
        staged_parts.append(
            assemble_staged_anchor_events(events, add_rule=add_rule)
        )
    staged = pd.concat(staged_parts, ignore_index=True, sort=False)
    staged_metrics = _summarize_staged(staged)
    cases = _case_catalog(events, staged)
    decision = _decision(metrics)
    portfolio_curve_parts: list[pd.DataFrame] = []
    portfolio_trade_parts: list[pd.DataFrame] = []
    portfolio_audits: list[dict[str, Any]] = []
    portfolio_metric_rows: list[dict[str, Any]] = []
    if decision["event_candidate"]:
        portfolio_events = events.loc[events["horizon"].eq(504)].copy()
        structures = (
            ("probe_25_only", None, 1.00),
            ("probe_25_repair_25", "no_new_low_20", 1.00),
            ("probe_25_reclaim_25", "range_mid_reclaim", 1.00),
            ("probe_25_only_core80_runner20", None, 0.80),
            ("probe_25_only_core90_runner10", None, 0.90),
        )
        for period, (start, end) in PORTFOLIO_PERIODS.items():
            effective_end = min(end, cutoff)
            for maximum_anchors in (20, 50, 100):
                for policy, add_rule, lock_fraction in structures:
                    portfolio_config = ProfitLockPortfolioConfig(
                        maximum_anchors=maximum_anchors,
                        target_anchor_fraction=2.0 / maximum_anchors,
                        profit_lock_fraction=lock_fraction,
                    )
                    for target in TARGETS:
                        print(
                            f"portfolio: {period} anchors={maximum_anchors} "
                            f"{policy} lock={lock_fraction:.0%} target={target:.0%}",
                            flush=True,
                        )
                        curve, trades, audit = simulate_profit_lock_portfolio(
                            daily,
                            portfolio_events,
                            calendar,
                            start_date=start,
                            end_date=effective_end,
                            target_return=target,
                            add_rule=add_rule,
                            config=portfolio_config,
                        )
                        curve["period"] = period
                        curve["panic_gate"] = "none"
                        curve["maximum_anchors"] = maximum_anchors
                        curve["policy"] = policy
                        curve["profit_lock_fraction"] = lock_fraction
                        curve["target_return"] = target
                        trades["period"] = period
                        trades["panic_gate"] = "none"
                        trades["maximum_anchors"] = maximum_anchors
                        trades["policy"] = policy
                        trades["profit_lock_fraction"] = lock_fraction
                        trades["target_return"] = target
                        portfolio_curve_parts.append(curve)
                        portfolio_trade_parts.append(trades)
                        portfolio_audits.append(
                            {
                                "period": period,
                                "panic_gate": "none",
                                "maximum_anchors": maximum_anchors,
                                "policy": policy,
                                "profit_lock_fraction": lock_fraction,
                                **audit,
                            }
                        )
                        portfolio_metric_rows.append(
                            {
                                "period": period,
                                "panic_gate": "none",
                                "maximum_anchors": maximum_anchors,
                                "policy": policy,
                                "profit_lock_fraction": lock_fraction,
                                "target_return": target,
                                **_portfolio_metrics(curve),
                                "entered_anchors": audit["entered_anchors"],
                                "added_tranches": audit["added_tranches"],
                                "skipped_capacity": audit["skipped_capacity"],
                            }
                        )
                for panic_gate in PANIC_GATES:
                    panic_events = portfolio_events.loc[
                        portfolio_events[panic_gate].fillna(False)
                    ]
                    policy = "probe_25_only_systemic_panic"
                    lock_fraction = 1.0
                    portfolio_config = ProfitLockPortfolioConfig(
                        maximum_anchors=maximum_anchors,
                        target_anchor_fraction=2.0 / maximum_anchors,
                    )
                    for target in TARGETS:
                        print(
                            f"portfolio: {period} gate={panic_gate} "
                            f"anchors={maximum_anchors} target={target:.0%}",
                            flush=True,
                        )
                        curve, trades, audit = simulate_profit_lock_portfolio(
                            daily,
                            panic_events,
                            calendar,
                            start_date=start,
                            end_date=effective_end,
                            target_return=target,
                            add_rule=None,
                            config=portfolio_config,
                        )
                        curve["period"] = period
                        curve["panic_gate"] = panic_gate
                        curve["maximum_anchors"] = maximum_anchors
                        curve["policy"] = policy
                        curve["profit_lock_fraction"] = lock_fraction
                        curve["target_return"] = target
                        trades["period"] = period
                        trades["panic_gate"] = panic_gate
                        trades["maximum_anchors"] = maximum_anchors
                        trades["policy"] = policy
                        trades["profit_lock_fraction"] = lock_fraction
                        trades["target_return"] = target
                        portfolio_curve_parts.append(curve)
                        portfolio_trade_parts.append(trades)
                        portfolio_audits.append(
                            {
                                "period": period,
                                "panic_gate": panic_gate,
                                "maximum_anchors": maximum_anchors,
                                "policy": policy,
                                "profit_lock_fraction": lock_fraction,
                                **audit,
                            }
                        )
                        portfolio_metric_rows.append(
                            {
                                "period": period,
                                "panic_gate": panic_gate,
                                "maximum_anchors": maximum_anchors,
                                "policy": policy,
                                "profit_lock_fraction": lock_fraction,
                                "target_return": target,
                                **_portfolio_metrics(curve),
                                "entered_anchors": audit["entered_anchors"],
                                "added_tranches": audit["added_tranches"],
                                "skipped_capacity": audit["skipped_capacity"],
                            }
                        )
        portfolio_curves = pd.concat(portfolio_curve_parts, ignore_index=True)
        portfolio_trades = pd.concat(portfolio_trade_parts, ignore_index=True)
        portfolio_metrics = pd.DataFrame(portfolio_metric_rows)
        decision = _portfolio_decision(portfolio_metrics, decision, panic_metrics)
    else:
        portfolio_curves = pd.DataFrame()
        portfolio_trades = pd.DataFrame()
        portfolio_metrics = pd.DataFrame()
    metadata = {
        "analysis_cutoff": cutoff,
        "targets": TARGETS,
        "horizons": HORIZONS,
        "rules": RESEARCH_RULES,
        "daily": daily_metadata,
        "signals_source": str(args.signals),
        "baseline_events_source": str(args.baseline_events),
        "benchmark_source": str(args.benchmark),
        "profit_event_rows": int(len(events)),
        "staged_anchor_rows": int(len(staged)),
        "case_rows": int(len(cases)),
        "breadth_rows": int(len(breadth)),
        "panic_gates": PANIC_GATES,
        "portfolio_capacity_neighborhood": [20, 50, 100],
        "portfolio_anchor_budget_rule": "2 / maximum_anchors",
        "portfolio_audits": portfolio_audits,
    }
    events.to_parquet(
        args.output_dir / "events.parquet", index=False, compression="zstd"
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    panic_metrics.to_csv(args.output_dir / "panic_event_metrics.csv", index=False)
    breadth.to_parquet(
        args.output_dir / "market_breadth_20d.parquet",
        index=False,
        compression="zstd",
    )
    staged.to_parquet(
        args.output_dir / "staged_events.parquet", index=False, compression="zstd"
    )
    staged_metrics.to_csv(args.output_dir / "staged_metrics.csv", index=False)
    cases.to_parquet(
        args.output_dir / "case_catalog.parquet", index=False, compression="zstd"
    )
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
        portfolio_metrics.to_csv(
            args.output_dir / "portfolio_metrics.csv", index=False
        )
    _write_json(args.output_dir / "decision.json", decision)
    _write_json(args.output_dir / "metadata.json", metadata)
    _write_report(
        args.output_dir / "report.md",
        metrics,
        staged_metrics,
        cases,
        decision,
        metadata,
        portfolio_metrics,
        panic_metrics,
    )
    print(f"event candidate: {decision['event_candidate']}", flush=True)
    print(f"selected structure: {decision['selected_structure']}", flush=True)
    print(f"report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
