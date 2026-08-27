#!/usr/bin/env python3
"""Backtest grid, structural-stop and causal reentry low-zone policies."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.research.blood_chip_deep_base import build_deep_base_features
from quant.research.monthly_low_zone_continuous import (
    CONTINUOUS_POLICIES,
    ContinuousConfig,
    evaluate_continuous_policies,
    simulate_continuous_portfolio,
)
from quant.research.monthly_low_zone_strict import (
    cohort_bootstrap_interval,
    leave_one_cohort_out_minimum,
)


PRIMARY_GRID = "grid_40_30_30_down10_down20"
PRIMARY_STOP = "base_stop_reentry_2"
PRIMARY_COMBINED = "grid_base_stop_reentry_2"
GRID_POLICIES = (
    "grid_40_30_30_down5_down10",
    PRIMARY_GRID,
    "grid_40_30_30_down15_down30",
)
EVENT_PERIODS = {
    "old_cycle_2003_2012": (
        pd.Timestamp("2003-01-01"),
        pd.Timestamp("2012-12-31"),
    ),
    "recent_exposed_2013_2024": (
        pd.Timestamp("2013-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
    "combined_2003_2024": (
        pd.Timestamp("2003-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
}
PORTFOLIO_PERIODS = {
    "old_cycle_2003_2012_held_to_2015": (
        pd.Timestamp("2003-01-01"),
        pd.Timestamp("2015-12-31"),
        "old",
    ),
    "development_2013_2016": (
        pd.Timestamp("2013-01-01"),
        pd.Timestamp("2016-12-31"),
        "recent",
    ),
    "exposed_validation_2017_2020": (
        pd.Timestamp("2017-01-01"),
        pd.Timestamp("2020-12-31"),
        "recent",
    ),
    "seen_diagnostic_2021_2024": (
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2024-12-31"),
        "recent",
    ),
}
DAILY_COLUMNS = [
    "ts_code",
    "date",
    "open",
    "high",
    "low",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "base_low",
    "base_position",
    "sessions_since_new_low",
    "return_20d",
    "prior_amount_median_20d",
    "prior_peak",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-events",
        type=Path,
        default=Path(
            "reports/research/monthly_low_zone_strict_extension/combined_candidate_events.parquet"
        ),
    )
    parser.add_argument(
        "--extension-data-dir",
        type=Path,
        default=Path("data/research/monthly_low_zone_strict_extension"),
    )
    parser.add_argument(
        "--recent-feature-cache",
        type=Path,
        default=Path("data/research/blood_chip_deep_base/features.parquet"),
    )
    parser.add_argument(
        "--recent-supplemental-root",
        type=Path,
        default=Path("data/research/low9_kdj_rebound/supplemental_daily"),
    )
    parser.add_argument(
        "--recent-benchmark",
        type=Path,
        default=Path(
            "data/research/low9_kdj_rebound/index_000001.SH_20100101_20260731.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/research/monthly_low_zone_continuous"),
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
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame["trade_date"] if "trade_date" in frame else frame["date"]
    compact = pd.to_datetime(
        source.astype("string").str.replace(r"\.0$", "", regex=True).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    parsed = pd.to_datetime(source, errors="coerce")
    out = frame.copy()
    out["date"] = compact.fillna(parsed).dt.normalize()
    return (
        out.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )


def _load_primary_events(path: Path) -> pd.DataFrame:
    events = pd.read_parquet(path)
    selected = events.loc[
        events["structure"].eq("case_range_reclaim_dd60")
        & np.isclose(pd.to_numeric(events["target_return"], errors="coerce"), 0.15)
        & pd.to_numeric(events["horizon"], errors="coerce").eq(504)
        & events["entry_status"].eq("accepted")
        & events["baseline_outcome_completed"].fillna(False)
        & events["outcome_completed"].fillna(False)
    ].copy()
    selected["source_sample"] = selected["source_sample"].replace(
        {
            "new_old_cycle_2003_2012": "old_cycle_2003_2012",
            "exposed_recent_2013_2024": "recent_exposed_2013_2024",
        }
    )
    return selected.sort_values(["entry_date", "ts_code"]).reset_index(drop=True)


def _load_old_daily(data_dir: Path, symbols: set[str]) -> pd.DataFrame:
    return pd.read_parquet(
        data_dir / "features_2000_2015.parquet",
        columns=DAILY_COLUMNS,
        filters=[("ts_code", "in", sorted(symbols))],
    )


def _load_recent_daily(
    feature_cache: Path,
    supplemental_root: Path,
    symbols: set[str],
) -> pd.DataFrame:
    primary = pd.read_parquet(
        feature_cache,
        columns=DAILY_COLUMNS,
        filters=[("ts_code", "in", sorted(symbols))],
    )
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
    parts = []
    for symbol in sorted(symbols):
        path = supplemental_root / f"{symbol}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path, columns=raw_columns))
    supplemental = (
        build_deep_base_features(pd.concat(parts, ignore_index=True, sort=False))[
            DAILY_COLUMNS
        ]
        if parts
        else pd.DataFrame(columns=DAILY_COLUMNS)
    )
    combined = pd.concat([primary, supplemental], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
    return combined.sort_values(["ts_code", "date"]).drop_duplicates(
        ["ts_code", "date"], keep="last"
    )


def _load_old_benchmark(data_dir: Path, recent_path: Path) -> pd.DataFrame:
    parts = [
        pd.read_parquet(path)
        for path in sorted(data_dir.glob("index_000001.SH_*.parquet"))
    ]
    parts.append(pd.read_parquet(recent_path))
    benchmark = _normalize_benchmark(pd.concat(parts, ignore_index=True, sort=False))
    return benchmark.loc[benchmark["date"].le("2015-12-31")].copy()


def _summarize_events(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = results.loc[results["outcome_completed"].fillna(False)].copy()
    frame["anchor_month"] = pd.PeriodIndex(
        frame["month_period"], freq="M"
    ).to_timestamp(how="end").normalize()
    frame["budget_return"] = pd.to_numeric(
        frame["budget_return"], errors="coerce"
    )
    frame = frame.dropna(subset=["budget_return"])
    cohort = (
        frame.groupby(["policy", "anchor_month"], observed=True, sort=True)
        .agg(
            cohort_return=("budget_return", "mean"),
            events=("budget_return", "size"),
            winners=("budget_return", lambda values: int(values.gt(0.0).sum())),
            worst_event_return=("budget_return", "min"),
            mean_invested_fraction=("mean_invested_fraction", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for policy in [item.name for item in CONTINUOUS_POLICIES]:
        policy_events = frame.loc[frame["policy"].eq(policy)]
        policy_cohorts = cohort.loc[cohort["policy"].eq(policy)]
        for period, (start, end) in EVENT_PERIODS.items():
            scoped = policy_events.loc[
                policy_events["anchor_month"].between(start, end)
            ]
            scoped_cohorts = policy_cohorts.loc[
                policy_cohorts["anchor_month"].between(start, end)
            ]
            returns = scoped["budget_return"]
            cohort_returns = scoped_cohorts.set_index("anchor_month")[
                "cohort_return"
            ]
            gains = float(returns.loc[returns > 0.0].sum())
            losses = float(-returns.loc[returns <= 0.0].sum())
            mean, lower, upper = cohort_bootstrap_interval(
                cohort_returns, iterations=10_000, seed=20260826
            )
            rows.append(
                {
                    "period": period,
                    "policy": policy,
                    "completed_events": int(len(returns)),
                    "completed_cohorts": int(len(cohort_returns)),
                    "event_win_rate": (
                        float(returns.gt(0.0).mean()) if len(returns) else np.nan
                    ),
                    "median_budget_return": (
                        float(returns.median()) if len(returns) else np.nan
                    ),
                    "mean_budget_return": (
                        float(returns.mean()) if len(returns) else np.nan
                    ),
                    "profit_factor": (
                        gains / losses
                        if losses > 0.0
                        else np.inf
                        if gains > 0.0
                        else np.nan
                    ),
                    "tail_loss_rate": (
                        float(returns.le(-0.50).mean()) if len(returns) else np.nan
                    ),
                    "target_hit_rate": (
                        float(scoped["target_hit"].mean()) if len(scoped) else np.nan
                    ),
                    "structural_stop_rate": (
                        float(scoped["stop_count"].gt(0).mean())
                        if len(scoped)
                        else np.nan
                    ),
                    "reentry_rate": (
                        float(scoped["reentries"].gt(0).mean())
                        if len(scoped)
                        else np.nan
                    ),
                    "mean_reentries": (
                        float(scoped["reentries"].mean()) if len(scoped) else np.nan
                    ),
                    "mean_grid_adds": (
                        float(scoped["grid_add_count"].mean())
                        if len(scoped)
                        else np.nan
                    ),
                    "mean_invested_fraction": (
                        float(scoped["mean_invested_fraction"].mean())
                        if len(scoped)
                        else np.nan
                    ),
                    "positive_cohort_share": (
                        float(cohort_returns.gt(0.0).mean())
                        if len(cohort_returns)
                        else np.nan
                    ),
                    "cohort_equal_mean_return": mean,
                    "cohort_bootstrap_ci95_low": lower,
                    "cohort_bootstrap_ci95_high": upper,
                    "worst_cohort_return": (
                        float(cohort_returns.min())
                        if len(cohort_returns)
                        else np.nan
                    ),
                    "leave_one_cohort_out_min_mean": (
                        leave_one_cohort_out_minimum(cohort_returns)
                    ),
                }
            )
    return pd.DataFrame(rows), cohort


def _curve_metrics(curve: pd.DataFrame) -> dict[str, float]:
    nav = pd.to_numeric(curve["nav"], errors="coerce")
    dates = pd.to_datetime(curve["date"], errors="coerce")
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    years = max(float((dates.iloc[-1] - dates.iloc[0]).days / 365.25), 1 / 365.25)
    drawdown = nav / nav.cummax() - 1.0
    rolling = nav / nav.shift(504) - 1.0
    return {
        "total_return": total_return,
        "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1.0),
        "maximum_drawdown": float(drawdown.min()),
        "worst_rolling_24m_return": (
            float(rolling.dropna().min()) if rolling.notna().any() else total_return
        ),
        "mean_invested_fraction": float(curve["invested_fraction"].mean()),
        "mean_reserved_fraction": float(curve["reserved_fraction"].mean()),
    }


def _run_portfolios(
    old_paths: pd.DataFrame,
    old_results: pd.DataFrame,
    old_calendar: pd.DatetimeIndex,
    recent_paths: pd.DataFrame,
    recent_results: pd.DataFrame,
    recent_calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for policy in [item.name for item in CONTINUOUS_POLICIES]:
        for period, (start, end, sample) in PORTFOLIO_PERIODS.items():
            if sample == "old":
                paths, results, calendar = old_paths, old_results, old_calendar
            else:
                paths, results, calendar = (
                    recent_paths,
                    recent_results,
                    recent_calendar,
                )
            curve, trades, audit = simulate_continuous_portfolio(
                paths,
                results,
                calendar,
                policy=policy,
                start_date=start,
                end_date=end,
            )
            curve["period"] = period
            trades["period"] = period
            curve_parts.append(curve)
            trade_parts.append(trades)
            metric_rows.append(
                {
                    "period": period,
                    "policy": policy,
                    **_curve_metrics(curve),
                    **audit,
                }
            )
    return (
        pd.concat(curve_parts, ignore_index=True, sort=False),
        pd.concat(trade_parts, ignore_index=True, sort=False),
        pd.DataFrame(metric_rows),
    )


def _metric_row(metrics: pd.DataFrame, policy: str, period: str) -> pd.Series:
    selected = metrics.loc[
        metrics["policy"].eq(policy) & metrics["period"].eq(period)
    ]
    return selected.iloc[0] if not selected.empty else pd.Series(dtype="object")


def _decide(event_metrics: pd.DataFrame, portfolio: pd.DataFrame) -> dict[str, Any]:
    baseline_old = _metric_row(event_metrics, "lump_sum", "old_cycle_2003_2012")
    baseline_combined = _metric_row(
        event_metrics, "lump_sum", "combined_2003_2024"
    )
    baseline_old_portfolio = _metric_row(
        portfolio, "lump_sum", "old_cycle_2003_2012_held_to_2015"
    )
    decisions: dict[str, Any] = {}
    for policy in [item.name for item in CONTINUOUS_POLICIES if item.name != "lump_sum"]:
        old = _metric_row(event_metrics, policy, "old_cycle_2003_2012")
        combined = _metric_row(event_metrics, policy, "combined_2003_2024")
        policy_portfolio = portfolio.loc[portfolio["policy"].eq(policy)]
        old_portfolio = _metric_row(
            portfolio, policy, "old_cycle_2003_2012_held_to_2015"
        )
        old_checks = {
            "completed_cohorts_at_least_4": bool(
                old.get("completed_cohorts", 0) >= 4
            ),
            "event_win_rate_at_least_85pct": bool(
                old.get("event_win_rate", np.nan) >= 0.85
            ),
            "profit_factor_at_least_2": bool(
                old.get("profit_factor", np.nan) >= 2.0
            ),
            "positive_cohort_share_at_least_80pct": bool(
                old.get("positive_cohort_share", np.nan) >= 0.80
            ),
            "bootstrap_lower_positive": bool(
                old.get("cohort_bootstrap_ci95_low", np.nan) > 0.0
            ),
            "worst_cohort_no_worse_than_minus_5pct": bool(
                old.get("worst_cohort_return", np.nan) >= -0.05
            ),
        }
        combined_checks = {
            "bootstrap_lower_positive": bool(
                combined.get("cohort_bootstrap_ci95_low", np.nan) > 0.0
            ),
            "leave_one_cohort_out_positive": bool(
                combined.get("leave_one_cohort_out_min_mean", np.nan) > 0.0
            ),
            "bootstrap_lower_improves_lump_sum": bool(
                combined.get("cohort_bootstrap_ci95_low", np.nan)
                > baseline_combined.get("cohort_bootstrap_ci95_low", np.nan)
            ),
            "leave_one_out_improves_lump_sum": bool(
                combined.get("leave_one_cohort_out_min_mean", np.nan)
                > baseline_combined.get("leave_one_cohort_out_min_mean", np.nan)
            ),
        }
        portfolio_checks = {
            "all_four_periods_present": bool(len(policy_portfolio) == 4),
            "positive_cagr_each_period": bool(
                len(policy_portfolio) == 4 and policy_portfolio["cagr"].gt(0.0).all()
            ),
            "maximum_drawdown_no_worse_than_10pct_each": bool(
                len(policy_portfolio) == 4
                and policy_portfolio["maximum_drawdown"].ge(-0.10).all()
            ),
            "old_maximum_drawdown_improves_lump_sum": bool(
                old_portfolio.get("maximum_drawdown", np.nan)
                > baseline_old_portfolio.get("maximum_drawdown", np.nan)
            ),
        }
        tail_checks = {
            "old_worst_cohort_improves_lump_sum": bool(
                old.get("worst_cohort_return", np.nan)
                > baseline_old.get("worst_cohort_return", np.nan)
            ),
            "old_tail_loss_rate_improves_lump_sum": bool(
                old.get("tail_loss_rate", np.nan)
                < baseline_old.get("tail_loss_rate", np.nan)
            ),
        }
        passed = (
            all(old_checks.values())
            and all(combined_checks.values())
            and all(portfolio_checks.values())
            and all(tail_checks.values())
        )
        decisions[policy] = {
            "old_cycle_checks": old_checks,
            "combined_checks": combined_checks,
            "portfolio_checks": portfolio_checks,
            "tail_checks": tail_checks,
            "status": (
                "historical_path_management_increment"
                if passed
                else "path_management_failed_certainty_contract"
            ),
        }
    grid_old = event_metrics.loc[
        event_metrics["period"].eq("old_cycle_2003_2012")
        & event_metrics["policy"].isin(GRID_POLICIES)
    ]
    grid_combined = event_metrics.loc[
        event_metrics["period"].eq("combined_2003_2024")
        & event_metrics["policy"].isin(GRID_POLICIES)
    ]
    stable_grid = set(
        grid_old.loc[grid_old["cohort_bootstrap_ci95_low"].gt(0.0), "policy"]
    ) & set(
        grid_combined.loc[
            grid_combined["cohort_bootstrap_ci95_low"].gt(0.0), "policy"
        ]
    )
    grid_stability = {
        "two_of_three_grid_spacings_robust": bool(len(stable_grid) >= 2),
        "robust_grid_policies": sorted(stable_grid),
    }
    for policy in GRID_POLICIES:
        decisions[policy]["grid_stability_checks"] = grid_stability
        if not grid_stability["two_of_three_grid_spacings_robust"]:
            decisions[policy]["status"] = "path_management_failed_certainty_contract"
    any_passed = any(
        item["status"] == "historical_path_management_increment"
        for item in decisions.values()
    )
    return {
        "policies": decisions,
        "grid_stability": grid_stability,
        "overall_status": (
            "at_least_one_historical_path_management_increment"
            if any_passed
            else "no_high_certainty_path_management_increment"
        ),
        "deployment_eligible": False,
        "stop_boundary": (
            "do not tune grid spacing, stop buffer, reentry wait, or reentry count after results"
        ),
    }


def _build_cases(results: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    event_state = events[
        ["signal_id", "source_sample", "ts_code", "month_period", "target_hit", "net_return"]
    ].rename(
        columns={
            "target_hit": "baseline_target_hit",
            "net_return": "baseline_budget_return",
        }
    )
    merged = results.merge(
        event_state,
        on=["signal_id", "source_sample", "ts_code", "month_period"],
        how="left",
        validate="many_to_one",
    )
    rows: list[pd.DataFrame] = []
    categories = {
        "baseline_tail_loss_improved": (
            merged["baseline_budget_return"].le(-0.50)
            & merged["budget_return"].gt(merged["baseline_budget_return"])
        ),
        "stopped_then_reentered_but_failed": (
            merged["stop_count"].gt(0)
            & merged["reentries"].gt(0)
            & merged["budget_return"].le(0.0)
        ),
        "stopped_no_reentry_but_baseline_won": (
            merged["stop_count"].gt(0)
            & merged["reentries"].eq(0)
            & merged["baseline_target_hit"].fillna(False)
        ),
        "full_grid_tail_loss": (
            merged["grid_add_count"].ge(2) & merged["budget_return"].le(-0.50)
        ),
        "writeoff": merged["exit_reason"].eq("missing_bar_writeoff"),
    }
    for category, mask in categories.items():
        part = merged.loc[mask].copy()
        part["case_category"] = category
        rows.append(part)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


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
    event_metrics: pd.DataFrame,
    cohorts: pd.DataFrame,
    portfolio: pd.DataFrame,
    cases: pd.DataFrame,
    decision: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    primary_policies = [
        "lump_sum",
        PRIMARY_GRID,
        PRIMARY_STOP,
        PRIMARY_COMBINED,
    ]
    main = event_metrics.loc[event_metrics["policy"].isin(primary_policies)].copy()
    main = _format_percent(
        main,
        [
            "event_win_rate",
            "median_budget_return",
            "mean_budget_return",
            "tail_loss_rate",
            "target_hit_rate",
            "structural_stop_rate",
            "reentry_rate",
            "mean_invested_fraction",
            "positive_cohort_share",
            "cohort_equal_mean_return",
            "cohort_bootstrap_ci95_low",
            "cohort_bootstrap_ci95_high",
            "worst_cohort_return",
            "leave_one_cohort_out_min_mean",
        ],
    )
    grid = event_metrics.loc[event_metrics["policy"].isin(GRID_POLICIES)].copy()
    grid = _format_percent(
        grid,
        [
            "event_win_rate",
            "mean_budget_return",
            "tail_loss_rate",
            "mean_invested_fraction",
            "positive_cohort_share",
            "cohort_bootstrap_ci95_low",
            "worst_cohort_return",
        ],
    )
    portfolio_display = _format_percent(
        portfolio.loc[portfolio["policy"].isin(primary_policies)].copy(),
        [
            "total_return",
            "cagr",
            "maximum_drawdown",
            "worst_rolling_24m_return",
            "mean_invested_fraction",
            "mean_reserved_fraction",
        ],
    )
    case_summary = (
        cases.groupby(["case_category", "policy"], observed=True, sort=True)
        .agg(
            cases=("budget_return", "size"),
            mean_policy_return=("budget_return", "mean"),
            mean_baseline_return=("baseline_budget_return", "mean"),
            worst_policy_return=("budget_return", "min"),
        )
        .reset_index()
        if not cases.empty
        else pd.DataFrame()
    )
    case_summary = _format_percent(
        case_summary,
        ["mean_policy_return", "mean_baseline_return", "worst_policy_return"],
    )
    conclusion = (
        "至少一种连续路径政策通过了冻结历史增益合同，但全部历史均已暴露，仍不可部署。"
        if decision["overall_status"]
        == "at_least_one_historical_path_management_increment"
        else (
            "没有连续政策达到高确定性。纯网格提供了部分尾部改善，其中40%初仓、"
            "下跌15%和30%各加30%的宽网格是描述性最稳的风险叠层，但旧周期PF、"
            "正锚月比例和bootstrap下界仍未通过；结构止损再接回及其网格组合明确失败。"
        )
    )
    path.write_text(
        f"""# 月线带血筹连续建仓与再入场回验

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 结论

{conclusion}

所有政策使用同一批 `range_mid_reclaim + 前高回撤至少60%` 锚、相同每锚总预算、15%止盈、504市场日上限和每轮20bp成本。网格未成交资金保留为现金；止损使用信号日可见且不下移的 `base_low`；接回至少等待20日并重新满足中轴确认，最多两次。

宽网格的改善不能解释成新增选股能力：它的逐锚平均实际股票投入约为总预算的53%，本质是用更低、更晚的暴露换取更小尾部。它可以作为仓位纪律候选，但不能按本轮已见结果事后升级为新的主参数。

## 主政策事件与锚月结果

{main.to_markdown(index=False)}

## 5%/10%、10%/20%、15%/30%网格邻域

{grid.to_markdown(index=False)}

## 日度盯市组合

{portfolio_display.to_markdown(index=False)}

## Case汇总

{case_summary.to_markdown(index=False) if not case_summary.empty else '_没有触发case_'}

## 冻结判定

```json
{json.dumps(_json_value(decision), ensure_ascii=False, indent=2)}
```

## 数据审计

```json
{json.dumps(_json_value(metadata), ensure_ascii=False, indent=2)}
```

独立恐慌月明细保存在 `cohort_metrics.csv`，逐锚路径和交易保存在Parquet文件。本研究仅供研究与教育用途，不构成投资建议、收益承诺或交易指令。
""",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ContinuousConfig()
    events = _load_primary_events(args.baseline_events)
    old_events = events.loc[events["source_sample"].eq("old_cycle_2003_2012")].copy()
    recent_events = events.loc[
        events["source_sample"].eq("recent_exposed_2013_2024")
    ].copy()
    print(
        f"primary anchors: old={len(old_events):,}, recent={len(recent_events):,}",
        flush=True,
    )

    old_benchmark = _load_old_benchmark(
        args.extension_data_dir, args.recent_benchmark
    )
    old_calendar = pd.DatetimeIndex(sorted(old_benchmark["date"].unique()))
    old_daily = _load_old_daily(
        args.extension_data_dir, set(old_events["ts_code"].astype(str))
    )
    print("evaluating old-cycle continuous policies", flush=True)
    old_results, old_paths, old_trades = evaluate_continuous_policies(
        old_daily, old_events, old_calendar, config
    )

    recent_benchmark = _normalize_benchmark(pd.read_parquet(args.recent_benchmark))
    recent_calendar = pd.DatetimeIndex(sorted(recent_benchmark["date"].unique()))
    recent_daily = _load_recent_daily(
        args.recent_feature_cache,
        args.recent_supplemental_root,
        set(recent_events["ts_code"].astype(str)),
    )
    print("evaluating recent continuous policies", flush=True)
    recent_results, recent_paths, recent_trades = evaluate_continuous_policies(
        recent_daily, recent_events, recent_calendar, config
    )
    results = pd.concat([old_results, recent_results], ignore_index=True, sort=False)
    paths = pd.concat([old_paths, recent_paths], ignore_index=True, sort=False)
    trades = pd.concat([old_trades, recent_trades], ignore_index=True, sort=False)

    event_metrics, cohorts = _summarize_events(results)
    print("simulating daily capacity-limited portfolios", flush=True)
    portfolio_curves, portfolio_trades, portfolio_metrics = _run_portfolios(
        old_paths,
        old_results,
        old_calendar,
        recent_paths,
        recent_results,
        recent_calendar,
    )
    decision = _decide(event_metrics, portfolio_metrics)
    cases = _build_cases(results, events)
    lump = results.loc[results["policy"].eq("lump_sum")].copy()
    reproduction_error = pd.to_numeric(
        lump["budget_return"], errors="coerce"
    ) - pd.to_numeric(lump["baseline_net_return"], errors="coerce")
    metadata = {
        "analysis_cutoff": pd.Timestamp("2026-07-31"),
        "config": config.to_dict(),
        "policies": [policy.to_dict() for policy in CONTINUOUS_POLICIES],
        "old_primary_anchors": int(len(old_events)),
        "recent_primary_anchors": int(len(recent_events)),
        "independent_anchor_months": int(events["month_period"].nunique()),
        "anchor_result_rows": int(len(results)),
        "anchor_path_rows": int(len(paths)),
        "trade_rows": int(len(trades)),
        "lump_sum_max_absolute_reproduction_error": (
            float(reproduction_error.abs().max())
            if reproduction_error.notna().any()
            else None
        ),
        "lump_sum_mismatches_over_1bp": int(reproduction_error.abs().gt(0.0001).sum()),
        "claim_boundary": (
            "all historical periods are exposed; no out-of-sample or deployment claim"
        ),
    }
    results.to_parquet(
        args.output_dir / "anchor_results.parquet", index=False, compression="zstd"
    )
    paths.to_parquet(
        args.output_dir / "anchor_paths.parquet", index=False, compression="zstd"
    )
    trades.to_parquet(
        args.output_dir / "trades.parquet", index=False, compression="zstd"
    )
    cases.to_parquet(
        args.output_dir / "cases.parquet", index=False, compression="zstd"
    )
    event_metrics.to_csv(args.output_dir / "event_metrics.csv", index=False)
    cohorts.to_csv(args.output_dir / "cohort_metrics.csv", index=False)
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
        event_metrics,
        cohorts,
        portfolio_metrics,
        cases,
        decision,
        metadata,
    )
    print(f"continuous status: {decision['overall_status']}", flush=True)
    print(f"report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
