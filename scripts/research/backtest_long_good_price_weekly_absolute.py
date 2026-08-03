"""Explore weekly historical percentiles and absolute good-price thresholds.

This is an exploratory extension of ``backtest_long_good_stock_price.py``.
It deliberately leaves the production long-pool rule unchanged until a rule
survives both the 2020-2023 validation period and the sealed 2024+ test period.

The quality gate is evaluated on month-end point-in-time data and carried
forward to weekly observations. That keeps the established three-year
business-stability window intact instead of accidentally treating 36 weeks as
36 months. Weekly PE/PB and price inputs use the last available trading day in
each ``W-FRI`` period.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest_long_dividend_quality import (
    filter_daily_basic_point_in_time,
    load_daily_basic_monthly,
    load_daily_monthly_features,
)
from backtest_long_good_stock_price import attach_forward_returns, period_label
from backtest_tea_master_long import (
    PROJECT_ROOT,
    add_historical_valuation_features,
    build_tea_scores,
    load_benchmark,
    prepare_data,
)


REPORT_DIR = PROJECT_ROOT / "reports/long_good_price_weekly_absolute"
HORIZONS = {"3m": 63, "6m": 126, "12m": 252, "24m": 504, "36m": 756}
WEEKS_PER_YEAR = 52
WINDOWS_YEARS = [
    (1, 3),
    (1, 5),
    (2, 5),
    (2, 7),
    (3, 5),
    (3, 7),
    (3, 10),
    (5, 10),
]


@dataclass(frozen=True)
class ResearchRule:
    name: str
    family: str
    description: str
    eligible_for_selection: bool = True


ABSOLUTE_RULES = [
    ResearchRule("all_good_stocks", "baseline", "好股票基线", False),
    ResearchRule("abs_pr_lt_075", "absolute", "类型加权双PR < 0.75"),
    ResearchRule("abs_pr_lt_1", "absolute", "类型加权双PR < 1"),
    ResearchRule("abs_pr_lt_15", "absolute", "类型加权双PR < 1.5"),
    ResearchRule("abs_pr_lt_2", "absolute", "类型加权双PR < 2"),
    ResearchRule("abs_profile_pr_lt_1", "absolute_profile", "按股票类型选择的PR < 1"),
    ResearchRule("abs_profile_pr_lt_15", "absolute_profile", "按股票类型选择的PR < 1.5"),
    ResearchRule("abs_both_pr_lt_1", "absolute", "PR-PE与PR-PB均 < 1"),
    ResearchRule("abs_both_pr_lt_15", "absolute", "PR-PE与PR-PB均 < 1.5"),
    ResearchRule("abs_pe15_pb15", "absolute", "PE < 15且PB < 1.5"),
    ResearchRule("abs_pe20_pb2", "absolute", "PE < 20且PB < 2"),
    ResearchRule("abs_pr_lt_1_guard", "absolute", "类型加权双PR < 1且长期结构未破坏"),
    ResearchRule("band_pr_lt_1", "absolute_band", "类型加权双PR < 1", False),
    ResearchRule("band_pr_1_to_2", "absolute_band", "类型加权双PR在[1,2]", False),
    ResearchRule("band_pr_gt_2", "absolute_band", "类型加权双PR > 2", False),
    ResearchRule("band_profile_pr_lt_1", "absolute_profile_band", "类型适配PR < 1", False),
    ResearchRule("band_profile_pr_1_to_2", "absolute_profile_band", "类型适配PR在[1,2]", False),
    ResearchRule("band_profile_pr_gt_2", "absolute_profile_band", "类型适配PR > 2", False),
]


HISTORICAL_RULES = [
    ResearchRule("hist_composite55", "historical", "历史价值分 >= 55"),
    ResearchRule("hist_composite60", "historical", "历史价值分 >= 60"),
    ResearchRule("hist_composite65", "historical", "历史价值分 >= 65"),
    ResearchRule("hist_triple_p50", "historical", "PE/PB/双PR历史分位均 <= 50%"),
    ResearchRule("hist_triple_p40", "historical", "PE/PB/双PR历史分位均 <= 40%"),
    ResearchRule("hist_composite60_guard", "historical", "历史价值分 >= 60且长期结构未破坏"),
    ResearchRule("hybrid_hist60_abs1", "hybrid", "历史价值分 >= 60且类型加权双PR < 1"),
    ResearchRule("hybrid_hist60_abs15", "hybrid", "历史价值分 >= 60且类型加权双PR < 1.5"),
    ResearchRule("hybrid_hist60_abs2", "hybrid", "历史价值分 >= 60且类型加权双PR < 2"),
    ResearchRule("hybrid_hist60_guard_abs2", "hybrid", "历史价值分 >= 60、双PR < 2且长期结构未破坏"),
    ResearchRule("hybrid_hist60_or_abs1", "hybrid", "历史价值分 >= 60或类型加权双PR < 1"),
]


RULE_LOOKUP = {rule.name: rule for rule in ABSOLUTE_RULES + HISTORICAL_RULES}


def trend_guard(frame: pd.DataFrame) -> pd.Series:
    return (
        (pd.to_numeric(frame["close"], errors="coerce") >= pd.to_numeric(frame["ma_120"], errors="coerce") * 0.90)
        & (pd.to_numeric(frame["ma_120_slope_20d"], errors="coerce") >= -0.06)
    )


def profile_adapted_pr(frame: pd.DataFrame) -> pd.Series:
    """Choose the economically relevant PR while preserving both raw values."""
    profile = frame["valuation_profile"].astype(str)
    weighted = pd.to_numeric(frame["pr"], errors="coerce")
    pr_pe = pd.to_numeric(frame["pr_pe"], errors="coerce")
    pr_pb = pd.to_numeric(frame["pr_pb"], errors="coerce")
    return weighted.where(
        profile.eq("balanced"),
        pr_pe.where(profile.eq("earnings_based"), pr_pb),
    )


def absolute_rule_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    pe = pd.to_numeric(frame["pe_ttm"], errors="coerce")
    pb = pd.to_numeric(frame["pb"], errors="coerce")
    pr = pd.to_numeric(frame["pr"], errors="coerce")
    pr_pe = pd.to_numeric(frame["pr_pe"], errors="coerce")
    pr_pb = pd.to_numeric(frame["pr_pb"], errors="coerce")
    profile_pr = profile_adapted_pr(frame)
    guard = trend_guard(frame)
    valid_pr = pr.gt(0)
    valid_profile_pr = profile_pr.gt(0)
    return {
        "all_good_stocks": pd.Series(True, index=frame.index),
        "abs_pr_lt_075": valid_pr & pr.lt(0.75),
        "abs_pr_lt_1": valid_pr & pr.lt(1.0),
        "abs_pr_lt_15": valid_pr & pr.lt(1.5),
        "abs_pr_lt_2": valid_pr & pr.lt(2.0),
        "abs_profile_pr_lt_1": valid_profile_pr & profile_pr.lt(1.0),
        "abs_profile_pr_lt_15": valid_profile_pr & profile_pr.lt(1.5),
        "abs_both_pr_lt_1": pr_pe.gt(0) & pr_pe.lt(1.0) & pr_pb.gt(0) & pr_pb.lt(1.0),
        "abs_both_pr_lt_15": pr_pe.gt(0) & pr_pe.lt(1.5) & pr_pb.gt(0) & pr_pb.lt(1.5),
        "abs_pe15_pb15": pe.gt(0) & pe.lt(15.0) & pb.gt(0) & pb.lt(1.5),
        "abs_pe20_pb2": pe.gt(0) & pe.lt(20.0) & pb.gt(0) & pb.lt(2.0),
        "abs_pr_lt_1_guard": valid_pr & pr.lt(1.0) & guard,
        "band_pr_lt_1": valid_pr & pr.lt(1.0),
        "band_pr_1_to_2": valid_pr & pr.ge(1.0) & pr.le(2.0),
        "band_pr_gt_2": valid_pr & pr.gt(2.0),
        "band_profile_pr_lt_1": valid_profile_pr & profile_pr.lt(1.0),
        "band_profile_pr_1_to_2": valid_profile_pr & profile_pr.ge(1.0) & profile_pr.le(2.0),
        "band_profile_pr_gt_2": valid_profile_pr & profile_pr.gt(2.0),
    }


def historical_rule_masks(frame: pd.DataFrame, minimum_weeks: int) -> dict[str, pd.Series]:
    has_history = pd.to_numeric(frame["valuation_history_points"], errors="coerce").ge(minimum_weeks)
    pe = pd.to_numeric(frame["pe_hist_percentile"], errors="coerce")
    pb = pd.to_numeric(frame["pb_hist_percentile"], errors="coerce")
    pr_percentile = pd.to_numeric(frame["pr_hist_percentile"], errors="coerce")
    score = pd.to_numeric(frame["historical_value_score"], errors="coerce")
    pr = pd.to_numeric(frame["pr"], errors="coerce")
    guard = trend_guard(frame)
    hist55 = has_history & score.ge(55)
    hist60 = has_history & score.ge(60)
    hist65 = has_history & score.ge(65)
    return {
        "hist_composite55": hist55,
        "hist_composite60": hist60,
        "hist_composite65": hist65,
        "hist_triple_p50": has_history & pe.le(50) & pb.le(50) & pr_percentile.le(50),
        "hist_triple_p40": has_history & pe.le(40) & pb.le(40) & pr_percentile.le(40),
        "hist_composite60_guard": hist60 & guard,
        "hybrid_hist60_abs1": hist60 & pr.gt(0) & pr.lt(1.0),
        "hybrid_hist60_abs15": hist60 & pr.gt(0) & pr.lt(1.5),
        "hybrid_hist60_abs2": hist60 & pr.gt(0) & pr.lt(2.0),
        "hybrid_hist60_guard_abs2": hist60 & pr.gt(0) & pr.lt(2.0) & guard,
        "hybrid_hist60_or_abs1": hist60 | (pr.gt(0) & pr.lt(1.0)),
    }


def merge_monthly_quality_asof(
    weekly: pd.DataFrame,
    monthly_scored: pd.DataFrame,
) -> pd.DataFrame:
    quality_columns = [
        "date",
        "ts_code",
        "is_good_stock",
        "good_stock_score",
        "profitability_score",
        "fundamental_growth_score",
        "balance_sheet_score",
        "business_stability_score",
        "good_stock_data_coverage",
        "roe",
        "netprofit_margin",
        "or_yoy",
        "basic_eps_yoy",
        "debt_to_assets",
        "listing_years",
    ]
    available = [column for column in quality_columns if column in monthly_scored.columns]
    right = monthly_scored[available].copy()
    left = weekly.drop(columns=[column for column in available if column not in {"date", "ts_code"}], errors="ignore").copy()
    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])
    merged_parts: list[pd.DataFrame] = []
    quality_by_symbol = {
        str(code): group.sort_values("date").drop(columns=["ts_code"])
        for code, group in right.groupby("ts_code", sort=False)
    }
    for code, group in left.groupby("ts_code", sort=False):
        quality = quality_by_symbol.get(str(code))
        if quality is None or quality.empty:
            continue
        merged = pd.merge_asof(
            group.sort_values("date"),
            quality,
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["ts_code"] = code
        merged_parts.append(merged)
    if not merged_parts:
        return pd.DataFrame()
    return pd.concat(merged_parts, ignore_index=True).sort_values(["date", "ts_code"]).reset_index(drop=True)


def prepare_weekly_good_stocks(
    start: str,
    end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    monthly, daily_returns, stock_basic, monthly_coverage = prepare_data(start, end)
    monthly_scored = build_tea_scores(
        monthly,
        valuation_window_months=84,
        valuation_minimum_months=24,
    )
    requested_start = pd.to_datetime(start, format="%Y%m%d")
    requested_end = pd.to_datetime(end, format="%Y%m%d") if end else None
    weekly_basic, weekly_basic_coverage = load_daily_basic_monthly(
        requested_start,
        requested_end,
        sampling="weekly",
    )
    weekly_price, _ = load_daily_monthly_features(
        requested_start,
        requested_end,
        stock_basic,
        candidate_symbols=None,
        include_daily_returns=False,
        sampling="weekly",
    )
    universe_config = SimpleNamespace(
        variant="tea",
        prefilter_min_dv_ttm=0.0,
        prefilter_min_total_mv=800000.0,
        prefilter_min_circ_mv=500000.0,
    )
    weekly_basic = filter_daily_basic_point_in_time(weekly_basic, universe_config)
    weekly = weekly_price.merge(
        weekly_basic.drop(columns=["trade_date"], errors="ignore"),
        on=["date", "ts_code"],
        how="inner",
    )
    weekly = merge_monthly_quality_asof(weekly, monthly_scored)
    weekly = weekly[weekly["is_good_stock"].fillna(False)].copy()
    coverage = {
        "monthly": monthly_coverage,
        "weekly_daily_basic": weekly_basic_coverage,
        "weekly_rows": int(len(weekly)),
        "weekly_dates": int(weekly["date"].nunique()),
        "weekly_symbols": int(weekly["ts_code"].nunique()),
        "quality_carry_forward": "latest_month_end_at_or_before_week_end",
    }
    benchmark = load_benchmark(requested_start, requested_end)
    return weekly, daily_returns, benchmark, coverage


def baseline_by_date(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        horizon: frame.groupby("date")[f"return_{horizon}"].mean()
        for horizon in HORIZONS
    }


def summarize_rule(
    frame: pd.DataFrame,
    *,
    rule: ResearchRule,
    minimum_years: int,
    maximum_years: int,
    baseline: dict[str, pd.Series],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    if frame.empty:
        return summary_rows, yearly_rows
    for period, period_frame in frame.groupby("period", sort=False):
        for horizon in HORIZONS:
            columns = ["date", "ts_code", f"return_{horizon}", f"excess_return_{horizon}", f"mae_{horizon}"]
            observations = period_frame[columns].copy()
            for column in columns[2:]:
                observations[column] = pd.to_numeric(observations[column], errors="coerce")
            observations = observations.dropna(subset=[f"return_{horizon}"])
            if observations.empty:
                continue
            names_per_week = observations.groupby("date")["ts_code"].nunique()
            weekly = observations.groupby("date").agg(
                portfolio_return=(f"return_{horizon}", "mean"),
                portfolio_excess=(f"excess_return_{horizon}", "mean"),
                portfolio_mae=(f"mae_{horizon}", "mean"),
            )
            weekly["baseline_return"] = baseline[horizon].reindex(weekly.index)
            weekly["baseline_delta"] = weekly["portfolio_return"] - weekly["baseline_return"]
            summary_rows.append(
                {
                    "rule": rule.name,
                    "family": rule.family,
                    "description": rule.description,
                    "eligible_for_selection": rule.eligible_for_selection,
                    "sampling": "weekly_last_trading_day",
                    "minimum_history_years": minimum_years,
                    "maximum_history_years": maximum_years,
                    "period": period,
                    "horizon": horizon,
                    "signals": int(len(observations)),
                    "weeks": int(len(weekly)),
                    "avg_names_per_week": float(names_per_week.mean()),
                    "mean_return": float(weekly["portfolio_return"].mean()),
                    "median_return": float(weekly["portfolio_return"].median()),
                    "positive_rate": float((weekly["portfolio_return"] > 0).mean()),
                    "p10_return": float(weekly["portfolio_return"].quantile(0.10)),
                    "mean_excess": float(weekly["portfolio_excess"].mean()),
                    "median_excess": float(weekly["portfolio_excess"].median()),
                    "baseline_delta": float(weekly["baseline_delta"].mean()),
                    "mean_mae": float(weekly["portfolio_mae"].mean()),
                    "p10_mae": float(weekly["portfolio_mae"].quantile(0.10)),
                }
            )
            annual = weekly.groupby(weekly.index.year).agg(
                mean_return=("portfolio_return", "mean"),
                mean_excess=("portfolio_excess", "mean"),
                baseline_delta=("baseline_delta", "mean"),
                weeks=("portfolio_return", "size"),
            )
            for year, values in annual.iterrows():
                yearly_rows.append(
                    {
                        "rule": rule.name,
                        "family": rule.family,
                        "minimum_history_years": minimum_years,
                        "maximum_history_years": maximum_years,
                        "period": period,
                        "horizon": horizon,
                        "year": int(year),
                        "weeks": int(values["weeks"]),
                        "mean_return": float(values["mean_return"]),
                        "mean_excess": float(values["mean_excess"]),
                        "baseline_delta": float(values["baseline_delta"]),
                    }
                )
    return summary_rows, yearly_rows


def validation_selection(summary: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    candidates = summary[
        (summary["period"] == "validation_2020_2023")
        & (summary["horizon"] == "12m")
        & summary["eligible_for_selection"].fillna(False)
        & (summary["avg_names_per_week"] >= 3)
        & (summary["weeks"] >= 100)
    ].copy()
    validation_years = yearly[
        (yearly["period"] == "validation_2020_2023")
        & (yearly["horizon"] == "12m")
    ].copy()
    annual = validation_years.groupby(
        ["rule", "minimum_history_years", "maximum_history_years"],
        as_index=False,
    ).agg(
        validation_years=("year", "nunique"),
        positive_excess_years=("mean_excess", lambda values: int((values > 0).sum())),
        positive_baseline_years=("baseline_delta", lambda values: int((values > 0).sum())),
        worst_year_excess=("mean_excess", "min"),
        worst_year_baseline_delta=("baseline_delta", "min"),
    )
    candidates = candidates.merge(
        annual,
        on=["rule", "minimum_history_years", "maximum_history_years"],
        how="left",
    )
    candidates = candidates[candidates["validation_years"].fillna(0) >= 4].copy()
    rank_metrics = {
        "mean_excess": 0.20,
        "median_excess": 0.10,
        "baseline_delta": 0.25,
        "positive_rate": 0.10,
        "p10_return": 0.15,
        "worst_year_excess": 0.10,
        "worst_year_baseline_delta": 0.10,
    }
    candidates["selection_score"] = 0.0
    for metric, weight in rank_metrics.items():
        candidates["selection_score"] += candidates[metric].rank(pct=True) * weight
    return candidates.sort_values(
        ["selection_score", "baseline_delta", "mean_excess"],
        ascending=False,
    ).reset_index(drop=True)


def pct(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(number) else f"{float(number):.2%}"


def build_report(
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    yearly: pd.DataFrame,
    coverage: dict,
    gap_summary: pd.DataFrame,
) -> str:
    selected = selection.iloc[0] if not selection.empty else None
    selected_rule = str(selected["rule"]) if selected is not None else "none"
    selected_min = int(selected["minimum_history_years"]) if selected is not None else 0
    selected_max = int(selected["maximum_history_years"]) if selected is not None else 0
    selected_rows = summary[
        (summary["rule"] == selected_rule)
        & (summary["minimum_history_years"] == selected_min)
        & (summary["maximum_history_years"] == selected_max)
    ].copy()
    validation_12m = selected_rows[
        (selected_rows["period"] == "validation_2020_2023")
        & (selected_rows["horizon"] == "12m")
    ]
    test_12m = selected_rows[
        (selected_rows["period"] == "test_2024_plus")
        & (selected_rows["horizon"] == "12m")
    ]
    validation_ok = bool(
        not validation_12m.empty
        and validation_12m.iloc[0]["mean_excess"] > 0
        and validation_12m.iloc[0]["baseline_delta"] > 0
    )
    test_ok = bool(
        not test_12m.empty
        and test_12m.iloc[0]["mean_excess"] > 0
        and test_12m.iloc[0]["baseline_delta"] > 0
    )
    promotion = "可进入下一轮独立验证" if validation_ok and test_ok else "不升级生产规则"
    validation_candidates = summary[
        (summary["period"] == "validation_2020_2023")
        & (summary["horizon"] == "12m")
        & summary["eligible_for_selection"].fillna(False)
    ]
    test_candidates = summary[
        (summary["period"] == "test_2024_plus")
        & (summary["horizon"] == "12m")
    ]
    paired_candidates = validation_candidates.merge(
        test_candidates,
        on=["rule", "minimum_history_years", "maximum_history_years"],
        suffixes=("_validation", "_test"),
    )
    both_pass = paired_candidates[
        (paired_candidates["mean_excess_validation"] > 0)
        & (paired_candidates["baseline_delta_validation"] > 0)
        & (paired_candidates["mean_excess_test"] > 0)
        & (paired_candidates["baseline_delta_test"] > 0)
    ]
    large_gap = gap_summary[
        gap_summary["period"].isin(["validation_2020_2023", "test_2024_plus"])
    ]["large_formula_gap_rate"]
    gap_range = (
        f"{large_gap.min():.1%}–{large_gap.max():.1%}"
        if not large_gap.empty
        else "-"
    )
    lines = [
        "# 长线好价格：周频分位与绝对阈值探索",
        "",
        "## 结论",
        "",
        f"- 验证集预选：`{selected_rule}`，最少 {selected_min} 年、最多 {selected_max} 年历史。",
        f"- 生产建议：**{promotion}**。2024+只用于本次封存样本外检查，观察后不可再次作为无污染调参集。",
        f"- 双重通过数：{len(both_pass)}。这里要求12个月收益在验证与测试两段都同时跑赢沪深300和好股票基线。",
        "- 参数研究建议：周频分位先以最少2年、最多5年为默认观察口径，7年仅作稳健性对照；2/5与2/7结果接近，增加到7年没有稳定改善。",
        "- 绝对阈值建议：PR<1只能标注“绝对低PR”，不能直接等同推荐；PR>2只能标注“高PR风险”，不能硬性否决。PE/PB绝对阈值更像防守性安全垫，但仍未在2024+跑赢基准。",
        "- 周频采用每个W-FRI周期最后交易日；好股票资格沿用当时最近的月末点时结果，避免把36个月经营稳定误算成36周。",
        "- 所有收益从下一市场交易日收盘开始，使用共同沪深300交易日历；结果先按信号周等权，再跨周统计。",
        "- 本轮同时测试绝对PR、绝对PE/PB、历史分位及二者组合。绝对PR采用行业类型加权后的双PR，两个原始PR仍保留。",
        f"- 双PR公式差距超过50%的比例在主要类型/阶段中为 {gap_range}，不宜只保留一个合成值。",
        "",
        "## 预选规则分阶段结果",
        "",
        "| 阶段 | 周期 | 周数 | 周均标的 | 平均收益 | 对沪深300 | 较好股基线 | P10收益 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.horizon} | {row.weeks} | {row.avg_names_per_week:.1f} | "
            f"{pct(row.mean_return)} | {pct(row.mean_excess)} | {pct(row.baseline_delta)} | {pct(row.p10_return)} |"
        )
    lines.extend(
        [
            "",
            "## 验证集排名前十",
            "",
            "| 排名 | 规则 | 家族 | 最少/最多历史 | 周均标的 | 收益 | 超额 | 较基线 | 正收益率 | 最差年度超额 |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in selection.head(10).iterrows():
        window = "不适用" if int(row["maximum_history_years"]) == 0 else f"{int(row['minimum_history_years'])}/{int(row['maximum_history_years'])}年"
        lines.append(
            f"| {index + 1} | {row['rule']} | {row['family']} | {window} | {row['avg_names_per_week']:.1f} | "
            f"{pct(row['mean_return'])} | {pct(row['mean_excess'])} | {pct(row['baseline_delta'])} | "
            f"{pct(row['positive_rate'])} | {pct(row['worst_year_excess'])} |"
        )
    absolute_bands = summary[
        summary["rule"].isin(
            [
                "band_pr_lt_1",
                "band_pr_1_to_2",
                "band_pr_gt_2",
                "band_profile_pr_lt_1",
                "band_profile_pr_1_to_2",
                "band_profile_pr_gt_2",
            ]
        )
        & summary["period"].isin(["validation_2020_2023", "test_2024_plus"])
        & (summary["horizon"] == "12m")
    ]
    lines.extend(
        [
            "",
            "## PR绝对区间诊断",
            "",
            "| 区间 | 阶段 | 周均标的 | 平均收益 | 对沪深300 | 较好股基线 | P10收益 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in absolute_bands.itertuples(index=False):
        lines.append(
            f"| {row.description} | {row.period} | {row.avg_names_per_week:.1f} | {pct(row.mean_return)} | "
            f"{pct(row.mean_excess)} | {pct(row.baseline_delta)} | {pct(row.p10_return)} |"
        )
    absolute_safety = summary[
        summary["rule"].isin(
            [
                "abs_pr_lt_1",
                "abs_profile_pr_lt_1",
                "abs_both_pr_lt_1",
                "abs_pe15_pb15",
                "abs_pe20_pb2",
            ]
        )
        & summary["period"].isin(["validation_2020_2023", "test_2024_plus"])
        & (summary["horizon"] == "12m")
    ]
    lines.extend(
        [
            "",
            "## 绝对阈值安全性对比",
            "",
            "| 规则 | 阶段 | 周均标的 | 平均收益 | 对沪深300 | 较好股基线 | 平均MAE | P10收益 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in absolute_safety.itertuples(index=False):
        lines.append(
            f"| {row.description} | {row.period} | {row.avg_names_per_week:.1f} | {pct(row.mean_return)} | "
            f"{pct(row.mean_excess)} | {pct(row.baseline_delta)} | {pct(row.mean_mae)} | {pct(row.p10_return)} |"
        )
    window_rows = summary[
        summary["rule"].isin(["hist_composite60", "hist_composite60_guard", "hybrid_hist60_abs2"])
        & (summary["period"] == "validation_2020_2023")
        & (summary["horizon"] == "12m")
    ]
    lines.extend(
        [
            "",
            "## 最少N年 / 最多M年敏感性",
            "",
            "| 规则 | N/M | 周均标的 | 收益 | 超额 | 较基线 | P10收益 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in window_rows.sort_values(["rule", "minimum_history_years", "maximum_history_years"]).itertuples(index=False):
        lines.append(
            f"| {row.rule} | {row.minimum_history_years}/{row.maximum_history_years}年 | {row.avg_names_per_week:.1f} | "
            f"{pct(row.mean_return)} | {pct(row.mean_excess)} | {pct(row.baseline_delta)} | {pct(row.p10_return)} |"
        )
    lines.extend(
        [
            "",
            "## 数据与限制",
            "",
            f"- 周截面数：{coverage.get('weekly_dates')}；好股票周信号行：{coverage.get('weekly_rows')}；股票数：{coverage.get('weekly_symbols')}。",
            f"- 日估值源起止：{coverage.get('weekly_daily_basic', {}).get('first_trade_date')} 至 {coverage.get('weekly_daily_basic', {}).get('last_trade_date')}。",
            "- 周频12/24/36个月信号高度重叠；周数不是独立样本数，本报告不把普通t统计显著性当作证据。",
            "- 绝对PR阈值是跨行业启发式。即使使用类型权重，也不能替代银行P/B-ROE、周期股中周期盈利或亏损成长公司的专用估值。",
            "- 本轮候选较多，存在多重比较风险；即使样本外同时为正，也只进入下一轮独立验证，不直接上线。",
            "- 周频相对日频把每年约252个估值点降到约52个，但相对现有月频约12个点仍增加约4.3倍计算量；它适合提高分位精度，不是最低成本方案。",
            "- 股票基础表的退市历史覆盖可能不完整，仍有幸存者偏差风险。",
            "- 本研究只评价好股票池中的价格筛选，不是单股目标价，也不构成交易建议。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(start: str, end: str | None) -> dict[str, object]:
    weekly_good, daily_returns, benchmark, coverage = prepare_weekly_good_stocks(start, end)
    return_keys = weekly_good[["date", "ts_code"]].drop_duplicates()
    forward = attach_forward_returns(
        return_keys,
        daily_returns,
        benchmark,
        horizons=HORIZONS,
    )
    forward["period"] = forward["date"].map(period_label)
    baseline = baseline_by_date(forward)
    all_summary: list[dict[str, object]] = []
    all_yearly: list[dict[str, object]] = []

    absolute_scored: pd.DataFrame | None = None
    for minimum_years, maximum_years in WINDOWS_YEARS:
        print(
            f"weekly percentiles: min={minimum_years}y max={maximum_years}y",
            flush=True,
        )
        scored = add_historical_valuation_features(
            weekly_good,
            window_months=maximum_years * WEEKS_PER_YEAR,
            minimum_months=minimum_years * WEEKS_PER_YEAR,
        )
        scored = scored.merge(forward, on=["date", "ts_code"], how="left", suffixes=("", "_return"))
        if absolute_scored is None:
            absolute_scored = scored
            masks = absolute_rule_masks(scored)
            for rule in ABSOLUTE_RULES:
                rows, yearly_rows = summarize_rule(
                    scored.loc[masks[rule.name]],
                    rule=rule,
                    minimum_years=0,
                    maximum_years=0,
                    baseline=baseline,
                )
                all_summary.extend(rows)
                all_yearly.extend(yearly_rows)
        masks = historical_rule_masks(scored, minimum_years * WEEKS_PER_YEAR)
        for rule in HISTORICAL_RULES:
            rows, yearly_rows = summarize_rule(
                scored.loc[masks[rule.name]],
                rule=rule,
                minimum_years=minimum_years,
                maximum_years=maximum_years,
                baseline=baseline,
            )
            all_summary.extend(rows)
            all_yearly.extend(yearly_rows)

    summary = pd.DataFrame(all_summary)
    yearly = pd.DataFrame(all_yearly)
    selection = validation_selection(summary, yearly)
    if absolute_scored is None:
        raise RuntimeError("No weekly valuation observations were scored")
    gap_source = absolute_scored.copy()
    gap_source["period"] = gap_source["date"].map(period_label)
    gap_source["large_formula_gap"] = pd.to_numeric(
        gap_source["pr_formula_gap"], errors="coerce"
    ).gt(0.50)
    gap_summary = (
        gap_source.groupby(["period", "valuation_profile"], as_index=False)
        .agg(
            observations=("ts_code", "size"),
            symbols=("ts_code", "nunique"),
            median_pr_pe=("pr_pe", "median"),
            median_pr_pb=("pr_pb", "median"),
            median_formula_gap=("pr_formula_gap", "median"),
            p75_formula_gap=("pr_formula_gap", lambda values: values.quantile(0.75)),
            large_formula_gap_rate=("large_formula_gap", "mean"),
        )
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_DIR / "summary.csv", index=False)
    yearly.to_csv(REPORT_DIR / "yearly.csv", index=False)
    selection.to_csv(REPORT_DIR / "validation_selection.csv", index=False)
    gap_summary.to_csv(REPORT_DIR / "pr_formula_gap_by_profile.csv", index=False)
    forward.to_parquet(REPORT_DIR / "weekly_good_stock_forward_returns.parquet", index=False)
    report = build_report(summary, selection, yearly, coverage, gap_summary)
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")

    selected_payload: dict[str, object] = {
        "sampling": "weekly_last_trading_day_W-FRI",
        "quality_gate": "latest_month_end_at_or_before_week_end",
        "selection_period": "2020-2023",
        "sealed_test_period": "2024+",
        "horizons": HORIZONS,
        "tested_windows_years": [list(item) for item in WINDOWS_YEARS],
        "absolute_rules": [asdict(item) for item in ABSOLUTE_RULES],
        "historical_rules": [asdict(item) for item in HISTORICAL_RULES],
    }
    if not selection.empty:
        selected = selection.iloc[0]
        selected_payload["selected"] = {
            "rule": str(selected["rule"]),
            "family": str(selected["family"]),
            "description": str(selected["description"]),
            "minimum_history_years": int(selected["minimum_history_years"]),
            "maximum_history_years": int(selected["maximum_history_years"]),
            "validation_selection_score": float(selected["selection_score"]),
        }
    (REPORT_DIR / "selected_rule.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "selected": selected_payload.get("selected"),
        "weekly_rows": coverage.get("weekly_rows"),
        "weekly_dates": coverage.get("weekly_dates"),
        "summary_rows": int(len(summary)),
        "selection_candidates": int(len(selection)),
        "report": str(REPORT_DIR / "report.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20130101")
    parser.add_argument("--end")
    args = parser.parse_args()
    print(json.dumps(run(args.start, args.end), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
