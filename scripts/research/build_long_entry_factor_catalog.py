#!/usr/bin/env python
"""Build the project-level factor inventory for the long-entry price model.

The inventory deliberately distinguishes factors that merely exist from factors
that are safe to train today.  It also records where a factor has already been
used so the long-entry research can reuse the project's prior model evidence
without silently inheriting a short-horizon target or a forward-looking field.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/long_entry_factor_inventory"


@dataclass(frozen=True)
class FactorRecord:
    factor: str
    group: str
    source: str
    frequency: str
    role: str
    status: str
    availability: str
    point_in_time_rule: str
    used_by: str = ""
    notes: str = ""


STATUS_ORDER = {
    "phase1_core": 0,
    "phase1_candidate": 1,
    "transform_required": 2,
    "phase2_research": 3,
    "shadow_only": 4,
    "benchmark_only": 5,
    "excluded": 6,
}


SELECTOR_PRODUCTION_FEATURES = [
    "selector_amplitude_1d",
    "selector_close_pos",
    "selector_excess_return_1d",
    "selector_gap_1d",
    "selector_high_1d",
    "selector_low_1d",
    "selector_market_dispersion_1d",
    "selector_market_down5_ratio_1d",
    "selector_market_mean_1d",
    "selector_market_mean_20d",
    "selector_market_mean_5d",
    "selector_market_median_1d",
    "selector_market_up5_ratio_1d",
    "selector_market_up_ratio_1d",
    "selector_market_up_ratio_20d",
    "selector_market_up_ratio_5d",
    "selector_positive_ratio_10d",
    "selector_positive_ratio_20d",
    "selector_positive_ratio_5d",
    "selector_return_10d",
    "selector_return_1d",
    "selector_return_20d",
    "selector_return_3d",
    "selector_return_5d",
    "selector_return_60d",
    "selector_turnover_relative_20d",
    "selector_turnover_relative_5d",
    "selector_volatility_10d",
    "selector_volatility_20d",
    "selector_volatility_5d",
    "selector_volatility_60d",
    "selector_volume_relative_10d",
    "selector_volume_relative_20d",
    "selector_volume_relative_5d",
    "selector_volume_relative_60d",
    "matched_count",
    "group__B2",
    "group__B3",
    "group__CHANGAN",
    "group__DOUBLE_YANG",
    "group__KENGQI",
    "group__LOW_PULLBACK",
    "group__RHYTHM_PLATFORM",
    "group__SB1",
    "group__STRONG_K",
    "group__SUPER_B1",
    "group__SUPPORT_PULLBACK",
    "group__TRIPLE_VOLUME_BREAKOUT",
    "group__VEGAS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the long-entry factor catalog.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_literal_list(path: Path, variable_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable_name for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == variable_name
        ):
            value = node.value
        if value is not None:
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
                raise TypeError(f"{path}:{variable_name} is not a list[str]")
            return parsed
    raise KeyError(f"{variable_name} not found in {path}")


def project_factor_group(factor: str) -> str:
    if factor.startswith("alpha"):
        return "technical_alpha"
    if factor.startswith(("volatility_", "downside_volatility_", "amplitude_", "atr_", "keltner_width")):
        return "price_risk"
    if factor.startswith(
        (
            "volume_",
            "turnover_",
            "ts_volume_",
            "obv",
            "ground_volume",
            "post_yidong",
        )
    ):
        return "liquidity_volume"
    if factor.startswith(("pe", "pb", "ps", "dv_")):
        return "valuation_absolute"
    if factor.startswith(("total_share", "float_share", "free_share", "total_mv", "circ_mv")):
        return "size_float_structure"
    if factor.startswith(("free_", "float_", "circ_")) and factor.endswith(("ratio", "20d")):
        return "size_float_structure"
    if factor.startswith(("yidong", "strong_yidong", "days_since", "b2_", "s1_", "sell_score")):
        return "technical_pattern"
    if factor in {"open", "high", "low", "close", "pre_close", "change", "price_level", "price_log"}:
        return "raw_price_scale"
    return "technical_trend_momentum"


RAW_SCALE_PREFIXES = (
    "ma_",
    "ema_",
    "bb_",
    "keltner_lower",
    "keltner_upper",
    "bbi",
    "parabolic_sar",
    "volume_ma",
    "volume_ema",
    "weekly_ma",
)
RAW_SCALE_FACTORS = {
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "price_level",
    "price_log",
    "obv",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
}


def project_factor_record(factor: str) -> FactorRecord:
    group = project_factor_group(factor)
    role = "timing"
    status = "phase1_candidate"
    notes = "B1 生产模型曾使用；周频仅保留信号日前可见值。"
    if group == "price_risk":
        role = "timing;drawdown_risk"
    elif group == "liquidity_volume":
        role = "timing;liquidity"
    elif group == "valuation_absolute":
        role = "relative_value"
        status = "phase1_core"
    elif group == "size_float_structure":
        role = "context;liquidity"
    if factor in RAW_SCALE_FACTORS or factor.startswith(RAW_SCALE_PREFIXES):
        status = "transform_required"
        notes += " 原始绝对量纲不直接入长线模型，先转为比率、斜率、对数或横截面排名。"
    return FactorRecord(
        factor=factor,
        group=group,
        source="daily;daily_basic;project_variable_library",
        frequency="daily_to_weekly",
        role=role,
        status=status,
        availability="production_available",
        point_in_time_rule="weekly_last_available_at_close",
        used_by="B1_production",
        notes=notes,
    )


def selector_records() -> list[FactorRecord]:
    records = []
    for factor in SELECTOR_PRODUCTION_FEATURES:
        conditional = factor == "matched_count" or factor.startswith("group__")
        market = factor.startswith("selector_market_")
        risk = "volatility" in factor or "amplitude" in factor or "down5" in factor
        records.append(
            FactorRecord(
                factor=factor,
                group=(
                    "short_signal_context"
                    if conditional
                    else "market_breadth_sentiment"
                    if market
                    else "price_risk"
                    if risk
                    else "technical_timing_normalized"
                ),
                source="selector_model_history;daily",
                frequency="daily_to_weekly",
                role="timing;drawdown_risk" if risk or market else "timing",
                status="phase2_research" if conditional else "phase1_candidate",
                availability="production_available",
                point_in_time_rule="signal_close_and_trailing_only",
                used_by="selector_buy_hold_production",
                notes=(
                    "只在同日命中对应短线策略时有定义，不作为长线估值的必要条件。"
                    if conditional
                    else "短线模型已验证过可计算性；长线目标需重新做周频样本外验证。"
                ),
            )
        )
    return records


def chan_records() -> list[FactorRecord]:
    manifest = PROJECT_ROOT / "reports/chan_daily/model_filter/chan_model_manifest.json"
    if manifest.exists():
        factors = json.loads(manifest.read_text(encoding="utf-8")).get("features", [])
    else:
        factors = read_literal_list(
            PROJECT_ROOT / "scripts/research/train_chan_daily_models.py", "BASE_FEATURES"
        )
    records = []
    for factor in factors:
        conditional = factor.startswith("chan_") or factor == "entry_gap_pct"
        top_list = factor.startswith("top_")
        market = factor.startswith(("market_", "limit_", "strong_up_"))
        records.append(
            FactorRecord(
                factor=factor,
                group=(
                    "chan_structure"
                    if conditional
                    else "event_moneyflow"
                    if top_list
                    else "market_breadth_sentiment"
                    if market
                    else "technical_timing_normalized"
                ),
                source="chan_model_dataset;daily_basic;top_list",
                frequency="daily_to_weekly",
                role="timing;drawdown_risk" if market else "timing",
                status=(
                    "shadow_only"
                    if top_list
                    else "phase2_research"
                    if conditional
                    else "phase1_candidate"
                ),
                availability="limited_history" if top_list else "research_available",
                point_in_time_rule="signal_close_and_trailing_only",
                used_by="chan_daily_research_model",
                notes=(
                    "本地龙虎榜仅覆盖 2026 年上半年，先影子记录，不参与历史训练。"
                    if top_list
                    else "缠论结构字段只对命中结构的样本有定义。"
                    if conditional
                    else "曾用于缠论短线模型，长线任务须重新验证。"
                ),
            )
        )
    return records


def spec_records(
    factors: Iterable[str],
    *,
    group: str,
    source: str,
    frequency: str,
    role: str,
    status: str,
    availability: str,
    point_in_time_rule: str,
    used_by: str = "",
    notes: str = "",
) -> list[FactorRecord]:
    return [
        FactorRecord(
            factor=factor,
            group=group,
            source=source,
            frequency=frequency,
            role=role,
            status=status,
            availability=availability,
            point_in_time_rule=point_in_time_rule,
            used_by=used_by,
            notes=notes,
        )
        for factor in factors
    ]


def long_existing_records() -> list[FactorRecord]:
    records: list[FactorRecord] = []
    records += spec_records(
        [
            "roe",
            "roe_waa",
            "roe_dt",
            "roa",
            "netprofit_margin",
            "grossprofit_margin",
            "profit_to_gr",
            "ar_turn",
            "inv_turn",
            "assets_turn",
        ],
        group="fundamental_profitability",
        source="fina_indicator",
        frequency="report_to_weekly",
        role="quality_gate;quality_context",
        status="phase1_core",
        availability="raw_available",
        point_in_time_rule="ann_date_lte_signal_date_latest_report",
        used_by="long_rule_or_available_raw",
        notes="按公告日向后合并；不得按报告期直接回填。",
    )
    records += spec_records(
        ["or_yoy", "basic_eps_yoy", "eps"],
        group="fundamental_growth",
        source="fina_indicator",
        frequency="report_to_weekly",
        role="quality_gate;quality_context",
        status="phase1_core",
        availability="production_available",
        point_in_time_rule="ann_date_lte_signal_date_latest_report",
        used_by="long_good_stock_rule",
        notes="增长率需缩尾并保留亏损/转盈状态标记。",
    )
    records += spec_records(
        ["debt_to_assets", "current_ratio", "quick_ratio"],
        group="fundamental_balance_sheet",
        source="fina_indicator",
        frequency="report_to_weekly",
        role="quality_gate;drawdown_risk",
        status="phase1_core",
        availability="production_available",
        point_in_time_rule="ann_date_lte_signal_date_latest_report",
        used_by="long_good_stock_rule",
        notes="金融行业须使用独立口径，不能与普通公司直接比较。",
    )
    records += spec_records(
        [
            "roe_volatility_36m",
            "margin_volatility_36m",
            "revenue_growth_volatility_36m",
            "dv_ttm_mean_36m",
            "dv_ttm_std_36m",
            "dv_ttm_stability_36m",
        ],
        group="fundamental_stability",
        source="fina_indicator;daily_basic",
        frequency="report_or_monthly_to_weekly",
        role="quality_gate;drawdown_risk",
        status="phase1_core",
        availability="production_available",
        point_in_time_rule="trailing_visible_observations_only",
        used_by="long_good_stock_rule;tea_master_long",
        notes="财务稳定度按报告频率计算，不能把 36 周误写为 36 个月。",
    )
    records += spec_records(
        ["cashflow_quality"],
        group="fundamental_cashflow",
        source="cashflow;income",
        frequency="report_to_weekly",
        role="quality_gate;drawdown_risk",
        status="phase1_candidate",
        availability="implemented_not_wired",
        point_in_time_rule="ann_date_lte_signal_date_same_report_identity",
        used_by="long_dividend_quality_experiment",
        notes="经营现金流/归母净利润；需处理季度累计值、分母接近零和金融行业。",
    )
    records += spec_records(
        [
            "pr_pe",
            "pr_pb",
            "pr",
            "pr_formula_gap",
            "pr_pe_weight",
            "pr_pb_weight",
            "valuation_profile",
            "roe_hist_percentile",
            "pe_hist_percentile",
            "pb_hist_percentile",
            "pr_pe_hist_percentile",
            "pr_pb_hist_percentile",
            "pr_hist_percentile",
            "historical_value_score",
            "valuation_history_points",
            "roe_history_points",
        ],
        group="valuation_pr_history",
        source="daily_basic;fina_indicator;stock_basic",
        frequency="weekly",
        role="relative_value;data_confidence",
        status="phase1_core",
        availability="production_available",
        point_in_time_rule="weekly_asof_plus_trailing_history_only",
        used_by="long_good_price_rule",
        notes="双 PR 始终保留；类型权重只作交互/分层，不删除任一原始 PR。",
    )
    records += spec_records(
        [
            "profitability_score",
            "fundamental_growth_score",
            "balance_sheet_score",
            "business_stability_score",
            "good_stock_score",
            "good_stock_data_coverage",
            "quality_score",
            "value_score",
            "trend_score",
            "risk_score",
            "volume_score",
            "tea_score",
            "satellite_score",
            "is_good_stock",
        ],
        group="legacy_composite_score",
        source="tea_master_long",
        frequency="weekly_or_monthly",
        role="gate;benchmark",
        status="benchmark_only",
        availability="production_available",
        point_in_time_rule="inherits_component_point_in_time_rules",
        used_by="tea_master_long;long_page",
        notes="作为基线、门控或解释字段；与底层因子同时入模会重复计权。",
    )
    records += spec_records(
        [
            "index_ma_60",
            "index_ma_120",
            "index_ma_120_slope_20d",
            "index_return_20d",
            "index_return_60d",
            "index_return_120d",
            "index_drawdown_60d",
            "index_overheat",
            "market_regime",
            "breadth_above_ma20",
            "liquidity_ratio_20d",
            "annualized_volatility_20d",
        ],
        group="market_regime",
        source="index_000300;all_stock_daily",
        frequency="daily_to_weekly",
        role="timing;drawdown_risk;calibration_context",
        status="phase1_core",
        availability="production_available",
        point_in_time_rule="signal_date_market_close_and_trailing_only",
        used_by="long_strategy;market_regime_service",
        notes="用于风险与阈值校准，不得替代个股估值。",
    )
    analyst_factors = [
        "analyst_report_count_180d",
        "analyst_org_count_180d",
        "analyst_institution_count_180d",
        "analyst_research_report_count_180d",
        "analyst_consensus_report_count_180d",
        "analyst_eps_mean_180d",
        "analyst_pe_mean_180d",
        "analyst_target_price_mean_180d",
        "analyst_net_profit_mean_180d",
        "analyst_revenue_mean_180d",
        "analyst_eps_revision_180d",
        "analyst_target_upside_180d",
        "analyst_forward_years_180d",
        "analyst_forward_eps_growth_180d",
        "analyst_forward_revenue_growth_180d",
        "analyst_forward_net_profit_growth_180d",
        "analyst_forward_pe_180d",
        "analyst_forward_eps_3y_mean_180d",
        "analyst_forward_eps_3y_variance_180d",
        "analyst_forward_eps_3y_years_180d",
        "analyst_forward_eps_3y_estimate_count_180d",
    ]
    analyst_factors += [
        f"analyst_forward_y{horizon}_{suffix}"
        for horizon in range(3)
        for suffix in (
            "year",
            "eps_mean_180d",
            "eps_std_180d",
            "eps_estimate_count_180d",
            "price_mean_180d",
            "price_std_180d",
            "price_estimate_count_180d",
        )
    ]
    records += spec_records(
        analyst_factors,
        group="analyst_point_in_time",
        source="analyst_forecasts",
        frequency="report_to_weekly",
        role="quality_context;expectation;data_confidence",
        status="phase1_candidate",
        availability="implemented_available",
        point_in_time_rule="report_date_lte_signal_date_180d_window_deduplicated",
        used_by="long_analyst_experiments;long_page_display",
        notes="进入模型前需按来源审计历史快照；研报文本和事后评价不使用。",
    )
    return records


def new_derivable_records() -> list[FactorRecord]:
    records: list[FactorRecord] = []
    records += spec_records(
        [
            "pr_pe_abs_band",
            "pr_pb_abs_band",
            "profile_pr_abs_band",
            "pe_abs_band",
            "pb_abs_band",
            "earnings_yield",
            "book_yield",
            "sales_yield",
            "pr_pe_log",
            "pr_pb_log",
            "pr_gap_direction",
        ],
        group="valuation_absolute_transforms",
        source="daily_basic;fina_indicator",
        frequency="weekly",
        role="relative_value",
        status="phase1_core",
        availability="derivable_now",
        point_in_time_rule="weekly_asof",
        notes="绝对阈值作为连续/分箱特征，不预设 PR<1 必然好或 PR>2 必然坏。",
    )
    valuation_windows = []
    for metric in ["pe", "pb", "pr_pe", "pr_pb", "profile_pr"]:
        valuation_windows.extend(f"{metric}_hist_pct_{years}y" for years in [2, 3, 5, 7, 10])
        valuation_windows.extend(f"{metric}_robust_z_{years}y" for years in [3, 5, 7, 10])
        valuation_windows.extend(
            [
                f"{metric}_distance_to_p10_5y",
                f"{metric}_distance_to_p25_5y",
                f"{metric}_distance_to_median_5y",
                f"{metric}_pct_short_long_gap_2y_7y",
            ]
        )
    records += spec_records(
        valuation_windows,
        group="valuation_multiwindow_history",
        source="weekly_daily_basic;point_in_time_financial",
        frequency="weekly",
        role="relative_value;data_confidence",
        status="phase1_core",
        availability="derivable_now",
        point_in_time_rule="trailing_2_to_10_years_minimum_2_years_no_future",
        notes="周采样；N/M 作为预注册候选，由走步验证选择，窗口外数据不参与。",
    )
    records += spec_records(
        [
            "valuation_valid_weeks_2y",
            "valuation_valid_weeks_5y",
            "valuation_valid_weeks_7y",
            "valuation_valid_weeks_10y",
            "valuation_missing_ratio_2y",
            "roe_age_days",
            "valuation_profile_onehot",
        ],
        group="valuation_data_confidence",
        source="weekly_daily_basic;fina_indicator;stock_basic",
        frequency="weekly",
        role="data_confidence;calibration_context",
        status="phase1_core",
        availability="derivable_now",
        point_in_time_rule="signal_date_and_trailing_only",
        notes="模型必须显式看到历史长度、缺失率和财报陈旧度。",
    )
    records += spec_records(
        [
            "close_to_ma20",
            "close_to_ma60",
            "close_to_ma120",
            "close_to_ma250",
            "ma120_slope_13w",
            "ma250_slope_26w",
            "price_hist_pct_2y",
            "price_hist_pct_5y",
            "drawdown_from_high_52w",
            "drawdown_from_high_3y",
            "distance_from_low_52w",
            "distance_from_low_3y",
            "return_13w",
            "return_26w",
            "return_52w",
            "return_104w",
            "momentum_52w_skip_4w",
            "weekly_volatility_13w",
            "weekly_volatility_52w",
            "weekly_downside_volatility_26w",
            "max_drawdown_trailing_52w",
        ],
        group="weekly_relative_price_state",
        source="daily_ohlcv",
        frequency="weekly",
        role="timing;drawdown_risk",
        status="phase1_core",
        availability="derivable_now",
        point_in_time_rule="week_close_and_trailing_only",
        notes="描述相对低位与下跌状态，不直接预测或拟合未来最低价。",
    )
    records += spec_records(
        [
            "ocf_to_revenue",
            "ocf_to_assets",
            "free_cashflow",
            "free_cashflow_margin",
            "free_cashflow_to_assets",
            "accruals_to_assets",
            "capex_to_revenue",
            "cash_to_assets",
            "goodwill_to_assets",
            "intangibles_to_assets",
            "inventory_to_assets",
            "fixed_assets_to_assets",
            "equity_ratio",
            "minority_interest_ratio",
            "effective_tax_rate",
            "operating_margin",
            "revenue_cagr_3y",
            "net_profit_cagr_3y",
            "roe_trend_3y",
            "gross_margin_trend_3y",
            "cashflow_quality_stability_3y",
        ],
        group="fundamental_quality_new",
        source="cashflow;income;balancesheet;fina_indicator",
        frequency="report_to_weekly",
        role="quality_gate;quality_context;drawdown_risk",
        status="phase1_candidate",
        availability="derivable_now",
        point_in_time_rule="ann_date_lte_signal_date_deduplicate_report_type",
        notes="累计季度值先转单季或 TTM；比率分母统一且按行业缩尾。",
    )
    records += spec_records(
        [
            "analyst_eps_dispersion_180d",
            "analyst_target_dispersion_180d",
            "analyst_revision_breadth_90d",
            "analyst_net_profit_revision_90d",
            "analyst_coverage_change_90d",
            "analyst_estimate_age_days",
            "analyst_source_diversity_180d",
            "analyst_forecast_year_completeness",
        ],
        group="analyst_new_transforms",
        source="analyst_forecasts",
        frequency="report_to_weekly",
        role="expectation;data_confidence;drawdown_risk",
        status="phase1_candidate",
        availability="derivable_after_source_audit",
        point_in_time_rule="report_date_lte_signal_date_identity_deduplicated",
        notes="预测分歧与上修方向优先于单一目标价；必须去除每日重复快照。",
    )
    records += spec_records(
        [
            "industry_pe_pct_rank",
            "industry_pb_pct_rank",
            "industry_pr_pe_pct_rank",
            "industry_pr_pb_pct_rank",
            "industry_roe_pct_rank",
            "industry_excess_return_13w",
            "industry_excess_return_52w",
        ],
        group="industry_relative",
        source="stock_basic;daily_basic;daily_ohlcv",
        frequency="weekly",
        role="relative_value;timing;calibration_context",
        status="phase2_research",
        availability="derivable_with_current_industry",
        point_in_time_rule="requires_dated_industry_membership_or_bias_disclosure",
        notes="当前 stock_basic 行业映射不是历史快照，历史回测存在重分类偏差。",
    )
    records += spec_records(
        [
            "large_net_amount_ratio",
            "large_net_3d_ratio",
            "large_net_5d_ratio",
            "moneyflow_net_ratio",
            "small_net_amount_ratio",
            "medium_net_amount_ratio",
            "large_flow_persistence_5d",
            "large_flow_price_divergence_5d",
        ],
        group="moneyflow_new",
        source="moneyflow",
        frequency="daily_to_weekly",
        role="timing;liquidity",
        status="phase1_candidate",
        availability="complete_trade_dates_2013_to_20260731",
        point_in_time_rule="trade_date_lte_signal_date",
        used_by="slow_money_follow_research",
        notes="已回填 3,296 个交易日；首轮按覆盖率准入，是否保留由走步验证决定。",
    )
    records += spec_records(
        [
            "top_list_count_20d",
            "top_list_net_ratio_20d",
            "top_list_positive_days_20d",
            "top_list_reason_concentration_60d",
        ],
        group="top_list_new",
        source="top_list",
        frequency="event_to_weekly",
        role="timing;event_risk",
        status="phase1_candidate",
        availability="complete_trade_dates_2013_to_20260731",
        point_in_time_rule="trade_date_lte_signal_date",
        notes="已回填 3,296 个交易日；事件稀疏，空分区表示当日无事件而非数据缺失。",
    )
    records += spec_records(
        [
            "holder_net_change_ratio_30d",
            "holder_net_change_ratio_90d",
            "holder_buy_event_count_180d",
            "holder_avg_price_gap",
            "holder_after_ratio_change_180d",
        ],
        group="holder_trade_new",
        source="holder_trade",
        frequency="event_to_weekly",
        role="quality_context;event_risk",
        status="phase1_candidate",
        availability="complete_2013_to_20260802",
        point_in_time_rule="ann_date_lte_signal_date",
        notes="169,817 条、覆盖 5,414 只股票；须规范增持/减持方向及同一事件修订。",
    )
    return records


def excluded_records() -> list[FactorRecord]:
    records: list[FactorRecord] = []
    records += spec_records(
        [
            "actual_eps",
            "actual_revenue",
            "actual_net_profit",
            "actual_pe",
            "evaluate_count",
            "future_return_t5_pct",
            "future_max_high_t5_pct",
            "future_return_52w",
            "future_mae_13w",
            "future_mae_26w",
            "future_cheaper_10pct_13w",
        ],
        group="label_or_hindsight",
        source="analyst_forecasts;research_labels",
        frequency="future",
        role="label_only",
        status="excluded",
        availability="present_or_derivable",
        point_in_time_rule="never_feature",
        notes="只能作标签或事后评价，严禁进入特征矩阵。",
    )
    records += spec_records(
        [
            "pledge_ratio",
            "pledge_ratio_change_13w",
            "pledge_ratio_change_52w",
            "pledge_event_count_26w",
            "pledge_release_ratio_26w",
        ],
        group="pledge_risk_new",
        source="pledge_stat",
        frequency="event_to_weekly",
        role="event_risk",
        status="phase1_candidate",
        availability="complete_stat_history_1724693_rows",
        point_in_time_rule="end_date_lte_signal_date",
        notes="质押统计覆盖 4,451 只有质押记录的股票；明细中 32 只存在源端重复尾页，仅作增强。",
    )
    records += spec_records(
        [
            "margin_balance",
            "margin_balance_change",
            "margin_buy_ratio_5d",
            "short_balance",
            "short_pressure_change_5d",
        ],
        group="margin_detail_new",
        source="margin_detail",
        frequency="daily_to_weekly",
        role="liquidity;event_risk",
        status="phase1_candidate",
        availability="complete_trade_dates_2013_to_20260731",
        point_in_time_rule="trade_date_lte_signal_date",
        notes="股票级明细已回填 6,574,332 条；不再使用仅交易所汇总的旧 margin 空壳表。",
    )
    records += spec_records(
        ["is_suspended_history", "is_st_history", "limit_distance_history"],
        group="unusable_source",
        source="tradability",
        frequency="daily",
        role="tradability_gate",
        status="excluded",
        availability="latest_snapshot_only",
        point_in_time_rule="not_historically_available",
        notes="当前只有 2026-07-31 单日快照，不能回填历史。",
    )
    return records


def merge_records(records: Iterable[FactorRecord]) -> list[FactorRecord]:
    merged: dict[str, FactorRecord] = {}
    for record in records:
        current = merged.get(record.factor)
        if current is None:
            merged[record.factor] = record
            continue
        status = min([current.status, record.status], key=STATUS_ORDER.get)

        def combine(left: str, right: str) -> str:
            values = []
            for value in [left, right]:
                values.extend(item.strip() for item in value.split(";") if item.strip())
            return ";".join(dict.fromkeys(values))

        merged[record.factor] = replace(
            current,
            source=combine(current.source, record.source),
            role=combine(current.role, record.role),
            status=status,
            availability=combine(current.availability, record.availability),
            point_in_time_rule=combine(current.point_in_time_rule, record.point_in_time_rule),
            used_by=combine(current.used_by, record.used_by),
            notes=" ".join(item for item in [current.notes, record.notes] if item),
        )
    return sorted(merged.values(), key=lambda item: (STATUS_ORDER[item.status], item.group, item.factor))


def build_catalog() -> list[FactorRecord]:
    project_factors = read_literal_list(
        PROJECT_ROOT / "src/quant/features/variable_library.py", "PROJECT_FACTOR_COLUMNS"
    )
    records: list[FactorRecord] = [project_factor_record(factor) for factor in project_factors]
    records += selector_records()
    records += chan_records()
    records += long_existing_records()
    records += new_derivable_records()
    records += excluded_records()
    return merge_records(records)


def parquet_metadata(path: Path) -> tuple[int, list[str]]:
    parquet = pq.ParquetFile(path)
    return parquet.metadata.num_rows, parquet.schema.names


def collect_data_coverage() -> dict[str, dict[str, object]]:
    sources = {
        "stock_basic": PROJECT_ROOT / "data/raw/stock_basic_history.parquet",
        "fina_indicator": PROJECT_ROOT / "data/raw/fina_indicator.parquet",
        "cashflow": PROJECT_ROOT / "data/raw/cashflow.parquet",
        "income": PROJECT_ROOT / "data/raw/income.parquet",
        "balancesheet": PROJECT_ROOT / "data/raw/balancesheet.parquet",
        "analyst_forecasts": PROJECT_ROOT / "data/raw/analyst_forecasts.parquet",
        "report_rc": PROJECT_ROOT / "data/raw/report_rc.parquet",
        "holder_trade": PROJECT_ROOT / "data/raw/holder_trade.parquet",
        "pledge_stat": PROJECT_ROOT / "data/raw/pledge_stat.parquet",
        "margin": PROJECT_ROOT / "data/raw/margin.parquet",
        "index_000300": PROJECT_ROOT / "data/raw/index_000300.SH.parquet",
    }
    coverage: dict[str, dict[str, object]] = {}
    for name, path in sources.items():
        if not path.exists():
            coverage[name] = {"exists": False, "rows": 0, "columns": []}
            continue
        rows, columns = parquet_metadata(path)
        coverage[name] = {"exists": True, "rows": rows, "columns": columns}
    for name, pattern in {
        "daily_basic": "*.parquet",
        "moneyflow": "*moneyflow_*.parquet",
        "top_list": "*top_list_*.parquet",
        "margin_detail": "*margin_detail_*.parquet",
        "tradability": "*.parquet",
    }.items():
        directory = PROJECT_ROOT / "data/raw" / name
        paths = sorted(directory.glob(pattern))
        rows = sum(parquet_metadata(path)[0] for path in paths)
        coverage[name] = {
            "exists": bool(paths),
            "files": len(paths),
            "rows": rows,
            "first_file": paths[0].name if paths else None,
            "last_file": paths[-1].name if paths else None,
            "columns": parquet_metadata(paths[-1])[1] if paths else [],
        }
    return coverage


def write_outputs(catalog: list[FactorRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "factor_catalog.csv"
    fields = list(asdict(catalog[0]).keys())
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in catalog)

    used_by_counts: Counter[str] = Counter()
    for record in catalog:
        used_by_counts.update(item for item in record.used_by.split(";") if item)
    summary = {
        "schema_version": "long_entry_factor_inventory_v1",
        "factor_count": len(catalog),
        "status_counts": dict(Counter(record.status for record in catalog).most_common()),
        "group_counts": dict(Counter(record.group for record in catalog).most_common()),
        "historical_use_counts": dict(used_by_counts.most_common()),
        "data_coverage": collect_data_coverage(),
        "guardrails": [
            "All features must be observable by the signal close.",
            "actual_* and future_* columns are never model features.",
            "Event and flow sources cover all 3,296 project trade dates from 2013 onward.",
            "Pledge-detail duplicate-tail anomalies remain explicit and are not treated as full coverage.",
            "Raw absolute price and volume levels require scale-free transforms.",
        ],
    }
    (output_dir / "factor_inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 长线建仓模型因子盘点",
        "",
        f"- 去重后因子/候选字段：{len(catalog)}",
        f"- B1 生产模型已用项目因子：{used_by_counts['B1_production']}",
        f"- 短线买入/持有生产模型已用：{used_by_counts['selector_buy_hold_production']}",
        f"- 缠论研究模型已用：{used_by_counts['chan_daily_research_model']}",
        "",
        "## 按训练状态",
        "",
        "| 状态 | 数量 | 含义 |",
        "| --- | ---: | --- |",
    ]
    status_meanings = {
        "phase1_core": "首轮必须纳入或作为硬门控",
        "phase1_candidate": "首轮候选，经覆盖率、相关性和走步验证筛选",
        "transform_required": "已有但必须先去量纲/归一化",
        "phase2_research": "条件性强或存在历史口径偏差",
        "shadow_only": "历史太短，只记录不训练",
        "benchmark_only": "仅作门控、解释或基线，避免重复计权",
        "excluded": "前视、事后或本地数据不可用",
    }
    counts = Counter(record.status for record in catalog)
    for status in STATUS_ORDER:
        lines.append(f"| `{status}` | {counts[status]} | {status_meanings[status]} |")
    lines += [
        "",
        "完整逐字段目录见 `factor_catalog.csv`，数据覆盖见 `factor_inventory_summary.json`。",
        "",
        "本目录仅用于研究设计，不构成投资建议。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    catalog = build_catalog()
    write_outputs(catalog, args.output_dir)
    print(
        json.dumps(
            {
                "status": "success",
                "factor_count": len(catalog),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
