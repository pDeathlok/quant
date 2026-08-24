"""Machine-readable ownership and point-in-time policy for project factors."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from quant.features.long_external_factors import LONG_EXTERNAL_FACTOR_COLUMNS
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
)
from quant.features.variable_library import DAILY_BASIC_SOURCE_COLUMNS, PROJECT_FACTOR_COLUMNS
from quant.research.right_side_unified_features import RULE_FEATURE_COLUMNS


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    family: str
    frequency: str
    source: str
    point_in_time: bool
    consumers: tuple[str, ...]
    role: str = "feature"
    canonical_name: str = ""
    layer: str = ""
    calculation_entrypoint: str = ""
    calculation_version: str = ""
    refresh_cadence: str = "on_demand"
    lifecycle: str = "research_candidate"


FACTOR_REGISTRY_SCHEMA_VERSION = "factor_registry_v2_governed"
LONG_PRODUCTION_FACTOR_SCHEMA_VERSION = "long-page-v2-governed-82"


_ANALYST_FACTOR_COLUMNS = (
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
    "analyst_forward_revenue_3y_mean_180d",
    "analyst_forward_revenue_3y_variance_180d",
    "analyst_forward_net_profit_3y_mean_180d",
    "analyst_forward_net_profit_3y_variance_180d",
    "analyst_eps_3y_dispersion_180d",
    *(
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
    ),
)

_LONG_ENGINEERED_COLUMNS = (
    "median_close_60",
    "ma_120_slope_20d",
    "dv_ttm_mean_36m",
    "dv_ttm_std_36m",
    "dv_ttm_stability_36m",
    "pr_pe_weight",
    "pr_pb_weight",
    "valuation_profile_code",
    "earnings_yield",
    "book_yield",
    "sales_yield",
    "roe_to_pe",
    "roe_to_pb",
    "dividend_to_volatility",
    "growth_quality_interaction",
    "leverage_adjusted_roe",
    "pr_absolute_lt_1",
    "pr_absolute_1_to_2",
    "pr_absolute_gt_2",
    "both_pr_absolute_lt_1",
    "market_return_13w",
    "market_return_26w",
    "market_return_52w",
    "market_drawdown_52w",
    "market_volatility_13w",
    "return_120d_minus_market",
    "industry_return_120d_minus_market",
    "industry_good_stock_count",
    "industry_positive_momentum_share",
)

_LONG_RANK_BASES = (
    "pe_ttm",
    "pb",
    "pr",
    "roe",
    "good_stock_score",
    "return_120d",
    "volatility_60d",
    "dv_ttm",
)
_LONG_INDUSTRY_BASES = (
    "pe_ttm",
    "pb",
    "pr",
    "roe",
    "or_yoy",
    "return_120d",
    "volatility_60d",
    "dv_ttm",
    "good_stock_score",
)
_LONG_INDUSTRY_MEAN_BASES = (
    "return_120d",
    "or_yoy",
    "roe",
    "volatility_60d",
    "dv_ttm",
    "pr",
)

# Stable annual quality factors built from the first locally available annual
# announcement.  Keeping the model-facing contract here prevents a research
# script from silently training on fields that are absent from the project
# registry.
LONG_ANNUAL_QUALITY_CASHFLOW_FACTOR_COLUMNS = (
    "annual_cashflow_quality",
    "annual_fcf_margin",
    "annual_accruals_to_assets",
    "cashflow_quality_3y",
    "free_cashflow_margin_3y",
    "accruals_to_assets_3y",
    "cfo_positive_share_5y",
)
LONG_ANNUAL_QUALITY_PERSISTENCE_FACTOR_COLUMNS = (
    "profit_positive_share_5y",
    "revenue_growth_positive_share_5y",
    "revenue_cagr_3y",
    "net_income_cagr_3y",
    "roe_mean_5y",
    "roe_std_5y",
)
LONG_ANNUAL_QUALITY_ASSET_FACTOR_COLUMNS = (
    "annual_goodwill_to_assets",
    "annual_inventory_to_assets",
    "annual_cash_to_assets",
    "fina_current_ratio",
    "fina_quick_ratio",
    "fina_ar_turn",
    "fina_inv_turn",
    "fina_assets_turn",
    "ar_turn_change_3y",
    "inv_turn_change_3y",
)
LONG_ANNUAL_QUALITY_RAW_FACTOR_COLUMNS = (
    "annual_cashflow_quality",
    "annual_fcf_margin",
    "annual_accruals_to_assets",
    "cashflow_quality_3y",
    "free_cashflow_margin_3y",
    "accruals_to_assets_3y",
    "profit_positive_share_5y",
    "cfo_positive_share_5y",
    "revenue_growth_positive_share_5y",
    "revenue_cagr_3y",
    "net_income_cagr_3y",
    "roe_mean_5y",
    "roe_std_5y",
    "annual_goodwill_to_assets",
    "annual_inventory_to_assets",
    "annual_cash_to_assets",
    "fina_current_ratio",
    "fina_quick_ratio",
    "fina_ar_turn",
    "fina_inv_turn",
    "fina_assets_turn",
    "ar_turn_change_3y",
    "inv_turn_change_3y",
)
LONG_ANNUAL_QUALITY_SCORE_FACTOR_COLUMNS = (
    "cashflow_quality_score",
    "cashflow_quality_coverage",
    "earnings_persistence_score",
    "earnings_persistence_coverage",
    "asset_quality_score",
    "asset_quality_coverage",
    "enhanced_good_stock_score",
    "enhanced_quality_coverage",
    "industry_value_score",
    "industry_value_coverage",
    "blended_value_score",
)
LONG_ANNUAL_QUALITY_FACTOR_COLUMNS = (
    *LONG_ANNUAL_QUALITY_RAW_FACTOR_COLUMNS,
    *LONG_ANNUAL_QUALITY_SCORE_FACTOR_COLUMNS,
)

LONG_FACTOR_COLUMNS = (
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
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "pr_pe",
    "pr_pb",
    "pr_formula_gap",
    "pr",
    "roe_hist_percentile",
    "roe_history_points",
    "pe_hist_percentile",
    "pb_hist_percentile",
    "pr_pe_hist_percentile",
    "pr_pb_hist_percentile",
    "pr_hist_percentile",
    "historical_value_score",
    "valuation_history_points",
    "pe_hist_percentile_5y",
    "pb_hist_percentile_5y",
    "pr_pe_hist_percentile_5y",
    "pr_pb_hist_percentile_5y",
    "pr_hist_percentile_5y",
    "historical_value_score_5y",
    "pe_hist_percentile_7y",
    "pb_hist_percentile_7y",
    "pr_pe_hist_percentile_7y",
    "pr_pb_hist_percentile_7y",
    "pr_hist_percentile_7y",
    "historical_value_score_7y",
    "pe_ttm_hist_percentile_5y",
    "pe_ttm_hist_percentile_7y",
    "roe_hist_percentile_5y",
    "roe_hist_percentile_7y",
    "valuation_history_points_5y",
    "valuation_history_points_7y",
    *_ANALYST_FACTOR_COLUMNS,
    *_LONG_ENGINEERED_COLUMNS,
    *(f"close_to_ma{window}" for window in (20, 60, 120)),
    "close_to_median60",
    *(f"{base}_cross_section_pct" for base in _LONG_RANK_BASES),
    *(f"{base}_industry_pct" for base in _LONG_INDUSTRY_BASES),
    *(f"industry_{base}_mean" for base in _LONG_INDUSTRY_MEAN_BASES),
    *(f"{base}_minus_industry" for base in _LONG_INDUSTRY_MEAN_BASES),
    *LONG_ANNUAL_QUALITY_FACTOR_COLUMNS,
)

# The long page currently and deterministically produces this exact contract.
# The remaining long factors stay registered as research candidates until their
# calculators are wired into the production snapshot and pass daily coverage.
LONG_PRODUCTION_FACTOR_COLUMNS = (
    *LONG_FACTOR_COLUMNS[:29],
    *(
        name
        for name in _ANALYST_FACTOR_COLUMNS
        if name != "analyst_eps_3y_dispersion_180d"
    ),
    *_LONG_ENGINEERED_COLUMNS[:7],
)
LONG_RESEARCH_FACTOR_COLUMNS = tuple(
    name for name in LONG_FACTOR_COLUMNS if name not in set(LONG_PRODUCTION_FACTOR_COLUMNS)
)

_EXTRA_DAILY_PROJECT_FACTORS = (
    "psy_12",
    "vr_6",
    "vr_12",
    "vr_24",
    "rsi_6",
    "rsi_24",
    "williams_r_14",
    "cmf",
    "eom",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "reversal_20d",
    "risk_adjusted_momentum",
    "price_volume_ratio",
    "limit_up_cnt_3d",
    "limit_up_cnt_5d",
    "limit_up_cnt_10d",
    "limit_up_cnt_20d",
    "limit_up_cnt_60d",
    "gt_9p5pct_cnt_3d",
    "gt_9p5pct_cnt_5d",
    "gt_9p5pct_cnt_10d",
    "gt_9p5pct_cnt_20d",
    "gt_9p5pct_cnt_60d",
)

_EXTRA_WEEKLY_LONG_FACTORS = LONG_EXTERNAL_FACTOR_COLUMNS

FACTOR_ALIAS_TARGETS = {
    "price_level": "close",
    "bb_middle": "ma_20",
    "kdj_k": "kdj_d_k",
    "kdj_d": "kdj_d_d",
    "kdj_j": "kdj_d_j",
    "rs_pct_chg_1d": "pct_chg",
    "rs_amplitude_pct": "amplitude_1",
    "rs_vol_ratio_5_inclusive": "volume_relative_5d",
    "rs_vol_ratio_20_inclusive": "volume_relative_20d",
    "rs_family_kdj_j": "kdj_d_j",
}

SELECTOR_LIVE_FACTOR_COLUMNS = (
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
    "matched_count",
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
)

CHAN_LIVE_FACTOR_COLUMNS = (
    "amount_rel20",
    "chan_center_width",
    "chan_score",
    "close_pos_20",
    "db_circ_mv_log",
    "db_float_mv_ratio",
    "db_free_float_share_ratio",
    "db_pb_inv",
    "db_pb_pct_rank",
    "db_pe_ttm_inv",
    "db_ps_ttm_inv",
    "db_total_mv_log",
    "db_total_mv_pct_rank",
    "db_turnover_rate",
    "db_turnover_rate_f",
    "db_turnover_rate_pct_rank",
    "db_volume_ratio",
    "db_volume_ratio_pct_rank",
    "limit_down_ratio_proxy",
    "limit_up_count_proxy",
    "limit_up_ratio_proxy",
    "ma10_dist",
    "ma20_dist",
    "ma20_slope_5d",
    "ma5_dist",
    "ma60_dist",
    "market_median_ret_1d",
    "market_panic_5d",
    "market_sentiment_5d",
    "market_up_ratio",
    "ret_10d",
    "ret_1d",
    "ret_20d",
    "ret_3d",
    "ret_5d",
    "strong_up_ratio_proxy",
    "top_list_count",
    "top_net_amount_ratio",
    "top_net_rate",
    "turnover_rate",
    "turnover_rate_ma20",
    "turnover_rate_rel20",
    "volatility_20d",
    "volume_rel20",
    "volume_rel5",
    "volume_z20",
)

PRODUCTION_REGISTRY_COLUMNS = tuple(
    dict.fromkeys(
        (
            *PROJECT_FACTOR_COLUMNS,
            *RULE_FEATURE_COLUMNS,
            *RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
            *SELECTOR_LIVE_FACTOR_COLUMNS,
            *CHAN_LIVE_FACTOR_COLUMNS,
            *LONG_PRODUCTION_FACTOR_COLUMNS,
        )
    )
)


def _daily_family(name: str) -> tuple[str, str]:
    if name in DAILY_BASIC_SOURCE_COLUMNS or name.startswith(
        ("turnover_", "total_mv", "circ_mv", "free_share", "float_share", "pe_", "pb_", "ps_")
    ):
        return "valuation_liquidity", "tushare_daily_basic"
    if name.startswith(("alpha",)):
        return "alpha", "tushare_daily_ohlcv"
    if name.startswith(("volume", "obv", "turnover")):
        return "volume_liquidity", "tushare_daily_ohlcv"
    if name.startswith(("volatility", "downside", "atr", "amplitude")):
        return "risk", "tushare_daily_ohlcv"
    if name.startswith(
        (
            "limit_up_cnt",
            "gt_9p5pct_cnt",
            "williams_r",
            "cmf",
            "eom",
            "price_volume_ratio",
            "psy_",
            "vr_",
        )
    ):
        return "momentum_timing", "tushare_daily_ohlcv"
    if name.startswith(("return", "momentum", "reversal", "bias", "rsi", "kdj", "macd")):
        return "momentum_timing", "tushare_daily_ohlcv"
    return "price_structure", "tushare_daily_ohlcv"


def build_factor_registry() -> tuple[FactorDefinition, ...]:
    definitions: list[FactorDefinition] = []
    for name in PROJECT_FACTOR_COLUMNS:
        family, source = _daily_family(name)
        consumers = [
            "short_models",
            "long_entry_weekly",
            "right_side_unified_shadow",
            "right_side_unified",
        ]
        if name in CHAN_LIVE_FACTOR_COLUMNS:
            consumers.append("score.chan")
        canonical_name = FACTOR_ALIAS_TARGETS.get(name, name)
        definitions.append(
            FactorDefinition(
                name=name,
                family=family,
                frequency="daily",
                source=source,
                point_in_time=True,
                consumers=tuple(consumers),
                role=("compatibility_alias" if canonical_name != name else "feature"),
                canonical_name=canonical_name,
                layer="project_daily",
                calculation_entrypoint=(
                    "quant.features.project_factor_layer.calculate_project_factor_frame"
                ),
                calculation_version="schema_resolved_at_runtime",
                refresh_cadence="trade_daily",
                lifecycle=(
                    "compatibility_alias" if canonical_name != name else "production_model"
                ),
            )
        )
    existing = {definition.name for definition in definitions}
    for name in RULE_FEATURE_COLUMNS:
        if name in existing:
            continue
        canonical_name = FACTOR_ALIAS_TARGETS.get(name, name)
        definitions.append(
            FactorDefinition(
                name=name,
                family="right_side_rule",
                frequency="daily",
                source="quant.research.right_side_unified_features.compute_right_side_rule_features",
                point_in_time=True,
                consumers=("right_side_unified_shadow", "right_side_unified"),
                role=("compatibility_alias" if canonical_name != name else "feature"),
                canonical_name=canonical_name,
                layer="right_side_rule",
                calculation_entrypoint=(
                    "quant.research.right_side_unified_features.compute_right_side_rule_features"
                ),
                calculation_version="right_side_rule_features_v2_118_20260813",
                refresh_cadence="trade_daily",
                lifecycle=(
                    "compatibility_alias"
                    if canonical_name != name
                    else "production_materialized"
                ),
            )
        )
        existing.add(name)
    for name in RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS:
        if name in existing:
            continue
        definitions.append(
            FactorDefinition(
                name=name,
                family="right_side_signal_identity",
                frequency="daily",
                source="canonical_right_side_signal_cache",
                point_in_time=True,
                consumers=("right_side_unified_shadow", "right_side_unified"),
                role="strategy_identity",
                canonical_name=name,
                layer="right_side_identity",
                calculation_entrypoint="canonical_right_side_signal_cache",
                calculation_version="right_side_unified_signal_v1_live_z_20260813",
                refresh_cadence="trade_daily",
                lifecycle="strategy_identity",
            )
        )
        existing.add(name)
    for name in SELECTOR_LIVE_FACTOR_COLUMNS:
        if name in existing:
            continue
        definitions.append(
            FactorDefinition(
                name=name,
                family="selector_live",
                frequency="daily",
                source="market_daily+daily_basic+strategy_signal_groups",
                point_in_time=True,
                consumers=("score.selector",),
                canonical_name=name,
                layer="selector_live",
                calculation_entrypoint="quant.webapp.services._selector_live_feature_rows",
                calculation_version="active_model_artifact_contract",
                refresh_cadence="trade_daily",
                lifecycle="production_model",
            )
        )
        existing.add(name)
    for name in CHAN_LIVE_FACTOR_COLUMNS:
        if name in existing:
            continue
        definitions.append(
            FactorDefinition(
                name=name,
                family="chan_live",
                frequency="daily",
                source="market_daily+daily_basic+top_list+market_cross_section",
                point_in_time=True,
                consumers=("score.chan",),
                canonical_name=name,
                layer="chan_live",
                calculation_entrypoint="scripts.research.refresh_chan_model_live_scores",
                calculation_version="active_model_artifact_contract",
                refresh_cadence="trade_daily",
                lifecycle="production_model",
            )
        )
        existing.add(name)
    for name in _EXTRA_DAILY_PROJECT_FACTORS:
        if name in existing:
            continue
        family, source = _daily_family(name)
        canonical_name = FACTOR_ALIAS_TARGETS.get(name, name)
        definitions.append(
            FactorDefinition(
                name=name,
                family=family,
                frequency="daily",
                source="quant.features.project_factor_layer",
                point_in_time=True,
                consumers=(),
                role=("compatibility_alias" if canonical_name != name else "feature"),
                canonical_name=canonical_name,
                layer="project_daily_candidate",
                calculation_entrypoint=(
                    "quant.features.project_factor_layer.calculate_legacy_market_factors"
                ),
                calculation_version="project-v4-causal-price-alpha",
                refresh_cadence="on_demand",
                lifecycle=(
                    "compatibility_alias" if canonical_name != name else "research_candidate"
                ),
            )
        )
        existing.add(name)
    for name in _EXTRA_WEEKLY_LONG_FACTORS:
        if name in existing:
            continue
        if name.startswith(("large_", "moneyflow_", "small_", "medium_")):
            family, source = "moneyflow", "quant.features.long_external_factors"
        elif name.startswith(("margin_", "short_")):
            family, source = "margin", "quant.features.long_external_factors"
        elif name.startswith("top_list_"):
            family, source = "top_list", "quant.features.long_external_factors"
        elif name.startswith("holder_"):
            family, source = "holder", "quant.features.long_external_factors"
        else:
            family, source = "pledge", "quant.features.long_external_factors"
        definitions.append(
            FactorDefinition(
                name=name,
                family=family,
                frequency="weekly",
                source=source,
                point_in_time=True,
                consumers=(),
                canonical_name=name,
                layer="long_external_candidate",
                calculation_entrypoint=(
                    "quant.features.long_external_factors.build_weekly_external_factor_cache"
                ),
                calculation_version="long-external-v1-pit",
                refresh_cadence="on_demand",
                lifecycle="research_candidate",
            )
        )
        existing.add(name)
    cashflow_factors = set(LONG_ANNUAL_QUALITY_CASHFLOW_FACTOR_COLUMNS)
    persistence_factors = set(LONG_ANNUAL_QUALITY_PERSISTENCE_FACTOR_COLUMNS)
    asset_factors = set(LONG_ANNUAL_QUALITY_ASSET_FACTOR_COLUMNS)
    quality_scores = set(LONG_ANNUAL_QUALITY_SCORE_FACTOR_COLUMNS)
    industry_quality_scores = {
        "industry_value_score",
        "industry_value_coverage",
        "blended_value_score",
    }
    for name in LONG_FACTOR_COLUMNS:
        if name in existing:
            continue
        if name.startswith("analyst_"):
            family, source = "analyst_expectation", "analyst_forecasts"
        elif name in cashflow_factors:
            family, source = (
                "cashflow_quality",
                "income+cashflow+balancesheet_first_announcement_pit",
            )
        elif name in persistence_factors:
            family, source = (
                "earnings_persistence",
                "fina_indicator+income+cashflow_first_announcement_pit",
            )
        elif name in asset_factors:
            family, source = (
                "asset_quality",
                "fina_indicator+balancesheet_first_announcement_pit",
            )
        elif name in industry_quality_scores:
            family, source = (
                "quality_relative_value",
                "registered_quality+daily_basic+current_industry_mapping",
            )
        elif name in quality_scores:
            family, source = (
                "long_quality_composite",
                "registered_annual_quality_factors+weekly_cross_section",
            )
        elif "hist_percentile" in name or name.startswith(("pr_", "historical_value", "valuation_history")):
            family, source = "valuation_history", "daily_basic+financial_pit"
        elif name in {"roe", "netprofit_margin", "or_yoy", "basic_eps_yoy", "debt_to_assets"}:
            family, source = "fundamental_quality", "financial_ann_date_pit"
        else:
            family, source = "good_stock_quality", "financial_ann_date_pit"
        is_production = name in LONG_PRODUCTION_FACTOR_COLUMNS
        definitions.append(
            FactorDefinition(
                name=name,
                family=family,
                frequency="weekly",
                source=source,
                point_in_time=True,
                consumers=("long_entry_weekly",)
                if is_production
                else ("long_entry_quality_shadow",),
                canonical_name=name,
                layer="long_snapshot" if is_production else "long_research",
                calculation_entrypoint=(
                    "quant.webapp.services._tea_master_live_scores"
                    if is_production
                    else "quant.features.long_weekly_factors.build_long_weekly_factor_frame"
                ),
                calculation_version=(
                    LONG_PRODUCTION_FACTOR_SCHEMA_VERSION if is_production else "research"
                ),
                refresh_cadence="trade_daily" if is_production else "on_demand",
                lifecycle="production_materialized" if is_production else "research_candidate",
            )
        )
    return tuple(definitions)


FACTOR_REGISTRY = build_factor_registry()


def registry_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(definition) for definition in FACTOR_REGISTRY])


def validate_registry() -> None:
    names = [definition.name for definition in FACTOR_REGISTRY]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate factor definitions: {duplicates}")
    unsafe = [definition.name for definition in FACTOR_REGISTRY if not definition.point_in_time]
    if unsafe:
        raise ValueError(f"non-point-in-time factors cannot enter the production registry: {unsafe}")
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}
    aliases = [
        definition for definition in FACTOR_REGISTRY if definition.role == "compatibility_alias"
    ]
    invalid_aliases = sorted(
        definition.name
        for definition in aliases
        if not definition.canonical_name
        or definition.canonical_name == definition.name
        or definition.canonical_name not in definitions
    )
    if invalid_aliases:
        raise ValueError(f"invalid compatibility aliases: {invalid_aliases}")
    invalid_canonical = sorted(
        definition.name
        for definition in FACTOR_REGISTRY
        if definition.role == "feature" and definition.canonical_name != definition.name
    )
    if invalid_canonical:
        raise ValueError(f"canonical factors must self-reference: {invalid_canonical}")
    missing_production = sorted(set(PRODUCTION_REGISTRY_COLUMNS) - set(definitions))
    if missing_production:
        raise ValueError(f"production factors missing from registry: {missing_production}")
    if len(LONG_PRODUCTION_FACTOR_COLUMNS) != 82:
        raise ValueError("long production factor contract must contain 82 columns")
    invalid_metadata = sorted(
        definition.name
        for definition in FACTOR_REGISTRY
        if not definition.layer
        or not definition.calculation_entrypoint
        or not definition.calculation_version
        or not definition.refresh_cadence
        or not definition.lifecycle
    )
    if invalid_metadata:
        raise ValueError(f"factor governance metadata is incomplete: {invalid_metadata}")
