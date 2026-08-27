"""Point-in-time weekly factors for long-horizon entry-price models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quant.features.project_factor_layer import PROJECT_FACTOR_SCHEMA_VERSION


IDENTIFIER_COLUMNS = {
    "date",
    "trade_date",
    "ts_code",
    "symbol",
    "name",
    "industry",
    "list_date",
    "valuation_profile",
    "factor_schema_version",
    "entry_date",
    "entry_price",
    "period",
}
LABEL_PREFIXES = (
    "label_",
    "future_",
    "return_13w",
    "return_26w",
    "return_52w",
    "mae_",
    "excess_return_",
    "benchmark_return_",
)

MONTHLY_VALUATION_HISTORY_COLUMNS: tuple[str, ...] = (
    "roe_hist_percentile",
    "roe_history_points",
    "pe_hist_percentile",
    "pb_hist_percentile",
    "pr_pe_hist_percentile",
    "pr_pb_hist_percentile",
    "pr_hist_percentile",
    "historical_value_score",
    "valuation_history_points",
)


def _rolling_last_percentile(
    values: pd.Series,
    *,
    window: int,
    minimum: int,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rolling(window, min_periods=minimum).rank(method="max", pct=True).mul(100.0)


def _add_dual_pr(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    roe_percent = pd.to_numeric(out["roe"], errors="coerce")
    roe_decimal = (roe_percent / 100.0).where(lambda values: values > 0)
    pe = pd.to_numeric(out["pe_ttm"], errors="coerce").where(lambda values: values > 0)
    pb = pd.to_numeric(out["pb"], errors="coerce").where(lambda values: values > 0)
    out["pr_pe"] = pe / roe_decimal / 100.0
    out["pr_pb"] = pb / roe_decimal.pow(2) / 100.0
    denominator = pd.concat([out["pr_pe"].abs(), out["pr_pb"].abs()], axis=1).max(axis=1)
    out["pr_formula_gap"] = (out["pr_pe"] - out["pr_pb"]).abs() / denominator.replace(0, np.nan)

    industry = out["industry"].astype(str)
    asset_based = industry.str.contains(
        "银行|证券|保险|多元金融|房地产|煤炭|石油|石化|钢铁|有色|建筑|建材|电力|公用事业|交通运输|港口|机场|高速|航运",
        regex=True,
        na=False,
    )
    earnings_based = industry.str.contains(
        "软件|互联网|传媒|医药|医疗|食品|饮料|家电|电子|计算机|通信|教育|旅游|酒店|商贸|零售|美容|服务",
        regex=True,
        na=False,
    ) & ~asset_based
    out["valuation_profile"] = "balanced"
    out.loc[asset_based, "valuation_profile"] = "asset_based"
    out.loc[earnings_based, "valuation_profile"] = "earnings_based"
    out["valuation_profile_code"] = out["valuation_profile"].map(
        {"balanced": 0.0, "asset_based": 1.0, "earnings_based": 2.0}
    )
    out["pr_pe_weight"] = 0.50
    out["pr_pb_weight"] = 0.50
    out.loc[asset_based, ["pr_pe_weight", "pr_pb_weight"]] = [0.30, 0.70]
    out.loc[earnings_based, ["pr_pe_weight", "pr_pb_weight"]] = [0.70, 0.30]
    out["pr"] = out["pr_pe"] * out["pr_pe_weight"] + out["pr_pb"] * out["pr_pb_weight"]
    return out


def add_weekly_valuation_history(
    frame: pd.DataFrame,
    *,
    minimum_years: int,
    maximum_years: int,
) -> pd.DataFrame:
    """Add trailing weekly valuation ranks for one N/M history contract."""

    if minimum_years < 1 or maximum_years < minimum_years:
        raise ValueError("valuation history requires 1 <= minimum_years <= maximum_years")
    out = _add_dual_pr(frame) if "pr_pe" not in frame.columns else frame.copy()
    window = maximum_years * 52
    minimum = minimum_years * 52
    suffix = f"{maximum_years}y"
    percentile_columns: list[str] = []
    for source in ("roe", "pe_ttm", "pb", "pr_pe", "pr_pb"):
        clean = pd.to_numeric(out[source], errors="coerce")
        if source != "roe":
            clean = clean.where(clean > 0)
        target = f"{source}_hist_percentile_{suffix}"
        out[target] = clean.groupby(out["ts_code"], sort=False).transform(
            lambda values: _rolling_last_percentile(values, window=window, minimum=minimum)
        )
        percentile_columns.append(target)
    out[f"pr_hist_percentile_{suffix}"] = (
        out[f"pr_pe_hist_percentile_{suffix}"] * out["pr_pe_weight"]
        + out[f"pr_pb_hist_percentile_{suffix}"] * out["pr_pb_weight"]
    )
    counts = []
    for source in ("pe_ttm", "pb", "pr_pe", "pr_pb"):
        clean = pd.to_numeric(out[source], errors="coerce").where(lambda values: values > 0)
        counts.append(
            clean.groupby(out["ts_code"], sort=False).transform(
                lambda values: values.rolling(window, min_periods=1).count()
            )
        )
    out[f"valuation_history_points_{suffix}"] = pd.concat(counts, axis=1).min(axis=1)
    out[f"historical_value_score_{suffix}"] = (
        100.0
        - out[
            [
                f"pe_ttm_hist_percentile_{suffix}",
                f"pb_hist_percentile_{suffix}",
                f"pr_hist_percentile_{suffix}",
            ]
        ]
        .mul([0.30, 0.25, 0.45])
        .sum(axis=1, min_count=3)
    ).clip(0, 100)
    # Stable public aliases omit the daily_basic source spelling.
    out[f"pe_hist_percentile_{suffix}"] = out[f"pe_ttm_hist_percentile_{suffix}"]
    return out.replace([np.inf, -np.inf], np.nan)


def add_monthly_valuation_history(
    frame: pd.DataFrame,
    *,
    window_months: int = 84,
    minimum_months: int = 24,
) -> pd.DataFrame:
    """Add the canonical unsuffixed PIT valuation-history factors.

    ``frame`` must contain one point-in-time observation per stock and month.
    Keeping this calculation in the factor layer lets research datasets
    materialize the same governed factors without depending on a live snapshot.
    """

    if minimum_months < 1 or window_months < minimum_months:
        raise ValueError(
            "monthly valuation history requires 1 <= minimum_months <= window_months"
        )
    required = {"date", "ts_code", "industry", "roe", "pe_ttm", "pb"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"monthly valuation history misses columns: {missing}")
    out = _add_dual_pr(frame.sort_values(["ts_code", "date"]))
    for source, target in (
        ("roe", "roe_hist_percentile"),
        ("pe_ttm", "pe_hist_percentile"),
        ("pb", "pb_hist_percentile"),
        ("pr_pe", "pr_pe_hist_percentile"),
        ("pr_pb", "pr_pb_hist_percentile"),
    ):
        clean = pd.to_numeric(out[source], errors="coerce")
        if source != "roe":
            clean = clean.where(clean > 0)
        out[target] = clean.groupby(out["ts_code"], sort=False).transform(
            lambda values: _rolling_last_percentile(
                values,
                window=window_months,
                minimum=minimum_months,
            )
        )
    out["pr_hist_percentile"] = (
        out["pr_pe_hist_percentile"] * out["pr_pe_weight"]
        + out["pr_pb_hist_percentile"] * out["pr_pb_weight"]
    )
    history_counts = []
    for source in ("pe_ttm", "pb", "pr_pe", "pr_pb"):
        clean = pd.to_numeric(out[source], errors="coerce").where(
            lambda values: values > 0
        )
        history_counts.append(
            clean.groupby(out["ts_code"], sort=False).transform(
                lambda values: values.rolling(window_months, min_periods=1).count()
            )
        )
    out["valuation_history_points"] = pd.concat(history_counts, axis=1).min(axis=1)
    out["roe_history_points"] = pd.to_numeric(
        out["roe"], errors="coerce"
    ).groupby(out["ts_code"], sort=False).transform(
        lambda values: values.rolling(window_months, min_periods=1).count()
    )
    out["historical_value_score"] = (
        100.0
        - out[
            ["pe_hist_percentile", "pb_hist_percentile", "pr_hist_percentile"]
        ]
        .mul([0.30, 0.25, 0.45])
        .sum(axis=1, min_count=3)
    ).clip(0, 100)
    return out.replace([np.inf, -np.inf], np.nan)


def add_long_entry_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric = lambda column: pd.to_numeric(out[column], errors="coerce")
    close = numeric("close")
    for window in (20, 60, 120):
        out[f"close_to_ma{window}"] = close / numeric(f"ma_{window}").replace(0, np.nan) - 1.0
    out["close_to_median60"] = close / numeric("median_close_60").replace(0, np.nan) - 1.0
    out["earnings_yield"] = 1.0 / numeric("pe_ttm").where(lambda values: values > 0)
    out["book_yield"] = 1.0 / numeric("pb").where(lambda values: values > 0)
    out["sales_yield"] = 1.0 / numeric("ps_ttm").where(lambda values: values > 0)
    out["roe_to_pe"] = numeric("roe") / numeric("pe_ttm").where(lambda values: values > 0)
    out["roe_to_pb"] = numeric("roe") / numeric("pb").where(lambda values: values > 0)
    out["dividend_to_volatility"] = numeric("dv_ttm") / numeric("volatility_60d").replace(0, np.nan)
    out["growth_quality_interaction"] = numeric("or_yoy") * numeric("roe") / 100.0
    out["leverage_adjusted_roe"] = numeric("roe") * (1.0 - numeric("debt_to_assets") / 100.0)
    out["pr_absolute_lt_1"] = numeric("pr").between(0, 1, inclusive="neither").astype(float)
    out["pr_absolute_1_to_2"] = numeric("pr").between(1, 2, inclusive="both").astype(float)
    out["pr_absolute_gt_2"] = numeric("pr").gt(2).astype(float)
    out["both_pr_absolute_lt_1"] = (
        numeric("pr_pe").between(0, 1, inclusive="neither")
        & numeric("pr_pb").between(0, 1, inclusive="neither")
    ).astype(float)
    if "analyst_forward_eps_3y_mean_180d" in out.columns:
        out["analyst_eps_3y_dispersion_180d"] = (
            numeric("analyst_forward_eps_3y_variance_180d").clip(lower=0).pow(0.5)
            / numeric("analyst_forward_eps_3y_mean_180d").abs().replace(0, np.nan)
        )
    cross_sectional = (
        "pe_ttm",
        "pb",
        "pr",
        "roe",
        "good_stock_score",
        "return_120d",
        "volatility_60d",
        "dv_ttm",
    )
    for column in cross_sectional:
        values = numeric(column)
        out[f"{column}_cross_section_pct"] = values.groupby(out["date"]).rank(
            method="average", pct=True
        )
    industry_group = [out["date"], out["industry"].fillna("未知").astype(str)]
    for column in (
        "pe_ttm",
        "pb",
        "pr",
        "roe",
        "or_yoy",
        "return_120d",
        "volatility_60d",
        "dv_ttm",
        "good_stock_score",
    ):
        values = numeric(column)
        out[f"{column}_industry_pct"] = values.groupby(industry_group).rank(
            method="average", pct=True
        )
    for column in ("return_120d", "or_yoy", "roe", "volatility_60d", "dv_ttm", "pr"):
        values = numeric(column)
        industry_mean = values.groupby(industry_group).transform("mean")
        out[f"industry_{column}_mean"] = industry_mean
        out[f"{column}_minus_industry"] = values - industry_mean
    out["industry_good_stock_count"] = close.groupby(industry_group).transform("count").astype(float)
    out["industry_positive_momentum_share"] = (
        numeric("return_120d").gt(0).astype(float).groupby(industry_group).transform("mean")
    )
    out["factor_schema_version"] = PROJECT_FACTOR_SCHEMA_VERSION
    return out.replace([np.inf, -np.inf], np.nan)


def build_long_weekly_factor_frame(
    weekly: pd.DataFrame,
    *,
    history_windows: Sequence[tuple[int, int]] = ((2, 5), (2, 7)),
) -> pd.DataFrame:
    out = weekly.sort_values(["ts_code", "date"]).copy()
    out = _add_dual_pr(out)
    for minimum_years, maximum_years in history_windows:
        out = add_weekly_valuation_history(
            out,
            minimum_years=minimum_years,
            maximum_years=maximum_years,
        )
    out = add_long_entry_interactions(out)
    return out.sort_values(["date", "ts_code"]).reset_index(drop=True)


def long_model_candidate_columns(frame: pd.DataFrame) -> list[str]:
    """Return every numeric, point-in-time column; no performance prefilter."""

    candidates: list[str] = []
    for column in frame.columns:
        if column in IDENTIFIER_COLUMNS or column.startswith(LABEL_PREFIXES):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            candidates.append(column)
    return candidates
