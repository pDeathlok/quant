"""Point-in-time annual quality factors for long-horizon A-share research.

The existing long screen uses the latest disclosed profitability, growth and
leverage fields.  This module adds slower annual evidence that is harder to
fake with a single reporting period: cash conversion, free cash flow,
profitability persistence and balance-sheet asset quality.

Only the first locally available announcement for each annual period is used.
That is intentionally conservative: later restatements are not backfilled into
earlier signals.  Every merged value is gated by ``available_at <= date``.
"""

from __future__ import annotations

from functools import reduce
from typing import Iterable

import numpy as np
import pandas as pd


FINANCIAL_INDUSTRY_PATTERN = "银行|证券|保险|多元金融|金融服务"


def _annual_source(
    frame: pd.DataFrame,
    *,
    prefix: str,
    value_columns: Iterable[str],
) -> pd.DataFrame:
    """Return first-published annual rows with source-specific column names."""

    if frame.empty:
        return pd.DataFrame(columns=["ts_code", "period_end", f"{prefix}_available_at"])
    out = frame.copy()
    end_text = out["end_date"].astype(str).str.replace(r"\.0$", "", regex=True).str[:8]
    out = out[end_text.str.endswith("1231", na=False)].copy()
    end_text = end_text.loc[out.index]
    out["period_end"] = pd.to_datetime(
        end_text, format="%Y%m%d", errors="coerce"
    )
    announcement_text = (
        out["ann_date"].astype(str).str.replace(r"\.0$", "", regex=True).str[:8]
    )
    out[f"{prefix}_available_at"] = pd.to_datetime(
        announcement_text, format="%Y%m%d", errors="coerce"
    )
    if "report_type" in out.columns:
        out = out[out["report_type"].astype(str).eq("1")]
    out = out.dropna(subset=["ts_code", "period_end", f"{prefix}_available_at"])
    out = out.sort_values(["ts_code", "period_end", f"{prefix}_available_at"])
    # First-published values are point-in-time safe.  A later correction is not
    # allowed to rewrite the feature seen at the original announcement date.
    out = out.drop_duplicates(["ts_code", "period_end"], keep="first")
    keep = ["ts_code", "period_end", f"{prefix}_available_at"]
    rename: dict[str, str] = {}
    for column in value_columns:
        if column in out.columns:
            keep.append(column)
            rename[column] = f"{prefix}_{column}"
    return out[keep].rename(columns=rename)


def _positive_share(values: pd.Series, window: int, minimum: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    positive = numeric.gt(0).astype(float).where(numeric.notna())
    return positive.rolling(window, min_periods=minimum).mean()


def _safe_cagr(values: pd.Series, years: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    previous = numeric.shift(years)
    valid = numeric.gt(0) & previous.gt(0)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[valid] = (numeric.loc[valid] / previous.loc[valid]).pow(1.0 / years) - 1.0
    return result


def build_annual_quality_events(
    fina_indicator: pd.DataFrame,
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
    balancesheet: pd.DataFrame,
) -> pd.DataFrame:
    """Build one point-in-time feature event per newly available annual report."""

    sources = [
        _annual_source(
            fina_indicator,
            prefix="fina",
            value_columns=(
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
            ),
        ),
        _annual_source(
            income,
            prefix="income",
            value_columns=("revenue", "n_income_attr_p"),
        ),
        _annual_source(
            cashflow,
            prefix="cashflow",
            value_columns=("n_cashflow_act", "c_pay_acq_const_fiolta"),
        ),
        _annual_source(
            balancesheet,
            prefix="balance",
            value_columns=(
                "total_assets",
                "total_liab",
                "total_hldr_eqy_exc_min_int",
                "money_cap",
                "inventories",
                "intan_assets",
                "goodwill",
            ),
        ),
    ]
    usable = [source for source in sources if not source.empty]
    if not usable:
        return pd.DataFrame(columns=["ts_code", "period_end", "annual_quality_available_at"])
    events = reduce(
        lambda left, right: left.merge(right, on=["ts_code", "period_end"], how="outer"),
        usable,
    )
    available_columns = [column for column in events if column.endswith("_available_at")]
    events["annual_quality_available_at"] = events[available_columns].max(axis=1)
    events = events.dropna(subset=["ts_code", "period_end", "annual_quality_available_at"])

    numeric_columns = [
        column
        for column in events.columns
        if column not in {"ts_code", "period_end", "annual_quality_available_at", *available_columns}
    ]
    for column in numeric_columns:
        events[column] = pd.to_numeric(events[column], errors="coerce")

    income_value = events.get("income_n_income_attr_p", pd.Series(np.nan, index=events.index))
    revenue = events.get("income_revenue", pd.Series(np.nan, index=events.index))
    cfo = events.get("cashflow_n_cashflow_act", pd.Series(np.nan, index=events.index))
    capex = events.get("cashflow_c_pay_acq_const_fiolta", pd.Series(np.nan, index=events.index))
    assets = events.get("balance_total_assets", pd.Series(np.nan, index=events.index))
    events["annual_free_cashflow"] = cfo - capex
    events["annual_cashflow_quality"] = cfo / income_value.where(income_value.abs().gt(0))
    events["annual_fcf_margin"] = events["annual_free_cashflow"] / revenue.where(revenue.gt(0))
    events["annual_accruals_to_assets"] = (income_value - cfo) / assets.where(assets.gt(0))
    events["annual_goodwill_to_assets"] = events.get(
        "balance_goodwill", pd.Series(np.nan, index=events.index)
    ) / assets.where(assets.gt(0))
    events["annual_inventory_to_assets"] = events.get(
        "balance_inventories", pd.Series(np.nan, index=events.index)
    ) / assets.where(assets.gt(0))
    events["annual_cash_to_assets"] = events.get(
        "balance_money_cap", pd.Series(np.nan, index=events.index)
    ) / assets.where(assets.gt(0))

    result = events.sort_values(["ts_code", "period_end"]).copy()
    net_income = pd.to_numeric(result.get("income_n_income_attr_p"), errors="coerce")
    revenue = pd.to_numeric(result.get("income_revenue"), errors="coerce")
    cfo = pd.to_numeric(result.get("cashflow_n_cashflow_act"), errors="coerce")
    free_cashflow = pd.to_numeric(result["annual_free_cashflow"], errors="coerce")
    assets = pd.to_numeric(result.get("balance_total_assets"), errors="coerce")
    roe = pd.to_numeric(result.get("fina_roe"), errors="coerce")
    code = result["ts_code"]

    def rolling(values: pd.Series, window: int, minimum: int, operation: str) -> pd.Series:
        grouped = values.groupby(code, sort=False).rolling(window, min_periods=minimum)
        calculated = getattr(grouped, operation)()
        return calculated.reset_index(level=0, drop=True).reindex(result.index)

    income_sum_3y = rolling(net_income, 3, 2, "sum")
    revenue_sum_3y = rolling(revenue, 3, 2, "sum")
    result["cashflow_quality_3y"] = rolling(cfo, 3, 2, "sum") / income_sum_3y.where(
        income_sum_3y.gt(0)
    )
    result["free_cashflow_margin_3y"] = rolling(
        free_cashflow, 3, 2, "sum"
    ) / revenue_sum_3y.where(revenue_sum_3y.gt(0))
    result["accruals_to_assets_3y"] = rolling(
        net_income - cfo, 3, 2, "sum"
    ) / rolling(assets, 3, 2, "mean").where(lambda values: values.gt(0))

    profit_positive = net_income.gt(0).astype(float).where(net_income.notna())
    cfo_positive = cfo.gt(0).astype(float).where(cfo.notna())
    previous_revenue = revenue.groupby(code, sort=False).shift(1)
    revenue_growth = (revenue / previous_revenue - 1.0).where(previous_revenue.gt(0))
    revenue_positive = revenue_growth.gt(0).astype(float).where(revenue_growth.notna())
    result["profit_positive_share_5y"] = rolling(profit_positive, 5, 3, "mean")
    result["cfo_positive_share_5y"] = rolling(cfo_positive, 5, 3, "mean")
    result["revenue_growth_positive_share_5y"] = rolling(revenue_positive, 5, 3, "mean")

    revenue_lag3 = revenue.groupby(code, sort=False).shift(3)
    income_lag3 = net_income.groupby(code, sort=False).shift(3)
    result["revenue_cagr_3y"] = np.where(
        revenue.gt(0) & revenue_lag3.gt(0),
        (revenue / revenue_lag3).pow(1.0 / 3.0) - 1.0,
        np.nan,
    )
    result["net_income_cagr_3y"] = np.where(
        net_income.gt(0) & income_lag3.gt(0),
        (net_income / income_lag3).pow(1.0 / 3.0) - 1.0,
        np.nan,
    )
    result["roe_mean_5y"] = rolling(roe, 5, 3, "mean")
    result["roe_std_5y"] = rolling(roe, 5, 3, "std")
    result["annual_history_years"] = rolling(net_income, 5, 1, "count")
    ar_turn = pd.to_numeric(result.get("fina_ar_turn"), errors="coerce")
    inv_turn = pd.to_numeric(result.get("fina_inv_turn"), errors="coerce")
    ar_lag3 = ar_turn.groupby(code, sort=False).shift(3)
    inv_lag3 = inv_turn.groupby(code, sort=False).shift(3)
    result["ar_turn_change_3y"] = ar_turn / ar_lag3.where(ar_lag3.gt(0)) - 1.0
    result["inv_turn_change_3y"] = inv_turn / inv_lag3.where(inv_lag3.gt(0)) - 1.0
    # Ignore stale backfilled periods announced after a newer annual period was
    # already known.  Without this guard, an IPO history row can replace the
    # latest annual state in a simple as-of merge.
    result = result.sort_values(["ts_code", "annual_quality_available_at", "period_end"])
    result["_latest_period"] = result.groupby("ts_code", sort=False)["period_end"].cummax()
    result = result[result["period_end"].eq(result["_latest_period"])].drop(columns="_latest_period")
    result = result.drop_duplicates(["ts_code", "annual_quality_available_at"], keep="last")
    return result.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def merge_annual_quality_asof(signals: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Merge annual quality events without allowing future announcements."""

    if signals.empty or events.empty:
        return signals.copy()
    event_columns = [
        column
        for column in events.columns
        if column not in {"period_end"} and not column.endswith("_available_at")
    ]
    event_columns = [*event_columns, "period_end", "annual_quality_available_at"]
    right_by_symbol = {
        str(code): group[event_columns]
        .sort_values("annual_quality_available_at")
        .drop_duplicates("annual_quality_available_at", keep="last")
        for code, group in events.groupby("ts_code", sort=False)
    }
    parts: list[pd.DataFrame] = []
    for code, group in signals.groupby("ts_code", sort=False):
        right = right_by_symbol.get(str(code))
        if right is None or right.empty:
            parts.append(group.copy())
            continue
        left = group.sort_values("date").copy()
        right = right.drop(columns="ts_code", errors="ignore")
        merged = pd.merge_asof(
            left,
            right,
            left_on="date",
            right_on="annual_quality_available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["ts_code"] = code
        parts.append(merged)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ts_code"]).reset_index(drop=True)


def _rank_score(
    frame: pd.DataFrame,
    column: str,
    *,
    higher_is_better: bool,
    industry_neutral: bool = False,
) -> pd.Series:
    source = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
    values = pd.to_numeric(source, errors="coerce")
    market = values.groupby(frame["date"]).rank(
        method="average", pct=True, ascending=higher_is_better
    )
    if not industry_neutral:
        return market.mul(100.0)
    industry = frame["industry"].fillna("未知").astype(str)
    groups = [frame["date"], industry]
    local = values.groupby(groups).rank(method="average", pct=True, ascending=higher_is_better)
    size = values.groupby(groups).transform("count")
    return local.where(size >= 5, market).mul(100.0)


def _weighted_score(components: dict[str, tuple[pd.Series, float]]) -> tuple[pd.Series, pd.Series]:
    numerator = None
    denominator = None
    total_weight = sum(weight for _, weight in components.values())
    for series, weight in components.values():
        usable = pd.to_numeric(series, errors="coerce")
        contribution = usable.fillna(0.0) * weight
        coverage = usable.notna().astype(float) * weight
        numerator = contribution if numerator is None else numerator + contribution
        denominator = coverage if denominator is None else denominator + coverage
    assert numerator is not None and denominator is not None
    return numerator / denominator.replace(0, np.nan), denominator / total_weight


def add_enhanced_long_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable quality, value and long-model scores on a 0-100 scale."""

    out = frame.copy()
    numeric = lambda column: pd.to_numeric(
        out[column] if column in out.columns else pd.Series(np.nan, index=out.index),
        errors="coerce",
    )
    financial = out["industry"].astype(str).str.contains(
        FINANCIAL_INDUSTRY_PATTERN, regex=True, na=False
    )
    cashflow_components = {
        "cash_conversion": (
            _rank_score(out, "cashflow_quality_3y", higher_is_better=True, industry_neutral=True),
            0.40,
        ),
        "free_cashflow": (
            _rank_score(out, "free_cashflow_margin_3y", higher_is_better=True, industry_neutral=True),
            0.25,
        ),
        "cfo_persistence": (
            _rank_score(out, "cfo_positive_share_5y", higher_is_better=True),
            0.20,
        ),
        "low_accrual": (
            _rank_score(out, "accruals_to_assets_3y", higher_is_better=False, industry_neutral=True),
            0.15,
        ),
    }
    out["cashflow_quality_score"], out["cashflow_quality_coverage"] = _weighted_score(
        cashflow_components
    )
    out.loc[financial, ["cashflow_quality_score", "cashflow_quality_coverage"]] = np.nan

    persistence_components = {
        "profit_years": (
            _rank_score(out, "profit_positive_share_5y", higher_is_better=True),
            0.35,
        ),
        "roe_level": (_rank_score(out, "roe_mean_5y", higher_is_better=True), 0.25),
        "roe_stability": (_rank_score(out, "roe_std_5y", higher_is_better=False), 0.20),
        "revenue_years": (
            _rank_score(out, "revenue_growth_positive_share_5y", higher_is_better=True),
            0.20,
        ),
    }
    out["earnings_persistence_score"], out["earnings_persistence_coverage"] = _weighted_score(
        persistence_components
    )

    asset_components = {
        "low_goodwill": (
            _rank_score(out, "annual_goodwill_to_assets", higher_is_better=False, industry_neutral=True),
            0.35,
        ),
        "receivable_turn": (
            _rank_score(out, "fina_ar_turn", higher_is_better=True, industry_neutral=True),
            0.25,
        ),
        "inventory_turn": (
            _rank_score(out, "fina_inv_turn", higher_is_better=True, industry_neutral=True),
            0.20,
        ),
        "low_debt": (
            _rank_score(out, "fina_debt_to_assets", higher_is_better=False, industry_neutral=True),
            0.20,
        ),
    }
    out["asset_quality_score"], out["asset_quality_coverage"] = _weighted_score(asset_components)
    out.loc[financial, ["asset_quality_score", "asset_quality_coverage"]] = np.nan

    quality_components = {
        "profitability": (numeric("profitability_score"), 0.30),
        "growth": (numeric("fundamental_growth_score"), 0.15),
        "balance": (numeric("balance_sheet_score"), 0.10),
        "stability": (numeric("business_stability_score"), 0.15),
        "cashflow": (out["cashflow_quality_score"], 0.15),
        "persistence": (out["earnings_persistence_score"], 0.10),
        "asset_quality": (out["asset_quality_score"], 0.05),
    }
    out["enhanced_good_stock_score"], out["enhanced_quality_coverage"] = _weighted_score(
        quality_components
    )

    value_components = {
        "pe": (
            100.0 - numeric("pe_ttm_industry_pct") * 100.0,
            0.30,
        ),
        "pb": (
            100.0 - numeric("pb_industry_pct") * 100.0,
            0.25,
        ),
        "pr": (
            100.0 - numeric("pr_industry_pct") * 100.0,
            0.45,
        ),
    }
    out["industry_value_score"], out["industry_value_coverage"] = _weighted_score(value_components)
    historical_value = numeric("historical_value_score_5y")
    # Industry value is an enhancement, not a new data-coverage gate.  Fall
    # back to the stock's own history where the industry cross-section lacks a
    # complete PE/PB/PR tuple.
    industry_value = out["industry_value_score"].fillna(historical_value)
    out["blended_value_score"] = historical_value * 0.70 + industry_value * 0.30

    close_to_ma120 = numeric("close_to_ma120")
    trend_position = (100.0 - (close_to_ma120.clip(-0.20, 0.50) - 0.05).abs() * 200.0).clip(0, 100)
    momentum = numeric("return_120d_cross_section_pct") * 100.0
    downside_safe = _rank_score(out, "downside_volatility_60d", higher_is_better=False)
    out["long_trend_safety_score"] = trend_position * 0.60 + momentum * 0.40
    out["rule_long_model_score"] = (
        out["enhanced_good_stock_score"] * 0.50
        + out["blended_value_score"] * 0.30
        + out["long_trend_safety_score"] * 0.10
        + downside_safe * 0.10
    )

    sufficient_history = numeric("annual_history_years").ge(3)
    cashflow_gate = (
        numeric("cashflow_quality_3y").ge(0.80)
        & numeric("cfo_positive_share_5y").ge(0.60)
    )
    goodwill_to_assets = numeric("annual_goodwill_to_assets")
    asset_gate = goodwill_to_assets.lt(0.30) | goodwill_to_assets.isna()
    profit_gate = numeric("profit_positive_share_5y").ge(0.80)
    out["cashflow_gate_08"] = financial | (sufficient_history & cashflow_gate)
    out["durability_gate"] = financial | (
        sufficient_history & cashflow_gate & profit_gate & asset_gate
    )
    return out.replace([np.inf, -np.inf], np.nan)
