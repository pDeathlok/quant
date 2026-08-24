"""Point-in-time overlays for the deep-base blood-chip research strategy.

The price layer remains frozen.  This module only asks whether information
known at the signal close -- financial durability, valuation, capital
pressure, broad-market repair and a bias-disclosed industry diagnostic -- can
remove terminally impaired bases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


POLICY_GATES: dict[str, tuple[str, ...]] = {
    "price_only": (),
    "survival": ("survival_gate",),
    "survival_value": ("survival_gate", "value_scale_gate"),
    "survival_capital": ("survival_gate", "capital_pressure_gate"),
    "survival_market": ("survival_gate", "market_repair_gate"),
    "auditable_combined": (
        "survival_gate",
        "value_scale_gate",
        "capital_pressure_gate",
        "market_repair_gate",
    ),
    "current_industry_diagnostic": ("survival_gate", "current_industry_repair_gate"),
    "combined_current_industry_diagnostic": (
        "survival_gate",
        "value_scale_gate",
        "capital_pressure_gate",
        "market_repair_gate",
        "current_industry_repair_gate",
    ),
}

POLICY_SCORE_COLUMNS: dict[str, tuple[str, ...]] = {
    "price_only": (),
    "survival": ("survival_score",),
    "survival_value": ("survival_score", "value_scale_score"),
    "survival_capital": ("survival_score", "capital_pressure_score"),
    "survival_market": ("survival_score", "market_repair_score"),
    "auditable_combined": (
        "survival_score",
        "value_scale_score",
        "capital_pressure_score",
        "market_repair_score",
    ),
    "current_industry_diagnostic": (
        "survival_score",
        "current_industry_repair_score",
    ),
    "combined_current_industry_diagnostic": (
        "survival_score",
        "value_scale_score",
        "capital_pressure_score",
        "market_repair_score",
        "current_industry_repair_score",
    ),
}

INDUSTRY_DIAGNOSTIC_POLICIES = frozenset(
    {"current_industry_diagnostic", "combined_current_industry_diagnostic"}
)
SELECTION_ELIGIBLE_POLICIES = frozenset(
    {
        "survival",
        "survival_value",
        "survival_capital",
        "survival_market",
        "auditable_combined",
    }
)


@dataclass(frozen=True)
class MultidimensionalGateConfig:
    """Frozen absolute gates for the multidimensional research iteration."""

    maximum_financial_age_days: int = 550
    minimum_annual_history_years: int = 3
    minimum_profit_positive_share: float = 0.60
    minimum_cfo_positive_share: float = 0.60
    minimum_total_market_value_ten_thousand: float = 100_000.0
    maximum_pb: float = 4.0
    maximum_ps_ttm: float = 3.0
    maximum_free_turnover_rate: float = 12.0
    maximum_pledge_ratio: float = 50.0
    minimum_holder_net_change_ratio_180d: float = -2.0
    minimum_market_return_20d: float = -0.08
    minimum_market_return_120d: float = -0.18
    minimum_market_close_to_ma250: float = 0.85
    minimum_industry_return_20d: float = -0.05
    minimum_industry_positive_share_20d: float = 0.35
    minimum_industry_constituents: int = 5
    price_score_weight: float = 0.65

    def __post_init__(self) -> None:
        if self.maximum_financial_age_days < 1:
            raise ValueError("maximum_financial_age_days must be positive")
        if self.minimum_annual_history_years < 1:
            raise ValueError("minimum_annual_history_years must be positive")
        if not 0.0 < self.price_score_weight <= 1.0:
            raise ValueError("price_score_weight must be in (0, 1]")
        for name in ("minimum_profit_positive_share", "minimum_cfo_positive_share"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_date(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True).str[:8]
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(values, errors="coerce")
    return parsed.fillna(fallback)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    source = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(source, errors="coerce")


def _merge_asof_by_symbol(
    signals: pd.DataFrame,
    events: pd.DataFrame,
    *,
    signal_date_column: str,
    event_date_column: str,
    allow_exact_matches: bool,
) -> pd.DataFrame:
    """Merge dated events without requiring global merge-asof sort quirks."""

    left = signals.copy().reset_index(drop=True)
    left[signal_date_column] = pd.to_datetime(left[signal_date_column], errors="coerce")
    left["_multidim_row_id"] = np.arange(len(left))
    right = events.copy()
    if right.empty:
        return left.drop(columns="_multidim_row_id")
    right[event_date_column] = pd.to_datetime(right[event_date_column], errors="coerce")
    right = right.dropna(subset=["ts_code", event_date_column])
    right_by_symbol = {
        str(code): group.sort_values(event_date_column).drop_duplicates(
            event_date_column, keep="last"
        )
        for code, group in right.groupby("ts_code", observed=True, sort=False)
    }
    parts: list[pd.DataFrame] = []
    for code, group in left.groupby("ts_code", observed=True, sort=False):
        event_group = right_by_symbol.get(str(code))
        if event_group is None or event_group.empty:
            parts.append(group.copy())
            continue
        event_group = event_group.drop(columns="ts_code", errors="ignore")
        parts.append(
            pd.merge_asof(
                group.sort_values(signal_date_column),
                event_group,
                left_on=signal_date_column,
                right_on=event_date_column,
                direction="backward",
                allow_exact_matches=allow_exact_matches,
            )
        )
    return (
        pd.concat(parts, ignore_index=True, sort=False)
        .sort_values("_multidim_row_id")
        .drop(columns="_multidim_row_id")
        .reset_index(drop=True)
    )


def merge_financial_survival_asof(
    signals: pd.DataFrame,
    annual_events: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the latest fully published annual-quality event to each signal."""

    if "signal_date" not in signals or "ts_code" not in signals:
        raise ValueError("signals require ts_code and signal_date")
    if annual_events.empty:
        out = signals.copy()
        out["financial_coverage"] = False
        return out
    required = {"ts_code", "annual_quality_available_at", "period_end"}
    missing = sorted(required - set(annual_events.columns))
    if missing:
        raise ValueError(f"annual events missing columns: {missing}")
    out = _merge_asof_by_symbol(
        signals,
        annual_events,
        signal_date_column="signal_date",
        event_date_column="annual_quality_available_at",
        allow_exact_matches=True,
    )
    out["financial_coverage"] = out["annual_quality_available_at"].notna()
    out["financial_age_days"] = (
        pd.to_datetime(out["signal_date"], errors="coerce")
        - pd.to_datetime(out["period_end"], errors="coerce")
    ).dt.days
    return out


def merge_daily_basic_on_signal_date(
    signals: pd.DataFrame,
    daily_basic: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the exact signal-close daily-basic cross-section."""

    out = signals.copy()
    if daily_basic.empty:
        out["daily_basic_coverage"] = False
        return out
    required = {"ts_code", "trade_date"}
    missing = sorted(required - set(daily_basic.columns))
    if missing:
        raise ValueError(f"daily_basic missing columns: {missing}")
    basic = daily_basic.copy()
    basic["signal_date"] = _normalize_date(basic["trade_date"])
    value_columns = [
        column
        for column in (
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
        )
        if column in basic.columns
    ]
    basic = basic[["ts_code", "signal_date", *value_columns]].drop_duplicates(
        ["ts_code", "signal_date"], keep="last"
    )
    basic = basic.rename(columns={column: f"basic_{column}" for column in value_columns})
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out = out.merge(basic, on=["ts_code", "signal_date"], how="left", validate="many_to_one")
    observed = [column for column in out.columns if column.startswith("basic_")]
    out["daily_basic_coverage"] = out[observed].notna().any(axis=1) if observed else False
    return out


def _holder_window_features(signals: pd.DataFrame, holder_trade: pd.DataFrame) -> pd.DataFrame:
    out = signals[["ts_code", "signal_date"]].copy().reset_index(drop=True)
    net_values = np.zeros(len(out), dtype=float)
    event_counts = np.zeros(len(out), dtype=int)
    latest_dates = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
    if holder_trade.empty:
        return pd.DataFrame(
            {
                "holder_net_change_ratio_180d": net_values,
                "holder_event_count_180d": event_counts,
                "holder_latest_available_at": latest_dates,
            }
        )
    events = holder_trade.copy()
    events["holder_available_at"] = _normalize_date(events["ann_date"])
    direction = events["in_de"].astype("string").str.upper().map({"IN": 1.0, "DE": -1.0})
    ratio = pd.to_numeric(events["change_ratio"], errors="coerce").abs()
    events["signed_change_ratio"] = direction * ratio
    events = (
        events.dropna(subset=["ts_code", "holder_available_at", "signed_change_ratio"])
        .groupby(["ts_code", "holder_available_at"], as_index=False, observed=True)[
            "signed_change_ratio"
        ]
        .sum()
    )
    event_groups = {
        str(code): group.sort_values("holder_available_at")
        for code, group in events.groupby("ts_code", observed=True, sort=False)
    }
    for code, signal_group in out.groupby("ts_code", observed=True, sort=False):
        group = event_groups.get(str(code))
        if group is None or group.empty:
            continue
        event_dates = group["holder_available_at"].to_numpy(dtype="datetime64[ns]")
        signed = group["signed_change_ratio"].to_numpy(dtype=float)
        cumulative = np.cumsum(signed)
        signal_dates = pd.to_datetime(signal_group["signal_date"]).to_numpy(dtype="datetime64[ns]")
        starts = signal_dates - np.timedelta64(180, "D")
        lo = np.searchsorted(event_dates, starts, side="right")
        hi = np.searchsorted(event_dates, signal_dates, side="right")
        totals = np.where(hi > 0, cumulative[np.maximum(hi - 1, 0)], 0.0)
        before = np.where(lo > 0, cumulative[np.maximum(lo - 1, 0)], 0.0)
        values = np.where(hi > lo, totals - before, 0.0)
        indices = signal_group.index.to_numpy()
        net_values[indices] = values
        event_counts[indices] = hi - lo
        valid = hi > 0
        latest = np.full(len(hi), np.datetime64("NaT"), dtype="datetime64[ns]")
        latest[valid] = event_dates[hi[valid] - 1]
        latest_dates[indices] = latest
    return pd.DataFrame(
        {
            "holder_net_change_ratio_180d": net_values,
            "holder_event_count_180d": event_counts,
            "holder_latest_available_at": latest_dates,
        }
    )


def merge_capital_pressure_asof(
    signals: pd.DataFrame,
    pledge_stat: pd.DataFrame,
    holder_trade: pd.DataFrame,
) -> pd.DataFrame:
    """Attach conservative pledge and 180-day disclosed holder-flow evidence."""

    out = signals.copy().reset_index(drop=True)
    if not pledge_stat.empty:
        required = {"ts_code", "end_date", "pledge_ratio"}
        missing = sorted(required - set(pledge_stat.columns))
        if missing:
            raise ValueError(f"pledge_stat missing columns: {missing}")
        pledge = pledge_stat[list(required)].copy()
        pledge["pledge_available_at"] = _normalize_date(pledge["end_date"])
        pledge["pledge_ratio"] = pd.to_numeric(pledge["pledge_ratio"], errors="coerce")
        pledge = (
            pledge.dropna(subset=["ts_code", "pledge_available_at", "pledge_ratio"])
            .groupby(["ts_code", "pledge_available_at"], as_index=False, observed=True)[
                "pledge_ratio"
            ]
            .max()
        )
        out = _merge_asof_by_symbol(
            out,
            pledge,
            signal_date_column="signal_date",
            event_date_column="pledge_available_at",
            allow_exact_matches=False,
        )
    if "pledge_ratio" not in out:
        out["pledge_ratio"] = np.nan
    if "pledge_available_at" not in out:
        out["pledge_available_at"] = pd.NaT
    out["pledge_observed"] = out["pledge_ratio"].notna()
    holder = _holder_window_features(out, holder_trade)
    out = pd.concat([out.reset_index(drop=True), holder.reset_index(drop=True)], axis=1)
    out["holder_activity_observed"] = out["holder_event_count_180d"].gt(0)
    return out


def add_market_repair_features(
    signals: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Attach same-close CSI300 trailing returns and long moving-average state."""

    market = benchmark.copy()
    if "trade_date" not in market or "close" not in market:
        raise ValueError("benchmark requires trade_date and close")
    market["signal_date"] = _normalize_date(market["trade_date"])
    market = market.dropna(subset=["signal_date"]).sort_values("signal_date")
    market["market_close"] = pd.to_numeric(market["close"], errors="coerce")
    market["market_return_20d"] = market["market_close"].pct_change(20, fill_method=None)
    market["market_return_120d"] = market["market_close"].pct_change(120, fill_method=None)
    market["market_ma250"] = market["market_close"].rolling(250, min_periods=250).mean()
    market["market_close_to_ma250"] = market["market_close"] / market["market_ma250"]
    values = market[
        [
            "signal_date",
            "market_close",
            "market_return_20d",
            "market_return_120d",
            "market_ma250",
            "market_close_to_ma250",
        ]
    ].drop_duplicates("signal_date", keep="last")
    out = signals.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out = out.merge(values, on="signal_date", how="left", validate="many_to_one")
    out["market_coverage"] = out["market_close_to_ma250"].notna()
    return out


def add_current_industry_repair_features(
    signals: pd.DataFrame,
    price_features: pd.DataFrame,
    stock_basic: pd.DataFrame,
) -> pd.DataFrame:
    """Add a current-mapping industry diagnostic that is never selection-safe."""

    required_features = {"ts_code", "date", "return_20d"}
    missing = sorted(required_features - set(price_features.columns))
    if missing:
        raise ValueError(f"price features missing columns: {missing}")
    if not {"ts_code", "industry"}.issubset(stock_basic.columns):
        raise ValueError("stock_basic requires ts_code and industry")
    out = signals.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    mapping = stock_basic[["ts_code", "industry"]].drop_duplicates("ts_code", keep="last")
    mapping = mapping.rename(columns={"industry": "current_industry"})
    out = out.merge(mapping, on="ts_code", how="left", validate="many_to_one")
    needed_dates = pd.Index(out["signal_date"].dropna().unique())
    panel = price_features.loc[
        pd.to_datetime(price_features["date"], errors="coerce").isin(needed_dates),
        ["ts_code", "date", "return_20d"],
    ].copy()
    panel["signal_date"] = pd.to_datetime(panel.pop("date"), errors="coerce")
    panel = panel.merge(mapping, on="ts_code", how="left", validate="many_to_one")
    panel["return_20d"] = pd.to_numeric(panel["return_20d"], errors="coerce")
    panel = panel.dropna(subset=["signal_date", "current_industry", "return_20d"])
    panel["industry_positive_20d"] = panel["return_20d"].gt(0.0).astype(float)
    aggregates = (
        panel.groupby(["signal_date", "current_industry"], as_index=False, observed=True)
        .agg(
            current_industry_return_20d=("return_20d", "median"),
            current_industry_positive_share_20d=("industry_positive_20d", "mean"),
            current_industry_constituents=("return_20d", "count"),
        )
    )
    out = out.merge(
        aggregates,
        on=["signal_date", "current_industry"],
        how="left",
        validate="many_to_one",
    )
    out["current_industry_mapping_bias"] = True
    return out


def _mean_fixed_components(components: list[pd.Series]) -> pd.Series:
    if not components:
        raise ValueError("at least one component is required")
    numeric = [pd.to_numeric(component, errors="coerce").fillna(0.0) for component in components]
    return sum(numeric) / len(numeric)


def apply_multidimensional_gates(
    signals: pd.DataFrame,
    config: MultidimensionalGateConfig,
) -> pd.DataFrame:
    """Build explicit gates and bounded dimension scores without imputation-to-pass."""

    out = signals.copy()
    signal_date = pd.to_datetime(out["signal_date"], errors="coerce")
    available_source = (
        out["annual_quality_available_at"]
        if "annual_quality_available_at" in out
        else pd.Series(pd.NaT, index=out.index)
    )
    available_at = pd.to_datetime(available_source, errors="coerce")
    out["financial_point_in_time_ok"] = available_at.le(signal_date).fillna(False)
    financial_age = _numeric(out, "financial_age_days")
    annual_history = _numeric(out, "annual_history_years")
    profit_share = _numeric(out, "profit_positive_share_5y")
    cfo_share = _numeric(out, "cfo_positive_share_5y")
    latest_profit = _numeric(out, "income_n_income_attr_p")
    latest_cfo = _numeric(out, "cashflow_n_cashflow_act")
    out["survival_gate"] = (
        out["financial_point_in_time_ok"]
        & financial_age.between(0, config.maximum_financial_age_days, inclusive="both")
        & annual_history.ge(config.minimum_annual_history_years)
        & profit_share.ge(config.minimum_profit_positive_share)
        & cfo_share.ge(config.minimum_cfo_positive_share)
        & latest_profit.gt(0.0)
        & latest_cfo.gt(0.0)
    ).fillna(False)
    freshness = (1.0 - financial_age / config.maximum_financial_age_days).clip(0.0, 1.0)
    out["survival_score"] = _mean_fixed_components(
        [
            freshness.where(out["financial_point_in_time_ok"]),
            profit_share.clip(0.0, 1.0),
            cfo_share.clip(0.0, 1.0),
            latest_profit.gt(0.0).where(latest_profit.notna()).astype(float),
            latest_cfo.gt(0.0).where(latest_cfo.notna()).astype(float),
        ]
    )

    total_mv = _numeric(out, "basic_total_mv")
    pb = _numeric(out, "basic_pb")
    ps_ttm = _numeric(out, "basic_ps_ttm")
    turnover_f = _numeric(out, "basic_turnover_rate_f")
    cheap_enough = (pb.gt(0.0) & pb.le(config.maximum_pb)) | (
        ps_ttm.gt(0.0) & ps_ttm.le(config.maximum_ps_ttm)
    )
    out["value_scale_gate"] = (
        out.get("daily_basic_coverage", False)
        & total_mv.ge(config.minimum_total_market_value_ten_thousand)
        & cheap_enough
        & turnover_f.ge(0.0)
        & turnover_f.le(config.maximum_free_turnover_rate)
    ).fillna(False)
    pb_score = (1.0 - pb / config.maximum_pb).clip(0.0, 1.0).where(pb.gt(0.0))
    ps_score = (1.0 - ps_ttm / config.maximum_ps_ttm).clip(0.0, 1.0).where(ps_ttm.gt(0.0))
    cheap_score = pd.concat([pb_score, ps_score], axis=1).max(axis=1, skipna=True)
    scale_score = (
        np.log10(total_mv / config.minimum_total_market_value_ten_thousand)
        .replace([np.inf, -np.inf], np.nan)
        .add(1.0)
        .div(2.0)
        .clip(0.0, 1.0)
        .where(total_mv.gt(0.0))
    )
    turnover_score = (1.0 - turnover_f / config.maximum_free_turnover_rate).clip(0.0, 1.0)
    out["value_scale_score"] = _mean_fixed_components(
        [cheap_score, scale_score, turnover_score]
    )

    pledge = _numeric(out, "pledge_ratio")
    holder_net = _numeric(out, "holder_net_change_ratio_180d").fillna(0.0)
    pledge_pass = pledge.le(config.maximum_pledge_ratio) | pledge.isna()
    out["capital_pressure_gate"] = (
        pledge_pass & holder_net.ge(config.minimum_holder_net_change_ratio_180d)
    ).fillna(False)
    pledge_score = (1.0 - pledge / config.maximum_pledge_ratio).clip(0.0, 1.0)
    # Missing pledge observations pass the risk gate but receive no ranking bonus.
    pledge_score = pledge_score.where(pledge.notna())
    holder_score = (
        (holder_net - config.minimum_holder_net_change_ratio_180d)
        / abs(config.minimum_holder_net_change_ratio_180d)
    ).clip(0.0, 1.0)
    holder_score = holder_score.where(_numeric(out, "holder_event_count_180d").gt(0))
    out["capital_pressure_score"] = _mean_fixed_components([pledge_score, holder_score])

    market_20 = _numeric(out, "market_return_20d")
    market_120 = _numeric(out, "market_return_120d")
    market_ma = _numeric(out, "market_close_to_ma250")
    out["market_repair_gate"] = (
        out.get("market_coverage", False)
        & market_20.ge(config.minimum_market_return_20d)
        & market_120.ge(config.minimum_market_return_120d)
        & market_ma.ge(config.minimum_market_close_to_ma250)
    ).fillna(False)
    out["market_repair_score"] = _mean_fixed_components(
        [
            ((market_20 - config.minimum_market_return_20d) / 0.16).clip(0.0, 1.0),
            ((market_120 - config.minimum_market_return_120d) / 0.36).clip(0.0, 1.0),
            ((market_ma - config.minimum_market_close_to_ma250) / 0.30).clip(0.0, 1.0),
        ]
    )

    industry_return = _numeric(out, "current_industry_return_20d")
    industry_positive = _numeric(out, "current_industry_positive_share_20d")
    industry_count = _numeric(out, "current_industry_constituents")
    out["current_industry_repair_gate"] = (
        industry_count.ge(config.minimum_industry_constituents)
        & industry_return.ge(config.minimum_industry_return_20d)
        & industry_positive.ge(config.minimum_industry_positive_share_20d)
    ).fillna(False)
    out["current_industry_repair_score"] = _mean_fixed_components(
        [
            (
                (industry_return - config.minimum_industry_return_20d) / 0.20
            ).clip(0.0, 1.0),
            (
                (industry_positive - config.minimum_industry_positive_share_20d) / 0.50
            ).clip(0.0, 1.0),
        ]
    ).where(industry_count.ge(config.minimum_industry_constituents), 0.0)
    out["current_industry_mapping_bias"] = True
    out["auditable_combined_gate"] = (
        out["survival_gate"]
        & out["value_scale_gate"]
        & out["capital_pressure_gate"]
        & out["market_repair_gate"]
    )
    return out.replace([np.inf, -np.inf], np.nan)


def signals_for_policy(
    enriched_signals: pd.DataFrame,
    policy: str,
    config: MultidimensionalGateConfig,
) -> pd.DataFrame:
    """Filter one pre-registered policy and blend only its visible dimensions."""

    if policy not in POLICY_GATES:
        raise ValueError(f"unknown multidimensional policy: {policy}")
    mask = pd.Series(True, index=enriched_signals.index)
    for gate in POLICY_GATES[policy]:
        if gate not in enriched_signals:
            raise ValueError(f"enriched signals missing policy gate: {gate}")
        mask &= enriched_signals[gate].fillna(False).astype(bool)
    out = enriched_signals.loc[mask].copy()
    price_score = _numeric(out, "signal_score").clip(0.0, 1.0)
    out["price_signal_score"] = price_score
    score_columns = POLICY_SCORE_COLUMNS[policy]
    if score_columns:
        overlay = _mean_fixed_components([_numeric(out, column) for column in score_columns])
        out["multidimensional_overlay_score"] = overlay
        out["signal_score"] = (
            price_score * config.price_score_weight
            + overlay * (1.0 - config.price_score_weight)
        )
    else:
        out["multidimensional_overlay_score"] = np.nan
        out["signal_score"] = price_score
    out["research_policy"] = policy
    out["selection_eligible_policy"] = policy in SELECTION_ELIGIBLE_POLICIES
    out["current_industry_mapping_bias"] = policy in INDUSTRY_DIAGNOSTIC_POLICIES
    return out.reset_index(drop=True)
