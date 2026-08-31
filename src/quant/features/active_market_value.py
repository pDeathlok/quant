"""Transparent active-market-value proxies for stocks, markets, and indices.

Compass 0AMV is proprietary.  This module intentionally exposes a reproducible
proxy based on Tushare turnover and float market value instead of claiming
formula parity with the vendor index.  Every rolling calculation is causal and
every volume baseline excludes the current session.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Sequence

import numpy as np
import pandas as pd


ACTIVE_MARKET_VALUE_FEATURE_SCHEMA_VERSION = (
    "active_market_value_proxy_v1_20260831"
)
ACTIVE_SHARE_DECAY = 12.0 / 13.0
MIN_INDEX_CONSTITUENT_COVERAGE = 0.90


@dataclass(frozen=True)
class KeyIndexSpec:
    """Stable feature prefix and accepted Tushare codes for one key index."""

    slug: str
    ts_code: str
    aliases: tuple[str, ...] = ()


KEY_INDEX_SPECS: tuple[KeyIndexSpec, ...] = (
    KeyIndexSpec("sse_composite", "000001.SH"),
    KeyIndexSpec("szse_component", "399001.SZ"),
    KeyIndexSpec("chinext", "399006.SZ"),
    KeyIndexSpec("star50", "000688.SH"),
    KeyIndexSpec("sse50", "000016.SH"),
    KeyIndexSpec("csi300", "000300.SH", ("399300.SZ",)),
    KeyIndexSpec("csi500", "000905.SH"),
    KeyIndexSpec("csi1000", "000852.SH"),
    KeyIndexSpec("bse50", "899050.BJ"),
)

STOCK_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS: tuple[str, ...] = (
    "stock_active_share_ratio_13d_proxy",
    "stock_active_mv_proxy_cny",
    "stock_active_mv_log",
    "stock_active_mv_return_1d_pct",
    "stock_active_mv_return_5d_pct",
)
MARKET_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_active_mv_proxy_cny",
    "market_active_mv_log",
    "market_active_mv_ratio_proxy",
    "market_active_mv_return_1d_pct",
    "market_active_mv_return_5d_pct",
    "market_volume_ratio_prev20",
    "market_amount_ratio_prev20",
)
INDEX_FEATURE_SUFFIXES: tuple[str, ...] = (
    "return_1d_pct",
    "return_5d_pct",
    "volume_ratio_prev20",
    "amount_ratio_prev20",
    "active_share_ratio_13d_proxy",
    "active_mv_proxy_cny",
    "active_mv_log",
    "active_mv_return_1d_pct",
    "active_mv_return_5d_pct",
    "active_mv_coverage_ratio",
)
KEY_INDEX_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"index_{spec.slug}_{suffix}"
    for spec in KEY_INDEX_SPECS
    for suffix in INDEX_FEATURE_SUFFIXES
)
ACTIVE_MARKET_VALUE_RESEARCH_FEATURE_COLUMNS: tuple[str, ...] = (
    *STOCK_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS,
    *MARKET_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS,
    *KEY_INDEX_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class ActiveMarketValueFrames:
    """The stock-keyed and date-keyed outputs of the three-level calculator."""

    stock: pd.DataFrame
    market: pd.DataFrame
    indices: pd.DataFrame


def _normalize_keys(frame: pd.DataFrame, *, require_symbol: bool) -> pd.DataFrame:
    out = frame.copy()
    if "ts_code" not in out.columns and "symbol" in out.columns:
        out["ts_code"] = out["symbol"].astype(str)
    if require_symbol and "ts_code" not in out.columns:
        raise ValueError("active market value input misses ts_code/symbol")
    if "trade_date" not in out.columns and "date" in out.columns:
        out["trade_date"] = pd.to_datetime(
            out["date"], errors="coerce"
        ).dt.strftime("%Y%m%d")
    if "trade_date" not in out.columns:
        raise ValueError("active market value input misses trade_date/date")
    parsed = pd.to_datetime(
        out["trade_date"].astype(str).str.replace("-", "", regex=False),
        format="%Y%m%d",
        errors="coerce",
    )
    if parsed.isna().any():
        raise ValueError("active market value input has invalid trade_date values")
    out["trade_date"] = parsed.dt.strftime("%Y%m%d")
    out["date"] = parsed.dt.normalize()
    keys = ["trade_date"]
    if require_symbol:
        out["ts_code"] = out["ts_code"].astype(str)
        keys.insert(0, "ts_code")
    if out.duplicated(keys).any():
        raise ValueError(f"active market value input has duplicate keys: {keys}")
    return out.sort_values([*keys[:-1], "trade_date"], kind="stable").reset_index(
        drop=True
    )


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _safe_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.log(numeric.where(numeric > 0))


def _pct_change(values: pd.Series, periods: int) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").pct_change(
        periods, fill_method=None
    ) * 100.0


def _ratio_to_previous_mean(values: pd.Series, window: int = 20) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    denominator = numeric.shift(1).rolling(window, min_periods=window).mean()
    return numeric.div(denominator.replace(0, np.nan))


def compute_active_share_ratio(
    turnover_rate_pct: pd.Series,
    *,
    decay: float = ACTIVE_SHARE_DECAY,
) -> pd.Series:
    """Estimate the active float share with a causal CYF13-style recursion.

    Turnover is expressed in percentage points.  Missing observations publish
    ``NaN`` for that row but do not erase the last known state.  Turnover above
    100% is capped at one because this state is a share of float capital.
    """

    if not 0.0 <= decay < 1.0:
        raise ValueError("active share decay must be in [0, 1)")
    turnover = pd.to_numeric(turnover_rate_pct, errors="coerce") / 100.0
    result = pd.Series(np.nan, index=turnover.index, dtype=float)
    state = np.nan
    for index, raw_value in turnover.items():
        if not np.isfinite(raw_value):
            continue
        current = float(np.clip(raw_value, 0.0, 1.0))
        state = (
            current
            if not np.isfinite(state)
            else current + decay * (1.0 - current) * state
        )
        result.loc[index] = float(np.clip(state, 0.0, 1.0))
    return result


def compute_stock_active_market_value_features(
    daily_basic: pd.DataFrame,
) -> pd.DataFrame:
    """Return stock/date active-market-value proxy features.

    Tushare ``circ_mv`` is in ten-thousand CNY and ``turnover_rate`` is in
    percentage points.  The published active market value is standardized to
    CNY before logarithms and changes are calculated.
    """

    frame = _normalize_keys(daily_basic, require_symbol=True)
    required = {"turnover_rate", "circ_mv"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"stock active market value input misses columns: {missing}")
    frame["turnover_rate"] = _numeric(frame, "turnover_rate")
    frame["circ_mv"] = _numeric(frame, "circ_mv")
    frame["stock_active_share_ratio_13d_proxy"] = frame.groupby(
        "ts_code", sort=False
    )["turnover_rate"].transform(compute_active_share_ratio)
    frame["stock_active_mv_proxy_cny"] = (
        frame["circ_mv"]
        * 10_000.0
        * frame["stock_active_share_ratio_13d_proxy"]
    )
    frame["stock_active_mv_log"] = _safe_log(
        frame["stock_active_mv_proxy_cny"]
    )
    grouped = frame.groupby("ts_code", sort=False)["stock_active_mv_proxy_cny"]
    frame["stock_active_mv_return_1d_pct"] = grouped.transform(
        lambda values: _pct_change(values, 1)
    )
    frame["stock_active_mv_return_5d_pct"] = grouped.transform(
        lambda values: _pct_change(values, 5)
    )
    return frame[
        [
            "ts_code",
            "trade_date",
            "date",
            *STOCK_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS,
        ]
    ].replace([np.inf, -np.inf], np.nan)


def compute_market_active_market_value_features(
    daily_basic: pd.DataFrame,
    *,
    market_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate stock proxies into whole-market date-level features."""

    basic = _normalize_keys(daily_basic, require_symbol=True)
    stock = compute_stock_active_market_value_features(basic)
    stock_source = stock.merge(
        basic[["ts_code", "trade_date", "circ_mv"]],
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    active = stock_source.groupby("trade_date", sort=True)[
        "stock_active_mv_proxy_cny"
    ].sum(min_count=1)
    float_mv = (
        pd.to_numeric(stock_source["circ_mv"], errors="coerce")
        .groupby(stock_source["trade_date"])
        .sum(min_count=1)
        * 10_000.0
    )
    market = pd.DataFrame(
        {
            "trade_date": active.index,
            "market_active_mv_proxy_cny": active.to_numpy(dtype=float),
            "_market_float_mv_cny": float_mv.reindex(active.index).to_numpy(
                dtype=float
            ),
        }
    )
    market["date"] = pd.to_datetime(market["trade_date"], format="%Y%m%d")
    market["market_active_mv_log"] = _safe_log(
        market["market_active_mv_proxy_cny"]
    )
    market["market_active_mv_ratio_proxy"] = market[
        "market_active_mv_proxy_cny"
    ].div(market["_market_float_mv_cny"].replace(0, np.nan))
    market["market_active_mv_return_1d_pct"] = _pct_change(
        market["market_active_mv_proxy_cny"], 1
    )
    market["market_active_mv_return_5d_pct"] = _pct_change(
        market["market_active_mv_proxy_cny"], 5
    )

    if market_daily is not None and not market_daily.empty:
        daily = _normalize_keys(market_daily, require_symbol=True)
        volume_column = "volume" if "volume" in daily.columns else "vol"
        amount_column = "amount" if "amount" in daily.columns else "turnover"
        aggregates = pd.DataFrame({"trade_date": sorted(daily["trade_date"].unique())})
        for source, target in (
            (volume_column, "_market_volume"),
            (amount_column, "_market_amount"),
        ):
            if source not in daily.columns:
                aggregates[target] = np.nan
                continue
            values = pd.to_numeric(daily[source], errors="coerce")
            grouped_values = values.groupby(daily["trade_date"]).sum(min_count=1)
            aggregates[target] = aggregates["trade_date"].map(grouped_values)
        market = market.merge(
            aggregates,
            on="trade_date",
            how="left",
            validate="one_to_one",
        )
        market["market_volume_ratio_prev20"] = _ratio_to_previous_mean(
            market["_market_volume"]
        )
        market["market_amount_ratio_prev20"] = _ratio_to_previous_mean(
            market["_market_amount"]
        )
    else:
        market["market_volume_ratio_prev20"] = np.nan
        market["market_amount_ratio_prev20"] = np.nan

    return market[
        ["trade_date", "date", *MARKET_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS]
    ].replace([np.inf, -np.inf], np.nan)


def _canonical_index_codes() -> dict[str, str]:
    return {
        code: spec.ts_code
        for spec in KEY_INDEX_SPECS
        for code in (spec.ts_code, *spec.aliases)
    }


def _constituent_index_active_market_value(
    constituent_daily_basic: pd.DataFrame | None,
    index_weights: pd.DataFrame | None,
    target_dates: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Aggregate point-in-time constituent active values for index fallbacks."""

    if (
        constituent_daily_basic is None
        or constituent_daily_basic.empty
        or index_weights is None
        or index_weights.empty
    ):
        return {}
    stock_basic = _normalize_keys(constituent_daily_basic, require_symbol=True)
    stock_features = compute_stock_active_market_value_features(stock_basic)
    stock_values = stock_features.merge(
        stock_basic[["ts_code", "trade_date", "circ_mv"]],
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    stock_values["_float_mv_cny"] = _numeric(stock_values, "circ_mv") * 10_000.0

    weights = index_weights.copy()
    if "index_code" not in weights.columns and "ts_code" in weights.columns:
        weights = weights.rename(columns={"ts_code": "index_code"})
    if "con_code" not in weights.columns and "constituent_code" in weights.columns:
        weights = weights.rename(columns={"constituent_code": "con_code"})
    required = {"index_code", "con_code", "trade_date"}
    missing = sorted(required - set(weights.columns))
    if missing:
        raise ValueError(f"index weights input misses columns: {missing}")
    parsed = pd.to_datetime(
        weights["trade_date"].astype(str).str.replace("-", "", regex=False),
        format="%Y%m%d",
        errors="coerce",
    )
    if parsed.isna().any():
        raise ValueError("index weights input has invalid trade_date values")
    weights["trade_date"] = parsed.dt.strftime("%Y%m%d")
    weights["index_code"] = (
        weights["index_code"].astype(str).map(_canonical_index_codes()).fillna(
            weights["index_code"].astype(str)
        )
    )
    weights["con_code"] = weights["con_code"].astype(str)
    if weights.duplicated(["index_code", "con_code", "trade_date"]).any():
        raise ValueError("index weights input has duplicate index/constituent/date keys")
    weights["_member_weight"] = _numeric(weights, "weight")

    fallbacks: dict[str, pd.DataFrame] = {}
    for spec in KEY_INDEX_SPECS:
        spec_weights = weights[weights["index_code"].eq(spec.ts_code)].copy()
        spec_dates = target_dates[target_dates["ts_code"].eq(spec.ts_code)][
            ["trade_date"]
        ].drop_duplicates()
        if spec_weights.empty or spec_dates.empty:
            continue
        spec_dates = spec_dates.sort_values("trade_date").reset_index(drop=True)
        spec_dates["_target_datetime"] = pd.to_datetime(
            spec_dates["trade_date"], format="%Y%m%d"
        )
        snapshots = pd.DataFrame(
            {"_weight_date": sorted(spec_weights["trade_date"].unique())}
        )
        snapshots["_weight_datetime"] = pd.to_datetime(
            snapshots["_weight_date"], format="%Y%m%d"
        )
        date_map = pd.merge_asof(
            spec_dates,
            snapshots,
            left_on="_target_datetime",
            right_on="_weight_datetime",
            direction="backward",
        ).dropna(subset=["_weight_datetime"])
        memberships = date_map.merge(
            spec_weights.rename(columns={"trade_date": "_weight_date"}),
            on="_weight_date",
            how="left",
            validate="many_to_many",
        )
        memberships = memberships.merge(
            stock_values[
                [
                    "ts_code",
                    "trade_date",
                    "stock_active_mv_proxy_cny",
                    "_float_mv_cny",
                ]
            ],
            left_on=["con_code", "trade_date"],
            right_on=["ts_code", "trade_date"],
            how="left",
            validate="many_to_one",
        )
        available = memberships["stock_active_mv_proxy_cny"].notna()
        weighted = memberships["_member_weight"].notna().all()
        if weighted:
            total_weight = memberships.groupby("trade_date")["_member_weight"].sum(
                min_count=1
            )
            available_weight = memberships["_member_weight"].where(available).groupby(
                memberships["trade_date"]
            ).sum(min_count=1)
            coverage = available_weight.div(total_weight.replace(0, np.nan))
        else:
            coverage = available.groupby(memberships["trade_date"]).mean()
        active_mv = memberships["stock_active_mv_proxy_cny"].where(available).groupby(
            memberships["trade_date"]
        ).sum(min_count=1)
        float_mv = memberships["_float_mv_cny"].where(available).groupby(
            memberships["trade_date"]
        ).sum(min_count=1)
        complete = coverage.ge(MIN_INDEX_CONSTITUENT_COVERAGE)
        active_mv = active_mv.where(complete)
        active_ratio = active_mv.div(float_mv.replace(0, np.nan)).where(complete)
        fallbacks[spec.ts_code] = pd.DataFrame(
            {
                "trade_date": coverage.index,
                "_fallback_active_share_ratio": active_ratio.reindex(coverage.index),
                "_fallback_active_mv_cny": active_mv.reindex(coverage.index),
                "_fallback_coverage_ratio": coverage,
            }
        ).reset_index(drop=True)
    return fallbacks


def _empty_index_feature_frame(dates: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame({"trade_date": list(dates)})
    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    for column in KEY_INDEX_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS:
        out[column] = np.nan
    return out


def compute_key_index_active_market_value_features(
    index_daily: pd.DataFrame,
    index_daily_basic: pd.DataFrame | None = None,
    *,
    constituent_daily_basic: pd.DataFrame | None = None,
    index_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a wide date-level price, volume, and active-value index frame.

    ``index_dailybasic`` is the preferred source for index float market value
    and turnover.  When a provider does not publish those fields for an index,
    its active-value columns remain missing rather than being synthesized from
    index trading amount.
    """

    daily = _normalize_keys(index_daily, require_symbol=True)
    aliases = _canonical_index_codes()
    daily["ts_code"] = daily["ts_code"].map(aliases).fillna(daily["ts_code"])
    daily = daily[daily["ts_code"].isin({spec.ts_code for spec in KEY_INDEX_SPECS})]
    if daily.empty:
        return _empty_index_feature_frame(())
    if daily.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("index daily has duplicate canonical index/date keys")

    if index_daily_basic is None or index_daily_basic.empty:
        basic = pd.DataFrame(
            columns=["ts_code", "trade_date", "date", "float_mv", "turnover_rate"]
        )
    else:
        basic = _normalize_keys(index_daily_basic, require_symbol=True)
        basic["ts_code"] = basic["ts_code"].map(aliases).fillna(basic["ts_code"])
        basic = basic[basic["ts_code"].isin({spec.ts_code for spec in KEY_INDEX_SPECS})]
        if basic.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError("index dailybasic has duplicate canonical index/date keys")

    constituent_fallbacks = _constituent_index_active_market_value(
        constituent_daily_basic,
        index_weights,
        daily[["ts_code", "trade_date"]],
    )
    per_index: list[pd.DataFrame] = []
    for spec in KEY_INDEX_SPECS:
        current = daily[daily["ts_code"].eq(spec.ts_code)].copy()
        if current.empty:
            continue
        current = current.sort_values("trade_date", kind="stable").reset_index(drop=True)
        current_basic = basic[basic["ts_code"].eq(spec.ts_code)].copy()
        basic_columns = [
            column
            for column in ("trade_date", "float_mv", "circ_mv", "turnover_rate")
            if column in current_basic.columns
        ]
        if "trade_date" in basic_columns:
            current = current.merge(
                current_basic[basic_columns],
                on="trade_date",
                how="left",
                validate="one_to_one",
                suffixes=("", "_basic"),
            )
        fallback = constituent_fallbacks.get(spec.ts_code)
        if fallback is not None:
            current = current.merge(
                fallback,
                on="trade_date",
                how="left",
                validate="one_to_one",
            )
        close = _numeric(current, "close")
        volume = _numeric(current, "volume")
        if volume.isna().all():
            volume = _numeric(current, "vol")
        amount = _numeric(current, "amount")
        if amount.isna().all():
            amount = _numeric(current, "turnover")
        turnover_rate = _numeric(current, "turnover_rate")
        float_mv = _numeric(current, "float_mv")
        if float_mv.isna().all() and "circ_mv" in current.columns:
            # Stock daily_basic uses ten-thousand CNY; an explicitly supplied
            # constituent aggregate may follow that schema.
            float_mv = _numeric(current, "circ_mv") * 10_000.0
        direct_active_ratio = compute_active_share_ratio(turnover_rate)
        direct_active_mv = float_mv * direct_active_ratio
        fallback_active_ratio = _numeric(current, "_fallback_active_share_ratio")
        fallback_active_mv = _numeric(current, "_fallback_active_mv_cny")
        active_ratio = direct_active_ratio.combine_first(fallback_active_ratio)
        active_mv = direct_active_mv.combine_first(fallback_active_mv)
        coverage = pd.Series(
            np.where(direct_active_mv.notna(), 1.0, np.nan),
            index=current.index,
            dtype=float,
        ).combine_first(_numeric(current, "_fallback_coverage_ratio"))
        prefix = f"index_{spec.slug}_"
        output = pd.DataFrame(
            {
                "trade_date": current["trade_date"],
                f"{prefix}return_1d_pct": _pct_change(close, 1),
                f"{prefix}return_5d_pct": _pct_change(close, 5),
                f"{prefix}volume_ratio_prev20": _ratio_to_previous_mean(volume),
                f"{prefix}amount_ratio_prev20": _ratio_to_previous_mean(amount),
                f"{prefix}active_share_ratio_13d_proxy": active_ratio,
                f"{prefix}active_mv_proxy_cny": active_mv,
                f"{prefix}active_mv_log": _safe_log(active_mv),
                f"{prefix}active_mv_return_1d_pct": _pct_change(active_mv, 1),
                f"{prefix}active_mv_return_5d_pct": _pct_change(active_mv, 5),
                f"{prefix}active_mv_coverage_ratio": coverage,
            }
        )
        per_index.append(output)

    dates = sorted(daily["trade_date"].unique())
    if not per_index:
        return _empty_index_feature_frame(dates)
    merged = reduce(
        lambda left, right: left.merge(
            right, on="trade_date", how="outer", validate="one_to_one"
        ),
        per_index,
    )
    merged = _empty_index_feature_frame(dates)[["trade_date", "date"]].merge(
        merged,
        on="trade_date",
        how="left",
        validate="one_to_one",
    )
    for column in KEY_INDEX_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged[
        ["trade_date", "date", *KEY_INDEX_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS]
    ].replace([np.inf, -np.inf], np.nan)


def build_active_market_value_feature_frames(
    daily_basic: pd.DataFrame,
    *,
    market_daily: pd.DataFrame | None = None,
    index_daily: pd.DataFrame | None = None,
    index_daily_basic: pd.DataFrame | None = None,
    index_weights: pd.DataFrame | None = None,
) -> ActiveMarketValueFrames:
    """Build the stock, whole-market, and key-index feature layers together."""

    stock = compute_stock_active_market_value_features(daily_basic)
    market = compute_market_active_market_value_features(
        daily_basic, market_daily=market_daily
    )
    indices = (
        compute_key_index_active_market_value_features(
            index_daily,
            index_daily_basic,
            constituent_daily_basic=daily_basic,
            index_weights=index_weights,
        )
        if index_daily is not None and not index_daily.empty
        else _empty_index_feature_frame(market["trade_date"].tolist())
    )
    return ActiveMarketValueFrames(stock=stock, market=market, indices=indices)


def attach_active_market_value_features(
    rows: pd.DataFrame,
    frames: ActiveMarketValueFrames,
) -> pd.DataFrame:
    """Join stock- and date-level active-value features onto model rows."""

    out = _normalize_keys(rows, require_symbol=True)
    stock = frames.stock[
        ["ts_code", "trade_date", *STOCK_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS]
    ]
    market = frames.market[
        ["trade_date", *MARKET_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS]
    ]
    indices = frames.indices[
        ["trade_date", *KEY_INDEX_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS]
    ]
    out = out.merge(
        stock,
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    out = out.merge(market, on="trade_date", how="left", validate="many_to_one")
    out = out.merge(indices, on="trade_date", how="left", validate="many_to_one")
    return out


__all__ = [
    "ACTIVE_MARKET_VALUE_FEATURE_SCHEMA_VERSION",
    "ACTIVE_MARKET_VALUE_RESEARCH_FEATURE_COLUMNS",
    "ACTIVE_SHARE_DECAY",
    "ActiveMarketValueFrames",
    "INDEX_FEATURE_SUFFIXES",
    "KEY_INDEX_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS",
    "KEY_INDEX_SPECS",
    "KeyIndexSpec",
    "MARKET_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS",
    "MIN_INDEX_CONSTITUENT_COVERAGE",
    "STOCK_ACTIVE_MARKET_VALUE_FEATURE_COLUMNS",
    "attach_active_market_value_features",
    "build_active_market_value_feature_frames",
    "compute_active_share_ratio",
    "compute_key_index_active_market_value_features",
    "compute_market_active_market_value_features",
    "compute_stock_active_market_value_features",
]
