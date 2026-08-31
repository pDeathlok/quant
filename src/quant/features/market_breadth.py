"""Causal whole-market breadth features for cross-sectional research."""

from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_BREADTH_FEATURE_SCHEMA_VERSION = "market_breadth_v1_20260831"
MARKET_BREADTH_MA_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)
MARKET_BREADTH_HIGH_LOW_WINDOWS: tuple[int, ...] = (20, 60)

_BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_breadth_stock_count",
    "market_breadth_return_eligible_count",
    "market_breadth_up_count_1d",
    "market_breadth_down_count_1d",
    "market_breadth_flat_count_1d",
    "market_breadth_up_ratio_1d",
    "market_breadth_down_ratio_1d",
    "market_breadth_flat_ratio_1d",
    "market_breadth_advance_decline_diff_1d",
    "market_breadth_advance_decline_ratio_1d",
    "market_breadth_advance_decline_spread_1d",
    "market_breadth_return_mean_1d_pct",
    "market_breadth_return_median_1d_pct",
    "market_breadth_return_dispersion_1d_pct",
)
_STRONG_MOVE_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_breadth_up5_count_1d",
    "market_breadth_down5_count_1d",
    "market_breadth_up5_ratio_1d",
    "market_breadth_down5_ratio_1d",
    "market_breadth_up5_down5_diff_1d",
    "market_breadth_up5_down5_spread_1d",
)
_LIMIT_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_breadth_limit_up_count_proxy",
    "market_breadth_limit_down_count_proxy",
    "market_breadth_limit_up_ratio_proxy",
    "market_breadth_limit_down_ratio_proxy",
    "market_breadth_limit_up_down_diff_proxy",
    "market_breadth_limit_up_down_ratio_proxy",
    "market_breadth_limit_up_down_spread_proxy",
)
_MA_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column
    for window in MARKET_BREADTH_MA_WINDOWS
    for column in (
        f"market_breadth_ma{window}_eligible_count",
        f"market_breadth_above_ma{window}_count",
        f"market_breadth_above_ma{window}_ratio",
    )
)
_HIGH_LOW_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column
    for window in MARKET_BREADTH_HIGH_LOW_WINDOWS
    for column in (
        f"market_breadth_high_low_{window}d_eligible_count",
        f"market_breadth_new_high_{window}d_count",
        f"market_breadth_new_low_{window}d_count",
        f"market_breadth_new_high_{window}d_ratio",
        f"market_breadth_new_low_{window}d_ratio",
        f"market_breadth_new_high_low_diff_{window}d",
        f"market_breadth_new_high_low_ratio_{window}d",
        f"market_breadth_new_high_low_spread_{window}d",
    )
)
_ROLLING_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_breadth_up_ratio_ma5",
    "market_breadth_up_ratio_ma20",
    "market_breadth_up_ratio_change_1d",
    "market_breadth_up_ratio_change_5d",
    "market_breadth_advance_decline_spread_ma5",
    "market_breadth_advance_decline_spread_ma20",
)
MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS: tuple[str, ...] = (
    *_BASE_FEATURE_COLUMNS,
    *_STRONG_MOVE_FEATURE_COLUMNS,
    *_LIMIT_FEATURE_COLUMNS,
    *_MA_FEATURE_COLUMNS,
    *_HIGH_LOW_FEATURE_COLUMNS,
    *_ROLLING_FEATURE_COLUMNS,
)


def _normalize_market_daily(market_daily: pd.DataFrame) -> pd.DataFrame:
    frame = market_daily.copy()
    if "ts_code" not in frame.columns and "symbol" in frame.columns:
        frame["ts_code"] = frame["symbol"].astype(str)
    if "ts_code" not in frame.columns:
        raise ValueError("market breadth input misses ts_code/symbol")
    if "trade_date" not in frame.columns and "date" in frame.columns:
        frame["trade_date"] = frame["date"]
    required = {"trade_date", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"market breadth input misses columns: {missing}")
    parsed = pd.to_datetime(frame["trade_date"], errors="coerce")
    if parsed.isna().any():
        compact = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
        parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError("market breadth input has invalid trade_date values")
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["trade_date"] = parsed.dt.strftime("%Y%m%d")
    frame["date"] = parsed.dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("market breadth input has duplicate stock/date keys")
    return frame.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(
        drop=True
    )


def _return_1d_pct(frame: pd.DataFrame) -> pd.Series:
    calculated = frame.groupby("ts_code", sort=False)["close"].pct_change(
        fill_method=None
    ) * 100.0
    if "pct_chg" in frame.columns:
        reported = pd.to_numeric(frame["pct_chg"], errors="coerce")
        return reported.combine_first(calculated)
    if "pre_close" in frame.columns:
        previous = pd.to_numeric(frame["pre_close"], errors="coerce")
        reported = frame["close"].div(previous.replace(0, np.nan)).sub(1.0) * 100.0
        return reported.combine_first(calculated)
    return calculated


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce").div(
        pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    )


def compute_market_breadth_features(market_daily: pd.DataFrame) -> pd.DataFrame:
    """Build date-level breadth features from the full stock cross-section.

    A stock participates in return breadth only when its return is known and,
    when volume is supplied, its current-session volume is positive.  New
    highs/lows require a strict break of the preceding window, so an unchanged
    stock is never simultaneously classified as a new high and a new low.
    """

    frame = _normalize_market_daily(market_daily)
    frame["_ret_1d_pct"] = _return_1d_pct(frame)
    volume_column = "volume" if "volume" in frame.columns else "vol"
    if volume_column in frame.columns:
        volume = pd.to_numeric(frame[volume_column], errors="coerce")
        traded = volume.gt(0)
    else:
        traded = pd.Series(True, index=frame.index, dtype=bool)
    frame["_valid_close"] = frame["close"].notna()
    frame["_return_eligible"] = (
        frame["_valid_close"] & frame["_ret_1d_pct"].notna() & traded
    )
    eligible_return = frame["_ret_1d_pct"].where(frame["_return_eligible"])
    frame["_eligible_return"] = eligible_return
    frame["_up"] = frame["_return_eligible"] & eligible_return.gt(0)
    frame["_down"] = frame["_return_eligible"] & eligible_return.lt(0)
    frame["_flat"] = frame["_return_eligible"] & np.isclose(
        eligible_return.fillna(np.nan), 0.0, atol=1e-12, rtol=0.0
    )
    frame["_up5"] = frame["_return_eligible"] & eligible_return.ge(5.0)
    frame["_down5"] = frame["_return_eligible"] & eligible_return.le(-5.0)
    frame["_limit_up"] = frame["_return_eligible"] & eligible_return.ge(9.5)
    frame["_limit_down"] = frame["_return_eligible"] & eligible_return.le(-9.5)

    for window in MARKET_BREADTH_MA_WINDOWS:
        moving_average = frame.groupby("ts_code", sort=False)["close"].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        eligibility = frame["_valid_close"] & moving_average.notna() & traded
        frame[f"_ma{window}_eligible"] = eligibility
        frame[f"_above_ma{window}"] = eligibility & frame["close"].gt(moving_average)

    for window in MARKET_BREADTH_HIGH_LOW_WINDOWS:
        previous_high = frame.groupby("ts_code", sort=False)["close"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).max()
        )
        previous_low = frame.groupby("ts_code", sort=False)["close"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).min()
        )
        eligibility = (
            frame["_valid_close"]
            & previous_high.notna()
            & previous_low.notna()
            & traded
        )
        frame[f"_high_low_{window}d_eligible"] = eligibility
        frame[f"_new_high_{window}d"] = eligibility & frame["close"].gt(
            previous_high
        )
        frame[f"_new_low_{window}d"] = eligibility & frame["close"].lt(previous_low)

    aggregations: dict[str, tuple[str, str]] = {
        "market_breadth_stock_count": ("_valid_close", "sum"),
        "market_breadth_return_eligible_count": ("_return_eligible", "sum"),
        "market_breadth_up_count_1d": ("_up", "sum"),
        "market_breadth_down_count_1d": ("_down", "sum"),
        "market_breadth_flat_count_1d": ("_flat", "sum"),
        "market_breadth_up5_count_1d": ("_up5", "sum"),
        "market_breadth_down5_count_1d": ("_down5", "sum"),
        "market_breadth_limit_up_count_proxy": ("_limit_up", "sum"),
        "market_breadth_limit_down_count_proxy": ("_limit_down", "sum"),
        "market_breadth_return_mean_1d_pct": ("_eligible_return", "mean"),
        "market_breadth_return_median_1d_pct": ("_eligible_return", "median"),
        "market_breadth_return_dispersion_1d_pct": ("_eligible_return", "std"),
    }
    for window in MARKET_BREADTH_MA_WINDOWS:
        aggregations[f"market_breadth_ma{window}_eligible_count"] = (
            f"_ma{window}_eligible",
            "sum",
        )
        aggregations[f"market_breadth_above_ma{window}_count"] = (
            f"_above_ma{window}",
            "sum",
        )
    for window in MARKET_BREADTH_HIGH_LOW_WINDOWS:
        aggregations[f"market_breadth_high_low_{window}d_eligible_count"] = (
            f"_high_low_{window}d_eligible",
            "sum",
        )
        aggregations[f"market_breadth_new_high_{window}d_count"] = (
            f"_new_high_{window}d",
            "sum",
        )
        aggregations[f"market_breadth_new_low_{window}d_count"] = (
            f"_new_low_{window}d",
            "sum",
        )
    result = (
        frame.groupby("trade_date", sort=True)
        .agg(**aggregations)
        .reset_index()
    )
    result["date"] = pd.to_datetime(result["trade_date"], format="%Y%m%d")
    denominator = result["market_breadth_return_eligible_count"]
    for direction in ("up", "down", "flat"):
        result[f"market_breadth_{direction}_ratio_1d"] = _safe_ratio(
            result[f"market_breadth_{direction}_count_1d"], denominator
        )
    up = result["market_breadth_up_count_1d"]
    down = result["market_breadth_down_count_1d"]
    result["market_breadth_advance_decline_diff_1d"] = up - down
    result["market_breadth_advance_decline_ratio_1d"] = _safe_ratio(up, down)
    result["market_breadth_advance_decline_spread_1d"] = _safe_ratio(
        up - down, denominator
    )

    up5 = result["market_breadth_up5_count_1d"]
    down5 = result["market_breadth_down5_count_1d"]
    result["market_breadth_up5_ratio_1d"] = _safe_ratio(up5, denominator)
    result["market_breadth_down5_ratio_1d"] = _safe_ratio(down5, denominator)
    result["market_breadth_up5_down5_diff_1d"] = up5 - down5
    result["market_breadth_up5_down5_spread_1d"] = _safe_ratio(
        up5 - down5, denominator
    )

    limit_up = result["market_breadth_limit_up_count_proxy"]
    limit_down = result["market_breadth_limit_down_count_proxy"]
    result["market_breadth_limit_up_ratio_proxy"] = _safe_ratio(
        limit_up, denominator
    )
    result["market_breadth_limit_down_ratio_proxy"] = _safe_ratio(
        limit_down, denominator
    )
    result["market_breadth_limit_up_down_diff_proxy"] = limit_up - limit_down
    result["market_breadth_limit_up_down_ratio_proxy"] = _safe_ratio(
        limit_up, limit_down
    )
    result["market_breadth_limit_up_down_spread_proxy"] = _safe_ratio(
        limit_up - limit_down, denominator
    )

    for window in MARKET_BREADTH_MA_WINDOWS:
        result[f"market_breadth_above_ma{window}_ratio"] = _safe_ratio(
            result[f"market_breadth_above_ma{window}_count"],
            result[f"market_breadth_ma{window}_eligible_count"],
        )
    for window in MARKET_BREADTH_HIGH_LOW_WINDOWS:
        eligible = result[f"market_breadth_high_low_{window}d_eligible_count"]
        high = result[f"market_breadth_new_high_{window}d_count"]
        low = result[f"market_breadth_new_low_{window}d_count"]
        result[f"market_breadth_new_high_{window}d_ratio"] = _safe_ratio(
            high, eligible
        )
        result[f"market_breadth_new_low_{window}d_ratio"] = _safe_ratio(low, eligible)
        result[f"market_breadth_new_high_low_diff_{window}d"] = high - low
        result[f"market_breadth_new_high_low_ratio_{window}d"] = _safe_ratio(high, low)
        result[f"market_breadth_new_high_low_spread_{window}d"] = _safe_ratio(
            high - low, eligible
        )

    up_ratio = result["market_breadth_up_ratio_1d"]
    spread = result["market_breadth_advance_decline_spread_1d"]
    result["market_breadth_up_ratio_ma5"] = up_ratio.rolling(
        5, min_periods=5
    ).mean()
    result["market_breadth_up_ratio_ma20"] = up_ratio.rolling(
        20, min_periods=20
    ).mean()
    result["market_breadth_up_ratio_change_1d"] = up_ratio.diff(1)
    result["market_breadth_up_ratio_change_5d"] = up_ratio.diff(5)
    result["market_breadth_advance_decline_spread_ma5"] = spread.rolling(
        5, min_periods=5
    ).mean()
    result["market_breadth_advance_decline_spread_ma20"] = spread.rolling(
        20, min_periods=20
    ).mean()
    return result[
        ["trade_date", "date", *MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS]
    ].replace([np.inf, -np.inf], np.nan)


def attach_market_breadth_features(
    rows: pd.DataFrame,
    market_breadth: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one date-level market breadth row to every matching model row."""

    out = rows.copy()
    if "trade_date" not in out.columns and "date" in out.columns:
        out["trade_date"] = out["date"]
    if "trade_date" not in out.columns:
        raise ValueError("market breadth target rows miss trade_date/date")
    parsed = pd.to_datetime(out["trade_date"], errors="coerce")
    if parsed.isna().any():
        compact = out["trade_date"].astype(str).str.replace("-", "", regex=False)
        parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError("market breadth target rows have invalid trade_date values")
    out["trade_date"] = parsed.dt.strftime("%Y%m%d")
    features = market_breadth[
        ["trade_date", *MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS]
    ].copy()
    features["trade_date"] = pd.to_datetime(
        features["trade_date"], errors="coerce"
    ).dt.strftime("%Y%m%d")
    if features["trade_date"].isna().any():
        raise ValueError("market breadth feature rows have invalid trade_date values")
    if features.duplicated("trade_date").any():
        raise ValueError("market breadth feature rows have duplicate trade dates")
    return out.merge(features, on="trade_date", how="left", validate="many_to_one")


__all__ = [
    "MARKET_BREADTH_FEATURE_SCHEMA_VERSION",
    "MARKET_BREADTH_HIGH_LOW_WINDOWS",
    "MARKET_BREADTH_MA_WINDOWS",
    "MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS",
    "attach_market_breadth_features",
    "compute_market_breadth_features",
]
