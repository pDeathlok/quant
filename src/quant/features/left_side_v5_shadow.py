"""Causal feature and scoring adapter for the left-side v5 shadow model."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant.features.active_market_value import (
    compute_market_active_market_value_features,
    compute_stock_active_market_value_features,
)
from quant.research.left_side_regime_gate import compose_absolute_display_score


LEFT_SIDE_V5_SHADOW_SCHEMA_VERSION = "left-side-absolute-shadow-v5-score-v1"
LEFT_SIDE_V5_ACTIVE_FEATURE_COLUMNS: tuple[str, ...] = (
    "stock_active_share_ratio_13d_proxy",
    "stock_active_mv_log",
    "stock_active_mv_return_1d_pct",
    "stock_active_mv_return_5d_pct",
    "market_active_mv_log",
    "market_active_mv_ratio_proxy",
    "market_active_mv_return_1d_pct",
    "market_active_mv_return_5d_pct",
)
LEFT_SIDE_V5_REGIME_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_breadth_up_ratio_1d",
    "market_breadth_down_ratio_1d",
    "market_breadth_flat_ratio_1d",
    "market_breadth_advance_decline_ratio_1d",
    "market_breadth_advance_decline_spread_1d",
    "market_breadth_return_mean_1d_pct",
    "market_breadth_return_median_1d_pct",
    "market_breadth_return_dispersion_1d_pct",
    "market_breadth_up5_ratio_1d",
    "market_breadth_down5_ratio_1d",
    "market_breadth_up5_down5_spread_1d",
    "market_breadth_limit_up_ratio_proxy",
    "market_breadth_limit_down_ratio_proxy",
    "market_breadth_limit_up_down_spread_proxy",
    "market_breadth_up_ratio_ma5",
    "market_breadth_up_ratio_ma20",
    "market_breadth_up_ratio_change_1d",
    "market_breadth_up_ratio_change_5d",
    "market_breadth_advance_decline_spread_ma5",
    "market_breadth_advance_decline_spread_ma20",
    "market_active_mv_log",
    "market_active_mv_ratio_proxy",
    "market_active_mv_return_1d_pct",
    "market_active_mv_return_5d_pct",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce").div(
        pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    )


def _compute_selected_market_breadth_features(
    market_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Compute only the 20 breadth fields admitted by the shadow gate."""

    required = {"trade_date", "close"}
    missing = sorted(required - set(market_daily.columns))
    if missing:
        raise ValueError(f"left-side v5 market breadth misses columns: {missing}")
    frame = market_daily.copy()
    frame["trade_date"] = (
        frame["trade_date"].astype(str).str.replace("-", "", regex=False)
    )
    close = pd.to_numeric(frame["close"], errors="coerce")
    if "pct_chg" in frame:
        returns = pd.to_numeric(frame["pct_chg"], errors="coerce")
    else:
        previous = pd.to_numeric(frame["pre_close"], errors="coerce")
        returns = close.div(previous.replace(0.0, np.nan)).sub(1.0).mul(100.0)
    volume_column = "volume" if "volume" in frame else "vol"
    traded = pd.to_numeric(frame[volume_column], errors="coerce").gt(0.0)
    eligible = returns.notna() & close.notna() & traded
    work = pd.DataFrame(
        {
            "trade_date": frame["trade_date"],
            "ret": returns.where(eligible),
            "eligible": eligible,
            "up": eligible & returns.gt(0.0),
            "down": eligible & returns.lt(0.0),
            "flat": eligible & returns.abs().le(1e-12),
            "up5": eligible & returns.ge(5.0),
            "down5": eligible & returns.le(-5.0),
            "limit_up": eligible & returns.ge(9.5),
            "limit_down": eligible & returns.le(-9.5),
        }
    )
    out = (
        work.groupby("trade_date", sort=True)
        .agg(
            eligible=("eligible", "sum"),
            up=("up", "sum"),
            down=("down", "sum"),
            flat=("flat", "sum"),
            up5=("up5", "sum"),
            down5=("down5", "sum"),
            limit_up=("limit_up", "sum"),
            limit_down=("limit_down", "sum"),
            return_mean=("ret", "mean"),
            return_median=("ret", "median"),
            return_dispersion=("ret", "std"),
        )
        .reset_index()
    )
    denominator = out["eligible"]
    out["market_breadth_up_ratio_1d"] = _safe_ratio(out["up"], denominator)
    out["market_breadth_down_ratio_1d"] = _safe_ratio(out["down"], denominator)
    out["market_breadth_flat_ratio_1d"] = _safe_ratio(out["flat"], denominator)
    out["market_breadth_advance_decline_ratio_1d"] = _safe_ratio(out["up"], out["down"])
    out["market_breadth_advance_decline_spread_1d"] = _safe_ratio(
        out["up"] - out["down"], denominator
    )
    out["market_breadth_return_mean_1d_pct"] = out["return_mean"]
    out["market_breadth_return_median_1d_pct"] = out["return_median"]
    out["market_breadth_return_dispersion_1d_pct"] = out["return_dispersion"]
    out["market_breadth_up5_ratio_1d"] = _safe_ratio(out["up5"], denominator)
    out["market_breadth_down5_ratio_1d"] = _safe_ratio(out["down5"], denominator)
    out["market_breadth_up5_down5_spread_1d"] = _safe_ratio(
        out["up5"] - out["down5"], denominator
    )
    out["market_breadth_limit_up_ratio_proxy"] = _safe_ratio(out["limit_up"], denominator)
    out["market_breadth_limit_down_ratio_proxy"] = _safe_ratio(out["limit_down"], denominator)
    out["market_breadth_limit_up_down_spread_proxy"] = _safe_ratio(
        out["limit_up"] - out["limit_down"], denominator
    )
    up_ratio = out["market_breadth_up_ratio_1d"]
    spread = out["market_breadth_advance_decline_spread_1d"]
    out["market_breadth_up_ratio_ma5"] = up_ratio.rolling(5, min_periods=5).mean()
    out["market_breadth_up_ratio_ma20"] = up_ratio.rolling(20, min_periods=20).mean()
    out["market_breadth_up_ratio_change_1d"] = up_ratio.diff(1)
    out["market_breadth_up_ratio_change_5d"] = up_ratio.diff(5)
    out["market_breadth_advance_decline_spread_ma5"] = spread.rolling(5, min_periods=5).mean()
    out["market_breadth_advance_decline_spread_ma20"] = spread.rolling(20, min_periods=20).mean()
    return out[["trade_date", *LEFT_SIDE_V5_REGIME_FEATURE_COLUMNS[:20]]].replace(
        [np.inf, -np.inf], np.nan
    )


def build_left_side_v5_shadow_feature_frame(
    candidates: pd.DataFrame,
    *,
    market_daily: pd.DataFrame,
    daily_basic: pd.DataFrame,
    target_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Attach target-date active-value and whole-market breadth factors."""

    target = pd.to_datetime(target_date, errors="raise").normalize()
    required = {"ts_code", "trade_date", "date"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"left-side v5 candidates miss keys: {missing}")
    if candidates.empty:
        return candidates.assign(
            **{
                column: pd.Series(dtype=float)
                for column in dict.fromkeys(
                    (*LEFT_SIDE_V5_ACTIVE_FEATURE_COLUMNS, *LEFT_SIDE_V5_REGIME_FEATURE_COLUMNS)
                )
            }
        )

    stock = compute_stock_active_market_value_features(daily_basic)
    market_active = compute_market_active_market_value_features(
        daily_basic,
        stock_features=stock,
    )
    breadth = _compute_selected_market_breadth_features(market_daily)
    target_text = target.strftime("%Y%m%d")
    stock = stock[stock["trade_date"].eq(target_text)][
        ["ts_code", "trade_date", *LEFT_SIDE_V5_ACTIVE_FEATURE_COLUMNS[:4]]
    ]
    market_active = market_active[market_active["trade_date"].eq(target_text)][
        ["trade_date", *LEFT_SIDE_V5_ACTIVE_FEATURE_COLUMNS[4:]]
    ]
    breadth = breadth[breadth["trade_date"].eq(target_text)][
        ["trade_date", *LEFT_SIDE_V5_REGIME_FEATURE_COLUMNS[:20]]
    ]
    if len(market_active) != 1 or len(breadth) != 1:
        raise RuntimeError("left-side v5 market-state target row is unavailable")

    out = candidates.copy()
    out["ts_code"] = out["ts_code"].astype(str)
    out["trade_date"] = (
        out["trade_date"].astype(str).str.replace("-", "", regex=False)
    )
    out = out.merge(
        stock,
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        market_active,
        on="trade_date",
        how="left",
        validate="many_to_one",
    ).merge(
        breadth,
        on="trade_date",
        how="left",
        validate="many_to_one",
    )
    if len(out) != len(candidates) or out.duplicated(["ts_code", "date"]).any():
        raise RuntimeError("left-side v5 shadow join changed candidate keys")
    market_columns = list(LEFT_SIDE_V5_REGIME_FEATURE_COLUMNS)
    if out[market_columns].isna().all(axis=0).any():
        missing_market = out[market_columns].columns[
            out[market_columns].isna().all(axis=0)
        ].tolist()
        raise RuntimeError(
            f"left-side v5 market-state features are unavailable: {missing_market}"
        )
    return out


def score_left_side_v5_shadow(
    frame: pd.DataFrame,
    *,
    production_probability: np.ndarray,
    bundle: Mapping[str, Any],
) -> pd.DataFrame:
    """Score the observe-only blend without changing the production ordering."""

    if bundle.get("schema_version") != "left-side-absolute-shadow-v5-bundle-v1":
        raise ValueError("left-side v5 shadow bundle schema mismatch")
    values = np.asarray(production_probability, dtype=float)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("left-side v5 production probabilities are invalid")
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "shadow_v5_component_probability",
                "shadow_v5_blended_percentile",
                "shadow_v5_regime_score",
                "shadow_v5_trade_gate_open",
                "shadow_v5_absolute_score",
                "shadow_v5_schema_version",
            ]
        )

    component = bundle["component_model"]
    gate = bundle["regime_gate"]
    component_probability = np.asarray(component.predict_proba(frame), dtype=float)[:, 1]
    production_percentile = pd.Series(values).rank(method="average", pct=True).to_numpy(float)
    component_percentile = (
        pd.Series(component_probability).rank(method="average", pct=True).to_numpy(float)
    )
    production_weight = float(bundle["production_weight"])
    blended = (
        production_weight * production_percentile
        + (1.0 - production_weight) * component_percentile
    )
    regime_score = np.asarray(gate.predict_score(frame), dtype=float)
    gate_open = np.asarray(gate.predict_open(frame), dtype=bool)
    output = pd.DataFrame(index=frame.index)
    output["shadow_v5_component_probability"] = component_probability
    output["shadow_v5_blended_percentile"] = blended * 100.0
    output["shadow_v5_regime_score"] = regime_score
    output["shadow_v5_trade_gate_open"] = gate_open
    output["shadow_v5_absolute_score"] = compose_absolute_display_score(
        blended, gate_open
    )
    output["shadow_v5_schema_version"] = LEFT_SIDE_V5_SHADOW_SCHEMA_VERSION
    return output


__all__ = [
    "LEFT_SIDE_V5_ACTIVE_FEATURE_COLUMNS",
    "LEFT_SIDE_V5_REGIME_FEATURE_COLUMNS",
    "LEFT_SIDE_V5_SHADOW_SCHEMA_VERSION",
    "build_left_side_v5_shadow_feature_frame",
    "score_left_side_v5_shadow",
]
