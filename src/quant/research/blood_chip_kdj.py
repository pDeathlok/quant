"""Causal multi-timeframe KDJ overlay for the blood-chip research system."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


KDJ_COLUMNS = ("kdj_daily_j", "kdj_weekly_j", "kdj_monthly_j")
OverlayMode = Literal[
    "baseline_low_vol",
    "kdj_soft_priority",
    "daily_weekly_only",
    "triple_only",
]


def _kdj(frame: pd.DataFrame) -> pd.DataFrame:
    low9 = frame["adjusted_low"].rolling(9, min_periods=9).min()
    high9 = frame["adjusted_high"].rolling(9, min_periods=9).max()
    rsv = (
        (frame["adjusted_close"] - low9)
        / (high9 - low9).replace(0, np.nan)
        * 100.0
    )
    k = rsv.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    d = k.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    return pd.DataFrame({"k": k, "d": d, "j": 3.0 * k - 2.0 * d})


def _completed_period_j(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    bars = (
        frame.set_index("date")
        .resample(rule)
        .agg(
            adjusted_high=("adjusted_high", "max"),
            adjusted_low=("adjusted_low", "min"),
            adjusted_close=("adjusted_close", "last"),
        )
        .dropna()
    )
    bars["j"] = _kdj(bars)["j"]
    return bars[["j"]].reset_index()


def attach_completed_kdj(
    features: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Attach point-in-time daily and completed weekly/monthly J to signals.

    A weekly bar becomes usable on its Friday label and a monthly bar on its
    calendar month-end label.  Holiday-shortened periods therefore become
    visible on the next trading day, never before the period has completed.
    """

    out = signals.copy()
    if out.empty:
        for column in KDJ_COLUMNS:
            out[column] = pd.Series(dtype=float)
        out["kdj_negative_count"] = pd.Series(dtype="Int64")
        out["kdj_state"] = pd.Series(dtype=str)
        return out
    required_features = {
        "ts_code",
        "date",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
    }
    required_signals = {"ts_code", "signal_date"}
    missing_features = sorted(required_features - set(features.columns))
    missing_signals = sorted(required_signals - set(out.columns))
    if missing_features:
        raise ValueError(f"feature frame missing KDJ columns: {missing_features}")
    if missing_signals:
        raise ValueError(f"signal frame missing KDJ columns: {missing_signals}")

    signal_symbols = set(out["ts_code"].astype(str))
    feature_symbols = features["ts_code"].astype(str)
    panel = features.loc[feature_symbols.isin(signal_symbols), list(required_features)].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values(["ts_code", "date"])
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out["_kdj_order"] = np.arange(len(out), dtype=np.int64)
    pieces: list[pd.DataFrame] = []
    for symbol, prices in panel.groupby("ts_code", observed=True, sort=False):
        symbol = str(symbol)
        if symbol not in signal_symbols:
            continue
        targets = out.loc[out["ts_code"].astype(str).eq(symbol)].copy()
        prices = prices.sort_values("date").drop_duplicates("date", keep="last")
        daily = prices[["date"]].copy()
        daily["kdj_daily_j"] = _kdj(prices.reset_index(drop=True))["j"].to_numpy()
        weekly = _completed_period_j(prices, "W-FRI").rename(
            columns={"j": "kdj_weekly_j"}
        )
        monthly = _completed_period_j(prices, "ME").rename(
            columns={"j": "kdj_monthly_j"}
        )
        targets = pd.merge_asof(
            targets.sort_values("signal_date"),
            daily,
            left_on="signal_date",
            right_on="date",
            direction="backward",
        ).drop(columns="date")
        targets = pd.merge_asof(
            targets.sort_values("signal_date"),
            weekly,
            left_on="signal_date",
            right_on="date",
            direction="backward",
        ).drop(columns="date")
        targets = pd.merge_asof(
            targets.sort_values("signal_date"),
            monthly,
            left_on="signal_date",
            right_on="date",
            direction="backward",
        ).drop(columns="date")
        pieces.append(targets)
    if pieces:
        enriched = pd.concat(pieces, ignore_index=True, sort=False)
        out = enriched.sort_values("_kdj_order").reset_index(drop=True)
    negative = out[list(KDJ_COLUMNS)].lt(0)
    available = out[list(KDJ_COLUMNS)].notna().all(axis=1)
    out["kdj_negative_count"] = negative.sum(axis=1).astype("Int64")
    out["kdj_state"] = np.select(
        [
            ~available,
            negative.all(axis=1),
            negative["kdj_daily_j"] & negative["kdj_weekly_j"],
            negative.any(axis=1),
        ],
        [
            "unavailable",
            "triple_oversold",
            "daily_weekly_oversold",
            "partial_oversold",
        ],
        default="not_oversold",
    )
    return out.drop(columns="_kdj_order")


def attach_blood_chip_kdj_path(
    features: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Attach KDJ at both the fire-sale shock and later confirmation dates."""

    out = signals.copy().reset_index(drop=True)
    if out.empty:
        return attach_completed_kdj(features, out)
    if "shock_date" not in out:
        raise ValueError("signal frame missing KDJ path column: shock_date")
    out["_path_order"] = np.arange(len(out), dtype=np.int64)
    confirmation = out[["_path_order", "ts_code", "signal_date"]].copy()
    confirmation["_kdj_point"] = "confirmation"
    shock = out[["_path_order", "ts_code", "shock_date"]].rename(
        columns={"shock_date": "signal_date"}
    )
    shock["_kdj_point"] = "shock"
    attached = attach_completed_kdj(
        features,
        pd.concat([confirmation, shock], ignore_index=True, sort=False),
    )
    value_columns = [*KDJ_COLUMNS, "kdj_negative_count", "kdj_state"]
    for point in ("confirmation", "shock"):
        values = attached.loc[
            attached["_kdj_point"].eq(point), ["_path_order", *value_columns]
        ].rename(columns={column: f"{point}_{column}" for column in value_columns})
        out = out.merge(values, on="_path_order", how="left", validate="one_to_one")
    return out.drop(columns="_path_order")


def apply_kdj_overlay(signals: pd.DataFrame, mode: OverlayMode) -> pd.DataFrame:
    """Apply a pre-registered hard filter or soft ranking overlay."""

    required = set(KDJ_COLUMNS) | {"volatility_60d", "signal_date"}
    missing = sorted(required - set(signals.columns))
    if missing:
        raise ValueError(f"signal frame missing overlay columns: {missing}")
    out = signals.copy()
    daily_weekly = out["kdj_daily_j"].lt(0) & out["kdj_weekly_j"].lt(0)
    triple = daily_weekly & out["kdj_monthly_j"].lt(0)
    if mode == "daily_weekly_only":
        out = out.loc[daily_weekly].copy()
    elif mode == "triple_only":
        out = out.loc[triple].copy()
    elif mode not in {"baseline_low_vol", "kdj_soft_priority"}:
        raise ValueError(f"unknown KDJ overlay mode: {mode}")
    low_vol_priority = 1.0 - out.groupby(
        "signal_date", observed=True, sort=False
    )["volatility_60d"].rank(method="average", pct=True)
    if mode == "kdj_soft_priority":
        negative_count = out[list(KDJ_COLUMNS)].lt(0).sum(axis=1) / 3.0
        # KDJ breaks ties and nudges ordering; it cannot overpower the existing
        # low-volatility preference on its own.
        out["signal_score"] = 0.80 * low_vol_priority + 0.20 * negative_count
    else:
        out["signal_score"] = low_vol_priority
    out = out.sort_values(
        ["signal_date", "signal_score", "ts_code"],
        ascending=[True, False, True],
    )
    out["selection_rank"] = out.groupby(
        "signal_date", observed=True, sort=False
    ).cumcount() + 1
    return out.reset_index(drop=True)
