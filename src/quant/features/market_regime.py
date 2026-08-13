"""Daily market-regime classification from canonical project data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalize(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    required = {"trade_date", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["trade_date", "close"])
    return result.sort_values("trade_date").reset_index(drop=True)


def classify_market_regime(
    index_daily: pd.DataFrame,
    market_daily: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Classify risk-on/neutral/risk-off without looking beyond ``as_of``."""

    index = _normalize(index_daily, name="index_daily")
    market = _normalize(market_daily, name="market_daily")
    if "ts_code" not in market.columns:
        raise ValueError("market_daily missing required columns: ['ts_code']")
    requested_date = pd.Timestamp(as_of).normalize() if as_of is not None else None
    if requested_date is not None:
        index = index.loc[index["trade_date"] <= requested_date].copy()
        market = market.loc[market["trade_date"] <= requested_date].copy()
    if index["trade_date"].nunique() < 60:
        raise ValueError("index_daily requires at least 60 observations through as_of")
    if market.empty:
        raise ValueError("market_daily has no observations through as_of")

    index = index.drop_duplicates("trade_date", keep="last")
    latest_index_date = index["trade_date"].max()
    latest_market_date = market["trade_date"].max()
    if latest_index_date != latest_market_date:
        raise ValueError(
            "market regime terminal observation dates differ: "
            f"index={latest_index_date.date().isoformat()} "
            f"market={latest_market_date.date().isoformat()}"
        )
    if requested_date is not None and latest_index_date != requested_date:
        raise ValueError(
            "market regime inputs are stale for requested decision date: "
            f"requested={requested_date.date().isoformat()} "
            f"latest={latest_index_date.date().isoformat()}"
        )
    index["ma20"] = index["close"].rolling(20).mean()
    index["ma60"] = index["close"].rolling(60).mean()
    index["return"] = index["close"].pct_change()
    latest = index.iloc[-1]
    annualized_volatility = float(index["return"].tail(20).std(ddof=1) * np.sqrt(252))

    market["symbol_ma20"] = market.groupby("ts_code", sort=False)["close"].transform(
        lambda values: values.rolling(20).mean()
    )
    cross_section = market.loc[market["trade_date"] == latest_market_date]
    breadth = float((cross_section["close"] > cross_section["symbol_ma20"]).mean())

    liquidity_ratio: float | None = None
    if "amount" in market.columns:
        market["amount"] = pd.to_numeric(market["amount"], errors="coerce")
        daily_amount = market.groupby("trade_date")["amount"].sum(min_count=1)
        if len(daily_amount.dropna()) >= 20:
            denominator = float(daily_amount.tail(20).mean())
            liquidity_ratio = float(daily_amount.iloc[-1] / denominator) if denominator > 0 else None

    close_above_ma20 = bool(latest["close"] > latest["ma20"])
    ma20_above_ma60 = bool(latest["ma20"] > latest["ma60"])
    score = int(close_above_ma20) + int(ma20_above_ma60)
    score += 1 if breadth >= 0.55 else (-1 if breadth < 0.4 else 0)
    if liquidity_ratio is not None:
        score += 1 if liquidity_ratio >= 0.9 else -1
    if annualized_volatility > 0.30:
        score -= 1

    regime = "risk_on" if score >= 2 else ("risk_off" if score <= 0 else "neutral")
    return {
        "as_of": latest_index_date.strftime("%Y%m%d"),
        "regime": regime,
        "score": score,
        "signals": {
            "index_close": float(latest["close"]),
            "index_ma20": float(latest["ma20"]),
            "index_ma60": float(latest["ma60"]),
            "close_above_ma20": close_above_ma20,
            "ma20_above_ma60": ma20_above_ma60,
            "annualized_volatility_20d": annualized_volatility,
            "breadth_above_ma20": breadth,
            "liquidity_ratio_20d": liquidity_ratio,
        },
        "coverage": {
            "index_observations": int(index["trade_date"].nunique()),
            "index_observation_date": latest_index_date.strftime("%Y%m%d"),
            "market_symbols": int(cross_section["ts_code"].nunique()),
            "market_observation_date": latest_market_date.strftime("%Y%m%d"),
        },
        "method": "project_daily_close_ma_breadth_liquidity_v1",
    }
