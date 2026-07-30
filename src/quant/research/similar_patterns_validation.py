"""Point-in-time validation helpers for the similar-pattern decision desk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.research.similar_patterns import (
    SimilarPatternConfig,
    apply_probability_calibration,
    fit_probability_calibration,
    load_daily_file,
)


HORIZON_DAYS = {"next_1d": 1, "next_1m": 20, "next_3m": 60}
HORIZON_COLUMNS = {"next_1d": "fwd_1d", "next_1m": "fwd_20d", "next_3m": "fwd_60d"}


def build_market_regime(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify trailing benchmark state without using future rows."""
    out = frame.copy()
    if "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        raise ValueError("market frame requires trade_date or date")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    out["market_ret_20d"] = out["close"].pct_change(20)
    out["market_vol_20d"] = out["close"].pct_change().rolling(20).std()
    ma20 = out["close"].rolling(20).mean()
    risk_on = (out["close"] > ma20) & (out["market_ret_20d"] > 0)
    risk_off = (out["close"] < ma20) & (out["market_ret_20d"] < 0)
    out["market_regime"] = "neutral"
    out.loc[risk_on, "market_regime"] = "risk_on"
    out.loc[risk_off, "market_regime"] = "risk_off"
    return out[["date", "market_regime", "market_ret_20d", "market_vol_20d"]].reset_index(drop=True)


def load_market_regime(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "market_regime", "market_ret_20d", "market_vol_20d"])
    return build_market_regime(pd.read_parquet(path))


def build_industry_regime(
    daily_dir: Path,
    basic: pd.DataFrame,
    industry: str,
) -> pd.DataFrame:
    """Build an equal-weight trailing regime for one target industry."""
    if basic.empty or not industry:
        return pd.DataFrame(columns=["date", "industry_regime"])
    symbols = basic.loc[basic["industry"].fillna("").astype(str).eq(industry), "ts_code"].astype(str).tolist()
    returns: list[pd.Series] = []
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
    market = store.read_market_range(
        daily_dir.name,
        symbols=symbols,
        columns=["ts_code", "trade_date", "close"],
    )
    if not market.empty:
        if "trade_date" in market.columns:
            trade_dates = pd.to_datetime(
                market["trade_date"].astype(str),
                format="%Y%m%d",
                errors="coerce",
            )
            if "date" in market.columns:
                market["date"] = pd.to_datetime(market["date"], errors="coerce").fillna(trade_dates)
            else:
                market["date"] = trade_dates
        else:
            market["date"] = pd.to_datetime(market["date"], errors="coerce")
        market["close"] = pd.to_numeric(market["close"], errors="coerce")
        market = market.dropna(subset=["ts_code", "date", "close"])
        for symbol, daily in market.groupby("ts_code", sort=False):
            series = (
                daily.sort_values("date")
                .drop_duplicates("date", keep="last")
                .set_index("date")["close"]
                .pct_change()
                .rename(str(symbol))
            )
            returns.append(series)
    else:
        # Preserve compatibility with repositories that still only have the
        # pre-migration one-file-per-symbol layout.
        for symbol in symbols:
            path = daily_dir / f"{symbol}.parquet"
            if not path.exists():
                continue
            daily = load_daily_file(path)
            returns.append(daily.set_index("date")["close"].pct_change().rename(symbol))
    if not returns:
        return pd.DataFrame(columns=["date", "industry_regime"])
    industry_return = pd.concat(returns, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    equity = (1.0 + industry_return).cumprod()
    ret20 = equity.pct_change(20)
    ma20 = equity.rolling(20).mean()
    regime = pd.Series("neutral", index=equity.index, dtype=object)
    regime.loc[(equity > ma20) & (ret20 > 0)] = "risk_on"
    regime.loc[(equity < ma20) & (ret20 < 0)] = "risk_off"
    return pd.DataFrame({"date": regime.index, "industry_regime": regime.values}).reset_index(drop=True)


def filter_cases_mature_at_signal(
    cases: pd.DataFrame,
    signal_date: pd.Timestamp,
    horizon: str,
) -> pd.DataFrame:
    """Keep only cases whose outcome horizon was observable before the signal."""
    if horizon not in HORIZON_DAYS:
        raise ValueError(f"unsupported horizon: {horizon}")
    if cases.empty:
        return cases.copy()
    cutoff = pd.Timestamp(signal_date) - pd.offsets.BDay(HORIZON_DAYS[horizon])
    dates = pd.to_datetime(cases["date"], errors="coerce")
    column = HORIZON_COLUMNS[horizon]
    mask = dates.le(cutoff)
    if column in cases.columns:
        mask &= pd.to_numeric(cases[column], errors="coerce").notna()
    return cases.loc[mask].copy()


def apply_expanding_calibration(
    records: pd.DataFrame,
    *,
    min_samples: int = 20,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, object]]]]:
    """Calibrate each row from strictly earlier matured predictions only."""
    out = records.copy().sort_values(["symbol", "horizon", "signal_date"]).reset_index(drop=True)
    out["calibrated_up_probability"] = np.nan
    out["calibration_samples"] = 0
    calibrations: dict[str, dict[str, dict[str, object]]] = {}
    for (symbol, horizon), group in out.groupby(["symbol", "horizon"], sort=False):
        latest_calibration: dict[str, object] | None = None
        for row_idx in group.index:
            signal_date = pd.Timestamp(out.at[row_idx, "signal_date"])
            history = group.loc[
                (group["outcome_date"] <= signal_date)
                & group["actual_return"].notna()
                & group["raw_up_probability"].notna()
            ]
            calibration = fit_probability_calibration(
                history["raw_up_probability"].astype(float).tolist(),
                history["actual_return"].gt(0).tolist(),
                min_samples=min_samples,
            )
            calibrated = apply_probability_calibration(float(out.at[row_idx, "raw_up_probability"]), calibration)
            out.at[row_idx, "calibrated_up_probability"] = calibrated
            out.at[row_idx, "calibration_samples"] = int(len(history))
            latest_calibration = calibration
        calibrations.setdefault(str(symbol), {})[str(horizon)] = latest_calibration or {
            "status": "identity",
            "sample_count": 0,
            "x": [0.0, 100.0],
            "y": [0.0, 100.0],
        }
    return out, calibrations


def apply_global_expanding_calibration(
    records: pd.DataFrame,
    *,
    min_samples: int = 20,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Calibrate each horizon on the pooled watchlist history available at that date."""
    out = records.copy().sort_values(["horizon", "signal_date", "symbol"]).reset_index(drop=True)
    out["calibrated_up_probability"] = np.nan
    out["calibration_samples"] = 0
    calibrations: dict[str, dict[str, object]] = {}
    for horizon, group in out.groupby("horizon", sort=False):
        latest_calibration: dict[str, object] | None = None
        for row_idx in group.index:
            signal_date = pd.Timestamp(out.at[row_idx, "signal_date"])
            history = group.loc[
                (group["outcome_date"] <= signal_date)
                & group["actual_return"].notna()
                & group["raw_up_probability"].notna()
            ]
            calibration = fit_probability_calibration(
                history["raw_up_probability"].astype(float).tolist(),
                history["actual_return"].gt(0).tolist(),
                min_samples=min_samples,
            )
            out.at[row_idx, "calibrated_up_probability"] = apply_probability_calibration(
                float(out.at[row_idx, "raw_up_probability"]), calibration
            )
            out.at[row_idx, "calibration_samples"] = int(len(history))
            latest_calibration = calibration
        calibrations[str(horizon)] = latest_calibration or {
            "status": "identity",
            "sample_count": 0,
            "x": [0.0, 100.0],
            "y": [0.0, 100.0],
        }
    return out, calibrations


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def summarize_walk_forward_records(
    records: pd.DataFrame,
    *,
    transaction_cost: float,
) -> pd.DataFrame:
    """Summarize signal coverage, calibration, and cost-adjusted directional returns."""
    rows: list[dict[str, object]] = []
    for (symbol, horizon), group in records.groupby(["symbol", "horizon"], sort=False):
        mature = group[group["actual_return"].notna()].copy()
        actionable = mature[mature["signal"].isin(["bullish", "bearish"])].copy()
        predicted_up = actionable["signal"].eq("bullish")
        actual_up = actionable["actual_return"].gt(0)
        correct = predicted_up.eq(actual_up)
        directional = np.where(predicted_up, actionable["actual_return"], -actionable["actual_return"])
        cost_adjusted = np.asarray(directional, dtype=float) - float(transaction_cost)
        probability = mature["calibrated_up_probability"].astype(float) / 100.0
        outcome = mature["actual_return"].gt(0).astype(float)
        rows.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "signals": int(len(mature)),
                "actionable_signals": int(len(actionable)),
                "coverage": round(len(actionable) / len(mature) * 100, 2) if len(mature) else 0.0,
                "direction_accuracy": round(float(correct.mean()) * 100, 2) if len(actionable) else None,
                "brier_score": round(float(np.mean(np.square(probability - outcome))), 4) if len(mature) else None,
                "gross_directional_return": round(float(np.mean(directional)) * 100, 2) if len(actionable) else 0.0,
                "cost_adjusted_return": round(float(np.mean(cost_adjusted)) * 100, 2) if len(actionable) else 0.0,
                "max_drawdown": (
                    round(_max_drawdown(pd.Series(cost_adjusted)) * 100, 2)
                    if len(actionable) and horizon == "next_1d"
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def validation_config_payload(config: SimilarPatternConfig) -> dict[str, object]:
    return {
        "signal_bearish_max": config.signal_bearish_max,
        "signal_bullish_min": config.signal_bullish_min,
        "max_effective_cases": config.max_effective_cases,
        "max_events_per_date": config.max_events_per_date,
        "similarity_weight_power": config.similarity_weight_power,
        "transaction_cost": config.transaction_cost,
        "enable_risk_gate": config.enable_risk_gate,
    }
