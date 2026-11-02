"""Predeclared drawdown overlays on the public-strategy validation returns."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/public_strategy_validation_20260929"
FEE = 0.001

spec = importlib.util.spec_from_file_location("public_validation", ROOT / "scripts/research/validate_public_strategies.py")
validation = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(validation)


def overlay(returns: pd.Series, signal: pd.Series) -> pd.Series:
    """Signal at close t; change exposure at next close; earn it from t+2."""
    exposure = 1.0
    pending = None
    output = []
    for date, ret in returns.items():
        pnl = exposure * float(ret)
        if pending is not None:
            pnl -= FEE * abs(pending - exposure)
            exposure = pending
            pending = None
        if date in signal.index and pd.notna(signal.loc[date]):
            pending = float(signal.loc[date])
        output.append(pnl)
    return pd.Series(output, index=returns.index)


def run() -> dict:
    raw = pd.read_csv(BASE / "daily_returns.csv", parse_dates=["date"]).set_index("date")
    raw.index = pd.DatetimeIndex(raw.index)
    source = pd.read_parquet(BASE / "source/510300_SH_2018_2026.parquet")
    source["date"] = pd.to_datetime(source.trade_date)
    close = (source.set_index("date").sort_index().close
             * source.set_index("date").sort_index().adj_factor)
    market_trend = close > close.rolling(200, min_periods=200).mean()
    month_ends = raw.groupby(raw.index.to_period("M")).tail(1).index
    trend_signal = market_trend.reindex(month_ends).astype(float)

    portfolios = {}
    for name in ("dividend_quality_price_only", "etf_momentum_adjusted"):
        ret = raw[name]
        vol = ret.rolling(60, min_periods=60).std() * np.sqrt(252)
        vol_signal = (0.12 / vol).clip(lower=0.20, upper=1.0).reindex(month_ends)
        portfolios[name] = {
            "base": ret,
            "60pct_equity_40pct_cash": ret * 0.6,
            "market_200d_trend_cash": overlay(ret, trend_signal),
            "12pct_volatility_target_cash": overlay(ret, vol_signal),
        }
    rows = []
    for name, strategies in portfolios.items():
        for variant, series in strategies.items():
            for period, subset in (("full", series), ("2023_onward", series[series.index >= "2023-01-01"])):
                rows.append({"underlying": name, "overlay": variant, "period": period,
                             **validation.metrics(subset)})
    results = pd.DataFrame(rows)
    results.to_csv(BASE / "drawdown_overlay_results.csv", index=False)
    return {"results": rows, "notes": [
        "60/40 assumes daily constant exposure; incremental rebalancing cost is not modeled.",
        "Trend and volatility overlays change exposure at next close, charge 0.10% per changed notional.",
        "The 2023+ view is a later-period check, not a true untouched holdout: variants were chosen after seeing the first full-sample results.",
        "Stock strategy uses Tushare ex-rights-adjusted pct_chg proxy with the execution and data limits of the base validation.",
    ]}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
