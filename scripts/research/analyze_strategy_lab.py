"""Sensitivity checks and source-data case studies for strategy lab finalists."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_strategy_lab import (
    END, OUT, build_core_signals, core_prices, metrics, pullback, simulate_targets,
)

START = "2019-02-01"


def wilson(success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return math.nan, math.nan
    p = success / total
    divisor = 1 + z * z / total
    center = (p + z * z / (2 * total)) / divisor
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / divisor
    return center - half, center + half


def drawdown_case(name: str, daily: pd.DataFrame, prices: pd.DataFrame) -> dict:
    sample = daily.loc[START:END]
    equity = (1 + sample.daily_return).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    drawdown = equity / peak - 1
    trough = drawdown.idxmin()
    prior = equity.loc[:trough]
    high = peak.at[trough]
    peak_dates = prior.index[prior >= high - 1e-10]
    peak_day = peak_dates[-1] if len(peak_dates) else prior.index[0]
    recovery = equity.loc[trough:]
    recovered = recovery.index[recovery >= high]
    recovery_day = recovered[0] if len(recovered) else pd.NaT
    ma200 = prices["510300.SH"].rolling(200, min_periods=200).mean()
    return {
        "strategy": name, "peak_date": peak_day, "trough_date": trough,
        "recovery_date": recovery_day, "max_drawdown": float(drawdown.at[trough]),
        "calendar_days_peak_to_trough": (trough - peak_day).days,
        "trough_300_below_ma200": bool(prices.at[trough, "510300.SH"] < ma200.at[trough]),
        "trough_300_63d_return": float(prices["510300.SH"].pct_change(63, fill_method=None).at[trough]),
        "trough_nasdaq_63d_return": float(prices["513100.SH"].pct_change(63, fill_method=None).at[trough]),
        "trough_gold_63d_return": float(prices["518880.SH"].pct_change(63, fill_method=None).at[trough]),
        "trough_bond_63d_return": float(prices["511010.SH"].pct_change(63, fill_method=None).at[trough]),
    }


def main() -> None:
    prices = core_prices()
    signals = build_core_signals(prices)
    finalists = ("buy_nasdaq", "pullback_3pct_tp2_gold_300",
                 "pullback_2pct_tp2_gold_300", "invvol60_nasdaq",
                 "invvol60_nasdaq_trend", "fixed_nasdaq_25_gold25_bond50", "invvol60")
    runs = {}
    sensitivity, cases, annual_rows, worst_months = [], [], [], []
    for name in finalists:
        for scenario, fee, lag in (("base", 0.001, 1), ("double_cost", 0.002, 1),
                                    ("two_day_execution", 0.001, 2)):
            if name.startswith("pullback"):
                dip = 0.03 if "3pct" in name else 0.02
                run = pullback(prices, dip=dip, take_profit=0.02, max_hold=20,
                               candidates=("510300.SH", "518880.SH"), fee=fee, lag=lag)
            else:
                run = simulate_targets(prices, signals[name], fee=fee, lag=lag)
            sample = run.daily.loc[START:END]
            trade = run.trades
            sensitivity.append({
                "strategy": name, "scenario": scenario,
                **metrics(sample.daily_return, sample.exposure),
                "trade_count": int(len(trade)), "trade_win_rate": (
                    float((trade.net_return > 0).mean()) if len(trade) else None),
                "mean_trade_return": float(trade.net_return.mean()) if len(trade) else None,
                "turnover": run.turnover,
            })
            if scenario != "base":
                continue
            runs[name] = run
            cases.append(drawdown_case(name, run.daily, prices))
            grouped = sample.daily_return.groupby(sample.index.year)
            yearly = grouped.apply(lambda s: (1 + s).prod() - 1)
            for year, value in yearly.items():
                annual_rows.append({"strategy": name, "year": year,
                                    "trade_days": int(grouped.size().at[year]),
                                    "calendar_year_return": float(value)})
            monthly = (1 + sample.daily_return).groupby(sample.index.to_period("M")).prod() - 1
            for month, value in monthly.nsmallest(5).items():
                worst_months.append({"strategy": name, "month": str(month), "return": float(value)})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sensitivity).to_csv(OUT / "finalist_sensitivity.csv", index=False)
    pd.DataFrame(cases).to_csv(OUT / "drawdown_cases.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(OUT / "finalist_annual_returns.csv", index=False)
    pd.DataFrame(worst_months).to_csv(OUT / "worst_month_cases.csv", index=False)

    highwin = runs["pullback_3pct_tp2_gold_300"].trades.copy()
    signal_dates = pd.DatetimeIndex(highwin.entry_date).map(
        lambda date: prices.index[prices.index.get_loc(date) - 1])
    highwin["signal_date"] = signal_dates
    for code in ("510300.SH", "518880.SH"):
        mask = highwin.code == code
        history = prices[code]
        highwin.loc[mask, "signal_5d_return"] = (
            history.pct_change(5, fill_method=None).reindex(signal_dates[mask]).to_numpy())
        highwin.loc[mask, "signal_120d_return"] = (
            history.pct_change(120, fill_method=None).reindex(signal_dates[mask]).to_numpy())
        highwin.loc[mask, "signal_60d_vol"] = (
            history.pct_change(fill_method=None).rolling(60).std().mul(np.sqrt(252))
            .reindex(signal_dates[mask]).to_numpy())
    highwin.to_csv(OUT / "highwin_trades_enriched.csv", index=False)
    failures = highwin.nsmallest(10, "net_return")
    failures.to_csv(OUT / "highwin_worst_trades.csv", index=False)
    positive = int((highwin.net_return > 0).sum())
    interval = wilson(positive, len(highwin))
    print("high-win:", positive, "/", len(highwin), "Wilson 95%:", interval)
    print(pd.DataFrame(sensitivity)[["strategy", "scenario", "annual_return",
                                    "max_drawdown", "trade_win_rate"]].to_string(index=False))
    print(pd.DataFrame(cases).to_string(index=False))


if __name__ == "__main__":
    main()
