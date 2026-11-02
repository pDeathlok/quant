"""Point-in-time monthly dividend 4%-entry/3%-exit hysteresis comparison."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/public_strategy_validation_20260929/material_backtests"
spec = importlib.util.spec_from_file_location("public_validation", ROOT / "scripts/research/validate_public_strategies.py")
validation = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(validation)


def monthly_universe() -> pd.DataFrame:
    basic = pd.read_parquet(
        ROOT / "data/research/long_dividend_quality/daily_basic_monthly_period_ends_v1.parquet",
        columns=["ts_code", "trade_date", "dv_ttm", "pe_ttm"],
    )
    basic = basic[(basic.trade_date >= "20181201") & (basic.trade_date <= "20260924")].copy()
    basic["date"] = pd.to_datetime(basic.trade_date)
    fin = pd.read_parquet(ROOT / "data/raw/fina_indicator.parquet",
                          columns=["ts_code", "ann_date", "end_date", "roe", "basic_eps_yoy"])
    fin = fin[fin.ann_date.notna()].copy()
    fin["ann"] = pd.to_datetime(fin.ann_date)
    fin = fin.sort_values(["ts_code", "ann", "end_date"]).drop_duplicates(
        ["ts_code", "ann"], keep="last"
    )
    frame = pd.merge_asof(
        basic.sort_values(["date", "ts_code"]), fin.sort_values(["ann", "ts_code"]),
        left_on="date", right_on="ann", by="ts_code", direction="backward",
    )
    stock = pd.read_parquet(ROOT / "data/raw/stock_basic_history.parquet",
                            columns=["ts_code", "list_date"])
    stock["list_date"] = pd.to_datetime(stock.list_date, errors="coerce")
    frame = frame.merge(stock, on="ts_code", how="left")
    frame = frame[
        (frame.dv_ttm > 0) & (frame.dv_ttm < 15) & (frame.pe_ttm > 0)
        & (frame.roe > 6) & (frame.basic_eps_yoy > 1)
        & ((frame.date - frame.list_date) >= pd.Timedelta(days=365))
    ]
    return frame[["date", "ts_code", "dv_ttm", "roe", "basic_eps_yoy"]]


def targets_for(universe: pd.DataFrame, top_n: int, hysteresis: bool) -> tuple[dict[pd.Timestamp, dict[str, float]], pd.DataFrame]:
    holdings: list[str] = []
    targets, audit = {}, []
    for day, group in universe.groupby("date", sort=True):
        by_code = group.set_index("ts_code")
        if hysteresis:
            holdings = [s for s in holdings if s in by_code.index and by_code.at[s, "dv_ttm"] >= 3]
        else:
            holdings = []
        entry = group[group.dv_ttm >= 4].sort_values(["dv_ttm", "ts_code"], ascending=[False, True])
        for code in entry.ts_code:
            if len(holdings) >= top_n:
                break
            if code not in holdings:
                holdings.append(code)
        targets[day] = {s: 1 / top_n for s in holdings}
        audit.append({"date": day, "top_n": top_n, "hysteresis": hysteresis,
                      "positions": len(holdings), "target_exposure": len(holdings) / top_n,
                      "symbols": ",".join(holdings)})
    return targets, pd.DataFrame(audit)


def simulate(returns: pd.DataFrame, targets: dict[pd.Timestamp, dict[str, float]]) -> tuple[pd.Series, float, int]:
    weights: dict[str, float] = {}
    wealth = 1.0
    pending = None
    output, turnover_sum, missing_held_days = [], 0.0, 0
    for day, row in returns.iterrows():
        wealth_before = wealth
        missing_held_days += sum(pd.isna(row.get(s)) for s in weights)
        asset_return = {s: float(row[s]) if s in row.index and pd.notna(row[s]) else 0.0 for s in weights}
        gross = sum(w * asset_return[s] for s, w in weights.items())
        wealth *= 1 + gross
        if weights and 1 + gross > 0:
            weights = {s: w * (1 + asset_return[s]) / (1 + gross) for s, w in weights.items()}
        if pending is not None:
            desired = {s: w for s, w in pending.items() if s in row.index and pd.notna(row[s])}
            turnover = sum(abs(desired.get(s, 0.0) - weights.get(s, 0.0))
                           for s in set(desired) | set(weights))
            wealth *= 1 - 0.001 * turnover
            turnover_sum += turnover
            weights = desired
            pending = None
        if day in targets:
            pending = targets[day]
        output.append(wealth / wealth_before - 1)
    return pd.Series(output, index=returns.index), turnover_sum, missing_held_days


def run() -> pd.DataFrame:
    universe = monthly_universe()
    scenarios = {}
    audits = []
    for top_n in (10, 50):
        for hysteresis in (False, True):
            name = f"top{top_n}_{'4in3out' if hysteresis else 'monthly_reselect'}"
            scenarios[name], audit = targets_for(universe, top_n, hysteresis)
            audit["scenario"] = name
            audits.append(audit)
    symbols = set().union(*(set(target) for scenario in scenarios.values() for target in scenario.values()))
    etf = pd.read_parquet(ROOT / "reports/public_strategy_validation_20260929/source/510300_SH_2018_2026.parquet")
    dates = pd.DatetimeIndex(pd.to_datetime(etf.trade_date)).sort_values().unique()
    dates = dates[(dates >= pd.Timestamp("2019-01-01")) & (dates <= pd.Timestamp("2026-09-24"))]
    daily = validation.stock_returns(symbols, dates)
    OUT.mkdir(parents=True, exist_ok=True)
    results, all_returns = [], {}
    for name, targets in scenarios.items():
        series, turnover, missing = simulate(daily, targets)
        series = series.loc["2019-02-01":]
        all_returns[name] = series
        for period, subset in (("full", series), ("2023_onward", series.loc["2023-01-01":])):
            results.append({"scenario": name, "period": period, **validation.metrics(subset),
                            "turnover_notional_full": turnover, "missing_held_symbol_days_full": missing,
                            "months": len(targets)})
    pd.DataFrame(all_returns).to_csv(OUT / "dividend_4in3out_daily_returns.csv", index_label="date")
    pd.concat(audits).to_csv(OUT / "dividend_4in3out_signals.csv", index=False)
    result = pd.DataFrame(results)
    result.to_csv(OUT / "dividend_4in3out_comparison.csv", index=False)
    (OUT / "dividend_4in3out_notes.json").write_text(json.dumps({
        "source": "Tushare local daily_basic, fina_indicator and daily_partitioned pct_chg",
        "signal": "Month-end dividend yield >=4% to enter; held until <3% or quality fails; monthly selection alternative",
        "quality": "ROE>6, EPS YoY>1, PE>0, listed >=365d; exact author discretionary quality process unavailable",
        "execution": "Signal month-end close, rebalance next close, 0.10% per traded notional",
        "return_basis": "Tushare pct_chg uses ex-rights-adjusted pre_close; approximate dividend-adjusted return",
        "limitations": ["No historical ST flag or limit-order execution", "Dividend taxes and exact reinvestment not simulated",
                        "Quality factor and yield thresholds are a simplified proxy for author's discretionary method"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(run()[["scenario", "period", "annual_return", "max_drawdown", "turnover_notional_full"]].to_string(index=False))
