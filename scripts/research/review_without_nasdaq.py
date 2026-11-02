"""Rerank the existing lab without Nasdaq and retain a small optimization grid."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_strategy_lab as lab
from analyze_strategy_lab import wilson

OUT = lab.OUT / "without_nasdaq"
START = "2019-02-01"


def selection_dates(prices: pd.DataFrame, cadence: str) -> pd.DatetimeIndex:
    if cadence == "monthly":
        return lab.month_ends(prices.index)
    return pd.DatetimeIndex(prices.index.to_series().groupby(prices.index.to_period("W-FRI")).last())


def gold_targets(prices: pd.DataFrame, ma: int, cadence: str, target_vol: float | None = None):
    gold = prices["518880.SH"]
    trend = gold.rolling(ma).mean()
    vol = gold.pct_change(fill_method=None).rolling(60).std() * np.sqrt(252)
    targets = {}
    for date in selection_dates(prices, cadence):
        if date < pd.Timestamp("2019-01-01"):
            continue
        weight = (1.0 if target_vol is None else min(1.0, target_vol / max(vol.at[date], .005)))\
            if gold.at[date] > trend.at[date] else 0.0
        targets[date] = {code: value for code, value in
                         {"518880.SH": weight, "511010.SH": 1 - weight}.items() if value > 0}
    return targets


def split_initial_capital(run: lab.Run, fraction: float) -> lab.Run:
    wealth = fraction * run.daily.equity + 1 - fraction
    values = wealth.pct_change(fill_method=None)
    values.iloc[0] = wealth.iloc[0] - 1
    exposure = fraction * run.daily.equity / wealth * run.daily.exposure
    daily = run.daily.copy()
    daily["equity"], daily["daily_return"], daily["exposure"] = wealth, values, exposure
    return lab.Run(daily, run.trades.copy(), fraction * run.turnover)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = lab.core_prices()
    # Only this research process changes the candidate pool; source files stay unchanged.
    lab.RISKY_CODES = tuple(code for code in lab.RISKY_CODES if code != "513100.SH")
    generated = lab.build_core_signals(prices)
    nasdaq_free_pool = lab.RISKY_CODES
    lab.RISKY_CODES = tuple(code for code in nasdaq_free_pool if code != "513500.SH")
    no_us_240 = lab.build_core_signals(prices)["mom240_top1"]
    lab.RISKY_CODES = nasdaq_free_pool
    keep = ("mom120_top1", "mom120_top2", "mom240_top1", "risk_adj120_top1", "mom120_domestic_gate",
            "fixed_40_30_30", "invvol60", "invvol60_market_gate", "fixed_market_gate",
            "bond_gold_50_50", "fixed_25_25_50", "fixed_20_30_50")
    targets = {name: generated[name] for name in keep}
    for name, code in (("buy_300", "510300.SH"), ("buy_500", "510500.SH"), ("buy_chinext", "159915.SZ"),
                       ("buy_hangseng", "159920.SZ"), ("buy_sp500", "513500.SH"),
                       ("buy_gold", "518880.SH"), ("buy_bond", "511010.SH")):
        targets[name] = {date: {code: 1.0} for date in lab.month_ends(prices.index)
                         if date >= pd.Timestamp("2019-01-01")}
    baseline_names = set(targets)
    targets["mom240_no_us_top1"] = no_us_240
    for ma in (120, 200):
        for cadence in ("monthly", "weekly"):
            targets[f"gold_ma{ma}_{cadence}_bond"] = gold_targets(prices, ma, cadence)
    targets["gold_ma200_vol12_monthly_bond"] = gold_targets(prices, 200, "monthly", .12)
    market_on = prices["510300.SH"] > prices["510300.SH"].rolling(200).mean()
    for gold_cap in (None, .2, .3):
        name = "invvol_gate_300_only_to_bond" if gold_cap is None else f"invvol_gate_goldcap{int(gold_cap*100)}"
        desired = {}
        for date, weights in generated["invvol60"].items():
            w = weights.copy()
            if not market_on.at[date]:
                w["511010.SH"] += w.pop("510300.SH", 0)
            if gold_cap is not None and w["518880.SH"] > gold_cap:
                w["511010.SH"] += w["518880.SH"] - gold_cap
                w["518880.SH"] = gold_cap
            desired[date] = w
        targets[name] = desired
    assert all("513100.SH" not in weight for scenario in targets.values() for weight in scenario.values())
    pullbacks = {}
    for dip, tp, hold in ((.01, .02, 20), (.02, .02, 20), (.03, .02, 20), (.04, .02, 20),
                          (.02, .03, 20), (.04, .05, 30)):
        name = f"pullback_dip{int(dip*100)}_tp{int(tp*100)}_h{hold}"
        pullbacks[name] = {"dip": dip, "take_profit": tp, "max_hold": hold}
        baseline_names.add(name)
    pullbacks["pullback_dip2_tp2_gold_only"] = {"dip": .02, "take_profit": .02, "max_hold": 20,
                                              "candidates": ("518880.SH",)}
    baseline_names.add("pullback_dip2_tp2_gold_only")
    for vol in (.30, .35, .45):
        pullbacks[f"pullback_dip3_tp2_vol{int(vol*100)}"] = {
            "dip": .03, "take_profit": .02, "max_hold": 20, "max_vol": vol}
    pullbacks["pullback_dip3_tp2_rising_ma"] = {"dip": .03, "take_profit": .02,
                                               "max_hold": 20, "rising_ma": True}
    runs = {name: lab.simulate_targets(prices, values) for name, values in targets.items()}
    for name, config in pullbacks.items():
        config.setdefault("candidates", ("510300.SH", "518880.SH"))
        runs[name] = lab.pullback(prices, **config)
    runs["pullback_dip3_tp2_half_initial_cash"] = split_initial_capital(runs["pullback_dip3_tp2_h20"], .5)
    rows, annual = [], []
    for name, run in runs.items():
        rows.extend({"stage": "baseline" if name in baseline_names else "exploratory_optimization", **row}
                    for row in lab.summarize(name, run, START, lab.END))
        sample = run.daily.loc[START:lab.END]
        assert not sample.holdings.str.contains("513100", regex=False).any()
        sample.to_csv(OUT / f"{name}_daily.csv", index_label="date")
        if len(run.trades):
            run.trades.to_csv(OUT / f"{name}_trades.csv", index=False)
        for year, frame in sample.groupby(sample.index.year):
            annual.append({"strategy": name, "year": year, "days": len(frame),
                           "return": (1 + frame.daily_return).prod() - 1})
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(annual).to_csv(OUT / "annual_returns.csv", index=False)
    names = ("mom240_top1", "mom240_no_us_top1", "buy_gold", "pullback_dip3_tp2_h20", "pullback_dip3_tp2_vol30",
             "invvol60", "invvol60_market_gate", "invvol_gate_goldcap20",
             "gold_ma200_monthly_bond", "gold_ma200_vol12_monthly_bond", "invvol_gate_300_only_to_bond")
    sensitivity = []
    for name in names:
        for scenario, fee, lag in (("base", .001, 1), ("double_cost", .002, 1), ("two_day_execution", .001, 2)):
            run = lab.pullback(prices, **pullbacks[name], fee=fee, lag=lag) if name in pullbacks else\
                lab.simulate_targets(prices, targets[name], fee=fee, lag=lag)
            row = lab.summarize(name, run, START, lab.END)[0]
            sensitivity.append({"scenario": scenario, **row})
    pd.DataFrame(sensitivity).to_csv(OUT / "sensitivity.csv", index=False)
    trades = runs["pullback_dip3_tp2_h20"].trades
    wins = int(trades.net_return.gt(0).sum())
    info = {"start": START, "end": lab.END, "trading_days": len(prices.loc[START:lab.END]),
            "strategies": len(runs), "baseline_strategies": len(baseline_names),
            "Nasdaq_positions": 0, "new_independent_holdout": False,
            "highwin_wins": wins, "highwin_trades": len(trades), "highwin_wilson95": wilson(wins, len(trades))}
    (OUT / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    full = pd.DataFrame(rows).query("period=='full'")
    print(json.dumps(info), flush=True)
    print(full[["strategy", "stage", "annual_return", "max_drawdown", "closed_trades", "trade_win_rate",
                "worst_year", "annual_return_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
