"""Bounded second research round. All rules are retrospective, not unseen OOS."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_strategy_lab as lab
from review_without_nasdaq import selection_dates
from analyze_strategy_lab import wilson

OUT = lab.OUT / "optimization_round2"
START = "2019-02-01"
EQ, GOLD, BOND = "510300.SH", "518880.SH", "511010.SH"
POOL = tuple(c for c in lab.RISKY_CODES if c != "513100.SH")


def configs():
    rows = []
    def add(name, family, **kwargs):
        rows.append({"name": name, "family": family, **kwargs})
    add("rotation_baseline", "rotation")
    for budget in (.10, .12, .15):
        for cadence in ("monthly", "weekly"):
            add(f"rotation_vol{int(budget*100)}_{cadence}", "rotation", budget=budget, risk_cadence=cadence)
    add("rotation_vol12_top2", "rotation", budget=.12, top=2, risk_cadence="weekly")
    add("rotation_vol12_weekly_rank", "rotation", budget=.12, rank_cadence="weekly", risk_cadence="weekly")
    add("rotation_vol12_fast_exit", "rotation", budget=.12, risk_cadence="weekly", fast_exit=True)
    add("rotation_vol12_fastvol", "rotation", budget=.12, risk_cadence="weekly", fast_vol=True)
    add("rotation_vol12_no_us", "rotation", budget=.12, risk_cadence="weekly", no_us=True)
    add("gold_vol12_baseline", "gold", budget=.12, cadence="monthly")
    for budget in (.10, .12, .15):
        add(f"gold_vol{int(budget*100)}_weekly", "gold", budget=budget, cadence="weekly")
    add("gold_vol12_weekly_fastvol", "gold", budget=.12, cadence="weekly", fast_vol=True)
    add("gold_vol12_weekly_ma120", "gold", budget=.12, cadence="weekly", ma=120)
    add("pullback_baseline", "pullback")
    for stop in (.04, .06):
        add(f"pullback_stop{int(stop*100)}", "pullback", stop=stop)
    for hold in (10, 15):
        add(f"pullback_hold{hold}", "pullback", hold=hold)
    add("pullback_failure_exit", "pullback", failure=True)
    add("pullback_ma60_rising", "pullback", trend=True)
    add("pullback_confirmation", "pullback", confirmation=True)
    add("pullback_half_equity", "pullback", fraction=.5)
    add("pullback_vol12", "pullback", budget=.12)
    add("pullback_vol12_stop6", "pullback", budget=.12, stop=.06)
    add("stable_baseline", "stable", legacy=True)
    add("stable_goldcap20_baseline", "stable", cap=.20, eq_only=True)
    for cadence in ("monthly", "weekly"):
        for cap in (.20, .30):
            add(f"stable_owntrend_cap{int(cap*100)}_{cadence}", "stable", cap=cap, cadence=cadence)
    for budget in (.06, .08):
        add(f"stable_owntrend_vol{int(budget*100)}", "stable", cap=.30, cadence="weekly", budget=budget)
    add("stable_owntrend_bondtrend", "stable", cap=.30, cadence="weekly", bond_trend=True)
    add("stable_owntrend_corrstress", "stable", cap=.30, cadence="weekly", corr_stress=True)
    add("stable_owntrend_vol6_fastvol", "stable", cap=.30, cadence="weekly", budget=.06, fast_vol=True)
    return rows


class Features:
    def __init__(self, prices):
        self.p = prices
        self.r = prices.pct_change(fill_method=None).fillna(0)
        self.ma = {n: prices.rolling(n).mean() for n in (20, 60, 120, 200)}
        self.ret = {n: prices.pct_change(n, fill_method=None) for n in (5, 20, 60, 120, 240)}
        self.vol = {n: self.r.rolling(n).std() * np.sqrt(252) for n in (20, 60)}
        self.cov = {n: self.r.rolling(n).cov() * 252 for n in (20, 60)}
        self.corr = self.r[EQ].rolling(60).corr(self.r[GOLD])

    def portfolio_vol(self, day, weights, fast=False):
        codes = list(weights)
        if not codes:
            return 0.
        vector = np.array([weights[c] for c in codes])
        estimates = []
        for n in ((20, 60) if fast else (60,)):
            covariance = self.cov[n].loc[day].loc[codes, codes].to_numpy()
            estimates.append(np.sqrt(max(0., float(vector @ covariance @ vector))))
        return max(estimates)


def build_targets(f, config, *, signal_start="2019-01-01", initial_signal=None):
    p, family = f.p, config["family"]
    dates = selection_dates(p, config.get("cadence", config.get("risk_cadence", "monthly")))
    dates = dates[dates >= pd.Timestamp(signal_start)]
    if initial_signal is not None:
        dates = dates.union(pd.DatetimeIndex([pd.Timestamp(initial_signal)])).sort_values()
    targets = {}
    if family == "rotation":
        rank_dates = set(selection_dates(p, config.get("rank_cadence", "monthly")))
        if initial_signal is not None:
            rank_dates.add(pd.Timestamp(initial_signal))
        # Include month ends when risk is reviewed weekly.
        dates = dates.union(pd.DatetimeIndex(sorted(rank_dates))).sort_values()
        dates = dates[dates >= pd.Timestamp(signal_start)]
        chosen = {}
        pool = [c for c in POOL if not (config.get("no_us") and c == "513500.SH")]
        top = config.get("top", 1)
        for day in dates:
            if day in rank_dates:
                eligible = (f.ret[240].loc[day, pool] > 0) & (p.loc[day, pool] > f.ma[120].loc[day, pool])
                names = f.ret[240].loc[day, pool][eligible].dropna().nlargest(top).index
                chosen = {c: 1/top for c in names}
            w = chosen.copy()
            if config.get("fast_exit"):
                w = {c: weight for c, weight in w.items()
                     if not (p.at[day, c] < f.ma[60].at[day, c] and f.ret[20].at[day, c] < 0)}
            if "budget" in config:
                vol = f.portfolio_vol(day, w, config.get("fast_vol", False))
                scale = min(1., config["budget"] / max(.005, vol))
                w = {c: weight * scale for c, weight in w.items()}
                w[BOND] = 1 - sum(w.values())
            targets[day] = {c: v for c, v in w.items() if v > 1e-12}
    elif family == "gold":
        for day in dates:
            vol = f.vol[60].at[day, GOLD]
            if config.get("fast_vol"):
                vol = max(vol, f.vol[20].at[day, GOLD])
            weight = min(1., config["budget"] / max(.005, vol)) if p.at[day, GOLD] > f.ma[config.get("ma", 200)].at[day, GOLD] else 0.
            targets[day] = {c: v for c, v in {GOLD: weight, BOND: 1-weight}.items() if v > 1e-12}
    else:
        for day in dates:
            w = lab.capped_inverse_vol(f.vol[60].loc[day, [EQ, GOLD, BOND]])
            if config.get("legacy"):
                if p.at[day, EQ] <= f.ma[200].at[day, EQ]:
                    w = {GOLD: .5, BOND: .5}
            else:
                eq_on = p.at[day, EQ] > f.ma[200].at[day, EQ] if config.get("eq_only") else (
                    p.at[day, EQ] > f.ma[120].at[day, EQ] and f.ret[60].at[day, EQ] > 0)
                if not eq_on:
                    w[BOND] += w.pop(EQ, 0.)
                if not config.get("eq_only") and not (p.at[day, GOLD] > f.ma[200].at[day, GOLD] and f.ret[60].at[day, GOLD] > 0):
                    w[BOND] += w.pop(GOLD, 0.)
                cap = config.get("cap", .30)
                if w.get(GOLD, 0.) > cap:
                    w[BOND] += w[GOLD] - cap
                    w[GOLD] = cap
            if config.get("corr_stress") and f.ret[20].at[day, EQ] < 0 and f.ret[20].at[day, GOLD] < 0 and f.corr.at[day] > .3:
                for c in (EQ, GOLD):
                    if c in w:
                        w[c] *= .5  # Removed capital stays in cash.
            if config.get("bond_trend") and p.at[day, BOND] < f.ma[60].at[day, BOND] and f.ret[20].at[day, BOND] < 0:
                w.pop(BOND, None)
            if "budget" in config:
                vol = f.portfolio_vol(day, w, config.get("fast_vol", False))
                scale = min(1., config["budget"] / max(.005, vol))
                w = {c: weight * scale for c, weight in w.items()}
            targets[day] = {c: v for c, v in w.items() if v > 1e-12}
    return targets


def pullback_account(f, config, fee=.001, lag=1, *, signal_start="2019-01-01"):
    """Explicit shares/cash. Size only at entry, never rebalance inside a trade."""
    dates, p = f.p.index, f.p
    cash, shares, position, last_wealth, turnover = 1., 0., None, 1., 0.
    pending, candidate = None, None
    rows, trades = [], []
    entry_day, entry_price, entry_i, entry_cash, entry_signal = None, 0., -1, 0., None
    for i, day in enumerate(dates):
        just_exited = False
        if pending is not None and pending[0] == i:
            _, action, code, reason, signal_day, fraction = pending
            wealth = cash + (shares * p.at[day, position] if position else 0.)
            if action == "enter":
                entry_cash = wealth * fraction
                shares = entry_cash * (1-fee) / p.at[day, code]
                cash -= entry_cash
                position, entry_day, entry_price, entry_i, entry_signal = code, day, p.at[day, code], i, signal_day
                turnover += fraction
            else:
                proceeds = shares * p.at[day, position] * (1-fee)
                trades.append({"code": position, "entry_signal_date": entry_signal, "entry_date": entry_day,
                               "exit_signal_date": signal_day, "exit_date": day,
                               "net_return": proceeds/entry_cash - 1, "net_pnl": proceeds-entry_cash,
                               "entry_cash": entry_cash, "holding_days": i-entry_i, "exit_reason": reason})
                turnover += shares * p.at[day, position] / wealth
                cash += proceeds
                position, shares = None, 0.
                just_exited = True
            pending = None
        wealth = cash + (shares * p.at[day, position] if position else 0.)
        exposure = shares * p.at[day, position] / wealth if position else 0.
        if i >= 200 and day >= pd.Timestamp(signal_start) and i+lag < len(dates) and not just_exited and pending is None:
            if position:
                gross = p.at[day, position]/entry_price - 1
                reason = None
                if gross >= .02:
                    reason = "take_profit"
                elif p.at[day, position] < f.ma[200].at[day, position]:
                    reason = "trend_break"
                elif "stop" in config and gross <= -config["stop"]:
                    reason = "close_stop"
                elif config.get("failure") and i-entry_i >= 5 and gross < 0 and p.at[day, position] < f.ma[20].at[day, position]:
                    reason = "failure_exit"
                elif i-entry_i >= config.get("hold", 20):
                    reason = "max_hold"
                if reason:
                    pending = (i+lag, "exit", position, reason, day, 0.)
            else:
                eligible = [c for c in (EQ, GOLD) if p.at[day, c] > f.ma[200].at[day, c]
                            and f.ret[120].at[day, c] > 0 and f.vol[60].at[day, c] <= .30
                            and -.10 <= f.ret[5].at[day, c] <= -.03
                            and (not config.get("trend") or (p.at[day, c] > f.ma[60].at[day, c]
                                 and f.ma[60].at[day, c] > f.ma[60].iloc[i-20][c]))]
                code = None
                signal = day
                if config.get("confirmation"):
                    if candidate is not None:
                        prev_i, prev_code, prev_day, prev_px = candidate
                        # One next-session confirmation, recheck long trend and volatility.
                        if i == prev_i+1 and p.at[day, prev_code] > prev_px and p.at[day, prev_code] > f.ma[200].at[day, prev_code] and f.ret[120].at[day, prev_code] > 0 and f.vol[60].at[day, prev_code] <= .30:
                            code = prev_code
                        candidate = None
                    if code is None and eligible:
                        c = max(eligible, key=lambda c: f.ret[120].at[day, c])
                        candidate = (i, c, day, p.at[day, c])
                elif eligible:
                    code = max(eligible, key=lambda c: f.ret[120].at[day, c])
                if code:
                    fraction = config.get("fraction", 1.)
                    if "budget" in config:
                        fraction = min(1., config["budget"]/max(.005, f.vol[60].at[day, code]))
                    pending = (i+lag, "enter", code, "pullback", signal, fraction)
        rows.append({"date": day, "daily_return": wealth/last_wealth-1, "equity": wealth,
                     "exposure": exposure, "holdings": position or "", "cash": cash})
        last_wealth = wealth
    return lab.Run(pd.DataFrame(rows).set_index("date"), pd.DataFrame(trades), turnover)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rules = configs()
    (OUT / "candidate_rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=2))
    p = lab.core_prices()
    f = Features(p)
    # Baseline parity is a concrete accounting gate, including partial cash bookkeeping.
    old = lab.pullback(p, dip=.03, take_profit=.02, max_hold=20, max_vol=.30, candidates=(EQ, GOLD))
    new = pullback_account(f, {"family": "pullback"})
    parity = float((old.daily.equity-new.daily.equity).abs().max())
    assert parity < 1e-10
    assert len(old.trades) == len(new.trades)
    rows, years, sensitivity, cases = [], [], [], []
    for config in rules:
        name, family = config["name"], config["family"]
        target = None if family == "pullback" else build_targets(f, config)
        if target is not None:
            assert all("513100.SH" not in w and all(v >= 0 for v in w.values()) and sum(w.values()) <= 1+1e-10 for w in target.values())
            pd.DataFrame([{"signal_date": day, **w, "cash_weight": 1-sum(w.values())}
                          for day, w in target.items()]).fillna(0).to_csv(OUT / f"{name}_targets.csv", index=False)
        for scenario, fee, lag in (("base", .001, 1), ("double_cost", .002, 1), ("two_day_execution", .001, 2)):
            run = pullback_account(f, config, fee, lag) if family == "pullback" else lab.simulate_targets(p, target, fee=fee, lag=lag)
            sample = run.daily.loc[START:lab.END]
            assert not sample.holdings.str.contains("513100", regex=False).any()
            assert np.allclose((1+run.daily.daily_return).cumprod(), run.daily.equity)
            if family == "pullback" and len(run.trades):
                assert all(p.index.get_loc(t.entry_date)-p.index.get_loc(t.entry_signal_date) == lag and p.index.get_loc(t.exit_date)-p.index.get_loc(t.exit_signal_date) == lag for t in run.trades.itertuples())
            summary = lab.summarize(name, run, START, lab.END)
            for row, (_, lo, hi) in zip(summary, (("full", START, lab.END), ("early", START, "2022-12-31"), ("late", "2023-01-01", lab.END))):
                row["realized_vol"] = float(run.daily.loc[lo:hi, "daily_return"].std()*np.sqrt(252))
            # Asset enter/remove events in rebalanced portfolios are not account round trips.
            if family != "pullback":
                for row in summary:
                    for field in ("closed_trades", "trade_win_rate", "mean_trade_return", "mean_win", "mean_loss"):
                        row[field] = None
            sensitivity.append({"family": family, "scenario": scenario, **summary[0]})
            if scenario != "base":
                continue
            for row in summary:
                if family == "pullback":
                    trade = run.trades
                    lo, hi = (START, lab.END) if row["period"] == "full" else ((START, "2022-12-31") if row["period"] == "selection_2019_2022" else ("2023-01-01", lab.END))
                    trade = trade[(trade.entry_date >= pd.Timestamp(lo)) & (trade.exit_date <= pd.Timestamp(hi))]
                    pos, neg = trade.net_return[trade.net_return > 0], trade.net_return[trade.net_return <= 0]
                    interval = wilson(len(pos), len(trade))
                    row.update(profit_factor=float(pos.sum()/-neg.sum()) if neg.sum() < 0 else None,
                               worst_trade=float(trade.net_return.min()) if len(trade) else None,
                               win_ci_low=interval[0], win_ci_high=interval[1])
                rows.append({"family": family, **row})
            sample.to_csv(OUT / f"{name}_daily.csv", index_label="date")
            if family == "pullback":
                run.trades.to_csv(OUT / f"{name}_trades.csv", index=False)
            for year, frame in sample.groupby(sample.index.year):
                years.append({"strategy": name, "family": family, "year": year, "days": len(frame),
                              "return": (1+frame.daily_return).prod()-1})
            equity = (1+sample.daily_return).cumprod()
            peak = equity.cummax().clip(lower=1.)
            trough = (equity/peak-1).idxmin()
            peak_day = equity.loc[:trough].idxmax()
            cases.append({"strategy": name, "family": family, "peak": peak_day, "trough": trough,
                          "drawdown": (equity/peak-1).min(), "trough_holdings": sample.at[trough, "holdings"]})
        print(name, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(years).to_csv(OUT / "annual_returns.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "sensitivity.csv", index=False)
    pd.DataFrame(cases).to_csv(OUT / "drawdown_cases.csv", index=False)
    (OUT / "verification.json").write_text(json.dumps({"pullback_baseline_max_equity_error": parity,
        "baseline_closed_trade_count_equal": True, "equity_return_identity": True,
        "pullback_signal_execution_lag_checked": True, "nonnegative_unlevered_target_weights": True,
        "Nasdaq_positions": 0, "new_unseen_holdout": False, "rules": len(rules),
        "fee": .001, "cash_interest": 0, "start": START, "end": lab.END,
        "trading_days": len(p.loc[START:lab.END]),
        "target_engine": "previous-round proportional turnover cost convention; fractional ETF units"}, indent=2))
    print(pd.DataFrame(rows).query("period=='full'")[["strategy", "annual_return", "max_drawdown", "closed_trades", "trade_win_rate", "worst_year", "annual_return_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
