"""Rerun the frozen forty rules on an actual ten-calendar-year window."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_strategy_lab as lab
import optimize_without_nasdaq_round2 as opt
from analyze_strategy_lab import wilson
from fetch_strategy_ten_year import OUT, CODES

START, END = "2016-09-24", "2026-09-24"
PERIODS = (("ten_year", START, END), ("added_2016_2018", START, "2018-12-31"),
           ("old_window", "2019-02-01", END), ("recent_2023", "2023-01-01", END))


def prices():
    return lab.load_prices([OUT / "source" / f"{code.replace('.', '_')}.parquet" for code in CODES])


def statistics(config, run):
    output = []
    for label, lo, hi in PERIODS:
        daily = run.daily.loc[lo:hi]
        trades = run.trades
        if len(trades):
            trades = trades[(trades.entry_date >= daily.index.min()) & (trades.exit_date <= daily.index.max())]
        stats = lab.metrics(daily.daily_return, daily.exposure)
        years = (pd.Timestamp(hi)-pd.Timestamp(lo)).days / 365.2425
        stats.update(calendar_cagr=(1+stats["total_return"])**(1/years)-1,
                     realized_vol=daily.daily_return.std()*np.sqrt(252), rows=len(daily),
                     mean_cash_weight=1-daily.exposure.mean())
        if config["family"] == "pullback":
            win = trades.net_return.gt(0)
            low, high = wilson(int(win.sum()), len(trades))
            positive, negative = trades.net_return[win], trades.net_return[~win]
            stats.update(closed_trades=len(trades), wins=int(win.sum()), trade_win_rate=float(win.mean()),
                         mean_trade_return=trades.net_return.mean(), mean_win=positive.mean(), mean_loss=negative.mean(),
                         worst_trade=trades.net_return.min(), profit_factor=positive.sum()/-negative.sum() if negative.sum()<0 else None,
                         win_ci_low=low, win_ci_high=high)
        output.append({"strategy": config["name"], "family": config["family"], "period": label,
                       "start": str(daily.index.min().date()), "end": str(daily.index.max().date()), **stats})
    return output


def simulate(f, config, target, fee=.001, lag=1, signal_start=None):
    return opt.pullback_account(f, config, fee, lag, signal_start=signal_start) if config["family"] == "pullback" else\
        lab.simulate_targets(f.p, target, fee=fee, lag=lag)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = prices()
    f = opt.Features(p)
    initial_signal = p.index[p.index < pd.Timestamp(START)][-1]
    frozen = json.loads((lab.OUT / "optimization_round2/candidate_rules.json").read_text())
    assert frozen == opt.configs(), "Rules have changed from the prior frozen experiment."
    (OUT / "candidate_rules.json").write_text(json.dumps(frozen, indent=2))
    sample = p.loc[START:END]
    calendar = pd.read_parquet(OUT / "source/trade_calendar.parquet")
    open_days = pd.DatetimeIndex(pd.to_datetime(calendar[calendar.is_open.eq(1)].cal_date))
    open_days = open_days[(open_days >= START) & (open_days <= END)]
    omitted = open_days.difference(sample.index)
    comparison, annual, sensitivity, dd, checks = [], [], [], [], {}
    full_runs = {}
    for config in frozen:
        name, family = config["name"], config["family"]
        target = None if family == "pullback" else opt.build_targets(f, config, signal_start=initial_signal, initial_signal=initial_signal)
        if target is not None:
            assert all("513100.SH" not in w and sum(w.values()) <= 1+1e-10 and min(w.values(), default=0.) >= 0 for w in target.values())
            pd.DataFrame([{"signal_date": d, **w, "cash_weight": 1-sum(w.values())} for d,w in target.items()]).fillna(0).to_csv(OUT/f"{name}_targets.csv", index=False)
        for scenario, fee, lag in (("base", .001, 1), ("double_cost", .002, 1), ("two_day_execution", .001, 2)):
            run = simulate(f, config, target, fee, lag, initial_signal)
            assert np.allclose((1+run.daily.daily_return).cumprod(), run.daily.equity)
            assert np.allclose(run.daily.loc[:initial_signal, "equity"], 1.)
            assert not run.daily.holdings.str.contains("513100", regex=False).any()
            if family == "pullback":
                assert run.daily.exposure.between(0, 1+1e-10).all()
                if len(run.trades):
                    assert all(p.index.get_loc(t.entry_date)-p.index.get_loc(t.entry_signal_date) == lag and p.index.get_loc(t.exit_date)-p.index.get_loc(t.exit_signal_date) == lag for t in run.trades.itertuples())
            rows = statistics(config, run)
            sensitivity.append({"scenario": scenario, **rows[0]})
            if scenario != "base":
                continue
            full_runs[name] = run
            comparison.extend(rows)
            daily = run.daily.loc[START:END]
            daily.to_csv(OUT/f"{name}_daily.csv", index_label="date")
            if family == "pullback":
                run.trades.to_csv(OUT/f"{name}_trades.csv", index=False)
                final = run.daily.iloc[-1]
                marked_position = final.equity-final.cash
                # Cash PnL plus unsold mark less its original investment and buy fee.
                if marked_position == 0:
                    error = abs(final.equity-(1+run.trades.net_pnl.sum()))
                    assert error < 1e-10
                    checks[name] = float(error)
            for year, group in daily.groupby(daily.index.year):
                annual.append({"strategy":name,"family":family,"year":year,"days":len(group),
                               "return":(1+group.daily_return).prod()-1,"complete_year":2017<=year<=2025})
            eq=(1+daily.daily_return).cumprod(); drawdown=eq/eq.cummax().clip(lower=1)-1
            trough=drawdown.idxmin();peak=eq.loc[:trough].idxmax()
            dd.append({"strategy":name,"peak":peak,"trough":trough,"drawdown":drawdown.min(),
                       "trough_holdings":daily.at[trough,"holdings"]})
        print(name, flush=True)
    pd.DataFrame(comparison).to_csv(OUT/"comparison.csv", index=False)
    pd.DataFrame(annual).to_csv(OUT/"annual_returns.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/"sensitivity.csv", index=False)
    pd.DataFrame(dd).to_csv(OUT/"drawdown_cases.csv", index=False)
    # Reproduce old results on old inputs. This distinguishes a changed window from a code change.
    old_prices=lab.core_prices();old_f=opt.Features(old_prices)
    previous=pd.read_csv(lab.OUT/"optimization_round2/comparison.csv").query("period=='full'").set_index("strategy")
    parity={}
    for name in ("rotation_baseline","gold_vol12_baseline","pullback_baseline","pullback_vol12","stable_baseline","stable_goldcap20_baseline","stable_owntrend_vol6"):
        cfg=next(c for c in frozen if c["name"]==name)
        target=None if cfg["family"]=="pullback" else opt.build_targets(old_f,cfg)
        run=simulate(old_f,cfg,target,signal_start="2019-01-01")
        m=lab.metrics(run.daily.loc["2019-02-01":END].daily_return,run.daily.loc["2019-02-01":END].exposure)
        error=max(abs(m[k]-previous.at[name,k]) for k in ("annual_return","max_drawdown"))
        assert error<1e-10
        parity[name]=float(error)
    # Representative prefix check on extended history, including the newly added early period.
    prefix={};cut=pd.Timestamp("2018-12-28");short=opt.Features(p.loc[:cut])
    for name in ("rotation_vol12_top2","gold_vol12_baseline","pullback_confirmation","stable_owntrend_vol6"):
        cfg=next(c for c in frozen if c["name"]==name)
        target=None if cfg["family"]=="pullback" else opt.build_targets(short,cfg,signal_start=initial_signal,initial_signal=initial_signal)
        run=simulate(short,cfg,target,signal_start=initial_signal)
        # Dec28 truncation may introduce a synthetic month-end; its signal executes outside prefix.
        error=float((run.daily.equity-full_runs[name].daily.loc[:cut,"equity"]).abs().max())
        assert error<1e-10
        prefix[name]=error
    verify={"requested_start":START,"requested_end":END,"first_observation":str(sample.index.min().date()),
            "last_observation":str(sample.index.max().date()),"common_trading_days":len(sample),
            "SSE_open_days":len(open_days),"omitted_quote_dates":[str(d.date()) for d in omitted],
            "warmup_start":str(p.index.min().date()),"initial_signal":str(initial_signal.date()),
            "rules":len(frozen),"scenario_runs":len(sensitivity),"complete_years":"2017-2025",
            "annualization_trading_days":252,"calendar_cagr_also_reported":True,"fee":.001,
            "cash_interest":0,"Nasdaq_positions":0,"frozen_parameters_equal_previous_round":True,
            "prior_results_reproduction_errors":parity,"prefix_equity_errors":prefix,"closed_cash_pnl_errors":checks,
            "new_independent_holdout":False}
    (OUT/"verification.json").write_text(json.dumps(verify,indent=2))
    full=pd.DataFrame(comparison).query("period=='ten_year'")
    print(full[["strategy","annual_return","calendar_cagr","max_drawdown","closed_trades","trade_win_rate","worst_year","annual_return_std"]].to_string(index=False))
    print(json.dumps(verify,indent=2))


if __name__=="__main__":
    main()
