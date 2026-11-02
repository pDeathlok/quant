"""Audit representative temporal prefixes and record positive/negative cases."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

import optimize_without_nasdaq_round2 as opt
import run_strategy_lab as lab


def main():
    out = opt.OUT
    p = lab.core_prices()
    cut = pd.Timestamp("2023-12-29")
    f = opt.Features(p.loc[:cut])
    names = ("rotation_vol12_top2", "gold_vol10_weekly", "pullback_confirmation", "pullback_vol12", "stable_owntrend_vol6")
    prefix_errors = {}
    for cfg in opt.configs():
        if cfg["name"] not in names:
            continue
        run = opt.pullback_account(f, cfg) if cfg["family"] == "pullback" else lab.simulate_targets(f.p, opt.build_targets(f, cfg))
        saved = pd.read_csv(out/f'{cfg["name"]}_daily.csv', index_col="date", parse_dates=True)
        common = run.daily.index.intersection(saved.index)
        error = float((saved.loc[common, "equity"]-run.daily.loc[common, "equity"]).abs().max())
        assert error < 1e-10, (cfg["name"], error)
        prefix_errors[cfg["name"]] = error
    pairs = {"rotation_baseline": "mom240_top1", "gold_vol12_baseline": "gold_ma200_vol12_monthly_bond",
             "stable_baseline": "invvol60_market_gate", "stable_goldcap20_baseline": "invvol_gate_goldcap20"}
    old = pd.read_csv(lab.OUT/"without_nasdaq/comparison.csv").query("period=='full'").set_index("strategy")
    new = pd.read_csv(out/"comparison.csv").query("period=='full'").set_index("strategy")
    errors = {}
    for current, previous in pairs.items():
        error = float((new.loc[current, ["annual_return", "max_drawdown"]]-old.loc[previous, ["annual_return", "max_drawdown"]]).abs().max())
        assert error < 1e-10
        errors[current] = error
    # Full wealth minus marked open position reconciles with all closed cash PnL.
    cash_errors = {}
    for name in ("pullback_baseline", "pullback_half_equity", "pullback_vol12"):
        t = pd.read_csv(out/f"{name}_trades.csv")
        d = pd.read_csv(out/f"{name}_daily.csv")
        assert d.iloc[-1].exposure == 0
        error = abs(float(d.iloc[-1].equity)-(1+float(t.net_pnl.sum())))
        assert error < 1e-10
        cash_errors[name] = error
    verify = json.loads((out/"verification.json").read_text())
    verify.update(prefix_cut=str(cut.date()), truncated_history_equity_errors=prefix_errors,
                  previous_round_baseline_errors=errors, partial_cash_closed_pnl_errors=cash_errors)
    (out/"verification.json").write_text(json.dumps(verify, indent=2))

    # Compare every shared entry, also preserve new/missing entries rather than only losers.
    baseline = pd.read_csv(out/"pullback_baseline_trades.csv")
    comparisons = []
    for name in ("pullback_stop4", "pullback_hold10", "pullback_hold15", "pullback_ma60_rising"):
        t = pd.read_csv(out/f"{name}_trades.csv")
        merged = baseline[["code", "entry_date", "exit_date", "net_return"]].merge(
            t[["code", "entry_date", "exit_date", "net_return"]], on=["code", "entry_date"], how="outer", suffixes=("_base", "_variant"), indicator=True)
        merged["strategy"] = name
        merged["return_difference"] = merged.net_return_variant-merged.net_return_base
        comparisons.append(merged)
    pd.concat(comparisons, ignore_index=True).to_csv(out/"trade_case_comparison.csv", index=False)
    focus = []
    for name in ("rotation_baseline", "rotation_vol12_top2", "gold_vol12_baseline", "gold_vol15_weekly", "stable_baseline", "stable_owntrend_vol6"):
        targets = pd.read_csv(out/f"{name}_targets.csv", parse_dates=["signal_date"])
        focus.append(targets[(targets.signal_date >= "2026-01-01") & (targets.signal_date <= "2026-08-01")].assign(strategy=name))
    pd.concat(focus, ignore_index=True).fillna(0).to_csv(out/"case_2026_target_weights.csv", index=False)
    snapshots = []
    f = opt.Features(p)
    for day in ("2021-02-24", "2021-03-08", "2021-03-09", "2021-03-10", "2021-03-11", "2026-01-29", "2026-03-02", "2026-07-30"):
        d = pd.Timestamp(day)
        for code in (opt.EQ, "159915.SZ", opt.GOLD, opt.BOND):
            snapshots.append({"date": d, "code": code, "adjusted_close": p.at[d, code],
                              "daily_return": f.r.at[d, code], "return20": f.ret[20].at[d, code],
                              "above_ma200": p.at[d, code] > f.ma[200].at[d, code],
                              "vol60": f.vol[60].at[d, code]})
    pd.DataFrame(snapshots).to_csv(out/"case_price_context.csv", index=False)
    print(json.dumps(verify, indent=2))
    print(pd.concat(comparisons).query("strategy=='pullback_hold10'")[["code", "entry_date", "net_return_base", "net_return_variant", "_merge"]].to_string(index=False))


if __name__ == "__main__":
    main()
