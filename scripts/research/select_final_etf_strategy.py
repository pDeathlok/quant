"""Select one implementable ETF policy with PIT NAV gates and a cash ledger."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import explore_six_sleeves as prior

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/research/final_etf_strategy_20261111"
CODES = ("510300.SH", "510880.SH", "511010.SH", "511260.SH", "518880.SH",
         "159985.SZ", "159981.SZ", "159980.SZ")
LABELS = ("沪深300ETF", "红利ETF", "5年国债ETF", "10年国债ETF", "黄金ETF",
          "豆粕期货ETF", "能源化工期货ETF", "有色金属期货ETF")
START = pd.Timestamp("2021-01-04")
INITIAL = 1_000_000.
CONFIGS = {"defensive": np.array([.1, .1, .25, .25, .2, .1/3, .1/3, .1/3]),
           "balanced": np.array([.15, .15, .2, .2, .2, .1/3, .1/3, .1/3]),
           "simple": np.array([.15, .15, .25, .25, .2, 0., 0., 0.])}


@dataclass
class Panel:
    close: pd.DataFrame
    adjusted: pd.DataFrame
    amount: pd.DataFrame
    valid: pd.DataFrame
    eligible: pd.DataFrame
    premium: pd.DataFrame
    nav_dates: pd.DataFrame
    nav_ann: pd.DataFrame
    adv: pd.DataFrame


def known_nav_premium(nav: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """At close t, use only NAV announcements dated strictly before t.

    Match NAV to the price on its own valuation date, never divide today's price
    by yesterday's NAV. Same-day NAV may be published after the close.
    """
    f = nav.copy()
    f["nav_date"] = pd.to_datetime(f.nav_date, errors="coerce")
    f["ann_date"] = pd.to_datetime(f.ann_date, errors="coerce")
    f["unit_nav"] = pd.to_numeric(f.unit_nav, errors="coerce")
    f = f.dropna(subset=["nav_date", "ann_date", "unit_nav"])
    f = f[(f.unit_nav > 0) & (f.ann_date >= f.nav_date)]
    if f.duplicated(["nav_date", "ann_date"]).any():
        raise ValueError("Ambiguous NAV revisions")
    f["price"] = f.nav_date.map(close)
    f["premium"] = f.price/f.unit_nav-1
    # On a disclosure day keep its newest valuation. No NAV from future dates.
    f = f.dropna(subset=["premium"]).sort_values(["ann_date", "nav_date"]).drop_duplicates("ann_date", keep="last")
    result = pd.merge_asof(pd.DataFrame({"date": close.index}),
                          f[["ann_date", "nav_date", "premium"]],
                          left_on="date", right_on="ann_date", direction="backward", allow_exact_matches=False)
    return result.set_index("date")


def load_panel() -> Panel:
    directory = ROOT / "reports/research/six_sleeve_exploration_20261111/source"
    extra = ROOT / "reports/research/diversifiers_20261111/source"
    calendar = pd.read_parquet(directory/"510300_SH.parquet").trade_date
    dates = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    fields = {k: {} for k in ("close", "adjusted", "amount", "valid", "premium", "nav_dates", "nav_ann")}
    manifest = []
    for code in CODES:
        name = code.replace(".", "_")
        path = directory/f"{name}.parquet"
        if not path.exists(): path = extra/f"{name}.parquet"
        raw = pd.read_parquet(path); raw.index = pd.to_datetime(raw.trade_date)
        if raw.index.has_duplicates: raise ValueError("Duplicate prices")
        f = raw.reindex(dates)
        fields["valid"][code] = f.close.notna() & f.amount.gt(0)
        fields["close"][code] = f.close.ffill()
        fields["adjusted"][code] = (f.close*f.adj_factor).ffill()
        fields["amount"][code] = f.amount.fillna(0)*1000
        nav_path = OUT/"source"/f"{name}_nav.parquet"
        known = known_nav_premium(pd.read_parquet(nav_path), f.close)
        for field, col in [("premium", "premium"), ("nav_dates", "nav_date"), ("nav_ann", "ann_date")]:
            fields[field][code] = known[col]
        manifest.append({"code": code, "price_path": str(path), "price_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "nav_sha256": hashlib.sha256(nav_path.read_bytes()).hexdigest(),
                         "latest_price_date": str(raw.index.max().date()),
                         "valid_quotes": int(fields["valid"][code].sum())})
    tables = {k: pd.DataFrame(v, index=dates) for k, v in fields.items()}
    adv = tables["amount"].rolling(20, min_periods=20).median()
    age = tables["nav_dates"].apply(lambda x: (pd.Series(dates, index=dates)-x).dt.days)
    eligible = tables["valid"] & (tables["valid"].cumsum() >= 120) & (adv >= 20_000_000)
    eligible &= (age >= 0) & (age <= 7) & tables["premium"].le(.005) & tables["premium"].ge(-.5)
    (OUT/"source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return Panel(**tables, eligible=eligible, adv=adv)


def make_targets(panel: Panel, allocation: str, mode: str, *, momentum: int = 63,
                 ma: int = 120, vol_window: int = 60, vol_target: float = .08,
                 budget_override: np.ndarray | None = None, include_last: bool = False) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    prices = panel.adjusted
    returns = prices.pct_change(fill_method=None)
    momentum_values = prices.pct_change(momentum, fill_method=None)
    moving = prices.rolling(ma, min_periods=ma).mean()
    budget = CONFIGS[allocation] if budget_override is None else budget_override
    decision_days = prior.month_ends(prices.index)
    if include_last and len(prices)-1 not in decision_days: decision_days.append(len(prices)-1)
    targets, records = {}, []
    for t in decision_days:
        if prices.index[t] < START-pd.Timedelta(days=10): continue
        w = budget.copy()*panel.eligible.iloc[t].to_numpy()
        trend = (momentum_values.iloc[t] > 0) & (prices.iloc[t] > moving.iloc[t])
        if "trend" in mode:
            trend.iloc[[2, 3]] = True
            w *= trend.to_numpy()
        raw_cov = returns.iloc[t-vol_window+1:t+1].cov(min_periods=vol_window).to_numpy()
        cov = .75*raw_cov + .25*np.diag(np.diag(raw_cov))
        if not np.isfinite(cov).all():
            w[:] = 0
            estimated = 0.
        else:
            estimated = float(np.sqrt(max(w@cov@w*252, 0)))
            if "vol" in mode and estimated > vol_target:
                w *= vol_target/estimated
        targets[t] = w
        for j, code in enumerate(CODES):
            records.append({"signal_date": str(prices.index[t].date()), "code": code, "label": LABELS[j],
                            "budget": float(budget[j]), "target": float(w[j]),
                            "eligible": bool(panel.eligible.iloc[t,j]), "trend_pass": bool(trend.iloc[j]),
                            "momentum": float(momentum_values.iloc[t,j]), "ma": float(moving.iloc[t,j]),
                            "adjusted_close": float(prices.iloc[t,j]),
                            "known_premium": float(panel.premium.iloc[t,j]),
                            "nav_date": str(panel.nav_dates.iloc[t,j].date()) if pd.notna(panel.nav_dates.iloc[t,j]) else None,
                            "nav_announcement": str(panel.nav_ann.iloc[t,j].date()) if pd.notna(panel.nav_ann.iloc[t,j]) else None,
                            "median20_amount_CNY": float(panel.adv.iloc[t,j]),
                            "unscaled_portfolio_vol": estimated})
    return targets, pd.DataFrame(records)


def trade_fee(notional: float, multiplier: float = 1.) -> float:
    return (max(notional*.0002, 5.) + notional*.0008)*multiplier if notional > 1e-8 else 0.


def simulate(panel: Panel, targets: dict[int, np.ndarray], *, lag: int = 1,
             cost_multiplier: float = 1., initial: float = INITIAL) -> tuple[prior.Result, pd.DataFrame]:
    """Self-financing integer-lot trades; total-return marking approximates reinvestment."""
    p = panel.adjusted
    returns = p.pct_change(fill_method=None).fillna(0).to_numpy()
    close = panel.close.to_numpy(); valid = panel.valid.to_numpy()
    adv = panel.adv.shift(1).fillna(0).to_numpy()
    n, k = p.shape
    holdings, cash = np.zeros(k), initial
    daily, weights, turns, costs = np.zeros(n), np.zeros((n, k)), np.zeros(n), np.zeros(n)
    orders, skipped = [], []
    for t in range(p.index.searchsorted(START), n):
        before = cash+holdings.sum()
        holdings *= 1+returns[t]
        wealth = float(cash+holdings.sum())
        target = targets.get(t-lag)
        fees, traded = 0., 0.
        if target is not None:
            if target.min() < 0 or target.sum() > 1+1e-10: raise ValueError("Invalid target")
            desired = target*wealth
            # Selling first prevents accidental borrowing. Missing quotes never execute.
            for j in range(k):
                if holdings[j] <= desired[j]+1e-8: continue
                if not valid[t,j]:
                    skipped.append(f"{p.index[t].date()}:{CODES[j]}:sell"); continue
                gap = min(holdings[j]-desired[j], .01*adv[t,j])
                notional = np.floor(gap/(100*close[t,j]))*100*close[t,j]
                if target[j] == 0 and gap >= holdings[j]-1e-7:
                    notional = holdings[j]  # Exit odd/fractional residues from reinvestment approximation.
                if notional <= 0: continue
                fee = trade_fee(float(notional), cost_multiplier)
                if fee >= notional: continue
                holdings[j] -= notional; cash += notional-fee
                fees += fee; traded += notional
                orders.append({"date": str(p.index[t].date()), "signal_date": str(p.index[t-lag].date()),
                               "code": CODES[j], "side": "sell", "notional": float(notional), "fee": fee})
            gaps = np.maximum(desired-holdings, 0.)
            # Reserve maximum commissions and proportionately reduce unfunded requests.
            reserve = 5*cost_multiplier*int((gaps > 0).sum())
            scale = min(1., max(cash-reserve, 0)/(gaps.sum()*(1+.001*cost_multiplier))) if gaps.sum() else 0.
            for j in range(k):
                if gaps[j] <= 0: continue
                if not valid[t,j]:
                    skipped.append(f"{p.index[t].date()}:{CODES[j]}:buy"); continue
                amount = min(gaps[j]*scale, .01*adv[t,j])
                lot = 100*close[t,j]
                notional = float(np.floor(amount/lot)*lot)
                while notional > 0 and notional+trade_fee(notional, cost_multiplier) > cash+1e-8:
                    notional -= lot
                if notional <= 0: continue
                fee = trade_fee(notional, cost_multiplier)
                holdings[j] += notional; cash -= notional+fee
                fees += fee; traded += notional
                orders.append({"date": str(p.index[t].date()), "signal_date": str(p.index[t-lag].date()),
                               "code": CODES[j], "side": "buy", "notional": notional, "fee": fee,
                               "shares": round(notional/close[t,j], 8)})
        after = float(cash+holdings.sum())
        if cash < -1e-6 or holdings.min() < -1e-6: raise ValueError("Ledger borrowing invariant")
        daily[t] = after/before-1; weights[t] = holdings/after
        turns[t] = traded/wealth; costs[t] = fees/before
    idx = p.index
    return prior.Result(pd.Series(daily,index=idx), pd.DataFrame(weights,index=idx,columns=CODES),
                        pd.Series(turns,index=idx), pd.Series(costs,index=idx), skipped), pd.DataFrame(orders)


def qualify_and_score(r: pd.Series, turnover: pd.Series) -> dict:
    selection = r.loc["2021-01-04":"2025-12-31"]
    m = prior.metrics(selection, turnover.loc[selection.index])
    annual = (1+selection).groupby(selection.index.year).prod()-1
    qualifies = m["max_drawdown"] >= -.12 and m["worst_day"] >= -.04 and m["volatility"] <= .10 and annual.min() >= -.08
    score = float(annual.median() + .5*m["max_drawdown"] - .0005*m["annual_traded_notional"])
    return {**m, "worst_year": float(annual.min()), "median_year": float(annual.median()),
            "qualifies": bool(qualifies), "score": score}


def select_candidate(table: pd.DataFrame) -> tuple[str, bool]:
    qualified = table[table.all_execution_cases_pass].copy()
    passed = len(qualified) > 0
    if not passed:
        # Explicit fallback: smallest historical drawdown, then worst-day loss.
        selected = table.sort_values(["max_drawdown", "worst_day"], ascending=False).index[0]
        return str(selected), False
    top = qualified.score.max()
    near = qualified[qualified.score >= top-.003]
    selected = near.sort_values(["complexity", "annual_traded_notional", "max_drawdown"], ascending=[True, True, False]).index[0]
    return str(selected), True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel(); mask = panel.close.index >= START
    rows, net_returns, executions, audits, sims = [], {}, {}, {}, {}
    variations = [("base", {}), ("double_cost", {"cost_multiplier": 2.}), ("delay2", {"lag": 2})]
    for allocation in CONFIGS:
        for mode in ("fixed", "trend", "vol", "trendvol"):
            name = f"{allocation}_{mode}"
            targets, audit = make_targets(panel, allocation, mode)
            stats = {}
            for variant, kwargs in variations:
                result, orders = simulate(panel, targets, **kwargs)
                m = qualify_and_score(result.returns, result.turnover)
                rows.append({"strategy": name, "variant": variant, **m})
                stats[variant] = m
                if variant == "base":
                    net_returns[name] = result.returns.loc[mask]
                    sims[name] = result; executions[name] = orders; audits[name] = audit
            print(name, "selection CAGR", round(stats["base"]["cagr"],4), "MDD",round(stats["base"]["max_drawdown"],4),flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT/"all_selection_cases.csv", index=False)
    rank = comparison[comparison.variant.eq("base")].set_index("strategy").copy()
    rank["all_execution_cases_pass"] = comparison.groupby("strategy").qualifies.all()
    rank["complexity"] = [0 if x.endswith("_fixed") else 2 if x.endswith("_trendvol") else 1 for x in rank.index]
    chosen, passed = select_candidate(rank)
    rank["selected"] = rank.index == chosen
    rank.to_csv(OUT/"selection_ranking.csv")
    returns = pd.DataFrame(net_returns)
    # Same tradability and execution basis for the simple three-asset comparator.
    benchmark = np.array([1/3, 0., 1/3, 0., 1/3, 0., 0., 0.])
    bt, _ = make_targets(panel, "simple", "fixed", budget_override=benchmark)
    br, _ = simulate(panel, bt)
    returns["three_asset_equal"] = br.returns.loc[mask]
    sims["three_asset_equal"] = br
    returns.to_csv(OUT/"all_daily_returns.csv", index_label="date")
    annual = (1+returns).groupby(returns.index.year).prod()-1
    annual.to_csv(OUT/"annual_returns.csv", index_label="year")
    full = pd.DataFrame({n: {**prior.metrics(r, sims[n].turnover.loc[r.index]),
                              "average_cash": float(1-sims[n].weights.loc[r.index].sum(axis=1).mean())} for n,r in returns.items()}).T
    full.to_csv(OUT/"full_comparison.csv", index_label="strategy")
    diagnostics = pd.DataFrame({n: prior.metrics(r.loc["2026-01-01":]) for n,r in returns.items()}).T
    diagnostics.to_csv(OUT/"2026_diagnostic.csv", index_label="strategy")
    result = sims[chosen]
    result.weights.loc[mask].to_csv(OUT/"selected_holdings.csv", index_label="date")
    executions[chosen].to_csv(OUT/"selected_trades.csv", index=False)
    audits[chosen].to_csv(OUT/"selected_signals.csv", index=False)
    allocation, mode = chosen.split("_")
    targets, current = make_targets(panel, allocation, mode, include_last=True)
    current.to_csv(OUT/"selected_signals_with_monitor.csv", index=False)
    last_month = max(t for t in prior.month_ends(panel.close.index) if panel.close.index[t] >= START)
    latest = len(panel.close)-1
    inventory = []
    for j, code in enumerate(CODES):
        inventory.append({"code": code, "name": LABELS[j], "base_budget": float(CONFIGS[allocation][j]),
            "latest_close": float(panel.close.iloc[-1,j]), "last_month_target": float(targets[last_month][j]),
            "last_month_signal": str(panel.close.index[last_month].date()), "monitor_target_not_trade": float(targets[latest][j]),
            "latest_eligible": bool(panel.eligible.iloc[-1,j]), "known_premium": float(panel.premium.iloc[-1,j]),
            "known_nav_date": str(panel.nav_dates.iloc[-1,j].date()),
            "median20_amount_CNY": float(panel.adv.iloc[-1,j]),
            "minimum_100_shares_CNY": float(100*panel.close.iloc[-1,j]),
            "eligible_fraction_since_2021": float(panel.eligible.loc[mask,code].mean())})
    pd.DataFrame(inventory).to_csv(OUT/"pool_and_current_targets.csv", index=False)
    # Predeclared nearby-parameter diagnostics: no re-selection based on these results.
    nearby = [("base", {}), ("ma80_mom42", {"ma":80,"momentum":42}), ("ma160_mom84", {"ma":160,"momentum":84}),
              ("vol6_cov40", {"vol_target":.06,"vol_window":40}), ("vol10_cov120", {"vol_target":.10,"vol_window":120})]
    sensitivity = []
    for label, kwargs in nearby:
        tt, _ = make_targets(panel, allocation, mode, **kwargs)
        rr, _ = simulate(panel, tt)
        sensitivity.append({"variation":label, **prior.metrics(rr.returns.loc[mask], rr.turnover.loc[mask])})
    for gold in (.15, .25):
        budget = CONFIGS[allocation].copy(); difference = gold-budget[4]
        budget[4] = gold; budget[2:4] -= difference/2
        tt, _ = make_targets(panel, allocation, mode, budget_override=budget)
        rr, _ = simulate(panel, tt)
        sensitivity.append({"variation":f"gold_{gold:.0%}_bonds_offset",**prior.metrics(rr.returns.loc[mask], rr.turnover.loc[mask])})
    pd.DataFrame(sensitivity).to_csv(OUT/"selected_neighborhood.csv", index=False)
    summary = {"selected": chosen, "passed_predeclared_gates": passed, "selection_end":"2025-12-31",
        "diagnostic_2026_not_independent_oos": True, "start": str(START.date()), "end":str(panel.close.index[-1].date()),
        "reference_capital": INITIAL, "base_budget":dict(zip(CODES,CONFIGS[allocation].tolist())),
        "allocation":allocation,"management":mode,"ma":120,"momentum":63,"covariance_window":60,"target_volatility":.08,
        "premium_limit":.005,"nav_announcement_lag":"strictly earlier than signal date",
        "liquidity_min_median20_CNY":20_000_000,"maximum_daily_participation":.01,
        "commission_rate":.0002,"minimum_commission_CNY":5,"slippage_rate":.0008,"lot_size":100,
        "last_month_signal_date":str(panel.close.index[last_month].date()),
        "last_month_target":dict(zip(CODES,targets[last_month].tolist())),
        "monitor_date":str(panel.close.index[-1].date()),"monitor_target_not_order":dict(zip(CODES,targets[latest].tolist())),
        "full_metrics":full.loc[chosen].to_dict(),"selection_metrics":rank.loc[chosen].to_dict(),
        "2026_diagnostic":diagnostics.loc[chosen].to_dict(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256":hashlib.sha256((OUT/"protocol.md").read_bytes()).hexdigest(),
        "no_trade_execution":True,"not_production_config":True}
    # JSON null, rather than non-standard NaN, for metrics without turnover data.
    summary["2026_diagnostic"]["annual_traded_notional"] = None
    (OUT/"selected_strategy.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    print("SELECTED",chosen,"passed",passed)
    print(full.round(4).to_string())
    print(pd.DataFrame(inventory).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
