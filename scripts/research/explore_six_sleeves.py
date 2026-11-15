"""Causal, single-ledger ETF proxy experiment; not a microcap/factor backtest."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/research/six_sleeve_exploration_20261111"
CODES = ("518880.SH", "513500.SH", "159920.SZ", "511380.SH", "510880.SH",
         "512100.SH", "510300.SH", "510500.SH", "159915.SZ", "511010.SH")
SLEEVES = ("gold", "cross_border", "convertible", "dividend", "small_cap_proxy", "broad_rotation")
START = pd.Timestamp("2021-06-01")


@dataclass
class Result:
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    skipped: list[str]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    frames = {}
    for code in CODES:
        f = pd.read_parquet(OUT / "source" / f"{code.replace('.', '_')}.parquet")
        f.index = pd.to_datetime(f.trade_date)
        if f.index.has_duplicates:
            raise ValueError("Duplicate observations")
        frames[code] = f.sort_index()
    dates = frames["510300.SH"].index
    prices, valid, audit = {}, {}, []
    for code, f in frames.items():
        f = f.reindex(dates)
        raw = f.close * f.adj_factor
        prices[code] = raw.ffill()
        valid[code] = raw.notna() & f.amount.gt(0)
        missing = dates[(dates >= raw.first_valid_index()) & raw.isna()]
        r = prices[code].pct_change(fill_method=None)
        if r.abs().max() > .35:
            raise ValueError(f"Inspect adjustment outlier: {code}")
        audit.append({"code": code, "missing_quote_dates": missing.strftime("%Y-%m-%d").tolist(),
                      "largest_abs_daily_return": float(r.abs().max())})
    return pd.DataFrame(prices), pd.DataFrame(valid), audit


def month_ends(dates: pd.DatetimeIndex) -> list[int]:
    return [i for i in range(len(dates)-1) if dates[i].month != dates[i+1].month]


def simulate(prices: pd.DataFrame, valid: pd.DataFrame, targets: dict[int, np.ndarray],
             cost: float = .001, lag: int = 1, start: int = 0) -> Result:
    """Signal i -> execute i+lag close; holdings earn return before close trades.

    Trades solve the self-financing post-fee NAV equation; no implicit daily rebalancing.
    Missing prices are marked at last known price, and any affected rebalance is skipped.
    """
    r = prices.pct_change(fill_method=None).fillna(0).to_numpy()
    can_trade = valid.to_numpy()
    n, k = r.shape
    w = np.zeros(k)
    daily, weights, turns, fees = np.zeros(n), np.zeros((n, k)), np.zeros(n), np.zeros(n)
    skipped = []
    for t in range(start, n):
        gross = float(w @ r[t])
        pre = w * (1 + r[t]) / (1 + gross)
        target = targets.get(t-lag)
        alpha = 1.
        if target is not None:
            target = np.asarray(target)
            if (target < -1e-12).any() or target.sum() > 1 + 1e-10:
                raise ValueError("Invalid long-only target")
            affected = (target > 1e-12) | (pre > 1e-12)
            if not can_trade[t, affected].all():
                skipped.append(str(prices.index[t].date()))
                w = pre
            else:
                # Contractive fixed point: alpha = 1 - c * sum(|alpha*target-pre|).
                for _ in range(30):
                    updated = 1 - cost * np.abs(alpha*target-pre).sum()
                    if abs(updated-alpha) < 1e-14:
                        alpha = updated
                        break
                    alpha = updated
                turns[t] = np.abs(alpha*target-pre).sum()
                fees[t] = cost*turns[t]*(1+gross)
                w = target.copy()
        else:
            w = pre
        daily[t] = (1+gross)*alpha-1
        weights[t] = w
        if w.sum() > 1 + 1e-10 or not np.isfinite(daily[t]):
            raise ValueError("Ledger invariant failed")
    idx = prices.index
    return Result(pd.Series(daily, index=idx), pd.DataFrame(weights, index=idx, columns=prices.columns),
                  pd.Series(turns, index=idx), pd.Series(fees, index=idx), skipped)


def sleeve_targets(prices: pd.DataFrame, momentum: int = 63) -> dict[int, np.ndarray]:
    mom = prices.pct_change(momentum, fill_method=None)
    ma = prices.rolling(120, min_periods=120).mean()
    ix = {c: j for j, c in enumerate(CODES)}
    output = {}
    for t in month_ends(prices.index):
        mat = np.zeros((6, len(CODES)))
        for sleeve, candidates in [(0, ["518880.SH"]), (1, ["513500.SH", "159920.SZ"]),
                                  (5, ["510300.SH", "510500.SH", "159915.SZ"])]:
            eligible = [c for c in candidates if mom.iloc[t][c] > 0 and prices.iloc[t][c] > ma.iloc[t][c]]
            if eligible:
                winner = max(eligible, key=lambda c: mom.iloc[t][c])
                mat[sleeve, ix[winner]] = 1
        for sleeve, code in [(2, "511380.SH"), (3, "510880.SH"), (4, "512100.SH")]:
            if pd.notna(prices.iloc[t][code]):
                mat[sleeve, ix[code]] = 1
        output[t] = mat
    return output


def covariance(history: np.ndarray) -> np.ndarray:
    cov = np.cov(history, rowvar=False, ddof=1)
    cov = .75*cov + .25*np.diag(np.diag(cov))
    cov += np.diag(np.maximum(.05**2/252-np.diag(cov), 0))
    return cov


def erc(cov: np.ndarray) -> np.ndarray:
    """Solve the convex log-barrier risk-budget problem by coordinate descent."""
    scaled = cov / np.mean(np.diag(cov))
    n = len(cov)
    x = np.ones(n)/np.sqrt(n)
    for _ in range(10000):
        for j in range(n):
            a = scaled[j, j]
            off = scaled[j]@x - a*x[j]
            root = np.sqrt(off*off+4*a/n)
            x[j] = (2/n)/(root+off) if off >= 0 else (root-off)/(2*a)
        w = x/x.sum()
        contributions = w*(cov@w)/(w@cov@w)
        if np.max(abs(contributions - 1/n)) < 1e-9:
            break
    contributions = w*(cov@w)/(w@cov@w)
    if np.max(abs(contributions - 1/n)) > 1e-4:
        raise ValueError("ERC convergence failed")
    return w


def metrics(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    values = r.to_numpy()
    wealth = np.r_[1., np.cumprod(1+values)]
    dd = wealth / np.maximum.accumulate(wealth)-1
    years = (r.index[-1]-r.index[0]).days/365.25 + 1/252
    vol = float(r.std(ddof=1)*np.sqrt(252))
    return {"cagr": float(wealth[-1]**(1/years)-1), "total_return": float(wealth[-1]-1),
            "volatility": vol, "max_drawdown": float(dd.min()),
            "sharpe_rf0": float(r.mean()*252/vol) if vol > 0 else 0,
            "worst_day": float(r.min()),
            "annual_traded_notional": float(turnover.sum()/years) if turnover is not None else None}


def experiment(prices: pd.DataFrame, valid: pd.DataFrame, *, momentum: int = 63,
               lookback: int = 120, cost: float = .001, lag: int = 1,
               removed_sleeve: int | None = None) -> tuple[dict[str, Result], pd.DataFrame, pd.DataFrame]:
    st = sleeve_targets(prices, momentum)
    standalone = [simulate(prices, valid, {t: mat[j] for t, mat in st.items()}, cost, lag)
                  for j in range(6)]
    sr = pd.concat([x.returns for x in standalone], axis=1)
    sr.columns = SLEEVES
    names = ("six_equal", "six_erc", "six_erc_guarded", "four_equal", "three_asset_equal", "buy_hold_300")
    targets = {name: {} for name in names}
    records = []
    start = int(prices.index.searchsorted(START))
    for t, mat in st.items():
        if t < start-lag or t+lag < start:
            continue
        hist = sr.iloc[t-lookback+1:t+1].to_numpy()
        if len(hist) != lookback:
            raise ValueError("Insufficient covariance history")
        cov = covariance(hist)
        risk_w = erc(cov)
        alloc = {"six_equal": np.ones(6)/6, "six_erc": risk_w,
                 "six_erc_guarded": np.minimum(risk_w, [.3, .3, .3, .3, .1, .3]),
                 "four_equal": np.array([.25, .25, 0., .25, 0., .25])}
        guarded = alloc["six_erc_guarded"]
        equity = float(guarded[1:] @ mat[1:].sum(axis=1))
        if equity > .6:
            guarded[1:] *= .6/equity
        predicted_vol = float(np.sqrt(guarded@cov@guarded*252))
        if predicted_vol > .08:
            guarded *= .08/predicted_vol
        for name, budget in alloc.items():
            use = budget.copy()
            if removed_sleeve is not None:
                use[removed_sleeve] = 0  # Keep removed allocation in cash; do not refit.
            targets[name][t] = use@mat
            records.append({"date": str(prices.index[t].date()), "portfolio": name,
                            **dict(zip(SLEEVES, use.tolist())),
                            "actual_cash_target": float(1-(use@mat).sum()),
                            "model_volatility": float(np.sqrt(use@cov@use*252))})
        static = np.zeros(len(CODES))
        static[[0, 6, 9]] = 1/3
        targets["three_asset_equal"][t] = static
    initial_signal = start-lag
    # For lag=2, reuse last completed monthly signal at the evaluation boundary.
    for name in names[:-1]:
        if initial_signal not in targets[name]:
            prior = max(i for i in st if i <= initial_signal)
            # The common period starts June 1; delayed versions retain initial cash
            # until their first regular scheduled execution, instead of borrowing a signal.
            if prior+lag >= start and prior not in targets[name]:
                raise ValueError("Missing initial scheduled decision")
    hold = np.zeros(len(CODES)); hold[6] = 1
    targets["buy_hold_300"][initial_signal] = hold
    results = {name: simulate(prices, valid, target, cost, lag, start) for name, target in targets.items()}
    return results, sr, pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices, valid, audit = load_data()
    results, sleeves, allocations = experiment(prices, valid)
    mask = prices.index >= START
    returns = pd.DataFrame({name: result.returns[mask] for name, result in results.items()})
    returns.to_csv(OUT / "daily_returns.csv", index_label="date")
    sleeves.loc[mask].to_csv(OUT / "sleeve_returns.csv", index_label="date")
    allocations.to_csv(OUT / "allocation_signals.csv", index=False)
    table = pd.DataFrame({name: metrics(result.returns[mask], result.turnover[mask])
                          for name, result in results.items()}).T
    table["average_cash"] = [float(1-results[n].weights.loc[mask].sum(axis=1).mean()) for n in table.index]
    table.to_csv(OUT / "comparison.csv", index_label="portfolio")
    for name, result in results.items():
        result.weights.loc[mask].to_csv(OUT / f"{name}_holdings.csv", index_label="date")
        pd.DataFrame({"return": result.returns[mask], "traded_notional": result.turnover[mask],
                      "cost_fraction_start_nav": result.costs[mask]}).to_csv(OUT / f"{name}_ledger.csv")
    ((1+returns).groupby(returns.index.year).prod()-1).to_csv(OUT / "annual_returns.csv")
    monthly = (1+returns).groupby(returns.index.to_period("M")).prod()-1
    monthly.to_csv(OUT / "monthly_returns.csv")
    sleeves.loc[mask].corr().to_csv(OUT / "sleeve_correlation.csv")
    sections = []
    for label, lo, hi in [("202106_2023", "2021-06-01", "2023-12-31"),
                           ("2024_onward", "2024-01-01", "2026-11-10")]:
        for name in returns:
            sections.append({"segment": label, "portfolio": name, **metrics(returns.loc[lo:hi, name])})
    pd.DataFrame(sections).to_csv(OUT / "segments.csv", index=False)
    stress = []
    for label, lo, hi in [("2022", "2022-01-01", "2022-12-31"),
                          ("2024_early", "2024-01-01", "2024-02-08"),
                          ("2024_reversal", "2024-09-01", "2024-10-31"),
                          ("2025_april", "2025-04-01", "2025-04-30")]:
        for name in returns:
            stress.append({"window": label, "portfolio": name, **metrics(returns.loc[lo:hi, name])})
    pd.DataFrame(stress).to_csv(OUT / "historical_stress.csv", index=False)
    variations = [("base", {}), ("mom20", {"momentum": 20}), ("mom120", {"momentum": 120}),
                  ("cov60", {"lookback": 60}), ("cov252", {"lookback": 252}),
                  ("cost5bp", {"cost": .0005}), ("cost20bp", {"cost": .002}),
                  ("delay2", {"lag": 2})]
    sensitivity = []
    for label, parameters in variations:
        rr = results if label == "base" else experiment(prices, valid, **parameters)[0]
        for name in ("six_equal", "six_erc", "six_erc_guarded", "four_equal"):
            sensitivity.append({"variant": label, "portfolio": name,
                                **metrics(rr[name].returns[mask], rr[name].turnover[mask])})
        print(f"Completed {label}", flush=True)
    pd.DataFrame(sensitivity).to_csv(OUT / "sensitivity.csv", index=False)
    ablation = []
    for j, sleeve in enumerate(SLEEVES):
        rr = experiment(prices, valid, removed_sleeve=j)[0]
        for name in ("six_equal", "six_erc", "six_erc_guarded"):
            ablation.append({"removed_to_cash": sleeve, "portfolio": name,
                             **metrics(rr[name].returns[mask])})
    pd.DataFrame(ablation).to_csv(OUT / "ablations.csv", index=False)
    # Scenarios are assumed shocks on actual pre-shock ETF exposures, not forecasts.
    scenarios = {"equity_joint_selloff": [-.10, -.25, -.25, -.15, -.20, -.35, -.25, -.30, -.35, .02],
                 "small_cap_liquidity_shock": [0., -.05, -.05, -.10, -.08, -.45, -.10, -.20, -.25, 0.]}
    synthetic = []
    for name, result in results.items():
        for scenario, changes in scenarios.items():
            losses = result.weights.loc[mask].to_numpy() @ np.asarray(changes)
            synthetic.append({"portfolio": name, "assumed_scenario": scenario,
                              "loss_at_average_exposure": float(losses.mean()),
                              "worst_historical_exposure_loss": float(losses.min()),
                              "loss_at_last_exposure": float(losses[-1])})
    pd.DataFrame(synthetic).to_csv(OUT / "hypothetical_stress.csv", index=False)
    rolling = []
    for i, left in enumerate(SLEEVES):
        for right in SLEEVES[i+1:]:
            corr = sleeves[left].rolling(120).corr(sleeves[right]).loc[mask].dropna()
            rolling.append({"left": left, "right": right, "window": 120,
                            "minimum": float(corr.min()), "median": float(corr.median()),
                            "maximum": float(corr.max())})
    pd.DataFrame(rolling).to_csv(OUT / "rolling_correlation_summary.csv", index=False)
    worst_months = []
    for name in monthly:
        for date, value in monthly[name].nsmallest(5).items():
            worst_months.append({"portfolio": name, "month": str(date), "return": float(value)})
    pd.DataFrame(worst_months).to_csv(OUT / "worst_months.csv", index=False)
    summary = {"start": str(returns.index[0].date()), "end": str(returns.index[-1].date()),
               "observations": len(returns), "one_way_cost": .001,
               "proxy_only": True, "independent_out_of_sample": False,
               "missing_quote_audit": audit, "skipped_rebalances": {n: r.skipped for n, r in results.items()},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "protocol_sha256": hashlib.sha256((OUT / "protocol.md").read_bytes()).hexdigest(),
               "source_manifest_sha256": hashlib.sha256((OUT / "source/manifest.json").read_bytes()).hexdigest(),
               "versions": {name: importlib.metadata.version(name) for name in ("pandas", "numpy", "pyarrow")},
               "stress_shocks": dict(zip(scenarios, [dict(zip(CODES, x)) for x in scenarios.values()]))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(table.to_string())


if __name__ == "__main__":
    main()
