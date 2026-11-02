"""Reconcile dollar P&L by asset for the ETF stability candidate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_strategy_lab import OUT, build_core_signals, core_prices, metrics, simulate_targets


NAME = "invvol60_nasdaq_trend"
CODES = ("513100.SH", "518880.SH", "511010.SH")


def main() -> None:
    prices = core_prices()
    targets = build_core_signals(prices)[NAME]
    dates = prices.index
    returns = prices.pct_change(fill_method=None).fillna(0)
    executions = {dates[dates.get_loc(day) + 1]: target for day, target in targets.items()
                  if dates.get_loc(day) + 1 < len(dates)}
    weights: dict[str, float] = {}
    wealth = 1.0
    rows = []
    for day in dates:
        before = wealth
        contributions = {code: before * weights.get(code, 0) * returns.at[day, code]
                         for code in CODES}
        start_weights = {f"{code}_weight": weights.get(code, 0) for code in CODES}
        gross = sum(contributions.values()) / before
        wealth *= 1 + gross
        weights = {code: weight * (1 + returns.at[day, code]) / (1 + gross)
                   for code, weight in weights.items()}
        cost = 0.0
        if day in executions:
            desired = executions[day]
            turnover = sum(abs(weights.get(code, 0) - desired.get(code, 0))
                           for code in set(weights) | set(desired))
            cost = wealth * 0.001 * turnover
            wealth -= cost
            weights = desired.copy()
        rows.append({"date": day, "before": before, "equity": wealth,
                     **contributions, "cost": -cost, **start_weights})
    daily = pd.DataFrame(rows).set_index("date").loc["2019-02-01":"2026-09-24"]
    reference = simulate_targets(prices, targets).daily.reindex(daily.index)
    assert np.allclose(daily.equity, reference.equity, rtol=1e-12, atol=1e-12)
    assert abs(daily[list(CODES) + ["cost"]].to_numpy().sum()
               - (daily.equity.iloc[-1] - daily.before.iloc[0])) < 1e-10
    annual = []
    for year, frame in daily.groupby(daily.index.year):
        start = frame.before.iloc[0]
        annual.append({"year": year, "trade_days": len(frame),
                       **(frame[list(CODES) + ["cost"]].sum() / start).to_dict(),
                       "net_return": frame.equity.iloc[-1] / start - 1})
    total = daily[list(CODES) + ["cost"]].sum() / daily.before.iloc[0]
    summary = []
    for code in (*CODES, "cost"):
        summary.append({"asset": code, "contribution_to_initial_capital": total[code],
                        "share_of_net_profit": total[code] / total.sum(),
                        "average_start_weight": daily[f"{code}_weight"].mean()
                        if code != "cost" else np.nan})
    ablations = []
    for removed in (None, *CODES):
        candidate = targets if removed is None else {
            day: {code: value for code, value in target.items() if code != removed}
            for day, target in targets.items()}
        run = simulate_targets(prices, candidate)
        sample = run.daily.loc[daily.index.min():daily.index.max()]
        ablations.append({"scenario": "base" if removed is None else f"{removed}_to_cash",
                          **metrics(sample.daily_return, sample.exposure)})
    daily.to_csv(OUT / "stable_asset_attribution_daily.csv", index_label="date")
    pd.DataFrame(annual).to_csv(OUT / "stable_asset_attribution_yearly.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "stable_asset_attribution_summary.csv", index=False)
    pd.DataFrame(ablations).to_csv(OUT / "stable_asset_to_cash_ablations.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(annual).to_string(index=False))
    print(pd.DataFrame(ablations)[["scenario", "annual_return", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
