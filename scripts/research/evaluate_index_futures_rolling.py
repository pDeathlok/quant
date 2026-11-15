"""Walk-forward policy selection in one continuous one-lot futures account."""
import argparse
from bisect import bisect_left, bisect_right
from functools import lru_cache

import numpy as np
import pandas as pd

from backtest_index_futures import ROOT, metrics, simulate
from evaluate_index_futures_directions import CAPITAL, cash_risk, directional_panel

POLICIES = {
    "carry": ("carry", .0005, 10, "both", None),
    "carry_vol": ("carry", .0005, 10, "both", cash_risk),
    "lr5": ("score_direction_5", .20, 10, "both", None),
    "lr5_long": ("score_direction_5", .20, 10, "long", None),
    "lr60": ("score_direction_60", .30, 20, "both", None),
    "lr60_carry": ("lr60_carry", .30, 20, "both", None),
    "trend_carry_vol": ("trend_carry", .25, 10, "both", cash_risk),
}


def main(half_life=0):
    out = ROOT / ("rolling_selection" if not half_life
                  else f"rolling_selection_half_life_{half_life}m")
    out.mkdir(exist_ok=True)
    panel = directional_panel()
    groups = [list(g.itertuples(index=False)) for _, g in panel.groupby("entry_date")]
    dates = [g[0].entry_date for g in groups]
    exits = [g[0].exit_date for g in groups]
    start = bisect_left(dates, "20230101")
    assert dates[start] >= "20230101"
    evaluations, annual, audit = [], [], []
    for fee, margin in ((.0003, .15), (.0006, .25)):
        def replay(rows, policy, selector=None):
            score, threshold, cadence, side, filter_position = POLICIES[policy]
            return simulate(rows, score, threshold, 1, side, cadence, fee,
                            capital=CAPITAL, margin_rate=margin, enforce_margin=True,
                            lot_cap=1, position_filter=filter_position, policy_selector=selector)

        @lru_cache(maxsize=None)
        def rankings(day, months):
            cutoff = (pd.Timestamp(day) - pd.DateOffset(months=months)).strftime("%Y%m%d")
            # Strictly earlier exits are known before the current opening auction.
            past = groups[bisect_left(dates, cutoff):bisect_left(exits, day)]
            result = {}
            for name in POLICIES:
                ledger = replay(past, name)
                stats = metrics(ledger, CAPITAL)
                if half_life:
                    equity = np.r_[CAPITAL, ledger.equity.to_numpy()]
                    returns = np.diff(equity) / equity[:-1]
                    age_days = (pd.Timestamp(day) - pd.to_datetime(ledger.exit_date)).dt.days
                    weights = np.exp2(-age_days.to_numpy() / (half_life * 365.25 / 12))
                    mean = np.average(returns, weights=weights)
                    volatility = np.sqrt(np.average((returns - mean) ** 2, weights=weights))
                    stats["total_return"] = mean
                    stats["sharpe"] = np.sqrt(244) * mean / volatility if volatility else 0
                result[name] = (stats, int(ledger.forced_exit.sum()))
            return past[-1][0].exit_date, result

        def save(ledger, name, months=0, objective="fixed"):
            assert ledger.gross_lots.le(1).all()
            assert np.allclose(ledger.equity, CAPITAL + ledger.pnl.cumsum(), atol=1e-6, rtol=0)
            tag = f"{name}_{months}_{objective}_{margin:.2f}"
            ledger.to_csv(out / f"{tag}.csv", index=False)
            fields = dict(policy=name, months=months, objective=objective, margin=margin, fee=fee,
                          half_life_months=half_life)
            evaluations.append(dict(**fields, start=ledger.entry_date.iloc[0],
                                    end=ledger.exit_date.iloc[-1], forced=int(ledger.forced_exit.sum()),
                                    **metrics(ledger, CAPITAL)))
            equity = CAPITAL
            for year, rows in ledger.groupby(ledger.exit_date.str[:4]):
                path = np.r_[equity, rows.equity]
                annual.append(dict(**fields, year=year, total_return=path[-1] / equity - 1,
                                   max_drawdown=(path / np.maximum.accumulate(path) - 1).min(),
                                   fees=rows.fees.sum(), forced=int(rows.forced_exit.sum())))
                equity = path[-1]

        for name in POLICIES:
            save(replay(groups[start:], name), name)
        for months in (24, 36):
            for interval in (3, 6, 12, "close"):
                for objective in ("return", "stable"):
                    current = None
                    previous = set()
                    previous_period = None

                    def select(day, positions):
                        nonlocal current, previous, previous_period
                        exposure = {(code[:2], np.sign(lots)) for code, (lots, _) in positions.items()}
                        closed = bool(previous - exposure)
                        previous = exposure
                        period = pd.Timestamp(day).year * 12 + pd.Timestamp(day).month - 1
                        bucket = period // interval if interval != "close" else None
                        due = closed if interval == "close" else bucket != previous_period
                        previous_period = bucket
                        if current is not None and not due:
                            return None
                        last, candidates = rankings(day, months)
                        eligible = {name: stats for name, (stats, forced) in candidates.items()
                                    if forced == 0 and stats["active_days"] >= 20}
                        key = "total_return" if objective == "return" else "sharpe"
                        winner = max(eligible, key=lambda name: eligible[name][key]) if eligible else "carry"
                        audit.append(dict(date=day, last_scored_exit=last, policy=interval,
                                          months=months, objective=objective, margin=margin,
                                          previous=current, selected=winner,
                                          changed=current is not None and winner != current))
                        if winner == current:
                            return None
                        current = winner
                        return POLICIES[winner]

                    save(replay(groups[start:], "carry", select), str(interval), months, objective)
    pd.DataFrame(evaluations).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "annual.csv", index=False)
    selection = pd.DataFrame(audit)
    assert selection.last_scored_exit.lt(selection.date).all()
    selection.to_csv(out / "selection.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-life", type=int, choices=(0, 6, 12), default=0)
    main(parser.parse_args().half_life)
