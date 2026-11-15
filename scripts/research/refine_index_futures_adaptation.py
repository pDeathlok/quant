"""Capital-aware quarterly policy selection with preregistered switch hurdles."""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from backtest_index_futures import ROOT, metrics, simulate
from compare_index_futures_adaptation import (
    OUT as PREVIOUS, POLICIES, groups_for, history_before, replay as old_replay,
    trade_episodes, choose_policy, stability,
)
from evaluate_index_futures_directions import CAPITAL, load_bars, margin_reserve
from evaluate_index_futures_trade_wins import annual_stats, trade_stats

OUT = ROOT / "adaptation_funded"
NEW_METHODS = ("refit_executable", "reselect_capital", "reselect_guarded")
ALL_METHODS = ("fixed", "refit", "reselect", *NEW_METHODS)
SORTINO_HURDLE = .25
EQUITY_HURDLE = .005
COST_MULTIPLE = 3.
QUARTER_SESSIONS = 61


def affordable(row, direction, equity, fee, margin):
    notional = row.entry_open * row.multiplier
    opening_cost = notional * fee
    return (equity >= notional*margin + opening_cost
            and margin_reserve(row, direction, equity-opening_cost, margin))


def entry_candidates(rows, policy, equity, fee, margin, funded=True):
    candidates = []
    for row in rows:
        side = "long" if row.long_60 > row.short_60 else "short"
        confidence = max(row.long_60, row.short_60)
        if pd.isna(row.etf_ret_1) or confidence <= .45 or abs(row.long_60-row.short_60) < .10:
            continue
        if policy == "confidence60" and confidence < .60:
            continue
        if policy == "aux55" and getattr(row, f"win_{side}") < .55:
            continue
        if funded and not affordable(row, 1 if side == "long" else -1, equity, fee, margin):
            continue
        candidates.append((confidence, row.symbol, side))
    return sorted(candidates, key=lambda x: (-x[0], x[1]))


def switch_cost(rows, positions, equity, fee, margin):
    closing = sum(abs(lots)*price*fee for lots, price in positions.values())
    possible = [r.entry_open*r.multiplier for r in rows
                if any(affordable(r, side, equity-closing, fee, margin) for side in (1, -1))]
    # Next opening's instrument is unknown; use the largest currently affordable
    # notional rather than a future quote. This is a decision estimate, not a fee charge.
    return closing + max(possible)*fee if possible else None


def decide(scores, current, rows, positions, equity, fee, margin, guarded):
    selected, reason = choose_policy(scores, current)
    detail = dict(proposed=selected, estimated_switch_cost=np.nan, historical_quarter_edge=np.nan,
                  required_edge=np.nan, sortino_improvement=np.nan)
    if selected == current:
        return current, reason, detail
    cost = switch_cost(rows, positions, equity, fee, margin)
    if cost is None:
        return current, "no_currently_affordable_contract", detail
    detail["estimated_switch_cost"] = cost
    if not guarded:
        return selected, reason, detail
    candidate, incumbent = scores[selected], scores[current]
    edge = QUARTER_SESSIONS*(candidate["weighted_pnl"]-incumbent["weighted_pnl"])
    required = max(EQUITY_HURDLE*equity, COST_MULTIPLE*cost)
    improvement = candidate["weighted_sortino"]-incumbent["weighted_sortino"]
    detail.update(historical_quarter_edge=edge, required_edge=required, sortino_improvement=improvement)
    if improvement < SORTINO_HURDLE:
        return current, "insufficient_sortino_advantage", detail
    if edge < required:
        return current, "insufficient_net_advantage", detail
    if incumbent["trades"] >= 6 and candidate["max_drawdown"] < incumbent["max_drawdown"]-1e-12:
        return current, "worse_drawdown_than_incumbent", detail
    return selected, "passed_capital_and_switch_hurdles", detail


def funded_replay(groups, fee, margin, capital=CAPITAL, policy="base", decision=None, funded=True):
    by_day = {rows[0].entry_date: rows for rows in groups}
    step, opened = -1, 0
    current, quarter = policy, None
    events = []

    def select(day, positions, equity):
        nonlocal step, opened, current, quarter
        step += 1
        rows = by_day[day]
        if rows[0].quarter != quarter:
            if quarter is not None and decision is not None:
                wanted = decision(day, rows[0].trade_date, current, equity, positions, rows)
                if wanted != current:
                    events.append(dict(date=day, old=current, new=wanted,
                                       closed_for_switch=bool(positions), equity=equity))
                    current = wanted
                    quarter = rows[0].quarter
                    if positions:
                        return ("flat", .5, 1, "both", margin_reserve)
            quarter = rows[0].quarter
        if positions:
            code, (lots, _) = next(iter(positions.items()))
            row = next(r for r in rows if r.symbol == code[:2])
            opposing = row.short_20 if lots > 0 else row.long_20
            if step-opened >= 20 or (current == "opposing_exit" and opposing >= .35):
                return ("flat", .5, 1, "both", margin_reserve)
            return None
        candidates = entry_candidates(rows, current, equity, fee, margin, funded) if step % 5 == 0 else []
        if not candidates:
            return ("flat", .5, 1, "both", margin_reserve)
        _, symbol, side = candidates[0]
        opened = step
        return (f"{side}_{symbol}", .5, 10000, "both", margin_reserve)

    ledger = simulate(groups, "flat", .5, 1, "both", 1, fee, margin_rate=margin,
                      capital=capital, enforce_margin=True, lot_cap=1, state_policy_selector=select)
    assert ledger.gross_lots.le(1).all()
    np.testing.assert_allclose(ledger.equity, capital+ledger.pnl.cumsum(), atol=1e-6, rtol=0)
    return ledger, events


def capital_quality(ledger, day, capital, count):
    path = np.r_[capital, ledger.equity]
    returns = np.diff(path)/path[:-1]
    age = (pd.Timestamp(day)-pd.to_datetime(ledger.exit_date)).dt.days.to_numpy()
    weights = np.exp2(-age/365.25)
    mean = np.average(returns, weights=weights)
    downside = np.sqrt(np.average(np.minimum(returns, 0)**2, weights=weights))
    return dict(weighted_mean=mean, weighted_sortino=np.sqrt(244)*mean/max(downside, 1e-6),
                weighted_pnl=np.average(ledger.pnl, weights=weights),
                max_drawdown=(path/np.maximum.accumulate(path)-1).min(),
                total_return=path[-1]/capital-1, trades=count, days=len(ledger),
                active_days=int(ledger.gross_lots.gt(0).sum()), forced=int(ledger.forced_exit.sum()))


def cumulative_years(ledger):
    """Prefix of ONE continuously executed account, not independent yearly reruns."""
    for year in sorted(ledger.exit_date.str[:4].unique()):
        prefix = ledger[ledger.exit_date.str[:4].le(year)].reset_index(drop=True)
        path = np.r_[CAPITAL, prefix.equity]
        yield dict(year=year, asof=prefix.exit_date.iloc[-1], equity=path[-1],
                   cumulative_return=path[-1]/CAPITAL-1,
                   cumulative_drawdown=(path/np.maximum.accumulate(path)-1).min(),
                   cumulative_fees=prefix.fees.sum(), **stability(prefix))


def main():
    OUT.mkdir(exist_ok=True)
    protocol = dict(capital=CAPITAL, max_lots=1, starts=["20220101", "20230101", "20240101", "20250101", "20260101"],
                    methods=ALL_METHODS, fee=[.0003, .0006], margin=[.15, .25],
                    models="unchanged cached quarterly predictions from adaptation_comparison",
                    funded_selection="historical candidate accounts start at actual current equity; select among currently affordable entries",
                    guarded_sortino_hurdle=SORTINO_HURDLE, guarded_equity_hurdle=EQUITY_HURDLE,
                    guarded_cost_multiple=COST_MULTIPLE, quarter_sessions=QUARTER_SESSIONS,
                    guarded_additional_rule="historical drawdown no worse than incumbent when incumbent has >=6 trades",
                    risk_note="Thresholds are preregistered research settings, not user's accepted risk budget; 61-day historical edge is not guaranteed future return",
                    no_new_model_selection=True, historical_lookback_months=36,
                    inherited_state="no yearly/quarterly resets within each start; every independent start uses 500k",
                    limits="daily execution; quarter equity measured at that opening; same-opening execution approximation; no intraday limit-lock modeling")
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    panel = pd.read_parquet(PREVIOUS / "quarterly_predictions.parquet")
    opens = load_bars().open.to_dict()
    old_scores = pd.read_csv(PREVIOUS / "validation_audit.csv", dtype={"date": str})
    old_rank = {(day, fee, margin): g.set_index("policy").to_dict("index")
                for (day, fee, margin), g in old_scores.groupby(["date", "fee", "margin"])}
    audits, choices, summaries, annual, prefixes, all_trades, switch_events = [], [], [], [], [], [], []

    @lru_cache(maxsize=None)
    def history(cutoff):
        past = history_before(panel, cutoff)
        assert past.exit_date.max() <= cutoff
        return groups_for(past)

    @lru_cache(maxsize=512)
    def rankings(day, cutoff, equity, fee, margin):
        historical_groups = history(cutoff)
        scores = {}
        for policy in POLICIES:
            ledger, _ = funded_replay(historical_groups, fee, margin, capital=equity, policy=policy)
            trades = trade_episodes(ledger, opens, fee)
            scores[policy] = capital_quality(ledger, day, equity, len(trades))
            audits.append(dict(date=day, cutoff=cutoff, validation_initial_equity=equity, fee=fee, margin=margin,
                               policy=policy, last_validation_exit=ledger.exit_date.iloc[-1], **scores[policy]))
        assert historical_groups[-1][0].exit_date <= cutoff < day
        return scores

    for start in protocol["starts"]:
        sample = panel[panel.entry_date.ge(start)]
        groups = groups_for(sample)
        frozen = pd.read_parquet(PREVIOUS / f"frozen_{start[:4]}.parquet")
        fixed_groups = groups_for(frozen[frozen.entry_date.ge(start)])
        periods = list(sample.groupby("quarter", sort=True))
        for fee in (.0003, .0006):
            for margin in (.15, .25):
                current, old_schedule = "base", {}
                for i, (_, part) in enumerate(periods):
                    day = part.entry_date.min()
                    if i:
                        current, _ = choose_policy(old_rank[day, fee, margin], current)
                    old_schedule[day] = current
                for method in ALL_METHODS:
                    fields = dict(start=start, fee=fee, margin=margin, method=method)
                    if method in ("fixed", "refit", "reselect"):
                        ledger, events = old_replay(fixed_groups if method == "fixed" else groups,
                                                   fee, margin, schedule=old_schedule if method == "reselect" else None)
                    else:
                        def decision(day, cutoff, incumbent, equity, positions, rows):
                            scores = rankings(day, cutoff, equity, fee, margin)
                            selected, reason, detail = decide(scores, incumbent, rows, positions, equity,
                                                               fee, margin, method == "reselect_guarded")
                            choices.append(dict(**fields, date=day, cutoff=cutoff, current_equity=equity,
                                                incumbent=incumbent, selected=selected, changed=selected != incumbent,
                                                reason=reason, **detail))
                            return selected
                        ledger, events = funded_replay(groups, fee, margin,
                                                       decision=None if method == "refit_executable" else decision)
                    trades = trade_episodes(ledger, opens, fee)
                    summaries.append(dict(**fields, **metrics(ledger, CAPITAL), **trade_stats(trades), **stability(ledger),
                                          changes=len(events), switch_closures=sum(e["closed_for_switch"] for e in events)))
                    annual.extend(dict(**fields, **row) for row in annual_stats(ledger, trades))
                    prefixes.extend(dict(**fields, **row) for row in cumulative_years(ledger))
                    all_trades.append(trades.assign(**fields))
                    switch_events.extend(dict(**fields, **event) for event in events)
                    ledger.to_csv(OUT / f"ledger_{start}_{method}_{fee}_{margin}.csv", index=False)
                print("evaluated", start, fee, margin, flush=True)
        pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
        pd.DataFrame(audits).to_csv(OUT / "capital_validation_audit.csv", index=False)
        pd.DataFrame(choices).to_csv(OUT / "selection.csv", index=False)
    pd.DataFrame(annual).to_csv(OUT / "annual.csv", index=False)
    pd.DataFrame(prefixes).to_csv(OUT / "cumulative_by_year.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    pd.DataFrame(switch_events).to_csv(OUT / "switches.csv", index=False)


if __name__ == "__main__":
    main()
