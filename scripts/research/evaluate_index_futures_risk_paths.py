"""Symmetric, fixed risk overlays on the capital-aware quarterly selector.

These are exploratory historical replays, not untouched OOT. Parameters are
written to protocol.json before execution; no results-based parameter search.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from backtest_index_futures import ROOT, metrics, simulate
from compare_index_futures_adaptation import (
    OUT as PREDICTIONS, POLICIES, groups_for, history_before, stability, trade_episodes,
)
from evaluate_index_futures_directions import CAPITAL, load_bars, margin_reserve
from evaluate_index_futures_trade_wins import annual_stats, trade_stats
from refine_index_futures_adaptation import (
    OUT as BASELINE, affordable, capital_quality, decide, entry_candidates,
)
from index_futures_reentry import ReentryGate
from index_futures_regime import MODES as REGIME_MODES, entry_allowed, exit_required
from index_futures_state_rules import evidence_admits, holding_limit, profit_giveback

OUT = ROOT / "adaptation_risk_paths"
VARIANTS = ("control", "entry_risk", "loss_exit", "signal_exit", "combined")
RISK_FRACTION = .05


def minimum_equity(row, direction, fee, margin):
    notional = row.entry_open * row.multiplier
    adverse = max(.05, 3 * row.f_volatility_20)
    return notional * (fee + max(margin, (adverse + margin*(1-direction*adverse))/.8))


def risk_admits(row, equity):
    return (np.isfinite(row.f_volatility_20) and row.f_volatility_20 >= 0
            and 2 * row.f_volatility_20 * row.entry_open * row.multiplier <= RISK_FRACTION * equity)


def previous_close_pnl(equity, entry_equity, position, close, multiplier, fee):
    """Remove the known overnight mark; include a hypothetical close commission.

The account holds at most one contract, so its equity difference captures all
realized roll PnL and paid fees. Current open cancels algebraically: the stop
depends on yesterday's close, not today's/next day's price movement.
    """
    lots, marked_notional = position
    return equity-entry_equity + lots*(close*multiplier-marked_notional) - abs(lots)*close*multiplier*fee


def risk_replay(groups, opens, closes, fee, margin, capital=CAPITAL,
                policy="base", decision=None, variant="control", trace=False, exit_decider=None,
                state_exit_decider=None, entry_calendar=None, risk_overlay="none", principal_floor=0.,
                reentry_rule="none", regime_mode="none", market_regime=None, continuation_lookup=None,
                evidence_model=0, evidence_lookup=None, soft_exit=0, soft_lookup=None,
                evidence_names=("base","factor","interaction")):
    if regime_mode not in REGIME_MODES:
        raise ValueError(regime_mode)
    if exit_decider is not None and state_exit_decider is not None:
        raise ValueError("Choose one exit callback")
    by_day = {rows[0].entry_date: rows for rows in groups}
    step, opened, confirmations = -1, 0, 0
    current, quarter, entry_equity, opened_day = policy, None, capital, None
    halted = False
    profit_peak, soft_streak, entry_limit = 0., 0, 20
    gate = ReentryGate(reentry_rule)
    if risk_overlay not in ("none", "entry5", "stop5", "fuse20", "combined"):
        raise ValueError(risk_overlay)
    events, diagnostics = [], []
    flat = ("flat", .5, 1, "both", margin_reserve)

    def select(day, positions, equity):
        nonlocal step, opened, current, quarter, entry_equity, confirmations, opened_day, halted
        nonlocal profit_peak, soft_streak, entry_limit
        step += 1
        rows = by_day[day]
        gate.observe(day, step, positions, rows)
        record = dict(date=day, signal_date=rows[0].trade_date, equity=equity, policy=current,
                      held=next(iter(positions), ""), action="hold", raw_candidates=0,
                      affordable_candidates=0, risk_candidates=0, close_pnl=np.nan,
                      minimum_signal_equity=np.nan, minimum_any_equity=min(
                          minimum_equity(r, s, fee, margin) for r in rows for s in (1, -1)) if trace else np.nan)
        def finish(action, value):
            record["action"] = action
            record["policy"] = current
            if trace:
                diagnostics.append(record)
            return value
        # Principal protection uses yesterday's liquidatable close, not future
        # prices. A gap can overshoot it. The halt never resets at a quarter/year.
        close_liquidation = equity
        if positions:
            code, (lots, mark) = next(iter(positions.items()))
            row = next(r for r in rows if r.symbol == code[:2])
            closing_notional = closes[row.trade_date, code] * row.multiplier
            close_liquidation += lots * (closing_notional-mark) - abs(lots)*closing_notional*fee
        if risk_overlay in ("fuse20", "combined"):
            halted = halted or close_liquidation <= principal_floor
            record.update(principal_floor=principal_floor, close_liquidation=close_liquidation, halted=halted)
            if halted:
                return finish("principal_halt", flat)
        if rows[0].quarter != quarter:
            if quarter is not None and decision is not None:
                wanted = decision(day, rows[0].trade_date, current, equity, positions, rows)
                if wanted != current:
                    events.append(dict(date=day, old=current, new=wanted,
                                       closed_for_switch=bool(positions), equity=equity))
                    current = wanted
                    quarter = rows[0].quarter
                    if positions:
                        return finish("strategy_switch", flat)
            quarter = rows[0].quarter
        if current == "cash":
            if trace:
                due = step % 5 == 0 if entry_calendar is None else day in entry_calendar
                record.update(cash_mode=True, cash_signal_candidates=len(entry_candidates(rows, "base", equity, fee, margin, funded=False)) if due else 0,
                    cash_funded_candidates=len(entry_candidates(rows, "base", equity, fee, margin)) if due else 0)
            return finish("cash_wait", flat)
        if positions:
            code, position = next(iter(positions.items()))
            lots = position[0]
            row = next(r for r in rows if r.symbol == code[:2])
            opposing = row.short_20 if lots > 0 else row.long_20
            own = row.long_20 if lots > 0 else row.short_20
            net_close = previous_close_pnl(equity, entry_equity, position,
                                           closes[row.trade_date, code], row.multiplier, fee)
            record["close_pnl"] = net_close
            profit_peak = max(profit_peak, net_close)
            if risk_overlay in ("stop5", "combined") and net_close <= -RISK_FRACTION*entry_equity:
                return finish("budget_loss_exit", flat)
            confirmations = confirmations+1 if opposing >= .45 and opposing-own >= .10 else 0
            if step-opened >= 20:
                return finish("holding_limit", flat)
            if current == "opposing_exit" and opposing >= .35:
                return finish("native_opposing_exit", flat)
            if soft_exit:
                state=market_regime[row.trade_date,row.symbol]
                record.update(regime_state=state,profit_peak=profit_peak,entry_holding_limit=entry_limit)
                if soft_exit in (3,4,7,8) and step-opened>=entry_limit:
                    return finish("state_time_exit",flat)
                if soft_exit in (5,6,7) and profit_giveback(state,lots,profit_peak,net_close,entry_equity,soft_exit!=5):
                    return finish("profit_giveback_exit",flat)
                if soft_exit in (1,2,7):
                    model="base" if soft_exit==1 else "interaction"
                    forecast=soft_lookup[opened_day,day,row.symbol,model]
                    assert forecast.direction==lots
                    bps=round(fee*10000)
                    utility=getattr(forecast,f"pred_net_5_{bps}")-.5*getattr(forecast,f"pred_risk_5_{bps}")
                    soft_streak=soft_streak+1 if np.isfinite(utility) and utility<=0 else 0
                    record.update(soft_utility=utility,soft_confirmations=soft_streak)
                    if soft_streak>=2:
                        return finish("state_model_exit",flat)
            if regime_mode in ("exit", "both"):
                state = market_regime[row.trade_date, row.symbol]
                forecast = continuation_lookup[opened_day, day, row.symbol]
                assert forecast.direction == lots
                net = getattr(forecast, f"pred_net_5_{round(fee*10000)}")
                record.update(regime_state=state, regime_continuation=net)
                if exit_required(state, lots, net):
                    return finish("regime_exit", flat)
            if exit_decider is not None:
                should_exit, detail = exit_decider(day, opened_day, row, lots)
                record.update(detail)
                if should_exit:
                    return finish(detail.get("exit_action", "learned_exit"), flat)
            if state_exit_decider is not None:
                close_notional = closes[row.trade_date, code]*row.multiplier
                close_equity = equity+lots*(close_notional-position[1])
                should_exit, detail = state_exit_decider(day, opened_day, row, lots, close_equity)
                record.update(detail)
                if should_exit:
                    return finish(detail.get("exit_action", "calibrated_exit"), flat)
            if variant in ("loss_exit", "combined") and net_close <= -RISK_FRACTION*entry_equity:
                return finish("close_loss_exit", flat)
            if variant in ("signal_exit", "combined") and confirmations >= 2:
                return finish("confirmed_opposing_exit", flat)
            if not margin_reserve(row, lots, equity, margin):
                return finish("reserve_exit", None)
            return finish("hold", None)
        confirmations = 0
        entry_due = step % 5 == 0 if entry_calendar is None else day in entry_calendar
        if not entry_due:
            return finish("entry_schedule_wait", flat)
        raw = entry_candidates(rows, current, equity, fee, margin, funded=False)
        funded = entry_candidates(rows, current, equity, fee, margin)
        lookup = {r.symbol: r for r in rows}
        selected = [c for c in funded if (variant not in ("entry_risk", "combined") and risk_overlay not in ("entry5", "combined"))
                    or risk_admits(lookup[c[1]], equity)]
        record.update(raw_candidates=len(raw), affordable_candidates=len(funded), risk_candidates=len(selected))
        admitted = [c for c in selected if gate.admits(c[1], 1 if c[2] == "long" else -1, step)]
        if reentry_rule != "none":
            record.update(reentry_candidates=len(admitted), reentry_rejected=len(selected)-len(admitted),
                          reentry_substitution=bool(admitted and selected and admitted[0] != selected[0]))
        if regime_mode in ("entry", "both"):
            before = admitted
            admitted = [c for c in before if entry_allowed(
                market_regime[lookup[c[1]].trade_date, c[1]], 1 if c[2] == "long" else -1)]
            record.update(regime_rejected=len(before)-len(admitted),
                          regime_substitution=bool(admitted and before and admitted[0] != before[0]))
            if before and not admitted:
                return finish("regime_block", flat)
        if evidence_model:
            before=admitted
            model=evidence_names[evidence_model-1]
            bps=round(fee*10000)
            evidence={}
            for c in before:
                key=(day,c[1],1 if c[2]=="long" else -1,bps,model)
                values=evidence_lookup.get((*key,int(current=="opposing_exit")),evidence_lookup.get(key))
                if values is None:
                    raise AssertionError(f"Missing entry forecast: {key}, policy={current}")
                evidence[c[1]]=values
            admitted=[c for c in before if evidence_admits(*evidence[c[1]])]
            record.update(evidence_rejected=len(before)-len(admitted),
                evidence_fallback=sum(not np.isfinite(evidence[c[1]][0]) for c in before),
                evidence_substitution=bool(admitted and before and admitted[0]!=before[0]))
            if before and not admitted:
                return finish("evidence_block",flat)
        if raw:
            record["minimum_signal_equity"] = min(minimum_equity(lookup[s], 1 if side == "long" else -1,
                                                                 fee, margin) for _, s, side in raw)
        if not selected:
            return finish("no_signal" if not raw else "capital_block" if not funded else "risk_block", flat)
        if not admitted:
            return finish("reentry_block", flat)
        _, symbol, side = admitted[0]
        opened, entry_equity, opened_day = step, equity, day
        profit_peak, soft_streak = 0., 0
        entry_limit=5 if soft_exit==8 else 10 if soft_exit==3 else holding_limit(market_regime[lookup[symbol].trade_date,symbol],1 if side=="long" else -1) if soft_exit in (4,7) else 20
        record.update(selected_product=symbol, selected_side=side,
                      entry_two_sigma_fraction=2*lookup[symbol].f_volatility_20*
                      lookup[symbol].entry_open*lookup[symbol].multiplier/equity)
        if regime_mode != "none" or evidence_model or soft_exit:
            record["regime_state"] = market_regime[lookup[symbol].trade_date, symbol]
            record["entry_holding_limit"] = entry_limit
        if evidence_model:
            record.update(zip(("entry_probability","entry_pred_net","entry_pred_risk"),evidence[symbol]))
        return finish("enter", (f"{side}_{symbol}", .5, 10000, "both", margin_reserve))

    ledger = simulate(groups, "flat", .5, 1, "both", 1, fee, margin_rate=margin,
                      capital=capital, enforce_margin=True, lot_cap=1, state_policy_selector=select)
    assert ledger.gross_lots.le(1).all()
    np.testing.assert_allclose(ledger.equity, capital+ledger.pnl.cumsum(), atol=1e-6, rtol=0)
    return ledger, events, pd.DataFrame(diagnostics)


def first_year(ledger, start, opens, fee):
    """Causal first-year prefix with an additional year-end liquidation if needed."""
    prefix = ledger[ledger.exit_date.str[:4].eq(start[:4])].copy().reset_index(drop=True)
    if len(prefix) < len(ledger) and prefix.iloc[-1].positions:
        last = prefix.iloc[-1]
        code, sign = last.positions.split(":")
        multiplier = 300 if code[:2] in ("IF", "IH") else 200
        cost = abs(int(sign))*opens[last.exit_date, code]*multiplier*fee
        prefix.loc[prefix.index[-1], ["pnl", "equity"]] -= cost
        prefix.loc[prefix.index[-1], "fees"] += cost
    return prefix


def diagnose_trades(ledger, trades, trace, closes, opens, fee):
    result = []
    trace = trace.set_index("date")
    for trade in trades.itertuples():
        held = ledger[ledger.entry_date.ge(trade.entry) & ledger.entry_date.lt(trade.exit)]
        first_index = held.index[0]
        initial = CAPITAL if first_index == 0 else ledger.iloc[first_index-1].equity
        marks = [0.]
        for row in held.itertuples():
            code, sign = row.positions.split(":")
            direction = int(sign)
            close = closes[row.entry_date, code]
            mark = row.equity-initial - direction*(opens[row.exit_date, code]-close)*trade.multiplier
            if row.Index == len(ledger)-1:  # undo the engine's final-open liquidation
                mark += opens[row.exit_date, code]*trade.multiplier*fee
            marks.append(mark-close*trade.multiplier*fee)
        exit_reason = trace.loc[trade.exit, "action"] if trade.exit in trace.index else "sample_end"
        entry_trace = trace.loc[trade.entry]
        result.append(dict(entry=trade.entry, exit=trade.exit, entry_equity=initial,
                           entry_two_sigma_fraction=entry_trace.entry_two_sigma_fraction,
                           entry_policy=entry_trace.policy, exit_reason=exit_reason,
                           best_close_pnl=max(marks), worst_close_pnl=min(marks),
                           close_peak_giveback=max(marks)-trade.pnl))
    return trades.merge(pd.DataFrame(result), on=["entry", "exit"], validate="one_to_one") if result else trades


def main():
    OUT.mkdir(exist_ok=True)
    protocol = dict(capital=CAPITAL, max_lots=1, variants=VARIANTS, start_years=list(range(2022, 2027)),
                    fee=[.0003, .0006], margin=[.15, .25], risk_fraction=RISK_FRACTION,
                    entry_risk="2 * past20 daily volatility * one-lot notional <= 5% current equity",
                    loss_exit="net previous-close trade loss >= 5% entry equity, exit next open",
                    signal_exit="opposite20 >= .45 and opposite20-own20 >= .10 for 2 consecutive closes",
                    priority="strategy switch; 20-session limit; native exit; loss exit; signal exit; margin reserve",
                    controls="same models/labels/36m history/12m weights/quarterly guarded selection",
                    selection="historical candidates replay the SAME overlay at actual current account equity",
                    evaluation="single independent starting year and each independent start through data end",
                    limits="exploratory previously inspected history; no new OOT; daily next-open execution; no intraday guarantee",
                    risk_setting="5% is a fixed research scenario, not an agreed user risk budget or hard realized-loss cap")
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    panel = pd.read_parquet(PREDICTIONS / "quarterly_predictions.parquet")
    bars = load_bars()
    opens, closes = bars.open.to_dict(), bars.close.to_dict()
    audits, choices, summaries, annual, all_trades, events_all, traces = [], [], [], [], [], [], []
    matches = 0

    @lru_cache(maxsize=None)
    def history(cutoff):
        past = history_before(panel, cutoff)
        assert past.exit_date.max() <= cutoff
        return groups_for(past)

    @lru_cache(maxsize=512)
    def rankings(day, cutoff, equity, fee, margin, variant):
        groups = history(cutoff)
        scores = {}
        for policy in POLICIES:
            ledger, _, _ = risk_replay(groups, opens, closes, fee, margin, capital=equity,
                                       policy=policy, variant=variant)
            trades = trade_episodes(ledger, opens, fee)
            scores[policy] = capital_quality(ledger, day, equity, len(trades))
            audits.append(dict(date=day, cutoff=cutoff, validation_initial_equity=equity, fee=fee, margin=margin,
                               variant=variant, policy=policy, last_validation_exit=ledger.exit_date.iloc[-1], **scores[policy]))
        assert groups[-1][0].exit_date <= cutoff < day
        return scores

    for year in protocol["start_years"]:
        start = f"{year}0101"
        groups = groups_for(panel[panel.entry_date.ge(start)])
        for fee in protocol["fee"]:
            for margin in protocol["margin"]:
                for variant in VARIANTS:
                    fields = dict(start=start, fee=fee, margin=margin, variant=variant)
                    def decision(day, cutoff, incumbent, equity, positions, rows):
                        scores = rankings(day, cutoff, equity, fee, margin, variant)
                        selected, reason, detail = decide(scores, incumbent, rows, positions, equity, fee, margin, True)
                        choices.append(dict(**fields, date=day, cutoff=cutoff, current_equity=equity,
                                            incumbent=incumbent, selected=selected, reason=reason, **detail))
                        return selected
                    ledger, events, trace = risk_replay(groups, opens, closes, fee, margin,
                                                       decision=decision, variant=variant, trace=True)
                    if variant == "control":
                        old = pd.read_csv(BASELINE / f"ledger_{start}_reselect_guarded_{fee}_{margin}.csv",
                                          dtype={"entry_date": str, "exit_date": str}).fillna({"positions": ""})
                        pd.testing.assert_frame_equal(ledger, old, check_dtype=False, atol=1e-6, rtol=1e-12)
                        matches += 1
                    trades = trade_episodes(ledger, opens, fee)
                    all_trades.append(diagnose_trades(ledger, trades, trace, closes, opens, fee).assign(**fields))
                    annual.extend(dict(**fields, **row) for row in annual_stats(ledger, trades))
                    events_all.extend(dict(**fields, **event) for event in events)
                    traces.append(trace.assign(**fields))
                    for window, sample in (("to_latest", ledger), ("single_year", first_year(ledger, start, opens, fee))):
                        episodes = trade_episodes(sample, opens, fee)
                        summaries.append(dict(**fields, window=window, end=sample.exit_date.iloc[-1],
                                              **metrics(sample, CAPITAL), **trade_stats(episodes), **stability(sample)))
                    ledger.to_csv(OUT / f"ledger_{start}_{variant}_{fee}_{margin}.csv", index=False)
                print("evaluated", start, fee, margin, flush=True)
        pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
        pd.DataFrame(choices).to_csv(OUT / "selection.csv", index=False)
        pd.DataFrame(audits).to_csv(OUT / "validation_audit.csv", index=False)
    pd.DataFrame(annual).to_csv(OUT / "annual_marked.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(OUT / "daily_decisions.csv", index=False)
    pd.DataFrame(events_all).to_csv(OUT / "switches.csv", index=False)
    (OUT / "checks.json").write_text(json.dumps(dict(control_matches=matches, accounts=100,
                                                     evaluated_windows=len(summaries),
                                                     causal_validation=True, trade_ledger_reconciliation=True), indent=2))


if __name__ == "__main__":
    main()
