"""Frozen multi-start, equal-horizon policy selection on a fixed calendar.

Inner reference accounts start at CNY500k; live replay accounts keep their actual
wealth. Reference scores are not presented as forecasts at current account size.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from backtest_index_futures import ROOT, metrics
from compare_index_futures_adaptation import POLICIES, groups_for, stability, trade_episodes
from evaluate_index_futures_calibration import (OUT as CALIBRATION, VARIANTS, PROFILES,
    calibrated_callback, liquidated_prefix, start_specs, weighted_quantile)
from evaluate_index_futures_directions import CAPITAL, load_bars
from evaluate_index_futures_exit_value import exit_callback
from evaluate_index_futures_risk_paths import risk_replay, first_year
from evaluate_index_futures_startup import OUT as PREVIOUS, SOURCE, global_calendar
from evaluate_index_futures_trade_wins import annual_stats, trade_stats
from refine_index_futures_adaptation import switch_cost

OUT = ROOT / "multistart_selection"
NEW_MODES = ("multi_tail", "multi_funded")
MAIN_MODES = ("calendar", "calendar_history", *NEW_MODES)
ALL_PROFILES = (*PROFILES, (.0003, .25), (.0006, .15))
HORIZONS = (122, 244)
MIN_WINDOWS = 6
SCORE_HURDLE = .005
DATE_TYPES = {k: str for k in ("start", "end", "date", "cutoff", "entry", "exit", "year",
    "first_entry_date", "first_history_start", "last_history_end", "signal_date")}


def blocked_counts(trace):
    """Denominator: flat, scheduled days with a policy-qualified entry signal.

    Holding days, non-entry days, and no-signal days are not failed executions.
    Partially affordable candidate sets are executable and not counted blocked.
    """
    demand = trace.raw_candidates.gt(0)
    blocked = demand & trace.affordable_candidates.eq(0)
    return dict(signal_opportunities=int(demand.sum()), blocked_opportunities=int(blocked.sum()),
                blocked_ratio=float(blocked.sum()/demand.sum()) if demand.any() else np.nan)


def past_windows(windows, cutoff):
    beginning = (pd.Timestamp(cutoff)-pd.DateOffset(months=36)).strftime("%Y%m%d")
    return windows[windows.start.ge(beginning)&windows.end.le(cutoff)].copy()


def summarize_policy(windows, cutoff):
    """Each horizon gets half the weight; recent matured ends get more weight."""
    result = dict(ready=False, windows=len(windows), forced=int(windows.forced.sum()),
                  first_history_start=windows.start.min(), last_history_end=windows.end.max())
    parts = []
    for horizon in HORIZONS:
        part = windows[windows.horizon.eq(horizon)]
        result[f"windows_{horizon}"] = len(part)
        if len(part) < MIN_WINDOWS:
            return result
        age = (pd.Timestamp(cutoff)-pd.to_datetime(part.end)).dt.days.to_numpy()
        assert (age >= 0).all()
        weight = np.exp2(-age/365.25)
        normalized = part.total_return.to_numpy()*122/horizon
        demand = np.average(part.signal_opportunities, weights=weight)
        blocked = np.average(part.blocked_opportunities, weights=weight)
        parts.append(dict(q25=weighted_quantile(normalized, weight, .25),
                          median=weighted_quantile(normalized, weight, .5),
                          dd90=weighted_quantile(-part.max_drawdown.to_numpy(), weight, .9),
                          block=blocked/demand if demand > 0 else np.nan,
                          demand=demand, trades=np.average(part.trades*122/horizon, weights=weight),
                          active=np.average(part.active_days*122/horizon, weights=weight),
                          daily_pnl=np.average(part.total_return*CAPITAL/horizon, weights=weight)))
    result.update({key: float(np.mean([part[key] for part in parts])) for key in parts[0]})
    result["score"] = .5*(result["q25"]+result["median"])-.5*result["dd90"]
    result["ready"] = bool(np.isfinite(result["block"]))
    return result


def select_policy(scores, current, mode, *, initial=False, rows=None, positions=None,
                  equity=CAPITAL, fee=.0003, margin=.15):
    base = scores["base"]
    detail = dict(proposed=current, candidate_score=np.nan, incumbent_score=np.nan,
                  score_improvement=np.nan, estimated_switch_cost=np.nan,
                  historical_quarter_edge=np.nan, required_edge=np.nan)
    if not base.get("ready", False):
        return current, "insufficient_mature_windows", detail
    eligible = {}
    for policy, value in scores.items():
        if not value.get("ready", False):
            continue
        valid = (value["forced"] == 0 and value["median"] > 0
                 and value["trades"] >= max(2, .5*base["trades"])
                 and value["active"] >= max(10, .5*base["active"])
                 and value["dd90"] <= base["dd90"]+1e-12)
        if mode == "multi_funded":
            valid = valid and value["block"] <= base["block"]+1e-12
        if valid:
            eligible[policy] = value
    if not eligible:
        return current, "no_eligible_policy_keep_current", detail
    wanted = max(eligible, key=lambda name: (eligible[name]["score"], name == current, name))
    detail.update(proposed=wanted, candidate_score=scores[wanted]["score"],
                  incumbent_score=scores[current].get("score", np.nan))
    if initial or wanted == current:
        return wanted, "past_multistart_selection", detail
    candidate, incumbent = scores[wanted], scores[current]
    if not incumbent.get("ready", False):
        return current, "insufficient_incumbent_evidence", detail
    advantage = candidate["score"]-incumbent["score"]
    detail["score_improvement"] = advantage
    if advantage < SCORE_HURDLE:
        return current, "insufficient_robust_score_advantage", detail
    if incumbent["trades"] >= 2 and candidate["dd90"] > incumbent["dd90"]+1e-12:
        return current, "worse_dd90_than_incumbent", detail
    if mode == "multi_funded" and candidate["block"] > incumbent["block"]+1e-12:
        return current, "worse_execution_than_incumbent", detail
    cost = switch_cost(rows, positions, equity, fee, margin)
    if cost is None:
        return current, "no_currently_affordable_contract", detail
    edge = 61*(candidate["daily_pnl"]-incumbent["daily_pnl"])
    required = max(.005*equity, 3*cost)
    detail.update(estimated_switch_cost=cost, historical_quarter_edge=edge, required_edge=required)
    if edge < required:
        return current, "insufficient_net_advantage", detail
    return wanted, "passed_robust_and_cash_hurdles", detail


class Replay:
    def __init__(self):
        self.panel = pd.read_parquet(SOURCE/"quarterly_predictions.parquet")
        self.groups = groups_for(self.panel)
        self.calendar = global_calendar(self.panel)
        bars = load_bars()
        self.opens, self.closes = bars.open.to_dict(), bars.close.to_dict()
        forecast = pd.read_parquet(CALIBRATION/"calibrated_forecasts.parquet")
        keep = ["entry", "date", "symbol", "direction", "notional"]
        keep += [f"{p}_{b}" for b in (3, 6) for p in ("pred_net_5", "pred_risk_5", "cal_net", "risk90", "cal_ready")]
        self.lookup = {(r.entry, r.date, r.symbol): r for r in forecast[keep].itertuples(index=False)}

    def run(self, groups, fee, margin, variant, policy="base", decision=None):
        simple = exit_callback(self.lookup, "mean5", fee) if variant == "mean5" else None
        stateful = calibrated_callback(self.lookup, variant, fee) if variant in ("calibrated5", "tail5") else None
        return risk_replay(groups, self.opens, self.closes, fee, margin, policy=policy,
                           decision=decision, trace=True, variant=variant, exit_decider=simple,
                           state_exit_decider=stateful, entry_calendar=self.calendar)


def build_reference(variant):
    replay = Replay()
    records = []
    first_by_month = {}
    for i, group in enumerate(replay.groups):
        first_by_month.setdefault(group[0].entry_date[:6], i)
    for month, index in first_by_month.items():
        groups = replay.groups[index:index+max(HORIZONS)]
        if len(groups) < min(HORIZONS):
            continue
        for fee, margin in ALL_PROFILES:
            for policy in POLICIES:
                ledger, _, trace = replay.run(groups, fee, margin, variant, policy)
                assert trace.loc[trace.action.eq("enter"), "date"].isin(replay.calendar).all()
                for horizon in HORIZONS:
                    if len(ledger) < horizon:
                        continue
                    sample = liquidated_prefix(ledger, horizon, replay.opens, fee)
                    trades = trade_episodes(sample, replay.opens, fee)
                    stats = metrics(sample, CAPITAL)
                    records.append(dict(start=sample.entry_date.iloc[0], end=sample.exit_date.iloc[-1],
                        horizon=horizon, variant=variant, fee=fee, margin=margin, policy=policy,
                        total_return=stats["total_return"], max_drawdown=stats["max_drawdown"],
                        forced=int(sample.forced_exit.sum()), trades=len(trades),
                        active_days=int(sample.gross_lots.gt(0).sum()),
                        **blocked_counts(trace.iloc[:horizon])))
        print("reference", variant, month, len(records), flush=True)
    pd.DataFrame(records).to_csv(OUT/f"reference_{variant}.csv", index=False)


def run_year(year):
    replay = Replay()
    out = OUT/str(year)
    (out/"ledgers").mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(OUT/"reference_windows.csv", dtype=DATE_TYPES)
    summaries, annual_rows, trades_all, traces, decisions, audits = [], [], [], [], [], []
    old = pd.read_csv(PREVIOUS/str(year)/"summary.csv", dtype=DATE_TYPES)
    summaries.extend(old.to_dict("records"))
    annual_rows.extend(pd.read_csv(PREVIOUS/str(year)/"annual_marked.csv", dtype=DATE_TYPES).to_dict("records"))
    trades_all.append(pd.read_csv(PREVIOUS/str(year)/"trades.csv", dtype=DATE_TYPES))
    old_initial = pd.read_csv(PREVIOUS/str(year)/"startup_decisions.csv", dtype=DATE_TYPES)
    old_choices = pd.read_csv(PREVIOUS/str(year)/"selection.csv", dtype=DATE_TYPES)
    old_replays = 0

    @lru_cache(maxsize=None)
    def score_at(cutoff, fee, margin, variant):
        matching = reference[reference.fee.eq(fee)&reference.margin.eq(margin)&reference.variant.eq(variant)]
        past = past_windows(matching, cutoff)
        scores = {}
        for policy in POLICIES:
            sample = past[past.policy.eq(policy)]
            scores[policy] = summarize_policy(sample, cutoff)
            audits.append(dict(cutoff=cutoff, fee=fee, margin=margin, variant=variant, policy=policy, **scores[policy]))
        assert past.end.max() <= cutoff
        return scores

    for spec in start_specs(replay.panel, year):
        start = spec["start"]
        groups = [g for g in replay.groups if g[0].entry_date >= start]
        profiles = ALL_PROFILES if spec["annual"] else PROFILES
        variants = VARIANTS[:3] if spec["start_kind"] == "phase" else VARIANTS
        for fee, margin in profiles:
            for variant in variants:
                for mode in MAIN_MODES[:2]:
                    fields = dict(**spec, fee=fee, margin=margin, variant=variant, startup_mode=mode,
                                  first_entry_date=groups[0][0].entry_date)
                    def match(frame):
                        return frame.start.eq(start)&frame.fee.eq(fee)&frame.margin.eq(margin)&frame.variant.eq(variant)&frame.startup_mode.eq(mode)
                    initial = old_initial.loc[match(old_initial), "initial_policy"].item()
                    schedule = old_choices[match(old_choices)].set_index("date").selected.to_dict()
                    verify, _, trace = replay.run(groups, fee, margin, variant, initial,
                        lambda day, cutoff, current, equity, positions, rows: schedule.get(day, current))
                    original = pd.read_parquet(PREVIOUS/str(year)/"ledgers"/f"{start}_{variant}_{mode}_{fee}_{margin}.parquet")
                    pd.testing.assert_frame_equal(verify, original, check_dtype=False, atol=1e-6, rtol=1e-12)
                    old_replays += 1
                    for record in summaries:
                        if all(record[k] == fields[k] for k in ("start", "fee", "margin", "variant", "startup_mode")):
                            record.update(blocked_counts(trace.iloc[:int(record["days"])]))
                    if spec["annual"]:
                        traces.append(trace.assign(**fields))
                for mode in NEW_MODES:
                    fields = dict(**spec, fee=fee, margin=margin, variant=variant, startup_mode=mode,
                                  first_entry_date=groups[0][0].entry_date)
                    first = groups[0][0]
                    scores = score_at(first.trade_date, fee, margin, variant)
                    initial, reason, detail = select_policy(scores, "base", mode, initial=True)
                    decisions.append(dict(**fields, stage="startup", date=first.entry_date, cutoff=first.trade_date,
                                          equity=CAPITAL, incumbent="base", selected=initial, reason=reason, **detail))
                    def decision(day, cutoff, incumbent, equity, positions, rows):
                        assert cutoff < day
                        scores = score_at(cutoff, fee, margin, variant)
                        selected, reason, detail = select_policy(scores, incumbent, mode, rows=rows, positions=positions,
                                                                equity=equity, fee=fee, margin=margin)
                        decisions.append(dict(**fields, stage="quarter", date=day, cutoff=cutoff,
                                              equity=equity, incumbent=incumbent, selected=selected, reason=reason, **detail))
                        return selected
                    ledger, events, trace = replay.run(groups, fee, margin, variant, initial, decision)
                    assert trace.loc[trace.action.eq("enter"), "date"].isin(replay.calendar).all()
                    trades = trade_episodes(ledger, replay.opens, fee)
                    trades_all.append(trades.assign(**fields))
                    if spec["annual"]:
                        annual_rows.extend(dict(**fields, **r) for r in annual_stats(ledger, trades))
                        traces.append(trace.assign(**fields))
                    windows = [("to_latest", ledger)]
                    if spec["annual"]:
                        windows.append(("single_year", first_year(ledger, start, replay.opens, fee)))
                    windows.extend((f"sessions_{h}", liquidated_prefix(ledger, h, replay.opens, fee))
                                   for h in HORIZONS if len(ledger) >= h)
                    for window, sample in windows:
                        episodes = trade_episodes(sample, replay.opens, fee)
                        daily = trace.iloc[:len(sample)]
                        summaries.append(dict(**fields, window=window, end=sample.exit_date.iloc[-1],
                            **metrics(sample, CAPITAL), **trade_stats(episodes), **stability(sample),
                            capital_blocks=int(daily.action.eq("capital_block").sum()),
                            tail_exits=int(daily.action.eq("tail_budget_exit").sum()),
                            model_exits=int(daily.action.isin(["learned_exit", "calibrated_mean_exit"]).sum()),
                            changes=sum(e["date"] <= sample.entry_date.iloc[-1] for e in events),
                            **blocked_counts(daily)))
                    ledger.to_parquet(out/"ledgers"/f"{start}_{variant}_{mode}_{fee}_{margin}.parquet", index=False)
        pd.DataFrame(summaries).to_csv(out/"summary.csv", index=False)
        print("outer", year, start, flush=True)
    pd.DataFrame(annual_rows).to_csv(out/"annual_marked.csv", index=False)
    pd.concat(trades_all, ignore_index=True).to_csv(out/"trades.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(out/"annual_decisions.csv", index=False)
    pd.DataFrame(decisions).to_csv(out/"selection.csv", index=False)
    pd.DataFrame(audits).to_csv(out/"selection_audit.csv", index=False)
    (out/"checks.json").write_text(json.dumps(dict(replayed_fixed_calendar_controls=old_replays)))


def register():
    OUT.mkdir(exist_ok=True)
    protocol = dict(capital=CAPITAL, max_lots=1, frozen_calendar="20200102 anchor; every5 sessions",
        variants=VARIANTS, new_modes=NEW_MODES, controls="All3000 previous accounts retained; main controls calendar and calendar_history",
        inner="fixed policy, CNY500k flat startup, monthly starts in previous36months, complete122/244 sessions, terminal liquidation charged",
        temporal="inner end <= prior signal date < decision date; candidate predictions/labels/calibration unchanged genuine historical OOS",
        capital_distinction="inner standardized500k, not current-equity historical replay; outer continuous actual equity and affordable/switch cost checks",
        weighting="12m half-life by maturity/end; each horizon half weight; 244d returns linearly divided2 to122d scale, not compounded annualization",
        score="0.5*(weighted25th percentile + weighted median return) -0.5*weighted90th percentile drawdown; average horizon components",
        eligibility="atleast6 windows per horizon; zero forced episodes; median>0; trades>=max(2,.5base) and active>=max(10,.5base), normalized122d; DD90<=base",
        funded_ablation="multi_funded additionally block ratio<=base, and<=incumbent for switches; multi_tail ignores this gate",
        denominator="policy-qualified signals on flat scheduled entry days; no affordable candidate counts blocked; no-signal undefined not zero",
        switch="score advantage>=.005; DD90<=incumbent when incumbent trades>=2; current affordable contract;61day historical mean netPnL edge>=max(.005actual equity,3estimated switch friction)",
        startup="past-only score; no switch hurdle because no incumbent; unavailable=>base; later none eligible=>keep current",
        no_parameter_search=True, evaluation="same58 monthly+January5day grid; annualsingle/to_latest/equal122/244; fees3/6bps margins15/25%",
        limitations="overlapping historical starts not independent; past repeatedly inspected; daily execution not intraday liquidation guarantee; no refreshed market data")
    (OUT/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (OUT/"global_calendar.csv").write_bytes((PREVIOUS/"global_calendar.csv").read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", choices=VARIANTS)
    parser.add_argument("--year", type=int)
    args = parser.parse_args()
    if args.reference:
        build_reference(args.reference)
        return
    if args.year:
        run_year(args.year)
        return
    register()
    def worker(task):
        subprocess.run([sys.executable, __file__, *task], check=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(worker, [("--reference", v) for v in VARIANTS]))
    pd.concat([pd.read_csv(OUT/f"reference_{v}.csv", dtype=DATE_TYPES) for v in VARIANTS], ignore_index=True).to_csv(OUT/"reference_windows.csv", index=False)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(worker, [("--year", str(y)) for y in range(2022, 2027)]))
    for name in ("summary", "annual_marked", "trades", "annual_decisions", "selection", "selection_audit"):
        pd.concat([pd.read_csv(OUT/str(y)/f"{name}.csv", dtype=DATE_TYPES, low_memory=False)
                   for y in range(2022, 2027)], ignore_index=True).to_csv(OUT/f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
