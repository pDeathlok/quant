"""2x2 frozen startup/switch advantage and exact current-equity validation."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from evaluate_index_futures_multistart import (OUT as PREVIOUS, Replay, POLICIES, VARIANTS, PROFILES,
    ALL_PROFILES, CAPITAL, HORIZONS, DATE_TYPES, past_windows, summarize_policy, select_policy,
    blocked_counts, start_specs, liquidated_prefix, first_year, metrics, trade_episodes, trade_stats,
    annual_stats, stability, risk_replay, exit_callback, calibrated_callback, weighted_quantile)
from index_futures_equity_fast import FastInner

OUT = PREVIOUS.parent/"equity_advantage_selection"
NEW_MODES = ("score_gate", "actual_equity", "equity_score_gate")
MAIN_MODES = ("calendar", "multi_funded", *NEW_MODES)
GATE = .005


class EquityReplay(Replay):
    def run(self, groups, fee, margin, variant, policy="base", decision=None, capital=CAPITAL):
        simple = exit_callback(self.lookup, "mean5", fee) if variant == "mean5" else None
        stateful = calibrated_callback(self.lookup, variant, fee) if variant in ("calibrated5", "tail5") else None
        return risk_replay(groups, self.opens, self.closes, fee, margin, policy=policy, capital=capital,
            decision=decision, trace=True, variant=variant, exit_decider=simple,
            state_exit_decider=stateful, entry_calendar=self.calendar)


def equity_scores(windows, cutoff, capital):
    result = summarize_policy(windows, cutoff)
    if not result["ready"]:
        return result
    # summarize_policy's legacy currency conversion is500k. Returns here have
    # already been simulated at actual capital; convert back to actual cash.
    result["daily_pnl"] *= capital/CAPITAL
    for horizon in HORIZONS:
        part = windows[windows.horizon.eq(horizon)]
        age = (pd.Timestamp(cutoff)-pd.to_datetime(part.end)).dt.days.to_numpy()
        weight = np.exp2(-age/365.25)
        returns = part.total_return.to_numpy()*122/horizon
        result[f"score_{horizon}"] = .5*(weighted_quantile(returns, weight, .25)
            +weighted_quantile(returns, weight, .5)-weighted_quantile(-part.max_drawdown.to_numpy(), weight, .9))
    return result


def gated_select(scores, current, gated, *, initial=False, **kwargs):
    proposed, reason, detail = select_policy(scores, current, "multi_funded", initial=True)
    detail.update(advantage_122=np.nan, advantage_244=np.nan, horizon_gate_passed=False)
    if proposed == current:
        return current, reason, detail
    for h in HORIZONS:
        detail[f"advantage_{h}"] = scores[proposed].get(f"score_{h}", np.nan)-scores[current].get(f"score_{h}", np.nan)
    passed = all(detail[f"advantage_{h}"] >= GATE for h in HORIZONS)
    detail["horizon_gate_passed"] = passed
    if gated and not passed:
        return current, "insufficient_horizon_score_advantage", detail
    if initial:
        return proposed, reason, detail
    selected, reason, old_detail = select_policy(scores, current, "multi_funded", initial=False, **kwargs)
    detail.update(old_detail)
    return selected, reason, detail


def validate_fast():
    replay = EquityReplay()
    fast = FastInner(replay)
    reference = pd.read_csv(PREVIOUS/"reference_windows.csv", dtype=DATE_TYPES)
    columns = ["total_return", "max_drawdown", "trades", "active_days", "forced", "signal_opportunities", "blocked_opportunities"]
    for r in reference.itertuples(index=False):
        result, _ = fast.run(r.start, r.horizon, r.policy, r.variant, r.fee, r.margin, CAPITAL)
        np.testing.assert_allclose([result[k] for k in columns], [getattr(r,k) for k in columns], atol=1e-9, rtol=0)
    print("all frozen reference windows match", len(reference), flush=True)
    cash_cases, max_error = 0, 0.
    for capital in (265421.952, 350000., 499123.456, 750000., 1250000.):
        for start in ("20200102", "20230103", "20240902"):
            i = fast.index[start]
            for fee, margin in ALL_PROFILES:
                for variant in VARIANTS:
                    for policy in POLICIES:
                        result, daily = fast.run(i, 244, policy, variant, fee, margin, capital, trace=True)
                        ledger, _, trace = replay.run(replay.groups[i:i+244], fee, margin, variant, policy, capital=capital)
                        expected = np.column_stack([ledger.pnl, ledger.equity, ledger.gross_lots, ledger.fees,
                            ledger.forced_exit, ledger.settlement_breach, trace.raw_candidates, trace.affordable_candidates])
                        max_error = max(max_error, float(np.max(abs(daily[:, :8]-expected))))
                        np.testing.assert_allclose(daily[:, :8], expected, atol=1e-6, rtol=0)
                        positions = ["" if c<0 else f"{fast.codes[int(c)]}:{int(s):+d}" for c,s in daily[:,8:10]]
                        assert positions == ledger.positions.tolist()
                        assert result["trades"] == len(trade_episodes(ledger, replay.opens, fee))
                        cash_cases += 1
        print("cash parity", capital, cash_cases, max_error, flush=True)
    (OUT/"engine_checks.json").write_text(json.dumps(dict(reference_windows=len(reference),
        exact_cash_daily_parity_cases=cash_cases, max_daily_cash_error=max_error), indent=2))


def run_year(year):
    replay = EquityReplay()
    fast = FastInner(replay)
    out = OUT/str(year)
    (out/"ledgers").mkdir(parents=True, exist_ok=True)
    (out/"validation").mkdir(exist_ok=True)
    reference = pd.read_csv(PREVIOUS/"reference_windows.csv", dtype=DATE_TYPES)
    summaries = pd.read_csv(PREVIOUS/str(year)/"summary.csv", dtype=DATE_TYPES).to_dict("records")
    annual_rows = pd.read_csv(PREVIOUS/str(year)/"annual_marked.csv", dtype=DATE_TYPES).to_dict("records")
    all_trades = [pd.read_csv(PREVIOUS/str(year)/"trades.csv", dtype=DATE_TYPES)]
    audits, decisions, traces = [], [], []
    old_decisions = pd.read_csv(PREVIOUS/str(year)/"selection.csv", dtype=DATE_TYPES)
    matched = 0

    @lru_cache(maxsize=None)
    def template(cutoff, fee, margin, variant):
        selected = reference[reference.fee.eq(fee)&reference.margin.eq(margin)&reference.variant.eq(variant)]
        return past_windows(selected, cutoff)

    @lru_cache(maxsize=4096)
    def score_at(cutoff, fee, margin, variant, capital):
        assert capital > 0
        reference_part = template(cutoff, fee, margin, variant)
        key = json.dumps([cutoff, fee, margin, variant, capital.hex()])
        validation_id = hashlib.sha256(key.encode()).hexdigest()[:20]
        if capital == CAPITAL:
            windows = reference_part.copy()
        else:
            records = []
            for r in reference_part.itertuples(index=False):
                result, _ = fast.run(r.start, r.horizon, r.policy, variant, fee, margin, capital)
                records.append(result)
            windows = pd.DataFrame(records)
        assert windows.end.max() <= cutoff
        windows.assign(validation_capital=capital, cutoff=cutoff, fee=fee, margin=margin,
                       variant=variant).to_parquet(out/"validation"/f"{validation_id}.parquet", index=False)
        scores = {}
        for policy in POLICIES:
            scores[policy] = equity_scores(windows[windows.policy.eq(policy)], cutoff, capital)
            audits.append(dict(validation_id=validation_id, cutoff=cutoff, fee=fee, margin=margin,
                variant=variant, validation_capital=capital, policy=policy, **scores[policy]))
        return scores, validation_id

    for spec in start_specs(replay.panel, year):
        start = spec["start"]
        groups = [g for g in replay.groups if g[0].entry_date >= start]
        profiles = ALL_PROFILES if spec["annual"] else PROFILES
        variants = VARIANTS[:3] if spec["start_kind"] == "phase" else VARIANTS
        for fee, margin in profiles:
            for variant in variants:
                if spec["annual"]:
                    chosen = old_decisions[old_decisions.start.eq(start)&old_decisions.fee.eq(fee)&old_decisions.margin.eq(margin)
                        &old_decisions.variant.eq(variant)&old_decisions.startup_mode.eq("multi_funded")]
                    initial = chosen.loc[chosen.stage.eq("startup"), "selected"].item()
                    schedule = chosen[chosen.stage.eq("quarter")].set_index("date").selected.to_dict()
                    verify, _, _ = replay.run(groups, fee, margin, variant, initial,
                        lambda day, cutoff, current, equity, positions, rows: schedule.get(day, current))
                    old = pd.read_parquet(PREVIOUS/str(year)/"ledgers"/f"{start}_{variant}_multi_funded_{fee}_{margin}.parquet")
                    pd.testing.assert_frame_equal(verify, old, check_dtype=False, atol=1e-6, rtol=1e-12)
                    matched += 1
                for mode in NEW_MODES:
                    actual = mode != "score_gate"
                    gated = mode != "actual_equity"
                    fields = dict(**spec, fee=fee, margin=margin, variant=variant, startup_mode=mode,
                                  first_entry_date=groups[0][0].entry_date)
                    first = groups[0][0]
                    scores, validation_id = score_at(first.trade_date, fee, margin, variant, float(CAPITAL))
                    initial, reason, detail = gated_select(scores, "base", gated, initial=True)
                    decisions.append(dict(**fields, stage="startup", date=first.entry_date, cutoff=first.trade_date,
                        equity=CAPITAL, validation_capital=CAPITAL, validation_id=validation_id,
                        incumbent="base", selected=initial, reason=reason, **detail))
                    def decision(day, cutoff, incumbent, equity, positions, rows):
                        assert cutoff < day
                        capital = float(equity if actual else CAPITAL)
                        scores, validation_id = score_at(cutoff, fee, margin, variant, capital)
                        selected, reason, detail = gated_select(scores, incumbent, gated, rows=rows, positions=positions,
                            equity=equity, fee=fee, margin=margin)
                        decisions.append(dict(**fields, stage="quarter", date=day, cutoff=cutoff, equity=equity,
                            validation_capital=capital, validation_id=validation_id,
                            incumbent=incumbent, selected=selected, reason=reason, **detail))
                        return selected
                    ledger, events, trace = replay.run(groups, fee, margin, variant, initial, decision)
                    assert trace.loc[trace.action.eq("enter"), "date"].isin(replay.calendar).all()
                    trades = trade_episodes(ledger, replay.opens, fee)
                    all_trades.append(trades.assign(**fields))
                    if spec["annual"]:
                        annual_rows.extend(dict(**fields, **r) for r in annual_stats(ledger, trades))
                        traces.append(trace.assign(**fields))
                    windows = [("to_latest", ledger)]
                    if spec["annual"]:
                        windows.append(("single_year", first_year(ledger, start, replay.opens, fee)))
                    windows.extend((f"sessions_{h}", liquidated_prefix(ledger, h, replay.opens, fee)) for h in HORIZONS if len(ledger)>=h)
                    for window, sample in windows:
                        episodes = trade_episodes(sample, replay.opens, fee)
                        daily = trace.iloc[:len(sample)]
                        summaries.append(dict(**fields, window=window, end=sample.exit_date.iloc[-1],
                            **metrics(sample, CAPITAL), **trade_stats(episodes), **stability(sample),
                            capital_blocks=int(daily.action.eq("capital_block").sum()),
                            tail_exits=int(daily.action.eq("tail_budget_exit").sum()),
                            model_exits=int(daily.action.isin(["learned_exit", "calibrated_mean_exit"]).sum()),
                            changes=sum(e["date"]<=sample.entry_date.iloc[-1] for e in events), **blocked_counts(daily)))
                    ledger.to_parquet(out/"ledgers"/f"{start}_{variant}_{mode}_{fee}_{margin}.parquet", index=False)
        pd.DataFrame(summaries).to_csv(out/"summary.csv", index=False)
        print("evaluated", year, start, "validation", score_at.cache_info(), flush=True)
    for name, records in (("annual_marked", annual_rows), ("selection", decisions), ("selection_audit", audits)):
        pd.DataFrame(records).to_csv(out/f"{name}.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(out/"trades.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(out/"annual_decisions.csv", index=False)
    (out/"checks.json").write_text(json.dumps(dict(replayed_controls=matched, validations=score_at.cache_info().misses)))


def register():
    OUT.mkdir(exist_ok=True)
    protocol = dict(capital=CAPITAL, max_lots=1, calendar="fixed20200102 every5 sessions, identical to prior study",
        baseline="multi_funded:50w historical reference, no startup advantage floor, average switch score advantage>=.005",
        new_modes=NEW_MODES, design="2x2:stronger advantage gate yes/no; historical reference exact current equity yes/no",
        gate="both122d and244d score differences individually>=.005, at startup and quarterly switch; existing other gates unchanged",
        actual_equity="each historical window starts FLAT at precise current marked opening equity; rerun every fill, fee, reserve, stop and liquidation; no capital buckets or proportional scaling",
        information="startup only past reference plus known500k; quarter uses current open marked equity, historical window end<=previous signal date; no current settlement or next open in choice",
        historical="past36m monthly start complete122/244d windows; sameP25/median/P90 scoring,12m half-life and funding rejection gate; currency edge uses actual reference capital",
        frozen="models, labels, exits, policies, costs3/6bps, margins15/25%, coverage floors, switchcost3x/.5%equity, no parameter search",
        controls="all4500 previous accounts retained;100 annual multi_funded controls rerun; all11840 reference windows checked against optimized backend",
        evaluation="same58monthly/January5day/annual4profiles; single_year,to_latest,equal122/244; through20261106",
        limitations="already inspected overlapping history; no new untouchedOOT; daily fills and margin monitoring only")
    (OUT/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (OUT/"global_calendar.csv").write_bytes((PREVIOUS/"global_calendar.csv").read_bytes() if (PREVIOUS/"global_calendar.csv").exists()
        else (PREVIOUS.parent/"startup_consistency/global_calendar.csv").read_bytes())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args=parser.parse_args()
    if args.year:
        run_year(args.year)
        return
    register()
    validate_fast()
    if args.validate_only:
        return
    def worker(year):
        subprocess.run([sys.executable,__file__,"--year",str(year)],check=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(worker, range(2022,2027)))
    for name in ("summary","annual_marked","trades","annual_decisions","selection","selection_audit"):
        pd.concat([pd.read_csv(OUT/str(y)/f"{name}.csv",dtype=DATE_TYPES,low_memory=False) for y in range(2022,2027)],ignore_index=True).to_csv(OUT/f"{name}.csv",index=False)


if __name__=="__main__":
    main()
