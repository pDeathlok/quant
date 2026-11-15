"""Fixed calendar and past-only startup selection: preregistered 2x2 ablation."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from backtest_index_futures import ROOT, metrics
from compare_index_futures_adaptation import OUT as SOURCE, POLICIES, choose_policy, groups_for, history_before, stability, trade_episodes
from evaluate_index_futures_calibration import OUT as PREVIOUS, VARIANTS, PROFILES, calibrated_callback, liquidated_prefix, start_specs
from evaluate_index_futures_directions import CAPITAL, load_bars
from evaluate_index_futures_exit_value import exit_callback
from evaluate_index_futures_risk_paths import risk_replay, first_year
from evaluate_index_futures_trade_wins import annual_stats, trade_stats
from refine_index_futures_adaptation import capital_quality, decide

OUT = ROOT/"startup_consistency"
MODES = ("legacy", "calendar", "history", "calendar_history")
CALENDAR_ANCHOR = "20200102"


def global_calendar(panel):
    """Prefix-invariant session count; never optimize the offset on returns."""
    dates = sorted(d for d in panel.entry_date.unique() if d >= CALENDAR_ANCHOR)
    return frozenset(dates[::5])


def select_initial(scores):
    # Startup has no incumbent trade or switching friction. Apply the existing
    # absolute policy eligibility; switch-specific hurdles resume next quarter.
    return choose_policy(scores, "base")


def run_year(year):
    out = OUT/str(year)
    out.mkdir(exist_ok=True)
    (out/"ledgers").mkdir(exist_ok=True)
    panel = pd.read_parquet(SOURCE/"quarterly_predictions.parquet")
    calendar = global_calendar(panel)
    forecasts = pd.read_parquet(PREVIOUS/"calibrated_forecasts.parquet")
    keep = ["entry", "date", "symbol", "direction", "notional"]
    keep += [f"{prefix}_{bps}" for bps in (3, 6) for prefix in ("pred_net_5", "pred_risk_5", "cal_net", "risk90", "cal_ready")]
    lookup = {(r.entry, r.date, r.symbol): r for r in forecasts[keep].itertuples(index=False)}
    bars = load_bars()
    opens, closes = bars.open.to_dict(), bars.close.to_dict()
    old_summary = pd.read_csv(PREVIOUS/str(year)/"summary.csv", dtype={"start": str, "end": str, "first_entry_date": str})
    old_annual = pd.read_csv(PREVIOUS/str(year)/"annual_marked.csv", dtype={"start": str, "year": str, "first_entry_date": str})
    old_trades = pd.read_csv(PREVIOUS/str(year)/"trades.csv", dtype={"start": str, "entry": str, "exit": str, "first_entry_date": str})
    old_selection = pd.read_csv(PREVIOUS/str(year)/"selection.csv", dtype={"start": str, "date": str})
    audits, choices, startups, summaries, annual_rows, all_trades, traces = [], [], [], [], [], [], []
    summaries.extend(old_summary.assign(startup_mode="legacy").to_dict("records"))
    annual_rows.extend(old_annual.assign(startup_mode="legacy").to_dict("records"))
    all_trades.append(old_trades.assign(startup_mode="legacy"))
    control_replays = 0

    @lru_cache(maxsize=None)
    def history(cutoff):
        past = history_before(panel, cutoff)
        assert past.exit_date.max() <= cutoff
        return groups_for(past)

    def replay(groups, fee, margin, variant, fixed_calendar, capital=CAPITAL, policy="base", decision=None, trace=False):
        simple = exit_callback(lookup, "mean5", fee) if variant == "mean5" else None
        stateful = calibrated_callback(lookup, variant, fee) if variant in ("calibrated5", "tail5") else None
        return risk_replay(groups, opens, closes, fee, margin, capital=capital, policy=policy,
                           decision=decision, trace=trace, variant=variant, exit_decider=simple,
                           state_exit_decider=stateful, entry_calendar=calendar if fixed_calendar else None)

    @lru_cache(maxsize=8192)
    def rankings(day, cutoff, equity, fee, margin, variant, fixed_calendar):
        groups = history(cutoff)
        scores = {}
        for policy in POLICIES:
            ledger, _, _ = replay(groups, fee, margin, variant, fixed_calendar, capital=equity, policy=policy)
            episodes = trade_episodes(ledger, opens, fee)
            scores[policy] = capital_quality(ledger, day, equity, len(episodes))
            audits.append(dict(date=day, cutoff=cutoff, validation_initial_equity=equity, fee=fee, margin=margin,
                               variant=variant, fixed_calendar=fixed_calendar, policy=policy,
                               first_validation_entry=ledger.entry_date.iloc[0],
                               last_validation_exit=ledger.exit_date.iloc[-1], **scores[policy]))
        assert groups[-1][0].exit_date <= cutoff < day
        return scores

    for spec in start_specs(panel, year):
        start = spec["start"]
        groups = groups_for(panel[panel.entry_date.ge(start)])
        profiles = (*PROFILES, (.0003, .25), (.0006, .15)) if spec["annual"] else PROFILES
        variants = VARIANTS[:3] if spec["start_kind"] == "phase" else VARIANTS
        for fee, margin in profiles:
            for variant in variants:
                if spec["annual"]:
                    old = pd.read_parquet(PREVIOUS/str(year)/"ledgers"/f"{start}_{variant}_{fee}_{margin}.parquet")
                    old_choices = old_selection[old_selection.start.eq(start)&old_selection.fee.eq(fee)
                                                &old_selection.margin.eq(margin)&old_selection.variant.eq(variant)]
                    schedule = old_choices.set_index("date").selected.to_dict()
                    # Reexecute original fills/exits using already audited original
                    # quarterly decisions; this is a compatibility check, not a
                    # newly optimized decision schedule.
                    verify, _, _ = replay(groups, fee, margin, variant, False,
                                          decision=lambda day, cutoff, current, equity, positions, rows: schedule.get(day, current))
                    pd.testing.assert_frame_equal(verify, old, check_dtype=False, atol=1e-6, rtol=1e-12)
                    control_replays += 1
                for mode in MODES[1:]:
                    fixed = mode in ("calendar", "calendar_history")
                    warm = mode in ("history", "calendar_history")
                    fields = dict(**spec, fee=fee, margin=margin, variant=variant, startup_mode=mode,
                                  first_entry_date=groups[0][0].entry_date)
                    day, cutoff = groups[0][0].entry_date, groups[0][0].trade_date
                    initial, initial_reason = "base", "fixed_base_start"
                    if warm:
                        scores = rankings(day, cutoff, CAPITAL, fee, margin, variant, fixed)
                        initial, initial_reason = select_initial(scores)
                    startups.append(dict(**fields, date=day, cutoff=cutoff, initial_policy=initial,
                                         selection_reason=initial_reason, selection_used_past=warm,
                                         selection_capital=CAPITAL,
                                         last_history_exit=history(cutoff)[-1][0].exit_date if warm else None,
                                         next_scheduled_entry=min(d for d in calendar if d >= day) if fixed else day))
                    def decision(day, cutoff, incumbent, equity, positions, rows):
                        scores = rankings(day, cutoff, equity, fee, margin, variant, fixed)
                        selected, reason, detail = decide(scores, incumbent, rows, positions, equity, fee, margin, True)
                        choices.append(dict(**fields, date=day, cutoff=cutoff, equity=equity,
                                            incumbent=incumbent, selected=selected, reason=reason, **detail))
                        return selected
                    ledger, events, trace = replay(groups, fee, margin, variant, fixed, policy=initial, decision=decision, trace=True)
                    if fixed:
                        assert trace.loc[trace.action.eq("enter"), "date"].isin(calendar).all()
                    trades = trade_episodes(ledger, opens, fee)
                    all_trades.append(trades.assign(**fields))
                    if spec["annual"]:
                        annual_rows.extend(dict(**fields, **r) for r in annual_stats(ledger, trades))
                        traces.append(trace.assign(**fields))
                    windows = [("to_latest", ledger)]
                    if spec["annual"]:
                        windows.append(("single_year", first_year(ledger, start, opens, fee)))
                    for count in (122, 244):
                        if len(ledger) >= count:
                            windows.append((f"sessions_{count}", liquidated_prefix(ledger, count, opens, fee)))
                    for window, sample in windows:
                        episodes = trade_episodes(sample, opens, fee)
                        daily = trace[trace.date.le(sample.entry_date.iloc[-1])]
                        summaries.append(dict(**fields, window=window, end=sample.exit_date.iloc[-1],
                                              **metrics(sample, CAPITAL), **trade_stats(episodes), **stability(sample),
                                              capital_blocks=int(daily.action.eq("capital_block").sum()),
                                              tail_exits=int(daily.action.eq("tail_budget_exit").sum()),
                                              model_exits=int(daily.action.isin(["learned_exit", "calibrated_mean_exit"]).sum()),
                                              changes=sum(e["date"] <= sample.entry_date.iloc[-1] for e in events)))
                    ledger.to_parquet(out/"ledgers"/f"{start}_{variant}_{mode}_{fee}_{margin}.parquet", index=False)
        pd.DataFrame(summaries).to_csv(out/"summary.csv", index=False)
        print("evaluated startup", year, start, spec["start_kind"], flush=True)
    pd.DataFrame(audits).to_csv(out/"validation_audit.csv", index=False)
    pd.DataFrame(choices).to_csv(out/"selection.csv", index=False)
    pd.DataFrame(startups).to_csv(out/"startup_decisions.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(out/"annual_marked.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(out/"trades.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(out/"annual_decisions.csv", index=False)
    (out/"checks.json").write_text(json.dumps(dict(reexecuted_annual_controls=control_replays,
                                                  reused_legacy_accounts=int(old_summary.window.eq("to_latest").sum()),
                                                  accounts=sum(r["window"] == "to_latest" for r in summaries),
                                                  windows=len(summaries), validation_records=len(audits)), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    args = parser.parse_args()
    if args.year:
        run_year(args.year)
        return
    OUT.mkdir(exist_ok=True)
    panel = pd.read_parquet(SOURCE/"quarterly_predictions.parquet")
    dates = sorted(panel.entry_date.unique())
    calendar = global_calendar(panel)
    pd.DataFrame(dict(date=dates, index=range(len(dates)), entry_due=[d in calendar for d in dates])).to_csv(OUT/"global_calendar.csv", index=False)
    protocol = dict(capital=CAPITAL, max_lots=1, startup_modes=MODES, exit_variants=VARIANTS,
                    calendar_anchor=dates[0], calendar_rule="every fifth observed executable session, index mod5=0; never reset at quarter/year/account start",
                    startup_rule="select among original four policies using prior36m historical OOS funded replay at500k; same entry calendar and exit variant",
                    startup_eligibility="existing choose_policy criteria; score by12m-half-life weighted Sortino; tie prefer base; no eligible=>base",
                    startup_information="last validation exit <= previous signal date < actual account start; no current open/settle/future PnL in selection",
                    startup_switch_hurdle="no switch hurdle at initial selection because no incumbent; ordinary guarded switches thereafter unchanged",
                    carry="only chosen rule inherited; equity500k and flat positions at each startup; no import of prior profits or positions",
                    monthly_starts="same58 starts 2022-01..2026-10; five exit variants; 3bps15% and6bps25%",
                    phase_starts="same first5 January sessions2022..2026; original/control,loss_exit,mean5; same two profiles",
                    annual_profiles="same four cost/margin pairs at five January first starts", windows=["single_year", "to_latest", 122, 244],
                    legacy_reuse="all750 old accounts retained;100 annual accounts replayed with previously audited decision schedule",
                    no_parameter_search=True, limitations="previously inspected exploratory history; overlapping starts not independent; daily execution")
    (OUT/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    def worker(year):
        subprocess.run([sys.executable, __file__, "--year", str(year)], check=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(worker, range(2022, 2027)))
    for name in ("summary", "validation_audit", "selection", "startup_decisions", "annual_marked", "trades", "annual_decisions"):
        date_types = {key: str for key in ("start", "end", "date", "cutoff", "entry", "exit", "year",
                                          "first_entry_date", "first_validation_entry", "last_validation_exit",
                                          "last_history_exit", "next_scheduled_entry", "signal_date")}
        pd.concat([pd.read_csv(OUT/str(year)/f"{name}.csv", dtype=date_types, low_memory=False) for year in range(2022, 2027)], ignore_index=True).to_csv(OUT/f"{name}.csv", index=False)
    checks = [json.loads((OUT/str(year)/"checks.json").read_text()) for year in range(2022, 2027)]
    (OUT/"checks.json").write_text(json.dumps({key: sum(c[key] for c in checks) for key in checks[0]}, indent=2))


if __name__ == "__main__":
    main()
