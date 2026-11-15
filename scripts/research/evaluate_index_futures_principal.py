"""Predeclared loss prevention overlays with exact-equity inner validation."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from evaluate_index_futures_equity_gate import (
    OUT as PREVIOUS, EquityReplay, FastInner, equity_scores, gated_select,
    POLICIES, VARIANTS, PROFILES, ALL_PROFILES, CAPITAL, HORIZONS, DATE_TYPES,
    past_windows, start_specs, liquidated_prefix, first_year, metrics,
    trade_episodes, trade_stats, annual_stats, stability, blocked_counts,
    risk_replay, exit_callback, calibrated_callback,
)

OUT = PREVIOUS.parent / "principal_protection"
OVERLAYS = ("entry5", "stop5", "fuse20", "combined")
NEW_MODES = tuple("risk_"+v for v in OVERLAYS)
FLOOR = .8*CAPITAL


class PrincipalReplay(EquityReplay):
    def run(self, groups, fee, margin, variant, policy="base", decision=None,
            capital=CAPITAL, risk_overlay="none", principal_floor=FLOOR):
        simple = exit_callback(self.lookup, "mean5", fee) if variant == "mean5" else None
        stateful = calibrated_callback(self.lookup, variant, fee) if variant in ("calibrated5", "tail5") else None
        return risk_replay(groups, self.opens, self.closes, fee, margin, policy=policy, capital=capital,
            decision=decision, trace=True, variant=variant, exit_decider=simple,
            state_exit_decider=stateful, entry_calendar=self.calendar,
            risk_overlay=risk_overlay, principal_floor=principal_floor)


class HistoricalSelector:
    """Exact capital and fixed original-principal floor enter every inner replay."""
    def __init__(self, replay, out):
        self.replay, self.fast, self.out = replay, FastInner(replay), out
        self.reference = pd.read_csv(PREVIOUS.parent/"multistart_selection/reference_windows.csv", dtype=DATE_TYPES)
        self.audit = []
        if out is not None:
            (out/"validation").mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=None)
    def template(self, cutoff, fee, margin, variant):
        r = self.reference
        return past_windows(r[r.fee.eq(fee)&r.margin.eq(margin)&r.variant.eq(variant)], cutoff)

    @lru_cache(maxsize=4096)
    def score(self, cutoff, fee, margin, variant, capital, overlay):
        key = [cutoff, fee, margin, variant, float(capital).hex(), overlay, FLOOR]
        validation_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
        rows = []
        for r in self.template(cutoff, fee, margin, variant).itertuples(index=False):
            result, _ = self.fast.run(r.start, r.horizon, r.policy, variant, fee, margin, capital,
                risk_overlay=overlay, principal_floor=FLOOR)
            rows.append(result)
        windows = pd.DataFrame(rows)
        assert windows.end.max() <= cutoff
        if self.out is not None:
            windows.assign(validation_capital=capital, cutoff=cutoff, overlay=overlay,
                principal_floor=FLOOR).to_parquet(self.out/"validation"/f"{validation_id}.parquet", index=False)
        scores = {}
        for policy in POLICIES:
            scores[policy] = equity_scores(windows[windows.policy.eq(policy)], cutoff, capital)
            self.audit.append(dict(validation_id=validation_id, cutoff=cutoff, fee=fee, margin=margin,
                variant=variant, validation_capital=capital, overlay=overlay, policy=policy, **scores[policy]))
        return scores, validation_id

    def choose(self, cutoff, fee, margin, variant, capital, overlay, incumbent,
               initial=False, gated=False, rows=None, positions=None):
        scores, vid = self.score(cutoff, fee, margin, variant, float(capital), overlay)
        selected, reason, detail = gated_select(scores, incumbent, gated, initial=initial,
            rows=rows, positions=positions, equity=capital, fee=fee, margin=margin)
        return selected, dict(validation_id=vid, validation_capital=capital, reason=reason, **detail)


def validate():
    replay = PrincipalReplay()
    fast = FastInner(replay)
    ref = pd.read_csv(PREVIOUS.parent/"multistart_selection/reference_windows.csv", dtype=DATE_TYPES)
    cols = ["total_return", "max_drawdown", "trades", "active_days", "forced", "signal_opportunities", "blocked_opportunities"]
    for r in ref.itertuples(index=False):
        got, _ = fast.run(r.start, r.horizon, r.policy, r.variant, r.fee, r.margin, CAPITAL)
        np.testing.assert_allclose([got[c] for c in cols], [getattr(r,c) for c in cols], rtol=0, atol=1e-9)
    n, error = 0, 0.
    for capital in (350000., 500000., 735432.123):
        for start in ("20230103", "20240902"):
            i = fast.index[start]
            for fee, margin in PROFILES:
                for variant in VARIANTS:
                    for policy in POLICIES:
                        for overlay in OVERLAYS:
                            result, daily = fast.run(i, 244, policy, variant, fee, margin, capital, trace=True,
                                risk_overlay=overlay, principal_floor=FLOOR)
                            ledger, _, trace = replay.run(replay.groups[i:i+244], fee, margin, variant, policy,
                                capital=capital, risk_overlay=overlay)
                            expected = np.column_stack([ledger.pnl,ledger.equity,ledger.gross_lots,ledger.fees,
                                ledger.forced_exit,ledger.settlement_breach,trace.raw_candidates,trace.affordable_candidates])
                            error=max(error,float(abs(daily[:,:8]-expected).max()))
                            np.testing.assert_allclose(daily[:,:8],expected,atol=1e-6,rtol=0)
                            positions=["" if c<0 else f"{fast.codes[int(c)]}:{int(s):+d}" for c,s in daily[:,8:10]]
                            assert positions==ledger.positions.tolist()
                            assert result["trades"]==len(trade_episodes(ledger,replay.opens,fee))
                            n+=1
            print("validated",capital,start,n,flush=True)
    (OUT/"engine_checks.json").write_text(json.dumps(dict(reference_windows=len(ref),new_daily_cases=n,max_cash_error=error),indent=2))


def run_year(year):
    replay=PrincipalReplay()
    out=OUT/str(year)
    (out/"ledgers").mkdir(parents=True,exist_ok=True)
    (out/"traces").mkdir(exist_ok=True)
    selector=HistoricalSelector(replay,out)
    summaries=pd.read_csv(PREVIOUS/str(year)/"summary.csv",dtype=DATE_TYPES).to_dict("records")
    annual=pd.read_csv(PREVIOUS/str(year)/"annual_marked.csv",dtype=DATE_TYPES).to_dict("records")
    trades_all=[pd.read_csv(PREVIOUS/str(year)/"trades.csv",dtype=DATE_TYPES)]
    decisions=[]
    controls=0
    old_decisions=pd.read_csv(PREVIOUS/str(year)/"selection.csv",dtype=DATE_TYPES)
    for spec in start_specs(replay.panel,year):
        start=spec["start"]
        groups=[g for g in replay.groups if g[0].entry_date>=start]
        profiles=ALL_PROFILES if spec["annual"] else PROFILES
        variants=VARIANTS[:3] if spec["start_kind"]=="phase" else VARIANTS
        for fee,margin in profiles:
            for variant in variants:
                if spec["annual"]:
                    d=old_decisions[old_decisions.start.eq(start)&old_decisions.fee.eq(fee)&old_decisions.margin.eq(margin)&old_decisions.variant.eq(variant)&old_decisions.startup_mode.eq("actual_equity")]
                    initial=d.loc[d.stage.eq("startup"),"selected"].item()
                    schedule=d[d.stage.eq("quarter")].set_index("date").selected.to_dict()
                    ledger,_,_=replay.run(groups,fee,margin,variant,initial,
                        lambda day,*args:schedule.get(day,args[1]))
                    old=pd.read_parquet(PREVIOUS/str(year)/"ledgers"/f"{start}_{variant}_actual_equity_{fee}_{margin}.parquet")
                    pd.testing.assert_frame_equal(ledger,old,check_dtype=False,atol=1e-6,rtol=1e-12)
                    controls+=1
                for overlay,mode in zip(OVERLAYS,NEW_MODES):
                    fields=dict(**spec,fee=fee,margin=margin,variant=variant,startup_mode=mode,first_entry_date=groups[0][0].entry_date)
                    first=groups[0][0]
                    initial,detail=selector.choose(first.trade_date,fee,margin,variant,CAPITAL,overlay,"base",initial=True)
                    decisions.append(dict(**fields,stage="startup",date=first.entry_date,cutoff=first.trade_date,
                        equity=CAPITAL,incumbent="base",selected=initial,**detail))
                    def decision(day,cutoff,incumbent,equity,positions,rows):
                        selected,detail=selector.choose(cutoff,fee,margin,variant,equity,overlay,incumbent,rows=rows,positions=positions)
                        decisions.append(dict(**fields,stage="quarter",date=day,cutoff=cutoff,equity=equity,incumbent=incumbent,selected=selected,**detail))
                        return selected
                    ledger,events,trace=replay.run(groups,fee,margin,variant,initial,decision,risk_overlay=overlay)
                    assert trace.loc[trace.action.eq("enter"),"date"].isin(replay.calendar).all()
                    trades=trade_episodes(ledger,replay.opens,fee)
                    np.testing.assert_allclose(trades.pnl.sum() if len(trades) else 0,ledger.pnl.sum(),atol=1e-6,rtol=0)
                    trades_all.append(trades.assign(**fields))
                    if spec["annual"]:
                        annual.extend(dict(**fields,**r) for r in annual_stats(ledger,trades))
                    windows=[("to_latest",ledger)]
                    if spec["annual"]:
                        windows.append(("single_year",first_year(ledger,start,replay.opens,fee)))
                    windows.extend((f"sessions_{h}",liquidated_prefix(ledger,h,replay.opens,fee)) for h in HORIZONS if len(ledger)>=h)
                    for window,sample in windows:
                        t=trace.iloc[:len(sample)]
                        halted=t[t.action.eq("principal_halt")]
                        summaries.append(dict(**fields,window=window,end=sample.exit_date.iloc[-1],**metrics(sample,CAPITAL),
                            **trade_stats(trade_episodes(sample,replay.opens,fee)),**stability(sample),**blocked_counts(t),
                            risk_blocks=int(t.action.eq("risk_block").sum()),budget_exits=int(t.action.eq("budget_loss_exit").sum()),
                            halt_days=len(halted),halt_date=halted.date.iloc[0] if len(halted) else "",
                            capital_blocks=int(t.action.eq("capital_block").sum()),changes=sum(e["date"]<=sample.entry_date.iloc[-1] for e in events)))
                    filename=f"{start}_{variant}_{mode}_{fee}_{margin}.parquet"
                    ledger.to_parquet(out/"ledgers"/filename,index=False)
                    trace.to_parquet(out/"traces"/filename,index=False)
        print("evaluated",year,start,selector.score.cache_info(),flush=True)
    for name,rows in (("summary",summaries),("annual_marked",annual),("selection",decisions),("selection_audit",selector.audit)):
        pd.DataFrame(rows).to_csv(out/f"{name}.csv",index=False)
    pd.concat(trades_all,ignore_index=True).to_csv(out/"trades.csv",index=False)
    (out/"checks.json").write_text(json.dumps(dict(replayed_controls=controls,validations=selector.score.cache_info().misses)))


def register():
    OUT.mkdir(exist_ok=True)
    protocol=dict(capital=CAPITAL,max_lots=1,end="20261106",new_modes=NEW_MODES,
        comparison="all6750 previous accounts retained; new overlays on actual_equity selector without optional dual-horizon gate",
        entry5="2 * past20d daily volatility * one-lot current notional <=5% current equity, both sides; choose highest confidence among admissible",
        stop5="net trade loss at previous close >=5% entry equity: close next open; composes with all original exits; includes paid fees and estimated close fee",
        fuse20="previous-close net liquidatable equity <=400000: close next open and permanently remain cash within this account; no annual/quarterly reset",
        combined="entry5+stop5+fuse20",principal_floor="absolute400000 in every historical validation too; current equity is exact, no resetting floor to80% current equity",
        parameters="frozen before results;5% inherited risk scenario;20% new round-number scenario, not optimized or user-approved risk tolerance",
        historical="same past36m complete122/244d monthly histories, maturity cutoff,12m weights, same policy candidates and funding/switch gates; each overlay simulated identically inside and outside",
        evaluation="same monthly, January5day and annual4profile grid, single_year,to_latest,equal122/244; halted duration and trade coverage disclosed",
        no_topups="main comparisons start500k, no external funds; supplementation separate cash-accounting scenarios, no deposit counted as profit",
        limitation="exploratory reused history, overlapping starts not independent; daily next-open stops are not guaranteed loss caps; margin15/25%, fee3/6bps scenarios")
    (OUT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    (OUT/"global_calendar.csv").write_bytes((PREVIOUS/"global_calendar.csv").read_bytes())


def repair_metrics():
    """Regenerate summary-only marks from immutable cash ledgers, no reselection."""
    replay=PrincipalReplay()
    for year in range(2022,2027):
        path=OUT/str(year)/"summary.csv"
        frame=pd.read_csv(path,dtype=DATE_TYPES)
        for key,part in frame[frame.startup_mode.isin(NEW_MODES)].groupby(["start","variant","startup_mode","fee","margin"]):
            start,variant,mode,fee,margin=key
            ledger=pd.read_parquet(OUT/str(year)/"ledgers"/f"{start}_{variant}_{mode}_{fee}_{margin}.parquet")
            for i,row in part.iterrows():
                sample=(ledger if row.window=="to_latest" else first_year(ledger,start,replay.opens,fee)
                    if row.window=="single_year" else liquidated_prefix(ledger,int(row.window.split("_")[1]),replay.opens,fee))
                stats=metrics(sample,CAPITAL)
                np.testing.assert_allclose(stats["total_return"],sample.pnl.sum()/CAPITAL,atol=1e-9,rtol=0)
                for name,value in stats.items():frame.loc[i,name]=value
        frame.to_csv(path,index=False)
    pd.concat([pd.read_csv(OUT/str(y)/"summary.csv",dtype=DATE_TYPES) for y in range(2022,2027)],ignore_index=True).to_csv(OUT/"summary.csv",index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument("--year",type=int);p.add_argument("--validate-only",action="store_true");p.add_argument("--repair-metrics",action="store_true")
    args=p.parse_args()
    if args.repair_metrics:
        repair_metrics();return
    if args.year:
        run_year(args.year);return
    register();validate()
    if args.validate_only:return
    def worker(y):subprocess.run([sys.executable,__file__,"--year",str(y)],check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,range(2022,2027)))
    for name in ("summary","annual_marked","trades","selection","selection_audit"):
        pd.concat([pd.read_csv(OUT/str(y)/f"{name}.csv",dtype=DATE_TYPES,low_memory=False) for y in range(2022,2027)],ignore_index=True).to_csv(OUT/f"{name}.csv",index=False)


if __name__=="__main__":main()
