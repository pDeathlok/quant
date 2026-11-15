"""Frozen2x2x3 experiment: fixed/rolling × cash permission × reentry rule."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from evaluate_index_futures_principal import (
    OUT as PREVIOUS, PrincipalReplay, FastInner, POLICIES, VARIANTS, PROFILES,
    ALL_PROFILES, CAPITAL, HORIZONS, DATE_TYPES, past_windows, start_specs,
    liquidated_prefix, first_year, metrics, trade_episodes, trade_stats,
    annual_stats, stability, blocked_counts, risk_replay, exit_callback,
    calibrated_callback, equity_scores, gated_select,
)
from refine_index_futures_adaptation import switch_cost

OUT=PREVIOUS.parent/"reentry_cash"
RULES=("none","cooldown5","reset")
CONFIGS={}
for style in ("fixed","rolling"):
    for cash in (False,True):
        for rule in RULES:
            name=f"{style}_{'cash_' if cash else ''}{rule}"
            if style=="rolling" and not cash and rule=="none":name="actual_equity"
            CONFIGS[name]=(style,cash,rule)
NEW_MODES=tuple(m for m in CONFIGS if m!="actual_equity")
REFERENCE=PREVIOUS.parent/"multistart_selection/reference_windows.csv"
EQUITY=PREVIOUS.parent/"equity_advantage_selection"


class ReentryReplay(PrincipalReplay):
    def run(self,groups,fee,margin,variant,policy="base",decision=None,capital=CAPITAL,reentry_rule="none"):
        simple=exit_callback(self.lookup,"mean5",fee) if variant=="mean5" else None
        stateful=calibrated_callback(self.lookup,variant,fee) if variant in ("calibrated5","tail5") else None
        return risk_replay(groups,self.opens,self.closes,fee,margin,policy=policy,capital=capital,
            decision=decision,trace=True,variant=variant,exit_decider=simple,state_exit_decider=stateful,
            entry_calendar=self.calendar,reentry_rule=reentry_rule)


def eligible_policies(scores,style):
    base=scores["base"]
    if not base.get("ready",False):return {}
    names=("base",) if style=="fixed" else POLICIES
    return {p:scores[p] for p in names if scores[p].get("ready",False)
        and scores[p]["forced"]==0 and scores[p]["median"]>0
        and scores[p]["trades"]>=max(2,.5*base["trades"])
        and scores[p]["active"]>=max(10,.5*base["active"])
        and scores[p]["dd90"]<=base["dd90"]+1e-12
        and scores[p]["block"]<=base["block"]+1e-12}


def cash_select(scores,current,style,*,initial=False,rows=None,positions=None,equity=CAPITAL,fee=.0003,margin=.15):
    pool=eligible_policies(scores,style)
    detail=dict(proposed="cash",cash_score=0.,incumbent_score=scores.get(current,{}).get("score",0.),
        candidate_score=np.nan,estimated_switch_cost=np.nan,historical_quarter_edge=np.nan,required_edge=np.nan)
    incumbent_valid=current in pool and pool[current]["score"]>0
    positive={p:v for p,v in pool.items() if v["score"]>0}
    if not positive:return "cash","no_supported_positive_score",detail
    wanted=max(positive,key=lambda p:(positive[p]["score"],p==current,p))
    detail.update(proposed=wanted,candidate_score=positive[wanted]["score"])
    if incumbent_valid:
        if style=="fixed" or wanted==current:return current,"positive_incumbent_keep",detail
        selected,reason,old=gated_select(scores,current,False,initial=False,rows=rows,positions=positions,
            equity=equity,fee=fee,margin=margin)
        detail.update(old)
        return selected,reason,detail
    # Cash-to-risk activation inherits the existing .005 score and currency
    # edge thresholds. No margin/drawdown comparison to zero-risk cash, which
    # would make recovery logically impossible.
    cost=switch_cost(rows,positions or {},equity,fee,margin)
    edge=61*positive[wanted]["daily_pnl"]
    required=max(.005*equity,3*cost) if cost is not None else np.inf
    detail.update(estimated_switch_cost=cost,historical_quarter_edge=edge,required_edge=required)
    if positive[wanted]["score"]<.005:return "cash","cash_activation_score_hurdle",detail
    if cost is None:return "cash","cash_activation_not_affordable",detail
    if edge<required:return "cash","cash_activation_cost_hurdle",detail
    return wanted,"cash_activation_passed",detail


class Selector:
    def __init__(self,replay,out):
        self.fast=FastInner(replay);self.out=out;self.audit=[]
        self.reference=pd.read_csv(REFERENCE,dtype=DATE_TYPES)
        if out is not None:(out/"validation").mkdir(parents=True,exist_ok=True)

    @lru_cache(maxsize=None)
    def template(self,cutoff,fee,margin,variant):
        r=self.reference
        return past_windows(r[r.fee.eq(fee)&r.margin.eq(margin)&r.variant.eq(variant)],cutoff)

    @lru_cache(maxsize=8192)
    def score(self,cutoff,fee,margin,variant,capital,rule):
        vid=hashlib.sha256(json.dumps([cutoff,fee,margin,variant,float(capital).hex(),rule]).encode()).hexdigest()[:20]
        template=self.template(cutoff,fee,margin,variant)
        if capital==CAPITAL and rule=="none":windows=template.copy()
        else:
            records=[]
            for r in template.itertuples(index=False):
                result,_=self.fast.run(r.start,r.horizon,r.policy,variant,fee,margin,capital,reentry_rule=rule)
                records.append(result)
            windows=pd.DataFrame(records)
        assert windows.end.max()<=cutoff
        if self.out is not None:
            windows.assign(validation_capital=capital,cutoff=cutoff,reentry_rule=rule).to_parquet(self.out/"validation"/f"{vid}.parquet",index=False)
        scores={}
        for policy in POLICIES:
            scores[policy]=equity_scores(windows[windows.policy.eq(policy)],cutoff,capital)
            self.audit.append(dict(validation_id=vid,cutoff=cutoff,fee=fee,margin=margin,variant=variant,
                validation_capital=capital,reentry_rule=rule,policy=policy,**scores[policy]))
        return scores,vid

    def choose(self,cutoff,fee,margin,variant,capital,mode,incumbent,*,initial=False,rows=None,positions=None):
        style,cash,rule=CONFIGS[mode]
        scores,vid=self.score(cutoff,fee,margin,variant,float(capital),rule)
        kwargs=dict(initial=initial,rows=rows,positions=positions,equity=capital,fee=fee,margin=margin)
        if cash:selected,reason,detail=cash_select(scores,incumbent,style,**kwargs)
        else:selected,reason,detail=gated_select(scores,incumbent,False,**kwargs)
        return selected,dict(validation_id=vid,validation_capital=capital,reason=reason,**detail)


def validate():
    replay=ReentryReplay();fast=FastInner(replay);ref=pd.read_csv(REFERENCE,dtype=DATE_TYPES)
    cols=["total_return","max_drawdown","trades","active_days","forced","signal_opportunities","blocked_opportunities"]
    for r in ref.itertuples(index=False):
        got,_=fast.run(r.start,r.horizon,r.policy,r.variant,r.fee,r.margin,CAPITAL)
        np.testing.assert_allclose([got[c] for c in cols],[getattr(r,c) for c in cols],rtol=0,atol=1e-9)
    count=0;error=0.
    for capital in (265421.952,500000.,735432.123):
        for start in ("20200102","20230103","20240902"):
            i=fast.index[start]
            for fee,margin in PROFILES:
                for variant in VARIANTS:
                    for policy in POLICIES:
                        for rule in RULES:
                            result,daily=fast.run(i,244,policy,variant,fee,margin,capital,trace=True,reentry_rule=rule)
                            ledger,_,trace=replay.run(replay.groups[i:i+244],fee,margin,variant,policy,capital=capital,reentry_rule=rule)
                            expected=np.column_stack([ledger.pnl,ledger.equity,ledger.gross_lots,ledger.fees,
                                ledger.forced_exit,ledger.settlement_breach,trace.raw_candidates,trace.affordable_candidates])
                            error=max(error,float(abs(daily[:,:8]-expected).max()))
                            np.testing.assert_allclose(daily[:,:8],expected,atol=1e-6,rtol=0)
                            positions=["" if c<0 else f"{fast.codes[int(c)]}:{int(s):+d}" for c,s in daily[:,8:10]]
                            assert positions==ledger.positions.tolist()
                            assert result["trades"]==len(trade_episodes(ledger,replay.opens,fee))
                            count+=1
            print("validated",capital,start,count,flush=True)
    (OUT/"engine_checks.json").write_text(json.dumps(dict(reference_windows=len(ref),daily_cases=count,max_cash_error=error),indent=2))


def window_stats(ledger,trace,events,start,fee,replay):
    windows=[("to_latest",ledger)]
    if start.endswith("0101"):windows.append(("single_year",first_year(ledger,start,replay.opens,fee)))
    windows.extend((f"sessions_{h}",liquidated_prefix(ledger,h,replay.opens,fee)) for h in HORIZONS if len(ledger)>=h)
    for window,sample in windows:
        t=trace.iloc[:len(sample)]
        flat=sample.gross_lots.eq(0)
        run=(flat!=flat.shift()).cumsum()
        longest=int(flat.groupby(run).sum().max())
        stats=metrics(sample,CAPITAL)
        np.testing.assert_allclose(stats["total_return"],sample.pnl.sum()/CAPITAL,atol=1e-9,rtol=0)
        yield dict(window=window,end=sample.exit_date.iloc[-1],**stats,
            **trade_stats(trade_episodes(sample,replay.opens,fee)),**stability(sample),**blocked_counts(t),
            cash_days=int(t.policy.eq("cash").sum()),flat_ratio=float(flat.mean()),longest_flat=longest,
            reentry_blocks=int(t.action.eq("reentry_block").sum()),
            cash_signal_days=int(t.get("cash_signal_candidates",pd.Series(0,index=t.index)).gt(0).sum()),
            cash_funded_days=int(t.get("cash_funded_candidates",pd.Series(0,index=t.index)).gt(0).sum()),
            capital_blocks=int(t.action.eq("capital_block").sum()),
            changes=sum(e["date"]<=sample.entry_date.iloc[-1] for e in events),
            cash_resumptions=sum(e["date"]<=sample.entry_date.iloc[-1] and e["old"]=="cash" and e["new"]!="cash" for e in events),
            cash_pauses=sum(e["date"]<=sample.entry_date.iloc[-1] and e["new"]=="cash" for e in events))


def run_year(year):
    replay=ReentryReplay();out=OUT/str(year)
    for folder in ("ledgers","traces"):(out/folder).mkdir(parents=True,exist_ok=True)
    selector=Selector(replay,out)
    summaries=pd.read_csv(PREVIOUS/str(year)/"summary.csv",dtype=DATE_TYPES).to_dict("records")
    annual=pd.read_csv(PREVIOUS/str(year)/"annual_marked.csv",dtype=DATE_TYPES).to_dict("records")
    trades_all=[pd.read_csv(PREVIOUS/str(year)/"trades.csv",dtype=DATE_TYPES)]
    decisions=[];controls=0
    for spec in start_specs(replay.panel,year):
        start=spec["start"];groups=[g for g in replay.groups if g[0].entry_date>=start]
        profiles=ALL_PROFILES if spec["annual"] else PROFILES
        variants=VARIANTS[:3] if spec["start_kind"]=="phase" else VARIANTS
        for fee,margin in profiles:
            for variant in variants:
                for mode,(style,cash,rule) in CONFIGS.items():
                    if mode=="actual_equity" and not spec["annual"]:continue
                    fields=dict(**spec,fee=fee,margin=margin,variant=variant,startup_mode=mode,first_entry_date=groups[0][0].entry_date)
                    first=groups[0][0]
                    initial="cash" if cash else "base"
                    if cash or style=="rolling":
                        initial,detail=selector.choose(first.trade_date,fee,margin,variant,CAPITAL,mode,initial,
                            initial=True,rows=groups[0],positions={})
                        decisions.append(dict(**fields,stage="startup",date=first.entry_date,cutoff=first.trade_date,
                            equity=CAPITAL,incumbent="cash" if cash else "base",selected=initial,**detail))
                    def decision(day,cutoff,incumbent,equity,positions,rows):
                        selected,detail=selector.choose(cutoff,fee,margin,variant,equity,mode,incumbent,rows=rows,positions=positions)
                        decisions.append(dict(**fields,stage="quarter",date=day,cutoff=cutoff,equity=equity,
                            incumbent=incumbent,selected=selected,**detail))
                        return selected
                    ledger,events,trace=replay.run(groups,fee,margin,variant,initial,
                        decision if cash or style=="rolling" else None,reentry_rule=rule)
                    if mode=="actual_equity":
                        old=pd.read_parquet(EQUITY/str(year)/"ledgers"/f"{start}_{variant}_{mode}_{fee}_{margin}.parquet")
                        pd.testing.assert_frame_equal(ledger,old,check_dtype=False,atol=1e-6,rtol=1e-12)
                        controls+=1;continue
                    assert trace.loc[trace.action.eq("enter"),"date"].isin(replay.calendar).all()
                    trades=trade_episodes(ledger,replay.opens,fee)
                    np.testing.assert_allclose(trades.pnl.sum(),ledger.pnl.sum(),atol=1e-6,rtol=0)
                    trades_all.append(trades.assign(**fields))
                    if spec["annual"]:annual.extend(dict(**fields,**row) for row in annual_stats(ledger,trades))
                    summaries.extend(dict(**fields,**row) for row in window_stats(ledger,trace,events,start,fee,replay))
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
    protocol=dict(capital=CAPITAL,max_lots=1,end="20261106",configurations=CONFIGS,
        design="fixed base rule vs quarterly reselect x cash candidate off/on x no reentry/cooldown5/reset;12 cells including existing actual_equity",
        fixed="base entry rule permanently fixed; existing quarterly-trained prediction panel unchanged, not refitted in this study",
        cooldown="after every executed exit (profit/loss/model/margin/strategy), block same product+side for next5sessions; earliest6th if on global calendar; other products/opposite side unaffected",
        reset="same product+side barred until one fully observed post-exit close has no qualifying BASE signal; missing data does not reset; then normal policy threshold applies; rolls and quarter switches do not erase memory",
        cash="cash score0, no interest; startup and quarterly review; all candidate histories exact current equity and same reentry overlay, existing coverage/funding/DD eligibility",
        activation="eligible score>=.005 and61*historical dailycashPNL>=max(.005currentequity,3*estimated switchcost), currently affordable; same old hurdles",
        deactivation="incumbent loses eligibility or score<=0 and no candidate passes activation =>cash; if incumbent eligible withscore>0 retain unless old guardedswitch passes; no permanent stop",
        history="past36m completed122/244d monthly windows,12m half-life,flat reference starts at exact current equity; inner fixed candidates implement reentry, outer cash selector is not recursively run inside candidates",
        memory="outer account reentry memory persists through quarters, cash episodes and years; independent historical windows initialize flat with no prior episode",
        calendar="fixed20200102 every5sessions, exits daily, cash decisions only startup/quarter",
        frozen="all5exit variants/control models,labels,3/6bps,15/25%margin, same58monthly/Jan5day/annual4profiles; no new risk stop, no topup, no parameter search",
        measurement="yearly single-year/continuous/to_latest/equal122/244; flat duration, cash resumption, zero trade fraction; matched baseline daily positivePNL missed and negativePNL avoided while new account flat, not an executable cherry-picked return",
        controls="retain all9750 previous accounts;100 annual actual_equity controls reselect causally and replay; old11840 references and new daily parity",
        limits="overlapping historical starts not independent, extensively inspected history not newOOT; daily fill and margin scenarios only")
    (OUT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    (OUT/"global_calendar.csv").write_bytes((PREVIOUS/"global_calendar.csv").read_bytes())


def main():
    p=argparse.ArgumentParser();p.add_argument("--year",type=int);p.add_argument("--validate-only",action="store_true")
    args=p.parse_args()
    if args.year:run_year(args.year);return
    register();validate()
    if args.validate_only:return
    def worker(year):subprocess.run([sys.executable,__file__,"--year",str(year)],check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,range(2022,2027)))
    for name in ("summary","annual_marked","trades","selection","selection_audit"):
        pd.concat([pd.read_csv(OUT/str(y)/f"{name}.csv",dtype=DATE_TYPES,low_memory=False) for y in range(2022,2027)],ignore_index=True).to_csv(OUT/f"{name}.csv",index=False)


if __name__=="__main__":main()
