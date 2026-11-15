"""Fixed and rolling panic-feature ablations with original execution constraints."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from evaluate_index_futures_state_models import ModelReplay as OldReplay
from evaluate_index_futures_regime import (FastInner, POLICIES, PROFILES, ALL_PROFILES, CAPITAL,
    DATE_TYPES, past_windows, start_specs, window_stats, trade_episodes, annual_stats,
    risk_replay, exit_callback, equity_scores, gated_select, REFERENCE)
from index_futures_panic import OUT, PREVIOUS, ARMS, EVIDENCE_NAMES
from report_index_futures_reentry_cash import prefix, attribution

CONFIGS = {f"{style}_{arm}":(style,arm) for style in ("fixed","rolling") for arm in ARMS}


class PanicReplay(OldReplay):
    def __init__(self):
        super().__init__()
        self.evidence_names = EVIDENCE_NAMES
        frame = pd.read_parquet(OUT/"forecasts.parquet")
        for r in frame.itertuples(index=False):
            self.evidence_lookup[r.entry_date,r.symbol,r.direction,r.bps,r.model] = (r.probability,r.pred_net,r.pred_risk)

    def run(self, groups, fee, margin, variant="mean5", policy="base", decision=None, capital=CAPITAL, arm="baseline"):
        evidence,soft,hard = ARMS[arm]
        return risk_replay(groups,self.opens,self.closes,fee,margin,policy=policy,capital=capital,
            decision=decision,trace=True,variant=variant,exit_decider=exit_callback(self.lookup,"mean5",fee),
            entry_calendar=self.calendar,market_regime=self.market_regime,regime_mode=hard,
            continuation_lookup=self.lookup,evidence_model=evidence,evidence_lookup=self.evidence_lookup,
            soft_exit=soft,soft_lookup=self.soft_lookup,evidence_names=self.evidence_names)


class Selector:
    def __init__(self,replay,out):
        self.fast,self.out,self.audit = FastInner(replay),out,[]
        self.reference = pd.read_csv(REFERENCE,dtype=DATE_TYPES)
        if out is not None: (out/"validation").mkdir(parents=True,exist_ok=True)

    @lru_cache(maxsize=None)
    def template(self,cutoff,fee,margin):
        r = self.reference
        return past_windows(r[r.fee.eq(fee)&r.margin.eq(margin)&r.variant.eq("mean5")],cutoff)

    @lru_cache(maxsize=8192)
    def score(self,cutoff,fee,margin,capital,arm):
        vid = hashlib.sha256(json.dumps([cutoff,fee,margin,float(capital).hex(),arm]).encode()).hexdigest()[:20]
        evidence,soft,hard = ARMS[arm]
        windows = pd.DataFrame([self.fast.run(r.start,r.horizon,r.policy,"mean5",fee,margin,capital,
            evidence_model=evidence,soft_exit=soft,regime_mode=hard)[0]
            for r in self.template(cutoff,fee,margin).itertuples(index=False)])
        assert windows.end.max()<=cutoff
        if self.out is not None:
            windows.assign(validation_capital=capital,cutoff=cutoff,arm=arm).to_parquet(self.out/"validation"/f"{vid}.parquet",index=False)
        scores = {}
        for policy in POLICIES:
            scores[policy] = equity_scores(windows[windows.policy.eq(policy)],cutoff,capital)
            self.audit.append(dict(validation_id=vid,cutoff=cutoff,fee=fee,margin=margin,variant="mean5",
                validation_capital=capital,arm=arm,policy=policy,**scores[policy]))
        return scores,vid

    def choose(self,cutoff,fee,margin,capital,arm,current,initial=False,rows=None,positions=None):
        scores,vid = self.score(cutoff,fee,margin,float(capital),arm)
        selected,reason,detail = gated_select(scores,current,False,initial=initial,
            rows=rows,positions=positions,equity=capital,fee=fee,margin=margin)
        return selected,dict(validation_id=vid,validation_capital=capital,reason=reason,**detail)


def validate():
    replay = PanicReplay();fast = FastInner(replay)
    cases,error = 0,0.
    for capital in (265421.952,500000.,735432.123):
        for start in ("20230103","20240902","20260105"):
            i = fast.index[start]
            for fee,margin in PROFILES:
                for policy in ("base","opposing_exit"):
                    for arm,(evidence,soft,hard) in ARMS.items():
                        result,daily = fast.run(i,122,policy,"mean5",fee,margin,capital,trace=True,evidence_model=evidence,soft_exit=soft,regime_mode=hard)
                        ledger,_,trace = replay.run(replay.groups[i:i+122],fee,margin,policy=policy,capital=capital,arm=arm)
                        expected = np.column_stack([ledger.pnl,ledger.equity,ledger.gross_lots,ledger.fees,
                            ledger.forced_exit,ledger.settlement_breach,trace.raw_candidates,trace.affordable_candidates])
                        error = max(error,float(abs(daily[:,:8]-expected).max()))
                        np.testing.assert_allclose(daily[:,:8],expected,rtol=0,atol=1e-6)
                        positions = ["" if c<0 else f"{fast.codes[int(c)]}:{int(s):+d}" for c,s in daily[:,8:10]]
                        assert positions==ledger.positions.tolist()
                        assert result["trades"]==len(trade_episodes(ledger,replay.opens,fee))
                        cases += 1
    (OUT/"engine_checks.json").write_text(json.dumps(dict(daily_cases=cases,max_cash_error=error),indent=2))
    print("parity",cases,error,flush=True)


def run_year(year):
    replay = PanicReplay();out = OUT/str(year)
    for folder in ("ledgers","traces"): (out/folder).mkdir(parents=True,exist_ok=True)
    selector = Selector(replay,out)
    summaries,annual,trades_all,decisions,opportunities = [],[],[],[],[]
    controls = 0
    for spec in start_specs(replay.panel,year):
        start = spec["start"]
        groups = [g for g in replay.groups if g[0].entry_date>=start]
        for fee,margin in (ALL_PROFILES if spec["annual"] else PROFILES):
            baselines = {}
            for mode,(style,arm) in CONFIGS.items():
                fields = dict(**spec,fee=fee,margin=margin,variant="mean5",startup_mode=mode,arm=arm,first_entry_date=groups[0][0].entry_date)
                first = groups[0][0];initial = "base"
                if style=="rolling":
                    initial,detail = selector.choose(first.trade_date,fee,margin,CAPITAL,arm,"base",initial=True,rows=groups[0],positions={})
                    decisions.append(dict(**fields,stage="startup",date=first.entry_date,cutoff=first.trade_date,equity=CAPITAL,incumbent="base",selected=initial,**detail))
                def decision(day,cutoff,incumbent,equity,positions,rows):
                    selected,detail = selector.choose(cutoff,fee,margin,equity,arm,incumbent,rows=rows,positions=positions)
                    decisions.append(dict(**fields,stage="quarter",date=day,cutoff=cutoff,equity=equity,incumbent=incumbent,selected=selected,**detail))
                    return selected
                ledger,events,trace = replay.run(groups,fee,margin,policy=initial,decision=decision if style=="rolling" else None,arm=arm)
                if arm=="baseline":
                    old = pd.read_parquet(PREVIOUS/str(year)/"ledgers"/f"{start}_mean5_{style}_state_horizon_{fee}_{margin}.parquet")
                    pd.testing.assert_frame_equal(ledger,old,check_dtype=False,atol=1e-6,rtol=1e-12)
                    baselines[style] = ledger
                    controls += 1
                assert trace.loc[trace.action.eq("enter"),"date"].isin(replay.calendar).all()
                assert ledger.gross_lots.le(1).all()
                trades = trade_episodes(ledger,replay.opens,fee)
                np.testing.assert_allclose(trades.pnl.sum(),ledger.pnl.sum(),rtol=0,atol=1e-6)
                trades_all.append(trades.assign(**fields))
                if spec["annual"]:
                    annual.extend(dict(**fields,**r) for r in annual_stats(ledger,trades))
                for stats in window_stats(ledger,trace,events,start,fee,replay):
                    summaries.append(dict(**fields,**stats))
                    if arm!="baseline":
                        a = prefix(baselines[style],stats["window"],start,fee,replay.opens)
                        b = prefix(ledger,stats["window"],start,fee,replay.opens)
                        opportunities.append(dict(**fields,window=stats["window"],**attribution(a,b)))
                filename = f"{start}_mean5_{mode}_{fee}_{margin}.parquet"
                ledger.to_parquet(out/"ledgers"/filename,index=False)
                trace.to_parquet(out/"traces"/filename,index=False)
        print("panic evaluated",year,start,selector.score.cache_info(),flush=True)
    for name,records in (("summary",summaries),("annual_marked",annual),("selection",decisions),("selection_audit",selector.audit),("opportunity",opportunities)):
        pd.DataFrame(records).to_csv(out/f"{name}.csv",index=False)
    pd.concat(trades_all,ignore_index=True).to_csv(out/"trades.csv",index=False)
    (out/"checks.json").write_text(json.dumps(dict(replayed_controls=controls)))


def main():
    p = argparse.ArgumentParser();p.add_argument("--year",type=int);p.add_argument("--validate-only",action="store_true");p.add_argument("--skip-validation",action="store_true")
    args = p.parse_args()
    if args.year: run_year(args.year);return
    if not args.skip_validation: validate()
    if args.validate_only: return
    def worker(year): subprocess.run([sys.executable,__file__,"--year",str(year)],check=True)
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(worker,range(2022,2027)))
    for name in ("summary","annual_marked","trades","selection","selection_audit","opportunity"):
        pd.concat([pd.read_csv(OUT/str(y)/f"{name}.csv",dtype=DATE_TYPES,low_memory=False) for y in range(2022,2027)],ignore_index=True).to_csv(OUT/f"{name}.csv",index=False)


if __name__=="__main__": main()
