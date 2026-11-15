"""Across-date RankNet and calibrated entry thresholds, including all-cash fallback.

Uses independent rotating cash sleeves: left 6 overlapping sleeves, right 1;
20 slots per sleeve. A sleeve reinvests its own settled cash after its previous
cohort exits. Terminal capital return is compounded within sleeves; daily
diagnostic contributions use fixed initial notional, not marked-to-market NAV.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
import evaluate_local_ranknet_fusion as prior
from quant.research.local_ranknet import cross_date_pairs, derivatives
from evaluate_auxiliary_return_losses import intervals
from validate_left_side_disclosure_gain import sha256

OUT=ROOT/'reports/research/ranknet_cash_policy_v1'
THRESHOLDS=(-np.inf,0.,.002,.005,.01,.02,np.inf)
ALPHAS=(.25,.5,.75,1.)
CROSS_BANDS=(.0025,.005,.01)


def evaluate_policy(frame,score,threshold,calendar,sleeves,rank_score=None):
    candidates=frame[['symbol','date','outcome']].assign(score=score)
    candidates['rank_score']=np.asarray(score if rank_score is None else rank_score)
    candidates['maturity']=frame.maturity.to_numpy() if 'maturity' in frame else (frame.date+pd.Timedelta(days=1)).to_numpy()
    candidates['unfilled']=(frame.locked_limit_up.to_numpy() if 'locked_limit_up' in frame
                            else frame.at_upper.eq(True).to_numpy())
    picks=(candidates.loc[candidates.score.ge(threshold)]
           .sort_values(['date','rank_score','symbol'],ascending=[True,False,True])
           .groupby('date',sort=False).head(20).copy())
    picks['stress']=picks.outcome.fillna(-.10)
    picks['win']=picks.outcome.gt(0).astype(float)
    picks['filled']=(~picks.unfilled).astype(int)
    market_dates=pd.DatetimeIndex(calendar)
    busy=[pd.Timestamp.min]*sleeves
    wealth=np.ones(sleeves)
    active_slots=np.zeros(len(market_dates))
    accepted=[]
    blocked_attempts=0
    for date,group in picks.groupby('date',sort=True):
        i=market_dates.get_loc(date); sleeve=i%sleeves
        entry=market_dates[min(i+1,len(market_dates)-1)] if sleeves>1 else pd.Timestamp(date)
        if busy[sleeve]>entry:
            blocked_attempts+=len(group)
            continue
        accepted.append(group)
        # Reuse the sleeve only AFTER all selected old positions exit. Delayed
        # exits therefore block later entries; they never generate free capital.
        held=group.loc[~group.unfilled]
        if len(held):
            busy[sleeve]=pd.Timestamp(held.maturity.max())
            if held.outcome.isna().any():
                # No evidence of liquidation: retain the sleeve until the end
                # of this evaluation period, rather than invent sale proceeds.
                busy[sleeve]=max(busy[sleeve],market_dates[-1]+pd.Timedelta(days=1))
        wealth[sleeve]*=1+group.stress.sum()/20
        for row in held.itertuples():
            exit_date=(market_dates[-1]+pd.Timedelta(days=1)) if pd.isna(row.outcome) else row.maturity
            active_slots+=(market_dates>=entry)&(market_dates<exit_date)
    picks=pd.concat(accepted,ignore_index=True) if accepted else picks.iloc[:0]
    day=picks.groupby('date').agg(pnl=('stress','sum'),n=('symbol','size'),filled=('filled','sum'),wins=('win','sum'),unknown=('outcome',lambda x:int(x.isna().sum())))
    day=day.reindex(pd.DatetimeIndex(calendar,name='date'),fill_value=0)
    day['pnl']=day.pnl/(20*sleeves)
    day['active_slots']=active_slots
    # Unused slots remain cash; never redistribute them to surviving picks.
    total=int(day.n.sum())
    stats=dict(capital_pnl=float(wealth.mean()-1),fixed_notional_pnl=float(day.pnl.sum()),mean_daily_pnl=float(day.pnl.mean()),
               win_rate=float(day.wins.sum()/total) if total else np.nan,
               attempts=total,unknown=int(day.unknown.sum()),
               no_entry_days=float(day.n.eq(0).mean()),slot_utilization=float(day.n.mean()/20),
               flat_days=float(day.active_slots.eq(0).mean()),
               average_capital_deployed=float(day.active_slots.mean()/(20*sleeves)),
               delayed_capital_blocked_attempts=blocked_attempts,
               avg_net_per_attempt=float(picks.stress.mean()) if total else np.nan)
    return stats,day,picks


def calendar_for(frame,start,end):
    # Evaluate all market dates, including days with no candidates, only within
    # the available source period. No end-of-calendar rows are fabricated.
    cal=pd.to_datetime(pd.read_csv(ROOT/'data/research/left_side_disclosure_gain_20261029/calendar.csv').date)
    return cal.loc[cal.ge(start)&cal.lt(end)&cal.le(frame.date.max())].to_numpy()


def pick_policy(stats,names,mode):
    eligible=[n for n in names if stats[n]['capital_pnl']>0 and stats[n]['attempts']>=100]
    if mode=='win': eligible=[n for n in eligible if stats[n]['win_rate']>=.55]
    if not eligible: return 'cash'
    return max(eligible,key=lambda n:(stats[n]['win_rate'],stats[n]['capital_pnl']) if mode=='win' else (stats[n]['capital_pnl'],stats[n]['win_rate']))


def main():
    if not (prior.OUT/'complete.json').exists():
        raise RuntimeError('Complete corrected within-day experiment first')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'complete.json').unlink(missing_ok=True)
    prior.save_json(OUT/'protocol.json',dict(cross_bands=CROSS_BANDS,alphas=ALPHAS,thresholds=[str(t) for t in THRESHOLDS],
        grouping='Cross-date ONLY pairs with chronological OOF base expected-return difference <= bandwidth; absolute net outcomes, fixed horizon per strategy',
        calibration='First half of validation, labels mature strictly before selection start; isotonic maps each raw score to expected net return; no test fit',
        selection='Second half of validation; maximize fixed-capital stressed P&L or win with positive P&L and >=100 attempts; cash fallback',
        capital='20 slots per daily sleeve; left 6 rotating independent sleeves, right 1; each reinvests settled own capital only; unused slots cash; terminal wealth compounds per sleeve, daily contribution CI uses fixed initial notional, no MTM drawdown',
        costs='Existing 20bp round trip; unknown -10% terminal stress mark, not guaranteed bound; unresolved holdings freeze their whole sleeve through period end; no live execution',
        source_sha256=sha256(Path(__file__)),prior_protocol_sha256=sha256(prior.OUT/'protocol.json')))
    results=[];grids=[]
    for side,frame,_ in prior.datasets():
        folds=({'2024':('2023-01-01','2023-07-01','2024-01-01','2025-01-01'),
                '2025':('2024-01-01','2024-07-01','2025-01-01','2026-01-01'),
                '2026':('2025-01-01','2025-07-01','2026-01-01','2026-11-11')} if side=='left' else
               {'2025H2':('2025-01-01','2025-04-01','2025-07-01','2026-01-01'),
                '2026':('2025-07-01','2025-10-01','2026-01-01','2026-09-04')})
        for fold,(fit_end,select_start,test_start,test_end) in folds.items():
            print('CASH',side,fold,flush=True)
            contract=json.loads((prior.OUT/f'features_{side}_{fold}.json').read_text())
            first=contract['base50'];remaining=contract['remaining'];allcols=first+remaining
            oof=pd.read_parquet(prior.OUT/f'oof_{side}_{fold}.parquet')
            sub=oof.merge(frame,on=prior.KEYS,how='left',validate='one_to_one',suffixes=('_oof',''))
            assert sub.maturity.lt(fit_end).all()
            cal=frame.loc[frame.date.ge(fit_end)&frame.date.lt(select_start)&frame.maturity.lt(select_start)].copy()
            va=prior.evaluation_slice(frame,select_start,test_start)
            te=prior.evaluation_slice(frame,test_start,test_end)
            vcal=calendar_for(frame,select_start,test_start);tcal=calendar_for(frame,test_start,test_end)
            sleeves=6 if side=='left' else 1
            models={}
            for name in ['base50','single_all']+[f'{g}_{s}' for g in prior.BANDS for s in ['remaining','all']]:
                m=xgb.Booster();m.load_model(prior.OUT/f'model_{side}_{fold}_{name}.ubj')
                # DMatrix order follows the model's actual stored feature order.
                models[name]=m
            # A 120-tree single model controls for the 60+60 tree fusion budget.
            fit=frame.loc[prior.mature_before(frame,fit_end)]
            d=prior.matrix(fit,allcols);d.set_label(fit.outcome.clip(-.5,.5))
            w=(1/fit.groupby('date').date.transform('size')).to_numpy(copy=True);d.set_weight(w/w.mean())
            capacity=xgb.train({**prior.PARAMS,'objective':'reg:squarederror'},d,num_boost_round=2*prior.ROUNDS)
            capacity.save_model(OUT/f'model_{side}_{fold}_single_all120.ubj');models['single_all120']=capacity
            for band in CROSS_BANDS:
                pairs=cross_date_pairs(sub.date,sub.oof_score,sub.outcome,band)
                for scope,features in [('remaining',remaining),('all',allcols)]:
                    name=f'cross{band}_{scope}'
                    print(side,fold,name,len(pairs.left),'pairs',flush=True)
                    d=prior.matrix(sub,features);d.set_label(sub.outcome)
                    model=xgb.train({**prior.PARAMS,'base_score':0.,'disable_default_eval_metric':1},d,num_boost_round=prior.ROUNDS,obj=lambda pred,dm:derivatives(pred,pairs))
                    model.save_model(OUT/f'model_{side}_{fold}_{name}.ubj');models[name]=model
            calibrated={};calibrators={}
            for name,model in models.items():
                cols=model.feature_names
                raw=[model.predict(prior.matrix(d,cols),output_margin=True) for d in [cal,va,te]]
                known=cal.outcome.notna().to_numpy()
                weight=(1/cal.groupby('date').date.transform('size')).to_numpy()
                iso=IsotonicRegression(out_of_bounds='clip')
                iso.fit(raw[0][known],cal.outcome.to_numpy()[known].clip(-.5,.5),sample_weight=weight[known])
                calibrated[name]=(iso.predict(raw[1]),iso.predict(raw[2]))
                calibrators[name]={'x':iso.X_thresholds_.tolist(),'y':iso.y_thresholds_.tolist()}
            prior.save_json(OUT/f'calibrators_{side}_{fold}.json',calibrators)
            vbase,tbase=calibrated['base50'];blends={'base50':(vbase,tbase),'single_all':calibrated['single_all'],'single_all120':calibrated['single_all120']}
            for name,(v,t) in calibrated.items():
                if name in ['base50','single_all','single_all120']:continue
                for alpha in ALPHAS:
                    blends[f'{name}@{alpha}']=((1-alpha)*vbase+alpha*v,(1-alpha)*tbase+alpha*t)
            vsumm={};tsumm={};tdays={};picks={};names=[]
            for blend,(v,t) in blends.items():
                for threshold in THRESHOLDS:
                    name=f'{blend}#{threshold}'
                    vsumm[name],_,_=evaluate_policy(va,v,threshold,vcal,sleeves)
                    tsumm[name],tdays[name],picks[name]=evaluate_policy(te,t,threshold,tcal,sleeves)
                    names.append(name)
                    grids.append(dict(side=side,fold=fold,arm=name,**{f'validation_{k}':v for k,v in vsumm[name].items()},**{f'test_{k}':v for k,v in tsumm[name].items()}))
            vsumm['cash'],_,_=evaluate_policy(va,vbase,np.inf,vcal,sleeves)
            tsumm['cash'],tdays['cash'],picks['cash']=evaluate_policy(te,tbase,np.inf,tcal,sleeves)
            groups={'any':names,'base':[n for n in names if n.startswith('base50#')],
                    'day':[n for n in names if n.startswith('day_')],
                    'local':[n for n in names if n.startswith('local')],
                    'cross':[n for n in names if n.startswith('cross')],
                    'single_all':[n for n in names if n.startswith('single_all#')],
                    'capacity':[n for n in names if n.startswith('single_all120#')]}
            choices={f'{g}_{mode}':pick_policy(vsumm,ns,mode) for g,ns in groups.items() for mode in ['return','win']}
            choices['forced_base']='base50#-inf'
            prior.save_json(OUT/f'choices_{side}_{fold}.json',choices)
            for policy,arm in choices.items():
                basearm=choices['base_return'];diff=(tdays[arm].pnl-tdays[basearm].pnl).to_numpy();ci=intervals(diff)
                results.append(dict(side=side,fold=fold,policy=policy,arm=arm,**tsumm[arm],
                    vs_base_cash_capital_pp=float((tsumm[arm]['capital_pnl']-tsumm[basearm]['capital_pnl'])*100),
                    daily_delta_ci_low_bps=ci[0]*10000,daily_delta_ci_high_bps=ci[1]*10000))
                tdays[arm].to_csv(OUT/f'daily_{side}_{fold}_{policy}.csv')
                picks[arm].to_parquet(OUT/f'picks_{side}_{fold}_{policy}.parquet',index=False)
            pd.DataFrame(results).to_csv(OUT/'comparisons.csv',index=False)
            pd.DataFrame(grids).to_csv(OUT/'validation_grid.csv',index=False)
    prior.save_json(OUT/'complete.json',{'status':'complete','folds':5,'production_promoted':False})


if __name__=='__main__': main()
