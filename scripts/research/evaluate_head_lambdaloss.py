"""LambdaRank / NDCG-Loss2 / NDCG-Loss2++ on frozen causal research cohorts."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
import evaluate_local_ranknet_fusion as prior
from evaluate_ranknet_cash_policy import evaluate_policy,calendar_for,pick_policy,THRESHOLDS
from evaluate_auxiliary_return_losses import intervals
from quant.research.local_ranknet import make_pairs,derivatives,percentile_scores
from quant.research.head_lambdaloss import HeadLambdaObjective,fit_grade_cuts,gains,head_metrics
from validate_left_side_disclosure_gain import sha256

OUT=ROOT/'reports/research/head_lambdaloss_v1'
ARMS={'ranknet_full':('ranknet_full',20,'all','day'),
      'ranknet_head20':('ranknet_head',20,'all','day'),
      **{f'lambdarank{k}':('lambdarank',k,'all','day') for k in [5,10,20]},
      'lambdaloss2_20':('lambdaloss2',20,'all','day'),
      **{f'lambdaloss2pp{k}':('lambdaloss2pp',k,'all','day') for k in [5,10,20]},
      'lambdaloss2pp20_remaining':('lambdaloss2pp',20,'remaining','day'),
      'cross_ranknet20':('ranknet_head',20,'all','cross'),
      'cross_lambdaloss2pp20':('lambdaloss2pp',20,'all','cross')}
ALPHAS=(.5,1.)


def fold_bounds(side):
    return ({'2024':('2023-01-01','2023-07-01','2024-01-01','2025-01-01'),
             '2025':('2024-01-01','2024-07-01','2025-01-01','2026-01-01'),
             '2026':('2025-01-01','2025-07-01','2026-01-01','2026-11-11')} if side=='left' else
            {'2025H2':('2025-01-01','2025-04-01','2025-07-01','2026-01-01'),
             '2026':('2025-07-01','2025-10-01','2026-01-01','2026-09-04')})


def grouped_training(sub,kind):
    d=sub.copy()
    if kind=='day':d['_query']=d.date.astype(str)
    else:
        # A supplementary cross-date experiment: month x OOF return-score bin.
        # Groups are not defined by labels or current in-sample fitted scores.
        band=np.floor(d.oof_score.to_numpy()/.005).astype(int)
        d['_query']=d.date.dt.to_period('M').astype(str)+'|'+pd.Series(band,index=d.index).astype(str)
    d=d.sort_values(['_query','date','symbol']).reset_index(drop=True)
    q=pd.factorize(d._query,sort=False)[0]
    return d,q


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'complete.json').unlink(missing_ok=True)
    source_protocol=json.loads((prior.OUT/'protocol.json').read_text())
    for name,digest in source_protocol['input_hashes'].items():
        if sha256(ROOT/name)!=digest:raise RuntimeError(f'Frozen source changed: {name}')
    prior.save_json(OUT/'protocol.json',dict(arms=ARMS,params=prior.PARAMS,rounds=prior.ROUNDS,
        tail_samples=32,mu=5,alphas=ALPHAS,thresholds=[str(t) for t in THRESHOLDS],
        relevance='nonpositive/unknown return grade0; positive grades1..4 split by FIT-only positive-return quantiles50/80/95; gain=2^grade-1',
        weights='LambdaRank exact DeltaNDCG@k; Loss2 paper Eq15/18; Loss2++ full-rank rho+5delta Eq19 and head mask; equal normalized query weight',
        pairs='All current head-head pairs,32 uniformly sampled tail positions per head with inverse inclusion weights; rerank every boosting round; not all-pair exact paper reproduction',
        cross='month x 0.005-wide chronological OOF base expected-return bin; different-date pairs only; supplementary grouping, not the earlier exact-neighbourhood algorithm',
        temporal='Actual label-end purge; frozen features/base/OOF; separate calibration, selection, reused historical test; no production promotion',
        evaluation='Top1/5/10/20 raw and blended head metrics; max20 positions; iso net-return gate separate from raw-rank/percentile fusion; same cash/unknown-exit freeze evaluator for all controls',
        sources=['https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf','https://marc.najork.org/papers/cikm2018.pdf'],
        input_hashes=source_protocol['input_hashes'],
        code_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/quant/research/head_lambdaloss.py',ROOT/'scripts/research/evaluate_ranknet_cash_policy.py']}))
    summaries=[];heads=[];cash_rows=[];comparison=[];choices_all=[];times=[]
    for side,frame,_ in prior.datasets():
        if '--right-only' in sys.argv and side!='right':continue
        for fold,(fit_end,select_start,test_start,test_end) in fold_bounds(side).items():
            print('FOLD',side,fold,flush=True)
            contract=json.loads((prior.OUT/f'features_{side}_{fold}.json').read_text())
            first=contract['base50'];rest=contract['remaining'];cols=first+rest
            oof_path=prior.OUT/f'oof_{side}_{fold}.parquet'
            sub=pd.read_parquet(oof_path).merge(frame,on=prior.KEYS,validate='one_to_one',suffixes=('_oof',''))
            assert sub.outcome.notna().all() and sub.maturity.lt(fit_end).all()
            assert (sub.source_date.le(sub.date) if side=='left' else sub.source_date.lt(sub.date)).all()
            cuts=fit_grade_cuts(sub.outcome)
            cal=frame.loc[frame.date.ge(fit_end)&frame.date.lt(select_start)&frame.maturity.lt(select_start)&frame.outcome.notna()].copy()
            va=prior.evaluation_slice(frame,select_start,test_start)
            te=prior.evaluation_slice(frame,test_start,test_end)
            vcal=calendar_for(frame,select_start,test_start);tcal=calendar_for(frame,test_start,test_end)
            sleeves=6 if side=='left' else 1
            prior.save_json(OUT/f'contract_{side}_{fold}.json',dict(cuts=cuts.tolist(),train_rows=len(sub),calibration_rows=len(cal),selection_rows=len(va),test_rows=len(te),
                fit_end=fit_end,selection_start=select_start,test_start=test_start,test_end=test_end,actual_test_start=str(te.date.min()),actual_test_end=str(te.date.max()),
                train_label_max=str(sub.maturity.max()),cal_label_max=str(cal.maturity.max()),oof_sha256=sha256(oof_path),columns=cols))
            group_cache={kind:grouped_training(sub,kind) for kind in ['day','cross']}
            models={}
            for name,root,filearm in [('base50',prior.OUT,'base50'),('regression60',prior.OUT,'single_all'),('regression120',ROOT/'reports/research/ranknet_cash_policy_v1','single_all120')]:
                m=xgb.Booster();m.load_model(root/f'model_{side}_{fold}_{filearm}.ubj');models[name]=m
            for arm,(method,k,scope,kind) in ARMS.items():
                train,q=group_cache[kind];features=cols if scope=='all' else rest
                label=gains(train.outcome,cuts)
                d=prior.matrix(train,features);d.set_label(label)
                if method=='ranknet_full':
                    pairs=make_pairs(q,train.oof_score.to_numpy(),label)
                    objective=lambda z,dm:derivatives(z,pairs)
                else:
                    objective=HeadLambdaObjective(q,label,method,k=k,dates=train.date,cross_only=kind=='cross')
                stamp=time.monotonic()
                print('TRAIN',side,fold,arm,len(train),flush=True)
                model=xgb.train({**prior.PARAMS,'base_score':0.,'disable_default_eval_metric':1},d,num_boost_round=prior.ROUNDS,obj=objective)
                model.save_model(OUT/f'model_{side}_{fold}_{arm}.ubj');models[arm]=model
                times.append(dict(side=side,fold=fold,arm=arm,seconds=time.monotonic()-stamp,
                                  final_pairs=getattr(objective,'last_pairs',len(pairs.left) if method=='ranknet_full' else 0)))
                pd.DataFrame(times).to_csv(OUT/'training_times.csv',index=False)
            calibrated={};ranks={};raw={};head_days={};calibrators={}
            for name,model in models.items():
                values=[model.predict(prior.matrix(d,model.feature_names),output_margin=True) for d in [cal,va,te]]
                w=(1/cal.groupby('date').date.transform('size')).to_numpy()
                iso=IsotonicRegression(out_of_bounds='clip').fit(values[0],cal.outcome.to_numpy().clip(-.5,.5),sample_weight=w)
                calibrated[name]=(iso.predict(values[1]),iso.predict(values[2]))
                ranks[name]=(percentile_scores(va.date,values[1]),percentile_scores(te.date,values[2]))
                raw[name]=values[2]
                calibrators[name]=dict(x=iso.X_thresholds_.tolist(),y=iso.y_thresholds_.tolist(),
                                      test_outside_calibration=float(((values[2]<iso.X_thresholds_[0])|(values[2]>iso.X_thresholds_[-1])).mean()))
                for stage,fr,s in [('validation',va,values[1]),('test',te,values[2])]:
                    h=head_metrics(fr,s,cuts);h.to_csv(OUT/f'heads_{side}_{fold}_{name}_{stage}.csv',index=False)
                    for k,g in h.groupby('k'):
                        heads.append(dict(side=side,fold=fold,arm=name,stage=stage,k=k,ndcg=g.ndcg.mean(),stress_net=g.stress_net.mean(),win=g.win.mean(),unknown=g.unknown.sum(),dates=len(g),informative_dates=g.ndcg.notna().sum()))
                    if stage=='test':head_days[name]=h
            prior.save_json(OUT/f'calibrators_{side}_{fold}.json',calibrators)
            # Freeze a head-ranking choice using validation NDCG@20, before reporting tests.
            vh=pd.DataFrame(heads);vh=vh.loc[vh.side.eq(side)&vh.fold.eq(fold)&vh.stage.eq('validation')&vh.k.eq(20)]
            head_choices={family:vh.loc[vh.arm.isin(names)].sort_values(['ndcg','arm'],ascending=[False,True]).arm.iloc[0]
                          for family,names in {'ranknet':['ranknet_full','ranknet_head20'],
                                               'lambdarank':[a for a in ARMS if a.startswith('lambdarank')],
                                               'lambdaloss':[a for a in ARMS if a.startswith('lambdaloss') and 'remaining' not in a],
                                               'cross':['cross_ranknet20','cross_lambdaloss2pp20']}.items()}
            for family,name in head_choices.items():
                for k in [1,5,10,20]:
                    a=head_days[name].query('k==@k').set_index('date');b=head_days['ranknet_full'].query('k==@k').set_index('date')
                    diff=(a.stress_net-b.stress_net).to_numpy();ci=intervals(diff)
                    comparison.append(dict(side=side,fold=fold,family=family,selected=name,k=k,ndcg=a.ndcg.mean(),ndcg_delta=a.ndcg.mean()-b.ndcg.mean(),
                        net_delta_bps=diff.mean()*10000,win_delta_pp=(a.win-b.win).mean()*100,net_ci_low_bps=ci[0]*10000,net_ci_high_bps=ci[1]*10000))
            blends={n:(*calibrated[n],*ranks[n]) for n in ['base50','regression60','regression120']}
            for arm in ARMS:
                for alpha in ALPHAS:
                    blends[f'{arm}@{alpha}']=tuple((1-alpha)*b+alpha*a for b,a in zip((*calibrated['base50'],*ranks['base50']),(*calibrated[arm],*ranks[arm])))
            vst={};tst={};tdays={};tselected={};names=[]
            for blend,(v,t,vr,tr) in blends.items():
                for threshold in THRESHOLDS:
                    name=f'{blend}#{threshold}'
                    vst[name],_,_=evaluate_policy(va,v,threshold,vcal,sleeves,rank_score=vr)
                    tst[name],tdays[name],tselected[name]=evaluate_policy(te,t,threshold,tcal,sleeves,rank_score=tr)
                    names.append(name)
                    cash_rows.append(dict(side=side,fold=fold,arm=name,**{f'validation_{k}':v for k,v in vst[name].items()},**{f'test_{k}':v for k,v in tst[name].items()}))
            vst['cash'],_,_=evaluate_policy(va,np.zeros(len(va)),np.inf,vcal,sleeves)
            tst['cash'],tdays['cash'],tselected['cash']=evaluate_policy(te,np.zeros(len(te)),np.inf,tcal,sleeves)
            families={'base':[n for n in names if n.startswith('base50#')],
                      'capacity':[n for n in names if n.startswith('regression120#')],
                      'ranknet':[n for n in names if n.startswith('ranknet_')],
                      'lambdarank':[n for n in names if n.startswith('lambdarank')],
                      'lambdaloss':[n for n in names if n.startswith('lambdaloss') and 'remaining' not in n],
                      'remaining':[n for n in names if 'remaining' in n],
                      'cross':[n for n in names if n.startswith('cross_')],
                      'overall':names}
            choices={f'{f}_{mode}':pick_policy(vst,ns,mode) for f,ns in families.items() for mode in ['return','win']}
            prior.save_json(OUT/f'choices_{side}_{fold}.json',dict(head=head_choices,cash=choices))
            for policy,arm in choices.items():
                ref=choices['ranknet_return'];ci=intervals((tdays[arm].pnl-tdays[ref].pnl).to_numpy())
                summaries.append(dict(side=side,fold=fold,policy=policy,arm=arm,**tst[arm],vs_ranknet_capital_pp=(tst[arm]['capital_pnl']-tst[ref]['capital_pnl'])*100,
                                      daily_delta_ci_low_bps=ci[0]*10000,daily_delta_ci_high_bps=ci[1]*10000))
                tdays[arm].to_csv(OUT/f'daily_{side}_{fold}_{policy}.csv')
                tselected[arm].to_parquet(OUT/f'picks_{side}_{fold}_{policy}.parquet',index=False)
            # Raw predictions retained for independent rescoring / head audits.
            te[prior.KEYS+['source_date','maturity','outcome']].assign(**raw).to_parquet(OUT/f'predictions_{side}_{fold}.parquet',index=False)
            pd.DataFrame(summaries).to_csv(OUT/'cash_comparisons.csv',index=False)
            pd.DataFrame(heads).to_csv(OUT/'head_metrics.csv',index=False)
            pd.DataFrame(comparison).to_csv(OUT/'head_comparisons.csv',index=False)
            pd.DataFrame(cash_rows).to_csv(OUT/'validation_grid.csv',index=False)
    prior.save_json(OUT/'complete.json',dict(status='complete',model_runs=len(times),production_promoted=False))


if __name__=='__main__':main()
