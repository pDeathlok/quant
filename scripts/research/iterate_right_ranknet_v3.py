"""Right-side time-ordered data expansion, RankNet, auxiliary and context ablations."""
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
from evaluate_full_rank_comparison import calendar_for
from evaluate_ranknet_cash_policy import evaluate_policy,pick_policy,THRESHOLDS
from quant.research.local_ranknet import make_pairs,derivatives,percentile_scores
from quant.research.auxiliary_return_loss import LossWeights,terms,derivatives as aux_derivatives

SOURCE=ROOT/'reports/research/full_rank_comparison_v2'
OUT=ROOT/'reports/research/right_ranknet_iteration_v3'
FOLDS={'2026':('2025-10-01','2025-11-01','2026-01-01','2026-09-04'),
       '2026H2':('2026-01-01','2026-04-01','2026-07-01','2026-09-04')}
ARMS=['core129_more','regression_oldfit','regression_more','regression_same_oof','remaining_more',
      'ranknet_more','local05_more','hybrid_more','aux_absolute_more','aux_increment_more',
      'context_regression_more','context_ranknet_more']


def context_features(frame: pd.DataFrame) -> list[str]:
    added=[]
    for numerator,denominator in [('return_120d','volatility_60d'),('selector_return_60d','selector_volatility_60d')]:
        name=f'context__{numerator}_risk_scaled';frame[name]=frame[numerator]/frame[denominator].where(frame[denominator].abs().gt(1e-6));added.append(name)
    for stock in ['return_120d','volume_relative_20d','turnover_rate_ma20']:
        for market in ['market_return_13w','market_sentiment_5d']:
            name=f'context__{stock}_x_{market}';frame[name]=frame[stock]*frame[market];added.append(name)
    frame[added]=frame[added].replace([np.inf,-np.inf],np.nan).astype('float32')
    return added


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((SOURCE/'dataset_complete.json').read_text())
    if prior.sha256(SOURCE/'dataset_right.parquet')!=meta['dataset_hashes']['right']:raise ValueError('Source changed')
    frame=pd.read_parquet(SOURCE/'dataset_right.parquet').sort_values(['date','symbol']).reset_index(drop=True)
    added=context_features(frame)
    base=json.loads((SOURCE/'contract_right_2026.json').read_text())['scopes']['base']
    parameters={**prior.PARAMS,'nthread':2}
    prior.save_json(OUT/'protocol.json',dict(folds=FOLDS,arms=ARMS,new_columns=added,params=parameters,
        source_sha256=meta['dataset_hashes']['right'],selection='Calibration and policy selection precede test, actual-label maturity purge, all-cash fallback',
        extra_data='Fit through2025Q3 instead of2025H1; second walk-forward fit through2025Q4. Nonlocal methods additionally use2024Q1 mature rows omitted by prior OOF-only experiment.',
        novelty='Explicit risk-scaled stock returns and stock×market interactions; all source variables are admitted causal columns.',
        limitations='Historical windows have been researched before. Unknown exit sleeve freeze and-10% stress retained. No test-based model promotion.',
        script_sha256=prior.sha256(Path(__file__))))
    # Additional chronological OOF quarters for the local-pair experiment.
    oof=pd.read_parquet(SOURCE/'oof_right.parquet');pieces=[oof]
    for start,end in [('2025-07-01','2025-10-01'),('2025-10-01','2026-01-01')]:
        tr=frame.loc[frame.date.lt(start)&frame.maturity.lt(start)&frame.outcome.notna()]
        te=frame.loc[frame.date.ge(start)&frame.date.lt(end)]
        path=OUT/f'oof_{start}.ubj'
        dm=prior.matrix(tr,base);dm.set_label(tr.outcome.clip(-.5,.5));weight=1/tr.groupby('date').date.transform('size');dm.set_weight(weight/weight.mean())
        if path.exists():model=xgb.Booster();model.load_model(path)
        else:model=xgb.train({**parameters,'objective':'reg:squarederror'},dm,num_boost_round=60);model.save_model(path)
        pred=model.predict(prior.matrix(te,base))
        pieces.append(te[['symbol','date']].assign(oof_score=pred,oof_percentile=percentile_scores(te.date,pred)))
    oof=pd.concat(pieces,ignore_index=True)
    if oof.duplicated(['symbol','date']).any():raise ValueError('OOF overlap')
    oof.to_parquet(OUT/'oof.parquet',index=False)
    all_results=[];training=[]
    for fold,(fit_end,selection,test_start,test_end) in FOLDS.items():
        eligible=frame.loc[frame.date.lt(fit_end)&frame.maturity.lt(fit_end)&frame.outcome.notna()].copy()
        local=eligible.merge(oof,on=['symbol','date'],validate='one_to_one')
        old=eligible.loc[eligible.date.lt('2025-07-01')&eligible.maturity.lt('2025-07-01')]
        cal=frame.loc[frame.date.ge(fit_end)&frame.date.lt(selection)&frame.maturity.lt(selection)&frame.outcome.notna()]
        va=prior.evaluation_slice(frame,selection,test_start);te=prior.evaluation_slice(frame,test_start,test_end)
        full=[c for c in meta['columns'] if eligible[c].nunique()>1]
        scopes={'core':meta['core'],'all':full,'remaining':[c for c in full if c not in base],
                'context':full+[c for c in added if eligible[c].nunique()>1]}
        prior.save_json(OUT/f'contract_{fold}.json',dict(fit_end=fit_end,max_label_date=str(eligible.maturity.max()),
            calibration_end=selection,max_calibration_label_date=str(cal.maturity.max()),test_start=test_start,test_end=test_end,
            rows=dict(more=len(eligible),oldfit=len(old),oof=len(local),calibration=len(cal),validation=len(va),test=len(te)),scopes=scopes))
        statistics={};test_statistics={};picked={};predictions={};grid=[]
        for arm in ARMS:
            tr=old if arm=='regression_oldfit' else local if arm in ['local05_more','regression_same_oof'] else eligible
            scope='core' if arm=='core129_more' else 'remaining' if arm=='remaining_more' else 'context' if arm.startswith('context_') else 'all'
            cols=scopes[scope];path=OUT/f'{fold}_{arm}.ubj'
            dm=prior.matrix(tr,cols);dm.set_label(tr.outcome.clip(-.5,.5))
            w=(1/tr.groupby('date').date.transform('size')).to_numpy(copy=True);w/=w.mean()
            print('TRAIN',fold,arm,len(tr),len(cols),flush=True)
            if path.exists():model=xgb.Booster();model.load_model(path)
            elif 'ranknet' in arm or arm.startswith(('local05','hybrid')):
                pair_scores=tr.oof_percentile.to_numpy() if arm.startswith('local05') else np.zeros(len(tr))
                pairs=make_pairs(tr.date,pair_scores,tr.outcome,bandwidth=.05 if arm.startswith('local05') else None)
                target=tr.outcome.to_numpy()/max(float(tr.outcome.std()),.005)
                def objective(pred,dm):
                    g,h=derivatives(pred,pairs)
                    if arm.startswith('hybrid'):
                        g=g+.1*w*(pred-target.clip(-3,3));h=h+.1*w
                    return g,h
                model=xgb.train({**parameters,'base_score':0,'disable_default_eval_metric':1},dm,num_boost_round=60,obj=objective)
            elif arm.startswith('aux_'):
                targets=terms(tr.outcome.to_numpy(),tr.groupby('date').outcome.transform('mean').to_numpy(),max(float(tr.outcome.std()),.005))
                weights=LossWeights(absolute=.5,incremental=.25 if 'increment' in arm else 0,downside=1)
                model=xgb.train({**parameters,'base_score':0,'disable_default_eval_metric':1},dm,num_boost_round=60,
                    obj=lambda pred,dm:aux_derivatives(pred,targets,weights,w))
            else:
                dm.set_weight(w);model=xgb.train({**parameters,'objective':'reg:squarederror'},dm,num_boost_round=60)
            model.save_model(path)
            if model.feature_names!=cols:raise ValueError('Cached model input drift')
            scores=[model.predict(prior.matrix(part,cols),output_margin=True) for part in [cal,va,te]]
            calibration=IsotonicRegression(out_of_bounds='clip').fit(scores[0],cal.outcome.clip(-.5,.5),sample_weight=1/cal.groupby('date').date.transform('size'))
            gates=[calibration.predict(s) for s in scores[1:]]
            ranks=[percentile_scores(part.date,s) for part,s in zip([va,te],scores[1:])]
            predictions[arm]=scores[2]
            prior.save_json(OUT/f'calibration_{fold}_{arm}.json',dict(x=calibration.X_thresholds_.tolist(),y=calibration.y_thresholds_.tolist()))
            for threshold in THRESHOLDS:
                name=f'{arm}#{threshold}'
                v,_,_=evaluate_policy(va,gates[0],threshold,calendar_for(frame,selection,test_start),1,rank_score=ranks[0])
                t,_,p=evaluate_policy(te,gates[1],threshold,calendar_for(frame,test_start,test_end),1,rank_score=ranks[1])
                statistics[name]=v;test_statistics[name]=t;picked[name]=p
                grid.append(dict(name=name,**{f'validation_{k}':v for k,v in v.items()},**{f'test_{k}':v for k,v in t.items()}))
            training.append(dict(fold=fold,arm=arm,rows=len(tr),features=len(cols),used=len(model.get_score())))
            pd.DataFrame(training).to_csv(OUT/'training.csv',index=False)
        statistics['cash'],_,_=evaluate_policy(va,np.zeros(len(va)),np.inf,calendar_for(frame,selection,test_start),1)
        test_statistics['cash'],_,picked['cash']=evaluate_policy(te,np.zeros(len(te)),np.inf,calendar_for(frame,test_start,test_end),1)
        groups={a:[n for n in statistics if n.startswith(a+'#')] for a in ARMS};groups['overall']=list(statistics)
        for family,names in groups.items():
            for mode in ['return','win']:
                chosen=pick_policy(statistics,names,mode);p=picked[chosen];filled=p.loc[~p.unfilled]
                all_results.append(dict(fold=fold,policy=family+'_'+mode,selected=chosen,**test_statistics[chosen],filled=len(filled),
                    filled_win=float(filled.outcome.gt(0).mean()) if len(filled) else np.nan))
                p.to_parquet(OUT/f'picks_{fold}_{family}_{mode}.parquet',index=False)
        pd.DataFrame(grid).to_csv(OUT/f'grid_{fold}.csv',index=False)
        te[['symbol','date','outcome','maturity']].assign(**predictions).to_parquet(OUT/f'predictions_{fold}.parquet',index=False)
        pd.DataFrame(all_results).to_csv(OUT/'results.csv',index=False)
        print('FINISHED',fold,flush=True)
    prior.save_json(OUT/'complete.json',dict(status='complete',models=len(training),promoted=False))


if __name__=='__main__':main()
