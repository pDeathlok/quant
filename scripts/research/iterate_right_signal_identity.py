"""Predeclared signal-identity ablation; no mutation of the frozen v3 run."""
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
from iterate_right_ranknet_v3 import SOURCE,FOLDS,OUT as V3
import evaluate_local_ranknet_fusion as prior
from evaluate_full_rank_comparison import calendar_for
from evaluate_ranknet_cash_policy import evaluate_policy,pick_policy,THRESHOLDS
from quant.research.local_ranknet import make_pairs,derivatives,percentile_scores
from quant.research.auxiliary_return_loss import LossWeights,terms,derivatives as aux_derivatives
from quant.application.selector_ranking import DEFAULT_SELECTOR_RANKING_CONFIG

OUT=ROOT/'reports/research/right_signal_identity_v4'
ARMS=['signals_regression','signals_local05','signals_aux_absolute']


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((SOURCE/'dataset_complete.json').read_text())
    assert prior.sha256(SOURCE/'dataset_right.parquet')==meta['dataset_hashes']['right']
    frame=pd.read_parquet(SOURCE/'dataset_right.parquet').sort_values(['date','symbol']).reset_index(drop=True)
    signals=list(DEFAULT_SELECTOR_RANKING_CONFIG.supported_strategy_keys)
    assert len(signals)==14 and set(signals)<=set(frame) and frame.source_date.lt(frame.date).all()
    assert (frame[signals].isin([True,False,0,1]) | frame[signals].isna()).all().all()
    missing_signals=frame[signals].isna().sum().astype(int).to_dict()
    # Missing historical flags remain unknown; they must not become false signals.
    frame[signals]=frame[signals].astype('float32')
    prior.save_json(OUT/'protocol.json',dict(source_sha256=meta['dataset_hashes']['right'],signals=signals,folds=FOLDS,
        arms=ARMS,signal_missing_counts=missing_signals,missing_policy='Preserve NaN; never fill unknown flags with false',script_sha256=prior.sha256(Path(__file__)),
        selection='Separate arms and combined-v3 family selected solely by validation; same maturity, cash and capital rules',
        timing='Signal identities originate at screen/source_date strictly before entry/date',
        limitation='Exploratory ablation on previously examined periods; not a new untouched test or an incumbent production backtest'))
    oof=pd.read_parquet(V3/'oof.parquet');results=[];training=[]
    for fold,(fit_end,selection,test_start,test_end) in FOLDS.items():
        contract=json.loads((V3/f'contract_{fold}.json').read_text())
        cols=contract['scopes']['all']+signals
        fit=frame.loc[frame.date.lt(fit_end)&frame.maturity.lt(fit_end)&frame.outcome.notna()].copy()
        local=fit.merge(oof,on=['symbol','date'],validate='one_to_one')
        cal=frame.loc[frame.date.ge(fit_end)&frame.date.lt(selection)&frame.maturity.lt(selection)&frame.outcome.notna()]
        va=prior.evaluation_slice(frame,selection,test_start);te=prior.evaluation_slice(frame,test_start,test_end)
        previous=pd.read_csv(V3/f'grid_{fold}.csv');grid=[];stats={};test_stats={};picks={}
        for row in previous.to_dict('records'):
            stats[row['name']]={k.removeprefix('validation_'):v for k,v in row.items() if k.startswith('validation_')}
            test_stats[row['name']]={k.removeprefix('test_'):v for k,v in row.items() if k.startswith('test_')}
        for arm in ARMS:
            tr=local if arm=='signals_local05' else fit
            dm=prior.matrix(tr,cols);dm.set_label(tr.outcome.clip(-.5,.5))
            weights=(1/tr.groupby('date').date.transform('size')).to_numpy(copy=True);weights/=weights.mean()
            path=OUT/f'{fold}_{arm}.ubj';print('TRAIN',fold,arm,len(tr),len(cols),flush=True)
            params={**prior.PARAMS,'nthread':2}
            if arm=='signals_local05':
                pairs=make_pairs(tr.date,tr.oof_percentile,tr.outcome,bandwidth=.05)
                model=xgb.train({**params,'base_score':0,'disable_default_eval_metric':1},dm,num_boost_round=60,obj=lambda p,d:derivatives(p,pairs))
            elif arm=='signals_aux_absolute':
                targets=terms(tr.outcome.to_numpy(),tr.groupby('date').outcome.transform('mean').to_numpy(),max(float(tr.outcome.std()),.005))
                model=xgb.train({**params,'base_score':0,'disable_default_eval_metric':1},dm,num_boost_round=60,
                    obj=lambda p,d:aux_derivatives(p,targets,LossWeights(absolute=.5,incremental=0,downside=1),weights))
            else:
                dm.set_weight(weights);model=xgb.train({**params,'objective':'reg:squarederror'},dm,num_boost_round=60)
            model.save_model(path)
            raw=[model.predict(prior.matrix(part,cols),output_margin=True) for part in [cal,va,te]]
            calibrator=IsotonicRegression(out_of_bounds='clip').fit(raw[0],cal.outcome.clip(-.5,.5),sample_weight=1/cal.groupby('date').date.transform('size'))
            prior.save_json(OUT/f'{fold}_{arm}_calibration.json',dict(x=calibrator.X_thresholds_.tolist(),y=calibrator.y_thresholds_.tolist()))
            te[['symbol','date']].assign(score=raw[2]).to_parquet(OUT/f'{fold}_{arm}_predictions.parquet',index=False)
            for threshold in THRESHOLDS:
                name=f'{arm}#{threshold}'
                v,_,_=evaluate_policy(va,calibrator.predict(raw[1]),threshold,calendar_for(frame,selection,test_start),1,rank_score=percentile_scores(va.date,raw[1]))
                t,_,p=evaluate_policy(te,calibrator.predict(raw[2]),threshold,calendar_for(frame,test_start,test_end),1,rank_score=percentile_scores(te.date,raw[2]))
                stats[name]=v;test_stats[name]=t;picks[name]=p
                grid.append(dict(name=name,**{f'validation_{k}':value for k,value in v.items()},**{f'test_{k}':value for k,value in t.items()}))
            training.append(dict(fold=fold,arm=arm,rows=len(tr),features=len(cols),used_signals=[s for s in signals if s in model.get_score()],sha256=prior.sha256(path)))
        stats['cash'],_,_=evaluate_policy(va,np.zeros(len(va)),np.inf,calendar_for(frame,selection,test_start),1)
        test_stats['cash'],_,picks['cash']=evaluate_policy(te,np.zeros(len(te)),np.inf,calendar_for(frame,test_start,test_end),1)
        for family in [*ARMS,'combined_v3']:
            names=list(stats) if family=='combined_v3' else [name for name in stats if name.startswith(family+'#')]
            for mode in ['return','win']:
                chosen=pick_policy(stats,names,mode)
                results.append(dict(fold=fold,policy=f'{family}_{mode}',selected=chosen,**test_stats[chosen]))
                if chosen in picks:picks[chosen].to_parquet(OUT/f'picks_{fold}_{family}_{mode}.parquet',index=False)
        pd.concat([previous,pd.DataFrame(grid)],ignore_index=True).to_csv(OUT/f'grid_{fold}.csv',index=False)
        pd.DataFrame(results).to_csv(OUT/'results.csv',index=False)
        prior.save_json(OUT/'training.json',training)
    prior.save_json(OUT/'complete.json',dict(status='complete',models=len(training),promoted=False))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
