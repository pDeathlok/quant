"""One prespecified net-return metric adaptation after the NDCG experiment."""
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
from evaluate_head_lambdaloss import OUT as HEAD,fold_bounds,grouped_training
from evaluate_ranknet_cash_policy import evaluate_policy,calendar_for,pick_policy,THRESHOLDS
from quant.research.head_lambdaloss import head_metrics
from quant.research.local_ranknet import percentile_scores
from quant.research.return_lambdas import ReturnHeadObjective

OUT=HEAD/'return_metric'


def main():
    OUT.mkdir(exist_ok=True)
    prior.save_json(OUT/'protocol.json',dict(status='exploratory_extension_after_observed_ndcg_diagnostics',
        loss='absolute swap delta of equal-slot Top20 mean net return, returns clipped to +/-20%; query normalized; no within-head swap penalty',
        limitation='Same already-observed historical periods; no new independent confirmation',rounds=60,params=prior.PARAMS,alphas=[.5,1.],thresholds=[str(t) for t in THRESHOLDS]))
    heads=[];results=[]
    for side,frame,_ in prior.datasets():
        for fold,(fit_end,select_start,test_start,test_end) in fold_bounds(side).items():
            print('NET-LAMBDA',side,fold,flush=True)
            contract=json.loads((HEAD/f'contract_{side}_{fold}.json').read_text());cols=contract['columns'];cuts=contract['cuts']
            sub=pd.read_parquet(prior.OUT/f'oof_{side}_{fold}.parquet').merge(frame,on=prior.KEYS,validate='one_to_one',suffixes=('_oof',''))
            assert sub.maturity.lt(fit_end).all()
            sub,q=grouped_training(sub,'day')
            d=prior.matrix(sub,cols);d.set_label(sub.outcome)
            m=xgb.train({**prior.PARAMS,'base_score':0.,'disable_default_eval_metric':1},d,num_boost_round=60,obj=ReturnHeadObjective(q,sub.outcome))
            m.save_model(OUT/f'model_{side}_{fold}.ubj')
            cal=frame.loc[frame.date.ge(fit_end)&frame.date.lt(select_start)&frame.maturity.lt(select_start)&frame.outcome.notna()]
            va=prior.evaluation_slice(frame,select_start,test_start);te=prior.evaluation_slice(frame,test_start,test_end)
            raw=[m.predict(prior.matrix(f,cols),output_margin=True) for f in [cal,va,te]]
            iso=IsotonicRegression(out_of_bounds='clip').fit(raw[0],cal.outcome.clip(-.5,.5),sample_weight=1/cal.groupby('date').date.transform('size'))
            gate=(iso.predict(raw[1]),iso.predict(raw[2]));rank=(percentile_scores(va.date,raw[1]),percentile_scores(te.date,raw[2]))
            base=xgb.Booster();base.load_model(prior.OUT/f'model_{side}_{fold}_base50.ubj')
            base_raw=[base.predict(prior.matrix(f,base.feature_names)) for f in [va,te]]
            bc=json.loads((HEAD/f'calibrators_{side}_{fold}.json').read_text())['base50']
            bg=[np.interp(x,bc['x'],bc['y']) for x in base_raw]
            br=[percentile_scores(f.date,x) for f,x in zip([va,te],base_raw)]
            vcal=calendar_for(frame,select_start,test_start);tcal=calendar_for(frame,test_start,test_end);sleeves=6 if side=='left' else 1
            vs={};ts={};picks={};days={}
            for alpha in [.5,1.]:
                v,t=[(1-alpha)*b+alpha*s for b,s in zip(bg,gate)]
                vr,tr=[(1-alpha)*b+alpha*s for b,s in zip(br,rank)]
                for threshold in THRESHOLDS:
                    arm=f'net20@{alpha}#{threshold}'
                    vs[arm],_,_=evaluate_policy(va,v,threshold,vcal,sleeves,rank_score=vr)
                    ts[arm],days[arm],picks[arm]=evaluate_policy(te,t,threshold,tcal,sleeves,rank_score=tr)
            names=list(vs)
            vs['cash'],_,_=evaluate_policy(va,np.zeros(len(va)),np.inf,vcal,sleeves)
            ts['cash'],days['cash'],picks['cash']=evaluate_policy(te,np.zeros(len(te)),np.inf,tcal,sleeves)
            choices={mode:pick_policy(vs,names,mode) for mode in ['return','win']}
            prior.save_json(OUT/f'choices_{side}_{fold}.json',dict(choices=choices,calibration=dict(x=iso.X_thresholds_.tolist(),y=iso.y_thresholds_.tolist()),validation=vs))
            for policy,arm in choices.items():
                results.append(dict(side=side,fold=fold,policy=policy,arm=arm,**ts[arm]))
                picks[arm].to_parquet(OUT/f'picks_{side}_{fold}_{policy}.parquet',index=False)
                days[arm].to_csv(OUT/f'daily_{side}_{fold}_{policy}.csv')
            h=head_metrics(te,raw[2],cuts);h.to_csv(OUT/f'heads_{side}_{fold}.csv',index=False)
            for k,g in h.groupby('k'):heads.append(dict(side=side,fold=fold,k=k,ndcg=g.ndcg.mean(),stress_net=g.stress_net.mean(),win=g.win.mean(),unknown=g.unknown.sum()))
            pd.DataFrame(heads).to_csv(OUT/'head_metrics.csv',index=False);pd.DataFrame(results).to_csv(OUT/'cash_comparisons.csv',index=False)
    prior.save_json(OUT/'complete.json',dict(status='complete',model_runs=5,production_promoted=False))


if __name__=='__main__':main()
