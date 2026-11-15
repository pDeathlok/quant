"""Train real auxiliary losses; select weights strictly before each test period."""
from __future__ import annotations
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT), str(ROOT/'scripts/research')]
from evaluate_joinquant_candidates import cohorts
from quant.features.joinquant_candidates import PRICE, FINANCE
from quant.research.auxiliary_return_loss import LossWeights, terms, derivatives, daily_picks
from validate_left_side_disclosure_gain import sha256
from train_right_t012_full_registry import FOLDS, split_masks

OUT = ROOT/'reports/research/auxiliary_return_loss_v1'
FEATURES = ROOT/'reports/research/joinquant_candidates_v1/features.parquet'
ARMS = {'win': LossWeights(), 'absolute_01': LossWeights(absolute=.1),
        'absolute_03': LossWeights(absolute=.3), 'increment_01': LossWeights(incremental=.1),
        'increment_03': LossWeights(incremental=.3), 'downside_05': LossWeights(downside=.5),
        'joint': LossWeights(.1,.1,.5)}
PARAMS = dict(tree_method='hist', max_depth=3, min_child_weight=100, learning_rate=.06,
              reg_lambda=30, seed=73, nthread=4, base_score=0., disable_default_eval_metric=1)


def intervals(values: np.ndarray) -> list[float]:
    n=len(values)
    if n<40: return [float('nan')]*2
    rng=np.random.default_rng(73)
    means=[]
    for _ in range(1500):
        ids=(rng.integers(0,n,size=(int(np.ceil(n/10)),1))+np.arange(10))%n
        means.append(values[ids.ravel()[:n]].mean())
    return np.quantile(means,[.025,.975]).tolist()


def choose(validation: dict[str,pd.DataFrame]) -> dict[str,str]:
    # All rules use conservative unknown-exit stress, never test results.
    summary={a:(float(d.stress.mean()),float(d.stress_win.mean())) for a,d in validation.items()}
    net,win=summary['win']
    both=[a for a,(r,w) in summary.items() if r>=net and w>=win]
    ret=[a for a,(r,w) in summary.items() if w>=win-.02]
    wins=[a for a,(r,w) in summary.items() if r>=net]
    return {'balanced':max(both,key=lambda a:summary[a][0]),
            'return_priority':max(ret,key=lambda a:summary[a][0]),
            'win_priority':max(wins,key=lambda a:summary[a][1])}


def main() -> None:
    if '--allow-legacy-timing' not in sys.argv:
        raise SystemExit('Legacy experiment quarantined after timing/sample audit. Use evaluate_local_ranknet_fusion.py / evaluate_ranknet_cash_policy.py. --allow-legacy-timing only reproduces the superseded result; it does not repair PIT.')
    OUT.mkdir(parents=True,exist_ok=True)
    sets=cohorts()
    # For long, use actual absolute26week return net of an assumed20bp round trip.
    long_labels=pd.read_parquet(ROOT/'data/features/long_entry/weekly_training_v2.parquet',columns=['ts_code','date','return_26w']).rename(columns={'ts_code':'symbol'})
    sets['long'][0]['outcome']=sets['long'][0][['symbol','date']].merge(long_labels,on=['symbol','date'],validate='one_to_one').return_26w.to_numpy()-.002
    features=pd.read_parquet(FEATURES)
    protocol={'arms':{a:asdict(w) for a,w in ARMS.items()},'params':PARAMS,'rounds':80,
              'features_sha256':sha256(FEATURES),'baseline':'win-only XGBoost, identical features and rounds',
              'increment_definition':'realized net return minus same-date training candidate mean; target only, not an input',
              'selection':'validation-only; three predeclared preferences; no test tuning',
              'long':'actual26week stock return minus assumed20bps;2024/25 only;27week purge; overlapping labels',
              'limits':['Existing historical cohorts reused, no fresh prospective holdout','Financial vintage not verified','No live model promotion','Top20 cohort trade statistics, not portfolio NAV','Joint score is not a calibrated win probability']}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    metrics=[];selected=[]
    for side,(frame,base) in sets.items():
        frame=frame.merge(features,on=['symbol','date'],how='left',validate='one_to_one')
        folds=FOLDS if side=='right' else {str(y):y for y in ((2024,2025) if side=='long' else (2024,2025,2026))}
        for fold,bounds in folds.items():
            if side=='right':
                masks=split_masks(frame,bounds);fit=masks['fit'];val=masks['calibration']|masks['selection'];test=masks['diagnostic']
            else:
                boundary=pd.Timestamp(int(bounds),1,1);cut=boundary-pd.DateOffset(years=1)
                fit=frame.date.lt(cut)&frame.maturity.lt(cut)
                val=frame.date.ge(cut)&frame.date.lt(boundary)&frame.maturity.lt(boundary)
                test=frame.date.ge(boundary)&frame.date.lt(boundary+pd.DateOffset(years=1))
            fit &= frame.outcome.notna()
            test &= frame.maturity.le(pd.Timestamp('2026-11-10'))
            tr,va,te=frame.loc[fit].copy(),frame.loc[val].copy(),frame.loc[test].copy()
            cols=[c for c in dict.fromkeys([*base,*PRICE,*FINANCE]) if pd.api.types.is_numeric_dtype(tr[c]) and tr[c].notna().mean()>=.5 and tr[c].nunique()>1]
            matrix=lambda d:xgb.DMatrix(d[cols].astype('float32').replace([np.inf,-np.inf],np.nan),nthread=4)
            train,valid,testing=matrix(tr),matrix(va),matrix(te)
            net=tr.outcome.to_numpy(float);scale=max(.005,float(np.std(np.clip(net,-.5,.5))))
            target=terms(net,tr.groupby('date').outcome.transform('mean').to_numpy(float),scale)
            weight=(1/tr.groupby('date').date.transform('size')).to_numpy(copy=True);weight/=weight.mean()
            train.set_label(target[0])
            val_daily={};test_daily={};all_picks={};scores={}
            for arm,w in ARMS.items():
                print(side,fold,arm,'fit',len(tr),'val',len(va),'test',len(te),flush=True)
                model=xgb.train(PARAMS,train,num_boost_round=80,obj=lambda z,d:derivatives(z,target,w,weight))
                vs=model.predict(valid,output_margin=True);ts=model.predict(testing,output_margin=True)
                _,val_daily[arm]=daily_picks(va,vs)
                picks,day=daily_picks(te,ts);test_daily[arm]=day;all_picks[arm]=picks;scores[arm]=ts
                day.to_csv(OUT/f'daily_{side}_{fold}_{arm}.csv')
                known=picks.outcome.dropna();q=known.quantile(.05)
                metrics.append(dict(side=side,fold=fold,arm=arm,rows=len(te),features=len(cols),scale=scale,
                                    mean_net=day.net.mean(),win_rate=day.win.mean(),stress_net=day.stress.mean(),
                                    stress_win=day.stress_win.mean(),es5=known[known<=q].mean(),
                                    unknown=int(picks.outcome.isna().sum()),dates=len(day),complete_dates=int(day.net.notna().sum()),
                                    validation_net=val_daily[arm].stress.mean(),validation_win=val_daily[arm].stress_win.mean()))
                model.save_model(OUT/f'model_{side}_{fold}_{arm}.ubj')
                pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
            choices=choose(val_daily)
            # Include each fixed auxiliary arm for robustness inspection, separately from selected policies.
            for policy,arm in {**choices,**{f'fixed:{a}':a for a in ARMS if a!='win'}}.items():
                b=test_daily['win'];a=test_daily[arm]
                common=a.net.notna()&b.net.notna()
                dr=(a.loc[common,'net']-b.loc[common,'net']).to_numpy()
                dw=(a.loc[common,'win']-b.loc[common,'win']).to_numpy()
                ci=intervals(dr) if side!='long' else [float('nan')]*2
                wi=intervals(dw) if side!='long' else [float('nan')]*2
                selected.append(dict(side=side,fold=fold,policy=policy,arm=arm,paired_dates=int(common.sum()),
                                     net_delta_bps=float(dr.mean()*10000),win_delta_pp=float(dw.mean()*100),
                                     net_ci_low_bps=ci[0]*10000,net_ci_high_bps=ci[1]*10000,
                                     win_ci_low_pp=wi[0]*100,win_ci_high_pp=wi[1]*100,
                                     stress_delta_bps=float((a.stress-b.stress).mean()*10000),
                                     baseline_net=float(b.loc[common,'net'].mean()),candidate_net=float(a.loc[common,'net'].mean()),
                                     baseline_win=float(b.loc[common,'win'].mean()),candidate_win=float(a.loc[common,'win'].mean())))
                if policy in choices:
                    te[['symbol','date','outcome']].assign(score=scores[arm],baseline_score=scores['win']).to_parquet(OUT/f'predictions_{side}_{fold}_{policy}.parquet',index=False)
            pd.DataFrame(selected).to_csv(OUT/'comparisons.csv',index=False)
            (OUT/f'choices_{side}_{fold}.json').write_text(json.dumps(choices,indent=2))
    (OUT/'complete.json').write_text(json.dumps({'status':'complete','model_runs':len(metrics),'production_promoted':False}))


if __name__=='__main__':main()
