"""Preregistered policy-conditioned entry targets and chronological calibration."""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from backtest_index_futures import ROOT
from index_futures_state_models import OUT as PREVIOUS, entry_path, pipeline, add_regime_features
from index_futures_state_rules import holding_limit

OUT=ROOT/'policy_label_calibration'
HORIZONS={'h5':5,'h10':10,'h20':20,'state':None}
EVIDENCE_NAMES=('base','factor','interaction','five',*[x for h in HORIZONS for x in (f'matched_{h}',f'cal_{h}')], 'prob_state','return_state')
ARMS={'original':(0,0,'none'),'hard_both':(0,0,'both'),'previous_state':(3,4,'none')}
for h,soft in (('h5',8),('h10',3),('h20',0),('state',4)):
    for family,name in (('five','five'),('matched',f'matched_{h}'),('cal',f'cal_{h}')):
        ARMS[f'{family}_{h}']=(EVIDENCE_NAMES.index(name)+1,soft,'none')
for name in ('prob_state','return_state'):
    ARMS[name]=(EVIDENCE_NAMES.index(name)+1,4,'none')


def register():
    OUT.mkdir(exist_ok=True)
    protocol=dict(end='20261106',capital=500000,max_lots=1,arms=ARMS,
        scope='17 preregistered arms x fixed/rolling, mean5 native exit only; old control/loss/calibrated/tail results preserved, no claim of new full crossover',
        feature_model='frozen85 state-interaction features plus native opposing-exit policy indicator; separate long/short; LR C=.1, Ridge alpha=.1*n,36m mature history,12m half-life; no search',
        population='all original qualified candidate dates and original chosen side, not selected by portfolio fills or future outcomes; same candidate population for fixed5 and matched labels',
        labels='five:5session unconditional. matched: earliest of frozen mean5 two-negative exit, native opposing exit if current policy, uniform5/10/20 or entry-state locked20/10/5; include both roll legs, close liquidation marks and all fees. Hold current policy fixed for label; current account reserve/margin and future selector switches remain external overrides, explicitly audit actual-label divergence',
        maturity='label realized exit strictly before training/calibration cutoff; never fill censored targets with zero; prediction features signal-date only; old causal continuation forecasts reused, never refit them on future data',
        warmup='base needs122 distinct candidate dates and20 positive/20 negative rows; otherwise explicit prior interaction forecast fallback. Calibration needs60 distinct dates,at least2 forecast quarters,20 positive/20 negative; otherwise raw. Disclose side/year readiness',
        calibration='only previously issued quarterly out-of-time raw forecasts with matured targets from past24months. Twelve-month half-life, equal total weight per signal date. Probability: sigmoid of logit(raw),LogisticRegression C=1; nonpositive slope falls back weighted prevalence. Net and risk: affine slope with20 pseudo-observations at identity,reference scale.01,nonnegative slope capped2,shrunken bias; clip risk>=0. No OOT threshold fitting',
        ablations='fixed5 raw versus matched raw versus matched probability+net+risk calibration at each of4 deadlines; at state deadline also probability-only and net/risk-only calibration; preserve prior original/hard_both/state_horizon exactly',
        gate='unchanged probability>=.55 and net-.5*risk>0, allows countertrend; same gate for every arm, no year/product exceptions',
        replay='same fixed global5session entry calendar; daily exit; 500k,max1lot,no topups; same cost15/25margin profiles and monthly/Janphase/annual grids; inner validation at exact current equity using same new forecasts/exits; rolling only4 existing policies,not retrospective arm selection',
        evaluation='2022-2026 single-year,each-start-to-latest,122/244sessions,continuous annual,worststarts,flat and paired opportunity cost; candidate Brier/logloss/reliability/netbias/MAE byyear/side/state,actual label divergence',
        caveats='all reused history exploratory,not freshOOT; overlapping labels/windows not independent; calibration not a tail bound; conditional strategy label cannot guarantee future account-level selector/margin actions',
        reference='https://scikit-learn.org/stable/modules/calibration.html')
    target=OUT/'protocol.json';content=json.dumps(protocol,ensure_ascii=False,indent=2)
    if target.exists() and target.read_text()!=content:raise RuntimeError('Preregistered protocol changed')
    target.write_text(content)
    (OUT/'global_calendar.csv').write_bytes((PREVIOUS/'global_calendar.csv').read_bytes())
    files=[PREVIOUS/'entry_forecasts.parquet',PREVIOUS/'feature_sets.json',ROOT/'bull_bear_regime/states_200.parquet',
           ROOT/'adaptation_comparison/quarterly_predictions.parquet',ROOT/'exit_calibration_starts/calibrated_forecasts.parquet']
    (OUT/'sources.json').write_text(json.dumps([dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],indent=2))


def policy_exit_age(rows,start,side,bps,limit,native,lookup):
    """Conditional trade exit; no account-dependent funding or future selection."""
    first=rows[start];streak=0
    for age in range(1,limit+1):
        if start+age>len(rows):return None
        if age==limit:return age,'deadline'
        if start+age==len(rows):return None
        row=rows[start+age]
        if native and (row.short_20 if side>0 else row.long_20)>=.35:
            return age,'native_opposing'
        forecast=lookup[first.entry_date,row.entry_date,row.symbol]
        assert forecast.direction==side
        net=getattr(forecast,f'pred_net_5_{bps}')
        streak=streak+1 if np.isfinite(net) and net<=0 else 0
        if streak>=2:return age,'mean5'
    raise AssertionError('Unreachable')


def build_dataset():
    from evaluate_index_futures_regime import RegimeReplay
    replay=RegimeReplay();panel=replay.panel
    frozen=pd.read_parquet(ROOT/'bull_bear_regime/states_200.parquet')
    replay.market_regime={(r.trade_date,r.symbol):int(r.state)
        for r in frozen.dropna(subset=['state']).itertuples(index=False)}
    assert panel.exit_date.max()=='20261106'
    records=[]
    for product,part in panel.groupby('symbol'):
        rows=list(part.sort_values('entry_date').itertuples(index=False))
        for i,r in enumerate(rows):
            if pd.isna(r.etf_ret_1) or max(r.long_60,r.short_60)<=.45 or abs(r.long_60-r.short_60)<.1:continue
            side=1 if r.long_60>r.short_60 else -1
            state=replay.market_regime[r.trade_date,product]
            for target in ('five',*HORIZONS):
                limit=5 if target=='five' else HORIZONS[target] or holding_limit(state,side)
                for native in (0,1):
                    for bps in (3,6):
                        reason='fixed5';age=5
                        if target!='five':
                            result=policy_exit_age(rows,i,side,bps,limit,native,replay.lookup)
                            age,reason=result if result else (0,'censored')
                        label=entry_path(rows,i,side,bps/10000,replay.closes,age) if age else None
                        records.append(dict(entry_date=r.entry_date,symbol=product,direction=side,target=target,
                            native_exit=native,bps=bps,label_days=age if label else np.nan,label_reason=reason if label else 'censored',
                            **(label or dict(maturity=None,net=np.nan,risk=np.nan))))
        print('labels',product,len(records),flush=True)
    labels=pd.DataFrame(records)
    keys=['entry_date','symbol']
    # The source panel also has historical modeling targets. Only our explicit
    # policy targets may be used here; the source target is not an input feature.
    source=panel.drop(columns=[c for c in labels if c in panel and c not in keys])
    data=add_regime_features(source.merge(labels,on=keys,validate='one_to_many'),'trade_date')
    data.to_parquet(OUT/'dataset.parquet',index=False)
    return data


def time_weights(frame,cutoff):
    age=(pd.Timestamp(cutoff)-pd.to_datetime(frame.maturity)).dt.days.to_numpy()
    weights=np.exp2(-age/365.25)
    return weights/weights.mean() if len(weights) else weights


def affine_parameters(pred,actual,weights):
    """Shrunk monotone affine map, fixed identity prior; no outcome-selected scale."""
    sw=weights.sum();px=np.average(pred,weights=weights);yy=np.average(actual,weights=weights)
    variance=np.sum(weights*(pred-px)**2);covariance=np.sum(weights*(pred-px)*(actual-yy))
    slope=float(np.clip((covariance+20*.01**2)/(variance+20*.01**2),0,2))
    intercept=float(np.sum(weights*(actual-slope*pred))/(sw+20))
    return slope,intercept


def calibrate(history,test,cutoff):
    begin=(pd.Timestamp(cutoff)-pd.DateOffset(months=24)).strftime('%Y%m%d')
    past=history[history.trade_date.ge(begin)&history.maturity.lt(cutoff)&history.ready].copy()
    y=past.net.gt(0).astype(int)
    ready=past.trade_date.nunique()>=60 and past.quarter.nunique()>=2 and y.sum()>=20 and (1-y).sum()>=20
    out=test[['probability','pred_net','pred_risk']].to_numpy().copy()
    detail=dict(calibration_ready=bool(ready),cal_rows=len(past),cal_dates=past.trade_date.nunique(),
        cal_quarters=past.quarter.nunique(),cal_last_maturity=past.maturity.max() if len(past) else None)
    if not ready:return out,detail
    assert past.maturity.max()<cutoff and past.trade_date.max()<test.trade_date.min()
    # First equalize each signal date, then apply recency weighting. Repeated
    # product/native-policy outcomes must not count as independent date evidence.
    weights=1/past.groupby('trade_date').trade_date.transform('size').to_numpy()
    ages=(pd.Timestamp(cutoff)-pd.to_datetime(past.trade_date)).dt.days.to_numpy()
    weights*=np.exp2(-ages/365.25);weights*=past.trade_date.nunique()/weights.sum()
    logit=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    clf=LogisticRegression(C=1,max_iter=1000).fit(logit(past.probability.to_numpy()).reshape(-1,1),y,sample_weight=weights)
    a=float(clf.coef_[0,0]);b=float(clf.intercept_[0])
    if a<=0:
        a=0.;b=float(logit(np.average(y,weights=weights)))
    out[:,0]=1/(1+np.exp(-np.clip(a*logit(out[:,0])+b,-700,700)))
    detail.update(prob_slope=a,prob_intercept=b)
    for j,col,target in ((1,'pred_net','net'),(2,'pred_risk','risk')):
        slope,intercept=affine_parameters(past[col].to_numpy(),past[target].to_numpy(),weights)
        out[:,j]=slope*out[:,j]+intercept
        detail.update({target+'_slope':slope,target+'_intercept':intercept})
    out[:,2]=np.maximum(0,out[:,2])
    return out,detail


def train_models():
    data=build_dataset()
    features=json.loads((PREVIOUS/'feature_sets.json').read_text())['entry']['interaction']+['native_exit']
    old=pd.read_parquet(PREVIOUS/'entry_forecasts.parquet')
    old=old[old.model.eq('interaction')].set_index(['entry_date','symbol','direction','bps'])
    output=[];audits=[];raw_history={}
    for quarter,part in data.groupby('quarter',sort=True):
        cutoff=part.trade_date.min();begin=(pd.Timestamp(cutoff)-pd.DateOffset(months=36)).strftime('%Y%m%d')
        for side in (1,-1):
            for target in ('five',*HORIZONS):
                for bps in (3,6):
                    key=side,target,bps
                    sample=data[data.direction.eq(side)&data.target.eq(target)&data.bps.eq(bps)]
                    train=sample[sample.trade_date.ge(begin)&sample.maturity.lt(cutoff)]
                    test=sample[sample.quarter.eq(quarter)].copy()
                    if test.empty:continue
                    y=train.net.gt(0).astype(int)
                    ready=train.trade_date.nunique()>=122 and y.sum()>=20 and (1-y).sum()>=20
                    cols=['entry_date','trade_date','quarter','symbol','direction','native_exit','target','bps','maturity','net','risk','label_days','label_reason',
                          'regime_bull','regime_bear']
                    pred=test[cols].copy();pred['ready']=ready
                    fallback=old.reindex(pd.MultiIndex.from_frame(test[['entry_date','symbol','direction','bps']]))
                    pred[['probability','pred_net','pred_risk']]=fallback[['probability','pred_net','pred_risk']].to_numpy()
                    if ready:
                        assert train.maturity.max()<cutoff<=test.trade_date.min()
                        weights=time_weights(train,cutoff)
                        lr=pipeline(LogisticRegression(C=.1,max_iter=1000));rr=pipeline(Ridge(alpha=.1*len(train)))
                        lr.fit(train[features],y,standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
                        rr.fit(train[features],train[['net','risk']],standardscaler__sample_weight=weights,ridge__sample_weight=weights)
                        pred['probability']=lr.predict_proba(test[features])[:,1]
                        z=rr.predict(test[features]);pred['pred_net']=z[:,0];pred['pred_risk']=np.maximum(0,z[:,1])
                    raw_history.setdefault(key,[])
                    history=pd.concat(raw_history[key],ignore_index=True) if raw_history[key] else pred.iloc[:0]
                    calibrated,detail=calibrate(history,pred,cutoff)
                    # Do not transform old-model fallback with a calibrator trained
                    # on a different predictor's forecasts.
                    if not ready:
                        calibrated=pred[['probability','pred_net','pred_risk']].to_numpy();detail['calibration_ready']=False
                    pred['calibration_ready']=detail['calibration_ready']
                    pred[['cal_probability','cal_net','cal_risk']]=calibrated
                    raw_history[key].append(pred.copy())
                    pred['model']='five' if target=='five' else 'matched_'+target
                    output.append(pred)
                    audits.append(dict(quarter=quarter,direction=side,target=target,bps=bps,cutoff=cutoff,ready=ready,
                        train_rows=len(train),train_days=train.trade_date.nunique(),last_maturity=train.maturity.max(),**detail))
        print('trained policy labels',quarter,flush=True)
    pd.concat(output,ignore_index=True).to_parquet(OUT/'forecasts.parquet',index=False)
    pd.DataFrame(audits).to_csv(OUT/'training_audit.csv',index=False)
    (OUT/'features.json').write_text(json.dumps(features,indent=2))


if __name__=='__main__':
    register();train_models()
