"""Frozen-feature training for self-built index-option volatility ablations."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from quant.features.index_panic import option_features, QVIX_FEATURES
from index_futures_state_models import OUT as BASE, pipeline
from index_futures_panic import OUT as QVIX
from build_index_option_iv import OUT, END, register

EVIDENCE_NAMES=('base','factor','interaction','qvix','native_mf','native_atm','qvix_mo')
ARMS={name:(index,4,'none') for name,index in [('baseline',3),('qvix',4),('native_mf',5),('native_atm',6),('qvix_mo',7)]}
MODELS=('native_mf','native_atm','qvix_mo')


def build_features():
    indices=pd.read_parquet(OUT/'indices.parquet')
    indices=indices[indices.price.eq('close')&indices.trade_date.le(END)]
    assert not indices.duplicated(['trade_date','product']).any()
    spots={}
    for symbol in ('IF','IH','IC','IM'):
        s=pd.read_parquet(BASE.parent/f'{symbol}_spot.parquet')
        s=s[s.trade_date.le(END)].set_index('trade_date').sort_index()
        s['rv20']=np.log(s.close).diff().rolling(20).std()*np.sqrt(252)
        spots[symbol]=s
    result={}
    for method in ('mf','atm'):
        common=indices[indices['product'].eq('IO')].set_index('trade_date')[f'{method}_value']
        common=option_features(common,spots['IF'].rv20,spots['IF'].index).add_prefix('panic_common_')
        parts=[]
        for symbol,spot in spots.items():
            product={'IF':'IO','IH':'HO','IM':'MO'}.get(symbol)
            own=indices[indices['product'].eq(product)].set_index('trade_date')[f'{method}_value'] if product else pd.Series(dtype=float)
            own=option_features(own,spot.rv20,spot.index).add_prefix('panic_own_')
            f=common.reindex(spot.index).join(own)
            f['symbol']=symbol
            parts.append(f.reset_index())
        result[f'native_{method}']=pd.concat(parts,ignore_index=True)
    hybrid=pd.read_parquet(QVIX/'features.parquet')[['trade_date','symbol',*QVIX_FEATURES]].copy()
    mo=result['native_mf'].query("symbol == 'IM'").set_index('trade_date')
    mask=hybrid.symbol.eq('IM')
    own=[c for c in QVIX_FEATURES if c.startswith('panic_own_')]
    hybrid.loc[mask,own]=mo.reindex(hybrid.loc[mask,'trade_date'])[own].to_numpy()
    result['qvix_mo']=hybrid
    coverage=[]
    for name,f in result.items():
        f.to_parquet(OUT/f'features_{name}.parquet',index=False)
        f=f.assign(year=f.trade_date.str[:4])
        for (year,symbol),g in f[f.year.ge('2020')].groupby(['year','symbol']):
            coverage.append(dict(model=name,year=year,symbol=symbol,days=len(g),common_missing=g.panic_common_missing.mean(),own_missing=g.panic_own_missing.mean()))
        if name!='qvix_mo':
            for scope in ('own','common'):
                source=f[f'panic_{scope}_source_date'];assert source.dropna().lt(f.loc[source.notna(),'trade_date']).all()
    pd.DataFrame(coverage).to_csv(OUT/'feature_coverage.csv',index=False)
    return result


def train():
    register();feature_frames=build_features()
    entries=pd.read_parquet(BASE/'entry_dataset.parquet')
    base=json.loads((BASE/'feature_sets.json').read_text())['entry']['interaction']
    features=base+QVIX_FEATURES
    (OUT/'feature_sets.json').write_text(json.dumps({m:features for m in MODELS},indent=2))
    old=pd.read_parquet(QVIX/'forecasts.parquet');old=old[old.model.eq('qvix')].set_index(['entry_date','symbol','direction','bps'])
    audits,predictions,coefficients=[],[],[]
    with threadpool_limits(limits=1):
        for name in MODELS:
            panel=entries.merge(feature_frames[name][['trade_date','symbol',*QVIX_FEATURES]],on=['trade_date','symbol'],how='left',validate='many_to_one')
            for quarter,part in panel.groupby('quarter',sort=True):
                cutoff=part.trade_date.min();begin=(pd.Timestamp(cutoff)-pd.DateOffset(months=36)).strftime('%Y%m%d')
                for direction in (1,-1):
                    training=panel[panel.direction.eq(direction)&panel.trade_date.ge(begin)&panel.maturity.lt(cutoff)]
                    test=part[part.direction.eq(direction)]
                    age=(pd.Timestamp(cutoff)-pd.to_datetime(training.maturity)).dt.days.to_numpy()
                    weights=np.exp2(-age/365.25)
                    if len(weights):weights/=weights.mean()
                    for bps in (3,6):
                        columns=['probability','pred_net','pred_risk']
                        previous=old.reindex(pd.MultiIndex.from_arrays([test.entry_date,test.symbol,test.direction,np.full(len(test),bps)],names=old.index.names))[columns].to_numpy()
                        output=test[['entry_date','trade_date','symbol','direction','quarter','maturity',f'net_{bps}',f'risk_{bps}','panic_common_missing','panic_own_missing']].copy()
                        y=training[f'net_{bps}'].gt(0).astype(int)
                        ready=training.trade_date.nunique()>=244 and y.nunique()==2
                        output['bps'],output['model'],output['ready']=bps,name,ready
                        output[columns]=previous;output['fallback']='qvix_warmup'
                        if ready:
                            assert training.maturity.max()<cutoff<=test.trade_date.min()
                            lr=pipeline(LogisticRegression(C=.1,max_iter=1000));ridge=pipeline(Ridge(alpha=.1*len(training)))
                            lr.fit(training[features],y,standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
                            ridge.fit(training[features],training[[f'net_{bps}',f'risk_{bps}']],standardscaler__sample_weight=weights,ridge__sample_weight=weights)
                            output['probability']=lr.predict_proba(test[features])[:,1]
                            raw=ridge.predict(test[features]);output['pred_net'],output['pred_risk']=raw[:,0],np.maximum(0,raw[:,1])
                            output['fallback']='none'
                            missing=test.panic_common_missing.ne(0)
                            if name=='qvix_mo':missing |= test.symbol.eq('IM')&test.panic_own_missing.ne(0)
                            output.loc[missing,columns]=previous[missing.to_numpy()]
                            output.loc[missing,'fallback']='qvix_missing_iv'
                            coefficients.extend(dict(model=name,quarter=quarter,direction=direction,bps=bps,feature=f,coefficient=float(c)) for f,c in zip(features,lr[-1].coef_[0]))
                        audits.append(dict(model=name,quarter=quarter,direction=direction,bps=bps,cutoff=cutoff,begin=begin,rows=len(training),days=training.trade_date.nunique(),last_maturity=training.maturity.max(),ready=ready,features=len(features),fallback_rows=output.fallback.ne('none').sum()))
                        predictions.append(output)
                print('option models',name,quarter,flush=True)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'forecasts.parquet',index=False)
    pd.DataFrame(audits).to_csv(OUT/'training_audit.csv',index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'coefficients.csv',index=False)
    (OUT/'global_calendar.csv').write_bytes((BASE/'global_calendar.csv').read_bytes())


if __name__=='__main__':train()
