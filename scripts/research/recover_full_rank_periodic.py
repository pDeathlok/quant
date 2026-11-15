"""Second admission pass: reconstruct remaining price/dated-basic histories."""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
import evaluate_local_ranknet_fusion as prior

SOURCE=ROOT/'reports/research/full_rank_comparison_v1'
OUT=ROOT/'reports/research/full_rank_comparison_v2'
KEYS=['symbol','date']


def asof(keys, values, tolerance):
    keys=keys.assign(_row=np.arange(len(keys)),date=pd.to_datetime(keys.date).astype('datetime64[ns]'))
    values=values.assign(date=pd.to_datetime(values.date).astype('datetime64[ns]')).rename(columns={'date':'_available'})
    joined=pd.merge_asof(keys.sort_values(['date','symbol']),values.sort_values(['_available','symbol']),
        left_on='date',right_on='_available',by='symbol',direction='backward',allow_exact_matches=False,tolerance=pd.Timedelta(days=tolerance))
    return joined.sort_values('_row').drop(columns=['_row','_available']).reset_index(drop=True)


def main():
    assert (SOURCE/'complete.json').exists()
    OUT.mkdir(exist_ok=True)
    meta=json.loads((SOURCE/'dataset_complete.json').read_text());audit=pd.read_csv(SOURCE/'column_audit.csv')
    frames={s:pd.read_parquet(SOURCE/f'dataset_{s}.parquet') for s in ['left','right']}
    keys=pd.concat([f[['symbol','source_date']].rename(columns={'source_date':'date'}) for f in frames.values()]).drop_duplicates().reset_index(drop=True)
    extra=keys.copy();new=[];evidence={};inputs={}
    paths=sorted(p for p in (ROOT/'data/raw/daily_basic').glob('*.parquet') if p.stem.isdigit() and '20130101'<=p.stem<='20260904')
    files=pd.DataFrame({'path':paths,'date':pd.to_datetime([p.stem for p in paths])})
    for freq,tolerance in [('W-FRI',14),('M',45)]:
        picked=files.groupby(files.date.dt.to_period(freq)).tail(1)
        parts=[]
        for r in picked.itertuples():
            f=pd.read_parquet(r.path,columns=['ts_code','trade_date','pe_ttm','pb','dv_ttm'])
            f['date']=pd.Timestamp(r.date).to_period(freq).end_time.normalize()
            parts.append(f.rename(columns={'ts_code':'symbol'}));inputs[str(r.path.relative_to(ROOT))]=prior.sha256(r.path)
        values=pd.concat(parts,ignore_index=True).sort_values(['symbol','date']).reset_index(drop=True)
        if values.duplicated(KEYS).any():raise ValueError('Duplicate periodic source')
        cols=[]
        for source,prefix in [('pe_ttm','pe'),('pb','pb')]:
            clean=values[source].where(values[source]>0)
            for years in ([5,7] if freq=='W-FRI' else [None]):
                window=years*52 if years else 84; minimum=104 if years else 24
                name=f'{prefix}_hist_percentile'+(f'_{years}y' if years else '')
                values[name]=clean.groupby(values.symbol,sort=False).transform(lambda v:v.rolling(window,min_periods=minimum).rank(method='max',pct=True)*100)
                cols.append(name)
                if source=='pe_ttm' and years:
                    alias=f'pe_ttm_hist_percentile_{years}y';values[alias]=values[name];cols.append(alias)
        if freq=='W-FRI':
            dv=values.dv_ttm.groupby(values.symbol,sort=False)
            values['dv_ttm_mean_36m']=dv.transform(lambda v:v.rolling(156,min_periods=52).mean())
            values['dv_ttm_std_36m']=dv.transform(lambda v:v.rolling(156,min_periods=52).std())
            values['dv_ttm_stability_36m']=(values.dv_ttm_mean_36m/values.dv_ttm_std_36m.replace(0,np.nan)).clip(0,10)
            cols+=['dv_ttm_mean_36m','dv_ttm_std_36m','dv_ttm_stability_36m']
        aligned=asof(keys,values[KEYS+cols],tolerance)
        extra[cols]=aligned[cols].to_numpy();new+=cols
        evidence.update({c:('raw_daily_basic_period_history',f'strict_after_completed_{freq}', 'positivePE_PB_rolling_ranks_or_weekly_dividend_history') for c in cols})
        print('RECOVERED',freq,len(cols),flush=True)
    stock_path=ROOT/'data/raw/stock_basic_history.parquet'
    stock=pd.read_parquet(stock_path,columns=['ts_code','list_date']).rename(columns={'ts_code':'symbol'})
    stock['list_date']=pd.to_datetime(stock.list_date,errors='coerce')
    conflicts=stock.groupby('symbol').list_date.nunique().gt(1)
    stock=stock.loc[~stock.symbol.isin(conflicts.index[conflicts])].drop_duplicates('symbol')
    listed=keys.merge(stock,on='symbol',how='left',validate='many_to_one')
    extra['listing_years']=((listed.date-listed.list_date).dt.days/365.25).where(listed.date.ge(listed.list_date))
    new.append('listing_years');inputs[str(stock_path.relative_to(ROOT))]=prior.sha256(stock_path)
    evidence['listing_years']=('historical_stock_basic_IPO_date','IPO_date_not_after_signal','immutable_listing_date_only_no_current_status_or_industry')
    bench_path=ROOT/'reports/research/right_t012_index_sources_v1/000300.SH_index_daily.parquet'
    bench=pd.read_parquet(bench_path);bench['date']=pd.to_datetime(bench.trade_date);bench=bench.sort_values('date')
    equity=bench.close; cols=[]
    for weeks,days in [(13,63),(26,126),(52,252)]:
        c=f'market_return_{weeks}w';bench[c]=equity.pct_change(days,fill_method=None);cols.append(c)
    bench['market_drawdown_52w']=equity/equity.rolling(252,min_periods=63).max()-1
    bench['market_volatility_13w']=equity.pct_change(fill_method=None).rolling(63,min_periods=20).std()*np.sqrt(252)
    cols+=['market_drawdown_52w','market_volatility_13w']
    extra=extra.merge(bench[['date',*cols]],on='date',how='left',validate='many_to_one');new+=cols
    ret=pd.concat([f[['symbol','source_date','return_120d']].rename(columns={'source_date':'date'}) for f in frames.values()]).drop_duplicates(KEYS)
    extra=extra.merge(ret,on=KEYS,validate='one_to_one')
    extra['return_120d_minus_market']=extra.return_120d-extra.market_return_26w;extra=extra.drop(columns='return_120d')
    new.append('return_120d_minus_market');cols.append('return_120d_minus_market')
    inputs[str(bench_path.relative_to(ROOT))]=prior.sha256(bench_path)
    evidence.update({c:('CSI300_daily_history','signal_close','trailing63_126_252_sessions_no_financial_snapshot') for c in cols})
    # Rolling top-list statistics; missing daily files invalidate the whole window.
    top_paths=sorted(p for p in (ROOT/'data/raw/top_list').glob('*.parquet') if '20230901'<=p.stem.rsplit('_',1)[-1]<='20260904')
    top=pd.concat([pd.read_parquet(p,columns=['ts_code','trade_date','net_amount','amount','reason']) for p in top_paths],ignore_index=True)
    top['date']=pd.to_datetime(top.trade_date)
    calendar=pd.DatetimeIndex(files.date).sort_values()
    available=pd.DatetimeIndex(pd.to_datetime([p.stem.rsplit('_',1)[-1] for p in top_paths]))
    covered=pd.Series(calendar.isin(available).astype(int)).rolling(60,min_periods=60).sum().eq(60).to_numpy()
    query=keys.copy();query['_pos']=calendar.searchsorted(query.date)-1
    top['_pos']=calendar.searchsorted(top.date)
    top_groups={s:g.sort_values('_pos') for s,g in top.groupby('ts_code')};parts=[]
    cols=['top_list_count_20d','top_list_net_ratio_20d','top_list_positive_days_20d','top_list_reason_concentration_60d']
    for symbol,k in query.groupby('symbol',sort=False):
        k=k.copy();q=k._pos.to_numpy();v=np.zeros((len(k),4));events=top_groups.get(symbol)
        if events is not None:
            pos=events._pos.to_numpy();r=np.searchsorted(pos,q,side='right');l20=np.searchsorted(pos,q-19);l60=np.searchsorted(pos,q-59)
            v[:,0]=r-l20
            net=events.net_amount.to_numpy(float);amount=events.amount.to_numpy(float)
            ns=np.r_[0,np.nancumsum(net)];gross=np.r_[0,np.nancumsum(amount)];num=ns[r]-ns[l20];den=gross[r]-gross[l20]
            np.divide(num,den,out=v[:,1],where=den!=0)
            missing=np.r_[0,np.cumsum(~np.isfinite(net)|~np.isfinite(amount))]
            v[missing[r]-missing[l20]>0,1]=np.nan
            positive=np.unique(pos[net>0]);v[:,2]=np.searchsorted(positive,q,side='right')-np.searchsorted(positive,q-19)
            reason=pd.factorize(events.reason.fillna('unknown'))[0];maximum=np.zeros(len(k))
            for code in np.unique(reason):
                cumulative=np.r_[0,np.cumsum(reason==code)];maximum=np.maximum(maximum,cumulative[r]-cumulative[l60])
            np.divide(maximum,r-l60,out=v[:,3],where=(r-l60)>0)
        valid=(q>=0)&covered[np.maximum(q,0)]
        v[~valid]=np.nan
        parts.append(pd.DataFrame(v,index=k.index,columns=cols))
    result=pd.concat(parts).reindex(keys.index);extra[cols]=result;new+=cols
    evidence.update({c:('raw_daily_top_list','previous_session_and60file_coverage','full_market_event_windows_no_label_filtered_keys') for c in cols})
    inputs.update({str(p.relative_to(ROOT)):prior.sha256(p) for p in top_paths})
    assert len(new)==len(set(new))==22
    assert not set(new)&set(meta['columns'])
    for c,(source,timing,proof) in evidence.items():
        audit.loc[audit.name.eq(c),['admitted','source','timing','evidence','exclusion_reason']]=[True,source,timing,proof,'']
    blocked=~audit.admitted
    for prefix,reason in [('holder_','announcement_event_revision_archive_missing'),('pledge_','statistical_end_date_not_verified_publication_date')]:
        audit.loc[blocked&audit.name.str.startswith(prefix),'exclusion_reason']=reason
    industry=blocked&(audit.name.str.contains('industry')|audit.name.isin(['valuation_profile_code','pr_pe_weight','pr_pb_weight']))
    audit.loc[industry,'exclusion_reason']='historical_industry_classification_or_membership_not_versioned'
    mask=blocked&audit.exclusion_reason.eq('snapshot_only_publication_revision_or_historical_industry_not_certified')
    audit.loc[mask,'exclusion_reason']='statement_revision_archive_missing_or_dependent_financial_snapshot'
    allowed=audit.loc[audit.admitted,'name'].tolist()
    for side,frame in frames.items():
        frame=frame.merge(extra[KEYS+new].rename(columns={'date':'source_date'}),on=['symbol','source_date'],how='left',validate='one_to_one')
        frame[new]=frame[new].astype('float32').replace([np.inf,-np.inf],np.nan)
        frame.to_parquet(OUT/f'dataset_{side}.parquet',index=False)
        meta['dataset_hashes'][side]=prior.sha256(OUT/f'dataset_{side}.parquet')
        pd.DataFrame({'name':allowed,'coverage':frame[allowed].notna().mean(),'unique':frame[allowed].nunique()}).to_csv(OUT/f'coverage_{side}.csv',index=False)
    audit.to_csv(OUT/'column_audit.csv',index=False)
    for pattern in ['oof_*.parquet','oof_*.csv','oof_model_*.ubj','prefix_checks.csv','readiness_audit.json','source_recovery_audit.json','chan_prefix_audit.csv']:
        for path in SOURCE.glob(pattern):shutil.copy2(path,OUT/path.name)
    for side in ['left','right']:
        for fold in ['2025H2','2026']:
            for arm in ['base50','core129']:shutil.copy2(SOURCE/f'model_{side}_{fold}_{arm}.ubj',OUT/f'model_{side}_{fold}_{arm}.ubj')
    meta.update(columns=allowed,admitted=len(allowed),original_776_admitted=int((audit.admitted&audit.original_776).sum()),periodic_recovery_columns=new)
    meta['inputs'].update(inputs)
    prior.save_json(OUT/'dataset_complete.json',meta)
    prior.save_json(OUT/'periodic_recovery.json',dict(columns=new,source=str(SOURCE),listing_date_conflicts=int(conflicts.sum()),
        final_round='Same protocol; recover22 previously unverified but reconstructible columns; frozen core129/base50/OOF reused',
        script_hash=prior.sha256(Path(__file__))))
    print('RECOVERED FINAL',len(new),'TOTAL',len(allowed),flush=True)


if __name__=='__main__':main()
