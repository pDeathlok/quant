"""Local research reproductions of JoinQuant factors; never used by live scoring.

Price factors use complete observed daily bars. Financial events become visible
strictly after announcement. Retrospective provider revisions remain unverified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

VERSION = 'jq-local-research-v1'
PRICE = tuple([f'jq_DAVOL{w}' for w in (5, 10, 20)] +
              ['jq_turnover_volatility'] +
              [f'jq_VOL{w}' for w in (60, 120, 240)] +
              [f'jq_Skewness{w}' for w in (20, 60, 120)] +
              [f'jq_Kurtosis{w}' for w in (20, 60, 120)] +
              ['jq_VOSC', 'jq_WVAD', 'jq_MAWVAD', 'jq_MFI14',
               'jq_arron_up_25', 'jq_arron_down_25'])
FINANCE = ('jq_cash_earnings_to_price_ratio', 'jq_operating_profit_yield_proxy',
           'jq_operating_profit_ratio', 'jq_operating_revenue_growth_rate',
           'jq_operating_profit_growth_rate', 'jq_gross_margin_change_pp_proxy',
           'jq_OperatingCycle_proxy')


def _aroon(series: pl.Series, maximum: bool) -> pl.Series:
    values=series.to_numpy(); result=np.full(len(values),np.nan)
    if len(values)>=25:
        windows=np.lib.stride_tricks.sliding_window_view(values,25)
        distance=(np.argmax if maximum else np.argmin)(windows[:,::-1],axis=1)
        result[24:]=(25-distance)/25*100
        result[24:][~np.isfinite(windows).all(axis=1)]=np.nan
    return pl.Series(result)


def price_factors(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute within-stock factors; inputs are split-consistent OHLCV and true turnover."""
    d = bars.sort(['symbol', 'date'])
    t, v, r = pl.col('turnover_rate'), pl.col('volume'), pl.col('ret')
    expr = [(t.rolling_mean(w) / t.rolling_mean(120)).over('symbol').alias(f'jq_DAVOL{w}') for w in (5, 10, 20)]
    expr += [t.rolling_std(20).over('symbol').alias('jq_turnover_volatility')]
    expr += [t.rolling_mean(w).over('symbol').alias(f'jq_VOL{w}') for w in (60, 120, 240)]
    expr += [r.rolling_skew(w, bias=False).over('symbol').alias(f'jq_Skewness{w}') for w in (20, 60, 120)]
    expr += [r.rolling_kurtosis(w, fisher=True, bias=False).over('symbol').alias(f'jq_Kurtosis{w}') for w in (20, 60, 120)]
    expr += [((v.ewm_mean(span=12, adjust=False)-v.ewm_mean(span=26, adjust=False))/v.ewm_mean(span=12, adjust=False)*100).over('symbol').alias('jq_VOSC')]
    spread = pl.col('high')-pl.col('low')
    flow = (pl.col('close')-pl.col('open')) / pl.when(spread>0).then(spread).otherwise(None)*v
    expr += [flow.rolling_sum(6).over('symbol').alias('jq_WVAD')]
    for field, side in [('high','up'),('low','down')]:
        # Most recent tied extreme; 25 observations, today gets 100.
        expr += [pl.col(field).map_batches(
            lambda s, maximum=(field=='high'): _aroon(s, maximum),
            return_dtype=pl.Float64,
        ).over('symbol').alias(f'jq_arron_{side}_25')]
    d=d.with_columns(expr)
    tp=(pl.col('high')+pl.col('low')+pl.col('close'))/3
    d=d.with_columns(tp.alias('_tp')).with_columns(pl.col('_tp').diff().over('symbol').alias('_dtp'))
    pos=pl.when(pl.col('_dtp')>0).then(pl.col('_tp')*v).otherwise(0.).rolling_sum(14)
    neg=pl.when(pl.col('_dtp')<0).then(pl.col('_tp')*v).otherwise(0.).rolling_sum(14)
    d=d.with_columns((100*pos/(pos+neg)).over('symbol').alias('jq_MFI14'),
                     pl.col('jq_WVAD').rolling_mean(6).over('symbol').alias('jq_MAWVAD'))
    return d.select('symbol','date',*PRICE).with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).cast(pl.Float32).alias(c) for c in PRICE])


def financial_events(raw: pd.DataFrame, fields: list[str], *, ttm: bool) -> pd.DataFrame:
    """First published statements only, each new period needs its own visible inputs."""
    d=raw.copy()
    if 'report_type' in d: d=d[d.report_type.astype(str).eq('1')]
    d['available']=pd.to_datetime(d.ann_date,format='%Y%m%d',errors='coerce')
    if 'f_ann_date' in d:
        actual=pd.to_datetime(d.f_ann_date,format='%Y%m%d',errors='coerce')
        d['available']=pd.concat([d.available,actual],axis=1).max(axis=1)
    d['end_date']=d.end_date.astype(str)
    d=d.dropna(subset=['available']).sort_values('available',kind='stable').drop_duplicates(['ts_code','end_date'])
    out=[]
    for symbol,g in d.groupby('ts_code',sort=False):
        known={}; latest=''
        for row in g.to_dict('records'):
            period=row['end_date']; known[period]=row
            if period<latest: continue
            latest=period; year=int(period[:4]); prev=str(year-1)+period[4:]
            record={'symbol':symbol,'available':row['available']}
            for field in fields:
                value=row.get(field,np.nan)
                if ttm and not period.endswith('1231'):
                    value=value+known.get(str(year-1)+'1231',{}).get(field,np.nan)-known.get(prev,{}).get(field,np.nan)
                record[field]=value
                # Growth uses a complete previous-year TTM, available at this event.
                prior=known.get(prev,{}).get(field,np.nan)
                if ttm and not period.endswith('1231'):
                    prior=prior+known.get(str(year-2)+'1231',{}).get(field,np.nan)-known.get(str(year-2)+period[4:],{}).get(field,np.nan)
                record[field+'_growth']=value/prior-1 if prior>0 else np.nan
                record[field+'_delta']=value-prior
            out.append(record)
    return pd.DataFrame(out)


def align_events(keys: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Same-day announcements excluded even for close-of-day signals."""
    k=keys.copy(); e=events.copy()
    k['symbol']=k.symbol.astype(str); e['symbol']=e.symbol.astype(str)
    k['date']=pd.to_datetime(k.date).astype('datetime64[ns]')
    e['available']=pd.to_datetime(e.available).astype('datetime64[ns]')
    return pd.merge_asof(k.sort_values('date'),e.sort_values('available'),by='symbol',left_on='date',right_on='available',allow_exact_matches=False)
