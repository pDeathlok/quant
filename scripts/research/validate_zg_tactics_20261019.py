"""Frozen, exploratory price-event audit of zg-ai's 2026-10-18 notes.

Not an executable portfolio backtest: historical identity, limit queues, capacity,
and all source indicators are unavailable. Never substitute these for NAV.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/research/zg_tactics_20261019'
DESIGN = {
    'status': 'exploratory retrospective event study; no parameter fitting',
    'sample': '800 lowest SHA256(ts_code) among all observed SH/SZ A-share codes, including any delisted present in archive; not complete point-in-time universe',
    'dates': 'raw 20140101..20261016; events 20150101 onward; complete 20 subsequent stock observations required',
    'adjustment': 'causal continuous OHLC using previous close / pre_close cumulative factor',
    'white': 'EMA(EMA(C,10),10), from local trend-lines.md, not provided by linked page',
    'yellow_proxy': 'MA60; deliberately a proxy, not claimed source formula',
    'trend': 'white > MA60, C > MA60, at least 250 observed bars',
    'pullback': 'trend AND C <= white AND C < prior C AND L >= prior 20-observation low',
    'dry': 'pullback AND V < previous V AND V < previous 5-observation mean V',
    'b1_proxy': 'dry AND KDJ(9,3,3) J < 0',
    'b2_proxy': 'dry in previous 3 observations AND C/prior C-1 >4% AND V>prior V AND J<55 AND trend',
    's1_proxy': 'C<O AND C/O-1<=-4% AND H>=prior60high*.95; split V/prior20mean >=2 vs <2',
    'breakout': 'C>prior20high AND C>O AND C/priorC-1>4% AND V/prior20mean>=1.5; following observation bearish',
    'breakout_variants': 'prior high lookback 20 or 60; preceding bullish return >4% or >0%; four prespecified combinations',
    'volume_groups': 'bearish day V / breakout day V: [0,.5), [.5,.75), [.75,1), [1,infinity)',
    'failure_labels': 'bearish-day+5 or +20 close below frozen breakout neckline; all stock observations, not market sessions',
    'returns': 'signal+1 adjusted open -> signal+5/+20 adjusted close; report gross and fixed 30bps round-trip cost scenario',
    'dedup': 'per stock, per rule, first eligible event; next event more than20 observed bars later',
    'periods': '2015..2019, 2020..2023, 2024..2026; not untouched out of sample',
    'uncertainty': '95% moving 20-market-date block bootstrap of date-mean outcomes, 1000 replicates; date balanced, unadjusted multiple comparisons',
    'limitations': ['no source yellow or brick formula', 'no active market cap reconstruction', 'no tradeability/queue/identity/delisting-return reconstruction', 'archive survivor and coverage bias possible', 'event returns cannot imply strategy CAGR or max drawdown'],
}


def features(raw):
    d = raw.sort_values('date').reset_index(drop=True).copy()
    ratio = d.close.shift() / d.pre_close
    factor = ratio.where(ratio.gt(0) & np.isfinite(ratio), 1).fillna(1).cumprod()
    for c in ['open', 'high', 'low', 'close']:
        d[c] = d[c] * factor
    c, v = d.close, d.vol
    white = c.ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    yellow = c.rolling(60).mean()
    rsv = 100*(c-d.low.rolling(9).min())/(d.high.rolling(9).max()-d.low.rolling(9).min()).replace(0,np.nan)
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    j = 3*k-2*k.ewm(alpha=1/3,adjust=False).mean()
    ready = pd.Series(np.arange(len(d))>=250)
    trend = ready & white.gt(yellow) & c.gt(yellow)
    pullback = trend & c.le(white) & c.lt(c.shift()) & d.low.ge(d.low.shift().rolling(20).min())
    dry = pullback & v.lt(v.shift()) & v.lt(v.shift().rolling(5).mean())
    high_bear = ready & c.lt(d.open) & (c/d.open-1).le(-.04) & d.high.ge(d.high.shift().rolling(60).max()*.95)
    vr = v/v.shift().rolling(20).mean()
    masks = {
        'trend': trend, 'pullback': pullback, 'dry': dry,
        'b1_proxy': dry & j.lt(0),
        'b2_proxy': dry.shift().rolling(3).max().eq(1) & (c/c.shift()-1).gt(.04) & v.gt(v.shift()) & j.lt(55) & trend,
        's1_highvol': high_bear & vr.ge(2), 's1_other': high_bear & vr.lt(2),
    }
    for lookback in [20,60]:
        neck = d.high.shift().rolling(lookback).max()
        for threshold in [.04,0]:
            br = ready & c.gt(neck) & c.gt(d.open) & (c/c.shift()-1).gt(threshold) & vr.ge(1.5)
            bear = br.shift(fill_value=False) & c.lt(d.open)
            group = pd.cut(v/v.shift(), [0,.5,.75,1,np.inf],right=False, labels=['lt50','50to75','75to100','ge100'])
            tag = f'break{lookback}_ret{int(threshold*100)}'
            for label in ['lt50','50to75','75to100','ge100']:
                masks[f'{tag}_{label}'] = bear & group.eq(label)
            d[f'{tag}_neck'] = neck.shift()
    return d, masks


def extract(raw, symbol, verify=False):
    d, masks = features(raw)
    if verify and len(raw)>500:
        prefix, prefix_masks = features(raw.sort_values('date').iloc[:500])
        for name,mask in masks.items():
            np.testing.assert_array_equal(prefix_masks[name].to_numpy(),mask.iloc[:500].to_numpy())
        pd.testing.assert_frame_equal(prefix,d.iloc[:500].reset_index(drop=True))
    rows=[]
    for name,mask in masks.items():
        last=-10000
        for i in np.flatnonzero(mask.fillna(False).to_numpy()):
            if i-last<=20 or i+20>=len(d) or d.at[i,'date']<pd.Timestamp('2015-01-01'):
                continue
            last=i
            entry=d.at[i+1,'open']
            row={'symbol':symbol,'date':d.at[i,'date'],'rule':name,
                 'entry_date':d.at[i+1,'date'],'exit_date':d.at[i+20,'date'],
                 'ret5':d.at[i+5,'close']/entry-1,'ret20':d.at[i+20,'close']/entry-1,
                 'signal_close':d.at[i,'close'],'volume_ratio':d.at[i,'vol']/d.at[i-1,'vol']}
            if name.startswith('break'):
                tag=name.rsplit('_',1)[0]
                neck=d.at[i,f'{tag}_neck']
                row.update(neckline=neck,failed5=d.at[i+5,'close']<neck,failed20=d.at[i+20,'close']<neck)
            rows.append(row)
    return rows


def block_ci(g, column):
    a=g.groupby('date')[column].mean().dropna().to_numpy(float)
    if len(a)<20:return [None,None]
    rng=np.random.default_rng(20261019)
    starts=rng.integers(0,len(a),size=(1000,int(np.ceil(len(a)/20))))
    idx=(starts[:,:,None]+np.arange(20))%len(a)
    means=a[idx.reshape(1000,-1)[:,:len(a)]].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    design=OUT/'frozen_design.json'
    if design.exists():
        assert json.loads(design.read_text())==DESIGN
    else:design.write_text(json.dumps(DESIGN,ensure_ascii=False,indent=2))
    files=[f for f in sorted((ROOT/'data/raw/daily_partitioned').glob('*/data.parquet')) if '201401'<=f.parent.name.split('=')[1]<='202610']
    codes=set()
    for f in files:
        codes.update(pq.ParquetFile(f).read(columns=['ts_code']).column(0).unique().to_pylist())
    codes=[c for c in codes if c and ((c.endswith('.SH') and c[:1]=='6') or (c.endswith('.SZ') and c[:1] in ['0','3']))]
    selected=sorted(codes,key=lambda c:hashlib.sha256(c.encode()).hexdigest())[:800]
    (OUT/'sample_symbols.json').write_text(json.dumps(selected,indent=2))
    parts=[]
    columns=['ts_code','date','open','high','low','close','pre_close','vol']
    for f in files:
        t=pq.ParquetFile(f).read(columns=columns)
        t=t.filter(pc.is_in(t['ts_code'],value_set=__import__('pyarrow').array(selected)))
        parts.append(t.to_pandas())
    raw=pd.concat(parts,ignore_index=True)
    raw['date']=pd.to_datetime(raw.date).dt.normalize()
    raw=raw.loc[raw.date.le('2026-10-16')]
    assert not raw.duplicated(['ts_code','date']).any()
    assert raw[['open','high','low','close','pre_close']].gt(0).all().all()
    print('Loaded',len(raw),'rows',raw.ts_code.nunique(),'stocks',raw.date.min(),raw.date.max(),flush=True)
    rows=[]
    for n,(symbol,g) in enumerate(raw.groupby('ts_code',sort=True)):
        rows.extend(extract(g,symbol,verify=n<10))
        if n%200==0:print('Processed',n,flush=True)
    e=pd.DataFrame(rows)
    e.to_parquet(OUT/'events.parquet',index=False)
    e['period']=np.select([e.date.lt('2020-01-01'),e.date.lt('2024-01-01')],['2015-2019','2020-2023'],default='2024-2026')
    output=[]
    for period,f in [('all',e),*list(e.groupby('period'))]:
        for rule,g in f.groupby('rule'):
            row={'period':period,'rule':rule,'n':len(g),'stocks':g.symbol.nunique(),'dates':g.date.nunique()}
            for h in [5,20]:
                r=g[f'ret{h}']
                row.update({f'gross{h}':r.mean(),f'net{h}':r.mean()-.003,f'win{h}':(r>.003).mean(),f'median{h}':r.median()})
                if period=='all':
                    lo,hi=block_ci(g,f'ret{h}')
                    row.update({f'date_mean{h}':g.groupby('date')[f'ret{h}'].mean().mean(),f'date_ci_low{h}':lo,f'date_ci_high{h}':hi})
                if rule.startswith('break'):
                    row[f'failed{h}']=g[f'failed{h}'].astype(float).mean()
                    if period=='all':
                        lo,hi=block_ci(g,f'failed{h}')
                        row.update({f'fail_date_low{h}':lo,f'fail_date_high{h}':hi})
            output.append(row)
    summary=pd.DataFrame(output)
    summary.to_csv(OUT/'summary.csv',index=False)
    meta={'rows':len(raw),'universe_observed':len(codes),'stocks':raw.ts_code.nunique(),'data_start':str(raw.date.min().date()),'data_end':str(raw.date.max().date()),'last_event':str(e.date.max().date()),'events':len(e),'prefix_checks':10,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((OUT/'source.html').read_bytes()).hexdigest()}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(summary.loc[summary.period.eq('all'),['rule','n','net5','net20','win20','failed5','failed20']].to_string(index=False),flush=True)


if __name__=='__main__':main()
