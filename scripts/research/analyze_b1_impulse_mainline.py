"""Causal screenshot impulse/pullback B1 study, independent of frozen J<0 pool."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from quant.features.variable_library import build_continuous_ohlc
from quant.data.factors import KDJ
OUT=ROOT/'reports/research/b1_impulse_mainline_20261004'
PREV=ROOT/'reports/research/b1_mainline_leaders_20261004'
START=pd.Timestamp('2025-05-06'); END=pd.Timestamp('2026-03-27')
FACTORS=['j_depth','dry_volume','impulse_energy','wave_gain','pullback_shape','pullback_depth','ma60_position','sector_count']


def features(f):
    f=f.sort_values('date').reset_index(drop=True)
    p=build_continuous_ohlc(f)
    c,o,h,l=[p[x].to_numpy(dtype=float) for x in ['close','open','high','low']]
    v=f.volume.to_numpy(dtype=float)
    vol20=f.volume.shift(1).rolling(20,min_periods=20).mean().to_numpy()
    ma60=p.close.rolling(60,min_periods=60).mean().to_numpy()
    j=KDJ().compute(p).J.to_numpy()
    ret=np.r_[np.nan,c[1:]/c[:-1]-1]
    impulse=(v>=1.6*vol20)&(ret>=.05)&(c>o)&(v>0)
    rows=[]
    for t in np.flatnonzero((f.date>=START)&(f.date<=END)&(j<13)&(v>0)):
        anchors=np.flatnonzero(impulse[max(0,t-40):t])+max(0,t-40)
        if not len(anchors):continue
        a=int(anchors[-1]) # Latest impulse; no return-driven anchor search.
        if a==0:continue
        peak=a+int(np.argmax(c[a:t+1]))
        age=t-peak
        if age<1:continue
        pb=c[peak+1:t+1]
        rebound=np.max(pb/np.minimum.accumulate(pb)-1)
        straight=age<=10 and rebound<=.05 and c[t]>c[a-1]
        draw=1-c[t]/c[peak]
        vr=v[t]/v[a]; shrink=np.mean(v[peak+1:t+1])/v[a]
        dry=v[t]<=1.2*np.min(v[a+1:t+1]) and vr<.8
        bias=c[t]/ma60[t]-1
        near=-.15<=bias<=.05
        wave=c[peak]/c[a-1]-1
        energy=v[a]/vol20[a]
        scores=[20 if j[t]<=5 else 16 if j[t]<=9 else 12,
                15 if vr<.3 else 11 if vr<.5 else 6 if vr<.8 else 0,
                10 if energy>=2.5 else 8 if energy>=2 else 5,
                10 if wave>=.15 else 7 if wave>=.10 else 4 if wave>=.07 else 0,
                15 if shrink<.4 else 10 if shrink<.6 else 5 if shrink<.8 else 0,
                12 if .10<=draw<=.20 else 8 if .08<=draw<.10 or .20<draw<=.25 else 4 if .05<=draw<.08 else 0,
                10 if abs(bias)<=.05 else 6 if -.15<=bias<-.05 else 0]
        row=dict(symbol=f.symbol.iloc[0],date=f.date.iloc[t],j=j[t],impulse_date=f.date.iloc[a],peak_date=f.date.iloc[peak],pullback_bars=age,rebound=rebound,volume_ratio=vr,pullback_volume_ratio=shrink,impulse_multiple=energy,wave_return=wave,drawdown=draw,ma60_bias=bias,straight=straight,dry=dry,near_ma60=near,technical_gate=straight and dry and near)
        row.update(dict(zip(FACTORS[:7],scores)))
        for horizon in [5,10]:
            x=t+1+horizon
            row[f'return{horizon}']=c[x]/c[t+1]-1 if x<len(f) else np.nan
            row[f'exit{horizon}']=f.date.iloc[x] if x<len(f) else pd.NaT
        row['entry_date']=f.date.iloc[t+1] if t+1<len(f) else pd.NaT
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files=[p for p in sorted((ROOT/'data/raw/daily_partitioned').glob('year_month=*/data.parquet')) if '202401'<=p.parent.name.split('=')[1]<='202604']
    columns=['symbol','date','open','high','low','close','pre_close','volume']
    raw=pd.concat([pd.read_parquet(p,columns=columns) for p in files],ignore_index=True)
    raw.date=pd.to_datetime(raw.date).dt.normalize()
    assert not raw.duplicated(['symbol','date']).any()
    calendar=np.sort(raw.date.unique()); groups=raw.groupby('symbol',sort=True)
    chunks=[]; causal_checked=False
    for n,(symbol,f) in enumerate(groups,1):
        result=features(f)
        if len(result):chunks.append(result)
        if not causal_checked and len(result)>20:
            cutoff=pd.Timestamp('2025-10-31')
            prefix=features(f.loc[f.date.le(cutoff)])
            compare=[x for x in result.columns if not x.startswith('return') and not x.startswith('exit') and x!='entry_date']
            pd.testing.assert_frame_equal(prefix[compare].reset_index(drop=True),result.loc[result.date.le(cutoff),compare].reset_index(drop=True))
            causal_checked=True
        if n%1000==0:print(f'Features {n}/{groups.ngroups}',flush=True)
    candidates=pd.concat(chunks,ignore_index=True)
    assert causal_checked
    ix=np.searchsorted(calendar,candidates.date.to_numpy(),side='right')
    candidates=candidates.loc[candidates.entry_date.eq(calendar[np.minimum(ix,len(calendar)-1)])].copy()
    candidates['year']=candidates.date.dt.year
    members=pd.read_csv(PREV/'historical_members.csv'); members['year']=pd.to_datetime(members.snapshot_date).dt.year+1
    mainlines=pd.read_parquet(PREV/'daily_mainlines.parquet')
    mapped=candidates[['symbol','date','year']].merge(members[['symbol','year','index_code']],on=['symbol','year'],how='inner')
    counts=mapped.groupby(['date','index_code']).symbol.nunique().rename('candidate_count').reset_index()
    counts['sector_rank']=counts.groupby('date').candidate_count.rank(method='min',ascending=False)
    mapped=mapped.merge(counts,on=['date','index_code']).merge(mainlines[['date','index_code','mainline','mainline_score']],on=['date','index_code'],validate='many_to_one')
    mapped['sector_count']=np.where(mapped.sector_rank.eq(1),8,np.where(mapped.sector_rank.le(5),4,2))
    context=mapped.groupby(['symbol','date']).agg(mainline=('mainline','any'),sector_count=('sector_count','max'),mainline_score=('mainline_score','max')).reset_index()
    candidates=candidates.merge(context,on=['symbol','date'],how='left',validate='one_to_one')
    candidates['mainline']=candidates.mainline.fillna(False).astype(bool)
    candidates['sector_count']=candidates.sector_count.fillna(2)
    candidates['score100']=candidates[FACTORS].sum(axis=1)
    # Fixed hypotheses, no fitted coefficients: keep the three positive directions stated in screenshot.
    candidates['volume_score']=100*candidates[['dry_volume','impulse_energy','pullback_shape']].sum(axis=1)/40
    registry=pd.read_csv(PREV/'leader_evidence.csv')
    for col in ['available_at','valid_until']:registry[col]=pd.to_datetime(registry[col])
    evidence=mapped.merge(registry,on=['symbol','index_code'],how='inner')
    evidence=evidence.loc[evidence.date.gt(evidence.available_at)&evidence.date.le(evidence.valid_until)&evidence.mainline]
    leaderkeys=evidence[['symbol','date']].drop_duplicates().assign(leader_mainline=True)
    candidates=candidates.merge(leaderkeys,on=['symbol','date'],how='left',validate='one_to_one')
    candidates['leader_mainline']=candidates.leader_mainline.fillna(False).astype(bool)
    candidates.to_parquet(OUT/'candidates.parquet',index=False)
    mapped.to_parquet(OUT/'candidate_theme_context.parquet',index=False)
    mainlines.loc[mainlines.date.between(START,END)].to_csv(OUT/'historical_mainlines.csv',index=False)
    baseline=pd.read_parquet(PREV/'candidate_context.parquet')
    gate=candidates.technical_gate
    variants={
      'current_B1':None,
      'current_B1_mainline_priority':None,
      'impulse_J13_score100':(pd.Series(True,index=candidates.index),'score100',False),
      'add_straight_pullback':(candidates.straight,'score100',False),
      'add_dry_volume':(candidates.straight&candidates.dry,'score100',False),
      'five_conditions_score100':(gate,'score100',False),
      'five_conditions_mainline_score100':(gate&candidates.mainline,'score100',False),
      'five_conditions_volume_score':(gate,'volume_score',False),
      'five_conditions_mainline_volume':(gate&candidates.mainline,'volume_score',False),
      'mainline_priority_volume':(gate,'volume_score',True),
      'mainline_volume_J0':(gate&candidates.mainline&candidates.j.lt(0),'volume_score',False),
      'mainline_volume_leader':(gate&candidates.leader_mainline,'volume_score',False),
      'mainline_four_conditions':(candidates.straight&candidates.dry&candidates.mainline,'volume_score',False),
      'mainline_three_conditions':(candidates.straight&candidates.mainline,'volume_score',False),
      'four_conditions_mainline_priority':(candidates.straight&candidates.dry,'volume_score',True),
      'four_conditions_volume':(candidates.straight&candidates.dry,'volume_score',False),
    }
    rng=np.random.default_rng(20261004); metric=[]; events=[]
    for horizon in [5,10]:
      for year in [2025,2026]:
        eligible=candidates.date.dt.year.eq(year)&candidates[f'exit{horizon}'].le(END)&candidates[f'return{horizon}'].notna()
        local=candidates.loc[eligible].copy()
        dates=pd.Index(sorted(baseline.loc[baseline.date.dt.year.eq(year),'date'].unique()))
        # Same matured signal dates for both universes/horizons.
        dates=dates[dates<=calendar[np.searchsorted(calendar,np.datetime64(END),side='right')-horizon-2]]
        local=local.loc[local.date.isin(dates)]
        base=None
        for name,spec in variants.items():
          if spec is None:
            b=baseline.loc[baseline.date.isin(dates)]
            sortcols=['date','pred_unified_left_long_task_deep','symbol']; sortasc=[True,False,True]
            if name=='current_B1_mainline_priority':sortcols.insert(1,'mainline');sortasc.insert(1,False)
            s=b.sort_values(sortcols,ascending=sortasc).groupby('date').head(20).copy()
            if horizon==5:s['net_return']=s.terminal_return-.002
            else:
              # Frozen pool has only five-day outcome: rebuild ten-day returns separately below.
              lookup=outcomes10(raw,s[['symbol','date']],calendar)
              s=s.merge(lookup,on=['symbol','date'],validate='one_to_one');s['net_return']=s.return10-.002
            s=s.loc[s.net_return.notna()]
          else:
            mask,score,priority=spec
            s=local.loc[mask.reindex(local.index)].copy()
            cols=['date',score,'symbol'];asc=[True,False,True]
            if priority:cols.insert(1,'mainline');asc.insert(1,False)
            s=s.sort_values(cols,ascending=asc).groupby('date').head(20).copy()
            s['net_return']=s[f'return{horizon}']-.002
          s['variant']=name;s['horizon']=horizon;events.append(s)
          slots=s.groupby('date').net_return.sum().reindex(dates,fill_value=0)/20
          if base is None:base=slots
          delta=(slots-base).to_numpy();draws=[]
          for _ in range(1000):
            starts=rng.integers(0,len(delta),size=int(np.ceil(len(delta)/20)))
            indices=np.concatenate([(a+np.arange(20))%len(delta) for a in starts])[:len(delta)]
            draws.append(delta[indices].mean())
          net=s.net_return;loss=-net.loc[net<0].sum()
          metric.append(dict(year=year,horizon=horizon,variant=name,trades=len(s),stocks=s.symbol.nunique(),signal_days=len(dates),active_days=s.date.nunique(),avg_net=net.mean(),win_rate=net.gt(0).mean(),profit_factor=net.loc[net>0].sum()/loss if loss else np.nan,fill_rate=len(s)/(20*len(dates)),slot_return=slots.mean(),delta_slot=delta.mean(),ci_low=np.quantile(draws,.025),ci_high=np.quantile(draws,.975)))
    metrics=pd.DataFrame(metric);metrics.to_csv(OUT/'metrics.csv',index=False)
    pd.concat(events,ignore_index=True).to_parquet(OUT/'selected_events.parquet',index=False)
    ics=[]
    for universe,mask in [('impulse_J13',pd.Series(True,index=candidates.index)),('five_conditions',gate)]:
      for year in [2025,2026]:
        q=candidates.loc[mask&candidates.year.eq(year)&candidates.exit5.le(END)].copy()
        for factor in FACTORS:
          daily=[]
          for date,g in q.groupby('date'):
            if len(g)>=10 and g[factor].nunique()>1:
              daily.append(g[factor].rank().corr(g.return5.rank()))
          ics.append(dict(universe=universe,year=year,factor=factor,days=len(daily),rank_ic=np.mean(daily) if daily else np.nan))
    pd.DataFrame(ics).to_csv(OUT/'factor_ic.csv',index=False)
    checks=['unique candidate keys','strict impulse_date <= peak_date < signal_date','J<13','score100 in [0,100]','prefix invariance of technical features','entry next market session','historical membership only subsequent year','leader after disclosure','Top20 capacity']
    assert not candidates.duplicated(['symbol','date']).any()
    assert (candidates.impulse_date<=candidates.peak_date).all() and (candidates.peak_date<candidates.date).all()
    assert candidates.j.lt(13).all() and candidates.score100.between(0,100).all()
    assert pd.concat(events).groupby(['variant','horizon','date']).size().le(20).all()
    manifest=dict(status='success',start=str(START.date()),last_exit=str(END.date()),candidate_rows=len(candidates),technical_rows=int(gate.sum()),mainline_rows=int((gate&candidates.mainline).sum()),universe='all stocks in local daily bars, no frozen J<0 restriction',checks=checks,model='none; screenshot score and predefined three-volume-factor score',parameters={'impulse':'latest impulse in previous40 stock bars; volume>=1.6 previous20bar avg, return>=5%, bullish candle','pullback':'peak close after impulse, age1..10, maximum rebound from running low<=5%, close>pre-impulse close','dry':'current volume<=1.2 minimum volume strictly after impulse, current/impulse volume<.8','position':'close/MA60-1 in [-15%,+5%]','J':'project KDJ9/3/3 J<13'},limitations=['ten-theme mainline coverage only; fixed previous-year-end members','historical leadership registry only11companies','heuristic translation of qualitative N pattern, not unique screenshot interpretation','no independently untouched holdout; many exploratory comparisons','next-close event study; not production portfolio NAV/stoploss/execution','no historical ST/delisting/limit-up/suspension-exit filter; full raw-price universe is not production tradability universe','same stocks can repeat on adjacent days; overlapping outcome windows','volume is raw share-volume, possible share-count adjustment effects'])
    manifest['hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),PREV/'daily_mainlines.parquet',PREV/'historical_members.csv']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(metrics.loc[metrics.horizon.eq(5)].to_string(index=False),flush=True)


def outcomes10(raw,keys,calendar):
    parts=[]
    keygroups={symbol:k for symbol,k in keys.groupby('symbol')}
    for symbol,f in raw.loc[raw.symbol.isin(keygroups)].groupby('symbol',sort=False):
        k=keygroups[symbol]
        f=f.sort_values('date')
        c=build_continuous_ohlc(f).close
        q=pd.DataFrame({'symbol':symbol,'date':f.date,'return10':c.shift(-11)/c.shift(-1)-1,'exit10':f.date.shift(-11)})
        q.loc[q.exit10.gt(END),'return10']=np.nan
        parts.append(k.merge(q,on=['symbol','date'],how='left'))
    return pd.concat(parts,ignore_index=True)

if __name__=='__main__':main()
