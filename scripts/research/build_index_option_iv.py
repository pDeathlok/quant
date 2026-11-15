"""Reconstruct auditable 30-day index-option volatility from Tushare daily bars."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from quant.features.option_implied_volatility import term_volatility, interpolate_30

OUT=Path('data/research/index_futures/option_iv')
SOURCES=OUT/'sources'
END='20261106'
UNDERLYINGS={'IO':'IF','MO':'IM','HO':'IH'}


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(end=END,capital=500000,max_lots=1,
        downloaded='Tushare all opt_basic CFFEX/SSE with pagination to empty; CFFEX daily201912..20261106,SSE201502..20261106; Shibor2015..20261106. Immutable cached pages,duplicate/offset/date checks.',
        scope='Build IO/MO/HO European unadjusted index-option surfaces. SSE history retained and audited but excluded from self-IV features because latest adjusted ETF contract metadata cannot reconstruct point-in-time strikes safely. IC has commonIO factor only,own missing. No lookahead ETF adjustment reconstruction.',
        quality='Per date/expiry,listed<=date<=delist; code strike and C/P agree with metadata,positive trade volume/OI/price; close within daily high/low.7<=calendarDTE<=120. At least3 paired strikes. Forward=median of up to5 pairs with minimum abs(C-P); reject if forward90-10 spread >1% median. All rules chosen before strategy results.',
        rate='Same-date Shibor1w/2w/1m/3m/6m/1y simple annual rates linearly interpolated to DTE; r=log(1+simpleRate*DTE/365)/(DTE/365). Shibor is an unsecured rate proxy,not risk-free Treasury. No future rate filling.',
        variance='MF:OTM put belowK0,call aboveK0,mean paired prices atK0;2/T sum(deltaK/K^2 exp(rT)Q)-(F/K0-1)^2/T.>=3 OTM strikes each side,coverage at least2% either wing,max strike gap<=3 median grid steps,gross monotonicity tolerance0.4 index points,positive variance. Close not bid-ask midpoint; no zero-bid availability,so no claim of exactCboeVIX.',
        atm='Up to5 nearest paired strikes within5% of parity forward;invert discountedBlack76 OTM price,strict intrinsic/upper bounds;require3 finite IVs,take median,then square. No tail option requirement.',
        maturity='For each method select nearest valid maturities bracketing30calendar days,interpolate T*variance then divide30;exact30 uses one expiry. Never extrapolate or time-fill. Use actual metadata expiry date,close-to-close days/365.',
        sensitivity='Compute settlement-price MF/ATM as price-quality diagnostic only,not extra strategy arms or replacement selected by profits.',
        arms='baseline old85,state20/10/5; frozenQVIX93; native_mf93; native_atm93; qvix_mo93. Native commonIO,ownIO(IF)/HO(IH)/MO(IM),ICownmissing. qvix_mo keeps oldQVIX features except IMown replaced with MO MF. Same8 rank/change5/premium/missing features,252/min60;extra one session lag; no new interactions or threshold search.',
        fallback='Native common fully missing/insufficient history -> frozenQVIX forecast; native own missing -> training-only per-column imputation + flag. qvix_mo if IMown unavailable -> frozenQVIX forecast; if commonQVIX unavailable -> frozenQVIX forecast. Warmup<244maturedays -> frozenQVIX. Record all fallback rates.',
        models='same5session labels and85 original factors;quarterly separate long/short LR C=.1,Ridge alpha=.1*n,36m maturehistory,12m half-life. Original calendar,gates,mean5 protective exits,state horizon,reserves and fees unchanged.',
        replay='fixedbase first,quarterly4policy rolling second;exactcurrentequity inner122/244windows and old switching thresholds;500k,max1lot,no topups. Same2022..2026 monthly/Januaryphase grid,annual3/6bps x15/25margin,continuousannual,singleyear,toend,122/244,multi-startworst and opportunity attribution. Old+QVIX exact parity.',
        caveats='historical snapshots may be revised;close prices asynchronous and settlement model-based;not newOOT;overlappingwindows notindependent;missing is not zerofear;better source coverage is not guaranteed strategy improvement.')
    path=OUT/'protocol.json';text=json.dumps(protocol,ensure_ascii=False,indent=2)
    if path.exists() and path.read_text()!=text:raise RuntimeError('Frozen option-IV protocol changed')
    path.write_text(text)


def audit_downloads():
    rows=[];digests=[]
    calendar=set(pd.read_parquet(OUT.parent/'IF_spot.parquet').query('trade_date <= @END').trade_date)
    calendar_checks=[]
    for exchange in ('CFFEX','SSE'):
        basic=pd.read_parquet(SOURCES/f'contracts_{exchange}.parquet')
        frames=[pd.read_parquet(p) for p in sorted(SOURCES.glob(f'daily_{exchange}_*.parquet'))]
        daily=pd.concat(frames,ignore_index=True)
        assert not daily.duplicated(['ts_code','trade_date']).any()
        merged=daily.merge(basic[['ts_code','list_date','delist_date','opt_code']],on='ts_code',how='left',validate='many_to_one')
        merged['year']=merged.trade_date.str[:4]
        merged['bad_membership']=merged.list_date.isna()|merged.trade_date.lt(merged.list_date)|merged.trade_date.gt(merged.delist_date)
        if exchange=='CFFEX':
            for product in UNDERLYINGS:
                first=basic.loc[basic.ts_code.str.startswith(product),'list_date'].min()
                actual=set(daily.loc[daily.ts_code.str.startswith(product),'trade_date'])
                expected={date for date in calendar if date>=first}
                calendar_checks.append(dict(product=product,expected=len(expected),actual=len(actual),missing=sorted(expected-actual),off_calendar=sorted(actual-expected)))
        for (code,year),g in merged.groupby(['opt_code','year'],dropna=False):
            rows.append(dict(exchange=exchange,underlying=code,year=year,rows=len(g),days=g.trade_date.nunique(),contracts=g.ts_code.nunique(),bad_membership=int(g.bad_membership.sum()),zero_volume=float(g.vol.le(0).mean())))
    for path in sorted(SOURCES.glob('*.parquet')):
        digests.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(OUT/'history_coverage.csv',index=False)
    (OUT/'source_hashes.json').write_text(json.dumps(digests,indent=2))
    (OUT/'trading_calendar_audit.json').write_text(json.dumps(calendar_checks,indent=2))


def build():
    register();audit_downloads()
    contracts=pd.read_parquet(SOURCES/'contracts_CFFEX.parquet')
    parts=contracts.ts_code.str.extract(r'^(IO|MO|HO)(\d{4})-([CP])-(\d+(?:\.\d+)?)\.CFX$')
    contracts=contracts.assign(product=parts[0],code_side=parts[2],code_strike=pd.to_numeric(parts[3],errors='coerce'))
    contracts['metadata_valid']=contracts['product'].notna()&contracts.code_side.eq(contracts.call_put)&np.isclose(contracts.code_strike,contracts.exercise_price)&contracts.exercise_type.eq('欧式')
    contracts[~contracts.metadata_valid].to_csv(OUT/'invalid_contracts.csv',index=False)
    daily=pd.concat([pd.read_parquet(p) for p in sorted(SOURCES.glob('daily_CFFEX_*.parquet'))],ignore_index=True)
    chain=daily.merge(contracts[['ts_code','product','call_put','exercise_price','list_date','delist_date','maturity_date','metadata_valid']],on='ts_code',how='left',validate='many_to_one')
    chain=chain.rename(columns={'exercise_price':'strike'})
    chain['days']=(pd.to_datetime(chain.maturity_date)-pd.to_datetime(chain.trade_date)).dt.days
    chain['eligible']=chain.metadata_valid.fillna(False)&chain.trade_date.ge(chain.list_date)&chain.trade_date.le(chain.delist_date)&chain.days.between(7,120)
    chain=chain[chain.eligible].copy()
    rates=pd.read_parquet(SOURCES/'shibor.parquet').set_index('date')
    rates.index=rates.index.str.replace('-','',regex=False)
    assert rates.index.is_unique
    tenors=np.array([7,14,30,90,180,365])
    ratecols=['1w','2w','1m','3m','6m','1y']
    records=[]
    for (date,product,expiry),g in chain.groupby(['trade_date','product','maturity_date'],sort=True):
        days=int(g.days.iloc[0]);rate=np.nan
        if date in rates.index:
            quotes=rates.loc[date,ratecols].to_numpy(dtype=float)/100
            valid=np.isfinite(quotes)
            if valid.sum()>=2:
                simple=float(np.interp(days,tenors[valid],quotes[valid]))
                rate=float(np.log1p(simple*days/365)/(days/365))
        for price in ('close','settle'):
            result=term_volatility(g,days,rate,price)
            records.append(dict(trade_date=date,product=product,expiry=expiry,days=days,rate=rate,price=price,**result))
        if len(records)%2000==0:print('option terms',date,len(records),flush=True)
    terms=pd.DataFrame(records);terms.to_parquet(OUT/'term_audit.parquet',index=False)
    indices=[]
    for (date,product,price),g in terms.groupby(['trade_date','product','price']):
        row=dict(trade_date=date,product=product,symbol=UNDERLYINGS[product],price=price)
        for method in ('mf','atm'):
            row.update({f'{method}_{k}':v for k,v in interpolate_30(g,f'{method}_variance').items()})
        indices.append(row)
    indices=pd.DataFrame(indices)
    indices.to_parquet(OUT/'indices.parquet',index=False)
    indices.to_csv(OUT/'indices.csv',index=False)
    indices['year']=indices.trade_date.str[:4]
    indices.groupby(['year','product','price']).agg(days=('trade_date','size'),mf_days=('mf_value','count'),atm_days=('atm_value','count'),mf_median=('mf_value','median'),atm_median=('atm_value','median')).to_csv(OUT/'iv_coverage.csv')
    terms.assign(year=terms.trade_date.str[:4]).groupby(['year','product','price','reason']).size().rename('terms').to_csv(OUT/'quality_rejections.csv')
    print('IV built',len(indices),'daily rows',len(terms),'term rows',flush=True)


if __name__=='__main__':build()
