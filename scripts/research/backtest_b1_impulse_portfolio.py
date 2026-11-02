"""Simple executable cash/position accounting for impulse study, no production stops."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from quant.features.variable_library import build_continuous_ohlc
OUT=ROOT/'reports/research/b1_impulse_mainline_20261004'
START=pd.Timestamp('2025-05-06');END=pd.Timestamp('2026-03-27')


def simulate(signals,prices):
    cash=1.;holdings={};navs=[];trades=[]
    entries={d:g for d,g in signals.groupby('entry_date',sort=False)}
    for date,marks in prices.iterrows():
        # Last observed close marks suspended positions; sell only on an actual close.
        for symbol,pos in list(holdings.items()):
            if pd.notna(marks.get(symbol,np.nan)):
                pos['last']=float(marks[symbol]);pos['bars']+=1
            if pos['bars']>=5 and pd.notna(marks.get(symbol,np.nan)):
                assert date==pos['exit']
                proceeds=pos['qty']*pos['last'];cash+=proceeds*.999
                trades.append(dict(symbol=symbol,signal=pos['signal'],entry=pos['entry'],exit=date,net_return=proceeds*.999/pos['spent']-1))
                del holdings[symbol]
        value=cash+sum(p['qty']*p['last'] for p in holdings.values())
        size=value/100 # 20 entries/day x 5 holding intervals, each <=1% NAV.
        if date in entries:
            for row in entries[date].itertuples():
                if row.symbol in holdings or len(holdings)>=100:continue
                price=marks.get(row.symbol,np.nan)
                if not np.isfinite(price) or price<=0:continue
                spent=min(size,cash)
                if spent<1e-12:break
                cash-=spent
                holdings[row.symbol]=dict(qty=spent/(1.001*price),last=price,spent=spent,signal=row.date,entry=date,exit=row.exit_date,bars=0)
        nav=cash+sum(p['qty']*p['last'] for p in holdings.values())
        assert cash>=-1e-10 and nav>0 and len(holdings)<=100
        navs.append(dict(date=date,nav=nav,cash=cash,exposure=1-cash/nav,positions=len(holdings)))
    assert not holdings, 'Unmatured positions at end'
    return pd.DataFrame(navs),pd.DataFrame(trades)


def main():
    selected=pd.read_parquet(OUT/'selected_events.parquet')
    variants=['current_B1','current_B1_mainline_priority','add_straight_pullback','add_dry_volume','five_conditions_score100','five_conditions_mainline_score100','five_conditions_volume_score','mainline_priority_volume','mainline_four_conditions','mainline_three_conditions','four_conditions_mainline_priority','four_conditions_volume']
    selected=selected.loc[selected.horizon.eq(5)&selected.variant.isin(variants)].copy()
    # Harmonize outcome columns across frozen baseline and reconstructed candidates.
    selected['exit_date']=selected.exit_date.fillna(selected.exit5)
    selected=selected.loc[selected.exit_date.le(END)]
    cache=OUT/'portfolio_prices.parquet'
    parts=[];symbols=set(selected.symbol)
    for p in sorted((ROOT/'data/raw/daily_partitioned').glob('year_month=*/data.parquet')):
        if '202401'<=p.parent.name.split('=')[1]<='202604':
            f=pd.read_parquet(p,columns=['symbol','date','open','high','low','close','pre_close']);parts.append(f.loc[f.symbol.isin(symbols)])
    raw=pd.concat(parts,ignore_index=True);raw.date=pd.to_datetime(raw.date).dt.normalize()
    marks=[]
    for symbol,f in raw.groupby('symbol',sort=False):
        f=f.sort_values('date');p=build_continuous_ohlc(f)
        q=pd.DataFrame({'symbol':symbol,'date':f.date,'price':p.close})
        marks.append(q.loc[q.date.between(START,END)])
    prices=pd.concat(marks).pivot(index='date',columns='symbol',values='price').sort_index();prices.to_parquet(cache)
    rows=[];curves=[];tradeframes=[]
    for name in variants:
        nav,trades=simulate(selected.loc[selected.variant.eq(name)],prices)
        nav['variant']=name;trades['variant']=name;curves.append(nav);tradeframes.append(trades)
        for year in [2025,2026,'full']:
            q=nav if year=='full' else nav.loc[nav.date.dt.year.eq(year)]
            prev=nav.loc[nav.date<q.date.min()].tail(1)
            initial=float(prev.nav.iloc[0]) if len(prev) else 1.
            path=pd.Series(np.r_[initial,q.nav.to_numpy()]);dd=path/path.cummax()-1
            rows.append(dict(variant=name,year=year,return_net=q.nav.iloc[-1]/initial-1,max_drawdown=dd.min(),mean_exposure=q.exposure.mean(),end_nav=q.nav.iloc[-1],closed_trades=len(trades) if year=='full' else int(trades.exit.dt.year.eq(year).sum()) if len(trades) else 0))
    pd.DataFrame(rows).to_csv(OUT/'portfolio_metrics.csv',index=False)
    pd.concat(curves).to_csv(OUT/'portfolio_nav.csv',index=False)
    pd.concat(tradeframes).to_csv(OUT/'portfolio_trades.csv',index=False)
    (OUT/'portfolio_manifest.json').write_text(json.dumps(dict(status='success',cost='0.1% each side',position='at most20 original selected entries/day, 1% current NAV each, <=100 concurrent unique symbols; repeated held-stock signals skipped without refill',exit='entry close +5 stock sessions close; held across year boundary; mature by2026-03-27',checks=['nonnegative cash','positive NAV','position capacity','all positions exited at end'],limitations=['not production stable/aggressive stops','fractional shares; no lots, limit-price/volume execution or historicalST eligibility','continuous prices mark total-return proxy; volume unchanged','mainline filtering leaves cash; exposure materially differs']),ensure_ascii=False,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
