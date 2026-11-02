"""Return-seeking BYD expert bank and past-only constrained policy selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from quant.research.byd_allocation import signal_from_snapshot
from quant.research.byd_wear import wear_metrics


def growth_experts(market: pd.DataFrame, legacy: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    if len(market) != len(legacy):
        raise ValueError('Unaligned legacy experts')
    signal = signal_from_snapshot(market)
    close = signal.close
    sigma = (market.close/market.pre_close-1).rolling(20).std().fillna(0)
    mean20,mean60 = close.rolling(20).mean(),close.rolling(60).mean()
    # Fixed thresholds; probabilities are smooth scores, not calibrated forecasts.
    fast = expit((close/mean20-1)/.04)
    slow = expit((close/mean60-1)/.08)
    z = ((close-mean20)/close.rolling(20).std()).fillna(0)
    rebound = expit(close.pct_change(5)/.04)
    states = pd.DataFrame({'up_score':slow,'fast_score':fast,'sigma':sigma},index=market.index).fillna(.5)
    base = pd.DataFrame(index=market.index)
    for core,tilt in ((.35,.25),(.45,.25),(.45,.40)):
        base[f'soft_core{int(core*100)}_tilt{int(tilt*100)}'] = (core+tilt*(2*slow-1)).clip(0,1)
    base['fast_slow_core'] = (.40+.20*(2*slow-1)+.15*(2*fast-1)).clip(.05,.85)
    base['recovery_core'] = (.35+.20*(2*slow-1)+.20*rebound).clip(.10,.80)
    base['range_core'] = (.45-.10*z+.10*(2*slow-1)).clip(.15,.80)
    base['mixed_core'] = (base.fast_slow_core+base.range_core)/2
    raw_names=['original_range','reversal_5d','trend_120','trend_250d','relative_63d','ridge_21d','ridge_63d','tree_21d']
    base = pd.concat([base,legacy[raw_names]],axis=1)
    targets = pd.DataFrame({'cash':np.zeros(len(market))},index=market.index)
    groups = {}
    for gain in (1.2,1.5):
        suffix=f'x{int(gain*10)}'
        for name in base:
            targets[f'{name}_{suffix}'] = (gain*base[name]).clip(0,1)
        groups[f'legacy_{suffix}']=['cash']+[f'{name}_{suffix}' for name in raw_names[:5]]
        groups[f'growth_{suffix}']=['cash']+[f'{name}_{suffix}' for name in base.columns[:7]]
        groups[f'mixed_{suffix}']=['cash']+[f'{name}_{suffix}' for name in base.columns]
    # Unscaled direct rules also support the first optimization round.
    for name in base.columns[:7]:
        targets[name]=base[name]
    targets=targets.fillna(0).clip(0,1)
    targets.iloc[:269]=0
    return targets,groups,states


def constrained_choice(ledgers: dict[str,pd.DataFrame], year: int, keys: list[str]) -> tuple[str,list[dict]]:
    if year<2016 or not keys:
        raise ValueError('Three completed calendar years required')
    rows=[]
    for name in keys:
        ledger=ledgers[name]
        past=ledger[(ledger.date.dt.year>=year-3)&(ledger.date.dt.year<year)]
        if len(past)<600:
            raise ValueError('Incomplete prior validation window')
        m=wear_metrics(past)
        annual=[wear_metrics(past[past.date.dt.year.eq(y)])['cagr'] for y in range(year-3,year)]
        dd,vol,pos=abs(m['max_drawdown']),m['realized_volatility'],m['average_position']
        violation=max(0.,dd-.25)+max(0.,vol-.25)+max(0.,.25-pos)
        score=m['cagr']-.15*np.std(annual)
        rows.append({'year':year,'name':name,'past_cagr':m['cagr'],'past_drawdown':dd,
                     'past_volatility':vol,'past_position':pos,'violation':violation,
                     'feasible':violation<1e-12,'score':float(score)})
    best=max(rows,key=lambda r:(r['feasible'],-r['violation'],r['score']))
    for row in rows:
        row['selected']=row['name']==best['name']
    return best['name'],rows


def paired_return_bootstrap(a: pd.DataFrame, b: pd.DataFrame, name: str) -> dict:
    """Paired circular 20-day blocks on exactly the supplied evaluation dates."""
    aa=a.set_index('date')['return'];bb=b.set_index('date')['return']
    if not aa.index.equals(bb.index) or len(aa)==0:
        raise ValueError('Bootstrap dates differ or are empty')
    delta=(aa-bb).to_numpy();rng=np.random.default_rng(20261013)
    starts=rng.integers(0,len(delta),size=(2000,int(np.ceil(len(delta)/20))))
    indices=((starts[:,:,None]+np.arange(20))%len(delta)).reshape(2000,-1)[:,:len(delta)]
    low,high=np.quantile(delta[indices].mean(axis=1)*252,[.025,.975])
    return {'comparator':name,'annual_mean_difference':float(delta.mean()*252),
            'low95':float(low),'high95':float(high),'sessions':len(delta),
            'actual_start':str(aa.index[0].date()),'actual_end':str(aa.index[-1].date())}
