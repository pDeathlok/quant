"""Small tilts around equal allocation and benchmark-relative annual selection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from quant.research.domestic_basket import GROUPS
from quant.research.domestic_portfolio import Panel,portfolio_metrics


def bounded_budget(values: np.ndarray, available: np.ndarray, budget: float=1.) -> np.ndarray:
    """Scale positive six-basket preferences to a budget subject to twenty percent."""
    x=np.maximum(np.nan_to_num(values),0.)*available
    out=np.zeros(6);active=x>0;remaining=min(budget,float(active.sum())*.20)
    while active.any() and remaining>1e-12:
        allocation=remaining*x[active]/x[active].sum()
        indices=np.flatnonzero(active);full=allocation>.20+1e-12
        if not full.any():out[indices]=allocation;break
        capped=indices[full];out[capped]=.20;remaining-=.20*len(capped);active[capped]=False
    return out


def to_assets(panel: Panel, basket: np.ndarray) -> np.ndarray:
    out=np.zeros_like(panel.data['close']);eligible=panel.data['eligible'].astype(bool)
    for t in range(len(basket)):
        for g,group in enumerate(GROUPS):
            assets=np.array(group);usable=assets[eligible[t,assets]]
            if len(usable):out[t,usable]=basket[t,g]/len(usable)
    return out


def basket_scores(panel: Panel, asset_scores: np.ndarray) -> np.ndarray:
    score=np.zeros((len(panel.dates),6));eligible=panel.data['eligible'].astype(bool)
    for g,group in enumerate(GROUPS):
        a=list(group);valid=eligible[:,a]
        score[:,g]=(asset_scores[:,a]*valid).sum(axis=1)/np.maximum(valid.sum(axis=1),1)
    return score


def risk_guard(panel: Panel, weights: np.ndarray, multiplier: float) -> np.ndarray:
    f=panel.features;idx={n:i for i,n in enumerate(panel.feature_names)}
    weak=(f[:,1,idx['distance_120']]<-.03)&(f[:,1,idx['ret_20']]<-.06)
    result=weights.copy()
    for t in np.flatnonzero(weak):
        result[t,[0,1,2,5,6]]*=multiplier
        for a in (3,4):
            if panel.data['eligible'][t,a]:result[t,a]+=min(.20-result[t,a],1-result[t].sum())
    return result


def build_targets(panel: Panel, forecasts: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    n=len(panel.dates);f=panel.features;idx={k:i for i,k in enumerate(panel.feature_names)}
    available=np.stack([panel.data['eligible'][:,g].any(axis=1) for g in GROUPS],axis=1)
    base=np.tile(np.full(6,1/6),(n,1))
    targets={'equal':to_assets(panel,base)}
    for g in range(6):
        w=np.full((n,6),.20);w[:,g]=0;targets[f'omit_{g}']=to_assets(panel,w)
    targets['low_securities_core']=to_assets(panel,np.tile([.20,.15,.20,.20,.05,.20],(n,1)))
    for h in (63,120,250):
        signal=f[:,:,idx[f'ret_{h}']] / (.15 if h==63 else .30 if h==120 else .50)
        signal+=f[:,:,idx['distance_120']]/.12
        score=np.tanh(basket_scores(panel,signal))
        for amplitude in (.03,.06):
            proposal=base+amplitude*(score-score.mean(axis=1,keepdims=True))
            baskets=np.stack([bounded_budget(proposal[t],available[t],float(base[t,available[t]].sum())) for t in range(n)])
            targets[f'tilt{h}_{int(amplitude*100)}']=to_assets(panel,baskets)
    for name in ('ridge_21','ridge_excess21','tree_excess21'):
        score=np.tanh(basket_scores(panel,forecasts[name])/.001)
        proposal=base+.03*(score-score.mean(axis=1,keepdims=True))
        baskets=np.stack([bounded_budget(proposal[t],available[t],float(base[t,available[t]].sum())) for t in range(n)])
        targets[f'model_tilt_{name}']=to_assets(panel,baskets)
    for name in ('equal','tilt120_3'):
        for multiplier in (.70,.40):targets[f'{name}_guard{int(multiplier*100)}']=risk_guard(panel,targets[name],multiplier)
    for name,path in targets.items():
        if (path<0).any() or (path.sum(axis=1)>1+1e-9).any():raise RuntimeError(name)
        for g in GROUPS:
            if (path[:,g].sum(axis=1)>.20+1e-9).any():raise RuntimeError('Basket cap')
    return targets


def relative_choice(ledgers: dict[str,pd.DataFrame], year: int) -> tuple[str,list[dict]]:
    past={k:f[(f.date.dt.year>=year-3)&(f.date.dt.year<year)] for k,f in ledgers.items()}
    base=portfolio_metrics(past['expert_equal']);rows=[]
    for name,frame in past.items():
        m=portfolio_metrics(frame);delta=m['cagr']-base['cagr'];dd_gain=abs(base['max_drawdown'])-abs(m['max_drawdown'])
        rows.append({'year':year,'name':name,**m,'benchmark_cagr':base['cagr'],
                     'benchmark_drawdown':base['max_drawdown'],'cagr_gain':delta,'drawdown_gain':dd_gain,
                     'feasible':delta>1e-9 and dd_gain>1e-9,'score':delta+.25*dd_gain})
    feasible=[r for r in rows if r['feasible']]
    choice=max(feasible,key=lambda r:r['score'])['name'] if feasible else 'expert_equal'
    for r in rows:r['selected']=r['name']==choice
    return choice,rows
