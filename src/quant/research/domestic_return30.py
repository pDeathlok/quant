"""Return-first domestic policies, preserving six-basket concentration limits."""
from __future__ import annotations
import numpy as np
import pandas as pd
from quant.research.domestic_portfolio import Panel,portfolio_metrics
from quant.research.domestic_relative import basket_scores
from quant.research.domestic_basket import GROUPS

CORES={
    'lowsec':[.20,.15,.20,.20,.05,.20],
    'nosec':[.20,.20,.20,.20,0,.20],
    'offensive':[.20,.20,.20,.10,.10,.20],
    'nobond':[.20,.20,.20,0,.20,.20],
    'equal':[1/6]*6,
}


def distribute_broad(panel: Panel, baskets: np.ndarray, mode: str) -> np.ndarray:
    out=np.zeros_like(panel.data['close']);eligible=panel.data['eligible'].astype(bool)
    idx=panel.feature_names.index('ret_63')
    for t in range(len(baskets)):
        for g,group in enumerate(GROUPS):
            a=np.array(group);usable=a[eligible[t,a]]
            if not len(usable):continue
            if g==1 and mode!='blend':
                if mode=='csi300':chosen=1 if 1 in usable else int(usable[0])
                elif mode=='momentum63':chosen=int(usable[np.argmax(panel.features[t,usable,idx])])
                else:raise ValueError('Unknown broad mode')
                out[t,chosen]=baskets[t,g]
            else:out[t,usable]=baskets[t,g]/len(usable)
    return out


def base_targets(panel: Panel) -> dict[str,np.ndarray]:
    n=len(panel.dates);result={}
    for name,weights in CORES.items():
        b=np.tile(weights,(n,1))
        for mode in ('csi300','blend','momentum63'):result[f'{name}_{mode}']=distribute_broad(panel,b,mode)
    available=np.stack([panel.data['eligible'][:,g].any(axis=1) for g in GROUPS],axis=1)
    for h in (63,120,250):
        scores=basket_scores(panel,panel.features[:,:,panel.feature_names.index(f'ret_{h}')])
        b=np.zeros((n,6))
        for t in range(n):
            valid=np.flatnonzero(available[t]);selected=valid[np.argsort(scores[t,valid])[-5:]]
            b[t,selected]=.20
        for mode in ('csi300','blend','momentum63'):result[f'rank{h}_{mode}']=distribute_broad(panel,b,mode)
    for name,path in result.items():
        if (path<0).any() or (path.sum(axis=1)>1+1e-9).any():raise RuntimeError(name)
        if any((path[:,g].sum(axis=1)>.20+1e-9).any() for g in GROUPS):raise RuntimeError('Basket cap')
    return result


def return_choice(frames: dict[str,pd.DataFrame], year: int) -> tuple[str,list[dict]]:
    rows=[]
    for name,f in frames.items():
        past=f[(f.date.dt.year>=year-3)&(f.date.dt.year<year)]
        m=portfolio_metrics(past)
        rows.append({'year':year,'name':name,**m,'feasible':abs(m['max_drawdown'])<=.30+1e-9})
    feasible=[r for r in rows if r['feasible']]
    choice=max(feasible,key=lambda r:r['cagr'])['name'] if feasible else 'expert_v1'
    for r in rows:r['selected']=r['name']==choice
    return choice,rows
