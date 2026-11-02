"""Long-only asset concentration is unrestricted; total capital remains bounded."""
from __future__ import annotations
import numpy as np
from quant.research.domestic_portfolio import Panel


def rank_assets(panel: Panel, scores: np.ndarray, top: int, gate: np.ndarray=None) -> np.ndarray:
    out=np.zeros_like(panel.data['close']);eligible=panel.data['eligible'].astype(bool)
    if gate is None:gate=np.ones_like(eligible)
    for t in range(len(scores)):
        valid=np.flatnonzero(eligible[t]&gate[t]&np.isfinite(scores[t]))
        selected=valid[np.argsort(scores[t,valid])[-top:]]
        if len(selected):out[t,selected]=1/len(selected)
    return out


def build_targets(panel: Panel, forecasts: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    n=len(panel.dates);k=panel.data['close'].shape[1];eligible=panel.data['eligible'].astype(bool)
    f=panel.features;idx={name:i for i,name in enumerate(panel.feature_names)};result={}
    for a in range(k):
        w=np.zeros((n,k));w[:,a]=eligible[:,a];result[f'buy_{a}']=w
    for byd in (.25,.50,.75):
        w=np.zeros((n,k));w[:,0]=byd*eligible[:,0];w[:,3]=(1-byd)*eligible[:,3]
        result[f'byd_gold_{int(byd*100)}']=w
    for name,w in {'growth_core':[.5,.075,.075,.20,.10,0,.05],
                   'byd_gold_bond':[.4,0,0,.4,.2,0,0]}.items():result[name]=np.array(w)[None,:]*eligible
    for horizon in (63,120,250,'multi'):
        scores=(f[:,:,idx['ret_63']]+f[:,:,idx['ret_120']]+f[:,:,idx['ret_250']])/3 if horizon=='multi' else f[:,:,idx[f'ret_{horizon}']]
        for top in (1,2):
            for trend in (False,True):
                gate=f[:,:,idx['distance_120']]>0 if trend else None
                result[f'mom{horizon}_top{top}_{"trend" if trend else "relative"}']=rank_assets(panel,scores,top,gate)
    for horizon in (20,63,120,250):
        w=np.zeros((n,k));up=eligible[:,0]&(f[:,0,idx[f'distance_{horizon}']]>0)
        gold=(~up)&eligible[:,3]&(f[:,3,idx['distance_120']]>0)
        bond=(~up)&(~gold)&eligible[:,4]
        w[up,0]=1;w[gold,3]=1;w[bond,4]=1;result[f'byd_trend{horizon}_defense']=w
    for name in ('ridge_21','ridge_63','ridge_excess21','tree_excess21'):
        score=forecasts[name]
        for top in (1,2):result[f'{name}_top{top}']=rank_assets(panel,score,top,score>.0001)
    for name,w in result.items():
        if not np.isfinite(w).all() or (w<0).any() or (w.sum(axis=1)>1+1e-9).any():raise RuntimeError(name)
    return result
