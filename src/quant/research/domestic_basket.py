"""Six user-defined baskets, each optional and capped at twenty percent."""
from __future__ import annotations
import numpy as np
import pandas as pd
from quant.research.domestic_portfolio import Panel,CODES
from quant.research.byd_mwu import ROOT

GROUPS=((0,),(1,2),(3,),(4,),(5,),(6,))
NAMES=('BYD','broad','gold','bond','securities','dividend')


def project_baskets(values: np.ndarray) -> np.ndarray:
    w=np.maximum(np.nan_to_num(values),0.).copy()
    for group in GROUPS:
        a=list(group);total=w[a].sum()
        if total>.20:w[a]*=.20/total
    if w.sum()>1:w/=w.sum()
    return w


def ranked_targets(panel: Panel, scores: np.ndarray, threshold: float,
                   prior_byd: np.ndarray=None) -> np.ndarray:
    """Cash competes at a fixed positive hurdle; no forced defensive allocation."""
    output=np.zeros_like(scores);eligible=panel.data['eligible'].astype(bool)
    for t in range(len(scores)):
        remaining=1.;locked=prior_byd is not None
        if locked:
            output[t,0]=min(.20,max(0.,prior_byd[t])) if eligible[t,0] else 0.
            remaining-=output[t,0]
        ranked=[]
        for g,group in enumerate(GROUPS):
            if locked and g==0:continue
            a=np.array(group);valid=eligible[t,a]&np.isfinite(scores[t,a])&(scores[t,a]>threshold)
            if valid.any():ranked.append((float(scores[t,a[valid]].max()),g,a[valid]))
        for score,g,assets in sorted(ranked,key=lambda x:x[0],reverse=True):
            budget=min(.20,remaining)
            if budget<=1e-12:break
            relative=np.maximum(scores[t,assets],0.)
            output[t,assets]=budget*relative/relative.sum();remaining-=budget
    return output


def experts(panel: Panel, forecasts: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    n=len(panel.dates);eligible=panel.data['eligible']
    result={}
    for name,weights in {'balanced_cap':[.05,.12,.08,.20,.20,.05,.10],
                         'six_equal':[1/6,1/12,1/12,1/6,1/6,1/6,1/6],
                         'gold_bond_cap':[0,0,0,.20,.20,0,0],
                         'byd20_cash':[.20,0,0,0,0,0,0],
                         'cash':[0,0,0,0,0,0,0]}.items():
        result[name]=np.array(weights)[None,:]*eligible
    old=pd.read_csv(ROOT/'reports/research/byd_growth_20261013/ledger_base_previous_grid.csv',parse_dates=['signal_date'])
    # requested_target was computed after signal-date close in the old audited ledger.
    byd=old.drop_duplicates('signal_date').set_index('signal_date').requested_target.reindex(panel.dates).ffill().fillna(0).clip(0,.20).to_numpy()
    result['byd_grid_cap']=np.zeros_like(eligible);result['byd_grid_cap'][:,0]=byd*eligible[:,0]
    idx={name:i for i,name in enumerate(panel.feature_names)}
    for horizon in (63,120,250):
        score=panel.features[:,:,idx[f'ret_{horizon}']].copy()
        score[panel.features[:,:,idx['distance_120']]<=0]=-1
        result[f'momentum{horizon}']=ranked_targets(panel,score,0.)
    for name in ('ridge_5','ridge_21','ridge_63','ridge_excess21','tree_excess21'):
        horizon=5 if name=='ridge_5' else 63 if name=='ridge_63' else 21
        # Predictions are dailyized horizon labels, not annual forecasts.
        for bps in (20,50):
            result[f'{name}_hurdle{bps}']=ranked_targets(panel,forecasts[name],bps/10000/horizon)
    for name in ('ridge_21','ridge_excess21'):
        result[f'byd_grid_plus_{name}']=ranked_targets(panel,forecasts[name],.005/21,byd)
    for path in result.values():
        if not np.isfinite(path).all() or (path<0).any() or (path.sum(axis=1)>1+1e-9).any():raise RuntimeError('Invalid expert')
        for group in GROUPS:
            if (path[:,group].sum(axis=1)>.20+1e-9).any():raise RuntimeError('Basket cap')
    return result
