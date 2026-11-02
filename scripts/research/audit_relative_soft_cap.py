#!/usr/bin/env python3
"""Execution robustness only: identical drift-tolerance rule on frozen strategies."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from quant.research.byd_mwu import ROOT
from quant.research.byd_wear import WearConfig
from quant.research.domestic_portfolio import START,load_panel,simulate,portfolio_metrics
from quant.research.domestic_basket import GROUPS
from run_domestic_relative import dominance

OUT=ROOT/'reports/research/domestic_relative_risk_20261013/soft_cap_audit'


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'purpose':'common execution robustness, no new expert/model or optimization',
              'observed_original_results':True,'target_cap':.20,'execution_tolerance':.23,
              'repair':'open exceeds23%: reduce to22.5% on noncheck day; monthly target remains at most20%',
              'strategies':['core_vol16','equal_monthly','equal_weekly'],'cases':['base','severe','extreme','capacity_10m']}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    panel=load_panel()
    source=OUT.parent/'expert_targets.npz'
    with np.load(source) as c:targets={'core_vol16':c['core_vol16'],'equal_monthly':c['equal'],'equal_weekly':c['equal']}
    rows=[]
    for case,wear in {'base':WearConfig(),'severe':WearConfig(spread=.002,impact_eta=.5),'extreme':WearConfig(spread=.005,impact_eta=1),'capacity_10m':WearConfig(initial_cash=10_000_000)}.items():
        frames={}
        for name,path in targets.items():
            weekly=name=='equal_weekly'
            f,orders=simulate(panel,path,wear,frequency=5 if weekly else 21,deadband=.02 if weekly else .03,basket_groups=GROUPS,basket_limit=.23)
            frames[name]=f;f.to_csv(OUT/f'ledger_{case}_{name}.csv',index=False)
            orders.to_csv(OUT/f'orders_{case}_{name}.csv',index=False)
        for name,f in frames.items():
            for start in (START,'2018-01-01','2024-01-01'):
                block=f[f.date.ge(start)];m=portfolio_metrics(block)
                for baseline in ('equal_monthly','equal_weekly'):
                    b=portfolio_metrics(frames[baseline][frames[baseline].date.ge(start)])
                    rows.append({'case':case,'name':name,'start':start,'baseline':baseline,**m,**dominance(m,b),
                                 'max_actual_basket':float(block.basket_max_weight.max()),'over_tolerance_days':int(block.basket_excess.gt(1e-9).sum())})
    result=pd.DataFrame(rows);result.to_csv(OUT/'metrics.csv',index=False)
    short=load_panel(end_date='2019-12-31');n=len(short.dates)
    small,_=simulate(short,targets['core_vol16'][:n],WearConfig(),frequency=21,deadband=.03,basket_groups=GROUPS,basket_limit=.23)
    original=pd.read_csv(OUT/'ledger_base_core_vol16.csv',parse_dates=['date','signal_date'])
    np.testing.assert_allclose(small.equity,original.equity.iloc[:len(small)],rtol=1e-12,atol=1e-7)
    (OUT/'manifest.json').write_text(json.dumps({'account_prefix_pass':True,'source_sha256':hashlib.sha256((ROOT/'scripts/research/audit_relative_soft_cap.py').read_bytes()).hexdigest(),'target_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2)+'\n')
    print(result[(result.case=='base')&(result.start==START)&(result.baseline=='equal_weekly')][['name','cagr','max_drawdown','both_better','max_actual_basket','orders']].to_string(index=False))


if __name__=='__main__':main()
