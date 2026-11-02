#!/usr/bin/env python3
"""Declared second relative round after first-round results: constrain core risk."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from quant.research.byd_mwu import ROOT
from quant.research.byd_wear import WearConfig
from quant.research.byd_growth import paired_return_bootstrap
from quant.research.domestic_portfolio import START,load_panel,portfolio_metrics
from quant.research.domestic_relative import risk_guard,relative_choice
from run_domestic_portfolio import risk_cap
from run_domestic_relative import run,dominance
from run_byd_wear import chart,rolling_windows

PREV=ROOT/'reports/research/domestic_relative_20261013'
OUT=ROOT/'reports/research/domestic_relative_risk_20261013'


def make_targets(panel,core,equal):
    result={'equal':equal,'core':core}
    for scale in (.85,.90,.95):result[f'core_scale{int(scale*100)}']=core*scale
    for vol in (.14,.16,.18):result[f'core_vol{int(vol*100)}']=risk_cap(panel,core,vol)
    for scale in (.7,.4):result[f'core_guard{int(scale*100)}']=risk_guard(panel,core,scale)
    result['core_vol16_guard70']=risk_guard(panel,result['core_vol16'],.7)
    return result


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'previous_round_observed':True,'blind':False,'round':2,'start':START,
              'new_model_fits':0,'scales':[.85,.90,.95],'volatility_caps':[.14,.16,.18],
              'weak_equity_scales':[.7,.4],'combined_vol16_guard70':True,
              'target_each_basket_max':.20,'actual_drift_allowed':True,'frequency':21,'deadband':.03,
              'objective':'net CAGR higher AND absolute drawdown lower than equal',
              'initial_policy':'core_scale90 2014-2017','selection':'prior 3 completed years relative double dominance, otherwise monthly equal',
              'stopping':'fixed second round then stop; no third expansion based on results'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    panel=load_panel()
    with np.load(PREV/'expert_targets.npz') as cache:core=cache['low_securities_core'];equal=cache['equal']
    targets=make_targets(panel,core,equal);np.savez_compressed(OUT/'expert_targets.npz',**targets)
    protocol['candidates']=list(targets);(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cases={'base':WearConfig(),'severe':WearConfig(spread=.002,impact_eta=.5),
           'extreme':WearConfig(spread=.005,impact_eta=1),'capacity_10m':WearConfig(initial_cash=10_000_000)}
    routes={};scores=[];result=[];all_frames={}
    for case,wear in cases.items():
        frames={}
        for name,target in targets.items():frames[name]=run(panel,target,wear)[0]
        if case=='base':
            for year in range(2018,2027):
                routes[year],rows=relative_choice({f'expert_{k}':f for k,f in frames.items()},year);scores.extend(rows)
            (OUT/'routes_frozen.json').write_text(json.dumps(routes,indent=2)+'\n')
            pd.DataFrame(scores).to_csv(OUT/'annual_selection_scores.csv',index=False)
        selected=targets['core_scale90'].copy()
        schedule=panel.dates.to_series(index=np.arange(len(panel.dates))).shift(-1).fillna(panel.dates[-1]).dt.year.to_numpy()
        for year,key in routes.items():selected[schedule==year]=targets[key[7:]][schedule==year]
        frames['selected'],orders=run(panel,selected,wear)
        orders.to_csv(OUT/f'orders_{case}_selected.csv',index=False)
        frames['equal_weekly']=run(panel,equal,wear,True)[0]
        for name,frame in frames.items():
            frame.to_csv(OUT/f'ledger_{case}_{name}.csv',index=False);all_frames[case,name]=frame
            for start in (START,'2018-01-01','2024-01-01'):
                m=portfolio_metrics(frame[frame.date.ge(start)])
                for baseline in ('equal','equal_weekly'):
                    b=portfolio_metrics(frames[baseline][frames[baseline].date.ge(start)])
                    result.append({'case':case,'name':name,'start':start,'baseline':baseline,**m,**dominance(m,b)})
        print(f'{case}: complete',flush=True)
    table=pd.DataFrame(result);table.to_csv(OUT/'metrics.csv',index=False)
    rolling=[]
    for name in ('selected','core_scale90','core_vol16','equal','equal_weekly'):rolling.extend(rolling_windows(all_frames['base',name],name))
    r=pd.DataFrame(rolling);r.to_csv(OUT/'rolling_windows.csv',index=False);r.groupby(['name','years']).cagr.agg(['min','median','max']).to_csv(OUT/'rolling_summary.csv')
    paired=[]
    for start in (START,'2018-01-01','2024-01-01'):
        for name in ('selected','core_scale90','core_vol16'):
            a=all_frames['base',name];a=a[a.date.ge(start)];b=all_frames['base','equal'];b=b[b.date.ge(start)]
            paired.append({'start':start,'strategy':name,**paired_return_bootstrap(a,b,'equal')})
    pd.DataFrame(paired).to_csv(OUT/'paired_bootstrap.csv',index=False)
    pd.DataFrame([{'name':name,'year':year,**portfolio_metrics(block)} for (case,name),f in all_frames.items() if case=='base' for year,block in f.groupby(f.date.dt.year)]).to_csv(OUT/'yearly_metrics.csv',index=False)
    short=load_panel(end_date='2019-12-31');n=len(short.dates)
    small=make_targets(short,core[:n],equal[:n]);short_frames={}
    for name,path in small.items():
        np.testing.assert_allclose(path,targets[name][:n],atol=1e-12)
        f,_=run(short,path,WearConfig());short_frames[f'expert_{name}']=f
        pd.testing.assert_frame_equal(all_frames['base',name].iloc[:len(f)].reset_index(drop=True),f)
    for year in (2018,2019):
        key,_=relative_choice(short_frames,year)
        if key!=routes[year]:raise RuntimeError('Future-dependent annual selection')
    manifest={'independent_target_account_route_prefix_pass':True,'ledgers':len(all_frames),
              'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('scripts/research/run_domestic_relative_risk.py','src/quant/research/domestic_relative.py','src/quant/research/domestic_portfolio.py','scripts/research/run_domestic_relative.py')},
              'prior_targets_sha256':hashlib.sha256((PREV/'expert_targets.npz').read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    chart({n:all_frames['base',n] for n in ('selected','core_scale90','core_vol16','equal','equal_weekly')},OUT/'equity.svg','core risk budget versus equal baskets, relaxed realized cap')
    lines=['# 相对六篮子等权第二轮：核心风险预算','','本轮已观察第一轮，非盲测。固定核心为BYD20%、宽基15%、黄金20%、国债20%、证券5%、红利20%；候选比较85/90/95%总体仓位、14/16/18%过去63日波动预算和弱市控制，不重训模型。实际仓位漂移允许，目标篮子不超过20%。',
           '2014—2017初始化core_scale90；2018起仅用过去3完整年相对月度等权双优选择，失败回到等权；年度路线冻结后计算成本/容量压力。固定候选的全期诊断与年度流程分开，不声称事后双优即独立样本外成功。',
           'core_scale90具体目标：BYD18%、宽基13.5%（两ETF各6.75%）、黄金18%、国债18%、证券4.5%、红利18%、现金10%。每21交易日检查、3个百分点死区。两个等权对照分别为相同月度规则及5交易日/2点宽松规则。','','|成本|起点|方案|年化|最大回撤|收益增加|回撤减少|双优|订单数|','|---|---|---|---:|---:|---:|---:|---|---:|']
    for row in table[table.baseline.eq('equal')].itertuples():lines.append(f'|{row.case}|{row.start}|{row.name}|{row.cagr:.2%}|{row.max_drawdown:.2%}|{row.cagr_gain:.2%}|{row.drawdown_gain:.2%}|{row.both_better}|{row.orders}|')
    lines+=['','![净值](equity.svg)','','现金利息零，日线冲击/涨跌停近似未经真实队列校准，BYD分红税储备近似；重采样未校正多轮研究与多重比较。CAGR与回撤历史双优不保证未来；最终应同时检查完整期、2018起和成本压力。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(table[(table.case=='base')&(table.start==START)&(table.baseline=='equal')][['name','cagr','max_drawdown','both_better','orders']].to_string(index=False),flush=True)


if __name__=='__main__':main()
