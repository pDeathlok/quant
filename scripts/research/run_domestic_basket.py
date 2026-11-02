#!/usr/bin/env python3
"""Frozen dynamic basket allocation with explicit user-approved concentration limits."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from quant.research.byd_mwu import ROOT
from quant.research.byd_wear import WearConfig
from quant.research.byd_allocation import AllocationConfig
from quant.research.domestic_portfolio import START,load_panel,simulate,portfolio_metrics,mix_targets
from quant.research.domestic_basket import GROUPS,NAMES,experts
from quant.research.byd_growth import paired_return_bootstrap
from run_domestic_portfolio import annual_choice,risk_cap
from run_byd_wear import chart,rolling_windows

OUT=ROOT/'reports/research/domestic_basket20_20261013'
PREV=ROOT/'reports/research/domestic_portfolio_20261013'
GROWTH=ROOT/'reports/research/domestic_growth_20261013'


def run(panel,target,wear):
    return simulate(panel,target,wear,basket_groups=GROUPS)


def limit_metrics(frame):
    return {'basket_excess_days':int(frame.basket_excess.gt(1e-9).sum()),
            'max_actual_basket':float(frame.basket_max_weight.max()),
            'repair_days':int(frame.basket_repair_requested.sum())}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    bank={f'{method}_eta{eta}_vol{vol}':(AllocationConfig(method=method,eta=eta,share=.01,forget=.995),vol)
          for method in ('share','context','multiwindow') for eta in (.1,.3) for vol in (0.,.15)}
    protocol={'start':START,'blind':False,'previous_rounds_observed':True,
              'user_constraint':dict(zip(NAMES,[.20]*6)),'broad_combines':['510300.SH','510500.SH'],
              'cash_exempt':True,'new_fits':0,'models':'reuse mature-label rolling predictions audited in prior rounds',
              'hurdles_horizon_bps':[20,50],'frequency':5,'deadband':.02,
              'daily_repair':'open basket >20% triggers sale target <=19.5%, no deadband for affected basket; no guarantee on close due intraday/blocked orders',
              'selection':'prior 3 completed years, drawdown/volatility <=20% then CAGR minus annual std penalty; starts 2018',
              'initialization':'balanced_cap for 2014-2017','stress_routes_frozen':True,
              'candidate_configs':{k:{'config':cfg.to_dict(),'vol':vol} for k,(cfg,vol) in bank.items()},
              'stopping':'finish fixed expert/candidate set and all four cost/capacity cases; no result-driven expansion'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    panel=load_panel();forecasts={}
    for file in (PREV/'model_predictions.npz',GROWTH/'excess_predictions.npz'):
        with np.load(file) as c:forecasts.update({k:c[k] for k in c.files})
    targets=experts(panel,forecasts)
    protocol['experts']=list(targets);protocol['candidate_count']=len(targets)+len(bank)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    np.savez_compressed(OUT/'expert_targets.npz',**targets)
    first=int(panel.dates.searchsorted(pd.Timestamp(START)))
    cases={'base':WearConfig(),'severe':WearConfig(spread=.002,impact_eta=.5),
           'extreme':WearConfig(spread=.005,impact_eta=1),'capacity_10m':WearConfig(initial_cash=10_000_000)}
    routes={};scores=[];result=[];all_frames={};full_selected_targets={}
    for case,wear in cases.items():
        frames={};rewards=np.zeros((len(panel.dates),len(targets)));paths={};candidate_frames={}
        print(f'{case}: {len(targets)} expert accounts',flush=True)
        for e,(name,path) in enumerate(targets.items()):
            frame,orders=run(panel,path,wear);frames[name]=frame
            rewards[first:,e]=frame['return'].to_numpy()
            if case=='base':
                frame.to_csv(OUT/f'expert_{name}.csv',index=False)
        for key,(cfg,vol) in bank.items():
            path,_=mix_targets(panel,targets,rewards,cfg);paths[key]=risk_cap(panel,path,vol)
            if case=='base':candidate_frames[key]=run(panel,paths[key],wear)[0]
        if case=='base':
            candidate_frames.update({f'expert_{name}':f for name,f in frames.items()})
            pd.DataFrame([{'name':n,**portfolio_metrics(f),**limit_metrics(f)} for n,f in candidate_frames.items()]).to_csv(OUT/'candidate_diagnostic.csv',index=False)
            for year in range(2018,2027):
                routes[year],rows=annual_choice(candidate_frames,year);scores.extend(rows)
            (OUT/'routes_frozen.json').write_text(json.dumps(routes,indent=2)+'\n')
            pd.DataFrame(scores).to_csv(OUT/'annual_selection_scores.csv',index=False)
            print('Annual routes frozen',flush=True)
        selected=targets['balanced_cap'].copy()
        scheduled=panel.dates.to_series(index=np.arange(len(panel.dates))).shift(-1).fillna(panel.dates[-1]).dt.year.to_numpy()
        for year,key in routes.items():
            mask=scheduled==year
            selected[mask]=(targets[key[7:]] if key.startswith('expert_') else paths[key])[mask]
        selected_frame,orders=run(panel,selected,wear);full_selected_targets[case]=selected
        orders.to_csv(OUT/f'orders_{case}_selected.csv',index=False)
        final={'selected':selected_frame,**{n:frames[n] for n in ('balanced_cap','six_equal','byd20_cash','byd_grid_plus_ridge_excess21','ridge_excess21_hurdle50','momentum120')}}
        for name,frame in final.items():
            frame.to_csv(OUT/f'ledger_{case}_{name}.csv',index=False);all_frames[case,name]=frame
            for start in (START,'2018-01-01','2024-01-01'):
                block=frame[frame.date.ge(start)]
                result.append({'case':case,'name':name,'start':start,**portfolio_metrics(block),**limit_metrics(block)})
        print(f'{case}: completed',flush=True)
    table=pd.DataFrame(result);table.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame([{'name':name,'year':year,**portfolio_metrics(block)} for (case,name),f in all_frames.items() if case=='base' for year,block in f.groupby(f.date.dt.year)]).to_csv(OUT/'yearly_metrics.csv',index=False)
    rolling=[]
    for name in ('selected','balanced_cap','six_equal','byd20_cash'):rolling.extend(rolling_windows(all_frames['base',name],name))
    r=pd.DataFrame(rolling);r.to_csv(OUT/'rolling_windows.csv',index=False)
    r.groupby(['name','years']).cagr.agg(['min','median','max']).to_csv(OUT/'rolling_summary.csv')
    bootstrap=[]
    for start in (START,'2018-01-01','2024-01-01'):
        a=all_frames['base','selected'];a=a[a.date.ge(start)]
        for name in ('balanced_cap','six_equal','byd20_cash'):
            b=all_frames['base',name];b=b[b.date.ge(start)]
            bootstrap.append({'start':start,**paired_return_bootstrap(a,b,name)})
    pd.DataFrame(bootstrap).to_csv(OUT/'paired_bootstrap.csv',index=False)
    # Targets, forecasts and expert definitions must survive independent data truncation.
    short=load_panel(end_date='2019-12-31');n=len(short.dates)
    prefix_targets=experts(short,{k:v[:n] for k,v in forecasts.items()})
    for name in targets:np.testing.assert_allclose(prefix_targets[name],targets[name][:n])
    full_rewards=np.zeros((len(panel.dates),len(targets)));short_rewards=np.zeros((n,len(targets)))
    for e,name in enumerate(targets):
        f,_=run(short,prefix_targets[name],WearConfig());full_rewards[first:,e]=pd.read_csv(OUT/f'expert_{name}.csv')['return']
        short_rewards[first:,e]=f['return']
        np.testing.assert_allclose(full_rewards[:n,e],short_rewards[:,e],atol=1e-12)
    cfg,vol=next(iter(bank.values()))
    long_path,_=mix_targets(panel,targets,full_rewards,cfg);short_path,_=mix_targets(short,prefix_targets,short_rewards,cfg)
    np.testing.assert_allclose(long_path[:n],short_path,atol=1e-12)
    full_frame,_=run(panel,long_path,WearConfig());short_frame,_=run(short,short_path,WearConfig())
    pd.testing.assert_frame_equal(full_frame.iloc[:len(short_frame)].reset_index(drop=True),short_frame)
    manifest={'expert_learner_account_prefix_pass':True,'ledgers':len(all_frames),'candidate_count':len(targets)+len(bank),
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('src/quant/research/domestic_basket.py','src/quant/research/domestic_portfolio.py','scripts/research/run_domestic_basket.py','tests/test_domestic_basket.py')},
              'prediction_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PREV/'model_predictions.npz',GROWTH/'excess_predictions.npz')}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    chart({n:all_frames['base',n] for n in ('selected','balanced_cap','six_equal','byd20_cash')},OUT/'equity.svg','dynamic six baskets, target at most 20%, net costs')
    lines=['# 动态六篮子0—20%配置回测','','用户确认六篮子：BYD、宽基（沪深300+中证500）、黄金、债券、证券、红利，各目标0—20%，现金豁免。无融资/做空；允许不投资。历史已观察，本轮不是盲测。',
           f'预设{len(targets)}专家和{len(targets)+len(bank)}候选；模型复用上一轮成熟标签预测，本轮无新增训练；20/50bps持有期收益门槛过滤小预测。每年仅用前三完整年度选择，2014—2017以balanced_cap初始化。',
           'balanced_cap目标BYD5%、宽基20%（12/8）、黄金20%、国债20%、证券5%、红利10%、现金20%。six_equal六篮子各1/6，宽基内两ETF各1/12。BYD20%现金基准仅持有BYD20%。旧BYD网格信号裁剪至20%，不再声称原策略仓位保持不变。超额收益模型预测相对沪深300，不保证绝对收益为正。',
           '共享账户整手、最低佣金、股税、分红、参与率、涨跌停近似与前轮一致。每5交易日检查、2个百分点死区，每日开盘发现篮子超过20%则请求减到19.5%；实际收盘可能因盘中变化或无法成交超过20%，下表如实记录。', '',
           '|成本|起点|方案|年化净收益|波动率|最大回撤|实际超限日数|最大实际篮子|订单数|',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in table.itertuples():lines.append(f'|{row.case}|{row.start}|{row.name}|{row.cagr:.2%}|{row.volatility:.2%}|{row.max_drawdown:.2%}|{row.basket_excess_days}|{row.max_actual_basket:.2%}|{row.orders}|')
    lines+=['','![净值](equity.svg)','','限制：实际20%并非连续时间硬保证；ETF日线成交价格带和冲击未用订单队列/交割单校准。现金利息零，BYD分红税储备20%近似；基金成本不重复扣除。重采样不校正多重比较，候选诊断不能用事后全期最佳代替年度选型收益。目标15%/20%以完整期、2018起和压力情景共同评价。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(table[(table.case=='base')&(table.start==START)][['name','cagr','volatility','max_drawdown','basket_excess_days','max_actual_basket','orders']].to_string(index=False),flush=True)


if __name__=='__main__':main()
