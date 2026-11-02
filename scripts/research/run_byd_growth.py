#!/usr/bin/env python3
"""Three predeclared rounds of return-first constrained BYD optimization."""
from __future__ import annotations
import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
from quant.research.byd_allocation import AllocationConfig
from quant.research.byd_growth import growth_experts,constrained_choice,paired_return_bootstrap
from quant.research.byd_mwu import ROOT
from quant.research.byd_regime import regime_experts,mix_path
from quant.research.byd_wear import START,WearConfig,wear_expert_returns,wear_simulate,wear_metrics
from run_byd_wear import chart,rolling_windows

OUT=ROOT/'reports/research/byd_growth_20261013'


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    market=pd.read_parquet(ROOT/'reports/research/byd_mwu_20261012/market_snapshot.parquet')
    cached=pd.read_parquet(ROOT/'reports/research/byd_allocation_20261013/expert_targets.parquet')
    if not cached.date.equals(market.date):raise RuntimeError('Different snapshots')
    targets,groups,states=growth_experts(market,cached.drop(columns='date'))
    _,regimes=regime_experts(market)
    first=int(np.searchsorted(market.date.to_numpy(),np.datetime64(START)))
    bank={}; rounds={1:[],2:[],3:[]}
    direct=AllocationConfig(method='equal',deadband=.03,max_step=.20)
    for expert in list(targets.columns[-7:])+['original_range_x12','original_range_x15']:
        key=f'direct_{expert}';bank[key]=(direct,expert,None);rounds[1].append(key)
    for position in (1/3,.40,.50,.60):
        key=f'fixed_{position:.3f}';bank[key]=(direct,None,position);rounds[1].append(key)
    mixes=[]
    for group in groups:
        for method in ('share','context','multiwindow'):
            cfg=AllocationConfig(method=method,group=group,eta=.3,share=.01,forget=.995,
                                 deadband=.03,max_step=.20)
            key=f'{method}_{group}';bank[key]=(cfg,None,None);rounds[2].append(key);mixes.append(key)
    for key in mixes:
        for vol in (.20,.28):
            for band in (.03,.06):
                new=f'{key}_vol{int(vol*100)}_band{int(band*100)}'
                cfg=replace(bank[key][0],vol_target=vol,deadband=band,cost_gamma=.5)
                bank[new]=(cfg,None,None);rounds[3].append(new)
    sets={r:sum([rounds[i] for i in range(1,r+1)],[]) for r in rounds}
    protocol={'start':START,'data_end':str(market.date.iloc[-1].date()),'target_cagr':[.12,.15],
        'target_drawdown':[.20,.25],'blind':False,'candidate_count':len(bank),
        'candidate_bank':{k:{'config':cfg.to_dict(),'expert':expert,'fixed_position':p} for k,(cfg,expert,p) in bank.items()},
        'round_candidate_keys':sets,'training_constraints':{'max_drawdown':.25,'volatility':.25,'min_average_position':.25},
        'objective':'prior 3 complete calendar years CAGR minus 0.15 annual CAGR std; feasible first, otherwise minimum violation',
        'initial_policy':'direct_mixed_core','cost_routes_frozen':True,'stop':'complete all three predeclared rounds, no post-result candidates'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    targets.assign(date=market.date).to_parquet(OUT/'expert_targets.parquet',index=False)
    states.assign(date=market.date).to_parquet(OUT/'state_scores.parquet',index=False)
    cases={'base':WearConfig(),'severe':WearConfig(spread=.002,impact_eta=.5),'extreme':WearConfig(spread=.005,impact_eta=1),
           'capacity_10m':WearConfig(initial_cash=10_000_000)}
    routes={};metrics=[];selections=[];phases=[];all_ledgers={};candidate_ledgers={}
    for case,wear in cases.items():
        print(f'{case}: {targets.shape[1]} expert accounts',flush=True)
        rewards=wear_expert_returns(market,targets,wear)
        proposals={};w_paths={}
        for key in mixes:
            cfg=bank[key][0];names=groups[cfg.group]
            proposal,w=mix_path(targets[names],rewards[names],regimes,cfg,first)
            proposals[key]=proposal;w_paths[key]=pd.DataFrame(w,columns=names,index=market.index)
        for key,(cfg,expert,position) in bank.items():
            if expert:proposals[key]=targets[expert].to_numpy()
            elif position is not None:proposals[key]=np.full(len(market),position)
            elif key not in proposals:
                parent=key.split('_vol')[0];proposals[key]=proposals[parent]
                w_paths[key]=w_paths[parent]
        if case=='base':
            for count,(key,(cfg,_,_)) in enumerate(bank.items(),1):
                frame=wear_simulate(market,proposals[key],[cfg]*len(market),wear)
                candidate_ledgers[key]=frame
                if count%20==0:print(f'base candidates {count}/{len(bank)}',flush=True)
            candidate_rows=[]
            for key,frame in candidate_ledgers.items():
                for start in (START,'2016-01-01'):
                    candidate_rows.append({'name':key,'start':start,**wear_metrics(frame[frame.date.ge(start)])})
            pd.DataFrame(candidate_rows).to_csv(OUT/'candidate_metrics_diagnostic.csv',index=False)
            for round_no,keys in sets.items():
                routes[round_no]={}
                for year in range(2016,2027):
                    selected,rows=constrained_choice(candidate_ledgers,year,keys)
                    routes[round_no][year]=selected
                    selections.extend({'round':round_no,**row} for row in rows)
            (OUT/'routes_frozen.json').write_text(json.dumps(routes,indent=2))
            pd.DataFrame(selections).to_csv(OUT/'annual_selection_scores.csv',index=False)
            print('Annual routes frozen before stress and capacity evaluation',flush=True)
        ledgers={}
        scheduled_year=market.date.shift(-1).fillna(market.date.iloc[-1]).dt.year
        for r in sets:
            proposal=proposals['direct_mixed_core'].copy();cfgs=[direct]*len(market)
            chosen_weights=pd.DataFrame(0.,index=market.index,columns=targets.columns)
            for year,key in routes[r].items():
                mask=scheduled_year.eq(year).to_numpy();proposal[mask]=proposals[key][mask]
                for i in np.flatnonzero(mask):cfgs[i]=bank[key][0]
                if key in w_paths:
                    chosen_weights.loc[mask,w_paths[key].columns]=w_paths[key].loc[mask].to_numpy()
            name=f'round_{r}'
            ledgers[name]=wear_simulate(market,proposal,cfgs,wear)
            chosen_weights.assign(date=market.date).to_parquet(OUT/f'weights_{case}_{name}.parquet',index=False)
        for key in rounds[1]:
            if key.startswith('fixed_'):
                ledgers[key]=wear_simulate(market,proposals[key],[direct]*len(market),wear)
        for key in ('direct_mixed_core','direct_soft_core45_tilt25','direct_original_range_x15'):
            ledgers[key]=wear_simulate(market,proposals[key],[direct]*len(market),wear)
        # Same snapshot and prior frozen grid policy, as an external historical benchmark.
        if case!='capacity_10m':
            old=pd.read_csv(ROOT/f'reports/research/byd_wear_20261013/ledger_{case}_grid_search.csv',parse_dates=['date','signal_date'])
            ledgers['previous_grid']=old
        for name,frame in ledgers.items():
            if frame.cash.min() < -1e-6 or frame.shares.min()<0:raise RuntimeError('Account invariant failed')
            frame.to_csv(OUT/f'ledger_{case}_{name}.csv',index=False)
            for start in (START,'2016-01-01','2020-01-01','2024-01-01'):
                block=frame[frame.date.ge(start)]
                metrics.append({'case':case,'name':name,'start':start,**wear_metrics(block)})
            used_states=regimes[first-1:len(market)-1]
            for state,label in ((0,'down'),(1,'range'),(2,'up')):
                block=frame.loc[used_states==state]
                phases.append({'case':case,'name':name,'state':label,'days':len(block),
                               'annual_daily_mean':block['return'].mean()*252,'volatility':block['return'].std()*np.sqrt(252),
                               'average_position':block.position.mean()})
            all_ledgers[case,name]=frame
    result=pd.DataFrame(metrics);result.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(phases).to_csv(OUT/'phase_metrics.csv',index=False)
    pd.DataFrame([{'name':name,'year':year,**wear_metrics(block)} for (case,name),frame in all_ledgers.items() if case=='base'
        for year,block in frame.groupby(frame.date.dt.year)]).to_csv(OUT/'yearly_metrics.csv',index=False)
    roll=[]
    for name in ('round_1','round_2','round_3','fixed_0.333','fixed_0.500','previous_grid'):
        roll.extend(rolling_windows(all_ledgers['base',name],name))
    rolling=pd.DataFrame(roll);rolling.to_csv(OUT/'rolling_calendar_windows.csv',index=False)
    rolling.groupby(['name','years']).cagr.agg(['min','median','max']).to_csv(OUT/'rolling_summary.csv')
    bootstrap=[]
    for start in (START,'2016-01-01','2024-01-01'):
        for r in (1,2,3):
            a=all_ledgers['base',f'round_{r}'];a=a[a.date.ge(start)]
            for benchmark in ('fixed_0.333','fixed_0.500','previous_grid'):
                b=all_ledgers['base',benchmark];b=b[b.date.ge(start)]
                bootstrap.append({'start':start,'round':r,**paired_return_bootstrap(a,b,benchmark)})
    pd.DataFrame(bootstrap).to_csv(OUT/'paired_bootstrap.csv',index=False)
    # Exact prefix audit of experts, learner and ledger, not just lag column checks.
    cut=int(market.date.searchsorted(pd.Timestamp('2024-01-01')))
    short=market.iloc[:cut];st,sg,ss=growth_experts(short,cached.drop(columns='date').iloc[:cut])
    pd.testing.assert_frame_equal(st,targets.iloc[:cut]);pd.testing.assert_frame_equal(ss,states.iloc[:cut])
    short_rewards=wear_expert_returns(short,st,WearConfig())
    full_rewards=wear_expert_returns(market,targets,WearConfig())
    pd.testing.assert_frame_equal(short_rewards,full_rewards.iloc[:cut])
    key='share_mixed_x15';cfg=bank[key][0];names=groups[cfg.group]
    full_p,_=mix_path(targets[names],full_rewards[names],regimes,cfg,first)
    short_p,_=mix_path(st[names],short_rewards[names],regimes[:cut],cfg,first)
    np.testing.assert_allclose(short_p,full_p[:cut])
    frame=wear_simulate(short,short_p,[cfg]*cut)
    pd.testing.assert_frame_equal(frame,candidate_ledgers[key].iloc[:len(frame)].reset_index(drop=True))
    audit={'prefix_pass':True,'ledgers':len(all_ledgers),'candidates':len(bank),
           'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
              ('src/quant/research/byd_growth.py','scripts/research/run_byd_growth.py','src/quant/research/byd_wear.py','tests/test_byd_growth.py')},
           'market_sha256':hashlib.sha256((ROOT/'reports/research/byd_mwu_20261012/market_snapshot.parquet').read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(audit,indent=2))
    chart({n:all_ledgers['base',n] for n in ('round_1','round_2','round_3','fixed_0.500','previous_grid')},OUT/'equity.svg','return-first risk constraints, net costs')
    text=['# BYD 收益优先、约束风险：三轮优化','',
        '目标年化净收益12%-15%、最大回撤20%-25%。所有协议及候选在计算前固定。2012-2015固定mixed_core；2016起每年仅用此前3个完整年度選型。训练约束不保证外层风险上限。旧数据已被观察，不能称盲测。',
        '第一轮直接仓位策略，第二轮加入share/context/multiwindow融合，第三轮加入20%/28%波动上限及3%/6%不交易区间和成本控制；后轮保留前轮候选。无额外单日亏损惩罚，不再用短冷静期反复清仓。连续趋势分数不是校准后的上涨概率。',
        '基准本轮用3个百分点不交易区间、单次20个百分点仓位调整上限；旧网格流程保持原参数，因此应核对平均仓位与成本，不能只比较年化。现金利息为零、不融资、不做空。','',
        '|成本|起点|流程|累计净收益|年化|波动|最大回撤|平均仓位|订单数|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for case in cases:
        for start in (START,'2016-01-01','2024-01-01'):
            for name in ('round_1','round_2','round_3','fixed_0.333','fixed_0.500','previous_grid'):
                rows=result[(result.case==case)&(result.start==start)&(result.name==name)]
                if len(rows):
                    r=rows.iloc[0]
                    text.append(f'|{case}|{start}|{name}|{r.total_return:.2%}|{r.cagr:.2%}|{r.realized_volatility:.2%}|{r.max_drawdown:.2%}|{r.average_position:.2%}|{int(r.orders)}|')
    chosen=pd.DataFrame(selections);chosen=chosen[chosen.selected]
    text+=['','![连续账户净值](equity.svg)','',f'训练期选型不满足全部约束的轮次年份共{int((~chosen.feasible).sum())}个；具体见annual_selection_scores.csv。',
           'candidate_metrics_diagnostic.csv是候选诊断，不可用完整回测期最佳候选冒充部署收益。rolling_summary.csv与yearly_metrics.csv提供滚动/分年检验；paired_bootstrap.csv为20日区块、2000次的描述性置信区间，未校正反复研究和多重比较。',
           '严重/极端磨损分别为单边20/50bps加冲击eta0.5/1；冻结年度路由但重算专家净收益和在线权重。1000万元情景同样冻结路由。冲击模型未经交割单校准；期末持仓按市值计价，未强制卖出。',
           '前缀专家、在线反馈、权重、交易账本重放通过，全部账户现金股份非负。本轮执行完整预设集合后停止，是否达到目标据外层结果判定。']
    (OUT/'report.md').write_text('\n'.join(text),encoding='utf-8')
    print(result[(result.case=='base')&(result.start==START)][['name','cagr','realized_volatility','max_drawdown','average_position']].to_string(index=False),flush=True)

if __name__=='__main__':main()
