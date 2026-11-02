#!/usr/bin/env python3
"""Frozen regime/risk expert experiment; no hindsight state labels."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from quant.research.byd_allocation import AllocationConfig
from quant.research.byd_mwu import ROOT
from quant.research.byd_regime import regime_experts, mix_path, risk_simulate
from quant.research.byd_wear import START, WearConfig, wear_expert_returns, wear_metrics
from run_byd_wear import chart

OUT = ROOT/'reports/research/byd_regime_20261013'


def score(ledger: pd.DataFrame, year: int) -> float:
    blocks = []
    for previous in range(year-3,year):
        block = ledger[ledger.date.dt.year.eq(previous)]
        m = wear_metrics(block)
        blocks.append(m['cagr']-.5*m['realized_volatility']-abs(m['max_drawdown']))
    return float(np.mean(blocks)-.35*np.std(blocks))


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    market = pd.read_parquet(ROOT/'reports/research/byd_mwu_20261012/market_snapshot.parquet')
    targets, regimes = regime_experts(market)
    first = int(np.searchsorted(market.date.to_numpy(),np.datetime64(START)))
    bank = {}
    for method in ('share','context','multiwindow'):
        for vol in (.10,.15):
            for risk in (0,1,2):
                name = f'{method}_vol{int(vol*100)}_risk{risk}'
                cfg = AllocationConfig(method=method,eta=.3,share=.01,forget=.99,
                                       deadband=.04,max_step=.15,vol_target=vol,cost_gamma=.5)
                bank[name] = (cfg,float(risk), (0.,.12,.08)[risk])
    (OUT/'protocol.json').write_text(json.dumps({'start':START,'states':'MA120 distance +/-3% AND MA60 20-session slope +/-1%; otherwise range',
        'experts':list(targets),'candidates':{n:{'config':c.to_dict(),'downside_penalty':d,'account_stop':s} for n,(c,d,s) in bank.items()},
        'outer_start':2016,'score':'mean prior 3 annual CAGR - 0.5 volatility - abs drawdown, minus 0.35 score std',
        'blind':False,'stopping':'complete fixed 18 candidates and base/severe/extreme; no result-driven additions'},indent=2),encoding='utf-8')
    targets.assign(date=market.date).to_parquet(OUT/'expert_targets.parquet',index=False)
    cases={'base':WearConfig(),'severe':WearConfig(spread=.002,impact_eta=.5),'extreme':WearConfig(spread=.005,impact_eta=1)}
    routes = {}; rows=[]; phases=[]; all_ledgers={}; scores=[]
    default='share_vol10_risk1'
    for case,wear in cases.items():
        print(f'{case}: expert feedback and candidate paths',flush=True)
        rewards = wear_expert_returns(market,targets,wear)
        paths={}; weights={}; candidates={}
        for name,(cfg,downside,stop) in bank.items():
            proposal,w = mix_path(targets,rewards,regimes,cfg,first,downside)
            paths[name]=proposal;weights[name]=w
            if case=='base':
                candidates[name]=risk_simulate(market,proposal,[cfg]*len(market),regimes,wear,np.full(len(market),stop))
        if case=='base':
            for year in range(2016,2027):
                values={n:score(frame,year) for n,frame in candidates.items()}
                routes[year]=max(values,key=values.get)
                scores.extend({'year':year,'name':n,'score':v,'selected':n==routes[year]} for n,v in values.items())
            (OUT/'routes.json').write_text(json.dumps(routes,indent=2))
            pd.DataFrame(scores).to_csv(OUT/'inner_scores.csv',index=False)
        proposals=np.zeros(len(market)); configs=[]; stops=np.zeros(len(market)); selected_weights=np.zeros(targets.shape)
        for i,date in enumerate(market.date):
            name=routes.get(date.year,default);cfg,_,stop=bank[name]
            proposals[i]=paths[name][i];configs.append(cfg);stops[i]=stop;selected_weights[i]=weights[name][i]
        ledgers={'annual_risk_selected':risk_simulate(market,proposals,configs,regimes,wear,stops)}
        pd.DataFrame(selected_weights,columns=targets.columns).assign(date=market.date).to_parquet(OUT/f'weights_{case}.parquet',index=False)
        cfg=AllocationConfig(method='equal',deadband=.04,max_step=.15)
        for name in ('regime_switch','regime_vol_10','regime_vol_15','asset_drawdown_defense','trend_120','downtrend_defense','cash'):
            ledgers[name]=risk_simulate(market,targets[name].to_numpy(),[cfg]*len(market),regimes,wear,np.zeros(len(market)))
        for fraction in (1/3,.20,.25):
            name=f'constant_{fraction:.3f}'
            ledgers[name]=risk_simulate(market,np.full(len(market),fraction),[cfg]*len(market),regimes,wear,np.zeros(len(market)))
        ledgers['regime_account_stop']=risk_simulate(market,targets.regime_vol_15.to_numpy(),[cfg]*len(market),regimes,wear,np.full(len(market),.12))
        if case=='base':
            ledgers.update({n:candidates[n] for n in ('share_vol10_risk1','context_vol10_risk1','multiwindow_vol10_risk1')})
        for name,frame in ledgers.items():
            if frame.cash.min() < -1e-6 or frame.shares.min()<0:
                raise RuntimeError('Invalid account')
            frame.to_csv(OUT/f'ledger_{case}_{name}.csv',index=False)
            for start in (START,'2016-01-01','2020-01-01','2024-01-01'):
                block=frame[frame.date.ge(start)]
                m=wear_metrics(block)
                m.update(case=case,name=name,start=start,cash_days=float(block.position.lt(.01).mean()),halt_days=int(block.halted.sum()))
                rows.append(m)
            for state,label in ((0,'down'),(1,'range'),(2,'up')):
                block=frame[frame.regime.eq(state)]
                r=block['return']
                phases.append({'case':case,'name':name,'state':label,'days':len(block),
                    'annual_mean_return':float(r.mean()*252),'volatility':float(r.std()*np.sqrt(252)),
                    'mean_position':float(block.position.mean()),'negative_day_rate':float(r.lt(0).mean())})
            all_ledgers[case,name]=frame
    result=pd.DataFrame(rows);result.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(phases).to_csv(OUT/'phase_metrics.csv',index=False)
    pd.DataFrame([{'year':y,'name':n,**wear_metrics(block)} for (case,n),f in all_ledgers.items() if case=='base'
                  for y,block in f.groupby(f.date.dt.year)]).to_csv(OUT/'yearly_metrics.csv',index=False)
    # Causal prefix audit: future quotes cannot affect a past decision or fill.
    cfg,downside,stop=bank[default]
    cut=int(market.date.searchsorted(pd.Timestamp('2024-01-01')))
    short=market.iloc[:cut].copy();st,sr=regime_experts(short)
    pd.testing.assert_frame_equal(st,targets.iloc[:cut]);np.testing.assert_array_equal(sr,regimes[:cut])
    reward=wear_expert_returns(market,targets,cases['base'])
    proposal,_=mix_path(targets,reward,regimes,cfg,first,downside)
    pshort,_=mix_path(st,reward.iloc[:cut],sr,cfg,first,downside)
    np.testing.assert_allclose(proposal[:cut],pshort)
    full=risk_simulate(market,proposal,[cfg]*len(market),regimes,cases['base'],np.full(len(market),stop))
    prefix=risk_simulate(short,pshort,[cfg]*len(short),sr,cases['base'],np.full(len(short),stop))
    pd.testing.assert_frame_equal(full.iloc[:len(prefix)].reset_index(drop=True),prefix)
    chosen=['annual_risk_selected','regime_vol_10','regime_account_stop','constant_0.333','constant_0.200']
    chart({n:all_ledgers['base',n] for n in chosen},OUT/'equity.svg','regime and risk controls, base costs')
    text=['# BYD 场景专家与风险控制验证','',
        '上涨：股价高于120日均线3%且60日均线20日斜率超过1%；下跌条件相反；其余为震荡。收盘判定、下一报价日开盘执行，标签不采用事后趋势分段。',
        '新增现金、低仓位、60/120/250日迟滞趋势、上涨跟随、下跌防御、震荡反转、突破、10%/15%目标波动及标的回撤专家。账户回撤控制另行模拟：8%/12%触发，至少20报价日冷静期且上涨状态才能再入场，重新记录风险峰值；触发退出仍受真实成交约束，阈值不是最大回撤保证。',
        '18个候选提前固定；2016起仅用此前3个完整年度选择。年度选型与基础成本路由固定后再做严重/极端磨损，在线专家反馈分别重算。2012-2015统一使用share_vol10_risk1。目标波动是仓位上限，没有杠杆，实际波动不保证等于目标。现金利息设为零。',
        '所有旧数据已观察，非新盲测。基于日线模拟，冲击参数沿用假设，未用真实交割单校准。','',
        '|成本|区间|策略|累计净收益|年化|波动率|最大回撤|平均仓位|空仓天数比例|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for case in cases:
        for start in (START,'2016-01-01','2024-01-01'):
            for n in chosen+['regime_switch','share_vol10_risk1','context_vol10_risk1','multiwindow_vol10_risk1']:
                block=result[(result.case==case)&(result.start==start)&(result.name==n)]
                if len(block):
                    r=block.iloc[0]
                    text.append(f'|{case}|{start}|{n}|{r.total_return:.2%}|{r.cagr:.2%}|{r.realized_volatility:.2%}|{r.max_drawdown:.2%}|{r.average_position:.2%}|{r.cash_days:.2%}|')
    text += ['', '![净值](equity.svg)','',
        '阶段统计见phase_metrics.csv：使用上一收盘状态标记当天收益，只作分组均值/波动描述，不把不连续日期当独立复利净值。分年结果见yearly_metrics.csv，逐年候选评分见inner_scores.csv，权重见weights_base.parquet。',
        '前缀重放通过，未来报价不影响专家、在线权重、账户风控或已发生交易。所有账户现金与股份非负。期末按持仓市值计价，未强制清仓。']
    (OUT/'report.md').write_text('\n'.join(text),encoding='utf-8')
    sources=['src/quant/research/byd_regime.py','scripts/research/run_byd_regime.py','src/quant/research/byd_wear.py']
    (OUT/'manifest.json').write_text(json.dumps({'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'market_sha256':hashlib.sha256((ROOT/'reports/research/byd_mwu_20261012/market_snapshot.parquet').read_bytes()).hexdigest(),
        'causal_prefix_pass':True,'ledger_count':len(all_ledgers)},indent=2))
    print(result[(result.case=='base')&(result.start==START)][['name','total_return','cagr','realized_volatility','max_drawdown','average_position']].to_string(index=False),flush=True)

if __name__=='__main__':
    main()
