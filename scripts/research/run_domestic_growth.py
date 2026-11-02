#!/usr/bin/env python3
"""Second declared round: separate asset selection from risk normalization."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from quant.research.byd_allocation import AllocationConfig
from quant.research.byd_mwu import ROOT,_RidgeExpert
from quant.research.byd_growth import paired_return_bootstrap
from quant.research.byd_wear import WearConfig
from quant.research.domestic_portfolio import START,CODES,Panel,load_panel,matured_labels,capped_weights,simulate,portfolio_metrics,mix_targets
from run_domestic_portfolio import annual_choice,risk_cap
from run_byd_wear import chart,rolling_windows

PREVIOUS=ROOT/'reports/research/domestic_portfolio_20261013'
OUT=ROOT/'reports/research/domestic_growth_20261013'


def excess_models(panel: Panel) -> tuple[dict[str,np.ndarray],list[dict]]:
    h=21;labels=matured_labels(panel,h)
    labels=labels-labels[:,1:2]
    n,k=labels.shape;forecasts={name:np.zeros((n,k)) for name in ('ridge_excess21','tree_excess21')};audits=[]
    for name in forecasts:
        model=None
        for i in range(600,n):
            if model is None or (i-600)%21==0:
                end=i-h-1;begin=max(252,end-1259)
                y=labels[begin:end+1].reshape(-1);x=panel.features[begin:end+1].reshape(-1,panel.features.shape[2])
                mask=np.isfinite(y)
                if mask.sum()<600:continue
                model=HistGradientBoostingRegressor(max_iter=40,max_depth=3,max_leaf_nodes=8,min_samples_leaf=80,learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=20261013) if name.startswith('tree') else _RidgeExpert()
                model.fit(x[mask],y[mask]);audits.append({'model':name,'fit_date':str(panel.dates[i].date()),'label_end':str(panel.dates[end+h+1].date()),'samples':int(mask.sum())})
            if model is not None:forecasts[name][i]=model.predict(panel.features[i])/h
        print(f'{name}: past-only pooled training complete',flush=True)
    return forecasts,audits


def rank_targets(panel: Panel, score: np.ndarray, gate: np.ndarray, top: int, core: bool=False) -> np.ndarray:
    n,k=score.shape;target=np.zeros((n,k));eligible=panel.data['eligible'].astype(bool)
    base=np.array([.05,.15,.10,.25,.30,.05,.10])
    for i in range(n):
        available=eligible[i]&gate[i]&np.isfinite(score[i])&(score[i]>.0001)
        selected=np.flatnonzero(available)
        selected=selected[np.argsort(score[i,selected])[-top:]]
        raw=np.zeros(k);raw[selected]=1/top
        if core:raw=.5*raw+.5*base*eligible[i]
        w=capped_weights(raw)
        # Defense is selected separately; gold needs positive trend, bonds need history.
        distance=panel.features[i,:,panel.feature_names.index('distance_120')]
        if eligible[i,3] and distance[3]>0:w[3]+=min(.60-w[3],1-w.sum())
        if eligible[i,4]:w[4]+=min(.70-w[4],1-w.sum())
        target[i]=w
    return target


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    # Freeze this additional round before calculating its performance; prior results already observed.
    protocol={'assets':CODES,'start':START,'blind':False,'previous_round_observed':True,
        'new_labels':'net 21-day holding return minus simultaneous CSI300 ETF net holding return',
        'new_models':['ridge_excess21','tree_excess21'],'ridge_alpha':50,'tree_max_iter':40,
        'feature_definition':'same existing 21 state_features; no extra external features',
        'selection_change':'raw momentum/return rank, followed independently by fixed concentration and portfolio volatility constraints',
        'top_counts':[2,3],'core_blends':[0,.5],'model_horizons':[5,21,63],
        'portfolio_volatility_limits':[0,.15,.20],'methods':['share','multiwindow','context'],'etas':[.1,.3],
        'score':'same prior 3 completed years, drawdown and vol <=20% then CAGR minus annual std penalty',
        'stopping':'complete declared second round and base/severe/extreme/capacity, no more expansions based on results'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    panel=load_panel();extra,audit=excess_models(panel)
    pd.DataFrame(audit).to_csv(OUT/'model_training_audit.csv',index=False);np.savez_compressed(OUT/'excess_predictions.npz',**extra)
    with np.load(PREVIOUS/'model_predictions.npz') as cache:forecasts={k:cache[k] for k in cache.files}
    forecasts.update(extra)
    experts={};idx={n:i for i,n in enumerate(panel.feature_names)}
    for h in (63,120,250):
        score=panel.features[:,:,idx[f'ret_{h}']]
        gate=panel.features[:,:,idx['distance_120']]>0
        for top in (2,3):experts[f'raw_mom{h}_top{top}']=rank_targets(panel,score,gate,top)
    for name in ('ridge_5','ridge_21','ridge_63','ridge_excess21','tree_excess21'):
        for core in (False,True):
            experts[f'{name}_{"core" if core else "raw"}']=rank_targets(panel,forecasts[name],np.ones_like(forecasts[name],dtype=bool),3,core)
    with np.load(PREVIOUS/'expert_targets.npz') as cache:
        for name in ('balanced','growth','gold_bond'):experts[name]=cache[name]
    np.savez_compressed(OUT/'expert_targets.npz',**experts)
    bank={}
    for method in ('share','multiwindow','context'):
        for eta in (.1,.3):
            for vol in (0.,.15,.20):
                bank[f'{method}_eta{eta}_vol{vol}']=(AllocationConfig(method=method,eta=eta,share=.01,forget=.995),vol)
    protocol['expert_names']=list(experts);protocol['candidate_keys']=list(bank)+[f'expert_{n}' for n in experts]
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    routes={};selections=[];all_frames={};rows=[];cases={'base':WearConfig(),'severe':WearConfig(spread=.002,impact_eta=.5),'extreme':WearConfig(spread=.005,impact_eta=1),'capacity_10m':WearConfig(initial_cash=10_000_000)}
    first=int(panel.dates.searchsorted(pd.Timestamp(START)))
    for case,wear in cases.items():
        print(f'{case}: {len(experts)} expert shared accounts',flush=True)
        frames={};rewards=np.zeros((len(panel.dates),len(experts)))
        for e,(name,target) in enumerate(experts.items()):
            frame,orders=simulate(panel,target,wear);frames[name]=frame;rewards[first:,e]=frame['return'].to_numpy()
            if case=='base':frame.to_csv(OUT/f'expert_{name}.csv',index=False)
        paths={};candidate_frames={}
        for name,(cfg,vol) in bank.items():
            path,_=mix_targets(panel,experts,rewards,cfg);paths[name]=risk_cap(panel,path,vol)
            if case=='base':candidate_frames[name]=simulate(panel,paths[name],wear)[0]
        if case=='base':
            candidate_frames.update({f'expert_{n}':f for n,f in frames.items()})
            pd.DataFrame([{'name':n,**portfolio_metrics(f)} for n,f in candidate_frames.items()]).to_csv(OUT/'candidate_diagnostic.csv',index=False)
            for year in range(2018,2027):
                routes[year],selection=annual_choice(candidate_frames,year);selections.extend(selection)
            (OUT/'routes_frozen.json').write_text(json.dumps(routes,indent=2));pd.DataFrame(selections).to_csv(OUT/'annual_selection_scores.csv',index=False)
        target=experts['balanced'].copy();schedule=panel.dates.to_series(index=np.arange(len(panel.dates))).shift(-1).fillna(panel.dates[-1]).dt.year.to_numpy()
        for year,key in routes.items():
            mask=schedule==year;target[mask]=(experts[key[7:]] if key.startswith('expert_') else paths[key])[mask]
        selected,orders=simulate(panel,target,wear);orders.to_csv(OUT/f'orders_{case}.csv',index=False)
        final={'selected_round2':selected}
        for name in ('balanced','growth','raw_mom120_top2','ridge_21_raw','ridge_excess21_raw','tree_excess21_core'):final[name]=frames[name]
        old=pd.read_csv(PREVIOUS/f'ledger_{case}_selected_pipeline.csv',parse_dates=['date','signal_date']);final['selected_round1']=old
        for name,frame in final.items():
            frame.to_csv(OUT/f'ledger_{case}_{name}.csv',index=False)
            for start in (START,'2017-09-01','2018-01-01','2020-01-01','2024-01-01'):
                rows.append({'case':case,'name':name,'start':start,**portfolio_metrics(frame[frame.date.ge(start)])})
            all_frames[case,name]=frame
    result=pd.DataFrame(rows);result.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame([{'name':n,'year':year,**portfolio_metrics(b)} for (case,n),frame in all_frames.items() if case=='base' for year,b in frame.groupby(frame.date.dt.year)]).to_csv(OUT/'yearly_metrics.csv',index=False)
    roll=[]
    for n in ('selected_round2','selected_round1','balanced','raw_mom120_top2'):roll.extend(rolling_windows(all_frames['base',n],n))
    r=pd.DataFrame(roll);r.to_csv(OUT/'rolling_windows.csv',index=False);r.groupby(['name','years']).cagr.agg(['min','median','max']).to_csv(OUT/'rolling_summary.csv')
    bs=[]
    for start in (START,'2018-01-01','2024-01-01'):
        a=all_frames['base','selected_round2'];a=a[a.date.ge(start)]
        for n in ('balanced','growth','selected_round1'):
            b=all_frames['base',n];b=b[b.date.ge(start)];bs.append({'start':start,**paired_return_bootstrap(a,b,n)})
    pd.DataFrame(bs).to_csv(OUT/'paired_bootstrap.csv',index=False)
    cut=int(panel.dates.searchsorted(pd.Timestamp('2020-01-01')))
    prefix=Panel(panel.dates[:cut],{k:v[:cut] for k,v in panel.data.items()},panel.features[:cut],panel.feature_names,[e for e in panel.dividends if e['ex_index']<cut],panel.coverage)
    forecasts_prefix,_=excess_models(prefix)
    for name in extra:np.testing.assert_allclose(extra[name][:cut],forecasts_prefix[name],rtol=1e-10,atol=1e-10)
    for entry in audit:
        if pd.Timestamp(entry['label_end'])>pd.Timestamp(entry['fit_date']):raise RuntimeError('Future label included')
    chart({n:all_frames['base',n] for n in ('selected_round2','selected_round1','balanced','growth','raw_mom120_top2')},OUT/'equity.svg','domestic assets, return selection and risk budget')
    manifest={'model_prefix_pass':True,'labels_mature':True,'ledgers':len(all_frames),'experts':len(experts),'candidates':len(bank)+len(experts),
        'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/research/run_domestic_growth.py','src/quant/research/domestic_portfolio.py','scripts/research/run_domestic_portfolio.py']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    text=['# 境内多资产第二轮：收益选择与风险预算分离','',
        '第一轮结果已经观察，第二轮不是盲测。固定原资产池、仓位限制、交易成本与信号时间，仅用既有21特征；新增相对沪深300的21日超额收益label，滚动Ridge及小型树模型训练。新增纯收益/动量排名，风险预算独立施加；规则候选与模型候选共同参与过去3完整年度选型。训练风险约束不保证后续账户回撤上限。',
        '固定均衡配置为BYD5%、沪深30015%、中证50010%、黄金25%、国债30%、证券5%、红利10%；成长配置为10%、20%、15%、25%、15%、5%、10%。资产尚未满足252报价日预热的配额保留为现金。其余成本/公司行为/数据限制详见第一轮报告。','',
        '|成本|起点|方案|年化净收益|波动率|最大回撤|订单数|',
        '|---|---|---|---:|---:|---:|---:|']
    for case in cases:
        for start in (START,'2018-01-01','2024-01-01'):
            for n in ('selected_round2','selected_round1','balanced','growth','raw_mom120_top2','ridge_excess21_raw','tree_excess21_core'):
                row=result[(result.case==case)&(result.start==start)&(result.name==n)].iloc[0]
                text.append(f'|{case}|{start}|{n}|{row.cagr:.2%}|{row.volatility:.2%}|{row.max_drawdown:.2%}|{int(row.orders)}|')
    text+=['','![净值](equity.svg)','',
        '2014-08-01起最长真实交易期动态纳入新ETF；证券ETF2016-08-08上市，2017-09-01起共同成熟区间另列；2018起是过去3年逐年选型区间。没有拼接上市前指数作为成交价格。',
        '目标仍为扣费年化>15%且最大回撤<20%，是否达标需要同时检查完整期、逐年选型期及成本压力，不能根据某个好年份宣布成功。模型预测超额收益不代表能够锁定超额收益。',
        '市场费率、分红到账、份额折算和共享现金限制复用第一轮账本；ETF涨跌停仍按10%近似、缺少真实订单队列和校准的冲击参数，现金利息零，期末未强制卖出。配对重采样未校正多轮研究与多重比较。完整固定第二轮结束后停止，不继续看结果扩参。']
    (OUT/'report.md').write_text('\n'.join(text),encoding='utf-8')
    print(result[(result.case=='base')&(result.start==START)][['name','cagr','volatility','max_drawdown']].to_string(index=False))

if __name__=='__main__':main()
