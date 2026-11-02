#!/usr/bin/env python3
"""Frozen domestic multi-asset model and allocation experiment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from quant.research.byd_allocation import AllocationConfig
from quant.research.byd_growth import paired_return_bootstrap
from quant.research.byd_mwu import ROOT
from quant.research.byd_wear import WearConfig
from quant.research.domestic_portfolio import CODES,START,Panel,load_panel,fit_models,expert_targets,simulate,portfolio_metrics,mix_targets
from run_byd_wear import chart,rolling_windows

OUT=ROOT/'reports/research/domestic_portfolio_20261013'


def risk_cap(panel: Panel, targets: np.ndarray, limit: float) -> np.ndarray:
    if not limit:return targets
    returns=pd.DataFrame(panel.data['adjusted']).replace(0,np.nan).pct_change(fill_method=None).fillna(0).to_numpy()
    output=targets.copy()
    for i in range(63,len(targets)):
        past=returns[i-62:i+1]
        path=np.einsum('na,a->n',past,targets[i]);vol=path.std(ddof=1)*np.sqrt(252)
        if vol>limit:output[i]*=limit/vol
    return output


def annual_choice(ledgers: dict[str,pd.DataFrame], year: int) -> tuple[str,list[dict]]:
    rows=[]
    for name,frame in ledgers.items():
        past=frame[(frame.date.dt.year>=year-3)&(frame.date.dt.year<year)]
        m=portfolio_metrics(past)
        yearly=[portfolio_metrics(past[past.date.dt.year.eq(y)])['cagr'] for y in range(year-3,year)]
        violation=max(0.,abs(m['max_drawdown'])-.20)+max(0.,m['volatility']-.20)
        score=m['cagr']-.15*np.std(yearly)
        rows.append({'year':year,'name':name,**m,'violation':violation,'feasible':violation<1e-12,'score':score})
    best=max(rows,key=lambda r:(r['feasible'],-r['violation'],r['score']))
    for r in rows:r['selected']=r['name']==best['name']
    return best['name'],rows


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    bank={}
    for group in ('rules','all'):
        for method in ('share','multiwindow','context'):
            for eta in (.1,.3):
                for vol in (0.,.15):
                    key=f'{group}_{method}_eta{eta}_vol{vol}'
                    bank[key]=(AllocationConfig(method=method,eta=eta,share=.01,forget=.995),group,vol)
    protocol={'assets':CODES,'cash_interest':0.,'start':START,'outer_years':list(range(2018,2027)),
        'target_cagr':.15,'target_max_drawdown':.20,'blind':False,'candidate_count':len(bank),
        'candidates':{n:{'config':c.to_dict(),'group':g,'volatility_limit':v} for n,(c,g,v) in bank.items()},
        'models':['ridge_5','ridge_21','ridge_63','ridge_downside21','tree_downside21'],
        'model_hyperparameters':{'training_lookback':1260,'refit_every':21,'ridge_alpha':50,'tree_iterations':40,'tree_depth':3,'tree_leaves':8,'tree_l2':1,'early_stopping':False},
        'labels':'next-open to h+1 open return, less base roundtrip 16bps and BYD stamp; downside variant subtracts 0.5 maximum adverse excursion',
        'features':'reuse byd_allocation.state_features definitions, 21 features; no new external data features',
        'caps':{'BYD':.10,'securities':.15,'all_equities':.70,'gold':.60,'bond':.70},
        'selection':'prior 3 full calendar years: feasible drawdown and volatility <=20%, then CAGR -0.15 annual CAGR std',
        'frequency':5,'deadband':.02,'stress_routes_frozen':True,'stopping':'fixed bank, models and four cost/capacity cases; no post-result candidates'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    panel=load_panel();(OUT/'coverage.json').write_text(json.dumps(panel.coverage,indent=2))
    pd.DataFrame(panel.dividends).to_csv(OUT/'dividend_events.csv',index=False)
    model_cache=OUT/'model_predictions.npz';audit_cache=OUT/'model_training_audit.csv'
    if model_cache.exists():
        with np.load(model_cache) as cache:forecasts={k:cache[k] for k in cache.files}
        audit=pd.read_csv(audit_cache).to_dict('records')
    else:
        forecasts,audit=fit_models(panel);np.savez_compressed(model_cache,**forecasts);pd.DataFrame(audit).to_csv(audit_cache,index=False)
    if not all(pd.Timestamp(r['label_end'])<=pd.Timestamp(r['fit_date']) for r in audit):raise RuntimeError('Immature model labels')
    experts=expert_targets(panel,forecasts)
    rules={n:t for n,t in experts.items() if not n.startswith(('ridge','tree'))}
    np.savez_compressed(OUT/'expert_targets.npz',**experts)
    first=int(panel.dates.searchsorted(pd.Timestamp(START)));routes={};selections=[];all_frames={};metrics=[];candidates={};phase_rows=[]
    cases={'base':WearConfig(),'severe':WearConfig(spread=.002,impact_eta=.5),'extreme':WearConfig(spread=.005,impact_eta=1),'capacity_10m':WearConfig(initial_cash=10_000_000)}
    for case,wear in cases.items():
        print(f'{case}: {len(experts)} expert ledgers, shared cash execution',flush=True)
        expert_frames={};reward=np.zeros((len(panel.dates),len(experts)))
        for e,(name,target) in enumerate(experts.items()):
            frame,orders=simulate(panel,target,wear);expert_frames[name]=frame
            reward[first:,e]=frame['return'].to_numpy()
            if case=='base':
                frame.to_csv(OUT/f'expert_{name}.csv',index=False);orders.to_csv(OUT/f'expert_orders_{name}.csv',index=False)
        proposals={};weights={}
        for count,(key,(cfg,group,vol)) in enumerate(bank.items(),1):
            universe=experts if group=='all' else rules
            indices=[list(experts).index(n) for n in universe]
            target,w=mix_targets(panel,universe,reward[:,indices],cfg)
            proposals[key]=risk_cap(panel,target,vol);weights[key]=(w,list(universe))
            if case=='base':candidates[key]=simulate(panel,proposals[key],wear)[0]
        # Direct experts also participate in past-only selection; prevent forced complex blends.
        if case=='base':
            candidates.update({f'expert_{n}':frame for n,frame in expert_frames.items()})
            pd.DataFrame([{'name':n,**portfolio_metrics(f)} for n,f in candidates.items()]).to_csv(OUT/'candidate_metrics_diagnostic.csv',index=False)
            for year in range(2018,2027):
                selected,rows=annual_choice(candidates,year);routes[year]=selected;selections.extend(rows)
            (OUT/'routes_frozen.json').write_text(json.dumps(routes,indent=2));pd.DataFrame(selections).to_csv(OUT/'annual_selection_scores.csv',index=False)
            print('Past-only annual routes frozen',flush=True)
        target=experts['balanced'].copy();schedule=panel.dates.to_series(index=np.arange(len(panel.dates))).shift(-1).fillna(panel.dates[-1]).dt.year.to_numpy()
        used_weights=pd.DataFrame(0.,index=panel.dates,columns=list(experts))
        for year,key in routes.items():
            mask=schedule==year
            if key.startswith('expert_'):
                name=key[len('expert_'):];target[mask]=experts[name][mask];used_weights.loc[mask,name]=1
            else:
                target[mask]=proposals[key][mask];w,names=weights[key];used_weights.loc[mask,names]=w[mask]
        frames={};frames['selected_pipeline'],orders=simulate(panel,target,wear)
        orders.to_csv(OUT/f'orders_{case}_selected.csv',index=False);used_weights.to_parquet(OUT/f'weights_{case}.parquet')
        for name in ('balanced','growth','gold_bond','mom120_top2','ridge_21','ridge_downside21','tree_downside21'):
            frames[name]=expert_frames[name]
        for a in (3,4,5,6):
            t=np.zeros_like(target);t[:,a]=panel.data['eligible'][:,a]
            frames[f'buy_{CODES[a]}']=simulate(panel,t,wear)[0]
        for name,frame in frames.items():
            frame.to_csv(OUT/f'ledger_{case}_{name}.csv',index=False)
            for start in (START,'2017-09-01','2018-01-01','2020-01-01','2024-01-01'):
                block=frame[frame.date.ge(start)]
                metrics.append({'case':case,'name':name,'start':start,**portfolio_metrics(block)})
            for code in CODES:
                phase_rows.append({'case':case,'name':name,'code':code,'mean_position':float(frame[f'position_{code}'].mean())})
            all_frames[case,name]=frame
    result=pd.DataFrame(metrics);result.to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(phase_rows).to_csv(OUT/'position_attribution.csv',index=False)
    pd.DataFrame([{'name':name,'year':year,**portfolio_metrics(b)} for (case,name),f in all_frames.items() if case=='base'
                  for year,b in f.groupby(f.date.dt.year)]).to_csv(OUT/'yearly_metrics.csv',index=False)
    rolls=[]
    for name in ('selected_pipeline','balanced','growth','mom120_top2','gold_bond'):
        rolls.extend(rolling_windows(all_frames['base',name],name))
    rolling=pd.DataFrame(rolls);rolling.to_csv(OUT/'rolling_calendar_windows.csv',index=False)
    rolling.groupby(['name','years']).cagr.agg(['min','median','max']).to_csv(OUT/'rolling_summary.csv')
    paired=[]
    for start in (START,'2018-01-01','2024-01-01'):
        a=all_frames['base','selected_pipeline'];a=a[a.date.ge(start)]
        for name in ('balanced','growth','mom120_top2'):
            b=all_frames['base',name];b=b[b.date.ge(start)]
            paired.append({'start':start,**paired_return_bootstrap(a,b,name)})
    pd.DataFrame(paired).to_csv(OUT/'paired_bootstrap.csv',index=False)
    # Refit the pooled model to a shorter history and confirm identical forecasts.
    cut=int(panel.dates.searchsorted(pd.Timestamp('2020-01-01')))
    prefix=Panel(panel.dates[:cut],{k:v[:cut] for k,v in panel.data.items()},panel.features[:cut],panel.feature_names,
                 [e for e in panel.dividends if e['ex_index']<cut],panel.coverage)
    prefix_predictions,_=fit_models(prefix)
    for name in forecasts:np.testing.assert_allclose(prefix_predictions[name],forecasts[name][:cut],rtol=1e-10,atol=1e-10)
    pe=expert_targets(prefix,prefix_predictions)
    for name in experts:np.testing.assert_allclose(pe[name],experts[name][:cut])
    pf,_=simulate(prefix,pe['mom120_top2']);pd.testing.assert_frame_equal(pf,expert_frames_prefix(all_frames['base','mom120_top2'],len(pf)))
    manifest={'prefix_models_experts_account_pass':True,'labels_mature':True,'ledgers':len(all_frames),
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/quant/research/domestic_portfolio.py','scripts/research/run_domestic_portfolio.py','tests/test_domestic_portfolio.py']},
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'source').glob('*.parquet') if len(p.stem.split('_'))<=3}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    chart({n:all_frames['base',n] for n in ('selected_pipeline','balanced','growth','mom120_top2','gold_bond')},OUT/'equity.svg','domestic ETF + BYD, shared account net costs')
    text=['# 境内多资产与模型融合回测','',
        '资产：BYD、沪深300、中证500、黄金ETF、国债ETF、证券ETF、红利ETF、现金；无海外权益、无融资、无做空。2014-08-01起动态纳入满足252报价日预热的资产。证券ETF2016-08-08上市，预热后才进入策略；2017-09-01起是全资产已成熟的共同对照区间，2018起逐年选型用此前3个完整年度。',
        '新模型仅使用已有state_features定义的21个特征，滚动成熟label训练。5/21/63日下一开盘持有收益label扣除基础双边佣金滑点16bps及BYD卖出印花税；21日风险label另减半个最大不利波动。训练label成本为近似，实际表现仍按共享资金和真实交易重算。模型每21日重训一次、最多1260日历史，预测不含未来标签。',
        '单一共享现金账户，先卖后买、100份/股买入整手、最低佣金5元、万3佣金、基础单边5bps和动态冲击eta0.1，参与率上限为前日成交量1%。ETF不收股票印花税或股票过户费。每5交易日检查、2个百分点不交易区间。分红除息日确认应收、实际派息日转可用现金；基金费用已含在市场价格和收益中，不重复扣除。',
        '中证500ETF2015-04-14份额折算0.28032483，停牌日按机械除权价格估值、零碎份额向下取整。BYD沿用上一轮公司行为与20%分红税储备近似，并在除息日计入现金；其与ETF实际支付日口径不同需保留限制。缺报价只用过去价格估值、禁止成交。ETF涨跌停按10%价格带近似，未取得逐日完整官方限制/队列，实际可成交仍需进一步校准。','',
        '|成本|起点|方案|年化净收益|波动率|最大回撤|平均持仓|订单数|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for case in cases:
        for start in (START,'2018-01-01','2024-01-01'):
            for name in ('selected_pipeline','balanced','growth','mom120_top2','gold_bond','ridge_21','ridge_downside21','tree_downside21'):
                r=result[(result.case==case)&(result.start==start)&(result.name==name)].iloc[0]
                text.append(f'|{case}|{start}|{name}|{r.cagr:.2%}|{r.volatility:.2%}|{r.max_drawdown:.2%}|{r.average_position:.2%}|{int(r.orders)}|')
    text+=['','![净值](equity.svg)','',
        '候选完整期指标只作诊断，不能用事后最优候选替代selected_pipeline。全年选型基础成本路由冻结，严重20bps/eta0.5、极端50bps/eta1和1000万元本金下只更新实时专家反馈与真实账本。所有资产市值和应收股息计入期末，未强制清仓，现金利息为零。',
        '置信区间为20日块、2000次的配对重采样，未校正多重搜索及已有研究观察。黄金大涨等特定区间可能驱动结果，资产权重归因、滚动及逐年指标另附。新数据未留作完全未观察盲测，因此不能承诺未来年化15%/回撤20%。',
        '证券ETF上市依据：https://www.sse.com.cn/disclosure/fund/announcement/c/2016-08-03/512880_20160803_2.pdf',
        '份额折算依据：https://www.sse.com.cn/disclosure/fund/announcement/c/2015-04-15/510500_20150415_2.pdf',
        '行情及分红口径：https://tushare.pro/document/2?doc_id=127 、 https://tushare.pro/document/2?doc_id=120']
    (OUT/'report.md').write_text('\n'.join(text),encoding='utf-8')
    print(result[(result.case=='base')&(result.start==START)][['name','cagr','volatility','max_drawdown','average_position']].to_string(index=False),flush=True)


def expert_frames_prefix(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    return frame.iloc[:rows].reset_index(drop=True)

if __name__=='__main__':main()
