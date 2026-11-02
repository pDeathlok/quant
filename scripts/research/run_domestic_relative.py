#!/usr/bin/env python3
"""Benchmark-relative, low-turnover domestic allocation experiment."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from quant.research.byd_mwu import ROOT
from quant.research.byd_wear import WearConfig
from quant.research.byd_allocation import AllocationConfig
from quant.research.byd_growth import paired_return_bootstrap
from quant.research.domestic_portfolio import START,load_panel,simulate,portfolio_metrics,mix_targets
from quant.research.domestic_basket import GROUPS,NAMES
from quant.research.domestic_relative import build_targets,relative_choice
from run_domestic_portfolio import risk_cap
from run_byd_wear import chart,rolling_windows

OUT=ROOT/'reports/research/domestic_relative_20261013'
PREDICTIONS=(ROOT/'reports/research/domestic_portfolio_20261013/model_predictions.npz',ROOT/'reports/research/domestic_growth_20261013/excess_predictions.npz')


def run(panel,path,wear,weekly=False):
    # The user permits small realized drift: no emergency basket repair.
    return simulate(panel,path,wear,frequency=5 if weekly else 21,deadband=.02 if weekly else .03)


def dominance(a,b):
    return {'cagr_gain':a['cagr']-b['cagr'],
            'drawdown_gain':abs(b['max_drawdown'])-abs(a['max_drawdown']),
            'both_better':bool(a['cagr']>b['cagr']+1e-9 and abs(a['max_drawdown'])<abs(b['max_drawdown'])-1e-9)}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    bank={f'{method}_eta{eta}':AllocationConfig(method=method,eta=eta,share=.01,forget=.995)
          for method in ('share','context','multiwindow') for eta in (.1,.3)}
    protocol={'start':START,'history_observed':True,'blind':False,
              'objective':'higher net CAGR AND lower absolute maximum drawdown versus equal baskets, full history plus 2018 onwards',
              'target_caps':dict(zip(NAMES,[.20]*6)),'cash_exempt':True,'realized_drift_allowed':True,
              'frequency':21,'deadband':.03,'emergency_20percent_repair':False,
              'primary_baseline':'same monthly equal basket execution','secondary_baseline':'weekly equal, 5 days and 2ppt deadband, no emergency repair',
              'new_model_fits':0,'tilt_horizons':[63,120,250],'tilt_amplitudes':[.03,.06],
              'model_tilt_amplitude':.03,'weak_market':'CSI300 distance120<-3% AND ret20<-6%',
              'weak_equity_scales':[.7,.4],'volatility_caps':[.16,.18],
              'relative_selection':'prior 3 complete years net CAGR>equal and abs drawdown<equal; rank CAGR gain +.25 drawdown reduction, else equal',
              'initial_policy':'tilt120_3_guard70 for 2014-2017; results reported separately from outer selection',
              'mixes':{k:v.to_dict() for k,v in bank.items()},'stopping':'complete frozen candidates and four cases, no additional candidates based on performance'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    panel=load_panel();forecasts={}
    for file in PREDICTIONS:
        with np.load(file) as c:forecasts.update({k:c[k] for k in c.files})
    targets=build_targets(panel,forecasts)
    for name in ('equal','tilt120_3'):
        for limit in (.16,.18):targets[f'{name}_vol{int(limit*100)}']=risk_cap(panel,targets[name],limit)
    np.savez_compressed(OUT/'expert_targets.npz',**targets)
    protocol['experts']=list(targets);protocol['candidates']=len(targets)+len(bank)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    first=int(panel.dates.searchsorted(pd.Timestamp(START)))
    cases={'base':WearConfig(),'severe':WearConfig(spread=.002,impact_eta=.5),
           'extreme':WearConfig(spread=.005,impact_eta=1),'capacity_10m':WearConfig(initial_cash=10_000_000)}
    routes={};scores=[];rows=[];paired=[];all_frames={};all_experts={};selected_targets={}
    for case,wear in cases.items():
        frames={};rewards=np.zeros((len(panel.dates),len(targets)));paths={};candidates={}
        print(f'{case}: {len(targets)} experts',flush=True)
        for e,(name,path) in enumerate(targets.items()):
            f,orders=run(panel,path,wear);frames[name]=f;rewards[first:,e]=f['return'].to_numpy()
            if case=='base':f.to_csv(OUT/f'expert_{name}.csv',index=False)
        for key,cfg in bank.items():
            paths[key],_=mix_targets(panel,targets,rewards,cfg)
            if case=='base':candidates[key]=run(panel,paths[key],wear)[0]
        if case=='base':
            candidates.update({f'expert_{k}':v for k,v in frames.items()})
            baseline=portfolio_metrics(frames['equal'])
            pd.DataFrame([{'name':n,**portfolio_metrics(f),**dominance(portfolio_metrics(f),baseline)} for n,f in candidates.items()]).to_csv(OUT/'candidate_diagnostic.csv',index=False)
            for year in range(2018,2027):
                routes[year],r=relative_choice(candidates,year);scores.extend(r)
            (OUT/'routes_frozen.json').write_text(json.dumps(routes,indent=2)+'\n')
            pd.DataFrame(scores).to_csv(OUT/'annual_selection_scores.csv',index=False)
        selected=targets['tilt120_3_guard70'].copy()
        schedule=panel.dates.to_series(index=np.arange(len(panel.dates))).shift(-1).fillna(panel.dates[-1]).dt.year.to_numpy()
        for year,key in routes.items():
            mask=schedule==year;selected[mask]=(targets[key[7:]] if key.startswith('expert_') else paths[key])[mask]
        f,orders=run(panel,selected,wear);selected_targets[case]=selected
        orders.to_csv(OUT/f'orders_{case}_selected.csv',index=False)
        weekly,weekly_orders=run(panel,targets['equal'],wear,True)
        final={'selected':f,'equal_monthly':frames['equal'],'equal_weekly':weekly,
               **{name:frames[name] for name in ('low_securities_core','omit_4','tilt120_3','tilt120_3_guard70','model_tilt_ridge_excess21')}}
        for name,frame in final.items():
            frame.to_csv(OUT/f'ledger_{case}_{name}.csv',index=False);all_frames[case,name]=frame
            for start in (START,'2018-01-01','2024-01-01'):
                block=frame[frame.date.ge(start)];m=portfolio_metrics(block)
                for baseline_name in ('equal_monthly','equal_weekly'):
                    b=portfolio_metrics(final[baseline_name][final[baseline_name].date.ge(start)])
                    rows.append({'case':case,'name':name,'start':start,'baseline':baseline_name,**m,**dominance(m,b)})
        for name,frame in frames.items():all_experts[case,name]=frame
        print(f'{case}: done',flush=True)
    table=pd.DataFrame(rows);table.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame([{'name':name,'year':year,**portfolio_metrics(b)} for (case,name),frame in all_frames.items() if case=='base' for year,b in frame.groupby(frame.date.dt.year)]).to_csv(OUT/'yearly_metrics.csv',index=False)
    rolling=[]
    for name in ('selected','equal_monthly','equal_weekly','low_securities_core','omit_4'):rolling.extend(rolling_windows(all_frames['base',name],name))
    roll=pd.DataFrame(rolling);roll.to_csv(OUT/'rolling_windows.csv',index=False)
    roll.groupby(['name','years']).cagr.agg(['min','median','max']).to_csv(OUT/'rolling_summary.csv')
    for start in (START,'2018-01-01','2024-01-01'):
        for name in ('selected','low_securities_core','omit_4'):
            a=all_frames['base',name];a=a[a.date.ge(start)]
            b=all_frames['base','equal_monthly'];b=b[b.date.ge(start)]
            paired.append({'start':start,'strategy':name,**paired_return_bootstrap(a,b,'equal_monthly')})
    pd.DataFrame(paired).to_csv(OUT/'paired_bootstrap.csv',index=False)
    positions=[]
    for name in ('selected','equal_monthly','low_securities_core','omit_4'):
        f=all_frames['base',name]
        positions.append({'name':name,**f.filter(like='position_').mean().to_dict(),'cash_plus_receivable':1-f.position.mean()})
    pd.DataFrame(positions).to_csv(OUT/'average_positions.csv',index=False)
    # Independent truncation, then expert execution and learner replay.
    short=load_panel(end_date='2019-12-31');n=len(short.dates)
    small=build_targets(short,{k:v[:n] for k,v in forecasts.items()})
    for name in ('equal','tilt120_3'):
        for limit in (.16,.18):small[f'{name}_vol{int(limit*100)}']=risk_cap(short,small[name],limit)
    full_rewards=np.zeros((len(panel.dates),len(targets)));short_rewards=np.zeros((n,len(targets)))
    for e,name in enumerate(targets):
        np.testing.assert_allclose(small[name],targets[name][:n],atol=1e-12)
        sf,_=run(short,small[name],WearConfig());ff=all_experts['base',name]
        pd.testing.assert_frame_equal(ff.iloc[:len(sf)].reset_index(drop=True),sf)
        full_rewards[first:,e]=ff['return'];short_rewards[first:,e]=sf['return']
    cfg=next(iter(bank.values()));p,_=mix_targets(panel,targets,full_rewards,cfg);sp,_=mix_targets(short,small,short_rewards,cfg)
    np.testing.assert_allclose(p[:n],sp,atol=1e-12)
    selected_prefix,_=run(short,selected_targets['base'][:n],WearConfig())
    pd.testing.assert_frame_equal(all_frames['base','selected'].iloc[:len(selected_prefix)].reset_index(drop=True),selected_prefix)
    manifest={'prefix_targets_feedback_learner_account_pass':True,'ledgers':len(all_frames),
              'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('src/quant/research/domestic_relative.py','scripts/research/run_domestic_relative.py','src/quant/research/domestic_portfolio.py','tests/test_domestic_relative.py')},
              'prediction_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in PREDICTIONS}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    chart({name:all_frames['base',name] for name in ('selected','equal_monthly','equal_weekly','low_securities_core','omit_4')},OUT/'equity.svg','relative return and drawdown, relaxed actual weight cap')
    lines=['# 相对六篮子等权：宽松仓位、低换手动态配置','','用户放宽实际持仓20%至允许小幅漂移，目标权重仍为六篮子各0—20%、现金豁免。取消每日紧急减仓；目标转为净CAGR更高且最大回撤更低，不再以15%/20%作为必要条件。历史已观察，非盲测。',
           f'固定{len(targets)}专家、{len(targets)+len(bank)}候选，复用既有特征与成熟标签预测，无本轮新训练。主对照同月度等权，每21交易日检查、3个百分点死区；另列原5交易日/2个百分点宽松等权。所有成本/份额/共享现金与旧账本一致。',
           '2014—2017采用120日3个百分点软倾斜与0.7弱市权益缩放，2018起只用此前3完整年，候选当时同时优于同月度等权才入选，否则回等权。相对排序为CAGR增量+0.25回撤减少量。压力情景冻结年度路线，但重算专家反馈。',
           'low_securities_core篮子比例为BYD20%、宽基15%、黄金20%、国债20%、证券5%、红利20%；omit_4为排除证券、其余五篮子各20%。这两个固定对照不是年度动态流程。事后诊断中找到双优固定配置不等于独立样本外认证。','','|成本|起点|方案|年化净收益|最大回撤|相比月度等权收益增量|回撤减少|双优|订单数|','|---|---|---|---:|---:|---:|---:|---|---:|']
    for r in table[table.baseline.eq('equal_monthly')].itertuples():lines.append(f'|{r.case}|{r.start}|{r.name}|{r.cagr:.2%}|{r.max_drawdown:.2%}|{r.cagr_gain:.2%}|{r.drawdown_gain:.2%}|{r.both_better}|{r.orders}|')
    lines+=['','![净值](equity.svg)','','限制：初始段与历史均已经观察，没有真正未来盲测。日线冲击/价格带未校准真实队列；现金利息零，分红税储备近似。重采样未校正多重比较，只描述收益差；CAGR双优与回撤降低不代表未来必胜。完整期、2018起与成本压力应一起评判，不看近期或诊断表挑选胜者。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(table[(table.case=='base')&(table.start==START)&(table.baseline=='equal_monthly')][['name','cagr','max_drawdown','both_better','orders']].to_string(index=False),flush=True)


if __name__=='__main__':main()
