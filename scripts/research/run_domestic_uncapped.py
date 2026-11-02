#!/usr/bin/env python3
"""V3: unrestricted long-only concentration, 20% net CAGR and 30% drawdown goal."""
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
from quant.research.domestic_uncapped import build_targets
from quant.research.domestic_return30 import return_choice
from run_domestic_portfolio import risk_cap
from run_byd_wear import chart,rolling_windows

OUT=ROOT/'reports/research/domestic_uncapped_v3_20261013'
PRED=(ROOT/'reports/research/domestic_portfolio_20261013/model_predictions.npz',ROOT/'reports/research/domestic_growth_20261013/excess_predictions.npz')


def run(panel,path,wear):return simulate(panel,path,wear,frequency=5,deadband=.03)


def make_targets(panel,forecasts):
    raw=build_targets(panel,forecasts);result={}
    for name,path in raw.items():
        for vol in (0.,.25,.35):result[f'{name}_{"raw" if not vol else "vol"+str(int(vol*100))}']=risk_cap(panel,path,vol)
    result['cash']=np.zeros_like(panel.data['close'])
    return result


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    bank={f'{m}_eta{eta}':AllocationConfig(method=m,eta=eta,share=.01,forget=.995)
          for m in ('share','context','multiwindow') for eta in (.1,.3)}
    protocol={'version':3,'history_observed':True,'blind':False,'start':START,
              'target_net_cagr':.20,'target_drawdown':.30,'basket_cap_removed':True,'repair23_removed':True,
              'single_asset_limit':1.,'total_capital_limit':1.,'leverage':False,'shorting':False,
              'new_fits':0,'fixed_cores':12,'relative_rank_horizons':[63,120,250,'multi'],'top':[1,2],
              'trend_gate':'optional distance120 positive','byd_trend_horizons':[20,63,120,250],
              'model_horizons':['ridge_21','ridge_63','ridge_excess21','tree_excess21'],
              'volatility_budgets':[0,.25,.35],'frequency':5,'deadband':.03,
              'annual_selection':'past 3 completed years max drawdown<=30%, maximize net CAGR, cash feasible fallback',
              'initial_policy':'byd_gold_50_vol25 2014-2017','routes_frozen_before_stress':True,
              'fixed_research_selection':'highest full base CAGR among all-case full and 2018 drawdown<=30%; explicit hindsight',
              'stopping':'complete frozen set and all four cases; no new candidates after performance'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    panel=load_panel();forecasts={}
    for file in PRED:
        with np.load(file) as c:forecasts.update({k:c[k] for k in c.files})
    targets=make_targets(panel,forecasts);np.savez_compressed(OUT/'expert_targets.npz',**targets)
    protocol['expert_names']=list(targets);protocol['candidates']=len(targets)+len(bank)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    first=int(panel.dates.searchsorted(pd.Timestamp(START)))
    cases={'base':WearConfig(),'severe':WearConfig(spread=.002,impact_eta=.5),
           'extreme':WearConfig(spread=.005,impact_eta=1),'capacity_10m':WearConfig(initial_cash=10_000_000)}
    routes={};scores=[];diagnostics=[];all_candidates={};all_final={};mix_paths={}
    for case,wear in cases.items():
        frames={};paths={};rewards=np.zeros((len(panel.dates),len(targets)))
        print(f'{case}: {len(targets)} unconstrained concentration experts',flush=True)
        for e,(name,path) in enumerate(targets.items()):
            frame,orders=run(panel,path,wear);frames[f'expert_{name}']=frame;rewards[first:,e]=frame['return'].to_numpy()
            if case=='base':frame.to_csv(OUT/f'expert_{name}.csv',index=False)
            if (e+1)%40==0:print(f'{case}: expert {e+1}/{len(targets)}',flush=True)
        for name,cfg in bank.items():
            paths[name],_=mix_targets(panel,targets,rewards,cfg);frames[name]=run(panel,paths[name],wear)[0]
        for name,frame in frames.items():
            all_candidates[case,name]=frame
            for start in (START,'2018-01-01'):
                m=portfolio_metrics(frame[frame.date.ge(start)])
                diagnostics.append({'case':case,'name':name,'start':start,**m,'goal_met':m['cagr']>.20 and abs(m['max_drawdown'])<=.30})
        if case=='base':
            for year in range(2018,2027):routes[year],rows=return_choice(frames,year);scores.extend(rows)
            (OUT/'routes_frozen.json').write_text(json.dumps(routes,indent=2)+'\n');pd.DataFrame(scores).to_csv(OUT/'annual_selection_scores.csv',index=False)
        selected=targets['byd_gold_50_vol25'].copy()
        schedule=panel.dates.to_series(index=np.arange(len(panel.dates))).shift(-1).fillna(panel.dates[-1]).dt.year.to_numpy()
        for year,key in routes.items():
            mask=schedule==year;selected[mask]=(targets[key[7:]] if key.startswith('expert_') else paths[key])[mask]
        frame,orders=run(panel,selected,wear);all_final[case,'annual_v3']=frame
        orders.to_csv(OUT/f'orders_{case}_annual_v3.csv',index=False)
        # These are retained original versions, not changed to uncapped execution.
        all_final[case,'v1']=pd.read_csv(ROOT/f'reports/research/domestic_strategy_v1_20261013/ledger_{case}.csv',parse_dates=['date','signal_date'])
        all_final[case,'v2']=pd.read_csv(ROOT/f'reports/research/domestic_return30_v2_20261013/ledger_{case}_fixed_v2_candidate.csv',parse_dates=['date','signal_date'])
        all_final[case,'byd_buyhold']=frames['expert_buy_0_raw']
        mix_paths[case]=paths;print(f'{case}: complete',flush=True)
    diagnostic=pd.DataFrame(diagnostics);diagnostic.to_csv(OUT/'candidate_diagnostic.csv',index=False)
    feasible=[name for case,name in all_candidates if case=='base' and diagnostic.loc[diagnostic.name.eq(name),'max_drawdown'].abs().le(.30+1e-9).all()]
    fixed=max(feasible,key=lambda name:portfolio_metrics(all_candidates['base',name])['cagr']) if feasible else 'expert_cash'
    raw_best=max([name for case,name in all_candidates if case=='base'],key=lambda name:portfolio_metrics(all_candidates['base',name])['cagr'])
    (OUT/'research_choices.json').write_text(json.dumps({'best_all_case_dd30':fixed,'raw_highest_return':raw_best,'hindsight_research':True},indent=2)+'\n')
    rows=[]
    for case,wear in cases.items():
        path=targets[fixed[7:]] if fixed.startswith('expert_') else mix_paths[case][fixed]
        f,orders=run(panel,path,wear);all_final[case,'fixed_v3_candidate']=f
        orders.to_csv(OUT/f'orders_{case}_fixed_v3_candidate.csv',index=False)
        all_final[case,'raw_best_diagnostic']=all_candidates[case,raw_best]
        for name in ('annual_v3','fixed_v3_candidate','raw_best_diagnostic','v1','v2','byd_buyhold'):
            frame=all_final[case,name];frame.to_csv(OUT/f'ledger_{case}_{name}.csv',index=False)
            for start in (START,'2018-01-01','2024-01-01'):
                block=frame[frame.date.ge(start)];m=portfolio_metrics(block)
                rows.append({'case':case,'name':name,'start':start,**m,'goal_met':m['cagr']>.20 and abs(m['max_drawdown'])<=.30,
                             'max_actual_single_asset':float(block.filter(like='position_').max().max())})
    result=pd.DataFrame(rows);result.to_csv(OUT/'metrics.csv',index=False)
    roll=[]
    for name in ('annual_v3','fixed_v3_candidate','v1','v2'):roll.extend(rolling_windows(all_final['base',name],name))
    r=pd.DataFrame(roll);r.to_csv(OUT/'rolling_windows.csv',index=False);r.groupby(['name','years']).cagr.agg(['min','median','max']).to_csv(OUT/'rolling_summary.csv')
    pd.DataFrame([{'name':name,'year':year,**portfolio_metrics(b)} for (case,name),f in all_final.items() if case=='base' for year,b in f.groupby(f.date.dt.year)]).to_csv(OUT/'yearly_metrics.csv',index=False)
    paired=[]
    for start in (START,'2018-01-01','2024-01-01'):
        for name in ('annual_v3','fixed_v3_candidate'):
            a=all_final['base',name];a=a[a.date.ge(start)];b=all_final['base','v2'];b=b[b.date.ge(start)]
            paired.append({'start':start,'strategy':name,**paired_return_bootstrap(a,b,'v2')})
    pd.DataFrame(paired).to_csv(OUT/'paired_bootstrap.csv',index=False)
    short=load_panel(end_date='2019-12-31');n=len(short.dates);small=make_targets(short,{k:v[:n] for k,v in forecasts.items()})
    small_frames={};small_rewards=np.zeros((n,len(targets)));full_rewards=np.zeros((len(panel.dates),len(targets)))
    for e,name in enumerate(targets):
        np.testing.assert_allclose(small[name],targets[name][:n],atol=1e-12)
        f,_=run(short,small[name],WearConfig());small_frames[f'expert_{name}']=f
        full=all_candidates['base',f'expert_{name}'];pd.testing.assert_frame_equal(full.iloc[:len(f)].reset_index(drop=True),f)
        small_rewards[first:,e]=f['return'];full_rewards[first:,e]=full['return']
    for name,cfg in bank.items():
        p,_=mix_targets(short,small,small_rewards,cfg);long,_=mix_targets(panel,targets,full_rewards,cfg)
        np.testing.assert_allclose(p,long[:n],atol=1e-12);small_frames[name]=run(short,p,WearConfig())[0]
    for year in (2018,2019):
        key,_=return_choice(small_frames,year)
        if key!=routes[year]:raise RuntimeError('Future-dependent annual V3')
    manifest={'independent_prefix_and_annual_routes_pass':True,'final_ledgers':len(all_final),'candidates':len(targets)+len(bank),
              'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('src/quant/research/domestic_uncapped.py','scripts/research/run_domestic_uncapped.py','src/quant/research/domestic_portfolio.py','tests/test_domestic_uncapped.py')},
              'preserved_config_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'configs/strategies/domestic_low_securities_vol16_v1.json',ROOT/'configs/strategies/domestic_return30_v2.json')}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    chart({name:all_final['base',name] for name in ('annual_v3','fixed_v3_candidate','v1','v2','byd_buyhold')},OUT/'equity.svg','uncapped concentration V3, net return and drawdown')
    cfg={'strategy_id':'domestic_uncapped_v3','status':'research_candidate','target_net_cagr':.20,'target_max_drawdown':.30,
         'basket_cap':None,'total_capital_limit':1.,'leverage':False,'fixed_policy':fixed,'research_selection_uses_observed_history':True,
         'annual_policy_separate':'routes_frozen.json','frequency':5,'deadband':.03,'repair23':False,
         'report':str((OUT/'report.md').relative_to(ROOT)),'live_trading_enabled':False}
    (ROOT/'configs/strategies/domestic_uncapped_v3.json').write_text(json.dumps(cfg,indent=2)+'\n')
    lines=['# V3：移除集中度限制，目标净年化20%以上','','用户移除20%篮子限制，V3单标的可0—100%，无杠杆/做空，总投资<=100%；移除23%漂移修复。继续现有资产池与特征、共享资金、交易磨损和30%历史回撤约束。V1/V2原样保留，按原执行规则加载真实账本，未改为V3规则。',
           f'冻结{len(targets)}专家/{len(targets)+len(bank)}候选：单资产与集中配置、多周期前1/2轮动、BYD趋势与金债防御、成熟标签模型择一/二；0/25/35%波动预算，share/context/multiwindow融合。无新训练。历史已观察，非盲测。',
           'annual_v3在2014—2017初始化BYD黄金50/50与25%波动预算；2018起只按此前3完整年回撤<=30%候选最大化净CAGR，现金可行兜底。路线固定后计算成本/容量。',
           f'固定研究可行为{fixed}，在四成本容量情景、完整期与2018起都回撤<=30%的候选中按基础完整期收益排序；原始最高收益为{raw_best}。两者都是显式全历史诊断，不是年度过去数据决定或盲测。','','|成本|起点|方案|净年化|最大回撤|20%/30%达标|平均仓位|订单数|','|---|---|---|---:|---:|---|---:|---:|']
    for row in result.itertuples():lines.append(f'|{row.case}|{row.start}|{row.name}|{row.cagr:.2%}|{row.max_drawdown:.2%}|{row.goal_met}|{row.average_position:.2%}|{row.orders}|')
    lines+=['','![净值](equity.svg)','','专项测试与独立截断专家/反馈/学习器/真实账户/年度路线前缀检查见日志与manifest。日线冲击/价格带未校准队列；分红税储备近似，现金利息零；重采样未校正多轮研究。20%年化及30%回撤是验证目标，不能以近期区间、事后最佳或放宽成交假设宣称未来达标。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(result[(result.case=='base')&(result.start==START)][['name','cagr','max_drawdown','goal_met']].to_string(index=False),flush=True)
    print('research choices',fixed,raw_best,flush=True)


if __name__=='__main__':main()
