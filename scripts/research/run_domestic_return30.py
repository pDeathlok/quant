#!/usr/bin/env python3
"""Independent V2: maximize net return with a thirty-percent drawdown budget."""
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
from quant.research.domestic_return30 import base_targets,return_choice
from run_domestic_portfolio import risk_cap
from run_byd_wear import chart,rolling_windows

OUT=ROOT/'reports/research/domestic_return30_v2_20261013'
V1=ROOT/'reports/research/domestic_relative_risk_20261013'


def run(panel,path,wear,weekly=False):
    return simulate(panel,path,wear,frequency=5 if weekly else 21,deadband=.02 if weekly else .03,
                    basket_groups=GROUPS,basket_limit=.23)


def make_targets(panel,v1):
    raw=base_targets(panel);result={}
    for name,path in raw.items():
        vols=(.18,.22,0.) if not name.startswith('rank') else (.22,0.)
        for vol in vols:result[f'{name}_{"raw" if not vol else "vol"+str(int(vol*100))}']=risk_cap(panel,path,vol)
    result['v1']=v1
    return result


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    mixes={f'{m}_eta{eta}':AllocationConfig(method=m,eta=eta,share=.01,forget=.995)
           for m in ('share','context','multiwindow') for eta in (.1,.3)}
    protocol={'version':2,'history_observed':True,'blind':False,'v1_preserved':True,
              'objective':'maximize net CAGR, maximum absolute drawdown <=30%',
              'basket_target_caps':dict(zip(NAMES,[.20]*6)),'cash_exempt':True,'new_model_fits':0,
              'core_types':['lowsec','nosec','offensive','nobond','equal'],
              'broad_modes':['csi300','blend','momentum63'],'volatility_budgets':[.18,.22,0],
              'rank_top5_horizons':[63,120,250],'rank_volatility_budgets':[.22,0],
              'execution':{'frequency':21,'deadband':.03,'actual_repair_threshold':.23},
              'annual_selection':'past 3 completed years drawdown<=30%, highest net CAGR; none feasible falls back to V1',
              'initialization':'V1 for 2014-2017','routes_frozen_before_stress':True,
              'fixed_candidate_research_selection':'highest full base CAGR subject to full and 2018 onwards drawdown <=30% in ALL four cases; explicit hindsight, not outer return',
              'stopping':'complete frozen set and four cases; no new candidates after results'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    panel=load_panel()
    with np.load(V1/'expert_targets.npz') as c:v1=c['core_vol16']
    targets=make_targets(panel,v1)
    protocol['experts']=list(targets);protocol['candidates']=len(targets)+len(mixes)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    np.savez_compressed(OUT/'expert_targets.npz',**targets)
    first=int(panel.dates.searchsorted(pd.Timestamp(START)))
    routes={};selection_rows=[];candidate_metrics=[];candidate_frames={};final_frames={};case_paths={}
    cases={'base':WearConfig(),'severe':WearConfig(spread=.002,impact_eta=.5),
           'extreme':WearConfig(spread=.005,impact_eta=1),'capacity_10m':WearConfig(initial_cash=10_000_000)}
    for case,wear in cases.items():
        frames={};rewards=np.zeros((len(panel.dates),len(targets)));paths={}
        print(f'{case}: {len(targets)} expert shared accounts',flush=True)
        for e,(name,path) in enumerate(targets.items()):
            f,orders=run(panel,path,wear);frames[f'expert_{name}']=f;rewards[first:,e]=f['return'].to_numpy()
            if case=='base':f.to_csv(OUT/f'expert_{name}.csv',index=False)
        # Retained V1 replay must be bitwise-equivalent within CSV precision.
        retained=pd.read_csv(V1/f'soft_cap_audit/ledger_{case}_core_vol16.csv')
        np.testing.assert_allclose(frames['expert_v1'].equity,retained.equity,rtol=1e-12,atol=1e-7)
        for name,cfg in mixes.items():
            paths[name],_=mix_targets(panel,targets,rewards,cfg)
            frames[name]=run(panel,paths[name],wear)[0]
        for name,f in frames.items():
            candidate_frames[case,name]=f
            for start in (START,'2018-01-01'):
                candidate_metrics.append({'case':case,'name':name,'start':start,**portfolio_metrics(f[f.date.ge(start)])})
        if case=='base':
            for year in range(2018,2027):
                routes[year],rows=return_choice(frames,year);selection_rows.extend(rows)
            (OUT/'routes_frozen.json').write_text(json.dumps(routes,indent=2)+'\n')
            pd.DataFrame(selection_rows).to_csv(OUT/'annual_selection_scores.csv',index=False)
        selected=v1.copy();schedule=panel.dates.to_series(index=np.arange(len(panel.dates))).shift(-1).fillna(panel.dates[-1]).dt.year.to_numpy()
        for year,key in routes.items():
            mask=schedule==year;selected[mask]=(targets[key[7:]] if key.startswith('expert_') else paths[key])[mask]
        f,orders=run(panel,selected,wear);orders.to_csv(OUT/f'orders_{case}_annual_v2.csv',index=False)
        final_frames[case,'annual_v2']=f;final_frames[case,'v1']=frames['expert_v1']
        equal_path=base_targets(panel)['equal_blend']
        final_frames[case,'equal_weekly']=run(panel,equal_path,wear,True)[0]
        case_paths[case]=paths
        print(f'{case}: done',flush=True)
    diagnostic=pd.DataFrame(candidate_metrics);diagnostic.to_csv(OUT/'candidate_diagnostic.csv',index=False)
    feasible=[]
    for name in candidate_frames.keys():
        case,key=name
        if case!='base':continue
        check=diagnostic[diagnostic.name.eq(key)]
        if (check.max_drawdown.abs()<=.30+1e-9).all():feasible.append(key)
    # Research choice is explicitly based on observed history and is not annual V2.
    fixed=max(feasible,key=lambda n:portfolio_metrics(candidate_frames['base',n])['cagr']) if feasible else 'expert_v1'
    (OUT/'fixed_candidate_choice.json').write_text(json.dumps({'candidate':fixed,'hindsight_research_selection':True,'required_full_and_2018_all_cases_max_drawdown':.30},indent=2)+'\n')
    metrics=[]
    for case,wear in cases.items():
        path=targets[fixed[7:]] if fixed.startswith('expert_') else case_paths[case][fixed]
        f,orders=run(panel,path,wear);final_frames[case,'fixed_v2_candidate']=f
        orders.to_csv(OUT/f'orders_{case}_fixed_v2_candidate.csv',index=False)
        for name in ('annual_v2','fixed_v2_candidate','v1','equal_weekly'):
            frame=final_frames[case,name];frame.to_csv(OUT/f'ledger_{case}_{name}.csv',index=False)
            for start in (START,'2018-01-01','2024-01-01'):
                block=frame[frame.date.ge(start)];m=portfolio_metrics(block)
                ref=portfolio_metrics(final_frames[case,'v1'][final_frames[case,'v1'].date.ge(start)])
                positions=np.stack([block[[f'position_{list(panel.coverage)[a]["code"]}' for a in g]].sum(axis=1) for g in GROUPS],axis=1)
                metrics.append({'case':case,'name':name,'start':start,**m,'cagr_gain_vs_v1':m['cagr']-ref['cagr'],
                                'drawdown_under30':abs(m['max_drawdown'])<=.30+1e-9,'max_actual_basket':float(positions.max())})
    result=pd.DataFrame(metrics);result.to_csv(OUT/'metrics.csv',index=False)
    roll=[]
    for name in ('annual_v2','fixed_v2_candidate','v1','equal_weekly'):roll.extend(rolling_windows(final_frames['base',name],name))
    r=pd.DataFrame(roll);r.to_csv(OUT/'rolling_windows.csv',index=False);r.groupby(['name','years']).cagr.agg(['min','median','max']).to_csv(OUT/'rolling_summary.csv')
    pd.DataFrame([{'name':name,'year':year,**portfolio_metrics(b)} for (case,name),f in final_frames.items() if case=='base' for year,b in f.groupby(f.date.dt.year)]).to_csv(OUT/'yearly_metrics.csv',index=False)
    paired=[]
    for start in (START,'2018-01-01','2024-01-01'):
        for name in ('annual_v2','fixed_v2_candidate'):
            a=final_frames['base',name];a=a[a.date.ge(start)];b=final_frames['base','v1'];b=b[b.date.ge(start)]
            paired.append({'start':start,'strategy':name,**paired_return_bootstrap(a,b,'v1')})
    pd.DataFrame(paired).to_csv(OUT/'paired_bootstrap.csv',index=False)
    # Independent signal/target/feedback/learner/account and route prefixes.
    short=load_panel(end_date='2019-12-31');n=len(short.dates);small=make_targets(short,v1[:n])
    short_frames={};short_rewards=np.zeros((n,len(targets)));full_rewards=np.zeros((len(panel.dates),len(targets)))
    for e,name in enumerate(targets):
        np.testing.assert_allclose(small[name],targets[name][:n],atol=1e-12)
        f,_=run(short,small[name],WearConfig());short_frames[f'expert_{name}']=f
        full=candidate_frames['base',f'expert_{name}']
        pd.testing.assert_frame_equal(full.iloc[:len(f)].reset_index(drop=True),f)
        short_rewards[first:,e]=f['return'];full_rewards[first:,e]=full['return']
    for key,cfg in mixes.items():
        p,_=mix_targets(short,small,short_rewards,cfg);long,_=mix_targets(panel,targets,full_rewards,cfg)
        np.testing.assert_allclose(p,long[:n],atol=1e-12)
        short_frames[key]=run(short,p,WearConfig())[0]
    for year in (2018,2019):
        key,_=return_choice(short_frames,year)
        if key!=routes[year]:raise RuntimeError('Future-dependent V2 selection')
    manifest={'v1_exact_replay_pass':True,'independent_prefix_and_routes_pass':True,'final_ledgers':len(final_frames),
              'candidates':len(targets)+len(mixes),'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('src/quant/research/domestic_return30.py','scripts/research/run_domestic_return30.py','src/quant/research/domestic_portfolio.py','tests/test_domestic_return30.py')},
              'retained_v1_config_sha256':hashlib.sha256((ROOT/'configs/strategies/domestic_low_securities_vol16_v1.json').read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    chart({name:final_frames['base',name] for name in ('annual_v2','fixed_v2_candidate','v1','equal_weekly')},OUT/'equity.svg','return first V2 with thirty percent drawdown research budget')
    cfg={'strategy_id':'domestic_return30_v2','status':'research_candidate','objective_max_drawdown':.30,
         'retained_v1':'domestic_low_securities_vol16_v1','fixed_policy':fixed,
         'fixed_policy_selected_from_observed_history':True,'annual_policy_separate':'routes_frozen.json',
         'target_each_basket_max':.20,'actual_tolerance':.23,'frequency':21,'deadband':.03,
         'target_archive':str((OUT/'expert_targets.npz').relative_to(ROOT)),'report':str((OUT/'report.md').relative_to(ROOT)),
         'live_trading_enabled':False}
    (ROOT/'configs/strategies/domestic_return30_v2.json').write_text(json.dumps(cfg,indent=2)+'\n')
    lines=['# V2：回撤30%下收益优先','','用户确认保留V1，V2独立输出，不修改V1参数与报告。沿用现有资产/特征、六篮子目标0—20%、实际漂移23%修复、无杠杆/做空、共享现金和真实交易磨损。历史已观察，非盲测。',
           f'冻结{len(targets)}专家、{len(targets)+len(mixes)}候选；宽基300单持/300和500等分/63日动量择一；五种核心及63/120/250日排名前五篮子；18/22%波动预算或无缩放；share/context/multiwindow融合。无新特征、新模型训练。',
           'annual_v2：2014—2017初始化V1，2018起仅用前三完整年回撤<=30%的候选最大化净CAGR，失败回V1；基础年度路线在压力前冻结。训练约束不保证外层回撤<=30%。',
           f'fixed_v2_candidate为{fixed}：显式根据已观察的全历史选择，只在完整期及2018起的四种成本/容量情景均回撤<=30%的候选中按基础完整期CAGR排名。不能把它冒充年度过去数据选型收益或盲测。','','|成本|起点|方案|净年化|最大回撤|相比V1收益增加|回撤<=30%|订单数|','|---|---|---|---:|---:|---:|---|---:|']
    for row in result.itertuples():lines.append(f'|{row.case}|{row.start}|{row.name}|{row.cagr:.2%}|{row.max_drawdown:.2%}|{row.cagr_gain_vs_v1:.2%}|{row.drawdown_under30}|{row.orders}|')
    lines+=['','![净值](equity.svg)','','20项专项测试预期通过；执行后以测试记录为准。V1各成本容量账户精确重放通过；独立截断目标/反馈/学习器/账户/年度路线前缀通过。价格带/冲击和分红税储备仍为日线近似。重采样未校正多轮研究与多重比较；历史最高收益不是全局或未来最高，30%是历史评估/训练约束，不是实盘保证。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(result[(result.case=='base')&(result.start==START)][['name','cagr','max_drawdown','cagr_gain_vs_v1','drawdown_under30']].to_string(index=False),flush=True)
    print('Fixed research candidate:',fixed,flush=True)


if __name__=='__main__':main()
