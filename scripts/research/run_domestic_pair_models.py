#!/usr/bin/env python3
"""Declared second uncapped round: refit relative BYD/gold labels on existing features."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from quant.research.byd_mwu import ROOT,_RidgeExpert
from quant.research.byd_wear import WearConfig
from quant.research.byd_growth import paired_return_bootstrap
from quant.research.domestic_portfolio import START,load_panel,matured_labels,simulate,portfolio_metrics
from quant.research.domestic_return30 import return_choice
from run_domestic_portfolio import risk_cap
from run_byd_wear import chart

OUT=ROOT/'reports/research/domestic_pair_model_v3_20261013'
SPECS=(('ridge_relative21',21,False),('ridge_relative63',63,False),('ridge_risk63',63,True),('tree_relative63',63,False))


def fit(panel):
    n=len(panel.dates);x=panel.features[:,[0,3],:].reshape(n,-1);predictions={};audit=[]
    for name,h,risk in SPECS:
        labels=matured_labels(panel,h,risk);y=labels[:,0]-labels[:,3]
        output=np.zeros(n);model=None
        for i in range(600,n):
            if model is None or (i-600)%21==0:
                end=i-h-1;begin=max(252,end-1259);valid=np.isfinite(y[begin:end+1])
                if valid.sum()<252:continue
                model=HistGradientBoostingRegressor(max_iter=60,max_depth=2,max_leaf_nodes=6,min_samples_leaf=50,learning_rate=.05,l2_regularization=2.,early_stopping=False,random_state=20261013) if name.startswith('tree') else _RidgeExpert()
                model.fit(x[begin:end+1][valid],y[begin:end+1][valid])
                audit.append({'model':name,'fit_date':str(panel.dates[i].date()),'label_end':str(panel.dates[end+h+1].date()),'samples':int(valid.sum()),'input_features':x.shape[1]})
            if model is not None:output[i]=float(model.predict(x[i:i+1])[0])
        predictions[name]=output;print(f'{name}: mature relative-label fits complete',flush=True)
    return predictions,audit


def targets(panel,predictions):
    result={};scores=dict(predictions)
    # Horizons remain in holding-return units; long/short ensemble is an engineering score.
    scores['ridge_ensemble']=np.mean([predictions[n] for n in predictions if n.startswith('ridge')],axis=0)
    scores['all_ensemble']=np.mean(list(predictions.values()),axis=0)
    for name,score in scores.items():
        for amp in (.25,.45):
            w=np.zeros_like(panel.data['close']);byd=.5+amp*np.tanh(score/.10)
            w[:,0]=byd*panel.data['eligible'][:,0];w[:,3]=(1-byd)*panel.data['eligible'][:,3]
            for vol in (0.,.25,.35):result[f'{name}_amp{int(amp*100)}_{"raw" if not vol else "vol"+str(int(vol*100))}']=risk_cap(panel,w,vol)
    w=np.zeros_like(panel.data['close']);w[:,0]=.5*panel.data['eligible'][:,0];w[:,3]=.5*panel.data['eligible'][:,3]
    result['anchor']=w;result['cash']=np.zeros_like(w)
    return result


def run(panel,path,wear):return simulate(panel,path,wear,frequency=5,deadband=.03)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'round':2,'previous_results_observed':True,'blind':False,'target_net_cagr':.20,'target_max_drawdown':.30,
              'features':'existing21 BYD + existing21 gold, no external/new definitions',
              'labels':'net next-open holding return BYD minus gold; horizons21/63; risk63 subtracts half adverse excursion in each asset before difference',
              'training':{'ridge_alpha':50,'lookback':1260,'min_samples':252,'refit':21,'tree_iter':60,'tree_depth':2,'tree_min_leaf':50,'early_stopping':False},
              'weight_mapping':'0.5+amplitude*tanh(prediction/.10)','amplitudes':[.25,.45],'vol_caps':[0,.25,.35],
              'single_asset_cap':1.,'leverage':False,'frequency':5,'deadband':.03,
              'initial_policy':'anchor50/50 2014-2017','annual_selection':'past3 completed years drawdown<=30% max CAGR, cash fallback',
              'stopping':'finish fixed second-round set and four cases, no result-driven additional fits'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    panel=load_panel();pred,audit=fit(panel);np.savez_compressed(OUT/'predictions.npz',**pred)
    pd.DataFrame(audit).to_csv(OUT/'training_audit.csv',index=False)
    for a in audit:
        if pd.Timestamp(a['label_end'])>pd.Timestamp(a['fit_date']):raise RuntimeError('Immature label')
    expert=targets(panel,pred);np.savez_compressed(OUT/'expert_targets.npz',**expert)
    protocol['candidates']=list(expert);(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cases={'base':WearConfig(),'severe':WearConfig(spread=.002,impact_eta=.5),'extreme':WearConfig(spread=.005,impact_eta=1),'capacity_10m':WearConfig(initial_cash=10_000_000)}
    routes={};selection=[];diagnostic=[];frames={};final={}
    for case,wear in cases.items():
        bank={}
        for name,path in expert.items():
            f,_=run(panel,path,wear);bank[f'expert_{name}']=f;frames[case,name]=f
            for start in (START,'2018-01-01'):
                m=portfolio_metrics(f[f.date.ge(start)]);diagnostic.append({'case':case,'name':name,'start':start,**m})
        if case=='base':
            for year in range(2018,2027):routes[year],r=return_choice(bank,year);selection.extend(r)
            (OUT/'routes_frozen.json').write_text(json.dumps(routes,indent=2)+'\n');pd.DataFrame(selection).to_csv(OUT/'annual_selection_scores.csv',index=False)
        path=expert['anchor'].copy();schedule=panel.dates.to_series(index=np.arange(len(panel.dates))).shift(-1).fillna(panel.dates[-1]).dt.year.to_numpy()
        for year,key in routes.items():path[schedule==year]=expert[key[7:]][schedule==year]
        final[case,'annual_pair'],orders=run(panel,path,wear);orders.to_csv(OUT/f'orders_{case}_annual_pair.csv',index=False)
        final[case,'anchor']=frames[case,'anchor'];print(f'{case}: completed',flush=True)
    d=pd.DataFrame(diagnostic);d.to_csv(OUT/'candidate_diagnostic.csv',index=False)
    feasible=[name for name in expert if d[d.name.eq(name)].max_drawdown.abs().le(.30+1e-9).all()]
    fixed=max(feasible,key=lambda name:portfolio_metrics(frames['base',name])['cagr']) if feasible else 'cash'
    raw=max(expert,key=lambda name:portfolio_metrics(frames['base',name])['cagr'])
    (OUT/'research_choices.json').write_text(json.dumps({'fixed_risk_feasible':fixed,'raw_best':raw,'hindsight_selection':True},indent=2)+'\n')
    rows=[];paired=[]
    for case in cases:
        final[case,'fixed_pair']=frames[case,fixed];final[case,'raw_pair']=frames[case,raw]
        for name in ('annual_pair','fixed_pair','raw_pair','anchor'):
            f=final[case,name];f.to_csv(OUT/f'ledger_{case}_{name}.csv',index=False)
            for start in (START,'2018-01-01','2024-01-01'):
                m=portfolio_metrics(f[f.date.ge(start)]);rows.append({'case':case,'name':name,'start':start,**m,'goal_met':m['cagr']>.20 and abs(m['max_drawdown'])<=.30})
    for start in (START,'2018-01-01'):
        for name in ('annual_pair','fixed_pair'):
            a=final['base',name];a=a[a.date.ge(start)];b=final['base','anchor'];b=b[b.date.ge(start)]
            paired.append({'start':start,'strategy':name,**paired_return_bootstrap(a,b,'anchor')})
    pd.DataFrame(paired).to_csv(OUT/'paired_bootstrap.csv',index=False)
    result=pd.DataFrame(rows);result.to_csv(OUT/'metrics.csv',index=False)
    short=load_panel(end_date='2019-12-31');n=len(short.dates);small_pred,_=fit(short)
    for name in pred:np.testing.assert_allclose(small_pred[name],pred[name][:n],atol=1e-10,rtol=1e-10)
    small_targets=targets(short,small_pred);small_frames={}
    for name,path in small_targets.items():
        np.testing.assert_allclose(path,expert[name][:n],atol=1e-10)
        f,_=run(short,path,WearConfig());small_frames[f'expert_{name}']=f
        pd.testing.assert_frame_equal(frames['base',name].iloc[:len(f)].reset_index(drop=True),f)
    for year in (2018,2019):
        chosen,_=return_choice(small_frames,year)
        if chosen!=routes[year]:raise RuntimeError('Future-dependent pair selection')
    (OUT/'manifest.json').write_text(json.dumps({'independent_refit_target_account_route_prefix_pass':True,'mature_labels_pass':True,'candidate_count':len(expert),
                                               'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('scripts/research/run_domestic_pair_models.py','src/quant/research/domestic_portfolio.py','src/quant/research/domestic_uncapped.py')}},indent=2)+'\n')
    chart({n:final['base',n] for n in ('annual_pair','fixed_pair','anchor')},OUT/'equity.svg','relative BYD gold models, mature labels and costs')
    lines=['# V3第二轮：BYD/黄金相对收益模型','','在观察第一轮之后独立设计，本轮不是盲测。使用已有21特征分别取BYD/黄金拼接42列，实际重训21/63日相对收益、63日相对下行风险Ridge及63日小树；标签成熟后才进入训练。所有成本容量与无杠杆/无20%限制保持。',
           f'{len(expert)}固定候选；BYD目标0.5+amplitude*tanh(predicted_relative_return/.10)，幅度0.25/0.45，黄金剩余，风险预算0/25/35%。模型不可用时回50/50。',
           f'固定研究可行为{fixed}，原始最高收益为{raw}；事后根据已观察全历史筛选，不是年度选型收益。年度流程2018起只用前三完整年回撤<=30%最大化净CAGR，之前50/50初始化。','','|成本|起点|方案|净年化|最大回撤|20%/30%达标|','|---|---|---|---:|---:|---|']
    for r in result.itertuples():lines.append(f'|{r.case}|{r.start}|{r.name}|{r.cagr:.2%}|{r.max_drawdown:.2%}|{r.goal_met}|')
    lines+=['','![净值](equity.svg)','','独立截断数据重新训练、预测/权重/账户/年度路线前缀审计通过。风险预算不是实盘最大回撤保证；重采样未校正反复研究。若完整期不满足20%/30%，不得以近期或历史最佳风险超限值宣称达标。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(result[(result.case=='base')&(result.start==START)][['name','cagr','max_drawdown','goal_met']].to_string(index=False),flush=True)
    print('research choices',fixed,raw,flush=True)


if __name__=='__main__':main()
