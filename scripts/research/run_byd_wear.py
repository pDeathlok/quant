#!/usr/bin/env python3
"""Extend the frozen BYD experiment to 2012+, with decomposed trading wear."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant.research.byd_allocation import AllocationConfig, expert_groups, past_pruned_mask
from quant.research.byd_mwu import ROOT, metrics
from quant.research.byd_wear import (
    COST_COLUMNS, START, WearConfig, commission_rate, impact_rate, long_allocation_path,
    past_score, same_orders_gross, stamp_rate, transfer_rate, wear_expert_returns,
    wear_metrics, wear_simulate,
)
from run_byd_allocation import candidate_bank, markdown_table, write_json

OUT = ROOT / "reports/research/byd_wear_20261013"
PREVIOUS = ROOT / "reports/research/byd_allocation_20261013"
YEARS = tuple(range(2016, 2027))


def scenarios() -> dict[str, WearConfig]:
    return {
        "gross_adaptive": WearConfig(commission=0, minimum_commission=0, spread=0, impact_eta=0,
                                     historical_transfer=False, fixed_transfer=0, charge_stamp=False),
        "previous_flat": WearConfig(historical_transfer=False, impact_eta=0),
        "base": WearConfig(),
        "stress": WearConfig(spread=.001, impact_eta=.3),
        "severe": WearConfig(spread=.002, impact_eta=.5),
        "extreme": WearConfig(spread=.005, impact_eta=1),
        "historical_broker": WearConfig(historical_commission=True),
    }


def chart(ledgers: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    colors = ("#0f766e", "#2563eb", "#d97706", "#9333ea", "#64748b")
    curves = {name: np.r_[1., np.cumprod(1+frame["return"].to_numpy())] for name, frame in ledgers.items()}
    dates = next(iter(ledgers.values())).date
    low, high = min(.8, min(v.min() for v in curves.values())*.95), max(v.max() for v in curves.values())*1.05
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="570" viewBox="0 0 1140 570">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#334155}</style>',
             f'<text x="75" y="32" style="font-size:21px;font-weight:bold">BYD | {title} | net equity</text>']
    for value in np.linspace(low,high,6):
        y = 60+(high-value)/(high-low)*375
        parts += [f'<path d="M75 {y:.1f} H1110" stroke="#e2e8f0"/>',f'<text x="12" y="{y+5:.1f}">{value:.2f}x</text>']
    for j,(name,values) in enumerate(curves.items()):
        x = np.linspace(75,1110,len(values)); y = 60+(high-values)/(high-low)*375
        points = " ".join(f"{a:.1f},{b:.1f}" for a,b in zip(x,y))
        color = colors[j % len(colors)]
        parts += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{3 if j==0 else 1.8}"/>',
                  f'<text x="{75+(j%3)*355}" y="{515+(j//3)*25}" style="fill:{color}">{name}</text>']
    for ratio in (0,.25,.5,.75,1):
        date = dates.iloc[int(ratio*(len(dates)-1))]
        parts.append(f'<text x="{75+ratio*1035-20:.1f}" y="470">{date:%Y-%m}</text>')
    parts.append('</svg>');path.write_text("\n".join(parts),encoding="utf-8")


def rolling_windows(ledger: pd.DataFrame, name: str) -> list[dict]:
    dates = pd.DatetimeIndex(ledger.date)
    cumulative = np.r_[1., np.cumprod(1+ledger["return"].to_numpy())]
    monthly = pd.Series(np.arange(len(dates)), index=dates).groupby(dates.to_period("M")).last().to_numpy()
    result = []
    for horizon in (3,5):
        for end in monthly:
            cutoff = dates[end]-pd.DateOffset(years=horizon)
            if cutoff < pd.Timestamp(ledger.signal_date.iloc[0]):
                continue
            begin = int(dates.searchsorted(cutoff, side="right"))
            value = cumulative[end+1]/cumulative[begin]-1
            result.append({"name":name,"years":horizon,"end":str(dates[end].date()),
                           "total_return":float(value),"cagr":float((1+value)**(1/horizon)-1)})
    return result


def terminal_estimate(market: pd.DataFrame, ledger: pd.DataFrame, wear: WearConfig) -> dict:
    row,bar = ledger.iloc[-1],market.iloc[-1]
    sigma = float((market.close/market.pre_close-1).rolling(20).std().iloc[-1])
    q = int(row.shares);value = q*float(bar.close)
    date_int = int(bar.date.strftime("%Y%m%d"))
    impact = impact_rate(q, float(bar.volume_shares), sigma, wear)
    fees = max(value*commission_rate(date_int,wear),wear.minimum_commission) if q else 0.
    estimated = fees+value*(transfer_rate(date_int,wear)+stamp_rate(date_int,wear)+wear.spread+impact)
    return {"exit_cost_estimate":estimated,"exit_cost_fraction":estimated/row.equity,
            "exit_within_one_day_volume_cap":bool(q<=bar.volume_shares*wear.participation),
            "inventory_shares":q,"not_executed":True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    bank,families,_=candidate_bank()
    old=json.loads((PREVIOUS/"protocol.json").read_text())
    if old["candidates"]!={key:cfg.to_dict() for key,cfg in bank.items()}:
        raise RuntimeError("Unexpected change to the previously frozen candidate bank")
    cases=scenarios()
    write_json(out/"protocol.json",{
        "start":START,"outer_years":YEARS,"inner_years":"three complete years preceding each outer year, earliest 2013",
        "candidates_unchanged":True,"candidate_count":len(bank),"scenarios":{name:asdict(wear) for name,wear in cases.items()},
        "capitals":[1_000_000,10_000_000,100_000_000],"annual_routes_frozen_after_base":True,
        "blind_test":False,"stopping":"complete fixed bank and wear/capacity scenarios; no result-driven extensions",
    })
    market=pd.read_parquet(ROOT/"reports/research/byd_mwu_20261012/market_snapshot.parquet")
    targets=pd.read_parquet(PREVIOUS/"expert_targets.parquet").drop(columns="date")
    features=pd.read_parquet(PREVIOUS/"features.parquet").drop(columns="date")
    if not (pd.read_parquet(PREVIOUS/"expert_targets.parquet").date.equals(market.date)):
        raise RuntimeError("Snapshot/expert dates differ")
    groups=expert_groups(list(targets.columns));groups["pruned"]=tuple(targets.columns)
    first=int(np.searchsorted(market.date.to_numpy(),np.datetime64(START)))
    if first!=270 or not targets.iloc[:269].eq(0).all().all():
        raise RuntimeError("Unexpected earliest common warmup boundary")
    print(f"Long replay {START} to {market.date.iloc[-1].date()}, {len(market)-first} trading quotations",flush=True)
    base=cases["base"]
    print("Recalculating all 26 net expert accounts from 2012",flush=True)
    base_rewards=wear_expert_returns(market,targets,base)
    base_rewards.assign(date=market.date).to_parquet(out/"base_expert_returns.parquet",index=False)
    prune,pruning=past_pruned_mask(market,targets,base_rewards)
    write_json(out/"pruning_audit.json",pruning)
    cache={}

    def path_for(cfg:AllocationConfig,rewards:pd.DataFrame,mask:np.ndarray,store:dict)->tuple[np.ndarray,pd.DataFrame]:
        fields={k:v for k,v in cfg.to_dict().items() if k not in ("deadband","max_step","cost_gamma","risk","vol_target")}
        key=json.dumps(fields,sort_keys=True)
        if key not in store:
            names=groups[cfg.group]
            request,weights=long_allocation_path(market,targets.loc[:,names],rewards.loc[:,names],features,cfg,
                                                 prune=mask if cfg.group=="pruned" else None)
            store[key]=request,pd.DataFrame(weights,columns=names)
        return store[key]

    score_lookup={};inner_rows=[]
    for number,(key,cfg) in enumerate(bank.items(),1):
        request,_=path_for(cfg,base_rewards,prune,cache)
        ledger=wear_simulate(market,request,[cfg]*len(market),base)
        for year in YEARS:
            score=past_score(ledger,year);score_lookup[year,key]=score
            inner_rows.append({"year":year,"key":key,"score":score})
        if number%25==0:print(f"Base wear: evaluated {number}/{len(bank)} candidates",flush=True)
    pd.DataFrame(inner_rows).to_csv(out/"inner_scores.csv",index=False)
    routes={family:{year:max(keys,key=lambda key:score_lookup[year,key]) for year in YEARS}
            for family,keys in families.items()}
    standard=sorted({key for family,keys in families.items() if family!="grid_search" for key in keys})
    routes["selected_pipeline"]={year:max(standard,key=lambda key:score_lookup[year,key]) for year in YEARS}
    write_json(out/"routes_frozen.json",routes)
    pd.DataFrame([{"family":name,"year":year,"cutoff":f"{year-1}-12-31","inner_score":score_lookup[year,key],**bank[key].to_dict()}
                  for name,route in routes.items() for year,key in route.items()]).to_csv(out/"annual_selections.csv",index=False)
    print("2016+ annual routes frozen from past base-wear scores",flush=True)
    scheduled_year=market.date.shift(-1).fillna(market.date.iloc[-1]).dt.year.to_numpy()
    baseline=AllocationConfig(method="equal",deadband=.03,max_step=.1)
    fixed_policies={"original_range_matched":targets.original_range.to_numpy(),
                    "constant_one_third":np.full(len(market),1/3),"constant_half":np.full(len(market),.5),
                    "buy_hold":np.ones(len(market))}
    old_config=AllocationConfig(group="legacy",eta=.3)
    prominent=("grid_search","multiwindow","selected_pipeline","original_range_matched","constant_one_third","buy_hold")
    all_ledgers={};summaries=[];yearly=[];same_order=[];terminals=[];checks=[]

    def route_ledger(name:str,wear:WearConfig,rewards:pd.DataFrame,mask:np.ndarray,store:dict)->pd.DataFrame:
        if name in fixed_policies:
            cfg=replace(baseline,deadband=0,max_step=1) if name=="buy_hold" else baseline
            return wear_simulate(market,fixed_policies[name],[cfg]*len(market),wear)
        if name=="previous_frozen_MWU":
            request,weights=path_for(old_config,rewards,mask,store)
            return wear_simulate(market,request,[old_config]*len(market),wear,weights)
        initial=AllocationConfig()
        request,_=path_for(initial,rewards,mask,store);request=request.copy()
        configs=[initial]*len(market)
        for year,key in routes[name].items():
            selected=scheduled_year==year;cfg=bank[key]
            request[selected]=path_for(cfg,rewards,mask,store)[0][selected]
            for i in np.flatnonzero(selected):configs[i]=cfg
        return wear_simulate(market,request,configs,wear)

    def save_ledger(case:str,name:str,ledger:pd.DataFrame,wear:WearConfig)->None:
        ledger.to_csv(out/f"ledger_{case}_{name}.csv",index=False)
        if not np.isfinite(ledger.select_dtypes(include=np.number).to_numpy()).all():
            raise RuntimeError(f"Nonfinite financial result {case}/{name}")
        if not np.allclose(ledger.cost,ledger[list(COST_COLUMNS)].sum(axis=1),atol=1e-8):
            raise RuntimeError("Cost components do not reconcile")
        checks.append({"scenario":case,"name":name,"cash_nonnegative":bool(ledger.cash.min()>=-1e-6),
                       "shares_nonnegative":bool(ledger.shares.min()>=0),"cost_components_reconciled":True})
        for period,start in (("full_2012plus",START),("outer_2016plus","2016-01-01"),("2020plus","2020-01-01"),("2024plus","2024-01-01")):
            summaries.append({"scenario":case,"name":name,"period":period,"capital":wear.initial_cash,
                              **wear_metrics(ledger[ledger.date.ge(start)])})
        for year,frame in ledger.groupby(ledger.date.dt.year):
            yearly.append({"scenario":case,"name":name,"year":int(year),"capital":wear.initial_cash,**wear_metrics(frame)})
        terminals.append({"scenario":case,"name":name,"capital":wear.initial_cash,**terminal_estimate(market,ledger,wear)})

    main_names=list(routes)+list(fixed_policies)+["previous_frozen_MWU"]
    for case,wear in cases.items():
        print(f"Cost scenario {case}: rebuilding expert feedback and actual accounts",flush=True)
        rewards=base_rewards if case=="base" else wear_expert_returns(market,targets,wear)
        mask=prune if case=="base" else past_pruned_mask(market,targets,rewards)[0]
        store=cache if case=="base" else {}
        case_ledgers={}
        for name in main_names:
            ledger=route_ledger(name,wear,rewards,mask,store)
            save_ledger(case,name,ledger,wear);case_ledgers[name]=ledger
            if name in prominent:
                paired=same_orders_gross(market,ledger,wear)
                paired.to_csv(out/f"same_orders_gross_{case}_{name}.csv",index=False)
                net_return=metrics(ledger)["total_return"];gross_return=metrics(paired)["total_return"]
                same_order.append({"scenario":case,"name":name,"capital":wear.initial_cash,
                                   "same_orders_gross_return":gross_return,"net_return":net_return,
                                   "drag_return_points":gross_return-net_return,"trading_cost":float(ledger.cost.sum())})
        all_ledgers[case]=case_ledgers
    capacities=[]
    for capital in (10_000_000.,100_000_000.):
        wear=replace(base,initial_cash=capital)
        print(f"Capacity stress: capital={capital:,.0f}; annual parameters stay frozen",flush=True)
        rewards=wear_expert_returns(market,targets,wear)
        mask=past_pruned_mask(market,targets,rewards)[0];store={}
        for name in prominent:
            ledger=route_ledger(name,wear,rewards,mask,store)
            case=f"capital_{int(capital)}"
            save_ledger(case,name,ledger,wear)
            capacities.append({"name":name,"capital":capital,**wear_metrics(ledger)})
    capacities += [{"name":name,"capital":base.initial_cash,**wear_metrics(all_ledgers["base"][name])} for name in prominent]
    pd.DataFrame(capacities).to_csv(out/"capacity_metrics.csv",index=False)
    # Policy/ledger prefix audit independently replays a previously ended history.
    prefix=int(market.date.le("2023-12-31").sum())
    cfg=AllocationConfig(method="multiwindow",group="diverse")
    names=groups[cfg.group]
    full_request,_=long_allocation_path(market,targets.loc[:,names],base_rewards.loc[:,names],features,cfg)
    prefix_request,_=long_allocation_path(market.iloc[:prefix],targets.loc[:prefix-1,names],base_rewards.loc[:prefix-1,names],features.iloc[:prefix],cfg)
    np.testing.assert_allclose(full_request[:prefix],prefix_request,atol=1e-12)
    a=wear_simulate(market,full_request,[cfg]*len(market),base)
    b=wear_simulate(market.iloc[:prefix],prefix_request,[cfg]*prefix,base)
    pd.testing.assert_frame_equal(a.iloc[:len(b)],b)
    summary=pd.DataFrame(summaries);summary.to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(yearly).to_csv(out/"yearly_metrics.csv",index=False)
    pd.DataFrame(same_order).to_csv(out/"same_orders_cost_attribution.csv",index=False)
    pd.DataFrame(terminals).to_csv(out/"terminal_exit_estimates.csv",index=False)
    write_json(out/"ledger_checks.json",checks)
    rolling=[]
    for name in prominent:rolling+=rolling_windows(all_ledgers["base"][name],name)
    rolling_frame=pd.DataFrame(rolling);rolling_frame.to_csv(out/"rolling_calendar_windows.csv",index=False)
    rolling_summary=rolling_frame.groupby(["name","years"]).cagr.agg(["count","min","median","max"])
    rolling_summary["positive_fraction"]=rolling_frame.groupby(["name","years"]).cagr.apply(lambda values:float((values>0).mean()))
    rolling_summary.to_csv(out/"rolling_summary.csv")
    chart({name:all_ledgers["base"][name] for name in ("grid_search","multiwindow","selected_pipeline","constant_one_third","original_range_matched")},out/"long_equity.svg","2012-2026, decomposed base wear")
    chart({case:all_ledgers[case]["grid_search"] for case in ("gross_adaptive","base","stress","severe","extreme")},out/"wear_sensitivity.svg","grid search, frozen annual parameters, wear scenarios")
    # Compare all base methods with a risk-near simple baseline; descriptive only.
    rng=np.random.default_rng(20261013);bootstrap=[]
    reference=all_ledgers["base"]["constant_one_third"]["return"].to_numpy()
    for name in ("grid_search","multiwindow","selected_pipeline"):
        delta=all_ledgers["base"][name]["return"].to_numpy()-reference
        starts=rng.integers(0,len(delta),size=(2000,int(np.ceil(len(delta)/20))))
        indices=((starts[:,:,None]+np.arange(20))%len(delta)).reshape(2000,-1)[:,:len(delta)]
        low,high=np.quantile(delta[indices].mean(axis=1)*252,[.025,.975])
        bootstrap.append({"name":name,"annual_mean_difference":float(delta.mean()*252),"low95":float(low),"high95":float(high)})
    pd.DataFrame(bootstrap).to_csv(out/"paired_bootstrap_full.csv",index=False)
    manifest={"data_start":str(market.date.iloc[0].date()),"trade_start":START,"data_end":str(market.date.iloc[-1].date()),
              "raw_rows":len(market),"trade_quotations":len(market)-first,"candidates":len(bank),"experts":len(targets.columns),
              "outer_start":"2016-01-01","routes_use_future_scores":False,"blind_test":False,"production_changed":False,
              "scenario_count":len(cases),"ledger_count":len(checks),"scenarios":{name:asdict(wear) for name,wear in cases.items()},
              "verification":{"policy_and_account_prefix_equal":True,"same_orders_cost_identity":True,"cost_components_reconciled":True},
              "market_sha256":hashlib.sha256((ROOT/"reports/research/byd_mwu_20261012/market_snapshot.parquet").read_bytes()).hexdigest(),
              "sources":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                  ROOT/"src/quant/research/byd_wear.py",Path(__file__).resolve(),ROOT/"tests/test_byd_wear.py",ROOT/"docs/plans/2026-10-13-byd-long-history-wear.md")}}
    write_json(out/"manifest.json",manifest)
    headline=("grid_search","multiwindow","selected_pipeline","original_range_matched","previous_frozen_MWU","constant_one_third","constant_half","buy_hold")
    full=summary[summary.scenario.eq("base") & summary.period.eq("full_2012plus") & summary.name.isin(headline)]
    outer=summary[summary.scenario.eq("base") & summary.period.eq("outer_2016plus") & summary.name.isin(headline)]
    wear_table=summary[summary.period.eq("full_2012plus") & summary.name.eq("grid_search") & summary.scenario.isin(cases)].copy()
    wear_table["name"]=wear_table.scenario
    attribution=pd.DataFrame(same_order);attribution=attribution[attribution.scenario.eq("base")]
    attribution_lines="\n".join(f"- {row.name}：同订单毛收益{row.same_orders_gross_return:.2%}，净收益{row.net_return:.2%}，磨损拖累{row.drag_return_points:.2%}（百分点），累计交易成本{row.trading_cost:,.0f}元。" for row in attribution.itertuples())
    cost_lines=[]
    for name in ("grid_search","multiwindow","original_range_matched","constant_one_third"):
        row=full[full.name.eq(name)].iloc[0]
        cost_lines.append(f"- {name}：佣金{row.commission_cost:,.0f}，过户费{row.transfer_cost:,.0f}，印花税{row.stamp_cost:,.0f}，基础价差/滑点{row.spread_cost:,.0f}，动态冲击{row.impact_cost:,.0f}元；年化平均权益磨损{row.annualized_cost_fraction:.2%}，换手总量{row.turnover:.2f}倍。")
    capacity_lines="\n".join(f"- {row.name} / 本金{row.capital/1e4:,.0f}万元：净收益{row.total_return:.2%}，年化{row.cagr:.2%}，回撤{row.max_drawdown:.2%}，容量受限{row.capacity_limited_days}天，请求股数未完成比例{row.unfilled_share_fraction:.2%}。" for row in pd.DataFrame(capacities).sort_values(["name","capital"]).itertuples())
    bootstrap_lines="\n".join(f"- {row['name']}相对固定1/3：年化日均差{row['annual_mean_difference']:.2%}，描述性95%区间[{row['low95']:.2%}, {row['high95']:.2%}]。" for row in bootstrap)
    report=f"""# BYD 全历史与交易磨损回测

原始日线2011-06-30至{manifest['data_end']}，共{len(market)}条；预热269条后，2012-08-09收盘形成首个信号，2012-08-10起交易，共{len(market)-first}个有报价交易日，日历跨度约{full.calendar_years.iloc[0]:.2f}年。A股上市前不补造历史，也不把H股价格拼接进来。
26个专家及150个候选保持不变，学习起点从2015提前到2012；年度内层用该年前3个完整年度，外层从2016开始。2012-2015是预先固定的初始化流程，完整净值包含这段，不能全部称为独立样本外检验。

## 全段：2012-08-10至数据截止，基础磨损

{markdown_table(full)}

![完整历史净值](long_equity.svg)

年化收益按实际日历跨度；metrics.csv另保留cagr_252以便核对，Sharpe/波动使用252报价日惯例。与上一轮数字的变化同时包含评估起点、学习起点、年度选型起点和费用口径变化，不应全部归因于成本。完整长历史已被研究观察，仍不是新盲测。

## 逐年选型外层：2016起

{markdown_table(outer)}

2016起每年的参数只由之前3个完整年度选出。2012-2015初始化固定为diverse MWU，各方法的2016起真实账户连续重放，不拼接候选账户的收益。年度路由保存后，压力情景和资金规模不再次选型。模型沿用前轮滚动成熟标签预测缓存，1/21/63日模型在2014首次有足够样本时生效；5日模型约2013起，此前目标为0（现金），不存在上市时就有训练好的模型假设。

## 交易磨损定义与情景

基础佣金万3、最低5元，假定含经手费/监管费，避免重复扣费。深市过户费2015-08-01前按成交金额0.0255‰、此后0.02‰、2022-04-29起0.01‰双向；卖方印花税2023-08-28前1‰、此后0.5‰。个人佣金没有历史交割单，固定万3只是一项假设。
基础价差/滑点为单边5bps；动态冲击为eta乘过去20日标准差乘sqrt(订单股数/前一报价日成交股数)。base eta=.1；stress 10bps/eta.3；severe20bps/eta.5；extreme50bps/eta1。成交模拟只用前日量与已知波动，价格在历史涨跌停范围内，开盘涨停买单/跌停卖单不成交。
previous_flat是上一轮费用口径：统一过户费万0.2、固定5bps、无冲击，但本次学习和回测起点仍为2012。historical_broker是额外佣金压力假设：2015前千1、2015-2019万5、2020起万3，不是核验后的真实个人费率。

同一网格参数选型流程的全段成本敏感性：

{markdown_table(wear_table)}

![磨损敏感性](wear_sensitivity.svg)

每个成本情景重算专家净收益、在线权重、现金股数与成本控制，但年度配置由base冻结。gross_adaptive为零交易成本下的自适应路径，可能改变交易时点；分红税储备仍与净收益一致。因此它与base的差异不等于直接扣除的费用。

## 相同订单的毛净归因

保持净账本逐日成交股数不变，毛账本按开盘价成交、免交易费；节省现金不再投资。两个账本股份路径和公司行为一致，期末权益差严格等于累计交易成本。

{attribution_lines}

成本拆分（人民币）：

{chr(10).join(cost_lines)}

年化平均权益磨损是每日成本/开盘权益的均值乘252，反映换手负担；累计金额因账户净值路径而异，不能直接跨策略比较为相同本金百分比。整手与最低佣金、未成交机会损失不另用虚构收益扣除，而通过实际股数和订单记录反映。

## 资金规模与容量

{capacity_lines}

所有规模的容量都限制在前一报价日成交量1%，不是开盘盘口的真实可成交深度。请求订单、成交股数、资金不足、容量与涨跌停阻断逐日记录。期末为持仓市值，另在terminal_exit_estimates.csv估算清仓费用与一日容量可行性，不把数据截止后的清仓当作成交。

## 分年与滚动窗口

yearly_metrics.csv包含各情景各年度；rolling_calendar_windows.csv按月末计算日历3年和5年滚动窗口，rolling_summary.csv列最差、中位、最好年化与正收益占比。窗口重叠，不能当作独立样本。2020+与2024+分段仍保留在metrics.csv，避免完整期收益掩盖近期退化。

{bootstrap_lines}

20日分块bootstrap2000次，只描述既定历史差异，不校正历史观察、宏观风格及多重搜索，不构成未来收益保证。

## 验证与限制

完整学习路径/账本与2023截止前缀重放一致，毛净同股数归因和成本拆账逐日核对通过；{len(checks)}个真实成交账本现金与股数非负。缺失报价日不伪造成交，窗口按个股有报价日形成；完整期年化改为实际日历跨度。
历史日线有大量推算的10%涨跌停，分红到账日和20%现金/红股税储备沿用简化假设，不是FIFO税务结算。价差与冲击没有历史盘口标定，本文eta是敏感性假设；平方根模型不是Almgren-Chriss临时冲击的必然形式。没有盘口排队/真实分钟TWAP验证，1%日量容量也不认证开盘成交。这些研究限制在长历史尤其重要。
不新增测试结果驱动候选，不修改生产策略。完整长历史、2016起逐年评估与压力测试完成后停止，不宣称全局最优或未来收益收敛。

运行：`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/research/run_byd_wear.py`。
测试：`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -p test_byd_wear.py -v`。

## 来源

- [深交所：BYD A股2011-06-30上市](https://www.szse.cn/disclosure/notice/company/t20110628_508636.html)
- [深交所：2015年过户费调整](https://www.szse.cn/aboutus/trends/news/t20150701_518630.html)
- [政府公布2022年过户费调整](https://jrj.wuhan.gov.cn/ynzx_57/xwzx/202204/t20220429_1964160.shtml)
- [税务总局：2023年印花税减半](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html)
- [平方根冲击研究](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.1.021006)
"""
    (out/"report.md").write_text(report,encoding="utf-8")
    print(markdown_table(full),flush=True)
    print("Wear sensitivity:",flush=True);print(markdown_table(wear_table),flush=True)
    print(f"Saved {out/'report.md'}",flush=True)


if __name__=="__main__":
    main()
