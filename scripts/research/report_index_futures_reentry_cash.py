"""Factorial effects, idle opportunity attribution and whole-grid audit."""
import json
import os
from functools import lru_cache
import numpy as np
import pandas as pd
from evaluate_index_futures_reentry_cash import (
    OUT,PREVIOUS,EQUITY,CONFIGS,NEW_MODES,RULES,CAPITAL,DATE_TYPES,PROFILES,VARIANTS,
    ReentryReplay,liquidated_prefix,first_year,
)
from report_index_futures_multistart import grouped,pct,NAMES
from refine_index_futures_adaptation import entry_candidates

LABELS={}
for mode,(style,cash,rule) in CONFIGS.items():
    LABELS[mode]=( "固定" if style=="fixed" else "滚动")+("+现金" if cash else "")+"/"+dict(none="原入场",cooldown5="等待5日",reset="信号重置")[rule]


def mode_for(style,cash,rule):
    return next(m for m,c in CONFIGS.items() if c==(style,cash,rule))


def path_for(start,variant,mode,fee,margin):
    root=OUT if mode in NEW_MODES else EQUITY if mode in ("actual_equity","equity_score_gate") else PREVIOUS
    return root/start[:4]/"ledgers"/f"{start}_{variant}_{mode}_{fee}_{margin}.parquet"


@lru_cache(maxsize=48)
def ledger_at(start,variant,mode,fee,margin):
    return pd.read_parquet(path_for(start,variant,mode,fee,margin))


def prefix(ledger,window,start,fee,opens):
    if window=="to_latest":return ledger
    if window=="single_year":return first_year(ledger,start,opens,fee)
    return liquidated_prefix(ledger,int(window.split("_")[1]),opens,fee)


def effect_pairs():
    pairs=[]
    for style in ("fixed","rolling"):
        for rule in RULES[1:]:pairs.append(("reentry",mode_for(style,False,"none"),mode_for(style,False,rule)))
        for rule in RULES:pairs.append(("cash",mode_for(style,False,rule),mode_for(style,True,rule)))
    for cash in (False,True):
        for rule in RULES:pairs.append(("rolling",mode_for("fixed",cash,rule),mode_for("rolling",cash,rule)))
    return pairs


def attribution(a,b):
    np.testing.assert_array_equal(a.entry_date,b.entry_date)
    mask=a.gross_lots.gt(0)&b.gross_lots.eq(0)
    source=a.loc[mask,"pnl"]
    missed=float(source.clip(lower=0).sum())
    avoided=float(-source.clip(upper=0).sum())
    flat_fees=float(b.loc[mask,"pnl"].sum())
    other=float((b.pnl-a.pnl)[~mask].sum())
    net=float(b.pnl.sum()-a.pnl.sum())
    np.testing.assert_allclose(net,avoided-missed+flat_fees+other,rtol=0,atol=1e-6)
    return dict(missed_positive_pnl=missed,avoided_negative_pnl=avoided,
        skipped_baseline_net_pnl=missed-avoided,new_flat_fees=flat_fees,other_path_change=other,
        net_pnl_change=net,additional_flat_vs_held_days=int(mask.sum()))


def describe(data,keys):
    base=grouped(data,keys)
    extra=data.groupby(keys).agg(median_flat_ratio=("flat_ratio","median"),median_cash_ratio=("cash_ratio","median"),
        median_longest_flat=("longest_flat","median"),median_trades=("trades","median"),
        resumed_fraction=("cash_resumptions",lambda s:s.gt(0).mean()),median_reentry_blocks=("reentry_blocks","median"),
        never_traded_fraction=("trades",lambda s:s.eq(0).mean()),
        median_reentry_rejected_days=("reentry_rejected_days","median"),
        median_substitution_entries=("substitution_entries","median")).reset_index()
    return base.merge(extra,on=keys,validate="one_to_one")


def performance_table(lines,title,part,key,modes):
    lines.extend([title,"","| "+key+" | "+" | ".join(LABELS.get(m,m) for m in modes)+" |","|---|"+"---:|"*len(modes)])
    for value,g in part.groupby(key):
        g=g.set_index("startup_mode")
        lines.append(f"| {value} | "+" | ".join(f"{pct(g.loc[m,'total_return'])} / {pct(-g.loc[m,'max_drawdown'])}" for m in modes)+" |")
    lines.append("")


def charts(main):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/quant_adaptation_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    labels=["Original entries","Wait5","Signal reset","Cash + original","Cash + wait5","Cash + reset"]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for i,year in enumerate(("2025","2026")):
        for j,style in enumerate(("fixed","rolling")):
            modes=[m for m,c in CONFIGS.items() if c[0]==style]
            part=main[main.start_kind.eq("monthly")&main.start.str.startswith(year)&main.window.eq("to_latest")&main.variant.eq("mean5")&main.fee.eq(.0006)&main.margin.eq(.25)]
            ax=axes[i,j]
            for mode,label in zip(modes,labels):
                g=part[part.startup_mode.eq(mode)].sort_values("start")
                ax.plot(g.start.str[4:6].astype(int),100*g.total_return,label=label,marker="o",ms=3)
            ax.set_title(f"{year} starts | {style} policies |6bps/25% margin")
            ax.set_ylabel("Net return to2026-11-06 (%)");ax.set_xlabel("Startup month");ax.grid(alpha=.2);ax.axhline(0,c="grey",lw=.8)
    fig.suptitle("Reentry and recoverable cash: recent startup sensitivity",fontsize=14)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(OUT/"recent_fixed_vs_rolling.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for i,start in enumerate(("20230801","20260101")):
        for j,style in enumerate(("fixed","rolling")):
            modes=[m for m,c in CONFIGS.items() if c[0]==style]
            ax=axes[i,j]
            for mode,label in zip(modes,labels):
                l=ledger_at(start,"mean5",mode,.0006,.25)
                ax.plot(pd.to_datetime(l.exit_date),l.equity/10000,label=label,lw=1.3)
            ax.set_title(f"Start{start} | {style} | raw5,6bps/25% margin")
            ax.set_ylabel("Equity(CNY10k)");ax.grid(alpha=.2)
            locator=mdates.AutoDateLocator(minticks=3,maxticks=5)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle("Old frozen account and recent account: no topups",fontsize=14)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(OUT/"account_paths.png",dpi=160);plt.close(fig)


def main():
    data=pd.read_csv(OUT/"summary.csv",dtype=DATE_TYPES,low_memory=False)
    old=pd.read_csv(PREVIOUS/"summary.csv",dtype=DATE_TYPES,low_memory=False)
    keys=["start","fee","margin","variant","startup_mode"]
    sort=keys+["window"]
    retained=data[~data.startup_mode.isin(NEW_MODES)][old.columns]
    pd.testing.assert_frame_equal(retained.sort_values(sort).reset_index(drop=True),old.sort_values(sort).reset_index(drop=True),check_dtype=False,atol=1e-8,rtol=1e-10)
    trades=pd.read_csv(OUT/"trades.csv",dtype=DATE_TYPES,low_memory=False)
    decisions=pd.read_csv(OUT/"selection.csv",dtype=DATE_TYPES,low_memory=False)
    audit=pd.read_csv(OUT/"selection_audit.csv",dtype=DATE_TYPES,low_memory=False)
    annual=pd.read_csv(OUT/"annual_marked.csv",dtype=DATE_TYPES,low_memory=False)
    totals=data[data.window.eq("to_latest")]
    assert len(totals)==18000 and totals.groupby("startup_mode").size().eq(750).all()
    recon=totals.merge(trades.groupby(keys).pnl.sum().rename("net"),on=keys,how="left")
    np.testing.assert_allclose(recon.net.fillna(0),recon.total_return*CAPITAL,atol=1e-6,rtol=0)
    assert decisions.cutoff.lt(decisions.date).all() and audit.last_history_end.le(audit.cutoff).all()
    np.testing.assert_array_equal(decisions.equity,decisions.validation_capital)
    activated=decisions[decisions.reason.eq("cash_activation_passed")]
    assert activated.candidate_score.ge(.005-1e-12).all()
    assert activated.historical_quarter_edge.ge(activated.required_edge-1e-8).all()
    replay=ReentryReplay();main=data[data.startup_mode.isin(CONFIGS)].copy()
    main["start_year"]=main.start.str[:4]
    # Enrich old control for diagnostics without altering copied summary data.
    for i,r in main[main.startup_mode.eq("actual_equity")].iterrows():
        l=prefix(ledger_at(r.start,r.variant,r.startup_mode,r.fee,r.margin),r.window,r.start,r.fee,replay.opens)
        flat=l.gross_lots.eq(0);run=(flat!=flat.shift()).cumsum()
        main.loc[i,"flat_ratio"]=flat.mean();main.loc[i,"longest_flat"]=flat.groupby(run).sum().max()
        for c in ("cash_days","cash_resumptions","cash_pauses","reentry_blocks","cash_signal_days","cash_funded_days"):main.loc[i,c]=0
    main["cash_ratio"]=main.cash_days/main.days
    # A blocked top candidate can be replaced by another product. Count this
    # separately from all-candidate rejection, without changing execution.
    by_day={g[0].entry_date:g for g in replay.groups}
    main["reentry_rejected_days"]=0;main["substitution_entries"]=0
    for key,g in main[main.startup_mode.isin([m for m,c in CONFIGS.items() if c[2]!="none"])].groupby(["start","variant","startup_mode","fee","margin"]):
        start,variant,mode,fee,margin=key
        trace=pd.read_parquet(OUT/start[:4]/"traces"/f"{start}_{variant}_{mode}_{fee}_{margin}.parquet")
        rejected=trace.get("reentry_rejected",pd.Series(0,index=trace.index)).fillna(0).gt(0)
        substitutions=np.zeros(len(trace),dtype=int)
        for i,t in trace[rejected&trace.action.eq("enter")].iterrows():
            top=entry_candidates(by_day[t.date],t.policy,t.equity,fee,margin)[0]
            substitutions[i]=int((top[1],top[2])!=(t.selected_product,t.selected_side))
        for i,r in g.iterrows():
            main.loc[i,"reentry_rejected_days"]=int(rejected.iloc[:int(r.days)].sum())
            main.loc[i,"substitution_entries"]=int(substitutions[:int(r.days)].sum())
    main.to_csv(OUT/"factorial_summary.csv",index=False)
    monthly=main[main.start_kind.eq("monthly")]
    dist=describe(monthly,["startup_mode","variant","fee","margin","window"])
    cohorts=describe(monthly,["start_year","startup_mode","variant","fee","margin","window"])
    dist.to_csv(OUT/"start_distributions.csv",index=False);cohorts.to_csv(OUT/"year_cohorts.csv",index=False)
    phases=describe(main[main.annual|main.start_kind.eq("phase")],["start_year","startup_mode","variant","fee","margin","window"])
    phases.to_csv(OUT/"january_phase_sensitivity.csv",index=False)
    paired=[];opportunity=[]
    for effect,base,new in effect_pairs():
        a=main[main.startup_mode.eq(base)];b=main[main.startup_mode.eq(new)]
        pair=b.merge(a,on=["start","fee","margin","variant","window"],suffixes=("","_base"),validate="one_to_one")
        pair["effect"]=effect;pair["baseline_mode"]=base
        pair["return_change"]=pair.total_return-pair.total_return_base
        pair["drawdown_change"]=pair.max_drawdown-pair.max_drawdown_base
        paired.append(pair)
        for (start,variant,fee,margin),p in pair.groupby(["start","variant","fee","margin"]):
            full_a=ledger_at(start,variant,base,fee,margin);full_b=ledger_at(start,variant,new,fee,margin)
            for r in p.itertuples(index=False):
                aa=prefix(full_a,r.window,start,fee,replay.opens);bb=prefix(full_b,r.window,start,fee,replay.opens)
                values=attribution(aa,bb)
                np.testing.assert_allclose(values["net_pnl_change"],r.return_change*CAPITAL,atol=1e-6,rtol=0)
                opportunity.append(dict(effect=effect,baseline_mode=base,startup_mode=new,start=start,start_year=start[:4],
                    start_kind=r.start_kind,variant=variant,fee=fee,margin=margin,window=r.window,**values))
        print("attributed",effect,base,new,flush=True)
    paired=pd.concat(paired,ignore_index=True);paired.to_csv(OUT/"paired_effects.csv",index=False)
    op=pd.DataFrame(opportunity);op.to_csv(OUT/"opportunity_costs.csv",index=False)
    pair_keys=["effect","baseline_mode","startup_mode","start_year","variant","fee","margin","window"]
    effects=paired[paired.start_kind.eq("monthly")].groupby(pair_keys).agg(starts=("start","size"),
        median_return_change=("return_change","median"),return_improved_fraction=("return_change",lambda s:s.gt(1e-10).mean()),
        drawdown_improved_fraction=("drawdown_change",lambda s:s.gt(1e-10).mean()),drawdown_worsened_fraction=("drawdown_change",lambda s:s.lt(-1e-10).mean())).reset_index()
    opp=op[op.start_kind.eq("monthly")].groupby(pair_keys).agg(median_missed_positive=("missed_positive_pnl","median"),
        median_avoided_negative=("avoided_negative_pnl","median"),median_skipped_net=("skipped_baseline_net_pnl","median"),
        median_other_path_change=("other_path_change","median"),median_additional_flat_days=("additional_flat_vs_held_days","median")).reset_index()
    effects=effects.merge(opp,on=pair_keys,validate="one_to_one");effects.to_csv(OUT/"effects_by_year.csv",index=False)
    worst=[]
    for _,g in main.groupby(["startup_mode","fee","margin","window"]):
        for metric in ("total_return","max_drawdown"):
            worst.append(dict(g.loc[g[metric].idxmin()],criterion=metric))
    pd.DataFrame(worst).to_csv(OUT/"worst_all_starts.csv",index=False)
    forced=[]
    for r in totals[totals.startup_mode.isin(NEW_MODES)&totals.forced.gt(0)].itertuples(index=False):
        l=ledger_at(r.start,r.variant,r.startup_mode,r.fee,r.margin)
        for i in l.index[l.forced_exit]:
            code,side=l.positions.iloc[i-1].split(":")
            multiplier=300 if code[:2] in ("IF","IH") else 200
            required_margin=replay.opens[l.entry_date.iloc[i],code]*multiplier*r.margin
            before=CAPITAL if i==0 else l.equity.iloc[i-1]
            forced.append(dict(start=r.start,variant=r.variant,mode=r.startup_mode,fee=r.fee,margin=r.margin,
                date=l.entry_date.iloc[i],held=code,direction=int(side),required_margin=required_margin,
                margin_shortfall=max(0.,required_margin-before),before_equity=before,after_equity=l.equity.iloc[i],closing_fees=l.fees.iloc[i]))
    pd.DataFrame(forced).to_csv(OUT/"forced_exit_events.csv",index=False)
    checks=dict(accounts=len(totals),new_accounts=8250,windows=len(data),old_accounts_unchanged=9750,
        annual_controls_causal_reselected=sum(json.loads((OUT/str(y)/"checks.json").read_text())["replayed_controls"] for y in range(2022,2027)),
        decision_records=len(decisions),unique_validation_sets=audit.validation_id.nunique(),exact_current_equity=True,
        mature_history_only=True,all_trade_cash_reconciles=True,opportunity_rows=len(op),opportunity_cash_reconciles=True,
        new_forced=len(forced),cash_activations=len(activated),unit_tests=84,**json.loads((OUT/"engine_checks.json").read_text()))
    (OUT/"checks.json").write_text(json.dumps(checks,indent=2))
    lines=["# 股指期货：重新入场与可恢复现金候选","",
        "数据冻结截至2026-11-06；50万元、最多一手、全局固定日历、无外部补资。保留9750旧账户，新增8250账户；本轮2×2×3主实验含9000账户（含750个旧actual_equity对照）。费用3/6bps、保证金15/25%为研究情景。","",
        "## 预先固定的机制","",
        "- 固定：始终使用base入场规则，原季度重训预测面板不变；滚动：每季度按实际权益与已成熟历史窗口重选。两者不能混同固定模型权重。",
        "- 等待5日：每次实际退出后，同品种同方向在随后5个交易日不得重入，最早第6日且须在全局入场日；反向、其他品种仍可交易。",
        "- 信号重置：退出后，同品种同方向须至少一个已观察收盘的基础信号失效，再次满足策略条件才可进场；缺失数据不算失效。",
        "- 两种限制都覆盖盈利、亏损、模型、资金和策略切换退出；换月不触发退出，季度与现金期间不清除记忆。",
        "- 可恢复现金：现金零收益、零评分，不计利息；启动和每季度复核。策略不合格或评分≤0且无可启用候选时转现金；恢复需合格策略评分≥已有0.005门槛，并通过已有净现金优势/成本与可交易门槛。保持合格正评分的当前策略时沿用旧切换门槛。",
        "- 未放宽原最少交易/持仓、DD90、资金拒绝约束。现金自身不按交易次数筛掉；恢复候选不与现金的零回撤硬比较，否则会永远无法恢复。",
        "- 历史验证：过去36个月完整122/244日月度起点、12月近期权重，每窗从精确当前权益重放同一重新入场规则。内层验证固定候选，现金为外层选择机制，不递归回测历史现金选择器。独立历史窗初始为空仓且无旧交易记忆；真实外层账户记忆跨季度跨年持续。",
        "- 全部5个旧退出版本保留；本轮不增加止损阈值、不选择最佳等待天数、不按已知年份调门槛。","",
        "## 5日退出：完整244日等长窗口","","以下为全部月度起点，重叠窗口并非独立样本；每种方案47个完整窗口。","",
        "| 情景 | 方案 | 盈利比例 | 收益中位 | P10收益 | 最差收益 | 最差回撤 | 空仓比例中位 | 无交易账户数 |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    p=dist[dist.variant.eq("mean5")&dist.window.eq("sessions_244")]
    for r in p.itertuples():
        lines.append(f"| {r.fee*10000:.0f}bps/{r.margin:.0%} | {LABELS[r.startup_mode]} | "+" | ".join(pct(getattr(r,k)) for k in ("profitable_fraction","median_return","p10_return","worst_return","worst_drawdown","median_flat_ratio"))+f" | {r.zero_trade_accounts} |")
    lines += ["","## 逐年单年与各年起累计至今","","每格净收益/最大回撤。独立单年每年50万元；累计至共同终点保持权益与重新入场状态，不逐年重置。2026为部分年度。",""]
    for fee,margin in PROFILES:
        for variant in VARIANTS:
            for style in ("fixed","rolling"):
                modes=[m for m,c in CONFIGS.items() if c[0]==style]
                for window,label in (("single_year","独立单年"),("to_latest","各年起累计至今")):
                    part=main[main.annual&main.fee.eq(fee)&main.margin.eq(margin)&main.variant.eq(variant)&main.window.eq(window)]
                    performance_table(lines,f"### {fee*10000:.0f}bps/{margin:.0%} · {NAMES[variant]} · {style} · {label}",part,"start_year",modes)
    lines += ["## 5日退出：逐年等长窗口","","每格收益中位/最差收益/最差回撤，只纳入完整窗口；2026暂无完整244日新起点。",""]
    for fee,margin in PROFILES:
        for style in ("fixed","rolling"):
            modes=[m for m,c in CONFIGS.items() if c[0]==style]
            for window in ("sessions_122","sessions_244"):
                p=cohorts[cohorts.fee.eq(fee)&cohorts.margin.eq(margin)&cohorts.variant.eq("mean5")&cohorts.window.eq(window)]
                lines.extend([f"### {fee*10000:.0f}bps/{margin:.0%} · {style} · {window}","","| 年 | 起点数 | "+" | ".join(LABELS[m] for m in modes)+" |","|---|---:|"+"---:|"*len(modes)])
                for year,g in p.groupby("start_year"):
                    g=g.set_index("startup_mode")
                    lines.append(f"| {year} | {int(g.starts.iloc[0])} | "+" | ".join(" / ".join(pct(g.loc[m,k]) for k in ("median_return","worst_return","worst_drawdown")) for m in modes)+" |")
                lines.append("")
    lines += ["## 近两年月度起点：收益、现金和恢复","","累计至共同终点，收益中位可能来自不同于现金比例中位的账户。","",
        "| 年/情景 | 方案 | 盈利比例 | 收益中位 | 最差收益 | 最差回撤 | 空仓比例中位 | 现金决策比例中位 | 曾恢复交易比例 | 无交易数 |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    p=cohorts[cohorts.start_year.isin(["2025","2026"])&cohorts.variant.eq("mean5")&cohorts.window.eq("to_latest")]
    for r in p.itertuples():
        lines.append(f"| {r.start_year}/{r.fee*10000:.0f}bps/{r.margin:.0%} | {LABELS[r.startup_mode]} | "+" | ".join(pct(getattr(r,k)) for k in ("profitable_fraction","median_return","worst_return","worst_drawdown","median_flat_ratio","median_cash_ratio","resumed_fraction"))+f" | {r.zero_trade_accounts} |")
    lines += ["","## 滚动重选的独立增量","","同一现金机制和重新入场规则，滚动减固定。不是事后在两者中挑好年份；主展示5日退出、完整244日、联合压力。","",
        "| 起始年 | 滚动方案 | 起点数 | 收益差中位 | 收益改善比例 | 回撤改善比例 | 回撤恶化比例 |","|---|---|---:|---:|---:|---:|---:|"]
    p=effects[effects.effect.eq("rolling")&effects.variant.eq("mean5")&effects.fee.eq(.0006)&effects.margin.eq(.25)&effects.window.eq("sessions_244")]
    for r in p.itertuples():
        lines.append(f"| {r.start_year} | {LABELS[r.startup_mode]} | {r.starts} | "+" | ".join(pct(getattr(r,k)) for k in ("median_return_change","return_improved_fraction","drawdown_improved_fraction","drawdown_worsened_fraction"))+" |")
    lines += ["","## 错失收益与减少亏损：配对路径归因","",
        "在新方案空仓、对应原方案持仓的相同日期，把原方案逐日含费正损益累计为错失盈利，负损益绝对值累计为减少亏损；两者必须同时看。这是路径解释，不能当成可以择日交易获得的额外收益。新方案平仓费用和其他日期持仓/策略改变也计入对账；完整收益差=减少亏损−错失盈利+新增空仓日费用+其他路径差异。",
        "以下为2025/2026月度起点累计至今，5日退出联合压力；金额中位单位万元，各列中位不是同一个账户，不可直接相加。cash效果对照同一重新入场规则且不允许现金的方案；reentry效果对照同样固定/滚动的原入场方案。","",
        "| 年 | 改动 | 新方案 | 收益差中位 | 错失盈利中位 | 减少亏损中位 | 新增空仓对照持仓日中位 |","|---|---|---|---:|---:|---:|---:|"]
    p=effects[effects.effect.isin(["cash","reentry"])&effects.start_year.isin(["2025","2026"])&effects.variant.eq("mean5")&effects.fee.eq(.0006)&effects.margin.eq(.25)&effects.window.eq("to_latest")]
    for r in p.itertuples():
        lines.append(f"| {r.start_year} | {r.effect} | {LABELS[r.startup_mode]} | {pct(r.median_return_change)} | {r.median_missed_positive/10000:.2f} | {r.median_avoided_negative/10000:.2f} | {r.median_additional_flat_days:.0f} |")
    lines += ["","## 重新入场限制也会改变品种选择","",
        "只屏蔽同品种同方向，其他品种/反向可正常参与排名。因此不能只数最终空仓的次数：即使只增加少量空仓，也可能多次换选其他合约，进而改变整条资金路径。以下为5日退出、联合压力、58个月度起点累计至今的次数中位。","",
        "| 方案 | 有候选被屏蔽的入场日 | 全部可选信号被屏蔽而空仓 | 因首选被屏蔽改选其他候选 |","|---|---:|---:|---:|"]
    for mode,c in CONFIGS.items():
        if c[1] or c[2]=="none":continue
        r=dist[dist.startup_mode.eq(mode)&dist.variant.eq("mean5")&dist.fee.eq(.0006)&dist.margin.eq(.25)&dist.window.eq("to_latest")].iloc[0]
        lines.append(f"| {LABELS[mode]} | {r.median_reentry_rejected_days:.0f} | {r.median_reentry_blocks:.0f} | {r.median_substitution_entries:.0f} |")
    lines += ["","## 持续账户：近两年当年表现","","承接此前权益的当年收益/年内最大回撤，与年度重置不同。",""]
    for style in ("fixed","rolling"):
        modes=[m for m,c in CONFIGS.items() if c[0]==style]
        part=annual[annual.start.isin(["20220101","20250101"])&annual.year.ge("2025")&annual.side.eq("all")&annual.variant.eq("mean5")&annual.fee.eq(.0006)&annual.margin.eq(.25)&annual.startup_mode.isin(CONFIGS)].copy()
        part["起始年/当年"]=part.start.str[:4]+"/"+part.year
        performance_table(lines,f"### 联合压力 · {style}",part,"起始年/当年",modes)
    lines += ["## 全部退出、全部起点：最差持续账户","","分别搜索最差收益和最差回撤，可能不是同一账户；不能只展示5日退出。","",
        "| 情景 | 方案 | 最差收益 | 起点/退出 | 最差回撤 | 起点/退出 |","|---|---|---:|---|---:|---|"]
    for fee,margin in PROFILES:
        for mode in CONFIGS:
            g=main[main.startup_mode.eq(mode)&main.fee.eq(fee)&main.margin.eq(margin)&main.window.eq("to_latest")]
            a=g.loc[g.total_return.idxmin()];b=g.loc[g.max_drawdown.idxmin()]
            lines.append(f"| {fee*10000:.0f}bps/{margin:.0%} | {LABELS[mode]} | {pct(a.total_return)} | {a.start}/{a.variant} | {pct(-b.max_drawdown)} | {b.start}/{b.variant} |")
    lines += ["","## 旧方案对照仍保留","","主表滚动原入场即上轮actual_equity；以下为原优势门槛和本金研究对照，主展示联合压力、5日模型、独立单年。",""]
    legacy_modes=["actual_equity","equity_score_gate","risk_stop5","risk_fuse20","risk_combined"]
    legacy=data[data.startup_mode.isin(legacy_modes)&data.annual&data.variant.eq("mean5")&data.fee.eq(.0006)&data.margin.eq(.25)&data.window.eq("single_year")].copy();legacy["start_year"]=legacy.start.str[:4]
    performance_table(lines,"### 旧方案原样保留",legacy,"start_year",legacy_modes)
    lines += ["## 审计与局限","","```json",json.dumps(checks,ensure_ascii=False,indent=2),"```","",
        "所有收益与费用按50万元起始现金记账，最多一手；现金不计利息、没有补资。重新入场抑制不等于资本不足；资金拒绝比例仅统计实际允许交易时的合格信号，同时另报现金期间基础信号和资金可执行机会，避免用不交易美化执行率。",
        "全套历史已反复观察，重叠窗口不是独立证据；本轮预登记规则并保留全部结果，但仍非新增未触碰OOT。现金恢复只在季度复核，不承诺及时捕捉行情或避免季度内亏损；日线也不覆盖盘中缺口与无法成交。",
        "逐年等长分布、年初五日起点、资金/费用分离、交易明细、历史验证输入、全量归因见同目录CSV和各年validation目录。",""]
    if (OUT/"findings.md").exists():lines[4:4]=[(OUT/"findings.md").read_text(),""]
    (OUT/"report.md").write_text("\n".join(lines))
    charts(main)
    print(json.dumps(checks,ensure_ascii=False))
    print(dist[dist.variant.eq("mean5")&dist.fee.eq(.0006)&dist.margin.eq(.25)&dist.window.eq("sessions_244")].to_string(index=False))


if __name__=="__main__":main()
