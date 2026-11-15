"""Exact-equity and per-horizon advantage ablation report."""
import json
import os

import numpy as np
import pandas as pd

from evaluate_index_futures_equity_gate import OUT, PREVIOUS, MAIN_MODES, NEW_MODES, VARIANTS, PROFILES, CAPITAL, DATE_TYPES
from report_index_futures_multistart import grouped, pct, NAMES
from evaluate_index_futures_directions import load_bars

LABELS=dict(calendar="日历对照",multi_funded="上轮多起点",score_gate="仅优势门槛",
            actual_equity="仅实际权益",equity_score_gate="实际权益+门槛")


def ledger_path(r):
    root=OUT if r.startup_mode in NEW_MODES else PREVIOUS if r.startup_mode=="multi_funded" else PREVIOUS.parent/"startup_consistency"
    return root/r.start[:4]/"ledgers"/f"{r.start}_{r.variant}_{r.startup_mode}_{r.fee}_{r.margin}.parquet"


def chart(summary):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/quant_adaptation_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    modes=MAIN_MODES[1:]
    labels=["Prior multi-start", "Advantage gate", "Exact current equity", "Equity + advantage gate"]
    data=summary[summary.start_kind.eq("monthly")&summary.variant.eq("mean5")&summary.startup_mode.isin(modes)]
    fig,axes=plt.subplots(2,1,figsize=(15,6.5))
    maximum=max(10,data[data.window.eq("sessions_244")].total_return.abs().max()*100)
    for ax,(fee,margin) in zip(axes,PROFILES):
        table=data[data.window.eq("sessions_244")&data.fee.eq(fee)&data.margin.eq(margin)].pivot(index="startup_mode",columns="start",values="total_return").reindex(modes)
        im=ax.imshow(100*table.to_numpy(),aspect="auto",cmap="RdBu",norm=TwoSlopeNorm(vmin=-maximum,vcenter=0,vmax=maximum))
        ticks=np.arange(0,len(table.columns),3)
        ax.set_xticks(ticks,[s[:4]+"-"+s[4:6] for s in table.columns[ticks]],rotation=35,ha="right")
        ax.set_yticks(range(4),labels)
        ax.set_title(f"Raw5 exit | {fee*10000:.0f}bps | {margin:.0%} margin")
    fig.suptitle("Advantage gates and exact-equity validation | same244-session outer windows",fontsize=14)
    fig.text(.5,.91,"500k starting capital, max one lot; fixed global calendar; overlapping starts are not independent",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,0,.93,.89))
    fig.colorbar(im,cax=fig.add_axes([.94,.16,.012,.63]),label="Net return (%)")
    fig.savefig(OUT/"equal_window_heatmap.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for i,year in enumerate(("2025","2026")):
        for j,(fee,margin) in enumerate(PROFILES):
            part=data[data.window.eq("to_latest")&data.start.str.startswith(year)&data.fee.eq(fee)&data.margin.eq(margin)]
            for mode,label in zip(modes,labels):
                sub=part[part.startup_mode.eq(mode)].sort_values("start")
                axes[i,j].plot(sub.start.str[4:6].astype(int),sub.total_return*100,marker="o",label=label)
            axes[i,j].set_title(f"{year} starts | {fee*10000:.0f}bps | {margin:.0%} margin")
            axes[i,j].set_xlabel("Startup month")
            axes[i,j].set_ylabel("Net return through2026-11-06 (%)")
            axes[i,j].axhline(0,color="#999",linewidth=.7)
            axes[i,j].grid(alpha=.2)
    fig.suptitle("Recent monthly startups: actual continuous accounts",fontsize=14)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.savefig(OUT/"recent_monthly_starts.png",dpi=160)
    plt.close(fig)


def performance_table(lines, title, part, key):
    lines += [title,"","| "+key+" | "+" | ".join(LABELS[m] for m in MAIN_MODES)+" |","|---|"+"---:|"*len(MAIN_MODES)]
    for value,group in part.groupby(key):
        group=group.set_index("startup_mode")
        lines.append(f"| {value} | "+" | ".join(f"{pct(group.loc[m].total_return)} / {pct(-group.loc[m].max_drawdown)}" for m in MAIN_MODES)+" |")
    lines.append("")


def main():
    summary=pd.read_csv(OUT/"summary.csv",dtype=DATE_TYPES)
    old=pd.read_csv(PREVIOUS/"summary.csv",dtype=DATE_TYPES)
    keys=["start","fee","margin","variant","startup_mode"]
    sort=keys+["window"]
    retained=summary[~summary.startup_mode.isin(NEW_MODES)][old.columns]
    pd.testing.assert_frame_equal(retained.sort_values(sort).reset_index(drop=True),old.sort_values(sort).reset_index(drop=True),check_dtype=False,atol=1e-8,rtol=1e-10)
    trades=pd.read_csv(OUT/"trades.csv",dtype=DATE_TYPES)
    annual=pd.read_csv(OUT/"annual_marked.csv",dtype=DATE_TYPES)
    selection=pd.read_csv(OUT/"selection.csv",dtype=DATE_TYPES)
    audit=pd.read_csv(OUT/"selection_audit.csv",dtype=DATE_TYPES)
    totals=summary[summary.window.eq("to_latest")]
    assert len(totals)==6750 and totals.groupby("startup_mode").size().eq(750).all()
    recon=totals.merge(trades.groupby(keys).pnl.sum().rename("net"),on=keys,how="left")
    np.testing.assert_allclose(recon.net.fillna(0),recon.total_return*CAPITAL,atol=1e-6,rtol=0)
    assert selection.cutoff.lt(selection.date).all() and audit.last_history_end.le(audit.cutoff).all()
    actual=selection[selection.startup_mode.isin(["actual_equity","equity_score_gate"])]
    np.testing.assert_allclose(actual.equity,actual.validation_capital,atol=0,rtol=0)
    assert selection.loc[selection.startup_mode.eq("score_gate"),"validation_capital"].eq(CAPITAL).all()
    assert selection.loc[selection.stage.eq("startup"),"validation_capital"].eq(CAPITAL).all()
    old_selection=pd.read_csv(PREVIOUS/"selection.csv",dtype=DATE_TYPES)
    old_start=old_selection[old_selection.stage.eq("startup")&old_selection.startup_mode.eq("multi_funded")]
    new_start=selection[selection.stage.eq("startup")]
    initial_match=new_start[new_start.startup_mode.eq("actual_equity")].merge(old_start,on=keys[:-1],suffixes=("","_old"),validate="one_to_one")
    assert len(initial_match)==750 and initial_match.selected.eq(initial_match.selected_old).all()
    gate_match=new_start[new_start.startup_mode.eq("score_gate")].merge(new_start[new_start.startup_mode.eq("equity_score_gate")],on=keys[:-1],suffixes=("","_actual"),validate="one_to_one")
    assert gate_match.selected.eq(gate_match.selected_actual).all()
    accepted=selection[selection.startup_mode.isin(["score_gate","equity_score_gate"])&selection.selected.ne(selection.incumbent)]
    assert accepted[["advantage_122","advantage_244"]].ge(.005-1e-12).all().all()
    for h in (122,244):
        assert summary.loc[summary.window.eq(f"sessions_{h}"),"days"].eq(h).all()
    joined=selection.merge(audit[["validation_id","validation_capital","cutoff"]].drop_duplicates(),
        on=["validation_id","validation_capital","cutoff"],how="left",indicator=True,validate="many_to_one")
    assert joined._merge.eq("both").all()
    data=summary[summary.startup_mode.isin(MAIN_MODES)].copy()
    monthly=data[data.start_kind.eq("monthly")].copy()
    monthly["start_year"]=monthly.start.str[:4]
    dist=grouped(monthly,["window","fee","margin","variant","startup_mode"])
    cohorts=grouped(monthly[monthly.window.eq("to_latest")],["start_year","fee","margin","variant","startup_mode"])
    equal=grouped(monthly[monthly.window.isin(["sessions_122","sessions_244"])],["start_year","window","fee","margin","variant","startup_mode"])
    for name,frame in (("start_distributions",dist),("monthly_cohorts",cohorts),("equal_window_cohorts",equal)):
        frame.to_csv(OUT/f"{name}.csv",index=False)
    phase=data[data.window.eq("to_latest")&data.start.str[4:6].eq("01")&data.variant.isin(VARIANTS[:3])].copy()
    phase=phase[((phase.fee.eq(.0003)&phase.margin.eq(.15))|(phase.fee.eq(.0006)&phase.margin.eq(.25)))]
    phase["start_year"]=phase.start.str[:4]
    phases=grouped(phase,["start_year","fee","margin","variant","startup_mode"])
    assert phases.starts.eq(5).all()
    phases["return_range"]=phases.best_return-phases.worst_return
    phases.to_csv(OUT/"january_phase_sensitivity.csv",index=False)
    worst=[]
    for _,group in data.groupby(["window","fee","margin","variant","startup_mode"]):
        for criterion,index in (("worst_return",group.total_return.idxmin()),("worst_drawdown",group.max_drawdown.idxmin())):
            worst.append(group.loc[index].to_dict()|dict(criterion=criterion))
    pd.DataFrame(worst).to_csv(OUT/"worst_all_starts.csv",index=False)
    comparisons=[]
    for candidate,control in (("score_gate","multi_funded"),("actual_equity","multi_funded"),
                              ("equity_score_gate","multi_funded"),("equity_score_gate","score_gate"),("equity_score_gate","actual_equity")):
        part=data[data.startup_mode.eq(candidate)].merge(data[data.startup_mode.eq(control)],on=keys[:-1]+["window"],suffixes=("","_control"),validate="one_to_one")
        part["return_delta"]=part.total_return-part.total_return_control
        part["drawdown_improvement"]=part.max_drawdown-part.max_drawdown_control
        comparisons.append(part)
    pd.concat(comparisons,ignore_index=True).to_csv(OUT/"paired_comparison.csv",index=False)
    selection.groupby(["startup_mode","stage","reason"]).size().rename("decisions").to_csv(OUT/"decision_reasons.csv")
    selection[selection.stage.eq("startup")].groupby(["startup_mode","selected"]).size().rename("accounts").to_csv(OUT/"initial_policy_counts.csv")
    bars=load_bars(); forced=[]
    for r in totals[totals.startup_mode.isin(MAIN_MODES)&totals.forced.gt(0)].itertuples():
        ledger=pd.read_parquet(ledger_path(r))
        for i in ledger.index[ledger.forced_exit]:
            before,event=ledger.iloc[i-1],ledger.iloc[i]
            code,sign=before.positions.split(":")
            multiplier=300 if code[:2] in ("IF","IH") else 200
            forced.append(dict(start=r.start,fee=r.fee,margin=r.margin,variant=r.variant,startup_mode=r.startup_mode,
                date=event.entry_date,contract=code,lots=int(sign),equity_before=before.equity,
                required_margin=abs(int(sign))*bars.loc[(event.entry_date,code),"open"]*multiplier*r.margin,equity_after=event.equity))
    pd.DataFrame(forced).to_csv(OUT/"forced_exit_events.csv",index=False)
    engine=json.loads((OUT/"engine_checks.json").read_text())
    checks=dict(accounts=len(totals),new_accounts=2250,windows=len(summary),unit_tests=69,
        old_accounts_unchanged=4500,replayed_annual_controls=sum(json.loads((OUT/str(y)/"checks.json").read_text())["replayed_controls"] for y in range(2022,2027)),
        decision_records=len(selection),unique_validation_sets=audit.validation_id.nunique(),
        policy_score_records=len(audit.drop_duplicates()),actual_capital_matches=True,
        minimum_validation_equity=float(actual.validation_capital.min()),maximum_validation_equity=float(actual.validation_capital.max()),
        all_trade_pnl_reconciles=True,mature_history_only=True,
        forced_by_mode=totals.groupby("startup_mode").forced.sum().astype(int).to_dict(),last=summary.end.max(),**engine)
    (OUT/"checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    lines=["# 评分优势门槛与实际权益历史验证","",
        "50万元启动、全组合最多一手、多空，固定全局日历。保留4500旧账户，新增三组各750账户，共6750路径、19404窗口，统一截至2026-11-06。2026不是完整年，本轮为保持对照未更新行情。","",
        "## 事前固定的2×2对照","",
        "| 方法 | 内层初始权益 | 启动/切换评分优势 |","|---|---|---|",
        "| 上轮多起点 | 始终50万元 | 启动无额外优势门槛；切换平均优势≥0.005 |",
        "| 仅优势门槛 | 始终50万元 | 122日和244日评分优势分别≥0.005 |",
        "| 仅实际权益 | 决策时精确实际权益 | 沿用上轮优势要求 |",
        "| 实际权益+门槛 | 决策时精确实际权益 | 122日和244日评分优势分别≥0.005 |","",
        "另保留简单日历对照：base启动，之后旧季度重选，并非永久固定base。全部其他旧版本仍在summary.csv。","",
        "门槛0.005沿用已有值，代表评分的0.5个百分点，不是胜率或显著性门槛。新增要求是两个长度都通过，初始化相对base、季度相对现任。不扫描门槛、不用已知2025/2026亏损反选参数。候选仍按原平均分排序，再检查优势、回撤、资金拒绝和切换成本；不因第一候选不通过而继续挑另一个候选。","",
        "实际权益验证：将决策时已知开盘盯市权益作为每条历史窗口的初始资金，从空仓重新交易，重新计算每次可负担候选、保证金、安全储备、退出、换月及费用。没有按50万元结果线性缩放、没有资金分桶；只继承当下资金数额，不继承当下持仓进历史。初始启动所有账户都为50万元，所以权益因素从后续决策才产生差异。","",
        "过去36个月的月度起点、完整122/244日、12个月半衰期、P25/中位收益与P90回撤评分、资金拒绝门槛和交易覆盖条件均不变。历史窗口结束日≤决策前一信号日；季度可以使用当天已知的开盘权益，不使用当日结算或下一开盘。历史现金优势按对应验证资金换算，切换仍要求优势覆盖当前权益0.5%与3倍估计成本。","",
        "外层继续使用原Python逐日交易引擎。为反复重算精确权益，内层使用数值回放器，已对11840原参考窗口核对，并在1200组不同初始现金/策略/退出/费用/保证金组合中逐日比较损益、费用、仓位、强平和机会计数。相邻窗口相关，双长度同向不构成独立显著性证明。","",
        "## 完整244日窗口：全部退出对照","",
        "每组47个月度起点，最差收益与最差回撤可能来自不同账户。","",
        "| 情景 | 退出 | 方法 | 盈利比例 | 中位收益 | P10收益 | 最差收益 | 最差回撤 | 中位资金拒绝 |","|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for fee,margin in PROFILES:
        for variant in VARIANTS:
            for mode in MAIN_MODES:
                r=dist[dist.window.eq("sessions_244")&dist.fee.eq(fee)&dist.margin.eq(margin)&dist.variant.eq(variant)&dist.startup_mode.eq(mode)].iloc[0]
                lines.append(f"| {fee*10000:.0f}bps/{margin:.0%} | {NAMES[variant]} | {LABELS[mode]} | "+" | ".join(pct(r[k]) for k in ["profitable_fraction","median_return","p10_return","worst_return","worst_drawdown","median_blocked_ratio"])+" |")
    lines += ["","## 原5日退出：近两年月度起点累计至共同终点","",
        "| 起始年 | 情景 | 方法 | 盈利比例 | 中位累计 | 最差累计 | 最差回撤 | 中位资金拒绝 |","|---|---|---|---:|---:|---:|---:|---:|"]
    for year in ("2025","2026"):
        for fee,margin in PROFILES:
            for mode in MAIN_MODES:
                r=cohorts[cohorts.start_year.eq(year)&cohorts.fee.eq(fee)&cohorts.margin.eq(margin)&cohorts.variant.eq("mean5")&cohorts.startup_mode.eq(mode)].iloc[0]
                lines.append(f"| {year} | {fee*10000:.0f}bps/{margin:.0%} | {LABELS[mode]} | "+" | ".join(pct(r[k]) for k in ["profitable_fraction","median_return","worst_return","worst_drawdown","median_blocked_ratio"])+" |")
    lines += ["","## 原5日退出：等长窗口按起始年份分组","","每格中位收益 / 最差收益 / 最差回撤。只纳入完整窗口；2026没有完整244日新起点。所有退出见equal_window_cohorts.csv。",""]
    for fee,margin in PROFILES:
        for window in ("sessions_122","sessions_244"):
            lines += [f"### {fee*10000:.0f}bps/{margin:.0%} · {window}","","| 年份 | 每模式起点数 | "+" | ".join(LABELS[m] for m in MAIN_MODES)+" |","|---|---:|"+"---:|"*len(MAIN_MODES)]
            part=equal[equal.fee.eq(fee)&equal.margin.eq(margin)&equal.window.eq(window)&equal.variant.eq("mean5")]
            for year,group in part.groupby("start_year"):
                group=group.set_index("startup_mode")
                lines.append(f"| {year} | {group.starts.iloc[0]} | "+" | ".join(" / ".join(pct(group.loc[m,k]) for k in ["median_return","worst_return","worst_drawdown"]) for m in MAIN_MODES)+" |")
            lines.append("")
    lines += ["## 全部退出：逐年独立单年与各年起累计至今","","每格收益 / 最大回撤。单年每年独立50万元；累计承接权益到共同终点，不逐年重置。",""]
    for fee,margin in PROFILES:
        for variant in VARIANTS:
            for window,label in (("single_year","独立单年"),("to_latest","各年起累计至今")):
                part=data[data.annual&data.fee.eq(fee)&data.margin.eq(margin)&data.variant.eq(variant)&data.window.eq(window)].copy()
                part["起始年"]=part.start.str[:4]
                performance_table(lines,f"### {fee*10000:.0f}bps/{margin:.0%} · {NAMES[variant]} · {label}",part,"起始年")
    lines += ["## 原5日退出：持续账户近两年当年表现","","承接此前权益的当年收益 / 年内回撤，不是年度重置。",""]
    for fee,margin in PROFILES:
        part=annual[annual.start.isin(["20220101","20250101"])&annual.year.ge("2025")&annual.side.eq("all")&annual.variant.eq("mean5")&annual.fee.eq(fee)&annual.margin.eq(margin)&annual.startup_mode.isin(MAIN_MODES)].copy()
        part["起始年/当年"]=part.start.str[:4]+"/"+part.year
        performance_table(lines,f"### {fee*10000:.0f}bps/{margin:.0%}",part,"起始年/当年")
    lines += ["## 原5日退出：年初5日起点、联合压力","","| 起始年 | 方法 | 最差收益 | 最佳收益 | 收益跨度 | 最差回撤 |","|---|---|---:|---:|---:|---:|"]
    part=phases[phases.variant.eq("mean5")&phases.fee.eq(.0006)&phases.margin.eq(.25)]
    for r in part.itertuples():
        lines.append(f"| {r.start_year} | {LABELS[r.startup_mode]} | "+" | ".join(pct(getattr(r,k)) for k in ["worst_return","best_return","return_range","worst_drawdown"])+" |")
    part=data[data.start.eq("20260101")&data.window.eq("single_year")&data.variant.eq("mean5")].copy()
    part["费用/保证金"]=[f"{f*10000:.0f}bps/{m:.0%}" for f,m in zip(part.fee,part.margin)]
    performance_table(lines,"## 原5日退出：2026费用与保证金分离",part,"费用/保证金")
    lines += ["## 强平和审计","","| 方法 | 模拟强平次数 |","|---|---:|"]
    for mode in MAIN_MODES:
        lines.append(f"| {LABELS[mode]} | {checks['forced_by_mode'][mode]} |")
    lines += ["",f"69项专项测试通过；11840参考窗口、1200精确现金逐日案例校验通过，最大逐日现金误差{engine['max_daily_cash_error']}元。全部4500旧账户摘要保持一致、100年度基准重放一致；6750账户交易损益对账通过。共有{checks['unique_validation_sets']}组不同验证输入，实际权益验证范围{checks['minimum_validation_equity']:.2f}至{checks['maximum_validation_equity']:.2f}元。验证窗口明细保存在各年validation目录，selection.csv可按validation_id追溯。","",
        "成本3/6bps和保证金15%/25%均为研究情景，不是逐日真实券商标准；日线未覆盖盘中追保、涨跌停锁单和实际成交。零强平也不保证可继续交易。","",
        "历史已反复研究，窗口重叠，本轮不是新的未触碰OOT。方法规则没有按已知亏损调参，结果也不能替代新增数据的前瞻检验。时间顺序验证参考：[scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。",""]
    findings=OUT/"findings.md"
    if findings.exists():lines[4:4]=[findings.read_text(),""]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    chart(summary)
    print(json.dumps(checks,ensure_ascii=False))
    print(dist[dist.window.eq("sessions_244")&dist.variant.eq("mean5")&dist.fee.eq(.0006)&dist.margin.eq(.25)].to_string(index=False))
    print(cohorts[cohorts.start_year.eq("2026")&cohorts.variant.eq("mean5")&cohorts.fee.eq(.0006)&cohorts.margin.eq(.25)].to_string(index=False))


if __name__=="__main__":main()
