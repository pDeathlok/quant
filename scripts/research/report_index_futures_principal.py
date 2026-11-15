"""Risk-account attribution, start-grid comparison and transparent cash additions."""
import json
import os
import numpy as np
import pandas as pd
from evaluate_index_futures_principal import OUT, PREVIOUS, NEW_MODES, VARIANTS, PROFILES, DATE_TYPES, CAPITAL
from report_index_futures_multistart import grouped, pct, NAMES

MODES=("actual_equity","equity_score_gate",*NEW_MODES)
LABELS=dict(actual_equity="实际权益对照",equity_score_gate="实际权益+优势门槛",
    risk_entry5="入场波动5%",risk_stop5="逐笔止损5%",risk_fuse20="本金止损20%",risk_combined="三者组合")


def table(lines,title,part,key):
    lines.extend([title,"","| "+key+" | "+" | ".join(LABELS[m] for m in MODES)+" |","|---|"+"---:|"*len(MODES)])
    for value,g in part.groupby(key):
        g=g.set_index("startup_mode")
        lines.append(f"| {value} | "+" | ".join(f"{pct(g.loc[m,'total_return'])} / {pct(-g.loc[m,'max_drawdown'])}" for m in MODES)+" |")
    lines.append("")


def charts(summary):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/quant_adaptation_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels=["Exact equity baseline","Equity + score gate","Entry risk5%","Trade loss5%","Principal floor20%","Combined"]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for i,year in enumerate(("2025","2026")):
        for j,(fee,margin) in enumerate(PROFILES):
            data=summary[summary.variant.eq("mean5")&summary.start_kind.eq("monthly")&summary.window.eq("to_latest")&summary.start.str.startswith(year)&summary.fee.eq(fee)&summary.margin.eq(margin)]
            ax=axes[i,j]
            for mode,label in zip(MODES,labels):
                s=data[data.startup_mode.eq(mode)].sort_values("start")
                ax.plot(s.start.str[4:6].astype(int),100*s.total_return,marker="o",ms=3,label=label)
            ax.set_title(f"{year} starts | {fee*10000:.0f}bps, {margin:.0%} margin")
            ax.set_xlabel("Startup month");ax.set_ylabel("Net return to2026-11-06 (%)")
            ax.axhline(0,c="grey",lw=.8);ax.grid(alpha=.2)
    fig.suptitle("Raw5 exit: start sensitivity under fixed risk rules",fontsize=14)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(OUT/"recent_starts.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,10))
    paths=[("20230801","mean5",.0006,.25),("20240601","control",.0003,.15),("20230201","control",.0006,.25)]
    for ax,(start,variant,fee,margin) in zip(axes,paths):
        for mode,label in zip(MODES,labels):
            root=OUT if mode in NEW_MODES else PREVIOUS
            ledger=pd.read_parquet(root/start[:4]/"ledgers"/f"{start}_{variant}_{mode}_{fee}_{margin}.parquet")
            ax.plot(pd.to_datetime(ledger.exit_date),ledger.equity/10000,label=label,lw=1.3)
        ax.axhline(40,ls="--",c="grey",lw=.8)
        ax.set_title(f"Start{start} | {variant} | {fee*10000:.0f}bps, {margin:.0%} margin")
        ax.set_ylabel("Equity (CNY10k)");ax.grid(alpha=.2)
    fig.suptitle("Observed worst paths: no external cash in these comparisons",fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.06,1,.96));fig.savefig(OUT/"worst_paths.png",dpi=160);plt.close(fig)


def main():
    data=pd.read_csv(OUT/"summary.csv",dtype=DATE_TYPES)
    old=pd.read_csv(PREVIOUS/"summary.csv",dtype=DATE_TYPES)
    keys=["start","fee","margin","variant","startup_mode"]
    sort=keys+["window"]
    kept=data[~data.startup_mode.isin(NEW_MODES)][old.columns]
    pd.testing.assert_frame_equal(kept.sort_values(sort).reset_index(drop=True),old.sort_values(sort).reset_index(drop=True),check_dtype=False,atol=1e-8,rtol=1e-10)
    trades=pd.read_csv(OUT/"trades.csv",dtype=DATE_TYPES)
    selection=pd.read_csv(OUT/"selection.csv",dtype=DATE_TYPES)
    audit=pd.read_csv(OUT/"selection_audit.csv",dtype=DATE_TYPES)
    annual=pd.read_csv(OUT/"annual_marked.csv",dtype=DATE_TYPES)
    total=data[data.window.eq("to_latest")]
    assert len(total)==9750 and total.groupby("startup_mode").size().eq(750).all()
    recon=total.merge(trades.groupby(keys).pnl.sum().rename("net"),on=keys,how="left")
    np.testing.assert_allclose(recon.net.fillna(0),recon.total_return*CAPITAL,atol=1e-6,rtol=0)
    assert selection.cutoff.lt(selection.date).all() and audit.last_history_end.le(audit.cutoff).all()
    np.testing.assert_array_equal(selection.equity,selection.validation_capital)
    main=data[data.startup_mode.isin(MODES)].copy();main["start_year"]=main.start.str[:4]
    monthly=main[main.start_kind.eq("monthly")]
    dist=grouped(monthly,["startup_mode","variant","fee","margin","window"])
    cohorts=grouped(monthly,["start_year","startup_mode","variant","fee","margin","window"])
    dist.to_csv(OUT/"start_distributions.csv",index=False);cohorts.to_csv(OUT/"year_cohorts.csv",index=False)
    phases=grouped(main[main.annual|main.start_kind.eq("phase")],
        ["start_year","startup_mode","variant","fee","margin","window"])
    phases.to_csv(OUT/"january_phase_sensitivity.csv",index=False)
    worst=[]
    for k,g in main.groupby(["startup_mode","fee","margin","window"]):
        for metric in ("total_return","max_drawdown"):
            worst.append(dict(g.loc[g[metric].idxmin()],criterion=metric))
    pd.DataFrame(worst).to_csv(OUT/"worst_all_starts.csv",index=False)
    base=main[main.startup_mode.eq("actual_equity")]
    pair=main[main.startup_mode.isin(NEW_MODES)].merge(base,on=["start","fee","margin","variant","window"],suffixes=("","_base"),validate="many_to_one")
    pair["return_change"]=pair.total_return-pair.total_return_base
    pair["drawdown_change"]=pair.max_drawdown-pair.max_drawdown_base
    pair.to_csv(OUT/"paired_comparison.csv",index=False)
    impact=pair.groupby(["startup_mode","variant","fee","margin","window"]).agg(
        cases=("start","size"),return_improved=("return_change",lambda x:x.gt(1e-10).mean()),
        drawdown_improved=("drawdown_change",lambda x:x.gt(1e-10).mean()),
        drawdown_worsened=("drawdown_change",lambda x:x.lt(-1e-10).mean()),
        median_return_change=("return_change","median"),median_trades=("trades","median"),
        median_active_days=("active_days","median"),halted_fraction=("halt_days",lambda x:x.gt(0).mean()),
        median_halt_days=("halt_days","median"),max_halt_days=("halt_days","max")).reset_index()
    impact.to_csv(OUT/"risk_impact.csv",index=False)
    forced_records=[]
    for r in total[total.startup_mode.isin(NEW_MODES)&total.forced.gt(0)].itertuples(index=False):
        filename=f"{r.start}_{r.variant}_{r.startup_mode}_{r.fee}_{r.margin}.parquet"
        ledger=pd.read_parquet(OUT/r.start[:4]/"ledgers"/filename)
        trace=pd.read_parquet(OUT/r.start[:4]/"traces"/filename)
        for i in ledger.index[ledger.forced_exit]:
            t=trace.iloc[i]
            # Prior record has this open's marked equity, prior held contract.
            # Opening notional can be recovered from exit commission for this
            # one-lot, flat forced day; validate that no new exposure was opened.
            assert ledger.gross_lots.iloc[i]==0
            notional=ledger.fees.iloc[i]/r.fee
            forced_records.append(dict(start=r.start,mode=r.startup_mode,variant=r.variant,fee=r.fee,margin=r.margin,
                date=t.date,held=t.held,opening_equity=t.equity,required_margin=notional*r.margin,
                minimum_margin_shortfall=max(0.,notional*r.margin-t.equity),strategy_action=t.action,
                after_exit_equity=ledger.equity.iloc[i]))
    pd.DataFrame(forced_records).to_csv(OUT/"forced_exit_events.csv",index=False)
    engine=json.loads((OUT/"engine_checks.json").read_text())
    checks=dict(accounts=len(total),new_accounts=3000,windows=len(data),old_accounts_unchanged=6750,
        annual_controls_replayed=sum(json.loads((OUT/str(y)/"checks.json").read_text())["replayed_controls"] for y in range(2022,2027)),
        new_decisions=len(selection),unique_validation_sets=audit.validation_id.nunique(),all_trade_cash_reconciles=True,
        exact_current_equity=True,mature_history_only=True,global_calendar_unchanged=(OUT/"global_calendar.csv").read_bytes()==(PREVIOUS/"global_calendar.csv").read_bytes(),
        new_forced=int(total[total.startup_mode.isin(NEW_MODES)].forced.sum()),unit_tests=75,**engine)
    (OUT/"checks.json").write_text(json.dumps(checks,indent=2))
    lines=["# 股指期货：本金保护、最差账户与补资诊断","",
        "冻结行情截至2026-11-06。主研究50万元、最多一手、全局每5日入场日历、实际权益历史验证，禁止补资；费用/保证金3bps/15%和6bps/25%为情景。6750旧账户原样保留，新增3000账户；报告比较其中两个实际权益旧方案及四个新方案。","",
        "## 预先固定的规则与检验边界","",
        "- 入场波动5%：两倍过去20日日波动率×一手名义金额，不超过当时权益5%。这是单日波动筛选，不是最大损失保证。",
        "- 逐笔止损5%：前一日收盘含费净亏损达到入场权益5%，下一开盘退出，与原退出模型同时有效。",
        "- 本金止损20%：前一日可平仓净权益≤40万元，下一开盘退出，此账户此后永久现金。不会按季度或年度重新放行；独立新起点账户另计。",
        "- 组合：以上三者同时实施。所有参数先注册，再批量运行，没有按亏损时点、品种或多空方向调参。",
        "- 历史选择仍只看当时已成熟的36个月内122/244交易日多起点窗口；每个窗口从当前实际权益开始、同样执行风控。绝对本金底线仍为40万元，不改成当前权益80%。",
        "- 新规则均加在实际权益对照上；实际权益+优势门槛保留为独立对照，未把门槛和风控一起调优。相同模型、标签、成本、候选策略、覆盖与切换约束。",
        "- 暂停导致的低回撤必须连同交易覆盖、停机天数和收益报告。收盘触发下一开盘执行，跳空可能越过5%/20%阈值。","",
        "## 最差旧账户逐笔拆解","",
        "完整个案见diagnosis/cases.csv，逐笔入场概率、波动、日内/隔夜损益、首次止损触发与退出原因见diagnosis/trades.csv。仅对已观察个案解释，不用于选择参数。","",
        "### 2023-08启动、5日退出、6bps/25%：五笔交易","",
        "| 入场→退出 | 品种方向 | 入场权益(万元) | 两倍日波动/权益 | 净损益(万元) | 退出原因 | 首次5%止损执行日/损益(万元) |","|---|---|---:|---:|---:|---|---|"]
    dt=pd.read_csv(OUT/"diagnosis/trades.csv",dtype={"entry":str,"exit":str,"first_stop_date":str,"start":str})
    case=dt[dt.start.eq("20230801")&dt["mode"].eq("actual_equity")&dt.variant.eq("mean5")]
    for r in case.itertuples():
        stop=f"{str(r.first_stop_date).split('.')[0]} / {r.first_stop_pnl/10000:.2f}" if pd.notna(r.first_stop_date) else "未触发"
        lines.append(f"| {r.entry}→{r.exit} | {r.product} {'多' if r.direction>0 else '空'} | {r.entry_equity/10000:.2f} | {pct(r.entry_two_sigma_fraction)} | {r.pnl/10000:.2f} | {r.exit_reason} | {stop} |")
    lines += ["","首次止损损益是沿原持仓路径，按首次前收盘触发后的开盘平仓计算，含已付费用和退出费；不能直接与原收益差相加预测整套策略，因为后续入场和策略选择会改变。fixed_schedule_attribution.csv只做冻结原选择日期的解释性拆解，不当成可部署策略。","",
        "## 各年单年与累计至今","","每格净收益 / 最大回撤。独立单年每年从50万元开始；各年起累计至今不重置权益。2026是不完整年度。没有2022起逐年截断累计表。",""]
    for fee,margin in PROFILES:
        for variant in VARIANTS:
            for window,label in (("single_year","独立单年"),("to_latest","各年起累计至今")):
                part=main[main.annual&main.fee.eq(fee)&main.margin.eq(margin)&main.variant.eq(variant)&main.window.eq(window)]
                table(lines,f"### {fee*10000:.0f}bps/{margin:.0%} · {NAMES[variant]} · {label}",part,"start_year")
    lines += ["## 5日退出：等长窗口逐年起点分布","","每格中位收益 / 最差收益 / 最差回撤。仅完整窗口，2026尚无完整244日新起点。",""]
    for fee,margin in PROFILES:
        for window in ("sessions_122","sessions_244"):
            part=cohorts[cohorts.fee.eq(fee)&cohorts.margin.eq(margin)&cohorts.variant.eq("mean5")&cohorts.window.eq(window)]
            lines += [f"### {fee*10000:.0f}bps/{margin:.0%} · {window}","","| 起始年 | 起点数 | "+" | ".join(LABELS[m] for m in MODES)+" |","|---|---:|"+"---:|"*len(MODES)]
            for year,g in part.groupby("start_year"):
                g=g.set_index("startup_mode")
                lines.append(f"| {year} | {int(g.starts.iloc[0])} | "+" | ".join(" / ".join(pct(g.loc[m,k]) for k in ("median_return","worst_return","worst_drawdown")) for m in MODES)+" |")
            lines.append("")
    lines += ["## 近两年月度起点：5日退出，累计至今","","| 年 | 情景 | 规则 | 盈利起点比例 | 中位收益 | 最差收益 | 最差回撤 |","|---|---|---|---:|---:|---:|---:|"]
    p=cohorts[cohorts.start_year.isin(["2025","2026"])&cohorts.variant.eq("mean5")&cohorts.window.eq("to_latest")]
    for r in p.itertuples():
        lines.append(f"| {r.start_year} | {r.fee*10000:.0f}bps/{r.margin:.0%} | {LABELS[r.startup_mode]} | "+" | ".join(pct(getattr(r,k)) for k in ("profitable_fraction","median_return","worst_return","worst_drawdown"))+" |")
    lines += ["","## 持续账户：近两年的当年表现","","承接此前权益的当年收益/回撤，不同于年度50万元重置。",""]
    for fee,margin in PROFILES:
        p=annual[annual.start.isin(["20220101","20250101"])&annual.year.ge("2025")&annual.side.eq("all")&annual.variant.eq("mean5")&annual.fee.eq(fee)&annual.margin.eq(margin)&annual.startup_mode.isin(MODES)].copy()
        p["起始年/当年"]=p.start.str[:4]+"/"+p.year
        table(lines,f"### {fee*10000:.0f}bps/{margin:.0%}",p,"起始年/当年")
    lines += ["## 20%本金止损的代价与覆盖","","停机不等于模型准确度提升；这里统计新方案所有起点，包含年初日期扰动。详细匹配差值见paired_comparison.csv和risk_impact.csv。","",
        "| 规则 | 费用/保证金 | 退出 | 停机比例 | 停机天数中位/最大 | 交易数中位 | 相比实际权益对照收益改善比例 | 回撤改善/恶化比例 |","|---|---|---|---:|---:|---:|---:|---:|"]
    p=impact[impact.window.eq("to_latest")&impact.variant.eq("mean5")]
    for r in p.itertuples():
        lines.append(f"| {LABELS[r.startup_mode]} | {r.fee*10000:.0f}bps/{r.margin:.0%} | 5日 | {pct(r.halted_fraction)} | {r.median_halt_days:.0f}/{r.max_halt_days:.0f} | {r.median_trades:.0f} | {pct(r.return_improved)} | {pct(r.drawdown_improved)}/{pct(r.drawdown_worsened)} |")
    lines += ["","## 全部退出与全部起点的最差持续账户","",
        "收益最差与回撤最差可能来自不同账户，覆盖原退出、亏损退出、5日、校准5日和尾部退出，不只看主展示模型。","",
        "| 情景 | 规则 | 最差收益 | 起点/退出 | 最差回撤 | 起点/退出 |","|---|---|---:|---|---:|---|"]
    for fee,margin in PROFILES:
        for mode in MODES:
            g=main[main.startup_mode.eq(mode)&main.fee.eq(fee)&main.margin.eq(margin)&main.window.eq("to_latest")]
            a=g.loc[g.total_return.idxmin()];b=g.loc[g.max_drawdown.idxmin()]
            lines.append(f"| {fee*10000:.0f}bps/{margin:.0%} | {LABELS[mode]} | {pct(a.total_return)} | {a.start}/{a.variant} | {pct(-b.max_drawdown)} | {b.start}/{b.variant} |")
    lines += ["","## 补保证金/恢复交易：另行记账的诊断情景","",
        "原冻结账户已空仓，没有持仓保证金缺口，因此当时为避免强平必须追加的金额是0。开新仓需满足合约保证金、开仓成本和策略储备，不能与追保混为一谈。实际追保金额和时限取决于期货公司账户及合同；例如[中银期货说明](https://www.bocifco.com.cn/newsinfo.aspx?cid=3&id=16105)规定结算后风险率超过100%应及时补足或平仓，盘中行情急变也可能要求即刻处理。不存在统一的可拖延天数。","",
        "下表是额外补一次、之后不再补也不提现的独立路径。early_block：首次权益低于40万元且有效入场因资金被拒时；frozen_state：已观察冻结账户的条件续跑，仅做诊断，不是选时规则。minimum_resume补到当天最便宜合格信号的储备线+1元；restore_500k补回50万元。采用原退出和实际权益历史重选，未加本轮新风控。补资会改变成交及未来选择。","",
        "补资不计收益：期末权益=原50万+补资+累计净损益。资金占用口径A为第一笔新增持仓的平仓日；口径B为首次观察到可立即平仓、退还全部补资后仍余50万。B只是无提现路径上的可行时点，不保证取出后后续策略仍成立，也不代表此后持续维持该水平；未达到的只报告观察下限。","",
        "| 个案 | 情景 | 补资日 | 补资(万元) | 首笔平仓/交易日数 | 净损益不含入金(万元) | 期末权益(万元) | 首次可退补资且余50万/自然日 |","|---|---|---|---:|---|---:|---:|---|"]
    top=pd.read_csv(OUT/"diagnosis/topups.csv",dtype={"date":str,"first_exit":str,"recovery_to_500k_date":str})
    for r in top.itertuples():
        recovered=f"{r.recovery_to_500k_date}/{r.recovery_calendar_days:.0f}" if pd.notna(r.recovery_to_500k_date) else f"未达到，观察{r.observed_calendar_days}天"
        lines.append(f"| {r.case} | {r.scenario} | {r.date} | {r.added_cash/10000:.2f} | {r.first_exit}/{r.first_trade_sessions} | {r.total_net_profit_excluding_deposit/10000:.2f} | {r.ending_equity/10000:.2f} | {recovered} |")
    lines += ["","补资场景不计借款利息或机会成本；如果资金有成本，净结果应继续扣减。一次补至储备线并不能保证下一天仍有足够储备，价格、波动及保证金比例都可能变化；日线也不覆盖盘中保证金缺口。","",
        "## 审计与限制","",json.dumps(checks,ensure_ascii=False,indent=2),"",
        "规则先固定，但历史已被反复研究，不能称为新的未触碰OOT；重叠起点不是独立样本。20%本金底线不是20%回撤硬上限，盈利后回吐和跳空都可能导致更大峰值回撤。高保证金情景属于假设，不是券商逐日实际费率。",
        "全部逐年、方向、日历扰动与完整窗口结果保存在CSV；预测数据和日历延续旧研究，不更新行情，以保持可比性。",""]
    if (OUT/"findings.md").exists():lines[3:3]=[(OUT/"findings.md").read_text(),""]
    (OUT/"report.md").write_text("\n".join(lines))
    charts(data)
    print(json.dumps(checks))
    print(dist[dist.variant.eq("mean5")&dist.fee.eq(.0006)&dist.margin.eq(.25)&dist.window.eq("sessions_244")].to_string(index=False))


if __name__=="__main__":main()
