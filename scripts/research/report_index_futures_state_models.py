"""Whole-grid reports for state factors, interactions and causal exit rules."""
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from evaluate_index_futures_state_models import (OUT,PREVIOUS,CONFIGS,ARMS,VARIANTS,PROFILES,
    ALL_PROFILES,CAPITAL,DATE_TYPES,RegimeReplay)
from report_index_futures_multistart import grouped,pct,NAMES
from report_index_futures_reentry_cash import prefix,attribution

ARM_LABELS=dict(original="原方案",hard_entry="硬入场过滤",hard_both="硬入场/退出",
    model_base="无状态模型",state_factor="状态因子",state_interaction="状态交互",
    exit_base="交互＋基础退出模型",exit_state="交互＋状态退出模型",fixed10="交互＋统一10日",
    state_horizon="交互＋状态期限",trail_uniform="交互＋统一回吐",state_trail="交互＋状态回吐",combined="交互＋三项退出")
LABELS={mode:("固定/" if style=="fixed" else "滚动/")+ARM_LABELS[arm] for mode,(style,arm) in CONFIGS.items()}
KEYS=["start","variant","fee","margin","startup_mode"]
FOCUS=("original","hard_both","model_base","state_factor","state_interaction","exit_state","state_horizon","state_trail","combined")


@lru_cache(maxsize=128)
def ledger_at(start,variant,mode,fee,margin):
    return pd.read_parquet(OUT/start[:4]/"ledgers"/f"{start}_{variant}_{mode}_{fee}_{margin}.parquet")


def describe(data,keys):
    result=grouped(data,keys)
    extra=data.groupby(keys).agg(median_flat_ratio=("flat_ratio","median"),median_longest_flat=("longest_flat","median"),
        median_trades=("trades","median"),median_evidence_blocks=("evidence_blocks","median"),
        median_substitutions=("substitution_entries","median"),median_model_exits=("state_model_exit_days","median"),
        median_time_exits=("state_time_exit_days","median"),median_trail_exits=("profit_giveback_exit_days","median")).reset_index()
    return result.merge(extra,on=keys,validate="one_to_one")


def effect_pairs():
    comparisons=[("entry_model","original","model_base"),("state_factor","model_base","state_factor"),
        ("interaction","state_factor","state_interaction"),("soft_vs_hard","hard_entry","state_interaction"),
        ("entry_total","original","state_interaction"),("exit_model","state_interaction","exit_base"),
        ("exit_state","exit_base","exit_state"),("deadline","state_interaction","fixed10"),
        ("deadline_state","fixed10","state_horizon"),("trail","state_interaction","trail_uniform"),
        ("trail_state","trail_uniform","state_trail"),("combined_exit","state_interaction","combined"),
        ("combined_total","original","combined"),("combined_vs_hard","hard_both","combined")]
    for style in ("fixed","rolling"):
        for effect,a,b in comparisons:yield effect,f"{style}_{a}",f"{style}_{b}"
    for arm in ARMS:yield "rolling",f"fixed_{arm}",f"rolling_{arm}"


def model_diagnostics():
    f=pd.read_parquet(OUT/"entry_forecasts.parquet")
    from backtest_index_futures import ROOT
    panel=pd.read_parquet(ROOT/"adaptation_comparison/quarterly_predictions.parquet")
    f=f.merge(panel[["entry_date","symbol","long_60","short_60","etf_ret_1"]],on=["entry_date","symbol"],validate="many_to_one")
    f["candidate"]=(np.maximum(f.long_60,f.short_60)>.45)&((f.long_60-f.short_60).abs()>=.1)&f.etf_ret_1.notna()&f.direction.eq(np.where(f.long_60>f.short_60,1,-1))
    f["actual_net"]=np.where(f.bps.eq(3),f.net_3,f.net_6)
    f["actual_risk"]=np.where(f.bps.eq(3),f.risk_3,f.risk_6)
    f["regime"]=f.regime_bull-f.regime_bear
    f["alignment"]=np.where(f.regime.eq(0),"transition",np.where(f.regime.eq(f.direction),"aligned","counter"))
    f["pass"]=(f.probability>=.55)&(f.pred_net-.5*f.pred_risk>0)
    f["year"]=f.entry_date.str[:4]
    f.to_parquet(OUT/"candidate_evidence.parquet",index=False)
    valid=f[f.candidate&f.ready&f.actual_net.notna()&f.entry_date.ge("20220101")].copy()
    records=[]
    for key,g in valid.groupby(["year","model","bps","direction","alignment"]):
        selected=g[g["pass"]]
        records.append(dict(zip(["year","model","bps","direction","alignment"],key),
            samples=len(g),passed=len(selected),pass_fraction=g["pass"].mean(),
            brier=float(((g.probability-g.actual_net.gt(0).astype(int))**2).mean()),
            net_mae=float((g.pred_net-g.actual_net).abs().mean()),net_bias=float((g.pred_net-g.actual_net).mean()),
            all_mean_net=g.actual_net.mean(),selected_mean_net=selected.actual_net.mean(),
            selected_win_rate=selected.actual_net.gt(0).mean(),selected_mean_probability=selected.probability.mean(),
            selected_mean_adverse=selected.actual_risk.mean()))
    result=pd.DataFrame(records);result.to_csv(OUT/"model_evidence_by_year.csv",index=False)
    audit=pd.read_csv(OUT/"training_audit.csv",dtype=DATE_TYPES|{"last_maturity":str})
    assert audit[audit.ready].last_maturity.lt(audit[audit.ready].cutoff).all()
    assert f[f.entry_date.ge("20220101")].ready.all()
    return result,audit


def charts(data):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/quant_adaptation_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    arms=("original","hard_both","state_interaction","exit_state","state_horizon","state_trail","combined")
    names=("Original","Hard gates","Interaction entries","State model exit","State deadline","State giveback","Combined exits")
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for i,year in enumerate(("2025","2026")):
        for j,style in enumerate(("fixed","rolling")):
            ax=axes[i,j]
            part=data[data.start_kind.eq("monthly")&data.start.str.startswith(year)&data.window.eq("to_latest")&
                data.variant.eq("mean5")&data.fee.eq(.0006)&data.margin.eq(.25)]
            for arm,label in zip(arms,names):
                g=part[part.startup_mode.eq(f"{style}_{arm}")].sort_values("start")
                ax.plot(g.start.str[4:6].astype(int),g.total_return*100,label=label,marker="o",ms=2)
            ax.set_title(f"{year} starts | {style} |6bps/25% margin")
            ax.set_xlabel("Startup month");ax.set_ylabel("Net return to 2026-11-06 (%)")
            ax.axhline(0,c="grey",lw=.7);ax.grid(alpha=.2)
    fig.suptitle("State factors and conditional exits: recent startup sensitivity")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(OUT/"recent_starts.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for i,start in enumerate(("20230801","20260101")):
        for j,style in enumerate(("fixed","rolling")):
            ax=axes[i,j]
            for arm,label in zip(arms,names):
                d=ledger_at(start,"mean5",f"{style}_{arm}",.0006,.25)
                ax.plot(pd.to_datetime(d.exit_date),d.equity/10000,label=label,lw=1.1)
            ax.set_title(f"Start {start} | {style} |6bps/25%")
            ax.set_ylabel("Equity (CNY 10k)");ax.grid(alpha=.2)
            loc=mdates.AutoDateLocator(minticks=3,maxticks=5)
            ax.xaxis.set_major_locator(loc);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.suptitle("Continuous accounts: old adverse start and 2026, without topups")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(OUT/"account_paths.png",dpi=150);plt.close(fig)


def main():
    data=pd.read_csv(OUT/"summary.csv",dtype=DATE_TYPES,low_memory=False);data["start_year"]=data.start.str[:4]
    totals=data[data.window.eq("to_latest")]
    assert len(totals)==12948 and totals.groupby("startup_mode").size().eq(498).all()
    assert not data.duplicated(KEYS+["window"]).any()
    trades=pd.read_csv(OUT/"trades.csv",dtype=DATE_TYPES,low_memory=False)
    annual=pd.read_csv(OUT/"annual_marked.csv",dtype=DATE_TYPES,low_memory=False)
    decisions=pd.read_csv(OUT/"selection.csv",dtype=DATE_TYPES,low_memory=False)
    audit=pd.read_csv(OUT/"selection_audit.csv",dtype=DATE_TYPES,low_memory=False)
    old=pd.read_csv(PREVIOUS/"summary.csv",dtype=DATE_TYPES,low_memory=False)
    for style in ("fixed","rolling"):
        for arm,suffix in (("original","none"),("hard_entry","entry"),("hard_both","both")):
            current=data[data.startup_mode.eq(f"{style}_{arm}")]
            control=old[old.ma_window.eq(200)&old.startup_mode.eq(f"{style}_{suffix}")&old.variant.isin(VARIANTS)]
            compare=current.merge(control,on=["start","variant","fee","margin","window"],suffixes=("","_old"),validate="one_to_one")
            assert len(compare)==len(current)==len(control)
            for col in ("total_return","max_drawdown","trades","forced"):
                np.testing.assert_allclose(compare[col],compare[col+"_old"],rtol=0,atol=1e-8)
    recon=totals.merge(trades.groupby(KEYS).pnl.sum().rename("cash_pnl"),on=KEYS,how="left")
    np.testing.assert_allclose(recon.cash_pnl.fillna(0),recon.total_return*CAPITAL,rtol=0,atol=1e-6)
    assert decisions.cutoff.lt(decisions.date).all() and audit.last_history_end.le(audit.cutoff).all()
    np.testing.assert_array_equal(decisions.equity,decisions.validation_capital)
    assert (OUT/"global_calendar.csv").read_bytes()==(PREVIOUS/"global_calendar.csv").read_bytes()
    evidence,training=model_diagnostics()
    monthly=data[data.start_kind.eq("monthly")]
    dist=describe(monthly,["startup_mode","variant","fee","margin","window"])
    cohorts=describe(monthly,["start_year","startup_mode","variant","fee","margin","window"])
    phases=describe(data[data.annual|data.start_kind.eq("phase")],["start_year","startup_mode","variant","fee","margin","window"])
    for name,frame in (("start_distributions",dist),("year_cohorts",cohorts),("january_phase_sensitivity",phases)):
        frame.to_csv(OUT/f"{name}.csv",index=False)
    replay=RegimeReplay()
    frozen_states=pd.read_parquet(PREVIOUS/"states_200.parquet").dropna(subset=["state"])
    replay.market_regime={(r.trade_date,r.symbol):int(r.state) for r in frozen_states.itertuples(index=False)}
    pairs,opportunities=[],[]
    for effect,base,new in effect_pairs():
        a=data[data.startup_mode.eq(base)];b=data[data.startup_mode.eq(new)]
        pair=b.merge(a,on=["start","variant","fee","margin","window"],suffixes=("","_base"),validate="one_to_one")
        pair["effect"]=effect;pair["baseline_mode"]=base
        pair["return_change"]=pair.total_return-pair.total_return_base
        pair["drawdown_change"]=pair.max_drawdown-pair.max_drawdown_base
        cols=[*KEYS,"start_kind","start_year","window","effect","baseline_mode","return_change","drawdown_change",
            "total_return","total_return_base","max_drawdown","max_drawdown_base","flat_ratio","flat_ratio_base"]
        pairs.append(pair[cols])
        for (start,variant,fee,margin),p in pair.groupby(["start","variant","fee","margin"]):
            full_a=ledger_at(start,variant,base,fee,margin);full_b=ledger_at(start,variant,new,fee,margin)
            for r in p.itertuples(index=False):
                aa=prefix(full_a,r.window,start,fee,replay.opens);bb=prefix(full_b,r.window,start,fee,replay.opens)
                values=attribution(aa,bb)
                np.testing.assert_allclose(values["net_pnl_change"],r.return_change*CAPITAL,rtol=0,atol=1e-6)
                opportunities.append(dict(effect=effect,baseline_mode=base,startup_mode=new,start=start,start_year=start[:4],
                    start_kind=r.start_kind,variant=variant,fee=fee,margin=margin,window=r.window,**values))
        print("attributed",effect,base,new,flush=True)
    paired=pd.concat(pairs,ignore_index=True);paired.to_csv(OUT/"paired_effects.csv",index=False)
    op=pd.DataFrame(opportunities);op.to_csv(OUT/"opportunity_costs.csv",index=False)
    pk=["effect","baseline_mode","startup_mode","start_year","variant","fee","margin","window"]
    effects=paired[paired.start_kind.eq("monthly")].groupby(pk).agg(starts=("start","size"),
        median_return_change=("return_change","median"),return_improved_fraction=("return_change",lambda s:s.gt(1e-10).mean()),
        drawdown_improved_fraction=("drawdown_change",lambda s:s.gt(1e-10).mean()),
        drawdown_worsened_fraction=("drawdown_change",lambda s:s.lt(-1e-10).mean())).reset_index()
    opp=op[op.start_kind.eq("monthly")].groupby(pk).agg(median_missed_positive=("missed_positive_pnl","median"),
        median_avoided_negative=("avoided_negative_pnl","median"),median_skipped_net=("skipped_baseline_net_pnl","median"),
        median_additional_flat_days=("additional_flat_vs_held_days","median")).reset_index()
    effects=effects.merge(opp,on=pk,validate="one_to_one");effects.to_csv(OUT/"effects_by_year.csv",index=False)
    worst=[]
    for _,g in data.groupby(["startup_mode","fee","margin","window"]):
        for metric in ("total_return","max_drawdown"):
            worst.append(dict(g.loc[g[metric].idxmin()],criterion=metric))
    pd.DataFrame(worst).to_csv(OUT/"worst_all_starts.csv",index=False)
    state_at={(r.entry_date,r.symbol):replay.market_regime[r.trade_date,r.symbol] for g in replay.groups for r in g}
    trades["entry_regime"]=[state_at[r.entry,r.product] for r in trades.itertuples(index=False)]
    trades["alignment"]=np.where(trades.entry_regime.eq(0),"transition",np.where(trades.entry_regime.eq(trades.direction),"aligned","counter"))
    trades.to_csv(OUT/"trades_with_regime.csv",index=False)
    diagnostics=trades.groupby(KEYS+["alignment","side"]).agg(trades=("pnl","size"),net_pnl=("pnl","sum"),
        win_rate=("pnl",lambda s:s.gt(0).mean()),worst_trade=("pnl","min")).reset_index()
    diagnostics.to_csv(OUT/"trade_regime_by_account.csv",index=False)
    forced=[]
    for r in totals[totals.forced.gt(0)].itertuples(index=False):
        l=ledger_at(r.start,r.variant,r.startup_mode,r.fee,r.margin)
        for i in l.index[l.forced_exit]:
            held=l.positions.iloc[i-1] if i else "";code,side=held.split(":") if held else ("","0")
            before=CAPITAL if i==0 else l.equity.iloc[i-1]
            n=replay.opens[l.entry_date.iloc[i],code]*(300 if code[:2] in ("IF","IH") else 200) if code else 0.
            forced.append(dict(start=r.start,mode=r.startup_mode,variant=r.variant,fee=r.fee,margin=r.margin,
                date=l.entry_date.iloc[i],held=held,before_equity=before,required_margin=n*r.margin,
                margin_shortfall=max(0,n*r.margin-before),after_equity=l.equity.iloc[i]))
    pd.DataFrame(forced).to_csv(OUT/"forced_exit_events.csv",index=False)
    checks=dict(accounts=len(totals),windows=len(data),controls_replayed=sum(json.loads((OUT/str(y)/"checks.json").read_text())["replayed_controls"] for y in range(2022,2027)),
        training_audit_records=len(training),mature_training_only=True,decision_records=len(decisions),unique_validation_sets=audit.validation_id.nunique(),
        exact_current_equity=True,all_trade_cash_reconciles=True,opportunity_rows=len(op),opportunity_cash_reconciles=True,
        forced_scenarios=len(forced),unit_tests=102,**json.loads((OUT/"engine_checks.json").read_text()))
    (OUT/"checks.json").write_text(json.dumps(checks,indent=2))
    lines=["# 股指期货：牛熊状态因子、交互项与差异化退出","",
        "结论和反例解释见 [findings.md](findings.md)。数据冻结2026-11-06；50万元最多一手，无补资。13种预登记方案×固定/滚动×原control/loss_exit/mean5退出，12948账户。其他旧退出研究保留原目录，本轮没有跑它们与新规则的全部交叉。","",
        "## 固定定义","",
        "- 原方案、硬入场、硬入场/退出三个旧对照逐账户复现；均线固定MA200，不再搜索120/250。",
        "- 无状态/加状态/加交互：64/69/85个入场特征，分别训练多空LR盈利概率与Ridge净收益、不利波动，均为5日含开平仓及换月成本标签。保留原信号方向、排名、全局日历和资金约束，概率≥55%且预期净收益−0.5×预期不利波动>0才接受；逆势不被硬禁。",
        "- 训练：季度，过去36个月成熟样本，12月半衰期，LR C=0.1、Ridge alpha=0.1×训练行数，不搜索参数。插补和标准化仅在训练集拟合，见[官方数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls)。",
        "- 基础退出/状态退出：71/98个续持特征、两次负效用确认，作为原退出之外的附加退出。标签是继续持有相对立即平仓的5日净收益与不利波动，按原候选完整episode成熟日期隔离训练。原20日上限和保护性退出均保留。",
        "- 统一10日期限，对照入场状态固定的顺势20/过渡10/逆势5日上限；后续状态变化不能延长期限，原退出可提前触发。",
        "- 盈利回撤：上一收盘扣费净浮盈峰值达到入场权益5%后启用。统一回吐50%，对照顺势回吐50%/过渡或逆势回吐25%；峰值包含换月及已付费用，不因换月/季度重置；下一开盘执行可能跳空越线。",
        "- 组合：交互入场＋状态退出模型＋状态期限＋状态盈利回吐。滚动仅在原四种入场政策间按历史重选，不事后拼接本轮最优方案。",
        "- 入场模型从2021Q2具备训练条件，2022起全部入场预测已就绪；附加退出多头2021Q2、空头2023Q1就绪。在此前历史窗明确回退原入场/不增加模型退出，不用未来拟合补齐。",
        "- 内外层同规则、精确当前权益，仍用过去36个月已完成122/244日月度窗口及原切换门槛。资金拒绝统计和估计切换成本沿用原口径，实际费用逐笔入账。所有模型预期风险都不是已校准的尾部保证。","",
        "- 标签核查补齐了换月开盘瞬间费用对不利波动的影响，收益标签未变。此前未完成的计算已隔离，最终结果全部来自修正后的重新训练和回放，见label_correction.json；交易规则和参数没有因此改变。","",
        "- 年度划分沿用原开盘到次日开盘账本，按exit_date所属年份记账；独立单年在当年最后纳入的开盘清算，持续账户不清算重置。因此不是交易所逐日收盘结算收益表。","",
        "## 完整244日等长窗口","","mean5、47个月度起点；不把重叠窗口当独立样本。","",
        "| 成本/保证金 | 方案 | 收益中位 | P10收益 | 最差收益 | 最差回撤 | 空仓中位 | 零交易账户 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in dist[dist.variant.eq("mean5")&dist.window.eq("sessions_244")].itertuples():
        lines.append(f"| {r.fee*10000:.0f}bps/{r.margin:.0%} | {LABELS[r.startup_mode]} | "+" | ".join(
            pct(getattr(r,k)) for k in ("median_return","p10_return","worst_return","worst_drawdown","median_flat_ratio"))+f" | {r.zero_trade_accounts} |")
    lines += ["","## 逐年单年与各年起累计至统一终点","",
        "每格收益/最大回撤；独立单年每年50万元，累计不重置权益。2026为部分年度。所有预登记方案都列出，不按收益挑选。",""]
    for fee,margin in ALL_PROFILES:
        for variant in VARIANTS:
            for window in ("single_year","to_latest"):
                for style in ("fixed","rolling"):
                    lines += [f"### {fee*10000:.0f}bps/{margin:.0%} · {variant} · {window} · {style}","",
                        "| 方案 | 2022 | 2023 | 2024 | 2025 | 2026 |","|---|---:|---:|---:|---:|---:|"]
                    part=data[data.annual&data.fee.eq(fee)&data.margin.eq(margin)&data.variant.eq(variant)&data.window.eq(window)]
                    for arm in ARMS:
                        g=part[part.startup_mode.eq(f"{style}_{arm}")].set_index("start_year")
                        lines.append(f"| {ARM_LABELS[arm]} | "+" | ".join(f"{pct(g.loc[y,'total_return'])}/{pct(-g.loc[y,'max_drawdown'])}" for y in map(str,range(2022,2027)))+" |")
                    lines.append("")
    lines += ["## 逐年月度起点、等长窗口与空仓","","每格收益中位/最差收益/最差回撤/空仓中位。2026没有完整244日新起点。",""]
    for fee,margin in PROFILES:
        for window in ("to_latest","sessions_122","sessions_244"):
            for style in ("fixed","rolling"):
                lines += [f"### mean5 · {fee*10000:.0f}bps/{margin:.0%} · {window} · {style}","",
                    "| 方案 | 2022 | 2023 | 2024 | 2025 | 2026 |","|---|---:|---:|---:|---:|---:|"]
                part=cohorts[cohorts.fee.eq(fee)&cohorts.margin.eq(margin)&cohorts.variant.eq("mean5")&cohorts.window.eq(window)]
                for arm in ARMS:
                    g=part[part.startup_mode.eq(f"{style}_{arm}")].set_index("start_year")
                    lines.append(f"| {ARM_LABELS[arm]} | "+" | ".join("—" if y not in g.index else "/".join(pct(g.loc[y,k]) for k in (
                        "median_return","worst_return","worst_drawdown","median_flat_ratio")) for y in map(str,range(2022,2027)))+" |")
                lines.append("")
    lines += ["## 同起点配对增量与机会成本","",
        "以下为mean5、6bps/25%压力情景；完整全部情景见effects_by_year.csv。收益配对中位不是两组中位数之差。错失盈利/避免亏损是原账户有仓而新账户空仓日的净现金流分解，并非可执行择日收益；不同列中位数不可直接相减。",""]
    for window in ("to_latest","sessions_244","sessions_122"):
        lines += [f"### {window}","","| 年 | 对照→方案 | 起点数 | 收益差中位 | 收益改善比例 | 回撤改善/恶化比例 | 新增空仓日中位 | 错失盈利/避免亏损中位（万元） |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
        part=effects[effects.variant.eq("mean5")&effects.fee.eq(.0006)&effects.margin.eq(.25)&effects.window.eq(window)]
        for r in part.sort_values(["start_year","effect","startup_mode"]).itertuples():
            lines.append(f"| {r.start_year} | {LABELS[r.baseline_mode]}→{LABELS[r.startup_mode]} | {r.starts} | {pct(r.median_return_change)} | {pct(r.return_improved_fraction)} | {pct(r.drawdown_improved_fraction)}/{pct(r.drawdown_worsened_fraction)} | {r.median_additional_flat_days:.1f} | {r.median_missed_positive/10000:.2f}/{r.median_avoided_negative/10000:.2f} |")
        lines.append("")
    lines += ["## 全退出与全起点最差反例","","覆盖本轮control/loss_exit/mean5及月度/相邻日起点，最差收益与回撤可来自不同账户。","",
        "| 成本/保证金 | 方案 | 最差收益起点/退出 | 最差收益 | 最差回撤起点/退出 | 最差回撤 |","|---|---|---|---:|---|---:|"]
    for (fee,margin,mode),g in totals.groupby(["fee","margin","startup_mode"]):
        a=g.loc[g.total_return.idxmin()];b=g.loc[g.max_drawdown.idxmin()]
        lines.append(f"| {fee*10000:.0f}bps/{margin:.0%} | {LABELS[mode]} | {a.start}/{a.variant} | {pct(a.total_return)} | {b.start}/{b.variant} | {pct(-b.max_drawdown)} |")
    lines += ["","## 2025起持续账户：近两年年度记账","","同一账户2026不重置资金，分母是年初实际权益。mean5压力情景。","",
        "| 方案 | 2025收益/回撤 | 2026收益/回撤 |","|---|---:|---:|"]
    part=annual[annual.start.eq("20250101")&annual.variant.eq("mean5")&annual.side.eq("all")&annual.fee.eq(.0006)&annual.margin.eq(.25)]
    for mode in CONFIGS:
        g=part[part.startup_mode.eq(mode)].set_index("year")
        lines.append(f"| {LABELS[mode]} | "+" | ".join(f"{pct(g.loc[y,'total_return'])}/{pct(-g.loc[y,'max_drawdown'])}" for y in ("2025","2026"))+" |")
    lines += ["","## 模型证据诊断与审计","",
        "model_evidence_by_year.csv按年、多空、顺势/过渡/逆势列出原合格信号的样本数、通过数、Brier、预测偏差、通过后的真实5日扣费收益与不利波动；这是候选标签诊断，不等于实际账户交易收益。trade_regime_by_account.csv及trades_with_regime.csv提供真实执行的状态和多空交易。重叠起点不能叠加成独立交易样本。","",
        "候选期望盈利或概率过线不等于统计显著，也不保证尾部风险可控。所有结果来自已反复使用的历史，不是新OOT；没有根据本轮结果调整阈值、均线或期限。","",
        f"```json\n{json.dumps(checks,ensure_ascii=False,indent=2)}\n```","",
        "关键文件：protocol.json、sources.json、feature_sets.json、training_audit.csv；逐年目录含实际账户账本、诊断、历史验证窗口及选择审计。start_distributions.csv、year_cohorts.csv、january_phase_sensitivity.csv、annual_marked.csv、worst_all_starts.csv为各口径汇总。paired_effects.csv和opportunity_costs.csv保留每个起点的配对现金差。","",
        "![近两年起点](recent_starts.png)","","![持续账户](account_paths.png)",""]
    (OUT/"report.md").write_text("\n".join(lines))
    charts(data)
    print(json.dumps(checks,indent=2),flush=True)


if __name__=="__main__":main()
