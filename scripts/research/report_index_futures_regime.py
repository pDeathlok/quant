"""Whole-grid regime ablations, matched effects and idle opportunity costs."""
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from evaluate_index_futures_regime import (OUT, PREVIOUS, CONFIGS, MA_WINDOWS, MODES,
    CAPITAL, DATE_TYPES, PROFILES, ALL_PROFILES, VARIANTS, RegimeReplay)
from report_index_futures_multistart import grouped, pct, NAMES
from report_index_futures_reentry_cash import prefix, attribution

LABELS={f"{style}_{mode}":("固定" if style=="fixed" else "滚动")+"/"+
        dict(none="原方案",entry="入场状态",exit="退出状态",both="两者结合")[mode]
        for style,mode in CONFIGS.values()}
KEYS=["start","variant","fee","margin","startup_mode","ma_window"]


@lru_cache(maxsize=64)
def ledger_at(start,variant,mode,fee,margin,ma_window=200):
    return pd.read_parquet(OUT/start[:4]/"ledgers"/
                          f"{start}_{variant}_{mode}_{fee}_{margin}_ma{ma_window}.parquet")


def describe(data,keys):
    base=grouped(data,keys)
    extra=data.groupby(keys).agg(median_flat_ratio=("flat_ratio","median"),
        median_longest_flat=("longest_flat","median"),median_trades=("trades","median"),
        median_regime_exits=("regime_exits","median"),median_regime_blocks=("regime_blocks","median"),
        median_regime_rejected_days=("regime_rejected_days","median"),
        median_substitution_entries=("substitution_entries","median")).reset_index()
    return base.merge(extra,on=keys,validate="one_to_one")


def effect_pairs():
    for style in ("fixed","rolling"):
        for mode in MODES[1:]:
            yield "state_overlay",f"{style}_none",f"{style}_{mode}"
    for mode in MODES:
        yield "rolling",f"fixed_{mode}",f"rolling_{mode}"


def charts(data):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/quant_adaptation_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    names=["Original","Entry gate","Exit gate","Both"]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for i,year in enumerate(("2025","2026")):
        for j,style in enumerate(("fixed","rolling")):
            ax=axes[i,j]
            part=data[data.start_kind.eq("monthly")&data.start.str.startswith(year)&
                data.window.eq("to_latest")&data.variant.eq("mean5")&data.fee.eq(.0006)&data.margin.eq(.25)&data.ma_window.eq(200)]
            for mode,label in zip(MODES,names):
                g=part[part.startup_mode.eq(f"{style}_{mode}")].sort_values("start")
                ax.plot(g.start.str[4:6].astype(int),100*g.total_return,label=label,marker="o",ms=3)
            ax.set_title(f"{year} starts | {style} | MA200 |6bps/25% margin")
            ax.set_xlabel("Startup month");ax.set_ylabel("Net return to 2026-11-06 (%)")
            ax.axhline(0,c="grey",lw=.8);ax.grid(alpha=.2)
    fig.suptitle("Spot-index regime gates: recent startup sensitivity",fontsize=14)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(OUT/"recent_starts.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for i,start in enumerate(("20230801","20260101")):
        for j,style in enumerate(("fixed","rolling")):
            ax=axes[i,j]
            for mode,label in zip(MODES,names):
                l=ledger_at(start,"mean5",f"{style}_{mode}",.0006,.25)
                ax.plot(pd.to_datetime(l.exit_date),l.equity/10000,label=label,lw=1.3)
            ax.set_title(f"Start {start} | {style} | MA200 |6bps/25%")
            ax.set_ylabel("Equity (CNY 10k)");ax.grid(alpha=.2)
            locator=mdates.AutoDateLocator(minticks=3,maxticks=5)
            ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle("Continuous equity without topups: old adverse start and 2026",fontsize=14)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(OUT/"account_paths.png",dpi=160);plt.close(fig)


def performance_table(lines,title,part,modes):
    lines.extend([title,"","| 起始年 | "+" | ".join(LABELS[m] for m in modes)+" |",
                  "|---|"+"---:|"*len(modes)])
    for year,g in part.groupby("start_year"):
        g=g.set_index("startup_mode")
        lines.append(f"| {year} | "+" | ".join(f"{pct(g.loc[m,'total_return'])} / {pct(-g.loc[m,'max_drawdown'])}" for m in modes)+" |")
    lines.append("")


def main():
    data=pd.read_csv(OUT/"summary.csv",dtype=DATE_TYPES,low_memory=False)
    data["start_year"]=data.start.str[:4]
    totals=data[data.window.eq("to_latest")]
    assert not data.duplicated(KEYS+["window"]).any()
    assert totals[totals.ma_window.eq(200)].groupby("startup_mode").size().eq(750).all()
    assert len(totals)==7992
    old=pd.read_csv(PREVIOUS/"summary.csv",dtype=DATE_TYPES,low_memory=False)
    for mode,oldmode in (("fixed_none","fixed_none"),("rolling_none","actual_equity")):
        compare=data[data.startup_mode.eq(mode)].merge(old[old.startup_mode.eq(oldmode)],
            on=["start","variant","fee","margin","window"],suffixes=("","_old"),validate="one_to_one")
        assert len(compare)==len(data[data.startup_mode.eq(mode)])
        for col in ("total_return","max_drawdown","trades","forced"):
            np.testing.assert_allclose(compare[col],compare[col+"_old"],atol=1e-8,rtol=0)
    trades=pd.read_csv(OUT/"trades.csv",dtype=DATE_TYPES,low_memory=False)
    decisions=pd.read_csv(OUT/"selection.csv",dtype=DATE_TYPES,low_memory=False)
    audit=pd.read_csv(OUT/"selection_audit.csv",dtype=DATE_TYPES,low_memory=False)
    annual=pd.read_csv(OUT/"annual_marked.csv",dtype=DATE_TYPES,low_memory=False)
    recon=totals.merge(trades.groupby(KEYS).pnl.sum().rename("cash_pnl"),on=KEYS,how="left")
    np.testing.assert_allclose(recon.cash_pnl.fillna(0),recon.total_return*CAPITAL,atol=1e-6,rtol=0)
    assert decisions.cutoff.lt(decisions.date).all() and audit.last_history_end.le(audit.cutoff).all()
    np.testing.assert_array_equal(decisions.equity,decisions.validation_capital)
    assert (OUT/"global_calendar.csv").read_bytes()==(PREVIOUS/"global_calendar.csv").read_bytes()
    monthly=data[data.start_kind.eq("monthly")]
    dist=describe(monthly,["ma_window","startup_mode","variant","fee","margin","window"])
    cohorts=describe(monthly,["start_year","ma_window","startup_mode","variant","fee","margin","window"])
    phase=describe(data[data.annual|data.start_kind.eq("phase")],
        ["start_year","ma_window","startup_mode","variant","fee","margin","window"])
    for name,frame in (("start_distributions",dist),("year_cohorts",cohorts),("january_phase_sensitivity",phase)):
        frame.to_csv(OUT/f"{name}.csv",index=False)
    replay=RegimeReplay()
    paired,opportunity,divergences=[],[],[]
    adverse=totals[totals.ma_window.eq(200)&totals.variant.eq("mean5")&totals.fee.eq(.0006)&totals.margin.eq(.25)]
    diagnostic_starts={"20230801","20250101","20260101"}
    diagnostic_starts.update(adverse.loc[adverse.groupby("startup_mode").total_return.idxmin(),"start"])
    join=["start","variant","fee","margin","window"]
    for ma in MA_WINDOWS:
        for effect,base,new in effect_pairs():
            if ma!=200 and new.endswith("none"):
                continue
            base_ma=200 if base.endswith("none") else ma
            a=data[data.startup_mode.eq(base)&data.ma_window.eq(base_ma)]
            b=data[data.startup_mode.eq(new)&data.ma_window.eq(ma)]
            pair=b.merge(a,on=join,suffixes=("","_base"),validate="one_to_one")
            pair["effect"]=effect;pair["baseline_mode"]=base
            pair["return_change"]=pair.total_return-pair.total_return_base
            pair["drawdown_change"]=pair.max_drawdown-pair.max_drawdown_base
            paired.append(pair)
            for (start,variant,fee,margin),p in pair.groupby(["start","variant","fee","margin"]):
                full_a=ledger_at(start,variant,base,fee,margin,base_ma)
                full_b=ledger_at(start,variant,new,fee,margin,ma)
                if ma==200 and variant=="mean5" and start in diagnostic_starts:
                    different=full_a.positions.ne(full_b.positions)
                    if different.any():
                        i=different.idxmax()
                        traces=[]
                        for m in (base,new):
                            traces.append(pd.read_parquet(OUT/start[:4]/"traces"/
                                f"{start}_{variant}_{m}_{fee}_{margin}_ma200.parquet"))
                        ta,tb=traces[0].iloc[i],traces[1].iloc[i]
                        states={symbol:replay.market_regime[ta.signal_date,symbol]
                                for symbol in ("IF","IH","IC","IM")}
                        divergences.append(dict(effect=effect,baseline_mode=base,startup_mode=new,start=start,
                            fee=fee,margin=margin,date=full_a.entry_date.iloc[i],signal_date=ta.signal_date,
                            baseline_position=full_a.positions.iloc[i],new_position=full_b.positions.iloc[i],
                            baseline_action=ta.action,new_action=tb.action,baseline_policy=ta.policy,new_policy=tb.policy,
                            before_equity=CAPITAL if i==0 else full_a.equity.iloc[i-1],
                            states=json.dumps(states),total_pnl_change=full_b.pnl.sum()-full_a.pnl.sum()))
                for r in p.itertuples(index=False):
                    aa=prefix(full_a,r.window,start,fee,replay.opens)
                    bb=prefix(full_b,r.window,start,fee,replay.opens)
                    values=attribution(aa,bb)
                    np.testing.assert_allclose(values["net_pnl_change"],r.return_change*CAPITAL,atol=1e-6,rtol=0)
                    opportunity.append(dict(effect=effect,baseline_mode=base,startup_mode=new,ma_window=ma,
                        start=start,start_year=start[:4],start_kind=r.start_kind,variant=variant,fee=fee,
                        margin=margin,window=r.window,**values))
            print("attributed",ma,effect,base,new,flush=True)
    paired=pd.concat(paired,ignore_index=True)
    pd.DataFrame(divergences).to_csv(OUT/"first_divergences.csv",index=False)
    paired.to_csv(OUT/"paired_effects.csv",index=False)
    op=pd.DataFrame(opportunity);op.to_csv(OUT/"opportunity_costs.csv",index=False)
    pairkeys=["effect","baseline_mode","startup_mode","ma_window","start_year","variant","fee","margin","window"]
    effects=paired[paired.start_kind.eq("monthly")].groupby(pairkeys).agg(starts=("start","size"),
        median_return_change=("return_change","median"),return_improved_fraction=("return_change",lambda s:s.gt(1e-10).mean()),
        drawdown_improved_fraction=("drawdown_change",lambda s:s.gt(1e-10).mean()),
        drawdown_worsened_fraction=("drawdown_change",lambda s:s.lt(-1e-10).mean())).reset_index()
    opp=op[op.start_kind.eq("monthly")].groupby(pairkeys).agg(
        median_missed_positive=("missed_positive_pnl","median"),median_avoided_negative=("avoided_negative_pnl","median"),
        median_skipped_net=("skipped_baseline_net_pnl","median"),median_other_path_change=("other_path_change","median"),
        median_additional_flat_days=("additional_flat_vs_held_days","median")).reset_index()
    effects=effects.merge(opp,on=pairkeys,validate="one_to_one")
    effects.to_csv(OUT/"effects_by_year.csv",index=False)
    worst=[]
    for _,g in data.groupby(["ma_window","startup_mode","fee","margin","window"]):
        for metric in ("total_return","max_drawdown"):
            worst.append(dict(g.loc[g[metric].idxmin()],criterion=metric))
    pd.DataFrame(worst).to_csv(OUT/"worst_all_starts.csv",index=False)
    # Entry states and completed-episode PnL: overlapping accounts are never independent trades.
    execution_state={}
    statecounts=[]
    for ma in MA_WINDOWS:
        s=pd.read_parquet(OUT/f"states_{ma}.parquet").dropna(subset=["state"])
        lookup={(r.trade_date,r.symbol):int(r.state) for r in s.itertuples(index=False)}
        for g in replay.groups:
            for r in g:
                execution_state[ma,r.entry_date,r.symbol]=lookup[r.trade_date,r.symbol]
        s["transition"]=s.groupby("symbol").state.diff().fillna(0).ne(0)
        s=s[s.trade_date.ge("20220101")].copy()
        s["year"]=s.trade_date.str[:4]
        statecounts.append(s.groupby(["year","symbol","state"]).agg(days=("state","size"),
            arrivals=("transition","sum")).reset_index().assign(ma_window=ma))
    pd.concat(statecounts).to_csv(OUT/"state_coverage.csv",index=False)
    trades["entry_regime"]=[execution_state[r.ma_window,r.entry,r.product] for r in trades.itertuples(index=False)]
    trades.to_csv(OUT/"trades_with_regime.csv",index=False)
    annual_trades=trades[trades.annual&trades.entry.str[:4].eq(trades.start.str[:4])].copy()
    # Annual-account trade exits can extend beyond Dec31: separate from year-end liquidated account returns.
    annual_trades["entry_year"]=annual_trades.entry.str[:4]
    byside=annual_trades.groupby(["ma_window","startup_mode","variant","fee","margin","entry_year","entry_regime","side"]).agg(
        trades=("pnl","size"),net_pnl=("pnl","sum"),win_rate=("pnl",lambda s:s.gt(0).mean()),worst_trade=("pnl","min")).reset_index()
    byside.to_csv(OUT/"entry_regime_side_diagnostics.csv",index=False)
    forced=[]
    for r in totals[totals.forced.gt(0)].itertuples(index=False):
        l=ledger_at(r.start,r.variant,r.startup_mode,r.fee,r.margin,r.ma_window)
        for i in l.index[l.forced_exit]:
            held=l.positions.iloc[i-1] if i else ""
            code,sign=held.split(":") if held else ("","0")
            n=replay.opens[l.entry_date.iloc[i],code]*(300 if code[:2] in ("IF","IH") else 200) if code else 0
            before=CAPITAL if i==0 else l.equity.iloc[i-1]
            forced.append(dict(start=r.start,variant=r.variant,mode=r.startup_mode,fee=r.fee,margin=r.margin,
                ma_window=r.ma_window,date=l.entry_date.iloc[i],held=held,before_equity=before,
                required_margin=n*r.margin,margin_shortfall=max(0,n*r.margin-before),after_equity=l.equity.iloc[i]))
    pd.DataFrame(forced).to_csv(OUT/"forced_exit_events.csv",index=False)
    checks=dict(accounts=len(totals),primary_accounts=int(totals.ma_window.eq(200).sum()),windows=len(data),
        all_controls_replayed=sum(json.loads((OUT/str(y)/"checks.json").read_text())["replayed_controls"] for y in range(2022,2027)),
        old_studies_retained_in_place=True,decision_records=len(decisions),unique_validation_sets=audit.validation_id.nunique(),
        exact_current_equity=True,mature_history_only=True,all_trade_cash_reconciles=True,
        opportunity_rows=len(op),opportunity_cash_reconciles=True,forced_scenarios=len(forced),unit_tests=92,
        **json.loads((OUT/"engine_checks.json").read_text()))
    (OUT/"checks.json").write_text(json.dumps(checks,indent=2))
    lines=["# 股指期货：各指数牛熊状态与入场/退出策略","",
        "结论与反例分析见 [findings.md](findings.md)。本文件保留全部预登记对照，不按结果筛选方案。","",
        "数据冻结截至2026-11-06；50万元最多一手、无补资、全局日历与实际权益历史验证不变。主版本200日均线6000账户，120/250日敏感性1992账户；全部1500原固定/滚动账户逐日复现。既有其他研究保留原目录。","",
        "## 预登记规则与边界","",
        "- 每个现货指数独立计算MA、20日均值ATR与MA的20日变化。收盘高于MA+ATR且斜率正为偏牛；低于MA−ATR且斜率负为偏熊；其余为过渡。信号日严格早于成交日，无事后牛熊标签、无向后填充。",
        "- 入场：偏牛过滤新空单，偏熊过滤新多单，过渡区不改；允许选择下一名可负担品种，所以不只改变空仓。",
        "- 退出：原退出全部保留；状态不支持持仓方向（包括过渡）且冻结原5日模型预计续持扣费增量≤0，下一开盘退出。均线变化本身不直接平仓、不直接反手。主mean5对照原本需要连续两次负预测，新规则只在不支持的状态下缩短为一次。",
        "- 本轮检验状态操作规则，不重新训练LR或新增状态交互项，也未放宽顺势持仓的20日上限。主MA200保留control/loss_exit/mean5/calibrated5/tail5五种退出；MA120/250只在mean5跑同一起点与情景，禁止按表现挑最佳均线。",
        "- 固定指base入场规则固定，原预测仍按既有季度模型更新。滚动启动/每季度仅用过去36个月已完成的122/244日月度历史窗，12月近期半衰期，精确当前权益，同一状态规则，沿用原候选及切换门槛。没有新增现金、重新入场、止损或补资。",
        "- 分界会滞后且震荡反复；日线成交及15%/25%保证金是情景，不能保证盘中无强平。已反复观察历史不再是新OOT，重叠起点与共享市场事件不是独立证据。","",
        "## 主版本：完整244日等长窗口","",
        "原5日退出；各月度起点同长比较。收益/回撤均扣费，空仓包括无信号、资金不足及状态过滤。","",
        "| 成本/保证金 | 方案 | 盈利起点比例 | 收益中位 | P10收益 | 最差收益 | 最差回撤 | 空仓比例中位 | 零交易数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    p=dist[dist.ma_window.eq(200)&dist.variant.eq("mean5")&dist.window.eq("sessions_244")]
    for r in p.itertuples():
        lines.append(f"| {r.fee*10000:.0f}bps/{r.margin:.0%} | {LABELS[r.startup_mode]} | "+" | ".join(
            pct(getattr(r,k)) for k in ("profitable_fraction","median_return","p10_return","worst_return","worst_drawdown","median_flat_ratio"))+f" | {r.zero_trade_accounts} |")
    lines += ["","## 逐年独立单年与各年起累计至统一终点","",
        "每格净收益/最大回撤。单年各自50万元并在年末清算；累计持有至2026-11-06、不重置权益。2026为部分年度。",""]
    primary=data[data.ma_window.eq(200)]
    for fee,margin in ALL_PROFILES:
        for variant in VARIANTS:
            for style in ("fixed","rolling"):
                modes=[f"{style}_{m}" for m in MODES]
                for window,label in (("single_year","单年"),("to_latest","累计至统一终点")):
                    part=primary[primary.annual&primary.fee.eq(fee)&primary.margin.eq(margin)&primary.variant.eq(variant)&primary.window.eq(window)]
                    performance_table(lines,f"### {fee*10000:.0f}bps/{margin:.0%} · {NAMES[variant]} · {style} · {label}",part,modes)
    lines += ["## 主版本：逐年月度起点与等长窗口","",
        "每格收益中位/最差收益/最差回撤。2026暂无完整244日新起点；不能拿短期累计与完整一年直接比较。",""]
    for fee,margin in PROFILES:
        for window in ("to_latest","sessions_122","sessions_244"):
            for style in ("fixed","rolling"):
                modes=[f"{style}_{m}" for m in MODES]
                lines += [f"### {fee*10000:.0f}bps/{margin:.0%} · mean5 · {window} · {style}","",
                    "| 起始年 | 起点数 | "+" | ".join(LABELS[m] for m in modes)+" |","|---|---:|"+"---:|"*4]
                part=cohorts[cohorts.ma_window.eq(200)&cohorts.variant.eq("mean5")&cohorts.fee.eq(fee)&cohorts.margin.eq(margin)&cohorts.window.eq(window)]
                for year,g in part.groupby("start_year"):
                    g=g.set_index("startup_mode")
                    lines.append(f"| {year} | {g.loc[modes[0],'starts']} | "+" | ".join(
                        "/".join(pct(g.loc[m,k]) for k in ("median_return","worst_return","worst_drawdown")) for m in modes)+" |")
                lines.append("")
    lines += ["## 逐年配对增量与机会成本","",
        "同一起点配对，收益中位差不是两个分布中位数之差。错失盈利与避免亏损按基准持仓而新方案空仓的日子统计；日度盈亏含费用，不是可实现的选择性交易。不同列的中位数通常来自不同账户，不能直接相加。完整逐账户归因能与收益差逐项对平。",""]
    for fee,margin in PROFILES:
        for window in ("to_latest","sessions_122","sessions_244"):
            lines += [f"### mean5 · {fee*10000:.0f}bps/{margin:.0%} · {window}","",
                "| 年 | 对照→方案 | 数量 | 收益配对增量中位 | 收益改善比例 | 回撤改善/恶化比例 | 新增空仓日中位 | 错失盈利中位/避免亏损中位（万元） |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
            part=effects[effects.ma_window.eq(200)&effects.variant.eq("mean5")&effects.fee.eq(fee)&effects.margin.eq(margin)&effects.window.eq(window)]
            for r in part.sort_values(["start_year","effect","startup_mode"]).itertuples():
                lines.append(f"| {r.start_year} | {LABELS[r.baseline_mode]}→{LABELS[r.startup_mode]} | {r.starts} | {pct(r.median_return_change)} | {pct(r.return_improved_fraction)} | {pct(r.drawdown_improved_fraction)}/{pct(r.drawdown_worsened_fraction)} | {r.median_additional_flat_days:.1f} | {r.median_missed_positive/10000:.2f}/{r.median_avoided_negative/10000:.2f} |")
            lines.append("")
    lines += ["## 参数敏感性：不选择最优均线","",
        "mean5全月度起点；列示原基准以及三条均线的所有新机制。每格收益中位/最差收益/最差回撤/空仓中位。",""]
    for fee,margin in PROFILES:
        for window in ("sessions_244","to_latest"):
            lines += [f"### {fee*10000:.0f}bps/{margin:.0%} · {window}","",
                "| MA | 方案 | 所有起点 | 2025起点 | 2026起点 |","|---:|---|---:|---:|---:|"]
            part=dist[dist.variant.eq("mean5")&dist.fee.eq(fee)&dist.margin.eq(margin)&dist.window.eq(window)]
            for r in part.sort_values(["startup_mode","ma_window"]).itertuples():
                cells=["/".join(pct(getattr(r,k)) for k in ("median_return","worst_return","worst_drawdown","median_flat_ratio"))]
                for year in ("2025","2026"):
                    c=cohorts[cohorts.start_year.eq(year)&cohorts.ma_window.eq(r.ma_window)&cohorts.startup_mode.eq(r.startup_mode)&cohorts.variant.eq("mean5")&cohorts.fee.eq(fee)&cohorts.margin.eq(margin)&cohorts.window.eq(window)]
                    cells.append("—" if c.empty else "/".join(pct(c.iloc[0][k]) for k in ("median_return","worst_return","worst_drawdown","median_flat_ratio")))
                lines.append(f"| {r.ma_window} | {LABELS[r.startup_mode]} | "+" | ".join(cells)+" |")
            lines.append("")
    lines += ["## 全退出版本、全起点的最差反例","",
        "跨control/loss_exit/mean5/calibrated5/tail5与月度/相邻日起点；最差收益与最差回撤可能不是同一账户。", "",
        "| 成本/保证金 | 方案 | 最差收益：起点/退出版本 | 净收益 | 最大回撤：起点/退出版本 | 最大回撤 |",
        "|---|---|---|---:|---|---:|"]
    for (fee,margin,mode),g in totals[totals.ma_window.eq(200)].groupby(["fee","margin","startup_mode"]):
        a=g.loc[g.total_return.idxmin()];b=g.loc[g.max_drawdown.idxmin()]
        lines.append(f"| {fee*10000:.0f}bps/{margin:.0%} | {LABELS[mode]} | {a.start}/{a.variant} | {pct(a.total_return)} | {b.start}/{b.variant} | {pct(-b.max_drawdown)} |")
    lines += ["","## 持续账户在近两年的年度表现","",
        "2025年初50万元启动，2026不重置本金；分母使用年度开始实际权益。下面为mean5与200日状态。", "",
        "| 成本/保证金 | 方案 | 记账年 | 年度收益 | 年内最大回撤 |","|---|---|---|---:|---:|"]
    part=annual[annual.start.eq("20250101")&annual.variant.eq("mean5")&annual.ma_window.eq(200)&annual.side.eq("all")]
    for r in part.itertuples():
        lines.append(f"| {r.fee*10000:.0f}bps/{r.margin:.0%} | {LABELS[r.startup_mode]} | {r.year} | {pct(r.total_return)} | {pct(-r.max_drawdown)} |")
    lines += ["","## 审计与文件","",f"```json\n{json.dumps(checks,ensure_ascii=False,indent=2)}\n```","",
        "- protocol.json / sources.json：运行前定义、冻结输入哈希；states_*.parquet / state_coverage.csv：每日状态与年度覆盖。",
        "- summary.csv：所有账户及窗口；annual_marked.csv：持续账户逐年、多空记账；year_cohorts.csv与january_phase_sensitivity.csv：月度与相邻日起点分布。",
        "- paired_effects.csv / effects_by_year.csv / opportunity_costs.csv：同起点增量与精确机会成本归因。",
        "- trades_with_regime.csv / entry_regime_side_diagnostics.csv：入场状态、多空、交易盈亏；后者按年初账户的交易入场年统计，跨年平仓不等于年末清算收益。重叠持续账户不应累计成独立交易样本。",
        "- worst_all_starts.csv / forced_exit_events.csv：全网格最差账户与强平场景，共享事件不能视为独立证据。",
        "- 每年目录含逐日账本、入场/退出诊断、精确权益历史验证窗、选择和评分审计。", "",
        "![近两年起点](recent_starts.png)","","![账户路径](account_paths.png)",""]
    (OUT/"report.md").write_text("\n".join(lines))
    charts(data)
    print(json.dumps(checks,indent=2),flush=True)


if __name__=="__main__":
    main()
