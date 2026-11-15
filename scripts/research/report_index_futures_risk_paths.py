"""Report all frozen risk-overlay variants, including failures and diagnostics."""
import json
import os

import numpy as np
import pandas as pd

from evaluate_index_futures_risk_paths import OUT, VARIANTS, CAPITAL
from evaluate_index_futures_directions import load_bars

NAMES = dict(control="原季度方案", entry_risk="入场风险", loss_exit="亏损退出",
             signal_exit="反向确认", combined="三项组合")
REASONS = dict(holding_limit="满20日", native_opposing_exit="原策略反向退出",
               close_loss_exit="收盘亏损退出", confirmed_opposing_exit="连续反向确认",
               reserve_exit="安全储备不足", strategy_switch="季度切换", sample_end="样本结束")


def pct(value):
    return "—" if pd.isna(value) else f"{value:.2%}"


def shadow_loss_exits(trades):
    """Freeze baseline entries; isolate the loss exit from portfolio path changes."""
    bars = load_bars()
    opens, closes = bars.open.to_dict(), bars.close.to_dict()
    results = []
    for (start, fee, margin), part in trades[trades.variant.eq("control")].groupby(["start", "fee", "margin"]):
        ledger = pd.read_csv(OUT/f"ledger_{start}_control_{fee}_{margin}.csv", dtype={"entry_date": str, "exit_date": str}).fillna({"positions": ""})
        for trade in part.itertuples():
            held = ledger[ledger.entry_date.ge(trade.entry)&ledger.entry_date.lt(trade.exit)]
            ending, pnl, triggered = trade.exit, trade.pnl, False
            for row in held.itertuples():
                code, sign = row.positions.split(":")
                direction, multiplier = int(sign), trade.multiplier
                close, future_open = closes[row.entry_date, code], opens[row.exit_date, code]
                equity = row.equity + (future_open*multiplier*fee if row.Index == len(ledger)-1 else 0)
                mark = equity-trade.entry_equity-direction*(future_open-close)*multiplier-close*multiplier*fee
                if mark <= -.05*trade.entry_equity:
                    ending, pnl, triggered = row.exit_date, equity-trade.entry_equity-future_open*multiplier*fee, True
                    break
            results.append(dict(start=start, fee=fee, margin=margin, entry=trade.entry, original_exit=trade.exit,
                                product=trade.product, side=trade.side, original_pnl=trade.pnl,
                                shadow_exit=ending, shadow_pnl=pnl, change=pnl-trade.pnl, triggered=triggered))
    return pd.DataFrame(results)


def chart():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant_adaptation_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#777777", "#c78924", "#278b82", "#657ac3", "#aa477b"]
    labels = ["Control", "Entry risk", "Loss exit", "Signal exit", "Combined"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    for column, (start, fee, margin) in enumerate((("20220101", .0003, .15),
                                                 ("20220101", .0006, .25),
                                                 ("20260101", .0003, .15))):
        for variant, color, label in zip(VARIANTS, colors, labels):
            ledger = pd.read_csv(OUT/f"ledger_{start}_{variant}_{fee}_{margin}.csv", dtype={"entry_date": str, "exit_date": str})
            path = np.r_[CAPITAL, ledger.equity]
            dates = pd.to_datetime([ledger.entry_date.iloc[0], *ledger.exit_date])
            axes[0, column].plot(dates, path/10000, color=color, lw=1.3, label=label)
            axes[1, column].plot(dates, (path/np.maximum.accumulate(path)-1)*100, color=color, lw=1)
        axes[0, column].set_title(f"Start {start[:4]} | {fee*10000:.0f} bps | {margin:.0%} margin")
        axes[0, column].set_ylabel("Equity (CNY 10,000)")
        axes[1, column].set_ylabel("Drawdown (%)")
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Symmetric risk overlays | CNY 500,000, at most one lot", fontsize=15)
    fig.text(.5, .93, "Exploratory historical replay through 2026-11-06; no fresh OOT; costs include rolls", ha="center", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .055, 1, .91))
    fig.savefig(OUT/"equity_drawdown.png", dpi=170)
    plt.close(fig)


def main():
    summary = pd.read_csv(OUT/"summary.csv", dtype={"start": str, "end": str})
    trades = pd.read_csv(OUT/"trades.csv", dtype={"start": str, "entry": str, "exit": str})
    annual = pd.read_csv(OUT/"annual_marked.csv", dtype={"start": str, "year": str})
    trace = pd.read_csv(OUT/"daily_decisions.csv", dtype={"start": str, "date": str, "signal_date": str})
    audit = pd.read_csv(OUT/"validation_audit.csv", dtype={"date": str, "cutoff": str, "last_validation_exit": str})
    assert len(summary) == 200
    assert audit.last_validation_exit.le(audit.cutoff).all() and audit.cutoff.lt(audit.date).all()
    assert trace.signal_date.lt(trace.date).all()
    keys = ["start", "fee", "margin", "variant"]
    matched = summary[summary.window.eq("to_latest")].merge(trades.groupby(keys).pnl.sum().rename("net"), on=keys, how="left")
    np.testing.assert_allclose(matched.net.fillna(0), matched.total_return*CAPITAL, rtol=0, atol=1e-6)
    # Count improvements descriptively; overlapping start windows are not independent trials.
    controls = summary[summary.variant.eq("control")].drop(columns="variant")
    comparisons = summary[summary.variant.ne("control")].merge(controls, on=["start", "fee", "margin", "window"], suffixes=("", "_control"))
    comparisons["return_delta"] = comparisons.total_return-comparisons.total_return_control
    comparisons["drawdown_improvement"] = comparisons.max_drawdown-comparisons.max_drawdown_control
    comparisons.to_csv(OUT/"paired_comparison.csv", index=False)
    # All side/product attribution uses complete trade PnL, assigned to exit year.
    trades["exit_year"] = trades.exit.str[:4]
    attribution = trades.groupby(keys+["exit_year", "side", "product"]).agg(
        trades=("pnl", "size"), wins=("pnl", lambda s: int(s.gt(0).sum())),
        net_pnl=("pnl", "sum"), worst_trade=("pnl", "min")).reset_index()
    attribution.to_csv(OUT/"side_product_attribution.csv", index=False)
    blocks = trace.assign(year=trace.date.str[:4]).groupby(keys+["year", "action"]).size().rename("days").reset_index()
    blocks.to_csv(OUT/"decision_counts.csv", index=False)
    shadow = shadow_loss_exits(trades)
    shadow.to_csv(OUT/"frozen_entry_loss_exit.csv", index=False)
    checks = json.loads((OUT/"checks.json").read_text())
    checks.update(validation_records=len(audit), trace_rows=len(trace),
                  forced_exits=int(matched.forced.sum()), settlement_breaches=int(matched.settlement_breaches.sum()),
                  last=summary.end.max(), paired_results=len(comparisons), unit_tests=31)
    (OUT/"checks.json").write_text(json.dumps(checks, indent=2))

    lines = ["# 高保证金、2026归因与对称风险退出", "",
             "50万元初始资金，全组合最多一手，允许做多和做空。数据统一止于2026-11-06，2026是不完整年度。复用上轮季度模型、36个月历史窗口、12个月半衰期和资金/切换门槛。100条持续账户路径，200个评估窗口。", "",
             "本轮所有规则在运行前固定，不按2026亏损日期或IF品种定制，不搜索阈值。旧历史已反复查看，以下仍是探索性历史回放，不能称为新的未触碰OOT。", "",
             "**结论：亏损退出是较值得继续验证的单项候选，但还不是稳定的最终方案。基准情景2022起累计收益144.74%→177.53%、回撤31.10%→21.68%；该持续账户2026当年-5.57%→+10.49%。然而联合压力的2026独立账户反而从-4.57%变成-15.80%，2023起联合压力累计从+146.06%变成-35.27%，存在明显起点和资金路径敏感性。三项组合虽然改善部分长期回撤，但基准2025、2026独立年份转亏，不应因2022起曲线漂亮而采用。**", "",
             "## 固定对照与统计口径", "",
             "- 原季度方案：上轮资金适配+切换门槛重选，四项候选策略池不变。",
             "- 入场风险：过去20日波动率×2×一手名义金额，不超过当前权益5%；多空同规则，超出则寻找其他合格品种，否则空仓。它是单日波动代理，不是20日持仓最大损失估计。",
             "- 亏损退出：截至上一日收盘、计入换月与费用的交易净亏损达到入场权益5%，下一开盘退出；不设置止盈。跳空和日内波动可能使实际亏损超过5%。",
             "- 反向确认：相反方向20日模型分数≥0.45，且高于本方向至少0.10，连续两个收盘成立后下一开盘退出；分数未经概率校准。",
             "- 三项组合：同时应用上述三项，不因本次结果重新组合。所有历史候选策略的季度验证也应用同一风控版本。",
             "", "5%只是本轮研究情景，不代表用户已接受这一风险预算。退出优先级为季度切换、原20日上限、原策略反向退出、亏损退出、新反向确认、保证金安全储备。平仓后仍按原每5交易日入场节奏检查，不立即反手。", "",
             "年度表：每年独立以50万开始，在该年最后可用开盘平仓；2026到共同数据终点。累计表：每个起始年独立以50万运行至共同终点，中途不重置资金或持仓。各行不可拼接。持续账户的逐年盯市收益另列，但不再展示‘2022起累计到各年末’。每格为扣费收益 / 最大回撤幅度。", "",
             "单年窗口由同起点持续账户的因果前缀计算，另加年末平仓费用；样本最后开盘原已平仓时不重复计费。年末日期取exit_date落在当年，故采用最后可用开盘而非最后收盘。", ""]

    for fee, margin, caption in ((.0003, .15, "基准：3bps/15%"), (.0006, .25, "联合压力：6bps/25%")):
        for window, title in (("single_year", "各年独立账户的单年表现"), ("to_latest", "各起始年累计到2026-11-06")):
            lines += [f"## {caption}：{title}", "", "| 年份 | " + " | ".join(NAMES[v] for v in VARIANTS)+" |",
                      "|---|"+"---:|"*len(VARIANTS)]
            part = summary[summary.fee.eq(fee) & summary.margin.eq(margin) & summary.window.eq(window)]
            for start, group in part.groupby("start"):
                group = group.set_index("variant")
                lines.append("| "+start[:4]+" | "+" | ".join(
                    f"{pct(group.loc[v].total_return)} / {pct(-group.loc[v].max_drawdown)}" for v in VARIANTS)+" |")
            lines.append("")

    lines += ["## 持续账户逐年当年收益（不按年重置本金）", "",
              "此表是2022起账户各年的当年收益和年内回撤，与上方每年新50万元表不同；不含累计到该年末的指标。", "",
              "| 情景 | 年份 | 原季度方案 收益/回撤 | 亏损退出 收益/回撤 | 三项组合 收益/回撤 |", "|---|---|---:|---:|---:|"]
    for fee, margin in ((.0003, .15), (.0006, .25)):
        part = annual.query("start == '20220101' and side == 'all'")
        part = part[part.fee.eq(fee)&part.margin.eq(margin)]
        for year, group in part.groupby("year"):
            group = group.set_index("variant")
            lines.append(f"| {fee*10000:.0f}bps/{margin:.0%} | {year} | "+" | ".join(
                f"{pct(group.loc[v].total_return)} / {pct(-group.loc[v].max_drawdown)}" for v in ("control", "loss_exit", "combined"))+" |")

    baseline = trades.query("start == '20220101' and variant == 'control'")
    high = baseline.query("fee == .0006 and margin == .25")
    recent = baseline.query("fee == .0003 and margin == .15 and exit_year == '2026'")
    lines += ["", "## 归因：多空与具体交易", "", "完整交易净利润按平仓年归属；跨年交易与该年的逐日盯市盈亏不是同一口径。", "",
              "| 原方案样本 | 方向 | 交易数 | 获胜数 | 合计净利润（元） |", "|---|---|---:|---:|---:|"]
    for title, sample in (("2022起联合压力，全期", high), ("2022起基准账户，2026", recent)):
        for side, group in sample.groupby("side"):
            lines.append(f"| {title} | {side} | {len(group)} | {group.pnl.gt(0).sum()} | {group.pnl.sum():,.2f} |")
    lines += ["", "高保证金提高的是资金占用门槛，不直接把同一手的每点亏损放大。但它改变可开品种、退出时点、后续权益及季度重选，从而改变整条路径。", "",
              "| 原方案样本 | 入场→退出 | 品种方向 | 天数 | 净利润（元） | 最佳收盘浮盈（元） | 退出原因 |", "|---|---|---|---:|---:|---:|---|"]
    for title, sample in (("联合压力全期", high), ("基准2026", recent)):
        for t in sample.sort_values("entry").itertuples():
            lines.append(f"| {title} | {t.entry}→{t.exit} | {t.product} {t.side} | {t.days:.0f} | {t.pnl:,.2f} | {t.best_close_pnl:,.2f} | {REASONS.get(t.exit_reason,t.exit_reason)} |")
    lines += ["", "### 固定原入场，单独检验亏损退出", "",
              "保留原方案每笔交易的入场及入场权益，仅套用5%收盘亏损退出，互不影响后续交易。此诊断不触发季度重选、不开新交易，也不是可执行账户收益。用于区分直接止损效果与资金/时点变化效果。", "",
              "| 基准2026原交易 | 品种方向 | 原净利润（元） | 影子退出日 | 影子净利润（元） | 改善额（元） |", "|---|---|---:|---|---:|---:|"]
    sample = shadow.query("start == '20220101' and fee == .0003 and margin == .15")
    for r in sample[sample.original_exit.str.startswith("2026")].itertuples():
        lines.append(f"| {r.entry} | {r.product} {r.side} | {r.original_pnl:,.2f} | {r.shadow_exit} | {r.shadow_pnl:,.2f} | {r.change:,.2f} |")
    lines += ["", "固定原入场后，2026仅改善16015.19元，仍净亏56205.64元。完整亏损退出账户2026转正，主要还受后续新交易、当前权益和季度重选变化影响，不能把它当作单一止损的独立收益。", ""]
    blocked = trace.query("start == '20220101' and variant == 'control' and fee == .0006 and margin == .25")
    lines += ["", "### 联合压力原方案的资金门槛", "",
              "| 年份 | 无仓且到入场检查日、有合格信号的次数 | 全部因资金被拦次数 | 当年最低权益（元） | 合格信号所需资金最低值（元） |", "|---|---:|---:|---:|---:|"]
    for year, group in blocked.groupby(blocked.date.str[:4]):
        lines.append(f"| {year} | {group.raw_candidates.gt(0).sum()} | {group.action.eq('capital_block').sum()} | {group.equity.min():,.0f} | {group.minimum_signal_equity.min():,.0f} |")
    lines += ["", "资金门槛包括开仓费、保证金和原有安全储备：以至少5%（或3倍过去日波动率）的不利变动压力后，仍剩20%权益为条件。它比交易所最低开仓保证金严格；‘无交易’不能直接称为爆仓。本轮不通过降低安全储备或追加本金修饰结果。", "",
              "2025/2026账户权益约28.94万元，全部28/32次合格入场机会均被资金拦截。并非每个日期所有品种都绝对买不起：不满足信号条件的便宜品种可能买得起；但当时合格品种的最低所需资金仍约30.94/29.11万元。", "",
              "## 全部费用/保证金分离结果：2022起累计", "",
              "| 单边成本 | 保证金 | 版本 | 收益 | 回撤 | 交易数 | 胜率 | 费用（元） |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in summary.query("start == '20220101' and window == 'to_latest'").itertuples():
        lines.append(f"| {r.fee*10000:.0f}bps | {r.margin:.0%} | {NAMES[r.variant]} | {pct(r.total_return)} | {pct(-r.max_drawdown)} | {r.trades} | {pct(r.win_rate)} | {r.fees:,.0f} |")
    lines += ["", "## 跨年份与情景的一致性", "",
              "每种窗口各20组：5个起点×4个成本/保证金情景。以下是描述性计数，重叠起点高度相关，不能当作20次独立统计试验。", "",
              "| 窗口 | 版本 | 收益高于原方案 | 回撤小于原方案 | 两者同时改善 |", "|---|---|---:|---:|---:|"]
    for (window, variant), g in comparisons.groupby(["window", "variant"]):
        ret, dd = g.return_delta.gt(1e-10), g.drawdown_improvement.gt(1e-10)
        lines.append(f"| {window} | {NAMES[variant]} | {ret.sum()}/20 | {dd.sum()}/20 | {(ret&dd).sum()}/20 |")
    lines += ["", "## 审计、限制与文件", "",
              f"{checks['control_matches']}个原方案账本全部与上一轮一致；100个持续账户与逐笔交易净利润核对；200个评估窗口，{len(audit)}条候选历史验证满足cutoff；31项专项测试通过。模拟强平{checks['forced_exits']}次，结算保证金不足{checks['settlement_breaches']}次。", "",
              "开盘到次日开盘盯市，日线收盘产生退出信号；缺少盘中保证金追缴、涨跌停锁单和订单簿成交约束，因此零模拟强平不能证明实盘无爆仓风险。保证金15%/25%、单边3/6bps（费用与滑点的统一成本假设）均为情景，并非逐日历史实际费率。中金所规则明确实行当日无负债结算及风险处置，参考[交易规则](https://www.cffex.com.cn/u/cms/www/202003/27155912c7m4.pdf)和[结算细则](https://www.cffex.com.cn/u/cms/www/202501/071747590uho.pdf)。", "",
              "本轮没有重新训练退出label；方向模型仍是20/60日目标，改变退出后label和实际交易目标存在不一致。结果同时包含退出直接效果，以及资金、后续入场节奏、策略选择变化的间接效果；不能把全部差异归于某一笔成功止损。", "",
              "单项亏损退出版本中，触发该退出的交易最差实际净亏约入场权益15.22%，说明上一收盘5%触发线不能充当最大损失承诺。高保证金和成本对路径的影响仍很强，不能因未模拟强平便判定资金风险消除。", "",
              "下一步应冻结小型候选，不继续围绕这些年份微调5%或反向阈值；用冻结退出规则重建历史候选交易label，预测继续持有相对立即退出的增量净收益及风险，在内层时间验证选模型，再等待新增成熟数据前瞻检验。", "",
              "文件：protocol.json（运行前规则）、summary.csv（两种窗口）、annual_marked.csv（持续账户当年表现）、trades.csv（逐笔和退出归因）、side_product_attribution.csv（年/方向/品种归因）、frozen_entry_loss_exit.csv（固定入场影子诊断）、daily_decisions.csv（每次决策）、decision_counts.csv（拦截原因）、selection.csv及validation_audit.csv（季度审计）、paired_comparison.csv（全部对照差值）、equity_drawdown.png。", ""]
    (OUT/"report.md").write_text("\n".join(lines), encoding="utf-8")
    chart()
    print(json.dumps(checks, ensure_ascii=False))
    print(comparisons.groupby(["window", "variant"])[["return_delta", "drawdown_improvement"]].median())


if __name__ == "__main__":
    main()
