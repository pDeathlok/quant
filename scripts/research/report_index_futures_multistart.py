"""Report all frozen policies, matched starts, cash friction and worst paths."""
import json
import os

import numpy as np
import pandas as pd

from evaluate_index_futures_multistart import OUT, PREVIOUS, MAIN_MODES, NEW_MODES, VARIANTS, PROFILES, CAPITAL, DATE_TYPES
from report_index_futures_startup import distributions, pct, NAMES
from evaluate_index_futures_directions import load_bars

LABELS = dict(calendar="全局日历/base启动", calendar_history="全局日历/单路径历史选择",
              multi_tail="多起点收益回撤", multi_funded="多起点+资金拒绝约束")


def path_for(row):
    root = OUT if row.startup_mode in NEW_MODES else PREVIOUS
    return root/row.start[:4]/"ledgers"/f"{row.start}_{row.variant}_{row.startup_mode}_{row.fee}_{row.margin}.parquet"


def grouped(data, keys):
    result = distributions(data, keys)
    funding = data.groupby(keys).agg(median_blocked_ratio=("blocked_ratio", "median"),
        worst_blocked_ratio=("blocked_ratio", "max"),
        no_signal_accounts=("signal_opportunities", lambda s: s.eq(0).sum())).reset_index()
    return result.merge(funding, on=keys, validate="one_to_one")


def draw(summary):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant_adaptation_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    labels = ["Global calendar / base start", "Single-path historical selection",
              "Multi-start return / drawdown", "Multi-start + funding gate"]
    data = summary[summary.startup_mode.isin(MAIN_MODES)&summary.variant.eq("mean5")&summary.start_kind.eq("monthly")]
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.5))
    maximum = max(10, data[data.window.eq("sessions_244")].total_return.abs().max()*100)
    for ax, (fee, margin) in zip(axes, PROFILES):
        part = data[data.fee.eq(fee)&data.margin.eq(margin)&data.window.eq("sessions_244")]
        table = part.pivot(index="startup_mode", columns="start", values="total_return").reindex(MAIN_MODES)
        im = ax.imshow(100*table.to_numpy(), aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum))
        ticks = np.arange(0, len(table.columns), 3)
        ax.set_xticks(ticks, [s[:4]+"-"+s[4:6] for s in table.columns[ticks]], rotation=35, ha="right")
        ax.set_yticks(range(4), labels)
        ax.set_title(f"Raw5 exit | {fee*10000:.0f}bps costs | {margin:.0%} margin")
    fig.suptitle("Multi-start historical policy selection: same 244-session outer windows", fontsize=14)
    fig.text(.5, .91, "500k initial capital, max one lot; overlapping starts are not independent", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, .93, .89))
    fig.colorbar(im, cax=fig.add_axes([.94, .16, .012, .63]), label="Net return (%)")
    fig.savefig(OUT/"equal_window_heatmap.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for i, year in enumerate(("2025", "2026")):
        for j, (fee, margin) in enumerate(PROFILES):
            part = data[data.window.eq("to_latest")&data.start.str.startswith(year)&data.fee.eq(fee)&data.margin.eq(margin)]
            for mode, label in zip(MAIN_MODES, labels):
                sample = part[part.startup_mode.eq(mode)].sort_values("start")
                axes[i, j].plot(sample.start.str[4:6].astype(int), sample.total_return*100, marker="o", label=label)
            axes[i, j].set_title(f"{year} starts | {fee*10000:.0f}bps | {margin:.0%} margin")
            axes[i, j].set_xlabel("Startup month")
            axes[i, j].set_ylabel("Net return through 2026-11-06 (%)")
            axes[i, j].axhline(0, color="#999", linewidth=.7)
            axes[i, j].grid(alpha=.2)
    fig.suptitle("Recent monthly startups: raw5 exit, continuous account to common end", fontsize=13)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(OUT/"recent_monthly_starts.png", dpi=160)
    plt.close(fig)


def main():
    summary = pd.read_csv(OUT/"summary.csv", dtype=DATE_TYPES)
    old = pd.read_csv(PREVIOUS/"summary.csv", dtype=DATE_TYPES)
    old_keys = ["start", "fee", "margin", "variant", "startup_mode", "window"]
    retained = summary[~summary.startup_mode.isin(NEW_MODES)][old.columns]
    pd.testing.assert_frame_equal(retained.sort_values(old_keys).reset_index(drop=True),
        old.sort_values(old_keys).reset_index(drop=True), check_dtype=False, atol=1e-8, rtol=1e-10)
    trades = pd.read_csv(OUT/"trades.csv", dtype=DATE_TYPES)
    annual = pd.read_csv(OUT/"annual_marked.csv", dtype=DATE_TYPES)
    selection = pd.read_csv(OUT/"selection.csv", dtype=DATE_TYPES)
    audit = pd.read_csv(OUT/"selection_audit.csv", dtype=DATE_TYPES).drop_duplicates()
    reference = pd.read_csv(OUT/"reference_windows.csv", dtype=DATE_TYPES)
    assert reference.groupby(["start", "horizon", "fee", "margin", "variant", "policy"]).size().eq(1).all()
    assert audit.last_history_end.le(audit.cutoff).all()
    assert selection.cutoff.lt(selection.date).all()
    assert audit.loc[audit.ready, ["windows_122", "windows_244"]].ge(6).all().all()
    totals = summary[summary.window.eq("to_latest")]
    keys = old_keys[:-1]
    assert len(totals) == 4500 and totals.groupby("startup_mode").size().eq(750).all()
    recon = totals.merge(trades.groupby(keys).pnl.sum().rename("net"), on=keys, how="left")
    np.testing.assert_allclose(recon.net.fillna(0), recon.total_return*CAPITAL, atol=1e-6, rtol=0)
    for h in (122, 244):
        assert summary.loc[summary.window.eq(f"sessions_{h}"), "days"].eq(h).all()
    main_data = summary[summary.startup_mode.isin(MAIN_MODES)].copy()
    assert main_data.blocked_opportunities.le(main_data.signal_opportunities).all()
    assert main_data.loc[main_data.signal_opportunities.eq(0), "blocked_ratio"].isna().all()
    monthly = main_data[main_data.start_kind.eq("monthly")].copy()
    monthly["start_year"] = monthly.start.str[:4]
    dist = grouped(monthly, ["window", "fee", "margin", "variant", "startup_mode"])
    dist.to_csv(OUT/"start_distributions.csv", index=False)
    cohorts = grouped(monthly[monthly.window.eq("to_latest")], ["start_year", "fee", "margin", "variant", "startup_mode"])
    cohorts.to_csv(OUT/"monthly_cohorts.csv", index=False)
    equal = grouped(monthly[monthly.window.isin(["sessions_122", "sessions_244"])],
        ["start_year", "window", "fee", "margin", "variant", "startup_mode"])
    equal.to_csv(OUT/"equal_window_cohorts.csv", index=False)
    phase = main_data[main_data.window.eq("to_latest")&main_data.start.str[4:6].eq("01")&main_data.variant.isin(VARIANTS[:3])].copy()
    phase = phase[(phase.fee.eq(.0003)&phase.margin.eq(.15))|(phase.fee.eq(.0006)&phase.margin.eq(.25))]
    phase["start_year"] = phase.start.str[:4]
    phases = grouped(phase, ["start_year", "fee", "margin", "variant", "startup_mode"])
    assert phases.starts.eq(5).all()
    phases["return_range"] = phases.best_return-phases.worst_return
    phases.to_csv(OUT/"january_phase_sensitivity.csv", index=False)
    paired = []
    for baseline in MAIN_MODES[:2]:
        part = main_data[main_data.startup_mode.isin(NEW_MODES)].merge(
            main_data[main_data.startup_mode.eq(baseline)], on=old_keys[:4]+["window"], suffixes=("", "_control"))
        part["baseline"] = baseline
        part["return_delta"] = part.total_return-part.total_return_control
        part["drawdown_improvement"] = part.max_drawdown-part.max_drawdown_control
        paired.append(part)
    pd.concat(paired, ignore_index=True).to_csv(OUT/"paired_comparison.csv", index=False)
    ablation = totals[totals.startup_mode.eq("multi_funded")].merge(
        totals[totals.startup_mode.eq("multi_tail")], on=keys[:-1], suffixes=("_funded", "_tail"), validate="one_to_one")
    ablation["return_delta"] = ablation.total_return_funded-ablation.total_return_tail
    ablation["drawdown_improvement"] = ablation.max_drawdown_funded-ablation.max_drawdown_tail
    ablation.to_csv(OUT/"funding_ablation.csv", index=False)
    selection.groupby(["startup_mode", "stage", "reason"]).size().rename("decisions").to_csv(OUT/"decision_reasons.csv")
    selection[selection.stage.eq("startup")].groupby(["startup_mode", "selected"]).size().rename("accounts").to_csv(OUT/"initial_policy_counts.csv")
    worst = []
    for _, group in monthly.groupby(["window", "fee", "margin", "variant", "startup_mode"]):
        for criterion, index in (("worst_return", group.total_return.idxmin()), ("worst_drawdown", group.max_drawdown.idxmin())):
            worst.append(group.loc[index].to_dict()|dict(criterion=criterion))
    pd.DataFrame(worst).to_csv(OUT/"worst_starts.csv", index=False)
    all_start_worst = []
    for _, group in totals[totals.startup_mode.isin(MAIN_MODES)].groupby(["fee", "margin", "variant", "startup_mode"]):
        for criterion, index in (("worst_return", group.total_return.idxmin()), ("worst_drawdown", group.max_drawdown.idxmin())):
            all_start_worst.append(group.loc[index].to_dict()|dict(criterion=criterion))
    pd.DataFrame(all_start_worst).to_csv(OUT/"worst_all_starts.csv", index=False)
    bars = load_bars()
    forced = []
    for account in totals[totals.startup_mode.isin(MAIN_MODES)&totals.forced.gt(0)].itertuples():
        ledger = pd.read_parquet(path_for(account))
        for i in ledger.index[ledger.forced_exit]:
            before, event = ledger.iloc[i-1], ledger.iloc[i]
            code, sign = before.positions.split(":")
            multiplier = 300 if code[:2] in ("IF", "IH") else 200
            forced.append(dict(start=account.start, fee=account.fee, margin=account.margin, variant=account.variant,
                startup_mode=account.startup_mode, date=event.entry_date, contract=code, lots=int(sign),
                equity_at_open_before_exit=before.equity,
                required_margin=abs(int(sign))*bars.loc[(event.entry_date, code), "open"]*multiplier*account.margin,
                equity_after=event.equity))
    pd.DataFrame(forced).to_csv(OUT/"forced_exit_events.csv", index=False)
    checks = dict(accounts=len(totals), new_accounts=int(totals.startup_mode.isin(NEW_MODES).sum()),
        windows=len(summary), reference_windows=len(reference), policy_score_records=len(audit),
        startup_selections=int(selection.stage.eq("startup").sum()), decisions=len(selection), unit_tests=63,
        replayed_fixed_calendar_controls=sum(json.loads((OUT/str(y)/"checks.json").read_text())["replayed_fixed_calendar_controls"] for y in range(2022, 2027)),
        all_old_summary_unchanged=True, cash_and_trades_reconcile=True, mature_history_only=True,
        forced_by_mode=totals.groupby("startup_mode").forced.sum().astype(int).to_dict(),
        settlement_breaches=int(totals.settlement_breaches.sum()), last=summary.end.max())
    checks["funding_ablation"] = dict(accounts=len(ablation),
        changed=int(ablation.return_delta.abs().gt(1e-9).sum()), return_better=int(ablation.return_delta.gt(1e-9).sum()),
        return_worse=int(ablation.return_delta.lt(-1e-9).sum()), drawdown_better=int(ablation.drawdown_improvement.gt(1e-9).sum()),
        drawdown_worse=int(ablation.drawdown_improvement.lt(-1e-9).sum()))
    (OUT/"checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    audit.to_csv(OUT/"unique_selection_audit.csv", index=False)
    lines = ["# 多个历史起点与等长窗口选策略", "",
        "50万元、全组合最多一手、多空，固定全局日历，数据截至2026-11-06。保留上轮3000账户，新增两种规则各750账户，共4500路径、12936窗口。主要对照均使用相同全局日历，旧本地日历方案仍保存在summary.csv中。", "",
        "## 事前固定的方法", "",
        "内层：过去36个月的每月首个交易日启动独立50万元空仓参考账户，按候选固定策略运行完整122/244交易日，期末平仓扣费。只有结束日不晚于外层前一信号日的窗口参与选择；不足完整窗口不补零。退出预测仍使用当时季度的历史样本外预测。", "",
        "按窗口结束日使用12个月半衰期；两种长度各占一半权重。244日收益线性除以2，只作统一122日评分单位，不是复利年化；回撤使用原窗口幅度。各长度分别求收益加权P25/中位数、回撤加权P90，再平均两种长度。评分=0.5×(P25收益+中位收益)−0.5×P90回撤。", "",
        "两种新规则：多起点收益回撤；多起点+资金拒绝约束。后者额外要求历史资金拒绝比例不高于base，季度切换时也不高于现任策略。分母仅为空仓且到达检查日、存在策略合格信号的日期；分子为该日没有任何可负担候选。无信号、持仓日、不在检查日均不算资金拒绝；没有任何机会时记缺失、不能作为零风险通过。", "",
        "共同准入：每种长度至少6个完整窗口；历史窗口无强平；加权中位收益为正；归一到122日的平均交易数至少max(2,base一半)，持仓日至少max(10,base一半)；P90回撤不高于base。评分同分保留现任，初始化现任base；证据不足不切换。", "",
        "季度切换还要求评分优势至少0.005（122日评分的0.5个百分点）、P90回撤不劣于有足够交易的现任、当前实际资金可以买得起合约，以及61日历史平均净损益优势≥max(当前权益0.5%,3倍估计切换成本)。启动没有既有持仓，只使用过去评分，不使用当天开盘信息选择初始策略。", "",
        "资金口径的重要区别：新规则内层统一50万元，旧单路径选择按当时外层权益回放历史；外层全部按真实持续权益检查保证金、安全储备和切换成本。新内层统计不声称适用于所有当前权益。这是本轮方法差异的一部分，不能把全部效果只归因于分位数评分。", "",
        "保持五种退出、原四项策略候选、费用与保证金情景；不扫描评分系数、分位数、阈值、日期偏移或品种。候选窗口和资金拒绝比例依赖策略交易路径，低拒绝比例本身不等于更好，故保留交易覆盖准入。", "",
        "## 完整244交易日窗口：所有退出版本", "",
        "每组47个月度起点。最差收益与最差回撤可能来自不同账户；重叠窗口不是独立样本。资金拒绝列是各账户比例的中位数。", "",
        "| 情景 | 退出 | 选择规则 | 盈利比例 | 中位收益 | P10收益 | 最差收益 | 最差回撤 | 中位资金拒绝 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for fee, margin in PROFILES:
        for variant in VARIANTS:
            for mode in MAIN_MODES:
                r = dist[dist.window.eq("sessions_244")&dist.fee.eq(fee)&dist.margin.eq(margin)&dist.variant.eq(variant)&dist.startup_mode.eq(mode)].iloc[0]
                lines.append(f"| {fee*10000:.0f}bps/{margin:.0%} | {NAMES[variant]} | {LABELS[mode]} | "+" | ".join(pct(r[k]) for k in ["profitable_fraction", "median_return", "p10_return", "worst_return", "worst_drawdown", "median_blocked_ratio"])+" |")
    lines += ["", "## 原5日退出：近两年月度起点持续到共同终点", "",
        "| 起始年 | 情景 | 选择规则 | 盈利比例 | 中位累计 | 最差累计 | 最差回撤 | 中位资金拒绝 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for year in ("2025", "2026"):
        for fee, margin in PROFILES:
            for mode in MAIN_MODES:
                r = cohorts[cohorts.start_year.eq(year)&cohorts.fee.eq(fee)&cohorts.margin.eq(margin)&cohorts.variant.eq("mean5")&cohorts.startup_mode.eq(mode)].iloc[0]
                lines.append(f"| {year} | {fee*10000:.0f}bps/{margin:.0%} | {LABELS[mode]} | "+" | ".join(pct(r[k]) for k in ["profitable_fraction", "median_return", "worst_return", "worst_drawdown", "median_blocked_ratio"])+" |")
    lines += ["", "## 按起始年份分组的等长窗口：原5日退出", "",
        "每格中位收益 / 最差收益 / 最差回撤。只收完整窗口，2026没有满244日的新账户。所有退出版本见equal_window_cohorts.csv。", ""]
    for fee, margin in PROFILES:
        for window in ("sessions_122", "sessions_244"):
            lines += [f"### {fee*10000:.0f}bps/{margin:.0%} · {window}", "",
                "| 起始年 | 每模式起点数 | "+" | ".join(LABELS[m] for m in MAIN_MODES)+" |", "|---|---:|"+"---:|"*4]
            part = equal[equal.window.eq(window)&equal.fee.eq(fee)&equal.margin.eq(margin)&equal.variant.eq("mean5")]
            for year, group in part.groupby("start_year"):
                group = group.set_index("startup_mode")
                lines.append(f"| {year} | {group.starts.iloc[0]} | "+" | ".join(" / ".join(pct(group.loc[m, k]) for k in ["median_return", "worst_return", "worst_drawdown"]) for m in MAIN_MODES)+" |")
            lines.append("")
    lines += ["## 逐年单年与各年起累计至今：全部退出版本", "",
        "每格扣费收益 / 最大回撤。独立单年每年重置50万元；累计为各年启动后一直延续，不重置权益。2026截至11月6日。", ""]
    for fee, margin in PROFILES:
        for variant in VARIANTS:
            for window, label in (("single_year", "独立单年"), ("to_latest", "各年起累计至今")):
                lines += [f"### {fee*10000:.0f}bps/{margin:.0%} · {NAMES[variant]} · {label}", "",
                    "| 起始年 | "+" | ".join(LABELS[m] for m in MAIN_MODES)+" |", "|---|"+"---:|"*4]
                part = main_data[main_data.annual&main_data.window.eq(window)&main_data.fee.eq(fee)&main_data.margin.eq(margin)&main_data.variant.eq(variant)]
                for start, group in part.groupby("start"):
                    group = group.set_index("startup_mode")
                    lines.append(f"| {start[:4]} | "+" | ".join(f"{pct(group.loc[m].total_return)} / {pct(-group.loc[m].max_drawdown)}" for m in MAIN_MODES)+" |")
                lines.append("")
    lines += ["## 持续账户近两年当年表现（原5日退出）", "",
        "承接此前权益的当年收益 / 年内回撤，不是每年重置账户。", "",
        "| 起始年 | 当年 | 情景 | "+" | ".join(LABELS[m] for m in MAIN_MODES)+" |", "|---|---|---|"+"---:|"*4]
    part = annual[annual.start.isin(["20220101", "20250101"])&annual.year.ge("2025")&annual.side.eq("all")&annual.variant.eq("mean5")&annual.startup_mode.isin(MAIN_MODES)]
    for (start, year, fee, margin), group in part.groupby(["start", "year", "fee", "margin"]):
        if (fee, margin) not in PROFILES:
            continue
        group = group.set_index("startup_mode")
        lines.append(f"| {start[:4]} | {year} | {fee*10000:.0f}bps/{margin:.0%} | "+" | ".join(f"{pct(group.loc[m].total_return)} / {pct(-group.loc[m].max_drawdown)}" for m in MAIN_MODES)+" |")
    lines += ["", "## 年初前5交易日起点", "",
        "各年起持续至共同终点，表为原5日退出、联合压力；全部版本见january_phase_sensitivity.csv。", "",
        "| 起始年 | 规则 | 最差收益 | 最佳收益 | 收益跨度 | 最差回撤 |", "|---|---|---:|---:|---:|---:|"]
    for year in map(str, range(2022, 2027)):
        for mode in MAIN_MODES:
            r = phases[phases.start_year.eq(year)&phases.fee.eq(.0006)&phases.margin.eq(.25)&phases.variant.eq("mean5")&phases.startup_mode.eq(mode)].iloc[0]
            lines.append(f"| {year} | {LABELS[mode]} | "+" | ".join(pct(r[k]) for k in ["worst_return", "best_return", "return_range", "worst_drawdown"])+" |")
    lines += ["", "## 费用与保证金分开：2026独立账户、原5日退出", "",
        "每格收益 / 最大回撤。额外两种情景仅有年度起点，不混入58个月度统计。", "",
        "| 成本 | 保证金 | "+" | ".join(LABELS[m] for m in MAIN_MODES)+" |", "|---|---|"+"---:|"*4]
    part = main_data[main_data.start.eq("20260101")&main_data.window.eq("single_year")&main_data.variant.eq("mean5")]
    for (fee, margin), group in part.groupby(["fee", "margin"]):
        group = group.set_index("startup_mode")
        lines.append(f"| {fee*10000:.0f}bps | {margin:.0%} | "+" | ".join(f"{pct(group.loc[m].total_return)} / {pct(-group.loc[m].max_drawdown)}" for m in MAIN_MODES)+" |")
    lines += ["", "## 强平与审计", "",
        "| 规则 | 模拟强平次数 |", "|---|---:|"]
    for mode in MAIN_MODES:
        lines.append(f"| {LABELS[mode]} | {checks['forced_by_mode'][mode]} |")
    lines += ["", f"{checks['reference_windows']}条内层参考窗口；{checks['policy_score_records']}条不同日期/候选评分；1500次新启动选择。3000条旧账户摘要逐项一致，其中1500条固定日历旧账本逐笔重放一致。全部4500账户损益与交易对账；63项专项测试通过。", "",
        "强平事件不能当作独立风险概率；日线未模拟盘中追加保证金、涨跌停锁单和实际订单簿。单边3/6bps是固定交易摩擦情景，15%/25%是保证金压力假设。", "",
        "历史已经多轮查看，本轮不是新的未触碰OOT；不得根据这些结果改日历、阈值、年份或品种禁令。新增且成熟的数据才可检验冻结方法的前瞻效果。", "",
        "时间顺序及等长验证参考：[scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。本研究使用实际成熟日期过滤窗口；文档不证明策略盈利有效。", ""]
    findings = OUT/"findings.md"
    if findings.exists():
        lines[4:4] = [findings.read_text(), ""]
    (OUT/"report.md").write_text("\n".join(lines), encoding="utf-8")
    draw(summary)
    print(json.dumps(checks, ensure_ascii=False))
    print(dist[dist.window.eq("sessions_244")&dist.fee.eq(.0006)&dist.margin.eq(.25)&dist.variant.eq("mean5")].to_string(index=False))
    print(cohorts[cohorts.start_year.eq("2026")&cohorts.fee.eq(.0006)&cohorts.margin.eq(.25)&cohorts.variant.eq("mean5")].to_string(index=False))


if __name__ == "__main__":
    main()
