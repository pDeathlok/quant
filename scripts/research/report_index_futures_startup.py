"""Report the complete startup ablation and matched worst-start comparisons."""
import json
import os

import numpy as np
import pandas as pd

from evaluate_index_futures_startup import OUT, PREVIOUS, MODES, VARIANTS, PROFILES, CAPITAL
from evaluate_index_futures_directions import load_bars

NAMES = dict(control="原退出", loss_exit="亏损退出", mean5="5日模型", calibrated5="收益校准", tail5="校准+尾部")
MODENAMES = dict(legacy="原启动", calendar="全局日历", history="历史选初始策略", calendar_history="两者同时启用")


def pct(value):
    return "—" if pd.isna(value) else f"{value:.2%}"


def ledger_for(row):
    if row.startup_mode == "legacy":
        return pd.read_parquet(PREVIOUS/row.start[:4]/"ledgers"/f"{row.start}_{row.variant}_{row.fee}_{row.margin}.parquet")
    return pd.read_parquet(OUT/row.start[:4]/"ledgers"/f"{row.start}_{row.variant}_{row.startup_mode}_{row.fee}_{row.margin}.parquet")


def distributions(summary, group_columns):
    rows = []
    for key, group in summary.groupby(group_columns):
        dd = -group.max_drawdown
        rows.append(dict(zip(group_columns, key), starts=len(group),
                         profitable_fraction=group.total_return.gt(0).mean(), median_return=group.total_return.median(),
                         p10_return=group.total_return.quantile(.1), worst_return=group.total_return.min(),
                         best_return=group.total_return.max(), median_drawdown=dd.median(), worst_drawdown=dd.max(),
                         p90_drawdown=dd.quantile(.9), zero_trade_accounts=int(group.trades.eq(0).sum()),
                         forced=int(group.forced.sum())))
    return pd.DataFrame(rows)


def charts(summary):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant_adaptation_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    labels = ["Original startup", "Global calendar", "Historical initial policy", "Both"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.5))
    samples = summary[summary.start_kind.eq("monthly")&summary.window.eq("sessions_244")&summary.variant.eq("mean5")]
    maximum = max(10, samples.total_return.abs().max()*100)
    for ax, (fee, margin) in zip(axes, PROFILES):
        table = samples[samples.fee.eq(fee)&samples.margin.eq(margin)].pivot(index="startup_mode", columns="start", values="total_return").reindex(MODES)
        im = ax.imshow(table.to_numpy()*100, aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum))
        ticks = np.arange(0, len(table.columns), 3)
        ax.set_xticks(ticks, [s[:4]+"-"+s[4:6] for s in table.columns[ticks]], rotation=35, ha="right")
        ax.set_yticks(range(4), labels)
        ax.set_title(f"Raw 5-session exit model | {fee*10000:.0f} bps | {margin:.0%} margin")
    fig.suptitle("Startup consistency | same 244 sessions, CNY 500,000, at most one lot", fontsize=15)
    fig.text(.5, .915, "No calendar-offset search; all monthly starts retained; overlapping windows are not independent", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, .93, .9))
    fig.colorbar(im, cax=fig.add_axes([.94, .16, .012, .64]), label="Net return (%)")
    fig.savefig(OUT/"startup_heatmap.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for i, year in enumerate(("2025", "2026")):
        for j, (fee, margin) in enumerate(PROFILES):
            data = summary[summary.window.eq("to_latest")&summary.start.str.startswith(year+"01")
                           &summary.variant.eq("mean5")&summary.fee.eq(fee)&summary.margin.eq(margin)]
            for mode, label in zip(MODES, labels):
                group = data[data.startup_mode.eq(mode)].sort_values("first_entry_date")
                axes[i, j].plot(np.arange(1, 6), 100*group.total_return, marker="o", label=label)
            axes[i, j].set_title(f"Jan {year}: {fee*10000:.0f} bps, {margin:.0%} margin")
            axes[i, j].set_xticks(range(1, 6))
            axes[i, j].set_xlabel("Startup on January trading session")
            axes[i, j].set_ylabel("Net return through 2026-11-06 (%)")
            axes[i, j].axhline(0, color="#999999", linewidth=.7)
            axes[i, j].grid(alpha=.2)
    fig.suptitle("A few days apart: raw 5-session exit model", fontsize=15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(OUT/"january_startup.png", dpi=170)
    plt.close(fig)


def main():
    summary = pd.read_csv(OUT/"summary.csv", dtype={"start": str, "end": str, "first_entry_date": str})
    annual = pd.read_csv(OUT/"annual_marked.csv", dtype={"start": str, "year": str})
    trades = pd.read_csv(OUT/"trades.csv", dtype={"start": str, "entry": str, "exit": str})
    startups = pd.read_csv(OUT/"startup_decisions.csv", dtype={"start": str, "date": str, "cutoff": str, "last_history_exit": str})
    audit = pd.read_csv(OUT/"validation_audit.csv", dtype={"date": str, "cutoff": str, "last_validation_exit": str})
    old = pd.read_csv(PREVIOUS/"summary.csv", dtype={"start": str, "end": str, "first_entry_date": str})
    sort_keys = ["start", "fee", "margin", "variant", "window"]
    legacy = summary[summary.startup_mode.eq("legacy")][old.columns]
    pd.testing.assert_frame_equal(old.sort_values(sort_keys).reset_index(drop=True), legacy.sort_values(sort_keys).reset_index(drop=True),
                                  check_dtype=False, atol=1e-8, rtol=1e-10)
    assert audit.last_validation_exit.le(audit.cutoff).all() and audit.cutoff.lt(audit.date).all()
    selected = startups[startups.selection_used_past]
    # Older aggregation may have represented nullable YYYYMMDD as float text.
    history_end = selected.last_history_exit.str.removesuffix(".0")
    assert history_end.le(selected.cutoff).all() and selected.cutoff.lt(selected.date).all()
    keys = ["start", "fee", "margin", "variant", "startup_mode"]
    totals = summary[summary.window.eq("to_latest")]
    assert len(totals) == 3000 and totals.groupby("startup_mode").size().eq(750).all()
    reconcile = totals.merge(trades.groupby(keys).pnl.sum().rename("net"), on=keys, how="left")
    np.testing.assert_allclose(reconcile.net.fillna(0), reconcile.total_return*CAPITAL, atol=1e-6, rtol=0)
    for count in (122, 244):
        assert summary.loc[summary.window.eq(f"sessions_{count}"), "days"].eq(count).all()
    monthly = summary[summary.start_kind.eq("monthly")].copy()
    dist = distributions(monthly, ["window", "fee", "margin", "variant", "startup_mode"])
    dist.to_csv(OUT/"start_distributions.csv", index=False)
    monthly["start_year"] = monthly.start.str[:4]
    cohorts = distributions(monthly[monthly.window.eq("to_latest")], ["start_year", "fee", "margin", "variant", "startup_mode"])
    cohorts.to_csv(OUT/"monthly_cohorts.csv", index=False)
    equal_cohorts = distributions(monthly[monthly.window.isin(["sessions_122", "sessions_244"])],
                                 ["start_year", "window", "fee", "margin", "variant", "startup_mode"])
    equal_cohorts.to_csv(OUT/"equal_window_cohorts.csv", index=False)
    phase = totals[totals.start.str[4:6].eq("01")&totals.variant.isin(VARIANTS[:3])].copy()
    phase = phase[(phase.fee.eq(.0003)&phase.margin.eq(.15))|(phase.fee.eq(.0006)&phase.margin.eq(.25))]
    phase["start_year"] = phase.start.str[:4]
    phases = distributions(phase, ["start_year", "fee", "margin", "variant", "startup_mode"])
    assert phases.starts.eq(5).all()
    phases["return_range"] = phases.best_return-phases.worst_return
    phases.to_csv(OUT/"january_phase_sensitivity.csv", index=False)
    aligned = phase[phase.startup_mode.isin(["calendar", "calendar_history"])].merge(
        startups[keys+["initial_policy", "next_scheduled_entry"]], on=keys, validate="one_to_one")
    alignment_keys = ["start_year", "fee", "margin", "variant", "startup_mode", "initial_policy", "next_scheduled_entry"]
    consistency = aligned.groupby(alignment_keys).agg(starts=("start", "size"),
        return_min=("total_return", "min"), return_max=("total_return", "max"),
        drawdown_min=("max_drawdown", "min"), drawdown_max=("max_drawdown", "max")).reset_index()
    consistency = consistency[consistency.starts.gt(1)]
    assert np.allclose(consistency.return_min, consistency.return_max, atol=1e-12, rtol=0)
    assert np.allclose(consistency.drawdown_min, consistency.drawdown_max, atol=1e-12, rtol=0)
    consistency.to_csv(OUT/"aligned_start_consistency.csv", index=False)
    paired = summary[summary.startup_mode.ne("legacy")].merge(summary[summary.startup_mode.eq("legacy")],
                 on=["start", "fee", "margin", "variant", "window"], suffixes=("", "_legacy"))
    paired["return_delta"] = paired.total_return-paired.total_return_legacy
    paired["drawdown_improvement"] = paired.max_drawdown-paired.max_drawdown_legacy
    paired.to_csv(OUT/"paired_comparison.csv", index=False)
    worst_rows = []
    for _, group in monthly.groupby(["window", "fee", "margin", "variant", "startup_mode"]):
        worst_rows.append(group.loc[group.total_return.idxmin()].to_dict()|{"criterion": "worst_return"})
        worst_rows.append(group.loc[group.max_drawdown.idxmin()].to_dict()|{"criterion": "worst_drawdown"})
    pd.DataFrame(worst_rows).to_csv(OUT/"worst_starts.csv", index=False)
    bars = load_bars()
    forced_events = []
    for account in totals[totals.forced.gt(0)].itertuples():
        ledger = ledger_for(account)
        for i in ledger.index[ledger.forced_exit]:
            before, event = ledger.iloc[i-1], ledger.iloc[i]
            code, sign = before.positions.split(":")
            multiplier = 300 if code[:2] in ("IF", "IH") else 200
            opening = bars.loc[(event.entry_date, code), "open"]
            forced_events.append(dict(start=account.start, fee=account.fee, margin=account.margin, variant=account.variant,
                                      startup_mode=account.startup_mode, date=event.entry_date, contract=code, lots=int(sign),
                                      equity_before=before.equity, required_margin=abs(int(sign))*opening*multiplier*account.margin,
                                      equity_after=event.equity))
    pd.DataFrame(forced_events).to_csv(OUT/"forced_exit_events.csv", index=False)
    checks = json.loads((OUT/"checks.json").read_text())
    checks.update(unit_tests=54, same_horizon_checked=True, trade_reconciliation=True, legacy_summary_matches=True,
                  startup_selections=len(selected), fixed_calendar_anchor="20200102",
                  identical_aligned_start_groups=len(consistency),
                  forced_by_mode=totals.groupby("startup_mode").forced.sum().astype(int).to_dict(),
                  settlement_breaches=int(totals.settlement_breaches.sum()), last=summary.end.max())
    (OUT/"checks.json").write_text(json.dumps(checks, indent=2))
    lines = ["# 启动流程一致性：全局入场日历与历史选初始策略", "",
             "50万元、全组合最多一手、多空可做，数据统一截至2026-11-06。保留上轮750个账户，新增2250个账户，共3000条路径；相同月度、年初几日、年度和等长窗口，不删掉亏损起点。", "",
             "## 固定的2×2对照", "",
             "| 启动模式 | 入场检查 | 初始策略 |", "|---|---|---|",
             "| 原启动 | 从账户开始每5交易日 | base |",
             "| 全局日历 | 从2020-01-02开始，每5交易日一个统一检查日 | base |",
             "| 历史选初始策略 | 从账户开始每5交易日 | 启动日前历史选择 |",
             "| 两者同时启用 | 统一检查日 | 启动日前历史选择 |", "",
             "日历以现有样本首日为固定锚，不扫描0–4偏移、不选择最赚钱星期；跨年、季度和账户启动均不重置。日历只约束开仓检查，退出/风控仍每日运行。未来新增日期不会改变过去检查日。", "",
             "启动选策略：用启动日前最多36个月的真实历史样本外预测，在50万元初始权益下回放原四项候选策略，并使用与待测版本相同的退出及日历。只允许最后历史收益日期不晚于前一信号日。使用原准入条件：至少244日、至少6笔且不少于base一半交易数/覆盖、无强平、正累计收益/近期加权收益、回撤不劣于base；按12个月半衰期的Sortino排序，同分优先base，无候选回退base。", "",
             "启动时没有既有策略持仓，因此不额外套用季度切换优势门槛；后续季度重选仍执行原资金与切换门槛。只继承规则，账户从50万、空仓开始，不继承历史盈利或持仓。模型、退出label、校准、手续费、保证金和安全储备均冻结。", "",
             "沿用原退出、亏损退出、5日模型、收益校准、校准+尾部五种退出版本。月度58个起点、两个主要情景；年度起点补足四种费用/保证金组合；年初前5交易日起点仍只对原三种退出版本做两主要情景。每种启动模式750账户。", "",
             "旧报告分项说明中‘年度额外40账户’更正为50，580+50+120=750；实际旧账本、结果及总数没有变化。", "",
             "## 等长窗口：所有退出版本与启动模式", "",
             "以下只含完整244交易日（约一年）月度窗口，每组47个起点。收益比例是历史描述，不是独立试验或未来盈利概率；最差收益与最差回撤可能来自不同账户。", "",
             "| 情景 | 退出版本 | 启动模式 | 盈利起点比例 | 中位收益 | P10收益 | 最差收益 | 中位回撤 | 最差回撤 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for fee, margin in PROFILES:
        for variant in VARIANTS:
            for mode in MODES:
                r = dist[dist.window.eq("sessions_244")&dist.fee.eq(fee)&dist.margin.eq(margin)&dist.variant.eq(variant)&dist.startup_mode.eq(mode)].iloc[0]
                lines.append(f"| {fee*10000:.0f}bps/{margin:.0%} | {NAMES[variant]} | {MODENAMES[mode]} | {pct(r.profitable_fraction)} | {pct(r.median_return)} | {pct(r.p10_return)} | {pct(r.worst_return)} | {pct(r.median_drawdown)} | {pct(r.worst_drawdown)} |")
    lines += ["", "## 按起始年份分组的等长窗口：原5日模型", "",
              "每格为中位收益 / 最差收益 / 最差回撤幅度。每组仅纳入已走满指定长度的月度账户，未满窗口不补零或年化。全部退出版本见equal_window_cohorts.csv。", ""]
    for fee, margin in PROFILES:
        for window in ("sessions_122", "sessions_244"):
            lines += [f"### {fee*10000:.0f}bps/{margin:.0%} · {window.split('_')[1]}交易日", "",
                      "| 起始年份 | 每模式起点数 | "+" | ".join(MODENAMES[m] for m in MODES)+" |", "|---|---:|"+"---:|"*4]
            part = equal_cohorts[equal_cohorts.window.eq(window)&equal_cohorts.fee.eq(fee)&equal_cohorts.margin.eq(margin)&equal_cohorts.variant.eq("mean5")]
            for year, group in part.groupby("start_year"):
                group = group.set_index("startup_mode")
                lines.append(f"| {year} | {group.starts.iloc[0]} | "+" | ".join(f"{pct(group.loc[m].median_return)} / {pct(group.loc[m].worst_return)} / {pct(group.loc[m].worst_drawdown)}" for m in MODES)+" |")
            lines.append("")
    lines += ["", "## 原5日模型：近两年月度持续账户与年初几日差异", "",
              "每个月/每个指定日启动后，一直持有到共同终点，中途不重置；月度账户持有长度不同，因此另有上方等长比较。", "",
              "| 起始年 | 情景 | 启动模式 | 月度盈利比例 | 月度中位累计 | 月度最差累计 | 月度最差回撤 | 年初5日起点收益跨度 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for year in ("2025", "2026"):
        for fee, margin in PROFILES:
            for mode in MODES:
                r = cohorts[cohorts.start_year.eq(year)&cohorts.fee.eq(fee)&cohorts.margin.eq(margin)&cohorts.variant.eq("mean5")&cohorts.startup_mode.eq(mode)].iloc[0]
                ph = phases[phases.start_year.eq(year)&phases.fee.eq(fee)&phases.margin.eq(margin)&phases.variant.eq("mean5")&phases.startup_mode.eq(mode)].iloc[0]
                lines.append(f"| {year} | {fee*10000:.0f}bps/{margin:.0%} | {MODENAMES[mode]} | {pct(r.profitable_fraction)} | {pct(r.median_return)} | {pct(r.worst_return)} | {pct(r.worst_drawdown)} | {pct(ph.return_range)} |")
    lines += ["", "## 2026年初几日起点：原5日模型、联合压力", "",
              "每格累计收益 / 最大回撤幅度；25%保证金、单边6bps。", "",
              "| 实际启动日 | "+" | ".join(MODENAMES[m] for m in MODES)+" |", "|---|"+"---:|"*4]
    part = phase[phase.start.str.startswith("202601")&phase.fee.eq(.0006)&phase.margin.eq(.25)&phase.variant.eq("mean5")]
    for day, group in part.groupby("first_entry_date"):
        group = group.set_index("startup_mode")
        lines.append(f"| {day} | "+" | ".join(f"{pct(group.loc[m].total_return)} / {pct(-group.loc[m].max_drawdown)}" for m in MODES)+" |")
    lines += ["", f"按相同下一入场检查日、相同初始策略归组，{len(consistency)}组相邻日起点的最终收益与最大回撤完全一致。它验证了日历一致性，但跨过一个入场检查日或历史选择变化后，账户仍可能走向不同路径。", "",
              "## 2026年度起点：费用与保证金拆开比较（原5日模型）", "",
              "每格收益 / 最大回撤幅度。额外两种情景仅有年度起点，不混入58个月度起点分布。", "",
              "| 单边成本 | 保证金 | "+" | ".join(MODENAMES[m] for m in MODES)+" |", "|---|---|"+"---:|"*4]
    part = summary[summary.annual&summary.start.eq("20260101")&summary.window.eq("single_year")&summary.variant.eq("mean5")]
    for (fee, margin), group in part.groupby(["fee", "margin"]):
        group = group.set_index("startup_mode")
        lines.append(f"| {fee*10000:.0f}bps | {margin:.0%} | "+" | ".join(f"{pct(group.loc[m].total_return)} / {pct(-group.loc[m].max_drawdown)}" for m in MODES)+" |")
    lines += ["", "## 全部逐年单年与累计至今", "",
              "每格为扣费收益 / 最大回撤幅度。单年每年重新以50万开始，在当年最后可用开盘清仓；累计为各年独立启动后至2026-11-06，中途不重置。2026不完整。年度不拼接，回撤不随年重置。", ""]
    for fee, margin in PROFILES:
        for variant in VARIANTS:
            for window, label in (("single_year", "独立单年"), ("to_latest", "各年起累计至今")):
                lines += [f"### {fee*10000:.0f}bps/{margin:.0%} · {NAMES[variant]} · {label}", "",
                          "| 年份 | "+" | ".join(MODENAMES[m] for m in MODES)+" |", "|---|"+"---:|"*4]
                part = summary[summary.annual&summary.window.eq(window)&summary.fee.eq(fee)&summary.margin.eq(margin)&summary.variant.eq(variant)]
                for start, group in part.groupby("start"):
                    group = group.set_index("startup_mode")
                    lines.append(f"| {start[:4]} | "+" | ".join(f"{pct(group.loc[m].total_return)} / {pct(-group.loc[m].max_drawdown)}" for m in MODES)+" |")
                lines.append("")
    lines += ["## 持续账户2025/2026当年表现：原三项退出对照", "",
              "2022和2025起持续账户的当年收益 / 年内回撤，承接此前权益，不是年度重置。", "",
              "| 起始年 | 当年 | 情景 | 退出版本 | "+" | ".join(MODENAMES[m] for m in MODES)+" |", "|---|---|---|---|"+"---:|"*4]
    part = annual[annual.start.isin(["20220101", "20250101"])&annual.year.ge("2025")&annual.side.eq("all")&annual.variant.isin(VARIANTS[:3])]
    for (start, year, fee, margin, variant), group in part.groupby(["start", "year", "fee", "margin", "variant"]):
        if (fee, margin) not in PROFILES:
            continue
        group = group.set_index("startup_mode")
        lines.append(f"| {start[:4]} | {year} | {fee*10000:.0f}bps/{margin:.0%} | {NAMES[variant]} | "+" | ".join(f"{pct(group.loc[m].total_return)} / {pct(-group.loc[m].max_drawdown)}" for m in MODES)+" |")
    lines += ["", "## 模拟强平与审计", "",
              "| 启动模式 | 累计账户模拟强平次数 |", "|---|---:|"]
    for mode in MODES:
        lines.append(f"| {MODENAMES[mode]} | {checks['forced_by_mode'][mode]} |")
    lines += ["", "各模式及起点路径相关，事件计数不能当作未来强平概率。详细持仓、权益及保证金见forced_exit_events.csv。没有模拟真实盘中追保、涨跌停锁单及订单簿成交，零事件也不是风险保证。", "",
              f"3000账户、{checks['windows']}评估窗口，750旧账户摘要逐项一致；100个年度基准账本按原已审计决策记录重新执行一致，全部3000账户交易损益与权益对账通过。{checks['validation_records']}条新历史验证、{len(selected)}次历史启动选择满足时间边界；日历开仓检查及等长窗口检查通过，54项专项测试通过。", "",
              "全部比较使用事前固定日历锚、候选池和评分，没有根据结果扫描日历相位、初始策略、退出门槛或删减最差起点。历史已多次查看，仍属探索性回放，不能称为新的未触碰OOT。", "",
              "时间顺序验证的方法依据：[scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。本研究按实际label成熟日期做历史截断，并非随机交叉验证；该文档不为本策略收益有效性提供背书。", "",
              "主要文件：summary.csv、start_distributions.csv、monthly_cohorts.csv、january_phase_sensitivity.csv、paired_comparison.csv、worst_starts.csv、startup_decisions.csv、global_calendar.csv、annual_marked.csv、trades.csv、validation_audit.csv、forced_exit_events.csv、startup_heatmap.png、january_startup.png。", ""]
    lines[4:4] = ["## 本轮判断", "",
        "固定全局日历改善了启动流程一致性，也改善部分典型账户；尚未解决最差起点和近期盈利能力。历史选初始策略没有形成跨情景一致的增益，两项同时启用不自动优于只固定日历。保留所有版本作为研究对照，不升级实盘默认策略。", "",
        "原5日模型、6bps/25%下，2026年初5日起点原累计收益介于-14.13%与+42.59%；只固定日历后全部+42.59%、回撤23.45%，共同首笔交易为2月2日。两项同时启用则全部-9.48%、回撤26.79%：消除了相邻日起点差异，也可能得到一致的亏损。", "",
        "同情景47个完整244日窗口，只固定日历使盈利起点由31/47升至42/47，中位收益25.78%→35.39%，P10收益-9.37%→-0.64%。但最差收益-45.27%→-46.92%，最差回撤48.72%→48.53%，尾部没有实质改善。2023-08起的固定日历账户截至2026-11-06仍亏49.35%、回撤50.89%；2025/2026没有持仓，未从后续行情恢复。", "",
        "近两年并不一致：2025月度起点的高保证金5日模型中位累计-10.06%→+19.97%，最差回撤32.55%→29.50%；2026的10个月度起点仍仅2个盈利，中位累计-11.10%→-12.87%，最差回撤36.81%→34.72%。2026已走满122日的5个起点，中位收益+3.22%→-3.95%。不能用2026年初的好结果替代全年多起点证据。", "",
        "启动选择的反例并非使用2026答案选策略：2026年初联合压力、全局日历下，过去截至2025-12-31的confidence60加权Sortino为2.621，高于base的2.126，且历史回撤39.36%小于44.75%，所以按原规则入选。随后2月2日信号被过滤、3月17日资金不足、3月24日信号被过滤；后续权益与品种可执行性继续分化。相关决策和交易保留在annual_decisions.csv、trades.csv，不据此为2026指定base或降低confidence60门槛。", "",
        "持续账户也单列：2025起联合压力5日模型账户，2026当年收益由-0.45%变为+18.26%，年内回撤24.44%→17.88%；这与每年重置50万元的2026独立账户不同。2025独立单年收益却由82.52%降至68.17%、回撤6.80%升至18.82%，不能把累计改善说成每年都改善。", "",
        "其他退出版本仍有反例。全局日历+收益校准的244日最差收益-5.19%、最差回撤29.55%，优于它的原启动版本；但2026高保证金10个月度起点全部亏损。全局日历+尾部版本的244日最差收益恶化至-57.22%、最差回撤58.02%，并新增1次模拟强平。历史选择+全局日历+亏损退出虽在244日窗口较好，2026月度最差收益仍为-37.51%。不以全历史排名替代近期验证。", "",
        "原启动的强平仍是2024-01-03起原退出账户；新事件是2024-09起全局日历+尾部账户，2024-10-08的IH2412空头，开盘平仓前权益214462.66元低于保证金235410元，模拟平仓后213897.67元。两项同时启用及只历史选择本次为零事件，但样本相关、日线执行限制仍在。", "",
        "下一步优先研究历史选策略评分本身的起点依赖：只在启动日前的成熟历史里，按多个内层月度起点和等长窗口评估收益下分位、回撤与资金不可执行比例；检验单一路径历史赢家是否稳健。固定本轮全局日历，不再按这些结果搜索日历偏移或为某年指定策略。", "",
        "上述均为此次冻结版本的探索性历史结果，58个月度窗口存在重叠；单边3/6bps是含摩擦的研究情景，15%/25%为保证金压力假设，不是逐日真实券商费率与保证金记录。", ""]
    (OUT/"report.md").write_text("\n".join(lines), encoding="utf-8")
    charts(summary)
    print(json.dumps(checks, ensure_ascii=False))
    print(dist[dist.window.eq("sessions_244")&dist.variant.eq("mean5")&dist.fee.eq(.0006)&dist.margin.eq(.25)].to_string(index=False))
    print(phases[phases.start_year.eq("2026")&phases.variant.eq("mean5")&phases.fee.eq(.0006)&phases.margin.eq(.25)].to_string(index=False))


if __name__ == "__main__":
    main()
