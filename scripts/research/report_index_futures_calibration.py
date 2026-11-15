"""Calibration coverage and start-date distributions, with all failures retained."""
import json
import os

import numpy as np
import pandas as pd

from evaluate_index_futures_calibration import OUT, VARIANTS, PROFILES, CAPITAL, episode_weights, weighted_quantile
from evaluate_index_futures_directions import load_bars

NAMES = dict(control="原方案", loss_exit="亏损退出", mean5="原5日模型", calibrated5="收益校准", tail5="校准+尾部约束")


def pct(value):
    return "—" if pd.isna(value) else f"{value:.2%}"


def prediction_metrics(frame):
    rows = []
    frame = frame[frame.date.ge("20220101")&frame.maturity.notna()].copy()
    frame["year"] = frame.date.str[:4]
    for (year, direction), group in frame.groupby(["year", "direction"]):
        for bps in (3, 6):
            ready = group[group[f"cal_ready_{bps}"]].dropna(subset=[f"risk90_{bps}", f"net_5_{bps}", f"risk_5_{bps}"])
            if ready.empty:
                continue
            w = episode_weights(ready)
            y, risk = ready[f"net_5_{bps}"], ready[f"risk_5_{bps}"]
            bound, lower = ready[f"risk90_{bps}"], ready[f"net10_{bps}"]
            detail = dict(year=year, direction=direction, bps=bps, states=len(ready), episodes=ready.episode.nunique(),
                          mean_actual_net=np.average(y, weights=w), mean_actual_risk=np.average(risk, weights=w),
                          risk90_coverage=np.average(risk.le(bound), weights=w), net10_coverage=np.average(y.ge(lower), weights=w),
                          risk90_average=np.average(bound, weights=w),
                          excess_above_bound=np.average(np.maximum(0, risk-bound), weights=w),
                          realized_risk90=weighted_quantile(risk, w, .9))
            for name, predicted in (("raw", ready[f"pred_net_5_{bps}"]), ("calibrated", ready[f"cal_net_{bps}"])):
                mse = np.average((predicted-y)**2, weights=w)
                detail.update({f"{name}_bias": np.average(predicted-y, weights=w),
                               f"{name}_mae": np.average(abs(predicted-y), weights=w),
                               f"{name}_mse": mse,
                               f"{name}_mse_skill_vs_zero": 1-mse/np.average(y*y, weights=w)})
            for name, predicted in (("raw", ready[f"pred_risk_5_{bps}"]), ("calibrated", ready[f"cal_risk_{bps}"])):
                detail[f"{name}_risk_bias"] = np.average(predicted-risk, weights=w)
                detail[f"{name}_risk_mae"] = np.average(abs(predicted-risk), weights=w)
            rows.append(detail)
    return pd.DataFrame(rows)


def distributions(summary):
    rows = []
    monthly = summary[summary.start_kind.eq("monthly")]
    for keys, group in monthly.groupby(["window", "fee", "margin", "variant"]):
        window, fee, margin, variant = keys
        dd = -group.max_drawdown
        rows.append(dict(window=window, fee=fee, margin=margin, variant=variant, starts=len(group),
                         profitable_fraction=group.total_return.gt(0).mean(), median_return=group.total_return.median(),
                         p10_return=group.total_return.quantile(.1), worst_return=group.total_return.min(),
                         best_return=group.total_return.max(), median_drawdown=dd.median(), p90_drawdown=dd.quantile(.9),
                         worst_drawdown=dd.max(), zero_trade_accounts=group.trades.eq(0).sum(),
                         median_capital_blocks=group.capital_blocks.median(), median_active_days=group.active_days.median()))
    return pd.DataFrame(rows)


def charts(summary):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant_adaptation_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    english = ["Original", "Loss exit", "Raw 5-day", "Calibrated", "Calibrated + tail"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.7))
    samples = summary[summary.start_kind.eq("monthly")&summary.window.eq("sessions_244")]
    maximum = max(10, samples.total_return.abs().max()*100)
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    for ax, (fee, margin) in zip(axes, PROFILES):
        table = samples[samples.fee.eq(fee)&samples.margin.eq(margin)].pivot(index="variant", columns="start", values="total_return").reindex(VARIANTS)
        im = ax.imshow(table.to_numpy()*100, aspect="auto", cmap="RdBu", norm=norm)
        ticks = np.arange(0, len(table.columns), 3)
        ax.set_xticks(ticks, [f"{s[:4]}-{s[4:6]}" for s in table.columns[ticks]], rotation=35, ha="right")
        ax.set_yticks(range(5), english)
        ax.set_title(f"{fee*10000:.0f} bps per side, {margin:.0%} margin | each cell starts at CNY 500,000")
    fig.suptitle("Start-date sensitivity: net return over the SAME 244 trading sessions", fontsize=15)
    fig.text(.5, .915, "Monthly starts; overlapping windows are not independent trials; no start selected by outcome", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, .93, .90))
    cax = fig.add_axes([.94, .16, .012, .64])
    fig.colorbar(im, cax=cax, label="Net return (%)")
    fig.savefig(OUT/"start_sensitivity.png", dpi=180)
    plt.close(fig)


def main():
    summary = pd.read_csv(OUT/"summary.csv", dtype={"start": str, "end": str, "first_entry_date": str})
    annual = pd.read_csv(OUT/"annual_marked.csv", dtype={"start": str, "year": str})
    trades = pd.read_csv(OUT/"trades.csv", dtype={"start": str, "entry": str, "exit": str})
    audit = pd.read_csv(OUT/"validation_audit.csv", dtype={"date": str, "cutoff": str, "last_validation_exit": str})
    calibration = pd.read_csv(OUT/"calibration_audit.csv", dtype={"cutoff": str, "last_maturity": str})
    forecasts = pd.read_parquet(OUT/"calibrated_forecasts.parquet")
    trace = pd.read_csv(OUT/"annual_decisions.csv", dtype={"date": str, "signal_date": str})
    assert audit.last_validation_exit.le(audit.cutoff).all() and audit.cutoff.lt(audit.date).all()
    ready = calibration[calibration.ready]
    assert ready.last_maturity.lt(ready.cutoff).all()
    assert trace.signal_date.lt(trace.date).all()
    keys = ["start", "fee", "margin", "variant"]
    continuous = summary[summary.window.eq("to_latest")]
    assert len(continuous) == 750
    reconcile = continuous.merge(trades.groupby(keys).pnl.sum().rename("net"), on=keys, how="left")
    np.testing.assert_allclose(reconcile.net.fillna(0), reconcile.total_return*CAPITAL, atol=1e-6, rtol=0)
    for n in (122, 244):
        assert summary.loc[summary.window.eq(f"sessions_{n}"), "days"].eq(n).all()
    quality = prediction_metrics(forecasts)
    quality.to_csv(OUT/"prediction_quality.csv", index=False)
    distribution = distributions(summary)
    distribution.to_csv(OUT/"start_distributions.csv", index=False)
    monthly = continuous[continuous.start_kind.eq("monthly")].copy()
    monthly["start_year"] = monthly.start.str[:4]
    cohorts = monthly.groupby(["start_year", "fee", "margin", "variant"]).agg(
        starts=("total_return", "size"), median_return=("total_return", "median"),
        worst_return=("total_return", "min"), best_return=("total_return", "max"),
        profitable_fraction=("total_return", lambda x: x.gt(0).mean()),
        worst_drawdown=("max_drawdown", lambda x: -x.min()), median_days=("days", "median")).reset_index()
    cohorts.to_csv(OUT/"monthly_cohorts.csv", index=False)
    phase = continuous[continuous.variant.isin(VARIANTS[:3])&(continuous.start.str[4:6].eq("01"))]
    phase = phase[((phase.fee.eq(.0003)&phase.margin.eq(.15))|(phase.fee.eq(.0006)&phase.margin.eq(.25)))].copy()
    phase["start_year"] = phase.start.str[:4]
    phase_table = phase.groupby(["start_year", "fee", "margin", "variant"]).agg(
        starts=("total_return", "size"), min_return=("total_return", "min"), max_return=("total_return", "max"),
        worst_drawdown=("max_drawdown", lambda x: -x.min())).reset_index()
    assert phase_table.starts.eq(5).all()
    phase_table["return_range"] = phase_table.max_return-phase_table.min_return
    phase_table.to_csv(OUT/"january_phase_sensitivity.csv", index=False)
    comparisons = []
    for reference in VARIANTS[:3]:
        base = summary[summary.variant.eq(reference)]
        compare = summary[summary.variant.isin(VARIANTS[3:])].merge(base, on=["start", "fee", "margin", "window"], suffixes=("", "_reference"))
        compare["reference"] = reference
        compare["return_delta"] = compare.total_return-compare.total_return_reference
        compare["drawdown_improvement"] = compare.max_drawdown-compare.max_drawdown_reference
        comparisons.append(compare)
    pd.concat(comparisons, ignore_index=True).to_csv(OUT/"paired_comparison.csv", index=False)
    checks = json.loads((OUT/"checks.json").read_text())
    checks.update(calibration_fits=int(calibration.ready.sum()), unit_tests=47,
                  monthly_starts=monthly.start.nunique(), phase_accounts=continuous.start_kind.eq("phase").sum().item(),
                  forced_exits=int(continuous.forced.sum()), settlement_breaches=int(continuous.settlement_breaches.sum()),
                  last=summary.end.max(), same_horizon_checked=True, trade_reconciliation=True)
    (OUT/"checks.json").write_text(json.dumps(checks, indent=2))
    forced_events = []
    bars = load_bars()
    for account in continuous[continuous.forced.gt(0)].itertuples():
        ledger = pd.read_parquet(OUT/account.start[:4]/"ledgers"/f"{account.start}_{account.variant}_{account.fee}_{account.margin}.parquet")
        for index in ledger.index[ledger.forced_exit]:
            before, event = ledger.iloc[index-1], ledger.iloc[index]
            code, side = before.positions.split(":")
            multiplier = 300 if code[:2] in ("IF", "IH") else 200
            quote = bars.loc[(event.entry_date, code)]
            previous_close = bars.loc[(before.entry_date, code), "close"]
            forced_events.append(dict(start=account.start, variant=account.variant, fee=account.fee, margin=account.margin,
                                      date=event.entry_date, code=code, side=int(side), equity_before=before.equity,
                                      required_margin=abs(int(side))*quote.open*multiplier*account.margin,
                                      prior_close=previous_close, current_open=quote.open,
                                      gap_pnl=int(side)*(quote.open-previous_close)*multiplier, fees=event.fees,
                                      equity_after=event.equity))
    pd.DataFrame(forced_events).to_csv(OUT/"forced_exit_events.csv", index=False)
    lines = ["# 收益校准、漂移、尾部风险与起点稳定性", "",
             "初始50万元、全组合最多一手，多空均允许；统一数据截至2026-11-06。保留原方案、5%亏损退出、原5日模型，新增收益校准及校准+尾部约束两个固定版本。750条持续账户路径，不追加资金、不借用另一账户的盈利。", "",
             "**结论：当前流程起点稳定性不足。原5日模型在联合压力情景，2026-01-05开始赚42.59%，晚一天开始亏10.51%；2026年10个月度起点只有2个盈利。校准改善预测误差，但没有稳定修复账户表现；尾部约束有收益/风险取舍，也出现明显失败。扩大起点后发现1次模拟强平，不能再沿用之前少数起点零强平的结论。**", "",
             "## 评测设计：不挑起点", "",
             "- 每月首个可交易日启动：2022-01至2026-10，共58个起点，5版本×2主要情景，共580账户。每年1月起点另补两种费用/保证金组合50账户。",
             "- 每年1月前5个交易日分别启动：三个原对照、两主要情景。首日已包含在月度账户，新增120账户。这个实验衡量相差几天的敏感性，不能把差异全归因于一手容量。",
             "- 各起点累计至今，以及固定122/244个交易日（约半年/一年）两种完整窗口。短于固定长度的起点不进入该表，不把短样本折算为全年。年度表仍为各年独立50万元；持续账户另列承接权益的当年收益。",
             "", "每个新账户以base策略、空仓和新的每5交易日入场时钟开始；模型训练日历固定，中途继承资金/持仓并按季度重选。因此起点差异包含初始入场时机、入场节奏、初始策略及整手资金门槛的共同影响。一个月开始的账户不是随后每月重置。重叠窗口高度相关，盈利起点比例不是未来盈利概率。", "",
             "## 校准、尾部与漂移的做法", "",
             "只使用此前24个月、完整label已成熟的原历史季度样本外预测。分多空、分3/6bps费用，12个月半衰期；按候选交易归一状态权重。至少122个不同成熟信号日、60个候选事件才能启用，否则延用原5日模型、不开启尾部约束。", "",
             "多头校准从2021Q4可用，空头从2023Q4可用；早期缺少样本的阶段不解释为校准成功。", "",
             "收益校准：加权拟合actual=a+b×raw_prediction，b限制在[0,1]，不通过反向模型或放大斜率追逐历史。风险均值同样校准。实际交易继续使用净增量预测连续两次≤0后次日开盘退出。", "",
             "尾部上界：原风险预测+过去样本外风险残差的加权90%分位数，且不低于校准风险均值。另计算净收益残差10%下界作诊断。残差来自原来的真实历史样本外预测，不是把当前校准器拟合后再对自己的训练残差评估。", "",
             "尾部交易约束：90%路径损失估计×上一收盘一手名义金额，超过上一收盘实际账户权益的5%，连续两次后退出；校准收益退出条件同样保留。当前开盘跳空从权益中剔除，风险决策只使用当时收盘已知信息。5%是固定研究预算，不是最大实际损失承诺。", "",
             "漂移监测：每季度截止时记录过去最近6个月相对更早18个月的真实历史预测偏差；另对已知持仓第5日的预测值和波动率计算分布变化。漂移指标仅用于诊断，本轮没有按看到的年度结果动态开关模型。", "",
             "这些是经验残差分位数，不是有分布无关保证的conformal区间；时间相关、状态重叠和分布漂移会破坏简单覆盖解释。需要看之后真实覆盖率。参考[分布漂移下预测区间研究](https://proceedings.neurips.cc/paper_files/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf)；本实验没有实现该论文的ACI算法，也不借用其保证。", "",
             "## 每年单年与各年起累计至今", "",
             "每格为扣费收益 / 最大回撤幅度。单年每年独立50万，在该年最后可用开盘平仓；累计各行也是独立50万，持有至共同终点。2026不完整。3/6bps是单边综合成本情景，15%/25%保证金另加原安全储备。", ""]
    for fee, margin in PROFILES:
        for window, title in (("single_year", "独立单年"), ("to_latest", "各年起累计至今")):
            lines += [f"### {fee*10000:.0f}bps/{margin:.0%}：{title}", "", "| 年份 | "+" | ".join(NAMES[v] for v in VARIANTS)+" |", "|---|"+"---:|"*5]
            part = summary[summary.annual&summary.window.eq(window)&summary.fee.eq(fee)&summary.margin.eq(margin)]
            for start, group in part.groupby("start"):
                group = group.set_index("variant")
                lines.append(f"| {start[:4]} | "+" | ".join(f"{pct(group.loc[v].total_return)} / {pct(-group.loc[v].max_drawdown)}" for v in VARIANTS)+" |")
            lines.append("")
    lines += ["## 换起点以后：相同持有长度的分布", "",
              "不使用累计至今混合不同长度来判断收益稳定性。‘最差回撤’取该组所有起点账户中的最大回撤幅度，‘最差收益’未必来自同一个账户。", "",
              "| 长度 | 情景 | 版本 | 起点数 | 盈利起点比例 | 中位收益 | P10收益 | 最差收益 | 中位回撤 | 最差回撤 |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    part = distribution[distribution.window.isin(["sessions_122", "sessions_244"])]
    for r in part.itertuples():
        if (r.fee, r.margin) not in PROFILES:
            continue
        lines.append(f"| {r.window} | {r.fee*10000:.0f}bps/{r.margin:.0%} | {NAMES[r.variant]} | {r.starts} | {pct(r.profitable_fraction)} | {pct(r.median_return)} | {pct(r.p10_return)} | {pct(r.worst_return)} | {pct(r.median_drawdown)} | {pct(r.worst_drawdown)} |")
    lines += ["", "## 近两年不同月份启动的持续账户", "",
              "各起点一直运行到2026-11-06，不按年重置；同年起点的持有长度仍不同，所以这是启动敏感性描述，不取代上面的等长窗口。", "",
              "| 起始年 | 情景 | 版本 | 起点数 | 盈利起点比例 | 中位累计收益 | 最差累计收益 | 最好累计收益 | 最差回撤 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in cohorts[cohorts.start_year.ge("2025")].itertuples():
        if (r.fee, r.margin) not in PROFILES:
            continue
        lines.append(f"| {r.start_year} | {r.fee*10000:.0f}bps/{r.margin:.0%} | {NAMES[r.variant]} | {r.starts} | {pct(r.profitable_fraction)} | {pct(r.median_return)} | {pct(r.worst_return)} | {pct(r.best_return)} | {pct(r.worst_drawdown)} |")
    lines += ["", "## 只把年初启动挪动几个交易日", "",
              "每组为1月前5个交易日分别投入50万、持续到终点。仅三个原对照；固定模型训练日历，但启动后入场节奏和权益路径不同。", "",
              "| 年份 | 情景 | 版本 | 最差累计收益 | 最好累计收益 | 收益跨度 | 最差回撤 |", "|---|---|---|---:|---:|---:|---:|"]
    for r in phase_table[phase_table.start_year.ge("2025")].itertuples():
        lines.append(f"| {r.start_year} | {r.fee*10000:.0f}bps/{r.margin:.0%} | {NAMES[r.variant]} | {pct(r.min_return)} | {pct(r.max_return)} | {pct(r.return_range)} | {pct(r.worst_drawdown)} |")
    lines += ["", "## 持续账户近两年当年收益与回撤", "",
              "| 起始年 | 当年 | 情景 | "+" | ".join(NAMES[v] for v in VARIANTS)+" |", "|---|---|---|"+"---:|"*5]
    part = annual[annual.start.isin(["20220101", "20250101"])&annual.year.ge("2025")&annual.side.eq("all")]
    for (start, year, fee, margin), group in part.groupby(["start", "year", "fee", "margin"]):
        if (fee, margin) not in PROFILES:
            continue
        group = group.set_index("variant")
        lines.append(f"| {start[:4]} | {year} | {fee*10000:.0f}bps/{margin:.0%} | "+" | ".join(f"{pct(group.loc[v].total_return)} / {pct(-group.loc[v].max_drawdown)}" for v in VARIANTS)+" |")
    lines += ["", "## 校准与尾部覆盖质量", "",
              "只比较校准已启用的相同状态；按候选交易归一，3bps。偏差=预测−实际；MSE降低不代表一定改善交易。90%路径风险上界及10%净收益下界的理想单侧覆盖均为90%，下表是随后实现的覆盖。", "",
              "| 年份 | 方向 | 原收益偏差 | 校准收益偏差 | 原MAE | 校准MAE | 校准/原MSE | 风险90%上界覆盖 | 净收益10%下界覆盖 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in quality[quality.bps.eq(3)].itertuples():
        lines.append(f"| {r.year} | {'多' if r.direction > 0 else '空'} | {pct(r.raw_bias)} | {pct(r.calibrated_bias)} | {pct(r.raw_mae)} | {pct(r.calibrated_mae)} | {r.calibrated_mse/r.raw_mse:.2f} | {pct(r.risk90_coverage)} | {pct(r.net10_coverage)} |")
    lines += ["", "2025/2026四个方向-年度组的校准收益MSE比原预测降低约23%–59%，但都仍略劣于零预测。2026多头90%风险上界实际只覆盖74.97%，净收益10%下界只覆盖70.36%，尾部仍被明显低估；2025多头风险覆盖98.88%则偏保守。总体覆盖不能代替逐年、分方向覆盖。", "",
              "## 扩大起点后发现的模拟强平", "",
              "| 账户起点 | 版本 | 情景 | 强平日期 | 合约方向 | 开盘权益（元） | 所需保证金（元） | 长假跳空盈亏（元） |", "|---|---|---|---|---|---:|---:|---:|"]
    for r in forced_events:
        lines.append(f"| {r['start']} | {NAMES[r['variant']]} | {r['fee']*10000:.0f}bps/{r['margin']:.0%} | {r['date']} | {r['code']} {r['side']:+d} | {r['equity_before']:,.2f} | {r['required_margin']:,.2f} | {r['gap_pnl']:,.2f} |")
    lines += ["", "此处账户并非归零，而是权益不足以支持现有一手保证金。该IF空单前一收盘4167.6、节后开盘4549，跳空本身亏114420元；开盘权益320758.92元低于所需保证金341175元。上一个结算点没有不足，说明日终检查和既定安全储备不能保证覆盖隔夜跳空。模拟按开盘成交，现实能否及时成交仍未建模。", "",
              "## 下一步的判断标准", "",
              "优先修正启动与执行流程的任意性：例如把入场节奏固定到全局交易日历、启动时仅凭此前历史决定初始策略，再以同一月度/几日起点网格检验。它们是待验证方向，不能根据本次哪个日期赚钱来选启动日。单手不可分割、资金阈值反馈和序贯策略切换仍需保留。", "",
              "校准方向可保留，但不能因为2025起累计盈利就升级。未来评估同时看等长窗口的下十分位收益、最差回撤、近期持续账户及尾部覆盖；阈值只能在历史内层确定，新数据作前瞻检验。当前尾部规则不作为默认风控：它在联合压力244日窗口把最差收益由原5日模型-45.27%改善到-21.80%，却把中位收益25.78%降到15.86%；基准2023起累计甚至亏61.62%。", "",
              "## 边界与审计", "",
              "月度盈利比例、分位数和最差表现均为已观察历史中的描述；窗口重叠，不能解释为未来成功概率、独立统计试验或收益保证。所有起点、窗口、阈值在本轮运行前固定，没有删除亏损起点，也没有以最后收益选启动月份。历史已反复查看，仍非新OOT。", "",
              f"{checks['accounts']}账户、{checks['windows']}窗口，{checks['control_matches']}个既有年度起点对照账本一致；{len(audit)}条历史重选验证和{len(ready)}项有效校准满足成熟日期边界；逐笔交易与净值核对；47项测试通过。模拟强平{checks['forced_exits']}次、结算不足{checks['settlement_breaches']}次；日线模型不能保证盘中追保、涨跌停和成交可执行性。", "",
              "主要文件：summary.csv（全部账户/窗口）、start_distributions.csv（等长起点分布）、monthly_cohorts.csv（各年月份起点）、january_phase_sensitivity.csv（几天之差）、annual_marked.csv（持续账户当年）、prediction_quality.csv、calibration_audit.csv、drift_at_cutoff.csv、calibrated_forecasts.parquet、trades.csv、forced_exit_events.csv、paired_comparison.csv、start_sensitivity.png。", ""]
    (OUT/"report.md").write_text("\n".join(lines), encoding="utf-8")
    charts(summary)
    print(json.dumps(checks, ensure_ascii=False))
    print(distribution[distribution.window.eq("sessions_244")].to_string(index=False))
    print(quality[quality.year.ge("2025")&quality.bps.eq(3)].to_string(index=False))


if __name__ == "__main__":
    main()
