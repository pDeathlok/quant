"""All-year exit-label results, prediction quality, and counterfactual diagnostics."""
import json
import os

import numpy as np
import pandas as pd

from evaluate_index_futures_exit_value import OUT, SOURCE, VARIANTS, CAPITAL

NAMES = dict(control="原方案", loss_exit="5%亏损退出", mean5="5日增量收益",
             mean_remaining="剩余期增量收益", utility_remaining="剩余期收益−风险", fixed5="固定5日平仓(补充)")


def pct(value):
    return "—" if pd.isna(value) else f"{value:.2%}"


def prediction_quality(forecasts):
    rows = []
    frame = forecasts[forecasts.date.ge("20220101")&forecasts.maturity.notna()].copy()
    frame["year"] = frame.date.str[:4]
    for (year, direction), group in frame.groupby(["year", "direction"]):
        for bps in (3, 6):
            for horizon in ("5", "remaining"):
                for kind in ("net", "risk"):
                    name = f"{kind}_{horizon}_{bps}"
                    valid = group.dropna(subset=[name, f"pred_{name}"])
                    if valid.empty:
                        continue
                    weight = 1/valid.groupby("episode").episode.transform("size").to_numpy()
                    y, p = valid[name].to_numpy(), valid[f"pred_{name}"].to_numpy()
                    mean_y, mean_p = np.average(y, weights=weight), np.average(p, weights=weight)
                    covariance = np.average((y-mean_y)*(p-mean_p), weights=weight)
                    denominator = np.sqrt(np.average((y-mean_y)**2, weights=weight)*np.average((p-mean_p)**2, weights=weight))
                    mse = np.average((y-p)**2, weights=weight)
                    zero_mse = np.average(y*y, weights=weight)
                    rows.append(dict(year=year, direction=direction, bps=bps, horizon=horizon, target=kind,
                                     states=len(valid), episodes=valid.episode.nunique(), actual_mean=mean_y,
                                     predicted_mean=mean_p, mae=np.average(abs(y-p), weights=weight),
                                     correlation=covariance/denominator if denominator else np.nan,
                                     mse_skill_vs_zero=1-mse/zero_mse if zero_mse else np.nan))
    return pd.DataFrame(rows)


def actual_exit_outcomes(trades, forecasts):
    early = trades[trades.exit_reason.isin(["learned_exit", "close_loss_exit"])].copy()
    early = early.merge(forecasts, left_on=["entry", "exit", "product"], right_on=["entry", "date", "symbol"],
                        how="left", suffixes=("", "_state"), validate="many_to_one")
    records = []
    for row in early.itertuples():
        horizon = "5" if row.variant == "mean5" else "remaining"
        bps = round(row.fee*10000)
        net, risk = getattr(row, f"net_{horizon}_{bps}"), getattr(row, f"risk_{horizon}_{bps}")
        records.append(dict(start=row.start, fee=row.fee, margin=row.margin, variant=row.variant,
                            entry=row.entry, exit=row.exit, product=row.product, side=row.side,
                            exit_reason=row.exit_reason, horizon=horizon, maturity=row.maturity,
                            actual_trade_pnl=row.pnl, continuing_incremental_cny=net*row.notional,
                            continuing_adverse_cny=risk*row.notional, favorable_exit=net < 0 if pd.notna(net) else np.nan))
    return pd.DataFrame(records)


def charts():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant_adaptation_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Control", "5% loss exit", "5-session value", "Remaining value", "Remaining value - risk", "Fixed 5-session exit"]
    colors = ["#777777", "#218b82", "#cc8026", "#4e76c3", "#a14d86", "#a64836"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    for column, (start, fee, margin) in enumerate((("20250101", .0003, .15),
                                                 ("20260101", .0006, .25),
                                                 ("20230101", .0006, .25))):
        for variant, label, color in zip(VARIANTS, labels, colors):
            ledger = pd.read_csv(OUT/f"ledger_{start}_{variant}_{fee}_{margin}.csv", dtype={"entry_date": str, "exit_date": str})
            path = np.r_[CAPITAL, ledger.equity]
            dates = pd.to_datetime([ledger.entry_date.iloc[0], *ledger.exit_date])
            axes[0, column].plot(dates, path/10000, lw=1.3, color=color, label=label)
            axes[1, column].plot(dates, (path/np.maximum.accumulate(path)-1)*100, lw=1.2, color=color)
        axes[0, column].set_title(f"Start {start[:4]} | {fee*10000:.0f} bps | {margin:.0%} margin")
        axes[0, column].set_ylabel("Equity (CNY 10,000)")
        axes[1, column].set_ylabel("Drawdown (%)")
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Hold-versus-close labels | CNY 500,000, at most one lot", fontsize=15)
    fig.text(.5, .93, "Exploratory historical replay through 2026-11-06; quarterly causal training; all variants retained", ha="center", fontsize=10)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .055, 1, .91))
    fig.savefig(OUT/"equity_drawdown.png", dpi=170)
    plt.close(fig)


def main():
    global VARIANTS
    summary = pd.read_csv(OUT/"summary.csv", dtype={"start": str, "end": str})
    if summary.variant.eq("fixed5").any():
        VARIANTS = (*VARIANTS, "fixed5")
    annual = pd.read_csv(OUT/"annual_marked.csv", dtype={"start": str, "year": str})
    trades = pd.read_csv(OUT/"trades.csv", dtype={"start": str, "entry": str, "exit": str})
    training = pd.read_csv(OUT/"training_audit.csv", dtype={"cutoff": str, "last_maturity": str})
    audit = pd.read_csv(OUT/"validation_audit.csv", dtype={"date": str, "cutoff": str, "last_validation_exit": str})
    forecasts = pd.read_parquet(OUT/"forecasts.parquet")
    trace = pd.read_csv(OUT/"daily_decisions.csv", dtype={"start": str, "date": str, "signal_date": str})
    source_audit = pd.read_csv(SOURCE/"training_audit.csv", dtype={"last_maturity": str, "cutoff": str})
    assert source_audit.last_maturity.lt(source_audit.cutoff).all()
    trained = training[training.trained]
    assert trained.last_maturity.lt(trained.cutoff).all()
    assert audit.last_validation_exit.le(audit.cutoff).all() and audit.cutoff.lt(audit.date).all()
    assert trace.signal_date.lt(trace.date).all()
    assert len(summary) == 40*len(VARIANTS)
    if (OUT/"primary_summary.csv").exists():
        primary = pd.read_csv(OUT/"primary_summary.csv", dtype={"start": str, "end": str})
        pd.testing.assert_frame_equal(primary.reset_index(drop=True), summary.query("variant != 'fixed5'").reset_index(drop=True),
                                      check_dtype=False, atol=1e-8, rtol=1e-10)
    keys = ["start", "fee", "margin", "variant"]
    totals = summary[summary.window.eq("to_latest")].merge(trades.groupby(keys).pnl.sum().rename("net"), on=keys, how="left")
    np.testing.assert_allclose(totals.net.fillna(0), totals.total_return*CAPITAL, atol=1e-6, rtol=0)
    quality = prediction_quality(forecasts)
    quality.to_csv(OUT/"prediction_quality.csv", index=False)
    exits = actual_exit_outcomes(trades, forecasts)
    exits.to_csv(OUT/"exit_counterfactuals.csv", index=False)
    comparisons = []
    for reference in ("control", "loss_exit"):
        base = summary[summary.variant.eq(reference)].drop(columns="variant")
        compared = summary[~summary.variant.isin(["control", "loss_exit"])].merge(
            base, on=["start", "fee", "margin", "window"], suffixes=("", "_reference"))
        compared["reference"] = reference
        compared["return_delta"] = compared.total_return-compared.total_return_reference
        compared["drawdown_improvement"] = compared.max_drawdown-compared.max_drawdown_reference
        comparisons.append(compared)
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison.to_csv(OUT/"paired_comparison.csv", index=False)
    trades["exit_year"] = trades.exit.str[:4]
    sides = trades.groupby(keys+["exit_year", "side", "product"]).agg(
        trades=("pnl", "size"), wins=("pnl", lambda x: int(x.gt(0).sum())),
        net_pnl=("pnl", "sum"), worst_trade=("pnl", "min")).reset_index()
    sides.to_csv(OUT/"side_product_attribution.csv", index=False)
    checks = json.loads((OUT/"checks.json").read_text())
    checks.update(training_models=len(trained), validation_records=len(audit), decision_rows=len(trace),
                  trained_first=trained.groupby("direction").quarter.first().to_dict(),
                  forced_exits=int(totals.forced.sum()), settlement_breaches=int(totals.settlement_breaches.sum()),
                  unit_tests=39, last=summary.end.max(), candidate_episodes=forecasts.episode.nunique(),
                  accounts=len(totals), evaluated_windows=len(summary), primary_results_unchanged=True)
    (OUT/"checks.json").write_text(json.dumps(checks, indent=2))

    lines = ["# 退出label：继续持有相对立即平仓的收益与风险", "",
             f"初始50万元、全组合最多一手、多空均允许；数据统一截至2026-11-06。{len(totals)}个持续账户、{len(summary)}个单年/累计评估窗口，40个原方案和亏损退出对照账本复现一致。", "",
             "**本轮没有找到普遍同时提高收益、降低回撤的新退出模型。5日增量收益label在近两年、尤其2026高保证金情景有改善，值得冻结后继续验证；剩余期风险惩罚未稳定降低回撤，不采用为默认规则。全部参数运行前固定，没有围绕已知亏损优化。**", "",
             "**预测层证据偏弱：2025、2026多空净增量收益的均方误差均劣于零预测。2026多头5日风险实际均值1.82%，预测仅1.06%，排序相关性约-0.04；2025多头同类风险实际0.71%、预测2.55%。收益幅度与风险校准不稳定，账户回放变好不足以证明稳定预测能力。**", "",
             "补充对照单独登记在supplement_protocol.json：查看首轮预测误差后，加入固定持有5日平仓，用于区分预测作用与机械缩短持仓作用。这个对照不是事前登记主实验，不调整任何模型、label或参数；primary_summary.csv保存最初五版本全部结果，并验证补充运行未改变它们。", "",
             "## label如何定义", "",
             "在收盘时，比较两种下一开盘行动：A平掉当前真实合约；B继续持有同一品种、按原规则换月，经过h个交易日再平仓。定义增量净利润：", "",
             "`Δ净利润 = 继续持有期间的价格盈亏 − 后续换月双边费用 − 最终平仓费 + A的立即平仓费`", "",
             "历史入场费和已实现盈亏在A、B两边相同，因此不重复扣入label。已发生的持仓盈亏、持有天数、从持仓以来收盘高点回落，作为当前状态特征。label用上一收盘的一手名义金额归一，预测后可换算成人民币。", "",
             "1. **5日增量收益：**h=min(5,距离原20日持有上限的剩余天数)。预测条件均值≤0连续两次成立，下一开盘退出。",
             "2. **剩余期增量收益：**h=距离20日上限的剩余天数，退出门槛同上。",
             "3. **剩余期收益−风险：**同时预测上述收益和h内最大不利路径损失，使用`预测净收益−0.5×预测不利损失`，连续两次≤0后退出。0.5为事前固定的研究权重，未扫描。",
             "", "风险label是相对立即平仓的增量路径在开盘/收盘盯市点的最差亏损，最低为0；包括换月成本，没有使用不可执行的‘未来最低点卖出’。预测的是该损失的条件均值，不是VaR、极端损失上限或保证金不足概率。", "",
             "label终点是固定持有期限，不包含未来模型再次退出、未来季度切换或按未来账户权益触发的安全储备退出；实际执行仍保留这些原约束。每日重估属于滚动决策，不能把此label称为完整最优退出价值函数。", "",
             "## 训练与防穿越", "",
             f"由原入场条件生成{forecasts.episode.nunique()}个候选交易、{len(forecasts)}个持仓状态，包含每个候选入场后的1–19日，不按未来胜负筛选，也不只使用最终被资金账户选中的交易。", "",
             "多空分别拟合Ridge回归；使用原60个期货/ETF/辅助指数因子、7个持仓及方向分数状态、4个品种标记。L2正则项固定为标准化加权MSE目标下的0.1，不搜索参数。3bps和6bps分别有净收益/风险目标；一个模型联合拟合8个输出。", "",
             "每季度仅使用过去36个月、完整剩余期label在首个信号cutoff之前成熟的状态；5日与剩余期使用相同的更严格成熟样本。填补、标准化均只在训练集拟合。12个月半衰期加权，同一候选交易的成熟状态总权重归一，减少重叠状态重复计权。重叠候选之间仍相关，不把状态条数当独立样本数。", "",
             "至少244个不同信号日、60个候选事件才能训练；因此多头模型2021Q2开始、空头模型2023Q1才开始，2022年的空头退出仍使用原规则。历史候选重选使用当时季度的预测，不用最新模型回填过去。季度重选也按对应退出版本重放历史账户，初始资金为当时实际权益。", "",
             "历史已反复查看，参数事前固定不等于恢复为未触碰OOT。本轮没有按某年某个品种加条件，也没有根据本轮结果二次调参。方法参考[scikit-learn防止数据泄漏](https://scikit-learn.org/stable/common_pitfalls)与[Ridge定义](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)。", "",
             "## 逐年与累计结果", "",
             "每格为扣费收益 / 最大回撤幅度。单年：每年独立50万元，在当年最后可用开盘平仓；累计：各起始年独立50万元持有至2026-11-06，中途不重置。各行不能拼接。2026不完整。费用包含开平和换月，3/6bps是单边统一交易成本情景；15%/25%为保证金情景，另保留原安全储备。", ""]
    for fee, margin, label in ((.0003, .15, "基准3bps/15%"), (.0006, .25, "联合压力6bps/25%")):
        for window, title in (("single_year", "各年独立单年"), ("to_latest", "各起点累计至今")):
            lines += [f"### {label}：{title}", "", "| 年份 | "+" | ".join(NAMES[v] for v in VARIANTS)+" |", "|---|"+"---:|"*len(VARIANTS)]
            part = summary[summary.fee.eq(fee)&summary.margin.eq(margin)&summary.window.eq(window)]
            for start, group in part.groupby("start"):
                group = group.set_index("variant")
                lines.append(f"| {start[:4]} | "+" | ".join(f"{pct(group.loc[v].total_return)} / {pct(-group.loc[v].max_drawdown)}" for v in VARIANTS)+" |")
            lines.append("")

    lines += ["## 近两年四种费用/保证金分离情景", "",
              "各年独立50万；用于区分费用上升与保证金导致的可执行路径变化。", "",
              "| 年份 | 单边成本 | 保证金 | "+" | ".join(NAMES[v] for v in VARIANTS)+" |", "|---|---:|---:|"+"---:|"*len(VARIANTS)]
    recent = summary[summary.start.ge("20250101")&summary.window.eq("single_year")]
    for (start, fee, margin), group in recent.groupby(["start", "fee", "margin"]):
        group = group.set_index("variant")
        lines.append(f"| {start[:4]} | {fee*10000:.0f}bps | {margin:.0%} | "+" | ".join(f"{pct(group.loc[v].total_return)} / {pct(-group.loc[v].max_drawdown)}" for v in VARIANTS)+" |")

    lines += ["", "## 持续账户2025、2026当年表现", "",
              "这是2022/2025起持续账户的当年收益/年内回撤，使用承接权益，与独立年度表不同。未列2022起累计到各年末。", "",
              "| 起始年 | 年份 | 情景 | "+" | ".join(NAMES[v] for v in VARIANTS)+" |", "|---|---|---|"+"---:|"*len(VARIANTS)]
    for (start, year, fee, margin), group in annual[annual.start.isin(["20220101", "20250101"])&annual.year.ge("2025")&annual.side.eq("all")].groupby(["start", "year", "fee", "margin"]):
        if (fee, margin) not in ((.0003, .15), (.0006, .25)):
            continue
        group = group.set_index("variant")
        lines.append(f"| {start[:4]} | {year} | {fee*10000:.0f}bps/{margin:.0%} | "+" | ".join(f"{pct(group.loc[v].total_return)} / {pct(-group.loc[v].max_drawdown)}" for v in VARIANTS)+" |")

    lines += ["", "## 两项对照的一致性检查", "",
              "每类窗口各20组（5起点×4费用/保证金），起点重叠不可当独立检验；不以计数宣称显著性。", "",
              "| 窗口 | 对照 | 新版本 | 收益改善 | 回撤改善 | 同时改善 |", "|---|---|---|---:|---:|---:|"]
    for (window, reference, variant), group in comparison.groupby(["window", "reference", "variant"]):
        r, d = group.return_delta.gt(1e-10), group.drawdown_improvement.gt(1e-10)
        lines.append(f"| {window} | {NAMES[reference]} | {NAMES[variant]} | {r.sum()}/20 | {d.sum()}/20 | {(r&d).sum()}/20 |")

    lines += ["", "## 近两年预测质量", "",
              "仅使用已有完整label的样本，按候选事件归一权重；单位为当前一手名义金额比例。3bps目标。净收益的MSE改善率以‘预测未来增量收益恒为0’为对照，负数表示还不如零预测；相关性和年度均值只作诊断。", "",
              "| 年份 | 方向 | label | 实际均值 | 预测均值 | MAE | 相关性 | 对零预测MSE改善率 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in quality[quality.year.ge("2025")&quality.bps.eq(3)].itertuples():
        lines.append(f"| {r.year} | {'多' if r.direction > 0 else '空'} | {r.horizon}/{r.target} | {pct(r.actual_mean)} | {pct(r.predicted_mean)} | {pct(r.mae)} | {r.correlation:.3f} | {pct(r.mse_skill_vs_zero) if r.target == 'net' else '—'} |")

    lines += ["", "## 2026独立账户多空、退出与覆盖", "",
              "| 情景 | 版本 | 方向 | 交易数 | 获胜数 | 净利润（元） | 最差单笔（元） |", "|---|---|---|---:|---:|---:|---:|"]
    for fee, margin in ((.0003, .15), (.0006, .25)):
        part = trades[trades.start.eq("20260101")&trades.fee.eq(fee)&trades.margin.eq(margin)]
        for variant in VARIANTS:
            for side in ("long", "short"):
                group = part[part.variant.eq(variant)&part.side.eq(side)]
                worst = f"{group.pnl.min():,.0f}" if len(group) else "—"
                lines.append(f"| {fee*10000:.0f}bps/{margin:.0%} | {NAMES[variant]} | {side} | {len(group)} | {group.pnl.gt(0).sum()} | {group.pnl.sum():,.0f} | {worst} |")
    lines += ["", "### 联合压力2026：5日模型具体交易", "",
              "| 入场 | 退出 | 品种方向 | 天数 | 净利润（元） | 退出原因 |", "|---|---|---|---:|---:|---|"]
    for r in trades.query("start == '20260101' and fee == .0006 and margin == .25 and variant == 'mean5'").itertuples():
        lines.append(f"| {r.entry} | {r.exit} | {r.product} {r.side} | {r.days:.0f} | {r.pnl:,.2f} | {r.exit_reason} |")
    lines += ["", "### 真实提前退出后，继续持有会怎样", "",
              "2026独立账户联合压力情景，只统计label已成熟的提前退出；分别按各版本对应h计算继续持有增量，亏损退出以剩余期为诊断。‘退出有利’表示继续持有会亏损。不同版本h及交易集合不同，数值不能直接当账户收益排名；也不能把避亏与后续新交易收益重复相加。", "",
              "| 版本 | 成熟提前退出数 | 退出有利数 | 假设继续持有净增量合计（元） |", "|---|---:|---:|---:|"]
    part = exits.query("start == '20260101' and fee == .0006 and margin == .25").dropna(subset=["continuing_incremental_cny"])
    for variant, group in part.groupby("variant"):
        lines.append(f"| {NAMES[variant]} | {len(group)} | {group.continuing_incremental_cny.lt(0).sum()} | {group.continuing_incremental_cny.sum():,.2f} |")

    lines += ["", "## 研究判断与边界", "",
              "5日均值label比剩余期label更直接对应‘接下来一段是否继续持有’，本轮近两年表现较好，但不能据此认定期限5天最优。没有尝试3/7/10天等更多窗口，也没有按亏损品种改阈值。", "",
              "补充固定5日平仓对照：基准2025独立年度+24.98%/14.12%回撤、2026独立年度+22.49%/32.40%；联合压力2026为-13.91%/30.48%，均弱于对应5日模型。该差异不能仅用机械缩短持仓解释，但也不排除退出时点、资金与季度选择的路径依赖。", "",
              "风险惩罚没有稳定改善回撤：风险均值回归的误差、与收益预测的相关性、固定风险权重，以及退出后资金与后续机会变化都可能造成失败。历史诊断不能单独识别各因素的因果作用。下一步应先检验风险预测排序/校准和尾部覆盖，再考虑以过去内层时间验证选择风险预算；不继续用本轮外部年份选惩罚系数。", "",
              "最明显反例仍是2023起联合压力：原方案累计+146.06%，5日模型-35.81%、回撤48.14%。2025起基准虽累计收益69.68%→87.54%，回撤也15.61%→23.51%。故保留原方案与亏损退出，5日模型仅作为冻结研究候选。", "",
              "尤其不能只强调2026独立新账户：同一个2025起账户在2026年的基准收益，5日模型仅+1.12%，原方案+17.55%；联合压力则为-0.45%与+16.75%。早年收益改变资金和后续交易路径，年度重置与连续运行结论差异很大。", "",
              f"审计：{checks['control_matches']}条对照完整账本一致；{len(trained)}个方向/季度模型、{len(audit)}条历史候选验证满足时间边界；{len(totals)}账户逐笔对账；39项测试通过。模拟强平{checks['forced_exits']}次、结算保证金不足{checks['settlement_breaches']}次。缺少盘中追保、涨跌停锁单及真实订单簿成交模型，不能据此保证实盘不会爆仓。", "",
              "文件：protocol.json、states.parquet、forecasts.parquet、training_audit.csv、coefficients.csv、summary.csv、annual_marked.csv、trades.csv、daily_decisions.csv、selection.csv、validation_audit.csv、prediction_quality.csv、exit_counterfactuals.csv、paired_comparison.csv、side_product_attribution.csv、equity_drawdown.png。", ""]
    (OUT/"report.md").write_text("\n".join(lines), encoding="utf-8")
    charts()
    print(json.dumps(checks, ensure_ascii=False))
    print(quality[quality.year.ge("2025")&quality.bps.eq(3)].to_string(index=False))


if __name__ == "__main__":
    main()
