"""Cumulative year-end and start-year-to-present review of funded adaptation."""
import json
import os

import numpy as np
import pandas as pd

from refine_index_futures_adaptation import OUT, PREVIOUS, ALL_METHODS, CAPITAL

NAMES = {"fixed": "固定", "refit": "原季度重训", "reselect": "原季度重选",
         "refit_executable": "资金适配重训", "reselect_capital": "资金适配重选",
         "reselect_guarded": "资金适配+切换门槛"}


def charts():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant_adaptation_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"fixed": ("Frozen", "#747781"), "refit": ("Quarterly refit", "#21849b"),
              "reselect": ("Original reselection", "#c58a30"),
              "reselect_guarded": ("Funded + switch hurdles", "#7251a1")}
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    for col, start in enumerate(("20220101", "20250101")):
        for method, (label, color) in styles.items():
            ledger = pd.read_csv(OUT / f"ledger_{start}_{method}_0.0003_0.15.csv", dtype={"entry_date": str, "exit_date": str})
            dates = pd.to_datetime([ledger.entry_date.iloc[0], *ledger.exit_date])
            path = np.r_[CAPITAL, ledger.equity]
            axes[0, col].plot(dates, path/10000, lw=1.5, label=label, color=color)
            axes[1, col].plot(dates, 100*(path/np.maximum.accumulate(path)-1), lw=1.2, color=color)
        axes[0, col].set_title(f"CNY 500,000 account from {start[:4]}")
        axes[0, col].set_ylabel("Equity (CNY 10,000)")
        axes[1, col].set_ylabel("Drawdown from running peak (%)")
        for ax in axes[:, col]:
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Capital-aware adaptation and switch hurdles | 1 lot maximum", fontsize=14)
    fig.text(.5, .925, "3 bps per side; 15% margin + reserve; through 2026-11-06; historical exploratory replay", ha="center", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, .91))
    fig.savefig(OUT / "equity_drawdown.png", dpi=180)
    plt.close(fig)


def main():
    summary = pd.read_csv(OUT / "summary.csv", dtype={"start": str})
    annual = pd.read_csv(OUT / "annual.csv", dtype={"start": str, "year": str})
    cumulative = pd.read_csv(OUT / "cumulative_by_year.csv", dtype={"start": str, "year": str, "asof": str})
    trades = pd.read_csv(OUT / "trades.csv", dtype={"start": str})
    audit = pd.read_csv(OUT / "capital_validation_audit.csv", dtype={"date": str, "cutoff": str, "last_validation_exit": str})
    selection = pd.read_csv(OUT / "selection.csv", dtype={"start": str, "date": str})
    assert len(summary) == 120
    keys = ["start", "fee", "margin", "method"]
    reconcile = summary.merge(trades.groupby(keys).pnl.sum().rename("trade_pnl"), on=keys, how="left")
    np.testing.assert_allclose(reconcile.trade_pnl.fillna(0), reconcile.total_return*CAPITAL, atol=1e-6, rtol=0)
    assert audit.last_validation_exit.le(audit.cutoff).all() and audit.cutoff.lt(audit.date).all()
    for _, group in cumulative.groupby(keys):
        assert group.sort_values("year").cumulative_drawdown.diff().dropna().le(1e-12).all()
    ends = summary.merge(cumulative.sort_values("year").groupby(keys).tail(1), on=keys)
    np.testing.assert_allclose(ends.total_return, ends.cumulative_return, atol=1e-12)
    np.testing.assert_allclose(ends.max_drawdown, ends.cumulative_drawdown, atol=1e-12)
    control_count = 0
    for start in ("20220101", "20250101"):
        for method in ("fixed", "refit", "reselect"):
            for fee in (.0003, .0006):
                for margin in (.15, .25):
                    new = pd.read_csv(OUT / f"ledger_{start}_{method}_{fee}_{margin}.csv", keep_default_na=False)
                    old = pd.read_csv(PREVIOUS / f"ledger_continuous_{start}_{method}_{fee}_{margin}.csv", keep_default_na=False)
                    pd.testing.assert_frame_equal(new, old, check_exact=False, atol=1e-6, rtol=1e-12)
                    control_count += 1
    checks = dict(accounts=len(summary), cumulative_rows=len(cumulative), validation_records=len(audit),
                  unchanged_controls=control_count, unit_tests=23,
                  forced_exits=int(summary.forced.sum()), last=cumulative["asof"].max())
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    def pct(x):
        return "—" if pd.isna(x) else f"{x:.2%}"
    selected = ("fixed", "refit", "reselect", "reselect_guarded")
    lines = ["# 当前资金适配与季度切换门槛", "",
             "初始50万元、全组合最多一手、允许多空。沿用上轮季度预测和四项策略池，仅改变执行可行性、历史验证起始资金与切换门槛。数据止于2026-11-06；2026不完整。完成120组账户回放，未按本轮结果搜索最优门槛。", "",
             "**本轮结论：切换门槛改善了2022起基准账户的累计收益和回撤，并降低2025起账户回撤；资金适配单独并未普遍改善基准结果。2026新起点不改善、高保证金长期账户仍亏损，因此尚未证明稳定优势。**", "",
             "## 两种累计口径同时提供", "",
             "A：同一个2022起账户，运行到每年末的累计收益与最大回撤；初始本金始终为50万，回撤包含更早年份的高点和低点，不按年清零，也不在年末强行平仓重开。B：分别在2022、2023、2024、2025、2026年以50万开始，一直运行到共同终点，比较起始年份敏感性。新起点也重置固定模型年代、持仓和入场节奏，不能只解释成资金差异。", "",
             "## A：2022起账户，累计至每个年末", "",
             "基准：单边名义金额3bps，保证金15%并保留原安全储备。每格为累计净收益 / 自起点以来最大回撤幅度。", "",
             "| 截止年 | 固定 | 原季度重训 | 原季度重选 | 资金适配+门槛 |", "|---|---:|---:|---:|---:|"]
    base_cumulative = cumulative.query("start == '20220101' and fee == .0003 and margin == .15")
    for year, group in base_cumulative.groupby("year"):
        cells = []
        for method in selected:
            row = group[group.method.eq(method)].iloc[0]
            cells.append(f"{pct(row.cumulative_return)} / {pct(-row.cumulative_drawdown)}")
        lines.append(f"| {year} | " + " | ".join(cells) + " |")
    lines += ["", "注意：累计回撤幅度不会因为后续创新高或一年没有亏损就缩小。2022起新方案2026当年仍为-5.57%，不能用累计+144.74%掩盖近期回落。", "",
              "## B：分别从各年份开始，累计到2026-11-06", "",
              "每格仍为累计净收益 / 全区间最大回撤幅度，各行相互独立，不能拼接。", "",
              "| 起始年 | 固定 | 原季度重训 | 原季度重选 | 资金适配+门槛 |", "|---|---:|---:|---:|---:|"]
    base_summary = summary.query("fee == .0003 and margin == .15")
    for start, group in base_summary.groupby("start"):
        cells = []
        for method in selected:
            row = group[group.method.eq(method)].iloc[0]
            cells.append(f"{pct(row.total_return)} / {pct(-row.max_drawdown)}")
        lines.append(f"| {start[:4]} | " + " | ".join(cells) + " |")
    lines += ["", "## 改了什么", "",
              "1. 资金适配重训：信号最高的品种若无法同时覆盖保证金、开仓费和安全储备，就从其他满足原信号规则的品种中选择可执行者；无可执行信号则空仓，不减少储备、不突破一手。原版只选最高信号后可能被资金约束拦截。", "",
              "2. 资金适配重选：每个更新时点读取已经盯市至当日开盘的实际账户权益，把它作为此前36个月候选验证账户的起始本金；历史路径仍逐日继承盈亏。历史验证也使用上述可执行品种选择。验证模型预测来自当时季度滚动训练，未用未来预测结果。", "",
              "3. 切换门槛：先保留上轮正收益、覆盖率、无强平及不劣于base回撤的过滤；再要求候选Sortino较当前策略提高至少0.25，过去加权日均净利润差×61超过max(当前权益×0.5%，预计切换费×3)。当前策略已有至少6笔历史交易时，还要求候选历史回撤不更差。无足够优势就延续当前策略。", "",
              "切换费估算=当前持仓平仓费+目前能负担品种中最大单手名义金额的开仓费；它是保守决策估算，不是实际额外收费。真实账本只扣实际开平和换月费用。61日历史均值折算也不是经过校准的未来利润预测。", "",
              "门槛在本轮运行前固定，未根据本轮各年收益反向寻找最优参数。风险调整得分、资本门槛及回撤守门共同作用，不能把全部改善归功于某一个数值。当前开盘权益/价格用于即时执行性检查，仍是日线同一开盘成交的理想化假设。", "",
              "## 分解对照：基准情景的所有六组", "",
              "| 起点 | 方法 | 累计收益 | 最大回撤幅度 | 交易数 | 规则切换数 | 切换平仓数 | 盈利月比例 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in base_summary.itertuples():
        lines.append(f"| {row.start[:4]} | {NAMES[row.method]} | {pct(row.total_return)} | {pct(-row.max_drawdown)} | {row.trades} | {row.changes} | {row.switch_closures} | {pct(row.positive_month_ratio)} |")
    lines += ["", "2022起、15%保证金的基准账户，资金适配本身没有改变原重训/原重选的实际路径；改善来自额外切换守门。规则切换由9降至5次，因切换平仓由6降至2次。2024年4月和7月因候选历史回撤劣于当时策略，拒绝了切换；这是本轮关键分歧，仍不能证明同类信号未来总有效。", "",
              "2025起新方案回撤降到15.61%，收益69.68%略低于原重选72.19%，体现收益与风险取舍；2026独立起点则与原重选相同且弱于固定模型，不能宣称已普遍优于固定策略。", "",
              "## 费用与保证金分离压力", "",
              "| 起点 | 费用bps | 保证金 | 方法 | 累计收益 | 回撤幅度 | 交易数 |", "|---|---:|---:|---|---:|---:|---:|"]
    for row in summary[summary.start.isin(["20220101", "20250101"]) & summary.method.isin(["refit", "refit_executable", "reselect", "reselect_guarded"])].itertuples():
        lines.append(f"| {row.start[:4]} | {row.fee*10000:.0f} | {pct(row.margin)} | {NAMES[row.method]} | {pct(row.total_return)} | {pct(-row.max_drawdown)} | {row.trades} |")
    lines += ["", "2022起联合压力情景（6bps、25%）新重选仍亏42.11%，虽较原重选亏45.51%有改善，仍不能视为可接受方案。资金适配重训将该情景交易数从22增至36，收益从-51.80%改善至-30.24%，但回撤仍约56.5%；让交易可执行不等于让交易盈利。", "",
              "## 逐年当年收益与回撤：与累计表分开", "",
              "2022起基准账户；每格是当年收益 / 年内回撤幅度。", "",
              "| 年份 | 固定 | 原季度重训 | 原季度重选 | 资金适配+门槛 |", "|---|---:|---:|---:|---:|"]
    sample = annual.query("start == '20220101' and fee == .0003 and margin == .15 and side == 'all'")
    for year, group in sample.groupby("year"):
        cells = []
        for method in selected:
            row = group[group.method.eq(method)].iloc[0]
            cells.append(f"{pct(row.total_return)} / {pct(-row.max_drawdown)}")
        lines.append(f"| {year} | " + " | ".join(cells) + " |")
    lines += ["", "## 审计与下一步", "",
              f"{len(summary)}账户逐笔净利润与逐日账本核对；{control_count}个上一轮完整对照账本一致；{len(audit)}条资金验证记录满足最后历史收益日期不晚于cutoff、cutoff早于决策日；{len(cumulative)}条累计年度记录的末值与总表一致，累计回撤不按年重置。23项专项测试通过。模型和训练窗口未改变。", "",
              "所有账户模拟强平次数为0；不意味着盘中不会追保或涨跌停无法平仓。成本与保证金是统一情景假设，并非逐日历史费率。历史已经反复研究，当前结果属于探索性回放，未形成新的独立OOT。", "",
              "下一步保留本轮方案作为研究候选，同时优先解释高保证金长期失败及2026年收益回落。新增规则须预先登记，不能围绕已知亏损日期事后补条件；后续新数据成熟后再做前瞻验证。", "",
              "输出：cumulative_by_year.csv（同一账户至各年末）、summary.csv（各起始年一直到最新）、annual.csv（当年及多空）、selection.csv（门槛审计）、capital_validation_audit.csv（资金验证）、trades.csv（逐笔），equity_drawdown.png（净值回撤图）。", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    charts()
    print(json.dumps(checks, ensure_ascii=False))
    print("Report:", OUT / "report.md")


if __name__ == "__main__":
    main()
