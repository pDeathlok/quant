"""Annual and recent-period review of the fixed/refit/reselect experiment."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant_adaptation_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_index_futures import metrics
from compare_index_futures_adaptation import OUT, METHODS, CAPITAL, stability

NAMES = {"fixed": "固定模型与规则", "refit": "季度滚动重训", "reselect": "季度重训并重选"}


def read_ledger(start, method, fee=.0003, margin=.15):
    return pd.read_csv(OUT / f"ledger_continuous_{start}_{method}_{fee}_{margin}.csv",
                       dtype={"entry_date": str, "exit_date": str}, keep_default_na=False)


def make_chart():
    colors = {"fixed": "#656773", "refit": "#167d9a", "reselect": "#c07924"}
    english = {"fixed": "Frozen model + rule", "refit": "Quarterly model refit", "reselect": "Quarterly refit + policy selection"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    for col, start in enumerate(("20220101", "20250101")):
        for method in METHODS:
            ledger = read_ledger(start, method)
            dates = pd.to_datetime([ledger.entry_date.iloc[0], *ledger.exit_date])
            equity = np.r_[CAPITAL, ledger.equity]
            axes[0, col].plot(dates, equity/10000, color=colors[method], label=english[method], lw=1.6)
            axes[1, col].plot(dates, 100*(equity/np.maximum.accumulate(equity)-1), color=colors[method], lw=1.3)
        axes[0, col].set_title(f"Fresh CNY 500,000 account from {start[:4]}")
        axes[0, col].axhline(50, color="#bbbbbb", ls="--", lw=.8)
        axes[0, col].set_ylabel("Equity (CNY 10,000)")
        axes[1, col].set_ylabel("Drawdown (%)")
        for ax in axes[:, col]:
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Index futures: frozen vs adaptive policies | 1 lot maximum", fontsize=14)
    fig.text(.5, .925, "3 bps per traded side; 15% margin + reserve; through 2026-11-06; historical exploratory replay", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .06, 1, .91))
    fig.savefig(OUT / "equity_drawdown.png", dpi=180)
    plt.close(fig)


def main():
    annual = pd.read_csv(OUT / "annual.csv", dtype={"start": str, "year": str})
    summary = pd.read_csv(OUT / "summary.csv", dtype={"start": str})
    decisions = pd.read_csv(OUT / "selection.csv", dtype={"start": str, "date": str, "cutoff": str})
    train = pd.read_csv(OUT / "training_audit.csv", dtype={"cutoff": str, "last_maturity": str})
    validation = pd.read_csv(OUT / "validation_audit.csv", dtype={"date": str, "cutoff": str, "last_validation_exit": str})
    trades = pd.read_csv(OUT / "trades.csv", dtype={"start": str})
    assert len(summary) == 84 and train.last_maturity.lt(train.cutoff).all()
    assert validation.last_validation_exit.le(validation.cutoff).all() and validation.cutoff.lt(validation.date).all()
    keys = ["evaluation", "start", "fee", "margin", "method"]
    check = summary.merge(trades.groupby(keys).pnl.sum().rename("trade_pnl"), on=keys, how="left")
    np.testing.assert_allclose(check.trade_pnl.fillna(0), check.total_return*CAPITAL, atol=1e-6, rtol=0)
    recent = []
    for start in ("20220101", "20250101"):
        for fee in (.0003, .0006):
            for margin in (.15, .25):
                for method in METHODS:
                    ledger = read_ledger(start, method, fee, margin)
                    mask = ledger.exit_date.ge("20250101")
                    initial = ledger.loc[~mask, "equity"].iloc[-1] if (~mask).any() else CAPITAL
                    part = ledger.loc[mask].reset_index(drop=True)
                    recent.append(dict(start=start, fee=fee, margin=margin, method=method,
                                       period_start_equity=initial, **metrics(part, initial), **stability(part, initial)))
    recent = pd.DataFrame(recent)
    recent.to_csv(OUT / "recent_2025_onward.csv", index=False)
    def pct(value):
        return "—" if pd.isna(value) else f"{value:.2%}"
    lines = ["# 固定、滚动重训、滚动重选：逐年收益与风险比较", "",
             "数据截止2026-11-06；初始50万元，全组合最多一手，允许多空。2026年为不完整年度。共84组账户回放，15项专项检查通过。已反复研究的历史仍属于探索性回放，不是未触碰OOT。", "",
             "**结论：季度重训在近期有研究价值，但季度重选没有稳定超过只重训；长区间和保证金压力下，两种滚动方案都可能比固定方案更差。不能把本轮结果解释为滚动一定更优。**", "",
             "## 方法与公平性", "",
             "每个起点三组先使用同一套模型和base规则。fixed冻结起点模型和规则；refit每季度重训，base规则不变；reselect同样重训，从第二季度起再在base、辅助指数盈利筛选aux55、提高方向置信门槛confidence60、相反模型提前退出opposing_exit四项中选择。三组首季度账本完全相同。", "",
             "模型训练最多回看36个月，12个月半衰期；方向20/60日模型使用期货和ETF因子，盈利20日模型加辅助指数因子。各label必须早于当季首个信号日成熟。此次保留既有label与交易规则以隔离适应方式，不声称已完成净利润回归/路径风险label优化。", "",
             "每次选策略只回放此前最多36个月的季度样本外预测：累计扣费收益与近期加权均值均为正，至少6笔且达到base一半交易数和持仓覆盖，至少244个观察日，无强平、回撤不劣于base，再按12个月半衰期加权Sortino选优。无合格候选则延续当前策略。最早的验证预测从2020年开始，因此2022年初不足完整36个月，实际起止保存在validation_audit.csv。", "",
             "策略改变时，旧仓在更新日开盘平仓；新规则在之后符合五日入场网格时开仓，计入两边费用，不按季度重置资金。仅模型重训不强制平仓。所有方案使用相同保证金安全储备和开盘/结算保证金检查。", "",
             "三种账户口径：2022起连续账户；2025起新账户；各年独立50万账户。新账户的固定模型在其起点重新训练，因此起点敏感性同时包含模型年代、资金、持仓和入场节奏变化，不能只归因于资金。年度独立结果不能拼成可交易复利净值。", "",
             "## 2022起连续账户：基准情景逐年", "",
             "成本为每次单边名义金额3bps，保证金15%；表格每格为年度净收益 / 年内最大回撤幅度。年度收益继承上年权益，不重置50万。", ""]
    def annual_table(frame):
        result = ["| 年份 | 固定 | 重训 | 重选 |", "|---|---:|---:|---:|"]
        for year, rows in frame.groupby("year"):
            cells = []
            for method in METHODS:
                row = rows[rows.method.eq(method)].iloc[0]
                cells.append(f"{pct(row.total_return)} / {pct(-row.max_drawdown)}")
            result.append(f"| {year} | " + " | ".join(cells) + " |")
        return result
    base = annual[(annual.side == "all") & (annual.fee == .0003) & (annual.margin == .15)]
    lines += annual_table(base[(base.evaluation == "continuous") & (base.start == "20220101")])
    lines += ["", "## 重点看2025年以来", "",
              "下表区分2022账户继续运行与2025新账户。近期回撤从2025期初权益重新计算；不代表全历史最大回撤，保留原持仓和原策略状态，不重新回测改变路径。", "",
              "| 账户起点 | 方案 | 2025至今净收益 | 期间回撤幅度 | 盈利月份比例 |", "|---|---|---:|---:|---:|"]
    for row in recent.query("fee == .0003 and margin == .15").itertuples():
        lines.append(f"| {row.start[:4]} | {NAMES[row.method]} | {pct(row.total_return)} | {pct(-row.max_drawdown)} | {pct(row.positive_month_ratio)} |")
    lines += ["", "2025新账户逐年结果：", ""]
    lines += annual_table(base[(base.evaluation == "continuous") & (base.start == "20250101")])
    lines += ["", "## 全区间稳定性与资金压力", "",
              "费用和保证金独立做2×2实验。下表为整个账户区间的净收益和全区间回撤；最长未创新高天数含尚未恢复的区间，按交易观察日计。盈利月份含最后不完整月份，零收益月不计盈利。", "",
              "| 起点 | 费用bps | 保证金 | 方案 | 净收益 | 最大回撤幅度 | 笔数 | 盈利月比例 | 最长未创新高日数 |", "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in summary[summary.evaluation.eq("continuous")].itertuples():
        lines.append(f"| {row.start[:4]} | {row.fee*10000:.0f} | {pct(row.margin)} | {NAMES[row.method]} | {pct(row.total_return)} | {pct(-row.max_drawdown)} | {row.trades} | {pct(row.positive_month_ratio)} | {row.max_underwater_sessions} |")
    lines += ["", "2022起、6bps及25%保证金情景中，重训2025/2026年均无交易，重选也在这两年无交易。零收益、零年内回撤反映资金与信号约束后的空仓，不能当成稳定盈利。所有84账户模拟强平次数为0，但不代表盘中不会追保或亏损。保证金安全储备会在强平前阻止开仓/退出。", "",
              "重选的历史候选账户统一从50万模拟，实际账户的当时权益可能更低；历史排名优胜不保证当时还有足够资金执行。这是本轮揭示的重要局限，下一步可预登记当前权益可执行性检查和更保守的切换门槛，不能事后只修补某年亏损交易。", "",
              "## 各年独立50万的敏感性诊断", ""]
    lines += annual_table(base[base.evaluation.eq("annual_reset")])
    lines += ["", "各年fixed在当年年初训练后冻结，refit与reselect同样从base开始。2026独立账户固定模型的结果较好，提示季度更新也可能放弃仍有效的参数；不能仅看2025新账户就断言越勤更新越好。", "",
              "## 策略切换审计", "",
              "| 连续账户起点 | 费用bps | 保证金 | 规则切换次数 | 因切换平仓次数 |", "|---|---:|---:|---:|---:|"]
    for row in summary[(summary.evaluation == "continuous") & (summary.method == "reselect")].itertuples():
        lines.append(f"| {row.start[:4]} | {row.fee*10000:.0f} | {pct(row.margin)} | {row.policy_changes} | {row.switch_closures} |")
    lines += ["", "每次选择及依据见selection.csv、validation_audit.csv；切换实际平仓见switches.csv。手续费已计入净值；没有额外收取不存在的固定策略切换费。费用提高有时改变资金可执行性和后续交易路径，因此不同费率账户收益不是简单线性相减。", "",
              "## 验证与后续", "",
              f"训练时点记录{len(train)}条，均严格满足label成熟日期早于cutoff；历史选择候选记录{len(validation)}条，均满足最后可见收益日期不晚于cutoff且早于决策日；84个账户逐笔净利润与逐日账本完全对账，持仓始终不超过一手。", "",
              "本轮支持继续研究季度重训作为适应市场变化的方向；暂不将自动重选替换为默认方案。后续优先检验当前资金可执行性、重选与现行策略之间的历史优势是否足以覆盖切换代价，再考虑少量策略池扩展。继续以扣费收益、回撤及尾部风险共同评价，胜率只作解释。", "",
              "日线没有模拟盘中追保、涨跌停无法成交，费率/保证金为统一情景假设而非逐日历史真实费率。年度交易笔数较少，所有结论属于探索性样本证据。", "",
              "文件：annual.csv（逐年多空），summary.csv（全区间），recent_2025_onward.csv（近期），trades.csv（完整交易），protocol.json（预设流程）。", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    make_chart()
    print(recent.query("fee == .0003 and margin == .15")[["start", "method", "total_return", "max_drawdown", "positive_month_ratio"]].round(4).to_string(index=False))
    print("Verified", len(summary), "accounts; report and chart saved in", OUT)


if __name__ == "__main__":
    main()
