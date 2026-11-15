"""Freeze the selected research policy, execution sensitivity and readable report."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import select_final_etf_strategy as research
from report_six_sleeves import markdown_table

OUT = research.OUT
ROOT = research.ROOT


def supplemental(summary: dict) -> pd.DataFrame:
    panel = research.load_panel()
    targets, _ = research.make_targets(panel, summary["allocation"], summary["management"])
    rows = []
    for capital in (100_000, 300_000, 1_000_000):
        for name, kwargs in (("base", {}), ("double_cost", {"cost_multiplier": 2.}), ("delay2", {"lag": 2})):
            result, _ = research.simulate(panel, targets, initial=capital, **kwargs)
            returns = result.returns.loc[research.START:]
            rows.append({"capital": capital, "execution": name,
                         **research.prior.metrics(returns, result.turnover.loc[returns.index]),
                         "average_cash": float(1-result.weights.loc[returns.index].sum(axis=1).mean())})
    table = pd.DataFrame(rows)
    table.to_csv(OUT/"capital_and_execution_sensitivity.csv", index=False)
    weights = pd.read_csv(OUT/"selected_holdings.csv", index_col=0)[list(research.CODES)]
    budget = np.array([summary["base_budget"][code] for code in research.CODES])
    shocks = {"inflation_rates_up": [-.25, -.25, -.04, -.08, -.15, .10, .10, .10],
              "liquidity_joint_selloff": [-.30, -.30, -.05, -.10, -.20, -.35, -.35, -.35]}
    stress = []
    for name, vector in shocks.items():
        loss = weights.to_numpy() @ np.array(vector)
        stress.append({"scenario": name, "full_budget_loss": float(budget @ np.array(vector)),
                       "mean_actual_exposure_loss": float(loss.mean()),
                       "worst_actual_exposure_loss": float(loss.min()),
                       "shocks": dict(zip(research.CODES, vector))})
    (OUT/"hypothetical_stress.json").write_text(json.dumps(stress, ensure_ascii=False, indent=2, allow_nan=False))
    return table


def plot(chosen: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/quant-six-sleeve-mpl")
    sys.path.append("/private/tmp/quant-six-sleeve-plotdeps")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.ticker import PercentFormatter
    font = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    fontManager.addfont(font)
    plt.rcParams.update({"font.family": FontProperties(fname=font).get_name(), "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
    returns = pd.read_csv(OUT/"all_daily_returns.csv", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for name, label, color in ((chosen, "最终V1：50%国债 / 20%股票 / 20%黄金 / 10%商品", "#087f8c"),
                               ("three_asset_equal", "比较基线：沪深300 / 黄金 / 5年国债等权", "#bd7b36")):
        nav = (1+returns[name]).cumprod()
        peak = nav.cummax().clip(lower=1.)
        axes[0].plot(nav.index, nav, label=label, color=color, lw=1.8)
        axes[1].plot(nav.index, nav/peak-1, color=color, lw=1.3)
    axes[0].set_ylabel("净值（起始1）")
    axes[1].set_ylabel("距历史高点回撤")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc="upper left", frameon=False, fontsize=10)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.axvspan(pd.Timestamp("2026-01-01"), returns.index[-1], color="#94a3b8", alpha=.12)
        ax.axvline(pd.Timestamp("2026-01-01"), color="#64748b", ls="--", lw=1)
    axes[0].text(pd.Timestamp("2026-01-15"), 1.02, "2026诊断区间\n非独立样本外", color="#475569", fontsize=10)
    fig.suptitle("最终ETF组合｜历史年化6.40%，最大回撤6.86%", fontsize=18)
    fig.supxlabel("2021-01-04—2026-11-10 · 100万元 · 计费用与整手约束 · 历史表现不是未来收益承诺", fontsize=11)
    fig.savefig(OUT/"final_strategy.png", dpi=170, facecolor="white")
    plt.close(fig)


def main() -> None:
    summary = json.loads((OUT/"selected_strategy.json").read_text())
    assert summary["selected"] == "defensive_fixed", "Report requires review if selection changes"
    sensitivity = supplemental(summary)
    pool = pd.read_csv(OUT/"pool_and_current_targets.csv")
    comparison = pd.read_csv(OUT/"full_comparison.csv", index_col=0)
    annual = pd.read_csv(OUT/"annual_returns.csv", index_col=0)[summary["selected"]]
    config = {
        "id": "etf_defensive_monthly_v1", "status": "research_frozen_not_live", "version_date": "2026-11-11",
        "data_end": summary["end"], "reference_capital_CNY": summary["reference_capital"],
        "base_weights": summary["base_budget"], "cash_return_assumption": 0,
        "rebalance": {"signal": "last_trading_day_of_month_close", "simulation_fill": "next_trading_day_close",
                      "initial_state": "cash", "unsatisfied_budget": "cash_no_redistribution",
                      "unfilled_orders": "no_automatic_retry_until_next_month", "shorting": False, "leverage": False},
        "eligibility": {"valid_trading_quotes_min": 120, "median20_amount_CNY_min": 20_000_000,
                        "current_valid_quote_required": True, "nav_announcement": "strictly_before_signal_date",
                        "nav_valuation_age_calendar_days_max": 7,
                        "premium": "close_on_NAV_valuation_date / unit_NAV - 1",
                        "premium_max": .005, "premium_data_sanity_min": -.5,
                        "missing_information": "target_cash", "complete_60_day_covariance_required": True},
        "execution": {"buy_lot": 100, "sell_first": True, "commission": .0002,
                      "minimum_commission_CNY": 5, "slippage": .0008,
                      "participation_of_prior20_median_max": .01, "borrowing": False,
                      "dividends": "adjusted_return_reinvestment_approximation"},
        "live_checks_not_backtested": ["account_permissions", "halt_and_price_limits", "bid_ask_spread",
                                       "valid_current_IOPV_and_premium_before_buy; skip_if_unavailable_or_abnormal"],
        "inactive_methods": ["trend_timing", "risk_parity", "volatility_targeting", "daily_stop_loss"],
        "latest_completed_month_signal": summary["last_month_signal_date"],
        "latest_completed_month_target": summary["last_month_target"],
        "monitor_only_date": summary["monitor_date"], "monitor_only_weights": summary["monitor_target_not_order"],
        "report": str(OUT/"report.md"), "backtest_summary": str(OUT/"selected_strategy.json"),
        "selection_script_sha256": summary["script_sha256"], "protocol_sha256": summary["protocol_sha256"]}
    config_path = ROOT/"configs/research/etf_defensive_monthly_v1.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False))
    inventory = pd.DataFrame({"ETF": pool.name, "代码": pool.code, "长期预算": pool.base_budget.map(lambda v:f"{v:.2%}"),
                              "10/30月末目标": pool.last_month_target.map(lambda v:f"{v:.2%}"),
                              "11/10已知历史溢价": pool.known_premium.map(lambda v:f"{v:.2%}"),
                              "20日成交额中位数（亿元）": pool.median20_amount_CNY.map(lambda v:f"{v/1e8:.2f}")})
    names = {"defensive_fixed": "最终V1固定配置", "defensive_trend": "相同预算＋趋势过滤",
             "defensive_vol": "相同预算＋波动率控制", "balanced_fixed": "股票30%固定配置",
             "simple_fixed": "无商品固定配置", "three_asset_equal": "三资产等权比较基线"}
    selected = comparison.loc[list(names)]
    results = pd.DataFrame({"组合": [names[k] for k in selected.index],
                            "年化收益": selected.cagr.map(lambda v:f"{v:.2%}").values,
                            "最大回撤": selected.max_drawdown.map(lambda v:f"{v:.2%}").values,
                            "年化波动": selected.volatility.map(lambda v:f"{v:.2%}").values,
                            "最差单日": selected.worst_day.map(lambda v:f"{v:.2%}").values})
    year_table = pd.DataFrame({"年份": [str(v) if v != 2026 else "2026截至11/10" for v in annual.index],
                               "收益": annual.map(lambda v:f"{v:.2%}").values})
    execution_names = {"base": "基准费用", "double_cost": "交易费用翻倍", "delay2": "多延迟一个交易日"}
    execution_table = pd.DataFrame({"资金（万元）": sensitivity.capital/10000,
                                   "执行": sensitivity.execution.map(execution_names),
                                   "年化收益": sensitivity.cagr.map(lambda v:f"{v:.2%}"),
                                   "最大回撤": sensitivity.max_drawdown.map(lambda v:f"{v:.2%}"),
                                   "平均现金": sensitivity.average_cash.map(lambda v:f"{v:.2%}")})
    plot(summary["selected"])
    report = f"""# ETF四类资产月度再平衡 V1

研究定版：2026-11-11；行情截至2026-11-10。选定 defensive_fixed，假设人民币100万元、无杠杆、月度再平衡。已完成研究与规则冻结，未下单、未接入生产。

## 最终决定

**国债50%、股票20%、黄金20%、非黄金商品10%。** 债券两只各25%，股票两只各10%，黄金20%，三个商品各10%/3。比例是资金预算，国债50%不代表承担50%的组合风险。债券共同承担人民币利率风险，八只ETF不等于八个独立风险来源。

本轮比较12个事先约定候选，最终选择固定配置。2021—2025的得分与执行扰动测试支持它；更复杂的择时/波动率管理没有带来足够改善。不是根据全期收益最高直接选冠军，也没有证明这是所有可交易ETF中的全局最优。

## 可交易ETF池与最近状态

{markdown_table(inventory)}

核验了基金身份、上市日期和最新成交记录，均为境内上市ETF二级市场证券；不要求直接开期货账户交易期货合约。具体账户权限与盘中可成交状态仍以券商和交易所为准。“有上市行情”不等于在任何价格都值得买。

豆粕159985在2026-10-30月末信号被过滤，目标0，约3.33%预算留现金。11/10监测仍不合格：可使用的11/9净值对应日溢价约3.40%，超过0.50%上限。其余ETF预算保持不变；整手、费用和持仓漂移会使实际现金不同于3.33%。11/10是监测，不能把它当成新增月度交易信号，也不是当前实时溢价。

## 冻结执行规则

1. 每月最后一个交易日收盘生成目标，下一交易日执行；回测采用下一交易日收盘。首期现金启动，组合初始就遵循上述日期规则。
2. 每只ETF至少120个有效报价；信号日有效成交；过去20个交易日成交额中位数≥2000万元；所需60日收益数据完整。没有完整数据则保留现金。
3. 仅使用公告日严格早于信号日的最近净值，估值日期不超过7个自然日；用净值估值当日收盘价计算历史已知溢价。溢价>0.50%、净值缺失或过期时，该ETF整份目标变为现金。低于-50%视为数据异常；该宽阈值不是认可大折价无风险。长假可能触发净值过期过滤。
4. 未达标的预算不分给其他ETF。按月重新判断；没有趋势择时、动态风险平价、每日止损或临时加杠杆。
5. 卖出先行，买入100份整数手，现金不足缩小买单；单只当日成交金额≤此前20日成交额中位数的1%。停牌/无有效报价不成交，未成交部分保留原持仓或现金，不在次日自动追单，下月重算。
6. 回测佣金单边2bp、单笔至少5元，加8bp滑点；内部管理和托管费用已反映在基金收益，不重复收取。现金收益设零。分红按复权收益近似再投资，退出时允许零碎尾数清仓，不等于逐笔登记结算模拟。
7. 实际买单还需检查最新有效IOPV、溢价、盘口价差、停牌和价格限制；IOPV不可用或异常时不买。**本轮没有盘中数据，这些执行检查没有被回测，不能宣称实盘必然复制日线结果。**

## 历史结果

统一窗口2021-01-04—2026-11-10；100万元、同一净值过滤和执行模型。最终V1累计收益43.77%、年化收益6.40%、最大回撤6.86%、年化波动5.29%、最差单日-3.12%，平均现金约5.99%。年买卖成交额合计约净值1.05倍，夏普1.25采用无风险利率0口径。

{markdown_table(results)}

![净值与回撤](final_strategy.png)

选择期仅使用2021—2025：最终V1年化7.43%、最大回撤2.96%。基准成本、费用翻倍、多延迟一天三种条件均通过预先设定的回撤≤12%、最差单日≥-4%、波动≤10%、完整年收益≥-8%门槛。所有12候选本次均通过，门槛只是排除条件，不是未来保证。

通过者按“年度收益中位数－0.5×最大回撤幅度－0.0005×年成交额/净值”排名；距最高分0.3个百分点内优先规则简单，再比换手和回撤。这些选择偏好是研究假设，不是统计显著性检验。

{markdown_table(year_table)}

**2026年收益只有0.52%，最大回撤6.86%，明显弱于选择期。** 2026行情在前轮已经看过，虽然本次不用于评分，仍不能称独立样本外。历史完整年度没有亏损，不代表未来每年盈利。

## 执行与资金敏感性

{markdown_table(execution_table)}

10万元也可模拟执行，但一手5年/10年国债ETF按11/10收盘约需1.41万/1.36万元，仓位颗粒明显偏粗。100万元只是参考资金，不是开户或申购门槛。实盘佣金由实际券商替换；未取得用户资金规模前，不输出具体下单手数。

附近权重诊断：黄金15%、减少部分改配国债时，年化5.86%、回撤4.94%；黄金25%、相应减国债时，年化6.96%、回撤8.82%。保持黄金20%，没有按诊断改选。最终是固定配置，改变动量/均线/波动率阈值不改变主要持仓规则；这些重复结果**不作为参数稳健性的证据**。

## 风险与边界

- 50%国债ETF仍有久期风险，利率上升会亏损；5年和10年国债不能视为完全独立资产。黄金和商品可能同时下跌，黄金20%仍是显著风险源。
- 商品ETF提供期货多头敞口，包含展期、跟踪误差和折溢价风险，不等于能做空的CTA。低相关不等于低波动，所以商品合计只给10%，其中能源化工仅3.33%。
- 人民币资产占主导；本版暂未覆盖海外股票和海外利率周期。跨境ETF溢价和时差更复杂，30年国债历史较短，小盘/转债又增加股票风险，因此不强行纳入。
- 池子事后选取，存在幸存者与研究者选择偏差；约六年窗口不能覆盖全部经济周期。月末估计溢价有滞后，拿不到历史盘中IOPV；成交额不是盘口深度，收盘成交、极端流动性与分红到账近似仍有误差。
- 假设股票-30%、5年国债-5%、10年国债-10%、黄金-20%、三类商品各-35%同时发生，按满额预算组合损失约**17.25%**。这是人为压力假设，不是历史事件或概率预测；历史6.86%回撤和12%筛选门槛都不是损失上限。

## 数据与复现

原始行情来自Tushare fund_daily/fund_adj，净值来自fund_nav；原始parquet、哈希、公告日期及信号记录均留存。基金身份由东方财富数据连接器核验，原始响应保存在source/fund_identity_evidence.json。基金净值公告与估值日期字段参见[Tushare净值文档](https://tushare.pro/document/2?doc_id=119)，成交额单位千元参见[Tushare行情文档](https://tushare.pro/document/2?doc_id=127)。本轮将成交额显式乘1000统一为元。

可复现命令（项目根目录，Python需numpy/pandas/pyarrow；绘图还需matplotlib与中文字体）：

```bash
python scripts/research/select_final_etf_strategy.py
python scripts/research/report_final_etf_strategy.py
python -m pytest -q tests/test_final_etf_strategy.py tests/test_six_sleeve_exploration.py
```

14项相关测试已通过，覆盖公告时点、净值与价格日期匹配、信号后成交、整手/费用/现金约束、缺失报价、未来数据不影响历史信号及选择规则。日线模型的测试通过不等于行情源独立审计或实盘收益验证。

交付：规则配置configs/research/etf_defensive_monthly_v1.json；selected_strategy.json为完整结果摘要，含候选研究阶段的非激活参数；all_selection_cases.csv和selection_ranking.csv保留全部候选及选择证据；pool_and_current_targets.csv为ETF池和状态；selected_signals.csv、selected_trades.csv、selected_holdings.csv为审计明细。只冻结这一套V1，不改现有生产策略。
"""
    (OUT/"report.md").write_text(report)
    print(f"Report: {OUT/'report.md'}")
    print(f"Frozen research config: {config_path}")


if __name__ == "__main__":
    main()
