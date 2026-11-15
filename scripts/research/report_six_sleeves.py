"""Produce a Chinese research note and standalone plots from the frozen experiment."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/research/six_sleeve_exploration_20261111"
LABELS = {"six_equal": "六方向等权", "six_erc": "六方向风险平价", "six_erc_guarded": "风险平价＋仓位约束",
          "four_equal": "四方向等权", "three_asset_equal": "沪深300＋黄金＋国债", "buy_hold_300": "沪深300持有"}
SLABELS = ["黄金择时", "跨境轮动", "转债ETF", "红利ETF", "小盘ETF代理", "宽基轮动"]


def markdown_table(frame: pd.DataFrame) -> str:
    names = list(frame.columns)
    return "\n".join(["| " + " | ".join(map(str, names)) + " |",
                       "| " + " | ".join(["---"]*len(names)) + " |"] +
                      ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)])


def draw() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/quant-six-sleeve-mpl")
    # Optional isolated plotting dependencies; the project environment is not modified.
    sys.path.append("/private/tmp/quant-six-sleeve-plotdeps")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.ticker import PercentFormatter
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font_path.exists():
        fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    r = pd.read_csv(OUT/"daily_returns.csv", index_col=0, parse_dates=True)
    r = pd.concat([pd.DataFrame(0., index=[r.index[0]-pd.Timedelta(days=1)], columns=r.columns), r])
    wealth = (1+r).cumprod()
    corr = pd.read_csv(OUT/"sleeve_correlation.csv", index_col=0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    names = ["six_equal", "six_erc", "six_erc_guarded", "three_asset_equal", "buy_hold_300"]
    colors = ["#64748b", "#2563eb", "#059669", "#d97706", "#b45369"]
    for name, color in zip(names, colors):
        axes[0,0].plot(wealth.index, wealth[name], label=LABELS[name], color=color, lw=1.5)
        dd = wealth[name]/wealth[name].cummax()-1
        axes[0,1].plot(dd.index, dd, label=LABELS[name], color=color, lw=1.1)
    axes[0,0].set_title("扣除交易摩擦后的净值（起点=1）", loc="left", weight="bold")
    axes[0,0].legend(fontsize=9, frameon=False)
    axes[0,1].set_title("最大回撤降低，仍可能出现单日大幅亏损", loc="left", weight="bold")
    axes[0,1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes[0]: ax.grid(alpha=.15)
    image = axes[1,0].imshow(corr, cmap="Blues", vmin=0, vmax=1)
    axes[1,0].set_xticks(range(6), SLABELS, rotation=25, ha="right")
    axes[1,0].set_yticks(range(6), SLABELS)
    for i in range(6):
        for j in range(6):
            axes[1,0].text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                           color="white" if corr.iloc[i,j]>.6 else "#172554")
    axes[1,0].set_title("袖套日收益相关性：转债与小盘为0.78", loc="left", weight="bold")
    fig.colorbar(image, ax=axes[1,0], shrink=.7)
    compare = pd.read_csv(OUT/"comparison.csv", index_col=0)
    x = np.arange(4)
    cols = ["six_equal", "six_erc", "six_erc_guarded", "three_asset_equal"]
    axes[1,1].bar(x-.18, compare.loc[cols,"cagr"], .35, label="复合年化收益", color="#2563eb")
    axes[1,1].bar(x+.18, -compare.loc[cols,"max_drawdown"], .35, label="最大回撤幅度", color="#e2a047")
    axes[1,1].set_xticks(x, ["六方向\n等权", "六方向\n风险平价", "风险平价\n加约束", "三资产\n等权"])
    axes[1,1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1,1].set_title("六方向没有明显胜过简单基线", loc="left", weight="bold")
    axes[1,1].legend(frameon=False)
    fig.suptitle("多策略配置探索｜ETF代理组合，不能代表原始六策略", fontsize=19, weight="bold")
    fig.supxlabel("2021-06-01—2026-11-10 · 单边交易摩擦10bp · 无杠杆 · 回顾性研究，非独立样本外", fontsize=11)
    fig.savefig(OUT/"research_overview.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    t = pd.read_csv(OUT/"comparison.csv", index_col=0)
    overview = pd.DataFrame({"组合": [LABELS[x] for x in t.index],
        "复合年化": t.cagr.map(lambda x:f"{x:.2%}").values,
        "最大回撤": t.max_drawdown.map(lambda x:f"{x:.2%}").values,
        "年化波动": t.volatility.map(lambda x:f"{x:.2%}").values,
        "夏普（现金收益0）": t.sharpe_rf0.map(lambda x:f"{x:.2f}").values,
        "最差单日": t.worst_day.map(lambda x:f"{x:.2%}").values,
        "平均现金": t.average_cash.map(lambda x:f"{x:.1%}").values})
    a = pd.read_csv(OUT/"annual_returns.csv", index_col=0)
    a.index = [f"{x}（部分年度）" if x in [2021,2026] else str(x) for x in a.index]
    a = a.rename(columns=LABELS).map(lambda x:f"{x:.2%}").reset_index(names="年份")
    s = pd.read_csv(OUT/"sensitivity.csv")
    sens = s[s.portfolio.eq("six_erc")][["variant","cagr","max_drawdown","worst_day"]].copy()
    sens.columns = ["单项变化","复合年化","最大回撤","最差单日"]
    for col in sens.columns[1:]:
        sens[col] = sens[col].map(lambda x:f"{x:.2%}")
    draw()
    report = f"""# 六方向＋风险平价：第一轮可复现探索

研究日期：2026-11-11。数据截至2026-11-10，共1320个评价交易日。用户允许先假设参数，本轮只研究、不接入生产、不交易。

## 结论

**值得继续做组合研究，但没有证据支持把截图中的六套策略直接一起开实盘。** 在这一组事前固定的代理规则下，风险平价改善了整体波动和回撤，收益提升很小；简单三资产基线的收益和夏普反而更高，但最大回撤较大。六方向并非六个独立收益来源，且宽基轮动可能放大反转日损失。

这是ETF代理组合的回顾性探索，不是原策略复现、独立样本外验证或未来收益预测。尤其不能用中证1000的结果推断真正微盘策略可行。

![净值、回撤、相关性及组合对照](research_overview.png)

## 本轮假设了什么

| 方向 | 实际实现 | 与截图的差异 |
| --- | --- | --- |
| 商品 | 黄金ETF 518880；63日动量为正且收盘高于120日均线才持有 | 只覆盖黄金，不代表原油、工业品等商品 |
| 跨境 | 513500标普500、159920恒生；满足同一趋势条件者中选63日涨幅最高者 | 中国境内ETF真实价格含汇率、溢价影响；没有额外溢价过滤 |
| 可转债 | 持有511380转债及可交换债ETF | 只代表转债市场暴露，不是双低或事件选债策略 |
| 红利现金流 | 持有510880上证红利ETF | 红利代理，未验证现金流质量选股 |
| 微盘 | 持有512100中证1000ETF | **是小盘代理，不是真正微盘**；不覆盖微盘退市与成交难题 |
| 宽基轮动 | 510300、510500、159915中，按同一趋势条件选63日涨幅最高者 | 假设的简单轮动，不是截图中未知原规则 |

月末生成信号，下一个交易日收盘执行，新仓从执行收盘之后获取收益。无合格信号留现金；现金收益设0；不加杠杆、不做空。模拟采用复权收盘价，头寸自然漂移，按实际成交金额单边收取0.10%综合摩擦。参考资金100万元，但采用连续权重，没有模拟整手、最低佣金和账户容量，因此收益率不构成100万元实盘撮合验证。

所有组合共用起点2021-06-01，之前数据只用于信号和协方差预热。基金上市前不虚构价格。基金内部费率已反映在价格中，不重复扣费。期末按市价估值，不强制卖出，因此未扣期末清仓成本。

风险平价用过去120交易日的**袖套净收益（包括空仓）**估计协方差，25%向对角收缩，袖套年化波动下限5%，求等风险贡献。它不同于简单波动率倒数配置。组合穿透到ETF后，在单一现金账本交易，避免重复计算两层手续费。

加约束版在月末目标上限制单袖套预算30%、小盘10%、黄金之外的股性ETF合计60%，模型目标波动最多8%，只降仓、不加杠杆；余款现金。约束后不再严格等风险贡献，也不保证真实波动或回撤不超限。两个调仓日之间仓位允许随价格漂移。

四方向等权只保留黄金、跨境、红利、宽基。三资产基线为沪深300ETF、黄金ETF、5年国债ETF各1/3，月末再平衡，费用和执行时序一致。

## 主要结果

{markdown_table(overview)}

复合年化按实际自然年跨度计算；波动和夏普按252交易日年化；最大回撤包含初始本金。三资产与六方向风险暴露不同，因此这个比较只表明复杂方案没有全面占优，不能把所有差异归因于配置算法。

- 六方向等权→风险平价：年化3.79%→4.03%，最大回撤15.75%→13.28%，年化波动11.60%→9.99%。改进主要体现为风险下降，收益只提高约0.24个百分点。
- 加约束后：年化3.73%，最大回撤11.63%，平均现金约32%。较低风险有相当一部分来自较低仓位，不是“免费提高收益”。
- 三资产基线：年化5.65%，夏普0.68；风险平价为4.03%、0.46。但三资产最大回撤16.17%，高于风险平价13.28%，不存在全面支配。
- 风险平价最差单日-8.12%，反而差于等权-7.13%。平均风险下降，不等于尾部风险必然下降。

## 真正值得注意的四个发现

### 1. 转债和小盘在这段历史中高度相关

袖套日收益相关性：转债ETF与中证1000为0.78；小盘与宽基轮动0.59；转债与宽基轮动0.54。黄金与其他袖套约0.12—0.17。它们有分散作用，但不能按名称把六份仓位视为六份独立风险。

### 2. 空仓时期的低波动，会低估重新入场时的风险

2024-09-30的风险平价信号把29.65%的资金分给宽基轮动，选择创业板ETF159915；按协议在国庆后的2024-10-08收盘执行。2024-10-09，该ETF复权收益约-16.28%，仅这一个仓位就拖累组合约4.83个百分点，组合当天亏损8.12%。

这里的机制是：风险模型使用含空仓时期的策略净值，策略过去没有持仓时很平静，但重新买入高波动资产后风险立刻改变。只用历史策略协方差不能完整反映新的持仓风险。下一轮应同时检查当前持仓穿透后的ETF协方差、条件波动、风险集中度和跳空情景，再确定仓位。

### 3. 这轮收益很依赖黄金，宽基轮动反而拖累

事后消融保持原配置预算，把被移除袖套留现金，不重新优化：风险平价去掉黄金后，年化从4.03%降至0.21%，最大回撤升至20.50%；去掉宽基轮动后，年化变为6.42%，最大回撤7.38%。

这说明收益并非六个模块均衡贡献。**不能据此直接发布“删除宽基”的新策略**：这是看完结果后的诊断，必须在后续冻结规则和新数据中验证。也不能据此预期黄金将来继续贡献相同收益。

### 4. 收益很受行情阶段和参数影响

2021年6月至2023年底，风险平价累计-3.88%；2024年以后累计+28.98%。它没有表现出每个阶段都稳定赚钱的特征。

{markdown_table(a)}

2021从6月1日开始；2026截至11月10日，均非完整年度。

## 参数、成本与执行敏感性

全部参数组事前列在protocol.md中；每次只改变一项，不根据结果挑选并重新包装“最优策略”。下表为风险平价版本：

{markdown_table(sens)}

mom20/mom120：动量20/120日；cov60/cov252：协方差60/252日；cost5bp/cost20bp：单边摩擦0.05%/0.20%；delay2：信号后第二个交易日收盘执行。其他参数不变。完整四类组合结果见sensitivity.csv。

协方差改成60日时，年化下降至2.59%，最差单日达到-10.22%；动量改成120日时年化提高至5.20%，但最差单日也达到-9.77%。更高收益不能代替尾部风险检验。延后执行这次恰好改善收益，也不能据此认定延迟本身有优势。

## 压力测试

- 2022全年：风险平价-9.95%，加约束-8.91%，三资产-3.23%。
- 2024年9—10月反转窗口：风险平价约-0.21%，三资产+10.28%，说明轮动可能错过快速反转，随后在高位入场。
- 2025年4月：风险平价整月+0.55%，但4月7日单日-5.73%。月度正收益会掩盖过程中承受的冲击。
- **人为假设，不是历史或预测**：黄金-10%，美股/港股-25%，转债-15%，红利-20%，小盘-35%，沪深300/500/创业板分别-25%/-30%/-35%，国债+2%。把该冲击施加于每一个历史实际持仓，风险平价在平均持仓上损失约16.34%，最不利历史持仓损失约23.28%；加约束分别约13.30%、17.48%。无盘中止损、救济成交或再平衡。
- 另一小盘流动性压力情景把小盘冲击设为-45%，并假设其他股票和转债同步不同幅度下跌，参数全部见summary.json。这只用于说明集中风险，不能替代微盘个股真实流动性回测。

## 数据与验证边界

数据来自Tushare fund_daily、fund_adj，10只ETF各自保存源文件、日期范围与SHA-256。基金身份另用东方财富连接器核对。原始输入保存在source/；本次重新下载与7只ETF旧快照的两千余日重叠区对账，6只收盘完全一致，创业板ETF最大历史修订0.001元，复权因子一致。

全输入期有3个缺报价日：513500在2022-03-29、512100在2022-09-02、159915在2021-02-08。按前值估值，禁止在缺报价日调仓，主组合未发生因此跳过调仓；缺报价的确切原因未核实，不能把估值平滑当真实低风险。

已通过7项检查：信号/成交收益时序、双边费用与现金自融资、持仓自然漂移、缺报价无法交易、风险贡献求解、初始损失纳入回撤、修改未来行情不改变过去信号或收益。

仍未完成：真实微盘的时点ST/退市/停牌/涨跌停撮合；转债双低及强赎事件；现金流质量因子；分钟级成交和资金容量；跨境溢价过滤；正式基金分红到账与再投资时点；历史所有可选ETF的无幸存者偏差筛选。单边10bp是研究假设，不能覆盖所有极端行情冲击。本研究使用当前可获得的历史修订数据，非逐日历史数据库快照。

## 下一版应如何收敛

1. 把简单三资产设为必须比较的基线，同时保留六方向等权，先证明复杂规则提供额外价值。
2. 优先修正“策略空仓低波动→重新入场高仓位”问题；用持仓穿透风险约束，单独复查2024-10-08/09这一类快速反转。
3. 对黄金依赖做历史分段和失效情景，避免把某一资产的阶段行情误认为稳定多策略收益。
4. 真微盘、双低转债、现金流选股单独建模，补足时点数据和成交约束后，才决定是否纳入组合。不为了凑六个模块而上线。
5. 冻结下一轮规则后做前向模拟，比较信号、成交、成本和回撤。短期模拟用于验证执行，不能单靠数月盈利证明长期有效。

本轮建议：**暂不启动六策略实盘。先验证穿透风险模型，并把微盘、转债选券、现金流选股的独立证据补齐。**

## 文件与复现

- protocol.md：查看结果之前冻结的假设与候选组。
- comparison.csv / annual_returns.csv / monthly_returns.csv：主结果和时间分段。
- allocation_signals.csv / *_holdings.csv / *_ledger.csv：逐月决策、每日实际权重、成本账本。
- sensitivity.csv / ablations.csv：全部预设敏感性和事后消融。
- historical_stress.csv / hypothetical_stress.csv：历史窗口和假设冲击；短窗口CSV年化字段仅为统一程序输出，不用于推断长期收益。
- source_reconciliation.csv / source/manifest.json / summary.json：数据与代码审计。
- scripts/research/fetch_six_sleeve_exploration.py：取数；scripts/research/explore_six_sleeves.py：计算；scripts/research/report_six_sleeves.py：本报告。

在仓库根目录执行：

```bash
python scripts/research/explore_six_sleeves.py
python -m pytest -q tests/test_six_sleeve_exploration.py
python scripts/research/report_six_sleeves.py
```

计算需numpy、pandas、pyarrow；图表另需matplotlib。本机绘图依赖隔离在/private/tmp/quant-six-sleeve-plotdeps，不修改项目环境。行情已经缓存，重算不需要联网；重新取数需要项目已配置的Tushare访问权限。

方法参考：[Tushare官方数据文档](https://tushare.pro/document/2)、[AQR风险平价与现实风险管理](https://www.aqr.com/Insights/Research/White-Papers/Risk-Parity-Risk-Management-and-the-Real-World)、[回测过拟合研究](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)。这些来源支持方法讨论，文中数值来自本轮本地计算。
"""
    (OUT/"report.md").write_text(report, encoding="utf-8")
    print(str(OUT/"report.md"))
    print(str(OUT/"research_overview.png"))


if __name__ == "__main__":
    main()
