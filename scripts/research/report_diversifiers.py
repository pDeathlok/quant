"""Write the follow-up diversification report from cached experimental results."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from report_six_sleeves import markdown_table

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/research/diversifiers_20261111"
NAMES = {"original_six_erc": "原六方向风险平价", "core80_cash20": "80%原组合＋20%现金",
         "core80_gov5_20": "80%原组合＋20%5年国债", "core80_gov10_20": "80%原组合＋20%10年国债",
         "core80_soymeal20": "80%原组合＋20%豆粕", "core80_energychem20": "80%原组合＋20%能源化工",
         "core80_metals20": "80%原组合＋20%有色金属", "core80_commodity_basket20": "80%原组合＋20%商品等权篮子",
         "core60_cash40": "60%原组合＋40%现金", "core60_gov5_20_gov10_20": "60%原组合＋20%5年国债＋20%10年国债",
         "core60_gov5_20_commodities20": "60%原组合＋20%5年国债＋20%商品篮子",
         "core80_gov30_20": "80%原组合＋20%30年国债"}


def plot(stats: pd.DataFrame, comparison: pd.DataFrame) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    colors = ["#059669", "#059669", "#2563eb", "#d97706", "#b45369"]
    x = np.arange(len(stats))
    axes[0].barh(x, stats.correlation_daily, color=colors)
    axes[0].set_yticks(x, stats.name)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#334155", lw=.8)
    axes[0].set_xlim(-.3, .65)
    for i, value in enumerate(stats.correlation_daily):
        axes[0].text(value+(.015 if value >= 0 else -.015), i, f"{value:.2f}", va="center",
                     ha="left" if value >= 0 else "right")
    axes[0].set_title("与原六方向风险平价的日收益相关性", loc="left")
    selected = ["original_six_erc", "core80_cash20", "core80_gov10_20", "core80_soymeal20", "core80_energychem20"]
    xs = np.arange(len(selected)); part = comparison.loc[selected]
    axes[1].bar(xs-.18, part.cagr, .35, label="复合年化收益", color="#2563eb")
    axes[1].bar(xs+.18, -part.max_drawdown, .35, label="最大回撤幅度", color="#e2a047")
    axes[1].set_xticks(xs, ["原组合", "20%\n现金", "20%\n10年国债", "20%\n豆粕", "20%\n能源化工"])
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("替换20%原组合后：低相关不保证低回撤", loc="left")
    axes[1].legend(frameon=False)
    fig.suptitle("低相关投资扩展｜国债优先，商品需检查自身风险", fontsize=19)
    fig.supxlabel("2021-06-01—2026-11-10 · 月度调仓 · 单边交易摩擦10bp · 固定比例研究，非建议仓位", fontsize=11)
    fig.savefig(OUT/"diversifiers.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    stats = pd.read_csv(OUT/"candidate_stats.csv")
    compare = pd.read_csv(OUT/"comparison.csv", index_col=0)
    overview = pd.DataFrame({"候选资产": stats.name, "代码": stats.code,
        "日相关性": stats.correlation_daily.map(lambda x:f"{x:.2f}"),
        "周相关性": stats.correlation_weekly.map(lambda x:f"{x:.2f}"),
        "原组合最差5%日的平均收益": stats.mean_return_on_core_worst5pct_days.map(lambda x:f"{x:.3%}"),
        "资产自身最大回撤": stats.max_drawdown.map(lambda x:f"{x:.2%}")})
    results = pd.DataFrame({"组合": [NAMES[x] for x in compare.index],
        "复合年化收益": compare.cagr.map(lambda x:f"{x:.2%}").values,
        "最大回撤": compare.max_drawdown.map(lambda x:f"{x:.2%}").values,
        "年化波动": compare.volatility.map(lambda x:f"{x:.2%}").values,
        "最差单日": compare.worst_day.map(lambda x:f"{x:.2%}").values})
    short = pd.read_csv(OUT/"gov30_short_window.csv", index_col=0)
    short_table = pd.DataFrame({"组合": [NAMES[x] for x in short.index],
        "复合年化收益": short.cagr.map(lambda x:f"{x:.2%}").values,
        "最大回撤": short.max_drawdown.map(lambda x:f"{x:.2%}").values})
    plot(stats, compare)
    text = f"""# 低相关投资扩展：国债、商品与另类策略

研究日期2026-11-11，数据截至2026-11-10。接续前轮六方向ETF代理组合，不改动生产配置或执行交易。

## 结论与研究优先级

**有值得加入的方向。优先把人民币国债加入原六方向组合；非黄金商品提供不同收益来源，但需要单独控制波动；CTA趋势和市场中性值得列入下一层候选，必须取得真实费后收益与可实施性证据。**

前轮国债只出现在三资产比较基线中，没有放入六方向组合。本轮把它正式作为新增候选测试。5年、10年、30年国债是同一利率风险的不同久期，不能算三种独立收益来源。

![相关性与组合效果](diversifiers.png)

## 境内ETF实际测得的结果

主窗口2021-06-01—2026-11-10，共1320个交易日。相关性相对于前轮六方向风险平价组合；相关性用ETF复权价格收益，组合绩效另计交易摩擦。原组合最差5%日为全样本事后识别的66天，仅用于诊断，不作为交易信号。

{markdown_table(overview)}

- **5/10年国债**：全期日相关性约-0.13/-0.16，周相关性约-0.21/-0.21；原组合最差交易日中，正收益占比约58%/64%，有一定分散作用，但不是每次股跌都上涨。
- **豆粕期货**：相关性约0.09，可补充农业供需、天气等风险来源；但自身最大回撤26.60%，原组合最差交易日平均约-0.20%，不是现金替代品或可靠的暴跌对冲。
- **能源化工期货**：相关性约0.19，但自身最大回撤48.36%，不能因低相关而给它和国债同等资金权重。
- **有色金属期货**：相关性约0.51，原组合最差交易日平均约-1.41%；受增长和风险偏好影响，对这套已有股票、转债和黄金的组合，分散优先级较低。

相关性会变：10年国债的120日滚动相关性约-0.50至+0.19；能源化工约-0.33至+0.56。全期相关性不能当固定常数用于未来配置。

## 加入以后是否真的改善组合

先固定“80%原组合＋20%候选”，同时做“80%原组合＋20%现金”的对照；另有40%新增资产的预先约定方案。比例是研究探针，不是推荐仓位。原组合内部本来就会持有现金，因此80%原组合不等于80%股票。

继承月末信号、下一交易日收盘执行、无杠杆、头寸自然漂移、单边10bp费用及期末不强制清仓。新增资产长期持有并月度回到预算，旧六方向信号和风险平价预算冻结，没有为了配合新资产重新优化。全部穿透到一个现金账本，未重复收费。

{markdown_table(results)}

**10年国债的增益相对清楚**：20%换成现金时年化3.27%、最大回撤10.74%；换成10年国债时年化4.15%、最大回撤9.87%。它比单纯降仓多提供了一些收益与风险缓冲。相对原组合，回撤从13.28%下降至9.87%，但其中很大一部分本来就能通过减少原组合仓位获得。

**增加40%国债敞口值得继续验证**：60%原组合＋20%5年国债＋20%10年国债，年化约4.00%、最大回撤6.55%；相同原组合比例、剩余现金时为2.49%和8.15%。这是该历史利率环境下的结果，债券票息和利率下降产生的资本利得不能简单外推。

**低相关不保证降低回撤**：20%能源化工使组合最大回撤扩大到16.93%；20%豆粕虽提高全期年化至5.48%，但最大回撤12.13%，也没有国债版稳定。豆粕在2021年6月—2023年阶段改善明显，2024年以后的组合年化7.36%，反而低于原组合9.31%；不能把阶段赢家直接当永久配置。

现金对照仍按前轮设为零息，不能等同于实际货币基金或国债逆回购。若采用真实费后现金管理收益，债券相对现金的收益优势可能缩小；本轮没有用假设货基收益冒充实测。

## 30年国债：可研究，但不应当作低波动现金

511090于2023-06-13上市，因此单独比较2023-07-03—2026-11-10，所有组合从同一时点现金启动：

{markdown_table(short_table)}

这个短窗口中30年国债与原组合日相关性约-0.21、周相关性-0.27，但它自身最大回撤约9.20%、年化波动6.41%。久期越长，对利率变化越敏感，受益于降息时弹性较大，利率上行时损失也可能更大。它不是保本工具，短历史也不覆盖完整利率周期。

同一短窗口，5年与10年国债相关性约0.83，10年与30年约0.87。增加期限品种主要是在调节久期，不是增加同等数量的独立风险来源。

## 其他值得纳入考量的方向

| 方向 | 为什么可能有帮助 | 主要失效机制及本轮状态 |
| --- | --- | --- |
| 现金管理、货币基金、短久期高质量债券 | 提供流动性和较低价格波动，缓解被迫卖出 | 收益低于风险资产的可能性；货基并非保本；现金零波动时相关系数不定义。本轮只用零息现金对照，尚未实测具体产品 |
| 多资产、多空CTA趋势 | 在商品、利率、汇率等市场按趋势做多或做空，收益机制与长期持有股票不同 | 震荡反复止损、快速V形反转、交易与展期成本、保证金风险；需要实际产品/执行序列，不能把商品多头ETF等同CTA |
| 股票市场中性 | 对冲部分市场涨跌暴露，尝试保留选股或相对价值收益 | 对冲不完全、风格拥挤、基差变化、借券与对冲成本；“市场中性”名称不保证低风险，本轮未获取可验证产品费后净值 |
| 海外高质量债券 | 不同利率周期与货币因素，可能补充国内资产 | 人民币计价收益会叠加汇率；未对冲汇率的低相关可能主要来自外汇；还需核验渠道、费用和额度 |
| REITs | 租金或基础设施经营现金流与普通股票有差异 | 上市价格仍受利率、经济和流动性影响；历史不长、分红需准确复权，不预设它是股市下跌保护工具 |
| 尾部保护（例如保护性看跌期权） | 为极端下跌提供较直接的保护机制 | 持续权利金成本、期限和执行价选择；需要衍生品能力，不按普通长期正收益资产处理 |

CTA研究支持其在部分持续性股票熊市中的分散价值，但不保证每次下跌都赚钱。[AQR风险缓释组合研究](https://www.aqr.com/insights/research/alternative-thinking/key-design-choices-when-building-a-risk-mitigating-portfolio)

市场中性策略旨在降低方向性市场暴露，实际仍需查看剩余风险与成本。[BlackRock市场中性策略说明](https://www.blackrock.com/us/financial-professionals/insights/market-neutral-investing)

商品期货ETF是商品期货多头风险暴露，展期、跟踪误差、保证金管理和二级市场折溢价都会影响收益。其风险高于低风险现金管理产品。[建信能源化工ETF基金年报](https://pdf.dfcfw.com/pdf/H2_AN202303301584686844_1.pdf)

## 如何进入下一轮研究

1. 先将5/10年国债合并视为一个“人民币利率”模块，选择久期，再研究与原组合的风险预算。30年作为久期扩展单列，不因本次表现较好而直接提高仓位。
2. 豆粕和商品篮子作为分散候选，按波动、压力损失及组合风险贡献限制，而非按产品数量平均分钱；核对历史折溢价和成交容量。
3. CTA与市场中性先找可投资、费后、带成立日期与退出限制的真实收益序列，并剔除仅保留存活产品的选择偏差；没有这些证据就不混入数值排名。
4. 同时检验通常时期、组合最差交易日、年度和行情分段；新增资产须优于同等比例现金对照，不能只追求低相关。

本轮支持的候选顺序是：**人民币国债 → 现金管理 → 经风险约束的非黄金商品 → 取得可靠净值后的CTA/市场中性**。这是一份研究顺序，不是即时买入清单。

## 数据、验证与复现

新增5只ETF的Tushare fund_daily/fund_adj原始行情及SHA-256在source/。基金全称、跟踪指数、上市日期通过东方财富连接器核对；10年国债基金身份可对照[上交所申购赎回清单](https://www.sse.com.cn/disclosure/fund/etflist/detail.shtml?fundid=511260)。

原组合在扩展引擎中逐日复现，最大数值误差小于1e-12；交易逻辑沿用前轮已通过7项检查的自融资与因果时序实现。新数据只有10年国债2018-08-17缺报价，在评价区间之外；本轮组合未因缺报价跳过调仓。原组合旧数据缺口及研究局限继续适用。

已有样本被用于前轮研究，不能把本轮称为独立样本外。标的选择、历史修订、ETF存续选择、收盘可成交、连续权重和现金零息都是限制。前轮原组合中微盘、小盘、双低、转债ETF等代理差异依旧存在。

复现命令（仓库根目录）：

```bash
python scripts/research/explore_diversifiers.py
python scripts/research/report_diversifiers.py
```

重新联网取数才需增加`--fetch`；已有数据重算无需联网。compare结果在comparison.csv，相关性及条件收益在candidate_stats.csv，30年国债只看gov30_short_window.csv和gov30_stats.csv；所有日期与原始源审计见summary.json，事前假设见protocol.md。
"""
    (OUT/"report.md").write_text(text, encoding="utf-8")
    print(OUT/"report.md")


if __name__ == "__main__":
    main()
