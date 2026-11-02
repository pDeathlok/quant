"""Produce a reviewable report from immutable completed scan diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "reports/research/scan_a_shares_validation_20261007_v3"
IND = ROOT / "reports/research/scan_a_shares_mainline_indices_20261007"


def table(frame: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def pct(value: float) -> str:
    return f"{value:.2%}"


def run() -> None:
    manifest = json.loads((RUN / "manifest.json").read_text())
    script = ROOT / "scripts/research/validate_scan_a_shares.py"
    if hashlib.sha256(script.read_bytes()).hexdigest() != manifest["script_sha256"]:
        raise ValueError("Completed run source differs from current script")
    shutil.copyfile(script, RUN / "script_source.py")
    audit_source = ROOT / "reports/research/scan_a_shares_validation_20261007_v2"
    for name in ["canonical_sql_audit.json", "canonical_sql_annual_inventory.csv"]:
        shutil.copyfile(audit_source / name, RUN / name)
    recall = pd.read_csv(RUN / "bull_coverage_details.csv")
    agg = recall.groupby(["scope", "window", "horizon", "multiple"]).agg(episodes=("symbol", "size"), captured=("captured", "sum"), early=("early_before_1_5x", "sum")).reset_index()
    agg["recall"] = agg.captured / agg.episodes
    agg["early_recall"] = agg.early / agg.episodes
    agg.to_csv(RUN / "bull_coverage_summary.csv", index=False)
    main = agg.loc[agg.scope.eq("known_non_ST") & agg.window.eq(20)].copy()
    main = main.assign(窗口=main.horizon.map({252: "约1年", 504: "约2年"}), 倍数=main.multiple.map(lambda n: f"{n}倍"), 牛股事件数=main.episodes, 扫描覆盖率=main.recall.map(pct), 早期覆盖率=main.early_recall.map(pct))[["窗口", "倍数", "牛股事件数", "扫描覆盖率", "早期覆盖率"]]
    precision = pd.read_csv(RUN / "selected_bull_precision.csv")
    p = precision.loc[precision.window.eq(20)].copy()
    p = p.assign(窗口=p.horizon.map({252: "约1年", 504: "约2年"}), 倍数=p.multiple.map(lambda n: f"{n}倍"), 成熟信号数=p.mature_signals, 后续曾达标=p.peak_hit_rate.map(pct), 窗口末仍达标=p.terminal_hit_rate.map(pct))[["窗口", "倍数", "成熟信号数", "后续曾达标", "窗口末仍达标"]]
    names = {"weekly_rotation": "每周换股对照", "trend_MA120_trail35": "开发期选出的趋势退出", "trend_MA120_trail35_risk_budget": "同一退出＋风险预算", "CSI300_price_index": "沪深300价格指数"}
    returns = pd.read_csv(RUN / "portfolio_summary.csv")
    r = returns.loc[returns.period.eq("full") & returns.strategy.isin(names)].copy()
    r = r.assign(方案=r.strategy.map(names), 累计净收益=r.total_return.map(pct), 年化收益=r.annual_return.map(pct), 最大回撤=r.max_drawdown.map(pct), 平均暴露=r.average_exposure.map(pct))[["方案", "累计净收益", "年化收益", "最大回撤", "平均暴露"]]
    stages = returns.loc[returns.strategy.isin(names) & ~returns.period.eq("full")].pivot(index="strategy", columns="period", values="total_return").reindex(names)
    stages.index = stages.index.map(names)
    stages = stages.rename(columns={"development": "2018–2021开发期", "chronological": "2022–2024后续诊断", "recent": "2025至今回看"})
    stages = stages[["2018–2021开发期", "2022–2024后续诊断", "2025至今回看"]].map(pct).reset_index(names="方案")
    trade_rows = []
    for name in names:
        path = RUN / f"trades_{name}.csv"
        if not path.exists():
            continue
        f = pd.read_csv(path)
        closed = f.loc[f.net_multiple.notna()]
        trade_rows.append({"方案": names[name], "已结束交易": len(closed), "未平仓": len(f)-len(closed), "盈利交易比例": pct(closed.net_multiple.gt(1).mean()), "实现3倍交易数": int(closed.net_multiple.ge(3).sum()), "实现5倍交易数": int(closed.net_multiple.ge(5).sum()), "持有天数中位": closed.holding_days.median()})
    trades = pd.DataFrame(trade_rows)
    trades.to_csv(RUN / "trade_summary.csv", index=False)
    # Exposure-matched reference: 50% daily-reset CSI300 price index, no interest.
    b = pd.read_parquet(ROOT / "data/raw/index_000300.SH.parquet")
    b["date"] = pd.to_datetime(b.trade_date.astype(str), format="%Y%m%d")
    b = b.sort_values("date").set_index("date")
    daily = b.close.pct_change(fill_method=None).loc["2018-01-01":"2026-09-30"].fillna(0)
    nav = (1 + daily*.5).cumprod()
    peaks = np.maximum.accumulate(np.r_[1., nav.to_numpy()])[1:]
    benchmark = {"total_return": float(nav.iloc[-1]-1), "annual_return": float(nav.iloc[-1]**(252/len(nav))-1), "max_drawdown": float((nav/peaks-1).min()), "exposure": .5, "notes": "50% daily reset CSI300 price index plus zero-yield cash; no dividends/rebalance costs; illustrative reference"}
    (RUN / "benchmark_half_exposure.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2))
    industries = pd.read_csv(IND / "summary.csv")
    i = industries.loc[industries.period.eq("full")].copy()
    i = i.assign(窗口=i.horizon.map(lambda h: f"{h}交易日"), 前三精确率=i.top3_precision.map(pct), 同日随机参照=i.matched_random_precision.map(pct), 前三覆盖率=i.top3_recall.map(pct), 所选平均后续收益=i.selected_mean_return.map(pct), 全行业平均=i.all_industry_mean_return.map(pct))[["窗口", "前三精确率", "同日随机参照", "前三覆盖率", "所选平均后续收益", "全行业平均"]]
    # Concrete successes and misses, all mechanically chosen rather than famous-stock anecdotes.
    candidates = recall.loc[recall.scope.eq("known_non_ST") & recall.window.eq(20) & recall.horizon.eq(504) & recall.multiple.eq(5)]
    examples = pd.concat([candidates.loc[candidates.captured & candidates.early_before_1_5x].nlargest(8, "peak_multiple"), candidates.loc[candidates.captured & ~candidates.early_before_1_5x].nlargest(8, "peak_multiple"), candidates.loc[~candidates.captured].nlargest(8, "peak_multiple")]).drop_duplicates(["symbol", "year"])
    examples.to_csv(RUN / "bull_examples_success_late_miss.csv", index=False)
    # Preserve original run metadata; add a correction instead of silently rewriting its history.
    addendum = {"original_manifest_preserved": True, "canonical_access_correction": "Original manifest retained an earlier sandbox-only unavailable flag. Escalated read-only MySQL audit succeeded; 10 sampled dates match mirror codes/close exactly. Not proof every field/day matches.", "raw_name_missing_correction": "Missing name fraction describes original price bar columns. Actual entry gating uses downloaded announced historical name intervals; see historical_status_coverage.csv.", "mainline_extension": str(IND.relative_to(ROOT)), "decision": "Do not promote mechanical module or experimental exit/position rules into a validated production strategy", "full_strategy_validated": False, "strict_executable_return_validated": False, "tests": "10 behavioral tests passed; 80-symbol causal prefix test passed; industry prefix test passed", "remaining_gaps": ["Full PIT policy, business exposure, industry monthly quantities/prices, counterparties, announcement timestamps and B/C signal reconstruction", "Exact board-specific historical price limits, intraday tradability, lots, capital capacity, cash/share corporate-action ledger", "Independently archived historical lifecycle/status and taxonomy vintages", "Untouched prospective holdout after rule freeze"]}
    (RUN / "validation_addendum.json").write_text(json.dumps(addendum, ensure_ascii=False, indent=2))
    report = f"""# A股扫描规则历史验证：第一轮模块诊断

生成日期：2026-10-07。行情截止：2026-09-30。研究区间：2018-01-02至2026-09-30，2123个市场交易日；2017年用于指标预热。

**结论：尚未达到严格验证通过条件，不能把原Skill或新增仓位/退出规则称为有效交易策略。** 已实际完成量价扫描、牛股后续走势、机械交易模拟以及行业趋势代理。完整A/B/C政策—产业—事件策略尚未重建；模拟净收益不等于经逐笔可成交核验的实际收益。

## 一、测试了什么

用户允许将3倍/5倍定义为股价达到买入起点的3/5倍，即+200%/+400%；以252/504市场交易日近似1/2年，采用后续最高**收盘价**。用户也允许定义缺失的退出规则并进行优化。

本轮先将可量化的扫描部分固定为：120条历史有效日线、近20日平均成交额≥5000万元、收盘高于MA60、20日或60日收益为正，每周取涨幅前20。两种涨幅窗口均展示，组合模拟使用20日窗口。本轮未重建政策关键词、产业量价/订单、主营收入敞口、对手方证据、重组概率与事件强弱，不能将该量价对照代替原完整策略。

退出实验为MA20/60/120×从持仓峰值收盘价回撤15%/25%/35%，共9组；初始3ATR止损幅度限制8%–20%，最长504交易日，无固定止盈。这里只是用户授权的机械研究定义，并非证明等同“三卖”“第五层”“甜点回踩”。买入使用信号后的次日开盘；收盘触发退出，次日开盘卖出。只按2018–2021开发期净年化选择最好的退出组合，不用后续牛股标签选参数。

**“最好的退出”仍然亏损。** 选中MA120＋35%峰值回撤，开发期累计-39.16%。2022–2024用于后续时间段诊断，2025以来用于近期回看；由于这些历史结果已被观察，均不得包装成新鲜盲测。暂未进行根据全样本结果反复调参。

## 二、能覆盖3倍、5倍牛股吗

为避免“只在满足扫描条件的股票里找赢家”，牛股分母不要求站上均线或已有正收益，只要求历史长度和流动性。每月月末设锚，次日开盘作为理论起点；每只股票/锚点年份/倍数/窗口保留后续最高倍数的一个事件。同一股票可出现在不同年份，不是全市场历史大底到大顶的穷尽统计。未来走势只用作评价标签，绝不参与当日选股。

下表使用锚点时有已知非ST状态的机会分母；全部状态口径另见bull_coverage_summary.csv。覆盖指在锚点之后、首次达到目标倍数之前至少进入过扫描名单；**早期覆盖**要求首次入选时尚未超过锚点起价的1.5倍。

{table(main)}

3倍/5倍事件的覆盖率不能理解为“买入后涨3倍/5倍的概率”。5倍事件往往持续更久、上涨更强，给动量扫描更多入选机会，因此其覆盖率高于3倍不构成早期识别能力的证据。早期覆盖只有约4%–7%，显示入选偏晚。

**从入选日起再涨3倍/5倍的概率：**

{table(p)}

最高收盘达标只说明期间曾有走势，窗口末达标也不是策略实现收益。重复周信号并非独立样本。近一年/两年未成熟信号分别单列，不当作失败；20日窗口252日评价为7857条成熟/8898条总信号，504日为6778/8898。详细入选日、首次目标达标日、剩余倍数和机械选择的成功/晚入选/遗漏样例均保留。

## 三、能命中主线吗

先补做2018年以来申万一级行业指数趋势代理：2018–2021使用2014分类列表，2022起使用2021分类列表；每日按20日涨幅分位65%＋60日涨幅分位35%排名，要求20日涨幅为正且收盘高于MA60，选前2。评价目标为同日可用行业中，未来20/60日涨幅前三；跨分类切换窗口不评价。32个不重复行业代码，未再优化参数。

{table(i)}

有行业排名信息，但不稳定。2022–2024的20日平均后续收益：所选-0.84%，全行业-0.33%；60日所选-2.05%，全行业-0.46%。全期60日所选0.93%也低于全行业0.98%。随机参照按同日有效行业数匹配；重叠日样本不是独立试验，本轮不据此宣称统计显著。

**这不能证明原“政策＋产业＋市场”主线规则有效。** 供应商行业行情默认2021版，旧指数历史是否被回溯改编尚未取得独立历史版本证明；一级行业也无法替代AI液冷、国产替代、基因测序等细分主题。详情和源数据留于../scan_a_shares_mainline_indices_20261007/。

另保留2024–2026年10个预先选定主题、上一年末成员名单的市场趋势试验：20日后涨幅前三精确率35.87%，60日27.33%，但所选平均后续收益均低于全主题平均。该主题试验有名单挑选及覆盖局限，不能外推至2018年以来全部主线。

## 四、模拟操作收益

20个持仓槽，总暴露上限50%。等权每次投入净值2.5%；风险预算方案每笔预算风险0.25%，权重=min(5%,0.25%/(初始止损距离＋1%缓冲))。并非Kelly估计，更未证明比原1/3试仓更优。

计入单边0.10%成本及按日期切换的卖出印花税。由于完整涨跌停可成交历史不足，保守使用开盘向上缺口≥4.8%拒买、向下缺口≤-4.8%延期卖出的**代理**；不能把它称作各板块真实涨跌停判定。遵守次日成交及T+1，停牌/缺失日持仓保留，已知退市按零回收压力处理；连续价格份额近似公司行动，未完整核对现金分红、配股及股份账簿。

{table(r)}

价格指数100%暴露，策略约30%–49%，不作风险相同的比较。补充50%每日重置沪深300＋零收益现金参照：累计{pct(benchmark['total_return'])}，年化{pct(benchmark['annual_return'])}，最大回撤{pct(benchmark['max_drawdown'])}；未含指数分红/再平衡成本，仅作说明。

分时段累计净收益：

{table(stages)}

完成交易统计：

{table(trades)}

趋势退出确有少数实现3倍/5倍交易，但不足以抵消多数亏损交易。未平仓交易不混入已实现胜率，期末市值进入组合净值。持有天数为市场交易日。风险预算降低损失的同时降低平均暴露，不能将亏损减少解释成稳定盈利优势。单边成本提高到0.30%时，趋势方案累计-74.42%，风险预算方案-55.93%。

## 五、数据审计与验证边界

- 读取约1056万条日线，5816个证券代码，1037万条满足OHLC/成交量有效性。源文件与脚本哈希已保存，实际每日覆盖见daily_coverage.csv。
- 原库244只2018年以来已知退市股仅覆盖14只，补取其余230只、309476条日线。补取37批1990–2026名称变更记录，共14229条。采用公告日及生效日控制历史ST状态，未知状态不允许进场；名称日线列缺失比例与补充后的实际状态覆盖是不同指标。没有独立旧版本档案证明状态和退市全集完全无遗漏。
- 当前MySQL只读访问已成功，10个抽样交易日与本地镜像的代码集合、收盘价完全一致。原manifest仍保留早期沙箱连接失败的旧标记；validation_addendum.json明确纠正，不能据此声称MySQL当前不可用。抽查不是所有字段/日期逐条认证。
- 80个证券的截断前后特征/历史状态一致性检验通过；10个行为测试通过，包括未来公司行动不改写过去、未来公告不提前激活、次日成交、T+1、缺口延期、未知状态拒买、退市损失、交易与净值核对及成熟窗口处理。行业指标截断一致性也通过。代码正确性检查不等于策略盈利验证。
- 历史逐日真实涨跌停、公告发布时间到开盘可用时点、全公司行动资金股份账、100股/板块手数、策略资金规模和冲击成本未完整建模。因此用户第3项“实际操作收益”目前仅交付有成本的机械模拟，严格可成交收益尚未完成。
- 历史政策、产业数据、业务映射、事件及重组信号未完整重建。因此不能声称完成原完整A/B/C规则的2018至今严格回测。

## 六、对Skill的处置

前次“文件格式、引用路径、安装内容已验证”只说明交付工程检查完成，并不能为仓位和交易规则背书。固定1/3试仓、风险预算以及退出规则需要在同一完整信号、同一资金约束下比较，本轮没有完成该对照。当前不再更改或提升正式Skill中的交易规则；上述机械定义和研究结果保留在研究目录。

完整验收仍需要：逐日只使用当时已经公开的政策/产业/公告重建A/B/C信号；准确业务映射与历史退市/ST全集；真实可成交和公司行动账簿；原仓位与候选仓位的同条件对照；规则冻结后的未触碰验证期。后续若改动规则，须另开版本，保留本轮失败结果与全部尝试数，不通过全历史反复调参制造一条漂亮收益曲线。

## 可复核文件与来源

- manifest.json / protocol.json / validation_addendum.json：范围、固定参数、输入哈希、修正说明。
- bull_coverage_summary.csv / bull_coverage_by_year.csv / bull_examples_success_late_miss.csv：总体/年度牛股覆盖与具体样例。
- selected_bull_precision.csv / selected_forward_outcomes.parquet / signals.parquet：入选后的走势、成熟度及完整信号。
- portfolio_summary.csv / nav_*.csv / trades_*.csv：净值、逐交易与分期收益。
- canonical_sql_audit.json / historical_status_coverage.csv / daily_coverage.csv：数据覆盖审计。
- script_source.py：完成本次运行的源代码快照。
- [Tushare日线字段说明](https://tushare.pro/document/2?doc_id=27)：成交额单位、除权前收盘定义。
- [名称变更接口](https://tushare.pro/document/2?doc_id=100)：历史名称、生效区间和公告日。
- [行业分类接口](https://tushare.pro/document/2?doc_id=181)、[行业日线说明](https://tushare.pro/document/2?doc_id=327)：分类版本和行情默认版本。
"""
    (RUN / "report.md").write_text(report, encoding="utf-8")
    print(str(RUN / "report.md"))


if __name__ == "__main__":
    run()
