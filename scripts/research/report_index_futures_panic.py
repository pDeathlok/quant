"""Yearly, continuous, matched-window and data-quality evidence for panic factors."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from index_futures_panic import OUT, PREVIOUS, END, QVIX_MAP

LABELS = {f"{style}_{arm}":f"{s}/{a}" for style,s in (("fixed","固定"),("rolling","重选"))
          for arm,a in (("baseline","原状态期限"),("qvix","加QVIX"),("stress","加价格压力"),("both","两者加入"))}


def pct(value): return f"{value:.2%}" if pd.notna(value) else "—"


def table(lines, headers, rows):
    lines += ["| "+" | ".join(headers)+" |", "|"+"---|"*len(headers)]
    lines += ["| "+" | ".join(map(str,row))+" |" for row in rows]
    lines.append("")


def source_quality():
    calendar = set(pd.read_parquet(PREVIOUS.parent/"IH_spot.parquet").trade_date)
    records = []
    for path in sorted((OUT/"sources").glob("qvix_*.parquet")):
        name = path.stem.replace("qvix_","")
        frame = pd.read_parquet(path)
        frame = frame[frame.trade_date.le(END)]
        invalid = frame[~frame.trade_date.isin(calendar)]
        invalid.to_csv(OUT/"sources"/f"off_calendar_{name}.csv",index=False)
        for year,g in frame.groupby(frame.trade_date.str[:4]):
            usable = g[g.trade_date.isin(calendar)]
            records.append(dict(series=name,year=year,source_rows=len(g),calendar_rows=len(usable),
                off_calendar_rows=len(g)-len(usable),first=usable.trade_date.min(),last=usable.trade_date.max()))
    pd.DataFrame(records).to_csv(OUT/"source_quality.csv",index=False)


def model_diagnostics():
    forecasts = pd.read_parquet(OUT/"forecasts.parquet")
    panel = pd.read_parquet(PREVIOUS.parent/"adaptation_comparison/quarterly_predictions.parquet")
    forecasts = forecasts.merge(panel[["entry_date","symbol","long_60","short_60","etf_ret_1"]],on=["entry_date","symbol"],validate="many_to_one")
    forecasts["candidate"] = (np.maximum(forecasts.long_60,forecasts.short_60)>.45)&((forecasts.long_60-forecasts.short_60).abs()>=.1)&forecasts.etf_ret_1.notna()&forecasts.direction.eq(np.where(forecasts.long_60>forecasts.short_60,1,-1))
    forecasts["actual_net"] = forecasts.net_3.where(forecasts.bps.eq(3),forecasts.net_6)
    forecasts["actual_risk"] = forecasts.risk_3.where(forecasts.bps.eq(3),forecasts.risk_6)
    forecasts["year"] = forecasts.entry_date.str[:4]
    records = []
    valid = forecasts[forecasts.candidate&forecasts.ready&forecasts.actual_net.notna()&forecasts.year.ge("2022")]
    for key,g in valid.groupby(["year","model","bps","direction"]):
        selected = g[g.probability.ge(.55)&(g.pred_net-.5*g.pred_risk).gt(0)]
        records.append(dict(zip(["year","model","bps","direction"],key),samples=len(g),passed=len(selected),
            brier=((g.probability-g.actual_net.gt(0))**2).mean(),net_mae=(g.pred_net-g.actual_net).abs().mean(),
            passed_label_win_rate=selected.actual_net.gt(0).mean(),passed_label_mean_net=selected.actual_net.mean(),
            own_missing=g.panic_own_missing.mean(),fallback=g.fallback.ne("none").mean()))
    pd.DataFrame(records).to_csv(OUT/"model_diagnostics.csv",index=False)
    return forecasts


def main():
    source_quality();model_diagnostics()
    d = pd.read_csv(OUT/"summary.csv",dtype={"start":str,"end":str})
    annual = pd.read_csv(OUT/"annual_marked.csv",dtype={"start":str,"year":str})
    d["start_year"] = d.start.str[:4]
    keys = ["startup_mode","fee","margin","window","start_year"]
    grouped = d[d.start_kind.eq("monthly")].groupby(keys).agg(accounts=("start","size"),
        median_return=("total_return","median"),q10_return=("total_return",lambda s:s.quantile(.1)),
        worst_return=("total_return","min"),worst_drawdown=("max_drawdown","min"),
        positive_fraction=("total_return",lambda s:s.gt(0).mean()),median_flat=("flat_ratio","median"),
        median_trades=("trades","median"),worst_blocked=("blocked_ratio","max"),forced_accounts=("forced",lambda s:s.gt(0).sum())).reset_index()
    grouped.to_csv(OUT/"grouped_by_year.csv",index=False)
    whole = d[d.start_kind.eq("monthly")].groupby(keys[:-1]).agg(accounts=("start","size"),
        median_return=("total_return","median"),q10_return=("total_return",lambda s:s.quantile(.1)),
        worst_return=("total_return","min"),worst_drawdown=("max_drawdown","min"),median_flat=("flat_ratio","median"),
        positive_fraction=("total_return",lambda s:s.gt(0).mean())).reset_index()
    whole.to_csv(OUT/"grouped_all.csv",index=False)
    checks = [json.loads((OUT/str(y)/"checks.json").read_text()) for y in range(2022,2027)]
    engines = json.loads((OUT/"engine_checks.json").read_text())
    unique = d.drop_duplicates(["start","startup_mode","fee","margin"])
    audit = pd.read_csv(OUT/"training_audit.csv",dtype={"cutoff":str,"last_maturity":str})
    assert audit.loc[audit.ready,"last_maturity"].lt(audit.loc[audit.ready,"cutoff"]).all()
    lines = ["# A股恐慌因子接入股指策略：QVIX与价格压力（2026-11-15）", "",
        "本轮仅新增因子，主对照为此前保留的固定政策＋季度模型重训＋状态期限20/10/5日。没有按测试年份调门槛或修改退出规则，也没有自动替换当前候选。", "",
        "研究价格和标签统一截到2026-11-06；2026不是完整年度。50万元、最多一手、多空均可、无补资；固定全局5日入场日历；原mean5退出、资金约束与换月成本不变。固定与重选都按季度重训；重选仅在原4种政策中，用已完成的历史多起点窗口、当前实际权益和原切换门槛选择。", "",
        f"完成{len(unique)}个账户、{len(d)}个统计窗口；原方案{sum(x['replayed_controls'] for x in checks)}条账户逐日账本一致；Python/C++核对{engines['daily_cases']}条路径，最大现金差{engines['max_cash_error']}元。", "",
        "## 数据与定义", "",
        "VIX是从期权价格提取的预期波动指标，并非预测下跌概率。AKShare提供第三方optbbs的QVIX系列；这里没有将其称为交易所官方iVIX，也没有确认其与Cboe编制方法完全一致。", "",
        "- IH用50ETF QVIX，IF用300ETF QVIX，IC用500ETF QVIX，IM用1000股指QVIX；另给各品种共同提供50ETF QVIX。每组取过去252日、至少60个有效值的历史分位，连续5日变化，QVIX/100减对应现货指数20日年化实际波动率，以及缺失标记。ETF QVIX与现货指数实际波动之差是代理口径，存在ETF跟踪差异。",
        "- QVIX在信号日期上再滞后一个交易日，之后才在下一开盘交易；缺失不前填、不后填。共同QVIX缺失时，QVIX模型回退原模型，联合模型回退价格压力模型。品种自身QVIX缺失保留标记，逐列用训练期中位数插补。",
        "- 自建价格压力分：下行波动（20日负对数收益平方均值的年化平方根）、60日收盘回撤、5日放量下跌（负对数收益×成交量/此前20日均量）分别取历史分位，再等权平均乘100。模型加入3个分位与压力分5日变化。它是价格压力代理，不是自制隐含波动率指数。",
        "- 采用冻结85项原因子，QVIX模型93项、价格压力89项、联合97项。仍用原5日净收益/不利波动标签，LR C=.1、Ridge alpha=.1*n；36个月成熟历史、多空分开、12个月权重半衰期。未做参数搜索、未改变55%概率和净收益−0.5×风险门槛。", "",
        "### 来源质量", "",
        "本次下载50/300/500ETF QVIX均延续到2026-11-13；1000股指QVIX最后非空日期为2026-03-13，300/50股指为2026-05-29，但后两类2026已大量断档，不能以最后日期冒充完整覆盖。源文件2026-11-02有冲突重复记录，全部隔离；非交易日记录通过交易日历排除，未自行修正日期。完整原始文件、SHA256、覆盖率及隔离记录均保留。", "",
        "Tushare opt_basic（SSE/CFFEX）与2026-11-06 opt_daily（SSE704条/CFFEX760条）权限实测通过。opt_basic这里只取1000条作连通性验证，不是完整合约库。没有将期权日线收盘直接冒充VIX报价中间价；完整自建隐波还需逐历史日期合约、行权价/调整、到期日、利率和报价质量验证。", ""]
    coverage = pd.read_csv(OUT/"coverage.csv")
    table(lines,["年","品种","共同QVIX因子缺失率","自身QVIX因子缺失率"],
        [(r.year,r.symbol,pct(r.common_missing),pct(r.own_missing)) for r in coverage[coverage.year.ge(2025)].itertuples()])
    lines += ["缺失率包含额外滞后、历史分位预热及5日连续性要求；不只是原始close为空的比例。", ""]
    for fee,margin in ((.0003,.15),(.0006,.25)):
        condition = d.fee.eq(fee)&d.margin.eq(margin)
        lines += [f"## {fee*10000:.0f}bps / {margin:.0%}保证金", "", "### 逐年独立单年：收益 / 最大回撤", ""]
        single = d[condition&d.window.eq("single_year")]
        table(lines,["启动年",*LABELS.values()],[(year,*[f"{pct(g.set_index('startup_mode').loc[m,'total_return'])} / {pct(-g.set_index('startup_mode').loc[m,'max_drawdown'])}" for m in LABELS]) for year,g in single.groupby("start_year")])
        lines += ["### 各年初启动累计至共同终点：收益 / 最大回撤", ""]
        cumulative = d[condition&d.window.eq("to_latest")&d.annual]
        table(lines,["启动年",*LABELS.values()],[(year,*[f"{pct(g.set_index('startup_mode').loc[m,'total_return'])} / {pct(-g.set_index('startup_mode').loc[m,'max_drawdown'])}" for m in LABELS]) for year,g in cumulative.groupby("start_year")])
        for window in ("sessions_122","sessions_244"):
            lines += [f"### {window}等长月度起点", ""]
            g = whole[whole.fee.eq(fee)&whole.margin.eq(margin)&whole.window.eq(window)]
            table(lines,["方案","起点数","收益中位","收益10%分位","最差收益","最大回撤","空仓中位"],
                [(LABELS[r.startup_mode],r.accounts,pct(r.median_return),pct(r.q10_return),pct(r.worst_return),pct(-r.worst_drawdown),pct(r.median_flat)) for r in g.itertuples()])
        lines += ["### 近两年月度启动累计至今", ""]
        g = grouped[grouped.fee.eq(fee)&grouped.margin.eq(margin)&grouped.window.eq("to_latest")&grouped.start_year.ge("2025")]
        table(lines,["启动年","方案","收益中位","最差收益","最大回撤","盈利起点比例","空仓中位"],
            [(r.start_year,LABELS[r.startup_mode],pct(r.median_return),pct(r.worst_return),pct(-r.worst_drawdown),pct(r.positive_fraction),pct(r.median_flat)) for r in g.sort_values(["start_year","startup_mode"]).itertuples()])
        lines += ["### 2025启动持续账户：年度收益 / 年内回撤", ""]
        g = annual[annual.start.eq("20250101")&annual.side.eq("all")&annual.fee.eq(fee)&annual.margin.eq(margin)]
        table(lines,["方案","记账年","年度收益","年内回撤"],[(LABELS[r.startup_mode],r.year,pct(r.total_return),pct(-r.max_drawdown)) for r in g.itertuples()])
        lines += ["### 所有月度和相邻日起点中的最差账户", ""]
        g = d[condition&d.window.eq("to_latest")]
        rows = []
        for mode,part in g.groupby("startup_mode"):
            worst = part.loc[part.total_return.idxmin()];dd = part.loc[part.max_drawdown.idxmin()]
            rows.append((LABELS[mode],worst.start,pct(worst.total_return),dd.start,pct(-dd.max_drawdown),int(part.forced.gt(0).sum())))
        table(lines,["方案","最差收益起点","最差收益","最大回撤起点","最大回撤","出现强平账户数"],rows)
    lines += ["## 多空与错失收益", "",
        "逐笔文件trades.csv包含多空、净盈亏与退出日期；annual_marked.csv包含同一持续账户逐年的多空交易胜率、盈亏比和净利润。胜率按已平仓完整交易归属，年度收益按盯市归属，两者不可直接相加。", "",
        "opportunity.csv按相同起点逐日对照：新增空仓期间，原方案盈利日计入错失收益，亏损日计入避损；另外列出新账户平仓费用和其他路径变化，四项与净收益差核对。它是路径归因，不能当作可单独交易的盈利。", ""]
    trades = pd.read_csv(OUT/"trades.csv",dtype={"start":str,"exit":str})
    t = trades[trades.start.eq("20250101")&trades.fee.eq(.0006)&trades.margin.eq(.25)].copy()
    t["year"] = t.exit.str[:4]
    tg = t.groupby(["startup_mode","year","side"]).agg(trades=("pnl","size"),wins=("pnl",lambda s:s.gt(0).sum()),net_pnl=("pnl","sum"),mean_pnl=("pnl","mean")).reset_index()
    tg.to_csv(OUT/"recent_continuous_sides.csv",index=False)
    table(lines,["2025持续账户方案","平仓年","方向","胜/笔数","净盈亏元"],[(LABELS[r.startup_mode],r.year,r.side,f"{r.wins}/{r.trades}",f"{r.net_pnl:,.0f}") for r in tg.itertuples()])
    lines += ["## 证据边界与复现", "",
        "所有方案共用已多次研究的历史，不是新OOT；月度窗口高度重叠。QVIX是本次获取的历史快照，无法证明历史每个数值从未修订；额外滞后只能降低发布时间风险。IM原始QVIX断档严重，本轮IM会较多依赖共同50ETF信号和缺失插补，不能把它解释为完整中证1000期权信息的检验。", "",
        "本轮只检验入场模型新增因子，未检验独立恐慌退出模型，结果不能推广为恐慌指标在所有策略中无效或有效。日线回测不能保证盘中不会穿越保证金要求。", "",
        "复现顺序：fetch_index_futures_panic.py → index_futures_panic.py → evaluate_index_futures_panic.py → report_index_futures_panic.py。目录protocol.json预先固定假设；训练成熟期检查、数据哈希、每笔交易和逐日账本均保留。", "",
        "来源：[AKShare QVIX文档](https://akshare.akfamily.xyz/data/index/index.html)；[AKShare官方实现](https://github.com/akfamily/akshare/blob/main/akshare/index/index_option_qvix.py)；[Tushare合约资料](https://tushare.pro/document/2?doc_id=158)；[Tushare期权日线](https://tushare.pro/document/2?doc_id=159)；[Cboe VIX定义](https://www.cboe.com/tradable_products/vix/faqs)。", ""]
    (OUT/"report.md").write_text("\n".join(lines))
    print("reported",len(unique),"accounts",len(d),"windows",flush=True)


if __name__=="__main__": main()
