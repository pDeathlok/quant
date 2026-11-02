"""Build the ten-year report directly from saved results, without retuning rules."""
import json
from pathlib import Path
import pandas as pd

import run_strategy_lab as lab
from fetch_strategy_ten_year import OUT

LABELS = {
    "rotation_baseline":"240日单资产月度轮动",
    "rotation_vol12_top2":"前二名轮动＋周度12%风险预算",
    "gold_vol12_baseline":"黄金MA200＋月度12%波动预算",
    "pullback_baseline":"原20日趋势回调",
    "pullback_vol12":"趋势回调＋入场12%波动预算",
    "pullback_hold15":"回调最长15日",
    "pullback_hold10":"回调最长10日",
    "pullback_confirmation":"回调次日反弹确认",
    "stable_baseline":"原三资产市场门控",
    "stable_goldcap20_baseline":"弱沪深300转国债＋黄金上限20%",
    "stable_owntrend_vol6":"三资产自身趋势＋周度6%风险预算",
}


def pct(value):
    return "—" if pd.isna(value) else f"{value:.2%}"


def table(headers, rows):
    return "| "+" | ".join(headers)+" |\n| "+" | ".join("---" for _ in headers)+" |\n"+"\n".join("| "+" | ".join(map(str,r))+" |" for r in rows)


def main():
    all_results=pd.read_csv(OUT/"comparison.csv")
    full=all_results.query("period=='ten_year'").set_index("strategy")
    old=pd.read_csv(lab.OUT/"optimization_round2/comparison.csv").query("period=='full'").set_index("strategy")
    annual=pd.read_csv(OUT/"annual_returns.csv")
    sensitivity=pd.read_csv(OUT/"sensitivity.csv")
    dd=pd.read_csv(OUT/"drawdown_cases.csv").set_index("strategy")
    verify=json.loads((OUT/"verification.json").read_text())
    names=list(LABELS)
    comparison=[]
    for n in names:
        r=full.loc[n]
        hit=f"{int(r.wins)}/{int(r.closed_trades)}={pct(r.trade_win_rate)}" if pd.notna(r.get('closed_trades')) else "—"
        comparison.append([LABELS[n],pct(old.at[n,"annual_return"]),pct(r.annual_return),pct(r.calendar_cagr),pct(r.max_drawdown),hit])
    period_rows=[]
    for n in ("rotation_baseline","rotation_vol12_top2","gold_vol12_baseline","pullback_baseline","pullback_vol12","stable_baseline","stable_goldcap20_baseline","stable_owntrend_vol6"):
        data=all_results[all_results.strategy.eq(n)].set_index('period')
        period_rows.append([LABELS[n],pct(data.at['added_2016_2018','annual_return']),pct(data.at['old_window','annual_return']),pct(data.at['recent_2023','annual_return'])])
    yearly=annual[annual.strategy.isin(names)].pivot(index='strategy',columns='year',values='return')
    yearly.to_csv(OUT/'finalist_annual_returns.csv')
    year_rows=[[LABELS[n]]+[pct(yearly.at[n,y]) for y in range(2016,2027)] for n in names]
    stable=[]
    for n in ('stable_baseline','stable_goldcap20_baseline','stable_owntrend_vol6'):
        r=full.loc[n];stable.append([LABELS[n],pct(r.annual_return),pct(r.max_drawdown),pct(r.worst_year),f"{r.annual_return_std*100:.2f}个百分点",f"{int(r.positive_years)}/{int(r.years_count)}"])
    stress=[]
    for n in ('rotation_baseline','rotation_vol12_top2','gold_vol12_baseline','pullback_baseline','pullback_hold15','pullback_confirmation','pullback_vol12','stable_baseline','stable_goldcap20_baseline','stable_owntrend_vol6'):
        frame=sensitivity[sensitivity.strategy.eq(n)].set_index('scenario')
        cost,lag=frame.loc['double_cost'],frame.loc['two_day_execution']
        stress.append([LABELS[n],f"{pct(cost.annual_return)} / {pct(cost.max_drawdown)}",f"{pct(lag.annual_return)} / {pct(lag.max_drawdown)}",pct(lag.get('trade_win_rate'))])
    ddrows=[[LABELS[n],str(dd.at[n,'peak'])[:10],str(dd.at[n,'trough'])[:10],pct(dd.at[n,'drawdown'])] for n in names]
    shortlist=json.loads((lab.OUT/'optimization_round2/frozen_shortlist.json').read_text())
    shortlist.update(historical_start='2016-09-24',historical_end='2026-09-24',review_version='ten_year_20261001',
                     parameters_changed=False,ranking_note='15-day highest eligible point hit rate; gold 12% monthly not 16% ten-year CAGR; keep primary/rival designations exploratory')
    (OUT/'frozen_shortlist.json').write_text(json.dumps(shortlist,ensure_ascii=False,indent=2))
    prefix='[十年数据与规则核验](verification.json)'
    text=f'''# 三类策略：十年窗口重算（2026-10-01）

## 1. 研究问题、范围与执行

按用户要求，观察窗口改成 **2016-09-24—2026-09-24，十个自然年**。2016-09-24为周末，首个观察交易日为9月26日；终点沿用前轮，使比较主要体现扩展历史。实际使用 **2,427个共同交易日**。2015-01-05起数据用于预热，2015年收益不计入结果，因此这不是包含2015股灾或2008危机的回测。

40条规则保持与前轮冻结清单完全相同，共计算120个费用/执行情景。单边0.10%、现金利息0、无杠杆、收盘信号后下一共同交易日收盘执行。配置/轮动在9月23日收盘生成初始目标、首日从现金买入，之后按原月/周周期更新；回调从9月23日起观察信号。全部排除纳指；标普500仍可被轮动选择，国内配置/黄金/回调不含美股。

## 2. 主要发现：十年数值替换前轮数值

{table(['策略','前轮年化（约7年8个月）','十年年化（252日口径）','十年自然年CAGR','十年最大回撤','十年已平仓胜率'],comparison)}

两种年化都从同一条扣费资金曲线复利计算。252日口径为净值^(252/2427)-1，保留与旧报告一致的换算方式；自然年CAGR为净值^(1/约10年)-1。实际每自然年交易日少于252，所以前者略高。后续对“十年收益”的理解应同时看自然年CAGR，不能把252日口径当成每个自然年平均赚到的比例。最大回撤按全期历史高点至后续低点计算；胜率只看完整持仓买卖，不把配置组合的成分调整当成独立交易。

- **年化最高仍是原240日轮动**：16.91%（自然年16.24%），最大回撤27.80%。前二名风险预算版年化12.78%、回撤14.32%，属于降低风险的折中。
- **黄金进攻方案需下调历史收益预期的参考值**：原16.14%的短窗口年化，在十年窗口为11.67%（自然年11.21%）；新增历史段表现弱。周度提高至15%预算的十年年化只有11.37%、回撤17.43%，并没有超过月度12%版。
- **高胜率没有维持80%以上**：原基线/入场限仓版均为31/41=75.61%，新增前段6笔仅2笔盈利。15日版33/42=78.57%为本轮至少30笔且平均交易净收益正的最高点，但回撤14.10%，年化2.76%；不能仅凭此换掉基线。反弹确认版27/35=77.14%、年化4.22%、回撤8.84%，可留作交易期望竞争者，规则并非本轮新挑。
- **稳定配置的跨年表现更值得保留**：原门控7.42%年化/6.45%回撤；黄金20%版5.50%/6.45%，九个完整年正；周度6%组合预算5.22%/5.12%，九年八年正。年度稳定与最小回撤应分别评价。

## 3. 证据：前段弱势不能被后段盈利掩盖

{table(['策略','新增2016—2018段年化','十年账户的2019以后段年化','2023以后段年化'],period_rows)}

这些是同一十年账户的连续分段收益，进入分段时可能已有持仓。它们与旧报告重新起算的短窗口还有初始仓位、首笔费用及日历差异，不能要求逐项完全等于旧报告。旧代码在旧数据/旧起点上复跑的七个主要候选年化和回撤最大误差不到1e-15，表明原结果可复现。

高胜率基线十年Wilson95%区间约60.66%—86.18%；15日版为64.06%—88.29%。仅41/42笔交易、且交易并非独立同分布、规则在已知历史上反复测试，这些区间仍不构成未来胜率保证。长期空仓也意味着十年不一定积累大量交易；基线平均实际仓位15.90%，入场限仓版11.61%。

## 4. 比较：完整年度与风险

完整年度采用 **2017—2025，共9年**；2016仅9月底至年底，2026仅至9/24。

{table(['国内配置','年化（252日）','最大回撤','最差完整年','完整年收益标准差','完整年盈利数'],stable)}

黄金20%版的九个完整年正收益，2016部分年仍为-2.40%；它不是所有任意起止区间都赚钱。6%组合预算仍是估计波动驱动的仓位缩放，不是5%回撤保证。

### 逐年收益

{table(['策略']+[str(y)+('（部分）' if y in (2016,2026) else '') for y in range(2016,2027)],year_rows)}

年度表很宽，可查看[逐年收益CSV](finalist_annual_returns.csv)。黄金2025年+50.39%，早期2017年-3.29%；不能从短窗口六个完整年都赚钱推断跨周期一直稳定。

## 5. 历史case与代价

### 2016年末标普ETF：场内价格含溢价风险

轮动的新最大回撤是2016-12-21—2017-01-24。Tushare记录513500收盘从1.982降至1.431，该段跌幅约27.80%；策略当时持有该ETF。该风险有当时证据：博时基金2016-12-24发布[交易风险提示公告](https://pdf.dfcfw.com/pdf/H2_AN201612240214942862_1.pdf)，说明二级市场明显溢价、QDII额度限制使申购暂停，并提示恢复申购后的溢价收窄风险。回测采用场内成交价格，不能用标普指数或基金净值替换这段损失。这里可以确认溢价风险存在，但没有逐日净值数据来精确分解27.80%中的溢价与标的资产贡献。

### 2016—2018回调：长期趋势成立也会失败

新增段基线六笔：2016-10-11黄金+2.29%；2016-11-16黄金-0.68%；2016-12-07沪深300-3.62%；2017-11-29沪深300-1.12%；2018-02-08沪深300-3.43%；2018-02-14沪深300+2.42%。四亏两赢，年化-1.96%。2018-02-08买入后次日跌破趋势、下一收盘2月12日退出，说明满足入场长期趋势条件仍会经历短期反转。

入场波动限仓在新增段年化-1.95%，同样没有变成盈利策略；当波动估计较低时仓位上限可能不约束，控制账户风险不能修复交易的负期望。新增亏损与后续回撤累计，使全期限仓版最大回撤升至8.98%，高于旧窗口5.72%。

### 黄金慢趋势：早期震荡是重要反例

月度12%黄金方案2016部分年度-8.69%、2017完整年-3.29%，新增段年化-2.13%；最大回撤发生于2016-11-09—2017-11-23，为12.14%。周度15%方案新增段年化-5.79%、十年最大回撤17.43%，更频繁调整和更高仓位未形成十年优势。

### 最大回撤发生区间

{table(['策略','此前高点','后续低点','最大回撤'],ddrows)}

回调的最长回撤区间跨过多笔交易和空仓期，不能误读成一笔持仓亏损。部分峰值/低点日期存在同值时取最早记录的约定。

## 6. 压力测试与备选方案

{table(['策略','费用翻倍：年化/回撤','两交易日执行：年化/回撤','两日执行胜率'],stress)}

费用翻倍仍保持正年化不代表能覆盖所有真实冲击成本。15日回调延迟后胜率73.17%，反弹确认版约70.97%（具体见完整CSV）；延迟后收益偶然增加也不能说执行更慢有优势。国债配置本身有价格风险，现金无利息会拖低部分版本收益。

## 7. 数据、实现核验与开放问题

- 优先沿用本地行情，补取2015—2017或2018年初重叠区数据，接口为[Tushare fund_daily](https://tushare.pro/document/2?doc_id=127)/[fund_adj](https://tushare.pro/document/2?doc_id=199)。7只ETF均自2015-01-05起有数据，重叠区新旧收盘相同、复权因子比例均为1；原缓存未被改写。
- SSE日历有2,429个交易日；159915在2021-02-08、513500在2022-03-29无fund_daily记录，已单日重试核对，复权因子有记录但行情仍为空。沿用共同报价日方法，明确排除这两天，未填造价格；下一日执行指下一共同报价日。这是执行近似，真实停牌/恢复及各资产自身日历需在成交仿真中进一步核对。
- 本轮日历只使用7只实际资产，2019以后比旧含纳指预处理的日历多2022-01-13一日；旧报告不能与“十年账户后半段”逐数字混作同一实验。
- 原规则参数冻结一致；扣费收益与净值恒等、无杠杆、无纳指、回调信号执行滞后以及已平仓现金盈亏核验通过。新数据截到2018-12-28后，四条代表规则此前净值与全历史结果一致。{prefix}。
- 未逐笔仿真盘口、ETF份额整数、QDII限购/溢价过滤或额外滑点；新增历史没有用事后事件修改入场规则。十年不含2015股灾和2008危机，也不是独立未观察样本外验证。

## 8. 当前保留顺序与实施

继续把240日轮动留作最高年化研究对照；需要降低回撤的轮动优先考察前二名风险预算版。黄金月度12%方案仍可作进攻候选，但十年自然年CAGR是11.21%，应替换此前16%左右的短窗口参考。高胜率候选按十年改成75.61%基线、78.57%十五日竞争者，不再描述为已验证80%以上。稳定配置在年际稳定优先时保留黄金20%版，回撤优先时保留周度6%预算版。各候选可能同时持有黄金，需要合并检查总敞口。

规则保存于[冻结清单](frozen_shortlist.json)，尚未改动生产策略。下一步保持规则固定，检验真实成交及新增时序数据；不因十年排名变化再追加参数搜索。

## 9. 文件与复现

[固定协议](protocol.md)、[全部40组及4个分段](comparison.csv)、[120个成本/执行情景](sensitivity.csv)、[所有逐年收益](annual_returns.csv)、[回撤案例](drawdown_cases.csv)、[数据覆盖与对账](source/coverage.json)、[缺报价重试](source/missing_observations.json)、[验证记录](verification.json)。每条策略逐日净值、目标权重和回调交易明细均在同目录。

[补数脚本](/Users/didi/Project/quant/scripts/research/fetch_strategy_ten_year.py)、[缺报价核对](/Users/didi/Project/quant/scripts/research/check_ten_year_missing.py)、[重算脚本](/Users/didi/Project/quant/scripts/research/run_strategy_ten_year.py)、[报告生成](/Users/didi/Project/quant/scripts/research/report_strategy_ten_year.py)。当前机器使用 /usr/bin/python3 执行；补数需要既有Tushare凭证，输出不含凭证。下载底稿已缓存，重算无需网络。
'''
    # Use measured delayed hit rate rather than hand-written assumptions.
    confirmation=sensitivity[(sensitivity.strategy=='pullback_confirmation')&(sensitivity.scenario=='two_day_execution')].iloc[0].trade_win_rate
    text=text.replace('反弹确认版约70.97%',f'反弹确认版{pct(confirmation)}')
    (OUT/'report.md').write_text(text)
    full.loc[names].to_csv(OUT/'finalists.csv',index_label='strategy')
    print('Ten-year report, finalist tables and unchanged shortlist saved.')


if __name__=='__main__':
    main()
