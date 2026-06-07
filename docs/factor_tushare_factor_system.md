# Tushare-only 天级别因子体系

更新日期：2026-06-06

本文档整理项目后续长期使用的天级别因子体系。原则是：

```text
1. 生产链路优先只依赖 Tushare。
2. 先做可解释、可复现、可回测的因子，再做复杂 ML / AutoAlpha。
3. 每个因子必须标明原始字段、Tushare 接口、是否有未来函数风险。
4. 因子扩展先服务 B1，再逐步沉淀为通用 Factor Factory。
```

Tushare 官方接口入口：

```text
数据接口总览: https://tushare.pro/document/2
历史日线 daily: https://tushare.pro/document/2?doc_id=27
每日指标 daily_basic: https://tushare.pro/document/2?doc_id=32
复权因子 adj_factor: https://tushare.pro/document/2?doc_id=28
股票技术面因子: https://tushare.pro/document/2?doc_id=296
每日筹码及胜率: https://tushare.pro/document/2?doc_id=293
每日筹码分布: https://tushare.pro/document/2?doc_id=294
个股资金流向: https://tushare.pro/document/2?doc_id=170
融资融券交易明细: https://tushare.pro/document/2?doc_id=59
财务指标数据: https://tushare.pro/document/2?doc_id=79
利润表: https://tushare.pro/document/2?doc_id=33
资产负债表: https://tushare.pro/document/2?doc_id=36
现金流量表: https://tushare.pro/document/2?doc_id=44
指数日线行情: https://tushare.pro/document/2?doc_id=95
指数成分和权重: https://tushare.pro/document/2?doc_id=96
申万行业分类: https://tushare.pro/document/2?doc_id=181
沪深股通持股明细: https://tushare.pro/document/2?doc_id=188
```

## 1. 因子宇宙

| 大类 | 本质 | 天级别优先级 | Tushare 实现状态 |
|---|---|---:|---|
| 量价因子 | 行为金融、供需结构 | 极高 | 已覆盖 daily |
| 动量因子 | 趋势延续 | 极高 | 已覆盖 daily |
| 反转因子 | 均值回归、超跌修复 | 极高 | 已覆盖 daily |
| 波动率因子 | 风险偏好、交易拥挤 | 高 | 已覆盖 daily |
| 流动性因子 | 资金活跃度、交易摩擦 | 极高 | daily + daily_basic 已覆盖基础版 |
| 资金流因子 | 主力行为、资金结构 | 极高 | 待接入 moneyflow |
| 行业因子 | Beta 来源、轮动结构 | 极高 | 待接入行业/指数数据 |
| 风格因子 | Barra 风险暴露 | 极高 | daily + daily_basic + 指数回归 |
| 基本面因子 | 长期定价、质量过滤 | 中 | 待接入 fina_indicator / 三表 |
| 宏观 Regime | 市场状态、因子切换 | 高 | 待接入宏观/利率/指数 |
| 情绪因子 | A 股短线风险偏好 | 高 | 待接入涨跌停、龙虎榜、资金流 |
| 另类/文本因子 | 非结构化 Alpha | 中高 | 暂不进入第一阶段 |
| ML 因子 | 自动组合非线性特征 | 极高 | 基于以上结构化因子训练 |

## 2. 当前项目已覆盖的数据层

### 2.1 Tushare daily

位置：

```text
data/raw/daily/*.parquet
```

字段：

```text
ts_code
trade_date
open
high
low
close
pre_close
change
pct_chg
vol
amount
```

项目标准化字段：

```text
date
symbol
volume = vol
name
```

可支持：

```text
收益率、动量、反转、均线、KDJ、RSI、BOLL、ATR、CCI、OBV、Alpha101/Alpha191 量价子集、波动率、K线结构、突破、趋势斜率。
```

### 2.2 Tushare daily_basic

位置：

```text
data/raw/daily_basic/YYYYMMDD.parquet
```

当前计划字段：

```text
turnover_rate       # 换手率
turnover_rate_f     # 自由流通换手率
volume_ratio        # 量比
pe
pe_ttm
pb
ps
ps_ttm
dv_ratio
dv_ttm
total_share
float_share
free_share
total_mv
circ_mv
```

可支持：

```text
真实换手率、真实量比、Size、市值、估值、股本结构、自由流通比例、股息率。
```

## 3. 量价因子

### 3.1 收益率

表达式：

```python
ret_1d = close / close.shift(1) - 1
ret_3d = close / close.shift(3) - 1
ret_5d = close / close.shift(5) - 1
ret_10d = close / close.shift(10) - 1
ret_20d = close / close.shift(20) - 1
ret_60d = close / close.shift(60) - 1
ret_120d = close / close.shift(120) - 1
```

Tushare 依赖：

```text
daily.close
```

状态：

```text
已实现：return_1d / return_5d / return_10d / return_60d / return_120d
建议补充：return_3d / return_20d
```

### 3.2 动量

表达式：

```python
mom_20 = close / close.shift(20) - 1
mom_60 = close / close.shift(60) - 1
mom_120 = close / close.shift(120) - 1
mom_skip_5_20 = close.shift(5) / close.shift(25) - 1
mom_accel = mom_5 - mom_20
risk_adj_mom_20 = ret_20 / volatility_20
```

Tushare 依赖：

```text
daily.close
```

状态：

```text
已实现：momentum_5d / momentum_20d / momentum_60d
建议补充：momentum_120d / mom_accel / risk_adj_mom
```

### 3.3 反转

表达式：

```python
reversal_1d = -ret_1d
reversal_3d = -ret_3d
reversal_5d = -ret_5d
oversold_5d = ret_5d < -0.10
rsi_reversal = RSI_6 or RSI_14
```

Tushare 依赖：

```text
daily.close
```

状态：

```text
已实现：reversal_5d / rsi_12
建议补充：reversal_1d / reversal_3d / RSI_6 / RSI_14
```

### 3.4 均线与趋势位置

表达式：

```python
ma_5 = mean(close, 5)
ma_20 = mean(close, 20)
ma_60 = mean(close, 60)
close_ma20_ratio = close / ma_20
close_ma60_ratio = close / ma_60
ma_bull = ma_5 > ma_10 > ma_20 > ma_60
ma_divergence = (ma_5 - ma_20) / ma_20
```

Tushare 依赖：

```text
daily.close
```

状态：

```text
已实现：ma_5 / ma_10 / ma_20 / ma_60 / ma_120 / ema_5 / ema_10 / ema_20
建议补充：close_ma20_ratio / close_ma60_ratio / ma_bull / ma_divergence
```

### 3.5 成交量

表达式：

```python
volume_relative_20d = volume / mean(volume, 20)
volume_relative_60d = volume / mean(volume, 60)
up_volume_strength = ret_1d * volume_relative_20d
pullback_shrink = -abs(ret_3d) / volume_relative_20d
obv = cumulative signed volume
```

Tushare 依赖：

```text
daily.vol
```

状态：

```text
已实现：obv / volume_relative_5d / volume_relative_20d / volume_relative_60d / volume_breakout_20d / volume_breakout_60d
```

注意：

```text
旧代码里的 turnover_ratio 实际是 volume / mean(volume, 60)，不是 Tushare daily_basic.turnover_rate。
当前新训练特征已改名为 volume_relative_60d，并用 daily_basic.turnover_rate 表示真实换手率。
```

### 3.6 VWAP

表达式：

```python
vwap_approx = amount / volume
vwap_typical = (high + low + close) / 3
vwap_deviation = (close - vwap) / vwap
vwap_trend_20 = vwap / vwap.shift(20) - 1
```

Tushare 依赖：

```text
daily.amount
daily.vol
daily.high / low / close
```

状态：

```text
当前 Alpha005 使用 typical price 近似 VWAP。
建议补充基于 amount / vol 的真实成交均价近似。
```

### 3.7 K 线结构

表达式：

```python
upper_shadow = (high - max(open, close)) / close
lower_shadow = (min(open, close) - low) / close
body = abs(close - open) / open
close_position = (close - low) / (high - low)
consecutive_up_days = rolling consecutive close > prev_close
```

Tushare 依赖：

```text
daily.open / high / low / close
```

状态：

```text
部分已隐含在 Alpha191 与 B1 条件中。
建议显式加入上影线、下影线、实体长度、收盘位置。
```

## 4. 波动率因子

表达式：

```python
volatility_20d = std(ret_1d, 20)
volatility_60d = std(ret_1d, 60)
downside_volatility = std(min(ret_1d, 0), window)
atr_14 = ATR(high, low, close, 14)
skew_20 = skew(ret_1d, 20)
kurt_20 = kurt(ret_1d, 20)
```

Tushare 依赖：

```text
daily.high / low / close
```

状态：

```text
已实现：volatility_20d / volatility_60d / downside_volatility_20d / downside_volatility_60d / atr_14
建议补充：skew_20 / kurt_20 / high_low_volatility
```

## 5. 流动性因子

### 5.1 真实换手率

表达式：

```python
turnover_rate
turnover_rate_f
turnover_rate_change_20 = turnover_rate / mean(turnover_rate, 20)
```

Tushare 依赖：

```text
daily_basic.turnover_rate
daily_basic.turnover_rate_f
```

状态：

```text
已加入 daily_basic 例行拉取。
建议进入下一版 B1 / 通用模型训练。
```

### 5.2 量比

表达式：

```python
volume_ratio
volume_ratio_change = volume_ratio / mean(volume_ratio, 20)
```

Tushare 依赖：

```text
daily_basic.volume_ratio
```

状态：

```text
已加入 daily_basic 例行拉取。
训练中命名为 ts_volume_ratio，避免和本地 volume_relative_* 混淆。
```

### 5.3 Amihud 非流动性

表达式：

```python
amihud = abs(ret_1d) / amount
amihud_20 = mean(amihud, 20)
```

Tushare 依赖：

```text
daily.amount
daily.close
```

状态：

```text
建议补充。
```

### 5.4 成交额

表达式：

```python
amount
log_amount = log(amount)
amount_relative_20 = amount / mean(amount, 20)
```

Tushare 依赖：

```text
daily.amount
```

状态：

```text
建议补充。
```

## 6. 资金流因子

表达式：

```python
moneyflow_net = buy_sm_amount + buy_md_amount + buy_lg_amount + buy_elg_amount
                - sell_sm_amount - sell_md_amount - sell_lg_amount - sell_elg_amount
large_net = buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount
large_net_ratio = large_net / amount
```

Tushare 依赖：

```text
moneyflow / 个股资金流向
沪深股通持股明细
融资融券交易明细
```

状态：

```text
未接入。
建议作为第二阶段扩展，先做低频滚动特征，不直接用于 B1 第一版重训。
```

原因：

```text
资金流数据容易受口径、更新时间和权限影响，必须单独审计覆盖率。
```

## 7. 趋势结构因子

表达式：

```python
slope_20 = slope(log(close), 20)
r2_20 = regression_r2(log(close), 20)
breakout_20 = close / rolling_max(high, 20)
breakout_60 = close / rolling_max(high, 60)
adx_14 = ADX(high, low, close, 14)
```

Tushare 依赖：

```text
daily.open / high / low / close
```

状态：

```text
已实现部分趋势类技术指标。
建议补充 slope / r2 / breakout。
```

## 8. 行业因子

表达式：

```python
industry_ret_20 = mean(ret_20 of industry constituents)
industry_rank_20 = cross_section_rank(industry_ret_20)
industry_amount = sum(amount by industry)
industry_amount_change = industry_amount / mean(industry_amount, 20)
industry_limit_up_count = count(limit_up by industry)
```

Tushare 依赖：

```text
stock_basic.industry
申万行业分类
申万行业成分
申万日线行情
涨跌停和炸板数据
每日涨跌停价格
```

状态：

```text
当前仅有 stock_basic 行业字段。
建议第三阶段接入申万行业分类和行业日线，先做行业动量、行业成交额、行业中性化。
```

## 9. Barra 风格因子

| 风格 | 表达式 | Tushare 来源 | 状态 |
|---|---|---|---|
| Size | log(total_mv) / log(circ_mv) | daily_basic.total_mv / circ_mv | daily_basic 已接入 |
| Momentum | ret_20 / ret_60 / ret_120 | daily.close | 已覆盖 |
| Residual Vol | 市场模型残差波动 | daily + 指数日线 | 待接入指数 |
| Liquidity | turnover_rate / amount / amihud | daily + daily_basic | 部分覆盖 |
| Value | PE / PB / PS / dividend | daily_basic | daily_basic 已接入 |
| Growth | revenue_yoy / profit_yoy | fina_indicator | 待接入 |
| Leverage | debt_to_assets 等 | fina_indicator / balancesheet | 待接入 |
| Beta | 个股相对指数回归 beta | daily + index_daily | 待接入 |

## 10. 基本面因子

### 10.1 估值

```python
pe
pe_ttm
pb
ps
ps_ttm
dv_ratio
dv_ttm
```

Tushare 依赖：

```text
daily_basic
```

状态：

```text
daily_basic 已接入，适合加入机器学习模型，但需要缺失值填充和极值处理。
```

### 10.2 成长与盈利质量

```python
revenue_yoy
profit_yoy
roe
roa
gross_margin
netprofit_margin
ocf_to_profit
debt_to_assets
```

Tushare 依赖：

```text
fina_indicator
income
balancesheet
cashflow
```

状态：

```text
未接入。
建议第四阶段做季度财务因子，严格按 ann_date / f_ann_date 防止未来函数。
```

## 11. 宏观 Regime

表达式：

```python
index_trend = index_close / index_ma60 - 1
market_volatility = std(index_ret, 20)
market_breadth = rising_stock_count / total_stock_count
credit_regime = social_financing_yoy
rate_regime = 10Y_yield
```

Tushare 依赖：

```text
指数日线行情
沪深市场每日交易统计
国债收益率曲线
宏观经济数据
```

状态：

```text
未接入。
建议优先用指数日线构建轻量 Regime，再接宏观数据。
```

## 12. 情绪因子

表达式：

```python
limit_up_count
limit_down_count
limit_up_open_rate
fail_board_rate
max_consecutive_board
lhb_net_buy
hot_topic_rank
```

Tushare 依赖：

```text
每日涨跌停价格
涨跌停和炸板数据
龙虎榜每日统计单
龙虎榜机构交易单
题材/概念板块数据
```

状态：

```text
未接入。
适合短线策略和 B1 风险过滤，但要单独做时间可得性审计。
```

## 13. Alpha101 / Alpha191

核心结构：

```text
输入: open / high / low / close / volume / vwap / returns
时间序列算子: mean / std / delta / delay / ts_rank / decay / corr / cov
横截面算子: rank / scale / neutralize
```

Tushare 依赖：

```text
daily
daily.amount / daily.vol 用于 vwap 近似
stock_basic.industry 用于行业中性化
daily_basic.total_mv 用于市值中性化
```

状态：

```text
项目已实现 Alpha101/Alpha191 子集。
下一步应补充横截面 rank 和行业/市值中性化版本。
```

## 14. ML 因子体系

建议模型输入分层：

```text
Level 1: daily 量价基础因子
Level 2: daily_basic 流动性/估值/市值因子
Level 3: 行业/风格/Regime 因子
Level 4: 资金流/情绪因子
Level 5: 文本/图谱/AutoAlpha
```

目标标签：

```text
未来 N 日收益
未来 N 日超额收益
未来 N 日最大上涨
未来 N 日最大回撤
先触发止盈/止损概率
```

B1 当前标签：

```text
label_t1_open_max_high_8pct
label_t1_open_max_high_10pct
label_t1_open_min_low_3pct_below_t0_low
```

## 15. 因子评价

必须产出：

```text
IC
RankIC
ICIR
分层回测
分行业 IC
分市值 IC
Decay
Turnover
覆盖率
缺失率
稳定性
```

最小实现：

```python
ic = corr(factor_t, future_return)
rank_ic = spearman_corr(factor_t, future_return)
icir = mean(ic_series) / std(ic_series)
```

注意：

```text
不能只看模型 AUC 或单次回测收益。
新增因子进入生产前，必须通过覆盖率、缺失率、IC、分层回测和样本外回测。
```

## 16. 中性化

优先级：

```text
1. 行业中性化
2. 市值中性化
3. Beta 中性化
```

Tushare 依赖：

```text
stock_basic.industry / 申万行业
daily_basic.total_mv / circ_mv
index_daily
```

建议：

```text
对横截面选股模型，行业和市值中性化比继续堆更多技术指标更重要。
```

## 17. 实盘约束

所有因子和模型都必须检查：

```text
1. 未来函数：尤其财务、公告、指数成分、行业成分、涨跌停数据。
2. 幸存者偏差：退市、ST、历史股票列表。
3. 涨跌停约束：不能假设涨停可买、跌停可卖。
4. 流动性约束：成交额、换手率、盘口可成交量。
5. 滑点和交易成本。
6. 数据更新时间：daily 15:00-16:00 入库，日内不可提前使用。
```

## 18. 本项目落地路线

### 阶段 1：已在推进

```text
Tushare daily
daily_basic
B1 Tushare-only 特征重训
特征缺失率审计
```

### 阶段 2：下一步优先

```text
增加 daily_basic 派生因子：
  turnover_rate_change_20
  ts_volume_ratio_change_20
  total_mv_log
  circ_mv_log
  free_share_ratio
  valuation_rank

增加日线派生因子：
  upper_shadow
  lower_shadow
  body_ratio
  close_position
  amount_relative_20
  amihud_20
  trend_slope_20
  trend_r2_20
  breakout_20
```

### 阶段 3：行业与 Regime

```text
申万行业分类
行业收益/成交额/涨停数
市场指数趋势
市场宽度
市场波动率
```

### 阶段 4：资金流与情绪

```text
moneyflow
沪深股通
融资融券
涨跌停/炸板
龙虎榜
```

### 阶段 5：基本面与另类数据

```text
fina_indicator
income / balancesheet / cashflow
公告/新闻 embedding
产业链/概念图谱
AutoAlpha
```

## 19. 当前建议

当前不建议一口气把所有因子塞进 B1。更稳的做法是：

```text
1. 先确认 Tushare daily + daily_basic 的覆盖率和缺失率。
2. 用 80~120 个高质量因子重训 B1。
3. 做样本外回测和分阶段回测。
4. 如果回撤没有改善，再引入行业/Regime 过滤。
5. 资金流、情绪、基本面因子单独做 ablation，不直接混入主模型。
```

原因：

```text
更多因子不一定带来更好实盘表现。
稳定性、低换手、低回撤和 Regime 适配，通常比盲目增加变量更重要。
```
