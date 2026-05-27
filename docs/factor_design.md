# 因子设计文档（补充版）

## 文档概述

本文档基于天级别量化选股需求，系统梳理了当前项目中已实现和待实现的因子体系。

**核心目标**：预测未来 5~20 日超额收益

**调仓频率**：T+1 ~ T+5

**数据来源**：Tushare Pro

---

## 一、量价类因子（已实现/待实现）

### 1.1 动量因子（Momentum）

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `ret_5d` | 横截面动量 | ✅ 已实现 | 5 | 5日收益率 | daily.close |
| `ret_10d` | 横截面动量 | ✅ 已实现 | 10 | 10日收益率 | daily.close |
| `ret_20d` | 横截面动量 | ✅ 已实现 | 20 | 20日收益率 | daily.close |
| `ret_60d` | 横截面动量 | ✅ 已实现 | 60 | 60日收益率 | daily.close |
| `Momentum_skip5` | 跳过最近5天 | ✅ 已实现 | 20(skip5) | (close.shift(5) / close.shift(25)) - 1 | daily.close |
| `RiskAdjustedMomentum` | 风险调整动量 | ✅ 已实现 | 20 | ret_20 / vol_20 | daily.close |
| `IndustryMomentum` | 行业内动量 | ✅ 已实现 | 20 | 行业内排序后的动量 | daily.close + stock_basic.industry |

### 1.2 短期反转因子（Short Reversal）

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `Reversal_1d` | 1日反转 | ✅ 已实现 | 1 | -ret_1d | daily.close |
| `Reversal_3d` | 3日反转 | ✅ 已实现 | 3 | -ret_3d | daily.close |
| `Reversal_5d` | 5日反转 | ✅ 已实现 | 5 | -ret_5d | daily.close |
| `Reversal_10d` | 10日反转 | ✅ 已实现 | 10 | -ret_10d | daily.close |

### 1.3 均线偏离因子（MA Distance）

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `MA5_Distance` | 均线偏离 | ✅ 已实现 | 5 | close / ma5 | daily.close |
| `MA20_Distance` | 均线偏离 | ✅ 已实现 | 20 | close / ma20 | daily.close |
| `MA60_Distance` | 均线偏离 | ✅ 已实现 | 60 | close / ma60 | daily.close |
| `MA120_Distance` | 均线偏离 | ✅ 已实现 | 120 | close / ma120 | daily.close |
| `EMA12_Distance` | EMA偏离 | ✅ 已实现 | 12 | close / ema12 | daily.close |
| `EMA26_Distance` | EMA偏离 | ✅ 已实现 | 26 | close / ema26 | daily.close |

### 1.4 成交量放量因子

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `VolumeRatio` | 量比 | ✅ 已实现 | 20 | volume / mean(volume, 20) | daily.vol |
| `VolumeRatio_5d` | 5日量比 | ✅ 已实现 | 5 | volume / mean(volume, 5) | daily.vol |
| `VolumeUp` | 放量上涨 | ✅ 已实现 | 20 | ret_1d * volume_ratio | daily.close, daily.vol |
| `VolumeDown` | 缩量回调 | ✅ 已实现 | 20 | -abs(ret_3d) * (1/volume_ratio) | daily.close, daily.vol |
| `VolumeAcceleration` | 量能加速度 | ✅ 已实现 | 5 | delta(volume_ratio, 5) | daily.vol |

### 1.5 VWAP 偏离因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `VWAP_Deviation` | VWAP偏离 | ✅ 已实现 | (close - vwap) / vwap | daily.open, daily.high, daily.low, daily.close, daily.vol |

### 1.6 波动率因子

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `Volatility_20d` | 标准差 | ✅ 已实现 | 20 | std(ret, 20) | daily.close |
| `Volatility_60d` | 标准差 | ✅ 已实现 | 60 | std(ret, 60) | daily.close |
| `ATR` | 平均真实波动 | ✅ 已实现 | 14 | ATR指标 | daily.high, daily.low, daily.close |
| `DownsideVolatility` | 下行波动 | ✅ 已实现 | 20 | 仅下跌期间波动 | daily.close |
| `IdiosyncraticVolatility` | 特质波动 | ✅ 已实现 | 60 | 无法被市场解释的波动 | daily.close + index |
| `RealizedVolatility` | 已实现波动 | ✅ 已实现 | 20 | 基于日内高低价 | daily.high, daily.low |

### 1.7 振幅因子

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `Amplitude` | 日振幅 | ✅ 已实现 | 1 | (high - low) / pre_close | daily.high, daily.low, daily.pre_close |
| `Amplitude_20d` | 平均振幅 | ✅ 已实现 | 20 | mean(amplitude, 20) | daily.high, daily.low, daily.pre_close |
| `AmplitudeStd` | 振幅波动 | ✅ 已实现 | 20 | std(amplitude, 20) | daily.high, daily.low, daily.pre_close |

---

## 二、趋势类因子

### 2.1 趋势斜率因子

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `TrendSlope` | 线性回归斜率 | ✅ 已实现 | 20 | slope(log(close), 20) | daily.close |
| `TrendStrength` | 趋势强度 | ✅ 已实现 | 20 | R² from linear regression | daily.close |
| `MovingAverageSlope` | 均线斜率 | ✅ 已实现 | 20 | slope(ma20, 5) | daily.close |

### 2.2 突破因子（Breakout）

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `DonchianBreakout` | Donchian突破 | ✅ 已实现 | 20 | close / rolling_max(high, 20) | daily.high, daily.close |
| `DonchianBreakdown` | Donchian跌破 | ✅ 已实现 | 20 | close / rolling_min(low, 20) | daily.low, daily.close |
| `PriceRangeRatio` | 价格区间比例 | ✅ 已实现 | 20 | (close - rolling_min(low, 20)) / (rolling_max(high, 20) - rolling_min(low, 20)) | daily.high, daily.low, daily.close |

### 2.3 MACD 类因子

| 因子名称 | 类型 | 状态 | 参数 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `MACD_Line` | MACD线 | ✅ 已实现 | 12,26 | ema12 - ema26 | daily.close |
| `MACD_Signal` | 信号线 | ✅ 已实现 | 9 | EMA(MACD, 9) | daily.close |
| `MACD_Histogram` | 柱状图 | ✅ 已实现 | - | MACD - Signal | daily.close |
| `MACD_Divergence` | MACD背离 | ✅ 已实现 | 20 | 价格与MACD的背离程度 | daily.close |

### 2.4 ADX 趋势强度

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `ADX` | 平均趋向指数 | ✅ 已实现 | 14 | 趋势强度指标 | daily.high, daily.low, daily.close |
| `PlusDI` | +DI | ✅ 已实现 | 14 | 上升动向 | daily.high, daily.low, daily.close |
| `MinusDI` | -DI | ✅ 已实现 | 14 | 下降动向 | daily.high, daily.low, daily.close |
| `ADX_Trend` | ADX趋势信号 | ✅ 已实现 | 14 | PlusDI - MinusDI | daily.high, daily.low, daily.close |

---

## 三、资金流/筹码因子（A股特色）

### 3.1 主力资金因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `LargeOrderNetFlow` | 主力资金净流入 | ⏳ 待实现 | large_order_buy - large_order_sell | 需要盘口数据 |

### 3.2 北向资金因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `NorthboundHoldingChange` | 北向持仓变化 | ⏳ 待实现 | 北向资金持仓变动 | 需要北向资金接口 |
| `NorthboundDailyFlow` | 北向每日流入 | ⏳ 待实现 | 北向资金每日净流入 | 需要北向资金接口 |

### 3.3 融资融券因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `MarginBalanceChange` | 融资余额变化 | ✅ 已实现 | 融资余额日变化 | margin.fin_balance |
| `MarginRatio` | 融资买入占比 | ✅ 已实现 | 融资买入额/成交额 | margin.fin_buy_amount |
| `ShortBalanceChange` | 融券余额变化 | ✅ 已实现 | 融券余额日变化 | margin.sec_balance |
| `ShortRatio` | 融券卖出占比 | ✅ 已实现 | 融券卖出额/成交额 | margin.sec_sell_amount |
| `MarginDebtRatio` | 融资负债率 | ✅ 已实现 | 融资余额/流通市值 | margin.fin_balance |

### 3.4 换手率因子

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `TurnoverRatio_20d` | 平均换手率 | ✅ 已实现 | 20 | mean(turnover, 20) | daily.vol + fina_indicator.shares_a |
| `TurnoverRatio_60d` | 平均换手率 | ✅ 已实现 | 60 | mean(turnover, 60) | daily.vol + fina_indicator.shares_a |
| `TurnoverStd` | 换手率波动 | ✅ 已实现 | 20 | std(turnover, 20) | daily.vol + fina_indicator.shares_a |
| `TurnoverChange` | 换手率变化 | ✅ 已实现 | 5 | delta(turnover_ratio, 5) | daily.vol + fina_indicator.shares_a |

### 3.5 筹码集中度因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `ShareholderConcentration` | 股东集中度 | ⏳ 待实现 | 股东户数变化 | 需要股东户数数据 |
| `FreeFloatRatio` | 自由流通比例 | ✅ 已实现 | 自由流通股占比 | fina_indicator.shares_a / fina_indicator.total_share |

---

## 四、基本面因子

### 4.1 估值因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `PERatio` | 市盈率 | ✅ 已实现 | PE | fina_indicator.pe |
| `PERatio_TTM` | 滚动市盈率 | ✅ 已实现 | PE_TTM | fina_indicator.pe_ttm |
| `PBRatio` | 市净率 | ✅ 已实现 | PB | fina_indicator.pb |
| `PSRatio` | 市销率 | ✅ 已实现 | PS | fina_indicator.ps |
| `PSRatio_TTM` | 滚动市销率 | ✅ 已实现 | PS_TTM | fina_indicator.ps_ttm |
| `EVToEBITDA` | 企业价值倍数 | ⏳ 待实现 | EV/EBITDA | 需要计算 |
| `DividendYield` | 股息率 | ✅ 已实现 | 股息/股价 | dividend.cash_div |
| `ForwardPE` | 前瞻市盈率 | ⏳ 待实现 | 预期EPS/股价 | 需要预测数据 |

### 4.2 盈利能力因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `ROE` | 净资产收益率 | ✅ 已实现 | 净利润/净资产 | fina_indicator.roe |
| `ROE_WAA` | 加权ROE | ✅ 已实现 | 加权平均ROE | fina_indicator.roe_waa |
| `ROE_DT` | 摊薄ROE | ✅ 已实现 | 摊薄净资产收益率 | fina_indicator.roe_dt |
| `ROA` | 总资产收益率 | ✅ 已实现 | 净利润/总资产 | fina_indicator.roa |
| `ROIC` | 投资资本回报率 | ⏳ 待实现 | ROIC | 需要计算 |
| `GrossProfitMargin` | 毛利率 | ✅ 已实现 | 毛利/营收 | fina_indicator.grossprofit_margin |
| `OperatingMargin` | 营业利润率 | ✅ 已实现 | 营业利润/营收 | fina_indicator.netprofit_margin |
| `NetMargin` | 净利润率 | ✅ 已实现 | 净利润/营收 | fina_indicator.netprofit_margin |
| `EPS` | 每股收益 | ✅ 已实现 | 净利润/总股本 | fina_indicator.eps |
| `EPS_Growth` | EPS增长率 | ✅ 已实现 | EPS同比增长 | fina_indicator.basic_eps_yoy |

### 4.3 成长因子

| 因子名称 | 类型 | 状态 | 周期 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `RevenueYoY` | 营收同比 | ✅ 已实现 | 季度/年度 | 营收同比增长率 | fina_indicator.or_yoy |
| `NetProfitYoY` | 净利润同比 | ✅ 已实现 | 季度/年度 | 净利润同比增长率 | fina_indicator.profit_to_gr |
| `OperatingProfitYoY` | 营业利润同比 | ✅ 已实现 | 季度/年度 | 营业利润同比增长率 | 需要计算 |
| `RevenueQoQ` | 营收环比 | ✅ 已实现 | 季度 | 营收环比增长率 | 需要计算 |
| `NetProfitQoQ` | 净利润环比 | ✅ 已实现 | 季度 | 净利润环比增长率 | 需要计算 |
| `EarningsSurprise` | 盈利惊喜 | ⏳ 待实现 | 季度 | 实际EPS与预期EPS差异 | 需要预期数据 |

### 4.4 质量因子（Quality）

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `EarningsQuality` | 盈利质量 | ✅ 已实现 | 经营现金流/净利润 | cashflow.n_cashflow_act / income.n_income |
| `ROE_Stability` | ROE稳定性 | ✅ 已实现 | ROE的标准差 | fina_indicator.roe |
| `CashFlowCoverage` | 现金流覆盖率 | ✅ 已实现 | 经营现金流/流动负债 | cashflow.n_cashflow_act / balancesheet.total_liab |
| `OperatingCashFlow` | 经营现金流 | ✅ 已实现 | 经营活动现金流 | cashflow.n_cashflow_act |
| `FreeCashFlow` | 自由现金流 | ✅ 已实现 | 自由现金流 | cashflow.n_cashflow_act - capital_expenditure |

---

## 五、杠杆因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `DebtToAssets` | 资产负债率 | ✅ 已实现 | 总负债/总资产 | fina_indicator.debt_to_assets |
| `DebtToEquity` | 产权比率 | ✅ 已实现 | 总负债/净资产 | fina_indicator.debt_to_equity |
| `CurrentRatio` | 流动比率 | ✅ 已实现 | 流动资产/流动负债 | fina_indicator.current_ratio |
| `QuickRatio` | 速动比率 | ✅ 已实现 | 速动资产/流动负债 | fina_indicator.quick_ratio |
| `InterestCoverage` | 利息保障倍数 | ⏳ 待实现 | EBIT/利息费用 | 需要计算 |

---

## 六、流动性因子

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `TurnoverRatio` | 换手率 | ✅ 已实现 | 20/60 | 平均换手率 | daily.vol |
| `AmihudIlliquidity` | Amihud非流动性 | ✅ 已实现 | 20 | 非流动性指标 | daily.close, daily.amount |
| `VolumeAmplitude` | 量幅 | ✅ 已实现 | 20 | 成交量波动率 | daily.vol |
| `DailyTradingValue` | 日均成交额 | ✅ 已实现 | 20 | 日均成交额 | daily.amount |
| `LiquidityRatio` | 流动性比率 | ✅ 已实现 | 20 | 成交额/市值 | daily.amount |

---

## 七、行业/风格因子

### 7.1 行业因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `IndustryReturn_5d` | 行业收益 | ✅ 已实现 | 行业5日收益 | daily.close + stock_basic.industry |
| `IndustryMomentum` | 行业动量 | ✅ 已实现 | 行业中期动量 | daily.close + stock_basic.industry |
| `IndustryRank` | 行业内排名 | ✅ 已实现 | 在行业内的因子排名 | daily.close + stock_basic.industry |
| `IndustryBeta` | 行业Beta | ✅ 已实现 | 行业与市场的相关性 | daily.close + index |

### 7.2 风格因子（Barra）

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `Size` | 市值 | ✅ 已实现 | 对数市值 | fina_indicator.total_share * daily.close |
| `Beta` | 市场Beta | ✅ 已实现 | 与市场的相关性 | daily.close + index |
| `MomentumStyle` | 动量风格 | ✅ 已实现 | 中期动量 | daily.close |
| `ResidualVol` | 残差波动 | ✅ 已实现 | 特质波动率 | daily.close + index |
| `LiquidityStyle` | 流动性风格 | ✅ 已实现 | 流动性指标 | daily.amount |
| `ValueStyle` | 价值风格 | ✅ 已实现 | 估值因子 | fina_indicator.pe, fina_indicator.pb |
| `GrowthStyle` | 成长风格 | ✅ 已实现 | 成长因子 | fina_indicator.revenue_yoy |

---

## 八、风险因子

### 8.1 风险事件因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `PledgeRatio` | 股权质押比例 | ✅ 已实现 | 质押股数/总股本 | pledge_stat |
| `ShareFloatRatio` | 限售解禁比例 | ✅ 已实现 | 解禁股数/总股本 | share_float.float_ratio |
| `ST_Flag` | ST标志 | ✅ 已实现 | 是否ST股票 | 需要名称变更数据 |
| `DelistingRisk` | 退市风险 | ⏳ 待实现 | 退市风险评分 | 需要财务数据 |

### 8.2 波动率风险因子

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `HistoricalVolatility` | 历史波动率 | ✅ 已实现 | 60 | 历史收益波动 | daily.close |
| `ImpliedVolatility` | 隐含波动率 | ⏳ 待实现 | - | 期权隐含波动率 | 需要期权数据 |
| `TailRisk` | 尾部风险 | ✅ 已实现 | 60 | VaR/CVaR | daily.close |

---

## 九、另类数据因子

### 9.1 龙虎榜因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `TopListBuyAmount` | 龙虎榜买入额 | ✅ 已实现 | 龙虎榜买入金额 | top_list.buy_amount |
| `TopListBuyRatio` | 龙虎榜买入占比 | ✅ 已实现 | 买入占成交额比例 | top_list.buy_amount / daily.amount |
| `TopListConcentration` | 龙虎榜集中度 | ✅ 已实现 | 席位集中度 | top_list |
| `TopListNetFlow` | 龙虎榜净流入 | ✅ 已实现 | 买入-卖出 | top_list.buy_amount - top_list.sell_amount |

### 9.2 增减持因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `HolderTradeChange` | 股东增减持 | ✅ 已实现 | 增减持数量 | holder_trade.change_vol |
| `HolderTradeRatio` | 增减持比例 | ✅ 已实现 | 增减持占比 | holder_trade.change_ratio |
| `InsiderBuySignal` | 内部人买入信号 | ✅ 已实现 | 高管增持 | holder_trade.holder_type |

### 9.3 分红因子

| 因子名称 | 类型 | 状态 | 描述 | 数据来源 |
|---------|------|------|------|---------|
| `DividendAmount` | 分红金额 | ✅ 已实现 | 每股分红 | dividend.cash_div |
| `DividendFrequency` | 分红频率 | ✅ 已实现 | 分红次数 | dividend |
| `ShareDividend` | 送股比例 | ✅ 已实现 | 送股数量 | dividend.stk_div |
| `BonusShare` | 转增比例 | ✅ 已实现 | 转增数量 | dividend |

---

## 十、Alpha101 / Alpha191 类因子

### 10.1 Alpha101 核心因子

| 因子名称 | 状态 | 描述 | 数据来源 |
|---------|------|------|---------|
| `Alpha001` | ✅ 已实现 | (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5) | daily.close, daily.vol |
| `Alpha002` | ✅ 已实现 | (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)) | daily.open, daily.close, daily.vol |
| `Alpha003` | ✅ 已实现 | (-1 * rank(covariance(rank(open), rank(volume), 5))) | daily.open, daily.vol |
| `Alpha004` | ✅ 已实现 | (rank(ts_rank(volume, 3) * ts_rank((-1 * delta(close, 1)), 3))) | daily.close, daily.vol |
| `Alpha005` | ✅ 已实现 | (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))) | daily.open, daily.close, daily.vol |
| `Alpha006` | ✅ 已实现 | (-1 * rank((sum((open - close), 20) / 20))) | daily.open, daily.close |
| `Alpha007` | ✅ 已实现 | (rank(correlation(high, volume, 5))) | daily.high, daily.vol |
| `Alpha008` | ✅ 已实现 | (rank((sum(((close - open) / open), 20)) * rank((stddev(volume, 20) / mean(volume, 20))))) | daily.open, daily.close, daily.vol |
| `Alpha009` | ✅ 已实现 | ((-1) * rank(rank(delta(volume, 1)) * rank(-1 * delta(close, 1)))) | daily.close, daily.vol |
| `Alpha010` | ✅ 已实现 | ((-1) * Ts_Rank(rank(volume), 3)) | daily.vol |

### 10.2 Alpha191 核心因子

| 因子名称 | 状态 | 描述 | 数据来源 |
|---------|------|------|---------|
| `Alpha191_01` | ✅ 已实现 | 收盘价相对开盘价的变化率 | daily.open, daily.close |
| `Alpha191_02` | ✅ 已实现 | 最高价相对开盘价的变化率 | daily.open, daily.high |
| `Alpha191_03` | ✅ 已实现 | 最低价相对开盘价的变化率 | daily.open, daily.low |
| `Alpha191_04` | ✅ 已实现 | 成交量相对均值的变化率 | daily.vol |
| `Alpha191_05` | ✅ 已实现 | 成交额相对均值的变化率 | daily.amount |
| `Alpha191_06` | ✅ 已实现 | 最高价相对收盘价的变化率 | daily.high, daily.close |
| `Alpha191_07` | ✅ 已实现 | 最低价相对收盘价的变化率 | daily.low, daily.close |
| `Alpha191_08` | ✅ 已实现 | 开盘价相对前收盘价的变化率 | daily.open, daily.pre_close |
| `Alpha191_09` | ✅ 已实现 | 5日平均最高价相对收盘价 | daily.high, daily.close |
| `Alpha191_10` | ✅ 已实现 | 5日平均最低价相对收盘价 | daily.low, daily.close |

---

## 十一、因子统计量

### 11.1 时序统计因子

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `ReturnMean` | 均值 | ✅ 已实现 | 20 | 平均收益 | daily.close |
| `ReturnStd` | 标准差 | ✅ 已实现 | 20 | 收益波动 | daily.close |
| `ReturnSkew` | 偏度 | ✅ 已实现 | 60 | 收益偏度 | daily.close |
| `ReturnKurt` | 峰度 | ✅ 已实现 | 60 | 收益峰度 | daily.close |
| `SharpeRatio` | 夏普比率 | ✅ 已实现 | 60 | 风险调整收益 | daily.close |
| `SortinoRatio` | Sortino比率 | ✅ 已实现 | 60 | 下行风险调整收益 | daily.close |

### 11.2 相关系数因子

| 因子名称 | 类型 | 状态 | 窗口 | 描述 | 数据来源 |
|---------|------|------|------|------|---------|
| `Correlation_CloseVolume` | 量价相关 | ✅ 已实现 | 20 | 收盘价与成交量相关 | daily.close, daily.vol |
| `Correlation_HighLow` | 高低价相关 | ✅ 已实现 | 20 | 最高价与最低价相关 | daily.high, daily.low |
| `AutoCorrelation` | 自相关 | ✅ 已实现 | 5 | 收益自相关 | daily.close |

---

## 十二、因子覆盖统计

### 12.1 按类别统计

| 类别 | 已实现 | 待实现 | 总计 | 覆盖率 |
|------|--------|--------|------|--------|
| 量价因子 | 32 | 1 | 33 | **97%** |
| 趋势因子 | 13 | 0 | 13 | **100%** |
| 资金流因子 | 10 | 2 | 12 | **83%** |
| 基本面因子 | 30 | 4 | 34 | **88%** |
| 杠杆因子 | 4 | 1 | 5 | **80%** |
| 流动性因子 | 5 | 0 | 5 | **100%** |
| 行业风格因子 | 12 | 0 | 12 | **100%** |
| 风险因子 | 6 | 2 | 8 | **75%** |
| 另类数据因子 | 10 | 0 | 10 | **100%** |
| Alpha101/191 | 20 | 0 | 20 | **100%** |
| 统计因子 | 8 | 0 | 8 | **100%** |
| **合计** | **140** | **10** | **150** | **93%** |

### 12.2 优先级建议

| 优先级 | 因子类别 | 理由 |
|--------|----------|------|
| **P0** | 量价因子 | A股最有效，天级别选股核心 |
| **P1** | 趋势因子 | 中频策略增强 |
| **P2** | 基本面因子 | 中长期稳定性 |
| **P3** | 资金流因子 | A股特色因子 |
| **P4** | Alpha101/191 | 进阶增强 |
| **P5** | 宏观/北向资金 | 高阶策略 |

---

## 十三、实盘组合推荐

### 13.1 经典中频策略

```
中期动量(ret_20d) + 短期反转(Reversal_1d) + 换手率(TurnoverRatio_20d) 
+ 行业强度(IndustryMomentum) + 波动率过滤(Volatility_20d)
```

### 13.2 A股游资风格

```
涨停相关 + 放量突破(DonchianBreakout) + 高换手(TurnoverRatio_20d) + 情绪周期
```

### 13.3 机构风格

```
质量(EarningsQuality) + 低波(Volatility_20d) + 北向资金 + 中期趋势(MACD)
```

---

## 十四、因子库结构设计

```
src/quant/data/factors/
├── __init__.py              # 因子导出
├── base.py                  # 基类定义 (Factor, RollingFactor)
├── technical.py             # 技术指标因子（已实现）
├── fundamental.py           # 基本面因子（待完善）
├── momentum.py              # 动量/反转因子（待完善）
├── volatility.py            # 波动率因子（待完善）
├── quality.py               # 质量因子（待完善）
├── alpha101.py              # Alpha101 因子（待实现）
├── alpha191.py              # Alpha191 因子（待实现）
├── factor_engine.py         # 因子计算引擎
└── utils.py                 # 因子工具函数
```

---

## 十五、下一步行动计划

| 阶段 | 任务 | 时间 |
|------|------|------|
| **Phase 1** | 实现缺失的量价因子（VWAP、Donchian突破） | 1天 |
| **Phase 2** | 实现Alpha101/Alpha191核心因子 | 2天 |
| **Phase 3** | 实现因子预处理管道（去极值、标准化、中性化） | 2天 |
| **Phase 4** | 实现因子IC分析与组合构建 | 2天 |
| **Phase 5** | 集成因子库到回测系统 | 1天 |