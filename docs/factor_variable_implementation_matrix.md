# 项目变量体系与实现状态

更新日期：2026-06-06

## 1. 文档目的

本文档是项目级变量总表，用于后续 review、扩展和清理。

阅读顺序建议：

```text
1. 本文档：看完整变量体系、实现状态、缺口原因。
2. docs/factor_variable_dictionary.md：看项目变量库当前 132 个已实现变量的逐字段公式。
3. docs/factor_tushare_factor_system.md：看更长期的 Tushare-only 因子路线图。
```

状态定义：

| 状态 | 含义 |
|---|---|
| 已实现-项目变量库 | 已进入 `src/quant/features/variable_library.py`，可被多个策略复用 |
| 已实现-辅助 | 代码已能计算，但暂未进入项目变量库 |
| 待实现-有数据 | Tushare 或本地数据已能支持，但尚未加工成变量 |
| 待实现-缺数据 | 需要新增 Tushare 接口或外部数据 |
| 暂缓 | 暂不建议进入当前项目变量库，避免噪声或工程复杂度过高 |

## 2. 当前项目变量库已实现变量

当前项目变量库变量数量：

```text
132
```

训练前校验结果：

```text
小样本构建：132/132 个项目变量均可产出
daily_basic 合并命中率：100.00%
当前没有启动正式训练
```

### 2.1 Tushare daily 原始行情

| 变量 | 中文含义 | 状态 | 实现逻辑 |
|---|---|---|---|
| `open` | 开盘价 | 已实现-项目变量库 | Tushare daily 原始字段 |
| `high` | 最高价 | 已实现-项目变量库 | Tushare daily 原始字段 |
| `low` | 最低价 | 已实现-项目变量库 | Tushare daily 原始字段 |
| `close` | 收盘价 | 已实现-项目变量库 | Tushare daily 原始字段 |
| `pre_close` | 昨收价 | 已实现-项目变量库 | Tushare daily 原始字段 |
| `change` | 涨跌额 | 已实现-项目变量库 | Tushare daily 原始字段 |
| `pct_chg` | 涨跌幅 | 已实现-项目变量库 | Tushare daily 原始字段 |

### 2.2 均线、趋势、摆动指标

| 变量 | 中文含义 | 状态 | 实现逻辑 |
|---|---|---|---|
| `ma_5`, `ma_10`, `ma_20`, `ma_60`, `ma_120` | 不同周期收盘价简单均线 | 已实现-项目变量库 | `rolling_mean(close,n)` |
| `ema_5`, `ema_10`, `ema_20` | 不同周期收盘价指数均线 | 已实现-项目变量库 | `ewm_mean(close,span=n)` |
| `bbi` | 多空指标 BBI | 已实现-项目变量库 | `(MA3+MA6+MA12+MA24)/4` |
| `bbi_ma60_diff`, `bbi_ma60_ratio` | BBI 相对 MA60 的价差和比例 | 已实现-项目变量库 | `bbi-ma_60`、`bbi/ma_60` |
| `kdj_d_k`, `kdj_d_d`, `kdj_d_j` | 日线 KDJ 的 K/D/J | 已实现-项目变量库 | 先用 `pre_close / 前一日 close` 连续化 OHLC，再计算 RSV、K、D、J |
| `kdj_w_k`, `kdj_w_d`, `kdj_w_j` | 周线 KDJ 的 K/D/J | 已实现-项目变量库 | 连续化 OHLC 后聚合为周线计算 KDJ，再对齐回日频 |
| `kdj_m_k`, `kdj_m_d`, `kdj_m_j` | 月线 KDJ 的 K/D/J | 已实现-项目变量库 | 连续化 OHLC 后聚合为月线计算 KDJ，再对齐回日频 |
| `macd_dif`, `macd_dea`, `macd_hist` | MACD 快慢线、信号线、柱状值 | 已实现-项目变量库 | `EMA12-EMA26`、DIF 的 EMA9、DIF-DEA |
| `rsi_12` | 12 日 RSI | 已实现-项目变量库 | 12 日平均上涨 / 平均下跌计算 |
| `bb_upper`, `bb_middle`, `bb_lower` | 布林带上中下轨 | 已实现-项目变量库 | 20 日均线 ± 2 倍标准差 |
| `atr_14` | 14 日平均真实波幅 | 已实现-项目变量库 | 14 日 TR 均值 |
| `cci` | 20 日顺势指标 | 已实现-项目变量库 | 典型价格偏离均值 / 平均绝对偏差 |
| `bias_6`, `bias_12`, `bias_24` | 乖离率 | 已实现-项目变量库 | `(close-ma_n)/ma_n*100` |
| `psy_24` | 24 日心理线 | 已实现-项目变量库 | 24 日上涨天数占比 |
| `mass_index` | 质量指数 | 已实现-项目变量库 | 高低价区间的双 EMA 比值求和 |
| `parabolic_sar` | 抛物线转向指标 | 已实现-项目变量库 | SAR 趋势递推 |
| `vortex_plus`, `vortex_minus` | 漩涡指标正负线 | 已实现-项目变量库 | VM / TR 的 14 日滚动比值 |
| `keltner_upper`, `keltner_lower`, `keltner_width` | 肯特纳通道 | 已实现-项目变量库 | 典型价格 EMA ± ATR EMA，及通道宽度 |

### 2.3 收益、动量、反转、波动

| 变量 | 中文含义 | 状态 | 实现逻辑 |
|---|---|---|---|
| `return_1d`, `return_5d`, `return_10d`, `return_60d`, `return_120d` | 不同周期收益率 | 已实现-项目变量库 | `close.pct_change(n)` |
| `momentum_5d`, `momentum_20d`, `momentum_60d` | 跳过最近 5 天的动量 | 已实现-项目变量库 | `close.shift(5)/close.shift(n+5)-1` |
| `reversal_5d` | 5 日反转 | 已实现-项目变量库 | `-close.pct_change(5)` |
| `volatility_20d`, `volatility_60d` | 年化波动率 | 已实现-项目变量库 | 收益率滚动标准差 * `sqrt(252)` |
| `downside_volatility_20d`, `downside_volatility_60d` | 年化下行波动率 | 已实现-项目变量库 | 仅保留负收益后计算滚动标准差 |
| `amplitude_1`, `amplitude_20` | 当日振幅 / 20 日平均振幅 | 已实现-项目变量库 | `(high-low)/pre_close*100` 及其滚动均值 |
| `price_level`, `price_log` | 股价水平 / 股价对数 | 已实现-项目变量库 | `close` / `log(close+1)` |

### 2.4 成交量与爆量结构

| 变量 | 中文含义 | 状态 | 实现逻辑 |
|---|---|---|---|
| `obv` | 能量潮 | 已实现-项目变量库 | 按涨跌方向累加或扣减成交量 |
| `volume_ma5`, `volume_ma10`, `volume_ma20`, `volume_ma60` | 成交量均线 | 已实现-项目变量库 | `rolling_mean(volume,n)` |
| `volume_ema5`, `volume_ema10`, `volume_ema20` | 成交量指数均线 | 已实现-项目变量库 | `ewm_mean(volume,span=n)` |
| `volume_relative_5d`, `volume_relative_20d`, `volume_relative_60d` | 成交量相对均量 | 已实现-项目变量库 | `volume/rolling_mean(volume,n)` |
| `volume_change_1d`, `volume_change_3d`, `volume_change_5d` | 成交量短期变化率 | 已实现-项目变量库 | `volume.pct_change(n)` |
| `volume_zscore_20d` | 成交量 20 日标准分 | 已实现-项目变量库 | `(volume-mean20)/std20` |
| `volume_breakout_20d`, `volume_breakout_60d` | 是否爆量及爆量强度 | 已实现-项目变量库 | `volume/rolling_max(volume.shift(1),n)`，大于 1 表示突破过去 n 日最高量 |
| `volume_price_strength_5d` | 量价强度 | 已实现-项目变量库 | `close.pct_change(5)*volume_relative_20d` |

说明：

```text
旧字段 turnover_ratio 已不再作为新模型变量。
新字段 volume_relative_60d 更准确地表达“成交量相对 60 日均量”。
真实换手率使用 Tushare daily_basic.turnover_rate。
```

### 2.5 Alpha101 / Alpha191

| 变量 | 中文含义 | 状态 | 实现逻辑 |
|---|---|---|---|
| `alpha003`, `alpha004`, `alpha005`, `alpha006`, `alpha009` | Alpha101 量价子集 | 已实现-项目变量库 | 基于开盘价、收盘价、成交量、近似 VWAP、排名、协方差等计算 |
| `alpha191_01`, `alpha191_02`, `alpha191_03` | 开高低收相对开盘价变化 | 已实现-项目变量库 | `(close/open-1)`、`(high/open-1)`、`(low/open-1)` |
| `alpha191_06`, `alpha191_07` | 高低价相对收盘价偏离 | 已实现-项目变量库 | `(high-close)/close`、`(low-close)/close` |
| `alpha191_09`, `alpha191_11`, `alpha191_12`, `alpha191_13`, `alpha191_15` | A 股量价结构因子 | 已实现-项目变量库 | 5 日均高价/收盘价、5 日均收盘价/开盘价、振幅、均振幅、收盘价/5 日均线 |

### 2.6 Tushare daily_basic 与派生

| 变量 | 中文含义 | 状态 | 实现逻辑 |
|---|---|---|---|
| `turnover_rate`, `turnover_rate_f` | 真实换手率、自由流通换手率 | 已实现-项目变量库 | Tushare daily_basic 原始字段 |
| `ts_volume_ratio` | Tushare 量比 | 已实现-项目变量库 | Tushare `volume_ratio` 重命名 |
| `pe`, `pe_ttm`, `pb`, `ps`, `ps_ttm` | 估值指标 | 已实现-项目变量库 | Tushare daily_basic 原始字段 |
| `dv_ratio`, `dv_ttm` | 股息率、滚动股息率 | 已实现-项目变量库 | Tushare daily_basic 原始字段 |
| `total_share`, `float_share`, `free_share` | 总股本、流通股本、自由流通股本 | 已实现-项目变量库 | Tushare daily_basic 原始字段 |
| `total_mv`, `circ_mv` | 总市值、流通市值 | 已实现-项目变量库 | Tushare daily_basic 原始字段 |
| `total_mv_log`, `circ_mv_log` | 市值对数 | 已实现-项目变量库 | `log(total_mv)`、`log(circ_mv)` |
| `free_share_ratio`, `float_share_ratio`, `float_mv_ratio`, `free_float_share_ratio` | 股本/市值结构比例 | 已实现-项目变量库 | 自由流通股本、流通股本、市值之间的比例 |
| `turnover_rate_ma5`, `turnover_rate_ma20`, `turnover_rate_rel20` | 换手率均值和相对强度 | 已实现-项目变量库 | 5/20 日均值，当前值 / 20 日均值 |
| `turnover_rate_f_ma5`, `turnover_rate_f_ma20`, `turnover_rate_f_rel20` | 自由流通换手率均值和相对强度 | 已实现-项目变量库 | 5/20 日均值，当前值 / 20 日均值 |
| `ts_volume_ratio_ma5`, `ts_volume_ratio_ma20`, `ts_volume_ratio_rel20` | Tushare 量比均值和相对强度 | 已实现-项目变量库 | 5/20 日均值，当前值 / 20 日均值 |
| `total_mv_change_20d`, `circ_mv_change_20d` | 市值 20 日变化 | 已实现-项目变量库 | `pct_change(20)` |
| `pe_ttm_inv`, `pb_inv`, `ps_ttm_inv` | 估值倒数 | 已实现-项目变量库 | `1/pe_ttm`、`1/pb`、`1/ps_ttm` |

## 3. 缺失率查验

查验范围：

```text
data/raw/daily_basic/*.parquet
交易日数量：585
总行数：3,155,903
股票数量：5,612
```

核心字段缺失率：

| 字段 | 缺失数 | 缺失率 | 观察 |
|---|---:|---:|---|
| `pe` | 707,297 | 22.4119% | 缺失较高，通常与亏损或不适用估值有关 |
| `pe_ttm` | 815,354 | 25.8358% | 缺失较高，通常与亏损或不适用估值有关 |
| `pb` | 19,417 | 0.6153% | 缺失很低 |
| `ps` | 1,110 | 0.0352% | 缺失很低 |
| `ps_ttm` | 2,278 | 0.0722% | 缺失很低 |
| `dv_ratio` | 319,288 | 10.1172% | 股息率缺失或为 0 都正常 |
| `dv_ttm` | 966,497 | 30.6251% | 缺失较高，和分红覆盖有关 |

结论：

```text
PE / PE_TTM 理论上对盈利公司应存在，但 Tushare daily_basic 中实际存在较高缺失。
这不是合并失败导致的：小样本 daily_basic 合并命中率为 100%。
更可能是亏损、停牌、特殊证券、北交所/新股、财务口径不适用等原因导致 Tushare 原始字段为空。
训练侧已加入 SimpleImputer(strategy="median")，可以处理缺失；但训练后需要单独观察这些变量的重要性。
```

## 4. 待实现变量体系

### 4.1 量价与技术增强

| 变量/变量组 | 中文含义 | 状态 | 实现逻辑 | 未实现原因 |
|---|---|---|---|---|
| `return_3d`, `return_20d` | 3 日、20 日收益率 | 待实现-有数据 | `close.pct_change(3/20)` | 当前变量库已有 1/5/10/60/120，暂未补入 |
| `reversal_1d`, `reversal_3d`, `reversal_20d` | 多周期反转 | 待实现-有数据 | `-close.pct_change(n)` | 当前仅入模 5 日反转 |
| `rsi_6`, `rsi_14`, `rsi_24` | 多周期 RSI | 已实现-辅助 | `RSI(n)` | 旧加工脚本能算，但项目变量库只保留 `rsi_12` |
| `close_ma20_ratio`, `close_ma60_ratio` | 收盘价相对均线位置 | 待实现-有数据 | `close/ma_n` | 可由已实现字段组合，尚未单独落列 |
| `ma_bull`, `ma_divergence` | 均线多头排列、均线乖离 | 待实现-有数据 | `ma5>ma10>ma20>ma60`，`(ma5-ma20)/ma20` | 尚未单独落列 |
| `macd_cross`, `macd_hist_change` | MACD 金叉/柱体变化 | 待实现-有数据 | `dif>dea and dif.shift(1)<=dea.shift(1)`，`macd_hist.diff()` | 当前先加入 MACD 基础三列 |
| `gap_open`, `upper_shadow`, `lower_shadow`, `body_ratio` | K 线形态 | 待实现-有数据 | 开盘跳空、上下影线、实体占比 | 文档规划中，尚未进入 B1 |

### 4.2 流动性、成交额、冲击成本

| 变量/变量组 | 中文含义 | 状态 | 实现逻辑 | 未实现原因 |
|---|---|---|---|---|
| `amount_ma20`, `amount_relative_20d` | 成交额均值和相对强度 | 待实现-有数据 | `rolling_mean(amount,20)`，`amount/amount_ma20` | 当前先补成交量结构，成交额尚未补 |
| `amihud_20d` | Amihud 非流动性 | 待实现-有数据 | `rolling_mean(abs(ret)/amount,20)` | 需要确认 amount 单位和极值处理 |
| `zero_volume_days_20d` | 近 20 日无成交天数 | 待实现-有数据 | `rolling_sum(volume<=0,20)` | A 股日线中通常很少，优先级低 |

### 4.3 资金流、筹码和融资融券

| 变量/变量组 | 中文含义 | 状态 | 实现逻辑 | 未实现原因 |
|---|---|---|---|---|
| `moneyflow_*` | 主力/小单/大单资金流 | 待实现-缺数据 | Tushare moneyflow 接口 | 例行任务尚未接入 |
| `chip_*` | 筹码集中度、获利比例 | 待实现-缺数据 | Tushare 每日筹码接口 | 例行任务尚未接入，需评估积分和稳定性 |
| `margin_*` | 融资融券余额和变化 | 待实现-缺数据 | Tushare margin detail | 例行任务尚未接入，仅部分股票适用 |

### 4.4 行业、指数、市场阶段

| 变量/变量组 | 中文含义 | 状态 | 实现逻辑 | 未实现原因 |
|---|---|---|---|---|
| `industry_return_5d/20d` | 行业短中期强弱 | 待实现-缺数据 | 申万行业分类 + 行业指数收益 | 需要接入行业分类和指数日线 |
| `industry_rank` | 行业内横截面排名 | 待实现-缺数据 | 按行业对收益、量能、估值排名 | 需要稳定行业映射 |
| `market_regime_*` | 市场阶段过滤 | 待实现-缺数据 | 沪深 300 / 中证 1000 / 全 A 指数趋势、波动、宽度 | 例行任务尚未接入指数日线 |
| `index_beta`, `residual_return` | 指数 Beta 和残差收益 | 待实现-缺数据 | 个股收益对指数收益滚动回归 | 需要指数数据 |

### 4.5 财务基本面

| 变量/变量组 | 中文含义 | 状态 | 实现逻辑 | 未实现原因 |
|---|---|---|---|---|
| `roe`, `roa`, `gross_margin`, `netprofit_margin` | 盈利能力 | 待实现-缺数据 | Tushare fina_indicator | 尚未接入财务指标例行任务 |
| `revenue_yoy`, `profit_yoy` | 成长性 | 待实现-缺数据 | Tushare fina_indicator / income | 尚未接入，且需处理公告日防未来函数 |
| `debt_to_asset`, `current_ratio` | 偿债能力 | 待实现-缺数据 | Tushare balancesheet / fina_indicator | 尚未接入，需按披露日对齐 |
| `operating_cashflow_quality` | 现金流质量 | 待实现-缺数据 | Tushare cashflow | 尚未接入，需按披露日对齐 |

### 4.6 情绪与交易约束

| 变量/变量组 | 中文含义 | 状态 | 实现逻辑 | 未实现原因 |
|---|---|---|---|---|
| `limit_up_down_*` | 涨跌停状态、连板、炸板 | 待实现-缺数据 | Tushare limit list / stk_limit | 尚未接入，且需要处理交易可成交性 |
| `northbound_holding_*` | 北向持股变化 | 待实现-缺数据 | Tushare 沪深股通持股明细 | 例行任务尚未接入 |
| `st_status`, `suspend_status` | ST、停牌、特殊状态 | 已实现-辅助 | 名称排除 ST/退市；停牌通过日线缺失体现 | 尚未作为单独入模变量 |

## 5. 当前建议

当前项目变量库建议：

```text
1. 各策略优先复用当前 132 个项目变量。
2. 保留 PE / PE_TTM，但训练报告中单独观察缺失率和特征重要性。
3. 不再使用旧 turnover_ratio 字段，改用 volume_relative_60d。
4. 暂不接入 moneyflow / chips / industry / financials，避免在同一轮同时引入太多数据接口变量。
5. 下一轮优先补行业和市场阶段变量，因为它们更可能改善回撤，而不是只改善单笔收益。
```
