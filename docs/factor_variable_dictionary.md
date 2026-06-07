# 项目变量库 132 个已实现变量字典

更新日期：2026-06-06

## 1. 说明

本文档记录当前项目变量库中已经实现并可被策略复用的 132 个变量。

当前 B1 只是第一个使用者，后续其他策略也应优先复用本文档和 `src/quant/features/variable_library.py` 中的变量定义。

数据源口径：

```text
Tushare daily:
  open, high, low, close, pre_close, change, pct_chg, vol, amount

Tushare daily_basic:
  turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm,
  dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv

本地标准化:
  vol -> volume
  daily_basic.volume_ratio -> ts_volume_ratio
```

公式记号：

```text
O = open
H = high
L = low
C = close
PC = pre_close
V = volume
TP = (H + L + C) / 3
ret_n = C / C.shift(n) - 1
rank(x) = 当前序列百分位排名
ts_rank(x, n) = n 日窗口内最后一个值的时序百分位排名
rolling_mean(x, n) = n 日滚动均值
rolling_std(x, n) = n 日滚动标准差
```

## 2. Tushare daily 原始行情字段

| 变量 | 中文含义 | 来源 | 计算逻辑 |
|---|---|---|---|
| `open` | 开盘价 | Tushare daily | 原始字段 `open` |
| `high` | 最高价 | Tushare daily | 原始字段 `high` |
| `low` | 最低价 | Tushare daily | 原始字段 `low` |
| `close` | 收盘价 | Tushare daily | 原始字段 `close` |
| `pre_close` | 昨收价 | Tushare daily | 原始字段 `pre_close` |
| `change` | 涨跌额 | Tushare daily | 原始字段 `change = close - pre_close` |
| `pct_chg` | 涨跌幅，单位百分比 | Tushare daily | 原始字段 `pct_chg` |

## 3. 均线与趋势类技术指标

| 变量 | 中文含义 | 来源 | 计算逻辑 |
|---|---|---|---|
| `ma_5` | 5 日简单移动均线 | 本地派生 | `rolling_mean(C, 5)` |
| `ma_10` | 10 日简单移动均线 | 本地派生 | `rolling_mean(C, 10)` |
| `ma_20` | 20 日简单移动均线 | 本地派生 | `rolling_mean(C, 20)` |
| `ma_60` | 60 日简单移动均线 | 本地派生 | `rolling_mean(C, 60)` |
| `ma_120` | 120 日简单移动均线 | 本地派生 | `rolling_mean(C, 120)` |
| `ema_5` | 5 日指数移动均线 | 本地派生 | `ewm_mean(C, span=5, adjust=True)` |
| `ema_10` | 10 日指数移动均线 | 本地派生 | `ewm_mean(C, span=10, adjust=True)` |
| `ema_20` | 20 日指数移动均线 | 本地派生 | `ewm_mean(C, span=20, adjust=True)` |
| `bbi` | 多空指标 BBI，综合 3/6/12/24 日均线观察短中期趋势 | 本地派生 | `(rolling_mean(C,3)+rolling_mean(C,6)+rolling_mean(C,12)+rolling_mean(C,24))/4` |
| `bbi_ma60_diff` | BBI 与 60 日均线的价差，用于衡量 BBI 相对中期均线的位置 | 本地派生 | `bbi-ma_60` |
| `bbi_ma60_ratio` | BBI 相对 60 日均线的比例，用于跨价格水平比较趋势强弱 | 本地派生 | `bbi/ma_60` |
| `kdj_d_k` | 日线 KDJ 的 K 值 | 本地派生 | 先用 `pre_close / 前一日 close` 将 OHLC 连续化，再计算 `RSV=(C-lowest(L,9))/(highest(H,9)-lowest(L,9))*100; K=ewm(RSV, alpha=1/3)` |
| `kdj_d_d` | 日线 KDJ 的 D 值 | 本地派生 | 基于连续 OHLC 的 K 值计算 `D=ewm(K, alpha=1/3)` |
| `kdj_d_j` | 日线 KDJ 的 J 值，衡量日线超买超卖 | 本地派生 | 基于连续 OHLC 计算 `J=3*K-2*D` |
| `kdj_w_k` | 周线 KDJ 的 K 值 | 本地派生 | 先将日线 OHLC 连续化，再按 `W-FRI` 聚合周 K 线：`O=first, H=max, L=min, C=last, V=sum`，再按 KDJ 公式计算，并向后对齐到每日 |
| `kdj_w_d` | 周线 KDJ 的 D 值 | 本地派生 | 周线 KDJ 的 `D=ewm(K, alpha=1/3)`，向后对齐到每日 |
| `kdj_w_j` | 周线 KDJ 的 J 值，衡量中短周期超买超卖 | 本地派生 | 周线 KDJ 的 `J=3*K-2*D`，向后对齐到每日 |
| `kdj_m_k` | 月线 KDJ 的 K 值 | 本地派生 | 先将日线 OHLC 连续化，再按月末聚合月 K 线：`O=first, H=max, L=min, C=last, V=sum`，再按 KDJ 公式计算，并向后对齐到每日 |
| `kdj_m_d` | 月线 KDJ 的 D 值 | 本地派生 | 月线 KDJ 的 `D=ewm(K, alpha=1/3)`，向后对齐到每日 |
| `kdj_m_j` | 月线 KDJ 的 J 值，衡量更大周期超买超卖 | 本地派生 | 月线 KDJ 的 `J=3*K-2*D`，向后对齐到每日 |
| `parabolic_sar` | 抛物线转向指标 | 本地派生 | 初始 `SAR=L[0]`，按趋势方向、极值点 EP、加速因子 AF=0.02 到 0.2 逐日递推 |
| `vortex_plus` | 漩涡指标正向线 | 本地派生 | `TR=max(H-L, abs(H-C.shift(1)), abs(L-C.shift(1))); +VM=abs(H-L.shift(1)); vortex_plus=sum(+VM,14)/sum(TR,14)` |
| `vortex_minus` | 漩涡指标负向线 | 本地派生 | `-VM=abs(L-H.shift(1)); vortex_minus=sum(-VM,14)/sum(TR,14)` |
| `mass_index` | 质量指数，观察高低价区间扩张 | 本地派生 | `range=H-L; ema1=ewm(range,9); ema2=ewm(ema1,9); mass_index=sum(ema1/ema2,25)` |

## 3.1 MACD 指标

| 变量 | 中文含义 | 来源 | 计算逻辑 |
|---|---|---|---|
| `macd_dif` | MACD 快慢线差值，也常称 DIF | 本地派生 | `EMA12=ewm_mean(C, span=12); EMA26=ewm_mean(C, span=26); macd_dif=EMA12-EMA26` |
| `macd_dea` | MACD 信号线，也常称 DEA | 本地派生 | `macd_dea=ewm_mean(macd_dif, span=9)` |
| `macd_hist` | MACD 柱状值 | 本地派生 | `macd_hist=macd_dif-macd_dea` |

## 4. 摆动、通道与偏离指标

| 变量 | 中文含义 | 来源 | 计算逻辑 |
|---|---|---|---|
| `rsi_12` | 12 日相对强弱指数 | 本地派生 | `delta=C.diff(); gain=max(delta,0); loss=max(-delta,0); RS=rolling_mean(gain,12)/rolling_mean(loss,12); RSI=100-100/(1+RS)` |
| `bb_upper` | 布林带上轨 | 本地派生 | `rolling_mean(C,20) + 2*rolling_std(C,20)` |
| `bb_middle` | 布林带中轨 | 本地派生 | `rolling_mean(C,20)` |
| `bb_lower` | 布林带下轨 | 本地派生 | `rolling_mean(C,20) - 2*rolling_std(C,20)` |
| `atr_14` | 14 日平均真实波幅 | 本地派生 | `TR=max(H-L, abs(H-C.shift(1)), abs(L-C.shift(1))); atr_14=rolling_mean(TR,14)` |
| `cci` | 20 日顺势指标 | 本地派生 | `TP=(H+L+C)/3; CCI=(TP-rolling_mean(TP,20))/(0.015*rolling_mad(TP,20))` |
| `bias_6` | 6 日乖离率 | 本地派生 | `(C-rolling_mean(C,6))/rolling_mean(C,6)*100` |
| `bias_12` | 12 日乖离率 | 本地派生 | `(C-rolling_mean(C,12))/rolling_mean(C,12)*100` |
| `bias_24` | 24 日乖离率 | 本地派生 | `(C-rolling_mean(C,24))/rolling_mean(C,24)*100` |
| `psy_24` | 24 日心理线，上涨天数占比 | 本地派生 | `rolling_sum(C>C.shift(1),24)/24*100` |
| `keltner_upper` | 肯特纳通道上轨 | 本地派生 | `EMA(TP,20) + 2*EMA(TR,20)` |
| `keltner_lower` | 肯特纳通道下轨 | 本地派生 | `EMA(TP,20) - 2*EMA(TR,20)` |
| `keltner_width` | 肯特纳通道宽度 | 本地派生 | `(keltner_upper-keltner_lower)/EMA(TP,20)` |

## 5. 收益、动量、反转与波动率

| 变量 | 中文含义 | 来源 | 计算逻辑 |
|---|---|---|---|
| `return_1d` | 1 日收益率 | 本地派生 | `C.pct_change(1)` |
| `return_5d` | 5 日收益率 | 本地派生 | `C.pct_change(5)` |
| `return_10d` | 10 日收益率 | 本地派生 | `C.pct_change(10)` |
| `return_60d` | 60 日收益率 | 本地派生 | `C.pct_change(60)` |
| `return_120d` | 120 日收益率 | 本地派生 | `C.pct_change(120)` |
| `momentum_5d` | 跳过最近 5 天后的 5 日动量 | 本地派生 | `C.shift(5)/C.shift(10)-1` |
| `momentum_20d` | 跳过最近 5 天后的 20 日动量 | 本地派生 | `C.shift(5)/C.shift(25)-1` |
| `momentum_60d` | 跳过最近 5 天后的 60 日动量 | 本地派生 | `C.shift(5)/C.shift(65)-1` |
| `reversal_5d` | 5 日反转因子 | 本地派生 | `-C.pct_change(5)` |
| `volatility_20d` | 20 日年化波动率 | 本地派生 | `rolling_std(C.pct_change(),20)*sqrt(252)` |
| `volatility_60d` | 60 日年化波动率 | 本地派生 | `rolling_std(C.pct_change(),60)*sqrt(252)` |
| `downside_volatility_20d` | 20 日年化下行波动率 | 本地派生 | `rolling_std(where(C.pct_change()<0, ret, 0),20)*sqrt(252)` |
| `downside_volatility_60d` | 60 日年化下行波动率 | 本地派生 | `rolling_std(where(C.pct_change()<0, ret, 0),60)*sqrt(252)` |
| `amplitude_1` | 当日振幅 | 本地派生 | `(H-L)/C.shift(1)*100` |
| `amplitude_20` | 20 日平均振幅 | 本地派生 | `rolling_mean((H-L)/C.shift(1)*100,20)` |
| `price_level` | 当前收盘价水平 | 本地派生 | `C` |
| `price_log` | 收盘价对数水平 | 本地派生 | `log(C+1)` |

## 6. 成交量与量价因子

| 变量 | 中文含义 | 来源 | 计算逻辑 |
|---|---|---|---|
| `obv` | 能量潮指标，按涨跌方向累加成交量 | 本地派生 | 若 `C>C.shift(1)`，`OBV=OBV.shift(1)+V`；若 `C<C.shift(1)`，`OBV=OBV.shift(1)-V`；否则不变 |
| `volume_ma5` | 5 日成交量均线 | 本地派生 | `rolling_mean(V,5)` |
| `volume_ma10` | 10 日成交量均线 | 本地派生 | `rolling_mean(V,10)` |
| `volume_ma20` | 20 日成交量均线 | 本地派生 | `rolling_mean(V,20)` |
| `volume_ma60` | 60 日成交量均线 | 本地派生 | `rolling_mean(V,60)` |
| `volume_ema5` | 5 日成交量指数均线 | 本地派生 | `ewm_mean(V, span=5)` |
| `volume_ema10` | 10 日成交量指数均线 | 本地派生 | `ewm_mean(V, span=10)` |
| `volume_ema20` | 20 日成交量指数均线 | 本地派生 | `ewm_mean(V, span=20)` |
| `volume_relative_5d` | 成交量相对 5 日均量 | 本地派生 | `V/rolling_mean(V,5)` |
| `volume_relative_20d` | 成交量相对 20 日均量 | 本地派生 | `V/rolling_mean(V,20)` |
| `volume_relative_60d` | 成交量相对 60 日均量，替代旧 `turnover_ratio` 命名 | 本地派生 | `V/rolling_mean(V,60)` |
| `volume_change_1d` | 成交量 1 日变化率 | 本地派生 | `V.pct_change(1)` |
| `volume_change_3d` | 成交量 3 日变化率 | 本地派生 | `V.pct_change(3)` |
| `volume_change_5d` | 成交量 5 日变化率 | 本地派生 | `V.pct_change(5)` |
| `volume_zscore_20d` | 20 日成交量标准分，衡量异常放量 | 本地派生 | `(V-rolling_mean(V,20))/rolling_std(V,20)` |
| `volume_breakout_20d` | 成交量相对过去 20 日最高量，>1 表示创 20 日量能新高 | 本地派生 | `V/rolling_max(V.shift(1),20)` |
| `volume_breakout_60d` | 成交量相对过去 60 日最高量，>1 表示创 60 日量能新高 | 本地派生 | `V/rolling_max(V.shift(1),60)` |
| `volume_price_strength_5d` | 5 日价格变化叠加成交量强度 | 本地派生 | `C.pct_change(5)*volume_relative_20d` |

## 7. Alpha101 量价因子

| 变量 | 中文含义 | 来源 | 计算逻辑 |
|---|---|---|---|
| `alpha003` | 开盘价排名与成交量排名的 5 日协方差，取负后再排名 | 本地派生 | `-rank(rolling_cov(rank(O), rank(V), 5))` |
| `alpha004` | 成交量时序排名与负价格变化时序排名的乘积排名 | 本地派生 | `rank(ts_rank(V,3) * ts_rank(-C.diff(1),3))` |
| `alpha005` | 开盘价相对 10 日 VWAP 均值偏离，并惩罚收盘价偏离 VWAP | 本地派生 | `rank(O-rolling_mean(VWAP,10)) * -abs(rank(C-VWAP))`，其中 `VWAP=(H+L+C)/3` |
| `alpha006` | 20 日平均开盘价减收盘价的反向排名 | 本地派生 | `-rank(rolling_mean(O-C,20))` |
| `alpha009` | 成交量变化与负价格变化的排名乘积，整体取负 | 本地派生 | `-rank(rank(V.diff(1))*rank(-C.diff(1)))` |

## 8. Alpha191 A 股量价因子

| 变量 | 中文含义 | 来源 | 计算逻辑 |
|---|---|---|---|
| `alpha191_01` | 收盘价相对开盘价涨跌幅 | 本地派生 | `(C-O)/O` |
| `alpha191_02` | 最高价相对开盘价涨幅 | 本地派生 | `(H-O)/O` |
| `alpha191_03` | 最低价相对开盘价跌幅 | 本地派生 | `(L-O)/O` |
| `alpha191_06` | 最高价相对收盘价偏离 | 本地派生 | `(H-C)/C` |
| `alpha191_07` | 最低价相对收盘价偏离 | 本地派生 | `(L-C)/C` |
| `alpha191_09` | 5 日均最高价相对收盘价 | 本地派生 | `rolling_mean(H,5)/C` |
| `alpha191_11` | 5 日均收盘价相对开盘价 | 本地派生 | `rolling_mean(C,5)/O` |
| `alpha191_12` | 当日高低价振幅相对收盘价 | 本地派生 | `(H-L)/C` |
| `alpha191_13` | 5 日平均振幅 | 本地派生 | `rolling_mean((H-L)/C,5)` |
| `alpha191_15` | 收盘价相对 5 日均线 | 本地派生 | `C/rolling_mean(C,5)` |

## 9. Tushare daily_basic 原始字段

| 变量 | 中文含义 | 来源 | 计算逻辑 |
|---|---|---|---|
| `turnover_rate` | 换手率，单位百分比 | Tushare daily_basic | 原始字段 `turnover_rate` |
| `turnover_rate_f` | 自由流通换手率，单位百分比 | Tushare daily_basic | 原始字段 `turnover_rate_f` |
| `ts_volume_ratio` | Tushare 量比 | Tushare daily_basic | 原始字段 `volume_ratio`，训练中重命名为 `ts_volume_ratio` |
| `pe` | 市盈率，总市值/净利润 | Tushare daily_basic | 原始字段 `pe` |
| `pe_ttm` | 滚动市盈率 TTM | Tushare daily_basic | 原始字段 `pe_ttm` |
| `pb` | 市净率 | Tushare daily_basic | 原始字段 `pb` |
| `ps` | 市销率 | Tushare daily_basic | 原始字段 `ps` |
| `ps_ttm` | 滚动市销率 TTM | Tushare daily_basic | 原始字段 `ps_ttm` |
| `dv_ratio` | 股息率，单位百分比 | Tushare daily_basic | 原始字段 `dv_ratio` |
| `dv_ttm` | 滚动股息率 TTM，单位百分比 | Tushare daily_basic | 原始字段 `dv_ttm` |
| `total_share` | 总股本，单位万股 | Tushare daily_basic | 原始字段 `total_share` |
| `float_share` | 流通股本，单位万股 | Tushare daily_basic | 原始字段 `float_share` |
| `free_share` | 自由流通股本，单位万股 | Tushare daily_basic | 原始字段 `free_share` |
| `total_mv` | 总市值，单位万元 | Tushare daily_basic | 原始字段 `total_mv` |
| `circ_mv` | 流通市值，单位万元 | Tushare daily_basic | 原始字段 `circ_mv` |

## 10. daily_basic 派生字段

| 变量 | 中文含义 | 来源 | 计算逻辑 |
|---|---|---|---|
| `total_mv_log` | 总市值对数 | daily_basic 派生 | `log(total_mv)`，0 值先转为缺失 |
| `circ_mv_log` | 流通市值对数 | daily_basic 派生 | `log(circ_mv)`，0 值先转为缺失 |
| `free_share_ratio` | 自由流通股本占总股本比例 | daily_basic 派生 | `free_share/total_share` |
| `float_share_ratio` | 流通股本占总股本比例 | daily_basic 派生 | `float_share/total_share` |
| `float_mv_ratio` | 流通市值占总市值比例 | daily_basic 派生 | `circ_mv/total_mv` |
| `free_float_share_ratio` | 自由流通股本占流通股本比例 | daily_basic 派生 | `free_share/float_share` |
| `turnover_rate_ma5` | 5 日平均真实换手率 | daily_basic 派生 | `rolling_mean(turnover_rate,5)`，最少 3 个有效值 |
| `turnover_rate_ma20` | 20 日平均真实换手率 | daily_basic 派生 | `rolling_mean(turnover_rate,20)`，最少 10 个有效值 |
| `turnover_rate_rel20` | 当前换手率相对 20 日均值 | daily_basic 派生 | `turnover_rate/turnover_rate_ma20` |
| `turnover_rate_f_ma5` | 5 日平均自由流通换手率 | daily_basic 派生 | `rolling_mean(turnover_rate_f,5)`，最少 3 个有效值 |
| `turnover_rate_f_ma20` | 20 日平均自由流通换手率 | daily_basic 派生 | `rolling_mean(turnover_rate_f,20)`，最少 10 个有效值 |
| `turnover_rate_f_rel20` | 当前自由流通换手率相对 20 日均值 | daily_basic 派生 | `turnover_rate_f/turnover_rate_f_ma20` |
| `ts_volume_ratio_ma5` | 5 日平均 Tushare 量比 | daily_basic 派生 | `rolling_mean(ts_volume_ratio,5)`，最少 3 个有效值 |
| `ts_volume_ratio_ma20` | 20 日平均 Tushare 量比 | daily_basic 派生 | `rolling_mean(ts_volume_ratio,20)`，最少 10 个有效值 |
| `ts_volume_ratio_rel20` | 当前 Tushare 量比相对 20 日均值 | daily_basic 派生 | `ts_volume_ratio/ts_volume_ratio_ma20` |
| `total_mv_change_20d` | 总市值 20 日变化率 | daily_basic 派生 | `total_mv.pct_change(20)` |
| `circ_mv_change_20d` | 流通市值 20 日变化率 | daily_basic 派生 | `circ_mv.pct_change(20)` |
| `pe_ttm_inv` | 滚动市盈率倒数，近似盈利收益率 | daily_basic 派生 | `1/pe_ttm`，0 值先转为缺失 |
| `pb_inv` | 市净率倒数，近似账面价值收益率 | daily_basic 派生 | `1/pb`，0 值先转为缺失 |
| `ps_ttm_inv` | 滚动市销率倒数，近似销售收益率 | daily_basic 派生 | `1/ps_ttm`，0 值先转为缺失 |

## 11. 训练前建议确认点

### 11.1 建议保留

以下变量建议保留：

```text
Tushare daily 原始行情字段
技术指标
收益/动量/反转/波动率
Alpha101 / Alpha191
daily_basic 原始字段
daily_basic 派生字段
```

原因是当前模型为树模型，且训练管线已加入中位数填充，对不同尺度和部分缺失的容忍度较高。

### 11.2 建议后续修正但不阻塞本轮确认

旧字段 `turnover_ratio` 已从新训练特征清单中移除，改为更准确的 `volume_relative_60d`。

`momentum_5d / momentum_20d / momentum_60d` 已修正为真正不同周期。

### 11.3 需要观察缺失率

以下字段可能天然缺失较多，训练后需要看缺失率和特征重要性：

```text
pe
pe_ttm
dv_ratio
dv_ttm
pe_ttm_inv
```

原因是亏损公司或未分红公司在估值/股息字段上可能为空或异常。
