# B1 模型变量来源审计

更新日期：2026-06-06

## 结论

当前生产版 B1 模型和下一版 Tushare-only 扩展模型都不使用 AkShare 独有字段。

当前 B1 候选信号与三个生产模型的入模变量，可以由 Tushare `daily` 日线行情字段、本地滚动技术指标、Tushare `stock_basic` 股票名称字段生成。下一版扩展模型额外接入 Tushare `daily_basic`，用于真实换手率、量比、估值和市值。

例行任务当前已经覆盖生产 B1 所需字段：

```text
data/raw/daily/*.parquet
```

该目录由 `src/quant/routine/data_refresh.py` 刷新，数据源为 Tushare。

扩展指标目录：

```text
data/raw/daily_basic/*.parquet
```

该目录由 `src/quant/routine/daily_basic_refresh.py` 刷新，数据源为 Tushare `daily_basic`。

## B1 候选信号变量

候选信号使用：

```text
open
high
low
close
pct_chg 或 close.pct_change()
name
```

派生变量：

```text
amplitude = (high - low) / low * 100
BBI = (MA3 + MA6 + MA12 + MA24) / 4
MA60 = close.rolling(60).mean()
BBI-MA60 = BBI - MA60
KDJ J = 本地由 high / low / close 计算
```

这些字段均由 Tushare `daily` 和 `stock_basic` 覆盖。

## 当前生产模型真实选中特征

读取模型：

```text
models/production/b1/up8_es.joblib
models/production/b1/up10.joblib
models/production/b1/down3_es.joblib
```

三个模型各自保留 50 个特征，合并去重后为 67 个真实选中特征：

```text
alpha003
alpha004
alpha005
alpha006
alpha009
alpha191_01
alpha191_02
alpha191_03
alpha191_06
alpha191_07
alpha191_09
alpha191_11
alpha191_12
alpha191_13
alpha191_15
amplitude_1
amplitude_20
atr_14
bb_lower
bb_middle
bb_upper
bias_12
bias_24
bias_6
cci
change
close
downside_volatility_20d
downside_volatility_60d
ema_10
ema_20
ema_5
high
kdj_j
keltner_lower
keltner_upper
keltner_width
low
ma_10
ma_120
ma_20
ma_5
ma_60
bbi
bbi_ma60_diff
bbi_ma60_ratio
mass_index
momentum_20d
momentum_5d
momentum_60d
obv
open
parabolic_sar
pct_chg
pre_close
price_level
price_log
psy_24
return_10d
return_120d
return_1d
return_5d
return_60d
reversal_5d
rsi_12
volume_relative_60d
volatility_20d
volatility_60d
vortex_minus
vortex_plus
```

## 字段来源分类

### Tushare daily 原始字段

```text
open
high
low
close
pre_close
change
pct_chg
volume / vol
amount
```

当前生产模型真实选中特征中，直接使用的 Tushare daily 原始字段是：

```text
open
high
low
close
pre_close
change
pct_chg
```

`volume / vol` 没有作为最终选中特征直接进入模型，但部分派生因子依赖它。

### 本地技术指标派生字段

以下字段由 Tushare 日线行情本地计算：

```text
ma_*
ema_*
rsi_12
kdj_d_k / kdj_d_d / kdj_d_j
kdj_w_k / kdj_w_d / kdj_w_j
kdj_m_k / kdj_m_d / kdj_m_j
macd_dif / macd_dea / macd_hist
bb_*
atr_14
cci
bias_*
psy_24
mass_index
parabolic_sar
vortex_*
keltner_*
amplitude_*
return_*
momentum_*
reversal_5d
volatility_*
downside_volatility_*
price_level
price_log
```

### 依赖成交量的派生字段

以下字段依赖 `volume / vol`：

```text
obv
alpha003
alpha004
alpha009
volume_relative_*
volume_change_*
volume_breakout_*
volume_zscore_20d
volume_price_strength_5d
```

注意：旧代码里的 `turnover_ratio` 不是 Tushare `daily_basic.turnover_rate`。新训练特征已改名为 `volume_relative_60d`，实际公式是：

```python
volume_relative_60d = volume / volume.rolling(60).mean()
```

因此它本质上是“成交量相对 60 日均量”，可以完全由 Tushare `daily.vol` 生成，不依赖 AkShare。

### 不依赖 AkShare 独有字段

当前生产模型没有使用以下 AkShare 常见独有字段：

```text
换手率
振幅
流通市值
总市值
市盈率
市净率
筹码分布
资金流
```

真实换手率、量比、市值、PE/PB 等字段已经优先接入 Tushare `daily_basic`，而不是依赖 AkShare。

## 例行任务覆盖情况

已覆盖：

```text
Tushare daily:
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

本地标准化:
  date
  symbol
  volume
  name
```

已扩展：

```text
Tushare daily_basic:
  turnover_rate
  turnover_rate_f
  volume_ratio
  pe
  pe_ttm
  pb
  ps
  ps_ttm
  total_share
  float_share
  free_share
  total_mv
  circ_mv
  dv_ratio
  dv_ttm

本地派生:
  total_mv_log
  circ_mv_log
  free_share_ratio
  float_share_ratio
```

## Tushare 替代性判断

| 变量类型 | 当前是否用到 | 当前例行是否覆盖 | 是否可用 Tushare 替代 |
|---|---:|---:|---:|
| OHLCV 日线 | 是 | 是 | 是，`daily` |
| 涨跌额/涨跌幅 | 是 | 是 | 是，`daily.change / pct_chg` |
| BBI/MA/KDJ/RSI/BOLL/ATR 等技术指标 | 是 | 是，本地计算 | 是，基于 `daily` 本地计算；也可未来对照 Tushare 技术面因子 |
| OBV/Alpha 成交量因子 | 是 | 是 | 是，基于 `daily.vol` |
| 当前 `volume_relative_60d` | 是 | 是 | 是，当前公式基于 `daily.vol` |
| 真实换手率 `turnover_rate` | 扩展模型会使用 | 是 | 是，`daily_basic` |
| 真实量比 `ts_volume_ratio` | 扩展模型会使用 | 是 | 是，`daily_basic.volume_ratio` |
| 市值/估值字段 | 扩展模型会使用 | 是 | 是，`daily_basic` |
| 筹码/资金流字段 | 否 | 否 | Tushare 有对应特色/资金流接口，但当前 B1 不使用 |

## 后续建议

1. 当前 B1 生产策略可继续使用 Tushare daily + daily_basic，不需要 AkShare 独有字段。
2. 新训练特征已将旧 `turnover_ratio` 重命名为 `volume_relative_60d`，避免误解为真实换手率。
3. daily_basic 扩展字段需要持续观察覆盖率，尤其是 PE、股息率等可能自然缺失的字段。
4. AkShare 不应作为当前 B1 生产特征的必要依赖。
