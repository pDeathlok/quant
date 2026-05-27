# 因子数据下载说明

## 概述

本文档说明如何使用 Tushare Pro 下载因子构建所需的全部数据。

**重要说明**：
- 积分仅作为权限约束，不会被消耗
- 5000积分权限已覆盖所有需要的接口

---

## 数据下载脚本

### 主脚本：`scripts/data/download_factor_data_full.py`

**功能**：下载所有因子构建所需的数据

**使用方法**：

```bash
# 下载所有数据（默认从2010年开始）
python scripts/data/download_factor_data_full.py

# 修改时间范围（在脚本中修改 start_date 参数）
python scripts/data/download_factor_data_full.py
```

---

## 数据覆盖范围

### 1. 股票基本信息

| 数据类型 | 接口 | 字段 | 用途 |
|---------|------|------|------|
| 股票列表 | `stock_basic` | ts_code, symbol, name, area, industry, list_date, market | 股票池构建、行业分类 |

### 2. 日线行情数据（量价因子）

| 数据类型 | 接口 | 字段 | 用途 |
|---------|------|------|------|
| 日线行情 | `daily` | open, high, low, close, pre_close, change, pct_chg, vol, amount | 量价因子、趋势因子、波动率因子、动量/反转因子 |

**覆盖因子**：
- 动量因子：ret_5d, ret_10d, ret_20d, ret_60d
- 反转因子：Reversal_1d, Reversal_3d, Reversal_5d
- 均线偏离：MA20_Distance, MA60_Distance
- 成交量放量：VolumeRatio, VolumeUp, VolumeDown
- 波动率：Volatility_20d, ATR, DownsideVolatility
- 振幅：Amplitude, Amplitude_20d
- 趋势：MACD, ADX, TrendSlope, DonchianBreakout

### 3. 财务指标数据（基本面因子）

| 数据类型 | 接口 | 字段 | 用途 |
|---------|------|------|------|
| 财务指标 | `fina_indicator` | eps, roe, roa, pe, pb, ps, debt_to_assets, current_ratio, revenue_yoy 等 | 估值、盈利、成长、质量、杠杆、流动性因子 |

**覆盖因子**：
- 估值：PE, PB, PS, EV/EBITDA
- 盈利：ROE, ROA, ROIC, EPS, GrossMargin, OperatingMargin, NetMargin
- 成长：RevenueYoY, NetProfitYoY, RevenueQoQ, NetProfitQoQ
- 质量：EarningsQuality, ROE_Stability
- 杠杆：DebtToAssets, DebtToEquity, CurrentRatio, QuickRatio
- 流动性：TurnoverRatio, AmihudIlliquidity

### 4. 三大报表数据

| 数据类型 | 接口 | 字段 | 用途 |
|---------|------|------|------|
| 利润表 | `income` | revenue, operate_profit, total_profit, n_income | 盈利能力、成长因子 |
| 资产负债表 | `balancesheet` | total_assets, total_liab, money_cap, inventories | 杠杆、流动性因子 |
| 现金流量表 | `cashflow` | n_cashflow_act, c_pay_acq_const_fiolta, net_profit | 质量因子 |

### 5. 资金流数据（资金流因子）

| 数据类型 | 接口 | 字段 | 用途 |
|---------|------|------|------|
| 融资融券 | `margin` | fin_buy_amount, fin_balance, sec_sell_amount | 融资融券因子 |
| 龙虎榜 | `top_list` | buy_amount, sell_amount, net_amount, reason | 龙虎榜因子、市场情绪 |

**覆盖因子**：
- 资金流：MarginBalanceChange, MarginRatio, ShortBalanceChange
- 另类数据：TopListBuyAmount, TopListBuyRatio, TopListConcentration

### 6. 股东数据（股东因子）

| 数据类型 | 接口 | 字段 | 用途 |
|---------|------|------|------|
| 增减持 | `stk_holdertrade` | holder_name, holder_type, change_vol, change_ratio | 股东行为因子 |

**覆盖因子**：
- 股东：ShareholderConcentration, FreeFloatRatio

### 7. 风险数据（风险因子）

| 数据类型 | 接口 | 字段 | 用途 |
|---------|------|------|------|
| 限售解禁 | `share_float` | float_date, float_share, float_ratio | 风险预警因子 |

### 8. 分红数据（分红因子）

| 数据类型 | 接口 | 字段 | 用途 |
|---------|------|------|------|
| 分红送股 | `dividend` | div_proc, stk_div, cash_div, record_date, ex_date | 分红能力因子 |

### 9. 市场基准数据（行业/风格因子）

| 数据类型 | 接口 | 字段 | 用途 |
|---------|------|------|------|
| 指数日线 | `index_daily` | ts_code, trade_date, open, high, low, close, vol, amount | 市场基准、行业因子 |

**覆盖因子**：
- 行业：IndustryReturn_5d, IndustryMomentum, IndustryRank
- 风格：Beta, MomentumStyle, ResidualVol

---

## 数据目录结构

```
data/
├── raw/                           # 原始数据
│   ├── stock_basic.parquet        # 股票基本信息
│   ├── fina_indicator.parquet     # 财务指标
│   ├── income.parquet             # 利润表
│   ├── balancesheet.parquet       # 资产负债表
│   ├── cashflow.parquet           # 现金流量表
│   ├── margin.parquet             # 融资融券
│   ├── top_list.parquet           # 龙虎榜
│   ├── holder_trade.parquet       # 增减持
│   ├── share_float.parquet        # 限售解禁
│   ├── dividend.parquet           # 分红送股
│   ├── index_000300.SH.parquet    # 沪深300指数
│   ├── index_000001.SH.parquet    # 上证指数
│   └── daily/                     # 日线行情（按股票代码存储）
│       ├── 000001.parquet
│       ├── 000002.parquet
│       └── ...
└── processed/                     # 处理后数据（待生成）
```

---

## 数据下载进度

### 当前状态

| 数据类型 | 状态 | 说明 |
|---------|------|------|
| 股票基本信息 | ✅ 完成 | 5,524 只股票 |
| 日线行情 | 🔄 进行中 | 正在下载全市场数据 |
| 财务指标 | ⏳ 待下载 | 需要逐只股票下载 |
| 利润表 | ⏳ 待下载 | 需要逐只股票下载 |
| 资产负债表 | ⏳ 待下载 | 需要逐只股票下载 |
| 现金流量表 | ⏳ 待下载 | 需要逐只股票下载 |
| 融资融券 | ⏳ 待下载 | 市场级数据 |
| 龙虎榜 | ⏳ 待下载 | 市场级数据 |
| 增减持 | ⏳ 待下载 | 市场级数据 |
| 限售解禁 | ⏳ 待下载 | 市场级数据 |
| 分红送股 | ⏳ 待下载 | 市场级数据 |
| 指数日线 | ⏳ 待下载 | 沪深300、上证指数 |

---

## 数据使用示例

### 读取日线数据

```python
import pandas as pd
from pathlib import Path

# 读取单只股票日线数据
df = pd.read_parquet("data/raw/daily/000001.parquet")

# 读取财务指标
fina = pd.read_parquet("data/raw/fina_indicator.parquet")

# 读取股票列表
stocks = pd.read_parquet("data/raw/stock_basic.parquet")
```

### 批量读取日线数据

```python
from pathlib import Path
import pandas as pd

daily_dir = Path("data/raw/daily")
all_daily = []

for file in daily_dir.glob("*.parquet"):
    df = pd.read_parquet(file)
    all_daily.append(df)

combined = pd.concat(all_daily, ignore_index=True)
```

---

## 因子数据映射

### 量价因子（26个）

| 因子 | 数据来源 | 字段 |
|------|---------|------|
| ret_5d, ret_10d, ret_20d, ret_60d | 日线行情 | close |
| Reversal_1d, Reversal_3d, Reversal_5d | 日线行情 | close |
| MA20_Distance, MA60_Distance | 日线行情 | close |
| VolumeRatio, VolumeUp, VolumeDown | 日线行情 | volume, close |
| Volatility_20d, ATR, DownsideVolatility | 日线行情 | high, low, close |
| MACD, ADX, TrendSlope | 日线行情 | close, high, low |

### 基本面因子（28个）

| 因子 | 数据来源 | 字段 |
|------|---------|------|
| PE, PB, PS, EV/EBITDA | 财务指标 | pe, pb, ps |
| ROE, ROA, ROIC, EPS | 财务指标 | roe, roa, eps |
| RevenueYoY, NetProfitYoY | 财务指标 | revenue_yoy |
| EarningsQuality | 财务指标+现金流量表 | n_cashflow_act, n_income |
| DebtToAssets, CurrentRatio | 财务指标 | debt_to_assets, current_ratio |

### 资金流因子（8个）

| 因子 | 数据来源 | 字段 |
|------|---------|------|
| MarginBalanceChange | 融资融券 | fin_balance |
| TopListBuyAmount | 龙虎榜 | buy_amount |

### 股东因子（7个）

| 因子 | 数据来源 | 字段 |
|------|---------|------|
| ShareholderConcentration | 增减持 | change_vol, change_ratio |

### 风险因子（7个）

| 因子 | 数据来源 | 字段 |
|------|---------|------|
| PledgeRatio | 限售解禁 | float_ratio |

### 分红因子（7个）

| 因子 | 数据来源 | 字段 |
|------|---------|------|
| DividendYield | 分红送股 | cash_div |

### 行业风格因子（9个）

| 因子 | 数据来源 | 字段 |
|------|---------|------|
| IndustryReturn_5d | 指数日线 | close |
| Beta | 日线行情+指数日线 | close |

---

## 注意事项

1. **数据更新**：建议定期更新数据，保持最新
2. **数据清洗**：使用前需进行数据清洗（去重、去缺失值等）
3. **复权处理**：日线数据建议使用前复权数据
4. **停牌处理**：注意处理停牌期间的数据
5. **ST股票**：根据策略需求决定是否过滤ST股票

---

## 下一步

数据下载完成后，可以进行：

1. **数据预处理**：去极值、标准化、中性化
2. **因子计算**：基于原始数据计算因子值
3. **因子分析**：IC分析、因子相关性分析
4. **因子组合**：构建多因子模型
5. **回测验证**：验证因子有效性