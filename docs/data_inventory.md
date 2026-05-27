# 数据目录说明

> 本文档说明 `data/` 下所有数据文件的来源、字段含义、覆盖范围。

## 总览

```
data/
├── cache/                    233 files  | AKShare 日线缓存
├── factors_raw/               15 files  | AKShare 因子辅助数据
├── raw/                       12 files + daily/ (5,534 files) | Tushare 原始数据
├── results/                    3 files  | 回测结果 / 因子测试
├── stocks/                   218 files  | AKShare 日线 (缓存副本)
├── stocks_daily/           4,598 files  | AKShare 日线 (主数据集)
└── stocks_daily_test/          0 files  | 空目录
```

**数据来源分两类**:
- **AKShare**: `cache/`, `stocks/`, `stocks_daily/`, `factors_raw/` — 免费、无 Token
- **Tushare**: `raw/` — 需要 Token，积分制权限

---

## 1. stocks_daily/ — 日线行情（主数据集）

| 属性 | 说明 |
|------|------|
| **来源** | AKShare (`stock_zh_a_daily`) |
| **股票数** | 4,598 只（沪深主板 + 创业板） |
| **时间范围** | 2010-01-04 ~ 2026-05-26 |
| **复权** | 前复权 (qfq) |
| **总记录** | 12,317,401 条 |
| **文件大小** | 516.8 MB |
| **命名** | `{代码}.parquet`，如 `000001.parquet` |

### 字段

| 字段 | 类型 | 含义 | 示例 |
|------|------|------|------|
| `date` | str | 交易日期 | `2010-01-04` |
| `open` | float | 开盘价 | `6.55` |
| `high` | float | 最高价 | `6.56` |
| `low` | float | 最低价 | `6.32` |
| `close` | float | 收盘价 | `6.33` |
| `volume` | float | 成交量（股） | `24,192,276` |
| `turnover` | float | 成交额（元） | `580,249,472` |
| `outstanding_share` | float | 流通股本（股） | `2,924,114,263` |
| `turnover_rate` | float | 换手率（%） | `0.83%` |
| `symbol` | str | 股票代码（纯6位） | `000001` |
| `name` | str | 股票名称 | `平安银行` |

### 示例

```python
import pandas as pd
df = pd.read_parquet("data/stocks_daily/000001.parquet")
#      date  open  high   low  close     volume     turnover  ...
# 0  2010-01-04  6.55  6.56  6.32   6.33 24192276.0 580249472  ...
```

---

## 2. factors_raw/ — 因子辅助数据

| 属性 | 说明 |
|------|------|
| **来源** | AKShare |
| **文件数** | 15 个 parquet |
| **总大小** | 20.6 MB |
| **总记录** | 215,904 条 |

### 2.1 财务数据

#### `financial_yjbb_multi.parquet`

| 属性 | 说明 |
|------|------|
| **来源** | AKShare `stock_yjbb_em` |
| **记录** | 87,225 行 × 17 列 |
| **覆盖** | 11,694 只股票，10 个报告期 |
| **报告期** | 2023Q3 ~ 2025Q4 |
| **大小** | 7.1 MB |

| 字段 | 含义 |
|------|------|
| `股票代码` | 6位代码 |
| `股票简称` | 名称 |
| `每股收益` | EPS |
| `营业总收入-营业总收入` | 营收总额 |
| `营业总收入-同比增长` | 营收同比增速（%） |
| `营业总收入-季度环比增长` | 营收环比增速（%） |
| `净利润-净利润` | 净利润额 |
| `净利润-同比增长` | 净利润同比增速（%） |
| `净利润-季度环比增长` | 净利润环比增速（%） |
| `每股净资产` | BVPS |
| `净资产收益率` | ROE（%） |
| `每股经营现金流量` | 每股经营现金流 |
| `销售毛利率` | 毛利率（%） |
| `所处行业` | 行业分类 |
| `最新公告日期` | 公告日期 |
| `report_date` | 报告期（如 20241231） |

#### `balance_sheet_zcfz_multi.parquet`

| 属性 | 说明 |
|------|------|
| **来源** | AKShare `stock_zcfz_em` |
| **记录** | 51,665 行 × 16 列 |
| **覆盖** | 5,215 只股票，10 个报告期 |
| **报告期** | 2023Q3 ~ 2025Q4 |
| **大小** | 5.3 MB |

| 字段 | 含义 |
|------|------|
| `股票代码` | 6位代码 |
| `股票简称` | 名称 |
| `资产-货币资金` | 现金及等价物 |
| `资产-应收账款` | 应收账款 |
| `资产-存货` | 存货 |
| `资产-总资产` | 总资产 |
| `资产-总资产同比` | 总资产同比增速（%） |
| `负债-应付账款` | 应付账款 |
| `负债-预收账款` | 预收账款 |
| `负债-总负债` | 总负债 |
| `负债-总负债同比` | 总负债同比增速（%） |
| `资产负债率` | 资产负债率（%） |
| `股东权益合计` | 股东权益 |
| `公告日期` | 公告日期 |
| `report_date` | 报告期 |

---

### 2.2 宏观经济数据

| 文件 | 来源 | 记录 | 关键字段 |
|------|------|------|---------|
| `macro_cpi.parquet` | `macro_china_cpi` | 220 行 | `全国-当月`, `全国-同比增长`, `全国-环比增长`, `城市/农村` 细分 |
| `macro_ppi.parquet` | `macro_china_ppi` | 244 行 | PPI 工业品出厂价格指数 |
| `macro_gdp.parquet` | `macro_china_gdp` | 81 行 | GDP 累计值及增速 |
| `macro_pmi.parquet` | `macro_china_pmi_yearly` | 250 行 | 制造业 PMI |
| `macro_money_supply.parquet` | `macro_china_money_supply` | 220 行 | M0/M1/M2 货币供应量 |
| `macro_shrzgm.parquet` | `macro_china_shrzgm` | 132 行 | 社融增量、人民币贷款、企业债券、股票融资 |
| `macro_lpr.parquet` | `macro_china_lpr` | 1,572 行 | LPR 1年/5年期利率 |
| `macro_cnbs.parquet` | `macro_cnbs` | 80 行 | 宏观杠杆率 |
| `macro_fx_reserves.parquet` | `macro_china_fx_reserves_yearly` | 132 行 | 外汇储备 |

### 2.3 利率/国债

| 文件 | 来源 | 记录 | 关键字段 |
|------|------|------|---------|
| `bond_yield_curve.parquet` | `bond_china_yield` | 738 行 | 各期限国债收益率曲线（3月/6月/1年/3年/5年/10年等） |
| `bond_zh_us_rate.parquet` | `bond_zh_us_rate` | 9,264 行 | 中美债券收益率对比 |

### 2.4 资金流向

| 文件 | 来源 | 记录 | 关键字段 |
|------|------|------|---------|
| `hsgt_fund_flow_summary.parquet` | `stock_hsgt_fund_flow_summary_em` | 4 行 | 沪深港通资金流向摘要（沪股通/深股通净流入） |

### 2.5 龙虎榜

| 文件 | 来源 | 记录 | 关键字段 |
|------|------|------|---------|
| `lhb_detail_multi_year.parquet` | `stock_lhb_detail_em` | 64,077 行 × 21 列 | 上榜日、代码、收盘价、涨跌幅、净买额、买入额、卖出额、成交额、换手率、流通市值、上榜原因、上榜后1/2/5/10日涨跌幅 |

---

## 3. cache/ — AKShare 数据缓存

| 属性 | 说明 |
|------|------|
| **来源** | AKShare (`stock_zh_a_daily`) |
| **文件数** | 233 个 parquet |
| **命名** | `{sh/sz}{代码}_{开始日期}_{结束日期}_qfq.parquet` |

### 字段

与 `stocks_daily/` 基本一致，多一个 `pct_change` 字段：

| 字段 | 含义 |
|------|------|
| `date`, `open`, `high`, `low`, `close`, `volume`, `amount` | 基本行情 |
| `outstanding_share` | 流通股本 |
| `turnover` | 成交额（换手率计算中间值） |
| `symbol` | 代码（带 sh/sz 前缀） |
| `pct_change` | 涨跌幅（%） |

### 特殊文件

| 文件 | 说明 |
|------|------|
| `tushare_stock_basic_all.parquet` | 股票基本信息（Tushare 源，5,524 只） |
| `tushare_index_*.parquet` | 指数数据（Tushare 源） |
| `tushare_finance_*.parquet` | 财务数据（Tushare 源） |

---

## 4. stocks/ — AKShare 日线缓存副本

| 属性 | 说明 |
|------|------|
| **来源** | AKShare (从 cache 复制) |
| **文件数** | 218 个 parquet |
| **命名** | `{sh/sz}{代码}.parquet` 或 `{sh/sz}{代码}_日期范围_qfq.parquet` |
| **字段** | 与 `cache/` 一致（11 列） |

> **注意**: 此目录与 `cache/` 存在数据重叠，可视为缓存的精简版本（去掉了文件名中的日期范围）。

---

## 5. raw/ — Tushare 原始数据

| 属性 | 说明 |
|------|------|
| **来源** | Tushare Pro |
| **文件数** | 12 个 parquet + daily/ 下 5,534 个 |
| **总大小** | ~70 MB |

### 5.1 日线行情（`raw/daily/`）

| 属性 | 说明 |
|------|------|
| **文件数** | 5,534 个（覆盖全市场 A 股，含北交所） |
| **命名** | `{代码}.{SH/SZ/BJ}.parquet` |
| **大小** | 单个 50KB ~ 250KB |

| 字段 | 类型 | 含义 |
|------|------|------|
| `ts_code` | str | Tushare 代码格式（如 `000001.SZ`） |
| `trade_date` | str | 交易日期（YYYYMMDD） |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `pre_close` | float | 昨收价 |
| `change` | float | 涨跌额 |
| `pct_chg` | float | 涨跌幅（%） |
| `vol` | float | 成交量（手） |
| `amount` | float | 成交额（千元） |

### 5.2 财务及参考数据（`raw/` 根目录）

| 文件 | 来源 | 记录 | 字段 |
|------|------|------|------|
| `stock_basic.parquet` | Tushare `stock_basic` | 5,524 行 | `ts_code, symbol, name, area, industry, list_date, market` |
| `fina_indicator.parquet` | Tushare `fina_indicator` | 192,951 行 | ROE/ROA/PE/PB/PS/EPS/毛利率/资产负债率/流动比率等 50+ 财务指标 |
| `income.parquet` | Tushare `income` | 153,189 行 | `revenue, operate_profit, total_profit, n_income, eps` 等 |
| `balancesheet.parquet` | Tushare `balancesheet` | 184,029 行 | `total_assets, total_liab, money_cap, inventories, goodwill` 等 |
| `cashflow.parquet` | Tushare `cashflow` | 236,771 行 | 经营活动/投资/筹资现金流各项 |
| `dividend.parquet` | Tushare `dividend` | 2,000 行 | `div_cash, stk_div, ex_date` 等 |
| `holder_trade.parquet` | Tushare `stk_holdertrade` | 3,000 行 | 增减持 `holder_name, change_vol, change_ratio` |
| `share_float.parquet` | Tushare `share_float` | 6,000 行 | 解禁 `float_date, float_share, float_ratio` |
| `margin.parquet` | Tushare `margin` | 4,000 行 | 融资融券 |
| `pledge_stat.parquet` | Tushare `pledge_stat` | 0 行（空文件） | 质押统计 |
| `index_000001.SH.parquet` | Tushare `index_daily` | 3,979 行 | 上证综指日线 |
| `index_000300.SH.parquet` | Tushare `index_daily` | 3,979 行 | 沪深300日线 |

---

## 6. results/ — 策略回测与因子测试结果

| 文件 | 说明 | 记录 | 关键字段 |
|------|------|------|---------|
| `b1_strategy_signals.parquet` | B1 策略信号 | 115 行 | OHLCV + `amplitude, bbi, ma60, K, D, J, prev_volume, signal(bool)` |
| `factors_test_result.parquet` | 因子测试1 | 118 行 | OHLCV + `MA5, MA20, EMA12, EMA26, MACD, Signal, Histogram, RSI, BollingerBands, ATR` |
| `factors_test_result_all.parquet` | 因子测试2 | 118 行 | OHLCV + `KDJ, WilliamsR, BIAS6, Momentum, PSY, VR, OBV, CCI, ADX` |

---

## 7. 数据对比

### 日线行情 — 三套数据源对比

| 维度 | `stocks_daily/` (AKShare) | `raw/daily/` (Tushare) | `cache/` (AKShare) |
|------|--------------------------|------------------------|-------------------|
| **来源** | AKShare | Tushare | AKShare |
| **股票覆盖** | 4,598 只 | 5,534 只（含北交所） | 233 只（子集） |
| **代码格式** | 纯6位 | `{代码}.{SH/SZ/BJ}` | `sh/sz{代码}` |
| **日期格式** | `YYYY-MM-DD` | `YYYYMMDD` | `YYYY-MM-DD` |
| **成交量单位** | 股 | 手（100股） | 股 |
| **成交额单位** | 元 | 千元 | 元 |
| **额外字段** | `name, turnover_rate, outstanding_share` | `pre_close, change, pct_chg` | `pct_change, outstanding_share` |
| **复权** | 前复权 | 前复权 | 前复权 |

### 财务数据 — AKShare vs Tushare

| 维度 | AKShare (`factors_raw/`) | Tushare (`raw/`) |
|------|-------------------------|-------------------|
| **业绩报表** | `yjbb` 10期全市场（87k 行，16 字段） | `fina_indicator` 50+ 字段（193k 行） |
| **资产负债表** | `zcfz` 10期全市场（52k 行，15 字段） | `balancesheet` 详细科目（184k 行） |
| **利润表** | 无 | `income` 详细科目（153k 行） |
| **现金流量表** | 无 | `cashflow` 详细科目（237k 行） |
| **优势** | 批量查询、字段精炼 | 字段更全、历史更深 |

---

## 8. 数据更新频率

| 数据 | 频率 | 说明 |
|------|------|------|
| 日线行情 | 每个交易日收盘后 | AKShare 当日可获取 |
| 财务报表 | 每季度 | 年报(3月)、一季报(4月)、中报(7月)、三季报(10月) |
| 宏观经济 | 月度 | 统计局/央行发布日 |
| 国债收益率 | 每日 | 中国债券信息网 |
| 龙虎榜 | 每个交易日 | 收盘后更新 |
| 北向资金 | 每个交易日 | 盘中实时、收盘汇总 |

---

## 9. 使用建议

### 日线行情 — 推荐使用 `stocks_daily/`

- 4,598 只股票全覆盖，字段包含名称和换手率
- 无 Tushare 积分消耗
- 代码格式统一（纯6位）

```python
import pandas as pd
df = pd.read_parquet("data/stocks_daily/000001.parquet")
df["return"] = df["close"].pct_change()
```

### 财务因子 — 推荐使用 AKShare (`factors_raw/`)

- 批量获取，无积分限制
- 字段已精炼（覆盖所有常用因子所需）

```python
import pandas as pd
yjbb = pd.read_parquet("data/factors_raw/financial_yjbb_multi.parquet")
# 计算 PE 因子
latest = yjbb[yjbb["report_date"] == "20241231"]
```

### 如需更详细财务科目 — 使用 Tushare (`raw/`)

- 利润表、资产负债表、现金流量表都有详细科目
- 注意 Tushare 积分限制

### 宏观择时因子

```python
import pandas as pd
cpi = pd.read_parquet("data/factors_raw/macro_cpi.parquet")
lpr = pd.read_parquet("data/factors_raw/macro_lpr.parquet")
m2 = pd.read_parquet("data/factors_raw/macro_money_supply.parquet")
```
