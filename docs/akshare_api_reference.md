# AKShare 接口文档（量化研究版）

> 数据源：东方财富、新浪、同花顺、雪球、交易所、集思录等公开数据源
> 适用场景：量化交易 / 金融工程 / 风控 / 宏观研究

---

## 目录

1. [安装与依赖](#1-安装与依赖)
2. [数据分类总览](#2-数据分类总览)
3. [A股数据](#3-a股数据)
4. [可转债](#4-可转债)
5. [指数数据](#5-指数数据)
6. [ETF / 基金](#6-etf--基金)
7. [期货期权](#7-期货期权)
8. [宏观经济](#8-宏观经济)
9. [利率 / 国债](#9-利率--国债)
10. [北向资金](#10-北向资金)
11. [行业板块 / 概念](#11-行业板块--概念)
12. [龙虎榜](#12-龙虎榜)
13. [新闻舆情](#13-新闻舆情)
14. [数据源稳定性](#14-数据源稳定性)
15. [常见问题与最佳实践](#15-常见问题与最佳实践)
16. [推荐架构](#16-推荐架构)
17. [接口速查总表](#17-接口速查总表)

---

## 1. 安装与依赖

```bash
pip install akshare --upgrade
```

推荐搭配：

```bash
pip install pandas pyarrow polars akshare
```

---

## 2. 数据分类总览

| 分类 | 模块前缀 | 说明 |
|------|----------|------|
| A股 | `stock_` | 行情、财务、板块、龙虎榜 |
| 指数 | `index_` | 宽基、行业、概念指数 |
| 可转债 | `bond_` | 转债行情、价值分析、强赎 |
| 基金 ETF | `fund_` | ETF、公募基金 |
| 期货 | `futures_` | 主力连续、实时行情 |
| 期权 | `option_` | ETF期权、个股期权 |
| 宏观 | `macro_` | GDP/CPI/PPI/PMI/M2/社融 |
| 利率 | `bond_china_` / `bond_zh_` | 国债收益率、中美利差 |
| 北向资金 | `stock_hsgt_` | 沪深港通、外资持股 |
| 行业板块 | `stock_board_` | 行业/概念板块及成分股 |
| 外汇 | `fx_` / `currency_` | 汇率数据 |
| 新闻 | `news_` | 财经新闻、个股新闻 |

---

## 3. A股数据

### 3.1 股票列表

```python
import akshare as ak

# 全A股列表（纯6位代码 + 名称）
df = ak.stock_info_a_code_name()
```

| code   | name   |
|--------|--------|
| 000001 | 平安银行 |

### 3.2 全市场实时行情（最核心）

```python
df = ak.stock_zh_a_spot_em()
```

**数据源**：东方财富
**包含字段**：最新价、涨跌幅、涨跌额、成交量、成交额、振幅、换手率、市盈率、市净率、总市值、流通市值

**适用**：
- 盘中实时监控
- 截面因子计算
- 选股筛选

**稳定性**：高（东方财富源）

### 3.3 创业板 / 科创板快照

```python
# 创业板
df = ak.stock_cy_a_spot_em()

# 科创板
df = ak.stock_kc_a_spot_em()
```

### 3.4 ST 股票

```python
df = ak.stock_zh_a_st_em()
```

**适用**：风控过滤、黑名单体系

### 3.5 股票列表（按交易所）

```python
# 沪市
df_sh = ak.stock_info_sh_name_code()

# 深市
df_sz = ak.stock_info_sz_name_code()
```

### 3.6 历史 K 线

#### 日线（推荐，字段最全）

```python
df = ak.stock_zh_a_hist(
    symbol="000001",          # 纯6位代码
    period="daily",
    start_date="20240101",
    end_date="20241231",
    adjust="qfq"              # qfq=前复权, hfq=后复权, ""=不复权
)
```

**返回字段（12个）**：

| 字段 | 含义 | 因子用途 |
|------|------|---------|
| 日期 | 交易日期 | 时间索引 |
| 股票代码 | 6位代码 | 股票标识 |
| 开盘 | 开盘价 | OHLC |
| 收盘 | 收盘价 | OHLC、收益率 |
| 最高 | 最高价 | OHLC、波动率 |
| 最低 | 最低价 | OHLC、波动率 |
| 成交量 | 成交股数 | 流动性因子 |
| 成交额 | 成交金额 | 流动性因子 |
| 振幅 | 日内振幅% | 波动率因子 |
| 涨跌幅 | 日涨跌幅% | 动量因子 |
| 涨跌额 | 涨跌金额 | 动量因子 |
| 换手率 | 日换手率% | 流动性因子 |

#### 备选接口

```python
df = ak.stock_zh_a_daily(
    symbol="sz000001",        # 需带 sh/sz 前缀
    start_date="20240101",
    end_date="20241231",
    adjust="qfq"
)
```

**返回字段（9个）**：date, open, high, low, close, volume, amount, outstanding_share, turnover
**特点**：多出 `outstanding_share`（流通股本），但少了振幅、涨跌幅、换手率
**稳定性**：有时不稳定

### 3.7 分钟级别行情

```python
df = ak.stock_zh_a_hist_min_em(
    symbol="000001",
    period="5"                 # 1/5/15/30/60 分钟
)
```

### 3.8 财务数据（三大报表）

#### 业绩报表（批量推荐）

```python
df = ak.stock_yjbb_em(date="20241231")
```

**返回字段**：
| 字段 | 含义 | 因子用途 |
|------|------|---------|
| 股票代码 | 6位代码 | 标识 |
| 每股收益 | EPS | PE 因子 |
| 营业总收入 | 营收总额 | PS 因子、成长 |
| 营业总收入-同比增长 | 营收增速 | 成长因子 |
| 营业总收入-季度环比增长 | 营收环比 | 成长因子 |
| 净利润 | 净利润额 | 盈利质量 |
| 净利润-同比增长 | 利润增速 | 成长因子 |
| 净利润-季度环比增长 | 利润环比 | 成长因子 |
| 每股净资产 | BVPS | PB 因子 |
| 净资产收益率 | ROE | 质量因子 |
| 每股经营现金流量 | 每股经营现金流 | 现金流质量 |
| 销售毛利率 | 毛利率% | 质量因子 |
| 所处行业 | 行业分类 | 行业因子 |
| 最新公告日期 | 公告日期 | 数据时效 |

#### 资产负债表（批量）

```python
df = ak.stock_zcfz_em(date="20241231")
```

**返回字段**：股票代码, 股票简称, 资产-总资产, 负债-总负债, 资产负债率, 股东权益合计, 资产-货币资金, 资产-应收账款, 资产-存货

#### 单只股票财务报表

```python
# 利润表（年度）
df = ak.stock_profit_sheet_by_yearly_em(symbol="SH600519")

# 资产负债表（报告期）
df = ak.stock_balance_sheet_by_report_em(symbol="SH600519")

# 现金流量表（季度）
df = ak.stock_cash_flow_sheet_by_quarterly_em(symbol="SH600519")
```

**注意**：单只接口代码格式需带交易所前缀（`SH`/`SZ` + 6位代码）

### 3.9 报告期格式

| 值 | 含义 |
|----|------|
| `20241231` | 年报 |
| `20240930` | 三季报 |
| `20240630` | 半年报 |
| `20240331` | 一季报 |

### 3.10 股票代码格式

| 格式 | 用途 | 示例 |
|------|------|------|
| 纯6位 | 批量查询（`stock_zh_a_hist`, `stock_yjbb_em`） | `000001` |
| `SH`+6位 | 单只财务接口（沪市） | `SH600519` |
| `SZ`+6位 | 单只财务接口（深市） | `SZ000001` |
| `sh`/`sz`+6位 | 部分历史接口 | `sz000001` |

---

## 4. 可转债

### 4.1 可转债列表

```python
df = ak.bond_zh_cov()
```

**包含**：转债代码、名称、正股代码、转股价、上市日期

### 4.2 可转债实时行情

```python
df = ak.bond_zh_hs_cov_spot()
```

**核心字段**：最新价、转股溢价率、双低值、成交额
**适用**：双低策略、可转债轮动、套利

### 4.3 可转债历史行情

```python
df = ak.bond_zh_hs_cov_daily(symbol="123107")
```

### 4.4 可转债价值分析

```python
df = ak.bond_cov_comparison()
```

**包含**：转股价值、溢价率、纯债价值

### 4.5 可转债强赎数据

```python
df = ak.bond_cb_redeem_jsl()
```

**数据源**：集思录 | **适用**：强赎风险监控

---

## 5. 指数数据

### 5.1 指数实时行情

```python
df = ak.stock_zh_index_spot()
```

**包含**：上证指数、深证成指、创业板指、沪深300 等

### 5.2 指数历史行情

```python
df = ak.index_zh_a_hist(
    symbol="000300",
    period="daily",
    start_date="20240101",
    end_date="20241231"
)
```

**常用指数代码**：

| 代码 | 名称 |
|------|------|
| `000001` | 上证指数 |
| `000300` | 沪深300 |
| `000905` | 中证500 |
| `000852` | 中证1000 |
| `399001` | 深证成指 |
| `399006` | 创业板指 |

### 5.3 行业指数

```python
df = ak.stock_board_industry_name_em()
```

### 5.4 概念板块

```python
df = ak.stock_board_concept_name_em()
```

### 5.5 板块成分股

```python
df = ak.stock_board_industry_cons_em(symbol="证券")
```

---

## 6. ETF / 基金

### 6.1 ETF 实时行情

```python
df = ak.fund_etf_spot_em()
```

### 6.2 ETF 历史行情

```python
df = ak.fund_etf_hist_em(symbol="159915")
```

### 6.3 公募基金列表

```python
df = ak.fund_name_em()
```

---

## 7. 期货期权

### 7.1 主力连续行情

```python
df = ak.futures_main_sina(symbol="RB0")   # 螺纹钢主力连续
```

### 7.2 期货实时行情

```python
df = ak.futures_zh_spot()
```

### 7.3 ETF 期权

```python
df = ak.option_current_em()
```

### 7.4 上证50ETF 期权

```python
df = ak.option_sse_spot_price_sina(symbol="510050")
```

---

## 8. 宏观经济

| 接口 | 说明 | 适用 |
|------|------|------|
| `macro_china_gdp()` | GDP | 经济周期 |
| `macro_china_cpi()` | CPI | 通胀因子 |
| `macro_china_ppi()` | PPI | 工业品价格 |
| `macro_china_shrzgm()` | 社融 | 信用周期 |
| `macro_china_money_supply()` | M2/M1 | 流动性 |
| `macro_china_lpr()` | LPR | 利率 |
| `macro_china_pmi_yearly()` | PMI | 景气度 |
| `macro_china_fx_reserves_yearly()` | 外汇储备 | 国际收支 |
| `macro_cnbs()` | 宏观杠杆率 | 金融稳定 |

**数据源**：国家统计局、央行、国家金融与发展实验室

---

## 9. 利率 / 国债

### 9.1 中美债券收益率

```python
df = ak.bond_zh_us_rate()
```

### 9.2 中国债券收益率曲线

```python
df = ak.bond_china_yield()
```

---

## 10. 北向资金

### 10.1 沪深港通资金流向

```python
df = ak.stock_hsgt_fund_flow_summary_em()
```

### 10.2 北向持股明细

```python
df = ak.stock_hsgt_hold_stock_em()
```

**适用**：外资因子、聪明钱跟踪

---

## 11. 行业板块 / 概念

### 11.1 行业板块列表

```python
df = ak.stock_board_industry_name_em()
```

### 11.2 概念板块列表

```python
df = ak.stock_board_concept_name_em()
```

### 11.3 板块成分股

```python
df = ak.stock_board_industry_cons_em(symbol="证券")
df = ak.stock_board_concept_cons_em(symbol="人工智能")
```

### 11.4 板块行情

```python
df = ak.stock_board_industry_hist_em(symbol="证券", period="daily")
df = ak.stock_board_concept_hist_em(symbol="人工智能", period="daily")
```

---

## 12. 龙虎榜

```python
df = ak.stock_lhb_detail_em(start_date="20240101", end_date="20240131")
```

**适用**：异动追踪、游资跟踪

---

## 13. 新闻舆情

```python
# 财经新闻
df = ak.news_cctv()

# 个股新闻
df = ak.stock_news_em(symbol="600519")
```

---

## 14. 数据源稳定性

| 数据源 | 稳定性 | 特点 |
|--------|--------|------|
| 东方财富 | **最稳定** | 主力数据源，推荐 |
| 新浪 | 较好 | 高频较快，偶发限流 |
| 同花顺 | 一般 | 偶发风控 |
| 雪球 | 一般 | 有时限流 |
| 集思录 | 较好 | 转债数据专源 |
| 交易所 | 最稳定 | 官方数据 |

---

## 15. 常见问题与最佳实践

### 15.1 被限流

**原因**：IP 限制、高频访问封禁、headers 校验
**解决**：
```python
import time
time.sleep(0.3)          # 请求间隔
```

大规模下载建议：
- 控制并发数（3-8 线程）
- 请求间隔（0.3-1 秒）
- 失败重试（指数退避）
- 本地 Parquet 缓存

### 15.2 推荐存储方案

| 场景 | 方案 |
|------|------|
| 日线行情 | Parquet + Pandas |
| 分钟线 | Parquet + Polars |
| 因子库 | DuckDB |
| 全量数据 | ClickHouse |

### 15.3 防封策略

```python
# 批量下载推荐参数
workers = 5                # 并发线程
sleep_between = 0.3        # 请求间隔
max_retries = 5            # 重试次数
```

---

## 16. 推荐架构

```
AKShare (数据采集)
   ↓
Parquet / ClickHouse (数据存储)
   ↓
Feature Engineering (特征工程)
   ↓
Factor Engine (因子计算)
   ↓
Strategy Engine (策略引擎)
   ↓
Backtest (回测)
   ↓
Execution (执行)
```

---

## 17. 接口速查总表

### 日常量化必备

| 接口 | 用途 | 数据源 | 稳定性 |
|------|------|--------|--------|
| `stock_zh_a_hist` | 日K线（12字段） | 东方财富 | ★★★★★ |
| `stock_zh_a_spot_em` | 全市场实时 | 东方财富 | ★★★★★ |
| `stock_zh_a_hist_min_em` | 分钟线 | 东方财富 | ★★★★★ |
| `stock_yjbb_em` | 业绩报表 | 东方财富 | ★★★★★ |
| `stock_zcfz_em` | 资产负债表 | 东方财富 | ★★★★★ |
| `stock_hsgt_hold_stock_em` | 北向持股 | 东方财富 | ★★★★ |
| `stock_board_industry_cons_em` | 板块成分 | 东方财富 | ★★★★ |
| `index_zh_a_hist` | 指数行情 | 东方财富 | ★★★★ |
| `bond_zh_hs_cov_spot` | 转债实时 | 东方财富 | ★★★★ |
| `fund_etf_spot_em` | ETF实时 | 东方财富 | ★★★★ |
| `macro_china_cpi` | CPI | 统计局 | ★★★★★ |
| `macro_china_lpr` | LPR | 央行 | ★★★★★ |

### 策略推荐组合

#### A股量化
```python
stock_zh_a_hist              # K线
stock_zh_a_spot_em           # 实时
stock_hsgt_hold_stock_em     # 北向
stock_yjbb_em                # 财务
```

#### 可转债
```python
bond_zh_hs_cov_spot          # 实时
bond_cov_comparison          # 价值分析
bond_cb_redeem_jsl           # 强赎
```

#### 宏观择时
```python
macro_china_cpi              # CPI
macro_china_shrzgm           # 社融
macro_china_money_supply     # M2
macro_china_lpr              # LPR
```

---

## AKShare vs Tushare

| 维度 | AKShare | Tushare |
|------|---------|---------|
| 免费 | **强** | 一般 |
| Token | 不需要 | 需要 |
| 稳定性 | 一般 | **强** |
| 宏观数据 | **很强** | 中等 |
| 高频 | 较强 | 较弱 |
| 财务数据 | 中等 | **很强** |
| 转债 | **很强** | 一般 |
| 社区活跃 | **很强** | 强 |

---

## 相关文档

- [AKShare 官方文档](https://akshare.akfamily.xyz/data/index.html)
- [AKShare GitHub](https://github.com/akfamily/akshare)
- [项目数据获取指南](./akshare_data_guide.md)
- [AKShare 速查手册](./akshare_cheatsheet.md)
- [策略开发指南](./strategy_development_guide.md)
