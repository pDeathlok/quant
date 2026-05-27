# Tushare Pro 完整接口文档（基于当前权限）

---

## 📖 官方文档入口

| 文档 | 链接 |
|------|------|
| 首页 | https://tushare.pro/document/2 |
| 接口列表 | https://tushare.pro/document/1?doc_id=108 |
| 通用行情 | https://tushare.pro/document/2?doc_id=14 |

---

## 🔐 当前权限覆盖

| 类别 | 内容 |
|------|------|
| **基础数据** | A股 / 基金 / 期货 / 期权 / 港股 / 美股 / 外汇 |
| **行情数据** | 日/周/月线行情 |
| **财务数据** | 三大报表（利润表、资产负债表、现金流量表） |
| **宏观经济** | GDP、CPI、PPI、M2、Shibor、LPR |
| **特殊数据** | ST股票、沪港通/深港通 |
| **参考数据** | 质押、解禁、回购、增减持、龙虎榜、融资融券 |

---

## 1. 初始化

### 1.1 安装

```bash
pip install tushare pandas
```

### 1.2 初始化 API

```python
import tushare as ts

# 设置Token
ts.set_token("wwqxe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq")

# 初始化接口
pro = ts.pro_api()
```

### 1.3 使用封装类（推荐）

```python
from quant.data import TushareDataFetcher

# 初始化
fetcher = TushareDataFetcher(token="wwqxe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq")

# 获取数据
df = fetcher.get_stock_daily("600000", "20240101", "20241231")
```

---

## 2. 基础数据接口

### 2.1 A股股票列表

**官方文档**: https://tushare.pro/document/2?doc_id=25

**接口**: `pro.stock_basic()`

**常用字段**:

| 字段 | 含义 |
|------|------|
| ts_code | 股票代码 |
| symbol | 股票代码简写 |
| name | 股票名称 |
| area | 地区 |
| industry | 行业 |
| market | 市场类型 |
| list_date | 上市日期 |

**示例**:

```python
df = pro.stock_basic(
    exchange='',
    list_status='L',
    fields='ts_code,symbol,name,area,industry,list_date'
)
```

### 2.2 基金基础信息

**官方文档**: https://tushare.pro/document/2?doc_id=123

**接口**: `pro.fund_basic()`

**示例**:

```python
df = pro.fund_basic(market='E')
```

### 2.3 期货基础信息

**官方文档**: https://tushare.pro/document/2?doc_id=135

**接口**: `pro.fut_basic()`

### 2.4 期权基础信息

**官方文档**: https://tushare.pro/document/2?doc_id=161

**接口**: `pro.opt_basic()`

### 2.5 港股基础信息

**官方文档**: https://tushare.pro/document/2?doc_id=196

**接口**: `pro.hk_basic()`

### 2.6 美股基础信息

**官方文档**: https://tushare.pro/document/2?doc_id=195

**接口**: `pro.us_basic()`

### 2.7 外汇基础信息

**官方文档**: https://tushare.pro/document/2?doc_id=218

**接口**: `pro.fx_obasic()`

---

## 3. 行情数据接口

### 3.1 日线行情

**官方文档**: https://tushare.pro/document/2?doc_id=27

**接口**: `pro.daily()`

**常用字段**:

| 字段 | 含义 |
|------|------|
| open | 开盘价 |
| high | 最高价 |
| low | 最低价 |
| close | 收盘价 |
| vol | 成交量(手) |
| amount | 成交额(千元) |

**示例**:

```python
df = pro.daily(
    ts_code='000001.SZ',
    start_date='20240101',
    end_date='20241231'
)
```

**积分消耗**: 1积分/次

### 3.2 周线行情

**官方文档**: https://tushare.pro/document/2?doc_id=144

**接口**: `pro.weekly()`

**示例**:

```python
df = pro.weekly(
    ts_code='000001.SZ',
    start_date='20240101',
    end_date='20241231'
)
```

### 3.3 月线行情

**官方文档**: https://tushare.pro/document/2?doc_id=145

**接口**: `pro.monthly()`

**示例**:

```python
df = pro.monthly(
    ts_code='000001.SZ',
    start_date='20240101',
    end_date='20241231'
)
```

### 3.4 通用行情接口（推荐）

**官方文档**: https://tushare.pro/document/1?doc_id=109

**接口**: `ts.pro_bar()`

**支持**:
- 股票、ETF、指数、期货、数字货币
- 前复权、后复权

**示例**:

```python
df = ts.pro_bar(
    ts_code='000001.SZ',
    adj='qfq',  # 前复权
    start_date='20240101',
    end_date='20241231'
)
```

**积分消耗**: 5积分/次（分钟线）

---

## 4. 财务数据（三大报表）

### 4.1 利润表

**官方文档**: https://tushare.pro/document/2?doc_id=33

**接口**: `pro.income()`

**常用字段**:

| 字段 | 含义 |
|------|------|
| revenue | 营业收入 |
| operate_profit | 营业利润 |
| total_profit | 利润总额 |
| n_income | 净利润 |

**示例**:

```python
df = pro.income(
    ts_code='600000.SH',
    start_date='20230101'
)
```

**积分消耗**: 50积分/次

### 4.2 资产负债表

**官方文档**: https://tushare.pro/document/2?doc_id=36

**接口**: `pro.balancesheet()`

**重要字段**:

| 字段 | 含义 |
|------|------|
| total_assets | 总资产 |
| total_liab | 总负债 |
| money_cap | 货币资金 |
| inventories | 存货 |

**积分消耗**: 50积分/次

### 4.3 现金流量表

**官方文档**: https://tushare.pro/document/2?doc_id=44

**接口**: `pro.cashflow()`

**重要字段**:

| 字段 | 含义 |
|------|------|
| n_cashflow_act | 经营现金流 |
| c_pay_acq_const_fiolta | 资本开支 |
| net_profit | 净利润 |

**示例**:

```python
df = pro.cashflow(
    ts_code='600000.SH',
    start_date='20230101'
)
```

**积分消耗**: 50积分/次

### 4.4 财务指标

**官方文档**: https://tushare.pro/document/2?doc_id=79

**接口**: `pro.fina_indicator()`

**常用指标**:

| 字段 | 含义 |
|------|------|
| roe | ROE |
| roa | ROA |
| grossprofit_margin | 毛利率 |
| debt_to_assets | 资产负债率 |

**积分消耗**: 50积分/次

---

## 5. 宏观经济数据

### 5.1 GDP

**官方文档**: https://tushare.pro/document/2?doc_id=221

**接口**: `pro.gdp()`

**积分消耗**: 0积分

### 5.2 CPI

**官方文档**: https://tushare.pro/document/2?doc_id=222

**接口**: `pro.cpi()`

**积分消耗**: 0积分

### 5.3 PPI

**官方文档**: https://tushare.pro/document/2?doc_id=223

**接口**: `pro.ppi()`

**积分消耗**: 0积分

### 5.4 M2

**官方文档**: https://tushare.pro/document/2?doc_id=224

**接口**: `pro.cn_m()`

**积分消耗**: 0积分

### 5.5 Shibor

**官方文档**: https://tushare.pro/document/2?doc_id=157

**接口**: `pro.shibor()`

**积分消耗**: 0积分

### 5.6 LPR

**官方文档**: https://tushare.pro/document/2?doc_id=325

**接口**: `pro.lpr()`

**积分消耗**: 0积分

---

## 6. ST 股票

**官方文档**: https://tushare.pro/document/2?doc_id=100

**接口**: `pro.namechange()`

**示例**:

```python
df = pro.namechange(ts_code='000001.SZ')
```

**积分消耗**: 10积分/次

---

## 7. 沪港通 / 深港通

**官方文档**: https://tushare.pro/document/2?doc_id=94

**接口**: `pro.hs_const()`

**示例**:

```python
df = pro.hs_const(hs_type='SH')  # SH: 沪港通, SZ: 深港通
```

**积分消耗**: 1积分/次

---

## 8. 参考数据

### 8.1 股权质押

**官方文档**: https://tushare.pro/document/2?doc_id=110

**接口**: `pro.pledge_stat()`

**积分消耗**: 10积分/次

### 8.2 限售解禁

**官方文档**: https://tushare.pro/document/2?doc_id=160

**接口**: `pro.share_float()`

**积分消耗**: 10积分/次

### 8.3 股票回购

**官方文档**: https://tushare.pro/document/2?doc_id=124

**接口**: `pro.repurchase()`

**积分消耗**: 10积分/次

### 8.4 增减持

**官方文档**: https://tushare.pro/document/2?doc_id=175

**接口**: `pro.stk_holdertrade()`

**积分消耗**: 10积分/次

### 8.5 龙虎榜

**官方文档**: https://tushare.pro/document/2?doc_id=106

**接口**: `pro.top_list()`

**积分消耗**: 10积分/次

### 8.6 融资融券

**官方文档**: https://tushare.pro/document/2?doc_id=58

**接口**: `pro.margin()`

**积分消耗**: 10积分/次

---

## 9. 天级别数据获取与建模准备

### 9.1 日线行情完整字段说明

**接口**: `pro.daily()`

| 字段 | 类型 | 含义 | 建模用途 |
|------|------|------|----------|
| ts_code | string | 股票代码 | 主键标识 |
| trade_date | string | 交易日期 | 时间索引 |
| open | float | 开盘价 | 特征/预测目标 |
| high | float | 最高价 | 波动特征 |
| low | float | 最低价 | 波动特征 |
| close | float | 收盘价 | 核心特征/标签 |
| pre_close | float | 昨收价 | 计算收益率 |
| change | float | 涨跌额 | 特征 |
| pct_chg | float | 涨跌幅(%) | 核心特征 |
| vol | float | 成交量(手) | 量能特征 |
| amount | float | 成交额(千元) | 量能特征 |

### 9.2 获取前复权日线数据（建模首选）

```python
import tushare as ts
import pandas as pd

ts.set_token("wwqxe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq")
pro = ts.pro_api()

# 方法1：使用pro_bar自动复权（推荐）
df = ts.pro_bar(
    ts_code='600000.SH',
    adj='qfq',              # 前复权: qfq, 后复权: hfq, 不复权: None
    start_date='20200101',
    end_date='20241231',
    ma=[5, 10, 20, 60]      # 自动计算均线
)

# 方法2：手动复权
daily = pro.daily(ts_code='600000.SH', start_date='20200101', end_date='20241231')
adj_factor = pro.adj_factor(ts_code='600000.SH', start_date='20200101', end_date='20241231')
df = daily.merge(adj_factor, on='trade_date')
df['close_qfq'] = df['close'] * df['adj_factor']
df['open_qfq'] = df['open'] * df['adj_factor']
df['high_qfq'] = df['high'] * df['adj_factor']
df['low_qfq'] = df['low'] * df['adj_factor']
```

### 9.3 批量获取多只股票日线数据

```python
def get_multi_stocks_daily(ts_codes, start_date, end_date):
    """批量获取多只股票日线数据"""
    all_data = []
    for code in ts_codes:
        try:
            df = ts.pro_bar(
                ts_code=code,
                adj='qfq',
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and len(df) > 0:
                df['ts_code'] = code  # 添加股票代码
                all_data.append(df)
        except Exception as e:
            print(f"获取 {code} 失败: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

# 使用示例
stocks = ['600000.SH', '000001.SZ', '600519.SH']
df = get_multi_stocks_daily(stocks, '20200101', '20241231')
```

### 9.4 数据预处理（建模必备）

```python
def preprocess_daily_data(df):
    """日线数据预处理"""
    # 1. 转换日期格式
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df.set_index('trade_date', inplace=True)
    
    # 2. 计算收益率特征
    df['return'] = df['close'].pct_change() * 100
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 3. 计算波动特征
    df['range'] = df['high'] - df['low']
    df['range_pct'] = (df['high'] - df['low']) / df['open'] * 100
    
    # 4. 计算量能特征
    df['volume_ratio'] = df['vol'] / df['vol'].rolling(20).mean()
    
    # 5. 计算移动平均
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    
    # 6. 计算布林带
    df['boll_up'] = df['ma20'] + 2 * df['close'].rolling(20).std()
    df['boll_down'] = df['ma20'] - 2 * df['close'].rolling(20).std()
    
    # 7. 计算MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # 8. 计算RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 9. 计算KDJ
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    df['k'] = ((df['close'] - low_min) / (high_max - low_min)) * 100
    df['d'] = df['k'].rolling(3).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    # 10. 计算未来收益（预测目标）
    df['future_1d_return'] = df['close'].pct_change(1).shift(-1) * 100
    df['future_5d_return'] = df['close'].pct_change(5).shift(-5) * 100
    df['future_20d_return'] = df['close'].pct_change(20).shift(-20) * 100
    
    # 11. 生成分类标签
    df['label_up'] = (df['future_5d_return'] > 3).astype(int)
    df['label_down'] = (df['future_5d_return'] < -3).astype(int)
    df['label_quality'] = pd.cut(
        df['future_5d_return'],
        bins=[-float('inf'), -5, -2, 2, 5, float('inf')],
        labels=[0, 1, 2, 3, 4]
    )
    
    # 12. 填充缺失值
    df = df.dropna()
    
    return df
```

### 9.5 构建建模数据集

```python
def build_model_dataset(ts_codes, start_date, end_date):
    """构建完整的建模数据集"""
    # 1. 获取基础数据
    df = get_multi_stocks_daily(ts_codes, start_date, end_date)
    
    if df is None or len(df) == 0:
        return None
    
    # 2. 预处理
    df = preprocess_daily_data(df)
    
    # 3. 添加股票基本信息
    stock_basic = pro.stock_basic(
        list_status='L',
        fields='ts_code,symbol,name,industry,list_date'
    )
    df = df.merge(stock_basic, on='ts_code', how='left')
    
    # 4. 添加财务指标（季度频率）
    fina = pro.fina_indicator(
        start_date=start_date,
        end_date=end_date,
        fields='ts_code,end_date,roe,roa,eps,pe,pb'
    )
    fina['end_date'] = pd.to_datetime(fina['end_date'], format='%Y%m%d')
    df = df.merge(fina, left_on=['ts_code', df.index.to_period('Q').to_timestamp()], 
                  right_on=['ts_code', 'end_date'], how='left')
    
    # 5. 保存数据集
    df.to_parquet(f"model_dataset_{start_date}_{end_date}.parquet")
    
    return df

# 使用示例
stocks = pro.stock_basic(list_status='L', fields='ts_code')['ts_code'].tolist()[:100]  # 取100只股票
dataset = build_model_dataset(stocks, '20200101', '20241231')
```

### 9.6 数据集特征列表

#### 价格特征
| 特征名 | 计算方式 | 用途 |
|--------|----------|------|
| open | 开盘价 | 特征 |
| high | 最高价 | 波动分析 |
| low | 最低价 | 波动分析 |
| close | 收盘价 | 核心特征 |

#### 收益特征
| 特征名 | 计算方式 | 用途 |
|--------|----------|------|
| return | 日收益率 | 因子计算 |
| log_return | 对数收益率 | 风险模型 |
| future_1d_return | 未来1日收益 | 预测目标 |
| future_5d_return | 未来5日收益 | 预测目标 |
| future_20d_return | 未来20日收益 | 预测目标 |

#### 技术指标
| 特征名 | 计算方式 | 用途 |
|--------|----------|------|
| ma5/ma10/ma20/ma60 | 移动平均 | 趋势判断 |
| boll_up/boll_down | 布林带 | 波动率 |
| macd/signal | MACD指标 | 趋势动量 |
| rsi | 相对强弱指数 | 超买超卖 |
| k/d/j | KDJ指标 | 超买超卖 |

#### 量能特征
| 特征名 | 计算方式 | 用途 |
|--------|----------|------|
| vol | 成交量 | 量能分析 |
| amount | 成交额 | 量能分析 |
| volume_ratio | 成交量/20日均量 | 量能对比 |

#### 财务特征
| 特征名 | 含义 | 用途 |
|--------|------|------|
| roe | 净资产收益率 | 盈利能力 |
| roa | 总资产收益率 | 盈利能力 |
| eps | 每股收益 | 盈利质量 |
| pe | 市盈率 | 估值 |
| pb | 市净率 | 估值 |

#### 标签特征
| 特征名 | 含义 | 用途 |
|--------|------|------|
| label_up | 未来5日涨幅>3% | 分类任务 |
| label_down | 未来5日跌幅<-3% | 分类任务 |
| label_quality | 5级质量标签 | 多分类任务 |
| future_5d_return | 连续收益值 | 回归任务 |

### 9.7 数据存储格式建议

```
data/
├── raw/                    # 原始数据
│   ├── daily/              # 日线数据
│   │   ├── 600000.SH.parquet
│   │   ├── 000001.SZ.parquet
│   │   └── ...
│   └── stock_basic.parquet # 股票基本信息
├── processed/              # 处理后数据
│   └── model_dataset.parquet  # 完整建模数据集
└── features/               # 特征数据
    ├── technical.parquet   # 技术指标
    ├── financial.parquet   # 财务指标
    └── macro.parquet       # 宏观因子
```

### 9.8 高频开发模板

#### 9.8.1 批量获取全市场日线

```python
import tushare as ts
import pandas as pd

ts.set_token("wwqxe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq")
pro = ts.pro_api()

# 获取股票列表
stocks = pro.stock_basic(list_status='L', fields='ts_code')

all_df = []
for code in stocks['ts_code']:
    try:
        df = pro.daily(
            ts_code=code,
            start_date='20240101',
            end_date='20241231'
        )
        all_df.append(df)
    except Exception as e:
        print(f"获取 {code} 失败: {e}")

# 合并数据
result = pd.concat(all_df)
result.to_parquet("all_stocks_daily.parquet")
```

### 9.2 获取财务三表

```python
# 获取财务数据
income = pro.income(ts_code='600000.SH')
balance = pro.balancesheet(ts_code='600000.SH')
cashflow = pro.cashflow(ts_code='600000.SH')

# 获取财务指标
indicators = pro.fina_indicator(ts_code='600000.SH')
```

### 9.3 获取宏观数据

```python
# 获取宏观经济数据
gdp = pro.gdp()
cpi = pro.cpi()
ppi = pro.ppi()
m2 = pro.cn_m()
shibor = pro.shibor()
lpr = pro.lpr()
```

### 9.4 因子研究数据获取

```python
def get_factor_data(ts_code, start_date, end_date):
    """获取因子研究所需数据"""
    # 行情数据
    daily = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    
    # 复权因子
    adj = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
    
    # 财务指标
    fina = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
    
    # 合并数据
    df = daily.merge(adj, on='trade_date', how='left')
    df = df.merge(fina, on='ts_code', how='left')
    
    return df
```

---

## 10. 权限与频率说明

### 10.1 当前权限覆盖范围

| 研究类型 | 是否覆盖 | 说明 |
|----------|----------|------|
| 常规低频量化研究 | ✅ | 完全覆盖 |
| 财务因子 | ✅ | 完全覆盖 |
| 基本面分析 | ✅ | 完全覆盖 |
| 宏观分析 | ✅ | 完全覆盖 |
| 风控策略研究 | ✅ | 完全覆盖 |
| 多资产研究 | ✅ | 完全覆盖 |

### 10.2 暂未覆盖的内容

| 类型 | 说明 |
|------|------|
| 分钟级行情 | 需要更高积分等级 |
| Tick数据 | 需要更高积分等级 |
| Level2数据 | 需要更高积分等级 |
| 高频数据 | 需要更高积分等级 |
| 部分特色因子 | 需要更高积分等级 |

### 10.3 积分消耗表

| 接口类型 | 消耗积分 | 频率限制 |
|----------|----------|----------|
| daily/weekly/monthly | 1分 | 无限制 |
| pro_bar(日线) | 1分 | 无限制 |
| pro_bar(分钟线) | 5分 | 100次/分钟 |
| index_daily | 1分 | 无限制 |
| stock_basic | 0分 | 无限制 |
| fina_indicator/income/balance/cashflow | 50分 | 10次/分钟 |
| gdp/cpi/ppi/cn_m/shibor/lpr | 0分 | 无限制 |
| margin/top_list/repurchase | 10分 | 无限制 |

---

## 11. 推荐核心接口

### 按场景推荐

| 场景 | 核心接口 |
|------|----------|
| **股票池构建** | `stock_basic` |
| **K线数据** | `daily` / `pro_bar` |
| **财务因子** | `fina_indicator` |
| **三大报表** | `income` / `balancesheet` / `cashflow` |
| **风险事件** | `pledge_stat` / `share_float` |
| **市场情绪** | `top_list` |
| **杠杆资金** | `margin` |
| **宏观因子** | `gdp` / `cpi` / `ppi` / `shibor` |

### 推荐的 Data Layer 架构

```
┌─────────────────────────────────────────┐
│              Data Layer                │
├─────────────────────────────────────────┤
│  股票基础数据  │  stock_basic          │
├─────────────────────────────────────────┤
│  行情数据     │  daily / pro_bar      │
├─────────────────────────────────────────┤
│  财务数据     │  fina_indicator       │
│              │  income/balance/cashflow│
├─────────────────────────────────────────┤
│  风险数据     │  pledge_stat          │
│              │  share_float           │
├─────────────────────────────────────────┤
│  市场数据     │  top_list / margin    │
├─────────────────────────────────────────┤
│  宏观数据     │  gdp/cpi/ppi/shibor   │
└─────────────────────────────────────────┘
```

---

## 12. 常见问题

### Q1: Token 如何设置？

```python
# 方式1：代码中设置
ts.set_token("wwqxe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq")

# 方式2：环境变量
export TUSHARE_TOKEN="wwqxe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq"
```

### Q2: 如何获取前复权数据？

```python
# 使用 pro_bar
df = ts.pro_bar(ts_code="600000.SH", adj="qfq")

# 或手动计算
daily = pro.daily(ts_code="600000.SH")
adj = pro.adj_factor(ts_code="600000.SH")
df = daily.merge(adj, on='trade_date')
df['close_qfq'] = df['close'] * df['adj_factor']
```

### Q3: 接口返回空数据？

**可能原因**:
1. 股票代码格式不正确
2. 日期范围没有数据
3. 积分不足
4. 接口调用频率超限

### Q4: 如何处理大量股票数据？

```python
# 分批获取，加入异常处理
stocks = pro.stock_basic(list_status='L')['ts_code'].tolist()

for i, code in enumerate(stocks):
    if i % 100 == 0:
        print(f"Processing {i}/{len(stocks)}")
    
    try:
        df = pro.daily(ts_code=code, start_date="20240101", end_date="20241231")
        df.to_parquet(f"data/{code}.parquet")
    except Exception as e:
        print(f"Error: {code} - {e}")
```

---

## 📞 联系方式

- **官方网站**: https://tushare.pro/
- **用户中心**: https://tushare.pro/usercenter
- **帮助文档**: https://tushare.pro/document/2

---

## 📋 用户信息

| 属性 | 值 |
|------|------|
| **Token** | wwqxe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq |
| **积分** | 5000 积分 |
| **有效期** | 2027-05-15 |

---

*文档生成日期: 2026-05-27*
*基于 Tushare Pro 官方文档整理*