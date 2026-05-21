# AKShare 数据获取指南

## 一、核心数据接口

### 1. 财务报表数据

#### 业绩报表（推荐）
```python
import akshare as ak

# 获取所有股票的业绩报表
df = ak.stock_yjbb_em(date="20241231")

# 返回字段：
# - 股票代码, 股票简称
# - 每股收益
# - 营业总收入-营业总收入, 营业总收入-同比增长, 营业总收入-季度环比增长
# - 净利润-净利润, 净利润-同比增长, 净利润-季度环比增长
# - 每股净资产, 净资产收益率
# - 每股经营现金流量
# - 销售毛利率
# - 所处行业
# - 最新公告日期
```

**适用因子：**
- 价值因子：每股收益、每股净资产
- 质量因子：净资产收益率、销售毛利率
- 成长因子：营业总收入同比增长、净利润同比增长
- 行业因子：所处行业

---

#### 资产负债表
```python
# 获取所有股票的资产负债表
df = ak.stock_zcfz_em(date="20241231")

# 返回字段：
# - 股票代码, 股票简称
# - 资产-货币资金, 资产-应收账款, 资产-存货, 资产-总资产
# - 资产-总资产同比
# - 负债-应付账款, 负债-预收账款, 负债-总负债
# - 负债-总负债同比
# - 资产负债率
# - 股东权益合计
```

**适用因子：**
- 杠杆因子：资产负债率
- 质量因子：总资产、股东权益合计

---

#### 单只股票财务报表
```python
# 利润表（按年度）
df = ak.stock_profit_sheet_by_yearly_em(symbol="SH600000")

# 资产负债表（按报告期）
df = ak.stock_balance_sheet_by_report_em(symbol="SH600000")

# 现金流量表（按季度）
df = ak.stock_cash_flow_sheet_by_quarterly_em(symbol="SH600000")
```

---

### 2. 行业分类数据

```python
# 从业绩报表中获取行业信息
df = ak.stock_yjbb_em(date="20241231")
industry_df = df[["股票代码", "股票简称", "所处行业"]]

# 返回字段：
# - 股票代码
# - 股票简称
# - 所处行业
```

**适用因子：**
- 行业因子：行业分类、行业中性化

---

### 3. 指数数据

```python
# 获取指数日线数据
df = ak.index_zh_a_hist(
    symbol="sh000300",  # 沪深300
    start_date="20230101",
    end_date="20231231"
)

# 返回字段：
# - 日期, 开盘, 最高, 最低, 收盘
# - 成交量, 成交额
```

**适用因子：**
- 市场基准收益率
- 相对强弱因子（RSTR）

---

## 二、数据接口与因子映射

| 数据类型 | AKShare 接口 | 关键字段 | 适用因子 |
|---------|-------------|---------|---------|
| **业绩报表** | `stock_yjbb_em(date)` | 每股收益、每股净资产、净资产收益率 | PERatio、ROE、ROA |
| **业绩报表** | `stock_yjbb_em(date)` | 营业总收入、净利润 | GrowthScore、RevenueGrowthAcceleration |
| **资产负债表** | `stock_zcfz_em(date)` | 总资产、总负债、股东权益 | DebtToEquity、Leverage |
| **资产负债表** | `stock_zcfz_em(date)` | 资产负债率 | DebtToEquity |
| **业绩报表** | `stock_yjbb_em(date)` | 所处行业 | IndustryFactor、IndustryDummy |
| **指数数据** | `index_zh_a_hist()` | 收盘价 | MarketReturn、RSTR |

---

## 三、完整数据获取示例

```python
import akshare as ak
import pandas as pd
from datetime import datetime

class FactorDataLoader:
    """因子数据加载器"""
    
    def __init__(self):
        self.report_date = "20241231"
    
    def load_financial_data(self) -> pd.DataFrame:
        """加载财务数据（业绩报表）"""
        df = ak.stock_yjbb_em(date=self.report_date)
        return df
    
    def load_balance_sheet(self) -> pd.DataFrame:
        """加载资产负债表"""
        df = ak.stock_zcfz_em(date=self.report_date)
        return df
    
    def load_industry_data(self) -> pd.DataFrame:
        """加载行业数据"""
        df = ak.stock_yjbb_em(date=self.report_date)
        return df[["股票代码", "股票简称", "所处行业"]]
    
    def load_index_data(self, symbol: str = "sh000300", 
                        start_date: str = "20230101", 
                        end_date: str = "20231231") -> pd.DataFrame:
        """加载指数数据"""
        df = ak.index_zh_a_hist(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        df["return"] = df["收盘"].pct_change()
        df["date"] = pd.to_datetime(df["日期"]).dt.date
        return df[["date", "return"]]
    
    def merge_all_data(self) -> pd.DataFrame:
        """合并所有数据"""
        # 加载财务数据
        financial = self.load_financial_data()
        
        # 加载资产负债表
        balance = self.load_balance_sheet()
        
        # 合并数据
        df = financial.merge(balance, on=["股票代码", "股票简称"], how="left")
        
        return df

# 使用示例
loader = FactorDataLoader()

# 1. 加载财务数据
financial_df = loader.load_financial_data()
print(f"财务数据: {financial_df.shape}")
print(financial_df.head())

# 2. 加载资产负债表
balance_df = loader.load_balance_sheet()
print(f"\n资产负债表: {balance_df.shape}")
print(balance_df.head())

# 3. 加载行业数据
industry_df = loader.load_industry_data()
print(f"\n行业数据: {industry_df.shape}")
print(industry_df.head())

# 4. 加载指数数据
index_df = loader.load_index_data()
print(f"\n指数数据: {index_df.shape}")
print(index_df.head())
```

---

## 四、因子计算示例

```python
from data.factors import (
    PERatio, ROE, MarketCap, Volatility,
    GrowthScore, IndustryFactor, FactorComposite
)

# 1. 计算价值因子
pe_factor = PERatio()
df["pe_ratio"] = df["每股收益"] / df["收盘价"]

# 2. 计算质量因子
roe_factor = ROE()
df["roe"] = df["净资产收益率"]

# 3. 计算成长因子
growth_factor = GrowthScore()
df["growth_score"] = (
    df["营业总收入-同比增长"] * 0.4 +
    df["净利润-同比增长"] * 0.6
)

# 4. 计算行业因子
industry_factor = IndustryFactor()
df["industry_code"] = industry_factor.compute(df["所处行业"])

# 5. 因子合成
composite = FactorComposite(
    factors=[PERatio(), ROE(), GrowthScore()],
    weights=[0.3, 0.4, 0.3],
    winsorize=True,
    standardize=True
)
df["composite_factor"] = composite.compute(df)
```

---

## 五、数据保存与加载

```python
# 保存数据
financial_df.to_parquet("./data/financial/financial_20241231.parquet")
balance_df.to_parquet("./data/financial/balance_20241231.parquet")
industry_df.to_parquet("./data/financial/industry_20241231.parquet")

# 加载数据
financial_df = pd.read_parquet("./data/financial/financial_20241231.parquet")
balance_df = pd.read_parquet("./data/financial/balance_20241231.parquet")
industry_df = pd.read_parquet("./data/financial/industry_20241231.parquet")
```

---

## 六、注意事项

1. **报告期格式**：使用 `YYYYMMDD` 格式，如 `20241231`、`20240930`、`20240630`、`20240331`

2. **股票代码格式**：
   - 业绩报表：纯数字（如 `600000`）
   - 单只股票报表：带交易所前缀（如 `SH600000`、`SZ000001`）

3. **数据更新频率**：
   - 财务数据：每季度更新（3/6/9/12月）
   - 行业数据：随财务数据更新
   - 指数数据：每日更新

4. **网络问题**：如遇到连接问题，可重试或使用缓存

---

## 七、已验证可用的接口

| 接口 | 状态 | 说明 |
|------|------|------|
| `stock_yjbb_em()` | ✅ 可用 | 业绩报表（推荐） |
| `stock_zcfz_em()` | ✅ 可用 | 资产负债表 |
| `stock_profit_sheet_by_yearly_em()` | ✅ 可用 | 单只股票利润表 |
| `stock_balance_sheet_by_report_em()` | ✅ 可用 | 单只股票资产负债表 |
| `stock_cash_flow_sheet_by_quarterly_em()` | ✅ 可用 | 单只股票现金流量表 |
| `index_zh_a_hist()` | ⚠️ 部分可用 | 指数数据（可能需要重试） |

---

## 八、下一步建议

1. **下载历史财务数据**：获取多个报告期的数据，计算同比增长率
2. **建立数据更新机制**：定期更新财务数据和行业分类
3. **数据质量控制**：检查数据缺失值、异常值
4. **因子数据库**：将计算好的因子保存到数据库，便于回测使用