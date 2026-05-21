# AKShare 速查手册

## 一、核心接口速查

### 1. 股票行情数据

| 接口 | 说明 | 参数 | 返回字段 |
|------|------|------|---------|
| `stock_zh_a_hist()` | A股历史行情 | `symbol`, `start_date`, `end_date`, `adjust` | 日期, 开盘, 最高, 最低, 收盘, 成交量, 成交额, 换手率 |

**示例：**
```python
df = ak.stock_zh_a_hist(
    symbol="600000",
    start_date="20230101",
    end_date="20231231",
    adjust="qfq"  # 前复权
)
```

---

### 2. 财务报表数据

#### 业绩报表（推荐）

| 接口 | 说明 | 参数 | 返回字段 |
|------|------|------|---------|
| `stock_yjbb_em()` | 业绩报表 | `date` (格式: YYYYMMDD) | 股票代码, 股票简称, 每股收益, 营业总收入, 净利润, 每股净资产, 净资产收益率, 每股经营现金流量, 销售毛利率, 所处行业 |

**示例：**
```python
df = ak.stock_yjbb_em(date="20241231")
```

**关键字段：**
- `每股收益` → PE 因子
- `每股净资产` → PB 因子
- `净资产收益率` → ROE 因子
- `营业总收入-同比增长` → 成长因子
- `净利润-同比增长` → 成长因子
- `销售毛利率` → 质量因子
- `所处行业` → 行业因子

---

#### 资产负债表

| 接口 | 说明 | 参数 | 返回字段 |
|------|------|------|---------|
| `stock_zcfz_em()` | 资产负债表 | `date` (格式: YYYYMMDD) | 股票代码, 股票简称, 资产-总资产, 负债-总负债, 资产负债率, 股东权益合计 |

**示例：**
```python
df = ak.stock_zcfz_em(date="20241231")
```

**关键字段：**
- `资产-总资产` → ROA 因子
- `负债-总负债` → 杠杆因子
- `资产负债率` → 杠杆因子
- `股东权益合计` → 质量因子

---

#### 单只股票财务报表

| 接口 | 说明 | 参数 | 返回字段 |
|------|------|------|---------|
| `stock_profit_sheet_by_yearly_em()` | 利润表（按年度） | `symbol` (格式: SH600000) | 多列财务指标 |
| `stock_balance_sheet_by_report_em()` | 资产负债表（按报告期） | `symbol` (格式: SH600000) | 多列财务指标 |
| `stock_cash_flow_sheet_by_quarterly_em()` | 现金流量表（按季度） | `symbol` (格式: SH600000) | 多列财务指标 |

**示例：**
```python
df = ak.stock_profit_sheet_by_yearly_em(symbol="SH600000")
df = ak.stock_balance_sheet_by_report_em(symbol="SH600000")
df = ak.stock_cash_flow_sheet_by_quarterly_em(symbol="SH600000")
```

---

### 3. 指数数据

| 接口 | 说明 | 参数 | 返回字段 |
|------|------|------|---------|
| `index_zh_a_hist()` | 指数历史行情 | `symbol`, `start_date`, `end_date` | 日期, 开盘, 最高, 最低, 收盘, 成交量, 成交额 |

**常用指数代码：**
- `sh000300` - 沪深300
- `sh000001` - 上证指数
- `sz399001` - 深证成指
- `sz399006` - 创业板指

**示例：**
```python
df = ak.index_zh_a_hist(
    symbol="sh000300",
    start_date="20230101",
    end_date="20231231"
)
df["return"] = df["收盘"].pct_change()  # 计算收益率
```

---

### 4. 行业分类数据

| 接口 | 说明 | 参数 | 返回字段 |
|------|------|------|---------|
| `stock_yjbb_em()` | 从业绩报表获取 | `date` | 股票代码, 股票简称, 所处行业 |

**示例：**
```python
df = ak.stock_yjbb_em(date="20241231")
industry_df = df[["股票代码", "股票简称", "所处行业"]]
```

---

## 二、因子数据映射表

### 价值因子

| 因子 | 所需数据 | AKShare 接口 | 字段 |
|------|---------|-------------|------|
| PE | 每股收益 / 股价 | `stock_yjbb_em()` | `每股收益` |
| PB | 每股净资产 / 股价 | `stock_yjbb_em()` | `每股净资产` |
| PS | 营业收入 / 市值 | `stock_yjbb_em()` | `营业总收入-营业总收入` |

### 质量因子

| 因子 | 所需数据 | AKShare 接口 | 字段 |
|------|---------|-------------|------|
| ROE | 净利润 / 净资产 | `stock_yjbb_em()` | `净资产收益率` |
| ROA | 净利润 / 总资产 | `stock_zcfz_em()` | `资产-总资产` |
| 毛利率 | 毛利 / 营业收入 | `stock_yjbb_em()` | `销售毛利率` |

### 成长因子

| 因子 | 所需数据 | AKShare 接口 | 字段 |
|------|---------|-------------|------|
| 营收增长 | 营收同比增长率 | `stock_yjbb_em()` | `营业总收入-同比增长` |
| 利润增长 | 净利润同比增长率 | `stock_yjbb_em()` | `净利润-同比增长` |
| 成长评分 | 综合评分 | `stock_yjbb_em()` | 多字段组合 |

### 杠杆因子

| 因子 | 所需数据 | AKShare 接口 | 字段 |
|------|---------|-------------|------|
| 资产负债率 | 总负债 / 总资产 | `stock_zcfz_em()` | `资产负债率` |
| 权益乘数 | 总资产 / 股东权益 | `stock_zcfz_em()` | `股东权益合计` |

### 行业因子

| 因子 | 所需数据 | AKShare 接口 | 字段 |
|------|---------|-------------|------|
| 行业编码 | 行业分类 | `stock_yjbb_em()` | `所处行业` |
| 行业虚拟变量 | 行业分类 | `stock_yjbb_em()` | `所处行业` |

---

## 三、常用代码片段

### 1. 获取财务数据

```python
import akshare as ak

# 获取最新财报
df = ak.stock_yjbb_em(date="20241231")

# 筛选特定股票
stock_data = df[df["股票代码"] == "600000"]

# 计算价值因子
stock_data["pe_ratio"] = stock_data["每股收益"] / stock_data["收盘价"]
```

### 2. 获取历史财务数据

```python
# 获取多个报告期数据
dates = ["20241231", "20240930", "20240630", "20240331"]
all_data = []

for date in dates:
    df = ak.stock_yjbb_em(date=date)
    df["report_date"] = date
    all_data.append(df)

historical_df = pd.concat(all_data, ignore_index=True)
```

### 3. 计算因子

```python
# 价值因子
df["pe_ratio"] = df["每股收益"] / df["收盘价"]
df["pb_ratio"] = df["每股净资产"] / df["收盘价"]

# 质量因子
df["roe"] = df["净资产收益率"]
df["gross_margin"] = df["销售毛利率"]

# 成长因子
df["revenue_growth"] = df["营业总收入-同比增长"]
df["profit_growth"] = df["净利润-同比增长"]

# 杠杆因子
df["debt_ratio"] = df["资产负债率"]
```

### 4. 行业中性化

```python
# 获取行业数据
industry_df = df[["股票代码", "所处行业"]]

# 行业编码
industry_map = {ind: i for i, ind in enumerate(df["所处行业"].unique())}
df["industry_code"] = df["所处行业"].map(industry_map)

# 行业中性化（示例）
from sklearn.linear_model import LinearRegression

X = df[["industry_code"]].values.reshape(-1, 1)
y = df["pe_ratio"].values

model = LinearRegression()
model.fit(X, y)
df["pe_ratio_neutralized"] = y - model.predict(X)
```

---

## 四、参数说明

### 报告期格式

| 格式 | 说明 | 示例 |
|------|------|------|
| `YYYYMMDD` | 年月日 | `20241231` (年报), `20240930` (三季报) |

### 股票代码格式

| 格式 | 说明 | 示例 |
|------|------|------|
| 纯数字 | 用于批量查询 | `600000` |
| 带前缀 | 用于单只股票查询 | `SH600000`, `SZ000001` |

### 复权类型

| 类型 | 说明 |
|------|------|
| `qfq` | 前复权（推荐） |
| `hfq` | 后复权 |
| `""` | 不复权 |

---

## 五、注意事项

1. **数据更新频率**
   - 财务数据：每季度更新（3/6/9/12月）
   - 行业数据：随财务数据更新
   - 指数数据：每日更新

2. **数据单位**
   - 金额单位：元
   - 比率单位：%
   - 股本单位：股

3. **网络问题**
   - 如遇连接问题，可重试
   - 建议使用缓存机制

4. **数据质量**
   - 检查缺失值
   - 处理异常值
   - 验证数据合理性

---

## 六、快速查找索引

### 按数据类型查找

- [股票行情](#1-股票行情数据)
- [业绩报表](#业绩报表推荐)
- [资产负债表](#资产负债表)
- [指数数据](#3-指数数据)
- [行业分类](#4-行业分类数据)

### 按因子类型查找

- [价值因子](#价值因子)
- [质量因子](#质量因子)
- [成长因子](#成长因子)
- [杠杆因子](#杠杆因子)
- [行业因子](#行业因子)

### 按接口名称查找

- [stock_yjbb_em](#业绩报表推荐)
- [stock_zcfz_em](#资产负债表)
- [stock_zh_a_hist](#1-股票行情数据)
- [index_zh_a_hist](#3-指数数据)

---

## 七、相关文档

- [AKShare 官方文档](https://akshare.akfamily.xyz/)
- [项目数据获取指南](./akshare_data_guide.md)
- [因子库文档](../data/factors/README.md)