# 活跃市值研究因子

本项目使用可复现的“活跃市值代理”，不宣称复刻指南针 0AMV 的专有公式。代理值以换手率估算近期仍在交易的流通股份比例，再乘流通市值；所有滚动计算仅使用当日及此前数据。

## 统一口径

设当日换手率的小数形式为 `h_t`，活跃股份比例为 `A_t`：

```text
A_t = h_t + (12 / 13) * (1 - h_t) * A_(t-1)
active_mv_t = float_mv_t * A_t
```

- `h_t` 限制在 `[0, 1]`；原始换手率字段以百分数输入。
- 缺失换手率时，当日结果为缺失，但不会清空此前状态。
- 个股 `circ_mv` 来自 Tushare `daily_basic`，原单位为万元，计算前统一换算为人民币元。
- 指数 `float_mv` 优先来自 Tushare `index_dailybasic`，按接口原始人民币元口径使用；不在该接口覆盖范围内的指数，使用当时已知的 `index_weight` 成分表聚合个股代理值。
- `*_ratio_prev20` 的分母是此前 20 个交易日均值，不包含当日，避免前视。

## 三层字段

### 个股层

每个 `(ts_code, trade_date)` 生成：

- `stock_active_share_ratio_13d_proxy`
- `stock_active_mv_proxy_cny`
- `stock_active_mv_log`
- `stock_active_mv_return_1d_pct`
- `stock_active_mv_return_5d_pct`

### 全市场层

先汇总当日全部个股活跃市值与流通市值，再生成：

- `market_active_mv_proxy_cny`、`market_active_mv_log`
- `market_active_mv_ratio_proxy`
- `market_active_mv_return_1d_pct`、`market_active_mv_return_5d_pct`
- `market_volume_ratio_prev20`、`market_amount_ratio_prev20`

这些字段以 `trade_date` 为键，可连接到同一交易日的每只股票样本。

### 关键指数层

当前固定覆盖上证综指、深证成指、创业板指、科创 50、上证 50、沪深 300、中证 500、中证 1000 和北证 50。每个指数生成：

- 1 日、5 日涨跌幅；
- 成交量、成交额相对此前 20 日均值；
- 活跃股份比例、活跃市值、活跃市值对数；
- 活跃市值 1 日、5 日变化率，以及成分数据覆盖率。

计算优先使用指数自身的 `float_mv` 和 `turnover_rate`。数据商未直接覆盖时，使用不晚于目标交易日的最新指数成分快照，汇总成分股活跃市值；覆盖权重低于 90% 时结果保留为空。两种来源都不可用时同样保留为空，绝不使用成交额冒充市值。

## 生命周期与使用限制

全部字段已登记到统一因子注册表，计算层为 `active_market_value_research`，schema 为 `active_market_value_proxy_v1_20260831`。当前角色为 `research_feature`、生命周期为 `research_candidate`，因此可用于左右侧、selector 和市场状态研究，但不会在未重训、未做样本外验证时静默进入现有生产模型。

原始活跃市值会受到上市公司数量、流通股本变更和指数成分调整影响。跨期建模优先使用活跃比例、对数和变化率；使用绝对值时，应同时控制总流通市值和样本覆盖率。

实现入口：`src/quant/features/active_market_value.py`。
