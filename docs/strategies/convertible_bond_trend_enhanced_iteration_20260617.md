# 可转债趋势增强策略迭代记录（2026-06-17）

## 策略来源

参考用户提供的“全市场可转债趋势增强策略”描述，但不复刻其中不可验证的私有指标。当前实现使用 Tushare 可转债日线可复现字段构造：

- 均线系统：`ma_3/5/10/15/20`
- 趋势强度：四段均线多头排列，每段 25 分，满分 100
- 短期动量：`return_5d`、`return_1d`
- 六脉神剑替代信号：站上 5 日线、均线多头、5 日收益、1 日收益等 6 个二值信号
- 波段状态：用 5/20 日均线、短期涨幅和 60 日价格位置判断 `买/卖/观望`
- 市场闸门：全市场双低中位数、20 日市场趋势、趋势广度

这条路线刻意区别于双低轮动：双低只把 `close + premium_rate` 作为核心估值排序，趋势增强以技术趋势和市场温度为主，双低只作为风险背景指标。

## 当前推荐参数

研究脚本中的 `trend_v6 weekly` 是当前相对稳健版本：

- 周频调仓
- `top_n = 12`
- `max_position_weight = 0.06`
- `min_trend_strength = 75`
- `0.5 <= return_5d <= 10`
- `-2 <= return_1d <= 3.5`
- `six_sword_daily >= 4`
- `price_position_60d <= 0.78`
- `market_median_double_low <= 138`
- `market_trend_20d >= -0.01`
- `market_trend_breadth >= 0.20`
- 个券过滤：`close <= 135`、`premium_rate <= 28`、成交额不低于 5000、评级 AA- 以上、排除强赎和未转股期

## 第一轮回测与 case 分析

第一轮直接按贴文逻辑构造趋势策略，结果偏弱：

- 2018 起点最佳：`trend_v4 weekly`，总收益 3.75%，年化 0.46%，最大回撤 -9.13%
- 2020 起点最佳仍为负：`trend_v4 weekly`，总收益 -4.58%，最大回撤 -12.01%
- 2024 起点最佳仍为负：`trend_v4 weekly`，总收益 -2.08%，最大回撤 -9.79%
- 日频版本明显受换手拖累，2024 起点部分组合平均换手超过 0.4

case 分析显示，差收益主要来自两类风险：

- 市场整体回落：例如 2024-10-09 全市场平均 1 日收益约 -4.28%，但策略仍满仓 20 只。
- 高位追涨：worst-day target 的 `price_position_60d` 中位数约 0.92，很多候选债已经处在 60 日高位附近。

## 第二轮调整

第二轮加入市场闸门和个券高位限制后，回撤明显下降：

| 起点 | 最佳版本 | 总收益 | 年化 | Sharpe | 最大回撤 | 平均换手 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2018-01-30 | trend_v6 weekly | 3.02% | 0.37% | 0.237 | -3.43% | 0.0068 |
| 2020-02-07 | trend_v6 weekly | 1.08% | 0.18% | 0.115 | -3.15% | 0.0092 |
| 2024-01-30 | trend_v2 daily | 1.35% | 0.59% | 0.125 | -5.32% | 0.1502 |
| 2024-01-30 | trend_v6 weekly | 0.53% | 0.23% | 0.093 | -3.15% | 0.0229 |

结论：市场闸门有效降低了回撤，但也显著降低暴露，导致收益偏平。当前策略更像低回撤趋势参与器，不适合作为追求高收益的独立主策略。

## 文件与复跑命令

- 策略实现：`src/quant/strategies/convertible_bond/trend_enhanced.py`
- 回测入口：`backtest_convertible_bond_trend_enhanced`
- 配置文件：`configs/strategies/convertible_bond_trend_enhanced.yaml`
- 迭代脚本：`scripts/research/iterate_convertible_bond_trend_enhanced.py`
- 汇总结果：`reports/convertible_bond/trend_enhanced_iteration/iteration_summary.csv`

复跑：

```bash
python scripts/research/iterate_convertible_bond_trend_enhanced.py
```

## 后续方向

如果继续优化，不建议继续简单放宽趋势追涨条件。更有价值的方向是：

- 引入持仓延续逻辑：买入条件严格、卖出条件钝化，减少信号抖动。
- 加入可转债指数或等权组合的市场择时，替代当前双低中位数闸门。
- 将趋势增强作为双低/网格策略的仓位加减模块，而不是独立选债主策略。
