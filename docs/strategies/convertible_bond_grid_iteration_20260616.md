# 可转债低位网格迭代记录

## 目标

在原双低轮动基础上，提高建仓门槛，允许长时间空仓，只在较确定的低位逐步建仓和加仓。研究区间从 2015 年以后开始，重点观察 2024 年以来的表现。

## 新增策略逻辑

新增模块：

- `src/quant/strategies/convertible_bond/grid.py`
- `scripts/research/iterate_convertible_bond_grid.py`

核心约束：

- 价格低位：`close <= max_entry_price`
- 溢价低位：`min_premium_rate <= premium_rate <= max_premium_rate`
- 双低值低位：`close + premium_rate <= max_double_low`
- 历史低位：`price_position_252 <= max_price_position_252`
- 高位回撤：`drawdown_from_252_high <= -min_drawdown_from_252_high`
- 企稳过滤：`momentum_20d >= min_momentum_20d`
- 风险过滤：排除强赎、未转股、评级不足、成交额不足、剩余规模不足

网格仓位：

- `close <= 106`: 满档
- `106 < close <= 110`: 0.75 档
- `110 < close <= 114`: 0.50 档
- `114 < close <= 118`: 0.25 档
- 深度低位和低溢价可小幅加档，但总仓位受 `max_total_weight` 限制

## 关键迭代

第一轮只做低位过滤和网格仓位，发现 2024 年 6-8 月坏日子集中，组合中出现较多负溢价、连续急跌标的。结论是：低位不等于确定性，需要加入“不要接急跌刀口”的企稳条件。

第二轮加入：

- `min_premium_rate = 0.0`
- `min_momentum_20d = -0.10` 或 `-0.08`

## 推荐候选

### grid_balanced_stabilized

参数：

- `top_n = 8`
- `max_total_weight = 0.75`
- `max_position_weight = 0.12`
- `max_entry_price = 116`
- `min_premium_rate = 0`
- `max_premium_rate = 24`
- `max_double_low = 138`
- `max_price_position_252 = 0.32`
- `min_drawdown_from_252_high = 0.08`
- `min_amount = 3000`
- `min_momentum_20d = -0.10`

2015-2026 周频回测：

- 总收益：7.46%
- 年化收益：0.66%
- 最大回撤：-9.46%
- Sharpe：0.24
- 平均仓位：15.80%
- 有持仓天数：1035

2024-2026 日频回测：

- 总收益：7.46%
- 年化收益：3.23%
- 最大回撤：-7.62%
- Sharpe：0.67
- 平均仓位：20.54%
- 有持仓天数：360

### grid_strict_stabilized

更保守，适合作为防守仓：

- 2024-2026 日频总收益：3.84%
- 最大回撤：-5.25%
- 平均仓位：13.91%

## 简单模型诊断

样本：2015-2026 可转债日截面，预测未来 20 日收益是否为正。

- Logistic recent AUC：0.5504
- 低价格、低溢价、低双低值为弱正向信号
- 20 日动量系数略正，支持“低位后企稳”过滤

模型信号偏弱，不建议直接机器学习下单。更合理的方向是把模型作为过滤/排序辅助，而不是替代规则。

## 当前结论

- 原始双低轮动收益更高，但回撤和换手更大。
- 低位网格能显著降低暴露和回撤，但长期收益会被空仓拖低。
- 近两三年，`balanced_stabilized` 比过度严格的 deep value 更有效。
- 下一步应重点研究市场状态过滤：当全市场可转债中位数或转债指数仍在快速下行时，即使个券很低，也降低仓位或延迟一到两周。

## 输出文件

- `reports/convertible_bond/grid_iteration/iteration_summary.csv`
- `reports/convertible_bond/grid_iteration/simple_model_diagnostics.json`
- `reports/convertible_bond/grid_iteration/grid_balanced_stabilized_20240101_20260616_daily/summary.json`
- `reports/convertible_bond/grid_iteration/grid_balanced_stabilized_20150101_20260616_weekly/summary.json`
