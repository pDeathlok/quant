# 可转债双低轮动策略

## 目标

构建一套基于 Tushare 的可转债每日选债和调仓流程，优先追求：

- 避开明显强赎、低评级、过小规模和流动性不足标的。
- 在可交易池内选择价格和转股溢价率都不高的“双低”转债。
- 输出目标组合权重和相对当前持仓的买卖权重差，便于后续接回测或券商交易。

## 数据源

- `cb_basic`: 债券基础信息、正股、剩余规模、评级、转股起始日、条款。
- `cb_daily`: 日行情、成交额、转股溢价率相关字段。
- `cb_call`: 赎回/强赎公告，用于风险过滤。
- `cb_share`: 转股结果和剩余规模跟踪，已在数据层封装，后续可用于规模更新校验。

## 策略流程

1. 合并当日 `cb_daily` 和 `cb_basic`，补充强赎风险标记。
2. 过滤价格、转股溢价率、成交额、剩余规模、评级、强赎风险、尚未进入转股期的标的。
3. 计算 `double_low = close + premium_rate`。
4. 按双低、流动性、剩余规模、短期动量加权评分。
5. 取前 `top_n`，按等权且不超过 `max_position_weight` 生成目标组合。
6. 和当前持仓权重比较，超过 `rebalance_threshold` 输出调仓指令。

## 默认参数

参数文件位于 `configs/strategies/convertible_bond_rotation.yaml`：

- 价格区间：100 到 135。
- 转股溢价率上限：35%。
- 成交额下限：1000。
- 剩余规模下限：1。
- 最低评级：AA-。
- 持仓数量：10。
- 单债权重上限：12%。

## 运行

设置 `TUSHARE_TOKEN` 后生成指定交易日计划：

```bash
python -m quant.routine.cli cb-plan --trade-date 20260616
```

输出：

- `data/routine/convertible_bond_plan_YYYYMMDD.json`
- `data/routine/convertible_bond_targets_YYYYMMDD.csv`

## 历史回测

使用 Tushare 尽量回溯可转债历史数据，并执行日频轮动回测：

```bash
python scripts/research/backtest_convertible_bond_rotation.py \
  --start-date 20180101 \
  --end-date 20260616 \
  --refresh
```

脚本会逐交易日拉取 `cb_daily`，同时缓存 `cb_basic` 和 `cb_call`：

- `data/convertible_bond/tushare/cb_basic_all.parquet`
- `data/convertible_bond/tushare/cb_daily_START_END.parquet`
- `data/convertible_bond/tushare/cb_call_START_END.parquet`

回测输出：

- `reports/convertible_bond/rotation/START_END_daily/summary.json`
- `reports/convertible_bond/rotation/START_END_daily/equity.csv`
- `reports/convertible_bond/rotation/START_END_daily/targets.csv`
- `reports/convertible_bond/rotation/START_END_daily/trades.csv`

回测假设：

- 当日收盘后形成信号，下一交易日按 close-to-close 收益计入组合。
- 每次调仓扣除手续费和滑点，默认各 2bp。
- 强赎公告在公告日之后持续作为风险剔除。
- `cb_basic` 中评级和剩余规模是 Tushare 当前/静态字段，不是完整 point-in-time 基本面。
