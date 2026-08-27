# 月线低位区间的因果形态约束验证计划

## Goal

在已经冻结的月线低9、系统性恐慌与个股深跌框架上，检验“更深的前高回撤”以及可在当时识别的双底、头肩底，是否能改善旧周期尾部损失和独立恐慌月份的收益可靠性。交易更少可以接受，但不能用减少样本伪装成确定性提高。

## Claim boundary

- 2013—2024 已经反复暴露，只能用于机制比较；2003—2012 是此前新增的反证周期，但也已在上一轮查看，因此本轮结果仍是历史诊断，不是新的样本外证据。
- 股票事件在同一恐慌月高度相关。主判断按 `month_period` 聚类，股票胜率和 PF 仅作路径描述。
- 形态只能减少既有锚，不能创造新的独立恐慌月。即使筛后 100% 胜率，只要锚月太少，也不能升级为“高确定性”。
- 不改线上策略；`deployment_eligible` 固定为 `false`。

## Frozen anchors and execution

1. 月线锚仅用 `monthly_low9`。
2. 锚日点时全市场 20 日正收益股票占比不高于 20%，20 日收益中位数不高于 -10%，截面股票数至少 500。
3. 个股相对锚日前历史峰值回撤至少 60% 为主值；50%/70%/80% 只作稳定性邻域，不能按结果换主值。
4. 形态的第一个拐点最早可在锚前 126 个市场交易日出现，但突破确认不得早于锚日，且最晚在锚后 126 个市场交易日确认；这允许月线锚所在月份已经形成第一只脚，又不读取过久以前的任意图形。
5. 确认日必须同时满足 `range_mid_reclaim`：至少 20 个交易日不创新低、20 日收益为正、收盘位置站回因果底部区间中轴；确认日的 20 日成交额中位数不低于 3,000 万元，且确认价相对锚日前高仍至少回撤 40%。形态因此是现有最强价格确认上的新增约束，而不是替代确认。
6. 信号日之后下一可交易日开盘，最长持有 504 个交易日，10%/15%/20% 止盈邻域，主值 15%，往返成本 20bp。
7. 与当前最强价格确认 `range_mid_reclaim` 同口径比较；不重新搜索市场宽度、财务或退出参数。

## Causal pivot contract

- 局部低点或高点使用左右各 3 个股票实际交易日。位置 `t` 的拐点只有在 `t+3` 收盘后才可知，确认日不得早于该可见日。
- 窗口内缺失或并列极值不算唯一拐点，避免停牌和连续同价制造伪形态。
- 所有高低点、颈线和突破只使用当日及此前的前复权价格；不得用确认日后的路径修改拐点。

## Frozen bullish patterns

### Double bottom

- 两个已经可见的局部低点，间隔 20—90 个市场交易日。
- 两底价格差不超过较低底部的 8%。
- 两底之间的最高复权收盘价为颈线；从较低底部到颈线的反弹幅度至少 10%。
- 第二底出现后，复权收盘价首次达到颈线的 101% 才确认。
- 确认还需满足连续至少 20 个市场交易日未创新低且 20 日收益为正，防止短促 V 型噪声。

### Inverse head and shoulders

- 三个已经可见的局部低点依次为左肩、头、右肩；相邻低点间隔各 10—60 个市场交易日。
- 两肩价格差不超过较低肩部的 10%；头部至少比两肩均价低 8%。
- 左肩—头、头—右肩区间的最高复权收盘价分别为两个颈线峰值，取二者较高值作为保守水平颈线。
- 右肩后复权收盘价首次达到颈线的 101% 才确认，并同样要求至少 20 个市场交易日未创新低且 20 日收益为正。

## Bearish head-and-shoulders handling

“头肩顶”方向看跌，不作为低位建仓确认。形态确认发生在跌破颈线时，把它加入买入条件在经济含义上相反；若将来验证退出管理，应在独立计划中把它作为入场后的提前退出规则，并重新处理卖出滑点、未完成路径和与 15% 止盈的先后顺序。本轮仅记录这一方向性否决，不用错误方向的条件扩大搜索空间。

## Frozen comparisons

- 规则：`range_mid_reclaim`、`double_bottom_breakout`、`inverse_head_shoulders_breakout`。
- 回撤层：50%/60%/70%/80%，主值 60%。
- 止盈：10%/15%/20%，主值 15%。
- 时间：旧周期 2003—2012、近周期 2013—2024、合并 2003—2024。
- 每条规则报告锚数、转化率、完成事件数、独立锚月数、股票胜率、PF、锚月均值、正锚月比例、锚月 bootstrap 95% 下界、最差锚月、逐一删除锚月后的最小均值。

## Decision rules

形态只在以下条件全部成立时标记为 `historical_pattern_increment`：

1. 60%/15% 主值在旧周期至少 3 个独立锚月、合并至少 8 个独立锚月；不足则只能标记样本不足。
2. 旧周期股票胜率至少 85%、PF 至少 2、正锚月占比至少 80%、bootstrap 下界大于 0、最差锚月不低于 -5%。
3. 合并样本的 bootstrap 下界和留一锚月最小均值均大于 0。
4. 相比同锚定义下的 `range_mid_reclaim`，旧周期最差锚月和 bootstrap 下界均改善；不能只提高相关股票事件胜率。
5. 50%/60%/70%/80% 至少三档的合并 bootstrap 下界和留一锚月最小均值为正；否则判定回撤边界脆弱。
6. 被形态删除的亏损和赢家都列出，不允许只展示成功图形。

即使达到 `historical_pattern_increment`，也不代表达到原研究要求的高确定性，因为独立恐慌月总数没有增加。若两种形态都失败或样本不足，停止继续调整底部误差、间距和颈线缓冲；结论为技术形态没有提供可证明的跨周期增益。

## Implementation

- 新增 `src/quant/research/monthly_low_zone_patterns.py`：冻结配置、延迟可见的局部拐点、双底和头肩底确认。
- 新增 `tests/research/test_monthly_low_zone_patterns.py`：未来信息隔离、几何约束、颈线突破和过期测试。
- 新增 `scripts/research/backtest_monthly_low_zone_chart_patterns.py`：复用已有旧/近周期缓存，生成形态信号、事件、分层指标和案例。
- 输出到 `reports/research/monthly_low_zone_chart_patterns/`，不覆盖既有报告。

## Verification

```bash
PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone_patterns.py
PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone_strict.py tests/research/test_monthly_low_zone_profit_lock.py tests/research/test_monthly_low_zone_confirmation.py tests/research/test_monthly_low_zone.py
PYTHONPATH=src python -m compileall -q src/quant/research/monthly_low_zone_patterns.py scripts/research/backtest_monthly_low_zone_chart_patterns.py tests/research/test_monthly_low_zone_patterns.py
PYTHONPATH=src python scripts/research/backtest_monthly_low_zone_chart_patterns.py
git diff --check
```

## Rollback

本轮不改生产代码、数据库和线上策略。撤回只删除本计划列出的新模块、测试、脚本与对应忽略报告目录。
