# 月线带血筹连续建仓、结构止损与再入场验证计划

## Goal

在当前 `range_mid_reclaim + 前高回撤至少60%` 的同一批月线低9锚上，检验固定总预算下的网格分批、跌破底部区间退出后再确认接回，以及二者组合，能否真实改善旧周期尾部、独立恐慌月可靠性和日度组合最大回撤，而不是仅靠少投入提高单笔胜率。

## Pre-conditions

- [x] `/Users/didi/Project/quant/reports/research/monthly_low_zone_strict_extension/combined_candidate_events.parquet` 存在，包含2003—2024的 `range_mid_reclaim` 完整路径。
- [x] `/Users/didi/Project/quant/data/research/monthly_low_zone_strict_extension/features_2000_2015.parquet` 存在，闭合旧周期504个交易日持有路径。
- [x] `/Users/didi/Project/quant/data/research/blood_chip_deep_base/features.parquet` 与退市补充历史存在，可闭合近周期路径至2026-07-31。
- [x] 事件主口径固定为市场正收益占比不高于20%、市场20日中位收益不高于-10%、前高回撤至少60%、15%止盈、504个市场交易日、往返20bp。

## Claim boundary

- 2003—2024均已暴露，本轮只判断历史机制是否改善，不构成样本外发现；`deployment_eligible` 固定为 `false`。
- 同一恐慌月股票高度相关，主统计单位仍是 `month_period`。股票事件和每一次再入场都不能冒充独立样本。
- 所有政策使用相同的每锚总预算。未投入的网格资金计为现金；必须报告“总预算收益”，不能只报告实际买入资金收益。
- 退市、长期停牌和缺失行情沿用原事件核销，不从网格或再入场样本中删除。

## Frozen state machine

### Common execution

1. 初始信号使用既有 `range_mid_reclaim`，下一只股票实际交易日开盘入场。
2. 每锚内部归一化预算为1；组合层每锚占入场前净值2.5%，最多20个活动锚，同一股票不重叠。
3. 买入限价在当日复权低价触及时成交；若开盘已低于限价，仍按预设限价成交，作为对买方更保守的价格。单价板不假定成交。
4. 15%止盈以当前全部持仓的复权加权平均成本为基准。发生新网格成交的同一天不允许止盈，避免用未知的日内高低顺序获得乐观成交；其他日期首次触及目标价时全部卖出。
5. 结构止损以每一轮首次入场信号日可见的 `base_low` 固定，不随后续下跌下移。收盘严格跌破该位置后，下一可交易日开盘全部卖出；卖出信号日不能同日再入。
6. 每次完整买卖按投入资金扣除20bp往返成本。504个市场交易日从最初入场起算，再入场不重置总期限。

### Frozen policies

- `lump_sum`：初始投入100%，不设结构止损或再入场；用于复现现有一次建仓基准。
- `grid_40_30_30_down10_down20`：初始40%，相对本轮首次入场价下跌10%再投入30%，下跌20%再投入剩余30%；不止损、不再入场。
- `grid_40_30_30_down5_down10` 与 `grid_40_30_30_down15_down30`：只作网格间距邻域，不能按结果替换10%/20%主值。
- `base_stop_reentry_2`：初始投入100%；跌破固定 `base_low` 后全部退出。至少等待20个市场交易日，且当日连续至少20日未创新低、20日收益为正、`base_position>=0.5`、20日成交额中位数不低于3,000万元、相对原锚前高仍回撤至少40%，才在下一开盘重新投入100%；最多接回2次。
- `grid_base_stop_reentry_2`：每一轮使用10%/20%的40%/30%/30%网格；跌破该轮固定 `base_low` 后全部退出，按相同条件最多接回2次。新一轮网格以接回开盘价重新锚定。

## Frozen metrics and decision

- 事件层：完成锚数、总预算胜率、总预算PF、平均/中位收益、平均投入比例、止盈率、止损率、再入场率、平均交易轮数、尾部损失率。
- 批次层：锚月等权收益、正锚月占比、10,000次锚月bootstrap区间、最差锚月、逐一删除锚月后的最小均值。
- 组合层：2003—2015、2013—2016、2017—2020、2021—2024分别重启100万元，报告CAGR、最大回撤、最差滚动24个月收益、平均实际股票仓位和资金利用率。
- case：全部核销；所有先止损后仍再接失败；止损避免的永久损失；止损后未接回但原策略最终止盈；网格满仓后亏损超过50%的事件。

连续政策只有在以下条件全部满足时标记 `historical_path_management_increment`：

1. 旧周期至少4个完成锚月，事件胜率至少85%、PF至少2、正锚月占比至少80%、bootstrap下界大于0、最差锚月不低于-5%。
2. 合并样本bootstrap下界和留一锚月最小均值均大于0，且都高于 `lump_sum`。
3. 旧周期和三个近周期组合的最大回撤均不低于-10%，并且每段CAGR为正。
4. 相比 `lump_sum`，旧周期最大回撤、最差锚月和尾部损失率均改善；不能只通过降低平均投入比例通过。
5. 三档网格间距中至少两档的旧周期和合并bootstrap下界均为正；否则网格判定脆弱。

若所有政策失败，不再搜索网格间距、止损缓冲或再入场等待天数；结论为价格路径管理不能把当前带血筹结构提升到高确定性。

## Steps

### Step 1 — 实现单锚连续状态机

**File:** `/Users/didi/Project/quant/src/quant/research/monthly_low_zone_continuous.py`

新增不可变配置、五个冻结政策、因果逐日状态机、逐锚日度净值与交易流水。状态必须显式区分 `waiting_initial`、`holding`、`stop_pending`、`waiting_reentry`、`completed`，并将未投入预算保留为现金。

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone_continuous.py` → 网格成交、同日顺序、固定结构止损、再入场等待、最大再入次数、永久核销和基准复现测试全部通过。

### Step 2 — 实现批次统计与组合模拟

**File:** `/Users/didi/Project/quant/src/quant/research/monthly_low_zone_continuous.py`

按 `month_period` 聚类事件，复用固定种子的锚月bootstrap；组合按入场日成交额降序处理容量，同一股票持仓重叠时拒绝新锚，每日按单锚归一化路径盯市。

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone_continuous.py` → 组合容量、内部现金、退出释放资金和实际股票仓位测试全部通过。

### Step 3 — 运行旧周期与近周期回验

**File:** `/Users/didi/Project/quant/scripts/research/backtest_monthly_low_zone_continuous.py`

读取已冻结的60%回撤主事件，只为相关股票载入日线；分别生成旧周期和近周期状态路径、事件指标、批次指标、日度组合、交易流水、case和判定，写入 `/Users/didi/Project/quant/reports/research/monthly_low_zone_continuous/`。

**Verify:** `PYTHONPATH=src python scripts/research/backtest_monthly_low_zone_continuous.py` → 生成 `report.md`、`decision.json`、`event_metrics.csv`、`cohort_metrics.csv`、`portfolio_metrics.csv`、`anchor_results.parquet`、`anchor_paths.parquet`、`trades.parquet` 与 `cases.parquet`。

### Step 4 — 回归与静态验证

**Files:** `/Users/didi/Project/quant/tests/research/test_monthly_low_zone_continuous.py`, `/Users/didi/Project/quant/tests/research/test_monthly_low_zone_profit_lock.py`

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone_continuous.py tests/research/test_monthly_low_zone_profit_lock.py tests/research/test_monthly_low_zone_strict.py
PYTHONPATH=src python -m compileall -q src/quant/research/monthly_low_zone_continuous.py scripts/research/backtest_monthly_low_zone_continuous.py tests/research/test_monthly_low_zone_continuous.py
git diff --check
```

预期：全部测试通过、编译无错误、`git diff --check`无输出。

## Rollback

本轮不修改线上策略、数据库或既有研究报告。撤回只删除本计划、新模块、新测试、新脚本及忽略目录 `/Users/didi/Project/quant/reports/research/monthly_low_zone_continuous/`；不得删除既有缓存、旧报告或用户文件。
