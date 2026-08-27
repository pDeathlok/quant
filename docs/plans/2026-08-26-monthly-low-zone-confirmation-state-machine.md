# 月线低位到右侧确认状态机持续验证计划

## Goal

把月线低9从直接买点改成低位状态锚，在最多126个市场交易日内等待“不再创新低、区间中轴收复、周线J修复、相对强度转正和市场修复”的因果确认；先验证确认层能否在跨阶段、按信号日期聚类后稳定提高12个月胜率，再决定是否进入分批组合。

## Pre-conditions

- [x] `/Users/didi/Project/quant/data/research/monthly_low_zone/monthly_features.parquet` 与 `weekly_features.parquet` 已按完成月线和完成 `W-FRI` 周线生成，研究截止日为 `2026-07-31`。
- [x] `/Users/didi/Project/quant/reports/research/monthly_low_zone/signals.parquet` 含前高回撤至少50%、36根完成月线、月成交额中位数至少3,000万元和12个月冷却后的月线低9锚。
- [x] `/Users/didi/Project/quant/data/research/blood_chip_deep_base/features.parquet` 含正式日线的因果连续价格、60日区间、20日不创新低、卖压份额、波动收缩和20日收益。
- [x] `/Users/didi/Project/quant/data/research/low9_kdj_rebound/supplemental_daily/` 的257只退市补充证券将用 `build_deep_base_features` 重新计算相同日线特征，避免只保留存续公司。
- [x] 当前四个未跟踪文件均属于上一轮月线研究；本轮不得覆盖或删除它们。

## Fixed research contract

### Anchor and confirmation window

- 主锚只使用上一轮统计样本充分且综合最优的 `monthly_low9`；`monthly_j_le_minus10` 仅作锚类型消融，不参与主结构选择。
- 锚在完整月末收盘后可见。确认搜索从锚后第5个市场交易日开始，到第126个市场交易日结束；超过窗口未确认记为过期，不把未来信息回填到锚。
- 每个锚每条规则只接受第一个确认日；确认收盘后在下一只股票可交易日开盘进入，沿用一字价格、最长5日入场延迟、停牌复牌和60日无行情核销口径。
- 所有候选保持锚时前高回撤至少50%，确认日成交额前20日中位数至少3,000万元，确认日仍较锚前高回撤至少40%，防止把已经大幅反弹的股票误称低位区间。

### Pre-registered confirmation rules

按以下固定次序做增量消融，不在查看验证结果后新增阈值：

1. `anchor_direct`：月线低9月末直接确认，作为上一轮基准。
2. `no_new_low_20`：确认日距最近60日新低至少20个股票交易日，且20日收益大于0。
3. `range_mid_reclaim`：规则2，且收盘位于当日60日高低区间中轴以上。
4. `range_mid_relative`：规则3，且股票20日收益高于上证综指同期20日收益。
5. `range_mid_weekly`：规则3，且最近已完成周线 `J>-10`、本周J高于前周J。
6. `range_mid_relative_weekly`：同时满足规则4和规则5。
7. `confirmed_market`：规则6，且上证综指收盘不低于120日均线、20日收益大于0。
8. `confirmed_market_exhaustion`：规则7，且20日下跌成交份额不高于此前60日、20日波动不高于此前60日。

周线状态只允许 `weekly_available_date <= confirmation_date`；月末位于周中时不得读取未完成周线。市场20日收益和120日均线只使用确认日及以前的指数收盘。

### Event outcomes and robustness

- 主选择期限为252个市场交易日；同时报告126和504日结果，收益扣20bp往返成本。
- 开发期 `2013-01-01..2016-12-31`、验证期 `2017-01-01..2020-12-31`、已见诊断期 `2021-01-01..2024-12-31`；2021年以后不参与规则选择。
- 每条规则报告锚数、确认数、确认率、确认等待日、等待期最深跌幅、完成事件、独立信号日期、胜率、中位收益、PF、超额胜率、MAE/MFE、日期等权收益及按日期聚类95%区间。
- 合理信号结构必须在开发和验证期各有至少100个完成事件、至少24个独立确认日期，并分别满足：胜率不低于55%、中位净收益大于0、PF不低于1.50、超额胜率不低于50%、日期聚类95%收益下界大于0。
- 通过开发和验证的候选还必须在已见诊断期胜率不低于50%、中位净收益大于0、PF不低于1.20，且参数邻域 `不创新低15/20/30日` 中至少两个版本保持正中位收益；否则只保留研究线索。
- 若没有信号层候选通过，不构建分批组合；继续下一轮机制迭代时必须先写新的冻结计划并披露验证期已被打开。

## Steps

### Step 1 — 用合成路径锁定状态机因果边界

**File:** `/Users/didi/Project/quant/tests/research/test_monthly_low_zone_confirmation.py`

新增以下测试：

- `test_confirmation_never_precedes_anchor_or_minimum_wait`
- `test_confirmation_uses_first_eligible_day_and_expires_after_126_sessions`
- `test_weekly_state_cannot_use_incomplete_week`
- `test_relative_strength_uses_only_same_day_known_benchmark_history`
- `test_confirmation_preserves_anchor_peak_and_limits_rebound_drawdown`
- `test_waiting_path_drawdown_is_measured_from_anchor_close`
- `test_anchor_direct_matches_existing_monthly_event_entry`
- `test_target_suspension_uses_recovery_open_through_existing_evaluator`

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone_confirmation.py` → `8 passed`。

### Step 2 — 构建确认状态与首个确认事件

**File:** `/Users/didi/Project/quant/src/quant/research/monthly_low_zone_confirmation.py`

新增不可变配置：

```python
@dataclass(frozen=True)
class MonthlyConfirmationConfig:
    minimum_wait_sessions: int = 5
    maximum_wait_sessions: int = 126
    minimum_sessions_since_new_low: int = 20
    minimum_confirmation_amount_thousand: float = 30_000.0
    maximum_confirmation_drawdown_from_anchor_peak: float = -0.40
    weekly_j_repair_threshold: float = -10.0
    benchmark_ma_sessions: int = 120
    round_trip_cost_bps: float = 20.0
    horizons: tuple[int, int, int] = (126, 252, 504)
```

公开函数固定为：

```python
def build_benchmark_confirmation_features(
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Return causal 20-session return and 120-session moving-average state."""


def generate_monthly_confirmation_signals(
    daily_features: pd.DataFrame,
    weekly_features: pd.DataFrame,
    monthly_anchors: pd.DataFrame,
    benchmark_features: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
    config: MonthlyConfirmationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return first confirmation per anchor/rule and anchor-level expiry diagnostics."""


def summarize_confirmation_events(
    events: pd.DataFrame,
    anchor_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Return period/rule/horizon metrics with signal-date clustered uncertainty."""
```

所有公开函数提供完整类型注解、输入列校验和无隐藏数据源的确定性输出。

### Step 3 — 读取全量存续与退市历史并生成研究产物

**File:** `/Users/didi/Project/quant/scripts/research/backtest_monthly_low_zone_confirmation.py`

脚本固定读取上述缓存，调用上一轮 `evaluate_monthly_low_zone_events` 统一执行，并输出：

```text
/Users/didi/Project/quant/reports/research/monthly_low_zone_confirmation/signals.parquet
/Users/didi/Project/quant/reports/research/monthly_low_zone_confirmation/anchor_diagnostics.parquet
/Users/didi/Project/quant/reports/research/monthly_low_zone_confirmation/events.parquet
/Users/didi/Project/quant/reports/research/monthly_low_zone_confirmation/metrics.csv
/Users/didi/Project/quant/reports/research/monthly_low_zone_confirmation/yearly_metrics.csv
/Users/didi/Project/quant/reports/research/monthly_low_zone_confirmation/case_catalog.parquet
/Users/didi/Project/quant/reports/research/monthly_low_zone_confirmation/case_summary.csv
/Users/didi/Project/quant/reports/research/monthly_low_zone_confirmation/decision.json
/Users/didi/Project/quant/reports/research/monthly_low_zone_confirmation/report.md
```

### Step 4 — 全量验证与选择

```bash
PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone_confirmation.py tests/research/test_monthly_low_zone.py tests/research/test_blood_chip_deep_base.py
PYTHONPATH=src python -m compileall -q src/quant/research/monthly_low_zone_confirmation.py scripts/research/backtest_monthly_low_zone_confirmation.py tests/research/test_monthly_low_zone_confirmation.py
PYTHONPATH=src python scripts/research/backtest_monthly_low_zone_confirmation.py
git diff --check
```

预期：所有因果测试通过，七个确认层与直接锚基准并列报告；仅当 `decision.json` 的 `selected_rule` 非空时进入分批组合研究。

### Step 5 — 用成败案例提出下一轮可证伪修正

**File:** `/Users/didi/Project/quant/scripts/research/backtest_monthly_low_zone_confirmation.py`

对252日主期限固定生成以下互斥或可重叠标签：

- `confirmed_winner`：已确认且净收益大于0。
- `confirmed_loser`：已确认且净收益不大于0。
- `confirmed_severe_adverse`：确认后MAE不高于-20%。
- `late_confirmation`：确认等待至少75个市场交易日。
- `expired_then_rebounded`：该规则过期，但直接锚252日净收益至少20%。
- `expired_avoided_loss`：该规则过期，且直接锚252日净收益不高于-20%。

案例目录必须保留证券、锚日、确认日、等待日、等待期跌幅、确认后MAE/MFE、252日净收益和超额收益。案例只用于解释机制和写下一轮冻结假设；当前轮不得根据个别案例回改阈值。

## Rollback

本轮不修改数据库、线上配置和既有策略。撤回时只删除本计划新增的模块、测试、脚本，以及被忽略的 `reports/research/monthly_low_zone_confirmation/`；不得删除上一轮月线研究文件或用户其他改动。

## Commit checkpoint

本轮不自动提交。若结构通过并经用户确认，建议提交：

```text
feat(research): validate monthly low-zone confirmation state machine
```
