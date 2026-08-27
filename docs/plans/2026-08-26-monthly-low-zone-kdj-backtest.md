# 月线低9与月周KDJ低位区间回测计划

## Goal

用因果连续价格在完整月线上识别前高腰斩后的月线低9、月线J深度负值及月周J共振，先以不依赖止损和分批技巧的下一开盘事件研究判断其12—24个月绝对胜率是否稳定高于现有深跌筑底体系，再决定是否值得进入分批组合研究。

## Pre-conditions

- [x] `/Users/didi/Project/quant/data/research/blood_chip_deep_base/features.parquet` 含本地正式日线的因果连续 OHLC、成交额与前高回撤，覆盖 `2010-01-04..2026-08-21`。
- [x] `/Users/didi/Project/quant/data/research/low9_kdj_rebound/supplemental_daily/` 含 257 只本地正式日线缺失的退市股历史，共约 61.9 万行；本研究纳入这些证券，避免只研究仍存续股票。
- [x] `/Users/didi/Project/quant/data/research/low9_kdj_rebound/index_000001.SH_20100101_20260731.parquet` 覆盖 `2010-01-04..2026-07-31`；研究截止日固定为 2026-07-31，排除未完成的 2026 年 8 月月线。
- [x] 项目 KDJ 公式固定为 9 期 RSV、K/D 各按 `alpha=1/3, adjust=False` 平滑、`J=3K-2D`。
- [x] “低9”固定为连续9根完成月线满足 `close[t] < close[t-4]`，只在计数首次等于9时触发。
- [x] 当前工作树无未提交改动；本轮只新增本计划列出的研究代码、测试、脚本和被忽略的研究产物。

## Fixed research contract

### Completed-period and point-in-time rules

- 日线先按 `previous_raw_close / current_pre_close` 只向后累乘，后续公司行动不得改写历史月线和周线。
- 月线信号只在该自然月最后一个市场交易日收盘后产生；月度 OHLC 可使用当月截至该日的全部已知个股行情。
- 周线使用 `W-FRI` 完成标签。若月末位于周中，标签晚于月末的当前周不得用于月末信号；节假日缩短周也采用该保守规则。
- 信号月个股最后成交距离市场月末不超过5个市场交易日；下一次个股开盘距离信号不超过5个市场交易日，且不是一字价格。
- 至少有36根完成月线，信号月日均成交额中位数不低于3,000万元。
- 前高只使用信号月以前的完成月线最高价；所有候选从前高回撤至少50%。跌幅只作门槛，不按更深回撤加分。

### Pre-registered signal rules

以下规则均叠加前高回撤、历史、流动性和新鲜度门槛；同一证券同一规则触发后冷却12根完成月线：

1. `monthly_low9`：月线低9完成。
2. `monthly_j_le_minus10`：月线 `J <= -10` 的状态起点。
3. `monthly_j_le_minus20`：月线 `J <= -20` 的状态起点。
4. `monthly_low9_j_negative`：月线低9且月线 `J < 0`。
5. `monthly_weekly_j_le_minus10`：月线与最近可见完整周线均 `J <= -10`。
6. `monthly_low9_weekly_j_le_minus10`：月线低9且最近可见完整周线 `J <= -10`。
7. `monthly_j_le_minus10_weekly_reclaim`：月线 `J <= -10`，且当月某根已完成周线从 `J <= -10` 上穿至 `J > -10`；用于比较“仍超卖”与“开始修复”。

不在查看2021年以后结果后新增阈值或删除规则。

### Event execution and outcomes

- 信号在月末收盘后确认，按下一可交易日开盘成交。
- 排除一字开盘；收益统一扣除20bp往返成本。
- 固定观察126、252、504个市场交易日，近似6、12、24个月；退出价使用目标市场交易日当日或之前最近的个股收盘。
- 若持有期间连续60个市场交易日无个股行情且之后截至目标日仍未恢复，按全额损失核销；若数据截止日前尚未走完目标持有期则标记未完成，不进入该期限统计。
- 同时报告净收益、正收益胜率、收益中位数、资金盈利因子、相对上证综指超额收益、超额胜率、MAE、MFE、信号日期等权均值和按信号日聚类的95%区间。

### Time segmentation and qualification

- 开发信号期：`2013-01-01..2016-12-31`。
- 验证信号期：`2017-01-01..2020-12-31`。
- 已见诊断期：`2021-01-01..2024-12-31`；每个持有期只统计在2026-07-31前已完成的事件，不称为盲测。
- 唯一选择期限为252个市场交易日。候选必须在开发和验证期各有至少100个已完成事件，并同时满足：净胜率不低于60%、中位净收益大于0、资金盈利因子不低于1.50、超额胜率不低于50%。
- 达标候选按两段最低净胜率、最低资金盈利因子、最低中位收益、事件总数排序；2021年以后诊断结果不参与选择。
- 若无候选通过，不进入分批组合实现，报告结论为“月周信号层未达到高胜率要求”。

## Steps

### Step 1 — 用合成月周路径锁定因果边界

**File:** `/Users/didi/Project/quant/tests/research/test_monthly_low_zone.py`

新增并验证以下测试：

- `test_monthly_low9_is_nine_completed_months_below_four_month_lag`
- `test_future_month_rows_do_not_change_prior_month_signal`
- `test_midweek_month_end_cannot_see_incomplete_week`
- `test_monthly_j_threshold_triggers_on_state_onset_with_twelve_month_cooldown`
- `test_weekly_reclaim_requires_a_completed_cross_within_signal_month`
- `test_drawdown_uses_only_prior_completed_month_peak`
- `test_entry_uses_next_open_and_rejects_long_suspension_or_one_price_bar`
- `test_unresolved_horizon_is_excluded_and_sixty_session_disappearance_is_written_off`

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone.py` → `8 passed`。

### Step 2 — 构建月周特征、冻结信号和长周期结果

**File:** `/Users/didi/Project/quant/src/quant/research/monthly_low_zone.py`

新增不可变配置：

```python
@dataclass(frozen=True)
class MonthlyLowZoneConfig:
    minimum_history_months: int = 36
    minimum_drawdown_from_prior_peak: float = 0.50
    minimum_median_daily_amount_thousand: float = 30_000.0
    maximum_signal_staleness_sessions: int = 5
    maximum_entry_delay_sessions: int = 5
    signal_cooldown_months: int = 12
    monthly_j_threshold: float = -10.0
    monthly_extreme_j_threshold: float = -20.0
    weekly_j_threshold: float = -10.0
    maximum_missing_market_sessions: int = 60
    round_trip_cost_bps: float = 20.0
    horizons: tuple[int, int, int] = (126, 252, 504)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
```

公开函数固定为：

```python
def build_monthly_weekly_features(
    daily: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return completed monthly features and completed W-FRI KDJ bars."""


def generate_monthly_low_zone_signals(
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    config: MonthlyLowZoneConfig,
) -> pd.DataFrame:
    """Return the seven pre-registered month-end signal families."""


def evaluate_monthly_low_zone_events(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    benchmark: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
    config: MonthlyLowZoneConfig,
) -> pd.DataFrame:
    """Resolve next-open entries and 126/252/504-session outcomes."""


def summarize_monthly_low_zone_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize rule, period and horizon with clustered path metrics."""
```

所有公开函数使用完整类型注解，输入缺列时抛出列名明确的 `ValueError`，不使用隐藏全局数据源。

### Step 3 — 读取正式数据、运行冻结比较并生成报告

**File:** `/Users/didi/Project/quant/scripts/research/backtest_monthly_low_zone.py`

CLI 固定支持：

```text
--feature-cache data/research/blood_chip_deep_base/features.parquet
--supplemental-root data/research/low9_kdj_rebound/supplemental_daily
--stock-basic-history data/raw/stock_basic_history.parquet
--benchmark data/research/low9_kdj_rebound/index_000001.SH_20100101_20260731.parquet
--output-dir reports/research/monthly_low_zone
--cache-dir data/research/monthly_low_zone
--build-cache
```

产物固定为：

```text
/Users/didi/Project/quant/reports/research/monthly_low_zone/signals.parquet
/Users/didi/Project/quant/reports/research/monthly_low_zone/events.parquet
/Users/didi/Project/quant/reports/research/monthly_low_zone/metrics.csv
/Users/didi/Project/quant/reports/research/monthly_low_zone/yearly_metrics.csv
/Users/didi/Project/quant/reports/research/monthly_low_zone/decision.json
/Users/didi/Project/quant/reports/research/monthly_low_zone/report.md
```

报告必须并列七条规则和三个持有期，披露退市补充覆盖、未完成事件、核销数量、事件重叠、信号日期数量及2021年以后非盲测属性。

### Step 4 — 完整验证

```bash
PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone.py tests/research/test_blood_chip_kdj.py tests/research/test_low9_kdj_rebound.py
PYTHONPATH=src python -m compileall -q src/quant/research/monthly_low_zone.py scripts/research/backtest_monthly_low_zone.py tests/research/test_monthly_low_zone.py
PYTHONPATH=src python scripts/research/backtest_monthly_low_zone.py --build-cache
git diff --check
```

预期：新增与既有KDJ/低9测试全部通过；报告与决策文件存在；如果没有规则满足开发和验证双段门槛，`decision.json` 的 `selected_rule` 必须为 `null`。

## Rollback

本研究不修改数据库、线上配置或生产策略。若实现需要撤回，只删除本计划新增的三个源文件和被忽略的 `data/research/monthly_low_zone/`、`reports/research/monthly_low_zone/`；不得回滚用户在同一工作树中的其他改动。

## Commit checkpoint

本轮不自动提交。用户确认后建议提交：

```text
feat(research): test monthly low9 and multi-timeframe KDJ lows
```
