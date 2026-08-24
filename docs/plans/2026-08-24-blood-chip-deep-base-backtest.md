# 深跌筑底型带血筹策略与长持回测计划

## Goal

在不修改现有线上“卖压冲击—吸收确认”策略的前提下，新增一套独立的“历史前高深跌—卖压耗尽—60日区间筑底—区间内分批建仓—1至2年持有”研究策略，并用因果日线、真实交易成本和分段组合回测判断其能否同时达到高胜率、正中位收益、长期超额收益与可接受回撤。

## Pre-conditions

- [x] `/Users/didi/Project/quant/data/raw/daily_partitioned/` 包含 200 个自然月分区，覆盖 `20100104..20260821`。
- [x] `/Users/didi/Project/quant/data/raw/index_000300.SH.parquet` 覆盖 `20100104..20260821`。
- [x] 日线包含 `ts_code/trade_date/open/high/low/close/pre_close/pct_chg/vol/amount`，`amount` 为 Tushare 千元口径。
- [x] `PYTHONPATH=src pytest -q tests/research/test_blood_chip.py tests/research/test_blood_chip_scale_in.py tests/test_blood_chip_long_plan.py tests/test_webapp_api.py tests/test_web_frontend.py` 输出 `169 passed`。
- [x] 当前工作树无未提交文件；本研究只新增本计划列出的代码、测试、脚本、文档以及被忽略的研究产物。

## Fixed research contract

### Point-in-time price and universe

- 使用项目现有因果连续 OHLC：公司行动只调整当日及以后价格，不用未来复权因子回写历史。
- 每个信号日只读取当日收盘后可见数据；首仓和后续加仓均在触发后的下一可交易日开盘执行。
- 至少需要 500 个历史交易日；过去 20 日成交额中位数不低于 3,000 万元。
- “前高”定义为该证券从本地数据起点至前一交易日的最高因果连续最高价；同时保存峰值日期和峰值距信号日的交易日数。
- 主比较预注册三档深跌门：从前高回撤至少 50%、65%、80%，不在看完诊断期后新增中间阈值。
- 峰值至少发生在 120 个交易日前；价格在对应深跌阈值下连续停留至少 40 个交易日，排除刚刚快速腰斩的下落过程。

### Exhausted selling and 60-session base

- 筑底窗口固定为 60 个交易日。
- 60 日最高价相对最低价的区间宽度不超过 25%。
- 60 日收盘收益位于 `[-12%, +12%]`，排除仍在单边下跌或已经明显启动的路径。
- 当前收盘位于区间 15%—65% 位置；首次建仓不追区间上沿。
- 当前价格不再创 60 日新低，距最近一次滚动新低至少 10 个交易日。
- 最近 20 日下跌日成交额占比必须不高于更早 60 日的 80%，或者绝对占比不高于 35%；该条件仅表示卖出成交收缩，不声称识别真实卖方身份。
- 最近 20 日年化波动不高于此前 60 日年化波动的 85%，要求价格和波动同时收敛。
- 同一证券信号冷却 120 个交易日；冷却期内不因条件反复满足重复建仓。
- 深跌幅度只作为准入门槛，不按“跌得越深越优”排序；同日容量优先分配给深跌持续更久、区间更紧、下跌成交占比更低、波动收缩更充分的候选。

### In-range staged construction

- 每个完整计划目标为组合权益的 10%，最多同时 10 个计划。
- 第一段 20%：信号后下一开盘缺口在 `[-7%, +7%]` 且不是一字涨停时成交，约占组合权益 2%。
- 第二段 30%：首仓后至少 10 个持有交易日，前一日收盘仍高于原筑底下沿 98%，并位于原区间 45% 以下；下一开盘仍在原区间下沿至中轴之间时成交。
- 第三段 50%：首仓后至少 20 个持有交易日，前一日收盘位于原区间 45%—90%，20 日收益为正；下一开盘未高于原区间上沿 3% 时成交。
- 所有加仓只允许在首仓后 120 个交易日内触发；没有出现确认则永久保留 20% 或 50% 部署，不强制建满。

### Exit and costs

- 原筑底下沿下方 10% 为灾难硬止损；跳空跌透时按开盘成交，一字跌停时延迟。
- 连续两日收盘低于原筑底下沿 97% 时，下一可交易日开盘结构性退出。
- 持仓连续 60 个市场交易日没有行情时按全额损失核销，防止退市或永久停牌按最后价格冻结并虚增长期胜率；60 日以内停牌等待复牌。
- 不设置固定止盈，分别比较首仓起算 250 和 500 个交易日到期退出。
- A 股 T+1；买卖各 5bp 滑点，佣金 3bp、卖出印花税 5bp、过户费 1bp、最低佣金 5 元、100 股整手。

### Time segmentation and decision gate

- 开发入场期：`2013-01-04..2016-12-30`，500 日持有最晚结果落在 2018 年末附近。
- 验证入场期：`2017-01-03..2020-12-31`，500 日持有最晚结果落在 2022 年末附近。
- 已见诊断入场期：`2021-01-04..2024-07-30`；`2024-07-30` 是终点 `2026-08-21` 前第 500 个沪深 300 交易日。
- 只使用开发期和验证期选择唯一候选；2021 年以后数据已被本项目其他研究观察，不称为独立盲测。
- 候选必须在开发期、验证期各有至少 40 笔已完成交易，并同时满足：胜率不低于 55%、中位净收益大于 0、资金盈利因子不低于 1.50、组合最大回撤不超过 35%。
- 达标候选先按两段最低胜率、两段最低资金盈利因子、两段年化收益几何均值、交易数排序；若没有候选全部达标，报告必须输出“保留研究，不替换线上策略”。

## Steps

### Step 1 — 用合成路径锁定深跌、筑底和因果信号

**File:** `/Users/didi/Project/quant/tests/research/test_blood_chip_deep_base.py`

新增以下测试并先确认在模块不存在时失败：

- `test_deep_drawdown_uses_only_prior_visible_peak`
- `test_future_rows_do_not_change_existing_deep_base_features`
- `test_signal_requires_threshold_duration_and_sixty_session_base`
- `test_falling_price_or_wide_range_does_not_signal`
- `test_recent_down_amount_and_volatility_must_contract`
- `test_signal_cooldown_suppresses_duplicate_base_entries`
- `test_first_second_and_third_tranches_trade_on_successive_next_opens`
- `test_unconfirmed_base_keeps_partial_position`
- `test_additions_never_chase_above_original_base`
- `test_structural_break_exits_on_next_open`
- `test_gap_through_hard_stop_uses_open_and_t_plus_one`
- `test_long_hold_exits_at_configured_session_count`
- `test_sixty_missing_market_sessions_write_position_off`

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_blood_chip_deep_base.py` → 完成 Step 2 后全部通过。

### Step 2 — 实现深跌筑底特征、信号和区间分批组合引擎

**File:** `/Users/didi/Project/quant/src/quant/research/blood_chip_deep_base.py`

新增不可变配置：

```python
@dataclass(frozen=True)
class DeepBaseSignalConfig:
    minimum_history_days: int = 500
    minimum_prior_amount_thousand: float = 30_000.0
    minimum_drawdown_from_peak: float = 0.50
    minimum_peak_age_sessions: int = 120
    minimum_deep_drawdown_sessions: int = 40
    base_window_sessions: int = 60
    maximum_base_range: float = 0.25
    minimum_base_return: float = -0.12
    maximum_base_return: float = 0.12
    minimum_base_position: float = 0.15
    maximum_base_position: float = 0.65
    minimum_sessions_since_new_low: int = 10
    maximum_recent_down_amount_share: float = 0.35
    maximum_down_amount_share_ratio: float = 0.80
    maximum_volatility_contraction_ratio: float = 0.85
    signal_cooldown_sessions: int = 120


@dataclass(frozen=True)
class DeepBaseExecutionConfig:
    initial_cash: float = 1_000_000.0
    maximum_positions: int = 10
    target_position_fraction: float = 0.10
    tranche_fractions: tuple[float, float, float] = (0.20, 0.30, 0.50)
    second_stage_minimum_sessions: int = 10
    third_stage_minimum_sessions: int = 20
    maximum_scale_in_sessions: int = 120
    hard_stop_below_base: float = 0.10
    structural_break_below_base: float = 0.03
    structural_break_sessions: int = 2
    maximum_missing_market_sessions: int = 60
    maximum_holding_sessions: int = 500
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    minimum_commission: float = 5.0
    slippage: float = 0.0005
    minimum_entry_gap: float = -0.07
    maximum_entry_gap: float = 0.07
    locked_limit_threshold: float = 0.048
    lot_size: int = 100
```

公开函数固定为：

```python
def build_deep_base_features(daily: pd.DataFrame) -> pd.DataFrame: ...
def generate_deep_base_signals(features: pd.DataFrame, config: DeepBaseSignalConfig) -> pd.DataFrame: ...
def run_deep_base_backtest(daily: pd.DataFrame, signals: pd.DataFrame, config: DeepBaseExecutionConfig, entry_start: str, entry_end: str) -> BloodChipBacktestResult: ...
def summarize_deep_base_result(result: BloodChipBacktestResult, benchmark: pd.DataFrame) -> dict[str, float | int]: ...
```

信号输出至少包含 `prior_peak/prior_peak_date/peak_age_sessions/drawdown_from_peak/deep_drawdown_sessions/base_low/base_high/base_mid/base_range/base_return/base_position/sessions_since_new_low/recent_down_amount_share/prior_down_amount_share/down_amount_share_ratio/volatility_20d/volatility_prior_60d/volatility_contraction_ratio/signal_score/signal_date/entry_date/entry_open`。交易输出额外包含三段成交日期、部署比例、原筑底上下沿、结构破位次数、退出原因和成本。

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_blood_chip_deep_base.py` → 12 项因果和执行测试全部通过。

### Step 3 — 实现冻结变体、分段回测和报告

**File:** `/Users/didi/Project/quant/scripts/research/backtest_blood_chip_deep_base.py`

CLI 固定支持：

```text
--daily-root data/raw/daily_partitioned
--benchmark data/raw/index_000300.SH.parquet
--output-dir reports/research/blood_chip_deep_base
--cache-dir data/research/blood_chip_deep_base
--build-cache
```

脚本固定比较六个变体：`dd50_hold250/dd50_hold500/dd65_hold250/dd65_hold500/dd80_hold250/dd80_hold500`。每个变体分别运行开发、验证、已见诊断三个入场期，输出交易数、胜率、Wilson 95% 下界、平均/中位净收益、资金加权单笔收益、资金盈利因子、总收益、年化收益、沪深300收益、最大回撤、止损率、结构破位率、平均持有期、平均部署比例、三段完成率及年度分布。

产物固定为：

```text
/Users/didi/Project/quant/reports/research/blood_chip_deep_base/metrics.csv
/Users/didi/Project/quant/reports/research/blood_chip_deep_base/metrics.json
/Users/didi/Project/quant/reports/research/blood_chip_deep_base/trades.parquet
/Users/didi/Project/quant/reports/research/blood_chip_deep_base/yearly_metrics.csv
/Users/didi/Project/quant/reports/research/blood_chip_deep_base/representative_cases.csv
/Users/didi/Project/quant/reports/research/blood_chip_deep_base/decision.json
/Users/didi/Project/quant/reports/research/blood_chip_deep_base/report.md
```

**Verify:** `PYTHONPATH=src python scripts/research/backtest_blood_chip_deep_base.py --help` → 返回 0 并显示五个参数。

### Step 4 — 运行全历史回验并执行冻结决策

首次运行：

```bash
PYTHONPATH=src python scripts/research/backtest_blood_chip_deep_base.py --build-cache
```

报告必须：

- 明确区分开发、验证和已见诊断，不把 2021 年以后称为盲测。
- 并列三档深跌阈值和两个持有期，不只展示胜出者。
- 明确报告是否有候选同时通过全部高胜率门槛。
- 若未通过，保留当前线上 `blood_chip_long_v3_kdj_path_annotation` 不变。
- 单列 80% 深跌样本数和胜率，样本不足时不因高收益个例升级。

**Verify:** `test -f reports/research/blood_chip_deep_base/report.md && test -f reports/research/blood_chip_deep_base/decision.json` → 返回 0；`decision.json` 包含 `selected_on_development_and_validation_only` 和 `deployment_decision`。

### Step 5 — 完整验证

按顺序运行：

```bash
PYTHONPATH=src pytest -q tests/research/test_blood_chip_deep_base.py
PYTHONPATH=src pytest -q tests/research/test_blood_chip.py tests/research/test_blood_chip_scale_in.py tests/test_blood_chip_long_plan.py
PYTHONPATH=src python -m compileall -q src/quant/research/blood_chip_deep_base.py scripts/research/backtest_blood_chip_deep_base.py tests/research/test_blood_chip_deep_base.py
ruff check src/quant/research/blood_chip_deep_base.py scripts/research/backtest_blood_chip_deep_base.py tests/research/test_blood_chip_deep_base.py
```

预期：所有新增测试和既有带血筹回归通过；`compileall` 无输出；`ruff` 无诊断。

## Commit checkpoint

本轮不自动提交。用户确认后建议提交：

```text
feat(research): backtest deep-drawdown blood-chip bases
```

## Follow-up — 二次探底后再加仓的机制迭代

首轮结果打开后，只用开发期和验证期 case 发现：`dd65_hold250` 的首仓未加仓样本胜率分别为 50.0% 和 45.0%，三段完成样本为 54.6% 和 47.1%，但只完成第二段的样本胜率仅 4.8% 和 5.3%。原“价格仍在区间下半部就加 30%”实际在放大继续破位的路径。

据此预注册唯一信号成熟度改动和两个退出对照，不再搜索连续阈值：

- 深跌门固定为前高下跌至少 65%，峰值距今至少 750 个交易日，深跌状态持续至少 120 个交易日。
- 第二段不在下半区直接加仓；必须先见到区间 45% 以下的二次探底，再重新站上区间中轴且 20 日收益转正，才在下一开盘加 30%。
- 第三段要求价格位于区间 70%—100% 且 20 日收益继续为正，仍不追出原区间。
- 对比保留“两日结构破位退出”和只保留“区间下沿下方 10% 灾难止损”两种退出；后者用于检验结构退出是否对月/年级持有过敏。
- 比较 250/500 日持有，共四个冻结变体；选择门槛和时间分段与首轮完全相同，2021 年以后仍不参与选择。

新增测试：

- `test_retest_reclaim_waits_while_price_remains_in_lower_half`
- `test_retest_reclaim_adds_after_midline_recovery`
- `test_hard_stop_only_policy_ignores_shallow_structural_break`

新增产物目录：`/Users/didi/Project/quant/reports/research/blood_chip_deep_base_iteration/`。

## Rollback

本研究不修改数据库、线上策略或已有研究模块。若实现或回测失败，仅删除以下新增内容，不覆盖任何既有文件：

```text
/Users/didi/Project/quant/src/quant/research/blood_chip_deep_base.py
/Users/didi/Project/quant/scripts/research/backtest_blood_chip_deep_base.py
/Users/didi/Project/quant/tests/research/test_blood_chip_deep_base.py
/Users/didi/Project/quant/docs/plans/2026-08-24-blood-chip-deep-base-backtest.md
/Users/didi/Project/quant/data/research/blood_chip_deep_base/
/Users/didi/Project/quant/reports/research/blood_chip_deep_base/
```
