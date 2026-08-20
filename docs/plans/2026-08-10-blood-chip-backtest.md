# 带血筹量价筛选与长周期回测实施计划

## Goal

用项目内 2013-01-04 至 2026-08-07 的点时 A 股日线和沪深 300 基准，构建“卖压冲击—筹码吸收”两阶段信号、允许止损后新事件重入的长周期组合回测，并只根据 2022 年及以前的案例完善规则，最后在 2023 年以后数据上做一次冻结参数盲测。

## Pre-conditions

- [x] `data/raw/daily_partitioned/year_month=201301/data.parquet` 至 `year_month=202608/data.parquet` 存在，合计 12,227,336 行、5,553 个历史证券。
- [x] `data/raw/index_000300.SH.parquet` 覆盖 `20100104` 至 `20260807`。
- [x] 日线字段包含 `ts_code/trade_date/open/high/low/close/pre_close/pct_chg/vol/amount`；`amount` 沿用 Tushare 千元口径。
- [x] 工作树已有大量用户修改；本任务只新建本计划所列代码/文档文件及被 `.gitignore` 排除的 `reports/`、`data/research/` 产物。
- [x] Python 环境可导入 pandas、numpy、pyarrow、pytest。

## Fixed research contract

- 信号只使用信号日收盘后可得数据；成交固定在下一可交易日开盘。
- 用截至前一交易日的 60 日 beta 剔除沪深 300 影响，不使用未来 beta。
- 公司行动采用项目现有的 causal continuous OHLC 口径：除权日在当日及之后调整，不回写过去价格。
- 基础流动性门为事件前 20 日成交额中位数不低于 3,000 万元。
- “卖压冲击”按 5 日残差跌幅、成交额放大、下跌价格冲击和 20 日回撤的当日截面分位合成。
- “吸收确认”必须发生在冲击后的第 2 至第 10 个交易日，使用事件低点反弹、下跌冲击衰减、3 日残差收益和 3 日收盘位置合成。
- 同一冲击事件只生成一次首次确认信号；止损后不永久拉黑证券，新冲击事件再次确认时允许重入。
- A 股 T+1；新买入仓位当天不可卖出。停牌无 bar 时不交易；一字涨停不买，一字跌停不卖。
- 默认组合 10 个等风险预算槽位；买卖各计 5bp 滑点，佣金 3bp、卖出印花税 5bp、过户费 1bp、最低佣金 5 元。
- 基线退出为 10% 硬止损和 120 个交易日到期，不预设止盈，以免截断长周期右尾。
- 开发期 `2014-01-01..2019-12-31`；案例迭代期 `2020-01-01..2022-12-30`；冻结盲测期 `2023-01-03..2026-02-06`。盲测结束日给后续最长持有期留足可观测路径。

## Steps

### Step 1 — 定义点时特征、事件和配置契约

**File:** `/Users/didi/Project/quant/src/quant/research/blood_chip.py`

新增以下不可变配置和结果类型：

```python
@dataclass(frozen=True)
class BloodChipSignalConfig:
    minimum_history_days: int = 120
    minimum_prior_amount_thousand: float = 30_000.0
    shock_score_threshold: float = 0.80
    maximum_residual_5d_percentile: float = 0.05
    minimum_amount_ratio_5d: float = 1.25
    minimum_impact_ratio_5d: float = 1.25
    event_quiet_days: int = 5
    minimum_absorption_day: int = 2
    maximum_absorption_day: int = 10
    minimum_rebound_from_event_low: float = 0.02
    maximum_impact_decay: float = 0.90
    minimum_residual_3d: float = 0.0
    minimum_clv_3d: float = -0.25
    absorption_score_threshold: float = 0.60
    minimum_return_120d: float | None = None
    maximum_return_120d: float | None = None
    maximum_volatility_60d: float | None = None
    minimum_market_return_60d: float | None = None

@dataclass(frozen=True)
class BloodChipBacktestConfig:
    initial_cash: float = 1_000_000.0
    maximum_positions: int = 10
    stop_loss: float = 0.10
    maximum_holding_days: int = 120
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    minimum_commission: float = 5.0
    slippage: float = 0.0005
    minimum_entry_gap: float = -0.07
    maximum_entry_gap: float = 0.07
    locked_limit_threshold: float = 0.048
    allow_reentry_after_stop: bool = True
    require_new_event_for_reentry: bool = True

@dataclass(frozen=True)
class BloodChipBacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    rejected_entries: pd.DataFrame
```

实现并完整标注类型的公共函数：

```python
def load_canonical_daily(root: str | Path, start_date: str, end_date: str) -> pd.DataFrame: ...
def load_benchmark(path: str | Path, start_date: str, end_date: str) -> pd.DataFrame: ...
def build_blood_chip_features(daily: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame: ...
def generate_blood_chip_signals(features: pd.DataFrame, config: BloodChipSignalConfig) -> pd.DataFrame: ...
def run_blood_chip_backtest(daily: pd.DataFrame, signals: pd.DataFrame, config: BloodChipBacktestConfig, entry_start: str, entry_end: str) -> BloodChipBacktestResult: ...
def summarize_blood_chip_result(result: BloodChipBacktestResult, benchmark: pd.DataFrame) -> dict[str, float | int]: ...
def analyze_blood_chip_cases(trades: pd.DataFrame) -> pd.DataFrame: ...
```

`build_blood_chip_features` 返回至少 `adjusted_open/high/low/close`、`market_beta_60d`、`residual_return_1d/3d/5d`、`amount_ratio_5d`、`impact_ratio_5d`、`drawdown_20d`、`return_120d`、`volatility_60d`、`market_return_60d` 和各截面分位。`generate_blood_chip_signals` 返回事件 ID、冲击日、确认日、下一开盘日以及冲击/吸收分项，保证每个 `ts_code + shock_event_id` 最多一条记录。

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_blood_chip.py -k 'features or signal'` → 点时不变性、事件去重、确认窗口和下一开盘日测试通过。

### Step 2 — 实现 T+1、止损、重入和组合资金曲线

**File:** `/Users/didi/Project/quant/src/quant/research/blood_chip.py`

`run_blood_chip_backtest` 按交易日顺序执行：用前一日收盘生成的 signal 在下一可交易日开盘申请买入；每个证券同时最多一个仓位；新买入当日不检查退出；后续日先检查一字跌停代理，再以“跳空开盘价优先、否则止损价”的保守方式执行硬止损；达到 120 个交易日时在下一可交易开盘退出。每笔交易保存 `shock_event_id/signal_date/entry_date/exit_date/exit_reason/reentry_number/gross_return/net_return/fees/maximum_adverse_excursion/maximum_favorable_excursion` 和全部入场特征。

重入状态固定为：若 `allow_reentry_after_stop=True`，止损后仍可接受该证券的新信号；若 `require_new_event_for_reentry=True`，新 signal 的 `shock_event_id` 必须大于上一笔已用事件。因持仓、槽位、缺失 bar、开盘跳空或一字涨停被拒绝的信号进入 `rejected_entries`，不得静默丢弃。

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_blood_chip.py -k 'backtest or reentry or limit'` → 下一开盘、T+1、止损跳空、一字跌停延迟、同事件不重入和新事件可重入测试通过。

### Step 3 — 增加可复现 CLI 与产物

**File:** `/Users/didi/Project/quant/scripts/research/backtest_blood_chip.py`

CLI 参数固定包含：

```text
--daily-root data/raw/daily_partitioned
--benchmark data/raw/index_000300.SH.parquet
--output-dir reports/research/blood_chip
--cache-dir data/research/blood_chip
--build-cache
--development-only
--frozen-holdout
```

脚本执行以下顺序：读取/复用特征缓存；输出基线在开发期和案例迭代期的指标；保存 `baseline_trades.parquet`、`baseline_equity.parquet`、`baseline_cases.csv`、`baseline_metrics.json`；只在非盲测数据上比较少量透明候选规则；将胜出配置写入 `frozen_config.json` 后，只有显式 `--frozen-holdout` 才读取并运行 2023 年后的盲测；最终生成 `report.md`。所有 JSON 写入数据日期、行数、证券数、复权口径、成交规则、参数、Git SHA（若可得）和运行时间。

**Verify:** `PYTHONPATH=src python scripts/research/backtest_blood_chip.py --help` → 返回 0 且显示上述参数。

### Step 4 — 用合成数据锁定因果与执行测试

**File:** `/Users/didi/Project/quant/tests/research/test_blood_chip.py`

新增以下独立测试：

```text
test_future_rows_do_not_change_existing_feature_values
test_signal_requires_shock_then_absorption_and_deduplicates_event
test_signal_quality_filters_use_only_entry_time_features
test_entry_uses_next_available_open
test_t_plus_one_blocks_same_day_stop
test_gap_through_stop_fills_at_open
test_locked_limit_down_delays_stop_exit
test_stop_does_not_permanently_blacklist_symbol
test_reentry_requires_a_new_shock_event
test_round_trip_costs_include_sell_stamp_tax
test_summary_total_return_includes_first_portfolio_day
test_case_analysis_uses_only_entry_time_features
```

所有 fixture 使用固定日期、固定价格和固定成交量；不访问网络、环境时间或随机数。

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_blood_chip.py` → 12 个新增测试全部通过。

### Step 5 — 基线回测和案例归因

运行：

```bash
PYTHONPATH=src python scripts/research/backtest_blood_chip.py --build-cache --development-only
```

在 2014–2019 和 2020–2022 分别报告交易数、胜率及 Wilson 下界、平均/中位净收益、盈亏比、收益因子、年化收益、最大回撤、平均持有期、止损率、重入交易数和相对沪深 300 收益。案例表只使用入场时特征，将赢家、普通亏损、止损、成功重入和失败重入按 `return_120d/volatility_60d/market_return_60d/shock_score/absorption_score/impact_decay/rebound_from_event_low` 分箱比较。

**Verify:** `test -f reports/research/blood_chip/baseline_metrics.json && test -f reports/research/blood_chip/baseline_cases.csv` → 返回 0；指标 JSON 中开发期和案例期交易数均大于 0。

### Step 6 — 冻结改进逻辑并做一次盲测

只从案例分析支持的三类可解释逻辑中选择，不做无限网格搜索：

```text
A. avoid_high_volatility：限制入场时 60 日年化波动，排除泡沫破裂式反抽。
B. avoid_overextended：限制入场前 120 日涨幅，避免过度延伸后的下跌中继。
C. avoid_extreme_spiral：要求沪深 300 的 60 日收益不处于极端下跌状态。
```

基线 case 显示亏损组的吸收分和反弹并不弱，因此放弃“继续提高吸收阈值”的原候选方向；最终冻结组合为 120 日涨幅不高于 50%、60 日年化波动不高于 55%、沪深 300 的 60 日收益不低于 -15%。

候选必须同时满足：开发期和案例期交易数均不少于 100；两个时期平均净收益均为正；案例期收益因子与最大回撤不劣于基线；改进不能只来自单一年份。按“案例期 Wilson 胜率下界、收益因子、平均净收益、交易数”依次排序，冻结唯一配置。冻结后运行：

```bash
PYTHONPATH=src python scripts/research/backtest_blood_chip.py --frozen-holdout
```

盲测结果无论好坏均写入报告，不再更改参数。报告并列基线、冻结策略和沪深 300，列出最具代表性的盈利、亏损、止损后成功重入、止损后再次失败案例。

**Verify:** `test -f reports/research/blood_chip/frozen_config.json && test -f reports/research/blood_chip/report.md` → 返回 0；`frozen_config.json` 的 `selected_without_holdout` 为 `true`。

### Step 7 — 完整验证

按顺序执行：

```bash
PYTHONPATH=src pytest -q tests/research/test_blood_chip.py
PYTHONPATH=src python -m compileall -q src/quant/research/blood_chip.py scripts/research/backtest_blood_chip.py tests/research/test_blood_chip.py
ruff check src/quant/research/blood_chip.py scripts/research/backtest_blood_chip.py tests/research/test_blood_chip.py
```

预期：定向测试全部通过；compileall 无输出且返回 0；ruff 对本任务新文件无诊断。若全量测试成本可接受，再运行 `PYTHONPATH=src pytest -q`，并把任何既有失败与新增回归分开报告。

## Commit checkpoint

本轮不自动提交。用户确认后建议提交信息：

```text
feat(research): backtest blood-chip absorption signals
```

## Follow-up — 事件路径与软波动约束

根据 case 复盘增加 `add_blood_chip_path_features`，比较冲击前 20 日、冲击 5 日和确认 3 日的波动、振幅、成交及下跌成交占比，并新增 `/Users/didi/Project/quant/scripts/research/analyze_blood_chip_exhaustion.py`。结论是绝对波动硬门可改为同日候选软排序；完整证据见 `/Users/didi/Project/quant/reports/research/blood_chip_exhaustion/report.md`。

## Rollback

本任务不迁移数据库、不改线上配置、不删除任何数据。回滚时只删除以下本任务新文件和被忽略的产物，不能覆盖任务开始前已有修改：

```text
/Users/didi/Project/quant/src/quant/research/blood_chip.py
/Users/didi/Project/quant/scripts/research/backtest_blood_chip.py
/Users/didi/Project/quant/scripts/research/analyze_blood_chip_exhaustion.py
/Users/didi/Project/quant/tests/research/test_blood_chip.py
/Users/didi/Project/quant/docs/plans/2026-08-10-blood-chip-backtest.md
/Users/didi/Project/quant/data/research/blood_chip/
/Users/didi/Project/quant/reports/research/blood_chip/
```
