# 带血筹多维生存过滤与周期修复回测计划

## Goal

固定上一轮表现最稳健但仍未达升级门槛的价格层 `dd65_mature_reclaim_hold250_hard_only`，只增加点时可得的公司生存、估值规模、资本压力、市场修复和行业修复维度，检验它们能否减少“横盘后继续死亡”的失败样本，并在开发期与验证期同时提高胜率、中位收益和资金盈利因子；本研究不修改线上带血筹策略。

## Pre-conditions

- [x] 深跌价格特征缓存覆盖 `2010-01-04..2026-08-21`，含 13,660,048 行。
- [x] 固定价格信号为：前高回撤至少 65%、峰值距今至少 750 个交易日、深跌持续至少 120 个交易日、60 日筑底、二次探底后中轴收复加仓、只保留灾难硬止损、250 个交易日持有。
- [x] `daily_basic` 共 3,311 个交易日文件，覆盖 `2013-01-04..2026-08-21`，可提供信号日收盘后的市值、PB、PS、PE、真实换手率和自由流通换手率。
- [x] `fina_indicator/income/cashflow/balancesheet` 分别含公告日与报告期；年度质量只使用每个报告期本地首次披露值，并要求最晚源公告日不晚于信号日。
- [x] `holder_trade` 覆盖 `2013-01-01..2026-08-12`，以公告日为可得日；`pledge_stat` 覆盖 `2014-03-07..2026-07-31`，同日记录保守地从下一交易日开始使用。
- [x] 沪深300覆盖 `2010-01-04..2026-08-21`。
- [x] 只有当前 `stock_basic.industry`，没有历史行业成员表；行业修复结果必须标记 `current_industry_mapping_bias`，不得参与正式候选选择。

## Fixed research contract

### Point-in-time dimensions

1. **公司生存能力**：最近已披露年度报告距信号日不超过 550 天；至少有 3 个年度历史；最近年度归母净利润与经营现金流均为正；近 5 年盈利为正比例、经营现金流为正比例均不低于 60%。缺失不填成通过。
2. **估值与可持续交易规模**：信号日总市值不低于 10 亿元；PB 为正且不高于 4，或 PS_TTM 为正且不高于 3；自由流通换手率不高于 12%。只用信号日收盘后可见的 `daily_basic`。
3. **资本压力**：最近可见质押比例不高于 50%；无质押记录视为“未观察到高质押”并单列覆盖率。最近 180 天公告的股东净增减持比例不低于 -2%；无增减持公告视为 0。`IN` 为正、`DE` 为负。
4. **市场修复**：沪深300信号日 20 日收益不低于 -8%、120 日收益不低于 -18%，且收盘不低于 250 日均线的 85%。这是一道防系统性加速下跌门，不要求牛市。
5. **行业修复诊断**：按当前行业映射计算信号日行业成分股 20 日收益中位数与上涨占比，要求中位数不低于 -5%、上涨占比不低于 35%。因历史映射不可靠，只做诊断。

所有连续特征保留原值、覆盖标记和可得日。多维分数只用于同日容量排序：价格分占 65%，已覆盖且通过程度的公司/估值/资本/市场分占 35%；缺失维度不获得正向加分。

### Pre-registered policies

- `price_only`：固定价格层基线。
- `survival`：价格层 + 公司生存门。
- `survival_value`：生存 + 估值与规模门。
- `survival_capital`：生存 + 资本压力门。
- `survival_market`：生存 + 市场修复门。
- `auditable_combined`：生存 + 估值规模 + 资本压力 + 市场修复；唯一可参与升级选择的完整多维候选。
- `current_industry_diagnostic`：生存 + 当前行业修复，仅诊断。
- `combined_current_industry_diagnostic`：完整多维 + 当前行业修复，仅诊断。

不根据 2021 年以后结果追加阈值、删除困难年份或增加新组合。价格执行层统一使用 250 日持有、硬止损、二次探底收复加仓，不再比较退出和持有期。

### Time segmentation and decision gate

- 开发入场期：`2013-01-04..2016-12-30`。
- 验证入场期：`2017-01-03..2020-12-31`。
- 已见诊断入场期：`2021-01-04..2024-07-30`，不称为盲测。
- 只允许 `survival/survival_value/survival_capital/survival_market/auditable_combined` 参与升级选择；行业诊断永不入选。
- 每个候选在开发和验证期各至少 40 笔完成交易，并同时满足：胜率不低于 55%、中位净收益大于 0、资金盈利因子不低于 1.50、组合最大回撤不超过 35%。
- 若没有候选全部达标，输出“保留研究，不替换线上策略”；不得用综合排序替代硬门槛。

## Steps

### Step 1 — 锁定点时合并与门槛语义

**File:** `/Users/didi/Project/quant/tests/research/test_blood_chip_multidimensional.py`

新增合成测试：财报在公告日前不可见、后续更正不回填；质押同日不可见；180 日增减持窗口不读取未来公告；市场特征只用当日及以前；缺失财务不能通过生存门；行业策略必须标记为诊断；未来数据追加不改变历史门槛。

### Step 2 — 实现多维信号富化和策略门

**File:** `/Users/didi/Project/quant/src/quant/research/blood_chip_multidimensional.py`

实现：

```python
@dataclass(frozen=True)
class MultidimensionalGateConfig: ...

def merge_financial_survival_asof(signals, annual_events): ...
def merge_daily_basic_on_signal_date(signals, daily_basic): ...
def merge_capital_pressure_asof(signals, pledge_stat, holder_trade): ...
def add_market_repair_features(signals, benchmark): ...
def add_current_industry_repair_features(signals, price_features, stock_basic): ...
def apply_multidimensional_gates(signals, config): ...
```

输出必须包含各源可得日、原始值、覆盖布尔值、五个门槛、审计型综合门、诊断型行业门和多维排序分。

### Step 3 — 实现冻结策略比较和报告

**File:** `/Users/didi/Project/quant/scripts/research/backtest_blood_chip_multidimensional.py`

读取已有价格特征缓存，生成固定价格信号，按信号日只加载所需 `daily_basic` 截面，构建年度财务事件、资本压力、市场与当前行业特征，运行八个预注册策略。产物：

```text
reports/research/blood_chip_multidimensional/coverage.csv
reports/research/blood_chip_multidimensional/signal_diagnostics.parquet
reports/research/blood_chip_multidimensional/metrics.csv
reports/research/blood_chip_multidimensional/trades.parquet
reports/research/blood_chip_multidimensional/yearly_metrics.csv
reports/research/blood_chip_multidimensional/decision.json
reports/research/blood_chip_multidimensional/report.md
```

报告并列展示每个维度造成的样本保留率、失败样本削减率、开发/验证/已见诊断表现和年度稳定性；行业结果必须带偏差警告。

### Step 4 — 验证

```bash
PYTHONPATH=src pytest -q tests/research/test_blood_chip_multidimensional.py
PYTHONPATH=src pytest -q tests/research/test_blood_chip_deep_base.py tests/research/test_blood_chip.py tests/research/test_blood_chip_scale_in.py tests/test_blood_chip_long_plan.py tests/test_long_quality_factors.py
PYTHONPATH=src python -m compileall -q src/quant/research/blood_chip_multidimensional.py scripts/research/backtest_blood_chip_multidimensional.py tests/research/test_blood_chip_multidimensional.py
PYTHONPATH=src python scripts/research/backtest_blood_chip_multidimensional.py
```

## Commit checkpoint

本轮不自动提交。用户确认后建议提交：

```text
feat(research): test multidimensional blood-chip survival gates
```
