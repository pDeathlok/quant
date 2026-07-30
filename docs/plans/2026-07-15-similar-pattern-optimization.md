# 相似走势优化与 2025 年后走步验证实施计划

## Goal

在不覆盖当前工作区未提交刷新改动的前提下，为相似走势决策台加入观望区、事件去重、自适应样本、非线性权重、市场与行业状态、风险闸门、分周期样本、概率校准和成本后验证，并对 `002594.SZ`、`002788.SZ` 从 `2025-01-01` 起做点时可用的走步检验。

## Pre-conditions

- [x] `git status --short` 已确认工作区存在用户的刷新任务改动；本任务只增量修改相似走势路径，不还原这些改动。
- [x] `data/raw/daily/002594.SZ.parquet`、`data/raw/daily/002788.SZ.parquet`、`data/raw/index_000300.SH.parquet` 存在。
- [x] `data/research/similar_patterns/vector_cache/c4898d1efeed/` 包含现有全市场向量缓存。
- [ ] `PYTHONPATH=src pytest -q tests/research/test_similar_patterns.py` 在修改前可运行。

## Steps

### Step 1 — 用失败测试固定优化契约

**File:** `/Users/didi/Project/quant/tests/research/test_similar_patterns.py`

增加独立测试，断言：同日/同行业重复案例被去重；每日期事件上限生效；相似度边际采用平方权重；有效样本量可计算；`45%～55%` 为观望；放量破位否决弱看涨；风险状态匹配改变权重；概率校准可序列化并插值；不同周期只使用在信号日已经成熟的案例。

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_similar_patterns.py` → 新增测试失败，失败原因是优化函数尚不存在。

### Step 2 — 实现纯算法优化层

**File:** `/Users/didi/Project/quant/src/quant/research/similar_patterns.py`

新增带完整类型标注的配置字段与纯函数：

```python
def optimize_similar_cases(
    cases: pd.DataFrame,
    config: SimilarPatternConfig,
    *,
    target_date: pd.Timestamp,
    target_industry: str,
    target_market_regime: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return deduplicated, adaptively capped cases with forecast_weight."""


def classify_forecast_signal(
    up_probability: float | None,
    snapshot: dict[str, float | str | None],
    market_regime: str,
    config: SimilarPatternConfig,
) -> dict[str, object]:
    """Return bullish, bearish, or observe after applying risk gates."""


def fit_probability_calibration(
    probabilities: list[float],
    outcomes: list[bool],
    *,
    min_samples: int = 20,
) -> dict[str, object]:
    """Fit a monotonic calibration curve or an identity fallback."""


def apply_probability_calibration(
    probability: float | None,
    calibration: dict[str, object] | None,
) -> float | None:
    """Apply a serialized interpolation curve."""
```

修改 `summarize_forecast`，当存在 `forecast_weight` 时使用该权重；没有该列时保持现有 `similarity` 行为，确保 API 向后兼容。

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_similar_patterns.py` → 全部通过。

### Step 3 — 实现点时走步验证器

**Files:**

- `/Users/didi/Project/quant/src/quant/research/similar_patterns_validation.py`
- `/Users/didi/Project/quant/scripts/research/validate_similar_patterns.py`
- `/Users/didi/Project/quant/tests/research/test_similar_patterns_validation.py`

验证器按每 5 个交易日一个锚点生成目标向量；候选案例必须满足 `candidate_date <= signal_date - horizon`，分别构造 1、20、60 日样本池。沪深300状态使用当时的 MA20、MA60、20日收益和20日波动计算；同行业案例加权，跨行业案例降权。概率校准只使用当前锚点之前已经成熟的预测。交易验证使用收盘信号、下一交易日收益、双边状态切换成本 `0.10%`，并输出覆盖率、方向准确率、Brier、成本后收益、最大回撤和相对持有收益。

**Verify:**

- `PYTHONPATH=src pytest -q tests/research/test_similar_patterns_validation.py` → 全部通过。
- `PYTHONPATH=src python scripts/research/validate_similar_patterns.py --targets 002594.SZ 002788.SZ --start-date 2025-01-01 --anchor-step 5` → 生成 CSV、JSON、Markdown 和 calibration JSON。

### Step 4 — 接入 API 与页面

**Files:**

- `/Users/didi/Project/quant/src/quant/webapp/services.py`
- `/Users/didi/Project/quant/web/index.html`
- `/Users/didi/Project/quant/web/app.js`
- `/Users/didi/Project/quant/web/styles.css`
- `/Users/didi/Project/quant/tests/test_webapp_api.py`

服务层读取验证产物中的校准曲线，对当前优化样本输出 `optimized_forecast`、`decision`、`effective_sample_size`、`market_regime`、`risk_gate` 和 `validation_summary`。页面展示校准概率、看涨/看跌/观望、有效样本、风险否决原因，以及 2025 年后验证的覆盖率、准确率和成本后表现；原始概率继续保留用于审计。

**Verify:**

- `PYTHONPATH=src pytest -q tests/test_webapp_api.py -k similar` → 相似走势 API 测试通过。
- 浏览器打开 `http://127.0.0.1:8088` 的“相似走势决策台” → 两只自选股均展示优化信号和验证摘要，控制台无新增错误。

### Step 5 — 全量回归与结果复核

**Verify:**

- `PYTHONPATH=src pytest -q tests/research/test_similar_patterns.py tests/research/test_similar_patterns_validation.py tests/test_webapp_api.py` → 全部通过。
- `ruff check src/quant/research/similar_patterns.py src/quant/research/similar_patterns_validation.py scripts/research/validate_similar_patterns.py tests/research/test_similar_patterns.py tests/research/test_similar_patterns_validation.py` → 无错误。
- 抽查每只股票首尾锚点，确认候选截止日期早于对应结果成熟日，不存在未来函数。

## Rollback

- 本任务不执行数据库迁移、外部写入或提交操作。
- 若算法测试失败，只删除本任务新增文件，并用反向补丁恢复本任务在 `similar_patterns.py`、`services.py`、`index.html`、`app.js`、`styles.css` 和测试文件中的增量块；不得使用 `git reset --hard` 或覆盖工作区已有修改。
- 验证产物位于 `reports/similar_patterns/validation_2025/`，删除该目录不会影响现有页面缓存和向量缓存。
