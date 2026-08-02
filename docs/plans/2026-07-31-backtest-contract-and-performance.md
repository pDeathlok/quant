# 统一回测契约与绩效分析实施计划

## Goal

在不改变 `BacktestEngine.run()` 原始返回值的前提下，将 akquant 的净值、收益、持仓、订单、成交、交易、成本和基准收益适配为项目自有的稳定契约，并让绩效指标统一从该契约计算。

## Pre-conditions

- [x] `python -c 'import akquant; print(akquant.__version__)'` 输出 `0.2.36`。
- [x] `git status --short` 显示的既有未提交修改仅位于 Web 相关文件，本计划不修改这些文件。
- [x] `BacktestResult` 提供 `equity_curve_daily`、`daily_returns`、`positions_df`、`orders_df`、`executions_df` 和 `trades_df`。

## Steps

### Step 1 — 定义项目自有回测结果契约

**File:** `/Users/didi/Project/quant/src/quant/backtest/artifacts.py`

实现不可变的 `BacktestArtifacts` 数据类，字段固定为：

```python
@dataclass(frozen=True)
class BacktestArtifacts:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    orders: pd.DataFrame
    executions: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.DataFrame
    benchmark_returns: Optional[pd.Series] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_akquant(
        cls,
        result: Any,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> "BacktestArtifacts":
        equity_curve = _series_attribute(result, "equity_curve_daily", "equity")
        returns = _series_attribute(result, "daily_returns", "return")
        if not equity_curve.empty:
            returns = equity_curve.pct_change(fill_method=None).fillna(0.0)
            returns.name = "return"
        executions = _frame_attribute(result, "executions_df")
        orders = _frame_attribute(result, "orders_df")
        return cls(
            equity_curve=equity_curve,
            returns=returns,
            positions=_frame_attribute(result, "positions_df"),
            orders=orders,
            executions=executions,
            trades=_frame_attribute(result, "trades_df"),
            costs=_build_cost_ledger(executions, orders),
            benchmark_returns=benchmark_returns,
            metadata={"initial_cash": float(getattr(result, "initial_cash", 0.0) or 0.0)},
        )
```

构造阶段复制全部 Pandas 对象、排序并去重时间索引，将 metadata 转为只读映射。成本台账固定包含 `timestamp`、`order_id`、`symbol`、`commission`、`stamp_tax`、`transfer_fee`、`slippage_cost` 和 `total_cost`；缺失的成本分项填 0，不从成交价差猜测滑点。

**Verify:** `PYTHONPATH=src pytest -q tests/test_backtest_artifacts.py` → 适配、复制隔离、成本合计和基准保留测试全部通过。

### Step 2 — 扩充绩效分析且区分收益期数与交易笔数

**File:** `/Users/didi/Project/quant/src/quant/analysis/performance.py`

保留 `PerformanceAnalyzer(returns, benchmark=None)` 的兼容调用，增加 `trades`、`costs`、`periods_per_year` 和 `risk_free_rate` 参数，并提供契约构造器：

```python
@classmethod
def from_artifacts(cls, artifacts: Any) -> "PerformanceAnalyzer":
    return cls(
        returns=artifacts.returns,
        benchmark=artifacts.benchmark_returns,
        trades=artifacts.trades,
        costs=artifacts.costs,
    )
```

`summary()` 固定输出复合收益、年化收益、年化波动率、Sharpe、Sortino、Calmar、最大回撤、最大回撤持续期、正收益期占比、收益期数、真实交易笔数、期间盈亏因子、95% VaR/CVaR 和总成本。存在基准时额外输出基准收益、超额收益、跟踪误差、信息比率、Beta 和年化 Alpha。

`trade_count` 只等于交易表行数；没有交易表时为 0，不再用收益序列长度冒充交易笔数。为兼容旧消费者，继续输出 `volatility` 和 `win_rate` 两个别名。

**Verify:** `PYTHONPATH=src pytest -q tests/test_performance_analysis.py` → 指标公式、空输入、真实交易数和基准对齐测试全部通过。

### Step 3 — 将稳定契约接入 BacktestEngine

**File:** `/Users/didi/Project/quant/src/quant/backtest/engine.py`

`run()` 继续返回 akquant `BacktestResult`，成功后同步设置 `self._artifacts`：

```python
self._result = result
self._artifacts = BacktestArtifacts.from_akquant(
    result,
    benchmark_returns=benchmark,
)
```

新增只读 `artifacts` 属性和 `get_artifacts()`；`get_metrics()` 改为 `PerformanceAnalyzer.from_artifacts(self._artifacts).summary()`。报告生成仍委托 akquant，保持现有 CLI 行为。

**Files:**

- `/Users/didi/Project/quant/src/quant/backtest/__init__.py`
- `/Users/didi/Project/quant/src/quant/__init__.py`

导出 `BacktestArtifacts`，供研究脚本和后续组合层复用。

**Verify:** `PYTHONPATH=src pytest -q tests/test_backtest_engine_contract.py tests/test_quant.py` → 原始结果返回兼容、契约可访问和标准指标测试全部通过。

### Step 4 — 文档化稳定输出

**Files:**

- `/Users/didi/Project/quant/README.md`
- `/Users/didi/Project/quant/docs/architecture.md`

记录回测结果的两层接口：`run()` 返回底层结果以兼容现有策略，`engine.artifacts` 是应用和研究脚本应依赖的项目稳定契约；说明期间胜率与逐笔胜率是不同口径。

**Verify:** `rg -n 'BacktestArtifacts|收益期数|交易笔数' README.md docs/architecture.md` → 两份文档均能定位到新契约说明。

### Step 5 — 完整验证

按顺序执行：

```bash
PYTHONPATH=src pytest -q tests/test_backtest_artifacts.py tests/test_performance_analysis.py tests/test_backtest_engine_contract.py tests/test_quant.py
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m compileall -q src tests
ruff check src tests
```

预期：新增定向测试全部通过；全量测试无新增失败；compileall 无输出并返回 0；ruff 无新增诊断。

## Commit checkpoint

本轮不自动提交。用户确认后可使用提交信息：

```text
feat(backtest): standardize artifacts and performance metrics
```

## Rollback

本轮不包含数据库迁移、删除或外部写入。需要回滚时，仅删除新测试、新计划和 `src/quant/backtest/artifacts.py`，再恢复本轮对 `engine.py`、导出文件和文档的局部修改；不得覆盖任务开始前已有的 Web 工作区修改。
