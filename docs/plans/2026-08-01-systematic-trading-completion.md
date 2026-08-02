# 系统化交易能力补全实施计划

## Goal

在不引入第二套数据、回测或工作流框架的前提下，补齐 A 股可交易性、组合构建、防泄漏验证、因子归因、风险与容量、研究审计、模拟交易和统一市场状态，并将需要更新的数据接入现有每日 Tushare 流水线。

## Pre-conditions

- [x] 仓库根目录为 `/Users/didi/Project/quant`。
- [x] `pyproject.toml` 已包含 Pandas、NumPy、SciPy、scikit-learn、Tushare、SQLAlchemy 和 akquant，本计划核心实现不增加运行时依赖。
- [x] 任务开始前已有 Web 修改位于 `src/quant/webapp/services.py`、`tests/test_web_frontend.py`、`tests/test_webapp_api.py`、`web/app.js`、`web/index.html`，本计划不覆盖这些修改。
- [x] `PYTHONPATH=src pytest -q` 在前序阶段输出 `384 passed`。
- [ ] 正式执行在线每日刷新时必须设置 `TUSHARE_TOKEN`；单元测试使用 FakePro，不访问网络。

## Steps

### Step 1 — 建立每日 A 股可交易性数据集

**Files:**

- `/Users/didi/Project/quant/src/quant/data/tradability.py`
- `/Users/didi/Project/quant/src/quant/routine/reference_data_refresh.py`
- `/Users/didi/Project/quant/tests/test_tradability.py`
- `/Users/didi/Project/quant/tests/test_reference_data_refresh.py`

项目数据集固定为一日一个 Parquet 分区：

```python
TRADABILITY_COLUMNS = (
    "trade_date",
    "ts_code",
    "pre_close",
    "up_limit",
    "down_limit",
    "is_suspended",
    "is_st",
    "st_type",
    "list_date",
    "market",
)

def build_daily_tradability(
    *,
    trade_date: str,
    stock_basic: pd.DataFrame,
    limits: pd.DataFrame,
    suspensions: pd.DataFrame,
    st_stocks: pd.DataFrame,
    minimum_coverage_rate: float = 0.98,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate, join and audit one point-in-time A-share trading universe."""
```

`refresh_reference_data()` 在刷新 `stock_basic` 后调用 Tushare `stk_limit(trade_date=...)`、`suspend_d(suspend_type="S", trade_date=...)` 和 `stock_st(trade_date=...)`，原子写入 `data/raw/tradability/YYYYMMDD.parquet`。相同交易日重跑覆盖同一分区，不追加重复记录；覆盖率低于 `ROUTINE_TRADABILITY_MIN_COVERAGE_RATE=0.98` 时步骤失败并阻断下游。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_tradability.py tests/test_reference_data_refresh.py
```

预期：构建、空停牌/ST、重复行、低覆盖率、同日重跑和 manifest 测试全部通过。

### Step 2 — 将可交易性接入撮合前校验

**Files:**

- `/Users/didi/Project/quant/src/quant/backtest/tradability.py`
- `/Users/didi/Project/quant/src/quant/backtest/engine.py`
- `/Users/didi/Project/quant/src/quant/backtest/artifacts.py`
- `/Users/didi/Project/quant/tests/test_backtest_tradability.py`

实现引擎无关判定：

```python
@dataclass(frozen=True)
class TradabilityDecision:
    allowed: bool
    reason: str | None = None

class AShareTradabilityPolicy:
    def check_order(
        self,
        *,
        trade_date: str,
        symbol: str,
        side: str,
        price: float,
    ) -> TradabilityDecision:
        """Reject suspended orders and buys/sells pinned at the relevant limit."""
```

由于底层 akquant 当前没有稳定的自定义撮合回调，第一阶段由项目策略适配层提供显式判定，并将数据覆盖范围写入 `BacktestArtifacts.metadata["tradability"]`；不得通过删除行情行伪造停牌或成交。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_backtest_tradability.py tests/test_a_share_execution.py tests/test_backtest_engine_contract.py
```

### Step 3 — 建立项目自有组合构建契约

**Files:**

- `/Users/didi/Project/quant/src/quant/portfolio/__init__.py`
- `/Users/didi/Project/quant/src/quant/portfolio/construction.py`
- `/Users/didi/Project/quant/src/quant/portfolio/orders.py`
- `/Users/didi/Project/quant/tests/test_portfolio_construction.py`

固定接口：

```python
@dataclass(frozen=True)
class PortfolioConstraints:
    max_weight: float = 0.10
    max_industry_weight: float = 0.30
    max_turnover: float = 1.00
    cash_buffer: float = 0.02
    min_position_weight: float = 0.00
    long_only: bool = True

@dataclass(frozen=True)
class PortfolioResult:
    target_weights: pd.Series
    cash_weight: float
    turnover: float
    diagnostics: Mapping[str, object]

class PortfolioConstructor:
    def equal_weight(self, candidates: pd.DataFrame) -> PortfolioResult: ...
    def score_weight(self, candidates: pd.DataFrame) -> PortfolioResult: ...
    def inverse_volatility(self, returns: pd.DataFrame) -> PortfolioResult: ...
    def minimum_variance(self, returns: pd.DataFrame) -> PortfolioResult: ...
```

生产代码使用 SciPy SLSQP 并在求解后重新校验约束。`target_weights_to_orders()` 使用最新价格、现有持仓、资金和 `lot_size=100` 生成整手差额订单；先卖后买，不超现金。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_portfolio_construction.py
```

### Step 4 — 建立防泄漏验证协议

**Files:**

- `/Users/didi/Project/quant/src/quant/research/validation.py`
- `/Users/didi/Project/quant/src/quant/backtest/optimizer.py`
- `/Users/didi/Project/quant/tests/test_research_validation.py`

固定接口：

```python
@dataclass(frozen=True)
class TimeSplit:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

class PurgedWalkForwardSplitter:
    def __init__(
        self,
        *,
        train_periods: int,
        test_periods: int,
        purge_periods: int = 0,
        embargo_periods: int = 0,
        expanding: bool = False,
    ) -> None: ...

    def split(self, dates: Sequence[object]) -> list[TimeSplit]: ...
```

输入日期先去重排序；训练结束与测试开始之间保留 purge，测试结束后保留 embargo；任何折叠不得重叠。旧 `WalkForwardOptimizer` 改用此 splitter 和项目稳定绩效契约。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_research_validation.py tests/test_backtest_engine_contract.py
```

### Step 5 — 补齐因子评价与组合归因

**Files:**

- `/Users/didi/Project/quant/src/quant/analysis/factors.py`
- `/Users/didi/Project/quant/src/quant/analysis/attribution.py`
- `/Users/didi/Project/quant/tests/test_factor_analysis.py`
- `/Users/didi/Project/quant/tests/test_attribution.py`

`FactorAnalyzer` 输出截面 Pearson IC、Rank IC、分位数组合收益、多空收益、换手和 IC 衰减；所有 forward return 必须按 symbol 向后移动构建。`AttributionAnalyzer` 增加行业贡献、因子暴露、选择效应、配置效应和成本后贡献，并保持现有方法兼容。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_factor_analysis.py tests/test_attribution.py tests/test_performance_analysis.py
```

### Step 6 — 补齐市场状态、风险、压力和容量

**Files:**

- `/Users/didi/Project/quant/src/quant/features/market_regime.py`
- `/Users/didi/Project/quant/src/quant/risk/portfolio.py`
- `/Users/didi/Project/quant/src/quant/risk/manager.py`
- `/Users/didi/Project/quant/src/quant/risk/limits.py`
- `/Users/didi/Project/quant/tests/test_market_regime.py`
- `/Users/didi/Project/quant/tests/test_portfolio_risk.py`

市场状态只使用截至当日的 Tushare 指数、全市场宽度、波动率和流动性，输出 `risk_on/neutral/risk_off` 与证据字段。风险模块输出集中度、行业暴露、历史 VaR/CVaR、压力损失、持仓可变现天数和成交额参与率；`RiskManager.pre_order_check()` 实际执行总杠杆、单股、日亏损、回撤和成交量限制，拒绝原因可审计。

每日流水线在参考数据完成后生成 `data/features/market_regime/latest.json` 和日期快照；相同输入产生相同结果。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_market_regime.py tests/test_portfolio_risk.py tests/test_quant.py
```

### Step 7 — 统一研究运行审计

**Files:**

- `/Users/didi/Project/quant/src/quant/research/manifest.py`
- `/Users/didi/Project/quant/tests/test_research_manifest.py`

固定不可变清单：

```python
@dataclass(frozen=True)
class ResearchRunManifest:
    run_id: str
    created_at: str
    strategy: str
    data_as_of: str
    parameters: Mapping[str, object]
    random_seed: int | None
    code_revision: str | None
    execution_policy: Mapping[str, object]
    metrics: Mapping[str, object]
    artifacts: Mapping[str, str]
    artifact_sha256: Mapping[str, str]
```

写入采用临时文件原子替换；哈希从实际产物计算。不能读取 Git 时 `code_revision=None`，但不得伪造版本。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_research_manifest.py
```

### Step 8 — 重构模拟交易与日终对账

**Files:**

- `/Users/didi/Project/quant/src/quant/trading/broker.py`
- `/Users/didi/Project/quant/src/quant/trading/order_manager.py`
- `/Users/didi/Project/quant/src/quant/trading/reconciliation.py`
- `/Users/didi/Project/quant/tests/test_simulated_broker.py`

`SimulatedBroker` 使用注入价格估值，不再以固定 100 元估值；成交使用 `AShareExecutionConfig` 计算佣金、印花税、过户费，执行整手与 T+1，订单和账户状态可原子持久化。日终对账比较 broker 持仓、订单成交和目标组合，输出现金、持仓数量、成本与差异。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_simulated_broker.py tests/test_a_share_execution.py tests/test_portfolio_construction.py
```

### Step 9 — 文档、每日刷新和完整验收

**Files:**

- `/Users/didi/Project/quant/README.md`
- `/Users/didi/Project/quant/docs/architecture.md`
- `/Users/didi/Project/quant/docs/operations.md`
- `/Users/didi/Project/quant/.env.example`

文档记录数据来源、刷新顺序、覆盖率门禁、回填方法、研究协议和模拟交易限制。新增配置：

```dotenv
ROUTINE_TRADABILITY_MIN_COVERAGE_RATE=0.98
ROUTINE_TRADABILITY_RETRIES=3
ROUTINE_MARKET_REGIME_LOOKBACK_DAYS=252
```

**Verify:**

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m compileall -q src tests scripts
git diff --check
```

预期：全部测试通过，compileall 和 diff check 返回 0；真实 Tushare 刷新仅在用户环境已有 Token 和权限时执行。

## Commit checkpoints

本任务不自动提交。建议按以下边界提交：

```text
feat(data): refresh daily a-share tradability inputs
feat(portfolio): add constrained portfolio construction
feat(research): add purged validation and factor diagnostics
feat(risk): integrate portfolio risk and market regimes
feat(trading): harden simulated broker and reconciliation
```

## Rollback

- 本计划不执行数据库迁移、删除或外部写入。
- 新数据集是 `data/raw/tradability/YYYYMMDD.parquet` 和 `data/features/market_regime/*.json`；需要回滚代码时可停止生成，既有正式日线和 `daily_basic` 不受影响。
- 删除新增 Python 模块与对应测试，再恢复对 `reference_data_refresh.py`、`pipeline.py`、`risk`、`trading`、导出文件和文档的局部修改即可。
- 不使用 `git reset --hard`、`git checkout --` 或覆盖任务开始前的 Web 修改。
