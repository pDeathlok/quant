# 每日刷新声明式 DAG 架构与迁移计划

## Goal

把每日刷新从 `src/quant/webapp/services.py` 中的手工编排迁移为声明式 DAG：数据、因子、模型和产品节点统一声明依赖、质量门禁、增量范围、缓存身份与资源需求；新增模型或变量只需注册契约和计算入口，不再修改中央调度代码。

## Implementation Status (2026-09-01)

- 已完成统一 operation/checkpoint/ChangeSet/resource contracts、持久化 dataset revision 和 shadow plan。
- 已完成最新日行情严格完整性、停牌证据、最近 10 日行情回补、最近 5 个交易日 daily_basic 源端复核及 17:20 前重试。
- 已完成 signal source revision 快速校验；SQL 发布前不再重读 600 天全市场，进程任务默认按 16 股票批处理。
- 已完成共享项目特征的 B1/family/Z 消费方并集；left ranker 直接消费该缓存，只读取 18 个月规则窗口，不再重算六年项目因子。
- 已完成 Chan 增量日期判定和中枢线性扫描；仍保留 800 日 warm-up 以维护结构状态。
- 已完成 signal/project-feature 与 right/left/chan 的生产 DAG adapters；10 核预算下 right(6)+chan(4) 先并行，left(2) 等资源释放。
- 已完成 selector 单次 full snapshot 构建，同时覆盖 core/extended 节点。
- `ROUTINE_DAG_EXECUTOR=shadow` 为默认；未迁移的长线、可转债、BYD、相似走势 operation 继续走现有协调器，`enabled` 在全部 adapter 完成前 fail closed。

## Non-goals

- 不改变 `python3 scripts/run_daily_web_refresh.py` 入口和 Web API。
- 不改变现有页面数据协议。
- 不在迁移阶段重训任何模型。
- 不把研究任务自动提升为生产任务。

## Existing Assets

- `src/quant/application/daily_dependencies.py` 已声明节点、依赖、freshness 和 `IncrementalPolicy`。
- `src/quant/features/factor_registry.py` 已声明变量来源、消费者和 calculator。
- `src/quant/features/factor_execution.py` 已声明 calculator 的 executor 和 worker 上限。
- `src/quant/routine/daily_dependency_runtime.py` 已能生成依赖计划和验证模型输入合同。
- 当前缺口是：上述声明没有驱动实际执行，`src/quant/webapp/services.py` 仍手工控制顺序、并发、缓存和失败处理。

## Target Architecture

```text
run_daily_web_refresh.py / Web API
              |
              v
      DailyRefreshCoordinator
              |
              +--> DependencyRegistry: 节点和依赖
              +--> OperationRegistry: 入口、资源、缓存、重试
              +--> FactorRegistry: 变量到 calculator 的映射
              +--> DatasetRevisionStore: 数据集/分区 revision
              |
              v
          DagExecutor
       /       |        \
 resource   checkpoint   quality
  pool        store       gates
       \       |        /
              v
       OperationResult + ChangeSet
              |
              v
    现有 manifest / 页面产物 / snapshot
```

## Core Contracts

### Operation contract

**New file:** `/Users/didi/Project/quant/src/quant/routine/operation_contracts.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class CacheMode(str, Enum):
    NONE = "none"
    EXACT_DATE = "exact_date"
    APPEND_STATE = "append_state"
    PARTITION_REPLACE = "partition_replace"


class ExecutionMode(str, Enum):
    INLINE = "inline"
    THREAD = "thread"
    SUBPROCESS = "subprocess"


@dataclass(frozen=True)
class ResourceClaim:
    cpu_slots: int = 1
    io_slots: int = 0
    memory_mb: int = 256
    rate_limit_group: str | None = None
    requested_workers: int = 1
    max_workers: int = 1


@dataclass(frozen=True)
class CachePolicy:
    mode: CacheMode
    contract_version: str
    output_paths: tuple[str, ...] = ()
    state_path: str | None = None


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 1
    interval_seconds: float = 0.0
    retry_until: str | None = None
    retryable_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationDefinition:
    operation_id: str
    entrypoint: str
    produces: tuple[str, ...]
    execution_mode: ExecutionMode
    resources: ResourceClaim
    cache: CachePolicy
    retry: RetryPolicy = RetryPolicy()
    enabled: bool = True


@dataclass(frozen=True)
class OperationContext:
    target_trade_date: str
    scope: str
    granted_workers: int
    upstream_results: Mapping[str, Any]
    dirty_partitions: tuple[str, ...] = ()
    dirty_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationResult:
    status: str
    node_results: Mapping[str, Mapping[str, Any]]
    changed_partitions: tuple[str, ...] = ()
    changed_keys: tuple[str, ...] = ()
    output_fingerprints: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error_category: str | None = None
    error: str | None = None
```

所有生产 operation 必须返回 `OperationResult`；不得通过读取另一个 operation 的私有临时文件推断状态。

### Quality contract

**Change file:** `/Users/didi/Project/quant/src/quant/application/daily_dependencies.py`

给 `DependencyNode` 增加 `quality: DataQualityPolicy`，默认只校验现有 freshness；生产数据源显式声明：

```python
@dataclass(frozen=True)
class DataQualityPolicy:
    fail_closed: bool = True
    expected_key_mode: str = "none"
    absence_resolver: str | None = None
    maximum_unresolved_missing: int = 0
    required_columns: tuple[str, ...] = ()
    minimum_column_coverage: tuple[tuple[str, float], ...] = ()
    require_official_or_deterministic: bool = False
```

`data.market_daily` 使用 `expected_key_mode="listed_universe"`、`absence_resolver="tushare_suspend_d"`、`maximum_unresolved_missing=0`。停牌必须有来源证据；普通缺失每 10 分钟重试到 17:20，仍缺失则整个 DAG 失败。

`data.daily_basic` 行覆盖必须和当日日行情可交易股票一致；PE、股息率等业务上允许为空的字段保留单独覆盖率，`volume_ratio` 可接受 Tushare 官方值或以 `vol / prior_5_session_mean_vol` 确定性计算的值。

### Dataset revision and ChangeSet

**New file:** `/Users/didi/Project/quant/src/quant/data/dataset_revision_store.py`

MySQL 创建并维护：

```sql
CREATE TABLE IF NOT EXISTS routine_dataset_revisions (
  dataset_id VARCHAR(128) PRIMARY KEY,
  revision BIGINT NOT NULL,
  watermark DATE NULL,
  content_sha256 CHAR(64) NOT NULL,
  updated_at DATETIME(6) NOT NULL
);

CREATE TABLE IF NOT EXISTS routine_partition_revisions (
  dataset_id VARCHAR(128) NOT NULL,
  partition_key VARCHAR(64) NOT NULL,
  revision BIGINT NOT NULL,
  row_count INT NOT NULL,
  content_sha256 CHAR(64) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  PRIMARY KEY (dataset_id, partition_key)
);
```

写入数据前后按主键比较，只有内容变化才递增 revision，并返回完整 `ChangeSet(partitions, keys)`。SQL 和 Parquet 同时写成功后才提交 revision。信号发布前只比较 revision，不再重读 600 天全市场数据。

### Cache identity

**New file:** `/Users/didi/Project/quant/src/quant/routine/checkpoint_store.py`

统一缓存键为以下规范 JSON 的 SHA256：

```json
{
  "operation_id": "feature.strategy_signals",
  "contract_version": "3",
  "target_trade_date": "2026-08-31",
  "upstream_revisions": {"data.market_daily": 1842},
  "upstream_fingerprints": {},
  "dirty_partitions": ["20260831"],
  "parameters": {"factor_mode": "stateful"},
  "code_contract_sha256": "...",
  "model_artifact_sha256": null
}
```

缓存命中必须同时满足身份一致、输出存在、输出 hash 一致、质量门禁通过。operation 不能自行定义另一套 checkpoint 判断。

### Resource scheduler

**New file:** `/Users/didi/Project/quant/src/quant/routine/dag_executor.py`

- 从 `DependencyRegistry` 取 scope 所需节点，按 `operation_id` 折叠为一次执行，天然消除一个 operation 多个产品重复计算。
- 只有所有上游 operation 成功或命中缓存后才进入 runnable queue。
- 全局资源池默认 `cpu_slots=os.cpu_count()`、`io_slots=4`、`memory_mb=系统可用内存的 70%`。
- operation 获得 `granted_workers=min(requested_workers, max_workers, 可用 cpu_slots)`；内部代码必须使用该值，禁止再次从环境变量扩张 worker。
- `rate_limit_group=tushare` 使用共享 `RequestLimiter`；AkShare 使用独立组，可以和 Tushare 并行。
- runnable operation 能满足资源声明时立即并行；资源不足时按关键路径长度、`ui_order` 排序。
- 失败后取消所有依赖后继，但允许无依赖的审计/清理 operation 完成。

### Operation registry

**New file:** `/Users/didi/Project/quant/src/quant/routine/operation_registry.py`

注册现有 operation，注册项是唯一的执行配置。`services.py` 不再分配 right/left/chan workers。

首批资源声明：

| operation | cpu | io | workers | cache |
|---|---:|---:|---:|---|
| `refresh_market_daily` | 1 | 1 | 1 | partition_replace |
| `refresh_daily_basic` | 1 | 1 | 1 | partition_replace |
| `refresh_reference_inputs` | 1 | 1 | 1 | partition_replace |
| `refresh_strategy_signals` | 8 | 1 | 8 | append_state |
| `build_project_feature_cache` | 6 | 1 | 6 | exact_date |
| `run_right_side_unified` | 6 | 1 | 6 | exact_date |
| `run_left_side_unified` | 2 | 1 | 2 | exact_date |
| `refresh_chan_model_scores` | 4 | 1 | 4 | append_state |
| `build_selector_payload` | 1 | 1 | 1 | exact_date |
| `build_long_stock_pools` | 4 | 1 | 4 | exact_date |
| `refresh_similar_patterns` | 4 | 1 | 4 | exact_date |

## Extension Standard

新增变量：

1. 在 `src/quant/features/factor_registry.py` 或受 schema 校验的扩展配置中注册 `FactorDefinition`。
2. 指定已有 `calculator_id`；需要新 calculator 时只在 `factor_execution.py` 注册一次资源和分区轴。
3. calculator 必须支持 `OperationContext.dirty_partitions/dirty_keys`。
4. `validate_factor_execution_registry()` 必须证明每个生产变量都有 calculator、PIT 安全、缓存模式和消费者。

新增模型：

1. 模型 manifest 必须声明模型 ID、artifact hash、输入变量、score schema、目标产品和 operation entrypoint。
2. `daily_dependency_runtime.resolve_model_contracts()` 自动解析变量并向上追踪 calculator。
3. operation 配置只声明资源、缓存模式和入口；DAG 自动得到依赖、并行机会和失效范围。
4. 未注册变量、无 calculator、超过资源上限、无质量证据或缓存输出不完整时启动即失败。
5. 不允许为新模型在 `services.py` 增加 `if model_enabled` 分支。

## Migration Steps

### Step 1 - Contract and executor foundation

**Files:**

- `/Users/didi/Project/quant/src/quant/routine/operation_contracts.py`
- `/Users/didi/Project/quant/src/quant/routine/operation_registry.py`
- `/Users/didi/Project/quant/src/quant/routine/checkpoint_store.py`
- `/Users/didi/Project/quant/src/quant/routine/dag_executor.py`
- `/Users/didi/Project/quant/tests/test_daily_dag_executor.py`

先写测试覆盖：依赖顺序、独立节点并行、资源不超配、同 operation 去重、缓存身份、失败取消和 ChangeSet 传播。执行器先以 shadow 模式只输出计划，不写生产产物。

**Verify:**

```bash
pytest -q tests/test_daily_dag_executor.py tests/test_daily_dependency_runtime.py tests/test_factor_execution.py
```

Expected: 全部通过，且测试证明总并发 CPU claim 永不超过配置预算。

### Step 2 - Source quality and revision

**Files:**

- `/Users/didi/Project/quant/src/quant/routine/data_refresh.py`
- `/Users/didi/Project/quant/src/quant/routine/daily_basic_refresh.py`
- `/Users/didi/Project/quant/src/quant/routine/pipeline.py`
- `/Users/didi/Project/quant/src/quant/data/market_data_store.py`
- `/Users/didi/Project/quant/src/quant/data/dataset_revision_store.py`
- `/Users/didi/Project/quant/tests/test_data_refresh.py`
- `/Users/didi/Project/quant/tests/test_daily_basic_refresh.py`
- `/Users/didi/Project/quant/tests/test_dataset_revision_store.py`

日行情每天重取最近 10 个自然日，最新日只允许已确认停牌解释缺失；`daily_basic` 强制重取最近 5 个交易日，每周任务核对最近 120 天。变化日期和股票写入 `ChangeSet`。将 postflight 从只校验 watermark 改为同时校验 quality evidence。

**Verify:**

```bash
pytest -q tests/test_data_refresh.py tests/test_daily_basic_refresh.py tests/test_tushare_availability.py tests/test_dataset_revision_store.py tests/test_daily_dependency_runtime.py
```

Expected: 未解释缺失必须失败；确认停牌成功；历史修订递增 revision；相同数据重跑不递增 revision。

### Step 3 - Feature and model migration

**Files:**

- `/Users/didi/Project/quant/src/quant/routine/left_side_unified_production.py`
- `/Users/didi/Project/quant/scripts/research/refresh_b1_feature_cache.py`
- `/Users/didi/Project/quant/scripts/research/rebuild_strategy_signal_cache.py`
- `/Users/didi/Project/quant/scripts/research/refresh_chan_model_live_scores.py`
- `/Users/didi/Project/quant/src/quant/features/daily_factor_layer.py`
- `/Users/didi/Project/quant/tests/test_left_side_unified_production.py`
- `/Users/didi/Project/quant/tests/test_strategy_signal_cache_incremental.py`
- `/Users/didi/Project/quant/tests/test_research_market_backtest.py`

项目因子缓存候选集合改为所有下游消费者并集；左侧直接连接精确日期缓存，只计算自身 27 个规则变量并对模型输入做非空质量校验。信号和缠论都采用每股票 append state；历史 `ChangeSet` 只使受影响股票从最早变化日期重算。输出按年/日期分区，latest manifest 提供统一读取入口。

**Verify:**

```bash
pytest -q tests/test_left_side_unified_production.py tests/test_strategy_signal_cache_incremental.py tests/test_research_market_backtest.py tests/test_project_factor_layer.py
```

Expected: 左侧 `daily_basic` 主字段不再全空；不变前缀只处理最新一行；历史变更只重算相关股票。

### Step 4 - Product migration and duplicate removal

**Files:**

- `/Users/didi/Project/quant/src/quant/webapp/services.py`
- `/Users/didi/Project/quant/src/quant/routine/web_refresh_runner.py`
- `/Users/didi/Project/quant/src/quant/application/refresh_contracts.py`
- `/Users/didi/Project/quant/tests/test_webapp_api.py`
- `/Users/didi/Project/quant/tests/test_selector_production_postflight.py`

selector 只生成一次完整 payload，再由同一不可变结果派生 core、extended 和 snapshot。Web 服务仅订阅 executor 进度事件，不再包含任务顺序和 worker 分配。保留 `ROUTINE_DAG_EXECUTOR=legacy|shadow|enabled`，默认先 `shadow`。

**Verify:**

```bash
pytest -q tests/test_webapp_api.py tests/test_selector_production_postflight.py tests/test_refresh_contracts.py
```

Expected: 页面协议不变；selector 构建调用一次；进度步骤顺序仍符合合同。

### Step 5 - Cutover and benchmark

1. `ROUTINE_DAG_EXECUTOR=shadow python3 scripts/run_daily_web_refresh.py`：legacy 负责写入，DAG 只比较计划、缓存决策和质量证据。
2. 连续三个交易日要求所有节点的目标日期、行数、hash 和产品数量一致。
3. `ROUTINE_DAG_EXECUTOR=enabled python3 scripts/run_daily_web_refresh.py`：DAG 写入生产产物。
4. 记录每个 operation 的 wall time、CPU seconds、读写字节、cache mode、dirty key 数量。

**Acceptance:**

- 未解释日行情缺失为 0；否则整体失败并保留上一份成功快照。
- `daily_basic` 最近 5 个交易日确实访问源端，并报告变化日期。
- 左侧 89 个候选中业务可得的 turnover、volume ratio、市值字段有值，模型输入覆盖门禁通过。
- SQL 信号发布前不再发生 600 天全市场二次读取。
- selector full build 每次刷新只执行一次。
- 10 核机器任何时刻已授予 CPU slots 不超过 10。
- 不变数据同日重跑除质量检查外全部命中 checkpoint。
- 正常新增交易日只重算新增分区；历史修订只重算受影响 keys。
- 稳定交易日全流程目标耗时不高于 30 分钟，目标值 25 分钟。

## Rollback

- 设置 `ROUTINE_DAG_EXECUTOR=legacy` 立即恢复旧调度路径。
- 新 revision 表只保存元数据，回滚执行器时无需删除。
- 每个 operation 发布前保留现有原子写和 last-known-good；失败不能覆盖成功产物。
- 任何节点发生新旧结果差异时，保留 shadow diff，停止该节点切换，不影响已迁移且通过合同的节点。
