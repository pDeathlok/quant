# 因子计算 DAG 与并行治理

## 目标

因子业务含义、公式所有权、当前消费者和刷新生命周期是四个独立维度。任何策略都可以声明任意规范因子为依赖；`active_consumers` 只描述当前事实，不是访问白名单。

## 计算器注册表

`factor_registry.py` 管理每个因子的业务元数据，`factor_execution.py` 管理计算器级依赖与执行能力，`configs/factors/governance.json` 管理可变运行策略。

每个计算器声明 `produces`、`dependencies`、`factor_dependencies`、`partition_axis`、`executor`、worker 上限、物化方式和每日 DAG 节点。生产日更从所有 `refresh_cadence=trade_daily` 因子反向编译计算器闭包并进行拓扑排序。

新增因子若由已有计算器产出，只需注册字段并通过 `factor_overrides` 晋级；新公式仍必须实现或注册一个计算器，不能只靠配置凭空生成。

## 重复计算治理

右侧规则原先再次计算 `pct_chg`、`amplitude_1`、`volume_relative_5d`、`volume_relative_20d` 和 `kdj_d_j`。生产入口现在先运行项目计算器，再把这五个规范序列传给右侧规则计算器；standalone 研究调用保留本地回退，两条路径必须通过逐字段 parity 测试。

名称不同但口径相似的 selector/Chan 字段暂不直接合并。它们的窗口、候选股票池和横截面时点未全部相同，未经输入、窗口、复权、空值和边界 parity 证明，不得仅按名称判断为重复。

### 排序模型的共享执行契约

注册表回答“特征是什么、由谁负责”，不等于运行时已经去重。左侧 RankNet 与组合策略曾因缓存键包含整份股票和列清单，重复处理重叠股票。现在生产消费者只调用 [共享特征入口](../src/quant/routine/rank_feature_runtime.py)，声明目标日、股票、所需列，并传入调度器授予的 worker 数，不配置自己的缓存目录。

| 输出口径 | 放置和复用规则 |
|---|---|
| 单股历史、固定全市场横截面 | 共享块。键包含目标交易日、封存输入身份、公式与上下文；对已物化股票取所需列，只计算新增股票。横截面分母仍是完整市场，不是请求股票列表。 |
| 旧指标的窗口或锚点不同 | 独立语义和计算块。左侧旧版兼容列按需物化，不覆盖规范因子。 |
| 依赖候选池、策略分组或模型评分的排名 | 在对应评分层计算，不进入上述共享块。如果要跨消费者共享，必须先将总体及分组身份纳入计算契约，并验证等价。 |

生产日更必须使用封存版本；独立研究脚本可指定数据源和缓存，不得给生产补值。源文件修订、公式变更或缓存校验失败不能命中旧块。缓存只保存可重建的特征块，不替代发布快照；日常清理删除早于基准日减 14 个自然日的缓存，跳过持有计算锁的日期。

### 新模型和变量接入

1. 查注册表与现有计算器：复用规范列；仅在输入、时点、复权、窗口、总体或空值规则确有差异时定义新特征。
2. 在模型 artifact 声明所需列，在生产 DAG 声明输入依赖、计算归属和资源预算。新增排序特征应由共享运行时或注册计算器产出，评分层只取列并评分；禁止从页面请求触发历史计算。
3. 验证至少两组不同的候选范围和列集合：重叠股票不再调用同一计算器，新股票按缺口补算；投影与各自独立运行逐项一致（含 NaN、边界和横截面）；源修订和文件损坏能使旧缓存失效。
4. 在例行运行结果检查 `shared_blocks.<block>.computed_symbols` 和 `reused_symbols`，记录变化并比较步骤耗时；不能只报全局“缓存命中”。发布仍以同日精确覆盖和 postflight 为准。

本地验证：

```bash
PYTHONPATH=src python -m pytest -q tests/test_architecture_boundaries.py tests/test_rank_feature_snapshot.py
```

Source Quality CI 检查生产导入边界，并用封存测试数据验证复用和失效。导入检查不能识别用不同函数重写的近似公式，评审仍须核对归属和数值等价。

## 并行方案 review

- 按股票的 Pandas/NumPy 技术计算属于 CPU 密集任务，继续使用进程池。
- 数据下载、独立文件读取适合线程池；共享横截面排序必须在完整截面上执行，不按股票拆散。
- 日频全市场刷新由一次性提交全部 Future 改为 `workers * max_pending_multiplier` 有界提交，降低大股票池的父进程内存与调度开销。
- Web 日刷的模型评分、Chan 和右侧计算共享 `ROUTINE_CPU_WORKER_BUDGET`，两个并行阶段分别分配预算，避免各任务独立按上限启动造成 CPU 过量竞争。
- 默认预算和各计算器上限来自 `configs/factors/governance.json`；环境变量只用于单次部署覆盖。

## 配置变更

将研究候选晋级日更可在 `factor_overrides` 设置 `lifecycle`、`refresh_cadence` 和 `active_consumers`。计算器必须已启用并具有 `routine_node`，否则静态校验拒绝启动。调整进程池只修改对应 `calculators.<id>`；全局预算修改 `execution.global_cpu_worker_budget`。

## 线上旧别名审计

2026-08-25 的活动模型检查结果：右侧 `right_side_unified_canonical_v2` 和左侧 `left_side_unified_canonical_v4_group4` 不含历史别名。`models/production/b1/` 的五个旧模型仍含 `price_level`、`bb_middle`，旧 `models/production/right_side_unified/` 也仍含五个历史别名；两者都已退出活动 DAG，仅保留作受控回滚，当前入口分别使用左侧统一排序和 canonical 右侧模型。任何旧 artifact 重新接入前都必须先通过禁止别名门禁。
