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
