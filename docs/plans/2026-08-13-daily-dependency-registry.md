# 每日依赖注册表与增量刷新实施计划

## Goal

把当前分散在因子目录、策略 YAML、模型 artifact、Web 刷新步骤和例行脚本中的日更约束，收敛为一个可执行的四层合同：

`data_source -> feature -> model_score -> product`

合同必须能回答：当前生产版本实际使用什么、什么可以停更、哪些节点必须与目标交易日一致、哪些只需每日轮询或按 TTL 更新，以及一个输入变化后应增量重算哪些下游分区。

## Constraints

- 保留当前 FastAPI、页面与既有产物格式；依赖快照同时作为 preflight 规划和 postflight 强制门禁，不是只读旁路报告。
- 模型 required features 以当前 artifact 为事实来源，不在代码里复制 238 个特征名。
- 目标日使用行情刷新确认的 `expected_trade_date`，不使用系统自然日。
- 行情类节点要求 exact trade date；披露类节点要求 polled-through；周频参考库与静态模型分别按周期和 hash 管理。
- 回看窗口只决定读取历史，不等于重写历史；新增交易日默认只写目标分区。
- 当前工作区包含用户未提交改动；不重置、不覆盖不相关文件。

## Steps

### 1. 建立纯应用层依赖图

新增 `quant.application.daily_dependencies`：

- 四层节点、生命周期、频率、新鲜度与增量策略数据类；
- DAG 校验、scope 活跃根反向闭包、拓扑排序；
- `reuse / refresh / poll / refresh_if_changed / skip_unused` 规划；
- 从同一图派生页面刷新步骤，消除第二份 scope 硬编码。

### 2. 建立运行时模型合同编译器

新增 `quant.routine.daily_dependency_runtime`：

- B1/Z 从 `feature_names_in_` 读取特征；selector/Chan 从 bundle `features` 读取；
- 缓存 artifact size/mtime/hash 与逐模型特征；只有 artifact 变化才重新加载；
- 校验生产 manifest 中已有的 checksum；
- 发布 `data/contracts/daily_dependencies/latest.json`。
- 按 scope 保存独立基线，节点定义、配置文件和模型 artifact 哈希变化都会触发下游 dirty。

### 3. 固定当前生产与研究边界

- B1 147、Z 112、selector 49、Chan 46 由 artifact 动态写入快照；
- Tea/Tea-safe/v44、可转债、BYD、相似走势作为规则或运行时模型节点显式注册；
- long-entry 研究模型、tradability 研究标签、selector 训练历史与长线研究外部源标为 `research_only`，不进入生产日更闭包；
- 周频相似走势参考向量、BYD 分钟训练样本不冒充当日特征。

### 4. 接入现有流水线并减少无效更新

- 因子注册表快照同时发布依赖合同摘要；
- Web scope 进度由依赖图派生；`similar` scope 补齐行情刷新，不再绕过当日源；
- 生产例行 reference refresh 停止无活跃消费者的 tradability 与 long research external 日更；
- Chan 使用规范 `data/raw/top_list` 源；
- B1/Z 对当日候选取并集，一次计算项目因子，Z 优先复用统一缓存。
- 断点续跑消费 preflight dirty 集合；同日产物若模型、因子或策略合同变化也禁止复用，tail resume 不得绕过已变化的上游。

### 5. 验证

- DAG 错误、scope 闭包、研究依赖泄漏、增量 dirty 传播；
- 两类 artifact 特征提取、缓存命中、hash 变化与 checksum 失败；
- 页面步骤由图派生且 `similar` 包含共享行情；
- 生产快照列出 active/inactive 节点、特征使用情况、模型 hash 和增量动作；
- 定点测试、完整 pytest、compileall 与 `git diff --check`。

## Status

四层注册表、动态模型合同、scope 源裁剪、B1/Z 候选并集、Web preflight/postflight、断点失效和维护测试已落地。生产验证以 `docs/daily_dependency_registry.md` 的命令与 dated postflight 合同为准。

## Rollback

不删除旧数据产物。若运行时合同编译失败，任务必须保持 failed；修复注册/manifest 后原地增量重跑，不允许通过关闭 strict postflight 或恢复陈旧检查点绕过门禁，也不执行破坏性 Git 或数据回滚。
