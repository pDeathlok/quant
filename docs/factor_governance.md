# 因子治理与生命周期

本文定义项目全部因子的统一治理合同。目标不是把所有公式塞进同一个函数，而是让任何被生产计算、模型消费、页面展示或研究复用的字段，都能回答以下问题：它是什么、由谁计算、当前是否上线、多久更新、如何变更、何时可以停用。

## 当前治理基线

注册表 schema 为 `factor_registry_v2_governed`，当前共有 628 条记录。

| 维度 | 状态 | 数量 | 含义 |
| --- | --- | ---: | --- |
| 角色 | `feature` | 610 | 有独立语义的规范因子 |
| 角色 | `compatibility_alias` | 0 | 旧名称消费者已清零并从注册表删除 |
| 角色 | `strategy_identity` | 18 | 左/右策略身份字段 |
| 生命周期 | `production_model` | 236 | 被当前生产模型直接消费 |
| 生命周期 | `production_materialized` | 218 | 被生产规则、页面或快照直接物化 |
| 生命周期 | `research_candidate` | 156 | 已登记且可研究，未承诺生产日更 |
| 生命周期 | `compatibility_alias` | 0 | 当前没有活动兼容别名 |
| 生命周期 | `strategy_identity` | 18 | 随对应策略产物管理 |

刷新节奏共有 472 条 `trade_daily` 记录和 156 条 `on_demand` 记录。刷新节奏是声明，活动 DAG 才是执行依据；两者不一致时生产门禁必须失败，而不是静默漏算。

## 三个事实来源

| 事实来源 | 文件或产物 | 负责回答 |
| --- | --- | --- |
| 字段治理 | `src/quant/features/factor_registry.py` | 因子名称、规范映射、层级、公式入口、版本、生命周期和消费者 |
| 执行依赖 | `src/quant/application/daily_dependencies.py` | 哪些活动因子必须在目标交易日刷新，以及依赖哪些数据和计算器 |
| 模型输入 | 当前晋级模型 artifact | 当前模型真实需要哪些输入列、schema 和 artifact hash |

注册表不能替代执行 DAG，DAG 也不能引用未注册字段。模型 artifact 与注册表不一致时必须阻断评分或发布。

## 计算层级

| 计算层 | 注册记录数 | 责任边界 |
| --- | ---: | --- |
| `project_daily` | 145 | 规范化后的项目级日频量价与技术因子 |
| `project_daily_candidate` | 22 | 已有公式或明确口径、尚未晋级的日频候选 |
| `right_side_rule` | 113 | 右侧和混合策略独有规则字段；重复项目因子已直接复用 |
| `right_side_identity` | 14 | 右侧规则产物的身份字段 |
| `left_side_rule` | 27 | 四组左侧策略独有规则字段；另复用 2 个右侧基础字段 |
| `left_side_identity` | 4 | B1、SB1、SUPER_B1、LOW_PULLBACK 身份字段 |
| `selector_live` | 49 | selector 生产模型输入 |
| `chan_live` | 42 | Chan 独有生产输入；另有 4 个复用规范因子，所以模型合同共 46 列 |
| `long_snapshot` | 78 | 长线生产快照独有字段；另有 4 个复用规范因子，所以快照合同共 82 列 |
| `long_research` | 108 | 长线注册池中未进入生产快照的研究字段 |
| `long_external_candidate` | 26 | 分析师及外部事件类按需研究字段 |

同一规范因子可以被多个层复用，但注册表只保留一条记录，通过 `consumers` 描述使用方，避免按策略重复发明字段。

## 注册记录必填字段

| 字段 | 规则 |
| --- | --- |
| `name` | 全局唯一、稳定、可读的机器名 |
| `canonical_name` | 规范因子指向自身；兼容别名指向唯一规范因子 |
| `role` | `feature`、`compatibility_alias` 或 `strategy_identity` |
| `lifecycle` | 当前生命周期，决定是否允许进入生产合同 |
| `family` | 业务类别，例如价格结构、量能、估值、质量、分析师预期 |
| `layer` | 唯一主要计算层；跨层复用写入 `consumers` |
| `source` | 原始数据来源或上游语义来源 |
| `calculation_entrypoint` | 可定位的计算函数、服务入口或身份来源 |
| `calculation_version` | schema、materialization 或运行时解析版本 |
| `frequency` | 因子观察频率，例如 daily 或 weekly |
| `refresh_cadence` | `trade_daily` 或 `on_demand` |
| `point_in_time` | 是否满足信号时点可得性；生产因子必须为 true |
| `consumers` | 当前模型、策略、页面或研究消费者 |

## 生命周期状态机

```text
research_candidate
    | 生产规则/页面晋级
    v
production_materialized

research_candidate
    | 模型训练、评估、晋级
    v
production_model

production_model / production_materialized
    | 清空生产消费者并退出活动 DAG
    v
research_candidate

重复规范因子
    | 统一口径并迁移消费者
    v
compatibility_alias
    | 兼容窗口结束且消费者为零
    v
从注册表删除
```

`lifecycle` 是单值主状态。字段同时被模型和快照使用时，以 `production_model` 为主状态，并在 `consumers` 和执行 DAG 中保留全部生产依赖。

## 生命周期迁移门禁

### 候选晋级为生产物化

必须同时满足：

1. 数据源在生产环境可稳定获取，缺失、停牌、上市初期和修订行为有明确处理；
2. 公式只有一个规范计算入口，并固定 `calculation_version`；
3. 点时安全、回看窗口、复权规则、空值语义和边界条件已写入测试；
4. 目标交易日覆盖率、唯一键和合法空结果可被 manifest 验证；
5. 计算器和配置进入 `feature.*.contract_sources`；
6. 字段进入活动产品闭包，严格 postflight 能证明目标日已更新；
7. 生命周期改为 `production_materialized`，刷新节奏改为 `trade_daily`。

### 候选晋级为生产模型

除满足生产物化的数据和计算门禁外，还必须：

1. 使用同一 schema 重新训练和评估模型；
2. artifact manifest 固定输入列、schema、训练截止日、hash 和 release ID；
3. 模型覆盖、分布、缺列和 schema 不兼容测试通过；
4. 先发布新 artifact，再切换活动消费者；
5. 生命周期改为 `production_model`，不得让旧模型静默消费新公式。

### 生产因子降级为研究候选

必须按顺序执行：

1. 从模型 artifact、策略配置、页面和快照合同中移除全部生产消费者；
2. 从活动 DAG 反向闭包中移除对应依赖；
3. 证明没有其他生产字段共用该节点，或只裁掉可安全停止的分支；
4. 生命周期改为 `research_candidate`，刷新节奏改为 `on_demand`；
5. 完成一次严格 postflight，确认生产输出没有缺列或旧消费者。

禁止仅凭“某个模型没使用”停更因子，因为规则策略、页面和其他模型可能仍在消费。

### 重复因子迁移为兼容别名

确认两个字段在输入、窗口、复权、空值、精度和边界条件上完全一致后：

1. 选择稳定且语义清晰的 `canonical_name`；
2. 所有计算入口只产生规范字段；
3. 旧字段改为 `compatibility_alias`，运行时仅做映射；
4. 迁移模型、策略和页面消费者到规范字段；
5. 至少保留一个生产发布周期；消费者归零后方可删除别名。

名称相似但口径不同的字段不得合并，必须通过前缀、窗口或版本体现差异。

## 当前兼容别名

当前数量为 0。五个历史兼容名称的活动消费者已经清零，注册表只保留规范名称。旧 Parquet 只允许在数据读取边界执行一次“逐值及 NaN 位置一致性校验—改名—删除旧列”；训练、评分、artifact、策略配置和页面合同均禁止继续携带旧名称。

## 每日更新保障

生产日更遵循以下闭环：

1. 运行时从活动模型 artifact、策略配置和产品 root 编译活动依赖闭包；
2. 闭包中的生产字段必须全部存在于治理注册表；
3. `trade_daily` 因子必须发布目标交易日证据，不能只检查文件存在或 mtime；
4. 长线快照严格要求 `long-page-v2-governed-82` 的 82 个生产因子，缺一列即拒绝发布；
5. 注册表、计算器、配置或 artifact hash 变化会使相应节点及下游变脏；
6. strict postflight 只有在日期、schema、覆盖和最终产物全部通过时才提交基线。

`research_candidate` 和 `on_demand` 字段不进入默认日更。将候选写入注册表只代表可治理、可评估，不代表已经有生产 SLA。

## 公式与版本变更

生产公式禁止原地改变语义。修改窗口、复权、数据源、边界规则或空值处理时必须：

1. 提升计算或 materialization 版本；
2. 保留旧版本产物用于对照和回滚；
3. 增加逐字段 golden parity 或差异说明；
4. 对模型输入变化执行重训和重新晋级；
5. 让合同 hash 触发目标节点及下游重算；
6. 在同一提交更新治理文档和自动生成目录。

## 维护命令

注册表变更后重新生成完整目录：

```bash
PYTHONPATH=src python scripts/generate_factor_catalog.py
```

核心治理测试：

```bash
PYTHONPATH=src pytest -q \
  tests/test_project_factor_layer.py \
  tests/test_daily_dependencies.py \
  tests/test_daily_dependency_runtime.py \
  tests/test_long_external_factors.py
```

提交前至少确认：注册表无重名、别名目标存在、规范字段自指、生产合同字段全部已注册、治理元数据完整、长线生产合同为 82 列。上述静态约束由 `validate_registry()` 在导入注册表时执行。

## 完整目录

[完整因子目录](factor_catalog.md)由注册表自动生成，不手工编辑。目录包含每条记录的规范名称、业务语义类别、因子层级、生命周期、计算器、计算归属、物化方式、刷新节奏、当前消费者、来源和版本。

## 正交分类模型

- `semantic_category`：只描述策略无关的业务含义，例如趋势、波动风险、估值、盈利和资金流。
- `factor_level`：描述原子、派生、复合、信号或身份层级。
- `calculation_owner/calculator_id`：描述技术实现和唯一计算责任，不限制策略使用范围。
- `active_consumers`：描述当前线上事实，不是访问控制列表；新策略通过 DAG 声明依赖即可使用任何规范因子。
- `lifecycle/refresh_cadence`：决定是否进入活动日更闭包。

计算器依赖、配置化晋级、共享 worker 预算和并行边界见[因子计算 DAG 与并行治理](factor_execution_and_parallelism.md)。
