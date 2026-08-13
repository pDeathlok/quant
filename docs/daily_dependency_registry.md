# 每日依赖注册表

本文是生产日更依赖的维护合同。新增或修改数据源、因子、模型、策略和页面产物时，必须同步更新这里说明的可执行注册表；不能只在脚本、YAML 或页面服务中增加一条隐式调用。

## 唯一事实来源

项目保留三类职责不同的注册信息：

1. `src/quant/features/factor_registry.py` 登记因子字段的名称、采样频率、来源、点时属性和描述性消费者，是字段目录，不负责编排；是否属于当前版本、是否需要日更只能以活动 DAG 与 artifact 合同为准。
2. `src/quant/application/daily_dependencies.py` 是生产日更的可执行四层 DAG，固定节点依赖、生命周期、频率、新鲜度、增量键、回看窗口、配置/计算器来源和最终门禁。
3. 当前晋级模型 artifact 是模型输入列的事实来源。`src/quant/routine/daily_dependency_runtime.py` 只在 artifact 文件变化时加载模型，编译逐模型 required/effective features、SHA256 和活动产品闭包。

`src/quant/application/refresh_contracts.py` 的页面进度步骤由 DAG 派生，不再维护第二份 scope 顺序。运行时合同写入：

- `data/contracts/daily_dependencies/latest.json`
- `data/contracts/daily_dependencies/latest-<scope>.json`（各 scope 最近一次成功 strict postflight 的已提交增量比较基线；preflight/failed 不得覆盖）
- `data/contracts/daily_dependencies/YYYY-MM-DD-<scope>-preflight.json`
- `data/contracts/daily_dependencies/YYYY-MM-DD-<scope>-postflight.json`
- `data/contracts/daily_dependencies/model_contract_cache.json`

## 四层合同

```text
data_source -> feature -> model_score -> product
```

| 层级 | 必须登记的内容 | 成功证据 |
| --- | --- | --- |
| `data_source` | 数据所有者、轮询/交易日频率、分区键、覆盖要求、修订重叠窗 | 目标日分区、当日轮询水位或受控 TTL |
| `feature` | 上游源、字段目录、计算器、回看上下文、写入粒度 | 目标日 manifest/分区，required feature coverage |
| `model_score` | artifact 精确路径、manifest、特征提取方式、模型版本 | 目标日评分、运行时 artifact hash、输入覆盖；评分 manifest 另记录分布 |
| `product` | 页面/计划/快照入口、策略配置、依赖的模型分或规则特征 | `signal_date/trade_date/target_date == expected_trade_date` |

目标日期只能使用行情刷新确认的 `expected_trade_date`，不能使用系统自然日，也不能用下游产物自己的最新日期反推目标日。

## 当前生产边界

### 必须与目标交易日精确一致

- A 股日线、当前 scope 所需的 `daily_basic`、沪深300日线；
- B1/Z 活动候选信号和项目因子、B1/Z 模型分、买入/持有分、短线快照；
- Chan 当日候选、特征、模型分和策略产物；
- 长线价格/市场状态/因子截面与 Tea/Tea-safe/v44 产物；
- 可转债当日行情、网格计划和配债页面所需日频输入；
- BYD 当日日线特征与计划；
- 相似走势的当日目标向量、市场/行业状态和最终结果。

### 每日轮询，但允许当天没有新记录

- 财报：按 `ann_date` 点时 upsert；
- 研报：按 `report_date` 点时 upsert；
- `stock_basic`、上市/退市状态、可转债基础/赎回/配债事件；
- 事件源必须证明 `polled_through >= expected_trade_date`。只有真实 provider poll 成功（空结果也算）或复用同一目标日的成功检查点时才能推进；异常、降级或旧缓存 fallback 不得推进，也不得用父步骤的 `end_date` 冒充子源水位。数据表的最大事件日期早于目标日可以合法，不能据此误判陈旧。

### 不要求每日重算

- 相似走势历史参考向量：最多每 7 天更新，目标向量仍每日计算；
- BYD 5 分钟训练样本：新样本到达时更新，并受 60 日 SLA 约束；它不是当日特征；
- 模型训练、回测、校准和晋级：仅在研究/发布流程运行；
- selector 历史训练参考、long-entry v1/v2 研究模型、tradability 研究标签、margin/moneyflow/holder-trade 等无当前生产消费者的数据：不进入生产日更闭包。

当前模型合同由 artifact 动态得到：B1 为 147 列，Z 为 112 列且是 B1 子集，selector 为 49 列，Chan 为 46 列。只要 B1 仍为生产版本，147 个项目日频因子没有安全的列级停更空间。Chan 的三项 top-list 列目前在晋级 artifact 中有效重要性为零，因此 top-list 源会被活动闭包裁掉；以后新模型真正使用这些列，源会自动重新进入日更计划。

Z/Chan 目前仍从 `models/research` 路径被生产日更消费，注册表会把它们标成 `production_consumed_from_research_path` 技术债，但仍按生产模型严格编译特征和哈希。下一次晋级必须迁入 `models/production`，补齐正式 release manifest 后才能删除这项豁免。

因子字段的采样频率与生产物化频率是两个概念。例如财务字段按公告到达，长线选股截面仍要在每个交易日以 PIT 方式重新物化。当前长线节点读取 8 年估值上下文和 450 个自然日价格上下文，但只发布目标日分区。

## 增量计算规则

`IncrementalPolicy` 区分“读取多少历史”和“重写多少数据”：

- 新增当日行情默认只 upsert 目标交易日分区；
- rolling/周线计算可读取声明的 calendar-day、session 或 year 上下文，但只发布目标分区；
- `daily_basic` 检查最近 45 个自然日空洞，滚动特征读取至少 20 个交易日；
- 财报和事件源使用有限 overlap 窗口，按业务主键 upsert；
- B1/Z 先生成当日候选并集，只计算这些 symbol 的 147 因子；B1 原缓存仍保持 B1 gate 语义，Z 复用共享 sidecar；
- 模型 artifact 只有 size/mtime 变化时才重新计算 SHA256 和加载特征合同；未变化直接读缓存；
- `contract_sources` 中的数据采集/标准化实现、因子实现或策略配置内容变化，只 dirty 对应节点及下游，不重刷无关上游数据；
- 节点本身的 edge、freshness、incremental、artifact 等注册定义也参与合同哈希；只改注册表同样会 dirty 对应节点；
- Web 断点续跑实际消费 preflight 的 dirty 节点：模型/hash/配置变化时，即使旧产物日期也是今天，也禁止复用对应检查点；tail resume 只允许补尚未成功且未受上游变更影响的节点；
- scope 尚无已提交基线时，preflight 必须把活动闭包全部标记为 dirty；同一次运行的 postflight 必须携带该 preflight identity，刷新完成后才能清除预期 dirty 并提交新基线；
- 周频、静态和事件节点先做 `reuse/poll`，只有结果变化才向下游传播 dirty 分区。

严禁用“文件存在”“生成时间是今天”或 imputer 成功预测作为新鲜度证据。合法空候选也必须有目标日 manifest。

## 新增或修改时的强制步骤

### 新数据源

1. 在 `daily_dependencies.py` 增加 `data.*` 节点；
2. 声明 `Cadence`、`FreshnessMode`、evidence、主键、分区和 repair overlap；
3. 将采集器、标准化实现和影响数据口径的配置加入对应 `data.*.contract_sources`；
4. 在运行时适配器中增加可验证的 evidence，不得以空表或 mtime 冒充目标日；事件源必须发布自己的 `polled_through`，不能复用父步骤水位；
5. 只有活动产品闭包需要时才在 reference refresh 中启用；
6. 若是最终必需源，设置 `final_gate=True` 并添加 stale/empty/错日测试。

### 新因子或修改公式

1. 通用字段先登记到 `factor_registry.py`，策略私有字段使用稳定前缀；
2. 更新唯一计算入口和 schema/materialization version；
3. 将计算器/配置文件加入对应 `feature.*.contract_sources`；
4. 声明真实回看单位：`context_lookback_sessions`、`context_lookback_calendar_days` 或 `context_lookback_years`；
5. 添加目标日、缺列、全空、覆盖率和历史修订的测试；
6. 重新训练并晋级使用新 schema 的模型，不能让旧模型静默消费新口径。

### 新模型或模型晋级

1. 在 `model_score` 节点的 `ArtifactSpec` 登记精确 artifact 和 manifest 路径；
2. B1/Z 使用 `feature_names_in_`，bundle 模型提供稳定 `features`；
3. manifest 固定 artifact path/hash、训练截止日、feature schema 和 release ID；`ArtifactSpec.expected_schema` 必须与 bundle 的 `schema_version` 一致；
4. required features 必须全部可用；effective features 只用于裁掉确实无贡献的可选源，不能改变模型输入 shape；
5. 评分产物必须记录目标日、模型 release/hash、输入覆盖和分布；
6. 添加 checksum、artifact cache、schema 不兼容和目标日门禁测试。

### 新策略或修改策略配置

1. YAML 的 `release.lifecycle` 明确为 `production` 或 `research_only`；
2. 每个 YAML 必须声明 `release.lifecycle` 和 `release.dependency_nodes`；生产 YAML 列出的每个节点都必须存在，且该 YAML 必须出现在对应节点的 `contract_sources`，漏登记或错登记都会失败；
3. 新页面/计划增加 `product.*` 节点，并加入对应 scope root；
4. 规则使用哪些 feature/source 必须通过 edge 声明，不能只在服务函数中隐式读取；
5. 最终 payload 暴露统一目标日字段并设 `final_gate=True`；
6. 更新 scope 闭包、增量传播、合法空结果和断点续跑测试。

### 删除或停用

先把所有生产消费者移出活动闭包，再将节点标为 `research_only` 或 `retired`。只有反向闭包中没有生产消费者的整节点，才可以停止日更。不要仅凭“未出现在某个模型 artifact”删除规则策略或页面使用的字段。

## 发布前验证

至少运行：

```bash
PYTHONPATH=src pytest -q \
  tests/test_b1_gate.py \
  tests/test_daily_dependencies.py \
  tests/test_daily_dependency_runtime.py \
  tests/test_b1_training_pipeline.py \
  tests/test_project_factor_layer.py \
  tests/test_score_latest_strategy_models.py \
  tests/test_model_feature_coverage.py \
  tests/test_refresh_contracts.py \
  tests/test_reference_data_refresh.py \
  tests/test_data_refresh.py \
  tests/test_daily_basic_refresh.py \
  tests/test_market_regime.py \
  tests/test_chan_daily.py \
  tests/test_byd_workspace.py \
  tests/test_convertible_bond_grid_plan.py \
  tests/test_convertible_bond_workspace.py \
  tests/test_pipeline.py \
  tests/test_webapp_api.py \
  tests/test_data_quality_regressions.py \
  tests/test_research_market_backtest.py

PYTHONPATH=src python -m compileall -q src scripts
git diff --check
```

生产刷新结束后还要检查：

```bash
python -m json.tool data/contracts/daily_dependencies/latest.json
python -m json.tool data/routine/latest_refresh_status.json
```

`latest.json` 必须满足根字段 `schema_version == "daily_dependency_snapshot_v2"`、`identity_complete == true`、`baseline_committed == true`、`phase == "postflight"`、`status == "success"`、`freshness_audit.status == "success"`，且 `refresh_node_ids` 为空；`latest_refresh_status.json` 必须满足根字段 `status == "success"`、`result.dependency_postflight.status == "success"`、`result.dependency_postflight.baseline_committed == true`，且 `result.dependency_postflight.refresh_node_ids` 为空。若 exact-date 上游失败，任务必须保持 failed；断点续跑不能跳过陈旧源直接补尾段产品。

## 2026-08-13 验证基线

- 默认注册表覆盖 `all/short/chan/long/cb/cbAllotment/byd/similar` 八个 scope；
- 所有生产数据源的主要采集/标准化实现及生产策略 YAML 已被 `contract_sources` 覆盖，research-only YAML 未进入生产节点；
- 当前四类持久模型合同可解析，活动 required feature union 为 238 个唯一字段；
- B1/Z 候选并集 sidecar 必须同时有精确日期文件、147 列/schema 证明和 `candidate_coverage_status=complete`；合法的零候选也必须发布目标日 manifest；
- 单元/集成测试覆盖 DAG、动态 source pruning、artifact cache、配置哈希 dirty 传播、新鲜度汇总和 Web 前后置门禁。
- 以 2026-08-12 为目标交易日的严格 `all` postflight 已通过：37 个活动节点（11 个数据源、10 个特征、6 个模型分、10 个最终产物），32 个 final-gate 节点无失败，`refresh_node_ids=[]`；
- B1/Z 活动候选并集 460 只，457 只完成 147 因子，3 只均有 ST/退市政策排除原因且无无法解释的漏算；Z 为 112/112 输入有效，Chan 为 46/46 输入有效；
- 在仅含本次暂存内容的干净目录中，上述 20 个注册表与受影响回归文件实际结果为 `271 passed`；完整无凭证测试收集为 `568 passed, 10 failed`，10 项均是未改动的配债测试依赖本地 Tushare 部署资产，在纯 HEAD 中可复现；`compileall` 与 `git diff --cached --check` 同时通过。
