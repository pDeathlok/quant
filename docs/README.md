# Quant 文档中心

这里收录当前生产架构、运行手册和策略研究档案。README 是项目入口；本页用于帮助维护者快速找到正确的细节文档。

## 开始使用

- [项目 README](../README.md)：安装、快速启动、主要配置和常用命令。
- [开发、验证与提交](development.md)：环境、隔离回归、目录归属与提交约定。
- [研究与生产入口索引](research_workflows.md)：排名、模型应用、ETF 与股指期货的脚本和发布边界。
- [每日运行与故障排查](operations.md)：完整日常刷新、调度、状态检查、失败恢复和限流调优。
- [每日刷新效率实测](daily_refresh_performance_20261111.md)：并行配额、重复计算消除、缓存边界和完整运行计时。
- [左右侧模型迭代与发布验收](model_iteration_20261111.md)：左侧替换、右侧研究、时点审计及线上版本核验。
- [模型应用组合 v1](model_application_portfolio_v1_20261112.md) 与 [第二轮新样本验收](model_application_iteration_v2_20261112.md)：组合选择、默认接入与执行限制。
- [每日刷新覆盖与效率审查](reviews/2026-11-14-daily-refresh-review.md)：页面覆盖、输出 DAG、晚期评分与快照基准。
- [每日依赖注册表](daily_dependency_registry.md)：数据源、特征、模型分和最终产物的四层日更合同，以及新增因子/策略/模型时的强制登记流程。
- [API 参考](api.md)：FastAPI 路由、参数和刷新范围。

## 架构与数据

- [系统架构与数据流](architecture.md)：模块边界、每日工作区、独立研究页面、并行依赖和产物。
- [项目结构与 MySQL 存储](project_structure_and_storage.md)：SQL/Parquet 双写、快照和历史复盘。
- [A 股分析 Skill 集成](a_share_skill_integration.md)：复用正式行情、保存不可变个股底稿，以及财报/事件后的认知迭代。
- [数据存储审查](data_storage_review_20260617.md)：存储问题与阶段性审查结论。
- [项目重构计划](project_restructure_plan.md)：从研究脚本向例行生产结构迁移的历史计划。
- [模块化单体实施记录](plans/2026-07-30-modular-monolith-refactor.md)：应用/基础设施边界、工作区迁移、前端性能和完整验收记录。

## 因子与变量

- [因子治理与生命周期](factor_governance.md)：统一注册表、计算层、生产日更边界、生命周期状态机、迁移门禁和维护流程。
- [因子执行与并行治理](factor_execution_and_parallelism.md)：新模型与变量如何接入共享计算，复用边界、资源预算与必跑回归。
- [完整因子目录](factor_catalog.md)：从机器注册表自动生成的全部因子、兼容别名和身份字段清单。
- [每日统一因子层](daily_factor_layer.md)：日频项目因子、轻量缓存和长线快照的计算与物化边界。
- [Tushare 因子体系](factor_tushare_factor_system.md)
- [变量字典](factor_variable_dictionary.md)
- [变量实现矩阵](factor_variable_implementation_matrix.md)

## 策略档案

生产和重点研究策略统一保存在 [`docs/strategies/`](strategies/)：

- [B1 正式策略记录](strategies/b1_selected_strategy_record.md)
- [缠论日线策略](strategies/chan_daily_strategy.md)
- [茶师长线策略](strategies/tea_master_long_strategy_record.md)
- [可转债轮动策略](strategies/convertible_bond_rotation_strategy.md)
- [BYD 日内做T策略](strategies/byd_002594_minute_t_strategy.md)
- [扩展策略模型记录](strategies/extended_strategy_model_record_20260607.md)
- [ETF四类资产月度再平衡 V1（研究定版，保留待用）](strategies/etf_defensive_monthly_v1_record.md)
- [股指战法工作台](strategies/index_futures_workspace.md)：冻结历史、每日开仓预案和按需刷新。
- [股指期货研究](strategies/index_futures_research.md)、[恐慌因子](strategies/index_futures_panic_research.md)、[期权隐波](strategies/index_futures_option_iv_research.md) 与 [后续研究](strategies/index_futures_next_research.md)。

## 项目整理与历史归档

- [2026-11-15 项目整理记录](reviews/2026-11-15-project-organization.md)：本次归档、提交分组与验证范围。
- [华纬科技 2026-11-03 研究正文](research/2026-11-03-huawei-001380.md)：从本地临时目录归档的历史分析，保留原始截止日期和来源；本次未重新核验行情或投资结论。

带日期的迭代文档保留当时的参数、样本和判断，属于历史证据，不保证描述当前生产默认值。当前运行口径以代码、策略配置和本页列出的生产文档为准。

## 文档维护规则

1. README 只放新使用者五分钟内需要的信息。
2. 运行命令、环境变量或 API 变化时，同一个提交内更新对应文档。
3. 策略行为变化时，先更新策略档案，再更新生产配置和代码。
4. 历史研究文档不覆盖重写；新增结论使用新文档或明确标注“已废弃/已替代”。
5. 文档中的命令应从仓库根目录可执行，路径使用仓库相对路径。
6. 修改因子注册表时必须在同一提交运行 `PYTHONPATH=src python scripts/generate_factor_catalog.py`，保证完整目录与代码一致。
