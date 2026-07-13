# Quant 文档中心

这里收录当前生产架构、运行手册和策略研究档案。README 是项目入口；本页用于帮助维护者快速找到正确的细节文档。

## 开始使用

- [项目 README](../README.md)：安装、快速启动、主要配置和常用命令。
- [每日运行与故障排查](operations.md)：完整日常刷新、调度、状态检查、失败恢复和限流调优。
- [API 参考](api.md)：FastAPI 路由、参数和刷新范围。

## 架构与数据

- [系统架构与数据流](architecture.md)：模块边界、7 个工作区、并行依赖和产物。
- [项目结构与 MySQL 存储](project_structure_and_storage.md)：SQL/Parquet 双写、快照和历史复盘。
- [数据存储审查](data_storage_review_20260617.md)：存储问题与阶段性审查结论。
- [项目重构计划](project_restructure_plan.md)：从研究脚本向例行生产结构迁移的历史计划。

## 因子与变量

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

带日期的迭代文档保留当时的参数、样本和判断，属于历史证据，不保证描述当前生产默认值。当前运行口径以代码、策略配置和本页列出的生产文档为准。

## 文档维护规则

1. README 只放新使用者五分钟内需要的信息。
2. 运行命令、环境变量或 API 变化时，同一个提交内更新对应文档。
3. 策略行为变化时，先更新策略档案，再更新生产配置和代码。
4. 历史研究文档不覆盖重写；新增结论使用新文档或明确标注“已废弃/已替代”。
5. 文档中的命令应从仓库根目录可执行，路径使用仓库相对路径。
