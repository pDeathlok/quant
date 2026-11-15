# 研究脚本导航

脚本按现有名称保留，避免破坏脚本间导入、发布合同和历史复现路径。按业务流程查找入口，请使用 [研究与生产入口索引](../../docs/research_workflows.md)。

- `build_*` / `fetch_*` / `refresh_*`：数据集、输入与特征准备，可能联网或覆盖本地产物。
- `train_*` / `evaluate_*` / `compare_*` / `stress_*`：训练、回测、对照与压力检验。
- `audit_*` / `validate_*`：版本、覆盖、时点和发布验收；具体写入行为以脚本为准。
- `report_*` / `summarize_*`：报告与汇总。
- `stage_*` / `activate_*` / `revalidate_*`：候选准备、配置激活或缓存迁移，使用前核对目标、备份与门禁。

每日正式刷新使用 `scripts/run_daily_web_refresh.py`。不要直接遍历执行本目录，也不要假定每个脚本都实现 `--help`。独立实验所需可选依赖和提交前检查见 [开发说明](../../docs/development.md)。
