# 研究与生产入口索引

更新于 2026-11-15。本页梳理当前代码与配置，不重新认证历史收益，也不执行训练、数据抓取或发布。

## 股票排名与模型应用

| 工作流 | 主要入口 | 说明 |
| --- | --- | --- |
| 全量候选与特征 | `build_full_rank_dataset.py`、`augment_full_rank_sources.py`、`recover_full_rank_periodic.py` | 构建研究数据；先核对日期、覆盖和来源版本 |
| 排名目标与融合 | `evaluate_local_ranknet_fusion.py`、`evaluate_head_lambdaloss.py`、`evaluate_auxiliary_return_losses.py`、`evaluate_return_lambdas.py` | 可复用数学实现位于 `src/quant/research/`，融合运行时位于 `src/quant/ml/rank_fusion.py` |
| 左侧发布准备与验收 | `stage_left_ranknet_release.py`、`activate_left_ranknet_release.py`、`audit_left_ranknet_live.py`、`audit_left_ranknet_publication.py` | 生效配置为 `configs/strategies/left_side_unified.yaml`；发布与回滚配置在 `configs/releases/` |
| 模型应用 v1 | `audit_model_applications.py`、`validate_model_applications.py`、`select_dual_objective_portfolio.py`、`stage_application_portfolio.py` | 三路候选队列与资金回放；见 [v1 档案](model_application_portfolio_v1_20261112.md) |
| 模型应用 v2 | `prepare_iteration_development.py`、`model_iteration_v2_protocol.py`、`train_iteration_trees.py`、`train_iteration_neural.py`、`evaluate_iteration_holdout.py` | 冻结协议、训练、选择与新样本验收；见 [v2 档案](model_application_iteration_v2_20261112.md) |
| 标签与持仓政策 | `research_sample_labels_v1.py`、`compare_sample_label_production.py`、`stress_sample_labels_v1.py` | 区分研究标签、排名和真实账户执行 |
| ETF 资金修正 | `train_short_etf_rank.py` | 配置 `configs/strategies/short_etf_rank.json`；日更只读发布结果，不重新训练 |
| ETF 多资产研究 | `explore_six_sleeves.py`、`select_final_etf_strategy.py`、`report_final_etf_strategy.py` | [月度再平衡档案](strategies/etf_defensive_monthly_v1_record.md)，研究保留，未作为每日正式组合启用 |

表中脚本均位于 `scripts/research/`。许多是固定目录和参数的阶段性研究脚本，直接执行可能重写研究产物；执行前阅读对应档案与脚本入口，不能假定所有脚本都支持无副作用的 `--help`。

当前左侧采用冻结 RankNet 融合；右侧保持现有模型。短线应用组合配置是 `configs/strategies/application_portfolio.json`，每日运行入口在 `quant.routine.application_portfolio_production`。`scripts/run_application_portfolio.py` 是独立候选/影子评分入口，不等于正式页面发布；`scripts/run_left_ranknet_release.py --date YYYY-MM-DD` 会运行指定日期评分，应作为维护命令使用。

### 可选神经网络实验环境

v2 实验曾把 TabM 临时安装到 `.tmp/application_iteration_dependencies/`；该目录是第三方依赖安装结果，不纳入源码。优先在独立研究虚拟环境安装：

```bash
python -m pip install -e ".[dev,ml]"
python -m pip install -r requirements/research-iteration.txt
```

若需要复现原有目录布局，可在已安装 PyTorch 的研究环境执行：

```bash
python -m pip install --target .tmp/application_iteration_dependencies \
  -r requirements/research-iteration.txt
```

这是 TabM 及数值嵌入包的固定版本清单，不是完整环境锁；PyTorch、平台、随机种子和训练配置也影响复现。

## 股指期货与期权隐波

1. `fetch_index_futures.py`、`fetch_index_futures_panic.py`、`fetch_index_option_history.py` 获取各自研究输入。
2. `backtest_index_futures.py` 与 `evaluate_index_futures_*.py` 回放方向、期限、资金、退出、滚动、多起点及压力情景；`compare_index_futures_adaptation.py` / `refine_index_futures_adaptation.py` 比较季度适配方案。
3. `report_index_futures_*.py` 输出对应研究报告；需要绘图的脚本在研究环境另行安装 `matplotlib`。
4. `build_index_futures_workspace.py` 导出冻结历史页面；`build_index_futures_signal.py --refresh --publish` 单独刷新开仓预案。

详见 [研究主档案](strategies/index_futures_research.md)、[恐慌因子](strategies/index_futures_panic_research.md)、[期权隐波](strategies/index_futures_option_iv_research.md)、[工作台与更新约束](strategies/index_futures_workspace.md)、[后续研究问题](strategies/index_futures_next_research.md)。当前历史冻结至 2026-11-06；实时预案使用独立输入目录，未接入日常 `scope=all`。新季度需要新的成熟标签训练版本，按钮刷新不会自动延长研究历史。

## 运行边界

个股历史正文与结构化留痕另见 [华纬科技归档](../outputs/2026-11-03/huawei-001380/README.md)。下载材料和逐日行情继续保留在本地临时目录，正文及原有估值输入/结果纳入版本管理。

- 日常更新统一使用 `scripts/run_daily_web_refresh.py`；源和模型依赖由注册表选择，运行时不通过临时研究脚本补齐缺失数据。
- `stage_*`、`activate_*`、`revalidate_*` 涉及候选落盘、配置激活或缓存身份迁移，先检查源码中的目标路径、备份和校验条件。它们不是普通只读诊断命令。
- 数据及模型文件不随 Git 分发。复现实验应恢复相应版本的输入、模型、manifest 与报告证据，不能仅凭同名脚本推断结果一致。
- 历史运行耗时与验收见 [日更效率记录](daily_refresh_performance_20261111.md) 和 [11 月 14 日审查](reviews/2026-11-14-daily-refresh-review.md)。完整生产运行、局部基准、源码回归是三种不同证据。
