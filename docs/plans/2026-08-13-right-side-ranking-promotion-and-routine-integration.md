# 右侧统一排序模型重训、策略层验证与例行注册计划

## Goal

用完整 118 因子数据真实验证右侧统一模型相对旧 105 因子统一模型及同口径独立模型的排序增益；把买入卖出方案作为独立二层策略模型验证；只有排序门槛通过后，才将新模型以项目四层依赖结构进入每日影子迭代，并保留可回滚的线上切换合同。

## 决策口径

第一层模型只负责排序，不用绝对收益否决排序模型。主指标固定为同一测试事件上的 PR-AUC，辅助指标为 ROC-AUC、Top10% precision/lift、逐交易日 Top-K 标签命中率和逐策略 PR-AUC；收益只用于二层买卖策略，不进入第一层替换门槛。

第一层替换门槛全部是排序门槛：

- 118 因子模型相对同架构 105 因子模型在 A、B 两折的配对 `delta_pr_auc` 都大于 0；
- A/B 等权平均 `delta_pr_auc` 大于 0；
- A/B 等权平均 `delta_top10_lift` 不小于 0；
- 逐策略可评估切片的 A/B 宏观 `delta_pr_auc` 不小于 0；
- 事件覆盖率不低于 99%；
- 所有因子、标签、时间切分和第一层 OOF 合同通过审计。

置信区间只用于披露稳定性，不再用收益护栏阻止第一层替换。由于 2026 C 折已经在上一轮查看，本轮 C 只能作为描述性复核，不能重新命名为 untouched。A/B 纯排序门槛通过后即可替换“排序组件”；每日影子只验证数据、schema、freshness、artifact 和回滚链路可运行，不再追加收益门槛，也不把排序组件替换错误地解释成自动启用二层买卖策略。

## Pre-conditions

- [x] `df -h /Users/didi/Project/quant` 显示可用空间 134GiB，足够并存旧 105 数据与版本化 118 数据。
- [x] `PYTHONPATH=src pytest -q tests/test_right_side_*.py` 返回 `89 passed`。
- [x] `/Users/didi/Project/quant/data/research/right_side_unified/dataset_manifest.json` 固定旧数据截止 `2026-08-12`、`rule_feature_count=105`。
- [x] `/Users/didi/Project/quant/data/raw/daily_partitioned`、family cache、tradability 数据存在。
- [ ] 全量构建启动前再次检查没有 `validate_unified_right_side_models.py build-dataset` 进程；若无法读取进程表，检查目标 `.tmp` 文件和 manifest 更新时间没有变化。

## Step 1 — 冻结 105/118 因子版本与同架构消融

**Files:**

- `/Users/didi/Project/quant/src/quant/research/right_side_unified_features.py`
- `/Users/didi/Project/quant/src/quant/research/right_side_long_task.py`
- `/Users/didi/Project/quant/scripts/research/validate_unified_right_side_models.py`
- `/Users/didi/Project/quant/tests/test_right_side_factor_parity.py`
- `/Users/didi/Project/quant/tests/test_right_side_long_task.py`

在规则因子模块加入以下不可变版本合同：

```python
RULE_FEATURE_SCHEMA_VERSION = "right_side_rule_features_v2_118_20260813"
ADDED_RULE_FEATURE_COLUMNS_V2 = (
    "rs_vol_ratio_20_inclusive",
    "rs_family_kdj_j",
    "rs_recent_yin_count_4",
    "rs_close_to_ma60_pct",
    "rs_b1_support_ok",
    "rs_family_bbi_distance_pct",
    "rs_b3_small_pos_amp7",
    "rs_b3_broad_small_pos",
    "rs_b3_broad_calm_pullback",
    "rs_vegas_history_ok",
    "rs_vegas_tradable",
    "rs_vegas_signal",
    "rs_tvb_merged",
)
LEGACY_RULE_FEATURE_COLUMNS_V1 = tuple(
    column for column in RULE_FEATURE_COLUMNS
    if column not in ADDED_RULE_FEATURE_COLUMNS_V2
)
```

增加 `unified_long_task_deep_rule105` 实验臂；它与 `unified_long_task_deep` 使用相同样本、项目因子、task one-hot、XGBoost 参数、校准和阈值，只把规则列限制为 `LEGACY_RULE_FEATURE_COLUMNS_V1`。两个 manifest 必须写入 `rule_feature_schema_version`、`rule_feature_count`、`rule_feature_columns` 和 SHA256。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_right_side_factor_parity.py tests/test_right_side_long_task.py
```

预期：105/13/118 三个计数精确；两臂除规则列版本外训练合同一致；所有测试通过。

## Step 2 — 增加纯排序配对比较

**Files:**

- `/Users/didi/Project/quant/src/quant/research/right_side_factor_increment_comparison.py`
- `/Users/didi/Project/quant/tests/test_right_side_factor_increment_comparison.py`
- `/Users/didi/Project/quant/scripts/research/validate_unified_right_side_models.py`

公开接口固定为：

```python
def compare_rule_feature_versions(
    predictions: pd.DataFrame,
    *,
    candidate_column: str = "pred_unified_long_task_deep",
    baseline_column: str = "pred_unified_long_task_deep_rule105",
    label_column: str = "good_path5",
    top_fraction: float = 0.10,
    bootstrap_iterations: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    ...
```

输出按 `entry_mode/horizon/label/fold` 保存相同行的 `candidate/baseline PR-AUC`、`delta_pr_auc`、自然月块 95% CI、`delta_roc_auc`、`delta_top10_precision`、`delta_top10_lift` 和逐日 Top-K 标签命中率差。模块只读预测分数和标签，不读取收益列。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_right_side_factor_increment_comparison.py
```

预期：相同分数差为 0；更优合成排序所有 delta 为正；月份整块重采样保持事件成对；不同 fold/label 不混合。

## Step 2A — Beam Residual v3 规则因子筛选实验

把用户提供的 Beam Residual v3 作为独立实验臂，不能覆盖完整 118 因子主臂。与原方案“固定 baseline/context + 少量增量候选”的结构一致：固定 context 为项目公共因子、旧 105 个规则因子和 14 个 task identity；只有本轮新增的 13 个精确规则因子允许删除。搜索只读每个外层折训练期内部的滚动 validation 子折，外层测试年在特征子集、超参数和门控全部冻结前不可见。

后向 Beam 采用状态去重和结果缓存。正式适配保持全部 13 个候选，不做 Top-8 预筛或其他搜索空间裁剪；`beam_width=4`、`min_features=6`、`max_remove=10`，在 13 个候选上实际最大删除深度为 7。每个滚动窗 history/evaluation 分别最多 80,000/30,000 个事件，按交易月×稳定主策略×标签分层并以稳定 event hash、seed=42 确定性抽样。搜索模型固定 120 棵树、depth=4、early-stop=20。R0 为固定 context，R1 使用 R0 raw margin 作为 XGBoost `base_margin`，再按 `sigmoid(logit(R0_prob) + reliability × (R1_margin - R0_margin))` 组合；本项目没有同构的事前质量门，搜索固定 reliability=1，独立 pipeline-select 另比较 0.5 与 1。最终入选子集仍在完整 train/validation 上重训，并在完整外层 A/B 测试集评估。候选评分以 median `delta_pr_auc` 为主项，保留 v3 的 `5 × median(logloss improvement)`、折间标准差、最差折退化和特征数惩罚；ROC-AUC 仅作描述，不读取 terminal return、MFE、MAE 或任何买卖收益。所有原始/抽样行数、预算和滚动子折写入 manifest。

新增实验 `unified_long_task_deep_beam`，manifest 至少冻结：v3 adaptation schema、候选全集/hash、固定 context/hash、`test_data_used=false`、滚动子折、beam width、min features、max remove、访问组合、Top1 子集/hash、评分公式、置乱检验、独立 `pipeline_select` 和外层折。Validation 尾部保留完全独立的 pipeline-select 窗；Top1 冻结后才可读取。Beam Top1 只有在 provenance、标签置乱、pipeline-select 和 A/B 外层排序对照都通过后才有资格替换。

最终报告同时给出三组纯排序增量：118 vs 105、Beam vs 118、Beam vs 105。Beam 若失败，仅否定特征筛选臂，不影响完整 118 因子的主验证。

## Step 3 — 版本化重建 118 因子全量数据

旧目录完全保留。新产物固定写入：

- `/Users/didi/Project/quant/data/research/right_side_unified_v2_118/unified_right_side_dataset.parquet`
- `/Users/didi/Project/quant/data/research/right_side_unified_v2_118/unified_right_side_labels.parquet`
- `/Users/didi/Project/quant/data/research/right_side_unified_v2_118/dataset_manifest.json`
- `/Users/didi/Project/quant/reports/research/right_side_unified_v2_118/factor_coverage.csv`
- `/Users/didi/Project/quant/reports/research/right_side_unified_v2_118/sample_audit.json`

**Build:**

```bash
PYTHONPATH=src python scripts/research/validate_unified_right_side_models.py build-dataset \
  --start-date 2020-01-01 --end-date 2026-08-12 --workers 8 \
  --dataset-out data/research/right_side_unified_v2_118/unified_right_side_dataset.parquet \
  --labels-out data/research/right_side_unified_v2_118/unified_right_side_labels.parquet \
  --manifest-out data/research/right_side_unified_v2_118/dataset_manifest.json \
  --factor-audit-out reports/research/right_side_unified_v2_118/factor_coverage.csv \
  --sample-audit-out reports/research/right_side_unified_v2_118/sample_audit.json
```

**Verify:**

```bash
PYTHONPATH=src python scripts/research/validate_unified_right_side_models.py audit \
  --dataset data/research/right_side_unified_v2_118/unified_right_side_dataset.parquet \
  --labels data/research/right_side_unified_v2_118/unified_right_side_labels.parquet \
  --factor-audit reports/research/right_side_unified_v2_118/factor_coverage.csv \
  --sample-audit reports/research/right_side_unified_v2_118/sample_audit.json
```

预期：`rule_feature_count=118`；事件/标签键无重复；14 个策略均有事件；成熟标签缺失为 0；一字板进入成熟样本为 0；逐策略必需因子 missing/all-null/non-finite 为 0。

## Step 4 — 真实训练 105 vs 118 与排序决策

新模型和报告只写入版本化目录：

- `/Users/didi/Project/quant/models/research/right_side_unified_v2_118`
- `/Users/didi/Project/quant/reports/research/right_side_unified_v2_118/model_metrics.csv`
- `/Users/didi/Project/quant/reports/research/right_side_unified_v2_118/signal_metrics.csv`
- `/Users/didi/Project/quant/reports/research/right_side_unified_v2_118/test_predictions.parquet`
- `/Users/didi/Project/quant/reports/research/right_side_unified_v2_118/rule_factor_increment_ab.csv`
- `/Users/didi/Project/quant/reports/research/right_side_unified_v2_118/ranking_promotion_decision_ab.json`

**Train A/B:**

```bash
PYTHONPATH=src python scripts/research/validate_unified_right_side_models.py train \
  --dataset data/research/right_side_unified_v2_118/unified_right_side_dataset.parquet \
  --labels data/research/right_side_unified_v2_118/unified_right_side_labels.parquet \
  --entry-mode next_close --horizon 5 --label good_path5 --folds A B \
  --experiments independent unified_long_task_deep_rule105 unified_long_task_deep \
  --model-root models/research/right_side_unified_v2_118 \
  --metrics-out reports/research/right_side_unified_v2_118/model_metrics.csv \
  --signal-metrics-out reports/research/right_side_unified_v2_118/signal_metrics.csv \
  --predictions-out reports/research/right_side_unified_v2_118/test_predictions.parquet \
  --model-jobs 4
```

排序决策只读取 A/B。若通过，再单独跑 C 并在报告中标记 `descriptive_seen_period`，不修改 A/B 决策。

### Beam Residual v3 增量臂（主三臂后单任务运行）

额外实验臂 `unified_long_task_deep_beam` 忠实映射为：固定 context=`项目因子 + 105 旧规则因子 + 14 task one-hot`，候选仅为 13 个 v2 新增规则因子。外层 A/B 各自独立搜索，外层测试年完全不可见；validation 按时间保留 B1..B5，前三个滚动验证为 `B1→B2`、`B1+B2→B3`、`B1+B2+B3→B4`，B5 仅作 Top1 冻结后的独立 `pipeline_select`。主评分为 median ΔPR-AUC；logloss 进入 v3 稳定性公式，ROC-AUC 仅作描述，禁止读取收益。

冻结计算合同为：完整 13 候选全集、无预筛、`beam_width=4`、`min_features=6`、`max_remove=10`（effective depth=7）；每滚动 history 最多 80,000 event、evaluation 最多 30,000 event，按交易月×稳定主策略×label 分层，以 `(symbol,date,seed=42)` 稳定 hash 取样。主策略按 `RIGHT_SIDE_SIGNALS` 声明顺序取第一个 active signal。搜索使用 120 trees、depth 4、early stopping 20；最终选中子集仍在完整外层 train/validation 重训并在完整 outer test 上评价。单折预计访问 206–236 个组合、完成 618–708 次三窗 residual 拟合；A/B 以单任务顺序运行，不与二层 outcome/model 全量训练并发。manifest 必须记录原始/抽样行数、方法、seed、主策略规则、完整候选/hash、访问组合、选择列/hash、滚动窗、置乱和 pipeline-select。Beam 只有在两折 `development_gates_passed=true` 且 A/B 排序门槛通过时才可成为 selected candidate；旧 v1/bounded manifest 一律 fail-closed。

上线总决策还需同时披露三类排序证据：(a) 118/Beam vs 同架构 105；(b) 105/118/Beam vs 同口径 independent；(c) 最终候选 vs 当前 selector 消费的旧 artifact。旧 artifact 比较只在 A/B 的 exact symbol/date 交集和 legacy signal-timing-match 子集上进行，报告覆盖率与历史时点限制；当前事件覆盖约 A 29.7%、B 27.0%，禁止外推至全量事件。

## Step 5 — 构建窄表二层策略数据与真实验证

**Files:**

- `/Users/didi/Project/quant/src/quant/research/right_side_playbook_dataset.py`
- `/Users/didi/Project/quant/scripts/research/build_right_side_playbook_outcomes.py`
- `/Users/didi/Project/quant/scripts/research/validate_right_side_playbook_models.py`
- `/Users/didi/Project/quant/tests/test_right_side_playbook_dataset.py`
- `/Users/didi/Project/quant/tests/test_right_side_playbook_model.py`

宽事件因子表不复制九次。输出拆成：

- `playbook_events.parquet`：每 event 一行，包含 event key、fold、118 因子、14 身份、第一层 OOS 分数及 provenance；
- `playbook_outcomes.parquet`：每 event × action 一行，只含动作参数、T+1 可执行性、退出结果、净收益和 MAE；
- `playbook_dataset_manifest.json`：目录 hash、成本、数据截止、键数、成熟度和 OOF 合同。

二层 A→B 比较固定四臂：`no_trade`、`static_global`、`static_per_signal`、`shared_playbook_model`。收益和风险只在本步骤判定，不反向影响第一层排序替换。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_right_side_playbook_dataset.py tests/test_right_side_playbook_policy.py tests/test_right_side_playbook_model.py
```

预期：动作键唯一；T+1 不可执行只取消预选动作，不事后换动作；第一层分数只允许 earlier-fold OOF/test provenance；静态基线与模型使用相同事件和成本。

## Step 6 — 按项目四层结构注册每日影子迭代

此步骤只在 Step 4 排序门槛通过后启用；首次接入保持 shadow，不替换 selector 输出。

**Files:**

- `/Users/didi/Project/quant/configs/strategies/right_side_unified.yaml`
- `/Users/didi/Project/quant/src/quant/application/daily_dependencies.py`
- `/Users/didi/Project/quant/src/quant/application/refresh_contracts.py`
- `/Users/didi/Project/quant/src/quant/features/factor_registry.py`
- `/Users/didi/Project/quant/src/quant/routine/right_side_unified_shadow.py`
- `/Users/didi/Project/quant/src/quant/routine/pipeline.py`
- `/Users/didi/Project/quant/tests/test_daily_dependencies.py`
- `/Users/didi/Project/quant/tests/test_daily_dependency_runtime.py`
- `/Users/didi/Project/quant/tests/test_right_side_unified_shadow.py`

注册独立 shadow scope `rightSideShadow`，包含：

- `feature.right_side_unified_shadow`：T 日 v4 因果项目因子 + 118 规则因子；
- `score.right_side_unified_shadow`：版本化统一排序 artifact；
- `product.right_side_unified_shadow`：当日候选分数和 manifest，不被线上 selector 消费。

配置 `release.lifecycle=research_only`，manifest 固定 artifact SHA256、118 因子 hash、信号 schema、训练截止日和排序验证报告。每日运行必须经过 registry artifact feature extractor、factor registry point-in-time 校验、exact-trade-date freshness 和 final gate。

**Verify:**

```bash
PYTHONPATH=src pytest -q tests/test_daily_dependencies.py tests/test_daily_dependency_runtime.py tests/test_right_side_unified_shadow.py
```

预期：现有 `short` scope 闭包不变；`rightSideShadow` 四层拓扑完整；模型变更会 dirty score/product；缺 artifact、缺 118 因子、旧日期或错误 schema 均 fail closed。

## Step 7 — 排序组件生产替换开关

A/B 排序门槛通过、且至少一次完整每日影子运行通过 schema/freshness/artifact/final gate 后，执行一次显式 promotion：复制冻结 artifact 到 `/Users/didi/Project/quant/models/production/right_side_unified/ranking.joblib`，写 checksum manifest，把配置 lifecycle 改为 `production`，并让 `score.selector` 通过配置选择新 ranking score。旧 `score.z_skill` artifact、缓存和 scorer 保留一个回滚周期，不删除。影子运行是工程验收，不是新的收益或排序选模窗。

生产切换验收只看排序与运行合同：A/B 排序相关指标、逐策略排序护栏、同日候选覆盖、artifact/factor/schema freshness 和日常例行成功；买卖收益继续由二层策略配置独立迭代，未通过二层 A→B 验证前只保留研究/影子状态。

## 2026-08-13 实跑决策更新

- 完整 118 相对同架构 105 在 A/B 的平均 `delta_pr_auc=-0.00110655`、平均 `delta_top10_lift=-0.0393682`、策略宏平均 `delta_pr_auc=-0.00454774`，因此 118 全量直加臂失败，不进入替换。
- 105 统一长表模型相对同口径独立成员模型，A/B 平均 `delta_pr_auc=+0.0146482`、平均 `delta_top10_lift=+0.0634024`、策略宏平均 `delta_pr_auc=+0.00386780`、覆盖 100%，因此它是统一架构 shadow 候选。
- 105 与真实旧线上工件只能在约 27%–30% 的精确交集上比较，A/B 方向不一致；`unified_vs_independent` 的通过只授权 research shadow，不能被解释为 `replace_legacy_online`。生产替换需要单独的、显式的 legacy replacement decision；当前保持 false。
- Shadow 特征侧仍完整物化 118 因子以支持诊断/未来 Beam，105 工件只消费冻结的旧 105 规则列；两者的实际输入列和 hash 必须分别记录。
- Beam 实验升级为正式 v3 适配：13 个候选全量后向搜索，无 Top-8 预筛，`width=4/min_features=6/max_remove=10`（effective depth 7），增加独立 `pipeline_select`；旧 bounded v1 manifest 不允许 promotion。
- Beam Residual v3 正式 A/B 已完成：A/B 分别访问 206/208 个组合，置乱门均通过，但 A 的 pipeline-select 失败、B 的开发 logloss 稳定性失败；外层相对 105 的平均 `delta_pr_auc=-0.00309875`、平均 `delta_top10_lift=-0.0352558`，因此 Beam 不 promotion，研究 shadow 候选继续保持 105。

## Rollback

- Step 1–5 所有产物均写版本化新目录，失败时不覆盖旧 105 数据、旧报告或线上 artifact；删除新目录不是恢复线上所必需的操作。
- Shadow 注册使用独立 `rightSideShadow` scope，关闭该 scope 即停止影子，不影响 `short`。
- 生产切换若失败，将 `configs/strategies/right_side_unified.yaml` 的 lifecycle/consumer 恢复为 shadow，并把 selector source 切回 `score.z_skill`；旧 21 个 Z artifact 在整个观察周期保持原位。
- 不运行 `git reset --hard`、`git checkout --` 或覆盖现有生产 manifest。
