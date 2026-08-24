# 短线左右统一排序模型：生产发布与回滚合同

## 当前生产结构

短线候选只维护两套第一层排序模型：

- 左侧：B1、SB1、超级 B1、缩量回调低吸；
- 右侧：B2、B3、强 K/突破、双阳结构、长安战法、坑里起好货、维加斯隧道、三倍量突破；
- 混合：支撑回踩、节奏/平台。消融选择把两组混合策略并入右侧，因此总模型数仍为 2。

Selector 对同时命中左右信号的股票执行 `right_side_precedence`，不存在同一行被两套模型重复覆盖。第一层只输出排序分，不伪装成涨跌概率，也不选择买卖 playbook。

## 晋级证据

右侧 canonical 统一模型相对旧生产模型，在相同事件键、相同标签和相同折上：

| 折 | ΔPR-AUC |
| --- | ---: |
| A | +0.007339 |
| B | +0.000305 |

两折均不差，因此按排序门槛晋级。生产 artifact 为：

```text
models/production/right_side_unified_canonical_v2/ranking.joblib
models/production/right_side_unified_canonical_v2/manifest.json
```

左侧统一模型相对四组独立模型，在 B 选择折和 C 确认折分别提升 `+0.008384`、`+0.011663` PR-AUC，因此晋级：

```text
models/production/left_side_unified_canonical_v4_group4/ranking.joblib
models/production/left_side_unified_canonical_v4_group4/manifest.json
```

610 个 canonical 注册因子的完整候选实验在 A 折提升 `+0.003458`，但 B 折下降 `-0.001452`，按“每折不得劣于紧凑模型”的门槛停止，未读取 C、未替换生产模型。

## 版本化合同

- 项目因子：`project-v5-canonical-alias-free`，145 列；
- 右侧独有规则：`right_side_rule_features_v4_113_20260824`，113 列；
- 左侧独有规则：`left_side_rule_features_v2_27_20260824`，27 列，另直接复用 `rs_is_rise` 和 `rs_close_pos`；
- 右侧生产 artifact：`right-side-unified-ranking-production-v2-canonical-alias-free`；
- 左侧生产 artifact：`left-side-unified-ranking-production-v4-group4-canonical-alias-free`；
- 左侧每日分数：`left-side-unified-ranking-score-v4-group4`。

五个历史兼容别名不得出现在新训练输入、`feature_names_in_`、selected columns、artifact、manifest、策略配置、每日依赖或页面输出合同中。旧文件只允许在读取边界完成逐值和 NaN 位置一致性校验后迁移，进入模型前必须删除旧列。

## 排序归一化与策略阈值

右侧先使用模型内事件级 Platt calibration，再用冻结的 OOT-B 预测分布做 1001 点经验 CDF，输出 0–100 的 `ranking_score_normalized`。左侧使用当日横截面稳定平均百分位，同样输出 0–100，且不改变原始排序。

两侧生产阈值模式均为 `none_rank_only`：旧的固定涨跌概率阈值不再参与活动选股。B1 页面按左侧每日百分位取 Top20，再分别套用已注册的两套退出方案；其余 Selector 消费者按归一化分执行下游 Top-N。

买卖策略层保持独立。第一层排序模型晋级不等于二层 playbook 晋级；二层方案只有在冻结折上通过收益、回撤、覆盖与可执行约束后才能单独发布。

## 每日依赖与幂等更新

活动短线闭包只包含：

```text
feature.strategy_signals
  ├─ feature.right_side_unified ── score.right_side_unified ──┐
  └─ feature.left_side_unified  ── score.left_side_unified  ──┼─ score.selector
                                                               └─ product.b1_plan
```

`score.b1`、`score.z_skill` 和其共享的 `feature.project_daily` 已退出活动闭包，生命周期为 `retired/on_demand`；旧 artifact 仍保留回滚，但没有每日消费者。

左右生产链路均以目标日、输入文件 checksum、schema 和 artifact hash 做 checkpoint。左侧把特征输入指纹与评分输入指纹分离，模型替换只重算评分，不会无意义地重算六年因子；目标日及真实输入不变时返回 `checkpoint_reused=true`，不重写输出。

## 回滚与验收

旧右侧统一 artifact、旧 B1 五模型、旧 Z-skill 模型以及左侧 v3 artifact 均保持原路径不变。回滚必须通过显式配置切换完成，不能覆盖新 artifact 或在新模型中恢复旧名映射。

严格 postflight 输出位于：

```text
reports/production/short_side_two_unified_release_20260824/postflight.json
reports/production/short_side_two_unified_release_20260824/consumer_zero_report.json
reports/production/short_side_two_unified_release_20260824/canonical_feature_list.json
```

验收要求：两套活动 artifact 与每日分数 checksum 通过；required/effective features 完全解析；兼容别名数为 0；旧 B1/Z 活动消费者数为 0；所有 rollback artifact 仍存在。
