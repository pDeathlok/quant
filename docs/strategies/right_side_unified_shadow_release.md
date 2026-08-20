# 右侧统一排序模型：生产发布、影子验收与回滚合同

## 当前状态

`unified_long_task_deep_rule105` 已在 2026-08-13 经用户明确批准，以可回滚方式接管 14 个右侧/混合策略的候选排序。线上 selector 当前为 `right_side_unified`，生产 artifact 与最新日 feature/score manifest 均已 checksum 固定；`DUICHEN_VA/NANA/YIDONG_DILIAN` 继续消费旧 Z 分数。买入、卖出和 `no_trade` 属于独立的二层策略模型，本次没有晋级。

研究证据没有被改写：105 因子统一架构相对同样本 independent 的 architecture gate 通过（mean ΔPR-AUC `+0.014648`），完整 118 因子相对 105 的增量 gate 失败（mean ΔPR-AUC `-0.001107`），Beam Residual v3 相对 105 也失败，105 相对 legacy 线上 artifact 的精确交集 A/B 没有一致胜出。原始 composite decision 仍为 `replace_online=false`；独立的 `production_rollout_approval.json` 记录了维护简化的上线决定、风险确认、shadow 验收 checksum 和回滚方式。

105 research bundle 已完成 checksum 固定；Beam A/B 完成后仍选择 105。2026-08-13 最新行情与两套信号缓存对齐后，326 个候选全部生成生产特征并评分；含冻结归一化合同的生产工件 SHA256 为 `5267a4567739045d9cf43ff1d74c5cd6c467b27f712d5bb4bd5af75fa586a5aa`。`rightSideRankingCandidate` postflight 为 success、`baseline_committed=true`，五节点 freshness 全部通过。

排序分使用两段固定变换：模型内部先做事件级 Platt calibration，再使用 B 折 342,745 条 OOT 预测冻结的 1001 点经验 CDF，单调映射为 `ranking_score_normalized`（0–100 历史百分位）。原始 `ranking_score` 保留用于审计，Selector 只用归一化分排序。研究阶段 B 折阈值 `0.3490536111` 仅作为参考写入 manifest；生产合同为 `none_rank_only`，不按该阈值删除候选，最终选择仍由下游 Top-N 完成。

影子不是第二个统计门槛。它用于验收 schema、日期 freshness、artifact checksum、依赖脏传播和 final gate。生产审批与研究结论分开存档：维护者不得把用户的上线授权写回或覆盖研究 decision。

## 版本化合同

- 项目因子：`project-v4-causal-price-alpha`，147 列。
- 右侧规则因子：`right_side_rule_features_v2_118_20260813`，118 列。
- 模型因子并集：265 列；另带 14 个策略身份输入。
- 特征合同：`right-side-shadow-features-v1-project-v4-rule-v2-118`。
- artifact 合同：`right-side-unified-ranking-shadow-v1`。
- 265 因子顺序 hash：`6fc15ad06252b66b40f890165243f6b17cb74444909019155e12934683cace9c`。

147 列是注册与 sidecar 形状合同，不表示当前研究样本凭空拥有 147 列有效值。冻结训练集实际物化 112 个 OHLCV 项目因子；35 个未接入本轮训练的 `daily_basic` 项目因子显式为空。影子日更严格复现该语义，同时完整计算 118 个规则因子；不得为 35 列制造默认值，后续只有在重新构建全样本并训练验证后才可把它们改为有效输入。

合同源位于：

- `src/quant/features/right_side_factor_contract.py`
- `src/quant/features/factor_registry.py`
- `configs/strategies/right_side_unified.yaml`

所有 118 个规则因子均注册为 `right_side_rule`、`point_in_time=true`、consumer 同时覆盖 `right_side_unified_shadow` 和 dormant 的 `right_side_unified`；14 个身份列以 `strategy_identity` 角色单独注册，不伪装成连续市场因子。

## 四层影子拓扑

独立 scope 为 `rightSideShadow`：

```text
data.market_daily ───────────────┐
                                ├─ feature.right_side_unified_shadow
feature.strategy_signals ───────┘       │
                                        ▼
                            score.right_side_unified_shadow
                                        │
                                        ▼
                          product.right_side_unified_shadow
```

前三个影子专属节点均为 `research_only`。`score.selector` 不在该闭包内；`short` 和 `all` 的闭包也不包含任何 `right_side_unified_shadow` 节点。模型或配置 checksum 变化只会 dirty 影子 score/product。

## 工件和可观察性

影子 artifact 固定为：

```text
models/research/right_side_unified_v2_118/shadow/ranking.joblib
models/research/right_side_unified_v2_118/shadow/manifest.json
```

每日产物固定为：

```text
data/features/right_side_unified_shadow/latest_features.parquet
data/features/right_side_unified_shadow/feature_manifest.json
data/features/right_side_unified_shadow/latest_scores.parquet
data/features/right_side_unified_shadow/score_manifest.json
reports/research/right_side_unified_v2_118/shadow/latest_candidates.parquet
reports/research/right_side_unified_v2_118/shadow/product_manifest.json
reports/research/right_side_unified_v2_118/shadow/run_status.json
```

以下任一情况均 fail closed：artifact 缺失或 checksum 不符、artifact 绑定的 composite decision checksum 已变化、模型 schema 不符、118 规则因子不完整、模型要求未注册列、因子 hash 不符、feature/score parquet 与 manifest checksum 不符、日期不是目标交易日、候选键重复、候选覆盖不完整、分数越界或 final gate 未通过。失败只写入 `run_status.json`，不会调用或覆盖生产 selector。

## 启用步骤

1. 读取无歧义总决策 `production_replacement_decision_ab.json`；shadow 只消费 `shadow_candidate` 和 `selected_research_candidate`，绝不能把架构对照文件中的 `replace_online` 当成 legacy 生产授权。
2. 从当前冻结的 105 统一模型生成独立 shadow bundle（sidecar 仍完整物化 118 列，bundle 只消费 manifest 冻结的 105 规则列）：

   ```bash
   PYTHONPATH=src python -m quant.routine.right_side_unified_shadow \
     stage-shadow-release \
     --source-model models/research/right_side_unified_v2_118/next_close/h5/good_path5/B/unified_long_task_deep_rule105.joblib
   ```

3. 将 `configs/strategies/right_side_unified.yaml` 的 `routine.enabled` 显式改为 `true`。
4. 以独立研究任务运行目标交易日：

   ```bash
   PYTHONPATH=src python -m quant.routine.right_side_unified_shadow \
     run --target-date YYYY-MM-DD
   ```

   当例行开关仍关闭、只做一次工程验收时，可显式追加 `--force-enabled`。这个参数只进入 research-only 输出；若市场已有更新日期，它只允许回放一个真实存在的目标日行情分区，且仍要求信号、feature、score、product 全部精确等于同一目标日。正式例行不使用该参数，继续要求目标日为最新市场日。

5. 确认 dependency postflight `status=success`、`baseline_committed=true`，且 feature/score/product 三份 manifest 日期、schema 和 checksum 全部一致。

`quant.routine.pipeline.run_daily_pipeline()` 不调用影子入口。若由调度器例行执行，应把 `quant.routine.pipeline.run_right_side_shadow_routine()` 配为生产短线任务完成后的独立研究 job；影子失败不得改变正式任务状态。

## 生产运行与回滚

生产结构已启用：

- 开关合同：`configs/strategies/right_side_ranking_selector.yaml`；当前 `selector.ranking_source=right_side_unified` 且 `promotion.enabled=true`。
- 活动闭包：`feature.right_side_unified -> score.right_side_unified -> product.right_side_unified_adapter`；feature/score 同时进入默认 `short/all`，独立 scope `rightSideRankingCandidate` 用于发布验收。
- 生产工件：`models/production/right_side_unified/ranking.joblib` 和同目录 `manifest.json`。
- 每日分数：`data/features/right_side_unified/latest_scores.parquet`，唯一模型输出语义为 `ranking_score`；严禁映射为 `pred_up5/pred_up8/pred_down3`。
- 信号路由：晋级后 `score.selector` 同时依赖 `score.right_side_unified` 和 `score.z_skill`，但按信号互斥消费。新分数只接管 `B2/B3/KEY_K/VIOLENCE_K/PINGHANG/DOUBLE_GUN/CHANGAN/KENGQI/VEGAS/TRIPLE_VOLUME_BREAKOUT/GOLDEN_BOWL/ZAIHOU/BREATHING/YUEYUE`；`DUICHEN_VA/NANA/YIDONG_DILIAN` 继续消费旧 Z 分数。
- 工件隔离：上述 14 个接管信号与 3 个保留信号必须写入 production artifact manifest 和每日 score manifest 的 `replaced_signals/preserved_legacy_signals`。旧 Z 模型中 7 个被接管成员的 21 份 up/down 工件仅作 rollback 保留，不进入晋级后活动 artifact 合同；3 个低吸成员的 9 份工件继续生产消费。

每日运行先对目标日、行情月分区、两套信号缓存、生产 artifact、审批文件和合同源计算统一输入 fingerprint。目标日与 fingerprint 均未变化，且 feature/score checksum 可重开验证时，直接返回 `checkpoint_reused=true`，不重算也不重写；任一真实输入变化才重建。归一化参考及其 SHA 固定在生产 artifact，不能随每日候选池漂移。selector 快照缓存还会校验 `ranking_source`，禁止在切流后 fallback 到旧 legacy 快照。

若胜出的是 Beam 候选，shadow/production 封装要求项目 `BEAM_SCHEMA_VERSION`、`test_data_used=false`、`test_data_used_for_search=false`、候选与选中列表/hash、访问组合审计以及 permutation、独立 `pipeline_select`、汇总 development gate 全部一致且通过。当前冻结合同是固定 105 context、完整 13 增量 universe、无 prefilter、`width=4/min_features=6/max_remove=10`（13 候选下有效深度 7）的 Beam Residual v3 项目适配；旧 shortlist 或 `max_remove=2` 产物会被封装拒绝，不能以完整 v3 名义晋级。

`ranking_score` 只替换候选排序。现有 `opportunity_score/holding_score`、买入方案、卖出方案和二层 `playbook` 保持独立，不因第一层晋级被覆盖。

回滚只需在同一次评审变更中把 `selector.ranking_source` 恢复为 `legacy_z_skill` 并把 `promotion.enabled` 关闭，然后重启 `com.didi.quant.webapp`。不得删除旧工件，也不需要恢复 105 因子研究目录。
