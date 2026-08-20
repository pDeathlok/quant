# 右侧与混合策略统一模型验证合同

> 后续因子等价审计更新（2026-08-13）：本合同首轮数据和冻结模型使用 105 个规则因子；`right_side_factor_parity_v2` 已发现并补齐 13 个生产筛股原子/分支，当前代码合同为 118 个。已有冻结结果仍只代表旧 105 因子口径；任何“全部筛股逻辑已纳入”的新结论都必须先重建数据并重训。

## 目标与结论边界

本轮只在研究目录验证“14 个右侧/混合信号共享一个模型”是否优于同合同下重新训练的成员独立模型，不改线上 selector，不替换现有模型。

主决策基线是本轮重新训练的 `independent`：它与统一模型使用相同样本、标签、因子、走步切分和 XGBoost 参数。统一模型只有在 A/B 开发折稳定改善、冻结方案后 C 折仍改善，且逐信号宏观指标没有明显退化时，才值得进入组合级回测或影子运行。

当前收益指标是重叠事件的单笔统计，不是资金曲线；不能据此报告复利收益或最大回撤。即使模型排序改善，发布前仍需另做持仓、现金占用、涨跌停、滑点和逐日盯市完整的组合回测。

## 策略与信号事实来源

当前 Web 短线目录中的右侧和混合策略共 10 个组、14 个成员：

- 右侧：`B2`、`B3`、`KEY_K`、`VIOLENCE_K`、`PINGHANG`、`DOUBLE_GUN`、`CHANGAN`、`KENGQI`、`VEGAS`、`TRIPLE_VOLUME_BREAKOUT`；
- 混合：`GOLDEN_BOWL`、`ZAIHOU`、`BREATHING`、`YUEYUE`。

样本信号使用版本化合同 `right_side_unified_signal_v1_live_z_20260813`：

- `CHANGAN`、`PINGHANG`、`DOUBLE_GUN`、`GOLDEN_BOWL`、`BREATHING`、`KENGQI`、`ZAIHOU`、`YUEYUE`、`KEY_K`、`VIOLENCE_K` 按当前线上 detector 谓词逐股、逐日因果回放，不使用历史 Z 缓存的漂移规则，也不执行其 5 个交易日去重；
- `B2`、`B3`、`VEGAS`、`TRIPLE_VOLUME_BREAKOUT` 继续读取当前 Web selector 使用的 `data/features/b1/b1_family_rule_candidates.parquet`。B2/B3 对各子版本列做 OR，Vegas/三倍量突破读取生产配置生成的直接信号列；
- 同一股票同一日期多信号命中只保留一条事件，保存 14 个 multi-hot 列、B2/B3 子版本列和 `signal_count`，不重复放大样本权重。

旧 `data/features/z_skill_daily_candidates.parquet` 不再是主样本事实来源。

## 样本、入场与标签

决策时点固定为 T 日收盘后，特征最多使用 T 日数据。事件和标签分开流式写入并在成功后原子发布：

- `unified_right_side_dataset.parquet`：每个 `symbol/date` 一行事件和因子；
- `unified_right_side_labels.parquet`：每个事件按 `entry_mode/horizon` 展开长表，避免把同一大特征矩阵为两种入场和三个窗口重复六次。

两种入场分别训练：

- `next_open`：T+1 开盘买入；
- `next_close`：T+1 收盘买入，但特征仍截止 T 日。

A 股 T+1 约束下，最早可卖日为 T+2。标签按 3、5、10 个可卖交易日生成 `mfe`、`mae`、`terminal_return`、`hit_up3`、`hit_up5`、`hit_up8`、`hit_down3` 和 `good_path5`。其中主实验为 5 日 `good_path5`，`hit_up5`、`hit_up8` 作为上行标签敏感性分析；`hit_down3` 只作风险诊断，CLI 明确禁止把它作为买入排序标签。

标签严格使用交易所日历定位 T+1/T+2，不用个股本地 `shift(-1)` 跨过停牌日。缺少入场日、停牌、缺少完整未来窗口或窗口越过数据截止日的行均标记未成熟，所有结果标签保持空值，不能写成负样本。

T+1 一字涨停在训练前剔除：优先用历史 `up_limit` 与原始未复权 OHLC 精确判定；缺失时用原始 OHLC 一价且相对前收至少上涨 4.8% 的代理规则。每行保存 `locked_limit_up` 和 `locked_limit_source`。当前早期历史精确涨跌停价覆盖不足，因此代理期结果只能视为初步验证；正式发布判断前必须回填历史 `up_limit` 并复跑敏感性检查。普通高开或盘中打开涨停不在本轮“一字板”剔除范围内。

## 因子合同

主因子口径固定为 `project-v4-causal-price-alpha`。事件表保留 147 个项目因子合同列，当前主构建实际物化 112 个因果市场因子；`daily_basic` 估值、换手和市值数据不进入本轮主实验，manifest 固定记录 `daily_basic_included=false`。其余未物化项目列保持空值，训练时只从训练集按非空覆盖率至少 50% 接纳，不看验证或测试效果。

首轮固定物化 105 个右侧规则原子因子。后续等价审计补齐 B1/B2 隐含原子、B3 三个实际分支、Vegas 最终状态和三倍量 merged 状态后，当前合同为 118 个，覆盖 14 个信号的筛选状态。重建后的 118 个规则因子必须全部进入各模型，不进行表现筛选。

`factor_coverage.csv` 同时输出全体和逐信号的行数、非空率、唯一值数及 `ok/sparse/constant/all_null/missing` 状态。缺列或全空的信号必需因子会使审计失败；常数和稀疏列保留为诊断，因为命中某条规则后部分布尔谓词天然可能为常数。

## 走步验证与实验矩阵

时间折固定如下，训练集和验证集都按 `label_end_date` 清除跨边界标签窗口：

| 折 | 训练 | 验证 | 测试 | 用途 |
|---|---|---|---|---|
| A | 2020-2022 | 2023 | 2024 | 开发 |
| B | 2020-2023 | 2024 | 2025 | 开发与稳定性确认 |
| C | 2020-2024 | 2025 | 2026 截至数据截止日 | 方案冻结后的最终确认 |

每个验证年再按交易日顺序拆成三个互不重叠阶段：前 50% 只用于 XGBoost 早停，中间 25% 只用于 Platt 概率校准，最后 25% 只用于阈值选择。每段至少要求两个标签类别，验证期至少 60 个交易日。测试集不参与早停、校准、阈值或标签选择。

每个入场/标签组合比较五个口径：

1. `rule_only`：候选池常数基准率，不代表规则内部排序；
2. `independent`：逐成员按同合同重新训练，是主公平基线；
3. `unified_without_signal_id`：统一样本，使用项目因子和当前完整规则因子合同，不加入身份列；
4. `unified_with_signal_id`：在 3 上加入 14 个 multi-hot、B2/B3 子版本和池类型列；
5. `unified_balanced`：在 4 上按交易日和信号频率平衡权重。

独立成员训练样本少于 2,000、少数类不足或有效交易日不足时标记样本不足，并用训练期基准率回退；报告分别展示全部事件和“独立模型实际可训练行”的配对比较，防止把稀疏成员覆盖收益误称为纯模型提升。

模型指标包括 ROC-AUC、PR-AUC、Brier、Top-decile 精确率/lift 和逐信号指标。收益层只对每日预测 Top-K 事件统计扣除往返成本后的平均/中位单笔收益、胜率和交易级 PF。统一模型相对独立模型的 PR-AUC、Top-lift 和每日 Top-K 平均终值收益按相同测试行计算差值，并用自然月整块配对 bootstrap 给出区间；该区间只衡量测试期时间块不确定性，不包含重新训练不确定性。

## 旧线上工件重合基线

`src/quant/research/right_side_legacy_artifact_baseline.py` 提供只读诊断：加载 `models/research/z_skill/` 中 7 个成员 × `up5/up8/down3` 的 21 个旧工件，只在当前可成交事件与持久化旧因子表 `data/features/z_skill_model_dataset.parquet` 的精确 `symbol/date` 交集上评分，并单独输出旧因子覆盖率和旧信号时点重合率。

它不是主公平基线，原因是：旧工件依赖 `project-v1-latest-scale-global-rank`，不能用同名 v4 因子评分；旧信号含 5 日去重且多个 detector 已漂移；旧标签未执行本轮交易日历、成熟度和次日一字板合同；2025 年以前参与拟合或早停，2025 年以后虽曾作 OOT，但已用于后续工件/运行方案选择。`up5/up8` 仅在新样本门控后与 5 日开盘标签近似对应，旧 `down3` 与当前入场价 MAE 定义不兼容。该模块当前未接入主 CLI，结果只能报告“历史工件在可重合样本上的参考表现”，不能宣称干净样本外或全池覆盖。

## 实现文件与产物

- `src/quant/research/right_side_unified_signals.py`：14 信号的版本化事实来源与实时 Z 因果回放；
- `src/quant/research/right_side_unified_features.py`：当前 118 个规则因子、逐信号需求和覆盖审计；
- `src/quant/research/right_side_factor_parity.py`：生产谓词到模型因子的版本化等价合同；
- `src/quant/research/right_side_unified_labels.py`：交易日历、T+1 入场、一字板和成熟标签；
- `src/quant/research/right_side_unified.py`：去重、走步切分、权重、模型指标和交易级统计；
- `src/quant/research/right_side_paired_comparison.py`：相同测试行的月度整块配对比较；
- `src/quant/research/right_side_legacy_artifact_baseline.py`：旧 Z 工件交集诊断；
- `scripts/research/validate_unified_right_side_models.py`：分流构建、审计、训练和报告 CLI；
- `tests/test_right_side_unified.py`、`tests/test_right_side_paired_comparison.py`、`tests/test_right_side_legacy_artifact_baseline.py`：合同测试。

默认可再生产物：

- `data/research/right_side_unified/unified_right_side_dataset.parquet`
- `data/research/right_side_unified/unified_right_side_labels.parquet`
- `data/research/right_side_unified/dataset_manifest.json`
- `models/research/right_side_unified/`
- `reports/research/right_side_unified/factor_coverage.csv`
- `reports/research/right_side_unified/sample_audit.json`
- `reports/research/right_side_unified/model_metrics.csv`
- `reports/research/right_side_unified/signal_metrics.csv`
- `reports/research/right_side_unified/test_predictions.parquet`
- `reports/research/right_side_unified/paired_model_comparison.csv`
- `reports/research/right_side_unified/validation_report.md`

## 执行顺序

先验证合同并重建样本：

```bash
PYTHONPATH=src pytest -q tests/test_right_side_unified.py tests/test_right_side_paired_comparison.py tests/test_right_side_legacy_artifact_baseline.py
PYTHONPATH=src python scripts/research/validate_unified_right_side_models.py build-dataset --start-date 2020-01-01 --end-date 2026-08-12 --workers 8
PYTHONPATH=src python scripts/research/validate_unified_right_side_models.py audit
```

A/B 只用于开发。第一轮跑两种入场的主标签，再跑冻结前需要的上行标签敏感性：

```bash
PYTHONPATH=src python scripts/research/validate_unified_right_side_models.py train --entry-mode next_open --horizon 5 --label good_path5 --folds A B
PYTHONPATH=src python scripts/research/validate_unified_right_side_models.py train --entry-mode next_close --horizon 5 --label good_path5 --folds A B
PYTHONPATH=src python scripts/research/validate_unified_right_side_models.py train --entry-mode next_open --horizon 5 --label hit_up5 --folds A B
PYTHONPATH=src python scripts/research/validate_unified_right_side_models.py train --entry-mode next_open --horizon 5 --label hit_up8 --folds A B
```

根据 A/B 冻结入场、标签、模型臂和阈值流程后，只对冻结组合运行 C；下例以开盘买入、5 日 `good_path5` 为例：

```bash
PYTHONPATH=src python scripts/research/validate_unified_right_side_models.py train --entry-mode next_open --horizon 5 --label good_path5 --folds C
PYTHONPATH=src python scripts/research/validate_unified_right_side_models.py report
```

验收必须同时满足：事件键和标签键无重复；尾部未成熟标签为空；一字板不进入成熟训练样本；14 个信号均有覆盖记录；当前 118 个必需规则因子无缺失/全空；A/B 选择过程与 C 最终确认明确分离；报告不把交易级统计描述为组合资金曲线。

若统一模型没有稳定改善，保持线上不变。任何线上接入、旧模型替换或组合回测均另开实施任务并重新验收。
