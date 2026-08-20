# 右侧策略因子等价测试与二层交易方案模型实施计划

## 目标与边界

本轮完成两项研究验证，不改线上 selector、不替换现有模型、不发布交易方案：

1. 将 14 个右侧/混合策略从“规则因子列存在”升级为“生产筛股谓词逐项映射、边界值等价、因果可复现”的测试合同；
2. 在已冻结的统一选股分数之后增加一个研究型二层策略模型，从预注册的买入/卖出方案菜单中选择每个事件的可执行方案，并允许选择“不交易”。

第一层回答“这只股票是否值得进入候选排序”；第二层回答“在已进入候选的前提下，哪种入场与退出方案最合适”。二层模型不得改变或补造第一层信号，也不得使用 T 日收盘以后才知道的特征。

## 当前缺口

本轮开始时，`src/quant/research/right_side_unified_features.py` 将 14 个策略的 105 个规则因子列纳入统一模型，`SIGNAL_FEATURE_REQUIREMENTS` 也覆盖全部策略。现有测试已经验证列合同、非空覆盖、前缀因果性，并验证 10 个 Z 策略的 canonical 信号与当前 live detector 同日等价；B2、B3、Vegas、三倍量突破则按 Web 当前消费的 family cache 列合并。

但“信号等价”不等于“每一个原始筛股子条件都作为模型可见因子被证明等价”。现有测试仍缺：

- 每个生产谓词的稳定标识、阈值、方向和对应因子列；
- 阈值上下边界和历史状态锚点的逐项测试；
- family cache 四个策略的来源列、优化参数和配置版本证明；
- 训练列最终确实包含全部规则因子的端到端断言；本轮等价审计补齐后合同为 118 列；
- 逐策略命中切片中的有限值、非空率和退化状态审计。

旧 `train_z_skill_models_and_backtest.py` 只从网格中为每个策略静态选择一套 playbook，并非逐事件二层模型；它还不能直接作为本轮干净基线。

## 任务 1：生产谓词到模型因子的版本化合同

新增 `src/quant/research/right_side_factor_parity.py`：

- 定义 14 个策略的 `PredicateFactorContract`；
- 每个 predicate 记录生产事实来源、规则版本、阈值语义、所需历史、对应的连续 margin/状态/布尔因子；
- 区分 `live_detector` 与 `web_family_cache` 两类权威来源，禁止把旧 Z cache 与 live detector 混用；
- 提供合同审计函数，检查每个 predicate 至少有一个模型可见因子、所有映射因子均属于 `RULE_FEATURE_COLUMNS`、全部规则因子均被至少一个 predicate 或共享原子解释；
- 提供逐策略数据审计，输出 `event_rows/non_null_rate/finite_rate/unique_values/status`，空策略、全空或非有限必需因子直接失败。

修改 `src/quant/research/right_side_unified_features.py` 时只允许补充缺失的连续 margin 或精确状态，不重定义 canonical 信号。若某个生产布尔谓词目前只能从多个原子重建，合同必须列全依赖，不能用策略身份列冒充原始因子。

新增 `tests/test_right_side_factor_parity.py`：

- 合同恰好覆盖 14 个策略且无孤立 predicate/孤立规则因子；
- 10 个 Z 策略在合成边界样本和逐前缀真实样本上，因子重建谓词与 live detector/canonical flag 一致；
- B2/B3 的 family 子列 OR 与 Web 合同一致；Vegas 固定 optimized 参数；三倍量突破固定生产 YAML/config hash；
- 阈值等于、略低、略高三点测试；KENGQI/ZAIHOU/DOUBLE_GUN 等事件锚点使用同一真实事件索引；
- 训练特征选择端到端包含全部 `RULE_FEATURE_COLUMNS`，且 `unified_without_signal_id` 与带身份版本只差身份列。

## 任务 2：预注册可执行交易方案与反事实结果

新增 `src/quant/research/right_side_playbook_policy.py`：

- `EntryConstraint`：`next_open`/`next_close`、次日有交易记录、非停牌、非一字涨停；`next_open` 额外排除开盘即涨停而日线无法证明可在开盘价成交的事件；可选固定的高开/低开范围，但所有范围必须预注册；
- `ExitPolicy`：固定到期收盘、固定止盈止损、目标激活后的移动止盈；最早退出日为 T+2，遵守 A 股 T+1；
- `PlaybookSpec`：买入约束、退出规则、往返成本、日线同日止盈止损冲突规则和版本；
- 固定首轮小菜单，控制多重比较：`next_open/next_close × expiry_T3/T5 × TP4%-SL2%-T5 × TP6%-SL3%-T5`，再加一个 `NO_TRADE` 动作；
- 输出 event × playbook 长表，包括 `eligible/maturity_reason/entry_date/entry_price/exit_date/exit_price/exit_reason/gross_return/net_return/mae/holding_sessions/ambiguous_bar`；
- 同一日同时触发止盈和止损时采用预注册保守顺序（止损优先），并另行标记用于敏感性分析；尾部窗口不完整保持未成熟，不能当负样本。

新增 `tests/test_right_side_playbook_policy.py`：

- 次日一字板、开盘封板、停牌/缺失日、次日高低开门槛；
- next_open 与 next_close 入场价不同；
- T+1 买入当日不能卖，首个可卖日为 T+2；
- 到期、止盈、止损、跳空止损、移动止盈和同日双触发；
- 手续费/滑点只扣一次预注册往返成本；
- `NO_TRADE` 始终可选且净收益为 0；
- 未成熟标签为空、未来追加数据不改变已成熟历史结果。

## 任务 3：二层共享策略模型

新增 `src/quant/research/right_side_playbook_model.py`：

- 训练样本为 event × eligible playbook 长表；
- 输入包含 T 日可见且在训练折达到覆盖门槛的项目因子、118 个规则因子、14 个 signal identity、第一层 OOF/测试预测，以及 playbook one-hot/参数；不得使用真实未来收益、退出原因或由 T+1 行情生成的 eligibility 细节作为打分特征；
- 一个共享回归/排序模型在 T 日收盘后预测每个动作的预注册风险调整效用并先选定计划；`NO_TRADE` 效用固定为 0，模型预测不优于 0 时放弃交易；
- T+1 的可成交条件只在已选计划上执行：若条件失败则取消为 `NO_TRADE`，不得利用次日信息事后改选另一种入场方式。首轮不启用 fallback；以后若增加，只允许预注册、按时间可执行的单向 `next_open → next_close` fallback，不能在收盘后回选开盘动作；
- 首轮效用固定为 `net_return - 0.25 * abs(min(mae, 0))`，成本固定写入版本，不允许根据验证结果调参；
- 第一层预测必须是同折 OOF 或对应测试预测，禁止将训练内拟合分数喂给二层造成堆叠泄漏；
- 只在时间走步折中训练，先使用 A/B 做开发。既有 C 已被第一层研究查看，二层不得再把它声称为 untouched；最终确认依赖后续新增时间段影子数据。

新增 `scripts/research/validate_right_side_playbook_models.py`：

- `build-outcomes`：流式生成方案结果长表与 manifest；
- `audit`：检查键唯一、eligibility、成熟度、T+1、成本、动作覆盖和特征时点；
- `train --folds A B`：比较 `static_global`、`static_per_signal`、`shared_playbook_model` 和 `no_trade`；
- `report`：逐折、逐信号报告选中方案分布、覆盖率、净单笔收益、下行尾部、PF 和配对月块区间；明确这些仍是事件级统计，不是资金曲线。

新增 `tests/test_right_side_playbook_model.py`：

- T 日可以预选后来不可成交的动作，但 T+1 必须取消而不能事后换成另一动作；
- 所有预测效用不大于 0 时选择 `NO_TRADE`；
- 合成数据中模型能学习不同上下文对应不同方案；
- 同一事件的动作不能跨时间折；
- 训练输入拒绝未来结果列和非 OOF 第一层分数；
- artifact/manifest 固定记录特征、方案目录、成本、效用公式和数据截止日。

## 任务 4：验收与报告

先运行合同与单元测试：

```bash
PYTHONPATH=src pytest -q tests/test_right_side_factor_parity.py tests/test_right_side_playbook_policy.py tests/test_right_side_playbook_model.py
PYTHONPATH=src pytest -q tests/test_right_side_unified.py tests/test_right_side_paired_comparison.py tests/test_right_side_freeze_decision.py tests/test_right_side_c_confirmation.py
```

再运行小样本端到端 smoke，必须覆盖两个入场方式、至少三个退出规则、一个不可成交事件和 `NO_TRADE`。只有单测与 smoke 都通过后才允许构建全量 outcome 数据。

最终报告必须分别回答：

- 14 个策略的每个生产谓词是否有模型可见因子，哪些是精确连续 margin、哪些是布尔状态、哪些仍依赖 family cache；
- 二层模型相对静态全局/逐策略方案是否在同一事件、同一成本、同一月份上稳定改善；
- 改善来自更好的退出方案选择、入场方式选择还是更多 abstain，不能只给汇总收益；
- 日线无法解决的盘中成交顺序、历史涨停价覆盖和资金占用限制必须列为发布阻塞项。

验收通过仍只代表“值得影子运行”，不代表上线批准。

## 2026-08-13 实施结果

- 谓词合同覆盖 14 个策略、32 个稳定 predicate；10 个 Z 策略为 `exact_live`，B2/B3/Vegas/三倍量为当前生成器精确重建并保留历史 cache caveat；
- 规则因子从 105 增至 118，补齐 B1/B2 隐含原子、B3 三个 Web 分支、Vegas 历史/可交易/最终状态、三倍量 merged 状态，并修正 family 完整窗口口径和 YUEYUE 低价分母；
- 二层目录固定为 8 个交易动作加 `no_trade`，完整参数、顺序和 SHA256 可冻结；
- 二层决策采用 T 日预选、T+1 执行门失败则取消的时点合同；
- 全部 `test_right_side_*.py` 为 89 passed；另用 400 个真实 A/B 事件生成 3,600 个反事实动作行并完成 A→B 二层拟合烟测；
- 现有 1.4GB 事件表和已冻结模型仍是旧 105 因子口径。要比较新增因子的效果，必须重建事件数据并重新训练；本轮没有启动全量构建或生产发布。
