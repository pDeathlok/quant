# A 股研究复盘与认知迭代

## 1. 目标与边界

历史档案保存“当时知道什么、相信什么、为什么、哪些条件会推翻”，使同一标的下一次被分析时可以检验旧判断。历史回看适用于所有同标的重复任务，不限于财报或事件更新。档案不是事实真值库，不替代重新核验，也不允许用后来结果重写旧底稿。

默认项目位置：

```text
reports/a_shares/<代码.交易所>/
├── index.json
├── latest.json
└── records/
    └── <analysis_cutoff>-<content_hash>/
        ├── record.json
        └── report.md
```

`reports/` 已由项目忽略，不会把研究底稿、许可数据或敏感假设误提交到 Git。若用户要求可版本化共享，再明确选择脱敏后的独立目录。

## 2. 同标的分析前强制读取

唯一确认证券代码和本次 `analysis_cutoff` 后，在收集新材料或形成新结论前，先加载严格早于本次截止时点的历史上下文：

```bash
PYTHONPATH=src python -m quant.research.a_share_history context \
  --ticker 600519.SH \
  --before 2026-08-31T18:00:00+08:00 \
  --output /tmp/a_share_context_600519.json
```

`context` 沿 `baseline_record_id` 链返回最近一次完整覆盖、其后的所有跟进记录以及最近基线的完整正文。完整内容写入 `--output`，命令行只打印摘要；必须读取完整文件。`anchor_is_full_coverage=false` 表示历史链没有完整覆盖锚点，`chain_complete=false` 表示链断裂或异常，两者都必须披露。若项目版本尚无该子命令，才使用 `baseline`、`list` 和精确 `record_id` 的 `show` 组合逐条加载：

```bash
PYTHONPATH=src python -m quant.research.a_share_history baseline \
  --ticker 600519.SH \
  --before 2026-08-31T18:00:00+08:00

PYTHONPATH=src python -m quant.research.a_share_history show \
  --ticker 600519.SH \
  --record-id 20260430T180000p0800-abcdef1234
```

然后：

1. 同时加载最近完整覆盖、最近基线以及两者之间的所有更新；不得让最近一条窄主题报告遮蔽完整投资结论。
2. 冻结旧记录，不修订旧措辞。
3. 建立基线快照：旧截止、旧结论、旧论点原文与状态、三情景、估值输入、证伪条件、监测项和未解决问题。
4. 将新披露逐条与旧假设、证伪条件和预测比较。
5. 在新报告中明确列出维持、强化、弱化、失效、新增和尚待验证的部分。

只显示 `baseline_record_id`、只说“参考过往分析”或只复述本次结论，都不算完成历史回看。用户没有主动要求复盘，也不能跳过。

若没有基线，标记“首次建档”，不虚构市场共识或自己的旧观点。

迁移自旧 Codex 对话的记录可能存在结构化占位值或论点 ID 漂移。此时：

- 以 `report.md` 原文为权威，不把结构化缺失解释成旧报告没有相关判断。
- 同一 ID 的命题不同，按语义重新映射并披露冲突；不同 ID 的命题相同，保留旧命题并记录映射。
- 将映射持久化到 `revision.pillar_id_mappings`，不得只在正文临时解释。
- 最近基线是窄主题时，同时引用最近完整覆盖中的总体倾向、目标价/未估值原因和核心经营假设。

完成读取后只清理本次生成的临时上下文文件，不修改正式历史记录。

## 3. 认知变化分类

| 分类 | 含义 | 处理 |
|---|---|---|
| `new_information` | 当时不可知的新事实改变判断 | 更新结论，不能记作旧分析错误 |
| `prior_error` | 当时已有证据足以反驳，但旧分析遗漏或误判 | 明确承认错误和改进规则 |
| `model_limitation` | 模型结构、口径或敏感性不足 | 修订模型并保留前后结果 |
| `noise` | 新变化尚不足以影响长期逻辑 | 维持判断，写出继续观察条件 |
| `unresolved` | 证据冲突或关键输入仍缺失 | 降低置信度，不强行选边 |

BAD：

```text
业绩低于预期，因此原投资逻辑错误。
```

GOOD：

```text
P2 从 active 调整为 weakened。2026H1 毛利率低于原中性假设 2.1pct；
其中约 1.4pct 来自当时不可知的新促销政策（new_information），约 0.7pct
来自旧模型遗漏渠道返利敏感性（prior_error）。因此下调 FY2026E 利润率，
但品牌壁垒 P1 暂不变化。
```

## 4. 变化桥

每次同标的再次分析至少形成以下链条：

```text
旧记录与旧假设
→ 新事实及 available_at
→ 与旧预测/证伪条件的差异
→ 认知变化分类
→ 论点状态变化
→ 经营模型变化
→ 三情景与估值变化
→ 下一次可检验条件
```

使用跨期稳定的论点 ID。不得因为标题改写就生成全新 ID，也不得复用旧 ID 表示不同命题。

论点状态只用：

- `strengthened`
- `unchanged`
- `weakened`
- `invalidated`
- `new`

## 5. 事后复盘

复盘同时评价四个维度：

| 维度 | 问题 |
|---|---|
| 信息集 | 当时是否遗漏已经公开且可得的重要信息？ |
| 推理 | 因果链、替代解释和证伪条件是否合理？ |
| 校准 | 置信度、三情景范围和敏感性是否与不确定性匹配？ |
| 结果 | 后来经营、估值和价格结果如何？哪些是运气或外生冲击？ |

结果正确不等于推理正确；结果错误也不自动证明当时分析不合理。价格复盘必须分开基本面兑现、估值倍数变化、市场 beta 和公司行动，避免只看涨跌归因。

## 6. 保存契约

先生成模板：

```bash
PYTHONPATH=src python -m quant.research.a_share_history template \
  --ticker 600519.SH \
  --company-name 贵州茅台
```

研究包必填：

- `ticker`、`company_name`、带时区的 `analysis_cutoff`、`mode`。
- `trigger`：类型、摘要和来源引用。
- `conclusion`：倾向、置信度和摘要。
- `thesis`：稳定论点 ID、反方与证伪条件。
- `scenarios`：悲观、中性、乐观。
- `monitoring`、`evidence_ledger`、`report_markdown`。
- `data_snapshot.mcp_execution`：至少保存运行时是否加载、是否尝试、实际/成功调用次数、采用与复核字段、状态/失败分类和降级路径；没有调用也不能省略。

更新记录另填：

- `baseline_record_id`：严格早于本次截止时点；项目工具可自动挂接最近记录。
- `revision.trigger_summary`。
- `revision.new_facts`。
- `revision.belief_changes`。
- `revision.model_changes`。
- `revision.valuation_changes`。
- `revision.mistakes_and_lessons`。
- `revision.next_checks`。
- `revision.pillar_id_mappings`：仅在迁移旧记录存在 ID 漂移或冲突时填写；每项记录旧 `record_id`、旧论点 ID/原文、新论点 ID 和映射理由。

保存：

```bash
PYTHONPATH=src python -m quant.research.a_share_history save research_bundle.json
```

工具按截止时点和内容哈希创建不可变目录；相同内容重复保存是幂等操作。`latest.json` 只是指针，权威历史仍是 `records/` 和 `index.json`。

## 7. 更新报告最小结构

1. `baseline_record_id`、旧截止时点和旧结论。
2. 不超过 8 条新事实及各自 `available_at`。
3. 旧预测与实际/新信息差异。
4. 逐论点状态与认知变化分类。
5. 预测、三情景、估值和监测项的前后桥接。
6. 最强反方与尚未解决的问题。
7. 本次新记录 ID 和保存位置；保存失败时写明原因。

若用户只是再次说“分析某股”而没有给出明确事件，也必须使用上述结构；可以把触发类型记为定期复盘，但不得按首次建档重新生成一份与旧报告脱节的结论。

## 8. 完成检查

- [ ] 基线严格早于本次截止时点，没有未来记录参与比较。
- [ ] 唯一确认代码后、收集新材料前已查询同标的历史；命中后已加载完整 `record.json` 和 `report.md`。
- [ ] 报告不只展示基线 ID，还复述了旧结论、旧论点和旧三情景，并给出本次逐项处理。
- [ ] 没有覆盖或美化旧底稿。
- [ ] 每项认知变化都指向新证据或明确的旧错误。
- [ ] 论点 ID 跨期稳定，状态变化可追踪。
- [ ] 结果评价与推理质量分开。
- [ ] 新记录已实际保存并返回记录 ID，或如实披露保存失败。
