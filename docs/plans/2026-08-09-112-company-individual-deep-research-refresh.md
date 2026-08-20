# A 股 112 家逐公司 Deep 研究重做计划

**日期：** 2026-08-09
**研究批次：** `good_company_deep_20260809`
**状态：** 执行中
**前置产物：** `reports/good_company_deep_20260809/universe_112.csv`（112 个唯一证券）

## 目标与完成定义

逐一重新研究 112 家公司。每家公司都作为独立的 `analyze-a-shares` Deep 任务执行，不能把批量财务代理分、通用风险模板或统一情景参数包装成个股深度结论。旧的短报告和 2026-08-09 19:46:24 批次归档只作为历史基线，保持不可变。

一家公司的状态只有同时满足以下条件才可标为 `completed`：

1. 在新研究截止时点之前运行并完整读取 `a_share_history context`；正文明确复述旧结论、旧论点、旧三情景和本次认知变化。
2. 使用项目正式数据层读取证券身份、至少五个完整财年、最新一期财务、股本、日线和同步基准；适用时扩展到 7—10 年。
3. 完成妙想 MCP 最小实际调用；若当前任务未加载或调用失败，按 skill 契约记录状态和降级路径，不能伪报成功。
4. 核验审计意见、年报、最新季报/中报、重大公告和影响核心论点的行业一级信源；研报与一致预期保持二级来源属性。
5. GQS A1—G5 共 29 个评分项逐项保留实现证据、前瞻证据、反证、证伪条件、置信度和前瞻可靠度；正确计算 GQS-R、GQS-F、覆盖率、红线、关键模块门槛和分类上限。
6. 完成行业适配的主估值、交叉估值、悲观/中性/乐观情景、两变量敏感性与反向估值；最低证据门失败时显式写“暂不提供精确目标价”。
7. 完成不少于 120 根日线的量价快照和宽基/行业相对强弱；数据不足则披露缺口，不猜测图形。
8. 生成公司专属的 2—4 个投资论点、最强反方、催化剂、风险和 5—10 项监测清单，不得与其他公司的核心文本机械重复。
9. 报告写入 `reports/good_company_deep_20260809/individual_reports/<ticker>.md`，并通过 `a_share_history save` 追加新的不可变记录。
10. 报告契约、数值重算、来源账本、链接、重复文本和历史归档校验全部通过。

## 固定路径

- 母表：`reports/good_company_deep_20260809/universe_112.csv`
- Deep 报告：`reports/good_company_deep_20260809/individual_reports/<ticker>.md`
- 进度台账：`reports/good_company_deep_20260809/deep_research_progress.csv`
- 单股工作底稿：`reports/good_company_deep_20260809/workpapers/<ticker>/`
- 不可变历史：`reports/a_shares/<ticker>/records/<record_id>/`
- 聚合数据：`reports/good_company_deep_20260809/company_evaluations.json`
- 筛选页：`reports/good_company_deep_20260809/good_company_dashboard.html`
- 报告契约：`config/good_company_deep_report_contract_v1.json`
- 校验器：`scripts/research/validate_individual_deep_reports.py`
- 测试：`tests/research/test_individual_deep_report_contract.py`

## 研究顺序

先用四家公司校准四类方法，再按母表顺序逐家完成：

1. `002595.SZ 豪迈科技`：工业制造、工艺/客户认证稀缺性、P/E + FCF/DCF。
2. `603605.SH 珀莱雅`：品牌消费、品牌矩阵与渠道效率、P/E + DCF。
3. `601899.SH 紫金矿业`：强周期资源、成本曲线/储量/中周期盈利、NAV 或中周期 EV/EBITDA。
4. `002142.SZ 宁波银行`：银行、资产质量/负债基础、P/B—ROE + 剩余收益。
5. 其余 108 家按 `universe_112.csv` 中的证券代码升序执行；每完成一家立即校验、归档和更新台账，不等待整批结束。

用户进度更新节点固定为 4、25、50、75、100、112 家；若某家公司证据门未通过，也只标为 `blocked_evidence`，不计入 completed。

## 单家公司操作契约

以 `002595.SZ` 为例，其他公司只替换证券代码、公司名、行业 playbook 和工作目录：

```bash
mkdir -p reports/good_company_deep_20260809/workpapers/002595.SZ

PYTHONPATH=src python -m quant.research.a_share_history context \
  --ticker 002595.SZ \
  --before 2026-08-09T23:59:59+08:00 \
  --output /tmp/a_share_context_002595_SZ.json

PYTHONPATH=src python -m quant.research.a_share_skill_data \
  --help

python .agents/skills/analyze-a-shares/scripts/price_volume_snapshot.py \
  reports/good_company_deep_20260809/workpapers/002595.SZ/price_volume_input.json \
  --format markdown

python .agents/skills/analyze-a-shares/scripts/scenario_valuation.py \
  reports/good_company_deep_20260809/workpapers/002595.SZ/scenario_valuation_input.json \
  --format markdown

PYTHONPATH=src python -m quant.research.a_share_history save \
  reports/good_company_deep_20260809/workpapers/002595.SZ/research_bundle.json

PYTHONPATH=src python scripts/research/validate_individual_deep_reports.py \
  --ticker 002595.SZ \
  --report reports/good_company_deep_20260809/individual_reports/002595.SZ.md \
  --contract config/good_company_deep_report_contract_v1.json
```

`--before` 在实际执行时必须使用当家公司研究开始时生成的、晚于旧记录且带 `+08:00` 的精确截止时间；上面的 `23:59:59` 只限定本批次 2026-08-09 的最晚边界，不能用于晚于该时点才开始的研究。每家公司最终以其报告内保存的 `analysis_cutoff` 为准。

## Deep 报告结构

每份 Markdown 按以下顺序写作：

1. 标题、分析截止、目标日和证券确认。
2. 研究快照：价格/股本/财务可得时点、数据口径、MCP 审计、历史基线。
3. 结论先行：倾向、核心上行、核心下行、最强反方、改变条件。
4. 认知更新：旧论点到新事实、变化分类、状态和模型影响。
5. GQS：红线与上限、A—G 模块、29 项审计表、GQS-R/F、覆盖率和分类门。
6. 三情景：故事、经营/资本/估值输入、目标价或不可用原因、空间、股息和证伪条件。
7. 2—4 个公司专属投资论点及替代解释。
8. 业务、价值链、行业、竞争、经济性同行及排除理由。
9. 五至十年财务、ROIC/增量 ROIC、现金流/应计、资产负债表、治理与资本配置。
10. 主估值、交叉估值、完整桥接、敏感性和反向估值。
11. 量价、基准相对强弱、关键 K、波动和观察锚。
12. 催化剂、风险、监测清单、下一次更新日。
13. 完整来源账本、数据冲突、缺口、MCP 执行审计、免责声明和历史归档结果。

## 校验门

实现 `scripts/research/validate_individual_deep_reports.py`，至少检查：

- 文件存在且证券代码、公司名和分析截止唯一一致。
- 规定章节全部出现；GQS 29 个 ID 不缺失，权重合计 100。
- `realized_coverage < 60%` 或 B/D/E/G 任一覆盖低于 50% 时不得给正式分类。
- “优质/卓越”满足 GQS-R、覆盖率、confidence 和 B/D/E/G 模块最低分。
- 有数值目标价时存在同日未复权基准价、股本/稀释、情景输入、交叉估值和可复算结果。
- 悲观目标价不高于中性，中性不高于乐观；空间与含息回报能重算。
- MCP 运行时可用且任务需外部金融数据时，`actual_call_count > 0`。
- 至少一个 A 级法定/监管来源和一个公司/行业特定来源；正文引用可定位。
- 历史基线、认知变化与新 `record_id` 存在。
- 核心论点、催化剂、证伪条件和最强反方不能与其他公司高度重复。

测试命令：

```bash
PYTHONPATH=src pytest -q tests/research/test_individual_deep_report_contract.py
PYTHONPATH=src python scripts/research/validate_individual_deep_reports.py \
  --universe reports/good_company_deep_20260809/universe_112.csv \
  --reports-dir reports/good_company_deep_20260809/individual_reports \
  --progress reports/good_company_deep_20260809/deep_research_progress.csv \
  --contract config/good_company_deep_report_contract_v1.json
```

最终预期：`completed=112`、`failed=0`、`archive_errors=0`、`duplicate_core_text=0`、所有数值校验误差在合同容许范围内。

## 聚合页面刷新

只有台账为 `completed` 的新 Deep 报告可以覆盖聚合评分和估值字段。全部完成后重新生成：

```bash
PYTHONPATH=src python scripts/research/build_good_company_dashboard.py \
  --input reports/good_company_deep_20260809/company_evaluations.json \
  --output reports/good_company_deep_20260809/good_company_dashboard.html
```

筛选页保留公司名、代码、行业、综合分、A—G 分项、悲观/中性/乐观目标价与空间、置信度、证据覆盖率、个股 Markdown 链接和雪球链接。未通过估值证据门的公司显示 `暂不提供`，不得以 0 或通用倍数替代。

## 回滚与数据安全

- `reports/a_shares/` 只追加新记录，不改写旧 `record.json` 或 `report.md`。
- 更新单股 Markdown 前，历史保存器已保留旧批次正文；失败时将台账状态设为 `failed_validation`，不修改聚合数据。
- 原始 parquet、旧聚合 JSON/CSV 和旧 HTML 不在逐股研究阶段改写；直到报告通过校验后才逐项同步结构化字段。
- 临时 `/tmp/a_share_context_*` 只在完整读取并形成新记录后删除。
- 不执行 `git reset`、`git checkout --` 或删除旧历史目录。
