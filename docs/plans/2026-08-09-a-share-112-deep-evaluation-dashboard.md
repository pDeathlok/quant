# A 股 112 家“好公司”逐股评估与筛选台实施计划

**日期：** 2026-08-09
**研究截止：** 2026-08-09T19:46:24+08:00
**目标日期：** 2027-08-09
**负责人：** Codex
**状态：** 执行中

## 1. 目标与完成定义

把 `industry_shortlist.csv` 的 48 家与 `niche_capability_watchlist.csv` 的 75 家按证券代码去重，形成唯一的 112 家研究母表；对每家公司执行 `analyze-a-shares` 的完整覆盖契约，生成可审计个股档案和一个独立、可离线打开的 HTML 筛选台。

只有同时满足以下条件才算完成：

1. 母表严格为 112 个唯一 `ts_code`，并保留“48 家广谱池/75 家细分池”的来源标识。
2. 112 家均有业务、行业、财务质量、治理、GQS、估值、三情景、量价、风险、催化剂和证据账本字段。
3. 每家公司均有一个冻结在本次截止时点的 `reports/a_shares/<ticker>/records/<record_id>/` 历史档案。
4. 估值未通过最小证据门槛时显示 `unavailable` 和缺失原因，不允许补造目标价。
5. HTML 内嵌 112 家结构化数据，支持搜索、组合筛选、多列排序、查看详情、加入本地自选、导出筛选结果。
6. 自动校验通过：112/112 唯一、权重求和正确、目标价与空间可复算、所有有限数值合法、页面无控制台错误、桌面和移动端可用。

## 2. 输出文件

### 2.1 可复用代码与契约

- `src/quant/research/good_company_deep_evaluation.py`
  - 数据清洗、历史财务聚合、GQS 规则、行业估值路由、三情景计算、量价状态、报告渲染。
- `scripts/research/build_good_company_deep_evaluation.py`
  - 命令行入口；读取 112 家母表、原始数据和外部补充快照，生成全部产物。
- `scripts/research/build_good_company_dashboard.py`
  - 将已校验 JSON/CSV 渲染成单文件 HTML；禁止在前端重新计算财务结论。
- `config/good_company_deep_evaluation_v0_1.json`
  - 固定分析截止、目标日、评分阈值、行业估值族、情景参数和缺失值政策。
- `tests/research/test_good_company_deep_evaluation.py`
  - 单元与契约测试。

### 2.2 本次冻结产物

- `reports/good_company_deep_20260809/universe_112.csv`
- `reports/good_company_deep_20260809/company_evaluations.csv`
- `reports/good_company_deep_20260809/company_evaluations.json`
- `reports/good_company_deep_20260809/evidence_audit.csv`
- `reports/good_company_deep_20260809/validation_summary.json`
- `reports/good_company_deep_20260809/good_company_dashboard.html`
- `reports/good_company_deep_20260809/design/dashboard-concept.png`
- `reports/good_company_deep_20260809/design/dashboard-detail-concept.png`
- `reports/good_company_deep_20260809/qa/dashboard-desktop.png`
- `reports/good_company_deep_20260809/qa/dashboard-mobile.png`

每家公司的正式归档仍写入既有历史目录：

- `reports/a_shares/<ticker>/records/<record_id>/record.json`
- `reports/a_shares/<ticker>/records/<record_id>/report.md`

## 3. 统一数据契约

每个 `company_evaluations.json` 元素必须包含：

```text
identity:
  ts_code, symbol, name, exchange, industry, broad_industry,
  in_broad_48, in_niche_75, scarcity_hypothesis, valuation_family
cutoff:
  analysis_cutoff, target_date, price_date, finance_available_at,
  forecast_available_at, latest_annual_period, latest_interim_period
market:
  current_price, market_cap, total_shares, pe_ttm, pb, ps_ttm,
  return_20d, return_60d, return_120d, volatility_20d, drawdown_250d,
  ma20_gap, ma60_gap, volume_ratio_20d, technical_state
financials:
  revenue_latest, net_profit_latest, revenue_yoy, net_profit_yoy,
  revenue_cagr_3y, net_profit_cagr_3y, gross_margin, net_margin,
  roe, roa, roic, ocf_to_net_profit, fcf_margin, debt_to_assets,
  net_debt_to_ebitda, dividend_payout, dilution_3y
forecast:
  fy1, fy2, fy3, eps_fy1, eps_fy2, eps_fy3,
  net_profit_fy1, net_profit_fy2, net_profit_fy3,
  sample_count_fy2, dispersion_fy2, source, as_of
gqs:
  a_customer_business, b_scarcity_moat, c_growth_reinvestment,
  d_returns_profitability, e_cash_accounting, f_resilience_risk,
  g_governance_allocation, gqs_r, forward_adjustment, gqs_f,
  classification, hard_gate, coverage_ratio, confidence,
  score_evidence[], score_limitations[]
valuation:
  status, method_primary, method_crosscheck, forecast_basis,
  bear{}, base{}, bull{}, sensitivity{}, missing_reasons[]
  # 每个情景：conditions, earnings_or_cashflow, multiple_or_rate,
  # target_price, price_upside, dividend_return, total_return
research:
  stance, summary, thesis_pillars[], strongest_counterargument,
  falsifiers[], catalysts[], risks[], monitoring[], revision{}
evidence:
  sources[], primary_source_count, source_tier_mix,
  data_conflicts[], mcp_execution{}
```

数字字段用 JSON 数值或 `null`；不能使用 `NaN`、`Infinity`、字符串百分比。币种统一人民币，市值和财务金额统一元，页面展示层再格式化为亿元。

## 4. 数据源和时间边界

### 4.1 本地可复算数据

- 证券主表：`data/raw/stock_basic.parquet`
- 日行情：`data/raw/daily_partitioned/year_month=*/data.parquet`
- 宽基：`data/raw/index_000300.SH.parquet`
- 利润表：`data/raw/income.parquet`
- 资产负债表：`data/raw/balancesheet.parquet`
- 现金流量表：`data/raw/cashflow.parquet`
- 财务指标：`data/raw/fina_indicator.parquet`
- 机构预测：`data/raw/analyst_forecasts.parquet`

所有披露数据必须满足 `available_at <= 2026-08-09T19:46:24+08:00`。同一报告期有多个版本时按 `update_flag`、公告时间和可用时间保留截止时点前最新版本。

### 4.2 外部补充

- 妙想金融数据：最新股本、市值、P/E、P/B、预测口径、公告搜索。
- 交易所/巨潮资讯/公司官网：年报、最新季报或中报、审计意见、重大交易、处罚、质押、回购与分红。
- 行业官方或协会资料：只用于供需、竞争格局和稀缺性核验。

外部数据必须保存来源 URL、发布时间、保守可得时间、查询文本和冲突处理。搜索摘要不直接当作事实；二级来源仅可用于定位一级材料或填补低重要性背景。

## 5. 历史档案处理

分析前先对 112 个代码执行历史检查。已命中：

- `601899.SH 紫金矿业`：继承 2026-07-29 黄金股比较基线；重点复核产量、现金流、金铜锂暴露和周期估值。
- `603605.SH 珀莱雅`：继承 2026-07-17 至 2026-07-19 的完整研究链；重点复核主品牌修复、花知晓并表、销售费用率、股息和技术支撑。

其余 110 家为首次覆盖。历史结论不是当前事实；新记录必须列出 `new_facts`、`belief_changes`、`valuation_changes` 和 `next_checks`。

## 6. GQS 评分实现

严格采用 `good-company-scorecard.md` 的 100 分制：

- A 客户价值与商业模式：10
- B 稀缺性、护城河与竞争地位：20
- C 成长质量与再投资跑道：10
- D 资本回报与盈利能力：20
- E 现金流与会计质量：15
- F 抗风险性：10
- G 治理与资本配置：15

规则：

1. 每个维度先按 0–5 分、0.5 分步长评分，再乘维度权重。
2. 量化指标采用 3–5 年绝对值、稳定性和行业分位数；不能只用单年高点。
3. B 维度的稀缺性必须同时回答“为何稀缺、如何变现、如何验证、反方是什么”；只有标签无证据不得超过 3 分。
4. G 维度若缺少处罚、质押、关联交易和资本配置核验，最高 3 分。
5. 前瞻调整仅限 `[-5,+5]`，必须绑定未来 12–24 个月可验证事实。
6. 覆盖率低于 75% 时置信度不得为高；低于 60% 时分类最高为“观察池”。
7. 财务造假、持续经营重大不确定性、严重治理红旗或不可解释的大额现金流背离触发硬门槛。
8. GQS 不包含估值和股价；估值吸引力在独立字段中排序。

## 7. 行业估值路由

### 7.1 银行与金融租赁

- 主方法：P/B–ROE（或剩余收益）。
- 情景变量：目标年 BVPS、可持续 ROE、股权成本、长期增长率、资产质量。
- 禁止：EV/EBITDA。

### 7.2 资源与强周期

- 主方法：中周期盈利 × 正常化 P/E，或 NAV/EV–EBITDA（数据足够时）。
- 情景变量：商品价格、产量、单位成本、资本开支、净负债。
- 不得用景气高点利润直接乘成长股估值。

### 7.3 公用事业、交通基础设施与 REIT-like 资产

- 主方法：DCF/DDM 或正常化 EV/EBITDA。
- 情景变量：利用率、费率、资本开支、负债成本、分红率。

### 7.4 消费、工业制造与大多数服务业

- 主方法：目标年稀释 EPS × 合理 P/E。
- 交叉检验：正常化 FCF 收益率或简化 DCF。
- 倍数锚：行业中位、公司历史、增长/ROE/现金流/治理调整，且写出理由。

### 7.5 软件、半导体与成长科技

- 盈利成熟：目标年 EPS × P/E，FCF/DCF 交叉检验。
- 尚未成熟：EV/Sales 或 DCF，并明确利润率路径；不得强行套 P/E。

### 7.6 医药、器械与创新产品

- 成熟产品：P/E + DCF。
- 创新管线：rNPV/SOTP；若缺少概率、峰值销售和现金消耗证据则目标价不可用。

## 8. 三情景规则

所有情景目标日固定为 2027-08-09，不赋主观概率。悲观/中性/乐观都必须包含：经营条件、盈利或现金流驱动、估值参数、目标价、价差、股息回报和总回报。

最小证据门槛：

1. 证券身份唯一；
2. 有时间戳且不晚于截止时点的价格；
3. 有审计年报和最新季度/中报；
4. 已检查重大公告；
5. 有独立预测输入或清晰自建模型；
6. 股本与潜在稀释已核验；
7. 估值方法符合行业。

任何一项失败，`valuation.status = unavailable`，三个目标价均为 `null`，页面展示缺失原因。可以保留条件式情景，但不得显示伪精确价格。

## 9. 量价模块

量价只做市场确认，不进入 GQS：

- 20/60/120 日收益、20 日波动率、250 日回撤；
- MA20/MA60 距离、20 日量比；
- 沪深 300 相对强弱；
- 状态：`强势确认 / 修复中 / 区间震荡 / 弱势未确认 / 数据不足`。

不得据此覆盖基本面或估值结论。

## 10. HTML 筛选台

独立单文件、无 CDN、无后端依赖。主要交互：

1. 搜索证券代码/公司名。
2. 多选行业、GQS 分类、来源池、估值状态、技术状态。
3. 滑块或数值输入：最低 GQS-R/GQS-F、最低覆盖率、最低中性空间、最大悲观跌幅。
4. 任意列升降序；默认先按“证据覆盖达标 → GQS-F → 中性空间”排序。
5. 详情抽屉显示 A–G 维度条、三情景桥、核心逻辑、最强反方、证伪条件、风险、催化剂、监测项和来源链接。
6. 自选列表保存在浏览器 `localStorage`；可只看自选并导出当前筛选 CSV。
7. 明确区分：好公司质量分、估值空间、证据覆盖、技术确认，不合并成一个黑箱“买入分”。
8. 红色表示上涨、绿色表示下跌，符合 A 股阅读习惯；同时用正负号和文字避免只靠颜色表达。

## 11. 测试与验收

### 11.1 自动测试

```bash
PYTHONPATH=src pytest -q tests/research/test_good_company_deep_evaluation.py
PYTHONPATH=src python scripts/research/build_good_company_deep_evaluation.py \
  --analysis-cutoff 2026-08-09T19:46:24+08:00 \
  --target-date 2027-08-09 \
  --output-dir reports/good_company_deep_20260809
PYTHONPATH=src python scripts/research/build_good_company_dashboard.py \
  --input reports/good_company_deep_20260809/company_evaluations.json \
  --output reports/good_company_deep_20260809/good_company_dashboard.html
```

预期：测试全通过，构建命令返回 0，`validation_summary.json` 中：

```text
expected_companies = 112
unique_companies = 112
complete_records = 112
invalid_numeric_values = 0
scenario_recalculation_errors = 0
history_archive_errors = 0
```

### 11.2 浏览器验收

- 桌面：1440×1000；移动：390×844。
- 检查初始加载、公司搜索、行业筛选、GQS 排序、空间排序、详情抽屉、自选持久化、CSV 导出。
- 浏览器控制台无错误；所有 112 家行可访问；`unavailable` 估值不会被排序成最高空间。
- 用 `view_image` 同时检查 ImageGen 概念图和实现截图，记录 5 个视觉对照点与文案差异。

## 12. 分阶段执行与安全回滚

1. **锁定母表与截止时点**：只生成 `universe_112.csv`，校验 48+75-11=112。
2. **历史回读**：读取紫金矿业与珀莱雅全部旧记录，保存基线 ID。
3. **校准样本**：用豪迈科技、宁波银行、紫金矿业、珀莱雅覆盖制造、银行、资源、消费四类估值路由；先通过测试再扩展。
4. **批量数据与评分**：112 家统一读取本地数据；外部来源按小批次补充并记录调用审计。
5. **逐股归档**：先生成临时目录，校验后再调用历史保存器；重复执行按内容哈希幂等。
6. **页面实现**：ImageGen 概念确认后编码静态 HTML，再做真实浏览器验收。

构建脚本只写 `reports/good_company_deep_20260809/` 和新增历史记录，不修改原始 parquet，不覆盖既有历史记录。若构建失败，删除本次新输出目录即可回滚；历史目录采用追加式不可变记录，失败批次在写入前必须完成整体验证。

## 13. 已知限制和披露措辞

- 这是一套统一口径的深度初筛与研究底稿，不等于 112 篇完全依赖人工访谈的卖方深度报告。
- 稀缺性、治理和行业结构仍包含判断；页面必须展示证据覆盖率和反方观点。
- 一致预期是市场输入，不是事实；预测分歧和样本数必须一并显示。
- 目标价是条件式情景输出，不是收益承诺或交易指令。
- 截止时点之后发布的财报、公告、行情和新闻不得混入本次冻结结果。
