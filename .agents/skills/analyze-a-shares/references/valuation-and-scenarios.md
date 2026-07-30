# 估值与三情景价格框架

## 目录

1. 先选模型
2. 常用模型和公式
3. 三情景构建
4. 目标价、股息和概率
5. 敏感性与反向估值
6. 强制校验门
7. 计算脚本输入

## 1. 先选模型

根据价值驱动选择主方法和交叉方法，不以数据方便决定方法。

| 公司特征 | 主方法 | 常用交叉方法 | 避免 |
|---|---|---|---|
| 成熟、现金流可预测的非金融公司 | FCFF/FCFE DCF | forward P/E、EV/EBITDA | 用历史高增长永久外推 |
| 稳定盈利消费/制造 | forward P/E 或 DCF | PEG 只作辅助、EV/EBITDA | 忽视资本开支和营运资金 |
| 重资产、公用事业 | DCF/DDM、EV/EBITDA | P/B、股息率 | 把受监管回报当自由定价 |
| 银行 | P/B-ROE、股利/剩余收益 | forward P/E | 常规 EV/EBITDA、把存款当普通债务 |
| 保险 | P/EV、剩余收益/DDM | forward P/E、SOTP | 用普通工业企业 FCF |
| 券商 | 中周期 P/B-ROE | normalized P/E、SOTP | 用牛市峰值利润 |
| 周期/资源 | 中周期 EV/EBITDA、NAV | P/B、情景 DCF | 峰值低 P/E 判定便宜 |
| 多业务集团 | SOTP | DCF、整体倍数 | 用不匹配的单一倍数 |
| 亏损成长/软件 | EV/Sales 或单位经济 DCF | SOTP、远期盈利倍数 | 负 EPS 的 P/E |
| 创新药/二元项目 | 风险调整 NPV、SOTP | 管线交易/可比倍数 | 未经概率调整的峰值销售 |
| 房地产/资产型 | 调整 NAV、清算/情景价值 | P/B、现金流 | 忽视受限现金和表外义务 |
| 困境企业 | 概率情景、清算/重组价值 | 期权式上行 | 继续经营单点 P/E |

阅读 `<skill-dir>/references/sector-playbooks.md` 的行业 KPI 后再定输入。

## 2. 常用模型和公式

### 每股倍数

```text
P/E 目标价 = 目标期归母稀释 EPS × 目标 P/E
P/B 目标价 = 目标期归母 BVPS × 目标 P/B
P/S 目标价 = 目标期每股收入 × 目标 P/S
```

目标倍数应由同行同日同预测期、公司自身历史区间和基本面差异共同解释。增长、ROE、利润率、风险、资本强度和会计质量不同，不能直接套行业中位数。

### 企业价值倍数

```text
目标 EV = 目标期经营指标（如 EBITDA）× 目标倍数
目标普通股权价值 = 目标 EV
                   - 有息债务
                   - 少数股东权益
                   - 优先股及其他非普通股索取权
                   + 可自由支配现金
                   + 非经营性投资/资产
目标价 = 目标普通股权价值 / 目标期稀释后普通股数
```

明确经营指标和资产负债表桥接属于当前时点还是目标时点。若用目标期 EBITDA 与当前净债务，说明近一年现金生成和分红/资本开支是否足以忽略。

### FCFF DCF

```text
FCFF = EBIT × (1 - 经营税率) + 折旧摊销 - 资本开支 - 营运资本增加
WACC = E/(D+E) × Ke + D/(D+E) × Kd × (1-T)
终值（永续增长） = FCFF_(n+1) / (WACC - g)
EV = 显性期 FCFF 现值 + 终值现值
普通股权价值 = EV - 净债务 - 少数股东/优先权 + 非经营资产
```

把经营租赁、受限现金、养老金、关联财务资产和交叉持股按重要性调整。`g` 必须低于 WACC，并与长期名义经济/行业容量一致。终值占比过高时降低结论置信度并显示替代退出倍数结果。

### FCFE/DDM

```text
股权价值 = Σ FCFE_t / (1+Ke)^t + 终值/(1+Ke)^n
DDM 股权价值 = Σ 股利_t / (1+Ke)^t + 终值/(1+Ke)^n
Gordon 终值 = 下一期可持续股利 / (Ke - g)
```

只在股利/FCFE与可分配能力、监管资本和增长再投资一致时采用。高股息率本身不是可持续性的证据。

### 剩余收益

```text
股权价值 = 当前归母账面价值
           + Σ [(ROE_t - Ke) × 期初归母账面价值_t] / (1+Ke)^t
           + 持续期剩余收益现值
```

适用于账面价值有经济含义且资本约束重要的金融公司。先调整重大不良、减值不足、商誉或不可分配资本。

### SOTP

```text
整体 EV/股权价值 = Σ 各分部独立价值
                    - 总部成本现值
                    - 净债务/少数股东/优先权
                    + 非经营资产
目标价 = 普通股权价值 / 稀释后股数
```

每个分部选择纯业务同行和适用指标；处理分部间交易、未分配成本、税、控股折价和双重计算。

### 风险调整 NPV

对药物、矿权、牌照审批等离散项目，按阶段成功概率调整各状态现金流，再折现。不要同时在现金流中乘失败概率、又用包含同一特定失败风险的夸张折现率，避免双重计风险。

## 3. 三情景构建

### 3.1 先定义情景故事

三情景不是把 Excel 每个参数同时上调/下调。每个情景先写一段可发生的世界状态，再让变量联动：

- **中性**：最可辩护路径，不是乐观和悲观的机械平均。
- **悲观**：主要风险真实发生后的经营、融资、稀释和估值压缩；不是故意设成零价值。
- **乐观**：需求、执行或产品超预期但仍受产能、竞争、监管和资本约束；不是无限外推。

优先选择 2–4 个高解释力驱动：销量/渗透率、价格/产品结构、单位成本/利润率、资本开支/营运资金、ROE/信用成本、折现率/目标倍数。对相关变量使用一致方向，例如需求恶化通常同时影响销量、价格、利用率、库存和倍数。

### 3.2 强制假设表

每个情景至少披露：

| 类别 | 必填内容 |
|---|---|
| 故事与触发 | 什么条件构成该情景，何时可观察 |
| 经营期 | 收入/核心 KPI、利润率或 ROE、盈利/现金流 |
| 资本结构 | 净债务或监管资本、分红、资本开支、潜在融资与稀释 |
| 估值 | 模型、目标指标、倍数/Ke/WACC/g/分部假设 |
| 目标 | 目标日期始终必填；通过最低证据门时填目标价、相对基准价涨跌、预期股息与总回报，否则填估值公式、条件与缺口 |
| 证伪 | 哪个可观察阈值否定该情景 |

用相同目标日期和币种比较三情景。若不同情景采用不同方法，解释原因并展示可比桥接。

### 3.3 场景宽度

场景范围由业务风险决定，不预设固定 ±10%。依据历史波动、产能与成本弹性、行业周期、管理层指引区间、可比事件和资产负债表约束设定。

范围过窄会制造确定性；范围极宽但没有叙事只是免责声明。风险越离散、融资越脆弱或终值占比越高，越应降低置信度并扩大合理区间。

## 4. 目标价、股息和概率

先通过 `<skill-dir>/SKILL.md` 的最低证据门再发布任何数值目标价。未通过时仍保留三情景叙事和联动经营假设，但将目标价写为“暂不提供”，并给估值公式、触发条件和待补输入。发布数值目标价时，每个情景至少给出带期间、单位和来源/假设标记的一项收入或核心 KPI、一项利润率或 ROE、一项现金流/资本结构/稀释项，以及一项估值参数；不得仅改变倍数或折现率制造三情景。

### 基准和回报

```text
价格涨跌幅 = 目标价 / 基准价 - 1
目标期总回报 = (目标价 + 目标期每股现金股息 - 基准价) / 基准价
年化总回报 = (1 + 目标期总回报)^(365/目标天数) - 1
```

目标价使用未复权每股价格；不要把复权历史价与当前未复权价直接比较。现金股息只在除权/支付假设明确时计入总回报。

量价技术信号只用于描述市场确认、风险和监测条件，不得直接调整盈利预测、目标倍数、WACC/Ke、永续增长、目标价或无依据的情景概率。若基本面价值与价格趋势冲突，分列两种结论并降低相关置信度，不机械让一方覆盖另一方。

### 概率

概率不是必填。仅在情景互斥、近似穷尽且有历史频率、事件树或可解释判断时使用：

```text
概率加权目标价 = Σ(情景概率 × 情景目标价)
```

三情景若只是少数代表性叙事而非完整结果空间，概率不得加总为 100%，也不得计算“期望目标价”。即使可加权，概率加权值也不自动完成风险调整；说明折现率和情景概率如何处理风险，防止双重计算。

### 表达精度

目标价最多保留两位小数，通常按 0.1 元展示更符合估计精度。始终同时给出敏感性区间和置信度，避免把 12.37 元说成比 12.4 元更可靠。

## 5. 敏感性与反向估值

### 敏感性

选择对价值影响最大的两个独立变量，展示矩阵或范围：

- DCF：WACC/Ke × 永续增长或退出倍数。
- P/E：目标 EPS × 目标 P/E。
- 银行：目标 ROE/信用成本 × 目标 P/B 或 Ke。
- 周期：商品价格/单位成本 × 中周期倍数。
- 创新药：成功概率 × 峰值销售/利润率。

给出每个变量变化 1 个合理单位对目标价的影响，标出非线性和阈值。

### 反向估值

从当前 EV/市值反推模型中的单一关键变量：

- 当前价格隐含的长期收入增长与利润率。
- 当前 P/B 隐含的可持续 ROE 相对 Ke。
- 当前 EV 隐含的中周期商品价格或 EBITDA。
- 当前管线估值隐含的成功概率/峰值销售。

不要同时反推太多自由变量。将隐含值与公司历史、产能、行业容量、同行和政策约束比较。

## 6. 强制校验门

发布前逐项通过：

- [ ] 模型适用于行业；负 EPS 不用 P/E，银行/保险不用常规 EV 倍数。
- [ ] 预测指标和倍数属于同一期间，同行估值日在可比窗口内。
- [ ] 货币单位、总额/每股、归母/全体股东、合并范围一致。
- [ ] EV 到普通股权价值桥包含净债务、现金、少数股东、优先权和非经营资产。
- [ ] 使用目标期稀释股数，考虑转债、增发、期权、限制性股票和回购。
- [ ] `WACC/Ke > g`；折现名义/实际与现金流名义/实际一致。
- [ ] 通过最低证据门时，悲观目标价 ≤ 中性目标价 ≤ 乐观目标价；否则查明非线性或输入错误。
- [ ] 场景变化与业务叙事一致，不把所有参数无依据同向极化。
- [ ] 概率若存在，全部有依据并合计 100%；否则不提供加权值。
- [ ] 价格涨跌和含股息总回报分列，基准价格有时间戳。
- [ ] 至少一种交叉估值或明确“不适用”的理由。
- [ ] 最敏感假设、数据缺口和目标价置信度显著披露。
- [ ] 技术信号若被引用，仅作为独立市场确认/风险指标，没有直接改写估值输入或概率。

## 7. 计算脚本输入

`<skill-dir>/scripts/scenario_valuation.py` 接受 JSON。所有时间、单位和概率门必须显式填写；脚本拒绝自由文本币种、无时区价格、单位不明的总额和未经声明穷尽的概率加权。

### 根字段

| 字段 | 类型/取值 | 说明 |
|---|---|---|
| `company` | 非空文本 | 公司全称 |
| `ticker` | `dddddd.SH/SZ/BJ` | 已由官方信息确认的 A 股代码 |
| `as_of_date` | `YYYY-MM-DD` | 分析基准日；必须等于 `analysis_cutoff` 的本地日历日期 |
| `analysis_cutoff` | 带 UTC offset 的 ISO 8601 时间 | 例如 `2026-07-17T15:30:00+08:00` |
| `price_as_of` | 带 UTC offset 的 ISO 8601 时间 | 必须不晚于分析截止 |
| `current_price` | 正数 | 人民币/股 |
| `current_price_source` | 非空文本 | 行情页面、终端或接口及口径 |
| `price_basis` | `unadjusted` | 目标价比较只接受未复权基准价 |
| `target_date` | `YYYY-MM-DD` | 必须晚于 `as_of_date` |
| `currency` | `CNY` 或 `RMB` | 输入可用二者，脚本统一输出 `CNY` |
| `scenarios_exhaustive` | 布尔值 | 三情景是否互斥且近似穷尽 |
| `probability_basis` | 条件必填文本 | 只有提供概率时必填 |
| `scenarios` | 对象 | 必须且只包含 `bear/base/bull` |

每个情景先统一必填 `method`、`metric_period` 和 `bridge_as_of=YYYY-MM-DD`，再按四种方法填写：

| `method` | 额外必填字段 | 计算/口径约束 |
|---|---|---|
| `per_share_multiple` | `metric_name`, `metric_unit=CNY_per_share`, `metric_per_share`, `multiple`, `multiple_basis` | `metric_per_share × multiple`；指标、倍数与目标期必须匹配 |
| `enterprise_multiple` | `metric_name`, `metric_total`, `multiple`, `multiple_basis`, `total_value_unit`, `debt`, `cash`, `minority_interest`, `preferred_equity`, `non_operating_assets`, `diluted_shares`, `shares_unit`, `shares_period` | 先算 EV，再完成 EV 到普通股权价值桥并除以稀释股数；所有金额总额共用一个 `total_value_unit` |
| `equity_value` | `equity_value`, `total_value_unit`, `diluted_shares`, `shares_unit`, `shares_period`, `method_note` | 将已复核的整体普通股权价值转为每股价值；总额与股数单位必须成对 |
| `target_price` | `target_price`, `target_price_unit=CNY_per_share`, `method_note`, `model_type`, `model_reference`, `independent_check` | 只用于已在外部展示并独立复核计算链的复杂模型；脚本仅校验审计字段，不重算外部模型，不得用来绕过输入与证据门 |

总额/股数单位只接受成对的 `(CNY, shares)` 或 `(CNY_million, million_shares)`；脚本先转换到元和股，再计算人民币/股。禁止把不兼容单位拼接。

`target_price` 的 `model_type` 只接受 `dcf`、`sotp`、`residual_income`、`risk_adjusted_npv` 或 `other`；`model_reference` 指向报告附表、工作簿及工作表或其他可定位底稿，`independent_check` 写明第二次复算了什么。只有一句“已经复核”不构成充分审计说明。输出必须继续标记为“外部模型目标价”，不得暗示脚本已经重算。

每个情景可选填 `dividend_per_share`，但同时必须填写 `dividend_unit=CNY_per_share` 和 `dividend_period`。若填写 `probability`，三个情景必须全部填写、合计 1，且根字段必须声明 `scenarios_exhaustive=true` 并给出 `probability_basis`；否则脚本拒绝加权。

以下是虚构的架构示例；日期、价格和预测期间不代表当前市场数据，运行时必须替换并核验：

```json
{
  "company": "示例股份",
  "ticker": "600000.SH",
  "as_of_date": "2026-07-17",
  "analysis_cutoff": "2026-07-17T15:30:00+08:00",
  "price_as_of": "2026-07-17T15:00:00+08:00",
  "current_price": 10.0,
  "current_price_source": "交易所2026-07-17收盘价",
  "price_basis": "unadjusted",
  "target_date": "2027-07-17",
  "currency": "CNY",
  "scenarios_exhaustive": false,
  "scenarios": {
    "bear": {
      "method": "per_share_multiple",
      "metric_name": "稀释EPS",
      "metric_period": "FY2027E",
      "bridge_as_of": "2026-07-17",
      "metric_unit": "CNY_per_share",
      "metric_per_share": 0.60,
      "multiple": 10.0,
      "multiple_basis": "可比公司同期间中位数及风险折价",
      "dividend_per_share": 0.20,
      "dividend_unit": "CNY_per_share",
      "dividend_period": "目标期内"
    },
    "base": {
      "method": "enterprise_multiple",
      "metric_name": "EBITDA",
      "metric_period": "FY2027E",
      "metric_total": 1200,
      "multiple": 8.0,
      "multiple_basis": "同业务可比公司中位数",
      "bridge_as_of": "2026-07-17",
      "total_value_unit": "CNY_million",
      "debt": 2400,
      "cash": 800,
      "minority_interest": 100,
      "preferred_equity": 0,
      "non_operating_assets": 200,
      "diluted_shares": 720,
      "shares_unit": "million_shares",
      "shares_period": "FY2027E",
      "dividend_per_share": 0.25,
      "dividend_unit": "CNY_per_share",
      "dividend_period": "目标期内"
    },
    "bull": {
      "method": "equity_value",
      "metric_period": "2027-07-17 DCF",
      "bridge_as_of": "2026-07-17",
      "equity_value": 12000,
      "total_value_unit": "CNY_million",
      "diluted_shares": 700,
      "shares_unit": "million_shares",
      "shares_period": "FY2027E",
      "method_note": "五年FCFF DCF，计算链在研究底稿",
      "dividend_per_share": 0.30,
      "dividend_unit": "CNY_per_share",
      "dividend_period": "目标期内"
    }
  }
}
```

`target_price` 情景片段：

```json
{
  "method": "target_price",
  "metric_period": "2027-07-17 SOTP",
  "bridge_as_of": "2026-07-17",
  "target_price": 12.5,
  "target_price_unit": "CNY_per_share",
  "method_note": "分部价值、净债务和股数桥已在底稿复核",
  "model_type": "sotp",
  "model_reference": "valuation-workbook.xlsx#SOTP",
  "independent_check": "逐项复算分部价值、净债务桥和稀释股数"
}
```

运行：

```bash
python3 <skill-dir>/scripts/scenario_valuation.py inputs.json --format markdown
```

把脚本生成的目标价与独立手算或电子表格抽查一遍。假设的合理性始终由研究者负责。
