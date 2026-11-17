# Quant 策略工作台

Tushare-first 的量化研究与每日策略工作台，覆盖短线、缠论、长线、可转债、配债股、市场过渡和相似走势，并提供独立的股指战法研究页、操作计划和复盘记录。

> 本项目用于策略研究和数据分析，不构成投资建议。数据、模型、回测与页面信号都需要结合交易所公告和真实账户约束复核。

## 能力概览

- 增量刷新 Tushare 日线数据；MySQL 使用统一 `market_daily` 表，Parquet 使用年月分区镜像。
- 每日同步 `daily_basic`、`stock_basic`、沪深300、最近报告期财务数据，以及涨跌停、停牌、ST 的当日可交易快照。
- 复用正式日线生成每日 `risk_on/neutral/risk_off` 市场状态；无需接入第二套行情源。
- 构建 B1/B2/B3 与扩展策略特征、模型评分和每日候选池。
- 提供组合构建、防泄漏 Walk-Forward、因子诊断、绩效归因、风险容量、标准报告和可持久化模拟交易能力。
- 在 FastAPI + 静态前端工作台中查看策略结果、模型组合候选和 ETF 资金背景。
- 股指战法使用冻结研究历史与独立更新的开仓预案；操作计划和复盘笔记由用户维护。
- 按日期保存选股器和工作区快照，支持历史复盘。
- 每日流水线采用有依赖的并行编排，单个非核心工作区失败不会阻断其他工作区。
- 以模块化单体方式维护应用用例、存储适配和 HTTP 交付边界，避免工作区继续堆叠在 Web 服务层。
- JS/CSS 静态资源支持 gzip、模块预加载和显式缓存策略，桌面端与 390px 移动端均纳入真实浏览器验收。

## 前置条件

- 项目声明 Python 3.9 或更高版本；完整应用和新增研究脚本建议使用 Python 3.13。CI 的版本覆盖范围见 [开发说明](docs/development.md)。
- 正式刷新行情时需要有效的 `TUSHARE_TOKEN`。
- MySQL 为生产推荐项；本地开发可以仅使用 Parquet。

下列命令以 macOS/Linux Shell 为例；Windows 用户需要使用对应的虚拟环境激活和环境变量语法。

## 快速开始

在仓库根目录执行：

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]" && (test -f .env || cp .env.example .env)
PYTHONPATH=src python scripts/run_webapp.py
```

打开 [http://127.0.0.1:8088](http://127.0.0.1:8088)。API 健康检查位于 [http://127.0.0.1:8088/api/health](http://127.0.0.1:8088/api/health)，交互式 API 文档位于 [http://127.0.0.1:8088/docs](http://127.0.0.1:8088/docs)。

Git 不包含正式行情、模型权重与页面快照。首次启动不会自动生成历史研究成果；需要恢复对应产物并核对版本后才能获得完整策略输出，详见 [开发与验证](docs/development.md)。

首次进行真实数据刷新前，编辑 `.env` 并至少填写：

```dotenv
TUSHARE_TOKEN=your_token
```

## 每日任务

生产环境的每日更新统一走 Web 编排入口：

```bash
python scripts/run_daily_web_refresh.py
# 等价入口
PYTHONPATH=src python -m quant.routine.cli web-refresh
```

兼容命令 `daily --refresh-data` 现在也会委托给同一套 Web 编排；只有显式增加
`--direct-pipeline` 才会运行旧的进程内流水线，供维护和诊断使用。

执行顺序为：

1. 获取跨进程单实例锁，并用交易日历确认当天是否执行。
2. 复用健康的 Web 服务，通过 API 启动与页面“更新全部”一致的任务。
3. 按活动依赖闭包清理缓存并刷新共享 Tushare 行情、当前策略实际需要的 `daily_basic` 与参考数据；`tradability` 只在研究/显式回填任务中刷新。
4. 使用截至当日的沪深300、全市场宽度、波动率和成交额生成市场状态日期快照与 `latest.json`。
5. 先增量构建便宜的规则信号和 B1/Z 候选并集，再只为命中股票计算完整重因子，避免全市场重复计算和两套进程池争抢资源。
6. 共享数据封存后按依赖和统一资源配额启动下游任务；相似走势向量分析可提前执行，策略快照齐备后再补齐同日买入/持有评分和只读策略标记。排名、模型组合与页面产物按所选作用域执行。
7. 原子发布正式文件，并将步骤状态、起止时间、耗时和结果写入 `data/routine/<运行时间>_<run_id>/manifest.json`。

「更新全部」和短线页的「更新本页」会发布同日宽基 ETF 资金快照。快照使用 Tushare 日度数据和固定的 15 只 ETF，显示 5/20 交易日估算净申购、占期初规模比例、放量倍数和覆盖率。成交指标滞后一交易日，资金指标至少滞后两交易日并遵守净值公告日。接口随 `/api/selector/stocks` 返回 `etf_flow`，历史日期按当日快照读取。ETF 采集异常会记录到刷新结果，候选评分可继续完成。

左右侧正式排序共用同一日的 ETF 特征快照。模型发布文件为 `configs/strategies/short_etf_rank.json`，每侧对原排序分的修正上限为 4 个百分点。评分产物保留原分 `ranking_score_base`，并用 `ranking_score` 排序。ETF 数据缺失或基础模型版本不匹配时沿用原分。只有 ETF 输入或排序层变化时会重评排序并复用有效的个股特征缓存。排序层单独通过 `scripts/research/train_short_etf_rank.py` 训练，日常刷新直接读取发布文件。

网页「更新全部」和定时入口共用全量任务。任务先发布 7 个股票策略工作区，再于北京时间 18:00 后更新股指开仓预案。股指页也可手动刷新。股指历史研究、操作计划和复盘不由日更覆盖。项目提供每日任务入口，但不内置常驻调度器；生产环境需要由 cron、launchd 或其他调度平台每天调用上述统一入口。详细操作和故障处理见 [每日运行与故障排查](docs/operations.md)。

当前左侧使用 RankNet 融合，短线默认模型应用组合由 `configs/strategies/application_portfolio.json` 管理；发布/回滚和研究入口见 [研究工作流索引](docs/research_workflows.md)。排序分、校准概率与资金队列是不同口径，页面候选不代表已下单或真实账户持仓。

## 配置

复制 [`.env.example`](.env.example) 后按环境填写；`.env` 已被 Git 忽略。

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `TUSHARE_TOKEN` | 无 | 正式拉取 Tushare 数据时必填 |
| `MARKET_DATA_BACKEND` | `mysql` | `mysql`/`sql` 使用 SQL 主存储，其他值使用 Parquet |
| `MARKET_DATA_SQL_URL` | 无 | SQLAlchemy MySQL 连接串；未配置时可回落到 Parquet 镜像 |
| `MARKET_DATA_ROOT` | `data/raw` | Parquet 数据根目录 |
| `MARKET_DATA_MIRROR_PARQUET` | `1` | SQL 写入时是否同时更新 `data/raw/daily_partitioned/year_month=YYYYMM/data.parquet` |
| `MARKET_DATA_SQL_BATCH_SIZE` | `5000` | 全市场 MySQL 批量 upsert 每批行数 |
| `MARKET_DATA_SQL_CONNECT_TIMEOUT` | `10` | MySQL 连接超时，单位秒 |
| `MARKET_DATA_SQL_READ_TIMEOUT` | `60` | MySQL 读取超时，单位秒 |
| `MARKET_DATA_SQL_WRITE_TIMEOUT` | `60` | MySQL 写入超时，单位秒 |
| `ROUTINE_DAILY_WORKERS` | `4` | 日线刷新并发数 |
| `ROUTINE_DAILY_SLEEP` | `0.08` | Tushare 请求间隔，单位秒 |
| `ROUTINE_DAILY_AVAILABILITY_RETRY_FAILURES` | `12` | 历史日线或非缺失错误的失败预算；当日缺失采用 17:20 截止规则 |
| `ROUTINE_DAILY_AVAILABILITY_RETRY_INTERVAL` | `600` | 历史或非缺失错误的重试间隔，单位秒；当日缺失每 600 秒重试 |
| `ROUTINE_DAILY_BATCH_MIN_COVERAGE_RATE` | `0.995` | 每个交易日全市场批量响应的最低股票覆盖率；低于阈值拒绝发布 |
| `ROUTINE_DAILY_BASIC_WORKERS` | `4` | `daily_basic` 按交易日刷新的并发数 |
| `ROUTINE_DAILY_BASIC_SLEEP` | `0.25` | `daily_basic` 请求最小间隔，单位秒 |
| `ROUTINE_DAILY_BASIC_MIN_COVERAGE_RATE` | `0.98` | `daily_basic` 相对当日正式行情股票数的最低覆盖率 |
| `ROUTINE_TRADABILITY_MIN_COVERAGE_RATE` | `0.98` | `stk_limit` 相对当日有效证券范围的最低覆盖率；低于阈值阻断下游 |
| `ROUTINE_TRADABILITY_RETRIES` | `3` | 涨跌停、停牌和 ST 接口的单日重试次数 |
| `ROUTINE_MARKET_REGIME_LOOKBACK_DAYS` | `252` | 每日市场状态读取正式行情的自然日回看窗口，最小 90 |
| `ROUTINE_FEATURE_WORKERS` | `8` | 特征构建并发数 |
| `ROUTINE_TOTAL_WORKERS` | CPU 核数 | DAG 统一 CPU 配额；各任务及嵌套并行共同受此约束 |
| `ROUTINE_IO_SLOTS` | `4` | DAG 统一 I/O 配额；新左侧分区读取最多占 2 槽，配置为 1 时串行 |
| `ROUTINE_MEMORY_BUDGET_MB` | `8192` | 调度内存预算；新左侧声明 6144 MB，预算不足会阻断任务 |
| `ROUTINE_DB_CONNECTIONS` | `8` | DAG 数据库连接配额 |
| `ROUTINE_MODEL_SCORE_WORKERS` | `4` | Web 每日更新中策略模型评分的 worker 数；与缠论评分并行时上限为 4 |
| `ROUTINE_RIGHT_SIDE_WORKERS` | `6` | 右侧统一因子 worker 数；等待模型评分完成后与缠论 4 workers 并行，总上限为 10 |
| `ROUTINE_FEATURE_EXECUTOR` | `processes` | 特征计算执行器；CPU 密集计算默认使用多进程 |
| `ROUTINE_DAILY_BASIC_MIN_MATCH_RATE` | `0.98` | B1 增量特征与 `daily_basic` 的最低匹配率；低于阈值阻断发布 |
| `B1_FEATURE_MAX_SYMBOL_ERROR_RATE` | `0.001` | B1 特征构建单股异常率上限；超过即阻断发布 |
| `ROUTINE_CHAN_WORKERS` | `4` | 缠论增量候选扫描并发数；Web 每日更新与模型评分并行时上限为 4 |
| `ROUTINE_WEB_WORKSPACE_WORKERS` | `6` | 六个下游工作区的最大并发数 |
| `SIMILAR_PATTERN_CACHE_WORKERS` | `4` | 相似走势向量缓存并发数 |
| `SIMILAR_PATTERN_FORCE_VECTOR_CACHE` | 空 | 设为 `1` 时强制重建相似走势全市场参考库；日常无需设置 |

完整的重试和增量参数见 [每日运行与故障排查](docs/operations.md#资源与限流配置)。

## 常用命令

```bash
# 仅验证每日流水线编排，不访问外部行情源
PYTHONPATH=src python -m quant.routine.cli daily --skip-backtest

# 单独生成短线每日计划或 dashboard
PYTHONPATH=src python -m quant.routine.cli plan
PYTHONPATH=src python -m quant.routine.cli dashboard

# 首次启用时按正式日线中的交易日回填历史可交易快照；已有日期默认跳过
PYTHONPATH=src python -m quant.routine.tradability_refresh \
  --start 20200101 --end 20260731

# 本地事件驱动回测
PYTHONPATH=src python main.py backtest --strategy momentum --symbol 600000 \
  --start-date 20200101 --end-date 20231231

# 测试与静态检查；隔离数据根目录的完整命令见 docs/development.md
PYTHONPATH=src pytest -q
node --test tests/frontend/*.test.js
PYTHONPATH=src python -m compileall -q src tests scripts
python -m ruff check --select E9,F63,F7,F82 src tests scripts
```

## 配债股自选池复核任务

配债股出入池使用独立的两阶段脚本执行。默认命令只生成待确认名单，不修改自选池：

```bash
python scripts/review_convertible_bond_watchlist.py plan
```

脚本要求 `data/routine/convertible_bond_allotments_latest.json` 是当日全部事件源轮询成功的快照。如快照未更新，可在生成名单时显式刷新配债事件源；该操作调用每日更新和页面使用的同一个 `get_convertible_bond_allotments` 入口，统一执行公告轮询、质量门禁和快照落盘。随后仅从本次缓存生成包含已过登记日记录的复核全集，不会再次独立联网刷新：

```bash
python scripts/review_convertible_bond_watchlist.py plan --refresh
```

生成计划前会逐股比较每日页面结果和复核全集中的阶段、公告标题、公告日期和公告链接；两边任一股票缺失或公告证据不一致都会中止，不产出可执行计划。计划会写入 `data/routine/convertible_bond_watchlist_review_latest.json`，并输出待出池、待入池、待补配债标记和需人工复核四类名单。只有确认名单后，才使用计划输出的完整 `plan_id` 执行：

```bash
PYTHONPATH=src python scripts/review_convertible_bond_watchlist.py apply \
  --plan data/routine/convertible_bond_watchlist_review_latest.json \
  --confirm <plan_id>
```

`apply` 会拒绝错误确认码、非当日计划，以及生成计划后又被其他操作改动过的自选池。执行前会在同目录生成 `watchlist.before_cb_review_<时间>_<plan_id>.json` 备份，再原子更新正式自选池。入池条件是“上市委通过”或“同意注册”；股权登记日早于复核日才列为待出池。数据源缺失或阶段异常回退只会进入人工复核，不会自动出池。

每次确认执行还会同步所有有效配债股的任务信息，而不只处理新增股票：备注区块统一刷新阶段、配售参数、股权登记日、最新价格及日期、日/周/月 KDJ、公告标题和链接。原有人工备注及其他提醒保持不变；每只有效配债股都会确保存在一个可独立触发的 `日线J < 5` 提醒，已有相同单条件提醒时不重复添加。出池股票随自选池记录一并清理备注和提醒。

名单原子更新后，脚本会立即轻量刷新全部自选股的买入分、持有分、评分日期和特征质量，并更新 `data/research/similar_patterns/web_watchlist_analysis.json`；该步骤不重跑耗时的相似走势案例分析。评分必须覆盖全部自选股，否则脚本恢复执行前备份并报告失败，避免出现名单已加入但页面评分为空的半完成状态。

### 配债公告回归测试集

公告适配样本集中维护在 `tests/fixtures/convertible_bond_announcement_cases.json`，分为两类：

- `title_cases`：单条公告是否属于公开发行配债流程，以及对应阶段；
- `timeline_cases`：同一股票连续公告后的最终阶段，防止普通后续公告覆盖已经确认的更高阶段。

遇到新的公告措辞、漏报或误报时，先把真实标题加入该文件，再调整分类规则。定向可转债、并购支付对价型可转债，以及转股、付息、赎回等存量债券公告必须保留反例。执行：

```bash
PYTHONPATH=src python -m pytest -q tests/test_convertible_bond_announcement_cases.py
```

该测试文件由常规 `pytest` 自动发现，无需单独接入每日任务。

每周六 09:00 的 Codex 任务会刷新并核对公告证据、页面配债池、复核结果和自选池名单。若发现漏报、误报或两条结果不一致，任务应先将真实公告加入上述测试集，再修复共享分类代码，运行配债相关测试和完整测试套件；测试全部通过后重新生成待执行名单。自动任务不得执行 `apply`，自选池仍需人工确认后修改。

## 回测结果契约

`BacktestEngine.run()` 继续返回 akquant 原始结果，兼容已有策略和报告生成；新的应用、研究脚本与组合层应优先读取 `engine.artifacts`。`BacktestArtifacts` 固定提供：

- 日频净值与收益序列；
- 持仓、订单、成交和已平仓交易表；
- 佣金、印花税、过户费、滑点成本和总成本台账；
- 可选的基准收益与初始资金元数据。

`engine.get_metrics()` 从稳定契约计算统一指标。`period_count` 和 `positive_period_rate` 描述收益期数；`trade_count` 只统计交易表中的真实交易笔数，两种口径不能混用。底层引擎升级时，应只调整 `quant.backtest.artifacts` 的适配逻辑，不让工作区直接依赖 akquant 的具体字段。

`write_backtest_report()` 可将上述契约一次写成净值、收益、持仓、订单、成交、交易、成本、指标 JSON、自包含 HTML 和 `research_manifest.json`。研究清单记录数据/代码 SHA-256、参数、样本期、随机种子、Git 状态与依赖版本。

## A 股执行口径

CLI 本地回测默认使用项目自有 `AShareExecutionConfig`：佣金率 `0.03%`、卖出印花税 `0.05%`、双向过户费 `0.001%`、最低佣金 5 元、100 股整手、T+1，以及单根行情最多成交 10% 成交量。佣金和最低佣金是研究假设，可按真实账户覆盖；完整配置会写入 `engine.artifacts.metadata["execution_policy"]`，便于复现。

执行规则由本项目定义和校验，akquant 仅作为当前底层成交适配器。每日流程用 Tushare `stk_limit`、`suspend_d`、`stock_st` 和 `stock_basic` 生成 `data/raw/tradability/YYYYMMDD.parquet`；`AShareTradabilityPolicy` 在项目订单门禁和模拟券商中拒绝缺失数据、停牌、涨停买入、跌停卖出及上市前订单。由于 akquant 当前没有稳定的自定义撮合返回协议，原始 akquant 撮合仍只负责兼容回测，不能被当作这套门禁已经自动执行。

`SimulatedBroker` 复用相同执行配置和门禁，执行最低佣金、双向过户费、卖出印花税、整手、T+1 与成交量参与率，并用注入的最新价估值；持仓、订单、成交和当日可卖数量可以原子持久化，日终通过 `reconcile_account()` 对账。

数据接入遵循“项目能力优先、Tushare-first、显式降级”：业务层先读 `MarketDataStore` 的统一正式数据；缺失时由现有 `TushareDataFetcher` 和刷新流程补齐。仅当 Tushare 确实不提供所需字段时，才允许新增其他来源的独立适配器，并记录来源、字段口径、更新时间和降级原因，禁止静默拼接不同来源的同名字段。

## 项目结构

```text
src/quant/core/            项目路径与无副作用公共基础类型
src/quant/application/     API、CLI、调度共用的应用用例与刷新契约
src/quant/infrastructure/  文件与 SQL 快照等外部存储适配器
src/quant/data/            数据源、标准化和统一存储
src/quant/features/        项目级变量与特征
src/quant/strategies/      可复用策略实现
src/quant/routine/         每日任务和产物编排
src/quant/webapp/          FastAPI 路由与工作区服务
web/                       单页工作台前端
web/core/                  前端 API 客户端与格式化基础模块
scripts/research/          研究、训练、回测和校准脚本
configs/strategies/        已确认的生产策略配置
configs/releases/          发布候选与回滚配置
configs/research/          研究参数与候选方案
docs/strategies/           策略依据、口径和迭代记录
tests/                     单元、集成和 API 测试
```

`data/`、`models/`、`reports/` 和 `web/data/` 是本地运行产物，不提交到 Git。

应用依赖方向为 `interfaces -> application -> domain/data`，文件和 SQL 等技术细节由
`infrastructure` 适配。BYD、可转债和配债股已形成可独立测试的垂直工作区，Web 服务保留兼容
函数作为组合入口。详细边界和后续迁移规则见[系统架构与数据流](docs/architecture.md)。

## 文档

- [文档中心](docs/README.md)
- [开发、验证与提交](docs/development.md)
- [研究与生产入口索引](docs/research_workflows.md)
- [系统架构与数据流](docs/architecture.md)
- [API 参考](docs/api.md)
- [每日运行与故障排查](docs/operations.md)
- [每日依赖注册表](docs/daily_dependency_registry.md)
- [项目结构与 MySQL 存储](docs/project_structure_and_storage.md)
- [正式策略档案](docs/strategies/b1_selected_strategy_record.md)

## 开发约定

- 新策略进入例行任务前，必须在 `docs/strategies/` 记录数据口径、入场/离场规则、风险和回测结果。
- 新功能必须补充对应测试；涉及路由或工作区缓存时同时更新 API 或运维文档。
- 禁止提交 `.env`、Token、数据库密码、原始数据、模型和报告产物。
- 清理研究脚本前先确认已有正式策略档案或可复现的替代实现。
