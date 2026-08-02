# 每日运行与故障排查

本文面向日常维护者，说明如何执行、调度、验证和恢复每日刷新。

## 运行前检查

从仓库根目录执行：

```bash
source .venv/bin/activate
python -c "import pandas, tushare, fastapi; print('dependencies ok')"
PYTHONPATH=src python -c "import os; from quant.routine.paths import PROJECT_ROOT; print(PROJECT_ROOT, bool(os.getenv('TUSHARE_TOKEN')))"
```

期望看到项目根目录和 `True`。`quant.routine.paths` 会自动加载仓库根目录的 `.env`，且不会覆盖已经导出的环境变量。

服务健康检查：

```bash
curl --fail http://127.0.0.1:8088/api/health
```

期望响应：

```json
{"status":"ok","service":"quant-webapp"}
```

## 前端静态交付验证

HTML 每次重新验证，JS/CSS 缓存一小时并通过版本参数更新。服务对大于 1 KiB 的响应启用 gzip。
部署或重启后使用真实 GET 请求检查响应头：

```bash
curl --compressed -D - -o /dev/null http://127.0.0.1:8088/app.js
curl --compressed -D - -o /dev/null http://127.0.0.1:8088/styles.css
```

两项响应都应包含：

```text
content-encoding: gzip
cache-control: public, max-age=3600
vary: Accept-Encoding
```

如需比较传输量，分别请求 identity 与 gzip；不要用 `HEAD` 判断 gzip，因为无响应体时中间件可以不压缩：

```bash
curl -H 'Accept-Encoding: identity' -o /dev/null \
  -w 'bytes=%{size_download} time=%{time_total}\n' http://127.0.0.1:8088/app.js
curl --compressed -o /dev/null \
  -w 'bytes=%{size_download} time=%{time_total}\n' http://127.0.0.1:8088/app.js
```

## Web 服务常驻运行（macOS）

不要用普通终端后台进程长期托管 Web 服务；会话退出、开发工具回收子进程或机器重启后，进程不会自动恢复。项目提供 `launchd` 用户服务，登录后自动启动，退出时自动拉起：

```bash
scripts/webapp_service.sh install
scripts/webapp_service.sh status
```

日常管理：

```bash
scripts/webapp_service.sh restart
scripts/webapp_service.sh logs
scripts/webapp_service.sh stop
scripts/webapp_service.sh start
```

服务标识是 `com.didi.quant.webapp`，应用日志统一写入 `.run/webapp.log`，按 5MB 轮转并保留 2 个备份（总量约 15MB）。launchd 的标准输出和错误输出定向到 `/dev/null`，避免与应用日志重复累积；`scripts/webapp_service.sh logs` 可持续查看轮转日志。安装脚本会优先选用项目 `.venv`，其次选用已包含项目依赖的 Miniforge/Conda Python；也可用 `QUANT_PYTHON=/absolute/path/python` 显式指定。

更新服务启动参数或 launchd 配置后使用 `install`，不要只执行 `restart`，因为 `install` 会重新生成并加载 plist。若完整刷新正在运行，先等待终态，再执行安装，避免中断当日任务：

```bash
curl http://127.0.0.1:8088/api/selector/refresh-latest/status
scripts/webapp_service.sh install
scripts/webapp_service.sh status
curl --fail http://127.0.0.1:8088/api/health
```

## 执行每日任务

推荐生产命令：

```bash
python scripts/run_daily_web_refresh.py
# 等价：PYTHONPATH=src python -m quant.routine.cli web-refresh
```

- 该入口与页面“更新全部”共用 Web 编排、断点续跑和状态记录。
- 兼容命令 `daily --refresh-data` 会委托到这个入口，不再维护第二套生产步骤。
- 只有维护或诊断时才使用 `daily --direct-pipeline`；它不会替代生产调度。

命令结束时输出 manifest 路径。也可以查找最近一次：

```bash
find data/routine -name manifest.json -type f -print | sort | tail -1
```

打开该 JSON，确认关键步骤 `status` 为 `success`；主动跳过的数据刷新或回测会显示 `skipped`。

参考数据步骤还必须包含：

- `tradability`：`data/raw/tradability/YYYYMMDD.parquet`，覆盖率不得低于配置门槛；
- `market_regime`：`data/features/market_regime/YYYYMMDD.json` 与 `latest.json`，`as_of` 必须等于本次正式行情日期。

首次启用或需要重建历史回测输入时，按本地正式日线中实际存在的交易日回填。该命令只使用 Tushare；上市、退市和暂停上市证券会合并后再按每个日期过滤，已有分区默认跳过：

```bash
PYTHONPATH=src python -m quant.routine.tradability_refresh \
  --start 20200101 --end 20260731
```

需要覆盖重拉时显式增加 `--force`。每次回填会写入 `data/raw/source_audit/*_tradability/manifest.json`；任一日期失败时进程返回非零。大范围回填受 Tushare 权限和限频影响，先用一个月验证权限，再逐年执行。

## B1 正式模型兼容与发布

正式回测使用 `models/production/b1/` 下的五个统一模型：`up5_es`、`up8_es`、`up10_es`、`down2_es`、`down3_es`。模型必须使用与 `data/features/b1/training_xgb_project_vars.parquet` 一致的统一特征定义；生产目录只保留当前发布，旧模型不再作为运行时回退。

发布新模型时先写入候选目录并完成三层验证：模型 test/OOT AUC、2025 校准与 2026 独立验证、正式组合全量回测。正式组合还必须通过 `reports/b1/current/model_compatibility_audit.json` 中的样本量、平均收益和 PF 门禁，状态为 `valid` 后才允许生成每日计划。`configs/strategies/b1_selected.yaml` 是发布 ID、模型目录、阈值和退出规则的唯一正式来源；正式回测和 Web 每日计划共同消费它。

从已审计特征缓存快速重训候选模型：

```bash
PYTHONPATH=src:scripts/research python scripts/research/train_b1_tushare_models.py \
  --reuse-dataset \
  --dataset-out data/features/b1/training_xgb_project_vars.parquet \
  --model-dir models/research/b1_candidate \
  --report-dir reports/b1/research/b1_candidate
```

候选验证可通过 `B1_FORMAL_MODEL_DIR` 和 `B1_FORMAL_OUTPUT_DIR` 隔离模型与报告；不要直接覆盖 `reports/b1/current`：

```bash
B1_FORMAL_MODEL_DIR=models/research/b1_candidate \
B1_FORMAL_OUTPUT_DIR=reports/b1/candidate \
PYTHONPATH=src python -m quant.research.b1_formal_combos
```

兼容脚本 `scripts/research/analyze_b1_formal_combos.py` 只转发到上述包内任务；生产编排不再执行研究脚本路径。

## 调度说明

项目提供可重复执行的每日任务，但不内置 cron/launchd 常驻调度。调度器需要满足：

1. 工作目录是仓库根目录。
2. 使用安装了项目依赖的 Python。
3. 同一时刻最多一个完整任务，避免覆盖相同日期的缓存。
4. 将标准输出和错误输出写入受轮转管理的日志。
5. 在 A 股收盘且 Tushare 日线更新后执行。

调度命令本体为：

```bash
cd /absolute/path/to/quant && .venv/bin/python scripts/run_daily_web_refresh.py
```

把 `/absolute/path/to/quant` 替换为部署机器上的仓库绝对路径。不要把 Token 或数据库密码直接写进调度配置；保存在权限受控的 `.env` 中。

## 手工刷新页面工作区

在页面点击“更新全部”，或通过 API：

```bash
curl -X POST http://127.0.0.1:8088/api/selector/refresh-latest \
  -H 'Content-Type: application/json' \
  -d '{"scope":"all"}'
```

查询进度：

```bash
curl http://127.0.0.1:8088/api/selector/refresh-latest/status
```

只补跑单个工作区时，将作用域替换为 `short`、`chan`、`long`、`cb`、`cbAllotment`、`byd` 或 `similar`。后台状态同时持久化到 `data/routine/latest_refresh_status.json`。

## 例行一键刷新脚本

日常定时任务优先调用同一个脚本，让它自己完成交易日判断、服务检查/启动、触发“更新全部”、进度打印，以及失败后的自动重试：

```bash
cd /absolute/path/to/quant && python3 scripts/run_daily_web_refresh.py
```

也可以走现有 CLI：

```bash
cd /absolute/path/to/quant && PYTHONPATH=src python3 -m quant.routine.cli web-refresh
```

脚本默认行为：

1. 读取项目 `.env`。
2. 获取 `.run/daily_web_refresh.lock` 跨进程锁；已有任务运行时立即返回 `busy`，不启动第二套写盘任务。
3. 用 Tushare `trade_cal` 判断当天是否为 A 股交易日；非交易日跳过，无法可靠确认时按失败处理。
4. 默认复用健康的本地 Web 服务；仅显式传入 `--restart-service` 时才会在刷新前重启脚本托管的服务。
5. 检查 `http://127.0.0.1:8088/api/health` 和前端首页 `http://127.0.0.1:8088/`，确认前后端已就绪。
6. 触发 `POST /api/selector/refresh-latest`，作用域默认 `all`。
7. 轮询 `/api/selector/refresh-latest/status` 并打印进度。
8. 若终态为 `failed/error`，自动再次触发刷新。服务端会优先复用已有的断点续跑能力。
9. 每次终态都会保存独立的 `data/routine/<运行时间>_<run_id>/manifest.json`，其中包含每一步的状态、起止时间、耗时、结果和错误；`latest_refresh_status.json` 只作为最新状态指针。
10. 服务端刷新开始前自动清理缓存：手工回测生成的长线研究缓存只保留最近 2 组，相似走势正式向量只保留最新一套，smoke 测试向量缓存全部删除，Tushare 单股请求缓存保留最近 7 天。Tushare `daily_basic` 请求缓存也保留最近 7 天，但只有对应正式文件存在且非空时才删除。已被合并可转债日线覆盖的逐日请求缓存会删除；B1 时间戳研究报告每类保留最近 2 版，可重建的超大 trade-samples 只保留最新 1 版，内容相同的 `latest` 文件使用硬链接去重。策略快照保留 30 天、每个业务分组最多 10 个日期；workspace 快照保留 14 天、每组最多 3 个日期；数据源审计保留 30 天且最多 10 次；routine 历史运行保留 14 天且最多 5 次。每个业务分组最新一期始终保留，对应 MySQL 快照表同步执行相同规则。
11. 相似走势的全市场历史参考库每 7 天最多重建一次；每日任务仍会直接读取自选池股票的最新日线，现场计算目标向量并完成匹配。因此自选股信号按日更新，历史样本及其后续收益标签按周更新。

清理逻辑也会在 `daily` CLI 开始前执行。需要单独维护或立即释放空间时可运行：

```bash
cd /absolute/path/to/quant && PYTHONPATH=src python3 -m quant.routine.cli cache-cleanup
```

`cache-cleanup` 会真正删除满足条件的文件和 MySQL 快照行，不是预览命令。正式行情位于 `data/raw/` 或 MySQL 行情表，不会因为普通请求缓存过期而被删除；`daily_basic` 请求缓存只有在对应正式 Parquet 已存在且非空时才允许删除。

## 缓存清理生效验证

清理规则在完整刷新开始时执行。因此，如果代码是在当天刷新完成后才部署，空间不会立刻下降；下一次完整刷新开始后才会自动清理。部署后的下一个交易日按以下顺序验证：

```bash
# 1. 确认完整刷新已结束
curl http://127.0.0.1:8088/api/selector/refresh-latest/status

# 2. 查看主要目录占用
du -sh \
  data/cache/source_merge/tushare \
  data/research/long_dividend_quality \
  data/research/similar_patterns/vector_cache

# 3. smoke 缓存应不存在；find 无输出即符合预期
find data/research/similar_patterns -maxdepth 1 \
  -type d -name 'vector_cache*smoke*' -print

# 4. 参考库元数据应记录最近一次全量重建时间和 7 天周期
find data/research/similar_patterns/vector_cache -maxdepth 3 \
  -name _refresh_metadata.json -print
```

2026-07-21 优化部署前的实测基线和预期稳态如下。数值会随股票数量、Parquet 编码和配置变化，应作为容量量级而不是严格告警阈值：

| 内容 | 部署前 | 清理后/稳态预期 | 说明 |
| --- | ---: | ---: | --- |
| Tushare 请求缓存 | 约 3.29 GiB | 约 0.22–0.25 GiB | 单股日线及有正式副本的 `daily_basic` 保留 7 天 |
| 长线研究中间缓存 | 约 13.68 GiB | 约 0.6–0.8 GiB | 只保留最近 2 组；生产刷新不再新增大缓存 |
| 相似走势正式向量 | 约 1.91 GiB | 约 1.9 GiB | 只保留一个配置版本，每 7 天最多全量更新一次 |
| 相似走势 smoke 向量 | 约 0.21 GiB | 0 | 全部删除，不再纳入生产保留范围 |
| 快照、审计和 routine 历史 | 约 0.04 GiB | 约 0.01 GiB | 同时受天数和最大版本数限制 |
| **以上合计** | **约 19.13 GiB** | **约 2.8–3.0 GiB** | 预计释放约 16 GiB |

MySQL 快照执行相同的定期删除规则，但 InnoDB 删除行后通常先释放为表内可复用空间，数据库文件未必立即缩小；这不代表清理没有生效。

常用参数：

- `--max-attempts 3`：最多触发几次刷新。
- `--poll-seconds 20`：轮询间隔。
- `--no-progress-timeout 2100`：本地监控无进展超时秒数。
- `--log-file .run/daily_web_refresh.log`：自动启动 web 服务时的日志文件。
- `--pid-file .run/daily_web_refresh.pid`：常驻后台 web 服务的 pid 文件。
- `--restart-service`：仅在明确需要重启脚本托管的 Web 服务时使用；若已有刷新任务运行，脚本会跳过重启并接管监控。
- `--frontend-url http://127.0.0.1:8088/`：前端首页健康检查地址；默认由 `--base-url` 推导。

## 资源与限流配置

| 变量 | 默认值 | 调整建议 |
| --- | --- | --- |
| `ROUTINE_DAILY_LOOKBACK_DAYS` | `0` | 数据修订时增加回看天数 |
| `ROUTINE_DAILY_WORKERS` | `4` | Tushare 限频时降低 |
| `ROUTINE_DAILY_SLEEP` | `0.08` | 遇到限频时提高到 `0.25` 或更高 |
| `ROUTINE_DAILY_FINAL_RETRY_ROUNDS` | `2` | 最终失败较多时谨慎增加 |
| `ROUTINE_DAILY_FINAL_RETRY_WORKERS` | `4` | 限频时降低 |
| `ROUTINE_DAILY_FINAL_RETRY_SLEEP` | `0.8` | 最终重试仍限频时提高 |
| `ROUTINE_DAILY_BATCH_MIN_COVERAGE_RATE` | `0.995` | 单交易日全市场行情覆盖率门禁；低于阈值阻断发布 |
| `MARKET_DATA_SQL_BATCH_SIZE` | `5000` | 统一行情表批量 upsert 行数；遇到 `max_allowed_packet` 限制时降低 |
| `ROUTINE_DAILY_BASIC_WORKERS` | `4` | `daily_basic` 按交易日拉取的并发数；限频时降低 |
| `ROUTINE_DAILY_BASIC_SLEEP` | `0.25` | `daily_basic` 请求最小间隔；限频时提高 |
| `ROUTINE_DAILY_BASIC_RETRIES` | `3` | `daily_basic` 单日失败重试次数 |
| `ROUTINE_DAILY_BASIC_MIN_COVERAGE_RATE` | `0.98` | `daily_basic` 相对当日正式行情股票数的最低覆盖率 |
| `ROUTINE_TRADABILITY_MIN_COVERAGE_RATE` | `0.98` | `stk_limit` 对当日有效证券范围的覆盖率门禁；低于门槛不发布 |
| `ROUTINE_TRADABILITY_RETRIES` | `3` | 可交易性三个 Tushare 接口的单日重试次数 |
| `ROUTINE_TRADABILITY_RETRY_SLEEP` | `0.5` | 单接口线性退避的基础秒数 |
| `ROUTINE_TRADABILITY_BACKFILL_SLEEP` | `0.5` | 历史回填不同交易日之间的等待秒数 |
| `ROUTINE_MARKET_REGIME_LOOKBACK_DAYS` | `252` | 市场状态读取正式行情的自然日窗口；必须至少 90 |
| `TUSHARE_STOCK_BASIC_TTL_HOURS` | `24` | `stock_basic` 请求缓存最长有效时间；同一流水线内复用，避免重复请求 |
| `ROUTINE_FINANCIAL_PERIODS` | `4` | 每日通过 Tushare VIP 重拉的最近报告期数 |
| `ROUTINE_FINANCIAL_SLEEP` | `0.15` | 财务 VIP 请求之间的最小间隔秒数 |
| `ROUTINE_FEATURE_WORKERS` | `8` | 内存或 CPU 紧张时降低 |
| `ROUTINE_FEATURE_EXECUTOR` | `processes` | CPU 密集的特征计算默认使用多进程；调试时可改为 `threads` |
| `ROUTINE_DAILY_BASIC_MIN_MATCH_RATE` | `0.98` | B1 增量候选与 `daily_basic` 匹配率门禁；不建议调低 |
| `B1_FEATURE_MAX_SYMBOL_ERROR_RATE` | `0.001` | B1 特征构建允许的单股异常比例；超过即失败 |
| `ROUTINE_CHAN_WORKERS` | `4` | 缠论增量候选扫描并发数；Web 每日更新并行阶段上限为 4 |
| `ROUTINE_WEB_WORKSPACE_WORKERS` | `6` | 下游接口限频或机器负载高时降低 |
| `SIMILAR_PATTERN_CACHE_WORKERS` | `4` | 相似向量计算占用高时降低 |
| `SIMILAR_PATTERN_FORCE_VECTOR_CACHE` | 空 | 设为 `1` 可在下一次相似走势刷新时强制重建全市场参考库；正常每日任务无需设置 |

一次运行同时包含多层并发。B1 特征缓存与全市场规则信号各自会使用多进程，外层默认依次执行，避免两个进程池争抢同一批 CPU；可转债、配债股和 BYD 做T 在共享数据就绪后提前并行。每日计划、Dashboard、模型评分与缠论评分也会并行，其中两类评分各自最多使用 4 个 worker。不要盲目把每个并发参数都调大；优先观察内存、CPU、Tushare 限频和 MySQL 写入延迟。

日线正式存储为 MySQL `market_daily` 与年月分区 Parquet 镜像。每日刷新只更新当天所在月份，旧的 `data/raw/daily/*.parquet` 逐股票文件和 `daily_XXXXXX_XX` MySQL 分表不再使用。

B1-family 与 z-skill 的生产信号刷新一次读取统一行情分区，再按股票分组交给多进程计算。每日流程不自动物化逐股票因子缓存，避免数千个小文件和额外 I/O。

日线和 `daily_basic` 完成后，流水线还会刷新因子参考数据：`stock_basic` 每日缓存复用并同步，沪深300按最新日期回看 10 天增量合并，`stk_limit` / `suspend_d` / `stock_st` 组成当日可交易快照，`fina_indicator` / `income` / `cashflow` 通过 VIP 接口重拉最近 4 个报告期，并更新 `v44` 使用的全市场分析师一致预期快照。随后只读本地正式行情和沪深300生成市场状态，不再访问其他行情源。同一交易日重试会复用已成功的财务和分析师快照检查点。所有写入都使用业务唯一键去重和临时文件原子替换；审计位于 `data/raw/source_audit/*_reference_data/manifest.json`。

若任务在共享数据刷新之后失败，当天重试会直接复用已通过门禁的日线、`daily_basic` 和参考数据，从特征计算阶段续跑；复用前必须同时满足源清单预期交易日、本地统一行情日期和 `daily_basic` 日期一致。若核心与扩展股票池已经完成，则从下游工作区或快照阶段续跑。缺少源检查点或日期不一致时会重新执行源刷新，不从下游产物反推“源已成功”。跨日任务不会复用旧检查点。

长线、茶大和三倍量策略的当前生产选择也来自 `configs/strategies/*.yaml`；每日刷新清缓存时重新加载。可转债趋势增强 YAML 明确标记为 `research_only`，不会被误认为 Web 生产策略。

## 常见故障

### Tushare Token 缺失或失效

现象：数据刷新立即失败，日志包含 Token、权限或接口访问错误。

处理：

1. 检查 `.env` 中存在 `TUSHARE_TOKEN`，且没有多余引号或空格。
2. 重新执行健康检查中的 Python 环境命令，确认输出 `True`。
3. 使用较小范围手工刷新验证账号权限，再重跑每日任务。

### 请求限频、超时或连接重置

处理：

```dotenv
ROUTINE_DAILY_WORKERS=2
ROUTINE_DAILY_SLEEP=0.25
ROUTINE_DAILY_FINAL_RETRY_WORKERS=2
ROUTINE_DAILY_FINAL_RETRY_SLEEP=1.5
```

重跑同一任务是安全的；日线和工作区按稳定键覆盖或替换，不会产生无限重复记录。

### MySQL 不可用

现象：连接超时、认证失败，或页面无法命中历史工作区快照。

处理：

1. 检查 `MARKET_DATA_SQL_URL`、网络和账号权限。
2. 保持 `MARKET_DATA_MIRROR_PARQUET=1`，本地计算可继续使用 Parquet 镜像。
3. MySQL 恢复后重跑每日任务，补写当日快照。

未配置 SQL URL 时，行情层可以回落到 Parquet；依赖 SQL 的历史工作区快照不会持久化，页面可能重新计算。

### 单个 Tab 失败

每日流水线会继续执行其他非短线工作区，并在 manifest 的 `refresh_daily_web_workspaces` 下记录失败项和错误文本。

处理：

1. 找到失败的工作区键和错误。
2. 确认共享行情已经成功。
3. 启动 Web 服务后，用对应 `scope` 单独补跑。
4. 重新查询刷新状态，并检查页面更新时间。

### 刷新状态长期停在 running

后台状态有超时和进程中断识别，但异常退出后仍应核对：

```bash
curl http://127.0.0.1:8088/api/selector/refresh-latest/status
```

如果服务进程已经重启，状态会被标记为中断或过期；确认没有其他刷新进程后重新提交任务。

### 页面数据日期没有变化

依次检查：

1. 当天是否为交易日，Tushare 是否已经发布当日日线。
2. manifest 的 `refresh_data` 是否成功，最新行情日期是否前进。
3. 对应工作区步骤是否成功。
4. 页面是否读取了带旧参数的快照；使用页面刷新按钮强制重建。
5. 配债股和相似走势的 `generated_at` 是否为当天；两者首次打开会自动补刷过期缓存。

## 验证与回归

代码或配置变更后执行：

```bash
PYTHONPATH=src pytest -q
ruff check src tests
```

涉及页面/API 时，至少验证：

```bash
curl --fail http://127.0.0.1:8088/api/health
curl --fail http://127.0.0.1:8088/api/selector/refresh-latest/status
```

提交前确认 `git status --short` 不包含 `.env`、`data/`、`models/` 或 `reports/`。
