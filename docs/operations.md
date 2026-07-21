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
PYTHONPATH=src python -m quant.routine.cli daily --refresh-data --skip-backtest
```

- `--refresh-data`：真实访问 Tushare；省略时数据刷新为 dry-run，但后续仍使用本地数据重建产物。
- `--skip-backtest`：跳过正式组合回测，缩短日常刷新时间。
- 不加 `--skip-backtest` 时会运行 `scripts/research/analyze_b1_formal_combos.py`。

命令结束时输出 manifest 路径。也可以查找最近一次：

```bash
find data/routine -name manifest.json -type f -print | sort | tail -1
```

打开该 JSON，确认关键步骤 `status` 为 `success`；主动跳过的数据刷新或回测会显示 `skipped`。

## 调度说明

项目提供可重复执行的每日任务，但不内置 cron/launchd 常驻调度。调度器需要满足：

1. 工作目录是仓库根目录。
2. 使用安装了项目依赖的 Python。
3. 同一时刻最多一个完整任务，避免覆盖相同日期的缓存。
4. 将标准输出和错误输出写入受轮转管理的日志。
5. 在 A 股收盘且 Tushare 日线更新后执行。

调度命令本体为：

```bash
cd /absolute/path/to/quant && .venv/bin/python -m quant.routine.cli daily --refresh-data --skip-backtest
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
2. 用 Tushare `trade_cal` 判断当天是否为 A 股交易日；不是交易日或无法可靠确认时直接跳过。
3. 默认先按 `.run/daily_web_refresh.pid` 重启由该脚本启动的本地 web 服务。如果检测到 Web 服务由 `launchd` 等外部守护器托管，则保持现有服务，避免重复启动和端口冲突。
4. 检查 `http://127.0.0.1:8088/api/health` 和前端首页 `http://127.0.0.1:8088/`，确认前后端已就绪。
5. 触发 `POST /api/selector/refresh-latest`，作用域默认 `all`。
6. 轮询 `/api/selector/refresh-latest/status` 并打印进度。
7. 若终态为 `failed/error`，自动再次触发刷新。服务端会优先复用已有的断点续跑能力。
8. 服务端刷新开始前自动清理缓存：手工回测生成的长线研究缓存只保留最近 2 组，相似走势正式向量只保留最新一套，smoke 测试向量缓存全部删除，Tushare 单股请求缓存保留最近 7 天。Tushare `daily_basic` 请求缓存也保留最近 7 天，但只有对应 `data/raw/daily_basic/YYYYMMDD.parquet` 正式文件存在且非空时才删除旧缓存，避免误删唯一副本。策略快照保留 30 天、每个业务分组最多 10 个日期；workspace 快照保留 14 天、每组最多 3 个日期；数据源审计保留 30 天且最多 10 次；routine 历史运行保留 14 天且最多 5 次。每个业务分组最新一期始终保留，对应 MySQL 快照表同步执行相同规则。
9. 相似走势的全市场历史参考库每 7 天最多重建一次；每日任务仍会直接读取自选池股票的最新日线，现场计算目标向量并完成匹配。因此自选股信号按日更新，历史样本及其后续收益标签按周更新。

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
- `--no-restart-service`：调试时复用已有服务；例行任务不要加这个参数。
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
| `ROUTINE_FEATURE_WORKERS` | `8` | 内存或 CPU 紧张时降低 |
| `ROUTINE_WEB_WORKSPACE_WORKERS` | `6` | 下游接口限频或机器负载高时降低 |
| `SIMILAR_PATTERN_CACHE_WORKERS` | `4` | 相似向量计算占用高时降低 |
| `SIMILAR_PATTERN_FORCE_VECTOR_CACHE` | 空 | 设为 `1` 可在下一次相似走势刷新时强制重建全市场参考库；正常每日任务无需设置 |

一次运行同时包含多层并发。不要盲目把每个并发参数都调大；优先观察内存、CPU、Tushare 限频和 MySQL 写入延迟。

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
