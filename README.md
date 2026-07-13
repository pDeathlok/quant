# Quant 策略工作台

Tushare-first 的量化研究与每日策略工作台，覆盖短线、缠论、长线、可转债、配债股、BYD 做T 和相似走势 7 个工作区。

> 本项目用于策略研究和数据分析，不构成投资建议。数据、模型、回测与页面信号都需要结合交易所公告和真实账户约束复核。

## 能力概览

- 增量刷新 Tushare 日线数据，并支持 MySQL 主存储和 Parquet 镜像。
- 构建 B1/B2/B3 与扩展策略特征、模型评分和每日候选池。
- 在一个 FastAPI + 静态前端工作台中查看 7 类策略结果。
- 按日期保存选股器和工作区快照，支持历史复盘。
- 每日流水线采用有依赖的并行编排，单个非核心工作区失败不会阻断其他工作区。

## 前置条件

- Python 3.9 或更高版本。
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

首次进行真实数据刷新前，编辑 `.env` 并至少填写：

```dotenv
TUSHARE_TOKEN=your_token
```

## 每日任务

执行完整日常刷新，不运行正式组合回测：

```bash
PYTHONPATH=src python -m quant.routine.cli daily --refresh-data --skip-backtest
```

执行顺序为：

1. 串行刷新共享 Tushare 行情。
2. 并行构建特征缓存和规则信号缓存。
3. 生成模型评分、短线股票池和例行产物。
4. 并行刷新缠论、长线、可转债、配债股、BYD 做T、相似走势 6 个工作区。
5. 将每个步骤写入 `data/routine/<运行时间>/manifest.json`。

短线策略由主流水线生成，因此这套流程覆盖全部 7 个 Tab。项目提供每日任务入口，但不内置常驻调度器；生产环境需要由 cron、launchd 或其他调度平台每天调用上述命令。详细操作和故障处理见 [每日运行与故障排查](docs/operations.md)。

## 配置

复制 [`.env.example`](.env.example) 后按环境填写；`.env` 已被 Git 忽略。

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `TUSHARE_TOKEN` | 无 | 正式拉取 Tushare 数据时必填 |
| `MARKET_DATA_BACKEND` | `mysql` | `mysql`/`sql` 使用 SQL 主存储，其他值使用 Parquet |
| `MARKET_DATA_SQL_URL` | 无 | SQLAlchemy MySQL 连接串；未配置时可回落到 Parquet 镜像 |
| `MARKET_DATA_ROOT` | `data/raw` | Parquet 数据根目录 |
| `MARKET_DATA_MIRROR_PARQUET` | `1` | SQL 写入时是否同时保留 Parquet 镜像 |
| `MARKET_DATA_SQL_CONNECT_TIMEOUT` | `10` | MySQL 连接超时，单位秒 |
| `MARKET_DATA_SQL_READ_TIMEOUT` | `60` | MySQL 读取超时，单位秒 |
| `MARKET_DATA_SQL_WRITE_TIMEOUT` | `60` | MySQL 写入超时，单位秒 |
| `ROUTINE_DAILY_WORKERS` | `4` | 日线刷新并发数 |
| `ROUTINE_DAILY_SLEEP` | `0.08` | Tushare 请求间隔，单位秒 |
| `ROUTINE_FEATURE_WORKERS` | `8` | 特征构建并发数 |
| `ROUTINE_WEB_WORKSPACE_WORKERS` | `6` | 六个下游工作区的最大并发数 |
| `SIMILAR_PATTERN_CACHE_WORKERS` | `4` | 相似走势向量缓存并发数 |

完整的重试和增量参数见 [每日运行与故障排查](docs/operations.md#资源与限流配置)。

## 常用命令

```bash
# 仅验证每日流水线编排，不访问外部行情源
PYTHONPATH=src python -m quant.routine.cli daily --skip-backtest

# 单独生成短线每日计划或 dashboard
PYTHONPATH=src python -m quant.routine.cli plan
PYTHONPATH=src python -m quant.routine.cli dashboard

# 本地事件驱动回测
PYTHONPATH=src python main.py backtest --strategy momentum --symbol 600000 \
  --start-date 20200101 --end-date 20231231

# 测试与静态检查
PYTHONPATH=src pytest -q
ruff check src tests
```

## 项目结构

```text
src/quant/data/            数据源、标准化和统一存储
src/quant/features/        项目级变量与特征
src/quant/strategies/      可复用策略实现
src/quant/routine/         每日任务和产物编排
src/quant/webapp/          FastAPI 路由与工作区服务
web/                       单页工作台前端
scripts/research/          研究、训练、回测和校准脚本
configs/strategies/        已确认的生产策略配置
docs/strategies/           策略依据、口径和迭代记录
tests/                     单元、集成和 API 测试
```

`data/`、`models/`、`reports/` 和 `web/data/` 是本地运行产物，不提交到 Git。

## 文档

- [文档中心](docs/README.md)
- [系统架构与数据流](docs/architecture.md)
- [API 参考](docs/api.md)
- [每日运行与故障排查](docs/operations.md)
- [项目结构与 MySQL 存储](docs/project_structure_and_storage.md)
- [正式策略档案](docs/strategies/b1_selected_strategy_record.md)

## 开发约定

- 新策略进入例行任务前，必须在 `docs/strategies/` 记录数据口径、入场/离场规则、风险和回测结果。
- 新功能必须补充对应测试；涉及路由或工作区缓存时同时更新 API 或运维文档。
- 禁止提交 `.env`、Token、数据库密码、原始数据、模型和报告产物。
- 清理研究脚本前先确认已有正式策略档案或可复现的替代实现。
