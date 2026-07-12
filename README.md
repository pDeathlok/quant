# Quant 策略选股器

本项目已经从早期研究脚本整理为 Tushare-first 的量化策略工程。当前重点是策略选股器、例行数据刷新、B1/B2/B3/扩展策略复盘与每日候选池生成。

## 当前主线

```text
src/quant/routine/                 # 例行任务：拉取数据、生成每日计划、生成 dashboard
src/quant/webapp/                  # FastAPI 后端接口
web/                               # 前端页面
src/quant/data/market_data_store.py # MySQL/parquet 统一存储层
src/quant/features/                # 项目级变量库
src/quant/strategies/custom/       # B1 family 与扩展策略规则
scripts/research/                  # 正式研究、训练、回测脚本
configs/strategies/                # 已确认策略配置
docs/strategies/                   # 策略制定档案与复盘记录
```

本地数据、模型、报告产物不提交到 Git，统一由 `.gitignore` 排除：`data/`、`models/`、`reports/`、`web/data/`。

## 安装

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"  # 开发、测试和 lint
```

PyTorch 属于可选依赖，需要时使用 `python -m pip install -e ".[ml]"`。

## 环境变量

复制 `.env.example` 为本地 `.env` 后填写真实值，`.env` 不允许提交。

```bash
TUSHARE_TOKEN=...
MARKET_DATA_BACKEND=mysql
MARKET_DATA_SQL_URL=mysql+pymysql://quant_user:<db-password>@127.0.0.1:3306/quant?charset=utf8mb4
MARKET_DATA_MIRROR_PARQUET=1
```

说明：

- `MARKET_DATA_BACKEND=mysql` 是生产口径。
- `MARKET_DATA_MIRROR_PARQUET=1` 会在写入 MySQL 的同时保留 parquet 镜像，兼容当前仍按文件扫描的特征和复盘脚本。
- 如果本机暂未配置 MySQL URL，存储层会回落读取 parquet 镜像，便于本地调试。

## 启动选股器

```bash
PYTHONPATH=src python scripts/run_webapp.py
```

浏览器打开：

```text
http://127.0.0.1:8088
```

前端能力：

- 选择策略查看股票池。
- 选择 2026-06-01 以来的历史日期做轻量复盘。
- 点击“更新最新数据”启动后台 Tushare 数据刷新，并在完成后重新加载股票池。
- 同一策略家族命中数按家族去重；同一策略家族下不同买入操作会分别展示。

## 例行任务

刷新 Tushare 日线数据并生成每日股票池：

```bash
PYTHONPATH=src python -m quant.routine.cli daily --refresh-data --skip-backtest
```

每日任务同时刷新配债股工作区和相似走势决策台。Web 端也会校验 `generated_at`：若缓存
不是当天生成，首次打开对应 Tab 时会自动补刷，页面上的“刷新”按钮仍可用于手工强制更新。

手工补跑日线数据：

```bash
PYTHONPATH=src python -m quant.routine.data_refresh \
  --start 20100101 \
  --output-dir data/raw/daily \
  --workers 2 \
  --sleep 0.25 \
  --retries 3 \
  --retry-base-delay 2 \
  --retry-max-delay 60
```

增量行情默认保存不复权价格，特征层统一构建连续价格。若显式使用 `--adjust qfq` 或
`--adjust hfq`，刷新器会从该股票已有历史的第一天重新拉取，避免不同复权基准直接拼接。

## 本地事件回测

```bash
PYTHONPATH=src python main.py backtest \
  --strategy momentum \
  --symbol 600000 \
  --start-date 20200101 \
  --end-date 20231231
```

可通过 `--data /absolute/path/to/daily.parquet` 指定行情文件；省略时会从项目标准数据目录解析。

刷新 Tushare `daily_basic` 扩展因子：

```bash
PYTHONPATH=src python -m quant.routine.daily_basic_refresh \
  --start 20240101 \
  --workers 4 \
  --sleep 0.25 \
  --retries 3
```

## 主要 API

```text
GET  /api/health
GET  /api/selector/stocks?signal_date=2026-06-04&strategies=B1,B2
POST /api/selector/refresh-latest
GET  /api/selector/refresh-latest/status
GET  /api/b1/plan
POST /api/b1/plan/refresh
GET  /api/b1/history
GET  /api/research/b1
```

## 文档索引

```text
docs/project_restructure_plan.md
docs/project_structure_and_storage.md
docs/factor_tushare_factor_system.md
docs/factor_variable_implementation_matrix.md
docs/factor_variable_dictionary.md
docs/strategies/b1_selected_strategy_record.md
docs/strategies/extended_strategy_model_record_20260607.md
```

## 清理原则

已确认策略必须先写入 `docs/strategies/` 的策略档案，再删除历史模型和实验脚本。无效代码的判断标准：

- 仍依赖 AkShare 或旧 DataFetcher 的入口。
- 与当前 Tushare-only 数据层冲突。
- 只服务一次性研究且已有正式研究报告或策略档案替代。
- 会把本地数据、模型、报告或密钥误提交到 Git。
