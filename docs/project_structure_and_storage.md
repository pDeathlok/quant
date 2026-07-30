# 项目结构与 MySQL 存储说明

## 目标结构

项目长期按“数据层、变量层、策略层、研究层、应用层、文档层”维护。

```text
src/quant/data/          数据源、字段标准化、统一存储
src/quant/core/          仓库路径、公共类型和无副作用基础设施
src/quant/application/   API、CLI、调度共同消费的应用用例
src/quant/infrastructure/ 文件系统、SQL 等外部存储适配器
src/quant/features/      项目变量库
src/quant/strategies/    可复用策略规则
src/quant/routine/       例行任务编排
src/quant/webapp/        FastAPI 后端
web/                     前端选股器
web/core/                前端 API 客户端与无状态格式化模块
scripts/research/        研究、训练、回测脚本
configs/strategies/      已确认策略配置
docs/                    项目与策略文档
```

依赖约束：

```text
interfaces(webapp/cli) -> application -> data/features/strategies
routine                -> application -> data/features/strategies
scripts/research       -> application/data/features/strategies
webapp                 -> infrastructure -> MySQL/filesystem
```

`application` 不反向依赖 `webapp` 或 `routine`；`routine` 不依赖 `webapp`。项目路径统一从 `quant.core.paths` 派生，构造配置对象不会自动创建目录，运行入口需要写产物时再显式创建。

通用工作区快照位于 `src/quant/infrastructure/workspace_snapshots.py`。该仓储统一处理参数哈希、日期标准化、最近历史快照选择、原子文件写入及 MySQL 回退；Web 服务保留兼容函数，但不再内联存储实现。

工作区业务用例位于 `src/quant/application/workspaces/`。新增或迁移工作区时，应把纯业务流程和质量判断放在此目录，把行情、快照和缓存访问声明为依赖；`webapp.services` 只负责组合依赖和兼容 API。当前 BYD、可转债网格与配债股已完成该迁移。

本地运行产物不进入 Git：

```text
data/       原始行情、特征缓存、例行 manifest
models/     训练后的模型文件
reports/    研究报告与回测明细
web/data/   静态快照
```

## 数据源原则

当前生产数据源为 Tushare。旧 AkShare 拉取脚本和 `DataFetcher` 已清理，避免项目同时存在两套口径。

保留 `FactorDataAdapter` 对历史字段格式的适配能力，但它只负责字段转换，不再承担数据源切换。

## MySQL 存储

统一入口为：

```text
src/quant/data/market_data_store.py
```

环境变量：

```bash
MARKET_DATA_BACKEND=mysql
MARKET_DATA_SQL_URL=mysql+pymysql://quant_user:<db-password>@127.0.0.1:3306/quant?charset=utf8mb4
MARKET_DATA_MIRROR_PARQUET=1
```

写入逻辑：

- `MARKET_DATA_BACKEND=mysql` 时优先写 MySQL。
- `MARKET_DATA_MIRROR_PARQUET=1` 时同步写 parquet 镜像。
- 未配置 `MARKET_DATA_SQL_URL` 的本地环境会使用 parquet 镜像，便于开发和测试。

当前日线刷新已经通过统一存储层写入：

```text
src/quant/routine/data_refresh.py
```

每日股票池快照也会通过 MySQL 缓存，表名为：

```text
selector_snapshots
```

缓存粒度为：

```text
signal_date + strategies + include_extended
```

用途：

- 历史日期复盘优先读取 MySQL 快照，避免每次重新扫描全市场。
- 最新数据刷新完成后，会重新计算最新日期全策略股票池，并拆分写入 ALL 和每个单独策略的股票池快照。
- 如果本地没有配置 `MARKET_DATA_SQL_URL`，开发环境会临时写入 `data/selector_snapshots/*.json`，但生产复盘应使用 MySQL。

## 为什么暂时保留 parquet 镜像

部分研究脚本仍按目录扫描方式读取全市场日线或特征产物，例如：

```text
scripts/research/train_b1_tushare_models.py
scripts/research/train_z_skill_models_and_backtest.py  # 历史研究脚本，文件名暂保留兼容已有产物
src/quant/strategies/custom/z_skill_patterns.py        # 扩展策略规则模块，文件名暂保留兼容已有研究脚本
```

因此当前迁移策略是 MySQL 主存储 + parquet 镜像兼容。后续每次改造一个研究脚本，就把它从目录读取迁移到 `MarketDataStore.read_frame()` 或批量 SQL 查询。

这里的 Parquet 镜像与 `data/cache/source_merge/tushare/` 请求缓存用途不同：

- `data/raw/daily/`、`data/raw/daily_basic/` 是正式行情镜像，供生产特征和研究脚本读取。
- `data/cache/source_merge/tushare/` 只是按请求参数保存的可重建副本，用于减少短时间内重复请求。
- 请求缓存保留 7 天即可；删除旧请求缓存不会删除 MySQL 或 `data/raw/` 中的正式数据。
- 对 `daily_basic` 额外要求对应 `data/raw/daily_basic/YYYYMMDD.parquet` 存在且非空，才允许清理其旧请求缓存。

## 策略选股器复盘范围

前端复盘日期暂从 `2026-06-01` 开始。原因：

- 当前应用阶段只需要验证 6 月以来每日候选池。
- 更早历史复盘首次生成全市场扩展策略日线信号较慢。
- 后续会补日期级预计算缓存和收益曲线后再扩展日期范围。

## 提交安全要求

提交前必须确认：

- `.env`、真实 token、数据库密码没有被提交。
- `data/`、`models/`、`reports/` 没有进入暂存区。
- 新策略如果替代旧策略，先在 `docs/strategies/` 记录策略来源、变量、买入卖出逻辑、回测口径，再删除旧脚本或旧模型。
