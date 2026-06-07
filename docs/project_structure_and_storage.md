# 项目结构与 MySQL 存储说明

## 目标结构

项目长期按“数据层、变量层、策略层、研究层、应用层、文档层”维护。

```text
src/quant/data/          数据源、字段标准化、统一存储
src/quant/features/      项目变量库
src/quant/strategies/    可复用策略规则
src/quant/routine/       例行任务编排
src/quant/webapp/        FastAPI 后端
web/                     前端选股器
scripts/research/        研究、训练、回测脚本
configs/strategies/      已确认策略配置
docs/                    项目与策略文档
```

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
signal_date + strategies + include_z_skill
```

用途：

- 历史日期复盘优先读取 MySQL 快照，避免每次重新扫描全市场。
- 最新数据刷新完成后，会重新计算最新日期股票池并写入快照。
- 如果本地没有配置 `MARKET_DATA_SQL_URL`，开发环境会临时写入 `data/selector_snapshots/*.json`，但生产复盘应使用 MySQL。

## 为什么暂时保留 parquet 镜像

部分研究脚本仍按目录扫描方式读取全市场日线或特征产物，例如：

```text
scripts/research/train_b1_tushare_models.py
scripts/research/train_z_skill_models_and_backtest.py
src/quant/strategies/custom/z_skill_patterns.py
```

因此当前迁移策略是 MySQL 主存储 + parquet 镜像兼容。后续每次改造一个研究脚本，就把它从目录读取迁移到 `MarketDataStore.read_frame()` 或批量 SQL 查询。

## 策略选股器复盘范围

前端复盘日期暂从 `2026-06-01` 开始。原因：

- 当前应用阶段只需要验证 6 月以来每日候选池。
- 更早历史复盘首次生成 z-skill 日线信号较慢。
- 后续会补日期级预计算缓存和收益曲线后再扩展日期范围。

## 提交安全要求

提交前必须确认：

- `.env`、真实 token、数据库密码没有被提交。
- `data/`、`models/`、`reports/` 没有进入暂存区。
- 新策略如果替代旧策略，先在 `docs/strategies/` 记录策略来源、变量、买入卖出逻辑、回测口径，再删除旧脚本或旧模型。
