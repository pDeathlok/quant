# 数据与中间产物存放边界

本文档记录 2026-06-17 对当前项目数据产物的 review 结论，目标是让页面可复盘结果优先从 MySQL 读取，研究和训练中间件继续保留在文件系统。

## 应该放 MySQL

| 数据 | 当前位置 | 建议位置 | 原因 |
| --- | --- | --- | --- |
| 短线策略完整推荐快照 | `selector_snapshots` MySQL，失败时落 `data/selector_snapshots/*.json` | MySQL 主存，JSON 仅作为无 SQL 配置时兜底 | 页面按日期/策略组合读取，需要稳定复盘、快速命中和避免 schema key 变化导致重算 |
| 长线策略股票池快照 | `long_stock_pool_snapshots` MySQL，失败时落 `data/long_stock_pool_snapshots/*.json` | MySQL 主存，JSON 兜底 | 页面按日期和 variant 查询，属于用户可见策略结果 |
| 可转债策略计划 | `web_workspace_snapshots` MySQL | MySQL 主存 | Tab 进入会构建多套策略候选，适合首次生成后按 `trade_date + limit` 复用 |
| 配债股跟踪 payload | `web_workspace_snapshots` MySQL | MySQL 主存 | 页面按固定筛选参数读取，适合缓存最新 asof 结果，刷新时再重建 |
| BYD 日线做T计划 | `web_workspace_snapshots` MySQL | MySQL 主存 | 单票工作台页面反复进入不应重复计算；按持仓参数区分缓存 |
| 行情日线、基础数据 | `MarketDataStore` 对应 MySQL 表，镜像 parquet | MySQL 主存，parquet 镜像 | 多入口共享，增量刷新和线上读取需要一致；parquet 保留给研究脚本批量扫描 |
| 可复盘的页面级推荐 payload | 部分仍在 `data/web/*.json` 或策略模块输出 | 后续收敛到 MySQL 快照表 | 页面展示依赖、日期敏感，应该可按日期审计和回放 |

## 应该放文件

| 数据 | 当前位置 | 建议位置 | 原因 |
| --- | --- | --- | --- |
| 特征训练集、候选信号矩阵 | `data/features/**/*.parquet` | 文件 | 体积较大、列式扫描、研究脚本批量读取，parquet 更合适 |
| 研究样本与回测中间结果 | `data/research/**/*.parquet`、`reports/**/*.csv/json/md` | 文件 | 实验产物需要保留版本轨迹，不适合高频在线查询 |
| 模型训练报告、参数搜索结果 | `reports/**` | 文件 | 人读报告和实验审计为主，通常随代码版本一起管理 |
| Tushare 原始缓存镜像 | `data/cache/*.parquet` | 文件或淘汰 | 仅作为抓取缓存/离线兜底，不应成为页面主读取源 |
| 运行审计、失败清单、manifest | `data/audit`/对应脚本目录 | 文件 | 批处理运行记录，便于排障，不参与页面查询主路径 |

## 本次落位动作

- 新增 `scripts/research/backfill_web_strategy_snapshots.py`，用于按日期重建短线完整策略池，并写入 `selector_snapshots`。
- 同一脚本可同步补齐长线 `tea`、`tea_safe`、`v44` 股票池到 `long_stock_pool_snapshots`。
- 6 月补刷时使用当前 schema key 重新 upsert，旧 key 只作为兼容读取兜底，不再依赖它承载最新结果。
- 新增通用 `web_workspace_snapshots` cache-aside 层：可转债策略、配债股跟踪、BYD 日线做T计划在普通读取时优先命中 MySQL；`refresh=true` 时强制重算并覆盖快照。

## 后续建议

- 给 `selector_snapshots(signal_date, strategies_key, include_extended, updated_at)` 和 `long_stock_pool_snapshots(signal_date, variant, updated_at)` 增加组合索引；当前表规模不大，先不强制迁移。
- 对 `web_workspace_snapshots(workspace, params_key, snapshot_date)` 保持组合索引；如果后续 payload 增长明显，再拆出候选明细表。
- 研究脚本继续产出 parquet/csv/md，但凡被页面稳定读取的最终结果，应通过服务层写 MySQL 快照，避免页面直接依赖研究目录文件。
