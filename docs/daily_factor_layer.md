# 每日统一因子层

项目现在把“因子定义、计算、刷新、应用”拆成同一契约下的四层，而不是继续把 B1 候选缓存当作项目因子层：

- `factor_registry.py`：机器可读注册表，当前登记 147 个日频项目因子和 156 个长线因子字段；
- `project_factor_layer.py`：完整日频因子唯一计算入口，市场因子与 `daily_basic` 合并后严格输出 147 列；
- `daily_factor_layer.py`：门控和多策略共用的轻量滚动缓存，保留快速日刷能力；
- `long_weekly_factors.py`：公告日/报告日点时的长线周频估值、双 PR、历史分位、行业相对和研报因子。

跨数据源、特征、模型分和最终产物的执行依赖由
`src/quant/application/daily_dependencies.py` 统一登记；维护规则见
[每日依赖注册表](daily_dependency_registry.md)。`factor_registry.py` 只负责字段元数据，不能替代执行 DAG。

每日 Web 刷新还会把长线页面实际使用的点时截面原子发布到
`data/features/long/YYYYMMDD.parquet` 与 `latest.parquet`，最后再写页面股票池快照。
`latest.json` 记录交易日、schema、行数和路径；因子截面缺字段、股票重复或日期不一致时，
页面股票池不会被标记为刷新成功。

短线生产采用“先算便宜门控、命中后再算完整重因子”，避免每天先对全市场计算 Alpha、周/月指标。长线训练只在好股票池内按周末采样。两者共享注册、命名、样本准入和版本审计，但不强迫使用同一种物理物化粒度。

## 设计约束

- 原始行情仍是唯一事实来源，因子层可随时重建。
- 缓存键为 `factor_version / ts_code / year`，公式升级时递增版本，不原地污染旧口径；当前轻量缓存为 `v2-causal-price`，增量信号状态为 `signal-v2-causal-price`。
- 每个年度分区以 `(symbol, date)` 幂等替换；重复执行不会增加重复行。
- 手工研究需要物化因子时，可显式运行下方命令；该缓存可删除并从原始行情重建。
- 公共 KDJ、BBI、均线、量能、周/月线因子使用统一名称；z-skill 原有特殊口径使用 `z_` 前缀，消费时映射回旧字段，防止同名异义。
- 财务只允许 `ann_date <= signal_date`，研报只允许 `report_date <= signal_date`；训练标签不进入注册表。
- 模型首轮因子准入只看非空样本量、覆盖率与是否常量，不使用单因子收益或重要性预筛。
- `project-v4-causal-price-alpha` 修正两类历史泄漏：Alpha101 单股票实现改为过去 252 个交易日滚动时序排名；连续价格改为只从公司行动发生日向后累积的因果尺度，未来除权不再改写过去绝对均线。
- 已发布但未声明 schema 的旧短线模型只允许在显式 `project-v1-latest-scale-global-rank` 兼容模式下消费旧口径因子；旧模型与 v4 特征、v4 模型与旧特征都会硬失败，不能静默混用。正式日刷暂时固定旧发布版口径，新研究/重训默认固定 v4，待独立回测和发布门禁通过后再迁移生产。

## 手工预热或回填

```bash
PYTHONPATH=src:scripts/research python scripts/research/refresh_daily_factor_layer.py \
  --incremental-start-date 2026-07-21 \
  --workers 8 \
  --executor processes
```

日常不运行该命令：正式刷新会先发布 `data/features/factor_registry/latest.json`，随后刷新 B1、family、z-skill 和最新模型评分所需的日频因子应用物化；长线步骤再发布当日长线因子截面，并由同一截面生成页面股票池。生产信号刷新一次读取统一行情，并在每只股票的共享 DataFrame 上计算一次公共因子。预热命令只用于需要物化年度因子分区的手工研究，不是 Web 每日更新的前置条件。

可通过 `DAILY_FACTOR_ROOT` 临时切换缓存目录；默认目录为 `data/features/daily_factor_layer`。

## 长线研究源数据刷新

当前生产 Tea/Tea-safe/v44 不消费 `margin_detail`、`moneyflow`、`top_list` 和 `stk_holdertrade`，因此这些研究输入不再随正式日更无条件刷新。活动依赖注册表只在晋级生产消费者明确声明后启用对应源；事件和逐日分区仍分别按业务主键与 `(trade_date, ts_code)` 幂等更新。

质押统计和质押明细需要逐股票请求，不放进每日主链路。需要低频全量校验或恢复时运行：

```bash
PYTHONPATH=src python scripts/research/backfill_long_factor_sources.py \
  --datasets stock_universe holder_trade pledge_stat pledge_detail \
  --start 20130101
```

逐日历史源可用同一入口断点续跑：

```bash
PYTHONPATH=src python scripts/research/backfill_long_factor_sources.py \
  --datasets margin_detail moneyflow top_list \
  --start 20130101
```

每次运行在 `data/raw/source_audit/` 发布请求清单和 manifest。覆盖、字段非空率、重复键、错日分区及研报三年覆盖通过下方命令复核：

```bash
PYTHONPATH=src python scripts/research/audit_long_factor_sources.py
```

审计结果写入 `reports/long_entry_factor_inventory/source_coverage.json`。最新研报一致预期可用于当前池展示；`snapshot_only` 数据和只在当前日期获取的 DataYes 一致预期不得回填历史样本。

## 新增因子

后续策略不得在策略文件内复制通用 rolling/EMA/KDJ 计算。完整项目公式加入 `project_factor_layer.py`，轻量门控公式加入 `daily_factor_layer.py`，元数据同时登记到 `factor_registry.py`；只属于单一策略且口径特殊的字段应使用策略前缀，例如 `z_`。

修改公式时必须：

1. 增加对应 schema/materialization version；
2. 添加旧公式与新统一层的逐字段一致性测试；
3. 将计算器或配置文件加入相应 `feature.*.contract_sources`，并核对回看单位；
4. 先预热新版本，再切换消费者；
5. 运行 `tests/test_daily_dependencies.py`，确保生产策略、模型和最终产物没有漏登记。
