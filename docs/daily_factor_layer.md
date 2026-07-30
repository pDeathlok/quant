# 每日统一因子层

`daily_factor_layer` 统一 B1、family rules 与 z-skill 的日线技术因子口径。生产每日刷新默认在内存中计算，不为每只股票写入因子小文件。

## 设计约束

- 原始行情仍是唯一事实来源，因子层可随时重建。
- 缓存键为 `factor_version / ts_code / year`，公式升级时递增版本，不原地污染旧口径。
- 每个年度分区以 `(symbol, date)` 幂等替换；重复执行不会增加重复行。
- 手工研究需要物化因子时，可显式运行下方命令；该缓存可删除并从原始行情重建。
- 公共 KDJ、BBI、均线、量能、周/月线因子使用统一名称；z-skill 原有特殊口径使用 `z_` 前缀，消费时映射回旧字段，防止同名异义。

## 手工预热或回填

```bash
PYTHONPATH=src:scripts/research python scripts/research/refresh_daily_factor_layer.py \
  --incremental-start-date 2026-07-21 \
  --workers 8 \
  --executor processes
```

日常不运行该命令：生产信号刷新一次读取统一行情，并在每只股票的共享 DataFrame 上计算一次公共因子，再同时生成 family 与 z-skill 信号。预热命令只用于需要物化年度因子分区的手工研究，不是 Web 每日更新的前置条件。

可通过 `DAILY_FACTOR_ROOT` 临时切换缓存目录；默认目录为 `data/features/daily_factor_layer`。

## 新增因子

后续策略不得在策略文件内复制通用 rolling/EMA/KDJ 计算。通用公式加入
`src/quant/features/daily_factor_layer.py` 的注册列；只属于单一策略且口径特殊的字段应使用策略前缀，例如 `z_`。

修改公式时必须：

1. 增加 `FACTOR_LAYER_VERSION`；
2. 添加旧公式与新统一层的逐字段一致性测试；
3. 先预热新版本，再切换消费者。
