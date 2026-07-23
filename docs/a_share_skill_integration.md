# A 股分析 Skill：项目行情与研究历史集成

## 目标

`analyze-a-shares` 在本项目中复用现有 Tushare、MySQL 和 Parquet 配置，不创建第二套 Token、缓存或行情口径。每次完成的个股研究保存为不可变记录，供后续财报、公司事件、行业事件和定期复盘读取。

## 历史行情适配

入口：

```bash
PYTHONPATH=src python -m quant.research.a_share_skill_data \
  --ticker 600519.SH \
  --cutoff 2026-07-23T18:00:00+08:00 \
  --kind stock \
  --bars 250
```

实现约束：

- 通过 `quant.routine.paths.load_project_env()` 读取项目现有 `.env`。
- 通过 `MarketDataStoreConfig.from_env()` 使用已有 MySQL/Parquet 存储。
- 股票读取正式 `daily` 数据集；指数读取 `data/raw/index_<代码>.parquet`。
- 价格固定标为未复权；Tushare `vol` 映射为 `volume`，单位标为“手”。
- 每根日线保守按交易日 `16:00:00+08:00` 视为可得，严格排除晚于分析截止时点的数据。
- 本地行业指数不存在时返回明确错误，由研究报告披露缺口。

输出可直接传给：

```bash
python .agents/skills/analyze-a-shares/scripts/price_volume_snapshot.py \
  stock_ohlcv.json \
  --format markdown
```

## 个股研究档案

默认位置：

```text
reports/a_shares/<代码.交易所>/
```

该目录已被 Git 忽略。每条记录包含：

- `report.md`：当次完整报告。
- `record.json`：结论、论点、情景、监测、证据与认知变更账本。
- `index.json`：该股票全部历史记录索引。
- `latest.json`：最近记录指针，不替代历史。

生成研究包模板：

```bash
PYTHONPATH=src python -m quant.research.a_share_history template \
  --ticker 600519.SH \
  --company-name 贵州茅台
```

保存：

```bash
PYTHONPATH=src python -m quant.research.a_share_history save research_bundle.json
```

查询严格早于新分析截止时点的基线：

```bash
PYTHONPATH=src python -m quant.research.a_share_history baseline \
  --ticker 600519.SH \
  --before 2026-08-31T18:00:00+08:00
```

查看历史：

```bash
PYTHONPATH=src python -m quant.research.a_share_history list --ticker 600519.SH
PYTHONPATH=src python -m quant.research.a_share_history show --ticker 600519.SH
```

## 认知迭代规则

新记录自动挂接严格早于当前截止时点的最近记录。更新研究必须填写：

- 新事实。
- 旧论点的前后变化和分类。
- 经营模型变化。
- 三情景与估值变化。
- 旧判断错误与教训。
- 下一次检查项。

旧记录不可覆盖。相同输入重复保存时返回同一记录 ID；不同内容由内容哈希生成新记录。

## 验证

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_a_share_skill_data.py \
  tests/test_a_share_history.py

python3 .agents/skills/analyze-a-shares/scripts/validate_skill.py \
  .agents/skills/analyze-a-shares
```
