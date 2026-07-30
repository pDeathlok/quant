# 量化项目目录重构与例行执行计划

## 目标

把当前以研究脚本为主的 B1 项目，整理成“研究分析、例行生产、结果展示”三层结构。稳健版和进攻版都保留为正式策略，例行流程每天完成数据刷新、特征构建、策略运行、结果落盘和前端展示。

## 当前结构判断

当前项目已经具备完整研究资产：

- `src/quant/`：已有数据、因子、策略、回测、风控、交易等基础包。
- 根目录 `analyze_*.py`、`train_*.py`：主要是实验和分析脚本。
- `models/production/b1/`：当前 B1 生产模型。
- `data/features/b1/`：当前 B1 候选池和特征产物。
- `reports/b1/current/`：当前 B1 正式回测结果。
- `docs/`：策略说明和分析报告。
- `data/raw/daily/`：日线数据。

主要问题是研究脚本和生产流程混在一起，策略参数散落在脚本中，前端没有统一入口查看每日结果。

## 新目录规划

```text
/Users/didi/Project/quant
├── configs/
│   └── strategies/
│       └── b1_selected.yaml
├── docs/
│   ├── strategies/
│   │   └── b1_selected_strategy_record.md
│   └── project_restructure_plan.md
├── src/
│   └── quant/
│       ├── data/
│       ├── strategies/
│       ├── backtest/
│       ├── ml/
│       └── routine/
│           ├── cli.py
│           ├── dashboard.py
│           ├── paths.py
│           ├── pipeline.py
│           └── strategies.py
├── web/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── data/
│       └── dashboard.json
├── data/
│   └── features/
│       └── b1/
│           └── candidates_strict_no_volume_20240101.parquet
├── reports/
│   └── b1/
│       └── current/
│           ├── backtest.md
│           ├── summary.csv
│           └── trades.csv
└── models/
    └── production/
        └── b1/
            ├── up8_es.joblib
            ├── up10.joblib
            └── down3_es.joblib
```

## 已保留的正式策略

### B1 稳健版

```text
pred_up8_es >= 0.55
pred_down3_es <= 0.55
T+1 开盘买入
上涨 4% 后启动回撤止盈
从最高点回撤 2% 卖出
固定止损 1.5%
最晚 T+7 到期卖出
```

### B1 进攻版

```text
pred_up8_es >= 0.65
pred_down3_es <= 0.50
T+1 开盘买入
上涨 5% 后启动回撤止盈
从最高点回撤 2% 卖出
固定止损 1.5%
最晚 T+9 到期卖出
```

## 例行流程

### 1. 例行获取数据

入口：

```bash
PYTHONPATH=src python -m quant.routine.cli daily --refresh-data
```

当前接入点是：

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

数据源规则：

```text
1. 生产数据源使用 Tushare-only。
2. 日线行情来自 Tushare daily。
3. 真实换手率、量比、估值和市值来自 Tushare daily_basic。
4. 股票基础信息来自 Tushare stock_basic。
5. 每次刷新输出 source audit，用于检查覆盖率、失败项和自动重试情况。
```

对照审计输出：

```text
data/raw/source_audit/YYYYMMDD_HHMMSS/daily_source_audit.csv
data/raw/source_audit/YYYYMMDD_HHMMSS/failed_symbols.csv
data/raw/source_audit/YYYYMMDD_HHMMSS/manifest.json
data/raw/source_audit/YYYYMMDD_HHMMSS_daily_basic/daily_basic_audit.csv
```

并发和重试规则：

```text
1. 日线默认并发 workers=2，daily_basic 可按交易日并发补拉，但需要保留请求间隔和重试 buffer。
2. 全局请求间隔 sleep=0.25 秒，即使并发 worker 同时运行，也会排队错峰发起请求。
3. 限频、超时、连接重置等临时错误自动重试 3 次，使用指数退避和随机 jitter。
4. “未获取到数据”按停牌、退市或无行情处理，不反复重试，直接进入 failed_symbols.csv。
5. failed_symbols.csv 可作为下一次 --symbols-file 的输入，只补失败股票。
```

默认不访问外部数据源，使用 dry-run：

```bash
PYTHONPATH=src python -m quant.routine.cli daily
```

### 2. 构建特征变量

当前特征构建复用 B1 兼容研究脚本里的模型特征计算逻辑：

```bash
python scripts/research/analyze_b1_entry_exit_grid.py --candidate-mode strict_no_volume --entry-mode threshold
```

下一步要把 `calculate_minimal_model_features`、`predict_models`、`build_strict_b1_no_volume_candidates` 从研究脚本迁入 `src/quant/ml/` 或 `src/quant/routine/`，让根目录分析脚本只负责研究，不承担生产逻辑。

### 3. 例行跑选定策略

入口：

```bash
PYTHONPATH=src python -m quant.research.b1_formal_combos
```

输出：

```text
/Users/didi/Project/quant/reports/b1/current/summary.csv
/Users/didi/Project/quant/reports/b1/current/trades.csv
```

### 4. 生成前端数据

入口：

```bash
PYTHONPATH=src python -m quant.routine.cli dashboard
```

输出：

```text
/Users/didi/Project/quant/web/data/dashboard.json
```

### 5. 查看前端

启动：

```bash
python -m http.server 8092 --directory web
```

浏览器打开：

```text
http://127.0.0.1:8092
```

## 执行计划

### 阶段 1：结构收敛

- [x] 新增 `configs/strategies/b1_selected.yaml`，把稳健版、进攻版、旧基准配置化。
- [x] 新增 `src/quant/routine/`，提供例行 CLI、策略配置读取、看板 JSON 生成。
- [x] 新增 `web/` 静态前端，可以查看策略指标、最近信号、每日复盘和阶段对比。
- [x] 新增 `docs/project_restructure_plan.md`，记录目标结构和执行入口。
- [x] 新增 `docs/strategies/b1_selected_strategy_record.md`，记录 B1 稳健版和进攻版的制定依据，避免清理旧代码和旧模型后丢失上下文。
- [x] 将生产模型迁入 `models/production/b1/`。
- [x] 将正式回测结果迁入 `reports/b1/current/`。
- [x] 将当前候选池迁入 `data/features/b1/`。

### 阶段 2：生产化重构

- [ ] 把 `analyze_b1_entry_exit_grid.py` 中的候选池构建、模型预测、卖出模拟迁入 `src/quant/routine/b1_engine.py`。
- [x] 把 `analyze_b1_formal_combos.py` 改成只调用包内 `quant.research.b1_formal_combos`；共享退出模拟位于 `quant.research.b1_backtest`。
- [ ] 每次例行运行落盘到 `data/routine/YYYYMMDD_HHMMSS/`，包含 `signals.csv`、`summary.csv`、`manifest.json`。
- [ ] 前端改为读取最新 `data/routine/latest.json`，而不是固定读取 20260606 的回测文件。
- [ ] 清理旧脚本、旧模型、旧报告前，先确认对应策略已经写入 `docs/strategies/`。
- [x] 例行数据刷新改为 Tushare-only，并输出 source audit。

### 阶段 3：每日实盘候选

- [ ] 数据刷新后只对最新交易日生成信号。
- [ ] 前端增加“今日候选”视图，展示股票、行业、模型分、触发策略、建议买入价、止损价、回撤止盈参数。
- [ ] 增加策略开关：稳健版、进攻版、旧基准可以独立启用或关闭。

### 阶段 4：复盘和风控

- [ ] 增加按行业、日期、市场阶段聚合的复盘表。
- [ ] 增加最大单日候选数、单票冷却期、行业集中度限制。
- [ ] 增加一键导出 CSV 和 Markdown 日报。

## 当前验证命令

```bash
PYTHONPATH=src python -m quant.routine.cli dashboard
python -m py_compile src/quant/routine/*.py
PYTHONPATH=src python -m quant.research.b1_formal_combos
```

## 清理原则

后续删除历史资产前遵循以下顺序：

```text
1. 确认对应策略是否还会继续使用。
2. 如果继续使用，先写入 docs/strategies/ 下的策略档案。
3. 将生产模型放入 models/production/<strategy>/。
4. 将正式报告放入 reports/<strategy>/current/。
5. 再删除旧训练产物、旧临时脚本和旧报告。
```
