# B1 正式策略档案

## 1. 文档目的

本文档记录当前已经确认保留的 B1 策略：`B1 稳健版` 与 `B1 进攻版`。

项目后续会持续清理历史实验脚本、旧模型、旧报告和临时产物。为了避免清理后丢失策略制定依据，本文档固定记录：

- B1 原始信号如何定义。
- 使用了哪些模型预测值。
- 为什么选择当前买入阈值。
- 为什么选择当前卖出规则。
- 当前回测表现。
- 已知缺陷和后续迭代方向。

后续 review 或再次迭代时，应优先阅读本文档，再看 `configs/strategies/b1_selected.yaml` 和 `src/quant/routine/`。

B1 模型变量来源和 Tushare 覆盖审计见：

```text
docs/strategies/b1_feature_source_audit.md
```

## 2. B1 原始信号

B1 是一个偏超跌反弹的候选信号。当前正式版本已经删除了早期的成交量条件。

当前 B1 候选池使用的核心条件：

```text
当日涨跌幅在 -2% 到 +2% 之间
当日振幅 < 7%
BBI > MA60
KDJ J < 0
排除 ST、退市相关股票
不再要求“当日成交量大于上一日成交量”
```

删除成交量条件的原因：

```text
早期 B1 要求当日成交量大于上一日成交量。
后续分析认为这个条件会过早排除一部分缩量企稳的机会。
因此正式版本中将成交量放大从硬条件中删除，量价状态改为后续分析和辅助过滤项。
```

## 3. 买入逻辑

正式策略不再使用“每日排序买前 XX%”。

原因：

```text
真实交易中不一定每天都要交易。
如果当天没有足够好的机会，策略应该允许空仓。
因此买入规则改为固定阈值：
只要模型预测值达到阈值就买；
没有达到阈值就不买。
```

买入执行口径：

```text
T 日出现 B1 信号并满足模型阈值
T+1 开盘买入
```

## 4. 当前生产发布

当前唯一生产发布为 `b1-20260722`，使用五个同版本模型：

| 字段 | 含义 |
|---|---|
| `pred_up5_es` | T+1 开盘后未来区间上涨 5% 的概率 |
| `pred_up8_es` | T+1 开盘后未来区间上涨 8% 的概率 |
| `pred_up10_es` | T+1 开盘后未来区间上涨 10% 的概率 |
| `pred_down2_es` | T+1 开盘后先向下触发 2% 风险区间的概率 |
| `pred_down3_es` | T+1 开盘后先向下触发 3% 风险区间的概率 |

正式 YAML 同时声明发布 ID、模型目录、模型清单、回测摘要和兼容审计。正式回测与每日计划都从该 YAML 加载规则，不再各自维护阈值。

## 5. 正式保留策略

### 5.1 B1 稳健版

```text
策略 ID：b1_stable
入场：pred_up10_es >= 0.20 且 pred_down3_es <= 0.40
执行：T+1 开盘买入
退出：8% 固定止盈，1.5% 盘中硬止损，最晚 T+5 到期
定位：主策略，优先控制回撤和资金曲线稳定性
```

### 5.2 B1 进攻版

```text
策略 ID：b1_aggressive
入场：pred_up8_es >= 0.70 且 pred_down3_es <= 0.45
执行：T+1 开盘买入
退出：不设盘中止盈止损，最晚 T+9 收盘退出
定位：低频高弹性机会；必须接受更少样本和更大波动
```

没有任一固定阈值命中时，正式计划为空仓；不会用每日 TopN 补足候选，也不会回退到上一交易日信号。

## 6. 样本外回测表现

口径为 2025-01-01 以后、T+1 开盘买入、同日止损与止盈同时触发时按止损优先。2026-07-26 使用最新行情重跑：

| 策略 | 交易数 | 平均单笔收益 | 胜率 | 日度夏普 | 最大回撤 | 盈亏比 |
|---|---:|---:|---:|---:|---:|---:|
| B1 稳健版 | 5669 | 0.5866% | 43.18% | 0.5964 | -40.67% | 1.5471 |
| B1 进攻版 | 106 | 0.7932% | 45.28% | 2.3246 | -49.97% | 1.2001 |
| B1 旧基准 | 15069 | -0.7483% | 31.62% | -6.1371 | -94.76% | 0.6171 |

这里的最大回撤是“所有信号按日等权形成的研究曲线”回撤，尚未叠加真实组合的持仓上限、仓位预算和交易冲突约束，不能直接等同于实盘账户回撤。

## 7. 新旧模型同口径回验

同一候选池、同一最新行情、同一入场和退出规则下：

| 策略 | 版本 | 交易数 | 平均单笔收益 | 日度夏普 | 最大回撤 | 盈亏比 |
|---|---|---:|---:|---:|---:|---:|
| 稳健版 | 旧模型 | 5622 | 0.5933% | 0.1586 | -54.44% | 1.5543 |
| 稳健版 | `b1-20260722` | 5669 | 0.5866% | 0.5964 | -40.67% | 1.5471 |
| 进攻版 | 旧模型 | 93 | 0.6558% | 1.9877 | -60.89% | 1.1707 |
| 进攻版 | `b1-20260722` | 106 | 0.7932% | 2.3246 | -49.97% | 1.2001 |

新版稳健策略牺牲了约 0.0067 个百分点的平均单笔收益，但显著改善夏普与最大回撤；进攻版的收益、夏普、回撤和盈亏比均改善。因此生产只保留 `b1-20260722`。

## 8. 当前生产资产

```text
configs/strategies/b1_selected.yaml
models/production/b1/{up5_es,up8_es,up10_es,down2_es,down3_es}.joblib
models/production/b1/manifest.json
reports/b1/current/backtest.md
reports/b1/current/summary.csv
reports/b1/current/trades.csv
reports/b1/current/model_compatibility_audit.json
web/data/b1_daily_plan.json
web/data/dashboard.json
```

正式每日入口为：

```bash
python scripts/run_daily_web_refresh.py
```

发布审计必须为 `valid`，生产模型文件必须齐全。当天 B1 特征没有命中时发布空计划，而不是发布旧信号。

## 9. 例行数据源规则

当前生产 B1、例行任务和模型训练按 Tushare-only 口径运行：

```text
日线行情来自 Tushare daily。
每日指标来自 Tushare daily_basic。
股票名称、行业等基础信息来自 Tushare stock_basic。
正式策略判断、模型训练、候选复盘不依赖 AkShare 字段。
```

审计输出：

```text
data/raw/source_audit/YYYYMMDD_HHMMSS/daily_source_audit.csv
data/raw/source_audit/YYYYMMDD_HHMMSS/failed_symbols.csv
data/raw/source_audit/YYYYMMDD_HHMMSS/manifest.json
data/raw/source_audit/YYYYMMDD_HHMMSS_daily_basic/daily_basic_audit.csv
```

该审计用于后续检查：

```text
Tushare daily 每只股票是否刷新成功。
Tushare daily_basic 每个交易日是否刷新成功。
每只股票或交易日是否发生自动重试，查看 attempts 字段。
失败股票是否可直接用 failed_symbols.csv 做增量补跑。
```

## 10. 已知风险

```text
1. 当前回测是日线级模拟，无法知道盘中先触发止损还是止盈；同日同时触发时采用止损优先的保守假设。
2. 稳健版在 2024 测试段仍出现较大回撤，说明市场阶段过滤仍需继续研究。
3. 进攻版样本外交易数较少，不能只看平均收益，需要持续观察样本扩展后的稳定性。
4. 当前没有纳入真实交易成本、滑点、涨跌停无法成交等完整实盘约束。
5. daily_basic 的估值字段存在天然缺失，模型侧使用中位数填充；后续做因子评价时应单独观察缺失率和覆盖率。
```

## 11. 后续迭代方向

优先级从高到低：

```text
1. 增加市场阶段过滤，避免系统性下跌阶段连续止损。
2. 增加行业集中度和单日最大信号数限制。
3. 增加真实交易成本、滑点、涨跌停成交约束。
4. 将根目录研究脚本中的 B1 引擎迁入 src/quant/routine/b1_engine.py。
5. 前端增加每日候选、持仓跟踪和策略开关。
```

## 12. Review 指引

以后 review 当前策略时，按以下顺序看：

```text
1. docs/strategies/b1_selected_strategy_record.md
2. configs/strategies/b1_selected.yaml
3. src/quant/routine/strategies.py
4. src/quant/routine/dashboard.py
5. reports/b1/current/backtest.md
```

如果未来删除旧实验模型和旧研究脚本，本文档仍应保留。
