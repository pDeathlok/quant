# 带血筹递增分批建仓回测计划

## Goal

在不改变带血筹选股信号的前提下，对比一次性建仓、等比例确认式分批、20%/30%/50% 递增确认式分批和 20%/30%/50% 单纯下跌加仓，判断分批方式能否提高净收益并降低组合回撤。

## Pre-conditions

- [x] `/Users/didi/Project/quant/data/research/blood_chip/features.parquet` 存在，包含因果复权 OHLC、`residual_return_3d` 和成交额字段。
- [x] `/Users/didi/Project/quant/reports/research/blood_chip_exhaustion/path_signals.parquet` 存在，包含已冻结的 `rank_low_vol_no_cap` 信号特征。
- [x] 基准仍使用 `/Users/didi/Project/quant/data/raw/index_000300.SH.parquet`。

## Steps

### Step 1 — 固定分批执行配置和无未来函数测试

**File:** `/Users/didi/Project/quant/tests/research/test_blood_chip_scale_in.py`

- 验证首笔成交为目标仓位的 20%。
- 验证第二、三笔只能由前一交易日收盘条件触发，并在下一交易日开盘成交。
- 验证 20%/30%/50% 三笔的累计目标不超过单槽位预算。
- 验证止损日先退出、不会同日继续加仓。

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_blood_chip_scale_in.py` → 新测试先因模块不存在失败，完成 Step 2 后全部通过。

### Step 2 — 建立分批组合执行引擎

**File:** `/Users/didi/Project/quant/src/quant/research/blood_chip_scale_in.py`

实现四个冻结变体：

1. `one_shot`：100% 单次次日开盘建仓，作为现有策略对照。
2. `equal_confirmed`：33.33%/33.33%/33.34%。第二笔要求前一日收盘进入事件低点上方 3% 至信号收盘下方 3% 的吸筹区，且三日残差收益不低于 -1%；第三笔要求前一日重新站上信号收盘且三日残差收益大于 0。
3. `increasing_confirmed`：20%/30%/50%，触发条件与 `equal_confirmed` 完全相同，只改变资金比例。
4. `increasing_price_only`：20%/30%/50%。第二笔仅要求前一日收盘低于信号收盘 4%，第三笔仅要求低于信号收盘 8%，用于检验“越跌越加”的风险。

案例迭代追加两个因果对照：

5. `increasing_survival`：20% 首仓；至少存活 5 个交易日、守住信号价 95% 且三日残差收益转正后加 30%；至少存活 10 日、站回信号价且残差继续为正后加 50%。
6. `increasing_survival_risk_capped`：触发同上，但每次加仓后把止损提高到加权成本下方 10%，用于检验机械抬升止损是否过紧。

所有变体统一使用：最多 10 个持仓、信号后下一开盘成交、开盘缺口 ±7%、一字涨停不买、一字跌停延迟退出、首笔成交价下方 10% 固定止损、首笔起算最长 120 个交易日、佣金/印花税/过户费/滑点及新事件再入。

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_blood_chip_scale_in.py` → 全部通过。

### Step 3 — 运行固定分段比较并保存案例

**File:** `/Users/didi/Project/quant/scripts/research/backtest_blood_chip_scale_in.py`

信号固定为：`maximum_return_120d <= 50%`、`market_return_60d >= -15%`、`rebound_from_event_low <= 15%`，同日按 `volatility_60d` 从低到高排序。分别运行 2014–2019 研发期、2020–2022 迭代期和 2023–2026 已见诊断期，输出每期净收益、年化收益、最大回撤、胜率、平均单笔收益、盈利因子、平均资金部署比例、三段完成率及代表案例。

**Verify:** `PYTHONPATH=src python scripts/research/backtest_blood_chip_scale_in.py` → 生成 `/Users/didi/Project/quant/reports/research/blood_chip_scale_in/metrics.csv`、`trades.parquet` 和 `report.md`。

### Step 4 — 只在研发期和迭代期判断是否值得上线

判断条件：案例迭代得到的递增生存确认式分批必须同时满足 2020–2022 总收益高于一次性建仓、最大回撤绝对值不扩大超过 3 个百分点、资金加权盈利因子不下降，并且 2014–2019 资金加权单笔收益为正。2023–2026 仅作为已见诊断，不作为选择依据。普通按笔盈利因子会把小额试仓失败与完整仓位赢家等权，不用于递增加仓方案的最终判断。

**Verify:** `/Users/didi/Project/quant/reports/research/blood_chip_scale_in/report.md` 明确给出“上线 / 保留研究 / 否决”及原因；未达条件时不修改中长线线上执行逻辑。

## Rollback

本研究不修改线上策略。若实验实现或口径验证失败，仅删除新增的研究模块、测试、脚本和 `/Users/didi/Project/quant/reports/research/blood_chip_scale_in/` 产物；现有带血筹页面和每日快照不受影响。
