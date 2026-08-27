# 月线低位全市场宽度确认第二轮计划

## Goal

修正首轮用上证综指代表市场状态的风格错配：用当日流动性合格A股的20日收益中位数和上涨家数占比确认全市场修复，再检验月线低9锚、60日区间收复、个股相对宽度强度和周线J修复能否跨阶段形成合理结构。

## Pre-conditions

- [x] 首轮报告 `/Users/didi/Project/quant/reports/research/monthly_low_zone_confirmation/report.md` 已冻结且结论为无候选通过。
- [x] 首轮252日年度结果显示2017年月线低9直接锚353笔、胜率9.35%；上证指数确认未能避开该批次。
- [x] `/Users/didi/Project/quant/data/research/blood_chip_deep_base/features.parquet` 对每个历史交易日含股票20日收益与前20日成交额中位数，可在不使用未来信息的情况下构造横截面宽度。
- [x] 点时年度财务源位于 `/Users/didi/Project/quant/data/raw/fina_indicator.parquet`、`income.parquet`、`cashflow.parquet`、`balancesheet.parquet`，基本面消融只使用确认日已披露记录。

## Fixed second-round contract

### Causal breadth state

- 每个市场交易日只纳入 `return_20d` 有效且前20日成交额中位数至少3,000万元的证券。
- 当日有效证券至少500只；否则宽度状态缺失，不能确认。
- `breadth_median_return_20d` 为上述证券20日收益中位数；`breadth_positive_share_20d` 为20日收益大于0的比例。
- 全市场修复固定为中位收益大于0且上涨占比不低于55%。阈值不按2017—2024结果优化。

### Pre-registered breadth confirmation rules

沿用月线低9锚、5—126个市场日等待、确认日仍较锚前高回撤至少40%、下一开盘执行和三个持有期。新增规则：

1. `breadth_repair`：首轮 `range_mid_reclaim`，且全市场宽度修复。
2. `breadth_relative`：规则1，且股票20日收益高于横截面中位数。
3. `breadth_relative_weekly`：规则2，且最近完成周线 `J>-10`、J高于前周。
4. `breadth_relative_weekly_exhaustion`：规则3，且卖压份额与波动均不高于此前窗口。
5. `breadth_relative_weekly_survival`：规则3确认日使用点时年度数据，至少3年历史、最近利润与经营现金流为正、过去5年利润和经营现金流为正比例均至少60%。
6. `breadth_relative_weekly_exhaustion_survival`：规则4叠加规则5。

首轮八条规则保留为基准，但不因第二轮结果回改。

### Qualification

- 开发期至少60个完成事件，验证期至少100个完成事件；两期各至少24个独立确认日期。
- 两期分别要求胜率≥55%、中位净收益>0、PF≥1.50、超额胜率≥50%、日期聚类95%收益下界>0。
- 已见诊断期要求胜率≥50%、中位净收益>0、PF≥1.20。
- 通过者再运行不创新低15/20/30日邻域；至少两个邻域版本保持正中位收益且胜率不低于50%。

## Steps

### Step 1 — 锁定横截面宽度因果测试

**File:** `/Users/didi/Project/quant/tests/research/test_monthly_low_zone_confirmation.py`

新增：

- `test_breadth_uses_only_same_date_liquid_cross_section`
- `test_future_cross_section_rows_do_not_change_prior_breadth`
- `test_breadth_confirmation_waits_until_positive_share_reaches_55pct`

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone_confirmation.py` → `11 passed`。

### Step 2 — 扩展宽度特征与六条第二轮规则

**File:** `/Users/didi/Project/quant/src/quant/research/monthly_low_zone_confirmation.py`

新增公开函数：

```python
def build_market_breadth_features(
    daily_features: pd.DataFrame,
    config: MonthlyConfirmationConfig,
) -> pd.DataFrame:
    """Return same-date liquid-universe median return and positive share."""
```

配置新增 `minimum_breadth_constituents=500` 与 `minimum_breadth_positive_share=0.55`；宽度规则仍只取窗口内第一次满足日。

### Step 3 — 点时存活消融与第二轮报告

**File:** `/Users/didi/Project/quant/scripts/research/backtest_monthly_low_zone_confirmation.py`

CLI 新增 `--enable-breadth-rules`、`--enable-survival-ablation` 和 `--raw-dir data/raw`。第二轮输出目录固定为：

```text
/Users/didi/Project/quant/reports/research/monthly_low_zone_confirmation_breadth/
```

报告必须分列2017年度宽度规则胜率、宽度未确认后避免大亏/错过大涨数量，以及基本面存活对每期确认保留率的增量。

### Step 4 — 验证

```bash
PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone_confirmation.py tests/research/test_monthly_low_zone.py tests/research/test_blood_chip_deep_base.py
PYTHONPATH=src python scripts/research/backtest_monthly_low_zone_confirmation.py --enable-breadth-rules --enable-survival-ablation --output-dir reports/research/monthly_low_zone_confirmation_breadth
git diff --check
```

## Rollback

本轮不修改线上策略、数据库和第一轮产物。撤回时只删除第二轮被忽略的报告目录，并回退本计划明确新增的宽度规则代码；不得删除第一轮代码、测试或用户文件。

## Commit checkpoint

本轮不自动提交。若结构通过并经用户确认，建议提交：

```text
feat(research): add breadth confirmation to monthly low-zone anchors
```
