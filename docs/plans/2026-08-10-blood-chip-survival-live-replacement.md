# 带血筹生存确认递增加仓线上替换计划

## Goal

将中长线 Tab 的带血筹每日执行从 100% 一次性建仓直接替换为经回测通过的 20%/30%/50% 生存确认递增加仓，并在每日快照与页面上明确展示当前阶段、下一笔触发条件和阶段变化。

## Pre-conditions

- [x] `/Users/didi/Project/quant/reports/research/blood_chip_scale_in/report.md` 的结论为“建议进入线上灰度”，且 `increasing_survival` 在 2014–2019、2020–2022、2023–2026 三段的总收益均高于 `one_shot`。
- [x] `PYTHONPATH=src pytest -q tests/research/test_blood_chip_scale_in.py` 输出 `6 passed`。
- [x] `/Users/didi/Project/quant/src/quant/application/blood_chip_long_plan.py` 是带血筹每日计划的唯一应用层构建入口。
- [x] `/Users/didi/Project/quant/src/quant/webapp/services.py` 使用 `BLOOD_CHIP_LONG_SCHEMA_VERSION` 过滤并写入每日快照，升级版本即可停止读取旧逻辑快照。

## Steps

### Step 1 — 扩充分批回测结果的当前持仓状态

**File:** `/Users/didi/Project/quant/src/quant/research/blood_chip_scale_in.py`

在 `_ScalePosition` 中保存最后一个交易日的 `residual_return_3d`；在每日行情更新时同步该值；每笔 `end_of_data` 模拟持仓输出以下字段：

```python
{
    "entry_fill": position.entry_value / position.adjusted_units,
    "signal_close": position.signal_close,
    "stop_price": position.stop_price,
    "current_residual_return_3d": position.last_residual_return_3d,
    "next_stage_ready": position.ready_stage is not None,
    "tranches_filled": position.tranches_filled,
    "tranche_dates": "|".join(value.date().isoformat() for value in position.tranche_dates),
    "deployed_fraction": min(position.entry_value / position.target_budget, 1.0),
}
```

**Verify:** `PYTHONPATH=src pytest -q tests/research/test_blood_chip_scale_in.py` → 所有分批成交、止损优先、5/10 日确认测试通过。

### Step 2 — 替换每日计划执行引擎并升级快照契约

**File:** `/Users/didi/Project/quant/src/quant/application/blood_chip_long_plan.py`

执行器固定为：

```python
BLOOD_CHIP_SCALE_IN_POLICY = DEFAULT_SCALE_IN_POLICIES["increasing_survival"]
BLOOD_CHIP_LONG_SCHEMA_VERSION = "blood_chip_long_v2_survival_20_30_50"
```

`build_blood_chip_long_plan` 必须调用：

```python
result = run_blood_chip_scale_in_backtest(
    features,
    concrete,
    BLOOD_CHIP_BACKTEST_CONFIG,
    BLOOD_CHIP_SCALE_IN_POLICY,
    entry_start.date().isoformat(),
    asof.date().isoformat(),
)
```

候选输出 `initial_tranche_fraction=0.20`；模拟持仓输出 `tranches_filled`、`deployed_fraction`、`next_addition_fraction`、`next_trigger` 与 `next_stage_ready`；研究证据替换为生存确认方案的资金加权指标。

**Verify:** `PYTHONPATH=src pytest -q tests/test_blood_chip_long_plan.py` → 候选为 20% 首仓、持仓阶段映射和每日阶段推进断言全部通过。

### Step 3 — 展示递增加仓执行状态

**Files:**

- `/Users/didi/Project/quant/web/index.html`
- `/Users/didi/Project/quant/web/app.js`

页面固定展示：

```text
首仓 20%：信号次日开盘成交
第二段 30%：持仓至少 5 日、收盘不低于信号价 95%、三日残差收益不低于 0
第三段 50%：持仓至少 10 日、收盘不低于信号价、三日残差收益大于 0
```

持仓表列调整为“股票 / 阶段与部署 / 入场 / 持有日 / 估算收益 / 下一步 / 止损参考 / 再入次数”；每日迭代增加“加仓阶段推进”和“待次日加仓”。

**Verify:** `PYTHONPATH=src pytest -q tests/test_web_frontend.py` → 页面包含 20%/30%/50% 文案、阶段字段渲染和既有雪球链接断言。

### Step 4 — 验证 API、每日刷新与旧快照失效

**Files:**

- `/Users/didi/Project/quant/src/quant/webapp/services.py`
- `/Users/didi/Project/quant/tests/test_webapp_api.py`

不修改 API 路径；依靠 schema v2 让 `_read_blood_chip_long_snapshot` 忽略所有 v1 快照，下一次 `/long/blood-chip?refresh=true` 或每日 `long_stock_pool` 刷新生成新逻辑快照。

**Verify:**

- `PYTHONPATH=src pytest -q tests/test_blood_chip_long_plan.py tests/test_web_frontend.py tests/test_webapp_api.py` → 全部通过。
- `PYTHONPATH=src pytest -q tests/research/test_blood_chip.py tests/research/test_blood_chip_scale_in.py tests/test_blood_chip_long_plan.py tests/test_web_frontend.py tests/test_webapp_api.py` → 带血筹研究、应用、页面和 API 回归全部通过。

### Step 5 — 刷新本地服务并核验真实每日 payload

调用现有带血筹刷新 API 生成 schema v2 快照，确认：

```json
{
  "schema_version": "blood_chip_long_v2_survival_20_30_50",
  "strategy": {
    "scale_in_policy": "increasing_survival",
    "tranche_fractions": [0.2, 0.3, 0.5]
  }
}
```

**Verify:** 服务响应为 HTTP 200；页面能够读取当天 payload；旧 v1 JSON 文件可以保留，但不会被读取。

## Commit checkpoint

建议提交信息：`feat(long): replace blood-chip entries with survival scale-in`

本次不自动提交，除非用户另行要求。

## Rollback

- 将 `/Users/didi/Project/quant/src/quant/application/blood_chip_long_plan.py` 的执行器恢复为 `run_blood_chip_backtest`，并把 schema 升级为新的回滚版本，禁止重新启用已经生成的 v2 快照。
- 恢复 `/Users/didi/Project/quant/web/index.html` 与 `/Users/didi/Project/quant/web/app.js` 的旧持仓列和一次性建仓文案。
- 研究模块和回测报告保留，便于复盘；不删除任何历史快照或用户数据。
