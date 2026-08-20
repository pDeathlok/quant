# 带血筹修复策略接入中长线页实施计划

**目标：** 将带血筹的事件识别、次日买入观察、模拟持仓、止损与新事件再入状态，作为独立子策略接入中长线工作区，并纳入每日刷新。

**边界：** 页面展示研究策略的模拟状态，不读取或推断用户真实持仓；卖方身份不可由量价数据证明。执行保持 T+1、次日开盘、涨跌停与交易成本约束。

## 任务 1：固定每日信号语义

- 修改 `src/quant/research/blood_chip.py`，允许在显式参数开启时保留尚无次日开盘价的最新交易日信号。
- 在 `tests/research/test_blood_chip.py` 增加“最新日待执行信号默认不泄漏、显式开启才返回”的测试。
- 验证：`pytest -q tests/research/test_blood_chip.py`

## 任务 2：生成每日中长线计划

- 新增 `src/quant/application/blood_chip_long_plan.py`，从规范日线和沪深 300 数据生成候选、策略模拟持仓、当日退出及研究证据摘要。
- 候选使用已验证逻辑：120 日涨幅不超过 50%、市场 60 日收益不低于 -15%、事件低点反弹不超过 15%、同日按 60 日残差波动率由低到高排序。
- 新增 `tests/test_blood_chip_long_plan.py`，覆盖输出契约、排序、持仓与退出映射。

## 任务 3：快照和 API

- 在 `src/quant/webapp/services.py` 增加每日快照读取/刷新服务。
- 在 `src/quant/webapp/api.py` 增加 `GET /api/long/blood-chip`。
- 在 `tests/test_webapp_api.py` 覆盖参数传递和响应。

## 任务 4：中长线界面

- 在 `web/index.html` 增加“带血筹修复”子策略按钮与独立面板。
- 在 `web/app.js` 根据子策略切换接口，渲染候选、模拟持仓、退出/再入状态和研究边界。
- 在 `web/styles.css` 补充桌面与窄屏样式。
- 在 `tests/test_web_frontend.py` 固定按钮、面板和接口路由。

## 任务 5：每日迭代链路

- 将带血筹计划加入 `src/quant/webapp/services.py` 的长线刷新任务，并在 `src/quant/application/refresh_contracts.py` 标注刷新步骤。
- 更新相应刷新测试，保证“更新本页”和每日全量更新均执行。
- 验证：`pytest -q tests/test_workspace_refresh.py tests/test_refresh_contracts.py tests/test_daily_web_refresh_entrypoint.py`

## 任务 6：综合验证

- 运行：`pytest -q tests/research/test_blood_chip.py tests/test_blood_chip_long_plan.py tests/test_webapp_api.py tests/test_web_frontend.py tests/test_workspace_refresh.py tests/test_refresh_contracts.py`
- 对前端静态契约和真实最新快照各做一次检查；若本地数据不足，明确显示错误而不伪造候选。
