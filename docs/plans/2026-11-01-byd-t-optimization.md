# BYD 做T优化与验证

目标：统一盘前计划与回测成交规则，以扣费后收益、回撤和按季度前推验证选择500股做T规则，保留其他工作区修改。

前提：使用 `/Users/didi/miniforge3/bin/python`；正式5分钟训练数据须通过 `read_byd_training_frame` 校验，日线由现有 MarketDataStore 读取。保留原始数据，不修改60天数据刷新契约。

1. 修正 `/Users/didi/Project/quant/src/quant/research/byd_daily_t_plan.py` 的模拟：统一14:30停止开仓、14:50定时退出；目标价格按盘前固定价计算；保留买入条内不计算止盈；加入可选止损、跳空与同条双触发保守成交，以及空样本返回。记录成本、时间、最差交易与累计已结算做T盈亏回撤。
2. 将同一文件的模型改为按季度前推：每个季度只使用前季度末及以前的已完成样本；收益标签使用成交额归一的基点，训练/验证/2026监测分别报告；实盘使用与前推相同的训练截止规则。日线不足、无样本、最新特征缺失时明确停用。
3. 在 `/Users/didi/Project/quant/scripts/research/optimize_byd_daily_t.py` 比较有限候选：入场偏离0.6%/0.8%/1.0%，目标0.8%/1.0%/1.2%，无止损/2%止损；质量门槛10/20基点。仅按2023开发与2024—2025验证选择；2026用于冻结后监测，另加双倍滑点压力测试。增加深度1/2的4组简单模型对照，共40组。保存逐笔、候选、汇总和最新计划，避免覆盖旧报告。
4. 在 `/Users/didi/Project/quant/src/quant/application/workspaces/byd.py` 提升计划缓存版本，保留已有训练数据读取修改；等待文案区分规则失败、当日评分失败和持仓容量不足。
5. 在 `/Users/didi/Project/quant/tests/test_byd_daily_t_plan.py` 验证尾盘截止、固定目标、止损先行、空样本、季度训练截止和最新特征缺失；在 `/Users/didi/Project/quant/tests/test_byd_workspace.py` 验证新版本及等待文案。用 `PYTHONPATH=src python -m pytest -q tests/test_byd_daily_t_plan.py tests/test_byd_workspace.py tests/test_byd_t_backtest.py tests/test_byd_minute_t.py` 验证，预期全部通过。
6. 更新 `/Users/didi/Project/quant/docs/strategies/byd_002594_minute_t_strategy.md`，写清生效规则、真实数据截止、选择/监测分离、底仓及现金前提和验证限制。使用依赖注入生成10月30日新计划，验证JSON序列化和现有页面字段契约。

如候选未满足验证或2026监测标准，更新策略为暂停新开仓并保留研究结果，不通过放宽标准发布。未授权自动下单，本次只更新规则、计划及验证产物。无数据库迁移，不提交其他修改。

## 完成记录

40组候选无合格推广结果；旧模型算法在修正成交标签后重新拟合的对照也未通过原验证盈利因子要求。因此默认参数仅保留研究，关闭新开正T仓。相关117项测试通过；完整workspace JSON无NaN、缓存版本5、停用理由验证通过。页面更新使用单BYD发布流程，执行本轮preflight身份记录和postflight严格新鲜度审计；通用刷新流程的既有composition错误未修改。
