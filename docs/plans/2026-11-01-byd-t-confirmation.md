# BYD做T第二轮：确认入场与风险预算

目标：按用户确认的“控制回撤、扣费后稳定增益”，检验无需机器学习的盘中确认规则。保留上一轮失败结果、门槛和现有停用状态。

1. 新增 `/Users/didi/Project/quant/src/quant/research/byd_t_confirmation.py`：复用已发布5分钟训练数据和 `prepare_byd_intraday_bars` 的滞后特征；比较前收盘限价与VWAP下方反转确认两类入场；确认后下一根开盘成交，禁止跨午休、跨日挪用开盘价；每日最多一笔，当天14:50退出；ATR止盈止损、同条止损优先、开仓条不计止盈。每笔研究风险预算200元，股数100股取整、最多500股、现金上限50000元，真实费用参与股数计算。记录5分钟收盘盯市回撤和跳空超预算损失。
2. 新增 `/Users/didi/Project/quant/tests/test_byd_t_confirmation.py`：验证下一条成交、不可用午休未来开盘、止损先行、尾盘退出、现金与风险股数约束、未来K线不改历史信号、初始亏损计入回撤、缺少收益证据时不能合格。
3. 新增 `/Users/didi/Project/quant/scripts/research/optimize_byd_confirmation_t.py`：固定64组候选（两类入场、0.4/0.6 ATR偏离、0.4/0.6 ATR目标、0.3/0.5 ATR止损、11:00/14:00截止、两类滞后环境），先只用2025年底以前数据。保留2023开发至少40笔、2024—2025验证至少25笔及60%胜率、1.2盈利因子标准，增加验证各年盈利、邻近参数和双倍滑点检查；2026仅在冻结候选后监测。没有合格候选则不启用。
4. 结果保存 `/Users/didi/Project/quant/reports/byd_002594/t_confirmation_20261101/`，包含候选、逐笔、盯市曲线、按年结果和与上一轮的可比说明；更新 `/Users/didi/Project/quant/docs/strategies/byd_002594_minute_t_strategy.md`。只有满足完整证据和实时数据条件才调整执行状态。

验证命令：`PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest -q tests/test_byd_t_confirmation.py tests/test_byd_t_backtest.py tests/test_byd_daily_t_plan.py tests/test_byd_workspace.py`，预期全部通过。使用独立研究文件、保留原始输入和线上快照；不回滚其他工作区修改，不自动下单。

## 根据开发/验证期证据的扩展

首轮64组正T中，直接触价候选开发/验证均未同时盈利；同时要求“反转时仍处极端偏离区”的确认候选样本过少（开发1—3笔、验证0—6笔）。完整结果保存在 `reports/byd_002594/t_confirmation_20261101/contemporaneous_confirmation/`。

据此将确认定义改为“偏离触达后15分钟内出现回升/回落，下一根成交”，保留追价上限、午休失效、风险预算和盈利门槛。增加对称反T作为另一交易方向，固定其余网格形成128组，保存到 `reports/byd_002594/t_confirmation_20261101/armed_confirmation/`。2026仍只作冻结后监测；记录重复搜索局限，不声称新增128组是独立全新试验。增加邻近参数盈利比例、月块收益区间、剔除最佳交易和确认延迟压力测试，不能以低交易数的零回撤候选冒充最优。

## 本轮完成结果

最终128组无候选在开发/验证两段同时盈利，未启用。130项相关测试通过；风险预算、资金上限、每日一笔、底仓恢复、实时页面维持停用均通过检查。已知开盘成交与盘中触价的先后逻辑已分开验证。结果与失败回放均已归档，不修改线上生产开仓规则。
