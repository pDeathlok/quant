# Kronos近三个月月度/周度滚动更新验证

根目录：`/Users/didi/Project/quant`。复用 `reports/research/kronos_multiscale_expanded_v2` 的冻结编码器、embedding、候选和标签，研究更新频率，不重新预训练编码器或改生产。

固定2026年原匹配样本作为已观察的时间外复核，不将其称作全新封存测试。每月/每周第一个可用信号交易日开盘前重训，训练信号日期在该日之前三个自然月内，且exit_date必须严格早于重训日。5日标签导致窗口尾部约6个交易日不可用。

对照：原模型、原冻结序列、原冻结融合；月/周滚动融合（冻结子分，融合层用整个三个月成熟样本）；月/周滚动序列（MLP用整个三个月成熟样本）；月/周联合滚动融合（前两个月训练辅助head，最后一个月用辅助head生成时间外子分训练融合层，然后最终head用三个月全部成熟标签重训）；联合滚动另输出允许空仓版本。

所有新增融合统一使用当日原分数分位数及3个序列子分；标明它与旧冻结融合原始分数输入不同。每次从固定随机种子初始化MLP，20轮，不根据2026调参。融合用标准化+LogisticRegression(C=.1)，仍用5日净盈利标签。空仓阈值只在该次重训的末月辅助head时间外预测上选择：候选阈值0.50至0.80，样本至少30、胜率>=60%、Wilson下限>=50%、平均净收益>0且PF>1，否则现金。

实现文件：`/Users/didi/Project/quant/src/quant/research/kronos_rolling.py`（时间窗口与重训）；`/Users/didi/Project/quant/scripts/research/run_kronos_rolling.py`（预测、同口径资金回测与报告）；`/Users/didi/Project/quant/tests/test_kronos_rolling.py`（周/月边界、成熟标签、末月stacking purge、未来标签篡改不影响旧窗口、样本不足现金）。

先执行 `PYTHONPATH=src .venv-kronos/bin/python -m pytest tests/test_kronos_rolling.py -q`，要求全通过。然后执行 `PYTHONPATH=src .venv-kronos/bin/python scripts/research/run_kronos_rolling.py --source reports/research/kronos_multiscale_expanded_v2 --output reports/research/kronos_rolling_quarter_v1`，要求产生重训审计、逐事件预测、每期样本数、同口径逐日资金曲线、成交记录、AUC/IC/胜率/净收益/回撤/敞口/空仓比例及配对区块区间。重建行情后先核对原baseline曲线，要求绝对差小于1e-8。

研究产物写入新目录，拒绝覆盖。保存参数、源数据及代码哈希、每次模型权重；无自动提交或生产发布。回滚不涉及现有模型，只需停止本研究进程，保留产物。

## 执行完成

输出到 `/Users/didi/Project/quant/reports/research/kronos_rolling_quarter_v1`。每侧月更8次、周更31次；成熟训练样本中位数左侧约1220/1236条、右侧2513/2496条。39次更新均使用成熟标签；现金选择在末月内额外做有purge的时间切分，保持该次holdout对应gate模型。

近三个月月/周更新均未稳定改善；纯序列两侧周更均比月更更差。联合融合周更比月更好，但仍不及旧冻结融合。训练内AUC中位数左侧0.62、右侧0.59，紧随预测期接近随机。结论不支持单靠提高更新频率解决当前表征与泛化问题。13项测试与代码检查通过，原baseline账户复现误差<1e-8。保留报告、每次权重、预测和重训审计，没有发布生产。
