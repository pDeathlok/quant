# 尾盘买入、下一交易日卖出：研究实现方案

目标：独立建立 tail_overnight 研究模块，复用项目 145 个统一因子作为 T-1 输入，保留四类股票池来源，支持收盘代理与真实盘中快照两种数据口径。

## 数据和执行合同

- T 表示买入交易日；T+1 是全市场下一交易日。不得使用单只股票下一条记录替代下一交易日。
- 股票池取好股票、短线、长线、自选四者并集；多重归属保留。日频池及因子严格在 T-1 对齐，自选当前名单仅用于当前截面。
- 历史好股票标准复用现有 `_long_good_stock_assessment`。页面短线/长线存在展示截断，不能把页面显示条数当作全池。
- 收盘代理模式允许 T 收盘价及全天量价用于探索，输出明确标为不可直接等同 14:30 实盘。盘中模式必须使用实际时间戳、统一单位的快照，禁止从当天最终 OHLC 反推 14:30。
- 14:30 已交易 210 分钟，线性全天量估计为累计量乘 240/210；这是基线假设。后续用截至 T-1 的同股/行业日内成交曲线估计，报告估计误差。
- T-1 因子保持原始注册名并加 lag1 前缀，禁止自动把所有数值列当特征；仅训练集决定缺失值填充及特征准入。
- 标签：次日开盘/收盘收益、高点 MFE、低点 MAE；上涨 0.5%/1%/2%/3% 与下跌 1%/2%/3% 阈值；收益区间及分位回归。
- 高低点仅作标签，不作理想卖出收益。日线无法判断止盈止损先后。首版以次日开盘卖出为基线，次日无法成交标为未解决退出，不能从结果中无声删除或宣称组合收益。
- 交易成本情景 10/20/30 bps，不宣称是所有券商真实费率。涨跌停用精确限价优先，缺失时披露保守价格代理。除权隔夜样本先标记隔离，不直接将原始价格跳变当收益。

## 实施顺序

1. 在 `/Users/didi/Project/quant/tests/test_tail_overnight.py` 写防泄漏、缺失交易日、快照时点、双池去重、成交限制、成本、缺失标签的合同测试。
2. 在 `/Users/didi/Project/quant/src/quant/research/tail_overnight.py` 实现股票池并集、严格 T-1 合并、盘中特征折算、多目标标签和时间切分。
3. 在 `/Users/didi/Project/quant/scripts/research/run_tail_overnight.py` 接入本地历史因子、日行情、最新四类池，生成审计及按时间留出的模型实验。按用户补充，使用历史左侧/右侧短线规则事件、月度长线筛选结果及日度长线底稿重建历史池；自选仅使用当前真实名单。
4. 比较纯 T-1 与 T-1 + 当日收盘代理因子。固定历史训练/验证/测试边界，清除标签终点跨越边界的样本；不得随机切分。输出标签、预测、分组覆盖和成本敏感度。
5. 将实测结果、数据不足及后续上线条件写到 `/Users/didi/Project/quant/docs/strategies/tail_overnight_strategy.md`。

验证命令：`PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest -q tests/test_tail_overnight.py`，预期全部通过；`/Users/didi/miniforge3/bin/python scripts/research/run_tail_overnight.py --help`，预期正常退出并列出可复跑参数。

本次只新增独立研究文件和研究产物；不迁移数据库，不替换现有生产模型。回退时移除本方案列明的新增文件与 `reports/research/tail_overnight` 中本次产物即可。验证后的可选提交标题：`feat(research): add causal tail overnight research pipeline`；本次不自动提交已有工作区修改。
