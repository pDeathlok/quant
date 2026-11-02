# 尾盘选股模型第二轮

用户已明确接受以收盘价及全天量价代理 14:30 截面。本轮直接按这一统一假设评估，不以缺少真实 14:30 数据阻止训练。

使用已生成的历史并集数据；T-1 因子继续严格滞后。当日候选截面的涨跌幅、成交额、换手率和波动率排名作为新特征，不宣称候选池统计是全市场统计。

目标为次日开盘净收益、次日收盘净收益、次日触及 +1%/+2% 后退出否则收盘退出的净收益，以及次日触及 -2% 的分类风险。冲高退出若开盘已经高于止盈线按开盘计；无需假设止盈与止损的先后。成交仍按研究价格口径，不接券商。

2024 年拟合；2025 年上半年校准风险概率；2025 年下半年选择目标、股票池、Top-K、最低预测净收益和风险门槛；2026 年只在方案选定后做一次本轮评估。2026 年已在第一轮研究中看过，因此本轮不宣称完全未触碰的盲测。

搜索允许空仓，以固定 Top-K 槽位计现金权重。验证选择必须至少覆盖 40 个交易日和 100 条候选，使用验证期日收益均值减去两倍日均值标准误作为排序准则；不根据测试结果改参数。所有试验保留，不保证找到盈利策略。

新增 `/Users/didi/Project/quant/src/quant/research/tail_overnight_selection.py`、`/Users/didi/Project/quant/scripts/research/train_tail_overnight_selection.py` 和 `/Users/didi/Project/quant/tests/test_tail_overnight_selection.py`。输出第二轮模型、全部验证搜索、锁定方案、2026 评估和最新可用截面的候选名单至 `reports/research/tail_overnight_v2`。

验证：`PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest -q tests/test_tail_overnight.py tests/test_tail_overnight_selection.py`；新增测试覆盖跳空止盈、未触及止盈、缺失标签、现金槽位与空仓、严格选股门槛。只更改本研究文件，不提交已有工作区修改。
