# BYD 收益优先、约束风险的优化

## 目标和前置条件

研究年化净收益12%-15%、最大回撤20%-25%的可行性，不承诺达到目标。数据沿用 `/Users/didi/Project/quant/reports/research/byd_mwu_20261012/market_snapshot.parquet`，覆盖2011-06-30至2026-10-12；交易从2012-08-10开始。旧数据已观察，不称新盲测。复用历史税费、冲击、容量、整手和现金约束。

## 文件与步骤

1. `/Users/didi/Project/quant/src/quant/research/byd_growth.py`：建立保留底仓的连续趋势概率策略、短期反转及复苏策略；旧规则和成熟标签模型保留，并设置1.2/1.5倍目标仓位版本，最高100%，不融资。每个变换后的专家独立重算扣费反馈。无单日亏损线性惩罚，无固定短冷静期反复清仓。
2. `/Users/didi/Project/quant/scripts/research/run_byd_growth.py`：在读出结果前保存完整候选清单。分三轮固定集合：直接仓位策略，在线融合，波动上限与低换手控制。每轮2016起以前三个完整年度选择，优先训练期回撤<=25%、波动<=25%、平均仓位>=25%的候选，再按年化及年度稳定性排序；无可行候选时标注违规并选择超限最少候选，不声称约束保证。
3. 三轮连续账户按同一预设初始策略重放，基础成本冻结年度路由后再做严重和极端磨损、1000万元容量、3/5年滚动窗口及配对20日区块重采样。比较固定1/3、40%、50%、60%仓位及旧网格流程；不使用回测期最佳候选替代逐年选型流程。
4. `/Users/didi/Project/quant/tests/test_byd_growth.py`：检验专家前缀不变、概率与目标仓位边界、变换保留现金、约束可行性选择、未来年度收益不能影响历史选型。运行 `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -p test_byd_growth.py -v`，预期全部通过。
5. 运行 `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/research/run_byd_growth.py`。输出 `/Users/didi/Project/quant/reports/research/byd_growth_20261013/`：协议、候选指标、年度选型、真实账本、阶段指标、滚动窗口、压力测试、置信区间、来源哈希与报告。

## 停止与回滚

完成三轮预先固定集合即停止。若后两轮按相同外层评分未改善，明确报告停止理由；不追加看结果后的参数直到达标。所有改动为新增研究文件，不改交易生产配置，无发布或下单。回滚仅移除本轮新增文件，旧研究保留。
