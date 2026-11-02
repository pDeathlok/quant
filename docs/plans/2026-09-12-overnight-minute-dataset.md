# 真正收盘前可见的分钟训练样本

## 目标与边界

冻结90股，使用14:55决策时保守已完成的14:50及以前5分钟数据，加严格早于当日的145列项目因子。买入参考价为当日实际收盘，仅用于标签；预测时不能读取它。标签为次日每5分钟检查收盘价、触发后下一根开盘代理退出，以及预先约定的日终退出。行情不一致、公司行动、未知退出单列，不据其事后筛掉信号。

BaoStock下载未完成前仅做smoke数据衔接检查，必须在manifest标partial，不进行全90股效果宣称。当前自选池的历史选择偏差保留披露。

## 可复现步骤

工作目录：`/Users/didi/Project/quant`。

1. `test -f /Users/didi/Project/quant/reports/research/kronos_watchlist_multitask_v1/watchlist_snapshot.json`应返回0。分钟源仅读取完成的YYYY.parquet，不读取pending文件。
2. `/Users/didi/miniforge3/bin/python scripts/research/fetch_overnight_watchlist_limits.py`补全冻结90股2024-01-01至2026-09-11准确限价，独立目录`/Users/didi/Project/quant/reports/research/overnight_watchlist_limits_2024_2026`；不复用旧短期候选池的不完整交集，不修改正式原始库。
3. `PYTHONPATH=src /Users/didi/miniforge3/bin/python scripts/research/build_overnight_minute_dataset.py --smoke --output /Users/didi/Project/quant/reports/research/overnight_minute_dataset_smoke_v1`仅对已完整下载三年分块的股票做数据衔接。保存逐股结果、源哈希、输入不可用记录和标签状态；不启动模型效果测试。
4. `PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest -q tests/test_overnight_minutes.py`应通过新增的严格lag测试、未来K线变动不影响输入、缺失数据不编造、下一根才执行及分钟内不确定性测试。
5. 所有90股源数据完成后，去掉--smoke，输出到独立`/Users/didi/Project/quant/reports/research/overnight_minute_dataset_v1`，缺完整下载标记则拒绝正式构建。之后再预注册模型比较及验证门槛。

## 时点合同

- 决策记录首先由分钟输入和此前因子生成；不能用次日是否交易决定当日是否有样本。
- 因子从正式原始数据重新计算，不从按未来标签筛过的Kronos训练缓存提取。
- 准确上限与当前快照价格可用于预先门控；最终收盘是否涨停只能作为后续执行风险标记，不能替代快照门控。
- 因子缺失保持NaN，模型阶段只在训练分区拟合处理参数。
- 标签的可信状态与净收益代理分开保存。没有可信标签不等于没有发生交易。

无迁移、删除或生产配置更改，无需回滚。新产物与失败实验一并保留；不自动提交工作区其他改动。
