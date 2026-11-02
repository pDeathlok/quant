# 隔夜机会持续研究

## 用户最新输入口径（优先于下面旧阶段的分钟路线）

用户明确不需要分钟线，使用收盘数据研究；实际应用时使用14:30数据近似。因此后续训练与回测以截至当日收盘的日线及由其计算的特征为输入，预测下一交易日价格/固定退出结果，不继续新增分钟数据或分钟路径特征，也不把额外分钟权限作为推进条件。不能误退化为全部只用T-1日线的模型。

实盘以14:30快照近似当天日线；训练完整日线与实盘未完成日线存在输入差异，特别是当日收盘参考价、最高/最低和累计量额。该近似已由用户指定，研究需明确标注，不把收盘数据回测宣称为14:30已经可观测的无误差实盘验证。未来信息检查仍禁止使用下一日结果做买入选择，并保留按日期划分、锁定规则、真实命中率、成本和执行风险审查。

新增分钟路径构建已按要求中断（exit130），`reports/research/overnight_intraday_path_dataset_v1/STOPPED_BY_USER.md`禁止自动恢复。既有分钟研究和旧产物保留，不删除、不用于替代用户指定的日线主线。

目标：收盘参考价买入、下一交易日按预先锁定规则卖出，寻找事件命中率达到80%或90%的股票子集，并验证净收益及执行条件。研究目标保持活动状态，不将有限样本中的高命中率视为完成。

## 第一轮：复核已有短期模型

最新完成：`overnight_current_full_factors_v1`完整145列、1246611键重建正常结束；`overnight_same_day_rebuilt_limit_gated_v1`同169输入名重训完成。42条样本支持规则40bps均为负，最好0.5%止盈命中128/156=82.05%但平均净收益20bps为-0.3762%、40bps为-0.5762%；亏损单平均-3.5578%，尾部损失仍抵消多数小盈利。1%目标最好42/59=71.19%，20bps为-0.3146%。未达标。

执行缺口已补：旧限价缓存仅覆盖买入日候选，另从Tushare补齐全Q4候选125087个卖出日键、59日、缺失0，独立保存在`overnight_exit_limit_history_2025q4_v1`。`overnight_close_exit_limit_audit_v2/v3`分别核查旧/新模型固定Top1/Top5，保留跌停收盘回退亏损，不用未来执行标记删除样本。后续新策略应继续标注这些成交不确定性。仍只使用日线，不恢复分钟路线。

当前日线因子补全（2026-09-12续）：旧当日因子只从策略事件缓存连接，训练段候选覆盖约48.9%，并非所有候选日期都有计算。先用`build_overnight_current_market_factors.py`对全部5542只、1246611个候选键重算110项量价因子；再用`build_overnight_current_basic_factors.py`接入35项daily_basic及派生因子。后者从共享`derive_daily_basic_features`复用原公式，向历史回读滚动预热（保留pct_change默认历史填充语义所需的前置市值），保持缺失键和缺失值。不能将有行误称为每个变量均非空。

补全产物必须验证候选及输入哈希、键唯一性、覆盖率后，才运行`audit_overnight_lag_only.py --same-day-features --limit-targets --exact-entry-gate --rebuilt-current-factors`，新目录`overnight_same_day_rebuilt_limit_gated_v1`。保持前轮模型超参、169个特征名、日期切分及止盈规则不变，只改变当日因子来源。抽样388个重叠键发现绝对价格尺度与累计OBV起点不同，故本轮是重建来源整体对照，并非纯缺失覆盖消融；不能把效果变化全归于补齐缺失。Q4仍为重复使用的探索验证，不是独立确认，补全及测试通过不代表命中率达标。

只读来源：`/Users/didi/Project/quant/reports/research/holding_event_probability_v2/refit_probabilities.parquet` 与 `/Users/didi/Project/quant/reports/research/tail_overnight/dataset.parquet`。

固定原有条件：预测high_up1概率≥0.8，当日涨跌幅绝对值<0.095；不再搜索此轮阈值。退出规则为开盘超过1%则开盘退出，否则盘中超过1%按1%退出，未触及则收盘退出。另做不给开盘超额收益、触及需超过5/10bps的敏感性分析。成本固定比较20和40bps。所有成交均标为日线价格近似。

验证重点：缺失结果必须保留；公司行动异常不得当成未交易；原有4.8%执行代理只能报告，不能冒充真实涨跌停；按日期计算集中度与日等权收益，单笔均值单独展示。2026H2已经被观察，只能作探索，不升级为独立确认。

执行：`PYTHONPATH=src /Users/didi/miniforge3/bin/python scripts/research/audit_overnight_opportunity.py`。预期输出新目录内协议、逐笔结果、敏感性汇总和逐日分解。目录已存在时拒绝覆盖。

后续顺序：先审计高分集中日期及预测校准，再比较直接盈利标签、既有事件模型和Kronos；仅在历史验证窗口选规则，记录全部尝试。新数据与分钟数据可得性另行盘点。没有独立确认不宣称达标；不修改生产策略，不发出交易指令。

无需数据库迁移或回滚：第一轮仅新建研究文件，原数据和模型保持不变。阶段结束保留产物供复核，不自动提交工作区其他变更。

## 第二轮：直接盈利目标

入口：`/Users/didi/Project/quant/scripts/research/train_overnight_direct.py`，产物独立放入`/Users/didi/Project/quant/reports/research/overnight_direct_v1`。先训练2024，2025H1仅校准，2025H2锁定规则，2026只评估锁定规则；标签终点严格早于下一阶段。2026已经被研究，仍不是新的独立确认。

两个目标分别为1%止盈否则收盘，以及次日收盘退出；每个目标训练盈利分类器及净收益回归器，共4个模型。120轮、15叶、叶样本至少200、L2=30、学习率0.06、随机种子20260912，每个训练日期等权；去掉池身份输入。概率门槛[0.5,0.6,0.7,0.8,0.85,0.9,0.95]，最低净收益[0,0.002,0.005]，每日最多[1,3,5]只。锁定规则要求验证期至少50条、20日期、无未知标签、盈利率≥80%且扣成本日均值减两倍标准误为正，然后选覆盖最高者。不合格则不在确认集另选参数。

验证：`PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest -q tests/test_overnight_opportunity_audit.py tests/test_tail_overnight_selection.py`，预期9项通过。新增测试检查开盘改善、成交穿透敏感性、未知退出保留、按日仓位限制及选股不读取未来标签。

## 第三轮：真实涨停买入门控

已对第二轮107条候选完成Tushare准确限价检查。39条收于涨停；剩余68条盈利率77.94%，原始83.96%的表面达标不能作为停止依据。

`/Users/didi/Project/quant/scripts/research/fetch_overnight_limit_history.py`按既有648个历史候选日期获取准确限价，仅缓存到研究目录。若单次达到官方5800行上限，对未覆盖候选逐只补查，避免无声截断。零返回保留覆盖缺口，不造值。

完成后运行：`PYTHONPATH=src /Users/didi/miniforge3/bin/python scripts/research/train_overnight_direct.py --exact-limits --output /Users/didi/Project/quant/reports/research/overnight_direct_v2`。算法及选规则网格保持第二轮相同，先按当日准确上限剔除收于涨停及限价未知的样本，再训练/校准/选股。下一日未知结果仍保留。该门控只解决日线近似的一个问题，不声称尾盘完整数据已经时点可得。

新增精确上限与缺失门控测试后，同一验证命令预期10项通过。
