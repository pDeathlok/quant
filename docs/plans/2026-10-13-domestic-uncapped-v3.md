# V3：移除集中度约束，目标净年化20%以上、回撤30%以内

用户移除每篮子20%限制，其他授权不变：现有七个标的/六类资产、允许现金、不融资不做空，最大回撤继续30%。V1/V2独立保留，不改旧参数或账本。移除V3的20%目标投影与23%执行修复，单标的可0—100%，总投资<=100%。

新增 `/Users/didi/Project/quant/src/quant/research/domestic_uncapped.py`。固定12种基准：七单资产、BYD/黄金25/75、50/50、75/25、BYD50%宽基15%黄金20%国债10%红利5%、BYD40%黄金40%国债20%。规则轮动：63/120/250日及三窗口平均收益排序、前1/2标的、无绝对趋势门槛或120日均线门槛，共16；BYD20/63/120/250日趋势、退出后黄金120日趋势正则黄金否则国债，共4；既有Ridge21/63与超额Ridge21/树21预测前1/2标的，共8。40基础专家各施加0/25%/35%波动预算，共120专家。预先固定6种share/context/multiwindow融合（eta0.1/0.3）。不新增外部特征，不重训缓存模型。

新增 `/Users/didi/Project/quant/scripts/research/run_domestic_uncapped.py`。共享资金实际整手、最低佣金、公司行为、前日成交量参与率限制、下一开盘执行；5交易日/3个百分点死区。2014—2017初始化BYD/黄金50/50加25%波动预算，2018起只在前三完整年回撤<=30%的候选中最大化净CAGR；无可行者回现金。年度路线基础固定后运行severe/extreme/1000万元。固定研究候选明确按已观察全历史基础CAGR、且所有四情景完整/2018期回撤<=30%排序；标注历史搜索，不冒充样本外。绝对达标需基础扣费CAGR>20%且回撤<=30%，稳健达标还需压力与2018期支持；没有候选时明确未达标。

专项 `/Users/didi/Project/quant/tests/test_domestic_uncapped.py` 验证可以100%单标的、没有遗留20%投影、总资金上限、上市预热、趋势退出与现金、独立截断前缀。运行 `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -p 'test_domestic*.py' -v` 预期全部通过；执行同环境 `python3 scripts/research/run_domestic_uncapped.py` 输出 `/Users/didi/Project/quant/reports/research/domestic_uncapped_v3_20261013/` 协议、候选、路线、账本、收益风险、容量、来源哈希、图表及V3研究配置。

预设集合完成后停止，不以反复扩参把历史收益凑到20%。结果同时列最佳可行、原始最高收益（可能风险超限）及年度真实选型。若缺少达标证据，不称V3目标已完成。仅新增研究文件，不替换V1/V2或生产，不下单。
