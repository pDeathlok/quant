# 境内多资产及模型融合研究

目标仍为扣费年化>15%、最大回撤<20%，是否达到由历史验证判断。资产范围为BYD、沪深300ETF510300、中证500ETF510500、黄金ETF518880、国债ETF511010、证券ETF512880、红利ETF510880，加现金；不包含海外资产、不融资。

使用基金真实上市后的OHLC、成交量、复权因子及分红，股票复用已冻结快照。日历逐日对齐，缺报价只能用过去收盘估值、不可成交；证券ETF上市前不能持有。共享现金和真实份额，卖出先于买入，100份整手，费用至少5元，ETF不套用股票印花税；模型与信号复权收益不代替真实成交价格。

新增 `/Users/didi/Project/quant/src/quant/research/domestic_portfolio.py` 与 `/Users/didi/Project/quant/scripts/research/run_domestic_portfolio.py`。复用既有特征定义：2/5/20/63/120/250日收益和均线偏离、20/63日波动、20日价格区间、252日回撤、成交量比、相对沪深300强弱、基准趋势。新模型的label为下一开盘至5/21/63报价日后的收益，并比较21日收益减下行损失的效用label。每21交易日滚动拟合，成熟标签结束日<=拟合日；只用过去1260日；线性Ridge与小型梯度提升模型固定超参，不全历史搜索。

专家为静态股/金/债配置、多窗口绝对/相对动量、逆波动风险预算、低换手股金债轮动、模型配置、现金。所有专家先由真实成本账本产生反馈；MWU/share/multiwindow/context在资产权重向量层融合。每年配置仅按此前3完整年评价，完整候选和模型超参在结果前冻结。

输出 `/Users/didi/Project/quant/reports/research/domestic_portfolio_20261013/`，包括数据覆盖/缺失、公司行为、模型成熟标签审计、候选参数、逐年配置、共享账本、订单、压力成本、滚动及配对区块结果、图表和报告。基准为固定配置及单资产买持。基准和组合采用同一时间区间与费用规则。真实成交范围内最长历史和全资产成熟后的共同区间分别报告，不拼接上市前指数冒充交易回测。

验证命令：`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -p test_domestic_portfolio.py -v`，预期通过共享现金、整数成交、无上市前持仓、ETF税费、公司行为、前缀/标签成熟、权重上限约束测试。运行 `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/research/run_domestic_portfolio.py` 完成预设集合后停止，无结果驱动扩参，无实盘下单。新增研究文件可独立删除，不改变已有生产配置。
