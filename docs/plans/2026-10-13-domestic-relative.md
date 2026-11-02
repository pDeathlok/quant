# 相对六篮子等权的收益/回撤双目标

用户放宽实际仓位20%约束，允许少量价格漂移。六篮子定义仍为BYD、宽基（300+500）、黄金、国债、证券、红利，目标各0—20%，现金豁免。主要目标改为同区间净CAGR高于等权且最大回撤幅度小于等权，不再强制15%/20%绝对阈值。

新增 `/Users/didi/Project/quant/src/quant/research/domestic_relative.py`：确定性20%有界水位分配（保留不可投资的预算为现金）、已有63/120/250日特征软倾斜（幅度3/6个百分点）、Ridge21/Ridge超额21/树超额21的3个百分点倾斜、过去可见CSI300弱势（120日均线偏离<-3%且20日收益<-6%）将权益仓位乘0.7或0.4，剩余资金先补金债至20%后留现金、16%/18%波动上限。固定对照为等权、每次排除一个篮子的六个五篮子配置及低证券5%的固定核心。所有目标≤20%，不再每日紧急修复实际20%漂移。

新增 `/Users/didi/Project/quant/scripts/research/run_domestic_relative.py`，候选为直接专家与share/context/multiwindow（eta0.1/0.3）融合。动态候选每21交易日检查、3个百分点死区；主基准也采用同月度规则，另保留5交易日/2个百分点的宽松等权，以避免把执行变化当选资产alpha。2014—2017初始化固定为120日3点软倾斜加0.7弱市权益缩放；2018起只用此前3完整年对同月度等权比较，存在收益更高且回撤更低者才选择，否则退回月度等权。训练排序为CAGR增量+0.25*回撤降低量，基础路线在severe/extreme/1000万元之前冻结。固定历史已观察，不声称盲测。

专项文件 `/Users/didi/Project/quant/tests/test_domestic_relative.py` 验证水位分配、宽基合计上限、现金预算、不含未来的独立前缀、相对选型失败回退。执行 `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -p 'test_domestic*.py' -v` 预期所有专项通过，再运行同环境 `python3 scripts/research/run_domestic_relative.py`。输出 `/Users/didi/Project/quant/reports/research/domestic_relative_20261013/` 原始专家/候选、年度分数、路线、4种成本容量账本、相对双优标记、滚动窗口、配对重采样、平均资产权重、来源哈希和报告；同时检验2014完整期与2018起，不能用某一近期区间或事后最佳代替动态流程。

达到本轮固定集合完成条件后停止；没找到双优也据实记录，不循环扩参到历史必胜。仅新增研究文件，不改旧模块、生产策略、原报告或下单。不提交代码；撤销只删除本轮三个新文件、计划与报告目录。
