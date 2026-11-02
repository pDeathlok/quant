# V2：回撤30%约束下提高年化收益

用户保留V1，并要求新增收益优先版，最大回撤放宽到30%。沿用六篮子目标0—20%、现金豁免、小幅实际漂移23%修复、无杠杆/做空、现有资产和特征。V1配置及报告独立冻结，不覆盖。

新增 `/Users/didi/Project/quant/src/quant/research/domestic_return30.py`。五类核心篮子为低证券核心(20/15/20/20/5/20)、无证券(20/20/20/20/0/20)、进攻核心(20/20/20/10/10/20)、无国债(20/20/20/0/20/20)、均衡(各1/6)。宽基内部比较300单持、300/500等分、63日相对动量择一；风险预算比较18%、22%、不缩放。另比较63/120/250日正相对排名前五篮子（不以负收益自动空仓）、宽基三种分配、22%/不缩放，共18动态专家。45固定核心专家+18轮动专家+V1+等权周度比较，共65直接路径；融合share/context/multiwindow，eta0.1/0.3，共6配置。不得新增外部特征或根据本轮结果追加候选。

新增 `/Users/didi/Project/quant/scripts/research/run_domestic_return30.py`。2014—2017初始化V1，2018起只按过去3完整年最大回撤<=30%的候选最大化净CAGR；若无可行候选回V1。全年路线在压力情景前冻结。所有策略复用21交易日/3点死区与23%漂移修复；周度等权为5交易日/2点死区。基础/严重/极端/1000万元四情景重算，V1账本须与保留版本精确一致。

输出 `/Users/didi/Project/quant/reports/research/domestic_return30_v2_20261013/` 候选协议、完整账户、订单、年度评分、冻结路线、相对V1/等权改善、分年、滚动、重采样及来源SHA。提供年度选型V2和事后固定候选研究诊断两种结果，不能把全历史最佳冒充过去数据决定。若年度流程没有收益改进，据实说明；预设固定集合完成后停止，不为超过旧结果无限扩参。

专项 `/Users/didi/Project/quant/tests/test_domestic_return30.py` 验证六篮子上限、宽基择一、独立数据前缀、30%可行选型及失败退回。命令 `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -p 'test_domestic*.py' -v` 全部应通过；再执行同环境 `python3 scripts/research/run_domestic_return30.py`。仅新增研究文件和保留策略配置，不改生产/下单，不提交其他工作区改动。
