# 自选池 Kronos 次日多任务与信息增益验证

## 范围与预条件

只读当前 `/Users/didi/Project/quant/data/research/similar_patterns/watchlist.json` 的全部90只股票，冻结这次池快照。使用 `/Users/didi/Project/quant/data/raw/daily_partitioned` 的统一原始行情，样本截止2026-09-11。不写生产因子、策略、页面或旧模型。

模型权重与环境复用 `/Users/didi/Project/quant/configs/kronos_byd_walk_forward.yaml`。执行前验证原始模型哈希、日期排序、股票覆盖和价格合法性。

## 固定实验合同

1. `/Users/didi/Project/quant/src/quant/research/kronos_watchlist_multitask.py`：90日历史生成冻结Kronos最后位置表示，每个有效股票/交易日一条样本，不跨股票拼接。输入使用原始OHLCVA，归一化只看历史90日。回归监督次日真实原始四价相对今日收盘的对数变化；分桶监督以次日交易所pre_close为基准的经济涨跌幅，避免除权机械价差被算成方向机会。记录两个口径，不把预测价差当策略收益。
2. 复用项目标准145列因子计算器和daily_basic加载器，按预测日对齐；因子选择、均值、标准差只在训练区间拟合，缺失使用训练均值并附带缺失标记。794条注册字段生成逐项准入清单，本轮不回填实时模型评分、策略身份或缺乏历史发布时点的字段。
3. 最后历史表示与目标交易日的月、日、星期、距预测日天数拼接。目标日来自交易所日历；不把历史最后一天的时间戳替换为未来时间，不提供未来OHLC。日期只能表达季节性与休市间隔，本轮不声称已加入公司公告日历。
4. 六组同样本对照：A纯价格回归；B价格+开收分桶；C=B+目标日期；D=B+因子；E=B+因子+目标日期；F=E去掉Kronos表示。分桶边界为[-3,-1,-0.1,0.1,1,3]%，7类；loss为四价Smooth L1均值+0.3×两个分桶交叉熵均值。价格输出保持OHLC关系。A的未训练分类头不用于概率评价。
5. 2024/2025/2026三折滚动，训练区间沿用原实验。各验证年前半年用于选epoch，第三季度做温度校准，第四季度选择置信阈值；各测试年不参与任何选择。两个随机种子分别20260912/20260913，报告每个种子及平均概率；种子不视作独立市场样本。
6. 高置信事件为经济口径次日开盘或收盘涨幅至少1%、跌幅超过1%。阈值固定候选0.5/0.6/0.7/0.8/0.9/0.95；只允许Q4验证数据选阈值。候选准入至少50条信号、20个交易日、命中率80%、五交易日块bootstrap的90%下限70%、相对基础发生率提升1.2倍。测试报告覆盖率、命中、时间块区间、股票/日期集中度和跨折情况；允许没有合格信号。另报告固定阈值曲线，禁止事后最优阈值冒充确认性结果。

## 文件、验证与执行顺序

- 先写 `/Users/didi/Project/quant/tests/test_kronos_watchlist_multitask.py`，验证日期编码不读取价格、分桶边界、价格与辅助loss、训练期标准化、折内标签不交叉、非训练分类头不可报告概率、未来数据追加不重写因子、OHLC约束及checkpoint重载。
- 写独立配置 `/Users/didi/Project/quant/configs/kronos_watchlist_multitask.yaml` 和入口 `/Users/didi/Project/quant/scripts/research/run_kronos_watchlist_multitask.py`。冻结表示保存在独立实验目录，带数据和模型指纹；缓存完整性验证失败即拒绝复用。
- 执行以下检查，预期全部通过：

```bash
PYTHONPATH=src models/research/kronos/venv/bin/python -m pytest tests/test_kronos_watchlist_multitask.py tests/test_kronos_next_day_prices.py tests/test_kronos_walk_forward.py -q
models/research/kronos/venv/bin/ruff check src/quant/research/kronos_watchlist_multitask.py scripts/research/run_kronos_watchlist_multitask.py tests/test_kronos_watchlist_multitask.py
```

- 使用本机Metal GPU执行下列命令，预期完成90股样本审计、六组×两种子×三折的训练、概率校准及阈值锁定评估，保存模型、指标、抽样清单、因子准入表及报告：

```bash
PYTHONPATH=src models/research/kronos/venv/bin/python scripts/research/run_kronos_watchlist_multitask.py
```

## 限制与恢复

当前自选池回溯历史存在事后选股偏差；预训练历史重合和已有研究看过测试区间的风险保留。预测时点为收盘后且当日行情/daily_basic已可取得，不是收盘前可下单信号。90股同日相关，不能把行数当独立样本量；高命中不等于可执行收益。真实财报预约发布日期、复权公司行动日历、剩余因子和全参数微调仅在有合法历史可知数据及首轮证据后扩展，不混入当前对照。

所有新增产物放在独立目录，失败可重新运行未完成步骤；已完成实验不覆盖。无数据库迁移，无需破坏式回滚。本轮不自动提交Git。
