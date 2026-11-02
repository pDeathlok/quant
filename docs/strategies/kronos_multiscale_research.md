# Kronos 多周期量价研究

本实验使用 Kronos 的原始 TransformerBlock，重新训练连续量价输入投影和 EOS，不使用原模型来源日期不明的预训练权重。采用4层、128维、4个注意力头；这不是 Kronos-small 原版8层512维权重微调。

输入只有开高低收和成交量。日线128根、周线104根、月线60根，周期身份向量只标识频率。每个周期追加独立可训练 EOS，经过共享 Transformer 后取 EOS 向量。三个128维向量拼成384维；左右侧各自训练 MLP 输出净收益、盈利概率和下行风险子分，再与既有模型历史 OOT 分数训练逻辑回归融合层。

周/月采用上一完整自然周期，下一周期开始后才可用，保守地晚于该周期最后一个交易日。所有归一化只用当前历史窗口；未成熟的月线与周线不能提前聚合。预训练目标为各周期下一根 OHLCV，目标时间不晚于2022年底；2023只做无监督验证与checkpoint选择。2024训练子分，2025上半年训练融合及反馈规则，下半年选阈值和规则，2026做时间外复核。2026部分首轮结果已被查看，扩样结果不能再描述为完全未接触的测试。

原始日行情用当日 close/pre_close 串接连续价格指数，避免引用未来复权因子。此指数用于量价与价格收益研究，不等同于现金分红总回报账户。股票池固定为2022年12月已经出现的代码，按稳定哈希抽样；不会因后来表现或退市主动移除。缺少多周期历史和成熟标签的事件会被剔除，覆盖写入报告。

规则路径包括深度1至3的浅树蒸馏，以及用2025上半年输赢case反馈优化的盈利分类树。规则选择和序列/融合分的现金阈值都要求验证期样本量至少30、胜率至少60%、Wilson 95%下限至少50%、平均净收益为正、PF大于1。没有合格规则就持有现金，不将零交易标作高胜率。

组合以20个仓位为上限，单仓投入不超过当时净值的1/20，不重复持有同一只股票。次日收盘买入、5个交易日后收盘退出，涨停不能买，跌停或缺限价数据不能卖时延期；缺行情无法下单。费用买卖各10bps。股票可以零碎股成交；整手、滑点冲击、排队成交未模拟，实盘前需要补全。事件标签与受约束组合的收益分别报告。

执行环境：`/Users/didi/Project/quant/.venv-kronos`。测试命令：

```sh
PYTHONPATH=src .venv-kronos/bin/python -m pytest tests/test_kronos_multiscale.py -q
```

固定扩样实验命令（在项目根目录执行）：

```sh
PYTHONPATH=src .venv-kronos/bin/python scripts/research/run_kronos_multiscale.py --config configs/kronos_multiscale_expanded.yaml
PYTHONPATH=src .venv-kronos/bin/python scripts/research/summarize_kronos_multiscale.py reports/research/kronos_multiscale_expanded_v2
```

已有输出目录会拒绝覆盖；显式传 `--resume` 时核验配置、代码、行情来源哈希与事件表，再加载缓存。更换模型、样本或数据应使用新目录。训练模型、融合层、embedding键、交易明细、资金曲线、规则、反馈case、配对bootstrap区间及准入结论均保存在研究输出中。原有生产模型不变。
