# 周线方向、日线买点与近期加权研究

实验计划：`/Users/didi/Project/quant/docs/plans/2026-10-21-weighted-rolling-weekly-direction.md`。所有结果是研究回测，未改生产。

## 近期加权滚动

执行入口：`scripts/research/run_weighted_kronos.py`。复用冻结的三周期Kronos EOS，比较6/12个自然月、均匀权重/90自然日半衰期，月更MLP；加权版本增加周更轻量校准。周校准只用此前各期月度head的时间外分数和已成熟标签，不拿最新head重算历史。训练期、预测期和代码/数据来源均记录哈希。

```sh
PYTHONPATH=src .venv-kronos/bin/python scripts/research/run_weighted_kronos.py --source reports/research/kronos_multiscale_expanded_v2 --output reports/research/kronos_weighted_rolling_v1
```

该输出目录已经运行完成，重复运行请另取新目录。模型不修复冻结编码器的预训练缺陷。2025H2选择依据收益/PF，只是候选初筛，不等于通过60%胜率条件；现金策略另要求胜率及置信下限。

## 周线方向和日线买点

执行入口：`scripts/research/run_weekly_direction_research.py`。使用全部保留的1128830条历史候选，原日线候选及原始OOT分数按原样保留，额外施加周/日过滤；没有重新在所有股票上生成候选。预热行情2022—2023，研究行情2024—2026。次日收盘入场，固定5/10/20交易日持有，三种持有期使用同一个信号截止日以便比较。

完整周线的可见时间为下一自然周周一。因此周五收盘信号仍使用上一完整自然周，是保守的一天信号滞后，不使用未来整周数据。MACD参数12/26/9，柱为DIF-DEA；BOLL为20周均值±2倍标准差（ddof=0）。

周线门：DIF>DEA且DIF>0；前者且柱增加；周收盘>20周中轨且中轨高于4周前；周收盘突破上轨；周EMA10>EMA26且收盘高于EMA10、EMA10上涨；MACD多头和BOLL中轨上涨同时成立。

日线保留既有原始候选。左侧另比较5日下跌、站上MA60、缩量回踩，或MA20附近[-4%,2%]缩量并收于日振幅上半部；右侧另比较放量突破过去20日高，或阳线、收盘位置>=0.6且量比>=1。每侧54个固定组合。2024和2025H1均需足够样本、净收益/PF为正；2025H2需至少100事件、胜率>=60%、Wilson下限>=50%、PF>1.2，否则不准入且现金。探索候选与准入策略分别报告。

模型支线固定MACD+BOLL门，使用已知周/日特征和原模型当日分位数，月更浅层HistGradientBoosting，12个月窗口、90日时间衰减、每交易日总权重均衡，目标20日净盈利。阈值只在2025H2固定，2026不参与参数选择。

```sh
PYTHONPATH=src .venv-kronos/bin/python scripts/research/run_weekly_direction_research.py --output reports/research/weekly_direction_daily_entry_v1
PYTHONPATH=src .venv-kronos/bin/python scripts/research/audit_weekly_direction_research.py reports/research/weekly_direction_daily_entry_v1
```

已有数据集可通过 `--dataset /Users/didi/Project/quant/reports/research/weekly_direction_daily_entry_v1` 引用，在新输出目录复核。原始数据源、行情/标签覆盖、规则选择、逐笔成交、资金曲线、模型权重及case均保留。`source_snapshot`记录执行时源文件；之后的局部排序函数格式整理不改变选择规则。

## 限制与验证

2026此前已经观察，扩展股票只是同一时间区间的新增股票复核，不能替代独立时间段。事件标签有重叠、股票有共振，Wilson仅作初筛；多重比较未校正。组合最多20仓、同股不重复持有，涨停不能买，跌停或缺数据不能卖则延期；往返20bps。未建模整手、成交排队和冲击成本。低敞口带来的减损不等于alpha，模型与规则要在相同持有期对照。

```sh
PYTHONPATH=src .venv-kronos/bin/python -m pytest tests/test_trend_research.py tests/test_kronos_rolling.py tests/test_kronos_multiscale.py -q
```

17项因果时间边界、周线可见性、精确持有期、成熟标签、加权和资金约束测试通过。
