# 月线带血筹严格确定性结构验证计划

## Goal

在已经暴露的月线低9、系统性恐慌与等待型止盈结果上，不再追求交易次数，而是检验一个先验冻结的嵌套结构是否同时满足经济逻辑、批次级稳健性和可执行性：市场发生同步深跌，个股从历史高位大幅出清且基本生存能力仍在，随后价格停止创新低，才用小仓位等待中期反弹。

## Exposure and claim boundary

- 2013—2024 的价格路径已经在前几轮被查看，本轮所有阈值都只能形成“已暴露样本上的严格研究候选”，不能称为真正样本外发现。
- 2025—2026 截止 2026-07-31 的事件仍可能未走完 504 个交易日；未完成样本只报告当前状态，不进入胜率、PF 或收益置信区间。
- 同一恐慌月内的股票高度相关。股票数不能冒充独立样本数，主要统计单位冻结为锚定月份 `month_period`，股票事件指标仅作为交易路径描述。
- 无论历史胜率多高，只要独立恐慌月份不足、收益依赖单一危机批次、阈值轻微变化即翻转，或缺少新的完成样本，均不得升级为可实盘的高确定性策略。

## Frozen primary structure

1. 锚：只使用原始 `monthly_low9`，沿用至少 36 个月历史、月成交额中位数不低于 3,000 万元、同股 12 个月冷却和无未来数据的月线计算。
2. 系统性深恐慌：锚日点时可见的流动性全 A 横截面至少 500 只；20 日收益为正的股票占比不高于 20%，且横截面 20 日收益中位数不高于 -8%。两个条件必须同时满足。
3. 个股深度出清：锚月复权收盘价相对此前历史峰值回撤至少 70%。60% 与 80% 只用于阈值邻域，不参与事后选优。
4. 生存质量：截至实际入场信号日已经公布的最近年报距信号不超过 550 天，至少有 3 年历史；近 5 年归母净利润和经营现金流为正的年份占比均不低于 80%，最近年报归母净利润与经营现金流都为正。缺失财务数据视为不通过，不用当前财务状态回填历史。
5. 止跌确认：锚后最多等待 126 个市场交易日，只有 `no_new_low_20`（连续至少 20 个交易日不再创新低）成立才允许入场。直接锚入场与 `range_mid_reclaim` 只做机制消融，不替代主规则。
6. 执行：确认后下一只股票可交易日开盘；沿用一字板拒绝、最多 5 个市场交易日延迟、停牌恢复与 60 个市场日无行情核销。20bp 往返成本，最长持有 504 个市场交易日，首次触及入场复权价上方 15% 时按目标价止盈。
7. 仓位：每个确认锚固定使用当时净值 2.5%，最多 20 个未完成锚；不加仓、不补仓、不设会利用未来路径调参的机械止损。现金和未完成持仓逐日盯市。

## Frozen ablations, not a search grid

- 市场层：旧规则 `positive_share<=30% and median<=-5%` → 新的双重深恐慌规则，用于确认增益来自更深的系统性出清。
- 个股层：锚回撤至少 60% → 70% → 80%，要求方向总体稳定；主结论固定在 70%，不能按结果换档。
- 生存层：无财务门槛 → 原 60% 持续性门槛 → 新 80% 持续性门槛，记录各层删掉的赢家和永久损失者。
- 止跌层：锚后直接买入 → `no_new_low_20` → `range_mid_reclaim`；主结论固定为 `no_new_low_20`。
- 退出层：10%/15%/20% 是原计划已经登记的目标邻域，主结论固定为 15%。

## Frozen certainty checks

### Event and cohort level

- 主结构在 2013—2016、2017—2020、2021—2024 每段至少 2 个完成锚月，三段合计至少 12 个完成锚月；若达不到，只能标记“样本不足”，不能用股票事件数量放宽。
- 每段股票事件胜率不低于 85%，中位净收益为正，PF 不低于 2.0；无亏损时 PF 记为正无穷并同时报告样本数。
- 每段锚月等权平均净收益为正；全样本锚月正收益占比不低于 80%，最差锚月等权收益不低于 -5%。
- 以锚月为重采样单元、固定随机种子的 10,000 次 cluster bootstrap，其全样本平均收益 95% 区间下界必须大于 0。
- 逐一删除每个锚月后的全样本等权平均收益最小值必须大于 0，防止结论由单一危机月份支撑。
- 60%/70%/80% 三档回撤中至少两档的锚月平均收益、留一批次最小均值与 bootstrap 下界均为正；否则判定阈值脆弱。

### Portfolio level

- 2013—2016、2017—2020、2021—2024 三段分别重启 100 万元，CAGR 均为正，最大回撤均不低于 -10%。
- 每段最差滚动 24 个月收益不低于 -10%；因策略允许长时间空仓，不以“正滚动窗口占比”惩罚没有交易的窗口。
- 总体最差锚月和最大单笔亏损必须由 case 解释；核销、停牌和后来退市不得从样本中删除。
- 2025—2026 当前持仓和未完成锚单列，不读取截止日后的退出，也不计入通过判据。

### Final status

- `historically_robust_research_candidate`：全部历史严格检查通过，但因历史已暴露，仍只允许观察或极小试验仓。
- `promising_but_independent_sample_insufficient`：收益结构好但独立锚月不足。
- `strict_structure_failed_robustness`：任一核心收益、尾部、批次依赖或阈值稳定性检查失败。
- `deployment_eligible` 本轮固定为 `false`；至少需要冻结规则后新增 2 个互不相邻、均已完成的系统性恐慌锚月，才允许重新评估。

## Fixed case review

- 列出全部负收益锚月及其最大亏损股票，不只展示成功案例。
- 展示收益最高的 3 个锚月，用来识别是否由单一牛市反弹主导。
- 对财务生存门槛删掉的股票分别统计“删掉的最终赢家”和“删掉的尾部损失”。
- 对等待确认分别统计错过的止盈事件、避免的尾部亏损和确认后仍亏损案例。
- 单列 2025—2026 未完成信号的当前浮动状态和剩余观察期。

## Implementation

- 新增 `src/quant/research/monthly_low_zone_strict.py`：冻结配置、嵌套门槛、Wilson 区间、锚月聚类 bootstrap、留一批次与严格判定。
- 新增 `tests/research/test_monthly_low_zone_strict.py`：因果财务门槛、双重恐慌、主结构、聚类区间、留一批次和样本不足判定。
- 新增 `scripts/research/backtest_monthly_low_zone_strict.py`：复用既有事件路径，读取点时年报，生成嵌套消融、组合曲线、case、决策与报告。
- 输出到 `reports/research/monthly_low_zone_strict/`，不覆盖此前报告。

## Verification

```bash
PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone_strict.py
PYTHONPATH=src pytest -q tests/research/test_monthly_low_zone_profit_lock.py tests/research/test_monthly_low_zone_confirmation.py tests/research/test_monthly_low_zone.py
PYTHONPATH=src python -m compileall -q src/quant/research/monthly_low_zone_strict.py scripts/research/backtest_monthly_low_zone_strict.py tests/research/test_monthly_low_zone_strict.py
PYTHONPATH=src python scripts/research/backtest_monthly_low_zone_strict.py
git diff --check
```

## Rollback

本轮不改线上策略、数据库或既有报告。撤回只删除本计划列出的三个新代码文件和忽略目录 `reports/research/monthly_low_zone_strict/`，不得删除此前研究或用户文件。

## Case-driven historical extension (added after the frozen primary failed)

首个严格主结构在 2013—2024 有 101 个完成股票事件、股票胜率 93.07%，但只有 8 个独立锚月；锚月 bootstrap 下界为 -6.11%，最差锚月 -28.38%，因此按冻结标准失败。失败批次说明 `no_new_low_20` 只是暂时不跌，并不等于价格已经重新获得底部区间支撑。

已登记的 `range_mid_reclaim` 消融在回撤 70% 且生存门槛 80% 时只有 6 个锚月，但 6 个月全部为正、最差月 +6.49%。继续查看冻结邻域后发现，财务门槛从严格价格样本中删除了 90 个最终赢家、2 个尾部损失和 5 个其他损失，不能证明它提高了历史尾部确定性。因此下一步不在 2013—2024 继续选最优格，而冻结一个机制更清楚、样本稍多的 case 修正版，交给 2000—2012 的新增长历史检验：

- 市场锚：正收益占比不高于 20%，且横截面 20 日中位收益不高于 -10%；这是比首轮 -8% 更深的系统性恐慌，而不是利用 15% 上涨家数边界剔除单月。
- 个股锚：前高回撤至少 60%；50%/70%/80% 全部作为稳定性邻域，主值不按新增历史结果切换。
- 价格生存确认：`range_mid_reclaim`，即至少 20 个交易日不创新低、20 日收益转正且重新站回因果底部区间中轴。
- 不使用 5 年财务持续性硬门槛。永久损失由完整退市样本、-100% 核销、每锚 2.5% 和最多 20 锚承担；财务状态仍保留为 case 诊断，不能因在已见数据上表现差而改成新的事后财务阈值。
- 退出仍固定 504 个市场交易日、15% 止盈和 20bp 成本；10%/20% 仅为退出邻域。

2000—2012 历史在本轮冻结后才补取，使用每日全市场截面以保留当时上市与后来退市股票，不用当前成分股列表重建过去。至少要求旧周期有 5 个完成锚月、正锚月占比不低于 80%、bootstrap 下界为正、最差月不低于 -5%；与 2013—2024 合并后至少 12 个锚月。50%/60%/70%/80% 中至少三档合并 bootstrap 下界和留一批次最小均值为正。即使全部通过，也只升级为跨周期历史研究结构；因修正规则来自已见 case，`deployment_eligible` 继续固定为 `false`。

新增 `scripts/research/extend_monthly_low_zone_history.py`，以年度分片、可续跑方式补取并审计 2000—2015 日线和上证指数；2013—2015 只用于让截至 2012 年的锚完成最长 126 日等待与 504 日持有，不作为新增信号期。新增 `scripts/research/backtest_monthly_low_zone_strict_extension.py`，重建旧周期月线锚、区间中轴确认、完整事件与组合，并与 2013—2024 case 修正版合并判定。数据和报告分别写入 `data/research/monthly_low_zone_strict_extension/` 与 `reports/research/monthly_low_zone_strict_extension/`。

## Final falsification boundary (added after the old cycle rejected the case correction)

旧周期主结构只有 4 个独立锚月，股票胜率 75%、PF 1.10、锚月均值 -0.22%、bootstrap 下界 -11.30%、最差月 -16.74%；因此不允许把2013—2024的高胜率外推为高确定性。最后只验证与失败机制直接对应、此前已有因果定义的四种更严格确认：全市场宽度修复、个股相对宽度修复、再叠加周线修复、指数修复；另把原点时财务生存 60%/80% 作为诊断重新放回。若这些现成机制仍不能让旧周期达到胜率 85%、PF 2、正锚月 80%、bootstrap 下界为正和最差月不低于 -5%，停止继续切分参数。

该轮由 `scripts/research/backtest_monthly_low_zone_strict_falsification.py` 固化到 `reports/research/monthly_low_zone_strict_falsification/`。它不是新一轮选优；任何只剩一两个旧锚月的 100% 胜率结果均按样本不足处理。若全部失败，最终结论必须是“没有达到高确定性的带血筹结构”，而不是继续寻找恰好删除 2012 年亏损月的边界。
