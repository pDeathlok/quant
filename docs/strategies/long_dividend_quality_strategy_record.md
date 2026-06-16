# L1 长期股息质量策略档案

## 1. 文档目的

本文档记录一个偏长期持仓的策略系统：`L1 市场趋势质量长期版`。

该策略不是 B1/B2 这类短线反弹策略的延长版，而是一个组合级长期系统。目标是：

- 用更谨慎的买点建立长期核心仓。
- 通过市场趋势、质量、估值、趋势、风险和股息稳定性因子提高持仓稳定性。
- 用底仓和做T仓拆分降低持仓波动。
- 在基本面、股息吸引力或长期趋势失效时减仓或清仓。

本文档仅记录系统设计和研究口径，不构成具体投资建议。实盘前必须完成历史回测、样本外验证、交易成本模拟和风险压力测试。

## 2. 当前研究结论

当前建议采用的策略版本：

```text
v19_style_grid_full_entry_sleeve
```

核心变化：

```text
市场趋势决定买入谨慎度和目标仓位。
股息率不再作为硬性决定因素，只作为参考因子。
股息评分由当前 dv_ttm 和 36 个月股息率稳定性共同决定。
成长股不并入防守核心池，而是在 risk_on 市场通过上限明确的成长袖珍仓参与。
券商预测/一致预期数据按 report_date <= signal_date 的 as-of 规则进入评分，避免时间穿越。
Datayes/萝卜投研和 AkShare 作为当前主要数据源；Tushare report_rc 暂停补数。
Datayes 当前更适合作为历史覆盖、分析师关注度和质量排序参考；真正逐篇历史前瞻预测后续优先从萝卜投研/东方财富/巨潮等逐篇研报源补。
买入仍保持谨慎，但在 risk_on 市场允许优质标的略高于目标价建仓。
持仓采用状态机，买入后不要求每月重新满足买入条件，只有长期失效才退出。
执行层保留月度目标仓位建仓，持仓过程中按成长、稳健价值、高波动、质量核心四类使用不同网格/做T阈值。
```

回测区间：

```text
2013-01-04 至 2026-06-12
```

核心结果：

| 版本 | 定位 | 年化收益 | 最大回撤 | Sharpe | 平均持仓数 | 平均仓位 |
|---|---|---:|---:|---:|---:|---:|
| `baseline` | 股息硬门槛 + 月度重新选股 | 4.32% | -11.00% | 0.68 | 2.52 | 23.74% |
| `v3_stateful_hold` | 谨慎买入 + 状态机持有 | 5.11% | -16.82% | 0.52 | 5.93 | 46.52% |
| `v4_market_regime` | 市场趋势分层 + 股息稳定性参考 | 8.84% | -24.64% | 0.74 | 9.00 | 51.27% |
| `v5_industry_cap` | v4 + 简单行业数量上限 | 8.40% | -25.98% | 0.70 | 8.40 | 51.21% |
| `v6_t_overlay` | v4 + 保守做T，卖出 30% 交易仓 | 8.47% | -22.83% | 0.77 | 9.00 | 49.05% |
| `v7_t_overlay_light` | v4 + 轻量做T，卖出 20% 交易仓 | 8.74% | -23.64% | 0.76 | 9.00 | 50.46% |
| `v8_growth_sleeve` | risk_on 增加成长袖珍仓 | 9.29% | -33.08% | 0.71 | 12.48 | 52.55% |
| `v9_growth_sleeve_capped` | 成长袖珍仓限仓版 | 10.08% | -32.98% | 0.76 | 11.90 | 52.39% |
| `v10_bluechip_growth_sleeve` | 大市值成长放宽版 | 8.41% | -34.09% | 0.65 | 11.57 | 51.57% |
| `v11_analyst_growth_sleeve` | 成长袖珍仓 + 券商预测 as-of | 10.20% | -28.77% | 0.82 | 10.54 | 50.67% |
| `v13_mega_quality_growth_sleeve` | 放宽高质量大盘成长入口 | 9.80% | -33.68% | 0.76 | 14.11 | 51.24% |
| `v14_selective_mega_growth_sleeve` | risk_on 下选择性 mega 成长替换 | 7.58% | -26.05% | 0.69 | 9.30 | 47.87% |
| `v15_analyst_quality_rank_sleeve` | v11 + 分析师/高质量成长排序加分 | 10.39% | -28.78% | 0.83 | 10.47 | 50.58% |
| `v16_mega_rank_growth_sleeve` | mega 成长同池竞争，非强插 | 9.62% | -28.73% | 0.77 | 10.73 | 50.75% |
| `v17_staged_entry_sleeve` | v15 + 慢速分批建仓 | 7.79% | -24.14% | 0.77 | 10.47 | 40.99% |
| `v18_style_grid_overlay_sleeve` | v17 + 风格化网格/做T | 7.60% | -22.53% | 0.81 | 10.47 | 39.22% |
| `v19_style_grid_full_entry_sleeve` | v15 + 风格化网格/做T，不放慢建仓 | 10.37% | -27.79% | 0.86 | 10.47 | 49.06% |
| `v20_fast_staged_grid_sleeve` | 快速分批建仓 + 风格化网格/做T | 8.76% | -24.98% | 0.83 | 10.47 | 43.52% |
| `v21_forecast_dual_score_sleeve` | 预测成长/预测价值直接进入核心评分 | 7.24% | -20.52% | 0.74 | 5.68 | 39.50% |
| `v22_forecast_dual_score_grid_sleeve` | v21 + 风格化网格/做T | 7.17% | -20.11% | 0.75 | 5.68 | 38.75% |
| `v23_forecast_rank_grid_sleeve` | 预测成长/价值较重排序增强 + 做T | 7.69% | -27.52% | 0.71 | 9.02 | 47.35% |
| `v24_forecast_tiebreak_grid_sleeve` | 预测轻量 tie-break + 做T | 10.37% | -27.79% | 0.86 | 10.47 | 49.06% |
| `v25_forecast_guardrail_grid_sleeve` | 预测只做风控 + 趋势成长仓 | 9.59% | -30.50% | 0.77 | 10.74 | 50.70% |
| `v26_market_trend_compounder_sleeve` | 强资金趋势下扩复利成长仓 | 9.42% | -33.60% | 0.74 | 12.33 | 51.01% |
| `v27_cautious_compounder_sleeve` | 更谨慎复利成长入口 | 9.62% | -30.08% | 0.78 | 10.53 | 50.51% |
| `v28_overheat_guarded_compounder_sleeve` | 过热保护 + 复利成长仓 | 9.73% | -24.86% | 0.84 | 10.85 | 48.90% |
| `v29_overheat_throttle_grid_sleeve` | v19 选股 + 市场过热仓位节流 | 10.28% | -26.15% | 0.86 | 10.46 | 49.65% |
| `v30_concentrated_trend_sleeve` | 减少持仓 + 趋势仓位 | 7.71% | -19.61% | 0.81 | 5.40 | 37.76% |
| `v31_bull_bear_exposure_sleeve` | 牛市高仓 + 熊市近空仓 | 9.79% | -21.90% | 0.89 | 9.98 | 42.51% |
| `v32_empty_bear_sleeve` | 熊市/回撤信号强制空仓 | 8.97% | -21.22% | 0.77 | 11.92 | 51.33% |
| `v33_bull_boost_defensive_bear_sleeve` | 健康牛市加仓 + 熊市保留防御仓 | 11.30% | -25.24% | 0.95 | 9.92 | 49.10% |
| 沪深300 | 对照基准 | 5.07% | -46.70% | - | - | - |

结论：

```text
v33_bull_boost_defensive_bear_sleeve 当前最优。
它的年化收益约为沪深300的 2 倍，最大回撤明显低于沪深300。
v19_style_grid_full_entry_sleeve 保留为稳健基准版本。
v5 的简单行业数量截断没有改善风险收益，暂不采用。
v7_t_overlay_light 适合作为可选做T增强：略牺牲收益，改善回撤和 Sharpe。
v9_growth_sleeve_capped 提升收益但最大回撤扩大到约 -33%，不适合作为稳健默认。
v10 说明“只放宽大市值成长股”不能自动提升质量，暂不采用。
v11 完成券商预测数据的接入和时间安全验证，v15 在不扩容成长仓的前提下进一步改善排序。
v19 保持 v15 股票池不变，只在持仓执行层加入风格化网格/做T，使 Sharpe 从 0.83 提升到 0.86，最大回撤从 -28.78% 降至 -27.79%。
v17/v18/v20 的分批建仓都能降低回撤，但明显降低平均仓位和年化收益；当前不作为默认执行规则。
v21/v22/v23 说明：当前 Datayes/AkShare 的严格 as-of forward 预测覆盖太稀疏，不能让预测增长/预测估值直接主导历史选股。重权重预测会显著降低持仓数和收益；轻量 tie-break 的 v24 与 v19 持仓完全一致，说明预测信号尚不足以改变主组合。
v13/v14/v16 均能让比亚迪、宁德时代等高质量成长股更容易出现，但整体收益或 Sharpe 下降，因此不作为自动主规则。
v25/v26/v27 说明：把成长复利股入口做得更积极，确实能更早发现少数大牛股，但组合层会在 2015、2021 等强趋势后急跌阶段承受更大回撤。
v28 说明：加入市场过热/回撤保护后，最大回撤明显下降到约 -24.86%，但年化收益降到 9.73%，适合作为保守回撤研究版，不替代主策略。
v29 说明：只保留 v19 选股和做T，在市场过热或 60 日回撤恶化时节流仓位，年化收益 10.28%、最大回撤 -26.15%、Sharpe 0.86；这是当前最接近 v19 的风险优化候选。
v30 说明：减少持仓能显著降低回撤，但平均仓位和持股数过低，年化收益损失过大。
v31 说明：牛市高仓、熊市近空仓能把最大回撤降到 -21.90%，Sharpe 提升到 0.89，但熊市防御资产正收益被削弱，年化收益低于 v19。
v32 说明：熊市/回撤阶段强制空仓过度，收益和 Sharpe 都不如 v31/v33。
v33 说明：正确方向不是熊市完全空仓，而是健康牛市提高仓位、过热/回撤降仓、熊市保留少量高质量防御仓。该版本年化收益 11.30%、最大回撤 -25.24%、Sharpe 0.95，当前作为主推荐。
```

主要风险：

```text
组合在 2024-2026 对银行、港口、公用事业等高股息/低估值资产仍有较高暴露。
2015 年市场急跌期间出现最大回撤，说明 risk_on 状态下仍需进一步研究更快的市场风险切换。
2015 case 显示，沪深300在 2014-12 至 2015-05 已出现明显过热：60 日涨幅多次超过 30%，120 日涨幅最高超过 70%。仅用均线向上定义 risk_on 会在高位继续持有较高仓位。
当前已实现日级做T回测，但尚未实现日内分钟级做T或真实 T+1 执行约束。
Datayes consensus 的历史字段多为年度一致预期/实际值快照，不等同于逐篇研报的历史前瞻预测；后续仍需补齐真实 report_rc 或其它研报源。
严格使用 `report_date <= signal_date` 后，当前 forward 预测特征在 targets 中几乎没有历史覆盖：v24 中 `analyst_forward_growth_score` 只有 8 行非空、1 只股票。后续要想真正用券商盈利预测优化成长/价值筛选，需要继续补逐篇历史研报或历史一致预期快照。
```

更新：

```text
已实现日级做T回测版本 v6_t_overlay 和 v7_t_overlay_light。
当前建议使用 v33_bull_boost_defensive_bear_sleeve 作为主策略。
v19_style_grid_full_entry_sleeve 作为低复杂度基准，v29_overheat_throttle_grid_sleeve 作为仅仓位节流的保守候选。
若使用做T，优先使用 v7_t_overlay_light，而不是更激进的 v6_t_overlay。
若希望在牛市/趋势向上阶段进一步提高成长弹性，不建议直接启用 v13/v14/v16/v26；更合适的方向是把比亚迪、宁德时代、北方华创、中际旭创、阳光电源这类股票放入单独成长卫星组合，接受更高波动并设置独立仓位上限，而不是污染长期稳健主组合。
若账户偏保守，可参考 v20 的快速分批建仓，但应接受年化收益下降。
```

关键产物：

```text
configs/strategies/long_dividend_quality.yaml
scripts/research/backtest_long_dividend_quality.py
reports/long_dividend_quality/v19_style_grid_full_entry_sleeve/l1_dividend_quality_backtest.md
reports/long_dividend_quality/v19_style_grid_full_entry_sleeve/l1_dividend_quality_summary.json
reports/long_dividend_quality/v19_style_grid_full_entry_sleeve/l1_dividend_quality_targets.csv
reports/long_dividend_quality/v7_t_overlay_light/l1_dividend_quality_summary.json
```

## 3. 策略定位

策略 ID：

```text
l1_market_regime_quality
```

持仓周期：

```text
3 个月到数年
```

适用股票：

```text
质量较好、估值不过度透支、中长期趋势未破坏、风险可控的股票。
股息率和股息稳定性是加分项，但不是唯一选股条件。
市场趋势向上时，允许少量高 ROE、收入增长、EPS 增长和趋势强度都较好的成长股进入进攻袖珍仓。
```

策略性格：

```text
宁可错过，不追高。
达到目标价不等于立即买入。
股票先进入观察队列，等待更合适的回踩、折价或趋势确认买点。
```

## 4. 股票池过滤

候选池先排除明显不适合长期持仓的标的。

硬过滤：

```text
排除 ST、*ST、退市相关股票
上市时间不足 2 年的股票不进入核心池
近 60 日成交额或换手率过低的股票不进入核心池
total_mv 或 circ_mv 过小的股票不进入核心池
最近 120 日有效行情不足的股票不进入核心池
```

基础流动性与市值建议：

```text
circ_mv >= 50 亿
total_mv >= 80 亿
turnover_rate_ma20 >= 0.3
turnover_rate_ma20 <= 8.0
volatility_60d 不处于全市场最高 20%
```

上述阈值是第一版研究起点，后续应按 A 股分布和回测结果重新校准。

## 5. 长期评分

长期评分由 5 个子分数组成：

```text
long_score =
  15% dividend_score
+ 30% quality_score
+ 20% value_score
+ 20% trend_score
+ 15% risk_score
```

第一版可先使用当前项目已实现变量：

| 分数 | 变量 | 说明 |
|---|---|---|
| `dividend_score` | `dv_ttm`, `dv_ttm_stability_36m` | 股息率和 36 个月稳定性共同评分 |
| `quality_score` | `roe`, `netprofit_margin`, `or_yoy`, `debt_to_assets` | 盈利质量和负债约束 |
| `value_score` | `pe_ttm_inv`, `pb_inv`, `ps_ttm_inv` | 估值越合理越好 |
| `trend_score` | `close`, `ma_120`, `weekly_ma55_slope`, `return_120d` | 中长期趋势不能明显走坏 |
| `risk_score` | `volatility_60d`, `downside_volatility_60d`, `turnover_rate_ma20` | 惩罚高波动、流动性不足和过热换手 |

股息率评分不能简单取越高越好。异常高股息通常可能来自股价大跌、周期利润高点或一次性分红，应设置保护。当前采用：

```text
dv_ttm < 2%：不直接淘汰，但股息得分较低
2% <= dv_ttm <= 6%：正常加分区间
6% < dv_ttm <= 9%：谨慎加分，需要估值和趋势确认
dv_ttm > 9%：进入异常高股息检查，不直接加满分
dv_ttm_stability_36m 越高，说明股息率越稳定，股息得分越高
```

异常高股息检查：

```text
close > ma_120
weekly_ma55_slope >= 0
pe_ttm > 0
pb > 0
近 60 日未出现趋势性破位
```

## 6. 市场趋势分层

市场趋势使用沪深300指数：

```text
risk_on:
  close > ma120
  ma120_slope_20d > 0
  return_60d > 0

risk_off:
  close < ma120
  ma120_slope_20d < 0

neutral:
  非 risk_on 且非 risk_off
```

不同状态下的目标仓位和买入谨慎度：

| 市场状态 | 目标总仓位 | 买入分数 | 价格要求 |
|---|---:|---:|---|
| `risk_on` | 85% | `long_score >= 72` | `close <= target_price * 1.10` |
| `neutral` | 60% | `long_score >= 76` | `close <= target_price * 1.04` |
| `risk_off` | 30% | `long_score >= 84` | 必须满足回踩买入，且股息/质量分达标 |

## 7. 买入逻辑

策略采用观察队列，而不是信号触发后直接买入。

基础目标价定义：

```text
target_price = min(
  ma_60 * 1.02,
  ma_120 * 1.05,
  近 60 日中位收盘价
)
```

在 `risk_on` 中：

```text
close > ma_120
ma_120_slope_20d >= -0.01
long_score >= 72
close <= target_price * 1.10
```

在 `neutral` 中：

```text
close > ma_120
ma_120_slope_20d >= 0
long_score >= 76
close <= target_price * 1.04
```

在 `risk_off` 中：

```text
long_score >= 84
dividend_score >= 70
quality_score >= 65
close <= target_price
```

如果价格一路上涨没有回踩：

```text
不追买。
继续观察，直到下一次估值、股息率或均线回踩重新给出机会。
```

## 8. 持仓状态机

当前回测采用月度状态机：

```text
未持有股票：只有满足当前市场状态下的买入条件才可进入。
已持有股票：不要求每月重新满足买入条件，只检查长期失效条件。
组合补仓：每月用新候选补足最多 15 只。
```

退出规则：

```text
long_score < 58
或 risk_on 中 close < ma_120 * 0.94
或 neutral/risk_off 中 close < ma_120 * 0.98
或 risk_off 中 quality_score < 55
或 risk_off 中 risk_score < 35
```

## 9. 分批建仓与做T预留

## 10. 做T / 网格增强

当前已验证两种日级做T版本。

### 10.1 v6_t_overlay

规则：

```text
core_position = 70%
trading_position = 30%

卖出：
  close > ma20 * 1.08
  或 close > ma60 * 1.14
  则将该股仓位降到 core_position

买回：
  market_regime != risk_off
  close >= ma120 * 0.98
  且 close <= ma20 * 1.01 或 close <= ma60 * 1.03
  则买回到完整目标仓位
```

结果：

```text
年化收益：8.47%
最大回撤：-22.83%
Sharpe：0.77
做T交易次数：491
做T动作数：771
做T总换手：13.21
做T总成本：0.81%
```

结论：

```text
回撤和 Sharpe 改善，但过于频繁，收益牺牲较多。
```

### 10.2 v7_t_overlay_light

规则：

```text
core_position = 80%
trading_position = 20%

卖出：
  close > ma20 * 1.12
  或 close > ma60 * 1.18
  则将该股仓位降到 core_position

买回：
  market_regime != risk_off
  close >= ma120 * 0.98
  且 close <= ma20 * 1.015 或 close <= ma60 * 1.03
  则买回到完整目标仓位
```

结果：

```text
年化收益：8.74%
最大回撤：-23.64%
Sharpe：0.76
做T交易次数：275
做T动作数：422
做T总换手：4.78
做T总成本：0.29%
```

结论：

```text
v7_t_overlay_light 是当前推荐的可选做T增强。
它相比 v4_market_regime 年化收益略低，但最大回撤和 Sharpe 更好。
若实盘目标是稳定性优先，可启用 v7；若目标是长期总收益优先，保留纯 v4。
```

## 11. 成长袖珍仓增强

用户提出长期策略不能只关注确定性业绩，在市场趋势向上时成长股空间更大。当前已验证三个成长版本。

### 11.1 设计原则

成长股不直接放宽 L1 防守核心池，否则会把高估值、高波动带入默认长期策略。当前采用袖珍仓结构：

```text
核心仓：
  继续使用 v4_market_regime 的质量、估值、趋势、风险和股息稳定性框架。

成长袖珍仓：
  只在 risk_on 市场启用。
  目标总仓位上限 15%。
  最多 4 只。
  单股上限 5%。
  neutral 和 risk_off 不新增成长仓。
```

成长评分：

```text
growth_score =
  30% revenue_yoy
+ 25% eps_yoy
+ 20% roe
+ 15% return_120d
+ 10% ma120_slope_20d
```

成长入选条件：

```text
market_regime == risk_on
growth_score >= 78
roe >= 10
or_yoy >= 12
basic_eps_yoy >= 8
close > ma120
ma120_slope_20d > 0
```

### 11.2 回测结论

```text
v8_growth_sleeve:
  年化收益：9.29%
  最大回撤：-33.08%
  Sharpe：0.71

v9_growth_sleeve_capped:
  年化收益：10.08%
  最大回撤：-32.98%
  Sharpe：0.76

v10_bluechip_growth_sleeve:
  年化收益：8.41%
  最大回撤：-34.09%
  Sharpe：0.65

v11_analyst_growth_sleeve:
  年化收益：10.09%
  最大回撤：-28.92%
  Sharpe：0.79
  数据状态：仅 1 只股票 report_rc 覆盖，属于冒烟验证
```

结论：

```text
v9_growth_sleeve_capped 是当前成长增强中收益最高的版本。
但它把最大回撤从 v4 的 -24.64% 放大到 -32.98%。
因此 v9 适合作为进攻档或牛市增强档，不适合作为稳健默认版本。
v10 的大市值成长放宽没有改善结果，说明不能只因为公司优质或规模大就降低估值和风险约束。
v11 的实现路径正确，但需要补齐更广泛的券商预测数据后才能正式评价。
```

建议：

```text
稳健默认：v4_market_regime
稳健 + 做T：v7_t_overlay_light
进攻/牛市增强：v9_growth_sleeve_capped
研报预测研究版：v11_analyst_growth_sleeve
```

### 11.3 券商预测数据口径

Tushare `report_rc` 券商研报盈利预测接口已接入研究脚本，包含 EPS、PE、营收、净利润、目标价等字段。

防未来函数规则：

```text
可见性日期：report_date
调仓日 signal_date 只能使用 report_date <= signal_date 的研报。
预测期 quarter 只能表示预测对象，不允许作为可见性日期。
默认聚合窗口：signal_date 往前 180 天。
```

当前生成的预测特征：

```text
analyst_report_count_180d
analyst_org_count_180d
analyst_eps_mean_180d
analyst_pe_mean_180d
analyst_target_price_mean_180d
analyst_net_profit_mean_180d
analyst_revenue_mean_180d
analyst_eps_revision_180d
analyst_target_upside_180d
analyst_forecast_score
```

数据状态：

```text
已验证 report_rc 接口字段可用。
当前账号触发频控：按股票第二次请求提示 1次/分钟，日期区间请求提示 1次/小时。
Tushare report_rc 当前只拉取到 000001.SZ 样本。
AkShare 已补充全市场当前快照和少量历史个股研报样本。
由于历史 as-of 覆盖仍不足，不能据此正式评价 v11。
后续需要用断点续拉方式逐步补齐历史 report_date 数据，或更换更高频控权限的数据源。
```

补数方案：

```text
Tushare report_rc:
  优点：字段最完整，包含 EPS、PE、营收、净利润、目标价、预测期和研报发布日期。
  问题：当前账号频控较严，适合断点慢速补齐。

AkShare stock_profit_forecast_em:
  优点：一次分页即可获取全市场当前盈利预测快照。
  本轮验证：一次运行获取 2366 只股票、9464 行标准化预测记录。
  限制：属于当前快照，没有历史发布日期序列；只能从抓取日之后用于实盘/后续回测，不能回填 2013 年历史。

AkShare stock_research_report_em:
  优点：有个股研报发布日期和 EPS/PE 预测，可按 report_date 做 as-of。
  本轮验证：5 只股票样本中 3 只成功，新增 1353 行历史预测记录。
  限制：逐股票拉取，部分股票接口返回字段缺失，需要审计失败清单。

AkShare stock_rank_forecast_cninfo:
  优点：巨潮公开投资评级数据，含发布日期、评级、目标价上下限。
  本轮验证：2024-12-31 至 2025-01-01 两天新增 232 行，覆盖 198 只股票。
  限制：不含 EPS/PE/营收/利润，只适合补目标价空间和评级变化。
```

当前统一落库：

```text
data/raw/analyst_forecasts.parquet

source 分布：
  akshare_em_snapshot：9464 行，2366 只股票
  akshare_em_research：1353 行，3 只股票
  akshare_cninfo_rating：232 行，198 只股票
  tushare_report_rc：3899 行，1 只股票
```

重要约束：

```text
当前快照数据的 report_date 使用抓取日。
因此它不会在历史回测中提前可见，不会造成未来函数。
但它也不能被当作 2013-2026 全历史的预测数据补齐。
正式评价 v11 仍应优先补齐带历史 report_date 的 Tushare report_rc 或 AkShare 个股研报。
```

监控命令：

```text
python scripts/research/monitor_report_rc_refresh.py
```

## 12. 比亚迪案例说明

比亚迪 `002594.SZ` 数据完整，且进入了 v4 的预筛候选池，但没有进入主要持仓贡献。

诊断结果：

```text
评估月数：160
eligible 月数：0
最高 long_score：约 60.34
当前 v4 买入门槛：
  risk_on >= 72
  neutral >= 76
  risk_off >= 84
```

未入选原因：

```text
1. 股息率较低，历史 dv_ttm 均值约 0.44%，股息稳定性加分不足。
2. 估值长期偏高，PE/PB 分位在当前价值质量模型中不占优。
3. 2024-2025 趋势很强时，价格经常明显高于 target_price，谨慎买入条件不满足。
4. 2025 后部分月份 ma120_slope_20d 转弱，趋势分下降。
5. 该策略偏“质量 + 估值 + 趋势 + 风险 + 股息稳定性”的均衡型，不是高成长股趋势策略。
```

成长袖珍仓诊断：

```text
在 v8/v9/v10 中，比亚迪仍未进入最终持仓。
2024-10 曾满足 growth_entry。
当月 growth_score 约 83.61，long_score 约 63.20。
但成长袖珍仓只选 risk_on 下排名靠前的少数标的，比亚迪没有进入最终 top slots。
```

结论：

```text
比亚迪不是数据缺失导致没选中，而是策略风格不匹配。
当前 v9 已经覆盖“成长质量趋势”方向，但仍不为了单一个股强制入选。
如果希望覆盖比亚迪这类龙头成长股，应在 v9 之上继续研究“质量成长观察池”，
提高行业龙头、长期收入复合增长、盈利扩张稳定性和产业景气因子的权重。
不建议为了纳入单一个股而扭曲 L1 长期稳健策略。
```

单股目标仓位拆为核心仓和做T仓：

```text
core_position = 70%
trading_position = 30%
```

分三段建仓：

```text
第一笔：目标单股仓位的 30%
  条件：首次低位建仓信号成立

第二笔：目标单股仓位的 30%
  条件：买入后未跌破 ma_120，且回踩 ma_60 或 ma_120 后重新站稳

第三笔：目标单股仓位的 40%
  条件：长期评分仍 >= 80，股息率仍具吸引力，周线趋势确认
```

加仓禁区：

```text
close < ma_120
weekly_ma55_slope < 0
dv_ttm < 2%
long_score < 70
组合或市场风险状态为 RISK_OFF
```

## 7. 做T策略

做T只允许使用 `trading_position`，不得动用长期核心底仓。

高抛条件：

```text
close > ma_20 * 1.08
或 kdj_d_j > 90
或 单日放量长上影且 close 未创新高
```

高抛动作：

```text
卖出 trading_position 的 30% 到 50%
记录待买回份额和卖出参考价
```

低吸买回条件：

```text
close 回落到 ma_20 或 ma_60 附近
close >= ma_120
weekly_ma55_slope >= 0
long_score >= 70
```

禁止低吸：

```text
close < ma_120
long_score < 65
dividend_score < 60
触发基本面风险或组合风险降仓
```

## 8. 减仓与清仓

减仓条件：

```text
long_score < 70
或 dv_ttm < 2%
或 close 跌破 ma_60 后 5 个交易日未收回
```

减仓动作：

```text
先清 trading_position。
如果 20 个交易日内未修复，再降低 core_position。
```

清仓条件：

```text
long_score < 60
或 dividend_score < 50
或 close < ma_120 且连续 5 个交易日未收回
或 从持仓高点回撤 >= 25%
或 基本面、分红能力、财务质量出现重大恶化
```

清仓后进入冷却期：

```text
COOLDOWN 60 个交易日。
冷却期内不重新买入，除非长期评分重新进入前 10% 且价格回到目标区间。
```

## 9. 组合风控

建议第一版组合约束：

```text
最大持股数：8 到 15 只
单股目标仓位：5% 到 10%
单股最大仓位：12%
单行业最大仓位：25%
现金底线：15%
长期核心仓最大总仓位：70%
做T仓最大总仓位：20%
```

市场状态控制：

```text
RISK_ON：总仓位 70% 到 90%
NEUTRAL：总仓位 40% 到 70%
RISK_OFF：总仓位 10% 到 40%
```

再平衡：

```text
每月检查一次。
如果单股仓位偏离目标仓位超过 25%，触发再平衡。
优先用新增资金、分红和做T回笼资金调整，减少不必要卖出。
```

## 10. 回测注意事项

长期策略回测不能只看年化收益，至少需要输出：

```text
年化收益
最大回撤
最长回撤修复时间
股息贡献收益
价格贡献收益
换手率
单股集中度
行业集中度
胜率和盈亏比
持仓平均周期
现金占比
```

必须单独评估：

```text
不做T的纯核心仓版本
只使用做T仓的增强版本
股息率过滤开启/关闭对比
买入后等待回踩 vs 到目标价立即买入对比
```

## 11. 后续数据需求

当前项目已具备 `dv_ttm`、`dv_ratio`、估值、市值、趋势和波动率变量。为了提升长期策略质量，后续建议补充：

```text
ROE
ROA
毛利率
净利率
经营现金流 / 净利润
资产负债率
收入和利润 3 年稳定性
分红支付率
连续分红年数
股息增长率
```

这些变量补齐前，`quality_score` 应保持较低权重或用风险约束替代，避免把“高股息但质量恶化”的股票误判为长期机会。
