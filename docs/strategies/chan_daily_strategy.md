# 日线缠论结构策略

## 参考来源

- `waditu/czsc`: 参考其“分型/笔/中枢”和“信号-事件-交易”的建模方式，但本策略不直接依赖 `czsc`，避免引入 Rust 扩展和额外环境约束。
- 雪球文章 `8899955007/127630126`: 页面正文当前需要动态加载或登录访问；本策略只吸收“日线级别结构交易、买卖点分层”的思路。

## 策略定位

日线为主，偏右侧交易。策略不把一买作为主买点，而是将其作为观察信号；最终买入主要来自二买确认和三买确认。

## 结构识别

1. 对日线 OHLC 做连续复权口径处理。
2. 识别顶/底分型。
3. 将交替分型合成为笔，最小笔间隔默认 4 根日 K。
4. 用最近若干笔中连续三笔的价格重叠区识别中枢。
5. 在中枢、笔方向、均线趋势和 MACD 力度之间组合信号。

## 买点定义

- 一买观察：下跌笔创新低，但 MACD 柱力度弱于前一段下跌，且当日有反弹。只输出 `chan_buy1_watch`，不进入最终买入。
- 二买确认：前后两段下跌形成更高低点，价格重新站上 MA5/MA10，并首次突破最近下跌笔高点。
- 三买确认：价格向上离开中枢后回踩不进入中枢，再首次突破回踩下跌笔高点。

最终买入列为 `signal_chan_daily_long`。

## 卖出定义

输出 `signal_chan_daily_exit`：

- 收盘跌破最近中枢下沿；
- 或收盘跌破 20 日线 1.5%；
- 或上升笔创新高但 MACD 力度背驰，同时跌破 5 日线。

## 输出字段

- `chan_fx_mark`, `chan_fx_top`, `chan_fx_bottom`: 分型。
- `chan_stroke_direction`, `chan_stroke_end`, `chan_stroke_amplitude`: 笔。
- `chan_center_low`, `chan_center_high`, `chan_center_width`: 最近中枢。
- `chan_buy1_watch`, `chan_buy2_confirm`, `chan_buy3_confirm`: 买点分层。
- `signal_chan_daily_long`, `signal_chan_daily_exit`: 可回测信号。
- `chan_score`, `chan_buy_plan`, `chan_sell_plan`, `chan_structure_note`: 排名和交易说明。

## 风控建议

- 信号后次日若高开超过 3%，不追或降低仓位。
- 单票初始仓位建议低于普通趋势策略，先用 20%-30% 试仓。
- 跌破信号日中位价或中枢下沿时优先减仓。
- 市场指数处于日线下跌趋势时，只保留三买且降低仓位。

## 后续验证

1. 在沪深 A 股全市场做日线 T+1 开盘买入回测。
2. 分开评估二买、三买和二/三买合并结果。
3. 增加指数环境过滤：沪深 300 或中证 1000 的 MA20/MA60 趋势。
4. 与现有 `triple_volume_breakout` 策略做组合筛选，观察胜率和回撤变化。

## 模型筛选迭代

已新增 `scripts/research/train_chan_daily_models.py`，基于现有日线变量、缠论结构变量、T+1 开盘跳空、市场情绪代理变量训练三类 XGBoost 模型：

- `target_win10`: T+1 开盘买入后 10 日收益为正。
- `target_big10`: T+1 开盘买入后 10 日收益不低于 3%。
- `target_good`: 10 日收益不低于 2%，且 5 日收益不低于 -2%。

评估方式：

- `train`: 2023 年以前。
- `test`: 2023-2024 年。
- `oot`: 2025 年及以后。

当前最稳的候选筛选方向：

- `model_good_top10`: OOT 交易 1,948 条，10 日平均收益 3.41%，中位数 1.23%，胜率 56.93%，盈亏比 2.41。
- `buy3_model_good_top20`: OOT 交易 2,338 条，10 日平均收益 3.35%，中位数 0.78%，胜率 54.83%，盈亏比 2.31。
- `buy3_score_ge95_model_top50`: OOT 交易 4,567 条，10 日平均收益 3.27%，中位数 1.04%，胜率 55.88%，盈亏比 2.27。

推荐先采用：

1. 只交易三买或模型 Top10/Top20 高分池。
2. 主规则：`buy3_model_good_top20`。
3. 放宽规则：`buy3_score_ge95_model_top50`，用于候选不足时扩容。
4. 继续保留 T+1 高开过滤，高开超过 3% 降权，高开超过 6% 放弃。

case 分析发现：

- 赢家的 T 日前 5 日/20 日涨幅低于输家，说明三买不应追过热形态。
- 赢家所在日期的市场上涨家数比例更高，市场环境过滤有价值。
- 涨跌停情绪代理在模型重要性中排名靠前，后续应接入真实涨跌停数据替代代理口径。

情绪变量已新增 `src/quant/features/market_sentiment.py`：

- `limit_up_count_proxy`, `limit_up_ratio_proxy`
- `limit_down_ratio_proxy`
- `market_up_ratio`
- `market_sentiment_5d`, `market_panic_5d`
- `top_list_count`, `top_net_amount_ratio`, `top_net_rate`

龙虎榜缓存当前覆盖不足，净买字段在长期样本中稀疏；后续需要补齐 `data/raw/moneyflow` 或单独建设 `data/raw/top_list` 多年缓存，再重新训练模型。
