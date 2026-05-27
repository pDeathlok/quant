"""
右侧抄底策略 (Right Side Bottom Fishing Strategy)

策略逻辑：
1. 先跌完：前面有一波明显的下跌（跌幅超过一定比例）
2. 止跌：出现小阴小阳、十字星，不再创新低
3. 缩量：底部成交量明显变小，抛压衰竭
4. 右侧买点：
   - 出现中阳线/大阳线（涨幅超过一定比例）
   - 放量站上短期均线（5日/10日）
   - 突破小平台颈线

适用市场：A股
风格：趋势跟随 / 右侧交易
"""

import akquant as aq
from akquant import Bar, Strategy
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class RightSideBottomFishingStrategy(Strategy):
    """
    右侧抄底策略

    参数说明：
    - fall_threshold: 下跌幅度阈值（默认 0.15，即 15%）
    - consolidation_days: 盘整天数要求（默认 3）
    - volume_shrink_ratio: 缩量比例（相对于下跌段的平均成交量）
    - breakout_涨跌幅: 中阳线定义（默认 0.03，即 3%）
    - short_ma: 短期均线周期（默认 5）
    - medium_ma: 中期均线周期（默认 10）
    """

    def __init__(self,
                 fall_threshold: float = 0.15,
                 consolidation_days: int = 3,
                 volume_shrink_ratio: float = 0.5,
                 breakout_pct: float = 0.03,
                 short_ma: int = 5,
                 medium_ma: int = 10):
        super().__init__()

        # === 策略参数 ===
        self.fall_threshold = fall_threshold          # 下跌幅度阈值
        self.consolidation_days = consolidation_days  # 盘整天数要求
        self.volume_shrink_ratio = volume_shrink_ratio  # 缩量比例
        self.breakout_pct = breakout_pct            # 中阳线定义
        self.short_ma_period = short_ma              # 短期均线周期
        self.medium_ma_period = medium_ma            # 中期均线周期

        # === 预热期 ===
        self.warmup_period = max(60, short_ma * 2, medium_ma * 2)

        # === 状态变量 ===
        self.state = "watching"  # watching, bottom_found, ready_to_buy, in_position
        self.bottom_high = 0.0    # 底部区域的最高点（颈线）
        self.peak_price = 0.0   # 下跌前的最高价
        self.trough_price = 0.0  # 最近的最低价

        # === 日志开关 ===
        self.verbose = True

    def on_start(self) -> None:
        """策略启动时调用"""
        if self.verbose:
            print(f"[{self.now}] 右侧抄底策略启动")
            print(f"  下跌阈值: {self.fall_threshold:.1%}")
            print(f"  盘整天数: {self.consolidation_days}天")
            print(f"  缩量比例: {self.volume_shrink_ratio:.1%}")
            print(f"  突破涨幅: {self.breakout_pct:.1%}")

    def _is_small_candle(self, row: pd.Series) -> bool:
        """
        判断是否为小阴小阳/十字星
        条件：实体部分小于整根K线的30%
        """
        body = abs(row['close'] - row['open'])
        full_range = row['high'] - row['low']

        if full_range == 0:
            return True

        return body / full_range < 0.3

    def _is_doji(self, row: pd.Series) -> bool:
        """
        判断是否为十字星
        条件：开盘价和收盘价非常接近
        """
        body = abs(row['close'] - row['open'])
        full_range = row['high'] - row['low']

        if full_range == 0:
            return False

        return body / full_range < 0.1

    def _calculate_fall(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        计算下跌幅度
        返回：(下跌幅度, 峰值价格, 谷值价格)
        """
        if len(df) < 20:
            return 0.0, 0.0, 0.0

        # 取最近20天的高点和低点
        recent_20 = df.tail(20)
        peak = recent_20['high'].max()
        trough = recent_20['low'].min()

        # 计算从峰值到谷值的跌幅
        if peak > 0:
            fall_pct = (peak - trough) / peak
        else:
            fall_pct = 0.0

        return fall_pct, peak, trough

    def _check_volume_shrink(self, df: pd.DataFrame, bottom_start_idx: int) -> bool:
        """
        检查底部是否缩量
        底部成交量明显小于下跌段的平均成交量
        """
        if bottom_start_idx < 5:
            return False

        # 下跌段的平均成交量
        fall_volume = df['volume'].iloc[:bottom_start_idx].mean()

        # 底部区域的成交量
        bottom_volume = df['volume'].iloc[bottom_start_idx:].mean()

        if fall_volume == 0:
            return False

        return bottom_volume < fall_volume * self.volume_shrink_ratio

    def _is_steady_bottom(self, df: pd.DataFrame, recent_days: int = 5) -> bool:
        """
        判断是否为止跌区域（小阴小阳、十字星，不再创新低）
        """
        if len(df) < recent_days:
            return False

        recent = df.tail(recent_days)

        # 检查最后几天是否为小阴小阳或十字星
        small_candles = 0
        for _, row in recent.iterrows():
            if self._is_small_candle(row) or self._is_doji(row):
                small_candles += 1

        # 至少70%是小阴小阳/十字星
        if small_candles < recent_days * 0.7:
            return False

        # 不再创新低（最近5天最低价高于历史最低价）
        current_low = recent['low'].min()
        historical_low = df['low'].iloc[:-recent_days].min()

        return current_low >= historical_low * 0.98  # 允许2%的误差

    def _check_breakout(self, df: pd.DataFrame, neck_line: float) -> Tuple[bool, bool, float]:
        """
        检查是否突破颈线
        返回：(是否突破, 是否放量, 涨跌幅)
        """
        if len(df) < 2:
            return False, False, 0.0

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # 计算涨跌幅
        pct_change = (current['close'] - prev['close']) / prev['close']

        # 是否为中阳线/大阳线
        is_big_candle = pct_change >= self.breakout_pct

        # 是否突破颈线
        is_breakout = current['close'] > neck_line and current['open'] <= neck_line

        # 是否放量（站上均线时成交量大于5日均量）
        ma5_volume = df['volume'].rolling(5).mean().iloc[-1]
        is_volume_up = current['volume'] > ma5_volume * 1.2

        return is_breakout and is_big_candle, is_volume_up and is_big_candle, pct_change

    def _calculate_ma(self, df: pd.DataFrame, period: int) -> float:
        """计算均线值"""
        if len(df) < period:
            return 0.0
        return df['close'].rolling(period).mean().iloc[-1]

    def _is_above_ma(self, df: pd.DataFrame, period: int) -> bool:
        """判断是否站上均线"""
        if len(df) < period:
            return False
        ma = self._calculate_ma(df, period)
        return df['close'].iloc[-1] > ma

    def _detect_bottom_pattern(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        检测底部形态
        返回：(是否形成底部, 颈线价格)
        """
        if len(df) < 20:
            return False, 0.0

        # 1. 计算下跌幅度
        fall_pct, peak, trough = self._calculate_fall(df)

        # 必须有明显的下跌
        if fall_pct < self.fall_threshold:
            return False, 0.0

        # 2. 找到下跌结束的位置（最低点）
        # 从最低点往前数，计算盘整区域
        trough_idx = df['low'].iloc[-20:].idxmin()
        trough_position = df.index.get_loc(trough_idx)

        # 盘整区域起始位置
        consolidation_start = len(df) - 5

        # 3. 检查是否为止跌区域
        bottom_df = df.iloc[consolidation_start:]
        if not self._is_steady_bottom(df.iloc[:consolidation_start + 1]):
            return False, 0.0

        # 4. 检查是否缩量
        if not self._check_volume_shrink(df, consolidation_start):
            return False, 0.0

        # 5. 计算颈线（盘整区间的最高点）
        neck_line = bottom_df['high'].max()

        # 更新状态
        self.peak_price = peak
        self.trough_price = trough
        self.bottom_high = neck_line

        return True, neck_line

    def on_bar(self, bar: Bar) -> None:
        """
        核心交易逻辑
        """
        symbol = bar.symbol

        # === 1. 获取历史数据 ===
        history = self.get_history_df(
            count=self.warmup_period + 1,
            symbol=symbol,
            fields=["open", "high", "low", "close", "volume"]
        )

        if history is None or len(history) < self.warmup_period:
            return

        # === 2. 状态机逻辑 ===
        position = self.get_position(symbol)

        # 如果已持仓，检查是否需要止损/止盈
        if position > 0:
            # 止损：跌破颈线一定比例
            if bar.close < self.bottom_high * 0.95:
                if self.verbose:
                    print(f"[{bar.timestamp_iso}] 止损出局 - 跌破颈线")
                self.order_target_percent(0.0, symbol)
                self.state = "watching"
            return

        # === 3. 检测底部形态 ===
        if self.state == "watching":
            bottom_detected, neck_line = self._detect_bottom_pattern(history)

            if bottom_detected:
                self.state = "ready_to_buy"
                if self.verbose:
                    print(f"[{bar.timestamp_iso}] 底部形态形成 - 颈线: {neck_line:.2f}, "
                          f"峰值: {self.peak_price:.2f}, 谷值: {self.trough_price:.2f}")

        # === 4. 等待右侧买入信号 ===
        if self.state == "ready_to_buy":
            # 检查是否放量突破颈线
            breakout, volume_up, pct_change = self._check_breakout(history, self.bottom_high)

            # 同时检查是否站上短期均线
            above_ma5 = self._is_above_ma(history, self.short_ma_period)
            above_ma10 = self._is_above_ma(history, self.medium_ma_period)

            if breakout and above_ma5 and volume_up:
                if self.verbose:
                    print(f"[{bar.timestamp_iso}] 右侧买入信号！")
                    print(f"  涨幅: {pct_change:.2%}")
                    print(f"  放量: {'是' if volume_up else '否'}")
                    print(f"  站上5日均线: {'是' if above_ma5 else '否'}")
                    print(f"  站上10日均线: {'是' if above_ma10 else '否'}")

                # 买入
                self.order_target_percent(0.95, symbol)
                self.state = "in_position"
            else:
                # 如果太久没有突破（超过10天），重新等待
                if len(history) > 0:
                    recent_days = len(history) - history['high'].iloc[-10:].idxmax()
                    if recent_days > 10:
                        self.state = "watching"
                        if self.verbose:
                            print(f"[{bar.timestamp_iso}] 突破超时，重新等待")

    def on_order(self, order) -> None:
        """订单状态变化回调"""
        if order.status == aq.OrderStatus.Filled:
            if self.verbose:
                print(f"[{order.timestamp}] 订单成交 - {order.side} {order.symbol} "
                      f"数量: {order.quantity} 价格: {order.price:.2f}")

    def on_stop(self) -> None:
        """策略结束时调用"""
        if self.verbose:
            print(f"[{self.now}] 策略结束")


# ==============================================================================
# 回测运行代码
# ==============================================================================
def get_data(symbol: str = "600000",
             start_date: str = "20200101",
             end_date: str = "20231231") -> pd.DataFrame:
    """获取回测数据"""
    import akshare as ak

    print(f"正在获取 {symbol} 的历史数据...")

    if symbol.startswith("6"):
        market_symbol = f"sh{symbol}"
    else:
        market_symbol = f"sz{symbol}"

    df = ak.stock_zh_a_daily(
        symbol=market_symbol,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )

    df["symbol"] = symbol
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})

    print(f"数据获取完成 - {len(df)} 条记录")
    return df


def run_backtest(strategy_class,
                 data: pd.DataFrame,
                 initial_cash: float = 100_000.0,
                 commission_rate: float = 0.0003,
                 stamp_tax_rate: float = 0.001,
                 lot_size: int = 100) -> aq.BacktestResult:
    """运行回测"""
    print("\n" + "=" * 50)
    print("开始回测 - 右侧抄底策略")
    print("=" * 50)

    result = aq.run_backtest(
        strategy=strategy_class,
        data=data,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        lot_size=lot_size,
        fill_policy={
            "price_basis": "open",
            "bar_offset": 1,
            "temporal": "same_cycle"
        }
    )

    return result


def print_results(result: aq.BacktestResult) -> None:
    """打印回测结果"""
    print("\n" + "=" * 50)
    print("回测结果摘要")
    print("=" * 50)
    print(result)


if __name__ == "__main__":
    # === 配置参数 ===
    SYMBOL = "600000"
    START_DATE = "20200101"
    END_DATE = "20231231"
    INITIAL_CASH = 100_000.0

    # === 运行流程 ===
    df = get_data(SYMBOL, START_DATE, END_DATE)
    result = run_backtest(RightSideBottomFishingStrategy, df, initial_cash=INITIAL_CASH)
    print_results(result)