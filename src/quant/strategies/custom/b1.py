"""
B1 策略

策略条件：
1. 股票不包含ST
2. 涨跌幅在 -2% 到 +2% 之间
3. 当日振幅小于 7%
4. 当前的 BBI > 60日均线
5. KDJ J值 < 10
6. 成交量高于上一日
"""

from ..base import BaseStrategy
from ...data.factors import KDJ
import pandas as pd
from typing import Dict


class B1Strategy(BaseStrategy):
    """
    B1 策略实现

    入场条件：
    - 股票名称不包含 'ST'
    - 当日涨跌幅: -2% <= pct_change <= +2%
    - 当日振幅: (high - low) / low < 7%
    - BBI > MA60
    - KDJ J值 < -5
    - 当日成交量 > 上一日成交量

    出场条件（按优先级）：
    1. 长上影线: (high - close) > (close - low) 且收盘价较昨日跌超 1%，当日收盘卖出
    2. 止损: 价格跌破入场前一日最低价向下 2%
    3. 止盈: 价格在 BBI 线上方出现 2 次涨幅 > 3% 后，从最高点回撤 3% 离场
    4. 时间止损: 持有超过 N 天且从高点回撤 >= 1.5%
    """

    def __init__(
        self,
        hold_days: int = 5,
        stop_loss_pct: float = 0.02,
        take_profit_drawdown: float = 0.03,
        time_stop_drawdown: float = 0.015,
    ):
        super().__init__(name="B1")
        self.hold_days = hold_days
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_drawdown = take_profit_drawdown
        self.time_stop_drawdown = time_stop_drawdown
        self.kdj_factor = KDJ()
        self._price_cache = {}
        self._stop_loss_price: Dict[str, float] = {}
        self._highest_price: Dict[str, float] = {}  # 持仓期间最高价
        self._surge_count: Dict[str, int] = {}  # 持仓中涨幅 > 3% 的次数（BBI 上方）
        self._surge_peak: Dict[str, float] = {}  # 止盈激活后的最高点
        self._take_profit_active: Dict[str, bool] = {}  # 止盈是否已激活
    
    def _calculate_bbi(self, data: pd.DataFrame) -> pd.Series:
        """
        计算 BBI（多空指标）
        BBI = (MA3 + MA6 + MA12 + MA24) / 4
        """
        close = data["close"]
        ma3 = close.rolling(window=3).mean()
        ma6 = close.rolling(window=6).mean()
        ma12 = close.rolling(window=12).mean()
        ma24 = close.rolling(window=24).mean()
        bbi = (ma3 + ma6 + ma12 + ma24) / 4
        return bbi
    
    def _calculate_ma60(self, data: pd.DataFrame) -> pd.Series:
        """计算 60 日均线"""
        return data["close"].rolling(window=60).mean()
    
    def _calculate_amplitude(self, data: pd.DataFrame) -> pd.Series:
        """计算振幅 = (high - low) / low * 100"""
        return (data["high"] - data["low"]) / data["low"] * 100
    
    def on_start(self) -> None:
        """策略启动，启用历史数据追踪"""
        self.set_history_depth(60)

    def _is_st_stock(self, symbol: str, name: str = None) -> bool:
        """判断是否为 ST 股票"""
        # 从股票名称判断
        if name and ("ST" in name or "*ST" in name):
            return True
        # 从股票代码后缀判断（退市股）
        if symbol and ("ST" in symbol or "退" in symbol):
            return True
        return False
    
    def _get_j_value(self, symbol: str) -> float | None:
        """获取当前 KDJ J 值"""
        df = self.get_history_df(count=20, symbol=symbol)
        if df is None or len(df) < 20:
            return None
        kdj_result = self.kdj_factor.compute(df)
        return kdj_result["J"].iloc[-1]

    def should_entry(self, bar, current_pos: int) -> bool:
        """判断是否应该入场"""
        # 获取股票名称（如果可用）
        stock_name = getattr(bar, "name", "")
        
        # 条件1: 股票不包含 ST
        if self._is_st_stock(bar.symbol, stock_name):
            return False
        
        # 获取历史数据
        df = self.get_history_df(count=60, symbol=bar.symbol)
        if df is None or len(df) < 60:
            return False

        # 计算涨跌幅 (收盘价相对前一日)
        df["pct_change"] = df["close"].pct_change() * 100

        # 条件2: 涨跌幅在 -2% 到 +2% 之间
        pct_change = df.iloc[-1]["pct_change"]
        if not (-2 <= pct_change <= 2):
            return False

        # 条件3: 当日振幅小于 7%
        amplitude = self._calculate_amplitude(df).iloc[-1]
        if amplitude >= 7:
            return False
        
        # 条件4: BBI > MA60
        bbi = self._calculate_bbi(df).iloc[-1]
        ma60 = self._calculate_ma60(df).iloc[-1]
        if not (bbi > ma60):
            return False
        
        # 条件5: KDJ J值 < -5
        kdj_result = self.kdj_factor.compute(df)
        j_value = kdj_result["J"].iloc[-1]
        if not (j_value < -5):
            return False
        
        # 条件6: 成交量高于上一日
        if len(df) >= 2:
            current_volume = df.iloc[-1]["volume"]
            prev_volume = df.iloc[-2]["volume"]
            if not (current_volume > prev_volume):
                return False

        return True
    
    def should_exit(self, bar, current_pos: int) -> bool:
        """判断是否应该出场"""
        symbol = bar.symbol
        bars_held = self.bars_since_entry.get(symbol, 0)

        # 更新持仓期间最高价
        peak = self._highest_price.get(symbol, 0.0)
        if bar.high > peak:
            self._highest_price[symbol] = bar.high
            peak = bar.high

        # 计算从最高点的回撤比例
        drawdown = (peak - bar.close) / peak if peak > 0 else 0.0

        # 条件1: 长上影线 — 上影线 > 实体 且收盘价较昨日跌超 1%
        upper_shadow = bar.high - bar.close
        lower_part = bar.close - bar.low
        if upper_shadow > lower_part:
            hist = self.get_history_df(count=2, symbol=symbol)
            if hist is not None and len(hist) >= 2:
                prev_close = hist.iloc[-2]["close"]
                if (bar.close - prev_close) / prev_close < -0.01:
                    return True

        # 条件2: 止损 — 价格跌破入场前一日最低价向下 2%
        stop_price = self._stop_loss_price.get(symbol)
        if stop_price is not None and bar.low <= stop_price:
            return True

        # 条件3: 止盈 — BBI 上方出现 2 次涨幅 > 3% 后，从最高点回撤 3%
        df = self.get_history_df(count=2, symbol=symbol)
        if df is not None and len(df) >= 2:
            prev_close = df.iloc[-2]["close"]
            daily_return = (bar.close - prev_close) / prev_close

            # 计算当前 BBI
            hist = self.get_history_df(count=30, symbol=symbol)
            if hist is not None and len(hist) >= 24:
                bbi = self._calculate_bbi(hist).iloc[-1]
                above_bbi = bar.close > bbi
            else:
                above_bbi = False

            # BBI 上方且涨幅 > 3% 计数
            if above_bbi and daily_return > 0.03:
                count = self._surge_count.get(symbol, 0)
                self._surge_count[symbol] = count + 1
                # 达到 2 次后激活止盈跟踪，记录激活后的峰值
                if count + 1 >= 2 and not self._take_profit_active.get(symbol, False):
                    self._take_profit_active[symbol] = True
                    self._surge_peak[symbol] = bar.high

            # 止盈已激活后，跟踪峰值回撤
            if self._take_profit_active.get(symbol, False):
                if bar.high > self._surge_peak.get(symbol, 0):
                    self._surge_peak[symbol] = bar.high
                tp_drawdown = (self._surge_peak[symbol] - bar.close) / self._surge_peak[symbol]
                if tp_drawdown >= self.take_profit_drawdown:
                    return True

        # 条件4: 时间止损 — 持有超过 N 天且从高点回撤 >= time_stop_drawdown
        if bars_held >= self.hold_days and drawdown >= self.time_stop_drawdown:
            return True

        return False
    
    def do_entry(self, bar, quantity: int = 100):
        """执行入场，止损价 = 买入前一日最低价向下 2%"""
        df = self.get_history_df(count=2, symbol=bar.symbol)
        if df is not None and len(df) >= 2:
            prev_low = df.iloc[-2]["low"]
            self._stop_loss_price[bar.symbol] = prev_low * (1 - self.stop_loss_pct)
        self._highest_price[bar.symbol] = bar.high
        self._surge_count[bar.symbol] = 0
        self._surge_peak[bar.symbol] = 0.0
        self._take_profit_active[bar.symbol] = False
        super().do_entry(bar, quantity)

    def do_exit(self, bar):
        """执行出场"""
        self._stop_loss_price.pop(bar.symbol, None)
        self._highest_price.pop(bar.symbol, None)
        self._surge_count.pop(bar.symbol, None)
        self._surge_peak.pop(bar.symbol, None)
        self._take_profit_active.pop(bar.symbol, None)
        super().do_exit(bar)