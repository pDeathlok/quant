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


class B1Strategy(BaseStrategy):
    """
    B1 策略实现
    
    入场条件：
    - 股票名称不包含 'ST'
    - 当日涨跌幅: -2% <= pct_change <= +2%
    - 当日振幅: (high - low) / low < 7%
    - BBI > MA60
    - KDJ J值 < 10
    - 当日成交量 > 上一日成交量
    
    出场条件：
    - 持有超过 N 个交易日（默认 5 天）
    - 或 KDJ J值 > 80（超买）
    """
    
    def __init__(self, hold_days: int = 5):
        super().__init__(name="B1")
        self.hold_days = hold_days
        self.kdj_factor = KDJ()
        self._price_cache = {}  # 缓存历史价格数据
    
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
    
    def _is_st_stock(self, symbol: str, name: str = None) -> bool:
        """判断是否为 ST 股票"""
        # 从股票名称判断
        if name and ("ST" in name or "*ST" in name):
            return True
        # 从股票代码后缀判断（退市股）
        if symbol and ("ST" in symbol or "退" in symbol):
            return True
        return False
    
    def should_entry(self, bar, current_pos: int) -> bool:
        """判断是否应该入场"""
        # 获取股票名称（如果可用）
        stock_name = getattr(bar, "name", "")
        
        # 条件1: 股票不包含 ST
        if self._is_st_stock(bar.symbol, stock_name):
            return False
        
        # 获取历史数据
        history = self.get_history(bar.symbol, lookback=60)
        if history is None or len(history) < 60:
            return False
        
        # 转换为 DataFrame
        df = pd.DataFrame([{
            "date": h.date,
            "open": h.open,
            "high": h.high,
            "low": h.low,
            "close": h.close,
            "volume": h.volume,
            "pct_change": getattr(h, "pct_change", 0)
        } for h in history])
        
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
        
        # 条件5: KDJ J值 < 10
        kdj_result = self.kdj_factor.compute(df)
        j_value = kdj_result["J"].iloc[-1]
        if not (j_value < 10):
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
        bars_held = self.bars_since_entry.get(bar.symbol, 0)
        
        # 条件1: 持有超过指定天数
        if bars_held >= self.hold_days:
            return True
        
        # 获取历史数据计算 KDJ
        history = self.get_history(bar.symbol, lookback=20)
        if history is not None and len(history) >= 20:
            df = pd.DataFrame([{
                "date": h.date,
                "high": h.high,
                "low": h.low,
                "close": h.close
            } for h in history])
            
            kdj_result = self.kdj_factor.compute(df)
            j_value = kdj_result["J"].iloc[-1]
            
            # 条件2: KDJ J值 > 80（超买）
            if j_value > 80:
                return True
        
        return False
    
    def do_entry(self, bar, quantity: int = 100):
        """执行入场"""
        super().do_entry(bar, quantity)
    
    def do_exit(self, bar):
        """执行出场"""
        super().do_exit(bar)