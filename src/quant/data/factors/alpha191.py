#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alpha191 因子模块

基于国泰君安 Alpha191 体系实现的量价因子，更适合 A 股市场
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

from .base import Factor, RollingFactor


class Alpha191_01Factor(Factor):
    """Alpha191_01: 收盘价相对开盘价的变化率"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(close - open) / open"""
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1)
        
        return (df['close'] - df['open']) / df['open']


class Alpha191_02Factor(Factor):
    """Alpha191_02: 最高价相对开盘价的变化率"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(high - open) / open"""
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1)
        
        return (df['high'] - df['open']) / df['open']


class Alpha191_03Factor(Factor):
    """Alpha191_03: 最低价相对开盘价的变化率"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(low - open) / open"""
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1)
        
        return (df['low'] - df['open']) / df['open']


class Alpha191_04Factor(RollingFactor):
    """Alpha191_04: 成交量相对均值的变化率"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """volume / mean(volume, 20)"""
        # 使用标准字段 volume
        return df['volume'] / df['volume'].rolling(self.window).mean()


class Alpha191_05Factor(RollingFactor):
    """Alpha191_05: 成交额相对均值的变化率"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """turnover / mean(turnover, 20)"""
        # 使用标准字段 turnover
        if 'turnover' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        return df['turnover'] / df['turnover'].rolling(self.window).mean()


class Alpha191_06Factor(Factor):
    """Alpha191_06: 最高价相对收盘价的变化率"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(high - close) / close"""
        return (df['high'] - df['close']) / df['close']


class Alpha191_07Factor(Factor):
    """Alpha191_07: 最低价相对收盘价的变化率"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(low - close) / close"""
        return (df['low'] - df['close']) / df['close']


class Alpha191_08Factor(Factor):
    """Alpha191_08: 开盘价相对前收盘价的变化率"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(open - prev_close) / prev_close"""
        # 使用标准字段 prev_close
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1)
        if 'prev_close' not in df.columns:
            df['prev_close'] = df['close'].shift(1)
        
        return (df['open'] - df['prev_close']) / df['prev_close']


class Alpha191_09Factor(RollingFactor):
    """Alpha191_09: 5日平均最高价相对收盘价"""
    
    def __init__(self, window: int = 5):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """mean(high, 5) / close"""
        return df['high'].rolling(self.window).mean() / df['close']


class Alpha191_10Factor(RollingFactor):
    """Alpha191_10: 5日平均最低价相对收盘价"""
    
    def __init__(self, window: int = 5):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """mean(low, 5) / close"""
        return df['low'].rolling(self.window).mean() / df['close']


class Alpha191_11Factor(RollingFactor):
    """Alpha191_11: 5日平均收盘价相对开盘价"""
    
    def __init__(self, window: int = 5):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """mean(close, 5) / open"""
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1)
        
        return df['close'].rolling(self.window).mean() / df['open']


class Alpha191_12Factor(Factor):
    """Alpha191_12: 当日振幅"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(high - low) / close"""
        return (df['high'] - df['low']) / df['close']


class Alpha191_13Factor(RollingFactor):
    """Alpha191_13: 5日平均振幅"""
    
    def __init__(self, window: int = 5):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """mean((high - low) / close, 5)"""
        amplitude = (df['high'] - df['low']) / df['close']
        return amplitude.rolling(self.window).mean()


class Alpha191_14Factor(RollingFactor):
    """Alpha191_14: 成交量相对前5日均量的变化率"""
    
    def __init__(self, window: int = 5):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """volume / mean(volume.shift(1), 5)"""
        # 使用标准字段 volume
        return df['volume'] / df['volume'].shift(1).rolling(self.window).mean()


class Alpha191_15Factor(Factor):
    """Alpha191_15: 收盘价相对5日均线的偏离"""
    
    def __init__(self, window: int = 5):
        self.window = window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """close / mean(close, 5)"""
        return df['close'] / df['close'].rolling(self.window).mean()


# 导出因子类
__all__ = [
    'Alpha191_01Factor',
    'Alpha191_02Factor',
    'Alpha191_03Factor',
    'Alpha191_04Factor',
    'Alpha191_05Factor',
    'Alpha191_06Factor',
    'Alpha191_07Factor',
    'Alpha191_08Factor',
    'Alpha191_09Factor',
    'Alpha191_10Factor',
    'Alpha191_11Factor',
    'Alpha191_12Factor',
    'Alpha191_13Factor',
    'Alpha191_14Factor',
    'Alpha191_15Factor'
]