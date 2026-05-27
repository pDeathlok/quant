#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动量/反转因子模块
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional

from .base import Factor, RollingFactor


class ReturnFactor(Factor):
    """收益率因子"""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算收益率"""
        return df['close'].pct_change(self.window)


class MomentumFactor(RollingFactor):
    """动量因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算动量因子"""
        return df['close'].pct_change(self.window)


class MomentumSkip5Factor(RollingFactor):
    """跳过最近5天的动量因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算跳过最近5天的动量"""
        return (df['close'].shift(5) / df['close'].shift(25)) - 1


class RiskAdjustedMomentumFactor(RollingFactor):
    """风险调整动量因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算风险调整动量 = 收益 / 波动率"""
        returns = df['close'].pct_change()
        momentum = returns.rolling(self.window).mean() * self.window
        volatility = returns.rolling(self.window).std() * np.sqrt(self.window)
        return momentum / volatility.replace(0, np.nan)


class ReversalFactor(RollingFactor):
    """反转因子"""
    
    def __init__(self, window: int = 1):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算反转因子"""
        return -df['close'].pct_change(self.window)


class IndustryMomentumFactor(Factor):
    """行业内动量因子"""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def compute(self, df: pd.DataFrame, industry_df: pd.DataFrame = None) -> pd.Series:
        """计算行业内动量"""
        if industry_df is None:
            raise ValueError("需要提供行业信息")
        
        # 合并行业信息
        merged = df.reset_index().merge(industry_df[['ts_code', 'industry']], on='ts_code')
        
        # 计算每个股票的动量
        merged['momentum'] = merged.groupby('ts_code')['close'].pct_change(self.window)
        
        # 行业内排名
        merged['industry_momentum'] = merged.groupby(['industry', 'trade_date'])['momentum'].rank(pct=True)
        
        return merged.set_index(['ts_code', 'trade_date'])['industry_momentum']


class CrossSectionalMomentumFactor(Factor):
    """横截面动量因子"""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算横截面动量（用于多股票）"""
        if 'ts_code' not in df.columns:
            return df['close'].pct_change(self.window)
        
        # 计算每个股票的动量
        df = df.copy()
        df['momentum'] = df.groupby('ts_code')['close'].pct_change(self.window)
        
        # 横截面排名
        df['cs_momentum'] = df.groupby('trade_date')['momentum'].rank(pct=True)
        
        return df['cs_momentum']


class PriceTrendFactor(RollingFactor):
    """价格趋势因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算价格趋势（线性回归斜率）"""
        close = df['close']
        x = np.arange(self.window)
        
        def slope_func(y):
            if len(y) < self.window:
                return np.nan
            return stats.linregress(x, y)[0]
        
        return close.rolling(self.window).apply(slope_func)


class TrendStrengthFactor(RollingFactor):
    """趋势强度因子（R²）"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算趋势强度"""
        close = df['close']
        x = np.arange(self.window)
        
        def r2_func(y):
            if len(y) < self.window:
                return np.nan
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            return r_value ** 2
        
        return close.rolling(self.window).apply(r2_func)


class MovingAverageSlopeFactor(RollingFactor):
    """均线斜率因子"""
    
    def __init__(self, ma_window: int = 20, slope_window: int = 5):
        super().__init__(ma_window)
        self.slope_window = slope_window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算均线斜率"""
        ma = df['close'].rolling(self.window).mean()
        x = np.arange(self.slope_window)
        
        def slope_func(y):
            if len(y) < self.slope_window:
                return np.nan
            return stats.linregress(x, y)[0]
        
        return ma.rolling(self.slope_window).apply(slope_func)


# 导出因子类
__all__ = [
    'ReturnFactor',
    'MomentumFactor',
    'MomentumSkip5Factor',
    'RiskAdjustedMomentumFactor',
    'ReversalFactor',
    'IndustryMomentumFactor',
    'CrossSectionalMomentumFactor',
    'PriceTrendFactor',
    'TrendStrengthFactor',
    'MovingAverageSlopeFactor'
]