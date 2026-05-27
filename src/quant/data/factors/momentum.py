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
        
        # 获取索引名称
        index_names = df.index.names
        
        # 合并行业信息（支持标准字段 symbol 和 Tushare 字段 ts_code）
        df_reset = df.reset_index()
        key_col = 'symbol' if 'symbol' in df_reset.columns else 'ts_code'
        industry_key = 'symbol' if 'symbol' in industry_df.columns else 'ts_code'
        
        merged = df_reset.merge(
            industry_df.rename(columns={industry_key: key_col})[[key_col, 'industry']], 
            on=key_col
        )
        
        # 计算每个股票的动量
        merged['momentum'] = merged.groupby(key_col)['close'].pct_change(self.window)
        
        # 行业内排名（支持标准字段 date 和 Tushare 字段 trade_date）
        date_col = 'date' if 'date' in merged.columns else 'trade_date'
        merged['industry_momentum'] = merged.groupby(['industry', date_col])['momentum'].rank(pct=True)
        
        # 恢复原始索引
        return merged.set_index(index_names)['industry_momentum']


class CrossSectionalMomentumFactor(Factor):
    """横截面动量因子"""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算横截面动量（用于多股票）"""
        # 支持标准字段 symbol 和 Tushare 字段 ts_code
        symbol_col = 'symbol' if 'symbol' in df.columns else ('ts_code' if 'ts_code' in df.columns else None)
        
        if symbol_col is None:
            return df['close'].pct_change(self.window)
        
        # 计算每个股票的动量
        df = df.copy()
        df['momentum'] = df.groupby(symbol_col)['close'].pct_change(self.window)
        
        # 横截面排名（支持标准字段 date 和 Tushare 字段 trade_date）
        date_col = 'date' if 'date' in df.columns else ('trade_date' if 'trade_date' in df.columns else None)
        
        if date_col is not None:
            df['cs_momentum'] = df.groupby(date_col)['momentum'].rank(pct=True)
        else:
            df['cs_momentum'] = df['momentum'].rank(pct=True)
        
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