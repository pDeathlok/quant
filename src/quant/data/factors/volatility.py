#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
波动率因子模块
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional

from .base import Factor, RollingFactor


class VolatilityFactor(RollingFactor):
    """波动率因子（标准差）"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算收益率标准差"""
        returns = df['close'].pct_change()
        return returns.rolling(self.window).std()


class DownsideVolatilityFactor(RollingFactor):
    """下行波动率因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算仅下跌期间的波动率"""
        returns = df['close'].pct_change()
        downside_returns = returns.where(returns < 0, 0)
        return downside_returns.rolling(self.window).std()


class IdiosyncraticVolatilityFactor(RollingFactor):
    """特质波动率因子"""
    
    def __init__(self, window: int = 60, market_returns: Optional[pd.Series] = None):
        super().__init__(window)
        self.market_returns = market_returns
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算特质波动率"""
        returns = df['close'].pct_change()
        
        if self.market_returns is not None:
            # 与市场回归后的残差波动率
            merged = pd.DataFrame({
                'stock_ret': returns,
                'market_ret': self.market_returns.reindex(returns.index)
            }).dropna()
            
            if len(merged) >= self.window:
                def resid_vol(window_df):
                    if len(window_df) < 10:
                        return np.nan
                    beta, alpha = np.polyfit(window_df['market_ret'], window_df['stock_ret'], 1)
                    residuals = window_df['stock_ret'] - (alpha + beta * window_df['market_ret'])
                    return residuals.std()
                
                return merged.rolling(self.window).apply(resid_vol)['stock_ret']
            else:
                return pd.Series(np.nan, index=returns.index)
        else:
            # 如果没有市场数据，返回普通波动率
            return returns.rolling(self.window).std()


class RealizedVolatilityFactor(RollingFactor):
    """已实现波动率（基于日内高低价）"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算已实现波动率"""
        # 使用 Garman-Klass 波动率估计
        high = df['high']
        low = df['low']
        close = df['close']
        open_ = df['open'] if 'open' in df.columns else df['close'].shift(1)
        
        # Garman-Klass 公式
        ret = 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_) ** 2
        
        return ret.rolling(self.window).mean()


class TailRiskFactor(RollingFactor):
    """尾部风险因子（VaR）"""
    
    def __init__(self, window: int = 60, confidence: float = 0.05):
        super().__init__(window)
        self.confidence = confidence
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算 VaR（Value at Risk）"""
        returns = df['close'].pct_change()
        
        def var_func(window_returns):
            if len(window_returns) < self.window:
                return np.nan
            return np.percentile(window_returns, self.confidence * 100)
        
        return returns.rolling(self.window).apply(var_func)


class CVaRFactor(RollingFactor):
    """条件尾部风险因子（CVaR）"""
    
    def __init__(self, window: int = 60, confidence: float = 0.05):
        super().__init__(window)
        self.confidence = confidence
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算 CVaR（Conditional Value at Risk）"""
        returns = df['close'].pct_change()
        
        def cvar_func(window_returns):
            if len(window_returns) < self.window:
                return np.nan
            var = np.percentile(window_returns, self.confidence * 100)
            return window_returns[window_returns <= var].mean()
        
        return returns.rolling(self.window).apply(cvar_func)


class KurtosisFactor(RollingFactor):
    """峰度因子"""
    
    def __init__(self, window: int = 60):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算收益峰度"""
        returns = df['close'].pct_change()
        return returns.rolling(self.window).kurt()


class SkewnessFactor(RollingFactor):
    """偏度因子"""
    
    def __init__(self, window: int = 60):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算收益偏度"""
        returns = df['close'].pct_change()
        return returns.rolling(self.window).skew()


# 导出因子类
__all__ = [
    'VolatilityFactor',
    'DownsideVolatilityFactor',
    'IdiosyncraticVolatilityFactor',
    'RealizedVolatilityFactor',
    'TailRiskFactor',
    'CVaRFactor',
    'KurtosisFactor',
    'SkewnessFactor'
]