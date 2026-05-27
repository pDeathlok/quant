#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alpha101 因子模块

基于 WorldQuant Alpha101 体系实现的量价因子
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional

from .base import Factor, RollingFactor


class Alpha001Factor(RollingFactor):
    """Alpha001: 基于标准差和价格的非线性变换"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
        returns = df['close'].pct_change()
        std = returns.rolling(self.window).std()
        
        # 条件判断：如果收益为负，使用标准差；否则使用收盘价
        cond = returns < 0
        value = np.where(cond, std, df['close'])
        
        # 平方
        value_sq = value ** 2
        
        # 5天内的最大值位置
        argmax_5 = pd.Series(value_sq, index=df.index).rolling(5).apply(np.argmax)
        
        # 排名并减0.5
        result = argmax_5.rank(pct=True) - 0.5
        
        return result


class Alpha002Factor(RollingFactor):
    """Alpha002: 成交量变化率与价格变化率的相关性"""
    
    def __init__(self, window: int = 6):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
        # 使用标准字段名（数据已通过适配器自动适配）
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1)
        
        # 计算log成交量的2日变化（使用标准字段 volume）
        log_vol = np.log(df['volume'])
        delta_log_vol = log_vol.diff(2)
        
        # 计算开盘收益率
        open_ret = (df['close'] - df['open']) / df['open']
        
        # 排名
        rank_vol = delta_log_vol.rank(pct=True)
        rank_ret = open_ret.rank(pct=True)
        
        # 计算滚动相关性
        def corr_func(x):
            if len(x) < self.window:
                return np.nan
            return stats.pearsonr(x[:, 0], x[:, 1])[0]
        
        combined = pd.DataFrame({'vol': rank_vol, 'ret': rank_ret})
        result = combined.rolling(self.window).apply(
            lambda x: stats.pearsonr(x[:, 0], x[:, 1])[0] if len(x) >= self.window else np.nan,
            raw=True
        )['vol']
        
        return -result


class Alpha003Factor(RollingFactor):
    """Alpha003: 开盘价与成交量排名的协方差"""
    
    def __init__(self, window: int = 5):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(-1 * rank(covariance(rank(open), rank(volume), 5)))"""
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1)
        
        # 排名（使用标准字段 volume）
        rank_open = df['open'].rank(pct=True)
        rank_vol = df['volume'].rank(pct=True)
        
        # 计算滚动协方差
        cov = pd.DataFrame({'open': rank_open, 'vol': rank_vol}).rolling(self.window).cov()
        
        # 获取协方差值
        cov_values = cov.loc[(slice(None), 'open'), 'vol']
        cov_values.index = cov_values.index.droplevel(1)
        
        return -cov_values.rank(pct=True)


class Alpha004Factor(Factor):
    """Alpha004: 成交量和价格变化的时序排名乘积"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(rank(ts_rank(volume, 3) * ts_rank((-1 * delta(close, 1)), 3)))"""
        # 3天内成交量的时序排名（使用标准字段 volume）
        ts_rank_vol = df['volume'].rolling(3).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # 3天内负收益的时序排名
        neg_ret = -df['close'].diff(1)
        ts_rank_ret = neg_ret.rolling(3).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # 乘积并排名
        return (ts_rank_vol * ts_rank_ret).rank(pct=True)


class Alpha005Factor(RollingFactor):
    """Alpha005: VWAP偏离因子"""
    
    def __init__(self, window: int = 10):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1)
        
        # 计算VWAP（简化版）
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
        
        # 10天VWAP均值
        vwap_mean = df['vwap'].rolling(self.window).mean()
        
        # 开盘价与VWAP均值的差
        open_diff = df['open'] - vwap_mean
        
        # 收盘价与VWAP的差
        close_diff = df['close'] - df['vwap']
        
        return open_diff.rank(pct=True) * (-1 * close_diff.abs().rank(pct=True))


class Alpha006Factor(RollingFactor):
    """Alpha006: 平均开盘收益"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(-1 * rank((sum((open - close), 20) / 20)))"""
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1)
        
        return -((df['open'] - df['close']).rolling(self.window).mean()).rank(pct=True)


class Alpha007Factor(RollingFactor):
    """Alpha007: 最高价与成交量的相关性"""
    
    def __init__(self, window: int = 5):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(rank(correlation(high, volume, 5)))"""
        def corr_func(x):
            if len(x) < self.window:
                return np.nan
            return stats.pearsonr(x[:, 0], x[:, 1])[0]
        
        # 使用标准字段 volume
        combined = pd.DataFrame({'high': df['high'], 'vol': df['volume']})
        corr = combined.rolling(self.window).apply(
            lambda x: stats.pearsonr(x[:, 0], x[:, 1])[0] if len(x) >= self.window else np.nan,
            raw=True
        )['high']
        
        return corr.rank(pct=True)


class Alpha008Factor(RollingFactor):
    """Alpha008: 开盘收益与成交量波动的组合"""
    
    def __init__(self, window: int = 20):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """(rank((sum(((close - open) / open), 20)) * rank((stddev(volume, 20) / mean(volume, 20)))))"""
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1)
        
        # 开盘收益率之和
        open_ret_sum = (df['close'] - df['open']) / df['open']
        open_ret_sum = open_ret_sum.rolling(self.window).sum()
        
        # 成交量波动率与均值的比（使用标准字段 volume）
        vol_std = df['volume'].rolling(self.window).std()
        vol_mean = df['volume'].rolling(self.window).mean()
        vol_ratio = vol_std / vol_mean.replace(0, np.nan)
        
        return (open_ret_sum.rank(pct=True) * vol_ratio.rank(pct=True)).rank(pct=True)


class Alpha009Factor(Factor):
    """Alpha009: 成交量变化和价格变化的排名乘积"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """((-1) * rank(rank(delta(volume, 1)) * rank(-1 * delta(close, 1))))"""
        # 使用标准字段 volume
        vol_delta = df['volume'].diff(1)
        close_delta = -df['close'].diff(1)
        
        return -(vol_delta.rank(pct=True) * close_delta.rank(pct=True)).rank(pct=True)


class Alpha010Factor(Factor):
    """Alpha010: 成交量的时序排名"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """((-1) * Ts_Rank(rank(volume), 3))"""
        # 使用标准字段 volume
        rank_vol = df['volume'].rank(pct=True)
        ts_rank = rank_vol.rolling(3).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        return -ts_rank


# 导出因子类
__all__ = [
    'Alpha001Factor',
    'Alpha002Factor',
    'Alpha003Factor',
    'Alpha004Factor',
    'Alpha005Factor',
    'Alpha006Factor',
    'Alpha007Factor',
    'Alpha008Factor',
    'Alpha009Factor',
    'Alpha010Factor'
]