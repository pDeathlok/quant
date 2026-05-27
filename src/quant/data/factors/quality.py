#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
质量因子模块
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

from .base import Factor, RollingFactor


class EarningsQualityFactor(Factor):
    """盈利质量因子"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame, cashflow_df: pd.DataFrame = None) -> pd.Series:
        """计算盈利质量 = 经营现金流 / 净利润"""
        if cashflow_df is not None:
            # 合并现金流数据
            merged = df.reset_index().merge(
                cashflow_df[['ts_code', 'end_date', 'n_cashflow_act']],
                left_on=['ts_code', 'trade_date'],
                right_on=['ts_code', 'end_date'],
                how='left'
            ).set_index(df.index.names)
            
            # 计算盈利质量
            merged['earnings_quality'] = merged['n_cashflow_act'] / merged['n_income'].replace(0, np.nan)
            return merged['earnings_quality']
        else:
            # 如果没有现金流数据，返回NaN
            return pd.Series(np.nan, index=df.index)


class ROEStabilityFactor(RollingFactor):
    """ROE稳定性因子"""
    
    def __init__(self, window: int = 12):
        super().__init__(window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算ROE的标准差（越小越稳定）"""
        if 'roe' in df.columns:
            return df['roe'].rolling(self.window).std()
        else:
            return pd.Series(np.nan, index=df.index)


class CashFlowCoverageFactor(Factor):
    """现金流覆盖率因子"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame, balance_df: pd.DataFrame = None) -> pd.Series:
        """计算现金流覆盖率 = 经营现金流 / 流动负债"""
        if balance_df is not None and 'n_cashflow_act' in df.columns:
            merged = df.reset_index().merge(
                balance_df[['ts_code', 'end_date', 'total_liab']],
                left_on=['ts_code', 'trade_date'],
                right_on=['ts_code', 'end_date'],
                how='left'
            ).set_index(df.index.names)
            
            merged['cashflow_coverage'] = merged['n_cashflow_act'] / merged['total_liab'].replace(0, np.nan)
            return merged['cashflow_coverage']
        else:
            return pd.Series(np.nan, index=df.index)


class OperatingCashFlowFactor(Factor):
    """经营现金流因子"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """返回经营现金流"""
        if 'n_cashflow_act' in df.columns:
            return df['n_cashflow_act']
        else:
            return pd.Series(np.nan, index=df.index)


class FreeCashFlowFactor(Factor):
    """自由现金流因子"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算自由现金流 = 经营现金流 - 资本开支"""
        if 'n_cashflow_act' in df.columns and 'c_pay_acq_const_fiolta' in df.columns:
            return df['n_cashflow_act'] - df['c_pay_acq_const_fiolta']
        else:
            return pd.Series(np.nan, index=df.index)


class ProfitabilityFactor(Factor):
    """盈利能力综合因子"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算盈利能力综合得分"""
        factors = []
        
        if 'roe' in df.columns:
            factors.append(df['roe'].rank(pct=True))
        if 'roa' in df.columns:
            factors.append(df['roa'].rank(pct=True))
        if 'grossprofit_margin' in df.columns:
            factors.append(df['grossprofit_margin'].rank(pct=True))
        
        if factors:
            return pd.concat(factors, axis=1).mean(axis=1)
        else:
            return pd.Series(np.nan, index=df.index)


class QualityScoreFactor(Factor):
    """质量得分因子"""
    
    def __init__(self):
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算综合质量得分"""
        score = pd.Series(0.0, index=df.index)
        weight = 0
        
        # 盈利能力
        if 'roe' in df.columns:
            score += df['roe'].rank(pct=True) * 0.2
            weight += 0.2
        
        # 盈利质量
        if 'earnings_quality' in df.columns:
            score += df['earnings_quality'].rank(pct=True) * 0.2
            weight += 0.2
        
        # 杠杆水平（越低越好）
        if 'debt_to_assets' in df.columns:
            score += (1 - df['debt_to_assets'].rank(pct=True)) * 0.2
            weight += 0.2
        
        # 现金流
        if 'n_cashflow_act' in df.columns:
            score += df['n_cashflow_act'].rank(pct=True) * 0.2
            weight += 0.2
        
        # ROE稳定性（标准差越小越好）
        if 'roe_std' in df.columns:
            score += (1 - df['roe_std'].rank(pct=True)) * 0.2
            weight += 0.2
        
        if weight > 0:
            return score / weight
        else:
            return pd.Series(np.nan, index=df.index)


# 导出因子类
__all__ = [
    'EarningsQualityFactor',
    'ROEStabilityFactor',
    'CashFlowCoverageFactor',
    'OperatingCashFlowFactor',
    'FreeCashFlowFactor',
    'ProfitabilityFactor',
    'QualityScoreFactor'
]