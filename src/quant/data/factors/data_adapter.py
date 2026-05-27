#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子数据适配器模块

用于统一不同数据源（Tushare、AKShare）的字段名称，确保因子计算的兼容性
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List


class FactorDataAdapter:
    """
    因子数据适配器
    
    将不同数据源的字段统一为标准字段名，便于因子计算
    支持自动识别数据源类型
    """
    
    # Tushare 特征字段标识
    TUSHARE_MARKERS = ['ts_code', 'trade_date', 'vol', 'amount', 'pct_chg', 'pre_close']
    
    # AKShare 特征字段标识  
    AKSHARE_MARKERS = ['股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额']
    
    # 字段映射表 - 扩展版
    DEFAULT_FIELD_MAPPING = {
        # Tushare 到标准字段
        'ts_code': 'symbol',
        'trade_date': 'date',
        'vol': 'volume',
        'amount': 'turnover',
        'pre_close': 'prev_close',
        'pct_chg': 'pct_change',
        
        # AKShare 到标准字段
        '股票代码': 'symbol',
        '日期': 'date',
        '开盘': 'open',
        '开盘价': 'open',
        '收盘': 'close',
        '收盘价': 'close',
        '最高': 'high',
        '最高价': 'high',
        '最低': 'low',
        '最低价': 'low',
        '成交量': 'volume',
        '成交额': 'turnover',
        '前收盘': 'prev_close',
        '涨跌幅': 'pct_change',
        
        # 其他常见字段名
        'adj_close': 'adj_close',
        'vwap': 'vwap',
        'open_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'close_price': 'close',
        'prev_close_price': 'prev_close',
        'volume_share': 'volume',
        'turnover_amt': 'turnover',
        'return_pct': 'pct_change',
    }
    
    # 标准字段列表
    STANDARD_FIELDS = [
        'symbol', 'date', 'open', 'high', 'low', 'close', 
        'prev_close', 'volume', 'turnover', 'pct_change'
    ]
    
    # 价格相关字段
    PRICE_FIELDS = ['open', 'high', 'low', 'close', 'prev_close']
    
    @classmethod
    def detect_source(cls, df: pd.DataFrame) -> str:
        """
        自动检测数据源类型
        
        Args:
            df: 输入数据
            
        Returns:
            数据源类型 ('tushare', 'akshare', 'unknown')
        """
        columns = set(df.columns.tolist())
        
        # 检查 Tushare 特征
        tushare_count = sum(1 for marker in cls.TUSHARE_MARKERS if marker in columns)
        
        # 检查 AKShare 特征
        akshare_count = sum(1 for marker in cls.AKSHARE_MARKERS if marker in columns)
        
        if tushare_count >= 3:
            return 'tushare'
        elif akshare_count >= 3:
            return 'akshare'
        else:
            return 'unknown'
    
    @classmethod
    def adapt(cls, df: pd.DataFrame, source: Optional[str] = None, 
              mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        适配数据到标准格式
        
        Args:
            df: 原始数据 DataFrame
            source: 数据源类型 ('tushare', 'akshare', 'custom', None)
                   如果为 None，自动检测
            mapping: 自定义字段映射
            
        Returns:
            适配后的 DataFrame，包含标准字段
        """
        df = df.copy()
        
        # 自动检测数据源类型
        if source is None:
            source = cls.detect_source(df)
        
        # 使用默认映射或自定义映射
        field_mapping = mapping or cls.DEFAULT_FIELD_MAPPING
        
        # 根据数据源类型进行特殊处理
        if source == 'tushare':
            return cls._adapt_tushare(df, field_mapping)
        elif source == 'akshare':
            return cls._adapt_akshare(df, field_mapping)
        else:
            return cls._adapt_generic(df, field_mapping)
    
    @classmethod
    def _adapt_tushare(cls, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """适配 Tushare 数据"""
        result = df.copy()
        
        # 重命名字段
        for old_name, new_name in mapping.items():
            if old_name in result.columns and new_name not in result.columns:
                result = result.rename(columns={old_name: new_name})
        
        # 处理 Tushare 特有字段转换
        if 'amount' in df.columns and 'turnover' not in result.columns:
            # Tushare 的 amount 是万元，转换为元
            result['turnover'] = df['amount'] * 10000
        
        # 转换日期格式
        if 'date' in result.columns:
            try:
                result['date'] = pd.to_datetime(result['date'], format='%Y%m%d')
            except:
                result['date'] = pd.to_datetime(result['date'])
        
        # 确保标准字段存在
        result = cls.ensure_standard_fields(result)
        
        return result
    
    @classmethod
    def _adapt_akshare(cls, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """适配 AKShare 数据"""
        result = df.copy()
        
        # 重命名字段
        for old_name, new_name in mapping.items():
            if old_name in result.columns and new_name not in result.columns:
                result = result.rename(columns={old_name: new_name})
        
        # 转换日期格式
        if 'date' in result.columns:
            result['date'] = pd.to_datetime(result['date'])
        
        # 确保标准字段存在
        result = cls.ensure_standard_fields(result)
        
        return result
    
    @classmethod
    def _adapt_generic(cls, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """通用适配"""
        result = df.copy()
        
        # 重命名字段
        for old_name, new_name in mapping.items():
            if old_name in result.columns and new_name not in result.columns:
                result = result.rename(columns={old_name: new_name})
        
        # 确保标准字段存在
        result = cls.ensure_standard_fields(result)
        
        return result
    
    @classmethod
    def ensure_standard_fields(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        确保 DataFrame 包含所有标准字段
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            包含所有标准字段的 DataFrame
        """
        result = df.copy()
        
        # 添加缺失的标准字段
        for field in cls.STANDARD_FIELDS:
            if field not in result.columns:
                if field == 'volume':
                    # 尝试从多种可能的字段获取
                    for alias in ['vol', '成交量', 'volume_share']:
                        if alias in result.columns:
                            result['volume'] = result[alias]
                            break
                elif field == 'turnover':
                    for alias in ['amount', '成交额', 'turnover_amt']:
                        if alias in result.columns:
                            result['turnover'] = result[alias]
                            break
                elif field == 'prev_close':
                    if 'close' in result.columns:
                        result['prev_close'] = result['close'].shift(1)
                    elif 'pre_close' in result.columns:
                        result['prev_close'] = result['pre_close']
                    elif '前收盘' in result.columns:
                        result['prev_close'] = result['前收盘']
                elif field == 'pct_change':
                    if 'close' in result.columns:
                        result['pct_change'] = result['close'].pct_change() * 100
                    elif 'pct_chg' in result.columns:
                        result['pct_change'] = result['pct_chg']
                    elif '涨跌幅' in result.columns:
                        result['pct_change'] = result['涨跌幅']
                elif field == 'date':
                    for alias in ['trade_date', '日期']:
                        if alias in result.columns:
                            result['date'] = pd.to_datetime(result[alias])
                            break
                elif field == 'symbol':
                    for alias in ['ts_code', '股票代码']:
                        if alias in result.columns:
                            result['symbol'] = result[alias]
                            break
        
        return result
    
    @classmethod
    def get_available_fields(cls, df: pd.DataFrame) -> List[str]:
        """
        获取数据中可用的标准字段
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            可用的标准字段列表
        """
        adapted = cls.adapt(df)
        return [f for f in cls.STANDARD_FIELDS if f in adapted.columns and not adapted[f].isna().all()]


class FactorDataLoader:
    """
    因子数据加载器
    
    统一加载不同数据源的数据并适配为标准格式
    """
    
    @staticmethod
    def load_from_tushare(file_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        从 Tushare 数据文件加载数据
        
        Args:
            file_path: 文件路径或 DataFrame
            
        Returns:
            适配后的标准格式 DataFrame
        """
        if isinstance(file_path, pd.DataFrame):
            df = file_path
        else:
            # 支持多种文件格式
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
                df = pd.read_pickle(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
        
        return FactorDataAdapter.adapt(df, source='tushare')
    
    @staticmethod
    def load_from_akshare(file_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        从 AKShare 数据文件加载数据
        
        Args:
            file_path: 文件路径或 DataFrame
            
        Returns:
            适配后的标准格式 DataFrame
        """
        if isinstance(file_path, pd.DataFrame):
            df = file_path
        else:
            # 支持多种文件格式
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
                df = pd.read_pickle(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
        
        return FactorDataAdapter.adapt(df, source='akshare')
    
    @staticmethod
    def load_from_custom(file_path: Union[str, pd.DataFrame], 
                         mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        从自定义数据源加载数据
        
        Args:
            file_path: 文件路径或 DataFrame
            mapping: 自定义字段映射
            
        Returns:
            适配后的标准格式 DataFrame
        """
        if isinstance(file_path, pd.DataFrame):
            df = file_path
        else:
            # 支持多种文件格式
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
                df = pd.read_pickle(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
        
        return FactorDataAdapter.adapt(df, source='custom', mapping=mapping)
    
    @staticmethod
    def load_auto(file_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        自动检测数据源并加载数据
        
        Args:
            file_path: 文件路径或 DataFrame
            
        Returns:
            适配后的标准格式 DataFrame
        """
        if isinstance(file_path, pd.DataFrame):
            df = file_path
        else:
            # 支持多种文件格式
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
                df = pd.read_pickle(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
        
        return FactorDataAdapter.adapt(df, source=None)


class FactorDataLoader:
    """
    因子数据加载器
    
    统一加载不同数据源的数据并适配为标准格式
    """
    
    @staticmethod
    def load_from_tushare(file_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        从 Tushare 数据文件加载数据
        
        Args:
            file_path: 文件路径或 DataFrame
            
        Returns:
            适配后的标准格式 DataFrame
        """
        if isinstance(file_path, pd.DataFrame):
            df = file_path
        else:
            df = pd.read_parquet(file_path)
        
        return FactorDataAdapter.adapt(df, source='tushare')
    
    @staticmethod
    def load_from_akshare(file_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        从 AKShare 数据文件加载数据
        
        Args:
            file_path: 文件路径或 DataFrame
            
        Returns:
            适配后的标准格式 DataFrame
        """
        if isinstance(file_path, pd.DataFrame):
            df = file_path
        else:
            df = pd.read_parquet(file_path)
        
        return FactorDataAdapter.adapt(df, source='akshare')
    
    @staticmethod
    def load_from_custom(file_path: Union[str, pd.DataFrame], 
                         mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        从自定义数据源加载数据
        
        Args:
            file_path: 文件路径或 DataFrame
            mapping: 自定义字段映射
            
        Returns:
            适配后的标准格式 DataFrame
        """
        if isinstance(file_path, pd.DataFrame):
            df = file_path
        else:
            df = pd.read_parquet(file_path)
        
        return FactorDataAdapter.adapt(df, source='custom', mapping=mapping)
