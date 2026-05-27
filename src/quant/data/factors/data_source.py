#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据源配置模块

支持动态切换数据源，统一管理 Tushare 和 AKShare 的数据获取
"""

import pandas as pd
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field

# 尝试导入 tushare 和 akshare（如果安装的话）
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


@dataclass
class DataSourceConfig:
    """
    数据源配置类
    
    统一管理不同数据源的配置参数
    """
    # 默认数据源类型
    default_source: str = 'tushare'
    
    # Tushare 配置
    tushare_token: Optional[str] = None
    tushare_pro: bool = True
    
    # AKShare 配置
    akshare_timeout: int = 30
    
    # 数据缓存配置
    cache_enabled: bool = True
    cache_dir: str = './data/cache'
    
    # 字段映射配置
    field_mapping: Dict[str, str] = field(default_factory=dict)
    
    # 数据格式配置
    date_format: str = '%Y%m%d'
    price_precision: int = 2


class DataSourceManager:
    """
    数据源管理器
    
    提供统一的数据获取接口，支持动态切换数据源
    """
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        """
        初始化数据源管理器
        
        Args:
            config: 数据源配置
        """
        self.config = config or DataSourceConfig()
        self._tushare_api = None
        self._init_tushare()
    
    def _init_tushare(self):
        """初始化 Tushare API"""
        if TUSHARE_AVAILABLE and self.config.tushare_token:
            if self.config.tushare_pro:
                self._tushare_api = ts.pro_api(self.config.tushare_token)
            else:
                ts.set_token(self.config.tushare_token)
                self._tushare_api = ts.pro_api()
    
    def get_stock_daily(self, ts_code: str, start_date: str = None, 
                        end_date: str = None, source: str = None) -> pd.DataFrame:
        """
        获取股票日线数据
        
        Args:
            ts_code: 股票代码（格式：000001.SZ）
            start_date: 开始日期（格式：YYYYMMDD）
            end_date: 结束日期（格式：YYYYMMDD）
            source: 数据源类型（'tushare', 'akshare'），如果为 None 使用默认配置
            
        Returns:
            包含日线数据的 DataFrame
        """
        source = source or self.config.default_source
        
        if source == 'tushare' and TUSHARE_AVAILABLE:
            return self._get_stock_daily_tushare(ts_code, start_date, end_date)
        elif source == 'akshare' and AKSHARE_AVAILABLE:
            return self._get_stock_daily_akshare(ts_code, start_date, end_date)
        else:
            raise ValueError(f"数据源 '{source}' 不可用或未安装")
    
    def _get_stock_daily_tushare(self, ts_code: str, start_date: str = None, 
                                  end_date: str = None) -> pd.DataFrame:
        """从 Tushare 获取日线数据"""
        if self._tushare_api is None:
            raise ValueError("Tushare API 未初始化，请设置 token")
        
        kwargs = {
            'ts_code': ts_code,
        }
        if start_date:
            kwargs['start_date'] = start_date
        if end_date:
            kwargs['end_date'] = end_date
        
        df = self._tushare_api.daily(**kwargs)
        
        # 按日期排序
        df = df.sort_values('trade_date', ascending=True)
        
        return df
    
    def _get_stock_daily_akshare(self, ts_code: str, start_date: str = None, 
                                  end_date: str = None) -> pd.DataFrame:
        """从 AKShare 获取日线数据"""
        # AKShare 的股票代码格式需要转换
        if ts_code.endswith('.SZ'):
            symbol = ts_code.replace('.SZ', 'sz')
        elif ts_code.endswith('.SH'):
            symbol = ts_code.replace('.SH', 'sh')
        else:
            symbol = ts_code
        
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust='qfq'  # 前复权
        )
        
        # 按日期排序
        df = df.sort_values('日期', ascending=True)
        
        return df
    
    def get_stock_list(self, source: str = None) -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            source: 数据源类型
            
        Returns:
            包含股票列表的 DataFrame
        """
        source = source or self.config.default_source
        
        if source == 'tushare' and TUSHARE_AVAILABLE:
            return self._get_stock_list_tushare()
        elif source == 'akshare' and AKSHARE_AVAILABLE:
            return self._get_stock_list_akshare()
        else:
            raise ValueError(f"数据源 '{source}' 不可用或未安装")
    
    def _get_stock_list_tushare(self) -> pd.DataFrame:
        """从 Tushare 获取股票列表"""
        if self._tushare_api is None:
            raise ValueError("Tushare API 未初始化，请设置 token")
        
        df = self._tushare_api.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry,list_date')
        return df
    
    def _get_stock_list_akshare(self) -> pd.DataFrame:
        """从 AKShare 获取股票列表"""
        df = ak.stock_zh_a_spot()
        # 选择常用字段
        if '代码' in df.columns:
            df = df.rename(columns={
                '代码': 'ts_code',
                '名称': 'name',
                '最新价': 'price',
                '涨跌幅': 'pct_change'
            })
        return df
    
    def get_index_daily(self, index_code: str, start_date: str = None, 
                        end_date: str = None, source: str = None) -> pd.DataFrame:
        """
        获取指数日线数据
        
        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            source: 数据源类型
            
        Returns:
            包含指数日线数据的 DataFrame
        """
        source = source or self.config.default_source
        
        if source == 'tushare' and TUSHARE_AVAILABLE:
            return self._get_index_daily_tushare(index_code, start_date, end_date)
        elif source == 'akshare' and AKSHARE_AVAILABLE:
            return self._get_index_daily_akshare(index_code, start_date, end_date)
        else:
            raise ValueError(f"数据源 '{source}' 不可用或未安装")
    
    def _get_index_daily_tushare(self, index_code: str, start_date: str = None, 
                                  end_date: str = None) -> pd.DataFrame:
        """从 Tushare 获取指数日线数据"""
        if self._tushare_api is None:
            raise ValueError("Tushare API 未初始化，请设置 token")
        
        kwargs = {
            'ts_code': index_code,
        }
        if start_date:
            kwargs['start_date'] = start_date
        if end_date:
            kwargs['end_date'] = end_date
        
        df = self._tushare_api.index_daily(**kwargs)
        df = df.sort_values('trade_date', ascending=True)
        
        return df
    
    def _get_index_daily_akshare(self, index_code: str, start_date: str = None, 
                                  end_date: str = None) -> pd.DataFrame:
        """从 AKShare 获取指数日线数据"""
        # AKShare 的指数代码格式
        symbol = index_code.replace('.SH', '').replace('.SZ', '')
        
        df = ak.index_zh_a_hist(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=''
        )
        
        df = df.sort_values('日期', ascending=True)
        
        return df
    
    def get_financial_report(self, ts_code: str, period: str = 'annual', 
                             source: str = None) -> pd.DataFrame:
        """
        获取财务报表数据
        
        Args:
            ts_code: 股票代码
            period: 周期类型（'annual' 年度, 'quarterly' 季度）
            source: 数据源类型
            
        Returns:
            包含财务报表数据的 DataFrame
        """
        source = source or self.config.default_source
        
        if source == 'tushare' and TUSHARE_AVAILABLE:
            return self._get_financial_report_tushare(ts_code, period)
        elif source == 'akshare' and AKSHARE_AVAILABLE:
            return self._get_financial_report_akshare(ts_code, period)
        else:
            raise ValueError(f"数据源 '{source}' 不可用或未安装")
    
    def _get_financial_report_tushare(self, ts_code: str, period: str = 'annual') -> pd.DataFrame:
        """从 Tushare 获取财务报表"""
        if self._tushare_api is None:
            raise ValueError("Tushare API 未初始化，请设置 token")
        
        # 获取利润表
        df = self._tushare_api.income(ts_code=ts_code)
        
        if period == 'quarterly':
            # 季度数据
            pass
        else:
            # 年度数据，取每年最后一个季度
            df['end_date'] = pd.to_datetime(df['end_date'])
            df['year'] = df['end_date'].dt.year
            df = df.groupby('year').last().reset_index()
        
        return df
    
    def _get_financial_report_akshare(self, ts_code: str, period: str = 'annual') -> pd.DataFrame:
        """从 AKShare 获取财务报表"""
        symbol = ts_code.replace('.SZ', '').replace('.SH', '')
        
        if period == 'quarterly':
            df = ak.stock_financial_report_sina(symbol=symbol, symbol_type='quarterly')
        else:
            df = ak.stock_financial_report_sina(symbol=symbol, symbol_type='yearly')
        
        return df
    
    def set_default_source(self, source: str):
        """
        设置默认数据源
        
        Args:
            source: 数据源类型（'tushare', 'akshare'）
        """
        if source not in ['tushare', 'akshare']:
            raise ValueError("数据源类型必须是 'tushare' 或 'akshare'")
        
        self.config.default_source = source
    
    def is_source_available(self, source: str) -> bool:
        """
        检查数据源是否可用
        
        Args:
            source: 数据源类型
            
        Returns:
            是否可用
        """
        if source == 'tushare':
            return TUSHARE_AVAILABLE
        elif source == 'akshare':
            return AKSHARE_AVAILABLE
        return False


# 全局数据源管理器实例
_global_data_source_manager = None


def get_data_source_manager() -> DataSourceManager:
    """
    获取全局数据源管理器实例
    
    Returns:
        数据源管理器实例
    """
    global _global_data_source_manager
    if _global_data_source_manager is None:
        _global_data_source_manager = DataSourceManager()
    return _global_data_source_manager


def init_data_source(config: Optional[DataSourceConfig] = None):
    """
    初始化数据源管理器
    
    Args:
        config: 数据源配置
    """
    global _global_data_source_manager
    _global_data_source_manager = DataSourceManager(config)
