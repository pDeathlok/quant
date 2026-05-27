"""
因子基类定义

所有因子都需要继承此基类并实现 compute 方法
支持自动适配不同数据源（Tushare、AKShare）的字段命名
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List
import pandas as pd

from .data_adapter import FactorDataAdapter


class Factor(ABC):
    """
    因子基类
    
    所有自定义因子都需要继承此类并实现 compute 方法
    
    支持自动适配不同数据源的字段命名（Tushare、AKShare等）
    通过数据适配器自动处理字段差异
    """
    
    # 字段别名映射（扩展版）
    _FIELD_ALIASES = {
        'close': ['close', '收盘价', 'adj_close', 'close_price'],
        'open': ['open', '开盘价', 'open_price'],
        'high': ['high', '最高价', 'high_price'],
        'low': ['low', '最低价', 'low_price'],
        'volume': ['volume', 'vol', '成交量', 'volume_share'],
        'turnover': ['turnover', 'amount', '成交额', 'turnover_amt'],
        'pct_change': ['pct_change', 'pct_chg', '涨跌幅', 'return_pct'],
        'prev_close': ['prev_close', 'pre_close', '前收盘', '前收盘价'],
        'symbol': ['symbol', 'ts_code', '股票代码'],
        'date': ['date', 'trade_date', '日期'],
        'vwap': ['vwap'],
    }
    
    # 标准字段列表
    _STANDARD_FIELDS = [
        'symbol', 'date', 'open', 'high', 'low', 'close', 
        'prev_close', 'volume', 'turnover', 'pct_change'
    ]
    
    def __init__(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        """
        初始化因子
        
        Args:
            name: 因子名称，如果为 None 则使用类名
            params: 因子参数
        """
        self.name = name or self.__class__.__name__
        self.params = params or {}
        self._data_adapter = FactorDataAdapter()
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        计算因子值
        
        Args:
            data: 包含价格数据的 DataFrame（已自动适配）
            
        Returns:
            因子值序列或 DataFrame
        """
        pass
    
    def __call__(self, data: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """支持直接调用计算"""
        # 自动适配数据字段
        data = self._adapt_data(data)
        return self.compute(data, **kwargs)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
    
    def _adapt_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        适配数据字段，统一不同数据源的字段命名
        
        使用 FactorDataAdapter 进行自动适配，支持 Tushare 和 AKShare 数据
        
        Args:
            data: 原始数据
            
        Returns:
            适配后的数据，包含标准字段
        """
        # 使用数据适配器进行自动适配
        return FactorDataAdapter.adapt(data)
    
    def _get_field(self, data: pd.DataFrame, field: str) -> pd.Series:
        """
        获取字段，支持别名查找
        
        Args:
            data: 数据 DataFrame
            field: 目标字段名
            
        Returns:
            字段数据
        """
        # 首先检查精确匹配
        if field in data.columns:
            return data[field]
        
        # 检查别名
        if field in self._FIELD_ALIASES:
            for alias in self._FIELD_ALIASES[field]:
                if alias in data.columns:
                    return data[alias]
        
        raise ValueError(f"数据中缺少字段 '{field}' 及其别名。可用字段: {list(data.columns)}")
    
    def _require_fields(self, data: pd.DataFrame, required_fields: List[str]) -> None:
        """
        检查必需字段是否存在
        
        Args:
            data: 数据 DataFrame
            required_fields: 必需字段列表
            
        Raises:
            ValueError: 如果缺少必需字段
        """
        missing_fields = []
        for field in required_fields:
            if field not in data.columns:
                # 检查别名
                has_alias = False
                if field in self._FIELD_ALIASES:
                    for alias in self._FIELD_ALIASES[field]:
                        if alias in data.columns:
                            has_alias = True
                            break
                if not has_alias:
                    missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"数据中缺少必需字段: {missing_fields}。可用字段: {list(data.columns)}")
    
    def _ensure_standard_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        确保数据包含所有标准字段
        
        Args:
            data: 输入数据
            
        Returns:
            包含标准字段的数据
        """
        return FactorDataAdapter.ensure_standard_fields(data)


class RollingFactor(Factor):
    """
    滚动窗口因子基类
    
    用于计算需要滚动窗口的因子（如均线、RSI等）
    """
    
    def __init__(self, window: int, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)
        self.window = window
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', window={self.window}, params={self.params})"