"""
因子基类定义

所有因子都需要继承此基类并实现 compute 方法
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd


class Factor(ABC):
    """
    因子基类
    
    所有自定义因子都需要继承此类并实现 compute 方法
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        初始化因子
        
        Args:
            name: 因子名称
            params: 因子参数
        """
        self.name = name
        self.params = params or {}
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值
        
        Args:
            data: 包含价格数据的 DataFrame
            
        Returns:
            因子值序列
        """
        pass
    
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        """支持直接调用计算"""
        return self.compute(data)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


class RollingFactor(Factor):
    """
    滚动窗口因子基类
    
    用于计算需要滚动窗口的因子（如均线、RSI等）
    """
    
    def __init__(self, name: str, window: int, params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)
        self.window = window
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        pass
