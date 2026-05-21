"""
策略模块

包含所有量化策略的定义和导出
"""

# 基类
from .base import BaseStrategy

# 动量策略
from .momentum import DualMAStrategy, MomentumStrategy, BreakoutStrategy

# 均值回归策略
from .mean_reversion import MeanReversionStrategy

# 用户自定义策略
from .custom import B1Strategy, TemplateStrategy

__all__ = [
    # 基类
    "BaseStrategy",

    # 动量策略
    "DualMAStrategy",
    "MomentumStrategy",
    "BreakoutStrategy",

    # 均值回归策略
    "MeanReversionStrategy",

    # 自定义策略
    "B1Strategy",
    "TemplateStrategy"
]