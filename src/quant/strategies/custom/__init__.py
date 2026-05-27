"""
策略模块 - 用户自定义策略
"""

from .b1 import B1Strategy
from .template import TemplateStrategy
from .right_side_bottom import RightSideBottomFishingStrategy

__all__ = ["B1Strategy", "TemplateStrategy", "RightSideBottomFishingStrategy"]