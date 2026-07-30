"""
Quant 量化系统

一个用于量化交易策略开发和回测的框架
"""

__version__ = "0.1.0"
__author__ = "Quant Team"

# 导出核心模块。策略和回测依赖 akquant，日常刷新入口不需要它们。
try:
    from .strategies import *
    from .backtest import BacktestEngine, GridSearchOptimizer
except ModuleNotFoundError as exc:
    if exc.name != "akquant":
        raise
from .data import DataStorage, TushareDataFetcher, MarketDataStore, MarketDataStoreConfig
from .analysis import PerformanceAnalyzer, AttributionAnalyzer
from .ml import (
    MLDataSet,
    ModelTrainer,
    B1QualityLabelMaker,
    B1ExitAwareLabelMaker,
    create_b1_labels
)

__all__ = [
    # 策略
    "BaseStrategy",
    "DualMAStrategy",
    "MomentumStrategy",
    "BreakoutStrategy",
    "MeanReversionStrategy",
    "B1Strategy",
    "TemplateStrategy",
    "RightSideBottomFishingStrategy",

    # 回测
    "BacktestEngine",
    "GridSearchOptimizer",

    # 数据
    "DataStorage",
    "TushareDataFetcher",
    "MarketDataStore",
    "MarketDataStoreConfig",

    # 分析
    "PerformanceAnalyzer",
    "AttributionAnalyzer",

    # 机器学习
    "MLDataSet",
    "ModelTrainer",
    "B1QualityLabelMaker",
    "B1ExitAwareLabelMaker",
    "create_b1_labels"
]
