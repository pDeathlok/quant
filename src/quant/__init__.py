"""
Quant 量化系统

一个用于量化交易策略开发和回测的框架
"""

__version__ = "1.0.0"
__author__ = "Quant Team"

# 导出核心模块
from .strategies import *
from .backtest import BacktestEngine, GridSearchOptimizer
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
