"""
因子模块导出

包含所有可用的因子类
"""

from .base import Factor, RollingFactor
from .technical import (
    # 技术指标因子
    MA, EMA, MACD, RSI, BollingerBands, ATR, VolumeRatio, ROC, Stochastic,
    KDJ, WilliamsR, BIAS, Momentum, PSY, VR, OBV, CCI, DMI,
    
    # 盈利惊喜因子
    NetProfitYoY, RevenueYoY, EPS, EarningsSurprise, RevenueSurprise,
    NetProfitQoQ, RevenueQoQ, NonRecurringProfitYoY, EarningsQuality, PEG,
    
    # 市值因子
    MarketCap, SizeDecile,
    
    # 价值因子
    PERatio, PBRatio, PSRatio, PCFRatio, EVToEBITDA, DividendYield,
    
    # 动量因子
    MomentumReturn, Reversal, IntermediateMomentum, RSTR,
    
    # 波动率因子
    Volatility, IdiosyncraticVolatility, HighLowVolatilityRatio, DownsideVolatility,
    
    # 质量因子
    ROE, ROA, ROIC, OperatingMargin, NetMargin, GrossProfitMargin,
    
    # 杠杆因子
    DebtToEquity, InterestCoverage,
    
    # 流动性因子
    TurnoverRatio, AmihudIlliquidity,
    
    # Rust 风格移动平均类
    SMMA, VWAP, EMAVolume,
    
    # Rust 风格波动类指标
    DonchianChannel, KeltnerChannel, MassIndex,
    
    # Rust 风格趋势类指标
    ADX, ParabolicSAR, VortexIndicator,
    
    # Rust 风格量价类指标
    ChaikinMoneyFlow, EaseOfMovement, VolumeWeightedMACD,
    
    # 成长因子扩展
    GrowthScore, RevenueGrowthAcceleration, ProfitGrowthQuality,
    
    # 因子工具函数
    winsorize_factor, standardize_factor, neutralize_factor,
    
    # 因子合成器
    FactorComposite,
    
    # 行业因子
    IndustryFactor, IndustryDummy
)

__all__ = [
    # 基类
    "Factor", "RollingFactor",
    
    # 技术指标因子
    "MA", "EMA", "MACD", "RSI", "BollingerBands", "ATR", "VolumeRatio", "ROC", "Stochastic",
    "KDJ", "WilliamsR", "BIAS", "Momentum", "PSY", "VR", "OBV", "CCI", "DMI",
    
    # 盈利惊喜因子
    "NetProfitYoY", "RevenueYoY", "EPS", "EarningsSurprise", "RevenueSurprise",
    "NetProfitQoQ", "RevenueQoQ", "NonRecurringProfitYoY", "EarningsQuality", "PEG",
    
    # 市值因子
    "MarketCap", "SizeDecile",
    
    # 价值因子
    "PERatio", "PBRatio", "PSRatio", "PCFRatio", "EVToEBITDA", "DividendYield",
    
    # 动量因子
    "MomentumReturn", "Reversal", "IntermediateMomentum", "RSTR",
    
    # 波动率因子
    "Volatility", "IdiosyncraticVolatility", "HighLowVolatilityRatio", "DownsideVolatility",
    
    # 质量因子
    "ROE", "ROA", "ROIC", "OperatingMargin", "NetMargin", "GrossProfitMargin",
    
    # 杠杆因子
    "DebtToEquity", "InterestCoverage",
    
    # 流动性因子
    "TurnoverRatio", "AmihudIlliquidity",
    
    # Rust 风格移动平均类
    "SMMA", "VWAP", "EMAVolume",
    
    # Rust 风格波动类指标
    "DonchianChannel", "KeltnerChannel", "MassIndex",
    
    # Rust 风格趋势类指标
    "ADX", "ParabolicSAR", "VortexIndicator",
    
    # Rust 风格量价类指标
    "ChaikinMoneyFlow", "EaseOfMovement", "VolumeWeightedMACD",
    
    # 成长因子扩展
    "GrowthScore", "RevenueGrowthAcceleration", "ProfitGrowthQuality",
    
    # 因子工具函数
    "winsorize_factor", "standardize_factor", "neutralize_factor",
    
    # 因子合成器
    "FactorComposite",
    
    # 行业因子
    "IndustryFactor", "IndustryDummy"
]
