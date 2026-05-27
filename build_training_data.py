#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 B1 策略构建训练数据

包含：
1. 多种技术指标因子
2. 动量因子
3. 波动率因子
4. 大盘趋势因子
5. 资金流因子
6. 基本面因子
7. 风险因子
8. 另类数据因子
9. 多种 label 定义

样本选择：只选择满足 B1 策略买入条件的样本
"""

import os
import sys
sys.path.insert(0, '/Users/didi/Project/quant/src')

import pandas as pd
import numpy as np
from datetime import datetime
from quant.data.factors import (
    FactorDataAdapter,
    MA, EMA, MACD, RSI, ATR, BollingerBands, KDJ, WilliamsR, BIAS,
    Momentum, ROC, Stochastic, OBV, CCI, DMI, ADX,
    Volatility, DownsideVolatility, IdiosyncraticVolatility,
    Factor, RollingFactor,
    EPSFromReport, BookValuePerShare, RevenueGrowth, NetProfitGrowth,
    GrossProfitMarginFromReport, ROEFromReport, RevenueQoQFromReport,
    NetProfitQoQFromReport, OperatingCashFlowPerShare, IndustryFromReport,
    TotalAsset, TotalDebt, EquityRatio, AssetGrowth, DebtToAssetRatio,
    CashOnHand, LogMarketCap, Amplitude, Amihud,
)
from quant.data.factors.technical import (
    PSY, VR, MassIndex, ParabolicSAR, VortexIndicator,
    ChaikinMoneyFlow, EaseOfMovement, KeltnerChannel,
)
from quant.data.factors.alpha101 import (
    Alpha001Factor, Alpha002Factor, Alpha003Factor, Alpha004Factor, Alpha005Factor,
    Alpha006Factor, Alpha007Factor, Alpha008Factor, Alpha009Factor, Alpha010Factor
)
from quant.data.factors.alpha191 import (
    Alpha191_01Factor, Alpha191_02Factor, Alpha191_03Factor, 
    Alpha191_04Factor, Alpha191_05Factor, Alpha191_06Factor,
    Alpha191_07Factor, Alpha191_08Factor, Alpha191_09Factor,
    Alpha191_10Factor, Alpha191_11Factor, Alpha191_12Factor,
    Alpha191_13Factor, Alpha191_14Factor, Alpha191_15Factor
)
from quant.data.factors.momentum import (
    ReturnFactor, MomentumSkip5Factor, RiskAdjustedMomentumFactor, ReversalFactor
)
from quant.ml.label_maker import create_b1_labels


class BBIFactor(Factor):
    """BBI 多空指标因子"""
    
    def __init__(self):
        super().__init__(name="BBI")
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ma3 = close.rolling(window=3).mean()
        ma6 = close.rolling(window=6).mean()
        ma12 = close.rolling(window=12).mean()
        ma24 = close.rolling(window=24).mean()
        return (ma3 + ma6 + ma12 + ma24) / 4


class AmplitudeFactor(RollingFactor):
    """振幅因子"""
    
    def __init__(self):
        super().__init__(name="Amplitude", window=1)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return (df["high"] - df["low"]) / df["low"] * 100


class VolumeChangeFactor(RollingFactor):
    """成交量变化因子"""
    
    def __init__(self):
        super().__init__(name="VolumeChange", window=2)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["volume"] / df["volume"].shift(1) - 1


class VolumeRatioFactor(RollingFactor):
    """量比因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(name="VolumeRatio", window=window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["volume"] / df["volume"].rolling(self.window).mean()


class TrendSlopeFactor(RollingFactor):
    """趋势斜率因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(name="TrendSlope", window=window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        from scipy import stats
        close = df["close"]
        result = close.rolling(self.window).apply(
            lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == self.window else np.nan,
            raw=True
        )
        return result


class TrendStrengthFactor(RollingFactor):
    """趋势强度因子（R²）"""
    
    def __init__(self, window: int = 20):
        super().__init__(name="TrendStrength", window=window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        from scipy import stats
        close = df["close"]
        result = close.rolling(self.window).apply(
            lambda x: stats.linregress(range(len(x)), x)[2] ** 2 if len(x) == self.window else np.nan,
            raw=True
        )
        return result


class DonchianBreakoutFactor(RollingFactor):
    """Donchian 突破因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(name="DonchianBreakout", window=window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] / df["high"].rolling(self.window).max()


class PriceRangeRatioFactor(RollingFactor):
    """价格区间比例因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(name="PriceRangeRatio", window=window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        rolling_high = df["high"].rolling(self.window).max()
        rolling_low = df["low"].rolling(self.window).min()
        return (df["close"] - rolling_low) / (rolling_high - rolling_low)


class MADistanceFactor(Factor):
    """均线偏离因子"""
    
    def __init__(self, window: int = 20):
        super().__init__(name=f"MA{window}Distance")
        self.window = window
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma = df["close"].rolling(self.window).mean()
        return df["close"] / ma


class VWAPFactor(Factor):
    """VWAP 因子"""
    
    def __init__(self):
        super().__init__(name="VWAP")
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return vwap


class VWAPDeviationFactor(Factor):
    """VWAP 偏离因子"""
    
    def __init__(self):
        super().__init__(name="VWAPDeviation")
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return (df["close"] - vwap) / vwap


class SharpeRatioFactor(RollingFactor):
    """夏普比率因子"""
    
    def __init__(self, window: int = 60):
        super().__init__(name="SharpeRatio", window=window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df["close"].pct_change()
        mean_ret = returns.rolling(self.window).mean()
        std_ret = returns.rolling(self.window).std()
        return mean_ret / std_ret


class ReturnSkewFactor(RollingFactor):
    """收益偏度因子"""
    
    def __init__(self, window: int = 60):
        super().__init__(name="ReturnSkew", window=window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df["close"].pct_change()
        return returns.rolling(self.window).skew()


class ReturnKurtFactor(RollingFactor):
    """收益峰度因子"""
    
    def __init__(self, window: int = 60):
        super().__init__(name="ReturnKurt", window=window)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df["close"].pct_change()
        return returns.rolling(self.window).kurt()


class TailRiskFactor(RollingFactor):
    """尾部风险因子（VaR）"""
    
    def __init__(self, window: int = 60, quantile: float = 0.05):
        super().__init__(name="TailRisk", window=window)
        self.quantile = quantile
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df["close"].pct_change()
        return returns.rolling(self.window).quantile(self.quantile)


def calculate_b1_signal(df: pd.DataFrame) -> pd.Series:
    """计算 B1 策略入场信号（放宽条件以获取更多样本）"""
    pct_change = df["close"].pct_change() * 100
    cond1 = (pct_change >= -3) & (pct_change <= 3)
    
    amplitude = (df["high"] - df["low"]) / df["low"] * 100
    cond2 = amplitude < 10
    
    bbi = BBIFactor().compute(df)
    ma60 = df["close"].rolling(window=60).mean()
    cond3 = bbi > ma60
    
    kdj = KDJ().compute(df)
    cond4 = kdj["J"] < 10
    
    cond5 = df["volume"] > df["volume"].shift(1) * 0.8
    
    signal = cond1 & cond2 & cond3 & cond4 & cond5
    return signal.astype(int)


def calculate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """计算多种 label（复用 B1 策略 LabelMaker）"""
    # 使用 label_maker 创建专业的 B1 策略 Label
    b1_labels = create_b1_labels(df, forward_days=5, exit_aware=True)
    
    # 添加一些额外的简单 Label
    extra_labels = pd.DataFrame(index=df.index)
    
    # 不同周期的收益
    extra_labels["label_1d_return"] = df["close"].pct_change(1).shift(-1)
    extra_labels["label_3d_return"] = df["close"].pct_change(3).shift(-3)
    extra_labels["label_10d_return"] = df["close"].pct_change(10).shift(-10)
    extra_labels["label_20d_return"] = df["close"].pct_change(20).shift(-20)
    
    # 不同周期的涨跌分类
    extra_labels["label_1d_up"] = (extra_labels["label_1d_return"] > 0).astype(int)
    extra_labels["label_3d_up"] = (extra_labels["label_3d_return"] > 0).astype(int)
    extra_labels["label_10d_up"] = (extra_labels["label_10d_return"] > 0).astype(int)
    extra_labels["label_20d_up"] = (extra_labels["label_20d_return"] > 0).astype(int)
    
    # B1 策略信号
    extra_labels["label_b1_signal"] = calculate_b1_signal(df)
    
    # 合并所有 Label
    labels = pd.concat([b1_labels, extra_labels], axis=1)
    
    # 重命名部分列以保持一致性
    labels = labels.rename(columns={
        'future_return': 'label_5d_return',
        'max_intraday': 'label_max_intraday',
        'max_return': 'label_max_return',
        'quality': 'label_quality',
        'is_good': 'label_is_good',
        'has_surge_5': 'label_has_surge_5',
        'has_surge_7': 'label_has_surge_7',
        'has_surge_9': 'label_has_surge_9',
        'tp_potential_5': 'label_tp_potential_5',
        'tp_potential_7': 'label_tp_potential_7',
        'quality_score': 'label_quality_score',
        'min_return': 'label_min_return',
        'exit_type': 'label_exit_type',
        'exit_is_profitable': 'label_exit_is_profitable'
    })
    
    return labels


def get_market_type(ts_code: str) -> str:
    """
    根据股票代码判断市场类型
    
    Args:
        ts_code: 股票代码，支持多种格式：
                 - Tushare格式: 000001.SZ, 600000.SH, 300001.SZ, 688001.SH
                 - 短格式: sh600000, sz000001, 600000, 000001
    
    Returns:
        市场类型: 沪市主板, 深市主板, 创业板, 科创板, 北交所
    """
    if pd.isna(ts_code):
        return '未知'
    
    ts_code = str(ts_code).strip().lower()
    
    # 格式1: sh600000 或 sz000001
    if ts_code.startswith('sh'):
        code = ts_code[2:]
        if code.startswith('60'):
            return '沪市主板'
        elif code.startswith('688'):
            return '科创板'
        elif code.startswith('4'):
            return '沪市B股'
    elif ts_code.startswith('sz'):
        code = ts_code[2:]
        if code.startswith('000') or code.startswith('001'):
            return '深市主板'
        elif code.startswith('002'):
            return '中小板'
        elif code.startswith('300') or code.startswith('301'):
            return '创业板'
        elif code.startswith('2'):
            return '深市B股'
    
    # 格式2: 600000.SH 或 000001.SZ
    if '.sh' in ts_code:
        code = ts_code.split('.')[0]
        if code.startswith('60'):
            return '沪市主板'
        elif code.startswith('688'):
            return '科创板'
        elif code.startswith('4'):
            return '沪市B股'
    elif '.sz' in ts_code:
        code = ts_code.split('.')[0]
        if code.startswith('000') or code.startswith('001'):
            return '深市主板'
        elif code.startswith('002'):
            return '中小板'
        elif code.startswith('300') or code.startswith('301'):
            return '创业板'
        elif code.startswith('2'):
            return '深市B股'
    elif '.bj' in ts_code:
        return '北交所'
    
    # 格式3: 纯数字代码
    code = ''.join([c for c in ts_code if c.isdigit()])
    if len(code) >= 6:
        code = code[:6]
        if code.startswith('60'):
            return '沪市主板'
        elif code.startswith('688'):
            return '科创板'
        elif code.startswith('000') or code.startswith('001'):
            return '深市主板'
        elif code.startswith('002'):
            return '中小板'
        elif code.startswith('300') or code.startswith('301'):
            return '创业板'
        elif code.startswith('8'):
            return '北交所'
    
    return '未知'


def get_market_code(market_type: str) -> int:
    """
    将市场类型转换为数字编码
    
    Args:
        market_type: 市场类型字符串
    
    Returns:
        数字编码: 1=沪市主板, 2=深市主板, 3=创业板, 4=科创板, 5=中小板, 6=北交所, 0=未知
    """
    market_map = {
        '沪市主板': 1,
        '深市主板': 2,
        '创业板': 3,
        '科创板': 4,
        '中小板': 5,
        '北交所': 6,
        '沪市B股': 7,
        '深市B股': 8,
        '未知': 0
    }
    return market_map.get(market_type, 0)


def calculate_market_value_factors(df: pd.DataFrame, float_df: pd.DataFrame = None) -> pd.DataFrame:
    """计算市值相关因子"""
    factors = pd.DataFrame(index=df.index)
    
    # 基于价格的市值估算（需要总股本数据才能计算真实市值）
    # 这里先添加一些与价格和成交量相关的因子作为替代
    
    # 股价水平因子
    factors["price_level"] = df["close"]
    factors["price_log"] = np.log(df["close"] + 1)
    factors["price_rank"] = df["close"].rank(pct=True)
    
    # 量价因子
    factors["price_volume_ratio"] = df["close"] / (df["volume"] + 1)
    factors["turnover_ratio"] = df["volume"] / (df["volume"].rolling(60).mean() + 1)
    
    # 流通市值估算（如果有流通股数据）
    if float_df is not None and len(float_df) > 0:
        float_df = float_df.copy()
        
        # 计算平均流通比例
        avg_float_ratio = float_df['float_ratio'].mean()
        
        # 估算流通市值（需要收盘价）
        # 流通市值 ≈ 收盘价 * 流通股数（万股）* 10000
        factors["float_ratio_avg"] = avg_float_ratio
        factors["estimated_float_market_cap"] = df["close"] * avg_float_ratio * 10000
        
        # 计算流通市值对数
        factors["estimated_float_market_cap_log"] = np.log(factors["estimated_float_market_cap"] + 1)
    
    # 价格与成交量的相关性
    factors["price_volume_corr_20"] = df["close"].rolling(20).corr(df["volume"])
    
    # 成交额因子
    if 'turnover' in df.columns:
        factors["turnover_amount"] = df["turnover"]
        factors["turnover_log"] = np.log(df["turnover"] + 1)
        factors["turnover_ratio_20"] = df["turnover"] / (df["turnover"].rolling(20).mean() + 1)
    
    return factors


def calculate_market_factors(df: pd.DataFrame, index_df: pd.DataFrame = None) -> pd.DataFrame:
    """计算大盘趋势因子"""
    factors = pd.DataFrame(index=df.index)
    
    if index_df is None or len(index_df) == 0:
        factors["market_return_5d"] = np.nan
        factors["market_return_10d"] = np.nan
        factors["market_return_20d"] = np.nan
        factors["market_volatility_20d"] = np.nan
        factors["market_trend"] = np.nan
        factors["market_ma5_distance"] = np.nan
        factors["market_ma10_distance"] = np.nan
        factors["market_ma20_distance"] = np.nan
        factors["market_rsi_14"] = np.nan
        factors["market_macd"] = np.nan
        return factors
    
    index_df = index_df.copy()
    
    # 处理日期列名
    if 'trade_date' in index_df.columns:
        index_df['date'] = pd.to_datetime(index_df['trade_date'])
    elif 'date' in index_df.columns:
        index_df['date'] = pd.to_datetime(index_df['date'])
    else:
        print("警告: 指数数据缺少日期列")
        factors["market_return_5d"] = np.nan
        factors["market_return_10d"] = np.nan
        factors["market_return_20d"] = np.nan
        factors["market_volatility_20d"] = np.nan
        factors["market_trend"] = np.nan
        factors["market_ma5_distance"] = np.nan
        factors["market_ma10_distance"] = np.nan
        factors["market_ma20_distance"] = np.nan
        factors["market_rsi_14"] = np.nan
        factors["market_macd"] = np.nan
        return factors
    
    index_df = index_df.sort_values('date').reset_index(drop=True)
    
    market_return_5d = index_df.set_index('date')['close'].pct_change(5)
    market_return_10d = index_df.set_index('date')['close'].pct_change(10)
    market_return_20d = index_df.set_index('date')['close'].pct_change(20)
    market_volatility_20d = index_df.set_index('date')['close'].pct_change().rolling(20).std()
    
    market_ma5 = index_df.set_index('date')['close'].rolling(5).mean()
    market_ma10 = index_df.set_index('date')['close'].rolling(10).mean()
    market_ma20 = index_df.set_index('date')['close'].rolling(20).mean()
    market_ma5_distance = index_df.set_index('date')['close'] / market_ma5
    market_ma10_distance = index_df.set_index('date')['close'] / market_ma10
    market_ma20_distance = index_df.set_index('date')['close'] / market_ma20
    
    market_trend = (market_ma5 > market_ma10) & (market_ma10 > market_ma20)
    market_trend = market_trend.astype(int)
    
    delta = index_df.set_index('date')['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    market_rsi_14 = 100 - (100 / (1 + rs))
    
    ema12 = index_df.set_index('date')['close'].ewm(span=12, adjust=False).mean()
    ema26 = index_df.set_index('date')['close'].ewm(span=26, adjust=False).mean()
    market_macd = ema12 - ema26
    
    df_dates = pd.to_datetime(df['date'])
    factors["market_return_5d"] = df_dates.map(market_return_5d).values
    factors["market_return_10d"] = df_dates.map(market_return_10d).values
    factors["market_return_20d"] = df_dates.map(market_return_20d).values
    factors["market_volatility_20d"] = df_dates.map(market_volatility_20d).values
    factors["market_trend"] = df_dates.map(market_trend).values
    factors["market_ma5_distance"] = df_dates.map(market_ma5_distance).values
    factors["market_ma10_distance"] = df_dates.map(market_ma10_distance).values
    factors["market_ma20_distance"] = df_dates.map(market_ma20_distance).values
    factors["market_rsi_14"] = df_dates.map(market_rsi_14).values
    factors["market_macd"] = df_dates.map(market_macd).values
    
    return factors


def calculate_akshare_fundamental_factors(df: pd.DataFrame, yjbb_path: str = None,
                                          zcfz_path: str = None) -> pd.DataFrame:
    """从 AKShare financial_yjbb_multi 和 balance_sheet_zcfz_multi 提取因子

    使用 merge_asof 按日期向前映射，防止穿越：
    1. 计算 publish_date = report_date + 3个月
    2. 只保留 publish_date <= 交易日的记录
    3. merge_asof 取 publish_date 最近的已发布报告
    """
    factors = pd.DataFrame(index=df.index)
    if 'date' not in df.columns or 'symbol' not in df.columns:
        return factors

    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'])

    def _merge_asof_fundamental(fund_df, report_date_col, publish_delay_months=3,
                                symbol_col='symbol', field_map=None):
        """用 merge_asof 将财务数据对齐到交易日，防止穿越。

        对每只股票单独做 merge_asof，确保日期对齐正确。
        防止穿越：publish_date = report_date + 延迟，只使用已发布的报告。
        """
        result_series = {dst: pd.Series(np.nan, index=df2.index) for dst in field_map.values()}

        fund = fund_df.copy()
        fund['publish_date'] = (pd.to_datetime(fund[report_date_col], format='%Y%m%d')
                                + pd.DateOffset(months=publish_delay_months))
        for col in field_map.keys():
            if col in fund.columns:
                fund[col] = pd.to_numeric(fund[col], errors='coerce')

        fund2 = fund.rename(columns={'publish_date': 'date'})

        for sym in df2[symbol_col].unique():
            sym_df = df2[df2[symbol_col] == sym][['date']].copy()
            sym_df = sym_df.sort_values('date').reset_index(drop=True)
            orig_idx = sym_df.index  # 这些是 df2 中的相对位置

            sym_fund = fund2[fund2[symbol_col] == sym].copy()
            if sym_fund.empty:
                continue

            sym_fund = sym_fund[['date'] + list(field_map.keys())].copy()
            sym_fund = sym_fund.sort_values('date').reset_index(drop=True)

            merged = pd.merge_asof(
                sym_df,
                sym_fund,
                on='date',
                direction='backward',
            )

            # 找到这只股票在 df2 中的索引
            sym_mask = df2[symbol_col] == sym
            sym_indices = df2.index[sym_mask]

            for src, dst in field_map.items():
                if src in merged.columns:
                    result_series[dst].loc[sym_indices] = merged[src].values

        for dst, s in result_series.items():
            factors[dst] = s

    if yjbb_path and os.path.exists(yjbb_path):
        yjbb = pd.read_parquet(yjbb_path)
        yjbb = yjbb.rename(columns={'股票代码': 'symbol'})

        field_map = {
            '每股收益': 'eps_report',
            '每股净资产': 'book_value_per_share',
            '营业总收入-同比增长': 'revenue_growth_yoy',
            '净利润-同比增长': 'net_profit_growth_yoy',
            '销售毛利率': 'gross_profit_margin_yjbb',
            '净资产收益率': 'roe_report',
            '营业总收入-季度环比增长': 'revenue_growth_qoq',
            '净利润-季度环比增长': 'net_profit_growth_qoq',
            '每股经营现金流量': 'ocf_per_share',
        }
        _merge_asof_fundamental(yjbb, 'report_date', 3, 'symbol', field_map)

    if zcfz_path and os.path.exists(zcfz_path):
        zcfz = pd.read_parquet(zcfz_path)
        zcfz = zcfz.rename(columns={'股票代码': 'symbol'})

        zcfz_field_map = {
            '资产-总资产': 'total_asset',
            '负债-总负债': 'total_debt',
            '资产负债率': 'debt_to_asset_ratio',
            '资产-货币资金': 'cash_on_hand',
            '资产-总资产同比': 'asset_growth_yoy',
        }
        _merge_asof_fundamental(zcfz, 'report_date', 3, 'symbol', zcfz_field_map)

    return factors


def calculate_macro_factors(df: pd.DataFrame, macro_dir: str = None) -> pd.DataFrame:
    """从 AKShare 宏观数据提取择时因子

    防止穿越：所有宏观数据使用 publish_date（发布日期）进行对齐，
    只使用已发布的数据，不引入未来信息。
    """
    import re
    factors = pd.DataFrame(index=df.index)
    if not macro_dir or not os.path.exists(macro_dir):
        return factors

    def _parse_chinese_month(s):
        """解析 '2026年04月份' 格式为日期。"""
        m = re.match(r'(\d{4})年(\d+)月份', str(s))
        if m:
            return pd.Timestamp(f'{m.group(1)}-{m.group(2)}-01')
        try:
            return pd.to_datetime(s)
        except Exception:
            return pd.NaT

    df['date'] = pd.to_datetime(df['date'])
    df_dates = df['date']

    # 辅助函数：按月对齐，防止穿越
    def _map_macro(series: pd.Series, dates: pd.Series, delay_months: int = 1) -> pd.Series:
        """将月度宏观数据映射到每日数据，考虑发布延迟。"""
        shifted = series.shift(delay_months)
        return dates.apply(
            lambda d: shifted[shifted.index <= d].iloc[-1]
            if not shifted[shifted.index <= d].empty else np.nan
        )

    # CPI（当月数据次月中旬发布，延迟1个月）
    cpi_path = os.path.join(macro_dir, 'macro_cpi.parquet')
    if os.path.exists(cpi_path):
        cpi = pd.read_parquet(cpi_path)
        cpi['month'] = cpi['月份'].apply(_parse_chinese_month)
        cpi = cpi.dropna(subset=['month'])
        cpi_val = pd.to_numeric(cpi['全国-同比增长'], errors='coerce')
        cpi_series = cpi_val.set_axis(cpi['month'])
        factors["macro_cpi"] = _map_macro(cpi_series, df_dates, delay_months=1)

    # M2（延迟1个月）
    m2_path = os.path.join(macro_dir, 'macro_money_supply.parquet')
    if os.path.exists(m2_path):
        m2 = pd.read_parquet(m2_path)
        m2['month'] = m2['月份'].apply(_parse_chinese_month)
        m2 = m2.dropna(subset=['month'])
        m2_col = '货币(M2)-同比增长' if '货币(M2)-同比增长' in m2.columns else 'M2-同比增长'
        if m2_col in m2.columns:
            m2_val = pd.to_numeric(m2[m2_col], errors='coerce')
            m2_series = m2_val.set_axis(m2['month'])
            factors["macro_m2_growth"] = _map_macro(m2_series, df_dates, delay_months=1)

    # LPR（每月20日公布，当日可用）
    lpr_path = os.path.join(macro_dir, 'macro_lpr.parquet')
    if os.path.exists(lpr_path):
        lpr = pd.read_parquet(lpr_path)
        lpr['date'] = pd.to_datetime(lpr['TRADE_DATE'])
        lpr1y = pd.to_numeric(lpr['LPR1Y'], errors='coerce')
        lpr5y = pd.to_numeric(lpr['LPR5Y'], errors='coerce')
        lpr_spread = lpr5y - lpr1y
        lpr_s = lpr_spread.set_axis(lpr['date'])
        factors["macro_lpr_spread"] = df_dates.apply(
            lambda d: lpr_s[lpr_s.index <= d].iloc[-1]
            if not lpr_s[lpr_s.index <= d].empty else np.nan)

    # 10年期国债收益率（每日发布，当日可用）
    bond_path = os.path.join(macro_dir, 'bond_yield_curve.parquet')
    if os.path.exists(bond_path):
        bond = pd.read_parquet(bond_path)
        if '日期' in bond.columns and '10年' in bond.columns:
            bond['date'] = pd.to_datetime(bond['日期'])
            bond_10y = pd.to_numeric(bond['10年'], errors='coerce')
            bond_s = bond_10y.set_axis(bond['date'])
            factors["macro_bond_10y"] = df_dates.apply(
                lambda d: bond_s[bond_s.index <= d].iloc[-1]
                if not bond_s[bond_s.index <= d].empty else np.nan)

    return factors


def calculate_factors(df: pd.DataFrame, index_df: pd.DataFrame = None,
                      fundamental_df: pd.DataFrame = None,
                      margin_df: pd.DataFrame = None,
                      akshare_yjbb_path: str = None,
                      akshare_zcfz_path: str = None,
                      macro_dir: str = None) -> pd.DataFrame:
    """计算所有因子"""
    factors = pd.DataFrame(index=df.index)
    
    # ========== 基础技术指标 ==========
    factors["ma5"] = MA(5).compute(df)
    factors["ma10"] = MA(10).compute(df)
    factors["ma20"] = MA(20).compute(df)
    factors["ma60"] = MA(60).compute(df)
    factors["ma120"] = MA(120).compute(df)
    
    factors["ema5"] = EMA(5).compute(df)
    factors["ema10"] = EMA(10).compute(df)
    factors["ema20"] = EMA(20).compute(df)
    
    macd_factor = MACD()
    factors["macd"] = macd_factor.compute(df)
    factors["macd_signal"] = macd_factor.compute_signal(df)
    factors["macd_hist"] = macd_factor.compute_histogram(df)
    
    factors["rsi_5"] = RSI(5).compute(df)
    factors["rsi_14"] = RSI(14).compute(df)
    factors["rsi_21"] = RSI(21).compute(df)
    
    factors["atr_14"] = ATR(14).compute(df)
    
    boll = BollingerBands()
    boll_result = boll.compute(df)
    factors["bb_upper"] = boll_result["upper"]
    factors["bb_middle"] = boll_result["middle"]
    factors["bb_lower"] = boll_result["lower"]
    factors["bb_width"] = (boll_result["upper"] - boll_result["lower"]) / boll_result["middle"]
    factors["bb_pct_b"] = (df["close"] - boll_result["lower"]) / (boll_result["upper"] - boll_result["lower"])
    
    kdj = KDJ().compute(df)
    factors["kdj_k"] = kdj["K"]
    factors["kdj_d"] = kdj["D"]
    factors["kdj_j"] = kdj["J"]
    
    factors["williams_r"] = WilliamsR().compute(df)
    
    factors["bias_6"] = BIAS(6).compute(df)
    factors["bias_12"] = BIAS(12).compute(df)
    factors["bias_24"] = BIAS(24).compute(df)
    
    factors["obv"] = OBV().compute(df)
    
    factors["cci"] = CCI().compute(df)
    
    dmi = DMI().compute(df)
    factors["dmi_pdi"] = dmi["+DI"]
    factors["dmi_mdi"] = dmi["-DI"]
    factors["dmi_adx"] = ADX().compute(df)
    
    factors["bbi"] = BBIFactor().compute(df)
    
    factors["amplitude"] = AmplitudeFactor().compute(df)
    factors["volume_change"] = VolumeChangeFactor().compute(df)
    
    # ========== 均线偏离因子 ==========
    factors["ma5_distance"] = MADistanceFactor(5).compute(df)
    factors["ma10_distance"] = MADistanceFactor(10).compute(df)
    factors["ma20_distance"] = MADistanceFactor(20).compute(df)
    factors["ma60_distance"] = MADistanceFactor(60).compute(df)
    factors["ma120_distance"] = MADistanceFactor(120).compute(df)
    
    # ========== 成交量因子 ==========
    factors["volume_ratio_5"] = VolumeRatioFactor(5).compute(df)
    factors["volume_ratio_10"] = VolumeRatioFactor(10).compute(df)
    factors["volume_ratio_20"] = VolumeRatioFactor(20).compute(df)
    
    factors["vwap_deviation"] = VWAPDeviationFactor().compute(df)
    
    # ========== 动量因子 ==========
    factors["momentum_5"] = Momentum(5).compute(df)
    factors["momentum_10"] = Momentum(10).compute(df)
    factors["momentum_20"] = Momentum(20).compute(df)
    factors["momentum_60"] = Momentum(60).compute(df)
    
    factors["roc_5"] = ROC(5).compute(df)
    factors["roc_10"] = ROC(10).compute(df)
    factors["roc_20"] = ROC(20).compute(df)
    
    factors["reversal_5"] = ReversalFactor(5).compute(df)
    factors["reversal_10"] = ReversalFactor(10).compute(df)
    
    factors["momentum_skip5"] = MomentumSkip5Factor().compute(df)
    
    # ========== 趋势因子 ==========
    factors["trend_slope_20"] = TrendSlopeFactor(20).compute(df)
    factors["trend_strength_20"] = TrendStrengthFactor(20).compute(df)
    factors["donchian_breakout_20"] = DonchianBreakoutFactor(20).compute(df)
    factors["price_range_ratio_20"] = PriceRangeRatioFactor(20).compute(df)
    
    # ========== 波动率因子 ==========
    factors["volatility_20"] = Volatility(20).compute(df)
    factors["volatility_60"] = Volatility(60).compute(df)
    factors["downside_vol_20"] = DownsideVolatility(20).compute(df)
    
    # ========== 统计因子 ==========
    factors["sharpe_ratio_60"] = SharpeRatioFactor(60).compute(df)
    factors["return_skew_60"] = ReturnSkewFactor(60).compute(df)
    factors["return_kurt_60"] = ReturnKurtFactor(60).compute(df)
    factors["tail_risk_60"] = TailRiskFactor(60).compute(df)
    
    # ========== Alpha101 因子 ==========
    factors["alpha001"] = Alpha001Factor(5).compute(df)
    factors["alpha002"] = Alpha002Factor(5).compute(df)
    factors["alpha003"] = Alpha003Factor(5).compute(df)
    factors["alpha004"] = Alpha004Factor().compute(df)
    factors["alpha005"] = Alpha005Factor(5).compute(df)
    factors["alpha006"] = Alpha006Factor(5).compute(df)
    factors["alpha007"] = Alpha007Factor(5).compute(df)
    factors["alpha008"] = Alpha008Factor(5).compute(df)
    factors["alpha009"] = Alpha009Factor().compute(df)
    factors["alpha010"] = Alpha010Factor().compute(df)
    
    # ========== Alpha191 因子 ==========
    factors["alpha191_01"] = Alpha191_01Factor().compute(df)
    factors["alpha191_02"] = Alpha191_02Factor().compute(df)
    factors["alpha191_03"] = Alpha191_03Factor().compute(df)
    factors["alpha191_04"] = Alpha191_04Factor(5).compute(df)
    factors["alpha191_05"] = Alpha191_05Factor(5).compute(df)
    factors["alpha191_06"] = Alpha191_06Factor().compute(df)
    factors["alpha191_07"] = Alpha191_07Factor().compute(df)
    factors["alpha191_08"] = Alpha191_08Factor().compute(df)
    factors["alpha191_09"] = Alpha191_09Factor().compute(df)
    factors["alpha191_10"] = Alpha191_10Factor().compute(df)
    factors["alpha191_11"] = Alpha191_11Factor().compute(df)
    factors["alpha191_12"] = Alpha191_12Factor().compute(df)
    factors["alpha191_13"] = Alpha191_13Factor().compute(df)
    factors["alpha191_14"] = Alpha191_14Factor(5).compute(df)
    factors["alpha191_15"] = Alpha191_15Factor().compute(df)

    # ========== 补充技术指标 ==========
    stoch = Stochastic(14).compute(df)
    factors["stoch_k"] = stoch["%K"]
    factors["stoch_d"] = stoch["%D"]
    factors["psy_12"] = PSY(12).compute(df)
    factors["vr_24"] = VR(24).compute(df)
    factors["mass_index"] = MassIndex().compute(df)
    factors["parabolic_sar"] = ParabolicSAR().compute(df)
    vortex = VortexIndicator().compute(df)
    factors["vortex_plus"] = vortex["+VI"]
    factors["vortex_minus"] = vortex["-VI"]
    factors["cmf"] = ChaikinMoneyFlow().compute(df)
    factors["eom"] = EaseOfMovement().compute(df)
    kc = KeltnerChannel().compute(df)
    factors["keltner_upper"] = kc["upper"]
    factors["keltner_lower"] = kc["lower"]
    factors["keltner_width"] = (kc["upper"] - kc["lower"]) / kc["middle"]

    # ========== AKShare 新增因子（基于 stocks_daily 字段） ==========
    # 这些因子需要 outstanding_share/turnover 字段，AKShare 数据才有
    if "outstanding_share" in df.columns:
        factors["log_market_cap"] = LogMarketCap().compute(df)
    if "high" in df.columns and "low" in df.columns:
        factors["amplitude_1"] = Amplitude(1).compute(df)
        factors["amplitude_20"] = Amplitude(20).compute(df)
    if "turnover" in df.columns:
        factors["amihud_20"] = Amihud(20).compute(df)

    # ========== AKShare 财务因子（yjbb 业绩报表） ==========
    ak_fund = calculate_akshare_fundamental_factors(df, akshare_yjbb_path, akshare_zcfz_path)
    for col in ak_fund.columns:
        factors[col] = ak_fund[col]

    # ========== 宏观因子 ==========
    macro = calculate_macro_factors(df, macro_dir)
    for col in macro.columns:
        factors[col] = macro[col]

    # ========== 大盘趋势因子 ==========
    market_factors = calculate_market_factors(df, index_df)
    for col in market_factors.columns:
        factors[col] = market_factors[col]
    
    # ========== 基本面因子 ==========
    if fundamental_df is not None and len(fundamental_df) > 0:
        fundamental_factors = calculate_fundamental_factors(df, fundamental_df)
        for col in fundamental_factors.columns:
            factors[col] = fundamental_factors[col]
    
    # ========== 融资融券因子 ==========
    if margin_df is not None and len(margin_df) > 0:
        margin_factors = calculate_margin_factors(df, margin_df)
        for col in margin_factors.columns:
            factors[col] = margin_factors[col]
    
    return factors


def calculate_fundamental_factors(df: pd.DataFrame, fundamental_df: pd.DataFrame) -> pd.DataFrame:
    """计算基本面因子（Tushare fina_indicator）"""
    factors = pd.DataFrame(index=df.index)

    fundamental_df = fundamental_df.copy()
    fundamental_df['date'] = pd.to_datetime(fundamental_df['ann_date'])

    # 按股票匹配，使用 merge_asof 防止穿越
    ts_col = 'ts_code'
    sym_col = 'symbol'
    if ts_col in fundamental_df.columns and sym_col in df.columns:
        # 转换股票代码格式匹配
        df_codes = df[sym_col].str[:6].unique()
        fund_matched = fundamental_df[fundamental_df['ts_code'].str[:6].isin(df_codes)]
    else:
        fund_matched = fundamental_df

    field_cols = ['pe', 'pb', 'ps', 'roe', 'roa', 'grossprofit_margin',
                  'netprofit_margin', 'debt_to_assets', 'current_ratio']

    dst_map = {
        'pe': 'pe_ratio', 'pb': 'pb_ratio', 'ps': 'ps_ratio',
        'roe': 'roe', 'roa': 'roa',
        'grossprofit_margin': 'gross_profit_margin',
        'netprofit_margin': 'net_profit_margin',
        'debt_to_assets': 'debt_to_assets',
        'current_ratio': 'current_ratio',
    }

    for col in field_cols:
        if col not in fund_matched.columns:
            continue
        # 取每只股票的最新一期报告
        latest = fund_matched.sort_values('date').groupby(ts_col, as_index=False).last()
        dst = dst_map[col]
        for _, row in latest.iterrows():
            code = str(row[ts_col])[:6]
            val = row[col]
            mask = df[sym_col].str[:6] == code if sym_col in df.columns else pd.Series(False, index=df.index)
            factors.loc[mask, dst] = val

    return factors


def calculate_margin_factors(df: pd.DataFrame, margin_df: pd.DataFrame) -> pd.DataFrame:
    """计算融资融券因子"""
    factors = pd.DataFrame(index=df.index)
    
    margin_df = margin_df.copy()
    margin_df['date'] = pd.to_datetime(margin_df['trade_date'])
    margin_df = margin_df.sort_values('date').reset_index(drop=True)
    
    if 'fin_balance' in margin_df.columns:
        fin_balance = margin_df.set_index('date')['fin_balance']
        df_dates = pd.to_datetime(df['date'])
        factors["margin_balance"] = df_dates.map(fin_balance).values
    
    if 'fin_buy_amount' in margin_df.columns:
        fin_buy = margin_df.set_index('date')['fin_buy_amount']
        df_dates = pd.to_datetime(df['date'])
        factors["margin_buy_amount"] = df_dates.map(fin_buy).values
    
    if 'fin_buy_value' in margin_df.columns and 'fin_sell_value' in margin_df.columns:
        margin_df['net_margin_flow'] = margin_df['fin_buy_value'] - margin_df['fin_sell_value']
        net_flow = margin_df.set_index('date')['net_margin_flow']
        df_dates = pd.to_datetime(df['date'])
        factors["net_margin_flow"] = df_dates.map(net_flow).values
    
    return factors


def build_training_data(data_path: str, output_path: str = None,
                        index_path: str = None,
                        fundamental_path: str = None,
                        margin_path: str = None,
                        float_path: str = None,
                        akshare_yjbb_path: str = None,
                        akshare_zcfz_path: str = None,
                        macro_dir: str = None):
    """
    构建训练数据

    Args:
        data_path: 输入数据路径（支持 CSV、Parquet、Pickle）
        output_path: 输出路径，默认为 data/training_data/
        index_path: 指数数据路径
        fundamental_path: 基本面数据路径（Tushare fina_indicator）
        margin_path: 融资融券数据路径
        float_path: 流通股数据路径
        akshare_yjbb_path: AKShare 业绩报表路径（financial_yjbb_multi.parquet）
        akshare_zcfz_path: AKShare 资产负债表路径（balance_sheet_zcfz_multi.parquet）
        macro_dir: AKShare 宏观数据目录（factors_raw/）
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(data_path), "training_data")
    os.makedirs(output_path, exist_ok=True)
    
    print(f"正在读取数据: {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.pkl') or data_path.endswith('.pickle'):
        df = pd.read_pickle(data_path)
    else:
        raise ValueError("不支持的数据格式，请使用 CSV、Parquet 或 Pickle")
    
    print("正在适配数据...")
    df = FactorDataAdapter.adapt(df)
    
    if 'date' not in df.columns:
        raise ValueError("数据中缺少 'date' 列")
    
    df = df.sort_values('date').reset_index(drop=True)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= '2015-01-01']
        print(f"筛选后数据范围: {df['date'].min()} 至 {df['date'].max()}")
    
    # 添加市场分类信息
    print("正在添加市场分类信息...")
    if 'ts_code' in df.columns:
        df['market_type'] = df['ts_code'].apply(get_market_type)
        df['market_code'] = df['market_type'].apply(get_market_code)
    elif 'symbol' in df.columns:
        df['market_type'] = df['symbol'].apply(get_market_type)
        df['market_code'] = df['market_type'].apply(get_market_code)
    print(f"市场分布: {df['market_type'].value_counts().to_dict()}")
    
    # 加载指数数据
    index_df = None
    if index_path and os.path.exists(index_path):
        print(f"正在加载指数数据: {index_path}")
        index_df = pd.read_parquet(index_path)
        print(f"指数数据记录数: {len(index_df)}")
    
    # 加载基本面数据
    fundamental_df = None
    if fundamental_path and os.path.exists(fundamental_path):
        print(f"正在加载基本面数据: {fundamental_path}")
        fundamental_df = pd.read_parquet(fundamental_path)
        if 'ts_code' in fundamental_df.columns and 'symbol' in df.columns:
            symbol = df['symbol'].iloc[0] if len(df) > 0 else None
            if symbol:
                fundamental_df = fundamental_df[fundamental_df['ts_code'].str.startswith(symbol[:6])]
        print(f"基本面数据记录数: {len(fundamental_df)}")
    
    # 加载融资融券数据
    margin_df = None
    if margin_path and os.path.exists(margin_path):
        print(f"正在加载融资融券数据: {margin_path}")
        margin_df = pd.read_parquet(margin_path)
        if 'ts_code' in margin_df.columns and 'symbol' in df.columns:
            symbol = df['symbol'].iloc[0] if len(df) > 0 else None
            if symbol:
                margin_df = margin_df[margin_df['ts_code'].str.startswith(symbol[:6])]
        print(f"融资融券数据记录数: {len(margin_df)}")
    
    # 加载流通股数据
    float_df = None
    if float_path and os.path.exists(float_path):
        print(f"正在加载流通股数据: {float_path}")
        float_df = pd.read_parquet(float_path)
        if 'ts_code' in float_df.columns and 'symbol' in df.columns:
            symbol = df['symbol'].iloc[0] if len(df) > 0 else None
            if symbol:
                float_df = float_df[float_df['ts_code'].str.startswith(symbol[:6])]
        print(f"流通股数据记录数: {len(float_df)}")
    
    print("正在计算因子...")
    factors = calculate_factors(
        df, index_df, fundamental_df, margin_df,
        akshare_yjbb_path=akshare_yjbb_path,
        akshare_zcfz_path=akshare_zcfz_path,
        macro_dir=macro_dir,
    )
    
    # 计算市值因子
    print("正在计算市值因子...")
    market_value_factors = calculate_market_value_factors(df, float_df)
    factors = pd.concat([factors, market_value_factors], axis=1)
    
    print("正在计算 labels...")
    labels = calculate_labels(df)
    
    print("正在合并数据...")
    training_data = pd.concat([df, factors, labels], axis=1)
    
    # ========== 只保留满足 B1 策略买入条件的样本 ==========
    print("正在筛选满足 B1 策略买入条件的样本...")
    b1_signal = calculate_b1_signal(df)
    initial_len = len(training_data)
    training_data = training_data[b1_signal == 1].copy()
    print(f"B1 策略筛选: {initial_len} -> {len(training_data)}")
    
    if len(training_data) == 0:
        print("警告: 没有满足 B1 策略买入条件的样本！")
        return None
    
    # 保留关键列（包含市场分类）
    keep_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    if 'symbol' in training_data.columns:
        keep_columns.insert(1, 'symbol')
    if 'ts_code' in training_data.columns:
        keep_columns.insert(2, 'ts_code')
    if 'market_type' in training_data.columns:
        keep_columns.insert(3, 'market_type')
    if 'market_code' in training_data.columns:
        keep_columns.insert(4, 'market_code')
    keep_columns += [col for col in factors.columns]
    keep_columns += [col for col in labels.columns]
    training_data = training_data[keep_columns]
    
    # 删除包含 NaN 的行
    initial_len = len(training_data)
    training_data = training_data.dropna()
    print(f"删除 NaN 行: {initial_len} -> {len(training_data)}")
    
    # 保存数据
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_path, f'training_data_{timestamp}.parquet')
    training_data.to_parquet(output_file)
    print(f"训练数据已保存到: {output_file}")
    
    # 输出统计信息
    print("\n=== 训练数据统计 ===")
    print(f"样本数量: {len(training_data)}")
    print(f"因子数量: {len(factors.columns)}")
    print(f"Label 数量: {len(labels.columns)}")
    print(f"特征总数: {len(factors.columns) + len(labels.columns)}")
    print(f"\n市场分布:")
    print(training_data['market_type'].value_counts().to_dict())
    print(f"\nLabel 分布:")
    for col in labels.columns:
        if 'up' in col or 'signal' in col or 'over' in col:
            dist = training_data[col].value_counts().to_dict()
            print(f"  {col}: {dist}")
    
    return training_data


def generate_sample_data(n_days: int = 200, n_stocks: int = 1) -> pd.DataFrame:
    """
    生成模拟数据用于测试
    
    Args:
        n_days: 天数
        n_stocks: 股票数量
    
    Returns:
        模拟的股票日线数据
    """
    np.random.seed(42)
    
    all_data = []
    
    for stock_idx in range(n_stocks):
        symbol = f"00000{stock_idx + 1}.SZ"
        
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        close = 10 + np.cumsum(np.random.randn(n_days) * 0.02)
        open_price = close + np.random.randn(n_days) * 0.01
        high = np.maximum(close, open_price) + np.random.rand(n_days) * 0.05
        low = np.minimum(close, open_price) - np.random.rand(n_days) * 0.05
        volume = np.random.randint(1000000, 10000000, n_days)
        turnover = volume * close
        
        df = pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'turnover': turnover
        })
        
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)


def load_stock_data(data_dir: str, n_stocks: int = None, start_date: str = '2015-01-01') -> pd.DataFrame:
    """从目录加载股票数据"""
    import glob
    
    pattern = os.path.join(data_dir, '*.parquet')
    files = sorted(glob.glob(pattern))
    
    if n_stocks:
        files = files[:n_stocks]
    
    print(f"找到 {len(files)} 个股票数据文件")
    
    all_data = []
    for i, file in enumerate(files):
        try:
            df = pd.read_parquet(file)
            
            filename = os.path.basename(file)
            symbol = filename.replace('.parquet', '')
            
            if 'symbol' not in df.columns and 'ts_code' not in df.columns:
                df['symbol'] = symbol
            
            all_data.append(df)
            
            if (i + 1) % 50 == 0:
                print(f"已加载 {i + 1}/{len(files)} 个文件")
        except Exception as e:
            print(f"加载文件失败 {file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("没有成功加载任何数据文件")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"合并后总记录数: {len(combined)}")
    
    return combined


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='基于 B1 策略构建训练数据')
    parser.add_argument('--input', help='输入数据文件路径')
    parser.add_argument('--data_dir', help='股票数据目录路径')
    parser.add_argument('--output', help='输出目录路径')
    parser.add_argument('--index', help='指数数据路径', default='/Users/didi/Project/quant/data/raw/index_000001.SH.parquet')
    parser.add_argument('--fundamental', help='基本面数据路径', default='/Users/didi/Project/quant/data/raw/fina_indicator.parquet')
    parser.add_argument('--margin', help='融资融券数据路径', default='/Users/didi/Project/quant/data/raw/margin.parquet')
    parser.add_argument('--float', dest='float_path', help='流通股数据路径', default='/Users/didi/Project/quant/data/raw/share_float.parquet')
    parser.add_argument('--akshare-yjbb', help='AKShare 业绩报表路径', default='/Users/didi/Project/quant/data/factors_raw/financial_yjbb_multi.parquet')
    parser.add_argument('--akshare-zcfz', help='AKShare 资产负债表路径', default='/Users/didi/Project/quant/data/factors_raw/balance_sheet_zcfz_multi.parquet')
    parser.add_argument('--macro-dir', help='AKShare 宏观数据目录', default='/Users/didi/Project/quant/data/factors_raw')
    parser.add_argument('--sample', action='store_true', help='使用模拟数据进行测试')
    parser.add_argument('--n_days', type=int, default=200, help='模拟数据天数')
    parser.add_argument('--n_stocks', type=int, default=None, help='模拟数据股票数量或加载数据数量（不指定则加载全部）')

    args = parser.parse_args()
    
    if args.sample:
        print(f"生成模拟数据: {args.n_days} 天, {args.n_stocks} 只股票")
        df = generate_sample_data(args.n_days, args.n_stocks)
        
        temp_file = '/tmp/sample_data.parquet'
        df.to_parquet(temp_file)
        print(f"模拟数据已保存到: {temp_file}")
        
        build_training_data(temp_file, args.output)
    elif args.data_dir:
        print(f"从目录加载数据: {args.data_dir}")
        df = load_stock_data(args.data_dir, args.n_stocks)
        
        temp_file = '/tmp/loaded_data.parquet'
        df.to_parquet(temp_file)
        print(f"数据已保存到: {temp_file}")
        
        build_training_data(
            temp_file,
            args.output,
            index_path=args.index,
            fundamental_path=args.fundamental,
            margin_path=args.margin,
            float_path=args.float_path,
            akshare_yjbb_path=args.akshare_yjbb,
            akshare_zcfz_path=args.akshare_zcfz,
            macro_dir=args.macro_dir,
        )
    elif args.input:
        build_training_data(
            args.input,
            args.output,
            index_path=args.index,
            fundamental_path=args.fundamental,
            margin_path=args.margin,
            float_path=args.float_path,
            akshare_yjbb_path=args.akshare_yjbb,
            akshare_zcfz_path=args.akshare_zcfz,
            macro_dir=args.macro_dir,
        )
    else:
        print("请使用以下方式之一:")
        print("  1. --input 指定输入文件")
        print("  2. --data_dir 指定数据目录")
        print("  3. --sample 生成模拟数据进行测试")
        print("\n示例:")
        print("  python build_training_data.py --sample --n_days 200")
        print("  python build_training_data.py --data_dir ./data/stocks --n_stocks 10")
        print("  python build_training_data.py --input /path/to/data.parquet")
