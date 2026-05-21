"""
技术指标因子库

包含常用的技术分析指标：
- 移动平均线 (MA)
- 指数移动平均线 (EMA)
- 移动平均收敛发散 (MACD)
- 相对强弱指数 (RSI)
- 布林带 (Bollinger Bands)
- 平均真实波动幅度 (ATR)
- 成交量比率 (Volume Ratio)
"""

import pandas as pd
import numpy as np
from .base import Factor, RollingFactor


class MA(RollingFactor):
    """
    简单移动平均线 (Simple Moving Average)
    
    参数:
        window: 窗口大小（默认 20）
        field: 计算字段（默认 'close'）
    """
    
    def __init__(self, window: int = 20, field: str = "close"):
        super().__init__(name=f"MA{window}", window=window, params={"field": field})
        self.field = field
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算移动平均线"""
        return data[self.field].rolling(window=self.window).mean()


class EMA(RollingFactor):
    """
    指数移动平均线 (Exponential Moving Average)
    
    参数:
        window: 窗口大小（默认 20）
        field: 计算字段（默认 'close'）
        adjust: 是否调整（默认 True）
    """
    
    def __init__(self, window: int = 20, field: str = "close", adjust: bool = True):
        super().__init__(name=f"EMA{window}", window=window, params={"field": field, "adjust": adjust})
        self.field = field
        self.adjust = adjust
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算指数移动平均线"""
        return data[self.field].ewm(span=self.window, adjust=self.adjust).mean()


class MACD(Factor):
    """
    移动平均收敛发散指标 (MACD)
    
    参数:
        fast_window: 快速EMA窗口（默认 12）
        slow_window: 慢速EMA窗口（默认 26）
        signal_window: 信号线窗口（默认 9）
        field: 计算字段（默认 'close'）
    """
    
    def __init__(self, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9, field: str = "close"):
        super().__init__(name="MACD", params={
            "fast_window": fast_window, 
            "slow_window": slow_window,
            "signal_window": signal_window,
            "field": field
        })
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window
        self.field = field
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 MACD 线"""
        ema_fast = EMA(self.fast_window, self.field).compute(data)
        ema_slow = EMA(self.slow_window, self.field).compute(data)
        return ema_fast - ema_slow
    
    def compute_signal(self, data: pd.DataFrame) -> pd.Series:
        """计算信号线"""
        macd_line = self.compute(data)
        return macd_line.ewm(span=self.signal_window, adjust=True).mean()
    
    def compute_histogram(self, data: pd.DataFrame) -> pd.Series:
        """计算柱状图"""
        macd_line = self.compute(data)
        signal_line = self.compute_signal(data)
        return macd_line - signal_line


class RSI(RollingFactor):
    """
    相对强弱指数 (Relative Strength Index)
    
    参数:
        window: 窗口大小（默认 14）
        field: 计算字段（默认 'close'）
    """
    
    def __init__(self, window: int = 14, field: str = "close"):
        super().__init__(name=f"RSI{window}", window=window, params={"field": field})
        self.field = field
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算相对强弱指数"""
        prices = data[self.field]
        deltas = prices.diff()
        
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.rolling(window=self.window).mean()
        avg_loss = losses.rolling(window=self.window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class BollingerBands(Factor):
    """
    布林带指标 (Bollinger Bands)
    
    参数:
        window: 窗口大小（默认 20）
        num_std: 标准差倍数（默认 2）
        field: 计算字段（默认 'close'）
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0, field: str = "close"):
        super().__init__(name="BollingerBands", params={
            "window": window, 
            "num_std": num_std,
            "field": field
        })
        self.window = window
        self.num_std = num_std
        self.field = field
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算布林带（返回包含上轨、中轨、下轨的 DataFrame）"""
        prices = data[self.field]
        
        middle_band = prices.rolling(window=self.window).mean()
        std = prices.rolling(window=self.window).std()
        
        upper_band = middle_band + self.num_std * std
        lower_band = middle_band - self.num_std * std
        
        return pd.DataFrame({
            "upper": upper_band,
            "middle": middle_band,
            "lower": lower_band
        })


class ATR(RollingFactor):
    """
    平均真实波动幅度 (Average True Range)
    
    参数:
        window: 窗口大小（默认 14）
    """
    
    def __init__(self, window: int = 14):
        super().__init__(name=f"ATR{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算平均真实波动幅度"""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # 计算真实波动范围
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算平均真实波动幅度
        return true_range.rolling(window=self.window).mean()


class VolumeRatio(RollingFactor):
    """
    成交量比率 (Volume Ratio)
    
    参数:
        window: 窗口大小（默认 5）
    """
    
    def __init__(self, window: int = 5):
        super().__init__(name=f"VolumeRatio{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算成交量比率（当前成交量与过去N日均量的比值）"""
        volume = data["volume"]
        avg_volume = volume.rolling(window=self.window).mean()
        return volume / avg_volume


class ROC(RollingFactor):
    """
    变动率指标 (Rate of Change)
    
    参数:
        window: 窗口大小（默认 12）
        field: 计算字段（默认 'close'）
    """
    
    def __init__(self, window: int = 12, field: str = "close"):
        super().__init__(name=f"ROC{window}", window=window, params={"field": field})
        self.field = field
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算变动率指标"""
        prices = data[self.field]
        return (prices / prices.shift(self.window) - 1) * 100


class Stochastic(RollingFactor):
    """
    随机指标 (Stochastic Oscillator)
    
    参数:
        window: 窗口大小（默认 14）
        smooth_window: 平滑窗口（默认 3）
    """
    
    def __init__(self, window: int = 14, smooth_window: int = 3):
        super().__init__(name="Stochastic", window=window, params={"smooth_window": smooth_window})
        self.smooth_window = smooth_window
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算随机指标（返回 %K 和 %D）"""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # 计算 %K
        lowest_low = low.rolling(window=self.window).min()
        highest_high = high.rolling(window=self.window).max()
        k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # 计算 %D（%K 的移动平均）
        d = k.rolling(window=self.smooth_window).mean()
        
        return pd.DataFrame({"%K": k, "%D": d})


class KDJ(Factor):
    """
    KDJ 指标 (随机指标的改进版)
    
    参数:
        window: 窗口大小（默认 9）
        k_window: K 值平滑窗口（默认 3）
        d_window: D 值平滑窗口（默认 3）
    """
    
    def __init__(self, window: int = 9, k_window: int = 3, d_window: int = 3):
        super().__init__(name="KDJ", params={
            "window": window,
            "k_window": k_window,
            "d_window": d_window
        })
        self.window = window
        self.k_window = k_window
        self.d_window = d_window
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 KDJ 指标（返回 K、D、J）"""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # 计算 RSV (未成熟随机值)
        lowest_low = low.rolling(window=self.window).min()
        highest_high = high.rolling(window=self.window).max()
        rsv = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # 计算 K 值（RSV 的指数平滑）
        k = rsv.ewm(alpha=1/self.k_window, adjust=False).mean()
        
        # 计算 D 值（K 值的指数平滑）
        d = k.ewm(alpha=1/self.d_window, adjust=False).mean()
        
        # 计算 J 值 (J = 3K - 2D)
        j = 3 * k - 2 * d
        
        return pd.DataFrame({"K": k, "D": d, "J": j})


class WilliamsR(RollingFactor):
    """
    威廉指标 (Williams %R)
    
    参数:
        window: 窗口大小（默认 14）
    """
    
    def __init__(self, window: int = 14):
        super().__init__(name=f"WilliamsR{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算威廉指标"""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        lowest_low = low.rolling(window=self.window).min()
        highest_high = high.rolling(window=self.window).max()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r


class BIAS(RollingFactor):
    """
    乖离率 (BIAS)
    
    参数:
        window: 窗口大小（默认 6）
        field: 计算字段（默认 'close'）
    """
    
    def __init__(self, window: int = 6, field: str = "close"):
        super().__init__(name=f"BIAS{window}", window=window, params={"field": field})
        self.field = field
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算乖离率（当前价格与均线的偏离程度）"""
        prices = data[self.field]
        ma = prices.rolling(window=self.window).mean()
        
        bias = ((prices - ma) / ma) * 100
        
        return bias


class Momentum(RollingFactor):
    """
    动量指标 (Momentum)
    
    参数:
        window: 窗口大小（默认 12）
        field: 计算字段（默认 'close'）
    """
    
    def __init__(self, window: int = 12, field: str = "close"):
        super().__init__(name=f"Momentum{window}", window=window, params={"field": field})
        self.field = field
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算动量指标（当前价格与N日前价格的差值）"""
        prices = data[self.field]
        momentum = prices - prices.shift(self.window)
        
        return momentum


class PSY(RollingFactor):
    """
    心理线指标 (Psychological Line)
    
    参数:
        window: 窗口大小（默认 12）
    """
    
    def __init__(self, window: int = 12):
        super().__init__(name=f"PSY{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算心理线（上涨天数占比）"""
        close = data["close"]
        prev_close = close.shift(1)
        
        # 判断每天是否上涨
        up_days = (close > prev_close).rolling(window=self.window).sum()
        
        # 计算上涨天数占比
        psy = (up_days / self.window) * 100
        
        return psy


class VR(RollingFactor):
    """
    成交量变异率 (Volume Ratio)
    
    参数:
        window: 窗口大小（默认 24）
    """
    
    def __init__(self, window: int = 24):
        super().__init__(name=f"VR{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算成交量变异率"""
        close = data["close"]
        volume = data["volume"]
        prev_close = close.shift(1)
        
        # 上涨日成交量总和
        up_volume = volume.where(close > prev_close, 0).rolling(window=self.window).sum()
        
        # 下跌日成交量总和
        down_volume = volume.where(close < prev_close, 0).rolling(window=self.window).sum()
        
        # 计算 VR
        vr = (up_volume / down_volume) * 100
        
        return vr


class OBV(Factor):
    """
    能量潮指标 (On-Balance Volume)
    """
    
    def __init__(self):
        super().__init__(name="OBV")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算能量潮指标"""
        close = data["close"]
        volume = data["volume"]
        prev_close = close.shift(1)
        
        # 根据涨跌方向累加成交量
        obv = pd.Series(0.0, index=data.index)
        
        for i in range(1, len(data)):
            if close.iloc[i] > prev_close.iloc[i]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < prev_close.iloc[i]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv


class CCI(RollingFactor):
    """
    顺势指标 (Commodity Channel Index)
    
    参数:
        window: 窗口大小（默认 20）
    """
    
    def __init__(self, window: int = 20):
        super().__init__(name=f"CCI{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算顺势指标"""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # 计算典型价格
        tp = (high + low + close) / 3
        
        # 计算简单移动平均
        tp_ma = tp.rolling(window=self.window).mean()
        
        # 计算平均绝对偏差（手动实现，兼容新版本 pandas）
        def mad_func(x):
            return abs(x - x.mean()).mean()
        
        mad = tp.rolling(window=self.window).apply(mad_func)
        
        # 计算 CCI
        cci = (tp - tp_ma) / (0.015 * mad)
        
        return cci


class DMI(Factor):
    """
    动向指标 (Directional Movement Index)
    
    参数:
        window: 窗口大小（默认 14）
        adx_window: ADX 平滑窗口（默认 14）
    """
    
    def __init__(self, window: int = 14, adx_window: int = 14):
        super().__init__(name="DMI", params={
            "window": window,
            "adx_window": adx_window
        })
        self.window = window
        self.adx_window = adx_window
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 DMI 指标（返回 +DI、-DI、ADX）"""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # 计算上升动向 +DM 和下降动向 -DM
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = -minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # 计算真实波动范围 TR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算 +DI 和 -DI
        plus_di = 100 * (plus_dm.rolling(window=self.window).sum() / tr.rolling(window=self.window).sum())
        minus_di = 100 * (minus_dm.rolling(window=self.window).sum() / tr.rolling(window=self.window).sum())
        
        # 计算 ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(window=self.adx_window).mean()
        
        return pd.DataFrame({"+DI": plus_di, "-DI": minus_di, "ADX": adx})


# ==================== 盈利惊喜因子 (Earnings Surprise Factors) ====================

class NetProfitYoY(Factor):
    """
    净利润同比增长率 (Net Profit Year-on-Year Growth)
    
    参数:
        period: 周期类型（'quarterly' 季度, 'annual' 年度）
    """
    
    def __init__(self, period: str = "quarterly"):
        super().__init__(name=f"NetProfitYoY_{period}", params={"period": period})
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算净利润同比增长率"""
        if "net_profit" not in data.columns:
            raise ValueError("数据中缺少 'net_profit' 列")
        
        net_profit = data["net_profit"]
        
        # 根据周期类型选择偏移量
        if self.period == "quarterly":
            offset = 4  # 同比去年同期（季度数据）
        else:
            offset = 1  # 同比去年（年度数据）
        
        # 计算同比增长率
        yoy_growth = ((net_profit - net_profit.shift(offset)) / net_profit.shift(offset).abs()) * 100
        
        return yoy_growth


class RevenueYoY(Factor):
    """
    营业收入同比增长率 (Revenue Year-on-Year Growth)
    
    参数:
        period: 周期类型（'quarterly' 季度, 'annual' 年度）
    """
    
    def __init__(self, period: str = "quarterly"):
        super().__init__(name=f"RevenueYoY_{period}", params={"period": period})
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算营业收入同比增长率"""
        if "revenue" not in data.columns:
            raise ValueError("数据中缺少 'revenue' 列")
        
        revenue = data["revenue"]
        
        # 根据周期类型选择偏移量
        if self.period == "quarterly":
            offset = 4  # 同比去年同期（季度数据）
        else:
            offset = 1  # 同比去年（年度数据）
        
        # 计算同比增长率
        yoy_growth = ((revenue - revenue.shift(offset)) / revenue.shift(offset).abs()) * 100
        
        return yoy_growth


class EPS(Factor):
    """
    每股收益 (Earnings Per Share)
    
    参数:
        diluted: 是否计算稀释每股收益（默认 False）
    """
    
    def __init__(self, diluted: bool = False):
        super().__init__(name="EPS_Diluted" if diluted else "EPS", params={"diluted": diluted})
        self.diluted = diluted
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算每股收益"""
        if "net_profit" not in data.columns:
            raise ValueError("数据中缺少 'net_profit' 列")
        if "shares_outstanding" not in data.columns:
            raise ValueError("数据中缺少 'shares_outstanding' 列")
        
        net_profit = data["net_profit"]
        shares = data["shares_outstanding"]
        
        # 计算每股收益
        eps = net_profit / shares
        
        return eps


class EarningsSurprise(Factor):
    """
    盈利惊喜 (Earnings Surprise)
    
    计算实际盈利与市场预期的偏差百分比
    
    参数:
        period: 周期类型（'quarterly' 季度, 'annual' 年度）
    """
    
    def __init__(self, period: str = "quarterly"):
        super().__init__(name=f"EarningsSurprise_{period}", params={"period": period})
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算盈利惊喜百分比"""
        if "actual_eps" not in data.columns:
            raise ValueError("数据中缺少 'actual_eps' 列（实际每股收益）")
        if "estimated_eps" not in data.columns:
            raise ValueError("数据中缺少 'estimated_eps' 列（预期每股收益）")
        
        actual = data["actual_eps"]
        estimated = data["estimated_eps"]
        
        # 计算盈利惊喜百分比
        surprise = ((actual - estimated) / estimated.abs()) * 100
        
        return surprise


class RevenueSurprise(Factor):
    """
    营收惊喜 (Revenue Surprise)
    
    计算实际营收与市场预期的偏差百分比
    
    参数:
        period: 周期类型（'quarterly' 季度, 'annual' 年度）
    """
    
    def __init__(self, period: str = "quarterly"):
        super().__init__(name=f"RevenueSurprise_{period}", params={"period": period})
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算营收惊喜百分比"""
        if "actual_revenue" not in data.columns:
            raise ValueError("数据中缺少 'actual_revenue' 列（实际营收）")
        if "estimated_revenue" not in data.columns:
            raise ValueError("数据中缺少 'estimated_revenue' 列（预期营收）")
        
        actual = data["actual_revenue"]
        estimated = data["estimated_revenue"]
        
        # 计算营收惊喜百分比
        surprise = ((actual - estimated) / estimated.abs()) * 100
        
        return surprise


class NetProfitQoQ(Factor):
    """
    净利润环比增长率 (Net Profit Quarter-on-Quarter Growth)
    """
    
    def __init__(self):
        super().__init__(name="NetProfitQoQ")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算净利润环比增长率"""
        if "net_profit" not in data.columns:
            raise ValueError("数据中缺少 'net_profit' 列")
        
        net_profit = data["net_profit"]
        
        # 计算环比增长率（与上一季度相比）
        qoq_growth = ((net_profit - net_profit.shift(1)) / net_profit.shift(1).abs()) * 100
        
        return qoq_growth


class RevenueQoQ(Factor):
    """
    营业收入环比增长率 (Revenue Quarter-on-Quarter Growth)
    """
    
    def __init__(self):
        super().__init__(name="RevenueQoQ")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算营业收入环比增长率"""
        if "revenue" not in data.columns:
            raise ValueError("数据中缺少 'revenue' 列")
        
        revenue = data["revenue"]
        
        # 计算环比增长率（与上一季度相比）
        qoq_growth = ((revenue - revenue.shift(1)) / revenue.shift(1).abs()) * 100
        
        return qoq_growth


class NonRecurringProfitYoY(Factor):
    """
    扣非净利润同比增长率 (Non-Recurring Profit Year-on-Year Growth)
    
    参数:
        period: 周期类型（'quarterly' 季度, 'annual' 年度）
    """
    
    def __init__(self, period: str = "quarterly"):
        super().__init__(name=f"NonRecurringProfitYoY_{period}", params={"period": period})
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算扣非净利润同比增长率"""
        if "non_recurring_profit" not in data.columns:
            raise ValueError("数据中缺少 'non_recurring_profit' 列")
        
        profit = data["non_recurring_profit"]
        
        # 根据周期类型选择偏移量
        if self.period == "quarterly":
            offset = 4
        else:
            offset = 1
        
        # 计算同比增长率
        yoy_growth = ((profit - profit.shift(offset)) / profit.shift(offset).abs()) * 100
        
        return yoy_growth


class EarningsQuality(Factor):
    """
    盈利质量因子 (Earnings Quality)
    
    通过比较经营现金流与净利润来衡量盈利质量
    
    盈利质量 = 经营活动产生的现金流量净额 / 净利润
    """
    
    def __init__(self):
        super().__init__(name="EarningsQuality")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算盈利质量因子"""
        if "operating_cash_flow" not in data.columns:
            raise ValueError("数据中缺少 'operating_cash_flow' 列")
        if "net_profit" not in data.columns:
            raise ValueError("数据中缺少 'net_profit' 列")
        
        cash_flow = data["operating_cash_flow"]
        net_profit = data["net_profit"]
        
        # 计算盈利质量
        quality = cash_flow / net_profit.replace(0, np.nan)
        
        return quality


class PEG(Factor):
    """
    市盈率相对盈利增长比率 (PEG Ratio)
    
    PEG = PE / 盈利增长率
    
    参数:
        period: 周期类型（'quarterly' 季度, 'annual' 年度）
    """
    
    def __init__(self, period: str = "quarterly"):
        super().__init__(name=f"PEG_{period}", params={"period": period})
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 PEG 比率"""
        if "pe_ratio" not in data.columns:
            raise ValueError("数据中缺少 'pe_ratio' 列")
        if "net_profit" not in data.columns:
            raise ValueError("数据中缺少 'net_profit' 列")
        
        pe = data["pe_ratio"]
        net_profit = data["net_profit"]
        
        # 计算盈利增长率
        if self.period == "quarterly":
            offset = 4
        else:
            offset = 1
        
        growth_rate = (net_profit - net_profit.shift(offset)) / net_profit.shift(offset).abs()
        
        # 计算 PEG（增长率转换为百分比形式）
        peg = pe / (growth_rate * 100)
        
        return peg


# ==================== 多因子模型常用因子 ====================

# ---------- 市值因子 (Size) ----------

class MarketCap(Factor):
    """
    市值因子 (Market Capitalization)
    
    参数:
        log_transform: 是否对数化处理（默认 True）
    """
    
    def __init__(self, log_transform: bool = True):
        super().__init__(name="MarketCap_Log" if log_transform else "MarketCap", 
                         params={"log_transform": log_transform})
        self.log_transform = log_transform
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算市值因子"""
        if "market_cap" not in data.columns:
            raise ValueError("数据中缺少 'market_cap' 列")
        
        market_cap = data["market_cap"]
        
        if self.log_transform:
            # 对数化处理，使分布更接近正态
            return np.log(market_cap)
        return market_cap


class SizeDecile(Factor):
    """
    市值分位因子 (Size Decile)
    
    将股票按市值分为 10 个分组
    
    参数:
        ascending: 是否升序排列（默认 False，即大市值在前）
    """
    
    def __init__(self, ascending: bool = False):
        super().__init__(name="SizeDecile", params={"ascending": ascending})
        self.ascending = ascending
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算市值分位"""
        if "market_cap" not in data.columns:
            raise ValueError("数据中缺少 'market_cap' 列")
        
        market_cap = data["market_cap"]
        
        # 计算分位数（1-10）
        decile = pd.qcut(market_cap, 10, labels=False) + 1
        
        if not self.ascending:
            # 反转分位，大市值为高分位
            decile = 11 - decile
        
        return decile


# ---------- 价值因子 (Value) ----------

class PERatio(Factor):
    """
    市盈率因子 (Price-to-Earnings Ratio)
    
    参数:
        log_transform: 是否对数化处理（默认 True）
    """
    
    def __init__(self, log_transform: bool = True):
        super().__init__(name="PERatio_Log" if log_transform else "PERatio",
                         params={"log_transform": log_transform})
        self.log_transform = log_transform
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算市盈率"""
        if "pe_ratio" not in data.columns:
            raise ValueError("数据中缺少 'pe_ratio' 列")
        
        pe = data["pe_ratio"].replace(0, np.nan)
        
        if self.log_transform:
            return np.log(pe.abs()) * np.sign(pe)
        return pe


class PBRatio(Factor):
    """
    市净率因子 (Price-to-Book Ratio)
    
    参数:
        log_transform: 是否对数化处理（默认 True）
    """
    
    def __init__(self, log_transform: bool = True):
        super().__init__(name="PBRatio_Log" if log_transform else "PBRatio",
                         params={"log_transform": log_transform})
        self.log_transform = log_transform
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算市净率"""
        if "pb_ratio" not in data.columns:
            raise ValueError("数据中缺少 'pb_ratio' 列")
        
        pb = data["pb_ratio"].replace(0, np.nan)
        
        if self.log_transform:
            return np.log(pb.abs()) * np.sign(pb)
        return pb


class PSRatio(Factor):
    """
    市销率因子 (Price-to-Sales Ratio)
    
    参数:
        log_transform: 是否对数化处理（默认 True）
    """
    
    def __init__(self, log_transform: bool = True):
        super().__init__(name="PSRatio_Log" if log_transform else "PSRatio",
                         params={"log_transform": log_transform})
        self.log_transform = log_transform
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算市销率"""
        if "ps_ratio" not in data.columns:
            raise ValueError("数据中缺少 'ps_ratio' 列")
        
        ps = data["ps_ratio"].replace(0, np.nan)
        
        if self.log_transform:
            return np.log(ps.abs()) * np.sign(ps)
        return ps


class PCFRatio(Factor):
    """
    市现率因子 (Price-to-Cash Flow Ratio)
    
    参数:
        log_transform: 是否对数化处理（默认 True）
    """
    
    def __init__(self, log_transform: bool = True):
        super().__init__(name="PCFRatio_Log" if log_transform else "PCFRatio",
                         params={"log_transform": log_transform})
        self.log_transform = log_transform
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算市现率"""
        if "pcf_ratio" not in data.columns:
            raise ValueError("数据中缺少 'pcf_ratio' 列")
        
        pcf = data["pcf_ratio"].replace(0, np.nan)
        
        if self.log_transform:
            return np.log(pcf.abs()) * np.sign(pcf)
        return pcf


class EVToEBITDA(Factor):
    """
    企业价值倍数 (EV/EBITDA)
    
    参数:
        log_transform: 是否对数化处理（默认 True）
    """
    
    def __init__(self, log_transform: bool = True):
        super().__init__(name="EVToEBITDA_Log" if log_transform else "EVToEBITDA",
                         params={"log_transform": log_transform})
        self.log_transform = log_transform
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 EV/EBITDA"""
        if "ev_to_ebitda" not in data.columns:
            raise ValueError("数据中缺少 'ev_to_ebitda' 列")
        
        ev_ebitda = data["ev_to_ebitda"].replace(0, np.nan)
        
        if self.log_transform:
            return np.log(ev_ebitda.abs()) * np.sign(ev_ebitda)
        return ev_ebitda


class DividendYield(Factor):
    """
    股息率因子 (Dividend Yield)
    """
    
    def __init__(self):
        super().__init__(name="DividendYield")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算股息率"""
        if "dividend_yield" not in data.columns:
            raise ValueError("数据中缺少 'dividend_yield' 列")
        
        return data["dividend_yield"]


# ---------- 动量因子 (Momentum) ----------

class MomentumReturn(RollingFactor):
    """
    动量收益因子 (Momentum Return)
    
    计算过去 N 期的累积收益（不含最近一期，避免短期反转效应）
    
    参数:
        window: 计算窗口（默认 12）
        skip_recent: 跳过最近期数（默认 1，避免反转效应）
    """
    
    def __init__(self, window: int = 12, skip_recent: int = 1):
        super().__init__(name=f"Momentum_{window}_{skip_recent}", 
                         window=window, params={"skip_recent": skip_recent})
        self.skip_recent = skip_recent
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算动量因子"""
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        
        close = data["close"]
        
        # 计算累积收益（跳过最近一期）
        momentum = (close.shift(self.skip_recent) / close.shift(self.window + self.skip_recent)) - 1
        
        return momentum


class Reversal(RollingFactor):
    """
    反转因子 (Reversal)
    
    计算短期反转收益（A股反转效应强于动量效应）
    
    参数:
        window: 计算窗口（默认 1，月度反转）
    """
    
    def __init__(self, window: int = 1):
        super().__init__(name=f"Reversal_{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算反转因子"""
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        
        close = data["close"]
        
        # 反转因子 = -短期收益（负收益表示反转）
        reversal = -(close / close.shift(self.window) - 1)
        
        return reversal


class IntermediateMomentum(RollingFactor):
    """
    中期动量因子 (Intermediate Momentum)
    
    A股中期（3-6个月）动量可能有效
    
    参数:
        window: 计算窗口（默认 6）
    """
    
    def __init__(self, window: int = 6):
        super().__init__(name=f"IntermediateMomentum_{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算中期动量"""
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        
        close = data["close"]
        
        # 计算中期累积收益
        momentum = (close / close.shift(self.window)) - 1
        
        return momentum


class RSTR(RollingFactor):
    """
    相对强弱因子 (Relative Strength)
    
    相对于市场基准的超额收益
    
    参数:
        window: 计算窗口（默认 12）
    """
    
    def __init__(self, window: int = 12):
        super().__init__(name=f"RSTR_{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算相对强弱因子"""
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        if "market_return" not in data.columns:
            raise ValueError("数据中缺少 'market_return' 列（市场基准收益）")
        
        close = data["close"]
        market_return = data["market_return"]
        
        # 计算个股累积收益
        stock_return = close / close.shift(self.window) - 1
        
        # 计算市场累积收益
        market_cum_return = (1 + market_return).rolling(window=self.window).apply(np.prod) - 1
        
        # 相对强弱 = 个股收益 - 市场收益
        rstr = stock_return - market_cum_return
        
        return rstr


# ---------- 波动率因子 (Volatility) ----------

class Volatility(RollingFactor):
    """
    波动率因子 (Volatility)
    
    计算收益率的标准差
    
    参数:
        window: 计算窗口（默认 20）
        annualize: 是否年化（默认 True）
    """
    
    def __init__(self, window: int = 20, annualize: bool = True):
        super().__init__(name=f"Volatility_{window}", 
                         window=window, params={"annualize": annualize})
        self.annualize = annualize
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算波动率"""
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        
        returns = data["close"].pct_change().dropna()
        
        # 计算滚动标准差
        vol = returns.rolling(window=self.window).std()
        
        if self.annualize:
            # 年化波动率（假设252个交易日）
            vol = vol * np.sqrt(252)
        
        return vol


class IdiosyncraticVolatility(RollingFactor):
    """
    特质波动率因子 (Idiosyncratic Volatility)
    
    计算个股收益中无法被市场解释的波动部分（"特质波动率之谜"）
    
    参数:
        window: 计算窗口（默认 60）
    """
    
    def __init__(self, window: int = 60):
        super().__init__(name=f"IdiosyncraticVolatility_{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算特质波动率"""
        if "return" not in data.columns:
            raise ValueError("数据中缺少 'return' 列")
        if "market_return" not in data.columns:
            raise ValueError("数据中缺少 'market_return' 列")
        
        returns = data["return"]
        market_returns = data["market_return"]
        
        def calc_idio_vol(window_data):
            if len(window_data) < 2:
                return np.nan
            # 简单市场模型回归
            y = window_data["return"].values
            x = window_data["market_return"].values
            if len(y) < 2 or np.std(x) == 0:
                return np.nan
            beta = np.cov(x, y)[0, 1] / np.var(x)
            alpha = np.mean(y) - beta * np.mean(x)
            # 计算残差标准差
            residuals = y - (alpha + beta * x)
            return np.std(residuals)
        
        # 应用滚动窗口计算
        df = pd.DataFrame({"return": returns, "market_return": market_returns})
        idio_vol = df.rolling(window=self.window).apply(calc_idio_vol)["return"]
        
        return idio_vol


class HighLowVolatilityRatio(RollingFactor):
    """
    高低波动比因子 (High-Low Volatility Ratio)
    
    基于每日最高价和最低价计算的波动率指标
    
    参数:
        window: 计算窗口（默认 20）
    """
    
    def __init__(self, window: int = 20):
        super().__init__(name=f"HighLowVolatilityRatio_{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算高低波动比"""
        if "high" not in data.columns:
            raise ValueError("数据中缺少 'high' 列")
        if "low" not in data.columns:
            raise ValueError("数据中缺少 'low' 列")
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # 计算真实波动范围
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算平均真实波动范围
        atr = tr.rolling(window=self.window).mean()
        
        # 高低波动比 = ATR / 平均收盘价
        avg_close = close.rolling(window=self.window).mean()
        ratio = atr / avg_close
        
        return ratio


class DownsideVolatility(RollingFactor):
    """
    下行波动率因子 (Downside Volatility)
    
    仅计算下跌期间的波动率
    
    参数:
        window: 计算窗口（默认 20）
        annualize: 是否年化（默认 True）
    """
    
    def __init__(self, window: int = 20, annualize: bool = True):
        super().__init__(name=f"DownsideVolatility_{window}", 
                         window=window, params={"annualize": annualize})
        self.annualize = annualize
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算下行波动率"""
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        
        returns = data["close"].pct_change()
        
        # 仅考虑负收益
        downside_returns = returns.where(returns < 0, 0)
        
        # 计算下行标准差
        downside_vol = downside_returns.rolling(window=self.window).std()
        
        if self.annualize:
            downside_vol = downside_vol * np.sqrt(252)
        
        return downside_vol


# ---------- 质量因子 (Quality) ----------

class ROE(Factor):
    """
    净资产收益率 (Return on Equity)
    """
    
    def __init__(self):
        super().__init__(name="ROE")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 ROE"""
        if "roe" not in data.columns:
            raise ValueError("数据中缺少 'roe' 列")
        
        return data["roe"]


class ROA(Factor):
    """
    总资产收益率 (Return on Assets)
    """
    
    def __init__(self):
        super().__init__(name="ROA")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 ROA"""
        if "roa" not in data.columns:
            raise ValueError("数据中缺少 'roa' 列")
        
        return data["roa"]


class ROIC(Factor):
    """
    投资资本回报率 (Return on Invested Capital)
    """
    
    def __init__(self):
        super().__init__(name="ROIC")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 ROIC"""
        if "roic" not in data.columns:
            raise ValueError("数据中缺少 'roic' 列")
        
        return data["roic"]


class OperatingMargin(Factor):
    """
    营业利润率 (Operating Margin)
    """
    
    def __init__(self):
        super().__init__(name="OperatingMargin")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算营业利润率"""
        if "operating_margin" not in data.columns:
            raise ValueError("数据中缺少 'operating_margin' 列")
        
        return data["operating_margin"]


class NetMargin(Factor):
    """
    净利润率 (Net Margin)
    """
    
    def __init__(self):
        super().__init__(name="NetMargin")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算净利润率"""
        if "net_margin" not in data.columns:
            raise ValueError("数据中缺少 'net_margin' 列")
        
        return data["net_margin"]


class GrossProfitMargin(Factor):
    """
    毛利率 (Gross Profit Margin)
    """
    
    def __init__(self):
        super().__init__(name="GrossProfitMargin")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算毛利率"""
        if "gross_profit_margin" not in data.columns:
            raise ValueError("数据中缺少 'gross_profit_margin' 列")
        
        return data["gross_profit_margin"]


# ---------- 杠杆因子 (Leverage) ----------

class DebtToEquity(Factor):
    """
    资产负债率 (Debt-to-Equity Ratio)
    """
    
    def __init__(self):
        super().__init__(name="DebtToEquity")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算资产负债率"""
        if "debt_to_equity" not in data.columns:
            raise ValueError("数据中缺少 'debt_to_equity' 列")
        
        return data["debt_to_equity"]


class InterestCoverage(Factor):
    """
    利息保障倍数 (Interest Coverage Ratio)
    """
    
    def __init__(self):
        super().__init__(name="InterestCoverage")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算利息保障倍数"""
        if "interest_coverage" not in data.columns:
            raise ValueError("数据中缺少 'interest_coverage' 列")
        
        return data["interest_coverage"]


# ---------- 流动性因子 (Liquidity) ----------

class TurnoverRatio(RollingFactor):
    """
    换手率因子 (Turnover Ratio)
    
    参数:
        window: 计算窗口（默认 20）
    """
    
    def __init__(self, window: int = 20):
        super().__init__(name=f"TurnoverRatio_{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算换手率"""
        if "turnover" not in data.columns:
            raise ValueError("数据中缺少 'turnover' 列")
        
        turnover = data["turnover"]
        
        # 计算平均换手率
        avg_turnover = turnover.rolling(window=self.window).mean()
        
        return avg_turnover


class AmihudIlliquidity(RollingFactor):
    """
    Amihud 非流动性因子 (Amihud Illiquidity)
    
    参数:
        window: 计算窗口（默认 20）
    """
    
    def __init__(self, window: int = 20):
        super().__init__(name=f"AmihudIlliquidity_{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 Amihud 非流动性指标"""
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        if "volume" not in data.columns:
            raise ValueError("数据中缺少 'volume' 列")
        if "market_cap" not in data.columns:
            raise ValueError("数据中缺少 'market_cap' 列")
        
        returns = data["close"].pct_change().abs()
        volume = data["volume"]
        market_cap = data["market_cap"]
        
        # Amihud 非流动性 = |收益| / (成交额/市值)
        # 成交额 = 成交量 * 收盘价
        turnover_value = volume * data["close"]
        illiquidity = returns / (turnover_value / market_cap)
        
        # 计算滚动平均值
        avg_illiquidity = illiquidity.rolling(window=self.window).mean()
        
        return avg_illiquidity


# ==================== Rust 风格技术指标扩展 ====================

# ---------- 移动平均类 ----------

class SMMA(RollingFactor):
    """
    平滑移动平均线 (Smoothed Moving Average)
    
    SMMA = (Previous SMMA * (n - 1) + Current Price) / n
    
    参数:
        window: 窗口大小（默认 30）
        field: 计算字段（默认 'close'）
    """
    
    def __init__(self, window: int = 30, field: str = "close"):
        super().__init__(name=f"SMMA{window}", window=window, params={"field": field})
        self.field = field
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算平滑移动平均线"""
        prices = data[self.field]
        
        # 初始化第一个值为 SMA
        smma = pd.Series(index=data.index)
        smma.iloc[:self.window] = prices.iloc[:self.window].rolling(window=self.window).mean()
        
        # 递推计算后续值
        for i in range(self.window, len(data)):
            smma.iloc[i] = (smma.iloc[i-1] * (self.window - 1) + prices.iloc[i]) / self.window
        
        return smma


class VWAP(RollingFactor):
    """
    成交量加权平均价格 (Volume Weighted Average Price)
    
    参数:
        window: 窗口大小（默认 20）
    """
    
    def __init__(self, window: int = 20):
        super().__init__(name=f"VWAP{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 VWAP"""
        if "high" not in data.columns or "low" not in data.columns:
            raise ValueError("数据中缺少 'high' 或 'low' 列")
        if "volume" not in data.columns:
            raise ValueError("数据中缺少 'volume' 列")
        
        high = data["high"]
        low = data["close"]  # 使用 close 替代典型价格中的 close
        close = data["close"]
        volume = data["volume"]
        
        # 典型价格
        tp = (high + low + close) / 3
        
        # VWAP = Sum(TP * Volume) / Sum(Volume)
        vwap = (tp * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        
        return vwap


class EMAVolume(RollingFactor):
    """
    成交量指数移动平均 (EMA Volume)
    
    参数:
        window: 窗口大小（默认 14）
    """
    
    def __init__(self, window: int = 14):
        super().__init__(name=f"EMAVolume{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算成交量 EMA"""
        if "volume" not in data.columns:
            raise ValueError("数据中缺少 'volume' 列")
        
        volume = data["volume"]
        return volume.ewm(span=self.window, adjust=False).mean()


# ---------- 波动类指标 ----------

class DonchianChannel(RollingFactor):
    """
    唐奇安通道 (Donchian Channel)
    
    参数:
        window: 窗口大小（默认 20）
    """
    
    def __init__(self, window: int = 20):
        super().__init__(name=f"DonchianChannel{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算唐奇安通道（返回上轨、中轨、下轨）"""
        if "high" not in data.columns:
            raise ValueError("数据中缺少 'high' 列")
        if "low" not in data.columns:
            raise ValueError("数据中缺少 'low' 列")
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # 上轨 = 最高价的 rolling max
        upper = high.rolling(window=self.window).max()
        
        # 下轨 = 最低价的 rolling min
        lower = low.rolling(window=self.window).min()
        
        # 中轨 = (上轨 + 下轨) / 2
        middle = (upper + lower) / 2
        
        # 突破信号
        long_signal = (close > upper.shift(1)).astype(int)
        short_signal = (close < lower.shift(1)).astype(int)
        
        return pd.DataFrame({
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "long_signal": long_signal,
            "short_signal": short_signal
        })


class KeltnerChannel(RollingFactor):
    """
    肯特纳通道 (Keltner Channel)
    
    参数:
        window: 窗口大小（默认 20）
        atr_multiplier: ATR 乘数（默认 2.0）
    """
    
    def __init__(self, window: int = 20, atr_multiplier: float = 2.0):
        super().__init__(name=f"KeltnerChannel{window}", 
                         window=window, params={"atr_multiplier": atr_multiplier})
        self.atr_multiplier = atr_multiplier
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算肯特纳通道"""
        if "high" not in data.columns or "low" not in data.columns:
            raise ValueError("数据中缺少 'high' 或 'low' 列")
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # 计算典型价格的 EMA
        tp = (high + low + close) / 3
        ema = tp.ewm(span=self.window, adjust=False).mean()
        
        # 计算 ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.window, adjust=False).mean()
        
        # 上下轨
        upper = ema + self.atr_multiplier * atr
        lower = ema - self.atr_multiplier * atr
        
        return pd.DataFrame({
            "upper": upper,
            "middle": ema,
            "lower": lower
        })


class MassIndex(RollingFactor):
    """
    质量指数 (Mass Index)
    
    参数:
        window: EMA 窗口（默认 9）
        signal_window: 信号窗口（默认 25）
    """
    
    def __init__(self, window: int = 9, signal_window: int = 25):
        super().__init__(name=f"MassIndex_{window}_{signal_window}", 
                         window=window, params={"signal_window": signal_window})
        self.signal_window = signal_window
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算质量指数"""
        if "high" not in data.columns or "low" not in data.columns:
            raise ValueError("数据中缺少 'high' 或 'low' 列")
        
        high = data["high"]
        low = data["low"]
        
        # 计算价格范围
        range_hl = high - low
        
        # EMA of range
        ema1 = range_hl.ewm(span=self.window, adjust=False).mean()
        ema2 = ema1.ewm(span=self.window, adjust=False).mean()
        
        # Mass Index = Sum(EMA1 / EMA2) over signal_window
        mass_ratio = ema1 / ema2
        mass_index = mass_ratio.rolling(window=self.signal_window).sum()
        
        return mass_index


# ---------- 趋势类指标 ----------

class ADX(RollingFactor):
    """
    平均方向指数 (Average Directional Index)
    
    参数:
        window: 窗口大小（默认 14）
    """
    
    def __init__(self, window: int = 14):
        super().__init__(name=f"ADX{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算 ADX"""
        if "high" not in data.columns or "low" not in data.columns:
            raise ValueError("数据中缺少 'high' 或 'low' 列")
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # +DM 和 -DM
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = -minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # TR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # +DI 和 -DI
        plus_di = 100 * (plus_dm.rolling(window=self.window).sum() / tr.rolling(window=self.window).sum())
        minus_di = 100 * (minus_dm.rolling(window=self.window).sum() / tr.rolling(window=self.window).sum())
        
        # DX 和 ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(window=self.window).mean()
        
        return adx


class ParabolicSAR(Factor):
    """
    抛物线转向指标 (Parabolic SAR)
    
    参数:
        acceleration: 加速因子（默认 0.02）
        max_acceleration: 最大加速因子（默认 0.2）
    """
    
    def __init__(self, acceleration: float = 0.02, max_acceleration: float = 0.2):
        super().__init__(name="ParabolicSAR", params={
            "acceleration": acceleration,
            "max_acceleration": max_acceleration
        })
        self.acceleration = acceleration
        self.max_acceleration = max_acceleration
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算抛物线转向指标"""
        if "high" not in data.columns or "low" not in data.columns:
            raise ValueError("数据中缺少 'high' 或 'low' 列")
        
        high = data["high"].values
        low = data["low"].values
        
        n = len(data)
        sar = np.zeros(n)
        sar[0] = low[0]
        
        # 初始方向（假设上升趋势）
        trend = 1  # 1 = 上升, -1 = 下降
        af = self.acceleration
        ep = high[0]  # 极值点
        
        for i in range(1, n):
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            
            if trend == 1:
                # 上升趋势
                if low[i] < sar[i]:
                    # 趋势反转
                    trend = -1
                    sar[i] = ep
                    ep = low[i]
                    af = self.acceleration
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + self.acceleration, self.max_acceleration)
            else:
                # 下降趋势
                if high[i] > sar[i]:
                    # 趋势反转
                    trend = 1
                    sar[i] = ep
                    ep = high[i]
                    af = self.acceleration
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + self.acceleration, self.max_acceleration)
        
        return pd.Series(sar, index=data.index)


class VortexIndicator(RollingFactor):
    """
    漩涡指标 (Vortex Indicator)
    
    参数:
        window: 窗口大小（默认 14）
    """
    
    def __init__(self, window: int = 14):
        super().__init__(name=f"VortexIndicator{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算漩涡指标"""
        if "high" not in data.columns or "low" not in data.columns:
            raise ValueError("数据中缺少 'high' 或 'low' 列")
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # 计算 TR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # +VM 和 -VM
        plus_vm = (high - low.shift()).abs()
        minus_vm = (low - high.shift()).abs()
        
        # +VI 和 -VI
        plus_vi = plus_vm.rolling(window=self.window).sum() / tr.rolling(window=self.window).sum()
        minus_vi = minus_vm.rolling(window=self.window).sum() / tr.rolling(window=self.window).sum()
        
        return pd.DataFrame({"+VI": plus_vi, "-VI": minus_vi})


# ---------- 量价类指标 ----------

class ChaikinMoneyFlow(RollingFactor):
    """
    蔡金资金流指标 (Chaikin Money Flow)
    
    参数:
        window: 窗口大小（默认 20）
    """
    
    def __init__(self, window: int = 20):
        super().__init__(name=f"ChaikinMoneyFlow{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算蔡金资金流"""
        if "high" not in data.columns or "low" not in data.columns:
            raise ValueError("数据中缺少 'high' 或 'low' 列")
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        if "volume" not in data.columns:
            raise ValueError("数据中缺少 'volume' 列")
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        
        # Money Flow Multiplier
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        
        # Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # CMF = Sum(MF Volume) / Sum(Volume)
        cmf = mf_volume.rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        
        return cmf


class EaseOfMovement(RollingFactor):
    """
    简易波动指标 (Ease of Movement)
    
    参数:
        window: 窗口大小（默认 14）
    """
    
    def __init__(self, window: int = 14):
        super().__init__(name=f"EaseOfMovement{window}", window=window)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算简易波动指标"""
        if "high" not in data.columns or "low" not in data.columns:
            raise ValueError("数据中缺少 'high' 或 'low' 列")
        if "volume" not in data.columns:
            raise ValueError("数据中缺少 'volume' 列")
        
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        
        # 距离移动
        distance_moved = ((high + low) / 2) - ((high.shift() + low.shift()) / 2)
        
        # 箱体比率
        box_ratio = (volume / 100000000) / (high - low)
        
        # EOM = Distance Moved / Box Ratio
        eom = distance_moved / box_ratio
        
        # 平滑
        eom_ema = eom.ewm(span=self.window, adjust=False).mean()
        
        return eom_ema


class VolumeWeightedMACD(Factor):
    """
    成交量加权 MACD (Volume-Weighted MACD)
    
    参数:
        fast_window: 快速 EMA 窗口（默认 12）
        slow_window: 慢速 EMA 窗口（默认 26）
        signal_window: 信号 EMA 窗口（默认 9）
    """
    
    def __init__(self, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9):
        super().__init__(name=f"VolumeWeightedMACD_{fast_window}_{slow_window}_{signal_window}", params={
            "fast_window": fast_window,
            "slow_window": slow_window,
            "signal_window": signal_window
        })
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算成交量加权 MACD"""
        if "close" not in data.columns:
            raise ValueError("数据中缺少 'close' 列")
        if "volume" not in data.columns:
            raise ValueError("数据中缺少 'volume' 列")
        
        close = data["close"]
        volume = data["volume"]
        
        # 成交量加权价格
        vw_price = (close * volume).cumsum() / volume.cumsum()
        
        # 计算 EMA
        ema_fast = vw_price.ewm(span=self.fast_window, adjust=False).mean()
        ema_slow = vw_price.ewm(span=self.slow_window, adjust=False).mean()
        
        # MACD 线
        macd = ema_fast - ema_slow
        
        # 信号线
        signal = macd.ewm(span=self.signal_window, adjust=False).mean()
        
        # 柱状图
        histogram = macd - signal
        
        return pd.DataFrame({
            "macd": macd,
            "signal": signal,
            "histogram": histogram
        })


# ---------- 成长因子扩展 ----------

class GrowthScore(Factor):
    """
    成长综合评分因子 (Growth Score)
    
    综合考虑营收增长、盈利增长、现金流增长
    """
    
    def __init__(self):
        super().__init__(name="GrowthScore")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算成长综合评分"""
        scores = []
        
        # 营收增长率
        if "revenue_growth" in data.columns:
            scores.append(data["revenue_growth"])
        
        # 净利润增长率
        if "net_profit_growth" in data.columns:
            scores.append(data["net_profit_growth"])
        
        # 经营现金流增长率
        if "cash_flow_growth" in data.columns:
            scores.append(data["cash_flow_growth"])
        
        # EPS 增长率
        if "eps_growth" in data.columns:
            scores.append(data["eps_growth"])
        
        if not scores:
            raise ValueError("数据中缺少成长相关列（revenue_growth, net_profit_growth, cash_flow_growth, eps_growth）")
        
        # 等权综合评分
        composite = sum(scores) / len(scores)
        
        return composite


class RevenueGrowthAcceleration(Factor):
    """
    营收增长加速度 (Revenue Growth Acceleration)
    
    衡量增长速度的变化
    """
    
    def __init__(self):
        super().__init__(name="RevenueGrowthAcceleration")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算营收增长加速度"""
        if "revenue_growth" not in data.columns:
            raise ValueError("数据中缺少 'revenue_growth' 列")
        
        growth = data["revenue_growth"]
        
        # 加速度 = 增长率的变化
        acceleration = growth.diff()
        
        return acceleration


class ProfitGrowthQuality(Factor):
    """
    盈利增长质量因子 (Profit Growth Quality)
    
    衡量盈利增长的可持续性
    """
    
    def __init__(self):
        super().__init__(name="ProfitGrowthQuality")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算盈利增长质量"""
        if "net_profit" not in data.columns:
            raise ValueError("数据中缺少 'net_profit' 列")
        if "operating_cash_flow" not in data.columns:
            raise ValueError("数据中缺少 'operating_cash_flow' 列")
        
        net_profit = data["net_profit"]
        cash_flow = data["operating_cash_flow"]
        
        # 计算盈利增长
        profit_growth = net_profit.pct_change()
        
        # 计算现金流增长
        cash_flow_growth = cash_flow.pct_change()
        
        # 质量评分 = 现金流增长 / 盈利增长（衡量盈利的现金支撑）
        quality = cash_flow_growth / profit_growth.replace(0, np.nan)
        
        return quality


# ---------- 因子工具函数 ----------

def winsorize_factor(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize 去极值
    
    参数:
        series: 输入序列
        lower: 下分位数（默认 1%）
        upper: 上分位数（默认 99%）
    
    返回:
        处理后的序列
    """
    if len(series.dropna()) < 10:
        return series
    
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def standardize_factor(series: pd.Series) -> pd.Series:
    """
    Z-score 标准化
    
    参数:
        series: 输入序列
    
    返回:
        标准化后的序列
    """
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val == 0:
        return pd.Series(0.0, index=series.index)
    
    return (series - mean_val) / std_val


def neutralize_factor(series: pd.Series, market_cap: pd.Series) -> pd.Series:
    """
    市值中性化
    
    通过回归去除市值对因子的影响
    
    参数:
        series: 因子序列
        market_cap: 市值序列
    
    返回:
        中性化后的残差序列
    """
    # 创建数据框，删除缺失值
    df = pd.DataFrame({"factor": series, "market_cap": np.log(market_cap)}).dropna()
    
    if len(df) < 2:
        return pd.Series(np.nan, index=series.index)
    
    # 简单线性回归
    x = df["market_cap"].values
    y = df["factor"].values
    
    # 计算斜率和截距
    n = len(x)
    if n < 2:
        return pd.Series(np.nan, index=series.index)
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    
    if denominator == 0:
        return pd.Series(np.nan, index=series.index)
    
    beta = numerator / denominator
    alpha = y_mean - beta * x_mean
    
    # 计算残差
    residuals = y - (alpha + beta * x)
    
    # 重建结果序列
    result = pd.Series(np.nan, index=series.index)
    result.loc[df.index] = residuals
    
    return result


class FactorComposite(Factor):
    """
    因子合成器 (Factor Composite)
    
    将多个因子合成为一个综合因子
    
    参数:
        factors: 因子类列表
        weights: 权重列表（等权为 None）
        winsorize: 是否去极值（默认 True）
        standardize: 是否标准化（默认 True）
        neutralize: 是否中性化（默认 False）
    """
    
    def __init__(self, factors: list, weights: list = None, 
                 winsorize: bool = True, standardize: bool = True, neutralize: bool = False):
        super().__init__(name="FactorComposite", params={
            "factor_names": [f.name for f in factors],
            "weights": weights,
            "winsorize": winsorize,
            "standardize": standardize,
            "neutralize": neutralize
        })
        self.factors = factors
        self.weights = weights if weights else [1/len(factors)] * len(factors)
        self.winsorize = winsorize
        self.standardize = standardize
        self.neutralize = neutralize
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """合成多个因子"""
        if len(self.factors) != len(self.weights):
            raise ValueError("因子数量与权重数量不匹配")
        
        composite = pd.Series(0.0, index=data.index)
        
        for factor, weight in zip(self.factors, self.weights):
            # 计算因子值
            factor_values = factor.compute(data)
            
            # 去极值
            if self.winsorize:
                factor_values = winsorize_factor(factor_values)
            
            # 标准化
            if self.standardize:
                factor_values = standardize_factor(factor_values)
            
            # 中性化（仅对需要的因子）
            if self.neutralize and "market_cap" in data.columns:
                factor_values = neutralize_factor(factor_values, data["market_cap"])
            
            # 加权求和
            composite += factor_values * weight
        
        return composite


# ---------- 行业因子 ----------

class IndustryFactor(Factor):
    """
    行业因子 (Industry Factor)
    
    将行业分类转换为数值因子
    
    参数:
        industry_column: 行业列名（默认 'industry'）
        method: 编码方法（'onehot' 或 'ordinal'，默认 'ordinal'）
    """
    
    def __init__(self, industry_column: str = "industry", method: str = "ordinal"):
        super().__init__(name=f"IndustryFactor_{method}", params={
            "industry_column": industry_column,
            "method": method
        })
        self.industry_column = industry_column
        self.method = method
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算行业因子"""
        if self.industry_column not in data.columns:
            raise ValueError(f"数据中缺少 '{self.industry_column}' 列")
        
        industry = data[self.industry_column]
        
        if self.method == "ordinal":
            # 序号编码
            industry_codes = industry.astype("category").cat.codes
            return industry_codes
        
        elif self.method == "onehot":
            # 独热编码（返回 DataFrame）
            raise NotImplementedError("独热编码需要返回 DataFrame，请使用 IndustryOneHot 类")
        
        else:
            raise ValueError(f"未知的编码方法: {self.method}")


class IndustryDummy(Factor):
    """
    行业虚拟变量生成器 (Industry Dummy Variables)
    
    参数:
        industry_column: 行业列名（默认 'industry'）
    """
    
    def __init__(self, industry_column: str = "industry"):
        super().__init__(name="IndustryDummy", params={"industry_column": industry_column})
        self.industry_column = industry_column
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成行业虚拟变量"""
        if self.industry_column not in data.columns:
            raise ValueError(f"数据中缺少 '{self.industry_column}' 列")
        
        # 生成独热编码
        dummies = pd.get_dummies(data[self.industry_column], prefix="industry")
        
        return dummies
