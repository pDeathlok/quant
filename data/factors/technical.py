import pandas as pd
import numpy as np
from .base import TechnicalFactor


class MA(TechnicalFactor):
    def __init__(self, period: int = 5):
        super().__init__(f"MA_{period}")
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].rolling(window=self.period).mean()


class MACD(TechnicalFactor):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f"MACD_{fast}_{slow}_{signal}")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ema_fast = data["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = data["close"].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        return macd_line - signal_line


class RSI(TechnicalFactor):
    def __init__(self, period: int = 14):
        super().__init__(f"RSI_{period}")
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class BollingerBands(TechnicalFactor):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(f"BB_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        sma = data["close"].rolling(window=self.period).mean()
        std = data["close"].rolling(window=self.period).std()
        upper = sma + (std * self.std_dev)
        lower = sma - (std * self.std_dev)
        return (data["close"] - lower) / (upper - lower)


class VolumeRatio(TechnicalFactor):
    def __init__(self, period: int = 5):
        super().__init__(f"VOL_RATIO_{period}")
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        vol_ma = data["volume"].rolling(window=self.period).mean()
        return data["volume"] / vol_ma


class ATR(TechnicalFactor):
    def __init__(self, period: int = 14):
        super().__init__(f"ATR_{period}")
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.period).mean()
        return atr
