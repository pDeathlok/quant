from ..base import BaseStrategy
import numpy as np


class MeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        lookback_period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        holding_bars: int = 10
    ):
        super().__init__("MeanReversionStrategy")
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.holding_bars = holding_bars

    def should_entry(self, bar, current_pos: int) -> bool:
        if current_pos > 0:
            return False

        history = self.get_history(bar.symbol, self.lookback_period + 1)
        if len(history) < self.lookback_period + 1:
            return False

        recent_prices = history["close"].iloc[-self.lookback_period:]
        sma = recent_prices.mean()
        std = recent_prices.std()

        if std == 0:
            return False

        z_score = (bar.close - sma) / std
        oversold = z_score < -self.entry_threshold

        return oversold

    def should_exit(self, bar, current_pos: int) -> bool:
        if current_pos == 0:
            return False

        bars_held = self.bars_since_entry.get(bar.symbol, 0)
        if bars_held >= self.holding_bars:
            return True

        history = self.get_history(bar.symbol, self.lookback_period + 1)
        if len(history) < self.lookback_period + 1:
            return False

        recent_prices = history["close"].iloc[-self.lookback_period:]
        sma = recent_prices.mean()
        std = recent_prices.std()

        if std == 0:
            return False

        z_score = (bar.close - sma) / std
        mean_reverted = abs(z_score) < self.exit_threshold

        return mean_reverted
