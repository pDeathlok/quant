from ..base import BaseStrategy
from typing import Dict


class MomentumStrategy(BaseStrategy):
    def __init__(
        self,
        fast_ma_period: int = 5,
        slow_ma_period: int = 20,
        holding_bars: int = 5
    ):
        super().__init__("MomentumStrategy")
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.holding_bars = holding_bars

    def should_entry(self, bar, current_pos: int) -> bool:
        if current_pos > 0:
            return False

        history = self.get_history(bar.symbol, self.slow_ma_period + 1)
        if len(history) < self.slow_ma_period + 1:
            return False

        fast_ma = history["close"].iloc[-self.fast_ma_period:].mean()
        slow_ma = history["close"].iloc[-self.slow_ma_period:].mean()

        prev_fast_ma = history["close"].iloc[-self.fast_ma_period-1:-1].mean()
        prev_slow_ma = history["close"].iloc[-self.slow_ma_period-1:-1].mean()

        golden_cross = prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma
        return golden_cross

    def should_exit(self, bar, current_pos: int) -> bool:
        if current_pos == 0:
            return False

        bars_held = self.bars_since_entry.get(bar.symbol, 0)
        if bars_held >= self.holding_bars:
            return True

        history = self.get_history(bar.symbol, self.slow_ma_period + 1)
        if len(history) < self.slow_ma_period + 1:
            return False

        fast_ma = history["close"].iloc[-self.fast_ma_period:].mean()
        slow_ma = history["close"].iloc[-self.slow_ma_period:].mean()

        prev_fast_ma = history["close"].iloc[-self.fast_ma_period-1:-1].mean()
        prev_slow_ma = history["close"].iloc[-self.slow_ma_period-1:-1].mean()

        death_cross = prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma
        return death_cross
