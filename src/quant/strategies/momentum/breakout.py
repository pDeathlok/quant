from ..base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    def __init__(
        self,
        lookback_period: int = 20,
        volume_multiplier: float = 1.5,
        holding_bars: int = 5
    ):
        super().__init__("BreakoutStrategy")
        self.lookback_period = lookback_period
        self.volume_multiplier = volume_multiplier
        self.holding_bars = holding_bars

    def should_entry(self, bar, current_pos: int) -> bool:
        if current_pos > 0:
            return False

        history = self.get_history(bar.symbol, self.lookback_period + 1)
        if len(history) < self.lookback_period + 1:
            return False

        recent = history.iloc[-self.lookback_period:]

        highest_high = recent["high"].max()
        avg_volume = recent["volume"].mean()

        price_breakout = bar.close > highest_high
        volume_confirm = bar.volume > avg_volume * self.volume_multiplier

        return price_breakout and volume_confirm

    def should_exit(self, bar, current_pos: int) -> bool:
        if current_pos == 0:
            return False

        bars_held = self.bars_since_entry.get(bar.symbol, 0)
        return bars_held >= self.holding_bars
