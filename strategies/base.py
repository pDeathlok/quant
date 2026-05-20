from akquant import Strategy, OrderStatus
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import pandas as pd


class BaseStrategy(Strategy, ABC):
    def __init__(self, name: str = None):
        self._name = name or self.__class__.__name__
        self.positions: Dict[str, int] = {}
        self.entry_prices: Dict[str, float] = {}
        self.trade_log: List[Dict] = []
        self.bars_since_entry: Dict[str, int] = {}
        self._factor_cache: Dict[str, pd.DataFrame] = {}

    @property
    def name(self) -> str:
        return self._name

    def on_bar(self, bar):
        current_pos = self.get_position(bar.symbol)

        if current_pos == 0:
            if self.should_entry(bar, current_pos):
                self.do_entry(bar)
        else:
            bars_held = self.bars_since_entry.get(bar.symbol, 0) + 1
            self.bars_since_entry[bar.symbol] = bars_held

            if self.should_exit(bar, current_pos):
                self.do_exit(bar)

    def on_order(self, order):
        if order.status == OrderStatus.Filled:
            self._log_trade(order)

    @abstractmethod
    def should_entry(self, bar, current_pos: int) -> bool:
        pass

    @abstractmethod
    def should_exit(self, bar, current_pos: int) -> bool:
        pass

    def do_entry(self, bar, quantity: int = 100):
        self.buy(symbol=bar.symbol, quantity=quantity)
        self.entry_prices[bar.symbol] = bar.close

    def do_exit(self, bar):
        self.close_position(symbol=bar.symbol)
        if bar.symbol in self.entry_prices:
            del self.entry_prices[bar.symbol]
        if bar.symbol in self.bars_since_entry:
            del self.bars_since_entry[bar.symbol]

    def _log_trade(self, order):
        self.trade_log.append({
            "time": order.timestamp,
            "symbol": order.symbol,
            "side": str(order.side),
            "quantity": order.quantity,
            "price": order.price
        })

    def get_trade_log(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)
