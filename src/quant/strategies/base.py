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
        self._last_entry_date: Dict[str, str] = {}  # 防同一天重复买入

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

            # 跳过入场成交后的第一个 bar，避免同一根 bar 上入场和出场冲突
            if bars_held > 1 and self.should_exit(bar, current_pos):
                self.do_exit(bar)

            # 有持仓时也可再次买入（允许加仓/加仓信号），同一日不重复买入
            last_date = self._last_entry_date.get(bar.symbol, "")
            if last_date != bar.timestamp_str and self.should_entry(bar, current_pos):
                self.do_entry(bar)

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
        self._last_entry_date[bar.symbol] = bar.timestamp_str

    def do_exit(self, bar):
        self.close_position(symbol=bar.symbol)
        if bar.symbol in self.entry_prices:
            del self.entry_prices[bar.symbol]
        if bar.symbol in self.bars_since_entry:
            del self.bars_since_entry[bar.symbol]

    def _log_trade(self, order):
        self.trade_log.append({
            "time": order.created_at_str,
            "symbol": order.symbol,
            "side": str(order.side),
            "quantity": order.quantity,
            "price": order.price
        })

    def get_trade_log(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)
