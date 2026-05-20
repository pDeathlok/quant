from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    price: Optional[float] = None
    order_type: OrderType = OrderType.LIMIT
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.order_id is None:
            import uuid
            self.order_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float = 0.0
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Broker(ABC):
    @abstractmethod
    def send_order(self, order: Order) -> str:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def get_account(self) -> Dict:
        pass

    @abstractmethod
    def get_pending_orders(self) -> List[Order]:
        pass


class SimulatedBroker(Broker):
    def __init__(
        self,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.0003
    ):
        self._cash = initial_cash
        self._positions: Dict[str, int] = {}
        self._pending_orders: Dict[str, Order] = {}
        self._filled_orders: Dict[str, Fill] = {}
        self._commission_rate = commission_rate

    def send_order(self, order: Order) -> str:
        self._pending_orders[order.order_id] = order
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            return True
        return False

    def get_positions(self) -> Dict[str, int]:
        return self._positions.copy()

    def get_account(self) -> Dict:
        total_value = self._cash
        for symbol, qty in self._positions.items():
            total_value += qty * 100
        return {
            "cash": self._cash,
            "total_value": total_value,
            "positions_value": total_value - self._cash
        }

    def get_pending_orders(self) -> List[Order]:
        return list(self._pending_orders.values())

    def simulate_fill(self, order_id: str, fill_price: float) -> Fill:
        if order_id not in self._pending_orders:
            raise ValueError(f"Order {order_id} not found")

        order = self._pending_orders[order_id]
        del self._pending_orders[order_id]

        commission = fill_price * order.quantity * self._commission_rate

        if order.side == OrderSide.BUY:
            cost = fill_price * order.quantity + commission
            if cost > self._cash:
                raise ValueError("Insufficient cash")
            self._cash -= cost
            self._positions[order.symbol] = self._positions.get(order.symbol, 0) + order.quantity
        else:
            if self._positions.get(order.symbol, 0) < order.quantity:
                raise ValueError("Insufficient position")
            revenue = fill_price * order.quantity - commission
            self._cash += revenue
            self._positions[order.symbol] -= order.quantity

        fill = Fill(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission
        )

        self._filled_orders[order_id] = fill
        return fill
