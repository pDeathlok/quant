from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Optional
from dataclasses import dataclass
from dataclasses import asdict, replace
from enum import Enum
from datetime import datetime
from pathlib import Path

from quant.backtest.execution import AShareExecutionConfig
from quant.backtest.tradability import AShareTradabilityPolicy
from quant.data.atomic_io import atomic_write_json


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
    stamp_tax: float = 0.0
    transfer_fee: float = 0.0
    slippage_cost: float = 0.0
    total_cost: float = 0.0
    trade_date: Optional[str] = None
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
        commission_rate: float | None = None,
        *,
        execution_config: AShareExecutionConfig | None = None,
        tradability_policy: AShareTradabilityPolicy | None = None,
        state_path: Path | str | None = None,
    ):
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        config = execution_config or AShareExecutionConfig()
        if commission_rate is not None:
            if commission_rate < 0:
                raise ValueError("commission_rate must be non-negative")
            config = replace(config, commission_rate=commission_rate)
        self._cash = initial_cash
        self._positions: Dict[str, int] = {}
        self._pending_orders: Dict[str, Order] = {}
        self._filled_orders: Dict[str, Fill] = {}
        self.execution_config = config
        self.tradability_policy = tradability_policy
        self.state_path = Path(state_path) if state_path is not None else None
        self._market_prices: Dict[str, float] = {}
        self._current_trade_date: str | None = None
        self._today_buys: Dict[str, int] = {}
        if self.state_path is not None and self.state_path.is_file():
            self._load_state()

    def send_order(self, order: Order) -> str:
        if order.quantity <= 0:
            raise ValueError("order quantity must be positive")
        self._pending_orders[order.order_id] = order
        self._persist_state()
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            self._persist_state()
            return True
        return False

    def get_positions(self) -> Dict[str, int]:
        return self._positions.copy()

    def get_account(self) -> Dict:
        total_value = self._cash
        for symbol, qty in self._positions.items():
            price = self._market_prices.get(symbol)
            if price is None:
                raise ValueError(f"missing market price for open position {symbol}")
            total_value += qty * price
        return {
            "cash": self._cash,
            "total_value": total_value,
            "positions_value": total_value - self._cash
        }

    def get_pending_orders(self) -> List[Order]:
        return list(self._pending_orders.values())

    def update_market_prices(self, prices: Mapping[str, float]) -> None:
        for symbol, value in prices.items():
            price = float(value)
            if price <= 0:
                raise ValueError(f"invalid market price for {symbol}")
            self._market_prices[str(symbol)] = price
        self._persist_state()

    def start_trading_day(self, trade_date: str) -> None:
        normalized = str(trade_date).replace("-", "")
        if len(normalized) != 8 or not normalized.isdigit():
            raise ValueError("trade_date must be YYYYMMDD")
        if self._current_trade_date != normalized:
            self._current_trade_date = normalized
            self._today_buys = {}
            self._persist_state()

    def simulate_fill(
        self,
        order_id: str,
        fill_price: float,
        *,
        trade_date: str | None = None,
        market_volume: float | None = None,
    ) -> Fill:
        if order_id not in self._pending_orders:
            raise ValueError(f"Order {order_id} not found")
        order = self._pending_orders[order_id]
        if fill_price <= 0:
            raise ValueError("fill_price must be positive")
        effective_date = str(trade_date or self._current_trade_date or "").replace("-", "")
        if not effective_date:
            raise ValueError("trade_date is required")
        self.start_trading_day(effective_date)
        if order.quantity % self.execution_config.lot_size != 0:
            full_liquidation = (
                order.side == OrderSide.SELL
                and order.quantity == self._positions.get(order.symbol, 0)
            )
            if not full_liquidation:
                raise ValueError(
                    f"order quantity must respect lot size {self.execution_config.lot_size}"
                )
        if market_volume is not None:
            if market_volume <= 0:
                raise ValueError("market_volume must be positive")
            if order.quantity / float(market_volume) > self.execution_config.volume_limit_pct:
                raise ValueError("order exceeds volume participation limit")

        price_multiplier = 1 + self.execution_config.slippage * (
            1 if order.side == OrderSide.BUY else -1
        )
        executed_price = float(fill_price) * price_multiplier
        slippage_cost = abs(executed_price - float(fill_price)) * order.quantity
        if self.tradability_policy is not None:
            decision = self.tradability_policy.check_order(
                trade_date=effective_date,
                symbol=order.symbol,
                side=order.side.value,
                price=executed_price,
            )
            if not decision.allowed:
                raise ValueError(decision.reason or "order is not tradable")

        notional = executed_price * order.quantity
        commission = max(
            notional * self.execution_config.commission_rate,
            self.execution_config.min_commission,
        )
        transfer_fee = notional * self.execution_config.transfer_fee_rate
        stamp_tax = (
            notional * self.execution_config.stamp_tax_rate
            if order.side == OrderSide.SELL
            else 0.0
        )
        total_cost = commission + transfer_fee + stamp_tax + slippage_cost

        if order.side == OrderSide.BUY:
            cost = notional + commission + transfer_fee
            if cost > self._cash:
                raise ValueError("Insufficient cash")
            self._cash -= cost
            self._positions[order.symbol] = self._positions.get(order.symbol, 0) + order.quantity
            self._today_buys[order.symbol] = self._today_buys.get(order.symbol, 0) + order.quantity
        else:
            position = self._positions.get(order.symbol, 0)
            available = position
            if self.execution_config.t_plus_one:
                available -= self._today_buys.get(order.symbol, 0)
            if available < order.quantity:
                suffix = " under T+1" if self.execution_config.t_plus_one else ""
                raise ValueError(f"Insufficient position{suffix}")
            if position < order.quantity:
                raise ValueError("Insufficient position")
            revenue = notional - commission - transfer_fee - stamp_tax
            self._cash += revenue
            self._positions[order.symbol] -= order.quantity
            if self._positions[order.symbol] == 0:
                del self._positions[order.symbol]

        fill = Fill(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=executed_price,
            commission=commission,
            stamp_tax=stamp_tax,
            transfer_fee=transfer_fee,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            trade_date=effective_date,
        )

        del self._pending_orders[order_id]
        self._filled_orders[order_id] = fill
        self._market_prices[order.symbol] = executed_price
        self._persist_state()
        return fill

    @staticmethod
    def _serialize_order(order: Order) -> dict:
        payload = asdict(order)
        payload["side"] = order.side.value
        payload["order_type"] = order.order_type.value
        payload["timestamp"] = order.timestamp.isoformat() if order.timestamp else None
        return payload

    @staticmethod
    def _serialize_fill(fill: Fill) -> dict:
        payload = asdict(fill)
        payload["side"] = fill.side.value
        payload["timestamp"] = fill.timestamp.isoformat() if fill.timestamp else None
        return payload

    def _persist_state(self) -> None:
        if self.state_path is None:
            return
        atomic_write_json(
            {
                "schema_version": "simulated-broker/v1",
                "cash": self._cash,
                "positions": self._positions,
                "pending_orders": [self._serialize_order(item) for item in self._pending_orders.values()],
                "filled_orders": [self._serialize_fill(item) for item in self._filled_orders.values()],
                "market_prices": self._market_prices,
                "current_trade_date": self._current_trade_date,
                "today_buys": self._today_buys,
                "execution_config": self.execution_config.to_dict(),
            },
            self.state_path,
        )

    def _load_state(self) -> None:
        import json

        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "simulated-broker/v1":
            raise ValueError("unsupported simulated broker state schema")
        stored_config = payload.get("execution_config")
        if isinstance(stored_config, dict):
            self.execution_config = AShareExecutionConfig(**stored_config)
        self._cash = float(payload["cash"])
        self._positions = {str(key): int(value) for key, value in payload.get("positions", {}).items()}
        self._market_prices = {
            str(key): float(value) for key, value in payload.get("market_prices", {}).items()
        }
        self._current_trade_date = payload.get("current_trade_date")
        self._today_buys = {
            str(key): int(value) for key, value in payload.get("today_buys", {}).items()
        }
        self._pending_orders = {}
        for item in payload.get("pending_orders", []):
            order = Order(
                symbol=item["symbol"],
                side=OrderSide(item["side"]),
                quantity=int(item["quantity"]),
                price=item.get("price"),
                order_type=OrderType(item["order_type"]),
                stop_price=item.get("stop_price"),
                order_id=item["order_id"],
                timestamp=datetime.fromisoformat(item["timestamp"]) if item.get("timestamp") else None,
            )
            self._pending_orders[order.order_id] = order
        self._filled_orders = {}
        for item in payload.get("filled_orders", []):
            fill = Fill(
                **{
                    **item,
                    "side": OrderSide(item["side"]),
                    "timestamp": datetime.fromisoformat(item["timestamp"])
                    if item.get("timestamp")
                    else None,
                }
            )
            self._filled_orders[fill.order_id] = fill
