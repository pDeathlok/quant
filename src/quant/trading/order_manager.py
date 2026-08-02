from typing import Dict, List, Optional, Callable
from .broker import Broker, Order, OrderSide
from quant.risk.manager import RiskManager
from quant.backtest.tradability import AShareTradabilityPolicy


class OrderManager:
    def __init__(
        self,
        broker: Broker,
        risk_manager: RiskManager = None,
        tradability_policy: AShareTradabilityPolicy | None = None,
    ):
        self.broker = broker
        self.risk_manager = risk_manager
        self.tradability_policy = tradability_policy
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self._order_callbacks: List[Callable] = []
        self.last_rejection_reason: str | None = None

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Optional[float] = None,
        *,
        trade_date: str | None = None,
        average_daily_volume: float | None = None,
    ) -> Optional[str]:
        self.last_rejection_reason = None
        account = self.broker.get_account()

        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price
        )

        if self.tradability_policy is not None:
            if trade_date is None or price is None:
                self.last_rejection_reason = "trade_date and price are required for tradability checks"
                return None
            decision = self.tradability_policy.check_order(
                trade_date=trade_date,
                symbol=symbol,
                side=side.value,
                price=price,
            )
            if not decision.allowed:
                self.last_rejection_reason = decision.reason
                return None

        if self.risk_manager:
            positions = self.broker.get_positions()
            order_dict = {
                "symbol": symbol,
                "side": side.value,
                "quantity": quantity,
                "price": price or 0
            }
            passed, msg = self.risk_manager.pre_order_check(
                order_dict,
                account["total_value"],
                positions,
                total_exposure=account.get("positions_value"),
                average_daily_volume=average_daily_volume,
            )
            if not passed:
                self.last_rejection_reason = msg
                return None

        order_id = self.broker.send_order(order)
        if order_id:
            self.pending_orders[order_id] = order

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        success = self.broker.cancel_order(order_id)
        if success and order_id in self.pending_orders:
            del self.pending_orders[order_id]
        return success

    def on_fill(self, fill):
        if fill.order_id in self.pending_orders:
            self.filled_orders[fill.order_id] = self.pending_orders[fill.order_id]
            del self.pending_orders[fill.order_id]

        for callback in self._order_callbacks:
            callback(fill)

    def add_order_callback(self, callback: Callable):
        self._order_callbacks.append(callback)

    def get_pending_orders(self) -> List[Order]:
        return list(self.pending_orders.values())

    def get_positions(self) -> Dict[str, int]:
        return self.broker.get_positions()

    def get_account(self) -> Dict:
        return self.broker.get_account()
