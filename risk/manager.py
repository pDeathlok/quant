from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RiskLimits:
    max_position_size: float = 0.1
    max_total_leverage: float = 1.0
    max_loss_per_day: float = 0.02
    max_drawdown: float = 0.1
    max_orders_per_minute: int = 30
    max_position_pct_per_symbol: float = 0.2


class RiskManager:
    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.daily_pnl: float = 0.0
        self.daily_high: float = 0.0
        self.peak_value: float = 0.0
        self.order_timestamps: list = []

    def pre_order_check(self, order: Dict, account_value: float, positions: Dict[str, int]) -> tuple[bool, str]:
        symbol = order.get("symbol")
        quantity = order.get("quantity", 0)
        price = order.get("price", 0)
        side = order.get("side", "")

        order_value = quantity * price

        if side.lower() == "buy":
            if order_value / account_value > self.limits.max_position_size:
                return False, f"Order value {order_value} exceeds max position size {self.limits.max_position_size}"

            current_pos_value = positions.get(symbol, 0) * price
            new_pos_value = current_pos_value + order_value
            if new_pos_value / account_value > self.limits.max_position_pct_per_symbol:
                return False, f"Position in {symbol} would exceed max per-symbol limit"

        if not self._check_order_rate():
            return False, "Order rate limit exceeded"

        return True, ""

    def pre_fill_check(self, fill_price: float, position: int, account_value: float) -> bool:
        return True

    def update_daily_pnl(self, pnl: float):
        self.daily_pnl += pnl

    def check_daily_loss(self, account_value: float) -> bool:
        if self.daily_high == 0:
            self.daily_high = account_value

        if account_value > self.daily_high:
            self.daily_high = account_value

        daily_loss = (self.daily_high - account_value) / self.daily_high
        return daily_loss > self.limits.max_loss_per_day

    def check_drawdown(self, current_value: float) -> bool:
        if self.peak_value == 0:
            self.peak_value = current_value

        if current_value > self.peak_value:
            self.peak_value = current_value

        drawdown = (self.peak_value - current_value) / self.peak_value
        return drawdown > self.limits.max_drawdown

    def _check_order_rate(self) -> bool:
        from datetime import datetime, timedelta

        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        self.order_timestamps = [ts for ts in self.order_timestamps if ts > cutoff]
        self.order_timestamps.append(now)

        return len(self.order_timestamps) <= self.limits.max_orders_per_minute

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_high = 0.0
        self.order_timestamps.clear()
