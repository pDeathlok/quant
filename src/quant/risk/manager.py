from dataclasses import dataclass
from typing import Dict


@dataclass
class RiskLimits:
    max_position_size: float = 0.1
    max_total_leverage: float = 1.0
    max_loss_per_day: float = 0.02
    max_drawdown: float = 0.1
    max_orders_per_minute: int = 30
    max_position_pct_per_symbol: float = 0.2
    max_volume_participation: float = 0.1

    def __post_init__(self):
        for name in (
            "max_position_size",
            "max_total_leverage",
            "max_loss_per_day",
            "max_drawdown",
            "max_position_pct_per_symbol",
            "max_volume_participation",
        ):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.max_orders_per_minute <= 0:
            raise ValueError("max_orders_per_minute must be positive")


class RiskManager:
    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.daily_pnl: float = 0.0
        self.daily_high: float = 0.0
        self.peak_value: float = 0.0
        self.order_timestamps: list = []

    def pre_order_check(
        self,
        order: Dict,
        account_value: float,
        positions: Dict[str, int],
        *,
        total_exposure: float | None = None,
        average_daily_volume: float | None = None,
    ) -> tuple[bool, str]:
        symbol = order.get("symbol")
        quantity = order.get("quantity", 0)
        price = order.get("price", 0)
        side = order.get("side", "")

        if account_value <= 0:
            return False, "Account value must be positive"
        if not symbol or side.lower() not in {"buy", "sell"}:
            return False, "Invalid symbol or side"
        if quantity <= 0 or price <= 0:
            return False, "Quantity and price must be positive"
        order_value = quantity * price

        if self.check_daily_loss(account_value):
            return False, "Daily loss limit exceeded"
        if self.check_drawdown(account_value):
            return False, "Drawdown limit exceeded"

        if side.lower() == "buy":
            if order_value / account_value > self.limits.max_position_size:
                return False, f"Order value {order_value} exceeds max position size {self.limits.max_position_size}"

            current_pos_value = positions.get(symbol, 0) * price
            new_pos_value = current_pos_value + order_value
            if new_pos_value / account_value > self.limits.max_position_pct_per_symbol:
                return False, f"Position in {symbol} would exceed max per-symbol limit"

            if total_exposure is not None:
                leverage = (float(total_exposure) + order_value) / account_value
                if leverage > self.limits.max_total_leverage:
                    return False, f"Total leverage {leverage:.2%} exceeds limit"

        if average_daily_volume is not None:
            if average_daily_volume <= 0:
                return False, "Average daily volume must be positive"
            participation = quantity / average_daily_volume
            if participation > self.limits.max_volume_participation:
                return False, f"Order volume participation {participation:.2%} exceeds limit"

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
