from abc import ABC, abstractmethod
from typing import Dict


class RiskLimit(ABC):
    @abstractmethod
    def check(self, **kwargs) -> tuple[bool, str]:
        pass


class PositionLimit(RiskLimit):
    def __init__(self, max_position_pct: float = 0.1):
        self.max_position_pct = max_position_pct

    def check(self, order_value: float, account_value: float, **kwargs) -> tuple[bool, str]:
        if self.max_position_pct > 0:
            if order_value / account_value > self.max_position_pct:
                return False, f"Position size {order_value/account_value:.2%} exceeds limit {self.max_position_pct:.2%}"
        return True, ""


class ExposureLimit(RiskLimit):
    def __init__(self, max_exposure: float = 1.0):
        self.max_exposure = max_exposure

    def check(self, total_exposure: float, account_value: float, **kwargs) -> tuple[bool, str]:
        if self.max_exposure > 0:
            exposure_ratio = total_exposure / account_value if account_value > 0 else 0
            if exposure_ratio > self.max_exposure:
                return False, f"Total exposure {exposure_ratio:.2%} exceeds limit {self.max_exposure:.2%}"
        return True, ""


class DrawdownLimit(RiskLimit):
    def __init__(self, max_drawdown: float = 0.1):
        self.max_drawdown = max_drawdown
        self.peak_value = 0.0

    def check(self, current_value: float, **kwargs) -> tuple[bool, str]:
        if self.peak_value == 0:
            self.peak_value = current_value

        if current_value > self.peak_value:
            self.peak_value = current_value

        drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0

        if drawdown > self.max_drawdown:
            return False, f"Drawdown {drawdown:.2%} exceeds limit {self.max_drawdown:.2%}"

        return True, ""

    def reset(self):
        self.peak_value = 0.0
