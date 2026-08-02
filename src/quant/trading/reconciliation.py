"""End-of-day account reconciliation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ReconciliationReport:
    balanced: bool
    cash_difference: float
    position_differences: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cash_difference", float(self.cash_difference))
        object.__setattr__(
            self,
            "position_differences",
            MappingProxyType(dict(self.position_differences)),
        )


def reconcile_account(
    *,
    expected_positions: Mapping[str, int],
    actual_positions: Mapping[str, int],
    expected_cash: float,
    actual_cash: float,
    cash_tolerance: float = 0.01,
) -> ReconciliationReport:
    if cash_tolerance < 0:
        raise ValueError("cash_tolerance must be non-negative")
    expected = {str(symbol): int(quantity) for symbol, quantity in expected_positions.items()}
    actual = {str(symbol): int(quantity) for symbol, quantity in actual_positions.items()}
    differences = {
        symbol: actual.get(symbol, 0) - expected.get(symbol, 0)
        for symbol in sorted(set(expected) | set(actual))
        if actual.get(symbol, 0) != expected.get(symbol, 0)
    }
    cash_difference = float(actual_cash) - float(expected_cash)
    return ReconciliationReport(
        balanced=not differences and abs(cash_difference) <= cash_tolerance,
        cash_difference=cash_difference,
        position_differences=differences,
    )
