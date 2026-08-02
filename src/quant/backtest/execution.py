"""Project-owned execution policies for market-specific backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class AShareExecutionConfig:
    """A-share transaction costs and trading constraints.

    The project owns and validates this policy. A backtest adapter may translate
    it into engine-specific parameters, but the external engine is not the
    source of truth for the policy.
    """

    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    min_commission: float = 5.0
    slippage: float = 0.0
    volume_limit_pct: float = 0.10
    lot_size: int = 100
    t_plus_one: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "commission_rate",
            "stamp_tax_rate",
            "transfer_fee_rate",
            "min_commission",
            "slippage",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")

        if not 0 < self.volume_limit_pct <= 1:
            raise ValueError("volume_limit_pct must be in (0, 1]")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Return engine-neutral policy values."""

        return asdict(self)

    def to_metadata(self) -> Dict[str, Any]:
        """Return an auditable representation stored with backtest artifacts."""

        return {
            "market": "a_share",
            "policy_version": "2026-07-31",
            **asdict(self),
        }
