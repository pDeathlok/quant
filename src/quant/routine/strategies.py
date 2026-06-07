from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EntryConfig:
    """Fixed-threshold B1 entry rule.

    B1 intentionally does not buy the daily top N candidates. A strategy only
    triggers when model scores pass explicit thresholds, allowing long empty
    periods when there is no strong opportunity.
    """

    min_up8: float
    max_down3: float | None = None


@dataclass(frozen=True)
class ExitConfig:
    """B1 production exit rule.

    Confirmed B1 variants use trailing exits plus fixed stop loss:
    buy at T+1 open, activate trailing after a target gain, sell on drawdown
    from peak, or exit at the configured maximum holding day.
    """

    kind: str
    hold_days: int
    take_profit: float | None = None
    stop_loss: float | None = None
    trail_drawdown: float | None = None

    @property
    def rule_name(self) -> str:
        target = f"target{self.take_profit:.0%}_" if self.take_profit is not None else ""
        trail = f"dd{self.trail_drawdown:.0%}_" if self.trail_drawdown is not None else ""
        stop = f"sl{self.stop_loss:.1%}".replace(".0%", "%") if self.stop_loss is not None else "nostop"
        return f"{self.kind}_{target}{trail}{stop}_T{self.hold_days + 1}"


@dataclass(frozen=True)
class StrategyConfig:
    """Long-lived strategy config used by routine runs and the frontend.

    The historical experiment scripts and obsolete models may be cleaned over
    time. The strategy rationale is preserved in
    docs/strategies/b1_selected_strategy_record.md.
    """

    id: str
    name: str
    enabled: bool
    entry: EntryConfig
    exit: ExitConfig
    description: str


def _strategy_from_dict(raw: dict[str, Any]) -> StrategyConfig:
    entry_raw = raw["entry"]
    exit_raw = raw["exit"]
    return StrategyConfig(
        id=str(raw["id"]),
        name=str(raw["name"]),
        enabled=bool(raw.get("enabled", True)),
        entry=EntryConfig(
            min_up8=float(entry_raw["min_up8"]),
            max_down3=float(entry_raw["max_down3"]) if "max_down3" in entry_raw else None,
        ),
        exit=ExitConfig(
            kind=str(exit_raw["kind"]),
            hold_days=int(exit_raw["hold_days"]),
            take_profit=float(exit_raw["take_profit"]) if "take_profit" in exit_raw else None,
            stop_loss=float(exit_raw["stop_loss"]) if "stop_loss" in exit_raw else None,
            trail_drawdown=float(exit_raw["trail_drawdown"]) if "trail_drawdown" in exit_raw else None,
        ),
        description=str(raw.get("description", "")),
    )


def load_strategy_configs(path: Path, include_disabled: bool = False) -> list[StrategyConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    strategies = [_strategy_from_dict(item) for item in payload.get("strategies", [])]
    if include_disabled:
        return strategies
    return [strategy for strategy in strategies if strategy.enabled]
