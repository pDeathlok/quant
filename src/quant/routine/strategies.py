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

    min_up5: float | None = None
    min_up8: float | None = None
    min_up10: float | None = None
    max_down2: float | None = None
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
    priority: int
    backtest_combo: str
    entry: EntryConfig
    exit: ExitConfig
    description: str


@dataclass(frozen=True)
class StrategyRelease:
    """One promoted B1 release consumed by backtests and daily selection."""

    id: str
    model_dir: str
    model_manifest: str
    model_names: tuple[str, ...]
    backtest_summary: str
    compatibility_audit: str
    strategies: tuple[StrategyConfig, ...]


def _optional_float(raw: dict[str, Any], key: str) -> float | None:
    value = raw.get(key)
    return float(value) if value is not None else None


def _strategy_from_dict(raw: dict[str, Any]) -> StrategyConfig:
    entry_raw = raw["entry"]
    exit_raw = raw["exit"]
    entry = EntryConfig(
        min_up5=_optional_float(entry_raw, "min_up5"),
        min_up8=_optional_float(entry_raw, "min_up8"),
        min_up10=_optional_float(entry_raw, "min_up10"),
        max_down2=_optional_float(entry_raw, "max_down2"),
        max_down3=_optional_float(entry_raw, "max_down3"),
    )
    if all(
        value is None
        for value in (
            entry.min_up5,
            entry.min_up8,
            entry.min_up10,
            entry.max_down2,
            entry.max_down3,
        )
    ):
        raise ValueError(f"strategy {raw.get('id')} has no entry threshold")
    return StrategyConfig(
        id=str(raw["id"]),
        name=str(raw["name"]),
        enabled=bool(raw.get("enabled", True)),
        priority=int(raw.get("priority", 100)),
        backtest_combo=str(raw.get("backtest_combo") or raw["id"]),
        entry=entry,
        exit=ExitConfig(
            kind=str(exit_raw["kind"]),
            hold_days=int(exit_raw["hold_days"]),
            take_profit=_optional_float(exit_raw, "take_profit"),
            stop_loss=_optional_float(exit_raw, "stop_loss"),
            trail_drawdown=_optional_float(exit_raw, "trail_drawdown"),
        ),
        description=str(raw.get("description", "")),
    )


def load_strategy_release(path: Path, include_disabled: bool = False) -> StrategyRelease:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    release_raw = payload.get("release") or {}
    required_release_fields = {
        "id",
        "model_dir",
        "model_manifest",
        "model_names",
        "backtest_summary",
        "compatibility_audit",
    }
    missing = sorted(required_release_fields - set(release_raw))
    if missing:
        raise ValueError(f"strategy release missing fields: {missing}")
    strategies = tuple(_strategy_from_dict(item) for item in payload.get("strategies", []))
    ids = [strategy.id for strategy in strategies]
    if len(ids) != len(set(ids)):
        raise ValueError("strategy ids must be unique")
    selected = strategies if include_disabled else tuple(
        strategy for strategy in strategies if strategy.enabled
    )
    if not selected:
        raise ValueError("strategy release has no enabled strategies")
    return StrategyRelease(
        id=str(release_raw["id"]),
        model_dir=str(release_raw["model_dir"]),
        model_manifest=str(release_raw["model_manifest"]),
        model_names=tuple(str(item) for item in release_raw["model_names"]),
        backtest_summary=str(release_raw["backtest_summary"]),
        compatibility_audit=str(release_raw["compatibility_audit"]),
        strategies=tuple(sorted(selected, key=lambda strategy: (strategy.priority, strategy.id))),
    )


def load_strategy_configs(path: Path, include_disabled: bool = False) -> list[StrategyConfig]:
    return list(load_strategy_release(path, include_disabled=include_disabled).strategies)
