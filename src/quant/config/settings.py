from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from quant.core.paths import PROJECT_ROOT, ProjectPaths


@dataclass
class Settings:
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    data_source: str = "tushare"
    default_commission: float = 0.0003
    default_slippage: float = 0.0
    default_initial_cash: float = 1000000.0

    broker_config: dict[str, Any] = field(default_factory=dict)
    strategy_config: dict[str, Any] = field(default_factory=dict)
    risk_limits: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        paths = ProjectPaths.from_root(self.project_root)
        self.project_root = paths.root
        self.data_dir = paths.data
        self.cache_dir = paths.cache
        self.log_dir = paths.logs
        self.reports_dir = paths.reports

    def ensure_runtime_directories(self) -> tuple[Path, Path, Path, Path]:
        """Create runtime roots only when an entry point explicitly requests it."""

        return ProjectPaths.from_root(self.project_root).ensure_runtime_directories()

    @classmethod
    def from_yaml(cls, config_path: str) -> "Settings":
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        settings = cls()
        if "broker" in config:
            settings.broker_config = config["broker"]
        if "strategy" in config:
            settings.strategy_config = config["strategy"]
        if "risk" in config:
            settings.risk_limits = config["risk"]
        return settings


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
