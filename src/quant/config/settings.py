from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
import yaml


@dataclass
class Settings:
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    data_source: str = "tushare"
    default_commission: float = 0.0003
    default_slippage: float = 0.0
    default_initial_cash: float = 1000000.0

    broker_config: Dict = field(default_factory=dict)
    strategy_config: Dict = field(default_factory=dict)
    risk_limits: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.cache_dir = self.data_dir / "cache"
        self.log_dir = self.project_root / "logs"
        self.reports_dir = self.project_root / "backtest" / "reports"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

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


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
