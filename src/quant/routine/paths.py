from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_project_env() -> None:
    """Load local .env values without overriding existing environment variables."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_project_env()
CONFIG_PATH = PROJECT_ROOT / "configs/strategies/b1_selected.yaml"
DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
CANDIDATE_PATH = PROJECT_ROOT / "data/features/b1/candidates_strict_no_volume_20240101.parquet"
ROUTINE_DIR = PROJECT_ROOT / "data/routine"
WEB_DATA_DIR = PROJECT_ROOT / "web/data"
REPORTS_DIR = PROJECT_ROOT / "reports/b1/current"
