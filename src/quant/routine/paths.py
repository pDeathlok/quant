from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs/strategies/b1_selected.yaml"
DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
CANDIDATE_PATH = PROJECT_ROOT / "data/features/b1/candidates_strict_no_volume_20240101.parquet"
ROUTINE_DIR = PROJECT_ROOT / "data/routine"
WEB_DATA_DIR = PROJECT_ROOT / "web/data"
REPORTS_DIR = PROJECT_ROOT / "reports/b1/current"
