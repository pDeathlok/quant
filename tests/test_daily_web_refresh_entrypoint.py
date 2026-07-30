from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/run_daily_web_refresh.py"
SPEC = importlib.util.spec_from_file_location("run_daily_web_refresh_test_module", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_daily_refresh_entrypoint_restarts_service_by_default() -> None:
    assert MODULE.daily_refresh_args([]) == ["--restart-service"]
    assert MODULE.daily_refresh_args(["--scope", "all"]) == [
        "--restart-service",
        "--scope",
        "all",
    ]


def test_daily_refresh_entrypoint_respects_explicit_service_policy() -> None:
    assert MODULE.daily_refresh_args(["--no-restart-service"]) == [
        "--no-restart-service",
    ]
    assert MODULE.daily_refresh_args(["--restart-service"]) == [
        "--restart-service",
    ]
