from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research/validate_drawdown_overlays.py"
spec = importlib.util.spec_from_file_location("validate_drawdown_overlays", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_overlay_changes_exposure_after_next_close() -> None:
    days = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.Series([0.0, 0.1, 0.1, 0.1], index=days)
    signal = pd.Series([0.0], index=[days[0]])
    result = module.overlay(returns, signal)
    assert list(result.round(8)) == [0.0, 0.099, 0.0, 0.0]
