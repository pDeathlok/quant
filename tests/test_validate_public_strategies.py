from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research/validate_public_strategies.py"
spec = importlib.util.spec_from_file_location("validate_public_strategies", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_signal_only_affects_returns_after_next_close() -> None:
    days = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"A": [100, 110, 121, 133.1]}, index=days)
    old_fee = module.FEE_ONE_WAY
    try:
        module.FEE_ONE_WAY = 0.0
        returns, turnover, count = module.simulate(prices, {days[0]: {"A": 1.0}})
    finally:
        module.FEE_ONE_WAY = old_fee
    assert list(returns.round(8)) == [0.0, 0.0, 0.1, 0.1]
    assert turnover == 1.0
    assert count == 1
