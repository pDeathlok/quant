import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_minute_evaluation_retains_unknown_and_execution_flags(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    module = importlib.import_module("train_overnight_minutes")
    picks = pd.DataFrame({"date": pd.to_datetime(["2025-10-09", "2025-10-10"]),
                          "net": [.001, np.nan], "entry_final_at_limit": [True, False],
                          "future_quality": ["ok", "source_price_conflict"]})
    result = module.evaluate(picks, pd.DatetimeIndex(picks.date), 1)
    assert result["signals"] == 2
    assert result["unknown"] == 1
    assert result["known_win"] == 1.
    assert result["win_missing_as_failure"] == .5
    assert result["cost40_known_win"] == 0.
    assert result["cost40_known_mean_net"] == pytest.approx(-.001)
    assert result["entry_final_at_limit"] == 1
    assert result["source_conflicts"] == 1
