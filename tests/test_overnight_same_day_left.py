import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_signal_metrics_use_current_day_equal_weight_and_keep_unknown(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    module = importlib.import_module("audit_overnight_same_day_left")
    days = pd.bdate_range("2025-07-01", periods=3)
    picks = pd.DataFrame({"date": [days[0], days[0], days[1]], "net": [.01, .03, -.01]})
    result = module.signal_metrics(picks, days)
    assert result["daily_penalized_mean"] == pytest.approx((.02-.01)/3)
    assert result["allocation"] == pytest.approx(2/3)
    picks.loc[0, "net"] = np.nan
    result = module.signal_metrics(picks, days)
    assert result["signals"] == 3 and result["unknown"] == 1
    assert result["daily_penalized_mean"] == pytest.approx(((-.1+.03)/2-.01)/3)


def test_same_day_signal_selection_excludes_future_quarter_labels(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    module = importlib.import_module("audit_overnight_same_day_left")
    f = pd.DataFrame({"date": pd.to_datetime(["2025-09-29", "2025-09-30", "2025-10-09"]),
                      "exit_date": pd.to_datetime(["2025-09-30", "2025-10-09", "2025-10-10"])})
    assert module.quarter_mask(f, "2025Q3").tolist() == [True, False, False]
    assert module.quarter_mask(f, "2025Q4").tolist() == [False, False, True]
