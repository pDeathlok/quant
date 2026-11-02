import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def test_close_exit_flags_do_not_reject_already_reached_limits(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    module = importlib.import_module("audit_overnight_close_exit_limits")
    frame = pd.DataFrame({"entry_close": [10.] * 5, "next_high": [10.1, 10., 10., 10., np.nan],
                          "next_close": [9., 9., 9.02, 9., 9.], "down_limit": [9., 9., 9., np.nan, 9.]})
    flags = module.exit_flags(frame, .005)
    assert flags.close_fallback.tolist() == [False, True, True, True, False]
    assert flags.fallback_at_lower.tolist() == [False, True, False, False, False]
    assert flags.fallback_limit_unknown.tolist() == [False, False, False, True, False]
