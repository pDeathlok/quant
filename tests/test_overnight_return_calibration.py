import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_return_calibration_rejects_future_exits_and_nonfinite_labels(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    fit = importlib.import_module("calibrate_overnight_unfiltered_returns").fit_return_calibrator
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-07-01", "2025-07-02"]),
                          "exit_date": pd.to_datetime(["2025-07-02", "2025-07-03"]),
                          "expected_net": [-.01, .001], "net": [-.02, .003]})
    model = fit(frame)
    np.testing.assert_allclose(model.predict([-.01, .001]), [-.02, .003])
    with pytest.raises(ValueError, match="2025Q3"):
        fit(frame.assign(exit_date=pd.Timestamp("2025-10-01")))
    with pytest.raises(ValueError, match="finite"):
        fit(frame.assign(net=np.nan))
    with pytest.raises(ValueError, match="2025Q3"):
        fit(frame.assign(date=pd.Timestamp("2026-01-05")))
