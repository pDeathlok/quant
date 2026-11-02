import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def test_relative_prices_are_anchor_invariant_and_do_not_use_labels(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    module = importlib.import_module("overnight_relative_prices")
    frame = pd.DataFrame({f"today_factor_{c}": [10., 20.] for c in (*module.PRICE_COLUMNS, *module.DROPPED)})
    frame["today_raw_close"] = [11., 21.]
    frame["today_factor_rsi_12"] = [40., 60.]
    columns = list(frame)
    frame["net"] = [1., -1.]
    a, cols = module.relative_prices(frame, columns)
    scaled = frame.copy()
    for c in (*module.PRICE_COLUMNS, "close"):
        scaled[f"today_factor_{c}"] *= pd.Series([3., 7.])
    scaled["net"] = [-100., 100.]
    b, cols2 = module.relative_prices(scaled, columns)
    assert cols == cols2
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert len(cols) == len(columns) - 3
    assert "net" not in cols
    assert a.today_raw_close.tolist() == [11., 21.]
    assert a.today_factor_rsi_12.tolist() == [40., 60.]
    frame["today_factor_close"] = [0., np.nan]
    bad, _ = module.relative_prices(frame, columns)
    assert bad.today_relative_ma_20.isna().all()
