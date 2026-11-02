from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.kronos_data_repair import fill_missing_cells


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.DataFrame({"ts_code": ["002594.SZ", "000001.SZ"], "trade_date": ["20260601"] * 2,
                           "open": [10., 10.], "high": [11., 11.], "low": [9., 9.], "close": [10., 10.],
                           "pre_close": [10., 10.], "vol": [100., 100.], "amount": [100., 100.]})
    original = source.assign(volume=source.vol, untouched="keep")
    original["amount"] = np.nan
    return original, source


def test_null_only_repair_is_scoped_and_idempotent() -> None:
    original, source = frames()
    result, changes, rejected = fill_missing_cells(original, source, {"002594.SZ"})
    assert len(changes) == 1 and not rejected
    assert result.amount.iloc[0] == 100 and pd.isna(result.amount.iloc[1])
    twice, changes, rejected = fill_missing_cells(result, source, {"002594.SZ"})
    pd.testing.assert_frame_equal(twice, result)
    assert not changes and not rejected


def test_conflicting_price_or_volume_units_cannot_silently_fill_amount() -> None:
    original, source = frames()
    source.loc[0, "vol"] *= 100
    result, changes, rejected = fill_missing_cells(original, source, {"002594.SZ"})
    assert not changes and rejected
    pd.testing.assert_frame_equal(result, original)
