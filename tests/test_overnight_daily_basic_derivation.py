from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from quant.features.variable_library import derive_daily_basic_features


def test_daily_basic_derivation_is_causal_and_matches_rolling_definition():
    dates = pd.bdate_range("2024-01-02", periods=60)
    frame = pd.DataFrame({
        "ts_code": ["A"] * 60, "trade_date": dates.strftime("%Y%m%d"),
        "turnover_rate": np.arange(1.0, 61.0), "volume_ratio": np.arange(1.0, 61.0),
        "total_mv": np.arange(100.0, 160.0), "circ_mv": np.arange(50.0, 110.0),
        "pe_ttm": [10.0] * 60, "pb": [2.0] * 60,
    })
    full = derive_daily_basic_features(frame)
    prefix = derive_daily_basic_features(frame.iloc[:40])
    assert_frame_equal(full.iloc[:40], prefix)
    warm = derive_daily_basic_features(frame.iloc[20:])
    assert_frame_equal(full.iloc[40:].reset_index(drop=True), warm.iloc[20:].reset_index(drop=True))
    assert full.loc[30, "turnover_rate_ma20"] == np.mean(np.arange(12.0, 32.0))
    assert np.isclose(full.loc[30, "total_mv_change_20d"], 130 / 110 - 1)
    assert full.loc[30, "pe_ttm_inv"] == 0.1
    assert "ts_volume_ratio" in full and "volume_ratio" not in full


def test_daily_basic_derivation_keeps_stocks_and_missing_history_separate():
    frame = pd.DataFrame({"ts_code": ["A", "B", "A"],
                          "trade_date": ["20240102", "20240103", "20240103"],
                          "turnover_rate": [1.0, 100.0, 3.0], "pe_ttm": [10.0, 0.0, np.nan]})
    result = derive_daily_basic_features(frame)
    assert len(result) == 3
    assert result.turnover_rate_ma5.isna().all()
    assert result.turnover_rate_ma20.isna().all()
    assert result.pe_ttm_inv.notna().sum() == 1
    assert result.loc[result.ts_code.eq("B"), "turnover_rate"].item() == 100.0
