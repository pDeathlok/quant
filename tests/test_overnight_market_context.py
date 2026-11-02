import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_market_context_is_causal_and_uses_same_day_breadth(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    module = importlib.import_module("overnight_market_context")
    dates = pd.bdate_range("2024-01-02", periods=35)
    bars = pd.DataFrame({"ts_code": ["000001.SZ", "600000.SH"] * 35,
        "trade_date": dates.repeat(2).strftime("%Y%m%d"), "open": [10.] * 70,
        "close": [11., 9.] * 35, "high": [12.] * 70, "low": [8.] * 70,
        "pre_close": [10.] * 70, "vol": [100.] * 70})
    full = module.market_context(bars, minimum_stocks=2)
    prefix = module.market_context(bars.iloc[:50], minimum_stocks=2)
    pd.testing.assert_frame_equal(full.iloc[:25], prefix)
    assert full.today_market_breadth.eq(.5).all()
    assert np.allclose(full.today_market_return_mean, 0.)
    assert full.today_market_breadth_mean_20.iloc[:19].isna().all()
    assert full.today_market_breadth_mean_20.iloc[19] == .5
    aliased = bars.assign(volume=bars.vol)
    aliased.loc[0, "vol"] = np.nan
    pd.testing.assert_frame_equal(full, module.market_context(aliased, minimum_stocks=2))
    # Fund rows must not change an A-share statistic.
    fund = bars.iloc[:1].assign(ts_code="510300.SH", close=12.)
    pd.testing.assert_frame_equal(full, module.market_context(pd.concat([bars, fund]), minimum_stocks=2))
    with pytest.raises(ValueError, match="Duplicate"):
        module.market_context(pd.concat([bars, bars.iloc[:1]]), minimum_stocks=2)
    with pytest.raises(ValueError, match="coverage"):
        module.market_context(bars.assign(vol=0.), minimum_stocks=2)
    with pytest.raises(ValueError, match="coverage"):
        module.market_context(bars.assign(vol=0., volume=100.), minimum_stocks=2)
