"""No source extrapolation, no future-dependent ranks, explicit availability lag."""
import numpy as np
import pandas as pd
import pytest

from quant.features.index_panic import past_percentile, price_stress, option_features


def test_rank_compares_only_prior_values_and_handles_ties():
    s = pd.Series([1., 2., 3., 2., 100.])
    r = past_percentile(s, window=3, minimum=3)
    assert r.iloc[:3].isna().all()
    assert r.iloc[3] == .5
    assert r.iloc[4] == 1.
    pd.testing.assert_series_equal(r.iloc[:4], past_percentile(s.iloc[:4],3,3))


def test_qvix_missing_is_not_filled_and_signal_has_extra_session_lag():
    dates = pd.Index(pd.bdate_range("2020-01-01",periods=110).strftime("%Y%m%d"))
    close = pd.Series(np.linspace(15,25,110),index=dates)
    close.iloc[90] = np.nan
    close.iloc[100:] = np.nan
    rv = pd.Series(.15,index=dates)
    f = option_features(close,rv,dates)
    assert f.loc[dates[89],"source_date"] == dates[88]
    assert f.loc[dates[91],"missing"] == 1
    assert pd.isna(f.loc[dates[91],"level"])
    assert f.loc[dates[92],"missing"] == 1  # five-day change does not bridge outage
    assert f.loc[dates[101]:,"level"].isna().all()
    assert np.isclose(f.loc[dates[89],"premium"],close.iloc[88]/100-.15)
    pd.testing.assert_frame_equal(f.iloc[:89],option_features(close.iloc[:89],rv.iloc[:89],dates[:89]))
    changed = close.copy(); changed.iloc[89:] = 10000
    pd.testing.assert_frame_equal(f.iloc[:90],option_features(changed,rv,dates).iloc[:90])


def test_duplicate_qvix_is_rejected():
    with pytest.raises(ValueError,match="Duplicate"):
        option_features(pd.Series([1,2],index=["20200101"]*2),pd.Series([.2],index=["20200101"]),pd.Index(["20200101"]))


def test_price_stress_prefix_is_invariant_to_future_crash():
    rng = np.random.default_rng(10)
    dates = pd.Index(pd.bdate_range("2019-01-01",periods=400).strftime("%Y%m%d"))
    spot = pd.DataFrame({"close":100*np.exp(np.cumsum(rng.normal(0,.01,400))),"vol":rng.uniform(100,200,400)},index=dates)
    full = price_stress(spot)
    pd.testing.assert_frame_equal(full.iloc[:300],price_stress(spot.iloc[:300]))
    spot.loc[dates[300]:,"close"] = 1
    pd.testing.assert_frame_equal(full.iloc[:300],price_stress(spot).iloc[:300])
    assert full.panic_price_score.dropna().between(0,100).all()
