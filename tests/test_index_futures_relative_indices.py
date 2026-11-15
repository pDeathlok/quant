"""Relative index factors must be causal and independent of index point scales."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/research"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("relative_indices", SCRIPTS / "evaluate_index_futures_relative_indices.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def prices():
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2019-01-01", periods=250).strftime("%Y%m%d")
    columns = list(research.TARGETS.values()) + list(research.EXTERNAL.values())
    return pd.DataFrame(np.exp(rng.normal(0, .01, (250, 7)).cumsum(axis=0)) * 1000,
                        index=pd.Index(index, name="trade_date"), columns=columns)


def test_future_price_changes_do_not_modify_past_factors():
    original = prices()
    changed = original.copy()
    changed.iloc[200:] *= np.arange(1, 8) * 10
    first = research.relative_features(original)
    second = research.relative_features(changed)
    pd.testing.assert_frame_equal(first[first.trade_date.lt(original.index[200])],
                                  second[second.trade_date.lt(original.index[200])])


def test_index_rebasing_does_not_change_factors():
    original = prices()
    changed = original.mul([.1, 2, 3, .4, 5, .6, 7], axis=1)
    first = research.relative_features(original)
    second = research.relative_features(changed)
    pd.testing.assert_frame_equal(first, second, check_exact=False, atol=1e-8, rtol=1e-8)


def test_missing_auxiliary_prices_are_not_backward_filled():
    original = prices()
    original.loc[original.index[180], "399006.SZ"] = np.nan
    factors = research.relative_features(original)
    assert factors[factors.trade_date.eq(original.index[180])].external_growth_ret_diff_5.isna().all()


def test_relative_return_is_a_dimensionless_difference():
    original = prices()
    target, benchmark = original.iloc[:, 0], original.iloc[:, 1]
    actual = research.relative_pair(target, benchmark)
    expected = np.log(target / target.shift(20)) - np.log(benchmark / benchmark.shift(20))
    np.testing.assert_allclose(actual.ret_diff_20, expected, atol=1e-12)
