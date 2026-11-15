import numpy as np
import pandas as pd
import pytest

from quant.ml.application_portfolio import combine_sleeve_nav, rank_sleeves

WEIGHTS = {'score_three_way': .25, 'score_return_path': .25, 'weighted_low_volatility': .5}


def candidates():
    return pd.DataFrame({'symbol': ['A', 'B', 'A', 'C'], 'date': ['2026-01-02'] * 4,
                         'side': ['left', 'left', 'right', 'right'],
                         'base_score': [.9, .4, .2, .7], 'path_score': [.2, .8, .6, .3],
                         'rank_score': [.1, .9, .5, .8], 'volatility_20d': [.2, .1, .3, .4]})


def test_capital_drift_is_preserved_without_daily_rebalancing():
    paths = {'a': pd.Series([2., 1.]), 'b': pd.Series([1., 2.])}
    result = combine_sleeve_nav(paths, {'a': .5, 'b': .5})
    np.testing.assert_allclose(result, [1.5, 1.5])
    # Averaging daily returns would incorrectly produce 1.875 at the end.
    assert result.iloc[-1] != 1.875


def test_mismatched_calendars_and_invalid_weights_are_rejected():
    with pytest.raises(ValueError, match='calendars'):
        combine_sleeve_nav({'a': pd.Series([1.], index=[1]), 'b': pd.Series([1.], index=[2])}, {'a': .5, 'b': .5})
    with pytest.raises(ValueError, match='weights'):
        combine_sleeve_nav({'a': pd.Series([1.])}, {'a': 2.})


def test_overlap_prefers_right_even_when_left_score_is_higher():
    output = rank_sleeves(candidates(), WEIGHTS)
    assert len(output) == 9
    assert output.loc[output.symbol.eq('A'), 'side'].eq('right').all()
    assert not output.duplicated(['sleeve', 'symbol', 'date']).any()


def test_future_and_input_order_do_not_change_earlier_rankings():
    frame = candidates()
    expected = rank_sleeves(frame, WEIGHTS)
    future = frame.assign(date='2026-01-05', volatility_20d=100., base_score=.01)
    output = rank_sleeves(pd.concat([frame, future]).sample(frac=1., random_state=7), WEIGHTS)
    output = output.loc[output.date.eq('2026-01-02')].reset_index(drop=True)
    pd.testing.assert_frame_equal(output, expected)


def test_volatility_rank_is_computed_within_side_before_overlap_removal():
    output = rank_sleeves(candidates(), WEIGHTS)
    actual = output.loc[output.sleeve.eq('weighted_low_volatility')].set_index('symbol').score
    assert np.isclose(actual['B'], .8 * .4 + .2 * .5)
    assert np.isclose(actual['A'], .8 * .2 + .2 * .5)


def test_duplicate_normalized_dates_and_nonfinite_scores_are_rejected():
    frame = candidates()
    duplicate = frame.iloc[[0]].assign(date='2026-01-02 12:00')
    with pytest.raises(ValueError, match='Unique'):
        rank_sleeves(pd.concat([frame, duplicate]), WEIGHTS)
    frame.loc[0, 'base_score'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        rank_sleeves(frame, WEIGHTS)
