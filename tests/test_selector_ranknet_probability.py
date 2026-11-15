import numpy as np
import pandas as pd

from scripts.research.build_selector_score_probability_bands import _ranknet_probability_sample


def test_outcome_availability_does_not_change_display_percentile():
    class FrozenModel:
        required_features = ('feature',)

        def predict(self, frame):
            assert set(frame.columns) == {'date', 'symbol', 'feature'}
            assert frame['symbol'].tolist() == ['a', 'b', 'c', 'd', 'e']
            return pd.DataFrame({'ranking_score': [0.1, 0.2, 0.3, 0.4, 0.5]})

    frame = pd.DataFrame({
        'symbol': ['pre', 'a', 'b', 'c', 'd', 'e', 'post'],
        'date': pd.to_datetime(['2025-12-31', *(['2026-01-05'] * 5), '2026-08-24']),
        'feature': 1.0,
        'outcome': [0.1, 0.1, np.nan, 0.0, -0.1, 0.1, 0.1],
        'maturity': pd.to_datetime(['2026-01-14'] * 5 + ['2026-09-10', '2026-08-31']),
        'maturity_reason': ['mature', 'mature', 'missing_entry_bar', 'locked_limit_up', 'mature', 'mature', 'mature'],
        'locked_limit_up': [False, False, False, True, False, False, False],
    })
    sample = _ranknet_probability_sample(frame, FrozenModel(), selection_end='2026-01-01',
                                        signal_end='2026-08-21', evaluation_asof='2026-09-04')
    assert sample['display_score'].tolist() == [0, 25, 50, 75, 100]
    assert sample['included'].tolist() == [True, False, True, True, False]
    observed = sample.loc[sample['included']]
    assert observed['display_score'].tolist() == [0, 50, 75]
    assert observed['outcome'].gt(0).tolist() == [True, False, False]
