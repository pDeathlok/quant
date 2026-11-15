"""Regression checks for outcome filtering and position-level capital isolation."""
import numpy as np
import pandas as pd

from scripts.research.research_sample_labels_v1 import ledger, sample_weights, slices


def test_boundary_filter_only_changes_training_rows():
    frame = pd.DataFrame({
        'date': pd.to_datetime(['2024-12-01', '2024-12-02', '2025-08-01', '2025-08-02']),
        'maturity': pd.to_datetime(['2024-12-10', '2024-12-11', '2025-08-10', '2026-01-05']),
        'planned_maturity': pd.to_datetime(['2024-12-10', '2024-12-11', '2025-08-10', '2025-08-11']),
        'outcome': [.001, .02, .001, -.2],
    })
    fit, _, _, test = slices(frame, ('2025-01-01', '2025-04-01', '2025-07-01', '2026-01-01'))
    keep, _ = sample_weights(fit, 'drop_005', '2025-01-01')
    assert keep.tolist() == [False, True]
    assert len(test) == 2
    assert test.iloc[0].outcome == .001
    assert pd.isna(test.iloc[1].outcome)


def test_unknown_exit_locks_only_one_position():
    rows = []
    for day in pd.date_range('2025-01-01', periods=9):
        for stock in range(20):
            rows.append({
                'date': day, 'symbol': str(stock), 'entry_date': day + pd.Timedelta(days=1),
                'maturity': day + pd.Timedelta(days=2), 'outcome': .01, 'unfilled': False,
            })
    frame = pd.DataFrame(rows)
    frame.loc[0, 'outcome'] = np.nan
    metrics, trades = ledger(frame, np.ones(len(frame)))
    assert metrics['filled'] == 180
    assert metrics['unknown'] == 1
    assert abs(metrics['capital_return'] - trades.contribution.sum()) < 1e-12


def test_unfilled_orders_have_zero_cost_and_zero_capital_lock():
    frame = pd.DataFrame({
        'date': pd.to_datetime(['2025-01-01']), 'symbol': ['A'],
        'entry_date': pd.to_datetime(['2025-01-02']),
        'maturity': pd.to_datetime(['2025-01-09']), 'outcome': [0.], 'unfilled': [True],
    })
    metrics, _ = ledger(frame, np.array([1.]))
    assert metrics['filled'] == 0
    assert abs(metrics['capital_return']) < 1e-12


def test_calibration_plateau_preserves_raw_ranking():
    frame = pd.DataFrame({
        'date': pd.to_datetime(['2025-01-01'] * 21), 'symbol': [str(i) for i in range(21)],
        'entry_date': pd.to_datetime(['2025-01-02'] * 21),
        'maturity': pd.to_datetime(['2025-01-09'] * 21), 'outcome': [.01] * 21, 'unfilled': [False] * 21,
    })
    _, trades = ledger(frame, np.ones(21), rank_scores=np.arange(21))
    assert '0' not in trades.symbol.tolist()
    assert trades.iloc[0].symbol == '20'
