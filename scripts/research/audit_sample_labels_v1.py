"""Verify saved predictions and summarize paired historical ablations."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_sample_labels_v1 as experiment


def main():
    out = experiment.OUT
    results = pd.read_csv(out / 'results.csv')
    training = pd.read_csv(out / 'training.csv')
    choices = json.loads((out / 'choices.json').read_text())
    checks, comparisons, selections = [], [], []
    for side in ['left', 'right']:
        frame = pd.read_parquet(out / f'dataset_{side}.parquet')
        for fold, bounds in experiment.FOLDS.items():
            fit, cal, valid, test = experiment.slices(frame, bounds)
            cohort = results.loc[results.side.eq(side) & results.fold.eq(fold)]
            dates = pd.bdate_range(test.date.min(), test.date.max())
            for arm in experiment.ARMS:
                model = xgb.Booster()
                model.load_model(out / f'model_{side}_{fold}_{arm}.ubj')
                stored = pd.read_parquet(out / f'predictions_{side}_{fold}_{arm}.parquet')
                pd.testing.assert_frame_equal(test[experiment.KEYS].reset_index(drop=True), stored[experiment.KEYS].reset_index(drop=True))
                rebuilt = model.predict(experiment.matrix(test, model.feature_names))
                np.testing.assert_array_equal(rebuilt, stored.raw_score)
                fit_row = training.loc[training.side.eq(side) & training.fold.eq(fold) & training.arm.eq(arm)].iloc[0]
                assert pd.Timestamp(fit_row.fit_label_end) < pd.Timestamp(bounds[0])
                assert pd.Timestamp(fit_row.cal_label_end) < pd.Timestamp(bounds[1])
                allowed = set(json.loads((experiment.SOURCE / 'dataset_complete.json').read_text())['core'])
                allowed.update(c for c in frame if c.startswith('event_'))
                assert set(model.feature_names) <= allowed
                checks.append({'side': side, 'fold': fold, 'arm': arm, 'prediction_rows': len(stored), 'passed': True})
            for row in cohort.loc[cohort.policy.eq('fixed_top20')].itertuples():
                control = {'path_boundary': 'good_path', 'rank_margin': 'rank'}.get(row.arm, 'return')
                baseline_path = out / f'trades_{side}_{fold}_{control}_fixed_top20.parquet'
                baseline = pd.read_parquet(baseline_path).groupby('date').contribution.sum().reindex(dates, fill_value=0)
                trades = pd.read_parquet(out / f'trades_{side}_{fold}_{row.arm}_fixed_top20.parquet')
                assert abs(trades.contribution.sum() - row.capital_return) < 1e-10
                daily = trades.groupby('date').contribution.sum().reindex(dates, fill_value=0)
                delta = (daily - baseline).to_numpy()
                rng = np.random.default_rng(73)
                starts = rng.integers(0, len(delta), size=(1000, int(np.ceil(len(delta)/10))))
                indices = (starts[:, :, None] + np.arange(10)) % len(delta)
                means = delta[indices.reshape(1000, -1)[:, :len(delta)]].mean(axis=1) * 10000
                comparisons.append({'side': side, 'fold': fold, 'arm': row.arm, 'control': control,
                    'capital_return': row.capital_return, 'daily_contribution_delta_bp': delta.mean()*10000,
                    'ci_low_bp': np.quantile(means, .025), 'ci_high_bp': np.quantile(means, .975),
                    'filled': row.filled, 'win_rate': row.win_rate, 'unknown': row.unknown})
            candidate_choices = [c for c in choices if c['side'] == side and c['fold'] == fold]
            grid = [(g['capital_return'], c['arm'], g['threshold']) for c in candidate_choices
                    for g in c['validation_grid'] if g['filled'] >= 100 and g['capital_return'] > 0]
            if grid:
                validation_return, selected_arm, threshold = max(grid, key=lambda value: value[0])
                row = cohort.loc[cohort.arm.eq(selected_arm) & cohort.policy.eq('validation_gate')].iloc[0]
                selections.append({'side': side, 'fold': fold, 'selected_arm': selected_arm,
                    'validation_return': validation_return, 'threshold': threshold,
                    'test_return': row.capital_return, 'test_filled': int(row.filled), 'test_win_rate': row.win_rate})
            else:
                selections.append({'side': side, 'fold': fold, 'selected_arm': 'cash', 'test_return': 0., 'test_filled': 0})
    pd.DataFrame(comparisons).to_csv(out / 'paired_comparisons.csv', index=False)
    pd.DataFrame(selections).to_csv(out / 'validation_selected.csv', index=False)
    sources_match = all(experiment.digest(p)==value for p,value in json.loads((out/'sources.json').read_text()).items())
    assert sources_match
    experiment.save('verification.json', {'passed': True, 'checks': checks,
        'tests': 'Four regression tests for train-only filtering, per-position freezing, no-fill cost and calibration plateau ranking.',
        'ci_scope': 'Exploratory 10-business-day circular bootstrap of signal-date settled capital contributions. Not MTM daily returns; not adjusted for multiple comparisons.',
        'script_sha256': experiment.digest(__file__), 'experiment_sha256': experiment.digest(experiment.__file__),
        'source_hashes_match': sources_match})
    print(pd.DataFrame(selections).to_string(index=False))
    print('VERIFIED', len(checks), 'models')


if __name__ == '__main__':
    main()
