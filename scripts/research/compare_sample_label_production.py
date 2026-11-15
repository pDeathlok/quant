"""Frozen production artifacts scored on the 2026 matched research cohorts."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_sample_labels_v1 as experiment
from quant.research.local_ranknet import percentile_scores


def main():
    rows = []
    comparisons = []
    selected = pd.read_csv(experiment.OUT / 'validation_selected.csv')
    for side in ['left', 'right']:
        frame = pd.read_parquet(experiment.OUT / f'dataset_{side}.parquet')
        test = experiment.slices(frame, experiment.FOLDS['2026'])[-1]
        if side == 'left':
            scores = []
            for name in ['base50', 'local10_all']:
                model = xgb.Booster()
                model.load_model(experiment.ROOT / f'models/production/left_side_ranknet_v6/{name}.ubj')
                scores.append(percentile_scores(test.date, model.predict(experiment.matrix(test, model.feature_names))))
            score = .5 * scores[0] + .5 * scores[1]
        else:
            artifact = joblib.load(experiment.ROOT / 'models/production/right_side_unified_canonical_v2/ranking.joblib')
            score = artifact['model'].predict_proba(test)[:, 1]
        assert np.isfinite(score).all()
        metrics, trades = experiment.ledger(test, score)
        trades.to_parquet(experiment.OUT / f'trades_{side}_2026_production_reference_fixed_top20.parquet', index=False)
        predictions = test[experiment.KEYS].copy()
        predictions['score'] = score
        predictions.to_parquet(experiment.OUT / f'production_reference_{side}_2026.parquet', index=False)
        rows.append({'side': side, 'fold': '2026', **metrics})
        choice = selected.loc[selected.side.eq(side) & selected.fold.eq('2026')].iloc[0]
        chosen_trades = pd.read_parquet(experiment.OUT / f'trades_{side}_2026_{choice.selected_arm}_validation_gate.parquet')
        days = pd.bdate_range(test.date.min(), test.date.max())
        delta = (chosen_trades.groupby('date').contribution.sum().reindex(days, fill_value=0)
                 - trades.groupby('date').contribution.sum().reindex(days, fill_value=0)).to_numpy()
        rng = np.random.default_rng(73)
        starts = rng.integers(0, len(delta), (1000, int(np.ceil(len(delta) / 10))))
        indices = ((starts[:, :, None] + np.arange(10)) % len(delta)).reshape(1000, -1)[:, :len(delta)]
        low, high = np.quantile(delta[indices].mean(axis=1) * 10000, [.025, .975])
        comparisons.append({'side': side, 'selected': choice.selected_arm,
            'capital_difference_pp': float(delta.sum() * 100),
            'daily_difference_bp': float(delta.mean() * 10000), 'ci_low_bp': low, 'ci_high_bp': high})
    pd.DataFrame(rows).to_csv(experiment.OUT / 'production_reference.csv', index=False)
    pd.DataFrame(comparisons).to_csv(experiment.OUT / 'production_paired_comparison.csv', index=False)
    experiment.save('production_reference_scope.json', {
        'left': 'Published v6 components on original reconstructed research features. Fit before2025-07 and selection before2026.',
        'right': 'Published foldC artifact, trained2020-2024, calibrated/selected2025; scores restricted pure-right cohort using research-materialized features.',
        'limitations': 'Matched cohort and ledger comparison. Research feature vintages and right feature pipeline have not been certified equal to production snapshots. 2025H2 excluded because published artifacts used later data for fitting/calibration/selection.'})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
