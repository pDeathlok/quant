"""Cost sensitivity with all model and policy choices held fixed."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_sample_labels_v1 as experiment


def main():
    rows = []
    selected = pd.read_csv(experiment.OUT / 'validation_selected.csv')
    for selection in selected.itertuples():
        frame = pd.read_parquet(experiment.OUT / f'dataset_{selection.side}.parquet')
        fit, _, _, test = experiment.slices(frame, experiment.FOLDS[selection.fold])
        candidates = [('return', 'fixed_top20', -np.inf), ('drop_010', 'fixed_top20', -np.inf),
                      ('good_path', 'fixed_top20', -np.inf), ('path_boundary', 'fixed_top20', -np.inf)]
        if selection.selected_arm != 'cash':
            candidates.append((selection.selected_arm, 'validation_gate', selection.threshold))
        if selection.fold == '2026':
            candidates.append(('production_reference', 'fixed_top20', -np.inf))
        for arm, policy, threshold in candidates:
            if arm == 'production_reference':
                prediction = pd.read_parquet(experiment.OUT / f'production_reference_{selection.side}_2026.parquet')
                score = prediction.score.to_numpy()
                rank = score
            else:
                prediction = pd.read_parquet(experiment.OUT / f'predictions_{selection.side}_{selection.fold}_{arm}.parquet')
                score = prediction['expected_net' if policy == 'validation_gate' else 'raw_score'].to_numpy()
                rank = prediction.raw_score.to_numpy()
            pd.testing.assert_frame_equal(test[experiment.KEYS].reset_index(drop=True), prediction[experiment.KEYS].reset_index(drop=True))
            eligible = None
            if arm == 'observable_filter':
                eligible = test.turnover_rate.ge(fit.turnover_rate.quantile(.1)) & test.volatility_20d.le(fit.volatility_20d.quantile(.9)) & test.event_age.eq(0)
            for cost in [.001, .002, .004, .006]:
                stressed = test.copy()
                known_fills = stressed.outcome.notna() & ~stressed.unfilled
                stressed.loc[known_fills, 'outcome'] -= cost - .002
                metrics, _ = experiment.ledger(stressed, score, threshold, eligible, rank)
                rows.append({'side': selection.side, 'fold': selection.fold, 'arm': arm,
                             'policy': policy, 'round_trip_cost_bp': cost * 10000, **metrics})
    pd.DataFrame(rows).to_csv(experiment.OUT / 'cost_sensitivity.csv', index=False)
    print('COST SCENARIOS', len(rows))


if __name__ == '__main__':
    main()
