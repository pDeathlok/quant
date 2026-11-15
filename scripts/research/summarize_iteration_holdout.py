"""Diagnostic ablations and paired uncertainty; never reselect on holdout."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from evaluate_iteration_holdout import initialize, replay
from select_iteration_applications import score_arm, metrics
from model_iteration_v2_protocol import OUT, BOUNDS
import validate_model_applications as app


def main():
    frame = initialize()
    start, end = [int(i) for i in app._CALENDAR.searchsorted(pd.to_datetime([BOUNDS['holdout_start'], BOUNDS['holdout_end']]))]
    validation = pd.read_csv(OUT / 'validation_applications.csv')
    rows = []
    for arm, group in validation.loc[validation.arm.ne('capital_mix')].groupby('arm'):
        # Validation-only arm selection. This table is diagnostic, not a release selector.
        selected = group.sort_values(['positive_dual_pass', 'selection_score', 'return', 'name'], ascending=[False, False, False, True]).iloc[0]
        source_arm = 'calibrator_only' if arm == 'frozen_return_old_cal' else arm
        parts = []
        for side in ['left', 'right']:
            predictions = pd.read_parquet(OUT / f'fresh_predictions_{side}_{source_arm}.parquet').assign(side=side)
            if arm == 'frozen_return_old_cal':
                cal = json.loads((ROOT / f'reports/research/sample_label_research_v1/calibrator_{side}_2026_return.json').read_text())
                predictions['expected_net'] = np.interp(predictions.raw_score, cal['x'], cal['y'])
            parts.append(predictions)
        scored = score_arm(frame, source_arm, pd.concat(parts, ignore_index=True))
        if pd.notna(selected.market_gate):
            scored = scored.loc[scored.market_breadth_above_ma20_ratio.ge(selected.market_gate)]
        policy = app.Policy(name=selected['name'], gate=-np.inf if pd.isna(selected.gate) else selected.gate, cost=.002)
        result, _, _ = replay(scored, policy, start, end)
        rows.append({'arm': arm, 'validation_selected_recipe': selected['name'],
                     'validation_return': selected['return'], 'validation_drawdown': selected.max_drawdown,
                     'holdout_return': result['return'], 'holdout_drawdown': result['max_drawdown'],
                     'holdout_closed': result['closed'], 'holdout_exposure': result['average_exposure']})
    pd.DataFrame(rows).to_csv(OUT / 'diagnostic_ablation_results.csv', index=False)
    paths = {name: pd.read_parquet(OUT / f'holdout_nav_{name}_20bp.parquet').nav.to_numpy() for name in ['baseline', 'v1', 'v2']}
    daily = {name: x / np.r_[1., x[:-1]] - 1 for name, x in paths.items()}
    rng = np.random.default_rng(7312)
    n = len(paths['baseline']); block = 5; draws = {name: [] for name in ['v1', 'v2']}
    for _ in range(4000):
        starts = rng.integers(0, n, size=(n + block - 1) // block)
        indices = np.concatenate([(i + np.arange(block)) % n for i in starts])[:n]
        base = metrics(np.cumprod(1 + daily['baseline'][indices]))
        for name in draws:
            stats = metrics(np.cumprod(1 + daily[name][indices]))
            draws[name].append([stats['return'] - base['return'], base['max_drawdown'] - stats['max_drawdown']])
    uncertainty = {name: {'paired_return_gain_ci95': np.quantile(values, [.025, .975], axis=0)[:, 0].tolist(),
                         'paired_drawdown_reduction_ci95': np.quantile(values, [.025, .975], axis=0)[:, 1].tolist(),
                         'bootstrap_joint_positive_fraction': float(np.mean(np.all(np.asarray(values) > 0, axis=1)))}
                   for name, values in draws.items()}
    (OUT / 'holdout_uncertainty.json').write_text(json.dumps({'block_sessions': block, 'samples': 4000, 'seed': 7312,
        'limitation': '43 sessions; conditional diagnostic, no correction for adaptive model selection or vendor vintage uncertainty.',
        'results': uncertainty}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(uncertainty, indent=2))


if __name__ == '__main__':
    main()
