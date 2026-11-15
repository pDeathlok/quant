"""Fixed development/holdout protocol for the second application iteration."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/research/model_iteration_v2'
TREE_ARMS = ['return_updated', 'window_6m', 'window_12m', 'window_24m', 'decay_180d',
             'utility_mae', 'utility_tail', 'multitask_tree', 'regime_experts',
             'industry_relative', 'event_execution', 'calibrator_only']
NEURAL_ARMS = ['mlp', 'tabm', 'tabm_multitask']
GATES = [None, 0., .002, .005, .01]
MARKET_GATES = [None, .4, .5]
BOUNDS = {'fit_end': '2026-04-01', 'calibration_end': '2026-07-01',
          'selection_end': '2026-09-04', 'holdout_start': '2026-09-04', 'holdout_end': '2026-11-12'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'bounds': BOUNDS, 'tree_arms': TREE_ARMS, 'neural_arms': NEURAL_ARMS,
        'expected_net_gates': GATES, 'market_breadth_gates': MARKET_GATES,
        'training': 'Outcomes must mature before each fit/calibration cutoff. Original129 features. Same XGB controls unless the named arm changes inputs/objective. Neural preprocessing fits training rows only.',
        'neural': {'seed': 73, 'epochs_max': 16, 'patience': 4, 'batch_size': 1024,
                   'hidden': 128, 'layers': 2, 'tabm_k': 8, 'learning_rate': .001,
                   'selection': 'Calibration-period loss early stopping only, never holdout.'},
        'selection': 'Use July1 through September3 only. Require >=100 closed trades, positive return and lower drawdown than production baseline. Maximize minimum(return improvement, drawdown reduction), then return. If none qualify, retain v1 frozen25/25/50 fallback.',
        'combinations': 'After selecting at most3 validation-qualified distinct model arms, compare equal initial capital mixtures of pairs and all3. Each sleeve keeps its own capital, positions and cash.',
        'holdout': 'New interval for this study, but retrospective data rather than future prospective observations. Open exactly once after choices saved. Never refit/select on holdout outcomes.',
        'acceptance': 'Report positive return AND drawdown below baseline separately from relative dual superiority. Compare to baseline and v1. Cost20bp and60bp. No claim of stable profitability from one short interval.',
        'fallback_authorization': 'User explicitly authorized replacing with existing dual-superior combination if further iteration cannot achieve positive return with controlled drawdown.',
        'industry': 'Use dated SW2021 membership with in/out intervals; current stock_basic industry prohibited as historical labels. Missing/ambiguous membership remains unknown.',
        'execution': 'Same next-close/five-market-session independent cash ledger. No trading orders are submitted by research or application activation.'}
    path = OUT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('Protocol already frozen with different content')
    path.write_text(json.dumps(protocol, indent=2))
    print(path)


if __name__ == '__main__':
    main()
