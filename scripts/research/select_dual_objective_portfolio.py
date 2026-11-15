"""Chronological dual-objective selection of independently funded strategy sleeves."""
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import validate_model_applications as previous

SOURCE = previous.OUT
OUT = ROOT / 'reports/research/dual_objective_portfolio_v1'
COMPONENTS = ['score_three_way', 'score_return_path', 'score_return_risk', 'weighted_low_volatility', 'hold_10']


def metrics(values):
    peak = np.maximum.accumulate(np.r_[1., values])[1:]
    return float(values[-1] - 1), float(-(values / peak - 1).min())


def weights_grid():
    return [np.array(parts, dtype=float) / 4 for parts in product(range(4), repeat=len(COMPONENTS))
            if sum(parts) == 4 and sum(p > 0 for p in parts) >= 2]


def load_paths(fold, stage):
    frames = [pd.read_parquet(SOURCE / f'nav_{fold}_{name}_{stage}.parquet') for name in COMPONENTS]
    baseline = pd.read_parquet(SOURCE / f'nav_{fold}_baseline_{stage}.parquet')
    for frame in frames:
        np.testing.assert_array_equal(frame.day, baseline.day)
    return np.column_stack([f.nav for f in frames]), baseline, frames


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'components': COMPONENTS, 'weights': [w.tolist() for w in weights_grid()],
        'allocation': 'Initial capital allocations, each sleeve compounds independently; no cross-sleeve cash transfer or daily rebalancing.',
        'selection': 'Within each fold validation only. Require return>baseline AND maximum drawdown<baseline. Maximize minimum of return improvement and drawdown reduction; break ties by return improvement then fewer sleeves.',
        'acceptance': 'Selected walk-forward policy must beat baseline on both metrics in each historical test fold; final2026 weights must also pass actual production-score replay.',
        'limitations': 'Historical data already explored.2025H2 uses its own earlier selected weights; final2026 weights are not retroactively claimed to have been selected in2025H2.',
    }
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows, selections, acceptance = [], [], []
    final_weights = None
    for fold in previous.prior.FOLDS:
        paths, baseline, _ = load_paths(fold, 'validation')
        base_return, base_drawdown = metrics(baseline.nav.to_numpy())
        eligible = []
        for number, weights in enumerate(weights_grid()):
            result, drawdown = metrics(paths @ weights)
            gain, risk_gain = result - base_return, base_drawdown - drawdown
            score = min(gain, risk_gain)
            rows.append({'fold': fold, 'candidate': number, 'weights': json.dumps(weights.tolist()),
                         'return': result, 'max_drawdown': drawdown, 'return_gain': gain,
                         'drawdown_reduction': risk_gain, 'dual_pass': gain > 0 and risk_gain > 0, 'score': score})
            if gain > 0 and risk_gain > 0:
                eligible.append((score, gain, -int((weights > 0).sum()), number, weights))
        if not eligible:
            raise RuntimeError(f'No validation-qualified dual winner for{fold}')
        winner = max(eligible, key=lambda item: item[:4])
        weights = winner[-1]
        selections.append({'fold': fold, 'candidate': winner[3], 'weights': dict(zip(COMPONENTS, weights)), 'validation_score': winner[0]})
        paths, baseline, frames = load_paths(fold, 'test')
        nav = paths @ weights
        result, drawdown = metrics(nav)
        base_return, base_drawdown = metrics(baseline.nav.to_numpy())
        cash = np.column_stack([f.cash for f in frames]) @ weights
        pd.DataFrame({'day': baseline.day, 'nav': nav, 'baseline_nav': baseline.nav, 'cash': cash}).to_csv(OUT / f'nav_{fold}.csv', index=False)
        acceptance.append({'scope': f'research_{fold}', 'return': result, 'max_drawdown': drawdown,
                           'baseline_return': base_return, 'baseline_max_drawdown': base_drawdown,
                           'return_gain_pp': (result-base_return)*100, 'drawdown_reduction_pp': (base_drawdown-drawdown)*100,
                           'average_exposure': float(np.mean(1-cash/nav)),
                           'passed': result > base_return and drawdown < base_drawdown})
        if fold == '2026':
            final_weights = weights
    frames = [pd.read_parquet(SOURCE / f'production_nav_{name}.parquet') for name in COMPONENTS]
    baseline = pd.read_parquet(SOURCE / 'production_nav_baseline.parquet')
    for frame in frames:
        np.testing.assert_array_equal(frame.day, baseline.day)
    nav = np.column_stack([f.nav for f in frames]) @ final_weights
    cash = np.column_stack([f.cash for f in frames]) @ final_weights
    result, drawdown = metrics(nav)
    base_return, base_drawdown = metrics(baseline.nav.to_numpy())
    acceptance.append({'scope': 'production_score_2026', 'return': result, 'max_drawdown': drawdown,
                       'baseline_return': base_return, 'baseline_max_drawdown': base_drawdown,
                       'return_gain_pp': (result-base_return)*100, 'drawdown_reduction_pp': (base_drawdown-drawdown)*100,
                       'average_exposure': float(np.mean(1-cash/nav)), 'passed': result>base_return and drawdown<base_drawdown})
    pd.DataFrame({'day': baseline.day, 'nav': nav, 'baseline_nav': baseline.nav, 'cash': cash}).to_csv(OUT / 'nav_production_2026.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'validation_grid.csv', index=False)
    pd.DataFrame(acceptance).to_csv(OUT / 'acceptance.csv', index=False)
    (OUT / 'selection.json').write_text(json.dumps(selections, indent=2))
    decision = {'passed': all(row['passed'] for row in acceptance), 'weights': dict(zip(COMPONENTS, final_weights)),
                'acceptance': acceptance, 'source_sha256': previous.prior.digest(__file__)}
    (OUT / 'decision.json').write_text(json.dumps(decision, indent=2))
    print(json.dumps(selections, indent=2))
    print(pd.DataFrame(acceptance).to_string(index=False))


if __name__ == '__main__':
    main()
