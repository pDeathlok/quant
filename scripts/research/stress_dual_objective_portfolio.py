"""Re-run frozen sleeve weights under costs; no test-period re-selection."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_model_applications as application
from validate_production_applications import initialize_production
from select_dual_objective_portfolio import OUT, metrics


def bootstrap(nav, baseline, seed=812, repetitions=2000):
    """Paired circular 10-session blocks; descriptive, not selection-adjusted."""
    returns = np.column_stack([nav / np.r_[1., nav[:-1]] - 1,
                               baseline / np.r_[1., baseline[:-1]] - 1])
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(returns), size=(len(returns) + 9) // 10)
        indexes = ((starts[:, None] + np.arange(10)) % len(returns)).ravel()[:len(returns)]
        paths = np.cumprod(1 + returns[indexes], axis=0)
        result, risk = metrics(paths[:, 0])
        base_result, base_risk = metrics(paths[:, 1])
        differences.append([result - base_result, base_risk - risk])
    differences = np.asarray(differences)
    return {'return_gain_95pct_interval': np.quantile(differences[:, 0], [.025, .975]).tolist(),
            'drawdown_reduction_95pct_interval': np.quantile(differences[:, 1], [.025, .975]).tolist(),
            'dual_positive_fraction': float((differences > 0).all(axis=1).mean()),
            'repetitions': repetitions, 'block_sessions': 10,
            'interpretation': 'Exploratory resampling, not a probability of future outperformance. No adjustment for policy/weight search.'}


def main():
    selections = json.loads((OUT / 'selection.json').read_text())
    specs = {p.name: p for p in application.policies()}
    rows, intervals, hashes, account_errors = [], {}, {}, []
    for scope in ['research_2025H2', 'research_2026', 'production_score_2026']:
        fold = '2025H2' if scope.endswith('2025H2') else '2026'
        if scope.startswith('production'):
            frame, hashes = initialize_production()
        else:
            application.initialize(fold)
            frame = application._FRAME
        weights = next(s['weights'] for s in selections if s['fold'] == fold)
        weights = {k: v for k, v in weights.items() if v > 0}
        groups = {name: application.candidates(frame, specs[name], application._THRESHOLDS, fold)
                  for name in ['baseline', *weights]}
        bounds = application.prior.FOLDS[fold]
        start, end = application._CALENDAR.searchsorted(pd.to_datetime(bounds[2:4]))
        for cost in [.001, .002, .004, .006]:
            navs, cash, results = {}, {}, {}
            for name, (candidates, coverage) in groups.items():
                result, path, trades = application.simulate(replace(specs[name], cost=cost), candidates,
                    application._MARKET, int(start), int(end), coverage)
                # Check the terminal cash/marked-position decomposition.
                assert abs(path.nav.iloc[-1] - path.cash.iloc[-1] - result['unresolved_position_value']) < 1e-9
                gross_sales = (trades.invested * (1 + trades.net) / (1 - cost / 2)).sum()
                total_invested = (result['turnover'] - gross_sales) * (1 + cost / 2)
                reconstructed = trades.pnl.sum() + result['unresolved_position_value'] - (total_invested - trades.invested.sum())
                account_errors.append(abs(reconstructed - result['return']))
                assert account_errors[-1] < 1e-9
                navs[name], cash[name], results[name] = path.nav.to_numpy(), path.cash.to_numpy(), result
            nav = sum(weights[name] * navs[name] for name in weights)
            portfolio_cash = sum(weights[name] * cash[name] for name in weights)
            result, risk = metrics(nav)
            base_result, base_risk = metrics(navs['baseline'])
            stress_return = sum(weights[n] * results[n]['total_loss_stress_return'] for n in weights)
            row = {'scope': scope, 'round_trip_cost_bp': cost * 10000, 'return': result, 'max_drawdown': risk,
                   'baseline_return': base_result, 'baseline_max_drawdown': base_risk,
                   'return_gain_pp': 100 * (result - base_result), 'drawdown_reduction_pp': 100 * (base_risk - risk),
                   'average_exposure': float(np.mean(1 - portfolio_cash / nav)),
                   'unresolved_zero_return': stress_return,
                   'baseline_unresolved_zero_return': results['baseline']['total_loss_stress_return'],
                   'dual_pass': result > base_result and risk < base_risk}
            rows.append(row)
            pd.DataFrame({'day': path.day, 'nav': nav, 'baseline_nav': navs['baseline'],
                          'cash': portfolio_cash}).to_csv(OUT / f'stress_nav_{scope}_{int(cost * 10000)}bp.csv', index=False)
            if cost == .002:
                original = pd.read_csv(OUT / ('nav_production_2026.csv' if scope.startswith('production') else f'nav_{fold}.csv'))
                np.testing.assert_allclose(nav, original.nav, atol=1e-12, rtol=0)
                intervals[scope] = bootstrap(nav, navs['baseline'])
            print(scope, cost, row['dual_pass'], flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'cost_stress.csv', index=False)
    (OUT / 'bootstrap.json').write_text(json.dumps(intervals, indent=2))
    (OUT / 'stress_verification.json').write_text(json.dumps({
        'all_cost_scenarios_dual_pass': all(r['dual_pass'] for r in rows), 'scenarios': len(rows),
        'base_cost_nav_replays_exact': True, 'production_artifact_hashes': hashes,
        'independent_account_checks': len(account_errors), 'maximum_account_error': max(account_errors),
        'script_sha256': application.prior.digest(__file__),
    }, indent=2))


if __name__ == '__main__':
    main()
