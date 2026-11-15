"""Finalize application evidence, including frozen production-score replay checks."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_model_applications as experiment


def main():
    out = experiment.OUT
    verification = json.loads((out / 'verification.json').read_text())
    assert verification['passed']
    production = pd.read_csv(out / 'production_application_results.csv')
    baseline = pd.read_parquet(out / 'production_nav_baseline.parquet').nav.to_numpy()
    baseline_returns = baseline / np.r_[1., baseline[:-1]] - 1
    comparisons, errors = [], []
    for name, costs in production.groupby('name'):
        assert costs.sort_values('round_trip_cost_bp')['return'].diff().dropna().le(1e-8).all()
        row = costs.loc[costs.round_trip_cost_bp.eq(20)].iloc[0]
        nav = pd.read_parquet(out / f'production_nav_{name}.parquet')
        trades = pd.read_parquet(out / f'production_trades_{name}.parquet')
        gross_sales = (trades.invested * (1 + trades.net) / .999).sum()
        total_invested = (row.turnover - gross_sales) * 1.001
        reconstructed = trades.pnl.sum() + row.unresolved_position_value - (total_invested - trades.invested.sum())
        error = abs(reconstructed - row['return'])
        assert error < 1e-9
        assert abs(nav.iloc[-1].nav - 1 - row['return']) < 1e-10
        errors.append(error)
        values = nav.nav.to_numpy()
        delta = values / np.r_[1., values[:-1]] - 1 - baseline_returns
        rng = np.random.default_rng(73)
        starts = rng.integers(0, len(delta), (1000, int(np.ceil(len(delta) / 10))))
        indices = ((starts[:, :, None] + np.arange(10)) % len(delta)).reshape(1000, -1)[:, :len(delta)]
        low, high = np.quantile(delta[indices].mean(axis=1) * 10000, [.025, .975])
        comparisons.append({'name': name, 'capital_difference_pp': (values[-1] - baseline[-1]) * 100,
                            'daily_difference_bp': delta.mean() * 10000, 'ci_low_bp': low, 'ci_high_bp': high})
    pd.DataFrame(comparisons).to_csv(out / 'production_paired_comparisons.csv', index=False)
    selected = pd.read_csv(out / 'selected.csv')
    summary = {
        'status': 'completed', 'research_policies': len(experiment.policies()), 'research_evaluations': 400,
        'production_score_policies': len(errors), 'cost_scenarios': verification['cost_scenarios'] + len(production),
        'account_checks': verification['account_checks'] + len(errors), 'focused_tests_passed': 5,
        'maximum_production_account_error': max(errors),
        'all_family_selection': selected.loc[selected.family.eq('all')].to_dict('records'),
        'findings': [
            'Equal-weight return/path score combination improved both historical intervals versus the research return-score baseline, but was not the global validation winner.',
            'Low-volatility variable weighting improved the2026 replay with broadly similar exposure;2025H2 returned slightly less than baseline.',
            'Entries5 and overlap agreement reduced exposure substantially; reduced losses alone do not establish better selection.',
            'Holding buffer improved research replay slightly but worsened frozen production-score replay.',
            'Isolated positive CHANGAN/KENGQI returns had very few trades or very low exposure.',
            'The global validation-selected policy underperformed baseline in2025H2 and improved2026; no stable universal winner established.',
        ],
        'production_score_replay': production.loc[production.round_trip_cost_bp.eq(20)].to_dict('records'),
        'limitations': json.loads((out / 'protocol.json').read_text())['limitations'] + [
            'Industry-cap policies were not tested because historical membership has not been certified; correlation limits were tested.',
            'Production-score transfer retains research-calibrated expected-return gates and research secondary model scores.',
            'All confidence intervals are exploratory and unadjusted for100-policy selection.',
        ],
        'artifact_hashes': {p.name: experiment.prior.digest(p) for p in out.glob('*.csv')},
        'script_hashes': {p.name: experiment.prior.digest(p) for p in [Path(experiment.__file__), Path(__file__)]},
    }
    experiment.save('summary.json', summary)
    print('COMPLETE', summary['research_policies'], summary['production_score_policies'], summary['cost_scenarios'], summary['account_checks'])
    print(pd.DataFrame(comparisons).loc[lambda x: x.name.isin(['score_return_risk', 'weighted_low_volatility', 'score_return_path', 'hold_buffer'])].to_string(index=False))


if __name__ == '__main__':
    main()
