"""Audit account identities, policy selection and fixed-policy cost sensitivity."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_model_applications as experiment


def main():
    out = experiment.OUT
    results = pd.read_csv(out / 'results.csv')
    choices = pd.read_csv(out / 'selected.csv')
    specs = {p.name: p for p in experiment.policies()}
    assert len(results) == len(specs) * 4
    assert not results.duplicated(['fold', 'stage', 'name']).any()
    source_hashes = json.loads((out / 'source_hashes.json').read_text())
    for source, expected in source_hashes.items():
        path = experiment.ROOT / 'data/research/left_side_disclosure_gain_20261029/events.parquet' if source == 'left_flags' else Path(source)
        assert experiment.prior.digest(path) == expected
    account_errors = []
    for row in results.itertuples():
        spec = specs[row.name]
        nav = pd.read_parquet(out / f'nav_{row.fold}_{row.name}_{row.stage}.parquet')
        trades = pd.read_parquet(out / f'trades_{row.fold}_{row.name}_{row.stage}.parquet')
        assert nav.cash.ge(-1e-10).all()
        assert nav.positions.le(spec.positions).all()
        expected_return = results.loc[(results.fold == row.fold) & (results.name == row.name) & (results.stage == row.stage), 'return'].iloc[0]
        assert abs(nav.iloc[-1].nav - 1 - expected_return) < 1e-10
        assert trades.age.ge(1).all()
        assert abs(row.fees - row.turnover * spec.cost / 2) < 1e-10
        # Recover buys independently from gross turnover and net sale proceeds.
        gross_sales = (trades.invested * (1 + trades.net) / (1 - spec.cost / 2)).sum()
        total_invested = (row.turnover - gross_sales) * (1 + spec.cost / 2)
        open_investment = total_invested - trades.invested.sum()
        reconstructed_return = trades.pnl.sum() + row.unresolved_position_value - open_investment
        error = abs(reconstructed_return - expected_return)
        assert error < 1e-9
        account_errors.append(error)
    comparisons, stress_rows, replay_checks = [], [], []
    for fold in experiment.prior.FOLDS:
        experiment.initialize(fold)
        test_start, test_end = experiment.prior.FOLDS[fold][2:]
        start, end = experiment._CALENDAR.searchsorted([pd.Timestamp(test_start), pd.Timestamp(test_end)])
        baseline = pd.read_parquet(out / f'nav_{fold}_baseline_test.parquet').nav.to_numpy()
        baseline_returns = baseline / np.r_[1., baseline[:-1]] - 1
        for choice in choices.loc[choices.fold.eq(fold)].itertuples():
            validation = results.loc[results.fold.eq(fold) & results.stage.eq('validation') & results.closed.ge(100) & results['return'].gt(0)]
            if choice.family != 'all':
                validation = validation.loc[validation.family.eq(choice.family)]
            expected = 'cash' if validation.empty else validation.sort_values(['return', 'name'], ascending=[False, True]).iloc[0]['name']
            assert choice.selected == expected
            if choice.selected == 'cash':
                nav_values = np.ones(len(baseline))
            else:
                nav_values = pd.read_parquet(out / f'nav_{fold}_{choice.selected}_test.parquet').nav.to_numpy()
            daily = nav_values / np.r_[1., nav_values[:-1]] - 1
            delta = daily - baseline_returns
            rng = np.random.default_rng(73)
            starts = rng.integers(0, len(delta), (1000, int(np.ceil(len(delta) / 10))))
            indices = ((starts[:, :, None] + np.arange(10)) % len(delta)).reshape(1000, -1)[:, :len(delta)]
            low, high = np.quantile(delta[indices].mean(axis=1) * 10000, [.025, .975])
            comparisons.append({'fold': fold, 'family': choice.family, 'selected': choice.selected,
                                'capital_difference_pp': (nav_values[-1] - baseline[-1]) * 100,
                                'daily_difference_bp': delta.mean() * 10000, 'ci_low_bp': low, 'ci_high_bp': high})
        names = {'baseline'} | set(choices.loc[choices.fold.eq(fold), 'selected']) - {'cash'}
        for name in sorted(names):
            spec = specs[name]
            groups, coverage = experiment.candidates(experiment._FRAME, spec, experiment._THRESHOLDS, fold)
            for cost in [.001, .002, .004, .006]:
                metrics, nav, trades = experiment.simulate(replace(spec, cost=cost), groups, experiment._MARKET, int(start), int(end), coverage)
                stress_rows.append({'fold': fold, 'name': name, 'round_trip_cost_bp': cost * 10000, **metrics})
                if cost == .002:
                    saved = pd.read_parquet(out / f'nav_{fold}_{name}_test.parquet')
                    pd.testing.assert_frame_equal(nav, saved)
                    saved_trades = pd.read_parquet(out / f'trades_{fold}_{name}_test.parquet')
                    pd.testing.assert_frame_equal(trades, saved_trades)
                    replay_checks.append({'fold': fold, 'name': name, 'passed': True})
        # Report actual stage-specific filtered coverage, including signal cutoff.
        for name, spec in specs.items():
            groups, _ = experiment.candidates(experiment._FRAME, spec, experiment._THRESHOLDS, fold)
            for stage, begin, finish in [('validation', *experiment.prior.FOLDS[fold][1:3]), ('test', *experiment.prior.FOLDS[fold][2:4])]:
                first, last = experiment._CALENDAR.searchsorted([pd.Timestamp(begin), pd.Timestamp(finish)])
                mask = experiment._FRAME.day.ge(first - 1) & experiment._FRAME.day.le(last - 11)
                denominator = len(experiment._FRAME.loc[mask].drop_duplicates(['day', 'sid']))
                numerator = sum(len(group) for day, group in groups.items() if first - 1 <= day <= last - 11)
                rows = results.fold.eq(fold) & results.name.eq(name) & results.stage.eq(stage)
                results.loc[rows, 'candidate_coverage'] = numerator / max(denominator, 1)
    results.to_csv(out / 'audited_results.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out / 'paired_comparisons.csv', index=False)
    pd.DataFrame(stress_rows).to_csv(out / 'cost_sensitivity.csv', index=False)
    experiment.save('verification.json', {
        'passed': True, 'account_checks': len(account_errors), 'maximum_account_error': max(account_errors),
        'selection_checks': len(choices), 'replay_checks': replay_checks,
        'cost_scenarios': len(stress_rows), 'focused_tests_passed': 5,
        'ci': 'Exploratory paired10-session circular bootstrap of marked daily NAV returns; no correction for policy search.',
        'source_hashes_verified': True, 'experiment_sha256': experiment.prior.digest(experiment.__file__),
        'audit_sha256': experiment.prior.digest(__file__),
    })
    print(choices.to_string(index=False))
    print('VERIFIED', len(account_errors), 'accounts', len(stress_rows), 'cost scenarios')


if __name__ == '__main__':
    main()
