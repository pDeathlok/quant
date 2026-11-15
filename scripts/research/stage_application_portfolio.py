"""Freeze the selected multi-sleeve candidate and verify full historical inference."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from quant.ml.application_portfolio import ApplicationPortfolio, combine_sleeve_nav
from quant.ml.rank_fusion import compact_booster, file_sha256
import validate_model_applications as application
from validate_production_applications import initialize_production
from select_dual_objective_portfolio import OUT


def main():
    decision = json.loads((OUT / 'decision.json').read_text())
    stress = json.loads((OUT / 'stress_verification.json').read_text())
    if not decision['passed'] or not stress['all_cost_scenarios_dual_pass']:
        raise RuntimeError('Historical dual-objective acceptance failed')
    folder = ROOT / 'models/candidates/application_portfolio_v1'
    folder.mkdir(parents=True, exist_ok=True)
    weights = {k: v for k, v in decision['weights'].items() if v > 0}
    components, inputs, checks = [], [], []
    frame, production_hashes = initialize_production()
    frame = frame.loc[frame.date.ge('2025-12-31')].copy()
    for side in ['left', 'right']:
        source = pd.read_parquet(application.prior.OUT / f'dataset_{side}.parquet')
        source = source.loc[source.date.ge('2025-12-31')].copy()
        for target in ['path_boundary', 'rank']:
            original_path = application.prior.OUT / f'model_{side}_2026_{target}.ubj'
            original = xgb.Booster()
            original.load_model(original_path)
            compact = compact_booster(original)
            output = folder / f'{side}_{target}.ubj'
            compact.save_model(output)
            raw = original.predict(application.prior.matrix(source, original.feature_names))
            reduced = compact.predict(application.prior.matrix(source, compact.feature_names))
            np.testing.assert_array_equal(raw, reduced)
            components.append({'side': side, 'target': target, 'file': output.name,
                               'sha256': file_sha256(output), 'features': compact.feature_names,
                               'training_source': str(original_path.relative_to(ROOT)),
                               'training_source_sha256': file_sha256(original_path)})
            checks.append({'side': side, 'target': target, 'rows': len(source),
                           'compaction_exact': True, 'features': len(compact.feature_names)})
        base = frame.loc[frame.side.eq(side), ['symbol', 'date', 's_return']].rename(columns={'s_return': 'base_score'})
        source = source.merge(base, on=['symbol', 'date'], validate='one_to_one').assign(side=side)
        inputs.append(source)
    manifest = {
        'schema_version': 'application-portfolio-v1', 'release_id': 'application_portfolio_v1',
        'status': 'research_candidate', 'weights': weights, 'auxiliary_models': components,
        'base_models': {str(Path(p).relative_to(ROOT)): sha for p, sha in production_hashes.items()},
        'account': {'allocation': 'Independent initial capital sleeves; no daily redistribution',
                    'max_positions_per_sleeve': 120, 'max_daily_entry_attempts_per_sleeve': 20,
                    'holding_market_sessions': 5, 'entry': 'next_session_close', 'overlap': 'right_side_precedence',
                    'default_round_trip_cost_bp': 20, 'unfilled_entry': 'skip_without_replacement',
                    'unfilled_exit': 'retain_and_retry', 'shares': 'fractional_research_account'},
        'validation': {'weights_selected_on': '2025-10-01 through 2025-12-31 research validation',
                       'acceptance_report': str(OUT.relative_to(ROOT) / 'decision.json'),
                       'acceptance_sha256': file_sha256(OUT / 'decision.json'),
                       'cost_stress_sha256': file_sha256(OUT / 'cost_stress.csv')},
        'activation': {'default_selector_changed': False,
                       'reason': 'Historical research inputs verified. Current daily feature parity and unseen forward performance not certified.'},
    }
    manifest_path = folder / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    runtime = ApplicationPortfolio.load(manifest_path)
    actual = runtime.predict(pd.concat(inputs, ignore_index=True))
    symbols = json.loads((application.OUT / 'market_index.json').read_text())['symbols']
    specs = {p.name: p for p in application.policies()}
    for name in weights:
        groups, _ = application.candidates(frame, specs[name], application._THRESHOLDS, '2026')
        reference = pd.DataFrame([{'date': application._CALENDAR[day], 'symbol': symbols[int(row.sid)],
                                   'score': row.score, 'side': row.side, 'priority': priority}
                                  for day, group in groups.items() for priority, row in enumerate(group, 1)])
        reference['date'] = reference.date.astype('datetime64[ns]')
        key = ['date', 'symbol', 'side']
        reference = reference.sort_values(key).reset_index(drop=True)
        result = actual.loc[actual.sleeve.eq(name)].sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(result[key], reference[key])
        np.testing.assert_allclose(result.score, reference.score, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(result.priority, reference.priority)
        checks.append({'sleeve': name, 'candidate_rankings': len(result), 'historical_rank_parity': True})
    paths = {name: pd.read_parquet(application.OUT / f'production_nav_{name}.parquet').set_index('day').nav for name in weights}
    nav = combine_sleeve_nav(paths, weights)
    original_nav = pd.read_csv(OUT / 'nav_production_2026.csv').set_index('day').nav
    np.testing.assert_allclose(nav, original_nav, atol=1e-14, rtol=0)
    date = pd.Timestamp('2026-08-21')
    actual.loc[actual.date.eq(date)].to_parquet(OUT / 'inference_example_20260821.parquet', index=False)
    verification = {'passed': True, 'checks': checks, 'capital_account_aggregation_parity': True,
                    'manifest_sha256': file_sha256(manifest_path), 'stage_script_sha256': file_sha256(Path(__file__)),
                    'runtime_sha256': file_sha256(ROOT / 'src/quant/ml/application_portfolio.py'),
                    'runner_sha256': file_sha256(ROOT / 'scripts/run_application_portfolio.py')}
    (OUT / 'release_verification.json').write_text(json.dumps(verification, indent=2))
    config = ROOT / 'configs/releases/application_portfolio_v1.json'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({'release_id': runtime.release_id, 'mode': 'research_candidate',
                                 'manifest': str(manifest_path.relative_to(ROOT)),
                                 'manifest_sha256': file_sha256(manifest_path), 'weights': weights,
                                 'verification': str((OUT / 'release_verification.json').relative_to(ROOT))}, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
