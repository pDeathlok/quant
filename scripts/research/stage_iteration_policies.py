"""Prepare reviewable application releases; do not activate before holdout decision."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from quant.ml.portfolio_policy import PortfolioPolicy
from quant.ml.rank_fusion import file_sha256
from model_iteration_v2_protocol import OUT, BOUNDS
from select_iteration_applications import score_arm
from validate_production_applications import initialize_production
import validate_model_applications as app

ACCOUNT = {'allocation': 'independent_initial_capital', 'positions_per_sleeve': 120,
           'daily_entry_attempts_per_sleeve': 20, 'holding_market_sessions': 5,
           'entry': 'next_session_close', 'exit': 'close_after_five_market_sessions',
           'unfilled_entry': 'skip_without_replacement', 'unfilled_exit': 'retain_and_retry',
           'overlap': 'right_precedence_after_admission', 'daily_rebalancing_between_sleeves': False}


def main():
    choice = json.loads((OUT / 'frozen_selection.json').read_text())
    folder = ROOT / 'models/candidates/application_policy_v1'
    folder.mkdir(parents=True, exist_ok=True)
    legacy = folder / 'legacy'
    legacy.mkdir(exist_ok=True)
    for source in (ROOT / 'models/candidates/application_portfolio_v1').glob('*'):
        if source.suffix in ['.json', '.ubj']:
            shutil.copy2(source, legacy / source.name)
    spec = {'schema_version': 'portfolio-policy-v1', 'release_id': 'application_portfolio_v1',
            'engine': 'legacy_combination', 'legacy_manifest': 'legacy/manifest.json',
            'weights': choice['fallback']['weights'], 'account': ACCOUNT,
            'labels': {'score_three_way': '三模型融合', 'score_return_path': '模型与路径融合',
                       'weighted_low_volatility': '模型与低波动'},
            'files': {str(p.relative_to(folder)): file_sha256(p) for p in legacy.glob('*')},
            'base_models': choice['production_artifact_hashes']}
    (folder / 'policy.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    PortfolioPolicy.load(folder / 'policy.json', ROOT)
    if not choice['candidate']:
        return
    folder = ROOT / 'models/candidates/application_policy_v2'
    folder.mkdir(parents=True, exist_ok=True)
    strategies, files = {}, {}
    for name, recipe in choice['recipes'].items():
        models = {}
        arm = recipe['arm']
        for side in ['left', 'right']:
            source = json.loads((OUT / f'models/{side}_{arm}.json').read_text())
            neural = arm in ['mlp', 'tabm', 'tabm_multitask']
            filenames = [f'{side}_{arm}.torchscript', source['preprocess']] if neural else [source['models'][0]['file']]
            for filename in filenames:
                shutil.copy2(OUT / 'models' / filename, folder / filename)
                files[filename] = file_sha256(folder / filename)
            models[side] = {'engine': 'torchscript' if neural else 'xgboost', 'file': filenames[0],
                            'features': source['features'], 'calibration': source['calibration']}
            if neural:
                models[side]['preprocess'] = source['preprocess']
        strategies[name] = {'arm': arm, 'expected_net_gate': recipe['gate'],
                            'market_breadth_gate': recipe['market_gate'], 'models': models}
    spec = {'schema_version': 'portfolio-policy-v1', 'release_id': 'application_portfolio_v2',
            'engine': 'gated_models', 'weights': choice['candidate']['weights'], 'strategies': strategies,
            'labels': {name: '尾部风险模型' if recipe['arm'] == 'utility_tail' else 'TabM 模型' for name, recipe in choice['recipes'].items()},
            'neural_python': '.venv-kronos/bin/python', 'files': files, 'account': ACCOUNT,
            'base_models': choice['production_artifact_hashes'],
            'selection_sha256': file_sha256(OUT / 'frozen_selection.json')}
    (folder / 'policy.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    runtime = PortfolioPolicy.load(folder / 'policy.json', ROOT)
    source, _ = initialize_production()
    source = source.loc[source.date.ge(BOUNDS['calibration_end']) & source.date.lt(BOUNDS['selection_end'])].copy()
    parts = []
    for side in ['left', 'right']:
        frame = pd.read_parquet(OUT / f'development_{side}.parquet')
        frame = frame.loc[frame.date.ge(BOUNDS['calibration_end']) & frame.date.lt(BOUNDS['selection_end'])]
        parts.append(frame.assign(side=side))
    result = runtime.predict(pd.concat(parts, ignore_index=True))
    symbols = json.loads((app.OUT / 'market_index.json').read_text())['symbols']
    checks = []
    for name, recipe in choice['recipes'].items():
        predictions = pd.concat([pd.read_parquet(OUT / f"predictions_{side}_{recipe['arm']}.parquet").assign(side=side)
                                 for side in ['left', 'right']], ignore_index=True)
        data = score_arm(source, recipe['arm'], predictions[['symbol', 'date', 'side', 'raw_score', 'expected_net']])
        if recipe['market_gate'] is not None:
            data = data.loc[data.market_breadth_above_ma20_ratio.ge(recipe['market_gate'])]
        groups, _ = app.candidates(data, app.Policy(gate=-np.inf if recipe['gate'] is None else recipe['gate']), app._THRESHOLDS, '2026')
        reference = pd.DataFrame([{'symbol': symbols[int(r.sid)], 'date': app._CALENDAR[day],
                                  'side': r.side, 'score': r.score, 'priority': i + 1}
                                 for day, group in groups.items() for i, r in enumerate(group)])
        reference['date'] = reference.date.astype('datetime64[ns]')
        key = ['symbol', 'date', 'side']
        expected = reference.sort_values(key).reset_index(drop=True)
        actual = result.loc[result.sleeve.eq(name)].sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[key], expected[key])
        np.testing.assert_array_equal(actual.score, expected.score)
        np.testing.assert_array_equal(actual.priority, expected.priority)
        checks.append({'sleeve': name, 'rows': len(actual), 'admission_and_ranking_exact': True})
    (OUT / 'policy_runtime_verification.json').write_text(json.dumps({'passed': True, 'checks': checks,
        'v1_policy_sha256': file_sha256(ROOT / 'models/candidates/application_policy_v1/policy.json'),
        'v2_policy_sha256': file_sha256(folder / 'policy.json')}, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
