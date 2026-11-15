"""Stage the validated left fusion, proving compact inference on every test row."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from quant.ml.rank_fusion import RankFusion, compact_booster, file_sha256
from quant.research.local_ranknet import percentile_scores

SOURCE = ROOT/'reports/research/full_rank_comparison_v2'
OUTPUT = ROOT/'models/production/left_side_ranknet_v6'


def main() -> None:
    verification = json.loads((SOURCE/'verification.json').read_text())
    if not verification['passed']:
        raise ValueError('Research verification is required')
    choices = json.loads((SOURCE/'choices_left_2026.json').read_text())
    if choices['ranknet_family_return'] != 'local10_all@0.5#-inf':
        raise ValueError('Approved research candidate changed')
    meta = json.loads((SOURCE/'dataset_complete.json').read_text())
    if file_sha256(SOURCE/'dataset_left.parquet') != meta['dataset_hashes']['left']:
        raise ValueError('Research data hash changed')
    if (OUTPUT/'manifest.json').exists() and json.loads((OUTPUT/'manifest.json').read_text()).get('lifecycle') == 'production':
        raise RuntimeError('Refusing to overwrite the active production release')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calibrators = json.loads((SOURCE/'calibrators_left_2026.json').read_text())
    frame = pd.read_parquet(SOURCE/'dataset_left.parquet')
    frame = frame.loc[frame.date.ge('2026-01-01')].sort_values(['date', 'symbol']).reset_index(drop=True)
    contract = json.loads((SOURCE/'contract_left_2026.json').read_text())
    required: set[str] = set()
    expected_score = np.zeros(len(frame)); expected_return = np.zeros(len(frame))
    specs = []
    for arm in ['base50', 'local10_all']:
        source_path = SOURCE/f'model_left_2026_{arm}.ubj'
        model = xgb.Booster(); model.load_model(source_path)
        compact = compact_booster(model)
        before = model.predict(xgb.DMatrix(frame[model.feature_names].astype('float32'), nthread=4))
        after = compact.predict(xgb.DMatrix(frame[compact.feature_names].astype('float32'), nthread=4))
        np.testing.assert_array_equal(before, after)
        path = OUTPUT/f'{arm}.ubj'; compact.save_model(path)
        calibration = {key: calibrators[arm][key] for key in ['x', 'y']}
        specs.append(dict(file=path.name, sha256=file_sha256(path), weight=.5, features=compact.feature_names,
                          source_model=str(source_path.relative_to(ROOT)), source_sha256=file_sha256(source_path),
                          original_inputs=len(model.feature_names), calibration=calibration))
        required.update(compact.feature_names)
        expected_score += .5*percentile_scores(frame.date, before)
        expected_return += .5*np.interp(before, calibration['x'], calibration['y'])
    manifest = dict(schema_version='rank-fusion-v1', release_id='left-ranknet-v6-20261111', lifecycle='staged',
        components=specs, required_features=sorted(required), minimum_expected_net_return=None,
        score_semantics='blend of within-signal-date raw component percentiles; not a probability',
        calibrated_score_semantics='net return including historical 20bp costs; calibration clips outside observed range',
        training_fit_end=contract['fit_end'], calibration_end=contract['select_start'],
        policy_selection_end=contract['test_start'], research_signal_end=str(frame.date.max()),
        user_authorized_replacement=True, rollback_config='configs/releases/left_side_unified_v4_rollback.yaml',
        rollback_artifact='models/production/left_side_unified_canonical_v4_group4/ranking.joblib',
        research_dataset_sha256=meta['dataset_hashes']['left'],
        published_validation_rows=len(frame), feature_builder_version='rank-feature-runtime-v1',
        daily_input_timing='signal close T; entry next close T+1',
        not_refitted_after_evaluation=True)
    manifest_path=OUTPUT/'manifest.json'; manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    result=RankFusion.load(manifest_path).predict(frame)
    np.testing.assert_array_equal(result.ranking_score,expected_score)
    np.testing.assert_allclose(result.expected_net_return,expected_return,rtol=1e-12,atol=1e-12)
    if not result.trade_gate_open.all():
        raise ValueError('Frozen rank-only threshold changed')
    report=dict(passed=True,rows=len(frame),required_features=len(required),component_inputs={s['file']:len(s['features']) for s in specs},
        exact_raw_prediction_parity=True,exact_fused_rank_parity=True,gate_parity=True,
        manifest_sha256=file_sha256(manifest_path),production_pointer_changed=False)
    (OUTPUT/'parity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
