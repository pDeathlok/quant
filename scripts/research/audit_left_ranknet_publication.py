"""Read-only acceptance of the committed daily generation and left replacement."""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY
from quant.application.left_side_ranking import load_left_side_ranking_config, load_left_side_ranking_scores
from quant.infrastructure.publication import PublicationStore
from quant.ml.rank_fusion import RankFusion, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True)
    args = parser.parse_args()
    target = pd.Timestamp(args.date).date().isoformat()
    status = json.loads((ROOT / 'data/routine/latest_refresh_status.json').read_text())
    assert status['status'] == 'success', status['status']
    assert pd.Timestamp(status['trade_date']).date().isoformat() == target
    pointer = json.loads((ROOT / 'data/publications/current.json').read_text())
    assert pointer['target_trade_date'] == target
    managed = {path for node in DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes.values() for path in node.publication_outputs}
    view = PublicationStore(ROOT, tuple(sorted(managed))).view()
    config = load_left_side_ranking_config()
    config = replace(config, paths=replace(config.paths, **{
        field.name: view.resolve(getattr(config.paths, field.name)) for field in fields(config.paths)
    }))
    assert config.release_id == 'left-ranknet-v6-20261111' and config.engine == 'rank_fusion'
    for path in [config.paths.feature_output, config.paths.feature_manifest, config.paths.score_output, config.paths.score_manifest]:
        assert path.is_file(), path
    scores, score_manifest = load_left_side_ranking_scores(target, config=config)
    features = pd.read_parquet(config.paths.feature_output)
    assert pd.to_datetime(features.date).eq(target).all()
    model = RankFusion.load(config.paths.artifact_manifest)
    expected = model.predict(features)
    assert set(scores) == set(expected.loc[expected.trade_gate_open, 'symbol'])
    actual = pd.read_parquet(config.paths.score_output).set_index('symbol').sort_index()
    expected = expected.loc[expected.trade_gate_open].set_index('symbol').sort_index()
    np.testing.assert_array_equal(actual.ranking_score_base, expected.ranking_score)
    np.testing.assert_array_equal(actual.expected_net_return, expected.expected_net_return)
    assert len(model.required_features) == 152
    if len(features):
        assert not features[list(model.required_features)].isna().all().any()
    feature_manifest=json.loads(config.paths.feature_manifest.read_text())
    compatibility=feature_manifest['selector_compatibility_features']
    assert len(compatibility)==27 and set(compatibility.values()).issubset(features)
    before_path=ROOT/'reports/research/left_ranknet_rollout/selector_compatibility_repair/before_latest_features.parquet'
    before=pd.read_parquet(before_path).set_index('symbol').sort_index()
    after=features.set_index('symbol').sort_index()
    pd.testing.assert_frame_equal(before[list(model.required_features)],after[list(model.required_features)])
    legacy=pd.read_parquet(before_path.with_name('consumer_preflight_features.parquet')).set_index('symbol').sort_index()
    np.testing.assert_array_equal(after[list(compatibility.values())].to_numpy(dtype='float32'),
                                  legacy[list(compatibility.values())].to_numpy(dtype='float32'))
    # Intermediate feature/score caches are mutable. The immutable page snapshots
    # are the public contract; inspect their embedded model identity and scores.
    snapshot_dir = view.resolve(ROOT / 'data/selector_snapshots')
    published_symbols = set()
    checked_snapshots = []
    right_hashes = set()
    artifact_sha = file_sha256(config.paths.artifact_manifest)
    for path in snapshot_dir.glob('*.json'):
        payload = json.loads(path.read_text())
        if payload.get('signal_date') != target:
            continue
        checked_snapshots.append(path.name)
        for row in payload.get('stocks', []):
            if row.get('ranking_source') == 'right_side_unified':
                right_hashes.add(row['ranking_model_artifact_sha256'])
            elif row.get('ranking_source') == 'left_side_unified':
                assert row['ranking_model_artifact_sha256'] == artifact_sha
                raw, normalized = scores[row['symbol']]
                assert np.isclose(row['ranking_score'], raw)
                assert np.isclose(row['ranking_score_normalized'], normalized)
                published_symbols.add(row['symbol'])
    assert checked_snapshots
    if len(scores):
        assert published_symbols
    report = dict(passed=True, date=target, generation=view.generation, run_id=status['run_id'],
                  release_id=config.release_id, feature_count=len(model.required_features),
                  candidates=len(features), accepted=len(scores), score_sha256=score_manifest['output_sha256'],
                  artifact_sha256=artifact_sha, published_left_symbols=sorted(published_symbols),
                  auxiliary_selector_columns=len(compatibility), rank_inputs_unchanged_by_compatibility_repair=True,
                  auxiliary_columns_match_independent_original_calculator=True,
                  checked_selector_snapshots=checked_snapshots, right_artifact_sha256=sorted(right_hashes),
                  etf_overlay=score_manifest['etf_rank'],
                  daily_steps=[{key: step.get(key) for key in ['key', 'status', 'elapsed_seconds', 'checkpoint_reused']}
                               for step in status['steps']])
    output = ROOT / f'reports/research/left_ranknet_rollout/published_{target}.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
