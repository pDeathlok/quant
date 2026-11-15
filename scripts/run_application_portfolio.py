"""Write three independent candidate rankings for replay or daily shadow use.

This command publishes no selector replacement and submits no trade orders.
Use --latest to materialize auxiliary inputs from the existing local sources.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.ml.application_portfolio import ApplicationPortfolio
from quant.ml.rank_fusion import RankFusion, file_sha256
from quant.routine.right_side_unified_production import _normalize_ranking_scores


def run(args):
    manifest_path = ROOT / 'models/candidates/application_portfolio_v1/manifest.json'
    release = json.loads((ROOT / 'configs/releases/application_portfolio_v1.json').read_text())
    if file_sha256(manifest_path) != release['manifest_sha256']:
        raise ValueError('Release manifest changed')
    spec = json.loads(manifest_path.read_text())
    for path, expected in spec['base_models'].items():
        if file_sha256(ROOT / path) != expected:
            raise ValueError(f'Base model changed; release revalidation required: {path}')
    runtime = ApplicationPortfolio.load(manifest_path)
    date = pd.Timestamp(args.date).normalize()
    right_path = ROOT / 'models/production/right_side_unified_canonical_v2/ranking.joblib'
    left_path = ROOT / 'models/production/left_side_ranknet_v6/manifest.json'
    right_artifact = joblib.load(right_path)
    left_model = RankFusion.load(left_path)
    hashes, pieces, materialization = {}, [], None
    if args.latest:
        from quant.application.selector_ranking import load_selector_ranking_config
        from quant.routine.rank_feature_runtime import materialize_rank_features
        paths = {'left': ROOT / 'data/features/left_side_ranknet_v6/latest_scores.parquet',
                 'right': load_selector_ranking_config(ROOT).paths.score_output}
        for side, path in paths.items():
            hashes[str(path)] = file_sha256(path)
            scores = pd.read_parquet(path)
            if set(pd.to_datetime(scores.date).dt.normalize()) != {date}:
                raise ValueError(f'{side} published scores are not exactly for {date.date()}')
            expected_hash = file_sha256(left_path if side == 'left' else right_path)
            if set(scores.model_artifact_sha256) != {expected_hash}:
                raise ValueError(f'{side} published model identity differs from pinned release')
            base = scores.ranking_score_base.to_numpy()
            if side == 'right':
                base = _normalize_ranking_scores(base, right_artifact['score_normalization']) / 100
            pieces.append(scores[['symbol', 'date']].assign(side=side, base_score=base))
        candidates = pd.concat(pieces, ignore_index=True)
        features, materialization = materialize_rank_features(str(date.date()), sorted(candidates.symbol.unique()),
            sorted(set(runtime.required_features('left') + runtime.required_features('right'))),
            root=ROOT, cache_root=ROOT / 'data/features/rank_runtime', workers=args.workers)
        candidates = candidates.merge(features, on=['symbol', 'date'], how='left', validate='many_to_one', indicator=True)
        if not candidates['_merge'].eq('both').all():
            raise ValueError('Missing materialized candidate rows')
        candidates = candidates.drop(columns='_merge')
    else:
        if not args.left_features or not args.right_features:
            raise ValueError('Provide both --left-features and --right-features, or --latest')
        for side, path in [('left', args.left_features), ('right', args.right_features)]:
            hashes[str(path)] = file_sha256(path)
            source = pd.read_parquet(path)
            source = source.loc[pd.to_datetime(source.date).dt.normalize().eq(date)].copy()
            if source.empty:
                raise ValueError(f'No {side} candidates on requested date')
            missing = set(runtime.required_features(side)) - set(source)
            if missing:
                raise ValueError(f'Missing {side} features: {sorted(missing)}')
            if side == 'left':
                base = left_model.predict(source).ranking_score.to_numpy()
            else:
                probabilities = right_artifact['model'].predict_proba(source)[:, 1]
                base = _normalize_ranking_scores(probabilities, right_artifact['score_normalization']) / 100
            pieces.append(source[['symbol', 'date', *runtime.required_features(side)]].assign(side=side, base_score=base))
        candidates = pd.concat(pieces, ignore_index=True)
    ranked = runtime.predict(candidates)
    for path, expected in hashes.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError('Input changed during inference; retry from a stable snapshot')
    output = args.output or ROOT / f'reports/research/dual_objective_portfolio_v1/shadow/{date:%Y%m%d}'
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(ranked, output / 'rankings.parquet', index=False)
    atomic_write_parquet(candidates, output / 'inputs.parquet', index=False)
    summary = {'release_id': runtime.release_id, 'date': str(date.date()),
               'mode': 'daily_shadow' if args.latest else 'historical_inference',
               'input_candidates': len(candidates), 'unique_symbols': candidates.symbol.nunique(),
               'rankings': len(ranked), 'weights': runtime.weights, 'input_sha256': hashes,
               'rankings_sha256': file_sha256(output / 'rankings.parquet'),
               'manifest_sha256': file_sha256(manifest_path),
               'features_materialized': materialization is not None,
               'feature_manifest': materialization,
               'execution': spec['account'],
               'interpretation': 'Independent sleeve priority queues, not order quantities. Existing holdings must be excluded before taking up to20 new candidates per sleeve. No existing account or live selector was modified.',
               'base_score_contract': 'Pinned left fusion score and right canonical CDF of unmodified base probability; ETF overlays excluded to match evaluated policy.'}
    atomic_write_json(summary, output / 'manifest.json')
    print(json.dumps({k: v for k, v in summary.items() if k != 'feature_manifest'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    parser.add_argument('--latest', action='store_true')
    parser.add_argument('--left-features', type=Path)
    parser.add_argument('--right-features', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    run(parser.parse_args())
