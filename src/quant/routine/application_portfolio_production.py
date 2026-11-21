"""Daily materialization of the active independently funded portfolio queues."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from quant.application.portfolio_application import load_portfolio_config
from quant.application.left_side_ranking import load_left_side_ranking_config
from quant.application.selector_ranking import load_selector_ranking_config
from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.infrastructure.publication import publication_path
from quant.ml.portfolio_policy import PortfolioPolicy
from quant.ml.rank_fusion import file_sha256
from quant.routine.rank_feature_runtime import materialize_production_rank_features
from quant.routine.right_side_unified_production import _normalize_ranking_scores

ROOT = Path(__file__).resolve().parents[3]


def run_portfolio(target_date: str, *, root: Path = ROOT, workers: int = 4, config: dict | None = None) -> dict:
    config = config or load_portfolio_config(root)
    if not config['enabled']:
        raise ValueError('Application portfolio is not active')
    policy_path = root / config['policy']
    if file_sha256(policy_path) != config['policy_sha256']:
        raise ValueError('Policy changed before application scoring')
    policy = PortfolioPolicy.load(policy_path, root)
    target = pd.Timestamp(target_date).normalize()
    left = load_left_side_ranking_config(project_root=root)
    right = load_selector_ranking_config(root)
    scores = {'left': publication_path(left.paths.score_output), 'right': publication_path(right.paths.score_output)}
    hashes = {str(p): file_sha256(p) for p in scores.values()}
    right_model = joblib.load(right.paths.artifact)
    pieces = []
    for side, path in scores.items():
        source = pd.read_parquet(path)
        if not source.empty and set(pd.to_datetime(source.date).dt.normalize()) != {target}:
            raise ValueError(f'{side} base candidate snapshot is stale')
        expected = policy.spec['base_models'].get(str(left.paths.artifact_manifest if side == 'left' else right.paths.artifact))
        if expected is None or (not source.empty and set(source.model_artifact_sha256) != {expected}):
            raise ValueError(f'{side} base candidate model identity changed')
        base = source.ranking_score_base.to_numpy()
        if side == 'right':
            base = _normalize_ranking_scores(base, right_model['score_normalization']) / 100
        pieces.append(source[['symbol', 'date']].assign(side=side, base_score=base))
    candidates = pd.concat(pieces, ignore_index=True)
    required = sorted(set(policy.required_features('left') + policy.required_features('right')))
    features, feature_manifest = materialize_production_rank_features(
        str(target.date()), sorted(candidates.symbol.unique()), required, root=root, workers=workers,
    )
    candidates = candidates.merge(features.copy(), on=['symbol', 'date'], how='left', validate='many_to_one', indicator=True)
    if not candidates['_merge'].eq('both').all():
        raise RuntimeError('Application feature coverage incomplete')
    candidates = candidates.drop(columns='_merge')
    ranked = policy.predict(candidates)
    if any(file_sha256(Path(p)) != expected for p, expected in hashes.items()):
        raise RuntimeError('Base candidates changed during portfolio scoring')
    if file_sha256(policy_path) != config['policy_sha256']:
        raise RuntimeError('Policy changed during portfolio scoring')
    output = {key: publication_path(root / config[key]) for key in ['rankings', 'inputs', 'manifest']}
    for path in output.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(candidates, output['inputs'], index=False)
    atomic_write_parquet(ranked, output['rankings'], index=False)
    conditions = {name: {'expected_net_minimum': strategy['expected_net_gate'],
                         'market_breadth_minimum': strategy['market_breadth_gate']}
                  for name, strategy in policy.spec.get('strategies', {}).items()}
    result = {'status': 'success', 'target_date': str(target.date()), 'release_id': policy.spec['release_id'],
              'policy_sha256': config['policy_sha256'], 'rankings_sha256': file_sha256(output['rankings']),
              'inputs_sha256': file_sha256(output['inputs']), 'input_score_hashes': hashes,
              'candidate_rows': len(candidates), 'unique_symbols': int(candidates.symbol.nunique()),
              'ranking_rows': len(ranked), 'weights': policy.spec['weights'], 'labels': policy.spec['labels'],
              'conditions': conditions, 'account': policy.spec['account'],
              'feature_manifest': feature_manifest, 'candidate_coverage_status': 'complete'}
    atomic_write_json(result, output['manifest'])
    return result


def run_operation(context):
    from quant.routine.operation_adapters import result_from_payload
    result = run_portfolio(context.target_trade_date, root=context.project_root or ROOT, workers=context.granted_workers)
    return result_from_payload(result, ('score.application_portfolio',), context)
