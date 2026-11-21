"""Production adapter for the separately versioned left RankNet fusion."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant.application.left_side_ranking import LeftSideRankingConfig, load_left_side_ranking_scores
from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.ml.rank_fusion import RankFusion, file_sha256
from quant.routine.rank_feature_runtime import materialize_production_rank_features


def run_left_ranknet(target_date: str, *, config: LeftSideRankingConfig, root: Path) -> dict[str,Any]:
    from quant.routine.left_side_unified_production import load_left_side_signal_universe, normalize_daily_percentile
    from quant.routine.short_etf_rank import rank_with_etf, model_identity, market_context_identity
    from quant.research.left_side_unified_features import LEFT_SIDE_RULE_FEATURE_COLUMNS
    target=pd.Timestamp(target_date).normalize();date=target.date().isoformat()
    model=RankFusion.load(config.paths.artifact_manifest)
    signals=load_left_side_signal_universe(target,config=config)
    compatibility={c:f'selector_legacy__{c}' for c in LEFT_SIDE_RULE_FEATURE_COLUMNS}
    features,feature_cache=materialize_production_rank_features(date,signals.symbol.tolist(),
        [*model.required_features,*compatibility.values()],
        root=root,workers=config.factor_workers)
    feature_frame=features.merge(signals,on=['symbol','date'],validate='one_to_one').copy()
    feature_frame['ts_code']=feature_frame.symbol;feature_frame['trade_date']=target.strftime('%Y%m%d')
    identity=dict(features=feature_cache['output_sha256'],release=file_sha256(config.paths.artifact_manifest),
                  signals=hashlib.sha256(pd.util.hash_pandas_object(signals,index=False).values.tobytes()).hexdigest(),
                  etf_model=model_identity(),etf_context=market_context_identity(date),
                  adapter=file_sha256(Path(__file__)),score_schema=config.score_schema_version)
    identity['scoring_implementation']={str(path.relative_to(Path(__file__).resolve().parents[2])):file_sha256(path)
        for path in [Path(__file__).resolve().parents[1]/'ml/rank_fusion.py',
                     Path(__file__).with_name('left_side_unified_production.py'),
                     Path(__file__).with_name('short_etf_rank.py'),
                     Path(__file__).resolve().parents[1]/'features/short_etf_rank.py']}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()
    if all(path.exists() for path in [config.paths.score_manifest,config.paths.score_output,config.paths.feature_manifest,config.paths.feature_output]):
        existing=json.loads(config.paths.score_manifest.read_text())
        previous_features=json.loads(config.paths.feature_manifest.read_text())
        if (existing.get('source_input_fingerprint')==fingerprint and existing.get('output_sha256')==file_sha256(config.paths.score_output)
                and previous_features.get('output_sha256')==file_sha256(config.paths.feature_output)):
            load_left_side_ranking_scores(date,config=config)
            return dict(status='success',target_date=date,checkpoint_reused=True,feature_checkpoint_reused=feature_cache['cache_hit'],
                        candidate_count=existing['candidate_count'],score=existing,
                        feature=json.loads(config.paths.feature_manifest.read_text()),adapter={'status':'success','target_date':date})
    feature_manifest=dict(status='success',target_date=date,candidate_coverage_status='complete',
        signal_candidate_count=len(signals),computed_candidate_count=len(feature_frame),factor_count=len(model.required_features),
        factor_contract_sha256=config.factor_contract_sha256,source_input_fingerprint=feature_cache['contract']['input_identity'],
        features=list(model.required_features),selector_compatibility_features=compatibility,
        policy_excluded_candidate_symbols=[],runtime_cache=feature_cache,
        created_at=datetime.now().isoformat(timespec='seconds'))
    atomic_write_parquet(feature_frame,config.paths.feature_output,index=False)
    feature_manifest['output_sha256']=file_sha256(config.paths.feature_output)
    atomic_write_json(feature_manifest,config.paths.feature_manifest)
    scores=model.predict(feature_frame)
    scores=scores.merge(signals,on=['symbol','date'],validate='one_to_one')
    scores['ts_code']=scores.symbol;scores['trade_date']=target.strftime('%Y%m%d')
    scores['ranking_score_base']=scores.ranking_score
    artifact_sha=file_sha256(config.paths.artifact)
    adjusted,etf=rank_with_etf('left',feature_frame,scores.ranking_score.to_numpy(),date,artifact_sha,project_root=root)
    scores['ranking_score']=adjusted;scores['ranking_score_normalized']=normalize_daily_percentile(adjusted)
    scores['model_artifact_sha256']=artifact_sha
    excluded=sorted(scores.loc[~scores.trade_gate_open,'symbol'].tolist())
    # Preserve rejected scores for auditing; only admitted candidates reach the
    # selector, using the existing explicit policy-exclusion contract.
    atomic_write_parquet(scores,config.paths.score_output.with_name('all_ranknet_scores.parquet'),index=False)
    accepted=scores.loc[scores.trade_gate_open].reset_index(drop=True)
    atomic_write_parquet(accepted,config.paths.score_output,index=False)
    manifest=dict(status='success',target_date=date,schema_version=config.score_schema_version,release_id=model.release_id,
        artifact_schema_version='rank-fusion-v1',artifact_sha256=artifact_sha,factor_contract_sha256=config.factor_contract_sha256,
        source_input_fingerprint=fingerprint,candidate_count=len(accepted),all_candidate_count=len(scores),
        policy_excluded_candidate_symbols=excluded,policy_excluded_candidate_count=len(excluded),
        gate_excluded_candidate_symbols=excluded,minimum_expected_net_return=model.minimum_expected_net_return,
        score_field=config.score_field,normalized_score_field=config.normalized_score_field,
        score_normalization={'method':'stable_average_rank','range':[0,100]},
        production_threshold_mode='none_rank_only' if model.minimum_expected_net_return is None else 'calibrated_net_return',
        selection_policy='downstream_top_n_after_cash_gate',selector_adapter_status='ready',
        shadow_v5={'status':'not_applicable','reason':'shadow was trained against the previous v4 model'},etf_rank=etf,
        created_at=datetime.now().isoformat(timespec='seconds'),output_sha256=file_sha256(config.paths.score_output))
    atomic_write_json(manifest,config.paths.score_manifest)
    load_left_side_ranking_scores(date,config=config)
    return dict(status='success',target_date=date,checkpoint_reused=False,feature_checkpoint_reused=feature_cache['cache_hit'],
                candidate_count=len(accepted),feature=feature_manifest,score=manifest,adapter={'status':'success','target_date':date})
