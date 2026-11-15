from dataclasses import replace
import hashlib
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from quant.ml.rank_fusion import file_sha256
from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG,load_left_side_ranking_scores


def test_cash_gate_excludes_all_rows_and_repeated_score_reuses_checkpoint(tmp_path,monkeypatch):
    from quant.routine import left_ranknet_production as live,left_side_unified_production as old,short_etf_rank as etf
    features=pd.DataFrame({'symbol':['000001.SZ','000002.SZ'],'date':pd.Timestamp('2026-11-10'),'signal':[-1.,1.]})
    model=xgb.train({'nthread':1,'max_depth':1},xgb.DMatrix(features[['signal']],label=[-.1,.1]),num_boost_round=2)
    artifact=tmp_path/'component.ubj';model.save_model(artifact)
    release=tmp_path/'manifest.json'
    release.write_text(json.dumps(dict(schema_version='rank-fusion-v1',release_id='cash-test',required_features=['signal'],
        minimum_expected_net_return=.2,components=[dict(file=artifact.name,sha256=file_sha256(artifact),features=['signal'],
            weight=1.,calibration=dict(x=[-1.,1.],y=[-.1,.1]))])))
    paths=replace(DEFAULT_LEFT_SIDE_RANKING_CONFIG.paths,artifact=release,artifact_manifest=release,
        feature_output=tmp_path/'features.parquet',feature_manifest=tmp_path/'features.json',
        score_output=tmp_path/'scores.parquet',score_manifest=tmp_path/'scores.json')
    config=replace(DEFAULT_LEFT_SIDE_RANKING_CONFIG,paths=paths,engine='rank_fusion',release_id='cash-test',
        factor_contract_sha256=hashlib.sha256(json.dumps(['signal']).encode()).hexdigest(),score_schema_version='left-side-rank-fusion-score-v1')
    signals=features[['symbol','date']].assign(B1=True,SB1=False,SUPER_B1=False,LOW_PULLBACK=False)
    monkeypatch.setattr(old,'load_left_side_signal_universe',lambda *a,**k:signals)
    cache={'output_sha256':'fixed-source','cache_hit':True,'contract':{'input_identity':'fixed'}}
    monkeypatch.setattr(live,'materialize_rank_features',lambda *a,**k:(features,cache))
    monkeypatch.setattr(etf,'model_identity',lambda:None)
    monkeypatch.setattr(etf,'market_context_identity',lambda _: 'fixed')
    monkeypatch.setattr(etf,'rank_with_etf',lambda side,frame,base,*a,**k:(base,{'status':'not_admitted'}))
    first=live.run_left_ranknet('2026-11-10',config=config,root=tmp_path)
    assert first['candidate_count']==0 and first['score']['all_candidate_count']==2
    assert first['score']['policy_excluded_candidate_symbols']==features.symbol.tolist()
    scores,manifest=load_left_side_ranking_scores('2026-11-10',config=config)
    assert scores=={} and manifest['production_threshold_mode']=='calibrated_net_return'
    second=live.run_left_ranknet('2026-11-10',config=config,root=tmp_path)
    assert second['checkpoint_reused']
    # Changing the release invalidates the prior score identity even if features
    # remain unchanged; a gate change must become observable to the selector.
    payload=json.loads(release.read_text());payload['minimum_expected_net_return']=None
    release.write_text(json.dumps(payload))
    third=live.run_left_ranknet('2026-11-10',config=config,root=tmp_path)
    assert not third['checkpoint_reused'] and third['candidate_count']==2
