"""Check complete candidate coverage and a no-work repeat of a release run."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from time import monotonic
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from quant.application.left_side_ranking import load_left_side_ranking_config,load_left_side_ranking_scores
from quant.routine.left_side_unified_production import run_left_side_production
from quant.ml.rank_fusion import RankFusion,file_sha256


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--date',required=True)
    parser.add_argument('--config',default='configs/releases/left_side_ranknet_v6.yaml')
    args=parser.parse_args();config=load_left_side_ranking_config(config_path=args.config)
    times=[];runs=[]
    for _ in range(2):
        started=monotonic();result=run_left_side_production(args.date,config=config)
        times.append(round(monotonic()-started,3));runs.append({k:result[k] for k in ['candidate_count','checkpoint_reused','feature_checkpoint_reused']})
    assert runs[-1]['checkpoint_reused'] and runs[-1]['feature_checkpoint_reused']
    frame=pd.read_parquet(config.paths.feature_output);scores,_=load_left_side_ranking_scores(args.date,config=config)
    model=RankFusion.load(config.paths.artifact_manifest);expected=model.predict(frame)
    assert set(scores)==set(expected.loc[expected.trade_gate_open,'symbol'])
    actual=pd.read_parquet(config.paths.score_output).set_index('symbol').sort_index()
    expected=expected.loc[expected.trade_gate_open].set_index('symbol').sort_index()
    np.testing.assert_array_equal(actual.ranking_score_base,expected.ranking_score)
    manifest=json.loads(config.paths.score_manifest.read_text())
    assert manifest['output_sha256']==file_sha256(config.paths.score_output)
    report=dict(passed=True,date=args.date,release_id=config.release_id,runs=runs,seconds=times,
                actual_feature_count=len(model.required_features),all_candidate_rows=len(frame),
                etf_overlay=manifest['etf_rank'],source_complete=True,
                score_sha256=manifest['output_sha256'],manifest_sha256=file_sha256(config.paths.artifact_manifest))
    out=ROOT/'reports/research/left_ranknet_rollout';out.mkdir(exist_ok=True)
    (out/f'live_{args.date}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
