"""Rebuild historical rows with the production calculator, without labels."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from quant.routine.rank_feature_runtime import materialize_rank_features
from quant.ml.rank_fusion import RankFusion


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--date',default='2026-06-30');parser.add_argument('--limit',type=int,default=6)
    args=parser.parse_args();folder=ROOT/'models/production/left_side_ranknet_v6'
    model=RankFusion.load(folder/'manifest.json')
    columns=['symbol','date',*model.required_features]
    original=pd.read_parquet(ROOT/'reports/research/full_rank_comparison_v2/dataset_left.parquet',columns=columns)
    original=original.loc[original.date.eq(args.date)].sort_values('symbol').head(args.limit).reset_index(drop=True)
    if original.empty:raise ValueError('No comparison candidates')
    actual,manifest=materialize_rank_features(args.date,original.symbol.tolist(),list(model.required_features),root=ROOT,
        cache_root=ROOT/'data/features/rank_runtime',workers=4,index_root=ROOT/'reports/research/right_t012_index_sources_v1',require_complete_context=False)
    actual=actual.sort_values('symbol').reset_index(drop=True)
    errors=[]
    for c in model.required_features:
        a=original[c].to_numpy();b=actual[c].to_numpy();match=np.isclose(a,b,rtol=1e-5,atol=1e-6,equal_nan=True)
        if not match.all():
            errors.append(dict(feature=c,rows=int((~match).sum()),original=a[~match].tolist(),runtime=b[~match].tolist()))
    report=dict(date=args.date,rows=len(actual),feature_count=len(model.required_features),mismatches=errors,passed=not errors,cache=manifest['cache_hit'])
    output=ROOT/'reports/research/rank_runtime_parity';output.mkdir(parents=True,exist_ok=True)
    (output/f'{args.date}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
