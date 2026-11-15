"""Revalidate unchanged non-left outputs after the measured I/O refactor.

The left feature producer must run again: its source enumeration and buffering
changed. Other operations retain identical mathematical code, inputs, models,
resource claims and outputs. Every metadata change is backed up and audited.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[2]
load_dotenv(ROOT/'.env')
sys.path.insert(0,str(ROOT/'src'))
from quant.ml.rank_fusion import file_sha256
from quant.routine.checkpoint_store import CheckpointStore
from quant.routine.default_operations import CORE_OPERATION_INPUTS,DEFAULT_DAILY_OPERATION_REGISTRY
from quant.routine.operation_contracts import OperationContext

OUT=ROOT/'reports/research/daily_refresh_performance'
ALLOWED={
    'src/quant/data/market_snapshot.py': {'_canonical_dates','read_pinned_parquet',
        'release_market_snapshot_reader','_capture_supplemental_dataset','export_market_snapshot'},
    'src/quant/routine/rank_feature_runtime.py': {'_read_candidate_partitions','_materialize_rank_features'},
    'src/quant/routine/production_dag.py': {'sealed_core_market_inputs'},
    'src/quant/routine/default_operations.py': set(),
}


def stable_ast(path: Path, allowed: set[str]) -> list[str]:
    nodes=[]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name in allowed:
            continue
        if isinstance(node,ast.Import) and [a.name for a in node.names]==['os']:
            continue
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_DIGEST_CACHE_SIZE' for t in node.targets):
            continue
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='_RESOURCE_PROFILES':
            assert isinstance(node.value,ast.Dict)
            pairs=[(k,v) for k,v in zip(node.value.keys,node.value.values)
                   if not isinstance(k,ast.Constant) or k.value!='run_left_side_unified']
            node.value.keys=[k for k,_ in pairs];node.value.values=[v for _,v in pairs]
        nodes.append(ast.dump(node,include_attributes=False))
    return nodes


def main() -> None:
    state=json.loads((ROOT/'data/routine/latest_refresh_status.json').read_text())
    assert state['status']=='success' and state['trade_date'].replace('-','')=='20261111'
    before=json.loads((OUT/'before_sources/before_hashes.json').read_text())
    for relative,allowed in ALLOWED.items():
        old=OUT/'before_sources'/Path(relative).name
        assert file_sha256(old)==before[relative]
        assert stable_ast(old,allowed)==stable_ast(ROOT/relative,allowed),relative
    store=CheckpointStore(ROOT,ROOT/'data/cache/routine_operations')
    prepared=[]
    for name in CORE_OPERATION_INPUTS:
        if name=='run_left_side_unified':
            continue
        definition=DEFAULT_DAILY_OPERATION_REGISTRY.definitions[name]
        path=store._checkpoint_path(name);saved=json.loads(path.read_text());old=saved['identity_payload']
        assert not any('left_side' in x or x=='data.rank_flows' for x in old['input_ids'])
        context=OperationContext(old['target_trade_date'],'all',1,{},
            upstream_fingerprints=old['upstream_fingerprints'],upstream_revisions=old['upstream_revisions'],
            parameters=old['parameters'],required_input_ids=tuple(old['input_ids']),project_root=ROOT)
        identity,new=store.build_identity(definition,context)
        assert {k for k in new if new[k]!=old.get(k)}=={'contract_hashes'},name
        changed={k for k in set(new['contract_hashes'])|set(old['contract_hashes'])
                 if new['contract_hashes'].get(k)!=old['contract_hashes'].get(k)}
        assert changed and changed<=set(ALLOWED),(name,changed)
        for relative in changed:
            assert old['contract_hashes'][relative]==before[relative]
        result=store.load(definition,saved['identity'])
        assert result is not None and result.status=='success',name
        backup=OUT/f'{name}_before_performance.json'
        assert not backup.exists(),'Single-use migration'
        prepared.append((definition,path,backup,saved,identity,new,result,changed))
    checks=[]
    for definition,path,backup,saved,identity,new,result,changed in prepared:
        backup.write_bytes(path.read_bytes())
        store.save(definition,identity,new,replace(result,contract_identity=identity,
            metrics={**result.metrics,'performance_only_revalidated':True}))
        assert store.load(definition,identity) is not None
        checks.append(dict(operation=definition.operation_id,old_identity=saved['identity'],new_identity=identity,
            changed_contract_files=sorted(changed),unchanged_output_checksums=saved['output_hashes']))
    report=dict(passed=True,left_cache_revalidated=False,checks=checks,
                reason='Only measured I/O/date conversion/bounded cache changes plus isolated left reads; no non-left formulas, parameters, inputs or outputs changed.',
                limitation='New daily source identities are still checked; this audit does not substitute for the next full refresh.')
    (OUT/'checkpoint_revalidation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':True,'operations':[c['operation'] for c in checks]},indent=2))


if __name__=='__main__':main()
