"""Activate the user-authorized release only between publication generations."""
from __future__ import annotations
import fcntl
import json
import sys
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from quant.data.atomic_io import atomic_write_json,atomic_write_text
from quant.ml.rank_fusion import RankFusion,file_sha256


def main() -> None:
    output=ROOT/'reports/research/left_ranknet_rollout'
    folder=ROOT/'models/production/left_side_ranknet_v6'
    manifest_path=folder/'manifest.json'
    candidate=ROOT/'configs/releases/left_side_ranknet_v6.yaml'
    active=ROOT/'configs/strategies/left_side_unified.yaml'
    rollback=ROOT/'configs/releases/left_side_unified_v4_rollback.yaml'
    for path in [folder/'parity.json',output/'staged_dag_audit.json',output/'live_2026-11-10.json',
                 ROOT/'reports/research/rank_runtime_parity/source_vintage_audit.json']:
        if not json.loads(path.read_text()).get('passed'):
            raise ValueError(f'Unsuccessful release prerequisite: {path}')
    RankFusion.load(manifest_path)
    lock=ROOT/'data/publications/writer.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('An existing daily publication is running; do not interrupt it')
        if active.read_bytes() not in [rollback.read_bytes(),candidate.read_bytes()]:
            raise RuntimeError('Active configuration changed outside this rollout')
        old_hash=file_sha256(manifest_path)
        manifest=json.loads(manifest_path.read_text())
        manifest.update(lifecycle='production',rollback_config=str(rollback.relative_to(ROOT)))
        atomic_write_json(manifest,manifest_path)
        parity=json.loads((folder/'parity.json').read_text())
        parity.setdefault('staged_manifest_sha256',old_hash)
        parity.update(manifest_sha256=file_sha256(manifest_path),production_pointer_changed=True)
        atomic_write_json(parity,folder/'parity.json')
        atomic_write_text(candidate.read_text(),active)
        record=dict(status='configured',release_id=manifest['release_id'],configured_at=datetime.now().isoformat(),
                    config_sha256=file_sha256(active),manifest_sha256=file_sha256(manifest_path),
                    rollback_config=str(rollback.relative_to(ROOT)),
                    live_publication_pending=True,user_authorized=True)
        atomic_write_json(record,output/'activation.json')
        print(json.dumps(record,indent=2))

if __name__=='__main__':main()
