"""Verify real completed checkpoints can serve short/all with identical inputs.

Read-only: reuse the completed sealed input identities, rehash current contracts
and materialized outputs, and call the production cache loader for both scopes.
This does not pretend a new source acquisition or provider request occurred.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')
sys.path.insert(0, str(ROOT / 'src'))
from quant.routine.checkpoint_store import CheckpointStore
from quant.routine.default_operations import DEFAULT_DAILY_OPERATION_REGISTRY, CORE_OPERATION_INPUTS
from quant.routine.operation_contracts import OperationContext


def main() -> None:
    status = json.loads((ROOT / 'data/routine/latest_refresh_status.json').read_text())
    assert status['status'] == 'success', status['status']
    store = CheckpointStore(ROOT, ROOT / 'data/cache/routine_operations')
    records = []
    for name in CORE_OPERATION_INPUTS:
        definition = DEFAULT_DAILY_OPERATION_REGISTRY.definitions[name]
        saved = json.loads(store._checkpoint_path(name).read_text())
        previous = saved['identity_payload']
        assert previous['target_trade_date'].replace('-', '') == status['trade_date'].replace('-', '')
        ids = []
        for scope in ['all', 'short']:
            context = OperationContext(target_trade_date=previous['target_trade_date'], scope=scope, granted_workers=1,
                                       upstream_results={}, upstream_fingerprints=previous['upstream_fingerprints'],
                                       upstream_revisions=previous['upstream_revisions'], parameters=previous['parameters'],
                                       required_input_ids=tuple(previous['input_ids']), project_root=ROOT)
            identity, _ = store.build_identity(definition, context)
            assert identity == saved['identity'], (name, scope, 'current contract drift')
            loaded = store.load(definition, identity)
            assert loaded is not None and loaded.status == 'success', (name, scope, 'invalid cache outputs')
            ids.append(identity)
        assert ids[0] == ids[1]
        records.append(dict(operation=name, all_hit=True, short_hit=True, identity=ids[0]))
    output = dict(passed=True, trade_date=status['trade_date'], checks=records,
                  limitation='Same verified sealed source identities; this is a real cache-loader audit, not a second full daily refresh.')
    path = ROOT / 'reports/research/left_ranknet_rollout/core_scope_cache.json'
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
