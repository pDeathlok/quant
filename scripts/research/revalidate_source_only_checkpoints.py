"""One-time audited cache migration after the empty-flow source validator fix.

Only signal/project calculators are eligible: neither consumes rank flows nor
executes source fetching. All their calculation code, parameters, input versions
and output bytes must be unchanged. The next DAG still checks fresh source
fingerprints; a data revision therefore remains a cache miss.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')
sys.path.insert(0, str(ROOT / 'src'))
from quant.ml.rank_fusion import file_sha256
from quant.routine.checkpoint_store import CheckpointStore
from quant.routine.default_operations import DEFAULT_DAILY_OPERATION_REGISTRY
from quant.routine.operation_contracts import OperationContext

OUT = ROOT / 'reports/research/left_ranknet_rollout/empty_flow_cache_repair'
SOURCE = 'src/quant/data/long_factor_backfill.py'


def main() -> None:
    assert json.loads((ROOT/'data/routine/latest_refresh_status.json').read_text())['status'] == 'failed'
    before = json.loads((OUT/'before.json').read_text())
    assert file_sha256(OUT/'before_long_factor_backfill.py') == before['sha256']
    allowed = {'_partition_valid', '_fetch_trade_date_partition'}
    def unchanged_nodes(path):
        return [ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if not isinstance(node, ast.FunctionDef) or node.name not in allowed]
    assert unchanged_nodes(ROOT/SOURCE) == unchanged_nodes(OUT/'before_long_factor_backfill.py')
    current_hash = file_sha256(ROOT/SOURCE)
    store = CheckpointStore(ROOT, ROOT/'data/cache/routine_operations')
    checks = []
    for name in ['refresh_strategy_signal_cache', 'refresh_active_project_features']:
        definition = DEFAULT_DAILY_OPERATION_REGISTRY.definitions[name]
        path = store._checkpoint_path(name)
        saved = json.loads(path.read_text()); old = saved['identity_payload']
        assert old['target_trade_date'] == '2026-11-11'
        assert 'data.rank_flows' not in old['input_ids']
        assert old['contract_hashes'][SOURCE] == before['sha256']
        context = OperationContext(old['target_trade_date'], 'all', 1, {},
                                   upstream_fingerprints=old['upstream_fingerprints'],
                                   upstream_revisions=old['upstream_revisions'], parameters=old['parameters'],
                                   required_input_ids=tuple(old['input_ids']), project_root=ROOT)
        identity, new = store.build_identity(definition, context)
        assert {key for key in new if new[key] != old.get(key)} == {'contract_hashes'}
        assert {key for key in set(new['contract_hashes']) | set(old['contract_hashes'])
                if new['contract_hashes'].get(key) != old['contract_hashes'].get(key)} == {SOURCE}
        result = store.load(definition, saved['identity'])
        assert result is not None and result.status == 'success', 'Changed outputs or invalid original lineage'
        backup = OUT/f'{name}_before.json'
        assert not backup.exists(), 'Migration is single use'
        backup.write_bytes(path.read_bytes())
        store.save(definition, identity, new, replace(result, contract_identity=identity,
            metrics={**result.metrics, 'source_validation_only_revalidated': True}))
        assert store.load(definition, identity) is not None
        checks.append(dict(operation=name, old_identity=saved['identity'], new_identity=identity,
                           unchanged_outputs=saved['output_hashes'], backup=str(backup.relative_to(ROOT))))
    report = dict(passed=True, changed_source=SOURCE, old_source_sha256=before['sha256'],
                  new_source_sha256=current_hash, changed_functions=sorted(allowed), checks=checks,
                  scope='Cache metadata only; no model predictions, input fingerprints or materialized output bytes changed.')
    (OUT/'checkpoint_revalidation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'passed': True, 'operations': [c['operation'] for c in checks]}))


if __name__ == '__main__':
    main()
