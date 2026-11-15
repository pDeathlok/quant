"""Verify a declaration lookup refactor and revalidate identical cached outputs.

This one-time migration never bypasses a changed source/output check. Only the
read_pinned_parquet declaration lookup may change; hashing and verified reads,
all mathematical code, model files, inputs and output checksums must be equal.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')
sys.path.insert(0, str(ROOT / 'src'))
from quant.ml.rank_fusion import file_sha256
from quant.routine.checkpoint_store import CheckpointStore
from quant.routine.default_operations import CORE_OPERATION_INPUTS, DEFAULT_DAILY_OPERATION_REGISTRY
from quant.routine.operation_contracts import OperationContext

OUT = ROOT / 'reports/research/left_ranknet_rollout/snapshot_lookup_optimization'
SOURCE = 'src/quant/data/market_snapshot.py'
BEFORE = Path('/tmp/quant_snapshot_lookup_before.py')


def main() -> None:
    status = json.loads((ROOT / 'data/routine/latest_refresh_status.json').read_text())
    assert status['status'] in {'success', 'failed'}
    assert status['trade_date'].replace('-', '') == '20261111'
    old_tree, new_tree = [ast.parse(p.read_text()) for p in [BEFORE, ROOT / SOURCE]]
    def unchanged(tree):
        return [ast.dump(n, include_attributes=False) for n in tree.body
                if not isinstance(n, ast.FunctionDef) or n.name != 'read_pinned_parquet']
    assert unchanged(old_tree) == unchanged(new_tree)
    assert ast.dump(old_tree) != ast.dump(new_tree)
    auxiliary_changes = {
        'src/quant/routine/rank_feature_runtime.py': ('/tmp/quant_before_selector_rank_feature_runtime.py',
                                                     {'_stock_task', '_materialize_rank_features'}),
        'src/quant/routine/left_ranknet_production.py': ('/tmp/quant_before_selector_left_ranknet_production.py',
                                                       {'run_left_ranknet'}),
        'src/quant/webapp/services.py': ('/tmp/quant_before_selector_services.py',
                                       {'_selector_production_snapshot_rows'}),
    }
    for relative, (before_path, allowed) in auxiliary_changes.items():
        def other_nodes(path):
            return [ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if not isinstance(n, ast.FunctionDef) or n.name not in allowed]
        assert other_nodes(Path(before_path)) == other_nodes(ROOT / relative), relative
    slim = json.loads(Path('/tmp/quant_snapshot_paths.json').read_text())
    root = Path('/sealed-lookup-equivalence').absolute()
    def prepare(tree):
        reader = SimpleNamespace(root=root, manifest=slim,
                                 _chunk_path=lambda chunk: root / chunk['path'])
        namespace = dict(Path=Path, pd=SimpleNamespace(DataFrame=object),
                         current_market_snapshot=lambda: reader, MarketSnapshotError=ValueError,
                         _read_verified_parquet=lambda path, sha, **kwargs: (str(path), sha, kwargs))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'read_pinned_parquet')
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<lookup-audit>', 'exec'), namespace)
        return namespace['read_pinned_parquet'], reader
    old_lookup, _ = prepare(old_tree)
    new_lookup, reader = prepare(new_tree)
    expected = {}
    for entry in slim['datasets'].values():
        for chunk in entry['files']:
            expected.setdefault(root / chunk['path'], chunk['sha256'])
    paths = list(expected)
    probes = [paths[i * (len(paths) - 1) // 31] for i in range(32)]
    results, times = [], []
    for lookup in [old_lookup, new_lookup]:
        start = perf_counter()
        results.append([lookup(path, columns=['ts_code']) for path in probes])
        times.append(perf_counter() - start)
    assert results[0] == results[1]
    for path, sha in expected.items():
        assert new_lookup(path) == (str(path), sha, {})
    for lookup in [old_lookup, new_lookup]:
        try:
            lookup(root / 'not-declared.parquet')
        except ValueError:
            pass
        else:
            raise AssertionError('Undeclared file accepted')
    assert len(reader._declared_file_index) == len(expected)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'before_market_snapshot.py').write_bytes(BEFORE.read_bytes())
    source_before, source_after = file_sha256(BEFORE), file_sha256(ROOT / SOURCE)
    store = CheckpointStore(ROOT, ROOT / 'data/cache/routine_operations')
    checks = []
    prepared = []
    # Validate every checkpoint before mutating any metadata.
    for name in CORE_OPERATION_INPUTS:
        if name == 'run_left_side_unified':
            continue  # Its extra selector compatibility columns MUST be recomputed.
        definition = DEFAULT_DAILY_OPERATION_REGISTRY.definitions[name]
        path = store._checkpoint_path(name)
        saved = json.loads(path.read_text()); old = saved['identity_payload']
        assert not any('left_side' in node for node in old['input_ids'])
        assert old['contract_hashes'][SOURCE] == source_before
        context = OperationContext(old['target_trade_date'], 'all', 1, {},
            upstream_fingerprints=old['upstream_fingerprints'], upstream_revisions=old['upstream_revisions'],
            parameters=old['parameters'], required_input_ids=tuple(old['input_ids']), project_root=ROOT)
        identity, new = store.build_identity(definition, context)
        assert {k for k in new if new[k] != old.get(k)} == {'contract_hashes'}
        changed = {k for k in set(new['contract_hashes']) | set(old['contract_hashes'])
                   if new['contract_hashes'].get(k) != old['contract_hashes'].get(k)}
        assert SOURCE in changed and changed <= {SOURCE, *auxiliary_changes}, (name, changed)
        for relative in changed - {SOURCE}:
            assert old['contract_hashes'][relative] == file_sha256(Path(auxiliary_changes[relative][0]))
        result = store.load(definition, saved['identity'])
        assert result is not None and result.status == 'success', name
        backup = OUT / f'{name}_before.json'
        assert not backup.exists(), 'Single-use migration'
        prepared.append((definition, path, backup, identity, new, result, saved))
    for definition, path, backup, identity, new, result, saved in prepared:
        backup.write_bytes(path.read_bytes())
        store.save(definition, identity, new, replace(result, contract_identity=identity,
            metrics={**result.metrics, 'snapshot_lookup_only_revalidated': True}))
        assert store.load(definition, identity) is not None
        checks.append(dict(operation=definition.operation_id, old_identity=saved['identity'],
                           new_identity=identity, unchanged_outputs=saved['output_hashes']))
    report = dict(passed=True, source=SOURCE, old_source_sha256=source_before, new_source_sha256=source_after,
                  changed_function='read_pinned_parquet', unchanged_other_ast=True,
                  source_snapshot_fingerprint=slim['source_fingerprint'], declared_files=len(expected),
                  exact_lookup_samples=len(probes), all_index_entries_verified=True,
                  old_lookup_seconds=times[0], new_lookup_seconds_including_index_build=times[1],
                  timing_scope='Declaration lookup only; not complete data reads or daily refresh.',
                  cache_checks=checks, left_cache_revalidated=False,
                  isolated_auxiliary_changes={p:sorted(v[1]) for p,v in auxiliary_changes.items()},
                  limitation='Revalidated existing identical inputs/outputs; no second full source acquisition or training run.')
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'cache_checks'}, indent=2))


if __name__ == '__main__':
    main()
