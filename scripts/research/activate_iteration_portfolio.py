"""Publish user-authorized application replacement with an explicit rollback record."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY, build_default_daily_dependency_registry
from quant.application.portfolio_application import attach_portfolio, CONFIG
from quant.data.atomic_io import atomic_write_json, atomic_write_text
from quant.infrastructure.publication import PublicationStore
from quant.ml.portfolio_policy import PortfolioPolicy
from quant.ml.rank_fusion import file_sha256


def main():
    out = ROOT / 'reports/research/model_iteration_v2'
    decision = json.loads((out / 'deployment_decision.json').read_text())
    selection = json.loads((out / 'frozen_selection.json').read_text())
    assert decision['selection_sha256'] == file_sha256(out / 'frozen_selection.json')
    assert decision['no_holdout_reselection']
    version = decision['release'].rsplit('_', 1)[-1]
    if version == 'v1':
        assert all(r['return_better'] and r['drawdown_better'] for r in decision['acceptance'] if r['name'] == 'v1_frozen_combination')
    else:
        assert decision['new_candidate_passed_positive_dual_both_costs']
    for path, expected in selection['production_artifact_hashes'].items():
        assert file_sha256(Path(path)) == expected
    assert json.loads((out / 'policy_runtime_verification.json').read_text())['passed']
    parity = json.loads((out / 'fresh_runtime_parity.json').read_text())
    assert parity['v1_score_mismatches'] == parity['v1_priority_mismatches'] == 0
    preview = out / 'live_preview'
    provenance = json.loads((preview / 'provenance.json').read_text())
    assert provenance['status'] == 'success' and provenance['release_id'] == decision['release']
    assert file_sha256(preview / 'rankings.parquet') == provenance['rankings_sha256']
    assert file_sha256(preview / 'inputs.parquet') == provenance['inputs_sha256']
    candidate = ROOT / f'models/candidates/application_policy_{version}'
    destination = ROOT / 'models/production' / decision['release']
    if destination.exists():
        assert file_sha256(destination / 'policy.json') == file_sha256(candidate / 'policy.json')
    else:
        shutil.copytree(candidate, destination)
    policy = PortfolioPolicy.load(destination / 'policy.json', ROOT)
    inputs = pd.read_parquet(preview / 'inputs.parquet')
    ranked = pd.read_parquet(preview / 'rankings.parquet')
    pd.testing.assert_frame_equal(policy.predict(inputs), ranked)
    config = json.loads((ROOT / CONFIG).read_text())
    if config['enabled']:
        raise RuntimeError('An application policy is already active; do not overwrite another activation')
    rollback_text = (ROOT / CONFIG).read_text()
    config.update(enabled=True, release_id=decision['release'], policy=str((destination / 'policy.json').relative_to(ROOT)),
                  policy_sha256=file_sha256(destination / 'policy.json'), active_from=provenance['target_date'])
    managed = {p for n in DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes.values() for p in n.publication_outputs}
    managed.update(config[key] for key in ['rankings', 'inputs', 'manifest'])
    store = PublicationStore(ROOT, tuple(sorted(managed)))
    generation = 'portfolio-' + datetime.now().strftime('%Y%m%d%H%M%S')
    with store.begin(generation) as staged:
        pointer = json.loads((store.directory / 'current.json').read_text())
        if pointer['generation'] != provenance['source_generation']:
            raise RuntimeError('New daily generation published since preview; rerun preview first')
        for source, expected in provenance['input_score_hashes'].items():
            assert file_sha256(Path(source)) == expected
        for key, filename in [('rankings', 'rankings.parquet'), ('inputs', 'inputs.parquet'), ('manifest', 'manifest.json')]:
            target = staged.resolve(ROOT / config[key]); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preview / filename, target)
        directory = staged.resolve(ROOT / 'data/selector_snapshots')
        updated = []; historical = []; full = []
        for path in directory.glob('*.json'):
            payload = json.loads(path.read_text())
            if payload.get('signal_date') != config['active_from']:
                historical.append((path, file_sha256(path))); continue
            before = payload.get('stocks', [])
            complete = set((payload.get('snapshot_scope') or {}).get('strategies', [])) == {'ALL'}
            enriched = attach_portfolio(payload, config=config, require_full_coverage=complete)
            assert enriched['stocks'] == before
            if complete:
                assert all(s['eligible_count'] == len(s['candidates']) for s in enriched['application_portfolio']['sleeves'])
                full.append(path.name)
            atomic_write_json(enriched, path); updated.append(path.name)
        assert full and updated
        assert all(file_sha256(p) == expected for p, expected in historical)
        # This release changes only the application outputs and today's embedded
        # selector queues; the previous daily generation remains the data source.
        audit = {'status': 'success', 'target_trade_date': config['active_from'], 'refresh_node_ids': [],
            'freshness_audit': {'status': 'success', 'failures': [],
                'checked': ['score.application_portfolio', 'product.selector_extended'],
                'scope': 'application_release', 'unchanged_daily_source_generation': pointer['generation'],
                'candidate_date_exact': bool(pd.to_datetime(inputs.date).eq(config['active_from']).all()),
                'policy_and_input_identities_checked': True, 'ranking_replay_exact': True,
                'full_snapshot_candidate_coverage': True},
            'updated_snapshots': updated, 'historical_snapshot_count_unchanged': len(historical),
            'decision_sha256': file_sha256(out / 'deployment_decision.json')}
        assert audit['freshness_audit']['candidate_date_exact']
        rollback = ROOT / 'configs/releases/application_portfolio_disabled_rollback.json'
        atomic_write_text(rollback_text, rollback)
        atomic_write_json(pointer, out / 'publication_pointer_before_activation.json')
        try:
            atomic_write_json(config, ROOT / CONFIG)
            registry = build_default_daily_dependency_registry()
            assert 'score.application_portfolio' in registry.required_node_ids('short')
            from quant.routine.default_operations import build_default_operation_registry
            build_default_operation_registry(registry).validate_against_dependencies(registry)
            store.commit(staged, audit)
        except BaseException:
            atomic_write_text(rollback_text, ROOT / CONFIG)
            raise
    record = {'status': 'published', 'release_id': decision['release'], 'generation': generation,
        'signal_date': config['active_from'], 'previous_generation': pointer['generation'], 'policy_sha256': config['policy_sha256'],
        'config_sha256': file_sha256(ROOT / CONFIG), 'rollback_config': str(rollback.relative_to(ROOT)),
        'weights': policy.spec['weights'], 'updated_snapshots': len(updated), 'historical_snapshots_unchanged': len(historical),
        'user_authorized': True, 'live_orders_submitted': False, 'api_acceptance_pending': True}
    atomic_write_json(record, out / 'activation.json')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
