import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.application.portfolio_application import load_portfolio_config, project_portfolio
from quant.ml.portfolio_policy import PortfolioPolicy
from quant.ml.rank_fusion import file_sha256


def test_admission_precedes_overlap_and_filters_preserve_global_ranks(monkeypatch, tmp_path):
    spec = {'weights': {'test': 1.}, 'strategies': {'test': {
        'models': {s: {'features': ['raw'], 'calibration': {'x': [0., 1.], 'y': [0., 1.]}}
                   for s in ['left', 'right']}, 'expected_net_gate': .5, 'market_breadth_gate': .4}}}
    policy = PortfolioPolicy(tmp_path, tmp_path, spec, None)
    monkeypatch.setattr(policy, '_predict_model', lambda frame, spec: frame.raw.to_numpy())
    frame = pd.DataFrame({'symbol': ['A', 'B', 'A', 'C'], 'date': ['2026-11-11'] * 4,
        'side': ['left', 'left', 'right', 'right'], 'raw': [.8, .9, .1, .9],
        'volatility_20d': [.2] * 4, 'market_breadth_above_ma20_ratio': [.5] * 4})
    output = policy.predict(frame)
    assert output.symbol.tolist() == ['B', 'C', 'A']
    assert output.loc[output.symbol.eq('A'), 'side'].iloc[0] == 'left'
    assert output.loc[output.symbol.eq('A'), 'score'].iloc[0] == .5
    future = frame.assign(date='2026-11-12', raw=.01)
    actual = policy.predict(pd.concat([frame, future]).sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(output.reset_index(drop=True), actual.loc[actual.date.eq('2026-11-11')].reset_index(drop=True))


def test_projection_does_not_rerank_reallocate_or_mutate():
    payload = {'sleeves': [{'initial_capital_weight': .25, 'eligible_count': 3,
        'candidates': [{'symbol': 'A', 'side': 'right', 'priority': 1},
                       {'symbol': 'B', 'side': 'left', 'priority': 2},
                       {'symbol': 'C', 'side': 'left', 'priority': 3}]}]}
    original = copy.deepcopy(payload)
    out = project_portfolio(payload, side='left', limit=1)
    assert out['sleeves'][0]['visible_eligible_count'] == 2
    assert out['sleeves'][0]['candidates'][0]['priority'] == 2
    assert out['sleeves'][0]['initial_capital_weight'] == .25
    assert payload == original


def test_config_rejects_changed_policy_and_path_escape(tmp_path):
    config = tmp_path / 'configs/strategies/application_portfolio.json'
    config.parent.mkdir(parents=True)
    policy = tmp_path / 'policy.json'
    policy.write_text('{}')
    value = dict(enabled=True, policy='policy.json', policy_sha256=file_sha256(policy),
                 rankings='rank.parquet', inputs='inputs.parquet', manifest='manifest.json')
    config.write_text(json.dumps(value))
    assert load_portfolio_config(tmp_path)['enabled']
    policy.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='identity'):
        load_portfolio_config(tmp_path)
    value['rankings'] = '../outside.parquet'
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='escapes'):
        load_portfolio_config(tmp_path)


def test_new_dag_node_owns_outputs_and_precedes_selector(monkeypatch):
    import quant.application.portfolio_application as app
    from quant.application.daily_dependencies import build_default_daily_dependency_registry
    config = dict(enabled=True, rankings='data/features/application_portfolio/latest_rankings.parquet',
        inputs='data/features/application_portfolio/latest_inputs.parquet', manifest='data/features/application_portfolio/manifest.json')
    monkeypatch.setattr(app, 'load_portfolio_config', lambda *args: config)
    registry = build_default_daily_dependency_registry()
    node = registry.nodes['score.application_portfolio']
    assert set(node.publication_outputs) == {config[k] for k in ['rankings', 'inputs', 'manifest']}
    assert 'score.application_portfolio' in registry.required_node_ids('short')
    # The production registry resolves the new callable through its entrypoint.
    assert node.operation_id == 'run_application_portfolio'


@pytest.mark.parametrize('manifest_date', ['2026-11-13', '2026-11-12'])
def test_portfolio_reads_only_injected_generation_and_keeps_date_gate(tmp_path, manifest_date):
    from quant.application.portfolio_application import attach_portfolio
    from quant.ml.rank_fusion import file_sha256
    import pandas as pd

    committed = tmp_path / 'generation'
    committed.mkdir()
    rankings_path = committed / 'rankings.parquet'
    pd.DataFrame([{'symbol': '000001.SZ', 'date': '2026-11-13', 'sleeve': 'left',
                   'priority': 1, 'side': 'left', 'score': 0.5}]).to_parquet(rankings_path)
    metadata = {'status': 'success', 'target_date': manifest_date, 'policy_sha256': 'policy',
                'rankings_sha256': file_sha256(rankings_path), 'weights': {'left': 1},
                'labels': {'left': 'Left'}, 'release_id': 'release', 'account': {}}
    (committed / 'manifest.json').write_text(json.dumps(metadata))
    (tmp_path / 'manifest.json').write_text('invalid live file')
    resolved = []

    def resolve(path):
        resolved.append(path)
        return committed / path.name

    kwargs = {'resolve_path': resolve, 'root': tmp_path, 'config': {
        'enabled': True, 'active_from': '2026-11-01', 'policy_sha256': 'policy',
        'manifest': 'manifest.json', 'rankings': 'rankings.parquet'}}
    payload = {'signal_date': '2026-11-13', 'stocks': [{'symbol': '000001.SZ', 'close': 10}]}
    if manifest_date != payload['signal_date']:
        with pytest.raises(RuntimeError, match='date or release'):
            attach_portfolio(payload, **kwargs)
    else:
        result = attach_portfolio(payload, **kwargs)
        assert result['application_portfolio']['sleeves'][0]['candidates'][0]['symbol'] == '000001.SZ'
    assert resolved == [tmp_path / 'manifest.json', tmp_path / 'rankings.parquet']
