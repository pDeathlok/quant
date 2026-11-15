from dataclasses import replace
from pathlib import Path
import json

import pandas as pd

import pytest

from quant.infrastructure.publication import PublicationView, publication_context
from quant.webapp import services


@pytest.mark.parametrize('mapped', [True, False])
def test_new_left_snapshot_uses_explicit_legacy_columns(monkeypatch, tmp_path, mapped):
    from quant.ml.rank_fusion import file_sha256
    frame=pd.DataFrame({'symbol':['000001.SZ'],'date':[pd.Timestamp('2026-11-11')],
                        'ls_close_to_bbi':[.9],'selector_legacy__ls_close_to_bbi':[.1]})
    path=tmp_path/'latest_features.parquet';frame.to_parquet(path,index=False)
    manifest=tmp_path/'feature_manifest.json'
    manifest.write_text(json.dumps(dict(status='success',target_date='2026-11-11',output_sha256=file_sha256(path),
        selector_compatibility_features={'ls_close_to_bbi':'selector_legacy__ls_close_to_bbi'} if mapped else {})))
    config=services.DEFAULT_LEFT_SIDE_RANKING_CONFIG
    monkeypatch.setattr(services,'DEFAULT_LEFT_SIDE_RANKING_CONFIG',replace(config,engine='rank_fusion',
        paths=replace(config.paths,feature_output=path,feature_manifest=manifest)))
    monkeypatch.setattr(services,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(services,'_selector_active_model_features',lambda:('ls_close_to_bbi',))
    if not mapped:
        with pytest.raises(RuntimeError,match='missing legacy selector feature mapping'):
            services._selector_production_snapshot_rows('2026-11-11')
    else:
        rows=services._selector_production_snapshot_rows('2026-11-11')
        assert rows['000001.SZ']['ls_close_to_bbi']==.1


@pytest.fixture
def published_payload(monkeypatch):
    monkeypatch.setattr(services, '_row_display_quality_gate', lambda row: True)
    monkeypatch.setattr(services, '_selector_score_probability_bands', lambda: {'available': False, 'calibrations': []})
    monkeypatch.setattr(services, '_apply_left_side_ranked_metrics', lambda rows: None)
    return {'signal_date': '2026-11-10', 'snapshot_scope': {'strategies': ['B1']}, 'stocks': [{
        'symbol': '000001.SZ', 'ranking_source': 'left_side_unified',
        'ranking_score': .73, 'ranking_score_normalized': 83., 'selector_score': 83.,
        'ranking_score_date': '2026-11-10', 'ranking_model_artifact_sha256': 'a'*64,
        'matched_count': 1, 'best_profit_factor': 1.2, 'signals': [],
        'buy_score_source': 'historical_return_model', 'hold_score_source': 'historical_return_model',
    }]}


def test_committed_snapshot_never_reads_mutable_rank_cache(monkeypatch, tmp_path, published_payload):
    def forbidden(*args, **kwargs):
        raise AssertionError('read-only display must not load the latest scoring cache')
    monkeypatch.setattr(services, 'apply_selector_ranking_source', forbidden)
    with publication_context(PublicationView(tmp_path, tmp_path/'publications', 'old-generation', ())):
        result = services._display_selector_payload(published_payload)
    row = result['stocks'][0]
    assert row['ranking_score'] == .73
    assert row['ranking_model_artifact_sha256'] == 'a'*64


@pytest.mark.parametrize('field,value', [('ranking_score_date', '2026-11-11'),
                                       ('ranking_score', float('nan')),
                                       ('ranking_model_artifact_sha256', '')])
def test_committed_snapshot_rejects_invalid_embedded_rank(tmp_path, published_payload, field, value):
    published_payload['stocks'][0][field] = value
    with publication_context(PublicationView(tmp_path, tmp_path/'publications', 'old-generation', ())):
        with pytest.raises(RuntimeError, match='published selector'):
            services._display_selector_payload(published_payload)


def test_new_left_model_cannot_inherit_old_fold_c_metrics(monkeypatch):
    monkeypatch.setattr(services, 'DEFAULT_LEFT_SIDE_RANKING_CONFIG', replace(services.DEFAULT_LEFT_SIDE_RANKING_CONFIG, engine='rank_fusion'))
    def forbidden(*args, **kwargs):
        raise AssertionError('new model must not read the old model performance CSV')
    monkeypatch.setattr(services.pd, 'read_csv', forbidden)
    services._left_side_ranked_metrics.cache_clear()
    try:
        assert services._left_side_ranked_metrics() is None
    finally:
        services._left_side_ranked_metrics.cache_clear()


def test_display_excludes_probability_bands_for_another_left_model(monkeypatch, tmp_path, published_payload):
    monkeypatch.setattr(services, 'apply_selector_ranking_source', lambda rows, *args: rows)
    monkeypatch.setattr(services, '_selector_score_probability_bands', lambda: {'available': True, 'calibrations': [
        {'key': 'model_left', 'source': {'artifact_sha256': 'b'*64}},
        {'key': 'model_left_v5_shadow', 'source': {'artifact_sha256': 'c'*64}},
        {'key': 'buy', 'source': {'artifact_sha256': 'd'*64}},
    ]})
    with publication_context(PublicationView(tmp_path, tmp_path/'publications', 'old-generation', ())):
        result = services._display_selector_payload(published_payload)
    assert [x['key'] for x in result['score_probability_bands']['calibrations']] == ['buy']


def test_rank_fusion_admitted_row_does_not_depend_on_old_strategy_metrics():
    # The model's cash gate already controls admission. An unrelated legacy
    # strategy backtest must not silently remove an admitted current candidate.
    row = {'ranking_source': 'left_side_unified', 'ranking_model_engine': 'rank_fusion',
           'signals': [{'strategy_family': 'SB1', 'metrics': None}]}
    assert services._row_display_quality_gate(row)


@pytest.mark.parametrize('active_key,artifact', [('model_left', 'old'), ('model_left_ranknet_v6', 'new')])
def test_probability_bands_follow_the_displayed_model_version(active_key, artifact):
    payload = {'available': True, 'calibrations': [
        {'key': 'model_left', 'source': {'artifact_sha256': 'old'}},
        {'key': 'model_left_ranknet_v6', 'source': {'artifact_sha256': 'new'}},
        {'key': 'model_right', 'source': {'artifact_sha256': 'right'}},
        {'key': 'buy', 'source': {}},
    ]}
    result = services._bind_selector_probability_bands(payload, [
        {'ranking_source': 'left_side_unified', 'ranking_model_artifact_sha256': artifact},
    ])
    assert [c['key'] for c in result['calibrations']] == [active_key, 'buy']


def test_generated_row_records_model_engine_before_publication():
    rows = [{'ranking_source': 'left_side_unified'}]
    services._annotate_selector_rank_release(rows)
    assert rows[0]['ranking_model_engine'] == 'rank_fusion'
    assert rows[0]['ranking_model_release_id'] == services.DEFAULT_LEFT_SIDE_RANKING_CONFIG.release_id
