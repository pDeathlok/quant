import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from quant.ml.rank_fusion import RankFusion, compact_booster, file_sha256


def test_tree_compaction_preserves_predictions_and_missing_routes(tmp_path):
    rng = np.random.default_rng(73)
    values = pd.DataFrame({'unused': np.ones(100), 'signal': rng.normal(size=100), 'empty': np.nan})
    values.loc[::7, 'signal'] = np.nan
    model = xgb.train({'max_depth': 2, 'nthread': 1}, xgb.DMatrix(values, label=values.signal.fillna(-3)), num_boost_round=6)
    compact = compact_booster(model)
    assert compact.feature_names == ['signal']
    np.testing.assert_array_equal(model.predict(xgb.DMatrix(values)), compact.predict(xgb.DMatrix(values[['signal']])))
    path = tmp_path/'model.ubj'
    compact.save_model(path)
    manifest = dict(schema_version='rank-fusion-v1', release_id='test', required_features=['signal'],
                    minimum_expected_net_return=0, components=[dict(file='model.ubj', sha256=file_sha256(path),
                    features=['signal'], weight=1, calibration={'x': [-3, 3], 'y': [-.1, .1]})])
    config = tmp_path/'manifest.json'; config.write_text(json.dumps(manifest))
    ranker = RankFusion.load(config)
    frame = values.assign(symbol=[str(i) for i in range(100)], date=pd.Timestamp('2026-01-05'))
    result = ranker.predict(frame)
    assert result.trade_gate_open.any() and (~result.trade_gate_open).any()
    with pytest.raises(ValueError, match='absent'):
        ranker.predict(frame.drop(columns='signal'))
    with pytest.raises(ValueError, match='Unique'):
        ranker.predict(pd.concat([frame, frame]))
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        RankFusion.load(config)


def test_kdj_readiness_cannot_be_unlocked_by_future_rows():
    from quant.features.variable_library import _align_resampled_kdj
    dates = pd.bdate_range('2024-01-02', '2025-05-01')
    values = 10 + np.sin(np.arange(len(dates))/5)
    frame = pd.DataFrame({'date': dates, 'open': values, 'high': values+1, 'low': values-1,
                          'close': values, 'volume': 100})
    short = frame[frame.date.lt('2024-09-01')]
    full = _align_resampled_kdj(frame, 'ME', 'kdj_m')
    prefix = _align_resampled_kdj(short, 'ME', 'kdj_m')
    pd.testing.assert_frame_equal(prefix, full.iloc[:len(short)])
    assert prefix.isna().all().all()
    assert full.iloc[-1].notna().all()


def test_publication_alignment_is_strict_prior_and_market_features_are_causal():
    from quant.features.rank_runtime import align_prior, market_features
    source = pd.DataFrame({'symbol':['000001.SZ']*2, 'available':pd.to_datetime(['2024-04-01','2024-04-02']), 'flow':[1.,99.]})
    keys = pd.DataFrame({'symbol':['000001.SZ']*2, 'date':pd.to_datetime(['2024-04-02','2024-04-03'])})
    assert align_prior(keys, source).flow.tolist() == [1.,99.]
    dates = pd.bdate_range('2023-10-01','2024-07-01')
    raw = pd.concat([pd.DataFrame({'symbol':symbol,'date':dates,'ts_code':symbol,'trade_date':dates.strftime('%Y%m%d'),
        'close':10+np.arange(len(dates))*.01+i,'pre_close':9.99+np.arange(len(dates))*.01+i,
        'pct_chg':.1,'volume':100.}) for i,symbol in enumerate(['000001.SZ','000002.SZ'])],ignore_index=True)
    full, ranks = market_features(raw)
    before = pd.Timestamp('2024-05-01')
    prefix, prefix_ranks = market_features(raw.loc[raw.date.lt(before)])
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True),full.loc[full.date.lt(before)].reset_index(drop=True))
    pd.testing.assert_frame_equal(prefix_ranks.reset_index(drop=True),ranks.loc[ranks.date.lt(before)].reset_index(drop=True))
