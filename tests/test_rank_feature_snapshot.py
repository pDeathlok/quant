"""Exercise the deployed feature calculator on sealed, complete toy sources."""
from pathlib import Path
import numpy as np
import pandas as pd


def test_rank_materialization_reads_sealed_sources_and_reuses_cache(tmp_path):
    from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
    from quant.data.market_data_store import MarketDataStore,MarketDataStoreConfig
    from quant.data.market_snapshot import export_market_snapshot,pinned_market_snapshot
    from quant.features.active_market_value import KEY_INDEX_SPECS
    from quant.ml.rank_fusion import RankFusion
    from quant.routine.rank_feature_runtime import materialize_rank_features

    target=pd.Timestamp('2026-11-10');dates=pd.bdate_range(end=target,periods=180)
    raw=tmp_path/'data/raw';prices=raw/'daily_partitioned/month=202611';prices.mkdir(parents=True)
    parts=[]
    for symbol in ['000001.SZ','000002.SZ']:
        close=10+np.arange(len(dates))*.01
        parts.append(pd.DataFrame({'ts_code':symbol,'symbol':symbol,'date':dates,'trade_date':dates.strftime('%Y%m%d'),
            'open':close-.01,'high':close+.1,'low':close-.1,'close':close,'pre_close':close-.01,
            'pct_chg':.1,'volume':1000.,'vol':1000.,'amount':10000.,'turnover':1.}))
    pd.concat(parts,ignore_index=True).to_parquet(prices/'all.parquet',index=False)
    sources={kind:raw/kind for kind in ['daily_basic','moneyflow','margin_detail','top_list']}
    for folder in sources.values():folder.mkdir()
    for date in dates[-61:]:
        keys={'ts_code':['000001.SZ','000002.SZ'],'trade_date':[date.strftime('%Y%m%d')]*2}
        pd.DataFrame({**keys,**{col:[1.,2.] for col in ['turnover_rate','turnover_rate_f','volume_ratio','pe_ttm','pb','ps_ttm','dv_ttm']},
            **{col:[100.,200.] for col in ['total_mv','circ_mv','total_share','float_share','free_share']}}).to_parquet(
                sources['daily_basic']/f'{date:%Y%m%d}.parquet',index=False)
        pd.DataFrame({**keys,**{col:1. for col in ['buy_sm_amount','sell_sm_amount','buy_md_amount','sell_md_amount',
            'buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount','net_mf_amount']}}).to_parquet(
                sources['moneyflow']/f'tushare_moneyflow_{date:%Y%m%d}.parquet',index=False)
        pd.DataFrame({**keys,**{col:1. for col in ['rzye','rqye','rzmre','rzche']}}).to_parquet(
                sources['margin_detail']/f'tushare_margin_detail_{date:%Y%m%d}.parquet',index=False)
        pd.DataFrame({**keys,'net_amount':1.,'amount':10.}).to_parquet(sources['top_list']/f'tushare_top_list_{date:%Y%m%d}.parquet',index=False)
    for spec in KEY_INDEX_SPECS:
        path=raw/f'index_{spec.ts_code}.parquet'
        parts[0].assign(ts_code=spec.ts_code,symbol=spec.ts_code).to_parquet(path,index=False)
        sources[f"index_{spec.ts_code.replace('.', '_')}"]=path
    # The toy root supplies version-marker paths; formulas execute the real
    # imported modules. This keeps test artifacts out of the production cache.
    for relative in ['src/quant/research/left_rule_replay.py','src/quant/research/right_side_unified_features.py',
                     'src/quant/research/left_side_unified_features.py',
                     'src/quant/features/market_breadth.py']:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture identity\n')
    store=MarketDataStore(MarketDataStoreConfig(backend='parquet',root=raw))
    sealed=export_market_snapshot(store,tmp_path/'sealed',datasets=('daily',*sources),supplemental_sources=sources)
    # Replacing a mutable source after acquisition must not affect inference.
    basic_path=sources['daily_basic']/f'{target:%Y%m%d}.parquet'
    old=pd.read_parquet(basic_path);old.assign(dv_ttm=999.).to_parquet(basic_path,index=False)
    required=list(RankFusion.load(DEFAULT_LEFT_SIDE_RANKING_CONFIG.paths.artifact_manifest).required_features)
    with pinned_market_snapshot(sealed.manifest_path):
        result,first=materialize_rank_features(str(target.date()),['000001.SZ','000002.SZ'],required,
            root=tmp_path,cache_root=tmp_path/'cache',workers=1)
        again,second=materialize_rank_features(str(target.date()),['000001.SZ','000002.SZ'],required,
            root=tmp_path,cache_root=tmp_path/'cache',workers=1)
        parallel,_=materialize_rank_features(str(target.date()),['000001.SZ','000002.SZ'],required,
            root=tmp_path,cache_root=tmp_path/'parallel-cache',workers=2)
        derived=['return_120d_minus_market','dividend_to_volatility']
        compact,_=materialize_rank_features(str(target.date()),['000001.SZ','000002.SZ'],derived,
            root=tmp_path,cache_root=tmp_path/'compact-cache',workers=1)
        minimal,_=materialize_rank_features(str(target.date()),['000001.SZ','000002.SZ'],['volatility_20d'],
            root=tmp_path,cache_root=tmp_path/'minimal-cache',workers=1)
    assert not first['cache_hit'] and second['cache_hit']
    np.testing.assert_allclose(result.dividend_to_volatility*result.volatility_60d,[1.,2.],rtol=1e-5)
    assert set(result.columns)=={'symbol','date',*required}
    pd.testing.assert_frame_equal(result,again)
    pd.testing.assert_frame_equal(result,parallel)
    pd.testing.assert_frame_equal(compact,result[['symbol','date',*sorted(derived)]])
    pd.testing.assert_frame_equal(minimal,result[['symbol','date','volatility_20d']])


def test_candidate_partition_reads_retain_target_universe_and_obey_io_grant(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import threading
    from quant.routine import rank_feature_runtime as runtime,resource_scheduler
    source=pd.DataFrame({'ts_code':['000001.SZ','000002.SZ'],'value':[1.,2.]})
    worker_threads=[]
    def read(path):
        worker_threads.append(threading.current_thread().name)
        return source.copy()
    monkeypatch.setattr(runtime,'read_pinned_parquet',read)
    monkeypatch.setattr(resource_scheduler,'current_resource_grant',lambda:SimpleNamespace(claim=SimpleNamespace(io_slots=1),granted_workers=4))
    result=runtime._read_candidate_partitions([tmp_path/'20261110.parquet',tmp_path/'20261111.parquet'],
        {'000001.SZ'},4,full_cross_section_date='20261111')
    assert result[0].ts_code.tolist()==['000001.SZ']
    assert result[1].ts_code.tolist()==['000001.SZ','000002.SZ']
    assert set(worker_threads)=={threading.current_thread().name}


def test_selector_compatibility_matches_old_production_without_changing_rank_inputs():
    from quant.routine.rank_feature_runtime import _stock_task
    from quant.routine.left_side_unified_production import _build_symbol_feature, PROJECT_FACTOR_COLUMNS, LEFT_SIDE_SIGNALS
    from quant.research.left_side_unified_features import LEFT_SIDE_RULE_FEATURE_COLUMNS
    dates=pd.bdate_range(end='2026-11-11',periods=400)
    close=10+np.sin(np.arange(400)/9)+np.arange(400)*.005
    raw=pd.DataFrame(dict(symbol='000001.SZ',ts_code='000001.SZ',date=dates,trade_date=dates.strftime('%Y%m%d'),
        open=close-.02,high=close+.2,low=close-.2,close=close,pre_close=np.r_[close[0],close[:-1]],
        volume=1000+np.arange(400)%17*20,vol=1000+np.arange(400)%17*20,amount=10000.,pct_chg=.1))
    target=dates[-1]
    model_columns=['ls_close_to_bbi']
    before=_stock_task((raw,target,model_columns))
    compatibility=[f'selector_legacy__{c}' for c in LEFT_SIDE_RULE_FEATURE_COLUMNS]
    after=_stock_task((raw,target,model_columns+compatibility))
    pd.testing.assert_frame_equal(before,after[before.columns])
    project=pd.DataFrame([{**dict.fromkeys(PROJECT_FACTOR_COLUMNS,0.),'symbol':'000001.SZ','ts_code':'000001.SZ',
                          'trade_date':'20261111','date':target,'factor_schema_version':'test'}])
    old=_build_symbol_feature('000001.SZ',raw,project,dict.fromkeys(LEFT_SIDE_SIGNALS,False),target)
    np.testing.assert_allclose(after[compatibility].to_numpy(dtype=float),old[list(LEFT_SIDE_RULE_FEATURE_COLUMNS)].to_numpy(dtype=float),
                               rtol=0,atol=0,equal_nan=True)
