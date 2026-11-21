"""Date- and input-version-aware materialization for deployed RankNet models.

This layer reads shared source data only. Fetch ownership stays in the daily
source DAG; absence of an input is an error, never a request from a scorer.
"""
from __future__ import annotations

import hashlib
import json
import inspect
import fcntl
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
from quant.features import rank_runtime as calc
from quant.features.active_market_value import KEY_INDEX_SPECS
from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.ml.rank_fusion import file_sha256
from quant.data.market_snapshot import current_market_snapshot, pinned_dataset_path, read_pinned_parquet


_LEGACY_PREFIX = "selector_legacy__"


def _dated(frame: pd.DataFrame) -> pd.DataFrame:
    result=frame.copy()
    if 'symbol' not in result:
        result['symbol']=result.ts_code.astype(str)
    if 'ts_code' not in result:
        result['ts_code']=result.symbol
    if 'date' not in result:
        result['date']=pd.to_datetime(result.trade_date.astype(str),format='mixed')
    result['date']=pd.to_datetime(result.date).astype('datetime64[ns]')
    date_text={d:pd.Timestamp(d).strftime('%Y%m%d') for d in result.date.unique()}
    result['trade_date']=result.date.map(date_text)
    return result


def _identity(paths: list[Path]) -> str:
    records=[]
    for path in sorted(set(paths)):
        stat=path.stat()
        records.append((str(path),stat.st_size,stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def _stock_task(task: tuple[pd.DataFrame, pd.Timestamp, list[str] | None]) -> pd.DataFrame:
    raw,target,required=task
    frame=calc.stock_price_features(raw)
    if required is None:
        return _target_numeric_features(frame, target)
    result=frame.loc[frame.date.eq(target),['symbol','date',*[c for c in required if c in frame]]].copy()
    compatibility=[c for c in required if c.startswith(_LEGACY_PREFIX)]
    if compatibility:
        legacy = _legacy_task((raw, target)).iloc[0]
        for column in compatibility:
            result[column]=legacy[column]
    return result


def _target_numeric_features(frame: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    columns = frame.select_dtypes(include=["number", "bool"]).columns.difference(calc.KEYS)
    return frame.loc[frame.date.eq(target), [*calc.KEYS, *columns]].copy()


def _legacy_task(task: tuple[pd.DataFrame, pd.Timestamp]) -> pd.DataFrame:
    from quant.data.source_merge import normalize_tushare_daily
    from quant.research.left_side_unified_features import (
        LEFT_SIDE_RULE_FEATURE_COLUMNS, compute_left_side_rule_features,
    )

    raw, target = task
    symbol = str(raw.symbol.iloc[0])
    normalized = normalize_tushare_daily(raw, symbol)
    history = normalized.loc[normalized.date.le(target)].tail(260).reset_index(drop=True)
    legacy = compute_left_side_rule_features(history).iloc[-1]
    return pd.DataFrame([{"symbol": symbol, "date": target, **{
        f"{_LEGACY_PREFIX}{column}": legacy[column] for column in LEFT_SIDE_RULE_FEATURE_COLUMNS
    }}])


def _map_stock_tasks(fn, tasks, workers: int) -> pd.DataFrame:
    if workers == 1:
        return pd.concat([fn(task) for task in tasks], ignore_index=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return pd.concat(pool.map(fn, tasks, chunksize=8), ignore_index=True)


def _read_shared_block(path: Path, contract: dict[str, Any]) -> pd.DataFrame | None:
    metadata = path.with_suffix(".json")
    try:
        saved = json.loads(metadata.read_text())
        if saved.get("contract") != contract or saved.get("sha256") != file_sha256(path):
            return None
        frame = pd.read_parquet(path)
        if (len(frame) != saved.get("rows") or frame.symbol.duplicated().any()
                or not frame.date.eq(pd.Timestamp(contract["target"])).all()):
            return None
        return frame
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def _extend_shared_block(
    path: Path,
    contract: dict[str, Any],
    symbols: list[str],
    columns: list[str],
    build: Callable[[list[str]], pd.DataFrame],
    validate_sources: Callable[[], None],
) -> tuple[pd.DataFrame, dict[str, int]]:
    cached = _read_shared_block(path, contract)
    if cached is not None and not set(columns) <= set(cached):
        raise RuntimeError(f"Unmaterialized rank features: {sorted(set(columns) - set(cached))}")
    available = set(cached.symbol) if cached is not None else set()
    missing = sorted(set(symbols) - available)
    if missing:
        fresh = build(missing)
        if (fresh.symbol.duplicated().any() or set(fresh.symbol) != set(missing)
                or not fresh.date.eq(pd.Timestamp(contract["target"])).all()):
            raise RuntimeError("Shared rank block did not cover the requested exact-date keys")
        if not set(columns) <= set(fresh):
            raise RuntimeError(f"Unmaterialized rank features: {sorted(set(columns) - set(fresh))}")
        features = fresh.columns.difference(calc.KEYS)
        fresh[features] = fresh[features].astype("float32").replace([np.inf, -np.inf], np.nan)
        if cached is not None and set(cached) != set(fresh):
            raise RuntimeError("Shared rank block schema differs across candidate partitions")
        cached = fresh if cached is None else pd.concat([cached, fresh], ignore_index=True)
        validate_sources()
        atomic_write_parquet(cached.sort_values("symbol"), path, index=False)
        atomic_write_json(
            {"contract": contract, "sha256": file_sha256(path), "rows": len(cached)},
            path.with_suffix(".json"),
        )
    assert cached is not None
    result = cached.loc[cached.symbol.isin(symbols), [*calc.KEYS, *columns]]
    result = result.sort_values("symbol").reset_index(drop=True)
    return result, {"computed_symbols": len(missing), "reused_symbols": len(symbols) - len(missing)}


def materialize_rank_features(target_date: str, symbols: list[str], required: list[str], *,
                              root: Path, cache_root: Path, workers: int = 4,
                              index_root: Path | None = None,
                              require_complete_context: bool = True) -> tuple[pd.DataFrame,dict[str,Any]]:
    """Coalesce concurrent requests before checking or computing shared caches."""
    lock_dir=cache_root/'locks';lock_dir.mkdir(parents=True,exist_ok=True)
    with (lock_dir/f'{pd.Timestamp(target_date):%Y%m%d}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            return _materialize_rank_features(target_date,symbols,required,root=root,cache_root=cache_root,
                workers=workers,index_root=index_root,require_complete_context=require_complete_context)
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def materialize_production_rank_features(
    target_date: str,
    symbols: list[str],
    required: list[str],
    *,
    root: Path,
    workers: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Production consumers declare their projection, not a cache namespace."""
    return materialize_rank_features(
        target_date, symbols, required, root=root,
        cache_root=root / "data/features/rank_runtime", workers=workers,
    )


def _read_candidate_partitions(paths: list[Path], symbols: set[str], workers: int, *,
                               full_cross_section_date: str | None = None,
                               columns: list[str] | None = None) -> list[pd.DataFrame]:
    from quant.infrastructure.publication import ContextThreadPoolExecutor
    from quant.routine.resource_scheduler import current_resource_grant
    grant = current_resource_grant()
    if grant is not None and grant.claim.io_slots < 1:
        raise RuntimeError('Rank partition reads require an IO resource grant')
    limit = min(workers, 2, grant.claim.io_slots if grant is not None else 2,
                grant.granted_workers if grant is not None else workers)
    def read(path: Path) -> pd.DataFrame:
        frame = read_pinned_parquet(path)
        if path.stem != full_cross_section_date:
            frame = frame.loc[frame.ts_code.isin(symbols)]
        return frame.reindex(columns=columns) if columns is not None else frame
    if limit <= 1:
        return [read(path) for path in paths]
    # Propagate the immutable source pin to each reader and stay inside the
    # parent's CPU/IO envelope. Filter before retaining or concatenating data.
    with ContextThreadPoolExecutor(max_workers=limit) as pool:
        return list(pool.map(read, paths))


def _materialize_rank_features(target_date: str, symbols: list[str], required: list[str], *,
                              root: Path, cache_root: Path, workers: int = 4,
                              index_root: Path | None = None,
                              require_complete_context: bool = True) -> tuple[pd.DataFrame,dict[str,Any]]:
    target=pd.Timestamp(target_date).normalize(); compact=target.strftime('%Y%m%d')
    if not 1<=workers<=16:
        raise ValueError('Rank feature workers must be in [1,16]')
    symbols=sorted(set(symbols));required=sorted(set(required))
    raw_root=root/'data/raw'
    pin=current_market_snapshot()
    def source_directory(kind: str) -> Path:
        return pinned_dataset_path(kind) if pin else raw_root/kind
    basic_paths=sorted(p for p in source_directory('daily_basic').glob('*.parquet') if p.stem.isdigit() and '20130101'<=p.stem<=compact)
    dates=pd.DatetimeIndex(pd.to_datetime([p.stem for p in basic_paths]))
    if target not in dates:
        raise RuntimeError(f'Daily basic source missing exact target date {compact}')
    weekly=pd.DataFrame({'path':basic_paths,'date':dates}).groupby(dates.to_period('W-FRI')).tail(1).path.tolist()
    daily_paths=[p for p in basic_paths if p.stem>='20181101']
    needed_basic=sorted(set(weekly+daily_paths))
    index_paths=[(index_root/f'{spec.ts_code}_index_daily.parquet') if index_root else raw_root/f'index_{spec.ts_code}.parquet' for spec in KEY_INDEX_SPECS]
    if pin:
        if index_root is not None:
            raise ValueError('Historical index override is forbidden under a production source pin')
        index_paths=[source_directory(f"index_{spec.ts_code.replace('.', '_')}")/f'index_{spec.ts_code}.parquet' for spec in KEY_INDEX_SPECS]
    price_paths=[] if pin else [p for p in (raw_root/'daily_partitioned').glob('*/*.parquet') if '201401'<=p.parent.name.split('=')[-1]<=compact[:6]]
    flow_paths={kind:sorted(p for p in source_directory(kind).glob(f'tushare_{kind}_*.parquet') if '20231001'<=p.stem.rsplit('_',1)[-1]<=compact)
                for kind in ['moneyflow','margin_detail']}
    top_paths=sorted(p for p in source_directory('top_list').glob('tushare_top_list_*.parquet') if '20230901'<=p.stem.rsplit('_',1)[-1]<=compact)
    if require_complete_context:
        reference=next(p for p in index_paths if '000300.SH' in p.name)
        calendar=pd.to_datetime(read_pinned_parquet(reference,columns=['trade_date']).trade_date.astype(str),format='mixed')
        expected={d.strftime('%Y%m%d') for d in sorted(set(calendar.loc[calendar.lt(target)]))[-60:]}
        absent_basic=expected-{p.stem for p in basic_paths}
        if len(expected)<60 or absent_basic:
            raise RuntimeError(f'Incomplete daily-basic trading context: {sorted(absent_basic)}')
        for name,paths in {**flow_paths,'top_list':top_paths}.items():
            absent=expected-{p.stem.rsplit('_',1)[-1] for p in paths}
            if absent:
                raise RuntimeError(f'Required {name} rolling context missing {len(absent)} trading days; source DAG must repair it')
    formula_paths=[Path(__file__),Path(calc.__file__),*sorted((root/'src/quant/features').glob('*.py')),
                   root/'src/quant/research/left_rule_replay.py',root/'src/quant/research/right_side_unified_features.py',
                   root/'src/quant/research/left_side_unified_features.py']
    source_paths=[*price_paths,*needed_basic,*index_paths,*top_paths,*sum(flow_paths.values(),[]),*formula_paths]
    observed_identity=_identity(source_paths)
    identity=(hashlib.sha256((json.dumps(pin.manifest['datasets'],sort_keys=True)+_identity(formula_paths)).encode()).hexdigest()
              if pin else observed_identity)
    shared_contract = dict(
        version=calc.VERSION, target=target.date().isoformat(), input_identity=identity,
        source_mode="sealed" if pin else "local", complete_context=require_complete_context,
        universe="canonical-market-cross-section",
    )
    mode = shared_contract["source_mode"]
    context = "strict" if require_complete_context else "relaxed"
    directory = cache_root / compact / f"shared-{mode}-{context}"
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=raw_root))
    legacy_columns = [column for column in required if column.startswith(_LEGACY_PREFIX)]
    base_columns = [column for column in required if not column.startswith(_LEGACY_PREFIX)]
    recent = None

    def validate_sources():
        if _identity(source_paths) != observed_identity:
            raise RuntimeError("Inputs changed during rank feature calculation; retry from stable sources")

    def read_prices(symbols):
        raw = _dated(store.read_market_range("daily", start_date="20140101", end_date=compact, symbols=symbols))
        if raw.duplicated(calc.KEYS).any() or not set(symbols) <= set(raw.loc[raw.date.eq(target), "symbol"]):
            raise RuntimeError('Rank candidate prices duplicated or missing on target date')
        if "vol" not in raw:
            raw["vol"] = raw.volume
        return raw

    def build_base(symbols):
        nonlocal recent
        keys = pd.DataFrame({"symbol": symbols, "date": target})
        raw = read_prices(symbols)
        if legacy_columns:
            recent = raw.sort_values("date").groupby("symbol", sort=False).tail(260)
        print('RANK FEATURES price',len(symbols),flush=True)
        tasks = ((group, target, None) for _, group in raw.groupby("symbol", sort=True))
        prices = _map_stock_tasks(_stock_task, tasks, workers)
        # Read candidate daily-basic history once; retain the complete target
        # cross section for ranks, and older weekly observations for PB ranks.
        basic_columns=['ts_code','trade_date','turnover_rate','turnover_rate_f','volume_ratio','total_mv','circ_mv',
                       'total_share','float_share','free_share','pe_ttm','pb','ps_ttm','dv_ttm']
        parts=_read_candidate_partitions(needed_basic,set(symbols),workers,
            full_cross_section_date=compact,columns=basic_columns)
        basic=_dated(pd.concat(parts,ignore_index=True))
        if basic.duplicated(calc.KEYS).any():
            raise RuntimeError('Duplicate raw basic source')
        basic_current=calc.basic_features(basic.loc[basic.date.ge('2018-11-01')])
        basic_current=basic_current.loc[basic_current.date.eq(target)&basic_current.symbol.isin(symbols)]
        result=keys.merge(prices,on=calc.KEYS,validate='one_to_one')
        def attach(other: pd.DataFrame, *, on: list[str] = calc.KEYS) -> None:
            nonlocal result
            columns = other.select_dtypes(include=["number", "bool"]).columns.difference(result.columns)
            if len(columns):
                result = result.merge(other[[*on, *columns]], on=on, how="left",
                                      validate="many_to_one" if on == ["date"] else "one_to_one")
        attach(basic_current)
        jq=calc.jq_price_features(raw,basic.loc[basic.symbol.isin(symbols)])
        attach(jq.loc[jq.date.eq(target)])
        attach(calc.periodic_pb_features(keys,basic,calendar=dates))
        print('RANK FEATURES market',flush=True)
        # Shared date-level cache is independent of candidate sets and models.
        market_code=inspect.getsource(calc.market_features)+calc.ASHARE
        daily_identity=json.dumps(pin.manifest['datasets']['daily'],sort_keys=True) if pin else _identity(price_paths)
        market_key=hashlib.sha256((daily_identity+market_code+file_sha256(root/'src/quant/features/market_breadth.py')).encode()).hexdigest()
        market_dir=cache_root/'market'/compact/market_key
        market_file=market_dir/'market.parquet';ranks_file=market_dir/'ranks.parquet'
        market_manifest=market_dir/'manifest.json'
        cached=json.loads(market_manifest.read_text()) if market_manifest.exists() else {}
        if (market_file.exists() and ranks_file.exists() and cached.get('market_sha256')==file_sha256(market_file)
                and cached.get('ranks_sha256')==file_sha256(ranks_file)):
            market=pd.read_parquet(market_file);ranks=pd.read_parquet(ranks_file)
        else:
            market_raw=_dated(store.read_market_range('daily',start_date='20180101',end_date=compact,
                       columns=('ts_code','trade_date','close','pre_close','pct_chg','volume','vol')))
            market,ranks=calc.market_features(market_raw)
            market=market.loc[market.date.eq(target)];ranks=ranks.loc[ranks.date.eq(target)]
            market_dir.mkdir(parents=True,exist_ok=True)
            atomic_write_parquet(market,market_file,index=False);atomic_write_parquet(ranks,ranks_file,index=False)
            atomic_write_json(dict(market_sha256=file_sha256(market_file),ranks_sha256=file_sha256(ranks_file)),market_manifest)
        attach(market,on=['date']);attach(ranks.loc[ranks.symbol.isin(symbols)])
        # Research cross-sectional ranks were computed on the observed A-share
        # universe, not on all securities returned by daily_basic.
        ranked_basic=basic.loc[basic.date.eq(target)&basic.symbol.isin(ranks.symbol)].copy()
        ranked_basic['rank']=np.log(ranked_basic.total_mv.replace(0,np.nan)).astype('float32').rank(pct=True)
        result['db_total_mv_pct_rank']=result.symbol.map(ranked_basic.set_index('symbol')['rank'])
        indices=_dated(pd.concat([read_pinned_parquet(p) for p in index_paths],ignore_index=True))
        indices=indices.loc[indices.date.le(target)&indices.date.ge('2018-01-01')]
        if set(indices.loc[indices.date.eq(target),'ts_code'])!={s.ts_code for s in KEY_INDEX_SPECS}:
            raise RuntimeError('Required index data incomplete on target date')
        index=calc.index_features(indices);attach(index.loc[index.date.eq(target)],on=['date'])
        result['return_120d_minus_market']=result.return_120d.astype('float32')-result.market_return_26w
        volatility=prices.set_index('symbol').volatility_60d.astype('float32')
        dv=basic_current.set_index('symbol').dv_ttm.astype('float32')
        result['dividend_to_volatility']=result.symbol.map(dv/volatility.replace(0,np.nan))
        for kind,paths in flow_paths.items():
            if not paths:
                raise RuntimeError(f'Missing required {kind} source')
            flows=pd.concat(_read_candidate_partitions(paths,set(symbols),workers),ignore_index=True)
            attach(calc.align_prior(keys,calc.flow_features(flows,kind)))
        events=_dated(pd.concat(_read_candidate_partitions(top_paths,set(symbols),workers),ignore_index=True))
        attach(calc.top_list_features(keys,events,dates,pd.DatetimeIndex(pd.to_datetime([p.stem.rsplit('_',1)[-1] for p in top_paths]))))
        return result

    def build_legacy(symbols):
        # Reuse bars from the base block when both are requested together.
        have = set(recent.symbol) if recent is not None else set()
        missing = sorted(set(symbols) - have)
        parts = []
        if have.intersection(symbols):
            parts.append(recent.loc[recent.symbol.isin(symbols)])
        if missing:
            parts.append(read_prices(missing))
        raw = pd.concat(parts, ignore_index=True)
        tasks = ((group, target) for _, group in raw.groupby("symbol", sort=True))
        return _map_stock_tasks(_legacy_task, tasks, workers)

    # These blocks use symbol-local history or a fixed market-wide observation
    # universe. Never add a caller-universe rank to a shared block.
    blocks = {}
    result = pd.DataFrame({"symbol": pd.Series(symbols, dtype="str"), "date": target})
    for block, columns, build in (("base", base_columns, build_base), ("legacy", legacy_columns, build_legacy)):
        if symbols and columns:
            rows, blocks[block] = _extend_shared_block(
                directory / f"{block}.parquet", {**shared_contract, "block": block},
                symbols, columns, build, validate_sources,
            )
            result = result.merge(rows, on=calc.KEYS, validate="one_to_one")
    result = result.reindex(columns=[*calc.KEYS, *required])
    validate_sources()
    # The per-consumer digest remains a Parquet digest, without retaining a
    # second on-disk copy for every distinct projection.
    digest = hashlib.sha256(result.to_parquet(index=False)).hexdigest()
    contract = dict(version=calc.VERSION, target=target.date().isoformat(), symbols=symbols,
                    features=required, input_identity=identity)
    manifest = dict(
        status="success", contract=contract, output_sha256=digest, rows=len(result),
        coverage=result[required].notna().mean().to_dict(),
        cache_hit=bool(blocks) and all(item["computed_symbols"] == 0 for item in blocks.values()),
        shared_blocks=blocks, input_files=len(source_paths),
    )
    return result,manifest
