"""Causal, label-free materializers shared by deployed ranking and replay."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from quant.data.source_merge import normalize_tushare_daily
from quant.features.active_market_value import compute_key_index_active_market_value_features, compute_stock_active_market_value_features
from quant.features.index_ma_state import compute_index_ma_state_features
from quant.features.joinquant_candidates import price_factors
from quant.features.long_external_factors import build_moneyflow_weekly, build_margin_weekly
from quant.features.market_breadth import compute_market_breadth_features
from quant.features.project_factor_layer import calculate_project_market_factors, calculate_legacy_market_factors
from quant.features.variable_library import build_continuous_ohlc, derive_daily_basic_features
from quant.research.left_rule_replay import replay_left_rules
from quant.research.right_side_unified_features import compute_right_side_rule_features

VERSION = 'rank-feature-runtime-v1'
KEYS = ['symbol', 'date']
ASHARE = r'(?:(?:00[0-3]|30[01])\d{3}\.SZ|(?:60[0135]|68[89])\d{3}\.SH|(?:[48]\d{5}|92\d{4})\.BJ)'


def stock_price_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.duplicated('date').any():
        raise ValueError('Duplicate daily prices')
    symbol = str(raw.symbol.iloc[0])
    raw = normalize_tushare_daily(raw, symbol).sort_values('date').reset_index(drop=True)
    # Name/status are intentionally absent: no historical ST classification.
    raw = raw.drop(columns=['name', 'industry'], errors='ignore')
    extras = calculate_legacy_market_factors(raw, factor_schema_version='current')
    core = calculate_project_market_factors(raw, symbol, factor_schema_version='current', legacy_factors=extras)
    core=pd.concat([core,extras],axis=1)
    core=core.loc[:,~core.columns.duplicated(keep='first')]
    # Legacy left baseline anchors its absolute prices in 2014; the additional
    # rule block was replayed from 2018. Keep both explicit, never use a rolling
    # warmup for absolute-price features.
    raw=raw.loc[raw.date.ge('2018-01-01')].reset_index(drop=True)
    right = compute_right_side_rule_features(raw)
    left = replay_left_rules(raw)
    close = build_continuous_ohlc(raw).close
    extra = pd.DataFrame(index=raw.index)
    extra['close_to_ma120'] = close/close.rolling(120).mean()-1
    extra['ma_120_slope_20d'] = close.rolling(120).mean().pct_change(20, fill_method=None)
    extra['ma20_slope_5d'] = close.rolling(20, min_periods=10).mean().pct_change(5, fill_method=None)
    extra['ma60_dist'] = close/close.rolling(60, min_periods=30).mean()-1
    pct = raw.pct_chg.astype(float)
    extra['selector_return_60d'] = np.expm1(np.log1p(pct/100).rolling(60).sum())*100
    for window in [5, 60]:
        extra[f'selector_volatility_{window}d'] = pct.rolling(window).std()
    extra=pd.concat([raw[KEYS],right,left,extra],axis=1)
    extra=extra.loc[:,~extra.columns.duplicated(keep='first')]
    columns=[c for c in extra if c not in core or c in KEYS]
    return core.merge(extra[columns],on=KEYS,how='left',validate='one_to_one')


def jq_price_features(raw: pd.DataFrame, basic: pd.DataFrame, *, anchor: str = '2018-11-01') -> pd.DataFrame:
    raw = raw.loc[raw.date.ge(anchor)].copy()
    bars = pl.from_pandas(raw[['symbol', 'date', 'open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol']])
    bars = bars.with_columns(pl.col('date').cast(pl.Datetime('ns')))
    b = pl.from_pandas(basic[['symbol', 'date', 'turnover_rate']]).with_columns(pl.col('date').cast(pl.Datetime('ns')))
    bars = bars.join(b, on=KEYS, how='left').sort(KEYS)
    bars = bars.with_columns(((pl.col('close')/pl.col('pre_close')).cum_prod()*pl.col('pre_close').first()).over('symbol').alias('_continuous'),
                             (pl.col('pct_chg')/100).alias('ret'))
    bars = bars.with_columns((pl.col('_continuous')/pl.col('close')).alias('_scale'))
    bars = bars.with_columns(*[(pl.col(c)*pl.col('_scale')).alias(c) for c in ['open','high','low','close']],
                            (pl.col('vol')/pl.col('_scale')).alias('volume'))
    return price_factors(bars).to_pandas()


def index_features(raw: pd.DataFrame) -> pd.DataFrame:
    states = compute_index_ma_state_features(raw)
    activity = compute_key_index_active_market_value_features(raw).drop(columns='trade_date')
    result = states.merge(activity, on='date', validate='one_to_one')
    bench = raw.loc[raw.ts_code.eq('000300.SH')].sort_values('date').copy()
    for weeks, days in [(13, 63), (26, 126)]:
        bench[f'market_return_{weeks}w'] = bench.close.pct_change(days, fill_method=None)
    bench['market_volatility_13w'] = bench.close.pct_change(fill_method=None).rolling(63, min_periods=20).std()*np.sqrt(252)
    return result.merge(bench[['date', 'market_return_13w', 'market_return_26w', 'market_volatility_13w']], on='date', validate='one_to_one')


def market_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Whole-universe ranks; candidate selection is deliberately not an input."""
    bars = raw.sort_values(KEYS).copy()
    if bars.duplicated(KEYS).any():
        raise ValueError('Duplicate market keys')
    selector=bars.groupby('date').pct_chg.mean().rolling(20).mean().rename('selector_market_mean_20d')
    bars=bars.loc[bars.symbol.str.fullmatch(ASHARE)|bars.symbol.eq('302132.SZ')].copy()
    breadth = compute_market_breadth_features(bars.loc[bars.date.ge('2024-01-01')]).drop(columns='trade_date')
    bars['_ret'] = (bars.close / bars.pre_close - 1).where(bars.pre_close.gt(0))
    # Use the exact canonical action adjustment, including its numerical tie
    # behavior. Compounding daily returns can perturb cross-sectional ties.
    adjustment = bars.groupby('symbol',sort=False).close.shift(1) / bars.pre_close
    bars['_adjustment'] = adjustment.where(np.isfinite(adjustment)&adjustment.gt(0),1.)
    bars['_close'] = bars.close * bars.groupby('symbol',sort=False)._adjustment.cumprod()
    bars['return_120d'] = bars.groupby('symbol', sort=False)._close.pct_change(120, fill_method=None).astype('float32')
    # The original first observed return is unavailable for new listings.
    first = bars.groupby('symbol', sort=False).cumcount().eq(0)
    bars['_volret'] = bars.groupby('symbol',sort=False)._close.pct_change(fill_method=None).mask(first)
    bars['volatility_60d'] = bars.groupby('symbol', sort=False)._volret.transform(lambda s: s.rolling(60).std()*np.sqrt(252)).astype('float32')
    for col in ['return_120d', 'volatility_60d']:
        bars[f'{col}_cross_section_pct'] = bars.groupby('date')[col].rank(pct=True)
    bars['_limit_up'] = (bars._ret*100).ge(9.5)
    aggregate = bars.groupby('date').agg(n=('_ret','count'), up=('_limit_up','sum'), mean=('pct_chg','mean'))
    aggregate['market_sentiment_5d'] = (aggregate.up/aggregate.n.replace(0,np.nan)).rolling(5,min_periods=2).mean()
    aggregate['selector_market_mean_20d'] = selector
    market = breadth.merge(aggregate[['market_sentiment_5d','selector_market_mean_20d']].reset_index(), on='date', validate='one_to_one')
    return market, bars[KEYS+['return_120d_cross_section_pct','volatility_60d_cross_section_pct']]


def basic_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.duplicated(['ts_code','trade_date']).any():
        raise ValueError('Duplicate daily basic keys')
    derived = derive_daily_basic_features(raw)
    derived['symbol'] = derived.ts_code
    derived['date'] = pd.to_datetime(derived.trade_date)
    derived['db_total_mv_pct_rank'] = derived.groupby('date').total_mv_log.rank(pct=True)
    active = compute_stock_active_market_value_features(raw).rename(columns={'ts_code':'symbol'})
    return derived.merge(active.drop(columns='trade_date'), on=KEYS, validate='one_to_one')


def periodic_pb_features(keys: pd.DataFrame, raw: pd.DataFrame, *, calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """One weekly observation, available strictly after completed week end."""
    dates = pd.DatetimeIndex(raw.date.unique()).sort_values() if calendar is None else calendar
    chosen = pd.DataFrame({'date':dates}).groupby(dates.to_period('W-FRI')).tail(1).date
    history = raw.loc[raw.date.isin(chosen), ['symbol','date','pb']].copy()
    history['available'] = history.date.dt.to_period('W-FRI').dt.end_time.dt.normalize()
    history = history.sort_values(['symbol','available']).reset_index(drop=True)
    columns = []
    for years in [5,7]:
        name = f'pb_hist_percentile_{years}y'
        history[name] = history.pb.where(history.pb.gt(0)).groupby(history.symbol,sort=False).transform(
            lambda s:s.rolling(years*52,min_periods=104).rank(method='max',pct=True)*100)
        columns.append(name)
    return align_prior(keys, history[['symbol','available',*columns]], tolerance=14)


def align_prior(keys: pd.DataFrame, source: pd.DataFrame, *, tolerance: int = 7) -> pd.DataFrame:
    left = keys.assign(_row=np.arange(len(keys)), date=pd.to_datetime(keys.date).astype('datetime64[ns]'))
    right = source.copy(); right['available'] = pd.to_datetime(right.available).astype('datetime64[ns]')
    left['symbol']=left.symbol.astype('str');right['symbol']=right.symbol.astype('str')
    if right.duplicated(['symbol','available']).any():
        raise ValueError('Duplicate publication observation')
    result = pd.merge_asof(left.sort_values(['date','symbol']), right.sort_values(['available','symbol']),
                           left_on='date',right_on='available',by='symbol',allow_exact_matches=False,
                           direction='backward',tolerance=pd.Timedelta(days=tolerance))
    return result.sort_values('_row').drop(columns=['_row','available']).reset_index(drop=True)


def flow_features(raw: pd.DataFrame, kind: str) -> pd.DataFrame:
    fields = (['buy_sm_amount','sell_sm_amount','buy_md_amount','sell_md_amount','buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount','net_mf_amount']
              if kind=='moneyflow' else ['rzye','rqye','rzmre','rzche'])
    raw = raw.sort_values(['ts_code','trade_date']).reset_index(drop=True)
    duplicates = raw.loc[raw.duplicated(['ts_code','trade_date'],keep=False)]
    conflicts = duplicates.groupby(['ts_code','trade_date'])[fields].nunique(dropna=False).gt(1).any(axis=1)
    raw = raw.drop_duplicates(['ts_code','trade_date']).reset_index(drop=True)
    bad = pd.MultiIndex.from_frame(raw[['ts_code','trade_date']]).isin(conflicts.index[conflicts])
    raw.loc[bad,fields] = np.nan
    keys=raw[['ts_code','trade_date']].rename(columns={'trade_date':'date'});keys.date=pd.to_datetime(keys.date)
    builder=build_moneyflow_weekly if kind=='moneyflow' else build_margin_weekly
    values=builder(keys,raw).rename(columns={'ts_code':'symbol','date':'available'})
    if kind=='moneyflow':
        missing=raw[fields[:-1]].isna().any(axis=1)
        values.loc[(missing|raw.net_mf_amount.isna()).to_numpy(),'moneyflow_net_ratio']=np.nan
        invalid=missing.groupby(raw.ts_code,sort=False).transform(lambda s:s.rolling(3,min_periods=1).max()).gt(0)
        values.loc[invalid.to_numpy(),'large_net_3d_ratio']=np.nan
        return values[['symbol','available','moneyflow_net_ratio','large_net_3d_ratio']]
    values['margin_balance_change']=raw.groupby('ts_code',sort=False).rzye.pct_change(20,fill_method=None).to_numpy()
    return values[['symbol','available','margin_balance_change','short_balance']]


def top_list_features(keys: pd.DataFrame, events: pd.DataFrame, calendar: pd.DatetimeIndex,
                      available: pd.DatetimeIndex) -> pd.DataFrame:
    covered=pd.Series(calendar.isin(available).astype(int)).rolling(60,min_periods=60).sum().eq(60).to_numpy()
    events=events.copy();events['_pos']=calendar.searchsorted(events.date)
    groups={s:g.sort_values('_pos') for s,g in events.groupby('ts_code')}
    result=keys.copy(); result['top_list_net_ratio_20d']=np.nan
    for symbol,group in keys.groupby('symbol'):
        positions=calendar.searchsorted(group.date)-1
        values=np.zeros(len(group));e=groups.get(symbol)
        if e is not None:
            p=e._pos.to_numpy();r=np.searchsorted(p,positions,side='right');l=np.searchsorted(p,positions-19)
            net=e.net_amount.to_numpy(float);gross=e.amount.to_numpy(float)
            ns=np.r_[0,np.nancumsum(net)];gs=np.r_[0,np.nancumsum(gross)]
            denominator=gs[r]-gs[l];np.divide(ns[r]-ns[l],denominator,out=values,where=denominator!=0)
            bad=np.r_[0,np.cumsum(~np.isfinite(net)|~np.isfinite(gross))];values[bad[r]-bad[l]>0]=np.nan
        values[(positions<0)|~covered[np.maximum(positions,0)]]=np.nan
        result.loc[group.index,'top_list_net_ratio_20d']=values
    return result
