"""Build an auditable next-session, flat-500k preview; never place orders.

Live inputs are isolated from frozen research. Quarterly models reuse the frozen
training vintage and must reproduce its predictions before scoring newer rows.
Run with --refresh to update Tushare inputs; otherwise use the isolated cache.
"""
import argparse
import json
import io
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits

from backtest_index_futures import ROOT, build_panel
from compare_index_futures_adaptation import mature_training
from evaluate_index_futures_lr import etf_features, ETFS
from test_index_futures_etf_filter import CACHE
from evaluate_index_futures_relative_indices import TARGETS, EXTERNAL, relative_features
from evaluate_index_futures_reversals import futures_features
from index_futures_state_models import add_regime_features, pipeline, OUT as STATE
from index_futures_regime import trend_states
from index_futures_state_rules import holding_limit
from evaluate_index_futures_risk_paths import minimum_equity
from refine_index_futures_adaptation import entry_candidates
from quant.features.index_panic import QVIX_MAP, option_features, price_stress

LIVE = ROOT / 'workspace_live'
SOURCES = LIVE / 'sources'
PROFILES = {'basic': (.0003, .15), 'stress': (.0006, .25)}
CAPITAL = 500000


def json_clean(value):
    return json.loads(pd.Series([value]).to_json(orient='values'))[0]


def refresh_sources():
    load_dotenv()
    SOURCES.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    # A closing bar is usable only after the closing session and publication buffer.
    cutoff = (now.date() if now.hour >= 18 else now.date()-timedelta(days=1)).strftime('%Y%m%d')
    end = (now.date()+timedelta(days=35)).strftime('%Y%m%d')
    session = requests.Session()

    def query(api, **params):
        response = session.post('https://api.tushare.pro', json={
            'api_name': api, 'token': os.environ['TUSHARE_TOKEN'], 'params': params}, timeout=40)
        response.raise_for_status()
        payload = response.json()
        if payload.get('code') != 0:
            raise RuntimeError(f'{api}: provider rejected request')
        frame = pd.DataFrame(payload['data']['items'], columns=payload['data']['fields'])
        if len(frame) >= 5000:
            raise RuntimeError(f'{api}: possible response truncation')
        return frame

    calendar = query('trade_cal', exchange='CFFEX', start_date='20190101', end_date=end)
    calendar = calendar.loc[calendar.is_open.eq(1)].sort_values('cal_date')
    latest = calendar.loc[calendar.cal_date.le(cutoff), 'cal_date'].max()
    if not isinstance(latest, str):
        raise RuntimeError('No complete session in exchange calendar')
    calendar.to_parquet(SOURCES/'calendar.parquet', index=False)

    def update(api, name, old, **params):
        path = SOURCES / name
        prior = pd.read_parquet(path) if path.exists() else old
        prior = prior.loc[prior.trade_date.le(latest)].copy()
        start = (pd.Timestamp(prior.trade_date.max())+pd.Timedelta(days=1)).strftime('%Y%m%d')
        if start <= latest:
            new = query(api, start_date=start, end_date=latest, **params)
            prior = pd.concat([prior, new], ignore_index=True)
        keys = ['trade_date']+(['ts_code'] if 'ts_code' in prior else [])
        if prior.duplicated(keys).any():
            raise RuntimeError(f'Duplicate source rows: {name}')
        prior.sort_values(keys).to_parquet(path, index=False)
        if prior.trade_date.max() != latest:
            raise RuntimeError(f'{name}: latest complete bar missing')
        return prior

    for symbol, code in TARGETS.items():
        contracts = query('fut_basic', exchange='CFFEX', fut_type='1', fut_code=symbol)
        contracts = contracts.loc[contracts.ts_code.str.fullmatch(rf'{symbol}\d{{4}}\.CFX')]
        contracts.to_parquet(SOURCES/f'{symbol}_contracts.parquet', index=False)
        old = pd.read_parquet(ROOT/f'{symbol}_daily.parquet')
        # One product spans only a handful of sessions here; query each live contract.
        parts = [old.loc[old.ts_code.eq(c)] for c in old.ts_code.unique()]
        active = contracts.loc[contracts.delist_date.ge(old.trade_date.max()) & contracts.list_date.le(latest)]
        for row in active.itertuples(index=False):
            prior = old.loc[old.ts_code.eq(row.ts_code)]
            start = (pd.Timestamp(prior.trade_date.max())+pd.Timedelta(days=1)).strftime('%Y%m%d') if len(prior) else row.list_date
            if start <= latest:
                parts.append(query('fut_daily', ts_code=row.ts_code, start_date=start, end_date=latest))
        bars = pd.concat(parts, ignore_index=True).sort_values(['trade_date','ts_code'])
        if bars.duplicated(['trade_date','ts_code']).any():
            raise RuntimeError('Duplicate futures bars')
        bars.to_parquet(SOURCES/f'{symbol}_daily.parquet', index=False)
        update('index_daily', f'{symbol}_spot.parquet', pd.read_parquet(ROOT/f'{symbol}_spot.parquet'), ts_code=code)
    for code in dict.fromkeys([*TARGETS.values(), *EXTERNAL.values()]):
        update('index_daily', f'index_{code}.parquet', pd.read_parquet(Path('data/raw')/f'index_{code}.parquet'), ts_code=code)
    for ticker in ETFS.values():
        code = f'{ticker}.SH'
        for api in ('fund_daily', 'fund_adj'):
            old = pd.concat(pd.read_parquet(p) for p in sorted((CACHE/api).glob(f'{code}_*.parquet')))
            update(api, f'{api}_{code}.parquet', old, ts_code=code)
    refresh_qvix()
    print('Live sources complete', latest, flush=True)


def refresh_qvix():
    """Refresh isolated vendor snapshots; never rewrite the frozen research source."""
    import akshare.index.index_option_qvix as qvix
    path = SOURCES/'qvix_audit.json'
    audit = {'retrieved_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
    try:
        raw = None
        for scheme in ('https','http'):
            try:
                response = requests.get(f'{scheme}://1.optbbs.com/d/csv/d/k.csv',timeout=(10,30))
                response.raise_for_status()
                candidate = pd.read_csv(io.BytesIO(response.content),encoding='gbk')
                if candidate.shape[1]<83 or len(candidate)<100:
                    raise ValueError('Unexpected QVIX dimensions')
                raw = candidate
                break
            except (requests.RequestException,ValueError):
                continue
        if raw is None:
            raise ValueError('QVIX source unavailable')
        original = qvix.__get_optbbs_daily
        qvix.__get_optbbs_daily = lambda:raw.copy()
        try:
            for name in set(QVIX_MAP.values()):
                frame = getattr(qvix,f'index_option_{name}_qvix')()
                frame['trade_date'] = pd.to_datetime(frame.date).dt.strftime('%Y%m%d')
                frame = frame.dropna(subset=['trade_date','close']).loc[lambda x:x.close.gt(0)]
                duplicated = frame.trade_date.duplicated(keep=False)
                frame.loc[duplicated].to_csv(SOURCES/f'quarantine_{name}.csv',index=False)
                frame.loc[~duplicated].sort_values('trade_date').to_parquet(SOURCES/f'qvix_{name}.parquet',index=False)
        finally:
            qvix.__get_optbbs_daily = original
        audit['status']='updated'
    except (requests.RequestException,ValueError,KeyError):
        audit['status']='unavailable_using_dated_cache'
    path.write_text(json.dumps(audit,ensure_ascii=False))


def qvix_series(name):
    path = SOURCES/f'qvix_{name}.parquet'
    if not path.exists():
        path = ROOT/f'panic_factors/sources/qvix_{name}.parquet'
    return pd.read_parquet(path).set_index('trade_date').close


def feature_panel():
    panel, _ = build_panel(source_root=SOURCES, include_latest=True)
    etfs = pd.concat(etf_features(symbol, ticker,
        bars=pd.read_parquet(SOURCES/f'fund_daily_{ticker}.SH.parquet'),
        adj=pd.read_parquet(SOURCES/f'fund_adj_{ticker}.SH.parquet')) for symbol,ticker in ETFS.items())
    bars = pd.concat(pd.read_parquet(SOURCES/f'{symbol}_daily.parquet') for symbol in TARGETS)
    future = futures_features(bars.set_index(['trade_date','ts_code']))
    prices = pd.concat([pd.read_parquet(SOURCES/f'index_{code}.parquet').set_index('trade_date').close.rename(code)
                       for code in dict.fromkeys([*TARGETS.values(),*EXTERNAL.values()])], axis=1).sort_index()
    panel = panel.merge(etfs, on=['symbol','trade_date'], validate='one_to_one').merge(
        future, on=['ts_code','trade_date'], validate='one_to_one').merge(
        relative_features(prices), on=['symbol','trade_date'], validate='one_to_one')
    states = pd.concat(trend_states(pd.read_parquet(SOURCES/f'{s}_spot.parquet')).assign(symbol=s) for s in TARGETS)
    panel = add_regime_features(panel, 'trade_date', states=states)
    panel = panel.merge(states[['trade_date','symbol','state']], on=['trade_date','symbol'], validate='one_to_one')
    spots = {s:pd.read_parquet(SOURCES/f'{s}_spot.parquet').set_index('trade_date').sort_index() for s in TARGETS}
    common_q = qvix_series('50etf')
    common = option_features(common_q, price_stress(spots['IH']).realized_vol20, spots['IH'].index)
    extras = []
    for s, spot in spots.items():
        own_q = qvix_series(QVIX_MAP[s])
        own = option_features(own_q, price_stress(spot).realized_vol20, spot.index)
        extras.append(common.add_prefix('panic_common_').join(own.add_prefix('panic_own_')).assign(symbol=s).reset_index())
    panel = panel.merge(pd.concat(extras), on=['trade_date','symbol'], validate='one_to_one')
    return panel.replace([np.inf,-np.inf], np.nan)


def fit_forecasts(test, quarter):
    old = pd.read_parquet(ROOT/'adaptation_comparison/quarterly_predictions.parquet')
    part = old.loc[old.quarter.eq(quarter)]
    if part.empty:
        raise RuntimeError('Quarter not trained in frozen research; a new mature-label training vintage is required')
    cutoff = part.trade_date.min()
    sets = json.loads((ROOT/'relative_indices/feature_sets.json').read_text())
    # Frozen rows are an independent prediction parity check of this serving path.
    check = part.loc[part.trade_date.eq(part.trade_date.max())].copy()
    errors = []
    for horizon in (20,60,'win'):
        maturity = 'profit_end' if horizon == 'win' else f'direction_{horizon}_end'
        features = sets['combined'] if horizon == 'win' else sets['base']
        train = mature_training(old, cutoff, maturity)
        age = (pd.Timestamp(cutoff)-pd.to_datetime(train[maturity])).dt.days.to_numpy()
        weights = np.exp2(-age/365.25); weights /= weights.mean()
        for side, direction in (('long',1),('short',-1)):
            label = train[f'profit_{side}'].astype(int) if horizon == 'win' else train[f'direction_{horizon}'].eq(direction).astype(int)
            model = pipeline(LogisticRegression(C=.1,max_iter=1000))
            model.fit(train[features],label,standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
            key = f'win_{side}' if horizon == 'win' else f'{side}_{horizon}'
            test[key] = model.predict_proba(test[features])[:,1]
            errors.append(float(abs(model.predict_proba(check[features])[:,1]-check[key]).max()))
    entries = pd.read_parquet(STATE/'entry_dataset.parquet').merge(
        pd.read_parquet(ROOT/'panic_factors/features.parquet'), on=['trade_date','symbol'], validate='many_to_one')
    base = json.loads((STATE/'feature_sets.json').read_text())['entry']['interaction']
    qfeatures = json.loads((ROOT/'panic_factors/feature_sets.json').read_text())['qvix']
    historical = pd.read_parquet(ROOT/'panic_factors/forecasts.parquet')
    predictions = []
    begin = (pd.Timestamp(cutoff)-pd.DateOffset(months=36)).strftime('%Y%m%d')
    for direction in (1,-1):
        train = entries.loc[entries.direction.eq(direction)&entries.trade_date.ge(begin)&entries.maturity.lt(cutoff)]
        age = (pd.Timestamp(cutoff)-pd.to_datetime(train.maturity)).dt.days.to_numpy()
        weights = np.exp2(-age/365.25); weights /= weights.mean()
        for bps in (3,6):
            result = test.copy().assign(direction=direction,bps=bps,fallback='none')
            check = entries.loc[entries.direction.eq(direction)&entries.trade_date.eq(part.trade_date.max())]
            checked = None
            for name, features in (('baseline',base),('qvix',qfeatures)):
                lr = pipeline(LogisticRegression(C=.1,max_iter=1000))
                ridge = pipeline(Ridge(alpha=.1*len(train)))
                lr.fit(train[features],train[f'net_{bps}'].gt(0).astype(int),standardscaler__sample_weight=weights,logisticregression__sample_weight=weights)
                ridge.fit(train[features],train[[f'net_{bps}',f'risk_{bps}']],standardscaler__sample_weight=weights,ridge__sample_weight=weights)
                def predict(frame):
                    raw = ridge.predict(frame[features]); raw[:,1] = np.maximum(0,raw[:,1])
                    return np.column_stack([lr.predict_proba(frame[features])[:,1],raw])
                values, check_values = predict(test), predict(check)
                if name == 'baseline':
                    fallback, checked = values.copy(), check_values.copy()
                else:
                    missing = test.panic_common_missing.ne(0).to_numpy()
                    values[missing] = fallback[missing]
                    result.loc[missing,'fallback'] = 'baseline_missing_common'
                    mask = check.panic_common_missing.ne(0).to_numpy()
                    check_values[mask] = checked[mask]
                    expected = historical.loc[historical.model.eq('qvix')&historical.bps.eq(bps)&historical.direction.eq(direction)].set_index(['entry_date','symbol']).reindex(pd.MultiIndex.from_frame(check[['entry_date','symbol']]))
                    errors.append(float(abs(check_values-expected[['probability','pred_net','pred_risk']].to_numpy()).max()))
                    result[['probability','pred_net','pred_risk']] = values
            predictions.append(result)
    if not np.isfinite(errors).all() or max(errors)>1e-8:
        raise RuntimeError(f'Serving model parity failed: {max(errors)}')
    return pd.concat(predictions,ignore_index=True), dict(quarter=quarter,cutoff=cutoff,max_parity_error=max(errors))


def evaluate_candidates(predictions, policy, fee, margin, capital=CAPITAL):
    """Known closing price is an estimate; actual next open must be rechecked."""
    one = predictions.drop_duplicates('symbol').copy()
    one['entry_open'] = one.close
    raw = {symbol for _,symbol,_ in entry_candidates(list(one.itertuples(index=False)),policy,capital,fee,margin,funded=False)}
    result = []
    for row in one.itertuples(index=False):
        direction = 1 if row.long_60>row.short_60 else -1
        evidence = predictions.loc[predictions.symbol.eq(row.symbol)&predictions.direction.eq(direction)].iloc[0]
        confidence = max(row.long_60,row.short_60)
        required = minimum_equity(row,direction,fee,margin)
        gates = dict(direction=row.symbol in raw,probability=bool(evidence.probability>=.55),
                     utility=bool(evidence.pred_net-.5*evidence.pred_risk>0),capital=bool(capital>=required))
        result.append(dict(symbol=row.symbol,contract=row.ts_code,side='long' if direction==1 else 'short',
            price=row.close,signal_date=row.trade_date,confidence=confidence,long_score=row.long_60,short_score=row.short_60,
            probability=evidence.probability,pred_net=evidence.pred_net,pred_risk=evidence.pred_risk,
            utility=evidence.pred_net-.5*evidence.pred_risk,minimum_equity=required,margin_cash=row.close*row.multiplier*margin,
            capital_gap=max(0,required-capital),max_entry_price=capital/required*row.close,
            state=int(row.state),holding_limit=holding_limit(int(row.state),direction),fallback=evidence.fallback,
            qvix_common_date=row.panic_common_source_date,qvix_own_date=row.panic_own_source_date,
            gates=gates,passes=bool(all(gates.values()) and np.isfinite([evidence.probability,evidence.pred_net,evidence.pred_risk,required]).all())))
    return sorted(result,key=lambda r:(-r['confidence'],r['symbol']))


def build():
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    panel = feature_panel()
    calendar = pd.read_parquet(SOURCES/'calendar.parquet').cal_date.sort_values()
    cutoff = (now.date() if now.hour>=18 else now.date()-timedelta(days=1)).strftime('%Y%m%d')
    expected = calendar.loc[calendar.le(cutoff)].max()
    latest = panel.groupby('symbol').trade_date.max().min()
    next_date = calendar.loc[calendar.gt(latest)].min()
    if not isinstance(next_date,str):
        raise RuntimeError('Exchange calendar must be refreshed')
    quarter = str(pd.Timestamp(next_date).to_period('Q'))
    sample = panel.loc[panel.trade_date.eq(latest)].copy()
    features = json.loads((ROOT/'relative_indices/feature_sets.json').read_text())['combined']
    if sample.symbol.nunique()!=4 or sample[features].isna().any().any():
        raise RuntimeError('Incomplete current futures/ETF/relative features')
    with threadpool_limits(limits=1):
        forecasts, audit = fit_forecasts(sample,quarter)
    frozen_calendar = pd.read_csv(STATE/'global_calendar.csv',dtype={'date':str})
    dates = calendar.loc[calendar.ge(frozen_calendar.date.min())].tolist()
    actual = [int(i%5==0) for i in range(len(dates))]
    mapping = dict(zip(dates,actual))
    assert all(bool(mapping[r.date])==bool(r.entry_due) for r in frozen_calendar.itertuples(index=False))
    due = bool(mapping[next_date])
    next_check = next(d for d in dates if d>=next_date and mapping[d])
    from evaluate_index_futures_panic import PanicReplay, Selector
    selector = Selector(PanicReplay(), None)
    profiles = {}
    for name,(fee,margin) in PROFILES.items():
        pred = forecasts.loc[forecasts.bps.eq(round(fee*10000))]
        scores, _ = selector.score(latest,fee,margin,float(CAPITAL),'qvix')
        policy, detail = selector.choose(latest,fee,margin,float(CAPITAL),'qvix','base',initial=True)
        profiles[name] = {}
        for mode,chosen in [('fixed_qvix','base'),('rolling_qvix',policy)]:
            candidates = evaluate_candidates(pred,chosen,fee,margin)
            eligible = next((r for r in candidates if r['passes']),None)
            fresh = latest == expected
            status = 'stale' if not fresh else 'calendar_wait' if not due else 'no_signal' if eligible is None else 'conditional'
            profiles[name][mode] = dict(status=status,policy=chosen,selection=scores if mode.startswith('rolling') else {},
                selection_reason=detail['reason'] if mode.startswith('rolling') else 'fixed_base',
                candidates=candidates,candidate=eligible,entry_due=due,next_check=next_check)
    snapshot = dict(version=1,generated_at=now.isoformat(),signal_date=latest,expected_session=expected,
        execution_date=next_date,valid_until=f'{next_date[:4]}-{next_date[4:6]}-{next_date[6:]}T09:30:00+08:00',
        capital=CAPITAL,max_lots=1,account='new_flat',model=audit,profiles=profiles,
        qvix_refresh=json.loads((SOURCES/'qvix_audit.json').read_text()) if (SOURCES/'qvix_audit.json').exists() else {'status':'dated_cache'},
        history_end=max(s.get('last_history_end','') for s in scores.values()),
        note='Closing-price preview only; no live holdings. Recheck actual opening price, broker margin and current equity before execution.')
    LIVE.mkdir(exist_ok=True)
    forecasts.to_parquet(LIVE/'forecasts.parquet',index=False)
    target = LIVE/'signal.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(json_clean(snapshot),ensure_ascii=False,allow_nan=False,indent=2))
    temporary.replace(target)
    print('Signal snapshot',latest,'next',next_date,'due',due,'parity',audit['max_parity_error'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    import fcntl
    LIVE.mkdir(parents=True,exist_ok=True)
    with (LIVE/'refresh.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.refresh: refresh_sources()
        build()
        if args.publish:
            from build_index_futures_workspace import main
            main()
