"""Round 2: confirmed first-wave structure + screenshot factors + B1 exit policies."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import runpy
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from quant.features.variable_library import build_continuous_ohlc
from quant.research.b1_wave_information import extract_wave_information, SCORE_COLUMNS
from quant.research.b1_backtest import ExitRule, simulate_exit

OUT = ROOT/'reports/research/b1_wave_round2_20261004'
PREV = ROOT/'reports/research/b1_mainline_leaders_20261004'
START, LAST_SIGNAL, END = pd.Timestamp('2025-05-06'), pd.Timestamp('2026-08-14'), pd.Timestamp('2026-08-31')
PRED = ROOT/'reports/research/left_side_unified_v3_group4_input_parity/test_predictions.parquet'


def design() -> dict:
    return dict(created_at=datetime.now(timezone.utc).isoformat(), status='research_only',
                periods={'earlier2025':'2025-05-06..2025-12-31', 'earlier2026':'2026-01-01..2026-03-31', 'later2026':'2026-04-01..2026-08-14 signal dates; account marked through2026-08-31'},
                mainline='unchanged ten-theme daily causal rule and prior-year-end membership; extend to Aug2026',
                structure='2-bar-confirmed pivot trough; first impulse within15bars and <=15% above trough; break prior10close high; no preceding5bar impulse; confirmed peak within20bars of trough; higher peak than prior confirmed high, pullback lows above trough; wave>=7% and directional efficiency>=.35',
                pullback='2..10bars, net decline/path length>=.5, max rebound<=5%, no second confirmed peak',
                dry='current<=1.2x post-impulse minimum and current/impulse<.8',
                scores='original eight-factor100 score; predefined volume-only40 scaled to100; model blend .8 daily model percentile + .2 screenshot percentile, fixed without fitting',
                entry='next market session close; no same-day sale; entry one-price bar excluded',
                exit={'stable':'TP8%, SL1.5% intraday stop-first, T+5 expiry', 'aggressive':'T+9 close expiry'},
                portfolio='<=20 selected entries/day, 1% current NAV budget each, max100 distinct held symbols; held duplicates skipped, no refill; 0.1% fee each side; cash retained',
                variants=['original','original_chart_blend','original_volume_blend','original_structure_priority','original_mainline_structure_priority','expanded_strict100','expanded_soft_volume','expanded_soft_mainline_volume'],
                restrictions='same signal/entry checks across groups; use archived ST/suspension where present, prior60price bars, no fabricated historical ST from current names',
                validation='rules frozen before this run; later2026 not used to set this round parameters, but was used in other project research; not globally untouched holdout')


def extend_mainlines(raw: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    universe = raw.loc[raw.symbol.isin(members.symbol)].sort_values(['symbol','date']).copy()
    universe['continuous'] = (universe.close/universe.pre_close).groupby(universe.symbol).cumprod()
    universe['above20'] = universe.continuous.gt(universe.groupby('symbol').continuous.transform(lambda s:s.rolling(20).mean()))
    universe['year'] = universe.date.dt.year
    joined = universe.merge(members[['year','symbol','index_code','index_name','snapshot_date']],on=['year','symbol'],how='inner')
    assert joined.snapshot_date.lt(joined.date).all()
    joined = joined.loc[joined.volume.gt(0)]
    panel = joined.groupby(['date','year','index_code','index_name']).agg(theme_return=('pct_chg','mean'),amount=('amount','sum'),breadth20=('above20','mean'),observed_members=('symbol','nunique')).reset_index()
    panel = panel.merge(members.groupby(['year','index_code']).symbol.nunique().rename('expected_members'),on=['year','index_code'])
    panel['member_coverage'] = panel.observed_members/panel.expected_members
    panel['theme_return'] /= 100
    panel['amount_share'] = panel.amount/panel.date.map(raw.groupby('date').amount.sum())
    market = pd.read_parquet(ROOT/'data/raw/index_000300.SH.parquet')
    market['date'] = pd.to_datetime(market.date if 'date' in market else market.trade_date.astype(str))
    market = market.sort_values('date').drop_duplicates('date').set_index('date')
    panel['market_ret20'] = panel.date.map(market.close.pct_change(20,fill_method=None))
    compute = runpy.run_path(str(ROOT/'scripts/research/analyze_b1_mainline_leaders.py'))['compute_mainlines']
    signals = compute(panel)
    for cut in ['2025-08-15','2026-03-31','2026-06-30']:
        prefix = compute(panel.loc[panel.date.le(cut)])
        pd.testing.assert_frame_equal(prefix[['date','index_code','mainline_score','mainline']],signals.loc[signals.date.le(cut),['date','index_code','mainline_score','mainline']].reset_index(drop=True))
    return signals


def signal_period(dates: pd.Series) -> pd.Series:
    return pd.Series(np.select([dates.le('2025-12-31'),dates.le('2026-03-31')],['earlier2025','earlier2026'],default='later2026'),index=dates.index)


def portfolio(events: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    entries = {d:g for d,g in events.groupby('entry_date',sort=False)}
    cash = 1.; held: dict[str,dict] = {}; curve = []; trades = []
    for date, marks in prices.iterrows():
        for symbol,pos in list(held.items()):
            mark = marks.get(symbol,np.nan)
            if np.isfinite(mark):pos['last'] = float(mark)
            if date == pos['exit_date']:
                proceeds = pos['qty']*pos['exit_price'];cash += proceeds*.999
                trades.append(dict(symbol=symbol,signal_date=pos['signal'],entry_date=pos['entry'],exit_date=date,net_return=proceeds*.999/pos['spent']-1,exit_type=pos['exit_type']))
                del held[symbol]
        value = cash+sum(p['qty']*p['last'] for p in held.values())
        budget = value/100
        for row in entries.get(date,pd.DataFrame()).itertuples():
            if row.symbol in held or len(held)>=100:continue
            price = marks.get(row.symbol,np.nan)
            if not np.isfinite(price) or price<=0:continue
            spent = min(budget,cash)
            if spent<1e-12:break
            cash -= spent
            held[row.symbol] = dict(qty=spent/(1.001*price),spent=spent,last=price,entry=date,signal=row.date,exit_date=row.exit_date,exit_price=row.exit_price,exit_type=row.exit_type)
        nav = cash+sum(p['qty']*p['last'] for p in held.values())
        assert cash>=-1e-10 and nav>0 and len(held)<=100
        curve.append(dict(date=date,nav=nav,exposure=1-cash/nav,positions=len(held)))
    # Unresolved positions stay marked in NAV; never erase their losses from the account.
    if held:curve[-1]['unresolved_positions'] = len(held)
    return pd.DataFrame(curve),pd.DataFrame(trades)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    specification = OUT/'frozen_design.json'
    if not specification.exists():specification.write_text(json.dumps(design(),ensure_ascii=False,indent=2))
    columns = ['symbol','date','open','high','low','close','pre_close','volume','amount','pct_chg']
    files = [p for p in sorted((ROOT/'data/raw/daily_partitioned').glob('year_month=*/data.parquet')) if '202401' <= p.parent.name.split('=')[1] <= '202608']
    raw = pd.concat([pd.read_parquet(p,columns=columns) for p in files],ignore_index=True)
    raw.date = pd.to_datetime(raw.date).dt.normalize()
    assert not raw.duplicated(['symbol','date']).any()
    calendar = np.sort(raw.date.unique())
    predictions = pd.read_parquet(PRED)
    baseline = predictions.loc[predictions.B1 & predictions.date.ge(START) & predictions.date.le(LAST_SIGNAL)].copy()
    waves = []; marks = []; causal_checks = 0; nonempty_prefix_checks = 0
    for n,(symbol,f) in enumerate(raw.groupby('symbol',sort=True),1):
        f = f.sort_values('date').reset_index(drop=True)
        w = extract_wave_information(f,START,LAST_SIGNAL)
        if len(w):waves.append(w)
        if len(w)>15 and causal_checks<10:
            for cutoff in ['2025-10-31','2026-03-31','2026-06-30']:
                prefix = extract_wave_information(f.loc[f.date.le(cutoff)],START,pd.Timestamp(cutoff))
                expected = w.loc[w.date.le(cutoff)].reset_index(drop=True)
                if prefix.empty and expected.empty:continue
                pd.testing.assert_frame_equal(prefix.reset_index(drop=True),expected)
                nonempty_prefix_checks += 1
            causal_checks += 1
        p = build_continuous_ohlc(f)
        q = pd.DataFrame({'symbol':symbol,'date':f.date,'price':p.close})
        marks.append(q.loc[q.date.between(START,END)])
        if n%1000==0:print(f'wave features {n}',flush=True)
    if not waves or causal_checks<10:raise RuntimeError('Insufficient causal checks/wave candidates')
    waves = pd.concat(waves,ignore_index=True)
    assert waves.peak_known_at.le(waves.date).all()
    assert waves.trough_date.lt(waves.impulse_date).all()
    members = pd.read_csv(PREV/'historical_members.csv');members.snapshot_date = pd.to_datetime(members.snapshot_date);members['year'] = members.snapshot_date.dt.year+1
    mainlines = extend_mainlines(raw,members)
    mainlines.to_parquet(OUT/'daily_mainlines.parquet',index=False)
    waves['year'] = waves.date.dt.year
    mapped = waves[['symbol','date','year']].merge(members[['year','symbol','index_code']],on=['year','symbol'],how='inner')
    counts = mapped.groupby(['date','index_code']).symbol.nunique().rename('candidate_count').reset_index()
    counts['rank'] = counts.groupby('date').candidate_count.rank(method='min',ascending=False)
    mapped = mapped.merge(counts,on=['date','index_code'])
    mapped['sector_score'] = np.where(mapped['rank'].eq(1),8,np.where(mapped['rank'].le(5),4,2))
    waves = waves.merge(mapped.groupby(['date','symbol']).sector_score.max().reset_index(),on=['date','symbol'],how='left')
    waves['sector_score'] = waves.sector_score.fillna(2)
    waves['score100'] = waves[SCORE_COLUMNS].sum(axis=1)+waves.sector_score
    waves['volume_score'] = waves[['dry_volume','impulse_energy','pullback_shape']].sum(axis=1)/40*100
    # Join mainline independently of wave presence, so original B1 candidates remain correctly classified.
    allkeys = pd.concat([baseline[['symbol','date']],waves[['symbol','date']]]).drop_duplicates();allkeys['year'] = allkeys.date.dt.year
    themes = allkeys.merge(members[['year','symbol','index_code']],on=['year','symbol'],how='inner')
    themes = themes.merge(mainlines[['date','index_code','mainline']],on=['date','index_code'],how='left')
    context = themes.groupby(['symbol','date']).mainline.any().reset_index()
    waves = waves.merge(context,on=['symbol','date'],how='left').fillna({'mainline':False})
    original = baseline.merge(waves.drop(columns=['year','mainline']),on=['symbol','date'],how='left',validate='one_to_one').merge(context,on=['symbol','date'],how='left')
    for name in ['strict_gate','soft_position_gate','mainline']:original[name] = original[name].fillna(False).astype(bool)
    for name in ['score100','volume_score']:original[name] = original[name].fillna(0)
    model_rank = original.groupby('date').pred_unified_left_long_task_deep.rank(pct=True)
    original['rank_original'] = model_rank
    original['rank_chart'] = .8*model_rank+.2*original.groupby('date').score100.rank(pct=True)
    original['rank_volume'] = .8*model_rank+.2*original.groupby('date').volume_score.rank(pct=True)
    original['structure_priority'] = original.soft_position_gate.astype(int)
    original['combined_priority'] = original.mainline.astype(int)*2+original.soft_position_gate.astype(int)
    waves.to_parquet(OUT/'wave_features.parquet',index=False)
    original.to_parquet(OUT/'original_candidate_context.parquet',index=False)
    keys = pd.concat([original[['symbol','date']],waves[['symbol','date']]]).drop_duplicates()
    # Point-in-time archive flags only; never use today's company name to label old ST status.
    archives = []
    for path in sorted((ROOT/'data/raw/tradability').glob('*.parquet')):
        d = pd.read_parquet(path)
        d['date'] = pd.to_datetime(d.trade_date.astype(str))
        archives.append(d[['ts_code','date','is_st','is_suspended']].rename(columns={'ts_code':'symbol'}))
    archive = pd.concat(archives,ignore_index=True).drop_duplicates(['symbol','date'],keep='last')
    signal_flags = archive.rename(columns={'is_st':'known_st','is_suspended':'known_suspended'})
    keys = keys.merge(signal_flags,on=['symbol','date'],how='left')
    futureframes = []
    keygroups = {s:k for s,k in keys.groupby('symbol',sort=False)}
    for symbol,f in raw.loc[raw.symbol.isin(keygroups)].groupby('symbol',sort=False):
        f = f.sort_values('date').reset_index(drop=True);p = build_continuous_ohlc(f)
        one_price = np.isclose(f.high,f.low,rtol=0,atol=1e-9) | f.volume.le(0)
        q = pd.DataFrame({'symbol':symbol,'date':f.date,'prior_bars':np.arange(len(f))})
        for day in range(1,17):
            q[f'date_t{day}'] = f.date.shift(-day)
            q[f'blocked_t{day}'] = pd.Series(one_price).shift(-day)
            for col in ['open','high','low','close']:q[f'{col}_t{day}'] = p[col].shift(-day)
        futureframes.append(keygroups[symbol].merge(q,on=['symbol','date'],how='left'))
    future = pd.concat(futureframes,ignore_index=True).copy()
    pos = np.searchsorted(calendar,future.date.to_numpy(),side='right')
    expected = calendar[np.minimum(pos,len(calendar)-1)]
    future['entry_date'] = future.date_t1
    future['entry_ok'] = (future.date_t1.eq(expected) & ~future.blocked_t1.fillna(True).astype(bool) & future.prior_bars.ge(60) & ~future.known_st.fillna(False).astype(bool) & ~future.known_suspended.fillna(False).astype(bool))
    entry_flags = archive.rename(columns={'date':'entry_date','is_st':'entry_known_st','is_suspended':'entry_known_suspended'})
    future = future.merge(entry_flags,on=['symbol','entry_date'],how='left',validate='many_to_one')
    future['entry_ok'] &= ~(future.entry_known_st.fillna(False)|future.entry_known_suspended.fillna(False)).astype(bool)
    execution_counts = dict(union_candidates=len(future),signal_archive_covered=int(future.known_st.notna().sum()),entry_archive_covered=int(future.entry_known_st.notna().sum()),entry_rejected=int((~future.entry_ok).sum()))
    # Primary entry matches frozen model's next-close task. Production exit parameters reused verbatim.
    future['entry_open'] = future.close_t1
    for day in range(2,17):
        blocked = future[f'blocked_t{day}'].fillna(True).astype(bool)
        for col in ['open','high','low','close']:future.loc[blocked,f'{col}_t{day}'] = np.nan
    policies = [ExitRule('stable','fixed',hold_days=4,take_profit=.08,stop_loss=.015),ExitRule('aggressive','expiry',hold_days=8)]
    selections = {}
    def choose(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return frame.sort_values(['date']+columns+['symbol'],ascending=[True]+[False]*len(columns)+[True]).groupby('date').head(20).copy()
    selections['original'] = choose(original,['rank_original'])
    selections['original_chart_blend'] = choose(original,['rank_chart'])
    selections['original_volume_blend'] = choose(original,['rank_volume'])
    selections['original_structure_priority'] = choose(original,['structure_priority','rank_original'])
    selections['original_mainline_structure_priority'] = choose(original,['combined_priority','rank_original'])
    selections['expanded_strict100'] = choose(waves.loc[waves.strict_gate],['score100'])
    selections['expanded_soft_volume'] = choose(waves.loc[waves.soft_position_gate],['volume_score'])
    selections['expanded_soft_mainline_volume'] = choose(waves.loc[waves.soft_position_gate],['mainline','volume_score'])
    prices = pd.concat(marks).pivot(index='date',columns='symbol',values='price').sort_index()
    curves = []; executed = []; metrics = []; all_events = []; bootstrap_rows = []
    # Compute outcomes once per policy across the union; do not filter rankings by realized outcomes.
    for policy in policies:
        result = simulate_exit(future,policy).rename(columns={'return_pct':'gross_pct'})
        unresolved = future.loc[~future.set_index(['symbol','date']).index.isin(result.set_index(['symbol','date']).index)]
        delayed = []
        unresolved_groups = {symbol:q for symbol,q in unresolved.groupby('symbol',sort=False)}
        for symbol,f in raw.loc[raw.symbol.isin(unresolved_groups)].groupby('symbol',sort=False):
            f = f.sort_values('date').reset_index(drop=True)
            p = build_continuous_ohlc(f)
            tradable = f.volume.gt(0) & ~np.isclose(f.high,f.low,rtol=0,atol=1e-9)
            for row in unresolved_groups[symbol].itertuples():
                expiry = getattr(row,f'date_t{policy.hold_days+1}')
                valid = f.date.ge(expiry) & f.date.le(END) & tradable if pd.notna(expiry) else pd.Series(False,index=f.index)
                if valid.any():
                    i = int(np.flatnonzero(valid)[0]); sale = float(p.close.iloc[i]); reason = 'delayed_expiry'
                    if policy.kind=='fixed':
                        stop = row.entry_open*(1-policy.stop_loss)
                        if p.low.iloc[i] <= stop:
                            sale = min(float(p.open.iloc[i]),stop); reason = 'delayed_stop_loss'
                        elif p.high.iloc[i] >= row.entry_open*(1+policy.take_profit):
                            sale = row.entry_open*(1+policy.take_profit); reason = 'delayed_take_profit'
                    delayed.append(dict(symbol=symbol,date=row.date,gross_pct=(sale/row.entry_open-1)*100,exit_date=f.date.iloc[i],exit_type=reason,exit_day=-1))
                else:
                    delayed.append(dict(symbol=symbol,date=row.date,gross_pct=np.nan,exit_date=pd.NaT,exit_type='unresolved',exit_day=-1))
        if delayed:result = pd.concat([result,pd.DataFrame(delayed)],ignore_index=True)
        result = result.merge(future[['symbol','date','entry_date','entry_open','entry_ok']],on=['symbol','date'],validate='one_to_one')
        result['exit_price'] = result.entry_open*(1+result.gross_pct/100)
        result['net_return'] = (1+result.gross_pct/100)*.999/1.001-1
        for name,selected in selections.items():
            ev = selected[['symbol','date']].merge(result,on=['symbol','date'],how='left',validate='one_to_one')
            ev = ev.loc[ev.entry_ok.fillna(False)&((ev.exit_date.le(END)&ev.exit_date.gt(ev.entry_date))|ev.exit_type.eq('unresolved'))].copy()
            ev['variant'] = name;ev['policy'] = policy.name;ev['period'] = signal_period(ev.date);all_events.append(ev)
            nav,trades = portfolio(ev,prices)
            nav['variant'] = name;nav['policy'] = policy.name;curves.append(nav)
            trades['variant'] = name;trades['policy'] = policy.name;executed.append(trades)
            for period in ['earlier2025','earlier2026','later2026','full']:
                dates = signal_period(nav.date)
                q = nav if period=='full' else nav.loc[dates.eq(period)]
                prior = nav.loc[nav.date.lt(q.date.min())].tail(1)
                initial = prior.nav.iloc[0] if len(prior) else 1.
                path = pd.Series(np.r_[initial,q.nav.to_numpy()]);dd = path/path.cummax()-1
                event = ev if period=='full' else ev.loc[ev.period.eq(period)]
                metrics.append(dict(policy=policy.name,variant=name,period=period,account_return=q.nav.iloc[-1]/initial-1,max_drawdown=-dd.min(),mean_exposure=q.exposure.mean(),events=len(event),avg_event_net=event.net_return.mean(),win_rate=event.net_return.dropna().gt(0).mean(),unresolved_events=int(event.exit_type.eq('unresolved').sum()),closed_trades=len(trades) if period=='full' else int(signal_period(trades.exit_date).eq(period).sum())))
    curves = pd.concat(curves,ignore_index=True);metrics = pd.DataFrame(metrics)
    rng = np.random.default_rng(20261004)
    for policy,g in curves.groupby('policy'):
        wide = g.pivot(index='date',columns='variant',values='nav')
        returns = wide.pct_change(fill_method=None);returns.iloc[0] = wide.iloc[0]-1
        for period in ['earlier2025','earlier2026','later2026']:
            q = returns.loc[signal_period(pd.Series(returns.index,index=returns.index)).eq(period)]
            for name in q.columns:
                delta = (q[name]-q.original).to_numpy();draws=[]
                for _ in range(1000):
                    starts = rng.integers(0,len(delta),size=int(np.ceil(len(delta)/20)))
                    ix = np.concatenate([(a+np.arange(20))%len(delta) for a in starts])[:len(delta)]
                    draws.append(delta[ix].mean())
                bootstrap_rows.append(dict(policy=policy,variant=name,period=period,mean_daily_excess=delta.mean(),ci_low=np.quantile(draws,.025),ci_high=np.quantile(draws,.975)))
    curves.to_csv(OUT/'portfolio_nav.csv',index=False)
    metrics.to_csv(OUT/'metrics.csv',index=False)
    pd.concat(executed,ignore_index=True).to_csv(OUT/'executed_trades.csv',index=False)
    pd.concat(all_events,ignore_index=True).to_parquet(OUT/'selected_events.parquet',index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUT/'bootstrap.csv',index=False)
    checks = [f'{causal_checks} stocks, {nonempty_prefix_checks} nonempty prefix checks for waves','3 cutoff checks for mainline','confirmed pivots available by signal date','rankings never use realized outcome','Top20 capacity','next-market-close entry','signal/entry archive flags when available','no entry on one-price bars; exits deferred on one-price bars','production exit parameters; stop before take-profit','nonnegative cash; max100 unique positions; unresolved holdings remain marked in NAV']
    assert not waves.duplicated(['symbol','date']).any()
    assert waves.score100.between(0,100).all()
    manifest = dict(status='success',frozen_design_hash=hashlib.sha256(specification.read_bytes()).hexdigest(),checks=checks,wave_rows=len(waves),strict_rows=int(waves.strict_gate.sum()),soft_rows=int(waves.soft_position_gate.sum()),execution_counts=execution_counts,
                    limitations=['historical ST/limit metadata archive coverage incomplete; one-price guard is conservative proxy, not full limit execution','ten themes fixed retrospectively; source historical constituents not independently archived','fixed .8/.2 blends and structure thresholds are new hypotheses; no hyperparameter optimization','later2026 not used for this round choice but seen in other project research','expanded candidates use standalone screenshot score, not newly trained production model','fees20bp round trip approximation, fractional shares and no impact costs','both price extremes same day use stop-first convention','NAV periods include positions carried from preceding period; signal event statistics grouped separately','delayed exits searched through full available data; unresolved positions stay marked, not erased'])
    manifest['hashes'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/quant/research/b1_wave_information.py',ROOT/'configs/strategies/b1_selected.yaml',PRED,PREV/'historical_members.csv']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(metrics.loc[metrics.period.isin(['full','later2026'])].to_string(index=False),flush=True)

if __name__ == '__main__':main()
