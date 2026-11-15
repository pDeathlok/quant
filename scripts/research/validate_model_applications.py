"""Frozen-model application experiments with next-close execution and a cash ledger."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/research')]
import research_sample_labels_v1 as prior
from quant.research.local_ranknet import percentile_scores

OUT = ROOT / 'reports/research/model_application_v1'
MODELS = ['return', 'good_path', 'path_boundary', 'drop_010', 'rank', 'rank_margin', 'quantile10', 'profit']
VARIABLES = ['turnover_rate', 'volatility_20d', 'volume_relative_20d', 'selector_return_20d',
             'market_breadth_above_ma20_ratio', 'event_age']
LEFT_SIGNALS = ['B1', 'SB1', 'SUPER_B1', 'LOW_PULLBACK']
RIGHT_SIGNALS = ['CHANGAN', 'PINGHANG', 'DOUBLE_GUN', 'GOLDEN_BOWL', 'BREATHING', 'KENGQI',
                 'ZAIHOU', 'YUEYUE', 'KEY_K', 'VIOLENCE_K', 'B2', 'B3', 'VEGAS', 'TRIPLE_VOLUME_BREAKOUT']


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@dataclass(frozen=True)
class Policy:
    name: str = 'baseline'
    family: str = 'baseline'
    score: str = 'return'
    scope: str = 'both'
    signal: str = ''
    filter: str = ''
    gate: float = -np.inf
    variable: str = ''
    overlap: str = 'right'
    positions: int = 120
    daily_entries: int = 20
    holding: int = 5
    weighting: str = 'equal'
    equal_sides: bool = False
    correlation_cap: bool = False
    buffer: bool = False
    exit_rule: str = ''
    gap_limit: float = np.inf
    cost: float = .002


def policies():
    base = Policy()
    specs = [base]
    for side in ['left', 'right', 'market_route']:
        specs.append(replace(base, name=f'scope_{side}', family='strategy', scope=side))
    for signal in ['L_' + s for s in LEFT_SIGNALS] + ['R_' + s for s in RIGHT_SIGNALS]:
        specs.append(replace(base, name=f'signal_{signal}', family='strategy', signal=signal))
    for mode in ['mean', 'best_expected', 'agreement']:
        specs.append(replace(base, name=f'overlap_{mode}', family='strategy', overlap=mode))
    for score in MODELS[1:] + ['return_path', 'return_rank', 'return_risk', 'three_way', 'consensus']:
        specs.append(replace(base, name=f'score_{score}', family='score', score=score))
    for filt in ['liquid', 'low_vol', 'first_event', 'breadth', 'liquid_low_vol']:
        specs.append(replace(base, name=f'filter_{filt}', family='variable', filter=filt))
    for variable in ['liquidity', 'low_volatility', 'momentum', 'reversal', 'volume']:
        specs.append(replace(base, name=f'weighted_{variable}', family='variable', variable=variable))
    for gate in [0., .002, .005, .01]:
        specs.append(replace(base, name=f'gate_{gate}', family='gate', gate=gate))
    for positions in [30, 60]:
        specs.append(replace(base, name=f'positions_{positions}', family='allocation', positions=positions))
    for count in [5, 10]:
        specs.append(replace(base, name=f'entries_{count}', family='allocation', daily_entries=count))
    specs.extend([
        replace(base, name='inverse_vol', family='allocation', weighting='inverse_vol'),
        replace(base, name='equal_sides', family='allocation', equal_sides=True),
        replace(base, name='correlation_cap', family='allocation', correlation_cap=True),
        replace(base, name='hold_buffer', family='turnover', buffer=True),
        replace(base, name='hold_3', family='exit', holding=3),
        replace(base, name='hold_10', family='exit', holding=10),
        replace(base, name='stop_take', family='exit', exit_rule='stop_take'),
        replace(base, name='trailing', family='exit', exit_rule='trailing'),
        replace(base, name='gap_3pct', family='execution', gap_limit=.03),
        replace(base, name='gap_5pct', family='execution', gap_limit=.05),
    ])
    # Predeclared, limited interaction grid. No test-dependent expansion.
    for score in ['return', 'return_path', 'three_way']:
        for filt in ['', 'liquid_low_vol', 'first_event', 'breadth']:
            for gate in [-np.inf, 0., .005]:
                if score == 'return' and not filt and gate == -np.inf:
                    continue
                specs.append(replace(base, name=f'combined_{score}_{filt or "all"}_{gate}',
                                     family='combined', score=score, filter=filt, gate=gate,
                                     weighting='inverse_vol', equal_sides=True))
    assert len({s.name for s in specs}) == len(specs)
    return specs


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    save('protocol.json', {
        'policies': [asdict(s) for s in policies()], 'folds': prior.FOLDS,
        'model_source': str(prior.OUT), 'selection': 'Validation return, at least100 closed positions. Cash if no positive candidate. Test choices frozen.',
        'account': 'Cash account, one position per symbol, fractional shares, fees at both ends, next-session close execution. Suspended/missing/one-price-down sales defer. No artificial proceeds from unknown exits.',
        'signal_tail': 'Stop new signals ten market sessions before each evaluation boundary for every policy. Still-held positions marked, not forcibly liquidated.',
        'exits': '3/5/10 market sessions; signal-close +5%/-3% or trailing5% triggers next close. Buffer can extend scheduled maturity to10 sessions when prior-close candidate stays within top40.',
        'risk': 'Correlation uses only60 prior sessions; >=20 joint returns; reject correlation>0.8. Historical industry membership not assumed.',
        'limitations': ['Previously explored historical periods, exploratory validation.',
            'Existing left/pure-right cohorts, not reconstructed full production universe.',
            'One-price limit proxies and closing prices, no queue or intraday fill model.',
            'Missing future prices retain last observed mark, with separate total-loss stress for unresolved positions.',
            'Fractional-share research account, no minimum commission or lot rounding.',
            'Per-fold research models are used for causal policy selection; current production artifacts are not used on their calibration periods.'],
    })
    raw_paths = json.loads((prior.OUT / 'raw_hashes.json').read_text())
    for path, expected in raw_paths.items():
        if prior.digest(path) != expected:
            raise ValueError(f'Raw vintage changed: {path}')
    columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'pre_close', 'volume']
    bars = pl.concat([pl.read_parquet(p, columns=columns) for p in raw_paths], how='vertical_relaxed').sort(['symbol', 'date'])
    bars = bars.with_columns((pl.col('close') / pl.col('pre_close')).cum_prod().over('symbol').alias('price'))
    symbols = sorted(bars['symbol'].unique().to_list())
    calendar = pd.DatetimeIndex(sorted(bars['date'].unique().to_list()))
    symbol_map = {symbol: i for i, symbol in enumerate(symbols)}
    shape = (len(calendar), len(symbols))
    price = np.full(shape, np.nan, dtype='float64')
    observed = np.zeros(shape, dtype=bool)
    buyable = observed.copy()
    sellable = observed.copy()
    gap = np.full(shape, np.nan, dtype='float32')
    for key, part in bars.partition_by('symbol', as_dict=True).items():
        sid = symbol_map[key[0]]
        days = calendar.get_indexer(pd.to_datetime(part['date'].to_numpy()))
        close, previous = part['close'].to_numpy(), part['pre_close'].to_numpy()
        one_price = np.isclose(part['high'].to_numpy(), part['low'].to_numpy(), atol=1e-6, rtol=0)
        valid = np.isfinite(close) & (close > 0) & (part['volume'].to_numpy() > 0)
        price[days, sid] = part['price'].to_numpy()
        observed[days, sid] = valid
        buyable[days, sid] = valid & ~(one_price & (close > previous))
        sellable[days, sid] = valid & ~(one_price & (close < previous))
        gap[days, sid] = part['open'].to_numpy() / previous - 1
    mark = pd.DataFrame(price).ffill().to_numpy()
    returns = price / np.roll(price, 1, axis=0) - 1
    returns[0] = np.nan
    np.savez_compressed(OUT / 'market.npz', price=price, mark=mark, buyable=buyable,
                        sellable=sellable, observed=observed, gap=gap, returns=returns)
    save('market_index.json', {'symbols': symbols, 'calendar': [d.isoformat() for d in calendar]})
    left_flags_path = ROOT / 'data/research/left_side_disclosure_gain_20261029/events.parquet'
    left_flags = pd.read_parquet(left_flags_path, columns=prior.KEYS + LEFT_SIGNALS)
    hashes = {'left_flags': prior.digest(left_flags_path)}
    fold_frames = {fold: [] for fold in prior.FOLDS}
    thresholds = {}
    for side in ['left', 'right']:
        source_path = prior.OUT / f'dataset_{side}.parquet'
        hashes[str(source_path)] = prior.digest(source_path)
        source = pd.read_parquet(source_path)
        if side == 'left':
            source = source.merge(left_flags, on=prior.KEYS, how='left', validate='one_to_one')
        for fold, bounds in prior.FOLDS.items():
            fit, _, _, _ = prior.slices(source, bounds)
            frame = source.loc[source.date.ge(bounds[1]) & source.date.lt(bounds[3])].copy()
            columns = prior.KEYS + VARIABLES
            thin = frame[columns].copy()
            thin['side'] = side
            thin['sid'] = thin.symbol.map(symbol_map).astype(int)
            thin['day'] = calendar.get_indexer(thin.date)
            for signal in LEFT_SIGNALS:
                thin['L_' + signal] = frame[signal].astype(float) if side == 'left' else 0.
            for signal in RIGHT_SIGNALS:
                thin['R_' + signal] = frame[signal].astype(float) if side == 'right' else 0.
            thresholds[f'{fold}_{side}'] = {
                'liquidity_floor': float(fit.turnover_rate.quantile(.1)),
                'volatility_ceiling': float(fit.volatility_20d.quantile(.9)),
                'volatility_floor': float(fit.volatility_20d.quantile(.1)),
            }
            for model_name in MODELS:
                path = prior.OUT / f'model_{side}_{fold}_{model_name}.ubj'
                model = xgb.Booster()
                model.load_model(path)
                raw = model.predict(prior.matrix(frame, model.feature_names))
                calibration = json.loads((prior.OUT / f'calibrator_{side}_{fold}_{model_name}.json').read_text())
                thin['s_' + model_name] = percentile_scores(frame.date, raw)
                thin['e_' + model_name] = np.interp(raw, calibration['x'], calibration['y'])
                hashes[str(path)] = prior.digest(path)
            for name, column in [('liquidity', 'turnover_rate'), ('low_volatility', 'volatility_20d'),
                                 ('momentum', 'selector_return_20d'), ('reversal', 'selector_return_20d'),
                                 ('volume', 'volume_relative_20d')]:
                value = thin.groupby('date')[column].rank(pct=True)
                thin['v_' + name] = 1 - value if name in ['low_volatility', 'reversal'] else value
            fold_frames[fold].append(thin)
    for fold, frames in fold_frames.items():
        combined = pd.concat(frames, ignore_index=True).sort_values(['date', 'symbol', 'side'])
        combined.to_parquet(OUT / f'scores_{fold}.parquet', index=False)
        save(f'cohort_{fold}.json', {'rows': len(combined), 'unique_events': len(combined.drop_duplicates(prior.KEYS)),
                                    'overlap_events': int(combined.duplicated(prior.KEYS).sum())})
    save('thresholds.json', thresholds)
    save('source_hashes.json', hashes)


def candidates(frame, spec, thresholds, fold):
    data = frame.copy()
    members = {'return_path': ['return', 'path_boundary'], 'return_rank': ['return', 'rank'],
               'return_risk': ['return', 'quantile10'], 'three_way': ['return', 'path_boundary', 'rank'],
               'consensus': ['return', 'path_boundary', 'rank']}.get(spec.score, [spec.score])
    data['score'] = data[['s_' + name for name in members]].mean(axis=1)
    data['expected'] = data[['e_' + name for name in members]].mean(axis=1)
    mask = pd.Series(True, index=data.index)
    if spec.score == 'consensus':
        mask &= data[['s_' + name for name in members]].min(axis=1).ge(.7)
    if spec.scope in ['left', 'right']:
        mask &= data.side.eq(spec.scope)
    if spec.scope == 'market_route':
        mask &= data.side.eq(np.where(data.market_breadth_above_ma20_ratio.ge(.5), 'right', 'left'))
    if spec.signal:
        mask &= data[spec.signal].eq(1)
    if spec.filter in ['liquid', 'liquid_low_vol']:
        floor = data.side.map({s: thresholds[f'{fold}_{s}']['liquidity_floor'] for s in ['left', 'right']})
        mask &= data.turnover_rate.ge(floor)
    if spec.filter in ['low_vol', 'liquid_low_vol']:
        ceiling = data.side.map({s: thresholds[f'{fold}_{s}']['volatility_ceiling'] for s in ['left', 'right']})
        mask &= data.volatility_20d.le(ceiling)
    if spec.filter == 'first_event':
        mask &= data.event_age.eq(0)
    if spec.filter == 'breadth':
        mask &= data.market_breadth_above_ma20_ratio.ge(.5)
    mask &= data.expected.ge(spec.gate)
    if spec.variable:
        data['score'] = .8 * data.score + .2 * data['v_' + spec.variable].fillna(.5)
    data = data.loc[mask].copy()
    if spec.overlap in ['mean', 'agreement']:
        groups = data.groupby(['day', 'sid'])
        data['score'] = groups.score.transform('mean')
        if spec.overlap == 'agreement':
            data = data.loc[groups.side.transform('size').eq(2)]
    order = ['day', 'sid', 'side']
    if spec.overlap == 'best_expected':
        order = ['day', 'sid', 'expected']
    data = data.sort_values(order).drop_duplicates(['day', 'sid'], keep='last')
    coverage = len(data) / max(1, len(frame.drop_duplicates(['day', 'sid'])))
    data = data.sort_values(['day', 'score', 'sid'], ascending=[True, False, True])
    groups = {int(day): group[['sid', 'score', 'volatility_20d', 'side']].to_records(index=False)
              for day, group in data.groupby('day', sort=True)}
    return groups, coverage


def correlated(sid, held, returns, day):
    if not held:
        return False
    window = returns[max(1, day - 59):day + 1]
    a = window[:, sid]
    for other in held:
        b = window[:, other]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() >= 20 and np.std(a[valid]) > 0 and np.std(b[valid]) > 0:
            if np.corrcoef(a[valid], b[valid])[0, 1] > .8:
                return True
    return False


def simulate(spec, groups, market, start, end, coverage):
    cash = 1.
    held = {}
    navs, trades = [], []
    entry_attempts = no_fills = delayed_sales = 0
    turnover = fees = exposure_sum = 0.
    days_without_positions = 0
    max_signal_day = end - 11
    for day in range(start, end):
        previous = day - 1
        ranked = groups.get(previous, [])
        ranks = {int(row.sid): i + 1 for i, row in enumerate(ranked)}
        # Orders determined at previous close. Trade availability is known only at execution.
        for sid, position in list(held.items()):
            age = day - position['entry_day']
            price_previous = market['mark'][previous, sid]
            return_previous = price_previous / position['entry_price'] - 1
            position['peak'] = max(position['peak'], price_previous)
            due = age >= spec.holding
            if spec.buffer and due and age < 10 and ranks.get(sid, 999999) <= 40:
                due = False
            if spec.exit_rule == 'stop_take' and age >= 1:
                due |= return_previous >= .05 or return_previous <= -.03
            if spec.exit_rule == 'trailing' and age >= 1:
                due |= price_previous < position['peak'] * .95
            if not due:
                continue
            if not market['sellable'][day, sid]:
                delayed_sales += 1
                continue
            value = position['shares'] * market['price'][day, sid]
            fee = value * spec.cost / 2
            cash += value - fee
            fees += fee
            turnover += value
            trades.append({'sid': sid, 'entry_day': position['entry_day'], 'exit_day': day,
                           'invested': position['invested'], 'net': (value - fee) / position['invested'] - 1,
                           'pnl': value - fee - position['invested'], 'age': age, 'side': position['side']})
            del held[sid]
        marked_value = sum(p['shares'] * market['mark'][day, sid] for sid, p in held.items())
        nav_before = cash + marked_value
        proposals = [row for row in ranked if int(row.sid) not in held]
        proposals = proposals[:min(spec.daily_entries, max(0, spec.positions - len(held)))]
        if previous > max_signal_day:
            proposals = []
        weights = np.ones(len(proposals))
        if spec.weighting == 'inverse_vol' and proposals:
            volatility = np.array([row.volatility_20d for row in proposals], dtype=float)
            volatility = np.nan_to_num(volatility, nan=.5)
            weights = 1 / np.clip(volatility, .1, 2.)
            weights = np.minimum(weights / weights.mean(), 2.)
        side_value = {s: sum(p['shares'] * market['mark'][day, sid] for sid, p in held.items() if p['side'] == s)
                      for s in ['left', 'right']}
        for row, weight in zip(proposals, weights):
            sid = int(row.sid)
            entry_attempts += 1
            if not market['buyable'][day, sid] or abs(market['gap'][day, sid]) > spec.gap_limit:
                no_fills += 1
                continue
            if spec.correlation_cap and correlated(sid, held, market['returns'], previous):
                continue
            budget = min(cash, nav_before / spec.positions * weight)
            if spec.equal_sides:
                budget = min(budget, max(0., nav_before / 2 - side_value[row.side]))
            if budget <= nav_before * 1e-8:
                continue
            fee = budget - budget / (1 + spec.cost / 2)
            value = budget - fee
            price = market['price'][day, sid]
            held[sid] = {'shares': value / price, 'entry_price': price, 'entry_day': day,
                         'invested': budget, 'peak': price, 'side': row.side}
            cash -= budget
            fees += fee
            turnover += value
            side_value[row.side] += value
        marked_value = sum(p['shares'] * market['mark'][day, sid] for sid, p in held.items())
        nav = cash + marked_value
        assert cash >= -1e-10 and np.isfinite(nav) and nav > 0
        exposure_sum += marked_value / nav
        days_without_positions += not held
        navs.append({'day': day, 'nav': nav, 'cash': cash, 'positions': len(held)})
    nav_frame = pd.DataFrame(navs)
    trade_frame = pd.DataFrame(trades, columns=['sid', 'entry_day', 'exit_day', 'invested', 'net', 'pnl', 'age', 'side'])
    values = nav_frame.nav.to_numpy()
    daily_returns = values / np.r_[1., values[:-1]] - 1
    peak = np.maximum.accumulate(np.r_[1., values])[1:]
    terminal_risk_value = sum(p['shares'] * market['mark'][end - 1, sid] for sid, p in held.items())
    metrics = {
        'return': values[-1] - 1, 'max_drawdown': float(np.min(values / peak - 1)),
        'sharpe': float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0.,
        'closed': len(trades), 'open': len(held), 'win_rate': float(trade_frame.net.gt(0).mean()) if len(trades) else None,
        'mean_holding_days': float(trade_frame.age.mean()) if len(trades) else None,
        'turnover': turnover, 'fees': fees, 'average_exposure': exposure_sum / len(values),
        'cash_days': days_without_positions / len(values), 'attempts': entry_attempts, 'no_fills': no_fills,
        'deferred_sale_days': delayed_sales, 'candidate_coverage': coverage,
        'unresolved_position_value': terminal_risk_value, 'total_loss_stress_return': values[-1] - terminal_risk_value - 1,
    }
    return metrics, nav_frame, trade_frame


_FRAME = _MARKET = _THRESHOLDS = _CALENDAR = _FOLD = None


def initialize(fold):
    global _FRAME, _MARKET, _THRESHOLDS, _CALENDAR, _FOLD
    _FOLD = fold
    _FRAME = pd.read_parquet(OUT / f'scores_{fold}.parquet')
    with np.load(OUT / 'market.npz') as source:
        _MARKET = {name: source[name] for name in source.files}
    _THRESHOLDS = json.loads((OUT / 'thresholds.json').read_text())
    _CALENDAR = pd.DatetimeIndex(json.loads((OUT / 'market_index.json').read_text())['calendar'])


def run_policy(spec):
    groups, coverage = candidates(_FRAME, spec, _THRESHOLDS, _FOLD)
    bounds = prior.FOLDS[_FOLD]
    rows = []
    for stage, begin, finish in [('validation', bounds[1], bounds[2]), ('test', bounds[2], bounds[3])]:
        start, end = _CALENDAR.searchsorted([pd.Timestamp(begin), pd.Timestamp(finish)])
        metrics, nav, trades = simulate(spec, groups, _MARKET, int(start), int(end), coverage)
        stem = f'{_FOLD}_{spec.name}_{stage}'
        nav.to_parquet(OUT / f'nav_{stem}.parquet', index=False)
        trades.to_parquet(OUT / f'trades_{stem}.parquet', index=False)
        rows.append({'fold': _FOLD, 'stage': stage, 'name': spec.name, 'family': spec.family, **metrics})
    return rows


def run():
    rows = []
    for fold in prior.FOLDS:
        with ProcessPoolExecutor(max_workers=4, initializer=initialize, initargs=(fold,)) as pool:
            for count, records in enumerate(pool.map(run_policy, policies(), chunksize=1)):
                rows.extend(records)
                pd.DataFrame(rows).to_csv(OUT / 'results.csv', index=False)
                if count % 10 == 0:
                    print('POLICIES', fold, count + 1, flush=True)
    result = pd.DataFrame(rows)
    choices = []
    for fold in prior.FOLDS:
        for family in ['all'] + sorted(result.family.unique().tolist()):
            eligible = result.loc[result.fold.eq(fold) & result.stage.eq('validation') & result.closed.ge(100) & result['return'].gt(0)]
            if family != 'all':
                eligible = eligible.loc[eligible.family.eq(family)]
            if eligible.empty:
                choices.append({'fold': fold, 'family': family, 'selected': 'cash', 'test_return': 0.})
                continue
            winner = eligible.sort_values(['return', 'name'], ascending=[False, True]).iloc[0]
            test = result.loc[result.fold.eq(fold) & result.stage.eq('test') & result.name.eq(winner['name'])].iloc[0]
            choices.append({'fold': fold, 'family': family, 'selected': winner['name'],
                            'validation_return': winner['return'], **{'test_' + key: test[key] for key in ['return', 'max_drawdown', 'closed', 'win_rate', 'turnover', 'average_exposure', 'open']}})
    pd.DataFrame(choices).to_csv(OUT / 'selected.csv', index=False)
    save('complete.json', {'policy_count': len(policies()), 'evaluations': len(rows), 'script_sha256': prior.digest(__file__)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        run()
