"""Open the new interval after selection; evaluate frozen recipes without tuning."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from quant.ml.rank_fusion import file_sha256
from model_iteration_v2_protocol import OUT, BOUNDS, TREE_ARMS, NEURAL_ARMS
from build_iteration_fresh_features import scan
from select_iteration_applications import metrics, score_arm
import validate_model_applications as app


def prepare_market():
    if not (OUT / 'frozen_selection.json').exists():
        raise RuntimeError('Holdout still sealed')
    if (OUT / 'holdout_market.npz').exists():
        return
    paths = sorted(p for p in (ROOT / 'data/raw/daily_partitioned').glob('*/*.parquet')
                   if '202401' <= p.parent.name.split('=')[-1] <= '202611')
    bars = scan(paths, ['symbol', 'date', 'open', 'high', 'low', 'close', 'pre_close', 'volume'])
    bars['date'] = pd.to_datetime(bars.date).astype('datetime64[ns]')
    bars = bars.loc[bars.date.lt(BOUNDS['holdout_end'])].sort_values(['symbol', 'date']).reset_index(drop=True)
    if bars.duplicated(['symbol', 'date']).any():
        raise ValueError('Duplicate holdout market prices')
    bars['price'] = (bars.close / bars.pre_close).groupby(bars.symbol).cumprod()
    symbols = sorted(bars.symbol.unique())
    calendar = pd.DatetimeIndex(sorted(bars.date.unique()))
    shape = (len(calendar), len(symbols))
    price, actual_close, gap = [np.full(shape, np.nan, dtype='float64') for _ in range(3)]
    observed, buyable, sellable = [np.zeros(shape, dtype=bool) for _ in range(3)]
    symbol_map = dict(zip(symbols, range(len(symbols))))
    for symbol, group in bars.groupby('symbol', sort=False):
        sid = symbol_map[symbol]
        days = calendar.get_indexer(group.date)
        valid = group.close.gt(0) & np.isfinite(group.close) & group.volume.gt(0)
        one_price = np.isclose(group.high, group.low, atol=1e-6, rtol=0)
        price[days, sid], actual_close[days, sid] = group.price, group.close
        observed[days, sid] = valid
        buyable[days, sid] = valid & ~(one_price & group.close.gt(group.pre_close))
        sellable[days, sid] = valid & ~(one_price & group.close.lt(group.pre_close))
        gap[days, sid] = group.open / group.pre_close - 1
    mark = pd.DataFrame(price).ffill().to_numpy()
    returns = price / np.roll(price, 1, axis=0) - 1
    returns[0] = np.nan
    np.savez_compressed(OUT / 'holdout_market.npz', price=price, mark=mark, actual_close=actual_close,
                        observed=observed, buyable=buyable, sellable=sellable, gap=gap, returns=returns)
    (OUT / 'holdout_market_index.json').write_text(json.dumps({'symbols': symbols,
        'calendar': [str(d.date()) for d in calendar],
        'raw_hashes': {str(p): file_sha256(p) for p in paths},
        'selection_sha256_before_opening': file_sha256(OUT / 'frozen_selection.json')}, indent=2))
    print('HOLDOUT MARKET READY', shape, flush=True)


def initialize():
    prepare_market()
    index = json.loads((OUT / 'holdout_market_index.json').read_text())
    app._CALENDAR = pd.DatetimeIndex(index['calendar'])
    symbol_map = dict(zip(index['symbols'], range(len(index['symbols']))))
    with np.load(OUT / 'holdout_market.npz') as source:
        app._MARKET = dict(source)
    app._THRESHOLDS = json.loads((app.OUT / 'thresholds.json').read_text())
    frames = []
    for side in ['left', 'right']:
        source = pd.read_parquet(OUT / f'fresh_inputs_{side}.parquet')
        frame = source[['symbol', 'date', 'side', 'base_score', 'volatility_20d', 'market_breadth_above_ma20_ratio',
                        'event_age', 'turnover_rate', 'volume_relative_20d', 'selector_return_20d']].copy()
        frame['sid'] = frame.symbol.map(symbol_map).astype(int)
        frame['day'] = app._CALENDAR.get_indexer(frame.date)
        frame['s_return'] = frame.base_score
        for arm in ['return', 'path_boundary', 'rank']:
            model = xgb.Booster()
            model.load_model(ROOT / f'reports/research/sample_label_research_v1/model_{side}_2026_{arm}.ubj')
            model.set_param({'nthread': 3})
            values = model.predict(xgb.DMatrix(source[model.feature_names].astype('float32'), nthread=3))
            cal = json.loads((ROOT / f'reports/research/sample_label_research_v1/calibrator_{side}_2026_{arm}.json').read_text())
            frame['e_' + arm] = np.interp(values, cal['x'], cal['y'])
            if arm != 'return':
                frame['s_' + arm] = pd.Series(values, index=frame.index).groupby(frame.date).rank(pct=True)
        frame['v_low_volatility'] = 1 - frame.groupby('date').volatility_20d.rank(pct=True)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_account(result, nav, trades, cost):
    sales = (trades.invested * (1 + trades.net) / (1 - cost / 2)).sum()
    invested = (result['turnover'] - sales) * (1 + cost / 2)
    reconstructed = trades.pnl.sum() + result['unresolved_position_value'] - (invested - trades.invested.sum())
    error = abs(reconstructed - result['return'])
    if error > 1e-9 or nav.cash.lt(-1e-10).any():
        raise ValueError('Holdout cash ledger audit failed')
    return error


def replay(frame, spec, start, end):
    groups, coverage = app.candidates(frame, spec, app._THRESHOLDS, '2026')
    result, nav, trades = app.simulate(spec, groups, app._MARKET, start, end, coverage)
    check_account(result, nav, trades, spec.cost)
    return {**result, **metrics(nav.nav)}, nav, trades


def mix(paths, results, weights):
    first = next(iter(paths.values()))
    nav = first[['day']].copy()
    for column in ['nav', 'cash']:
        nav[column] = sum(weights[n] * paths[n][column].to_numpy() for n in weights)
    stats = {**metrics(nav.nav), 'closed': sum(results[n]['closed'] for n in weights),
             'average_exposure': float(np.mean(1 - nav.cash / nav.nav)),
             'unresolved_position_value': sum(weights[n] * results[n]['unresolved_position_value'] for n in weights),
             'total_loss_stress_return': sum(weights[n] * results[n]['total_loss_stress_return'] for n in weights)}
    return stats, nav


def main():
    choice_path = OUT / 'frozen_selection.json'
    choice = json.loads(choice_path.read_text())
    selection_hash = file_sha256(choice_path)
    frame = initialize()
    frame.to_parquet(OUT / 'holdout_scores_base.parquet', index=False)
    start, end = app._CALENDAR.searchsorted(pd.to_datetime([BOUNDS['holdout_start'], BOUNDS['holdout_end']]))
    start, end = int(start), int(end)
    specs = {p.name: p for p in app.policies()}
    weights = choice['fallback']['weights']
    recipes = choice['recipes']
    scored = {}
    for arm in {r['arm'] for r in recipes.values()}:
        predictions = pd.concat([pd.read_parquet(OUT / f'fresh_predictions_{side}_{arm}.parquet').assign(side=side)
                                 for side in ['left', 'right']], ignore_index=True)
        scored[arm] = score_arm(frame, arm, predictions)
    rows = []
    exposure_fraction = json.loads((OUT / 'exposure_control.json').read_text())['baseline_initial_capital_fraction']
    for cost in [.002, .006]:
        baseline, baseline_nav, baseline_trades = replay(frame, replace(specs['baseline'], cost=cost), start, end)
        rows.append({'name': 'baseline', 'cost_bp': cost * 10000, **baseline})
        baseline_nav.to_parquet(OUT / f'holdout_nav_baseline_{int(cost * 10000)}bp.parquet', index=False)
        baseline_trades.to_parquet(OUT / f'holdout_trades_baseline_{int(cost * 10000)}bp.parquet', index=False)
        control = 1 - exposure_fraction + exposure_fraction * baseline_nav.nav
        rows.append({'name': 'baseline_cash_matched', 'cost_bp': cost * 10000, **metrics(control),
                     'average_exposure': float(np.mean(exposure_fraction * (baseline_nav.nav - baseline_nav.cash) / control))})
        paths, results = {}, {}
        for name in weights:
            result, nav, trades = replay(frame, replace(specs[name], cost=cost), start, end)
            paths[name], results[name] = nav, result
            trades.to_parquet(OUT / f'holdout_trades_{name}_{int(cost * 10000)}bp.parquet', index=False)
        old, old_nav = mix(paths, results, weights)
        rows.append({'name': 'v1_frozen_combination', 'cost_bp': cost * 10000, **old})
        old_nav.to_parquet(OUT / f'holdout_nav_v1_{int(cost * 10000)}bp.parquet', index=False)
        if choice['candidate']:
            paths, results = {}, {}
            for name, recipe in recipes.items():
                source = scored[recipe['arm']]
                if recipe['market_gate'] is not None:
                    source = source.loc[source.market_breadth_above_ma20_ratio.ge(recipe['market_gate'])]
                spec = app.Policy(name=name, gate=-np.inf if recipe['gate'] is None else recipe['gate'], cost=cost)
                result, nav, trades = replay(source, spec, start, end)
                paths[name], results[name] = nav, result
                rows.append({'name': 'component:' + name, 'cost_bp': cost * 10000, **result})
                nav.to_parquet(OUT / f'holdout_nav_{name}_{int(cost * 10000)}bp.parquet', index=False)
                trades.to_parquet(OUT / f'holdout_trades_{name}_{int(cost * 10000)}bp.parquet', index=False)
            candidate, candidate_nav = mix(paths, results, choice['candidate']['weights'])
            rows.append({'name': 'v2_preselected', 'cost_bp': cost * 10000, **candidate})
            candidate_nav.to_parquet(OUT / f'holdout_nav_v2_{int(cost * 10000)}bp.parquet', index=False)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'holdout_results.csv', index=False)
    acceptance = []
    for cost in [20, 60]:
        group = results.loc[results.cost_bp.eq(cost)].set_index('name')
        base = group.loc['baseline']
        for name in ['v1_frozen_combination', 'v2_preselected']:
            if name not in group.index:
                continue
            result = group.loc[name]
            acceptance.append({'name': name, 'cost_bp': cost, 'positive': bool(result['return'] > 0),
                               'return_better': bool(result['return'] > base['return']),
                               'drawdown_better': bool(result.max_drawdown < base.max_drawdown),
                               'return': result['return'], 'max_drawdown': result.max_drawdown})
    qualified = [r for r in acceptance if r['name'] == 'v2_preselected']
    use_new = len(qualified) == 2 and all(r['positive'] and r['return_better'] and r['drawdown_better'] for r in qualified)
    if file_sha256(choice_path) != selection_hash:
        raise RuntimeError('Frozen selection changed during holdout evaluation')
    decision = {'release': 'application_portfolio_v2' if use_new else 'application_portfolio_v1',
                'new_candidate_passed_positive_dual_both_costs': use_new, 'acceptance': acceptance,
                'selection_sha256': selection_hash, 'holdout_sessions': end - start,
                'reason': 'Frozen development candidate passed positive return and dual superiority at20/60bp.' if use_new else 'No preselected new candidate passed all holdout criteria; apply user-authorized v1 fallback.',
                'no_holdout_reselection': True, 'script_sha256': file_sha256(Path(__file__))}
    (OUT / 'deployment_decision.json').write_text(json.dumps(decision, indent=2))
    print(results[['name', 'cost_bp', 'return', 'max_drawdown', 'average_exposure']].to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    if '--market-only' in sys.argv:
        prepare_market()
    else:
        main()
