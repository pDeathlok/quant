"""Choose a model/application recipe using development dates only."""
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from quant.ml.rank_fusion import file_sha256
from model_iteration_v2_protocol import OUT, TREE_ARMS, NEURAL_ARMS, GATES, MARKET_GATES, BOUNDS
import validate_model_applications as app
from validate_production_applications import initialize_production


def metrics(nav):
    values = np.asarray(nav)
    peaks = np.maximum.accumulate(np.r_[1., values])[1:]
    return {'return': float(values[-1] - 1), 'max_drawdown': float(-(values / peaks - 1).min())}


def objective(result, baseline):
    gain = result['return'] - baseline['return']
    risk = baseline['max_drawdown'] - result['max_drawdown']
    passed = result['return'] > 0 and gain > 0 and risk > 0 and result['closed'] >= 100
    return {**result, 'return_gain': gain, 'drawdown_reduction': risk,
            'positive_dual_pass': passed, 'selection_score': min(gain, risk)}


def score_arm(frame, arm, predictions):
    result = frame.merge(predictions, on=['symbol', 'date', 'side'], how='left', validate='one_to_one')
    if result.raw_score.isna().any() or result.expected_net.isna().any():
        raise ValueError(f'Missing predictions for {arm}')
    result['s_return'] = result.groupby(['date', 'side']).raw_score.rank(method='average', pct=True)
    result['e_return'] = result.expected_net
    return result.drop(columns=['raw_score', 'expected_net'])


def replay(frame, *, name, gate=None, market_gate=None, cost=.002):
    source = frame
    if market_gate is not None:
        source = source.loc[source.market_breadth_above_ma20_ratio.ge(market_gate)]
    spec = app.Policy(name=name, gate=-np.inf if gate is None else gate, cost=cost)
    groups, coverage = app.candidates(source, spec, app._THRESHOLDS, '2026')
    start, end = app._CALENDAR.searchsorted(pd.to_datetime([BOUNDS['calibration_end'], BOUNDS['selection_end']]))
    result, nav, trades = app.simulate(spec, groups, app._MARKET, int(start), int(end), coverage)
    return {**result, **metrics(nav.nav)}, nav, trades


def main():
    for path in ['tree_complete.json', 'neural_complete.json']:
        if not (OUT / path).exists():
            raise RuntimeError(f'Finish all predefined experiments before selection: {path}')
    frame, hashes = initialize_production()
    frame = frame.loc[frame.date.ge(BOUNDS['calibration_end']) & frame.date.lt(BOUNDS['selection_end'])].copy()
    baseline, base_nav, _ = replay(frame, name='baseline')
    base_nav.to_parquet(OUT / 'validation_nav_baseline.parquet', index=False)
    specs = {p.name: p for p in app.policies()}
    weights = json.loads((ROOT / 'configs/releases/application_portfolio_v1.json').read_text())['weights']
    old_paths = {}
    start, end = app._CALENDAR.searchsorted(pd.to_datetime([BOUNDS['calibration_end'], BOUNDS['selection_end']]))
    old_closed = 0
    for name, weight in weights.items():
        groups, coverage = app.candidates(frame, specs[name], app._THRESHOLDS, '2026')
        result, nav, _ = app.simulate(specs[name], groups, app._MARKET, int(start), int(end), coverage)
        old_paths[name] = nav
        old_closed += result['closed']
    old_nav = base_nav[['day']].copy()
    old_nav['nav'] = sum(weights[n] * old_paths[n].nav for n in weights)
    old_nav['cash'] = sum(weights[n] * old_paths[n].cash for n in weights)
    old_nav.to_parquet(OUT / 'validation_nav_v1.parquet', index=False)
    rows, paths, recipes = [], {}, {}
    arm_winners = []
    for arm in TREE_ARMS + NEURAL_ARMS + ['frozen_return_old_cal']:
        parts = []
        for side in ['left', 'right']:
            source_arm = 'calibrator_only' if arm == 'frozen_return_old_cal' else arm
            part = pd.read_parquet(OUT / f'predictions_{side}_{source_arm}.parquet')
            if arm == 'frozen_return_old_cal':
                part['expected_net'] = part.expected_frozen
            parts.append(part[['symbol', 'date', 'raw_score', 'expected_net']].assign(side=side))
        scored = score_arm(frame, arm, pd.concat(parts, ignore_index=True))
        arm_rows = []
        for gate in GATES:
            for market_gate in MARKET_GATES:
                name = f'{arm}__net_{gate}__breadth_{market_gate}'
                result, nav, _ = replay(scored, name=name, gate=gate, market_gate=market_gate)
                record = objective({'name': name, 'arm': arm, 'gate': gate, 'market_gate': market_gate, **result}, baseline)
                rows.append(record)
                arm_rows.append(record)
                paths[name] = nav
                recipes[name] = {'arm': arm, 'gate': gate, 'market_gate': market_gate}
                nav.to_parquet(OUT / f'validation_nav_{name}.parquet', index=False)
        qualified = [r for r in arm_rows if r['positive_dual_pass']]
        if qualified and arm != 'frozen_return_old_cal':
            arm_winners.append(max(qualified, key=lambda r: (r['selection_score'], r['return'], r['name'])))
        pd.DataFrame(rows).to_csv(OUT / 'validation_applications.csv', index=False)
        print('VALIDATED', arm, 'positive dual', len(qualified), flush=True)
    eligible = sorted(arm_winners, key=lambda r: (r['selection_score'], r['return'], r['name']), reverse=True)[:3]
    qualified_choices = []
    for winner in arm_winners:
        qualified_choices.append({'metrics': winner, 'weights': {winner['name']: 1.}})
    for count in [2, 3]:
        for members in combinations(eligible, count):
            names = [r['name'] for r in members]
            values = sum(paths[n].nav.to_numpy() / count for n in names)
            result = objective({'name': '+'.join(names), 'arm': 'capital_mix', **metrics(values),
                                'closed': sum(r['closed'] for r in members)}, baseline)
            rows.append(result)
            if result['positive_dual_pass']:
                qualified_choices.append({'metrics': result, 'weights': {n: 1 / count for n in names}})
    winner = max(qualified_choices, key=lambda c: (c['metrics']['selection_score'], c['metrics']['return'], c['metrics']['name'])) if qualified_choices else None
    choice = {'source': 'validation_only', 'bounds': BOUNDS,
        'candidate': winner, 'fallback': {'release_id': 'application_portfolio_v1', 'weights': weights},
        'baseline': baseline, 'v1': {**metrics(old_nav.nav), 'closed': old_closed},
        'recipes': {name: recipes[name] for name in winner['weights']} if winner else {},
        'production_artifact_hashes': hashes, 'protocol_sha256': file_sha256(OUT / 'protocol.json'),
        'script_sha256': file_sha256(Path(__file__)),
        'new_interval_outcomes_opened': False}
    pd.DataFrame(rows).to_csv(OUT / 'validation_applications.csv', index=False)
    (OUT / 'frozen_selection.json').write_text(json.dumps(choice, indent=2))
    print(json.dumps(choice, indent=2))


if __name__ == '__main__':
    main()
