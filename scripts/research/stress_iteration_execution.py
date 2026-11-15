"""Execution sensitivity: entry lots and a conservative per-sleeve fee floor.

This is a scenario, not a broker fee schedule or full exchange execution model.
Total-return marks handle corporate actions; intraday queues remain unmodelled.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from evaluate_iteration_holdout import initialize
from select_iteration_applications import metrics, score_arm
from model_iteration_v2_protocol import OUT, BOUNDS
import validate_model_applications as app


def account(groups, capital, rate, start, end, symbols):
    cash = capital; held = {}; rows = []; pnl = []; skipped = 0; fees = 0.
    market = app._MARKET
    for day in range(start, end):
        for sid, p in list(held.items()):
            if day - p['entry'] < 5 or not market['sellable'][day, sid]:
                continue
            value = p['units'] * market['price'][day, sid]
            fee = max(5., value * rate / 2)
            cash += value - fee; fees += fee; pnl.append(value - fee - p['invested'])
            del held[sid]
        mark = sum(p['units'] * market['mark'][day, sid] for sid, p in held.items())
        nav_before = cash + mark
        proposals = [r for r in groups.get(day - 1, []) if int(r.sid) not in held][:min(20, max(0, 120 - len(held)))]
        if day - 1 > end - 11:
            proposals = []
        for row in proposals:
            sid = int(row.sid)
            if not market['buyable'][day, sid]:
                continue
            budget = min(cash, nav_before / 120)
            close = market['actual_close'][day, sid]
            lot = 200 if symbols[sid].startswith(('688', '689')) else 100
            quantity = int(max(0, min(budget - 5, budget / (1 + rate / 2))) / close / lot) * lot
            if quantity <= 0:
                skipped += 1; continue
            value = quantity * close; fee = max(5., value * rate / 2)
            invested = value + fee
            assert invested <= budget + 1e-7
            held[sid] = {'units': value / market['price'][day, sid], 'entry': day, 'invested': invested}
            cash -= invested; fees += fee
        value = sum(p['units'] * market['mark'][day, sid] for sid, p in held.items())
        nav = cash + value
        independent = capital + sum(pnl) + value - sum(p['invested'] for p in held.values())
        assert abs(nav - independent) <= 1e-6 and cash >= -1e-7
        rows.append({'nav': nav / capital, 'cash': cash / capital})
    frame = pd.DataFrame(rows)
    return frame, {'closed': len(pnl), 'lot_skips': skipped, 'fees_fraction': fees / capital,
                   'unresolved_fraction': value / capital}


def main():
    frame = initialize(); choice = json.loads((OUT / 'frozen_selection.json').read_text())
    start, end = [int(i) for i in app._CALENDAR.searchsorted(pd.to_datetime([BOUNDS['holdout_start'], BOUNDS['holdout_end']]))]
    symbols = json.loads((OUT / 'holdout_market_index.json').read_text())['symbols']
    groups = {}; specs = {p.name: p for p in app.policies()}
    for name in ['baseline', *choice['fallback']['weights']]:
        groups[name], _ = app.candidates(frame, specs[name], app._THRESHOLDS, '2026')
    for name, recipe in choice['recipes'].items():
        pred = pd.concat([pd.read_parquet(OUT / f"fresh_predictions_{s}_{recipe['arm']}.parquet").assign(side=s) for s in ['left', 'right']])
        scored = score_arm(frame, recipe['arm'], pred)
        if recipe['market_gate'] is not None:
            scored = scored.loc[scored.market_breadth_above_ma20_ratio.ge(recipe['market_gate'])]
        groups[name], _ = app.candidates(scored, app.Policy(gate=-np.inf if recipe['gate'] is None else recipe['gate']), app._THRESHOLDS, '2026')
    rows = []
    recipes = {'baseline': {'baseline': 1.}, 'v1': choice['fallback']['weights'], 'v2': choice['candidate']['weights']}
    for capital in [1_000_000, 5_000_000]:
        for rate in [.002, .006]:
            for name, weights in recipes.items():
                nav = cash = 0.; audits = []
                for sleeve, weight in weights.items():
                    path, audit = account(groups[sleeve], capital * weight, rate, start, end, symbols)
                    nav += weight * path.nav; cash += weight * path.cash; audits.append(audit)
                rows.append({'capital': capital, 'cost_bp': rate * 10000, 'name': name, **metrics(nav),
                             'average_exposure': float(np.mean(1 - cash / nav)),
                             'closed': sum(a['closed'] for a in audits), 'lot_skips': sum(a['lot_skips'] for a in audits),
                             'unresolved_fraction': sum(w * a['unresolved_fraction'] for w, a in zip(weights.values(), audits))})
    output = pd.DataFrame(rows)
    output.to_csv(OUT / 'execution_lot_fee_stress.csv', index=False)
    print(output.to_string(index=False))


if __name__ == '__main__':
    main()
