"""Reconstruct daily close marks for validation-selected research portfolios."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_sample_labels_v1 as experiment


def main():
    out = experiment.OUT
    paths = json.loads((out / 'raw_hashes.json').read_text())
    bars = pl.concat([pl.read_parquet(p, columns=['symbol', 'date', 'close', 'pre_close']) for p in paths], how='vertical_relaxed')
    bars = bars.sort(['symbol', 'date']).with_columns(
        (pl.col('close') / pl.col('pre_close')).cum_prod().over('symbol').alias('price'))
    prices = {key[0]: pd.Series(part['price'].to_numpy(), index=pd.to_datetime(part['date'].to_numpy()))
              for key, part in bars.partition_by('symbol', as_dict=True).items()}
    calendar = pd.DatetimeIndex(pd.read_parquet(out / 'calendar.parquet').date)
    selected = pd.read_csv(out / 'validation_selected.csv')
    reports = []
    for row in selected.itertuples():
        frame = pd.read_parquet(out / f'dataset_{row.side}.parquet')
        _, _, _, test = experiment.slices(frame, experiment.FOLDS[row.fold])
        start, end = experiment.FOLDS[row.fold][2:]
        days = calendar[(calendar >= start) & (calendar < end)]
        arms = [('return', 'fixed_top20'), ('return', 'validation_gate')]
        if row.fold == '2026':
            arms.append(('production_reference', 'fixed_top20'))
        if row.selected_arm != 'cash':
            arms.append((row.selected_arm, 'validation_gate'))
        for arm, policy in dict.fromkeys(arms):
            trades = pd.read_parquet(out / f'trades_{row.side}_{row.fold}_{arm}_{policy}.parquet')
            trades = trades.merge(test[experiment.KEYS + ['entry_date', 'maturity']], on=experiment.KEYS, validate='one_to_one')
            capital = np.ones(120) / 120
            available = np.full(120, np.datetime64('1900-01-01', 'ns'))
            pnl = np.zeros(len(days))
            mismatches = []
            unavailable_entry = 0
            for signal, group in trades.groupby('date', sort=True):
                entry = group.entry_date.dropna().iloc[0]
                free = list(np.flatnonzero(available <= entry.to_datetime64()))
                for trade in group.itertuples():
                    if not trade.filled:
                        continue
                    slot = free.pop(0)
                    notional = capital[slot]
                    assert abs(notional * trade.net - trade.contribution) < 1e-10
                    capital[slot] += trade.contribution
                    available[slot] = np.datetime64('2100-01-01') if trade.unknown else trade.maturity.to_datetime64()
                    series = prices[trade.symbol]
                    entry_price = series.get(entry, np.nan)
                    active = days >= entry
                    path_pnl = np.zeros(len(days))
                    if np.isfinite(entry_price):
                        marks = series.reindex(days, method='ffill').to_numpy()
                        path_pnl[active] = notional * (marks[active] / entry_price - 1 - .001)
                    else:
                        unavailable_entry += 1
                    if trade.unknown:
                        path_pnl[-1] = trade.contribution
                    else:
                        exit_price = series.get(trade.maturity, np.nan)
                        expected = exit_price / entry_price - 1 - .002
                        mismatches.append(abs(expected - trade.net))
                        path_pnl[days >= trade.maturity] = trade.contribution
                    pnl += path_pnl
            nav = pd.Series(1 + pnl, index=days)
            peak = nav.cummax().clip(lower=1.)
            assert abs(nav.iloc[-1] - capital.sum()) < 1e-10
            pd.DataFrame({'date': days, 'nav': nav.to_numpy()}).to_csv(out / f'nav_{row.side}_{row.fold}_{arm}_{policy}.csv', index=False)
            reports.append({'side': row.side, 'fold': row.fold, 'arm': arm, 'policy': policy,
                'terminal_return': nav.iloc[-1]-1, 'max_drawdown': (nav/peak-1).min(),
                'source_label_max_error': float(np.nanmax(mismatches)) if mismatches else 0.,
                'source_label_mismatch_count': int(np.sum(np.asarray(mismatches) > 1e-6)),
                'missing_entry_price': unavailable_entry,
                'interpretation': 'Current-vintage daily close marks; entry 10bp and exit additional10bp. Frozen label final proceeds override current close if revised. Unknown positions stressed -10% on final day.'})
    pd.DataFrame(reports).to_csv(out / 'nav_audit.csv', index=False)
    print(pd.DataFrame(reports).drop(columns='interpretation').to_string(index=False))


if __name__ == '__main__':
    main()
