"""Independently reproduce signal ablation calibration, choices and returns."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/research')]
from iterate_right_signal_identity import OUT, SOURCE, V3, FOLDS, ARMS
from evaluate_full_rank_comparison import calendar_for
from evaluate_ranknet_cash_policy import evaluate_policy
from quant.research.local_ranknet import percentile_scores
from quant.ml.rank_fusion import file_sha256


def main() -> None:
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert file_sha256(SOURCE / 'dataset_right.parquet') == protocol['source_sha256']
    assert file_sha256(ROOT / 'scripts/research/iterate_right_signal_identity.py') == protocol['script_sha256']
    assert json.loads((V3 / 'verification.json').read_text())['passed']
    frame = pd.read_parquet(SOURCE / 'dataset_right.parquet').sort_values(['date', 'symbol']).reset_index(drop=True)
    assert not frame.duplicated(['symbol', 'date']).any()
    assert frame.source_date.lt(frame.date).all()
    signals = protocol['signals']
    assert frame[signals].isna().sum().astype(int).to_dict() == protocol['signal_missing_counts']
    frame[signals] = frame[signals].astype('float32')
    results = pd.read_csv(OUT / 'results.csv')
    training = pd.DataFrame(json.loads((OUT / 'training.json').read_text()))
    hashes = {}
    summaries = []
    for fold, (fit_end, selection, test_start, test_end) in FOLDS.items():
        fit = frame.loc[frame.date.lt(fit_end) & frame.maturity.lt(fit_end) & frame.outcome.notna()]
        cal = frame.loc[frame.date.ge(fit_end) & frame.date.lt(selection) & frame.maturity.lt(selection) & frame.outcome.notna()]
        assert fit.maturity.max() < pd.Timestamp(fit_end)
        assert cal.maturity.max() < pd.Timestamp(selection)
        pieces = []
        for start, end in [(selection, test_start), (test_start, test_end)]:
            part = frame.loc[frame.date.ge(start) & frame.date.lt(end) & frame.planned_maturity.lt(end)].copy()
            part.loc[part.maturity.ge(end), 'outcome'] = np.nan
            pieces.append(part)
        grid = pd.read_csv(OUT / f'grid_{fold}.csv')
        original_grid = pd.read_csv(V3 / f'grid_{fold}.csv')
        validation = {row['name']: {k.removeprefix('validation_'): v for k, v in row.items() if k.startswith('validation_')}
                      for row in original_grid.to_dict('records')}
        replayed_picks = {}
        for arm in ARMS:
            path = OUT / f'{fold}_{arm}.ubj'
            model = xgb.Booster()
            model.load_model(path)
            model.set_param({'nthread': 2})
            row = training.loc[training.fold.eq(fold) & training.arm.eq(arm)].iloc[0]
            assert file_sha256(path) == row.sha256
            assert len(model.feature_names) == row.features
            assert set(signals) <= set(model.feature_names)
            hashes[path.name] = row.sha256
            raw = [model.predict(xgb.DMatrix(p[model.feature_names].astype('float32'), nthread=2), output_margin=True)
                   for p in [cal, *pieces]]
            iso = IsotonicRegression(out_of_bounds='clip').fit(raw[0], cal.outcome.clip(-.5, .5), sample_weight=1/cal.groupby('date').date.transform('size'))
            saved = json.loads((OUT / f'{fold}_{arm}_calibration.json').read_text())
            np.testing.assert_array_equal(iso.X_thresholds_, saved['x'])
            np.testing.assert_array_equal(iso.y_thresholds_, saved['y'])
            predictions = pd.read_parquet(OUT / f'{fold}_{arm}_predictions.parquet')
            pd.testing.assert_frame_equal(predictions[['symbol', 'date']].reset_index(drop=True), pieces[1][['symbol', 'date']].reset_index(drop=True))
            np.testing.assert_array_equal(raw[2], predictions.score)
            for item in grid.loc[grid.name.str.startswith(arm+'#')].to_dict('records'):
                threshold = float(item['name'].split('#')[1])
                for index, (prefix, start, end) in enumerate([('validation', selection, test_start), ('test', test_start, test_end)]):
                    part = pieces[index]
                    stats, _, picks = evaluate_policy(part, iso.predict(raw[index+1]), threshold, calendar_for(frame, start, end), 1,
                                                   rank_score=percentile_scores(part.date, raw[index+1]))
                    if prefix == 'test':
                        replayed_picks[item['name']] = picks
                    else:
                        validation[item['name']] = stats
                    for key in ['capital_pnl', 'attempts', 'win_rate', 'unknown']:
                        assert np.isclose(stats[key], item[f'{prefix}_{key}'], equal_nan=True)
        for result in results.loc[results.fold.eq(fold)].itertuples():
            family, mode = result.policy.rsplit('_', 1)
            names = list(validation) if family == 'combined_v3' else [n for n in validation if n.startswith(family+'#')]
            eligible = [n for n in names if validation[n]['capital_pnl'] > 0 and validation[n]['attempts'] >= 100
                        and (mode != 'win' or validation[n]['win_rate'] >= .55)]
            chosen = 'cash' if not eligible else max(eligible, key=lambda n: (
                (validation[n]['win_rate'], validation[n]['capital_pnl']) if mode == 'win'
                else (validation[n]['capital_pnl'], validation[n]['win_rate'])))
            assert result.selected == chosen, (fold, result.policy, result.selected, chosen)
            if chosen == 'cash':
                assert result.capital_pnl == 0 and result.attempts == 0
                filled = 0
                filled_win = np.nan
            else:
                selected = grid.loc[grid.name.eq(chosen)].iloc[0]
                for key in ['capital_pnl', 'attempts', 'win_rate', 'unknown']:
                    assert np.isclose(getattr(result, key), selected[f'test_{key}'], equal_nan=True)
                if chosen in replayed_picks:
                    held = replayed_picks[chosen].loc[lambda p: ~p.unfilled]
                    filled, filled_win = len(held), held.outcome.gt(0).mean()
                else:
                    # This winner is a v3 arm whose picks were already independently replayed.
                    previous = pd.read_csv(V3 / 'results.csv')
                    old = previous.loc[previous.fold.eq(fold) & previous.selected.eq(chosen)].iloc[0]
                    filled, filled_win = int(old.filled), old.filled_win
                assert filled <= result.attempts
            summaries.append({**result._asdict(), 'filled': filled, 'filled_win': filled_win})
    output = dict(passed=True, models=len(hashes), decisions=len(results), model_sha256=hashes,
                  checks=['immutable causal signal sources, unknown flags preserved', 'mature fit and calibration labels',
                          'exact model predictions and calibration', 'all new validation/test grid statistics replayed',
                          'validation-only family and combined selections'])
    (OUT / 'verification.json').write_text(json.dumps(output, indent=2))
    pd.DataFrame(summaries).drop(columns='Index').to_csv(OUT / 'audited_results.csv', index=False)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
