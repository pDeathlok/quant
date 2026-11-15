"""Fit predeclared tree ablations without reading the new-interval outcomes."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from quant.ml.rank_fusion import file_sha256
from model_iteration_v2_protocol import OUT, TREE_ARMS, BOUNDS

PARAMS = dict(tree_method='hist', max_depth=3, min_child_weight=100,
              learning_rate=.06, reg_lambda=30, seed=73, nthread=3, objective='reg:squarederror')


def matrix(frame, columns, labels=None, weights=None):
    return xgb.DMatrix(frame[columns].astype('float32').replace([np.inf, -np.inf], np.nan),
                       label=labels, weight=weights, nthread=3)


def calibrate(raw, frame):
    weights = 1 / frame.groupby('date').date.transform('size')
    model = IsotonicRegression(out_of_bounds='clip').fit(raw, frame.outcome.clip(-.5, .5), sample_weight=weights)
    return {'x': model.X_thresholds_.tolist(), 'y': model.y_thresholds_.tolist()}


def main():
    meta = json.loads((OUT / 'development_manifest.json').read_text())
    folder = OUT / 'models'
    folder.mkdir(exist_ok=True)
    audits = []
    for side in ['left', 'right']:
        frame = pd.read_parquet(OUT / f'development_{side}.parquet')
        fit = frame.loc[frame.date.lt(BOUNDS['fit_end']) & frame.maturity.lt(BOUNDS['fit_end']) & frame.outcome.notna()].copy()
        cal = frame.loc[frame.date.ge(BOUNDS['fit_end']) & frame.date.lt(BOUNDS['calibration_end'])
                       & frame.maturity.lt(BOUNDS['calibration_end']) & frame.outcome.notna()].copy()
        evaluation = frame.loc[frame.date.ge(BOUNDS['fit_end'])].copy()
        core = [c for c in meta['core'] if fit[c].nunique() > 1]
        for arm in TREE_ARMS:
            manifest_path = folder / f'{side}_{arm}.json'
            if manifest_path.exists() and (OUT / f'predictions_{side}_{arm}.parquet').exists():
                audits.append(json.loads(manifest_path.read_text())['audit'])
                continue
            print('TREE', side, arm, flush=True)
            started = time.monotonic()
            training = fit
            if arm.startswith('window_'):
                months = int(arm.split('_')[1].rstrip('m'))
                training = fit.loc[fit.date.ge(pd.Timestamp(BOUNDS['fit_end']) - pd.DateOffset(months=months))]
            columns = core + (meta['industry'] if arm == 'industry_relative' else meta['event'] if arm == 'event_execution' else [])
            columns = list(dict.fromkeys(columns))
            weights = (1 / training.groupby('date').date.transform('size')).to_numpy(copy=True)
            if arm == 'decay_180d':
                weights *= np.exp2(-(pd.Timestamp(BOUNDS['fit_end']) - training.date).dt.days.to_numpy() / 180)
            weights /= weights.mean()
            target = training.outcome.clip(-.5, .5).to_numpy(copy=True)
            if arm == 'utility_mae':
                target += .25 * training.mae.fillna(0).clip(-.5, 0).to_numpy()
            if arm == 'utility_tail':
                target -= .5 * np.maximum(-target - .03, 0)
            params = dict(PARAMS)
            if arm == 'multitask_tree':
                target = np.column_stack([target, training.mae.fillna(0).clip(-.5, 0), training.mfe.fillna(0).clip(0, .5)])
                params['multi_strategy'] = 'multi_output_tree'
            models = []
            if arm == 'calibrator_only':
                model = xgb.Booster()
                model.load_model(ROOT / f'reports/research/sample_label_research_v1/model_{side}_2026_return.ubj')
                model.set_param({'nthread': 3})
                columns = model.feature_names
                models.append(model)
            elif arm == 'regime_experts':
                for high in [False, True]:
                    mask = training.market_breadth_above_ma20_ratio.ge(.5).to_numpy() == high
                    if mask.sum() < 1000:
                        raise RuntimeError('Insufficient observations in market regime')
                    models.append(xgb.train(params, matrix(training.loc[mask], columns, target[mask], weights[mask]), num_boost_round=60))
            else:
                models.append(xgb.train(params, matrix(training, columns, target, weights), num_boost_round=60))
            def predict(part):
                values = [model.predict(matrix(part, columns)) for model in models]
                if arm == 'regime_experts':
                    return np.where(part.market_breadth_above_ma20_ratio.ge(.5), values[1], values[0])
                return values[0][:, 0] if values[0].ndim == 2 else values[0]
            calibration = calibrate(predict(cal), cal)
            raw = predict(evaluation)
            prediction = evaluation[['symbol', 'date']].copy()
            prediction['raw_score'] = raw
            prediction['expected_net'] = np.interp(raw, calibration['x'], calibration['y'])
            if arm == 'calibrator_only':
                frozen = json.loads((ROOT / f'reports/research/sample_label_research_v1/calibrator_{side}_2026_return.json').read_text())
                prediction['expected_frozen'] = np.interp(raw, frozen['x'], frozen['y'])
            prediction.to_parquet(OUT / f'predictions_{side}_{arm}.parquet', index=False)
            paths = []
            for index, model in enumerate(models):
                path = folder / f'{side}_{arm}_{index}.ubj'
                model.save_model(path)
                paths.append({'file': path.name, 'sha256': file_sha256(path)})
            audit = {'side': side, 'arm': arm, 'fit_rows': len(training), 'calibration_rows': len(cal),
                     'features': len(columns), 'seconds': time.monotonic() - started,
                     'training_label_max': str(training.maturity.max()), 'calibration_label_max': str(cal.maturity.max()),
                     'frozen_base_model': arm == 'calibrator_only'}
            if arm == 'calibrator_only':
                original = pd.read_csv(ROOT / 'reports/research/sample_label_research_v1/training.csv', dtype={'fold': str})
                original = original.loc[original.side.eq(side) & original.fold.eq('2026') & original.arm.eq('return')].iloc[0]
                audit.update(fit_rows=int(original.rows), training_label_max=original.fit_label_end,
                             base_refit=False, eligible_current_fit_rows=len(training))
            manifest_path.write_text(json.dumps({'arm': arm, 'side': side, 'features': columns, 'models': paths,
                'params': params, 'rounds': 60, 'calibration': calibration, 'audit': audit}, indent=2))
            audits.append(audit)
            pd.DataFrame(audits).to_csv(OUT / 'tree_training.csv', index=False)
    (OUT / 'tree_complete.json').write_text(json.dumps({'models': len(audits), 'script_sha256': file_sha256(Path(__file__))}, indent=2))


if __name__ == '__main__':
    main()
