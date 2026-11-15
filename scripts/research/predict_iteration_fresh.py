"""Score the sealed new interval only after development choices are frozen."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent), str(ROOT / '.tmp/application_iteration_dependencies')]
from model_iteration_v2_protocol import OUT, TREE_ARMS, NEURAL_ARMS


def prepare_inputs():
    import pyarrow.parquet as pq
    industry = pd.read_parquet(OUT / 'industry_features.parquet')
    calendar = pd.DatetimeIndex(pd.to_datetime([p.stem for p in sorted((ROOT / 'data/raw/daily_basic').glob('*.parquet'))
                                               if p.stem.isdigit() and p.stem >= '20240101']))
    for side in ['left', 'right']:
        frame = pd.read_parquet(OUT / f'fresh_features/inputs_{side}.parquet')
        frame = frame.drop(columns=[c for c in industry if c not in ['symbol', 'date'] and c in frame])
        frame = frame.merge(industry, on=['symbol', 'date'], how='left', validate='one_to_one')
        old = pd.read_parquet(OUT / f'development_{side}.parquet', columns=['symbol', 'date'])
        extra = []
        for path in sorted((ROOT / f'data/features/{side}_side_unified').glob('*_features.parquet')):
            if path.name[:8].isdigit() and '20260822' <= path.name[:8] < '20260904':
                extra.append(pd.read_parquet(path, columns=['symbol', 'date']))
        history = pd.concat([old, frame[['symbol', 'date']], *extra], ignore_index=True).drop_duplicates().sort_values(['symbol', 'date']).reset_index(drop=True)
        positions = pd.Series(calendar.get_indexer(history.date), index=history.index)
        history['event_gap'] = positions.groupby(history.symbol).diff().fillna(999)
        event = history.event_gap.gt(5).groupby(history.symbol).cumsum()
        history['event_age'] = history.groupby([history.symbol, event]).cumcount().astype(float)
        history['event_recent_count'] = positions.groupby(history.symbol).transform(lambda s: s.diff().le(5).rolling(5, min_periods=1).sum())
        columns = ['event_gap', 'event_age', 'event_recent_count']
        frame = frame.drop(columns=[c for c in columns if c in frame]).merge(history[['symbol', 'date', *columns]], on=['symbol', 'date'], validate='one_to_one')
        frame['execution_liquidity'] = frame.turnover_rate / frame.volatility_20d.clip(lower=.05)
        frame['execution_vol_volume'] = frame.volatility_20d * frame.volume_relative_20d
        frame.to_parquet(OUT / f'fresh_inputs_{side}.parquet', index=False)


def main(kind):
    if not (OUT / 'frozen_selection.json').exists():
        raise RuntimeError('New interval remains sealed until development selection is complete')
    if kind == 'tree':
        import xgboost as xgb
        prepare_inputs()
    else:
        import torch
        from train_iteration_neural import make_model, transform, predict
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    for side in ['left', 'right']:
        frame = pd.read_parquet(OUT / f'fresh_inputs_{side}.parquet')
        for arm in TREE_ARMS if kind == 'tree' else NEURAL_ARMS:
            spec = json.loads((OUT / f'models/{side}_{arm}.json').read_text())
            columns = spec['features']
            if kind == 'tree':
                inputs = xgb.DMatrix(frame[columns].astype('float32').replace([np.inf, -np.inf], np.nan), nthread=3)
                values = []
                for component in spec['models']:
                    model = xgb.Booster()
                    model.load_model(OUT / 'models' / component['file'])
                    model.set_param({'nthread': 3})
                    values.append(model.predict(inputs))
                if arm == 'regime_experts':
                    raw = np.where(frame.market_breadth_above_ma20_ratio.ge(.5), values[1], values[0])
                else:
                    raw = values[0][:, 0] if values[0].ndim == 2 else values[0]
            else:
                with np.load(OUT / 'models' / spec['preprocess']) as source:
                    preprocess = dict(source)
                inputs = transform(frame, columns, preprocess)
                model = make_model(arm, inputs.shape[1])
                model.load_state_dict(torch.load(OUT / 'models' / spec['weights'], map_location='cpu', weights_only=True))
                raw = predict(model, inputs)
            result = frame[['symbol', 'date']].copy()
            result['raw_score'] = raw
            result['expected_net'] = np.interp(raw, spec['calibration']['x'], spec['calibration']['y'])
            result.to_parquet(OUT / f'fresh_predictions_{side}_{arm}.parquet', index=False)
            print('FRESH PREDICT', side, arm, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['tree', 'neural'])
    main(parser.parse_args().kind)
