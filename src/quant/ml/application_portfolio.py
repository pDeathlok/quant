"""Frozen multi-strategy rankings and independently funded NAV aggregation.

Capital weights belong to separate sleeve accounts. They must never be used to
collapse the three rankings into one top-N list or to rebalance NAV every day.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from quant.ml.rank_fusion import file_sha256

SLEEVES = ('score_three_way', 'score_return_path', 'weighted_low_volatility')
KEYS = ['symbol', 'date', 'side']


def validate_weights(weights: dict[str, float]) -> None:
    values = np.asarray(list(weights.values()), dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any() or not np.isclose(values.sum(), 1):
        raise ValueError('Positive finite initial capital weights summing to one required')


def combine_sleeve_nav(paths: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Sum NAVs normalized to one at inception, preserving capital drift."""
    validate_weights(weights)
    if set(paths) != set(weights):
        raise ValueError('Exactly one path per funded sleeve required')
    index = next(iter(paths.values())).index
    if not len(index) or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError('Unique increasing NAV dates required')
    result = pd.Series(0., index=index, name='nav')
    for name, weight in weights.items():
        path = paths[name]
        if not path.index.equals(index):
            raise ValueError('Sleeve calendars differ')
        if not np.isfinite(path.to_numpy()).all() or path.le(0).any():
            raise ValueError('Positive finite sleeve NAVs required')
        result += weight * path
    return result


def rank_sleeves(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Consume same-day side-relative scores; retain right-side overlap precedence."""
    validate_weights(weights)
    if set(weights) != set(SLEEVES):
        raise ValueError('Unsupported sleeve contract')
    required = set(KEYS + ['base_score', 'path_score', 'rank_score', 'volatility_20d'])
    if required - set(frame):
        raise ValueError(f'Missing ranking inputs: {sorted(required - set(frame))}')
    data = frame[list(required)].copy().reset_index(drop=True)
    data['date'] = pd.to_datetime(data.date, errors='raise', format='mixed').dt.normalize()
    if data[KEYS].isna().any().any() or data.duplicated(KEYS).any() or not data.side.isin(['left', 'right']).all():
        raise ValueError('Unique, non-null symbol/date/side keys required')
    scores = data[['base_score', 'path_score', 'rank_score']].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('Scores must be finite and normalized to [0, 1]')
    if np.isinf(data.volatility_20d.to_numpy(dtype=float)).any():
        raise ValueError('Infinite volatility is invalid')
    low_vol = (1 - data.groupby(['date', 'side']).volatility_20d.rank(pct=True)).fillna(.5)
    values = {
        'score_three_way': data[['base_score', 'path_score', 'rank_score']].mean(axis=1),
        'score_return_path': data[['base_score', 'path_score']].mean(axis=1),
        'weighted_low_volatility': .8 * data.base_score + .2 * low_vol,
    }
    outputs = []
    for name in SLEEVES:
        ranked = data[KEYS].assign(sleeve=name, score=values[name], initial_capital_weight=weights[name])
        ranked = ranked.sort_values(['date', 'symbol', 'side']).drop_duplicates(['date', 'symbol'], keep='last')
        ranked = ranked.sort_values(['date', 'score', 'symbol'], ascending=[True, False, True])
        ranked['priority'] = ranked.groupby('date').cumcount() + 1
        outputs.append(ranked)
    return pd.concat(outputs, ignore_index=True)


@dataclass(frozen=True)
class ApplicationPortfolio:
    release_id: str
    weights: dict[str, float]
    models: dict[tuple[str, str], xgb.Booster]

    @classmethod
    def load(cls, manifest_path: Path) -> ApplicationPortfolio:
        manifest_path = manifest_path.resolve()
        spec = json.loads(manifest_path.read_text())
        if spec.get('schema_version') != 'application-portfolio-v1':
            raise ValueError('Unsupported application portfolio schema')
        weights = spec['weights']
        validate_weights(weights)
        if set(weights) != set(SLEEVES):
            raise ValueError('Unsupported sleeve contract')
        models = {}
        for component in spec['auxiliary_models']:
            path = (manifest_path.parent / component['file']).resolve()
            if not path.is_relative_to(manifest_path.parent) or file_sha256(path) != component['sha256']:
                raise ValueError('Invalid component path or checksum')
            model = xgb.Booster()
            model.load_model(path)
            model.set_param({'nthread': 1})
            if model.feature_names != component['features']:
                raise ValueError('Auxiliary feature contract mismatch')
            key = (component['side'], component['target'])
            if key in models:
                raise ValueError('Duplicate auxiliary model')
            models[key] = model
        if set(models) != {(side, target) for side in ['left', 'right'] for target in ['path_boundary', 'rank']}:
            raise ValueError('Incomplete auxiliary model set')
        return cls(spec['release_id'], weights, models)

    def required_features(self, side: str) -> list[str]:
        if side not in ['left', 'right']:
            raise ValueError('Unknown side')
        return sorted({'volatility_20d'} | {c for (s, _), model in self.models.items() if s == side for c in model.feature_names})

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Input base_score must come from the pinned official production scorer."""
        required = set(KEYS + ['base_score'])
        if required - set(frame):
            raise ValueError(f'Missing candidate keys/base score: {sorted(required - set(frame))}')
        if not frame.side.isin(['left', 'right']).all():
            raise ValueError('Unknown side')
        pieces = []
        for side, part in frame.groupby('side', sort=True):
            missing = set(self.required_features(side)) - set(part)
            if missing:
                raise ValueError(f'Missing {side} auxiliary features: {sorted(missing)}')
            part = part.reset_index(drop=True)
            result = part[KEYS + ['base_score', 'volatility_20d']].copy()
            result['date'] = pd.to_datetime(result.date, errors='raise', format='mixed').dt.normalize()
            for target, output in [('path_boundary', 'path_score'), ('rank', 'rank_score')]:
                model = self.models[side, target]
                inputs = part[model.feature_names].astype('float32')
                if np.isinf(inputs.to_numpy()).any():
                    raise ValueError('Infinite feature values are invalid; explicit NaN is allowed')
                raw = model.predict(xgb.DMatrix(inputs, nthread=1))
                result[output] = pd.Series(raw).groupby(result.date).rank(method='average', pct=True)
            pieces.append(result)
        if not pieces:
            raise ValueError('At least one candidate required')
        return rank_sleeves(pd.concat(pieces, ignore_index=True), self.weights)
