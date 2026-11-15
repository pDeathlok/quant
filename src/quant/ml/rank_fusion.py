"""Versioned RankNet inference with exact tree compaction and cash gates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compact_booster(model: xgb.Booster) -> xgb.Booster:
    """Remove unused input slots, preserving every split, leaf and base score."""
    payload = json.loads(bytes(model.save_raw(raw_format='json')))
    learner = payload['learner']
    trees = learner['gradient_booster']['model']['trees']
    used = sorted({int(feature) for tree in trees
                   for feature, child in zip(tree['split_indices'], tree['left_children'], strict=True)
                   if child != -1})
    if not used:
        raise ValueError('Cannot publish an input-free constant ranker')
    remap = {old: new for new, old in enumerate(used)}
    names = learner['feature_names']
    if not names or len(names) != model.num_features():
        raise ValueError('Named input contract required')
    learner['feature_names'] = [names[i] for i in used]
    if learner['feature_types']:
        learner['feature_types'] = [learner['feature_types'][i] for i in used]
    learner['learner_model_param']['num_feature'] = str(len(used))
    for tree in trees:
        tree['tree_param']['num_feature'] = str(len(used))
        tree['split_indices'] = [remap[index] if child != -1 else 0
                                 for index, child in zip(tree['split_indices'], tree['left_children'], strict=True)]
    result = xgb.Booster()
    result.load_model(bytearray(json.dumps(payload).encode()))
    result.set_param({'nthread': 4})
    return result


@dataclass(frozen=True)
class RankComponent:
    model: xgb.Booster
    weight: float
    calibration_x: np.ndarray
    calibration_y: np.ndarray


@dataclass(frozen=True)
class RankFusion:
    release_id: str
    components: tuple[RankComponent, ...]
    minimum_expected_net_return: float | None
    required_features: tuple[str, ...]

    @classmethod
    def load(cls, manifest_path: Path) -> RankFusion:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema_version') != 'rank-fusion-v1':
            raise ValueError('Unsupported rank fusion contract')
        components = []
        features: set[str] = set()
        for spec in manifest['components']:
            path = (manifest_path.parent / spec['file']).resolve()
            if not path.is_relative_to(manifest_path.parent.resolve()):
                raise ValueError('Component path escapes release directory')
            if file_sha256(path) != spec['sha256']:
                raise ValueError('Rank component checksum mismatch')
            model = xgb.Booster()
            model.load_model(path)
            model.set_param({'nthread': 1})
            if model.feature_names != spec['features']:
                raise ValueError('Rank component input mismatch')
            x = np.asarray(spec['calibration']['x'], dtype=float)
            y = np.asarray(spec['calibration']['y'], dtype=float)
            weight = float(spec['weight'])
            if (len(x) == 0 or len(x) != len(y) or not np.isfinite(x).all()
                    or not np.isfinite(y).all() or (np.diff(x) <= 0).any()
                    or (np.diff(y) < 0).any() or not np.isfinite(weight) or weight < 0):
                raise ValueError('Invalid calibration or component weight')
            components.append(RankComponent(model, weight, x, y))
            features.update(model.feature_names)
        if not components or not np.isclose(sum(c.weight for c in components), 1):
            raise ValueError('Fusion weights must sum to one')
        if sorted(features) != manifest['required_features']:
            raise ValueError('Fusion feature union mismatch')
        threshold = manifest['minimum_expected_net_return']
        if threshold is not None and not np.isfinite(float(threshold)):
            raise ValueError('Threshold must be finite or null for rank-only')
        return cls(manifest['release_id'], tuple(components), threshold, tuple(sorted(features)))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.required_features) - set(frame)
        if missing:
            raise ValueError(f'Required rank inputs absent: {sorted(missing)}')
        if not {'symbol', 'date'} <= set(frame) or frame.duplicated(['symbol', 'date']).any():
            raise ValueError('Unique symbol/date keys required')
        if pd.to_datetime(frame.date, errors='coerce').isna().any():
            raise ValueError('Invalid signal date')
        score = np.zeros(len(frame))
        expected = np.zeros(len(frame))
        if len(frame):
            for component in self.components:
                values = frame[component.model.feature_names].astype('float32')
                if np.isinf(values.to_numpy()).any():
                    raise ValueError('Infinite rank inputs are invalid; explicit NaNs are allowed')
                raw = component.model.predict(xgb.DMatrix(values, nthread=1))
                if not np.isfinite(raw).all():
                    raise ValueError('Nonfinite rank predictions')
                percentile = pd.Series(raw).groupby(pd.to_datetime(frame.date).to_numpy()).rank(method='average', pct=True)
                score += component.weight * percentile.to_numpy()
                expected += component.weight * np.interp(raw, component.calibration_x, component.calibration_y)
        gate = np.ones(len(frame), dtype=bool) if self.minimum_expected_net_return is None else expected >= self.minimum_expected_net_return
        return frame[['symbol', 'date']].assign(ranking_score=score, expected_net_return=expected, trade_gate_open=gate)
