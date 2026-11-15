"""Versioned multi-account application policies with explicit admission gates."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
import xgboost as xgb

from quant.ml.application_portfolio import ApplicationPortfolio, validate_weights
from quant.ml.rank_fusion import file_sha256

KEYS = ['symbol', 'date', 'side']


@dataclass
class PortfolioPolicy:
    root: Path
    folder: Path
    spec: dict
    legacy: ApplicationPortfolio | None

    @classmethod
    def load(cls, path: Path, root: Path) -> PortfolioPolicy:
        path = path.resolve()
        spec = json.loads(path.read_text())
        if spec.get('schema_version') != 'portfolio-policy-v1':
            raise ValueError('Unsupported portfolio policy schema')
        validate_weights(spec['weights'])
        for name, expected in spec['files'].items():
            artifact = (path.parent / name).resolve()
            if not artifact.is_relative_to(path.parent) or file_sha256(artifact) != expected:
                raise ValueError(f'Portfolio artifact identity mismatch: {name}')
        legacy = None
        if spec['engine'] == 'legacy_combination':
            legacy = ApplicationPortfolio.load(path.parent / spec['legacy_manifest'])
            if legacy.weights != spec['weights']:
                raise ValueError('Legacy capital weights changed')
        elif spec['engine'] != 'gated_models' or set(spec['strategies']) != set(spec['weights']):
            raise ValueError('Incomplete gated strategy definitions')
        return cls(root.resolve(), path.parent, spec, legacy)

    def required_features(self, side: str) -> list[str]:
        if self.legacy:
            return self.legacy.required_features(side)
        return sorted({'market_breadth_above_ma20_ratio', 'volatility_20d'} |
                      {c for strategy in self.spec['strategies'].values() for c in strategy['models'][side]['features']})

    def _predict_model(self, frame: pd.DataFrame, spec: dict) -> np.ndarray:
        values = frame[spec['features']].to_numpy(dtype=np.float32)
        if spec['engine'] == 'xgboost':
            model = xgb.Booster()
            model.load_model(self.folder / spec['file'])
            model.set_param({'nthread': 2})
            return model.predict(xgb.DMatrix(values, feature_names=spec['features'], nthread=2))
        if spec['engine'] != 'torchscript':
            raise ValueError('Unsupported inference engine')
        with np.load(self.folder / spec['preprocess'], allow_pickle=False) as preprocessing:
            absent = ~np.isfinite(values)
            values = np.where(absent, preprocessing['median'], values)
            values = np.clip((values - preprocessing['mean']) / preprocessing['scale'], -8, 8)
            values = np.column_stack([values, absent.astype(np.float32)]).astype(np.float32)
        # Keep the venv symlink path: resolving it would bypass its environment.
        python = (self.root / self.spec['neural_python']).absolute()
        if not python.is_file():
            raise ValueError('Configured neural inference environment is missing')
        scratch = self.root / '.tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='portfolio-inference-', dir=scratch) as directory:
            source, target = Path(directory) / 'input.npy', Path(directory) / 'output.npy'
            np.save(source, values, allow_pickle=False)
            result = subprocess.run([str(python), str(self.root / 'src/quant/ml/portfolio_neural_worker.py'),
                '--model', str(self.folder / spec['file']), '--input', str(source), '--output', str(target)],
                capture_output=True, text=True, timeout=180, check=False)
            if result.returncode or not target.exists():
                raise RuntimeError(f'Neural inference failed: {result.stderr[-1500:]}')
            return np.load(target, allow_pickle=False)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not set(KEYS) <= set(frame):
            raise ValueError('Candidate keys are missing')
        if frame.empty:
            return pd.DataFrame(columns=KEYS + ['sleeve', 'score', 'expected_net', 'priority', 'initial_capital_weight'])
        if self.legacy:
            return self.legacy.predict(frame)
        data = frame.copy().reset_index(drop=True)
        data['date'] = pd.to_datetime(data.date, errors='raise', format='mixed').dt.normalize()
        if data[KEYS].isna().any().any() or data.duplicated(KEYS).any() or not data.side.isin(['left', 'right']).all():
            raise ValueError('Unique non-null candidate keys required')
        outputs = []
        for name, strategy in self.spec['strategies'].items():
            pieces = []
            for side, part in data.groupby('side', sort=True):
                model = strategy['models'][side]
                missing = set(self.required_features(side)) - set(part)
                if missing:
                    raise ValueError(f'Missing portfolio inputs: {sorted(missing)}')
                raw = self._predict_model(part, model)
                if raw.ndim != 1 or len(raw) != len(part) or not np.isfinite(raw).all():
                    raise ValueError('Invalid model predictions')
                result = part[KEYS + ['market_breadth_above_ma20_ratio']].copy()
                result['raw_score'] = raw
                calibration = model['calibration']
                result['expected_net'] = np.interp(raw, calibration['x'], calibration['y'])
                result['score'] = result.groupby('date').raw_score.rank(method='average', pct=True)
                pieces.append(result)
            scored = pd.concat(pieces, ignore_index=True)
            mask = pd.Series(True, index=scored.index)
            if strategy['expected_net_gate'] is not None:
                mask &= scored.expected_net.ge(strategy['expected_net_gate'])
            if strategy['market_breadth_gate'] is not None:
                mask &= scored.market_breadth_above_ma20_ratio.ge(strategy['market_breadth_gate'])
            scored = scored.loc[mask].sort_values(['date', 'symbol', 'side']).drop_duplicates(['date', 'symbol'], keep='last')
            scored = scored.sort_values(['date', 'score', 'symbol'], ascending=[True, False, True])
            scored['priority'] = scored.groupby('date').cumcount() + 1
            scored['sleeve'] = name
            scored['initial_capital_weight'] = self.spec['weights'][name]
            outputs.append(scored[KEYS + ['sleeve', 'score', 'expected_net', 'priority', 'initial_capital_weight']])
        return pd.concat(outputs, ignore_index=True)
