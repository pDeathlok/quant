"""Export selected TabM to TorchScript, avoiding training-package dependencies."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.tmp/application_iteration_dependencies'), str(Path(__file__).resolve().parent)]
import numpy as np
import pandas as pd
import torch
from train_iteration_neural import make_model, transform, predict, digest
from model_iteration_v2_protocol import OUT


def main():
    torch.set_num_threads(2)
    choice = json.loads((OUT / 'frozen_selection.json').read_text())
    arms = {r['arm'] for r in choice['recipes'].values()} & {'mlp', 'tabm', 'tabm_multitask'}
    checks = []
    for side in ['left', 'right']:
        for arm in arms:
            spec = json.loads((OUT / f'models/{side}_{arm}.json').read_text())
            with np.load(OUT / 'models' / spec['preprocess']) as source:
                preprocessing = dict(source)
            frame = pd.read_parquet(OUT / f'development_{side}.parquet')
            frame = frame.loc[frame.date.ge('2026-04-01')]
            inputs = transform(frame, spec['features'], preprocessing)
            model = make_model(arm, inputs.shape[1])
            model.load_state_dict(torch.load(OUT / 'models' / spec['weights'], weights_only=True, map_location='cpu'))
            model.eval()
            exported = torch.jit.trace(model, inputs[:16], check_trace=True)
            path = OUT / f'models/{side}_{arm}.torchscript'
            exported.save(str(path))
            loaded = torch.jit.load(str(path))
            original, actual = predict(model, inputs), predict(loaded, inputs)
            np.testing.assert_array_equal(original, actual)
            checks.append({'side': side, 'arm': arm, 'rows': len(frame), 'exact_prediction_parity': True,
                           'file': path.name, 'sha256': digest(path)})
    (OUT / 'neural_export_verification.json').write_text(json.dumps({'passed': True, 'checks': checks}, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
