"""Matched MLP/official TabM/multitask ablations, CPU and fixed seed."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.tmp/application_iteration_dependencies'), str(Path(__file__).resolve().parent)]
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import torch
from torch import nn
from tabm import TabM
import tabm
from model_iteration_v2_protocol import OUT, NEURAL_ARMS, BOUNDS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_model(arm, width):
    if arm == 'mlp':
        return nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(.1),
                             nn.Linear(128, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 1))
    return TabM.make(n_num_features=width, d_out=3 if arm == 'tabm_multitask' else 1,
                     k=8, d_block=128, n_blocks=2, dropout=.1)


def transform(frame, columns, preprocess):
    values = frame[columns].to_numpy(dtype=np.float32)
    absent = ~np.isfinite(values)
    values = np.where(absent, preprocess['median'], values)
    values = np.clip((values - preprocess['mean']) / preprocess['scale'], -8, 8)
    return torch.from_numpy(np.column_stack([values, absent.astype(np.float32)]).astype(np.float32))


def predict(model, tensor):
    model.eval()
    values = []
    with torch.no_grad():
        for chunk in tensor.split(2048):
            value = model(chunk)
            if value.ndim == 3:
                value = value.mean(dim=1)
            values.append(value[:, 0].numpy() * .05)
    return np.concatenate(values)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    meta = json.loads((OUT / 'development_manifest.json').read_text())
    folder = OUT / 'models'
    folder.mkdir(exist_ok=True)
    all_audits = []
    for side in ['left', 'right']:
        frame = pd.read_parquet(OUT / f'development_{side}.parquet')
        fit = frame.loc[frame.date.lt(BOUNDS['fit_end']) & frame.maturity.lt(BOUNDS['fit_end']) & frame.outcome.notna()].copy()
        cal = frame.loc[frame.date.ge(BOUNDS['fit_end']) & frame.date.lt(BOUNDS['calibration_end'])
                       & frame.maturity.lt(BOUNDS['calibration_end']) & frame.outcome.notna()].copy()
        evaluation = frame.loc[frame.date.ge(BOUNDS['fit_end'])].copy()
        columns = [c for c in meta['core'] if fit[c].nunique() > 1]
        data = fit[columns].to_numpy(dtype=np.float32, copy=True)
        data[~np.isfinite(data)] = np.nan
        median = np.nanmedian(data, axis=0)
        filled = np.where(np.isnan(data), median, data)
        preprocess = {'median': median, 'mean': filled.mean(axis=0), 'scale': np.maximum(filled.std(axis=0), 1e-6)}
        np.savez(folder / f'{side}_neural_preprocess.npz', **preprocess)
        x_fit, x_cal, x_eval = [transform(part, columns, preprocess) for part in [fit, cal, evaluation]]
        y = torch.from_numpy(fit.outcome.clip(-.5, .5).to_numpy(dtype=np.float32) / .05)
        mae = torch.from_numpy(fit.mae.fillna(0).clip(-.5, 0).to_numpy(dtype=np.float32) / .05)
        good = torch.from_numpy(fit.good_path5.fillna(False).astype(float).to_numpy(dtype=np.float32))
        known = torch.from_numpy(fit.good_path5.notna().to_numpy(dtype=np.float32))
        weight = (1 / fit.groupby('date').date.transform('size')).to_numpy(dtype=np.float32)
        weights = torch.from_numpy(weight / weight.mean())
        cal_weight = (1 / cal.groupby('date').date.transform('size')).to_numpy()
        for arm in NEURAL_ARMS:
            manifest_path = folder / f'{side}_{arm}.json'
            if manifest_path.exists() and (OUT / f'predictions_{side}_{arm}.parquet').exists():
                all_audits.append(json.loads(manifest_path.read_text())['audit'])
                continue
            torch.manual_seed(73)
            np.random.seed(73)
            model = make_model(arm, x_fit.shape[1])
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            best_loss, best_epoch, stale, best_state = np.inf, -1, 0, None
            history = []
            started = time.monotonic()
            for epoch in range(16):
                model.train()
                training_loss = 0.
                for indexes in torch.randperm(len(fit)).split(1024):
                    prediction = model(x_fit[indexes])
                    if prediction.ndim == 2:
                        prediction = prediction.unsqueeze(1)
                    per_row = ((prediction[:, :, 0] - y[indexes, None]) ** 2).mean(dim=1)
                    if arm == 'tabm_multitask':
                        per_row += .25 * ((prediction[:, :, 1] - mae[indexes, None]) ** 2).mean(dim=1)
                        target = good[indexes, None].expand_as(prediction[:, :, 2])
                        per_row += .25 * nn.functional.binary_cross_entropy_with_logits(prediction[:, :, 2], target, reduction='none').mean(dim=1) * known[indexes]
                    loss = (per_row * weights[indexes]).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    training_loss += float(loss.detach()) * len(indexes)
                raw_cal = predict(model, x_cal)
                loss_cal = float(np.average((raw_cal - cal.outcome.clip(-.5, .5).to_numpy()) ** 2, weights=cal_weight))
                history.append({'epoch': epoch + 1, 'fit_loss': training_loss / len(fit), 'calibration_mse': loss_cal})
                print('NEURAL', side, arm, epoch + 1, round(loss_cal, 7), flush=True)
                if loss_cal < best_loss - 1e-8:
                    best_loss, best_epoch, stale = loss_cal, epoch + 1, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                if stale >= 4:
                    break
            model.load_state_dict(best_state)
            raw_cal = predict(model, x_cal)
            iso = IsotonicRegression(out_of_bounds='clip').fit(raw_cal, cal.outcome.clip(-.5, .5), sample_weight=cal_weight)
            calibration = {'x': iso.X_thresholds_.tolist(), 'y': iso.y_thresholds_.tolist()}
            raw = predict(model, x_eval)
            result = evaluation[['symbol', 'date']].copy()
            result['raw_score'], result['expected_net'] = raw, iso.predict(raw)
            result.to_parquet(OUT / f'predictions_{side}_{arm}.parquet', index=False)
            path = folder / f'{side}_{arm}.pt'
            torch.save(best_state, path)
            audit = {'side': side, 'arm': arm, 'fit_rows': len(fit), 'calibration_rows': len(cal),
                     'features': len(columns), 'input_width_including_missing_flags': x_fit.shape[1],
                     'epochs': len(history), 'best_epoch': best_epoch, 'seconds': time.monotonic() - started,
                     'fit_label_max': str(fit.maturity.max()), 'calibration_label_max': str(cal.maturity.max())}
            manifest_path.write_text(json.dumps({'arm': arm, 'side': side, 'features': columns,
                'weights': path.name, 'sha256': digest(path), 'preprocess': f'{side}_neural_preprocess.npz',
                'preprocess_sha256': digest(folder / f'{side}_neural_preprocess.npz'), 'calibration': calibration,
                'audit': audit, 'training_history': history, 'torch': torch.__version__, 'tabm': tabm.__version__,
                'multitask_loss': 'Return MSE + .25 MAE MSE + .25 good-path BCE. TabM losses average independent member losses; inference averages members.'}, indent=2))
            all_audits.append(audit)
            pd.DataFrame(all_audits).to_csv(OUT / 'neural_training.csv', index=False)
    (OUT / 'neural_complete.json').write_text(json.dumps({'models': len(all_audits), 'script_sha256': digest(__file__)}, indent=2))


if __name__ == '__main__':
    main()
