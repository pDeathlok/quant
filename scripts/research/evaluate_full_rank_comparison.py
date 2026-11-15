"""Same-cohort full-registry rank/fusion/cash comparison; research only."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/research')]
import evaluate_local_ranknet_fusion as prior
from build_full_rank_dataset import OUT
from evaluate_head_lambdaloss import grouped_training
from evaluate_ranknet_cash_policy import evaluate_policy, pick_policy, THRESHOLDS
from evaluate_auxiliary_return_losses import intervals
from quant.research.local_ranknet import make_pairs, cross_date_pairs, derivatives, percentile_scores
from quant.research.head_lambdaloss import HeadLambdaObjective, fit_grade_cuts, gains, head_metrics
from quant.research.auxiliary_return_loss import LossWeights, terms, derivatives as auxiliary_derivatives

FOLDS = {'2025H2': ('2025-01-01', '2025-04-01', '2025-07-01', '2026-01-01'),
         '2026': ('2025-07-01', '2025-10-01', '2026-01-01', '2026-09-04')}
ARMS = {
    'base50': ('regression', 'base', 'day'),
    'core129': ('regression', 'core', 'day'),
    'original_full': ('regression', 'original', 'day'),
    'full_regression': ('regression', 'all', 'day'),
    'full_capacity120': ('regression120', 'all', 'day'),
    'remaining_regression': ('regression', 'remaining', 'day'),
    'ranknet_all': ('ranknet', 'all', 'day'),
    'ranknet_remaining': ('ranknet', 'remaining', 'day'),
    'local05_all': ('local05', 'all', 'day'),
    'local10_all': ('local10', 'all', 'day'),
    'local05_remaining': ('local05', 'remaining', 'day'),
    'cross25_all': ('cross25', 'all', 'day'),
    'cross50_all': ('cross50', 'all', 'day'),
    'cross100_all': ('cross100', 'all', 'day'),
    'ranknet_head20': ('ranknet_head', 'all', 'day'),
    'lambdarank20_all': ('lambdarank', 'all', 'day'),
    'lambdarank20_remaining': ('lambdarank', 'remaining', 'day'),
    'lambdaloss2_all': ('lambdaloss2', 'all', 'day'),
    'lambdaloss2pp_all': ('lambdaloss2pp', 'all', 'day'),
    'lambdaloss2pp_remaining': ('lambdaloss2pp', 'remaining', 'day'),
    'cross_lambdaloss2pp': ('lambdaloss2pp', 'all', 'cross'),
    'aux_absolute_increment': ('auxiliary', 'all', 'day'),
}


def calendar_for(frame, start, end):
    # Include the execution/settlement tail after the last signal. Truncating at
    # the last signal would incorrectly move its next-close entry onto today.
    dates = pd.to_datetime(pd.read_csv(ROOT/'data/research/left_side_disclosure_gain_20261029/calendar.csv').date)
    return dates.loc[dates.ge(start) & dates.lt(end)].to_numpy()


def distribution(frame: pd.DataFrame, columns: list[str], boundaries: tuple, path: Path, tr: pd.DataFrame) -> None:
    fit_end, selection, test_start, test_end = boundaries
    stages = {'fit': tr, 'validation': prior.evaluation_slice(frame, selection, test_start), 'test': prior.evaluation_slice(frame, test_start, test_end)}
    records = []
    for c in columns:
        fit = tr[c].dropna().to_numpy()
        edges = np.unique(np.quantile(fit, np.linspace(0, 1, 11))) if len(fit) else np.array([])
        if len(edges) > 1:
            edges[0], edges[-1] = -np.inf, np.inf
        def hist(values):
            values = values.to_numpy(); finite = values[np.isfinite(values)]
            counts = np.r_[np.histogram(finite, edges)[0], len(values) - len(finite)] + .5
            return counts / counts.sum()
        p = hist(tr[c]) if len(edges) > 1 else None
        for stage, data in stages.items():
            q = hist(data[c]) if p is not None else None
            records.append(dict(name=c, stage=stage, coverage=data[c].notna().mean(), unique=data[c].nunique(),
                                psi=float(((q-p)*np.log(q/p)).sum()) if p is not None else None,
                                p01=data[c].quantile(.01), p50=data[c].quantile(.5), p99=data[c].quantile(.99)))
    pd.DataFrame(records).to_csv(path, index=False)


def main() -> None:
    meta = json.loads((OUT / 'dataset_complete.json').read_text())
    if not meta.get('augmentation_complete'):
        raise ValueError('Complete dated-source recovery before fitting')
    if not meta.get('readiness_repaired'):
        raise ValueError('Repair per-row readiness and regenerate OOF scores first')
    audit = pd.read_csv(OUT / 'column_audit.csv')
    original = set(audit.loc[audit.original_776, 'name'])
    prior.save_json(OUT / 'experiment_protocol.json', dict(arms=ARMS, folds=FOLDS, params=prior.PARAMS,
        alphas=[.5, 1], thresholds=[str(t) for t in THRESHOLDS],
        labels='Net terminal return after existing 20bp costs; actual maturity purge. Unknown exits retained at evaluation, stressed -10% and sleeve frozen.',
        calibration='Separate first validation quarter isotonic, last validation quarter selects thresholds/alpha; test only reports frozen choices.',
        oof='Rebuilt2024Q2+ quarterly baseline50 forecasts, fit only earlier2024+ rows with actual label maturity before forecast start. Fixed50 names; same features in every OOF block.',
        samples='All attempted candidates independent of future tradability. Same rows for every model in fold; models selected columns using FIT-only nonmissing/nonconstant check.',
        fusion='Calibrated expected net return for gate, within-date raw-score percentile blend for top20 priority.',
        cash='Cash allowed indefinitely; left 6 rotating sleeves, right 1,20 slots each. Unknown exit freezes sleeve; delayed exit blocks reuse. SAME-CLOSE exit/reentry proxy.',
        scope='Original776 audited PLUS341 newly registered index states and19JQ; full means all admitted/informative, not unsafe snapshots.',
        ndcg='Nonpositive grade0, fit-positive quantile grades1..4; head20; mu5;32 tail samples inverse probability weighted.',
        limitations=['Historical test periods already researched, not independent OOS discovery.', 'Terminal capital stress return is not MTM NAV or a live fill guarantee.', meta['limitations']],
        dataset_hashes=meta['dataset_hashes'], script_hash=prior.sha256(Path(__file__))))
    summaries=[]; head_rows=[]; grid=[]; admissions=[]; times=[]
    for side in ['left', 'right']:
        if f'--{side}-only' not in sys.argv and any(a in sys.argv for a in ['--left-only', '--right-only']):
            continue
        data_path = OUT / f'dataset_{side}.parquet'
        assert prior.sha256(data_path) == meta['dataset_hashes'][side]
        frame = pd.read_parquet(data_path).sort_values(['date', 'symbol']).reset_index(drop=True)
        for fold, bounds in FOLDS.items():
            fit_end, select_start, test_start, test_end = bounds
            print('FOLD', side, fold, flush=True)
            old_fold = '2026' if side == 'left' else fold
            contract = json.loads((prior.OUT / f'features_{side}_{old_fold}.json').read_text())
            first = contract['base50']
            oof_path = OUT / f'oof_{side}.parquet'
            oof = pd.read_parquet(oof_path)[prior.KEYS + ['oof_score', 'oof_percentile']]
            tr = frame.merge(oof, on=prior.KEYS, how='inner', validate='one_to_one')
            tr = tr.loc[tr.maturity.lt(fit_end) & tr.date.lt(fit_end) & tr.outcome.notna()].copy()
            assert len(tr) > 1000
            assert tr.source_date.le(tr.date).all() and tr.maturity.lt(fit_end).all()
            full = [c for c in meta['columns'] if tr[c].nunique() > 1]
            remaining = [c for c in full if c not in first]
            scopes = dict(base=first, core=meta['core'], original=[c for c in full if c in original or c.startswith('jq_')], all=full, remaining=remaining)
            for c in meta['columns']:
                admissions.append(dict(side=side, fold=fold, name=c, admitted=c in full, fit_coverage=tr[c].notna().mean(), fit_unique=tr[c].nunique(),
                                       reason='' if c in full else 'all_missing_or_constant_in_fit', original_776=c in original))
            pd.DataFrame(admissions).to_csv(OUT / 'fold_admission.csv', index=False)
            cal = frame.loc[frame.date.ge(fit_end) & frame.date.lt(select_start) & frame.maturity.lt(select_start) & frame.outcome.notna()].copy()
            va = prior.evaluation_slice(frame, select_start, test_start)
            te = prior.evaluation_slice(frame, test_start, test_end)
            cuts = fit_grade_cuts(tr.outcome)
            cache = {kind: grouped_training(tr, kind) for kind in ['day', 'cross']}
            prior.save_json(OUT / f'contract_{side}_{fold}.json', dict(scopes=scopes, rows=dict(fit=len(tr), calibration=len(cal), validation=len(va), test=len(te)),
                fit_label_max=str(tr.maturity.max()), calibration_label_max=str(cal.maturity.max()), actual_test_end=str(te.date.max()),
                oof_sha256=prior.sha256(oof_path), original_features=len(scopes['original']), expanded_features=len(full), base_warm_end=contract['warm_end'],
                fit_end=fit_end, select_start=select_start, test_start=test_start, test_end=test_end, grade_cuts=cuts.tolist()))
            distribution(frame, full, bounds, OUT / f'distribution_{side}_{fold}.csv', tr)
            raw={}; gates={}; ranks={}; head_days={}; calibrators={}
            for arm, (method, scope, grouping) in ARMS.items():
                train, q = cache[grouping]; cols = scopes[scope]
                model_path = OUT / f'model_{side}_{fold}_{arm}.ubj'
                started = time.monotonic()
                if model_path.exists():
                    model = xgb.Booster(); model.load_model(model_path)
                    assert model.feature_names == cols
                else:
                    print('TRAIN', side, fold, arm, len(train), len(cols), flush=True)
                    if method.startswith('regression'):
                        dm = prior.matrix(train, cols); dm.set_label(train.outcome.clip(-.5, .5))
                        w = 1 / train.groupby('date').date.transform('size'); dm.set_weight(w / w.mean())
                        model = xgb.train({**prior.PARAMS, 'objective': 'reg:squarederror'}, dm, num_boost_round=120 if method.endswith('120') else 60)
                    else:
                        label = gains(train.outcome, cuts)
                        dm = prior.matrix(train, cols); dm.set_label(label)
                        if method in ['ranknet', 'local05', 'local10']:
                            # Return ordering preserves differences between two losing trades.
                            band = {'local05': .05, 'local10': .1}.get(method)
                            pairs = make_pairs(q, train.oof_percentile, train.outcome, bandwidth=band)
                            objective = lambda z, d: derivatives(z, pairs)
                        elif method.startswith('cross'):
                            pairs = cross_date_pairs(train.date, train.oof_score, train.outcome, bandwidth=float(method[5:])/10000)
                            objective = lambda z, d: derivatives(z, pairs)
                        elif method == 'auxiliary':
                            net = train.outcome.to_numpy(); scale = max(float(np.std(net)), .005)
                            targets = terms(net, train.groupby('date').outcome.transform('mean').to_numpy(), scale)
                            weights = LossWeights(absolute=.25, incremental=.25, downside=1.)
                            w = (1 / train.groupby('date').date.transform('size')).to_numpy(copy=True); w /= w.mean()
                            objective = lambda z, d: auxiliary_derivatives(z, targets, weights, w)
                        else:
                            objective = HeadLambdaObjective(q, label, method, k=20, dates=train.date, cross_only=grouping=='cross')
                        model = xgb.train({**prior.PARAMS, 'base_score': 0., 'disable_default_eval_metric': 1}, dm, num_boost_round=60, obj=objective)
                    model.save_model(model_path)
                times.append(dict(side=side, fold=fold, arm=arm, features=len(cols), split_used_features=len(model.get_score(importance_type='weight')), seconds=time.monotonic()-started))
                values = [model.predict(prior.matrix(d, cols), output_margin=True) for d in [cal, va, te]]
                iso = IsotonicRegression(out_of_bounds='clip').fit(values[0], cal.outcome.to_numpy().clip(-.5, .5), sample_weight=1/cal.groupby('date').date.transform('size'))
                gates[arm] = [iso.predict(v) for v in values[1:]]
                ranks[arm] = [percentile_scores(f.date, v) for f, v in zip([va, te], values[1:])]
                raw[arm] = values[2]
                calibrators[arm] = dict(x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist(),
                    test_outside_calibration=float(((values[2] < iso.X_thresholds_[0]) | (values[2] > iso.X_thresholds_[-1])).mean()))
                for stage, f, v in [('validation', va, values[1]), ('test', te, values[2])]:
                    h = head_metrics(f, v, cuts)
                    h.to_csv(OUT / f'head_{side}_{fold}_{arm}_{stage}.csv', index=False)
                    if stage == 'test': head_days[arm] = h
                    for k, group in h.groupby('k'):
                        head_rows.append(dict(side=side, fold=fold, arm=arm, stage=stage, k=k, ndcg=group.ndcg.mean(),
                                              stress_net=group.stress_net.mean(), win=group.win.mean(), unknown=group.unknown.sum()))
            prior.save_json(OUT / f'calibrators_{side}_{fold}.json', calibrators)
            te[prior.KEYS + ['source_date', 'outcome', 'maturity']].assign(**raw).to_parquet(OUT / f'predictions_{side}_{fold}.parquet', index=False)
            vc = calendar_for(frame, select_start, test_start); tc = calendar_for(frame, test_start, test_end)
            sleeves = 6 if side == 'left' else 1
            vst={}; tst={}; daily={}; picked={}
            for arm in ARMS:
                for alpha in ([1.] if arm == 'base50' else [.5, 1.]):
                    vg, tg = [(1-alpha)*base + alpha*sub for base, sub in zip(gates['base50'], gates[arm])]
                    vr, ts = [(1-alpha)*base + alpha*sub for base, sub in zip(ranks['base50'], ranks[arm])]
                    for threshold in THRESHOLDS:
                        name = f'{arm}@{alpha}#{threshold}'
                        vst[name], _, _ = evaluate_policy(va, vg, threshold, vc, sleeves, rank_score=vr)
                        tst[name], daily[name], picked[name] = evaluate_policy(te, tg, threshold, tc, sleeves, rank_score=ts)
                        grid.append(dict(side=side, fold=fold, name=name, **{f'validation_{k}': v for k, v in vst[name].items()}, **{f'test_{k}': v for k, v in tst[name].items()}))
            vst['cash'], _, _ = evaluate_policy(va, np.zeros(len(va)), np.inf, vc, sleeves)
            tst['cash'], daily['cash'], picked['cash'] = evaluate_policy(te, np.zeros(len(te)), np.inf, tc, sleeves)
            names = list(vst)
            families = {a: [n for n in names if n.startswith(a+'@')] for a in ARMS}
            families.update({
                'ranknet_family': [n for n in names if any(n.startswith(a+'@') for a in ['ranknet_all', 'ranknet_remaining', 'local05_all', 'local10_all', 'local05_remaining'])],
                'lambdarank_family': [n for n in names if n.startswith('lambdarank')],
                'lambdaloss_family': [n for n in names if n.startswith('lambdaloss')],
                'cross_family': [n for n in names if n.startswith('cross')],
                'overall': names})
            choices = {f'{f}_{mode}': pick_policy(vst, ns, mode) for f, ns in families.items() for mode in ['return', 'win']}
            prior.save_json(OUT / f'choices_{side}_{fold}.json', choices)
            for policy, arm in choices.items():
                ref = choices['core129_return']
                ci = intervals((daily[arm].pnl - daily[ref].pnl).to_numpy())
                filled = picked[arm].loc[~picked[arm].unfilled]
                summaries.append(dict(side=side, fold=fold, policy=policy, selected=arm, **tst[arm],
                    filled_attempts=len(filled), filled_win=float(filled.outcome.gt(0).mean()) if len(filled) else np.nan,
                    known_filled=int(filled.outcome.notna().sum()),
                    versus_core_capital_pp=(tst[arm]['capital_pnl']-tst[ref]['capital_pnl'])*100,
                    daily_delta_ci_low_bps=ci[0]*10000, daily_delta_ci_high_bps=ci[1]*10000))
                daily[arm].to_csv(OUT / f'daily_{side}_{fold}_{policy}.csv')
                picked[arm].to_parquet(OUT / f'picks_{side}_{fold}_{policy}.parquet', index=False)
            pd.DataFrame(summaries).to_csv(OUT / 'cash_comparisons.csv', index=False)
            pd.DataFrame(head_rows).to_csv(OUT / 'head_metrics.csv', index=False)
            pd.DataFrame(grid).to_csv(OUT / 'validation_grid.csv', index=False)
            pd.DataFrame(times).to_csv(OUT / 'training_times.csv', index=False)
            print('FINISHED', side, fold, flush=True)
    prior.save_json(OUT / 'complete.json', dict(status='complete', model_runs=len(times), production_promoted=False, dataset_hashes=meta['dataset_hashes']))


if __name__ == '__main__':
    main()
