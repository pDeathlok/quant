"""Transfer preselected application policies to frozen production ranking outputs."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_model_applications as experiment
from quant.ml.rank_fusion import RankFusion
from quant.routine.right_side_unified_production import _normalize_ranking_scores


def initialize_production():
    """Load the frozen test cohort with the official production base scores."""
    experiment.initialize('2026')
    frame = experiment._FRAME
    hashes = {}
    for side in ['left', 'right']:
        source = pd.read_parquet(experiment.prior.OUT / f'dataset_{side}.parquet')
        source = source.loc[source.date.ge('2025-12-31')].copy()
        if side == 'left':
            path = experiment.ROOT / 'models/production/left_side_ranknet_v6/manifest.json'
            predictions = RankFusion.load(path).predict(source)
            score = predictions.ranking_score.to_numpy()
        else:
            path = experiment.ROOT / 'models/production/right_side_unified_canonical_v2/ranking.joblib'
            artifact = joblib.load(path)
            probabilities = artifact['model'].predict_proba(source)[:, 1]
            score = _normalize_ranking_scores(probabilities, artifact['score_normalization']) / 100
        hashes[str(path)] = experiment.prior.digest(path)
        mapping = source[experiment.prior.KEYS].copy()
        mapping['production_score'] = score
        target = frame.loc[frame.side.eq(side), experiment.prior.KEYS].merge(mapping, on=experiment.prior.KEYS, how='left', validate='one_to_one')
        # Only the test-period decisions are evaluated. Earlier scores remain unused.
        frame.loc[frame.side.eq(side), 's_return'] = target.production_score.to_numpy()
    return frame, hashes


def main():
    frame, hashes = initialize_production()
    choices = pd.read_csv(experiment.OUT / 'selected.csv')
    names = {'baseline', 'scope_left', 'scope_right', 'score_return_path', 'score_return_rank',
             'score_return_risk', 'score_three_way', 'weighted_low_volatility', 'weighted_liquidity',
             'overlap_mean', 'overlap_agreement', 'overlap_best_expected'}
    names |= set(choices.loc[choices.fold.eq('2026'), 'selected']) - {'cash'}
    specs = {p.name: p for p in experiment.policies()}
    start, end = experiment._CALENDAR.searchsorted(pd.to_datetime(['2026-01-01', '2026-09-04']))
    rows = []
    for name in sorted(names):
        spec = specs[name]
        groups, coverage = experiment.candidates(frame, spec, experiment._THRESHOLDS, '2026')
        for cost in [.001, .002, .004, .006]:
            metrics, nav, trades = experiment.simulate(replace(spec, cost=cost), groups, experiment._MARKET, int(start), int(end), coverage)
            rows.append({'name': name, 'round_trip_cost_bp': cost * 10000, **metrics})
            if cost == .002:
                nav.to_parquet(experiment.OUT / f'production_nav_{name}.parquet', index=False)
                trades.to_parquet(experiment.OUT / f'production_trades_{name}.parquet', index=False)
    pd.DataFrame(rows).to_csv(experiment.OUT / 'production_application_results.csv', index=False)
    experiment.save('production_application_scope.json', {
        'period': '2026-01-01 to2026-09-03', 'policies': sorted(names), 'hashes': hashes,
        'ranking': 's_return replaced by actual published left fusion score and official frozen right CDF score. Other scores remain frozen research model outputs.',
        'gate': 'Expected-return gates retain the independently calibrated research return estimator, for both sides. No good_path probability is treated as expected return.',
        'selection': 'Names fixed from research validation winners plus named diagnostic baselines. No production test winner promoted.',
        'limitations': 'Research candidate universe and feature materialization; production snapshot parity not certified. 2026 is a previously researched historical period.',
        'script_sha256': experiment.prior.digest(__file__),
    })
    print(pd.DataFrame(rows).query('round_trip_cost_bp==20')[['name', 'return', 'max_drawdown', 'closed', 'average_exposure']].to_string(index=False))


if __name__ == '__main__':
    main()
