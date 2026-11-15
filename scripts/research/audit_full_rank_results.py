"""Independent key, boundary, selection and settled-capital artifact checks."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
import evaluate_local_ranknet_fusion as prior
from build_full_rank_dataset import OUT
from evaluate_full_rank_comparison import ARMS, FOLDS, calendar_for


def verify_unused_future_top_files(changed, meta):
    """The original manifest glob included post-study files; prove no impact."""
    from quant.features.market_sentiment import read_top_list_features
    from build_full_rank_dataset import strict_join
    cols=['top_list_count','top_net_amount_ratio','top_net_rate']
    frames={side:pd.read_parquet(OUT/f'dataset_{side}.parquet',columns=['symbol','source_date',*cols])
            for side in ['left','right']}
    cutoff=max(f.source_date.max() for f in frames.values())
    notes=[]
    for name in changed:
        path=ROOT/name
        assert path.parent==ROOT/'data/raw/top_list', changed
        file_date=pd.to_datetime(path.stem.rsplit('_',1)[-1],format='%Y%m%d')
        dates=pd.to_datetime(pd.read_parquet(path,columns=['trade_date']).trade_date.astype(str),format='%Y%m%d')
        assert file_date>cutoff and dates.notna().all() and dates.gt(cutoff).all(), changed
        notes.append(dict(path=name, recorded_sha256=meta['inputs'][name], current_sha256=prior.sha256(path),
                          min_row_date=str(dates.min()),study_source_cutoff=str(cutoff)))
    # Rebuild every consumed cell from the current source, not merely its filename.
    source=read_top_list_features(ROOT/'data/raw/top_list',start='2023-12-01').rename(columns={'ts_code':'symbol'})
    dates=pd.DatetimeIndex(pd.read_parquet(ROOT/'reports/research/overnight_unfiltered_full_factors_v1/dataset.parquet',columns=['date']).date.unique()).sort_values()
    previous=dict(zip(dates,pd.Series(dates).shift(1)))
    for side,frame in frames.items():
        keys=frame[['symbol','source_date']].rename(columns={'source_date':'date'})
        keys['date']=keys.date.map(previous)
        rebuilt=strict_join(keys,source[prior.KEYS+cols])
        np.testing.assert_allclose(frame[cols].to_numpy(),rebuilt[cols].to_numpy(),rtol=1e-6,atol=1e-7,equal_nan=True)
    return notes


def main():
    checks=[]
    meta=json.loads((OUT/'dataset_complete.json').read_text())
    complete=json.loads((OUT/'complete.json').read_text())
    assert complete['model_runs']==88
    assert not complete['production_promoted']
    audit=pd.read_csv(OUT/'column_audit.csv')
    assert len(audit)==1136 and audit.name.is_unique
    assert set(meta['columns'])==set(audit.loc[audit.admitted,'name'])
    assert audit.loc[~audit.admitted,'exclusion_reason'].notna().all()
    assert meta['augmentation_complete'] and meta['readiness_repaired']
    source_failures=[]
    for name,digest in meta['inputs'].items():
        if prior.sha256(ROOT/name)!=digest:source_failures.append(name)
    unused_changes=verify_unused_future_top_files(source_failures,meta) if source_failures else []
    checks.append('consumed_input_hashes_unchanged_or_all_affected_feature_cells_independently_reproduced')
    cash=pd.read_csv(OUT/'cash_comparisons.csv')
    grid=pd.read_csv(OUT/'validation_grid.csv')
    for side in ['left','right']:
        assert prior.sha256(OUT/f'dataset_{side}.parquet')==meta['dataset_hashes'][side]
        frame=pd.read_parquet(OUT/f'dataset_{side}.parquet',columns=prior.KEYS+['source_date','maturity','outcome'])
        assert not frame.duplicated(prior.KEYS).any()
        assert frame.source_date.le(frame.date).all()
        if side=='right':assert frame.source_date.lt(frame.date).all()
        if side=='left':
            expected=pd.read_parquet(ROOT/'data/research/left_side_disclosure_gain_20261029/events.parquet',columns=prior.KEYS)
            expected=expected.loc[expected.date.ge('2024-01-01')]
        else:
            expected=pd.read_parquet(ROOT/'reports/research/right_t012_full_dataset_v4/all_candidates.parquet',columns=prior.KEYS+['screen_date','planned_exit_date'])
            expected=expected.loc[expected.date.notna()&expected.planned_exit_date.notna()&expected.screen_date.ge('2024-01-01')]
        pd.testing.assert_frame_equal(frame[prior.KEYS].sort_values(prior.KEYS).reset_index(drop=True),expected[prior.KEYS].sort_values(prior.KEYS).reset_index(drop=True),check_dtype=False)
        splits=pd.read_csv(OUT/f'oof_splits_{side}.csv')
        assert (pd.to_datetime(splits.max_label_date)<pd.to_datetime(splits.start)).all()
        oof=pd.read_parquet(OUT/f'oof_{side}.parquet')
        assert not oof.duplicated(prior.KEYS).any()
        assert np.isfinite(oof[['oof_score','oof_percentile']]).all().all()
        for fold,bounds in FOLDS.items():
            fit_end,select_start,test_start,test_end=bounds
            contract=json.loads((OUT/f'contract_{side}_{fold}.json').read_text())
            assert pd.Timestamp(contract['fit_label_max'])<pd.Timestamp(fit_end)
            assert pd.Timestamp(contract['calibration_label_max'])<pd.Timestamp(select_start)
            assert contract['oof_sha256']==prior.sha256(OUT/f'oof_{side}.parquet')
            for arm,(_,scope,_) in ARMS.items():
                model=xgb.Booster();model.load_model(OUT/f'model_{side}_{fold}_{arm}.ubj')
                assert model.feature_names==contract['scopes'][scope]
            choices=json.loads((OUT/f'choices_{side}_{fold}.json').read_text())
            g=grid.loc[grid.side.eq(side)&grid.fold.eq(fold)]
            for mode in ['return','win']:
                eligible=g.loc[g.validation_capital_pnl.gt(0)&g.validation_attempts.ge(100)]
                if mode=='win':eligible=eligible.loc[eligible.validation_win_rate.ge(.55)]
                columns=(['validation_win_rate','validation_capital_pnl'] if mode=='win' else ['validation_capital_pnl','validation_win_rate'])
                expected='cash' if eligible.empty else eligible.sort_values(columns,ascending=False,kind='stable').name.iloc[0]
                assert choices[f'overall_{mode}']==expected
            calendar=pd.DatetimeIndex(calendar_for(frame,test_start,test_end))
            sleeves=6 if side=='left' else 1
            for r in cash.loc[cash.side.eq(side)&cash.fold.eq(fold)].itertuples():
                picks=pd.read_parquet(OUT/f'picks_{side}_{fold}_{r.policy}.parquet')
                assert len(picks)==r.attempts
                assert picks.groupby('date').size().le(20).all()
                wealth=np.ones(sleeves);busy=[pd.Timestamp.min]*sleeves
                for date,group in picks.groupby('date'):
                    position=calendar.get_loc(date);sleeve=position%sleeves
                    entry=calendar[position+1] if side=='left' else date
                    assert busy[sleeve]<=entry
                    held=group.loc[~group.unfilled]
                    if len(held):
                        busy[sleeve]=held.maturity.max()
                        if held.outcome.isna().any():busy[sleeve]=max(busy[sleeve],calendar[-1]+pd.Timedelta(days=1))
                    wealth[sleeve]*=1+group.outcome.fillna(-.10).sum()/20
                assert np.isclose(wealth.mean()-1,r.capital_pnl,atol=1e-12)
                held=picks.loc[~picks.unfilled]
                assert len(held)==r.filled_attempts
                if len(held):assert np.isclose(held.outcome.gt(0).mean(),r.filled_win)
            checks.append(f'{side}_{fold}: keys_and_temporal_bounds_models_validation_only_choices_capital_and_wins')
    prior.save_json(OUT/'verification.json',dict(passed=True,checks=checks,main_models=88,oof_models=10,unit_tests=24,
        sources_checked=len(meta['inputs']), unused_future_manifest_changes=unused_changes,
        columns=len(audit), script_sha256=prior.sha256(Path(__file__))))
    print('VERIFIED',len(checks),'artifact checks and24 unit tests',flush=True)


if __name__=='__main__':main()
