"""Reconstruct selections, predictions and capital without using training outputs as truth."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
from iterate_right_ranknet_v3 import OUT,SOURCE,FOLDS,ARMS,context_features
from evaluate_full_rank_comparison import calendar_for
from quant.ml.rank_fusion import file_sha256


def main() -> None:
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert file_sha256(SOURCE/'dataset_right.parquet')==protocol['source_sha256']
    assert file_sha256(ROOT/'scripts/research/iterate_right_ranknet_v3.py')==protocol['script_sha256']
    frame=pd.read_parquet(SOURCE/'dataset_right.parquet').sort_values(['date','symbol']).reset_index(drop=True)
    assert not frame.duplicated(['symbol','date']).any()
    assert frame.source_date.lt(frame.date).all()
    context_features(frame)
    results=pd.read_csv(OUT/'results.csv');training=pd.read_csv(OUT/'training.csv')
    oof=pd.read_parquet(OUT/'oof.parquet');assert not oof.duplicated(['symbol','date']).any()
    hashes={};checks=[]
    for fold,(fit_end,select_start,test_start,test_end) in FOLDS.items():
        contract=json.loads((OUT/f'contract_{fold}.json').read_text())
        fit=frame.loc[frame.date.lt(fit_end)&frame.maturity.lt(fit_end)&frame.outcome.notna()]
        cal=frame.loc[frame.date.ge(fit_end)&frame.date.lt(select_start)&frame.maturity.lt(select_start)&frame.outcome.notna()]
        assert len(fit)==contract['rows']['more'] and len(cal)==contract['rows']['calibration']
        assert fit.maturity.max()<pd.Timestamp(fit_end) and cal.maturity.max()<pd.Timestamp(select_start)
        predictions=pd.read_parquet(OUT/f'predictions_{fold}.parquet')
        expected=frame.loc[frame.date.ge(test_start)&frame.date.lt(test_end)&frame.planned_maturity.lt(test_end)]
        pd.testing.assert_frame_equal(predictions[['symbol','date']].reset_index(drop=True),expected[['symbol','date']].reset_index(drop=True))
        for arm in ARMS:
            path=OUT/f'{fold}_{arm}.ubj';model=xgb.Booster();model.load_model(path)
            fresh=model.predict(xgb.DMatrix(expected[model.feature_names].astype('float32'),nthread=2),output_margin=True)
            np.testing.assert_array_equal(fresh,predictions[arm])
            hashes[path.name]=file_sha256(path)
            actual_train=fit.merge(oof,on=['symbol','date'],validate='one_to_one') if arm in ['local05_more','regression_same_oof'] else fit
            if arm=='regression_oldfit':actual_train=fit.loc[fit.date.lt('2025-07-01')&fit.maturity.lt('2025-07-01')]
            row=training.loc[training.fold.eq(fold)&training.arm.eq(arm)].iloc[0]
            assert len(actual_train)==row.rows and len(model.feature_names)==row.features
        grid=pd.read_csv(OUT/f'grid_{fold}.csv')
        calendar=pd.DatetimeIndex(calendar_for(frame,test_start,test_end))
        for result in results.loc[results.fold.eq(fold)].itertuples():
            family,mode=result.policy.rsplit('_',1)
            group=grid if family=='overall' else grid.loc[grid.name.str.startswith(family+'#')]
            eligible=group.loc[group.validation_capital_pnl.gt(0)&group.validation_attempts.ge(100)]
            if mode=='win':eligible=eligible.loc[eligible.validation_win_rate.ge(.55)]
            order=['validation_win_rate','validation_capital_pnl'] if mode=='win' else ['validation_capital_pnl','validation_win_rate']
            selected='cash' if eligible.empty else eligible.sort_values(order,ascending=False,kind='stable').name.iloc[0]
            assert result.selected==selected
            picks=pd.read_parquet(OUT/f'picks_{fold}_{result.policy}.parquet')
            assert len(picks)==result.attempts and not picks.duplicated(['symbol','date']).any()
            assert picks.groupby('date').size().le(20).all()
            wealth=1.;busy=pd.Timestamp.min
            for date,group in picks.groupby('date'):
                assert busy<=date
                held=group.loc[~group.unfilled]
                if len(held):
                    busy=held.maturity.max()
                    if held.outcome.isna().any():busy=max(busy,calendar[-1]+pd.Timedelta(days=1))
                wealth*=1+group.outcome.fillna(-.10).sum()/20
            assert np.isclose(wealth-1,result.capital_pnl,atol=1e-12)
            held=picks.loc[~picks.unfilled]
            assert len(held)==result.filled
            if len(held):assert np.isclose(held.outcome.gt(0).mean(),result.filled_win)
        checks.append(f'{fold}: mature labels, input scope, 12 exact predictions, 26 validation-only choices and settled-capital/win replays')
    # Additional OOF quarters must only see earlier matured labels.
    for start,end in [('2025-07-01','2025-10-01'),('2025-10-01','2026-01-01')]:
        earlier=frame.loc[frame.date.lt(start)&frame.maturity.lt(start)&frame.outcome.notna()]
        assert earlier.date.max()<pd.Timestamp(start) and earlier.maturity.max()<pd.Timestamp(start)
        model=xgb.Booster();path=OUT/f'oof_{start}.ubj';model.load_model(path)
        later=frame.loc[frame.date.ge(start)&frame.date.lt(end)]
        predicted=model.predict(xgb.DMatrix(later[model.feature_names].astype('float32'),nthread=2))
        actual=oof.merge(later[['symbol','date']],on=['symbol','date'],validate='one_to_one').sort_values(['date','symbol'])
        np.testing.assert_array_equal(predicted,actual.oof_score)
        hashes[path.name]=file_sha256(path)
    output=dict(passed=True,checks=checks,main_models=24,new_oof_models=2,source_sha256=protocol['source_sha256'],
                model_sha256=hashes,limitations=['The two test periods overlap; they are not independent replications.',
                'Frozen source ends September 2026; outcomes with unknown exits retain the conservative sleeve freeze.',
                'No candidate is promoted by inspecting its test result.'])
    (OUT/'verification.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
