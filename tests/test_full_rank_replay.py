import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant.research.left_rule_replay import replay_left_rules
from quant.research.left_side_unified_features import compute_left_side_rule_features


def test_array_replay_matches_canonical_and_is_prefix_invariant():
    rng = np.random.default_rng(91)
    n = 360
    close = 30*np.exp(np.cumsum(rng.normal(0, .045, n)))
    open_ = close*np.exp(rng.normal(0, .012, n))
    daily = pd.DataFrame(dict(date=pd.bdate_range('2020-01-01', periods=n), symbol='TEST',
        close=close, open=open_, high=np.maximum(close,open_)*1.01, low=np.minimum(close,open_)*.99,
        pre_close=np.r_[30,close[:-1]], volume=rng.lognormal(8, .8, n)))
    actual = replay_left_rules(daily)
    pd.testing.assert_frame_equal(actual, compute_left_side_rule_features(daily), check_exact=False, rtol=1e-12, atol=1e-12)
    for nrows in [15, 24, 30, 73, 220]:
        prefix = replay_left_rules(daily.iloc[:nrows])
        pd.testing.assert_frame_equal(prefix, actual.iloc[:nrows], check_exact=False, rtol=1e-12, atol=1e-12)
    # Future prices and extreme volume cannot rewrite earlier event/peak states.
    changed = daily.copy(); changed.loc[220:, ['open','high','low','close','volume']] *= 5
    pd.testing.assert_frame_equal(replay_left_rules(changed).iloc[:220], actual.iloc[:220])


def test_feature_publication_join_is_strict_and_keeps_all_keys():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/research'))
    from build_full_rank_dataset import strict_join
    keys = pd.DataFrame({'symbol':['A']*3,'date':pd.to_datetime(['2024-01-02','2024-01-03','2024-01-15'])})
    source = pd.DataFrame({'symbol':['A']*2,'date':pd.to_datetime(['2024-01-02','2024-01-03']),'factor':[2.,300.]})
    joined = strict_join(keys,source,prior_day=True)
    assert len(joined) == len(keys)
    assert np.isnan(joined.factor.iloc[0])
    assert joined.factor.iloc[1] == 2
    assert np.isnan(joined.factor.iloc[2])


def test_completed_period_readiness_ignores_future_history():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/research'))
    from audit_full_rank_readiness import readiness
    observations = pd.DataFrame({'symbol':'NEW', 'date':pd.bdate_range('2024-01-02','2025-04-30')})
    keys = pd.DataFrame({'symbol':'NEW','date':pd.to_datetime(['2024-09-20','2024-10-31'])})
    counts = readiness(keys, observations, 'M')
    np.testing.assert_array_equal(counts,[8,10])
    truncated = observations.loc[observations.date.le('2024-10-31')]
    np.testing.assert_array_equal(counts, readiness(keys,truncated,'M'))


def test_last_signal_keeps_next_close_entry_and_settlement_tail(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/research'))
    from evaluate_full_rank_comparison import calendar_for
    from evaluate_ranknet_cash_policy import evaluate_policy
    dates = pd.bdate_range('2024-01-03','2024-01-10')
    monkeypatch.setattr(pd,'read_csv',lambda path:pd.DataFrame({'date':dates}))
    frame=pd.DataFrame({'symbol':['A'],'date':pd.to_datetime(['2024-01-03']), 'outcome':[.1],
                        'maturity':pd.to_datetime(['2024-01-09']), 'locked_limit_up':[False]})
    calendar=calendar_for(frame,'2024-01-03','2024-01-11')
    _,day,_=evaluate_policy(frame,np.array([1.]),0.,calendar,6)
    assert day.loc['2024-01-03','active_slots']==0
    assert day.loc['2024-01-04','active_slots']==1
    assert day.loc['2024-01-09','active_slots']==0
    assert len(day)==6
