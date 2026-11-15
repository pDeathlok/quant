import numpy as np
import pandas as pd
from scipy.special import expit

from quant.research.local_ranknet import make_pairs, cross_date_pairs, derivatives, mature_before, percentile_scores, choose_blend


def test_exact_neighbourhood_no_cross_date_and_no_bin_boundary():
    dates=np.repeat(pd.date_range('2020-01-01',periods=2),4)
    scores=np.tile([.049,.051,.10,.9],2)
    outcome=np.arange(8.)
    pairs=make_pairs(dates,scores,outcome,.05,per_row=30)
    assert (dates[pairs.left]==dates[pairs.right]).all()
    assert (abs(scores[pairs.left]-scores[pairs.right]) <= .05+1e-12).all()
    assert (0,1) in set(zip(pairs.left,pairs.right))
    totals=pd.Series(pairs.weight).groupby(dates[pairs.left]).sum()
    np.testing.assert_allclose(totals,totals.iloc[0])


def test_pair_gradient_and_hessian_upper_bound():
    pairs=make_pairs(np.zeros(5),np.linspace(0,1,5),[.1,-.2,.3,.4,0.])
    z=np.array([.1,-.3,.4,.2,.05]);g,h=derivatives(z,pairs)
    def loss(x):
        margin=x[pairs.left]-x[pairs.right]
        return np.sum(pairs.weight*(np.logaddexp(0,margin)-pairs.target*margin))
    eps=1e-5
    numeric=np.array([(loss(z+np.eye(5)[i]*eps)-loss(z-np.eye(5)[i]*eps))/(2*eps) for i in range(5)])
    np.testing.assert_allclose(g,numeric,atol=1e-8)
    p=expit(z[pairs.left]-z[pairs.right]);exact=np.zeros((5,5))
    for i,j,w in zip(pairs.left,pairs.right,pairs.weight*p*(1-p)):
        v=np.eye(5)[i]-np.eye(5)[j];exact+=w*np.outer(v,v)
    assert np.linalg.eigvalsh(np.diag(h)-exact).min()>-1e-10
    assert abs(g.sum())<1e-10


def test_mature_purge_and_label_independent_score_grouping():
    d=pd.DataFrame({'date':pd.to_datetime(['2020-01-01']*3),'maturity':pd.to_datetime(['2020-01-03','2020-01-05','2020-01-06']),'outcome':[1.,2.,np.nan]})
    assert mature_before(d,'2020-01-05').tolist()==[True,False,False]
    np.testing.assert_allclose(percentile_scores(d.date,[3,1,2]),[1,1/3,2/3])
    # Choice consumes validation only; a return increase that lowers win is rejected.
    v={k:pd.DataFrame({'stress':[r],'stress_win':[w]}) for k,r,w in [('base50',.01,.6),('a',.02,.59),('b',.015,.61)]}
    assert choose_blend(v)=='b'


def test_cross_date_neighbours_and_date_balancing():
    dates=np.repeat(np.arange(3),4)
    scores=np.linspace(-.01,.01,12)
    outcome=np.sin(np.arange(12))
    pairs=cross_date_pairs(dates,scores,outcome,.005,per_row=30)
    assert (dates[pairs.left]!=dates[pairs.right]).all()
    assert (abs(scores[pairs.left]-scores[pairs.right])<=.005+1e-12).all()
    totals=pd.Series(pairs.weight).groupby(dates[pairs.left]).sum()
    np.testing.assert_allclose(totals,totals.iloc[0])


def test_cash_threshold_does_not_redistribute_unused_slots():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
    from evaluate_ranknet_cash_policy import evaluate_policy, pick_policy
    dates=pd.date_range('2024-01-01',periods=3)
    frame=pd.DataFrame({'symbol':['a','b'],'date':[dates[0]]*2,'outcome':[.1,-.1],'at_upper':[False,False]})
    stats,day,picks=evaluate_policy(frame,[.02,-.02],.01,dates,1)
    assert len(picks)==1
    np.testing.assert_allclose(stats['capital_pnl'],.1/20)
    assert stats['flat_days']==2/3
    empty,_,_=evaluate_policy(frame,[.02,-.02],np.inf,dates,1)
    assert empty['capital_pnl']==0 and empty['flat_days']==1
    assert pick_policy({'bad':{'capital_pnl':-.01,'attempts':200,'win_rate':.8}},['bad'],'return')=='cash'


def test_delayed_labels_stay_in_test_pool_and_capital_remains_locked():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
    from evaluate_local_ranknet_fusion import evaluation_slice
    from evaluate_ranknet_cash_policy import evaluate_policy
    cal=pd.date_range('2024-01-01',periods=10)
    d=pd.DataFrame({'symbol':['a','b'],'date':[cal[0],cal[6]],'outcome':[.1,.2],
                    'maturity':[cal[9],cal[8]],'planned_maturity':[cal[5],cal[8]],'locked_limit_up':[False,False]})
    truncated=evaluation_slice(d,cal[0],cal[8])
    assert len(truncated)==1 and truncated.outcome.isna().all()
    stats,_,picks=evaluate_policy(d,[.1,.1],0,cal,6)
    assert picks.symbol.tolist()==['a']
    assert stats['delayed_capital_blocked_attempts']==1
    unknown=d.copy();unknown.loc[0,'outcome']=np.nan;unknown.loc[0,'maturity']=cal[2]
    stats,_,picks=evaluate_policy(unknown,[.1,.1],0,cal,6)
    assert picks.symbol.tolist()==['a']
    assert stats['unknown']==1
