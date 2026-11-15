import numpy as np
import pandas as pd
import pytest
from quant.research.head_lambdaloss import HeadLambdaObjective,fit_grade_cuts,gains,head_metrics,metric_weights


def test_lambdarank_weight_equals_actual_ndcg_swap_including_boundary():
    g=np.array([1.,7.,3.,15.]);k=2
    disc=np.where(np.arange(1,5)<=k,1/np.log2(np.arange(1,5)+1),0.)
    ideal=np.sort(g)[::-1]@disc
    for i,j in [(0,1),(1,2),(0,3),(2,3)]:
        swapped=g.copy();swapped[i],swapped[j]=g[j],g[i]
        expected=abs(swapped@disc-g@disc)/ideal
        assert metric_weights(i+1,j+1,g[i],g[j],ideal,'lambdarank',k)==pytest.approx(expected)


def test_lambdaloss2_and_plus_plus_match_paper_equations():
    ra,rb=2,7;ga,gb=7.,1.;den=20.
    delta=1/np.log2(6)-1/np.log2(7)
    rho=1/np.log2(3)-1/np.log2(8)
    assert metric_weights(ra,rb,ga,gb,den,'lambdaloss2',5)==pytest.approx(delta*6/20)
    assert metric_weights(ra,rb,ga,gb,den,'lambdaloss2pp',5)==pytest.approx((rho+5*delta)*6/20)
    assert metric_weights(6,7,ga,gb,den,'lambdaloss2pp',5)==0


@pytest.mark.parametrize('method',['ranknet_head','lambdarank','lambdaloss2','lambdaloss2pp'])
def test_gradients_query_isolation_dynamic_head_and_hessian_bound(method):
    group=np.repeat([0,1],4);gain=np.array([0,1,3,7,0,3,1,7.])
    obj=HeadLambdaObjective(group,gain,method,k=2,tail_samples=10)
    z=np.array([.8,.6,.4,.1,.9,.7,.2,-.2]);a,b,y,w=obj.weighted_pairs(z)
    assert (group[a]==group[b]).all()
    assert (np.isin(a,[0,1,4,5])|np.isin(b,[0,1,4,5])).all()
    gg,hh=obj(z)
    def loss(x):
        u,v,t,weight=obj.weighted_pairs(x);m=x[u]-x[v]
        return np.sum(weight*(np.logaddexp(0,m)-t*m))
    eps=1e-6
    numeric=np.array([(loss(z+np.eye(8)[i]*eps)-loss(z-np.eye(8)[i]*eps))/(2*eps) for i in range(8)])
    np.testing.assert_allclose(gg,numeric,atol=1e-7)
    from scipy.special import expit
    p=expit(z[a]-z[b]);exact=np.zeros((8,8))
    for i,j,c in zip(a,b,w*p*(1-p)):
        v=np.eye(8)[i]-np.eye(8)[j];exact+=c*np.outer(v,v)
    assert np.linalg.eigvalsh(np.diag(hh)-exact).min()>-1e-8
    assert abs(gg.sum())<1e-8
    za=-z;aa,bb,_,_=obj.weighted_pairs(za)
    assert (np.isin(aa,[2,3,6,7])|np.isin(bb,[2,3,6,7])).all()
    assert set(zip(a,b))!=set(zip(aa,bb))


def test_cross_dates_all_negative_and_relevance_no_future_fit():
    obj=HeadLambdaObjective(np.zeros(6),np.arange(6.),'lambdaloss2pp',2,dates=[1,1,2,2,3,3],cross_only=True)
    a,b,_,_=obj.weighted_pairs(np.arange(6.));dates=np.repeat([1,2,3],2)
    assert (dates[a]!=dates[b]).all()
    zero=HeadLambdaObjective(np.zeros(4),np.zeros(4),'lambdarank',2)
    grad,hess=zero(np.arange(4.));assert not grad.any() and np.isfinite(hess).all()
    cuts=fit_grade_cuts([-.1,0,.01,.02,.03,.04]);frozen=cuts.copy()
    g=gains([-.01,0,np.nan,.01,1.],cuts)
    assert (g[:3]==0).all() and g[-1]==15
    np.testing.assert_array_equal(cuts,frozen)


def test_unknown_selected_not_removed_from_head():
    d=pd.DataFrame({'symbol':['a','b','c'],'date':pd.to_datetime(['2024-01-01']*3),'outcome':[np.nan,.1,-.1]})
    result=head_metrics(d,[3,2,1],[.01,.02,.03],ks=(1,2)).set_index('k')
    assert result.loc[1,'unknown']==1 and result.loc[1,'stress_net']==-.1
    assert result.loc[1,'ndcg']==0


def test_calibrated_gate_does_not_destroy_raw_head_order():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
    from evaluate_ranknet_cash_policy import evaluate_policy
    dates=pd.date_range('2024-01-01',periods=2)
    d=pd.DataFrame({'symbol':[f'{i:03}' for i in range(25)],'date':[dates[0]]*25,
                    'outcome':[.1]*25,'at_upper':[False]*25})
    _,_,p=evaluate_policy(d,np.ones(25)*.01,0,dates,1,rank_score=np.arange(25))
    assert p.symbol.iloc[0]=='024' and '000' not in set(p.symbol)
    _,_,empty=evaluate_policy(d,np.ones(25)*-.01,0,dates,1,rank_score=np.arange(25))
    assert empty.empty


def test_financial_swap_lambdas_distinguish_losses_and_ignore_internal_swaps():
    from quant.research.return_lambdas import ReturnHeadObjective
    obj=ReturnHeadObjective([0]*4,[-.01,-.02,-.1,.1],k=2)
    z=np.array([4.,3.,2.,1.]);a,b,y,w=obj.weighted_pairs(z)
    assert not ((a<2)&(b<2)).any()
    assert (0,2) in set(zip(a,b))  # Both lose money, but by different amounts.
    g,h=obj(z);eps=1e-5
    def loss(x):
        i,j,t,v=obj.weighted_pairs(x);m=x[i]-x[j]
        return np.sum(v*(np.logaddexp(0,m)-t*m))
    numeric=[(loss(z+np.eye(4)[i]*eps)-loss(z-np.eye(4)[i]*eps))/(2*eps) for i in range(4)]
    np.testing.assert_allclose(g,numeric,atol=1e-8)
