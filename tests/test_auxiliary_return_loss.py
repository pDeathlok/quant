import numpy as np
import pandas as pd
from quant.research.auxiliary_return_loss import LossWeights, terms, joint_loss, derivatives, daily_picks


def test_joint_derivatives_match_numerical_loss():
    z = np.array([-2., -.2, .3, 1.6])
    t = terms(np.array([-.04, -.01, .02, .08]), np.zeros(4), .03)
    w = LossWeights(.3, .1, .5)
    grad, hess = derivatives(z, t, w, np.ones(4))
    eps = 1e-4
    np.testing.assert_allclose(grad, (joint_loss(z+eps,t,w)-joint_loss(z-eps,t,w))/(2*eps), rtol=1e-6)
    np.testing.assert_allclose(hess, (joint_loss(z+eps,t,w)-2*joint_loss(z,t,w)+joint_loss(z-eps,t,w))/eps**2, rtol=1e-5)
    assert (hess > 0).all()


def test_unknown_exit_not_removed_before_selection():
    d = pd.DataFrame({'date': pd.to_datetime(['2025-01-01']*3), 'symbol': ['a','b','c'], 'outcome': [np.nan,.1,-.1]})
    chosen, day = daily_picks(d, np.array([3.,2.,1.]), k=1)
    assert chosen.symbol.tolist() == ['a']
    assert day.net.isna().all()
    assert day.stress.iloc[0] == -.1
