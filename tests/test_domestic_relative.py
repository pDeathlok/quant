import unittest
import numpy as np
import pandas as pd
from quant.research.domestic_relative import bounded_budget,build_targets,relative_choice
from quant.research.domestic_portfolio import load_panel,simulate
from quant.research.domestic_basket import GROUPS


class RelativeTests(unittest.TestCase):
    def test_bounded_budget(self):
        w=bounded_budget(np.array([100.,5,4,3,2,1]),np.ones(6,dtype=bool))
        self.assertAlmostEqual(w.sum(),1.);self.assertTrue((w<=.20+1e-12).all())
        w=bounded_budget(np.ones(6),np.array([1,1,0,0,0,0]),.5)
        self.assertAlmostEqual(w.sum(),.4)
        self.assertFalse(bounded_budget(np.zeros(6),np.ones(6,dtype=bool)).any())

    def test_target_prefix_and_caps(self):
        p=load_panel();short=load_panel(end_date='2019-12-31');n=len(short.dates)
        forecasts={k:np.zeros_like(p.data['close']) for k in ('ridge_21','ridge_excess21','tree_excess21')}
        full=build_targets(p,forecasts);prefix=build_targets(short,{k:v[:n] for k,v in forecasts.items()})
        for name,w in full.items():
            np.testing.assert_allclose(w[:n],prefix[name]);self.assertTrue((w.sum(axis=1)<=1+1e-12).all())
            for g in GROUPS:self.assertTrue((w[:,g].sum(axis=1)<=.20+1e-12).all())

    def test_relative_choice_falls_back(self):
        dates=pd.bdate_range('2015-01-01','2017-12-31');n=len(dates)
        f=pd.DataFrame({'date':dates,'signal_date':dates-pd.Timedelta(days=1),'return':np.full(n,.0003),
                        'position':.9,'cost':0.,'orders':0,'capacity_limited':False})
        chosen,rows=relative_choice({'expert_equal':f,'worse':f.assign(return_=0).assign(**{'return':-.0001})},2018)
        self.assertEqual(chosen,'expert_equal');self.assertFalse(any(r['feasible'] for r in rows))

if __name__=='__main__':unittest.main()
