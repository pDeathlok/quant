import unittest
import numpy as np
import pandas as pd
from quant.research.domestic_basket import GROUPS,project_baskets,ranked_targets
from quant.research.domestic_portfolio import load_panel,simulate,Panel


class BasketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.panel=load_panel()

    def test_broad_group_cap(self):
        w=project_baskets(np.ones(7))
        self.assertLessEqual(w[1]+w[2],.20+1e-12);self.assertLessEqual(w.sum(),1+1e-12)
        np.testing.assert_array_equal(project_baskets(np.zeros(7)),np.zeros(7))

    def test_cash_and_locked_byd(self):
        p=self.panel;score=np.full_like(p.data['close'],-1.)
        w=ranked_targets(p,score,0.);self.assertFalse(w.any())
        score[:,3]=1.;w=ranked_targets(p,score,0.,np.full(len(score),.15))
        np.testing.assert_allclose(w[-1,[0,3]],[.15,.20]);self.assertAlmostEqual(w[-1].sum(),.35)

    def test_rank_and_group_constraint(self):
        p=self.panel;score=np.tile(np.arange(1,8),(len(p.dates),1));w=ranked_targets(p,score,0.)
        for g in GROUPS:self.assertTrue((w[:,g].sum(axis=1)<=.20+1e-12).all())
        self.assertTrue((w.sum(axis=1)<=1+1e-12).all())

    def test_daily_repair_and_prefix(self):
        p=self.panel;w=np.zeros_like(p.data['close']);w[:,0]=.20
        full,orders=simulate(p,w,frequency=20,basket_groups=GROUPS)
        self.assertTrue(full.basket_repair_requested.any())
        self.assertGreater(len(orders),0)
        cut=p.dates.get_loc('2020-01-02');short=Panel(p.dates[:cut],{k:v[:cut] for k,v in p.data.items()},p.features[:cut],p.feature_names,[e for e in p.dividends if e['ex_index']<cut],p.coverage)
        prefix,_=simulate(short,w[:cut],frequency=20,basket_groups=GROUPS)
        pd.testing.assert_frame_equal(full.iloc[:len(prefix)].reset_index(drop=True),prefix)

    def test_reject_invalid_target(self):
        p=self.panel;w=np.zeros_like(p.data['close']);w[:,1:3]=.15
        with self.assertRaises(ValueError):simulate(p,w,basket_groups=GROUPS)

if __name__=='__main__':unittest.main()
