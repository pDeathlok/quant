import unittest
import numpy as np
from quant.research.domestic_portfolio import load_panel
from quant.research.domestic_uncapped import rank_assets,build_targets


class UncappedTests(unittest.TestCase):
    def test_full_single_asset_and_cash(self):
        p=load_panel();score=np.zeros_like(p.data['close']);score[:,0]=1
        w=rank_assets(p,score,1)
        self.assertEqual(w[-1,0],1.);self.assertAlmostEqual(w[-1].sum(),1.)
        self.assertFalse(rank_assets(p,score,1,np.zeros_like(score,dtype=bool)).any())

    def test_future_independent_targets(self):
        p=load_panel();short=load_panel(end_date='2019-12-31');n=len(short.dates)
        forecasts={k:np.zeros_like(p.data['close']) for k in ('ridge_21','ridge_63','ridge_excess21','tree_excess21')}
        full=build_targets(p,forecasts);prefix=build_targets(short,{k:v[:n] for k,v in forecasts.items()})
        self.assertEqual(len(full),40)
        for name,w in full.items():
            np.testing.assert_allclose(w[:n],prefix[name]);self.assertTrue((w.sum(axis=1)<=1+1e-12).all())
            self.assertTrue((w[~p.data['eligible'].astype(bool)]==0).all())
        self.assertTrue(full['buy_0'][-1,0]>.2)

if __name__=='__main__':unittest.main()
