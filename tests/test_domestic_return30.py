import unittest
import numpy as np
import pandas as pd
from quant.research.domestic_portfolio import load_panel
from quant.research.domestic_basket import GROUPS
from quant.research.domestic_return30 import base_targets,distribute_broad,return_choice


class Return30Tests(unittest.TestCase):
    def test_independent_prefix_and_caps(self):
        full=load_panel();short=load_panel(end_date='2019-12-31');n=len(short.dates)
        targets=base_targets(full);small=base_targets(short)
        for name,w in targets.items():
            np.testing.assert_allclose(w[:n],small[name]);self.assertTrue((w.sum(axis=1)<=1+1e-12).all())
            for g in GROUPS:self.assertTrue((w[:,g].sum(axis=1)<=.20+1e-12).all())

    def test_broad_selection_uses_known_signal(self):
        p=load_panel();b=np.zeros((len(p.dates),6));b[:,1]=.20
        w=distribute_broad(p,b,'momentum63');i=-1;j=p.feature_names.index('ret_63')
        chosen=1+int(p.features[i,2,j]>p.features[i,1,j])
        self.assertAlmostEqual(w[i,chosen],.20);self.assertEqual(w[i,3-chosen],0.)

    def test_30percent_choice_and_fallback(self):
        d=pd.bdate_range('2015-01-01','2017-12-31');n=len(d)
        def frame(r):return pd.DataFrame({'date':d,'signal_date':d-pd.Timedelta(days=1),'return':r,'position':1.,'cost':0.,'orders':0,'capacity_limited':False})
        base=frame(np.full(n,.0002));bad=np.full(n,.002);bad[10]=-.40
        chosen,_=return_choice({'expert_v1':base,'high_dd':frame(bad)},2018)
        self.assertEqual(chosen,'expert_v1')
        good=frame(np.full(n,.0003));chosen,_=return_choice({'expert_v1':base,'better':good},2018)
        self.assertEqual(chosen,'better')
        chosen,_=return_choice({'expert_v1':frame(bad)},2018);self.assertEqual(chosen,'expert_v1')

if __name__=='__main__':unittest.main()
