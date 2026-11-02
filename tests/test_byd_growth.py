import unittest
import numpy as np
import pandas as pd
from quant.research.byd_mwu import ROOT
from quant.research.byd_growth import growth_experts,constrained_choice,paired_return_bootstrap


class GrowthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market=pd.read_parquet(ROOT/'reports/research/byd_mwu_20261012/market_snapshot.parquet')
        cls.legacy=pd.read_parquet(ROOT/'reports/research/byd_allocation_20261013/expert_targets.parquet').drop(columns='date')

    def test_causal_prefix(self):
        full,groups,states=growth_experts(self.market,self.legacy)
        short,g,s=growth_experts(self.market.iloc[:800],self.legacy.iloc[:800])
        pd.testing.assert_frame_equal(short,full.iloc[:800])
        pd.testing.assert_frame_equal(s,states.iloc[:800]);self.assertEqual(g,groups)

    def test_targets_and_cash(self):
        t,_,s=growth_experts(self.market,self.legacy)
        self.assertTrue(t.ge(0).all().all());self.assertTrue(t.le(1).all().all())
        self.assertTrue(t.cash.eq(0).all());self.assertTrue(t.iloc[:269].eq(0).all().all())
        self.assertTrue(s.up_score.between(0,1).all())

    def test_smooth_core_does_not_exit_on_down_state(self):
        t,_,_=growth_experts(self.market,self.legacy)
        self.assertGreaterEqual(t.soft_core45_tilt25.iloc[269:].min(),.20-1e-10)

    def ledger(self, daily, position):
        dates=pd.bdate_range('2013-01-01','2016-12-31')
        n=len(dates)
        r=np.full(n,daily); equity=np.cumprod(1+r)*1e6
        frame=pd.DataFrame({'date':dates,'signal_date':dates-pd.Timedelta(days=1),'return':r,
            'equity':equity,'position':position,'turnover':0.,'trade_shares':0.,'cost':0.,'cost_fraction':0.,
            'requested_shares':0.,'unfilled_shares':0.,'capacity_limited':False,'cash_limited':False,'blocked':''})
        for name in ('commission_cost','transfer_cost','stamp_cost','spread_cost','impact_cost'):frame[name]=0.
        return frame

    def test_feasible_candidate_beats_high_return_near_cash(self):
        ledgers={'cash_like':self.ledger(.001,.03),'participating':self.ledger(.0003,.40)}
        chosen,rows=constrained_choice(ledgers,2016,list(ledgers))
        self.assertEqual(chosen,'participating')
        self.assertFalse(rows[0]['feasible'])

    def test_bootstrap_uses_supplied_period_before_2020(self):
        a=self.ledger(.0004,.40);b=self.ledger(.0003,.40)
        result=paired_return_bootstrap(a,b,'b')
        self.assertEqual(result['sessions'],len(a))
        self.assertEqual(result['actual_start'],'2013-01-01')
        self.assertAlmostEqual(result['annual_mean_difference'],.0252)

    def test_future_returns_cannot_change_choice(self):
        ledgers={'a':self.ledger(.0004,.40),'b':self.ledger(.0003,.40)}
        before,_=constrained_choice(ledgers,2016,list(ledgers))
        ledgers['b'].loc[ledgers['b'].date.dt.year.eq(2016),'return']=.10
        after,_=constrained_choice(ledgers,2016,list(ledgers));self.assertEqual(before,after)

if __name__=='__main__':unittest.main()
