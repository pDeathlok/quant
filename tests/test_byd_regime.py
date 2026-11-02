"""Causal regime and risk-control invariants."""
import unittest
import numpy as np
import pandas as pd
from quant.research.byd_allocation import AllocationConfig
from quant.research.byd_mwu import ROOT
from quant.research.byd_regime import regime_experts, mix_path, risk_simulate
from quant.research.byd_wear import START, WearConfig


class RegimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market=pd.read_parquet(ROOT/'reports/research/byd_mwu_20261012/market_snapshot.parquet').iloc[:850].copy()
        cls.targets,cls.regimes=regime_experts(cls.market)

    def test_experts_are_bounded_and_cash_explicit(self):
        self.assertTrue(self.targets.ge(0).all().all())
        self.assertTrue(self.targets.le(1).all().all())
        self.assertTrue(self.targets.cash.eq(0).all())
        self.assertTrue(self.targets.iloc[:269].eq(0).all().all())

    def test_downtrend_experts_exit(self):
        down=self.regimes==0
        self.assertGreater(down.sum(),0)
        self.assertTrue(self.targets.loc[down,'regime_switch'].eq(0).all())
        self.assertTrue(self.targets.loc[down,'downtrend_defense'].eq(0).all())

    def test_volatility_caps_only_reduce_regime_proposal(self):
        self.assertTrue((self.targets.regime_vol_10<=self.targets.regime_switch).all())
        self.assertTrue((self.targets.regime_vol_10<=self.targets.regime_vol_15).all())

    def test_future_prices_cannot_relabel_past(self):
        short=self.market.iloc[:600]
        targets,states=regime_experts(short)
        pd.testing.assert_frame_equal(targets,self.targets.iloc[:600])
        np.testing.assert_array_equal(states,self.regimes[:600])

    def test_cash_account_no_orders_cost_or_drawdown(self):
        cfg=AllocationConfig(method='equal')
        ledger=risk_simulate(self.market,np.zeros(len(self.market)),[cfg]*len(self.market),
            self.regimes,WearConfig(),np.full(len(self.market),.08))
        self.assertTrue(ledger.trade_shares.eq(0).all())
        self.assertTrue(ledger.cost.eq(0).all())
        self.assertTrue(ledger.equity.eq(1_000_000).all())
        self.assertFalse(ledger.halted.any())

    def test_stop_uses_prior_equity_and_reentry_has_cooldown(self):
        cfg=AllocationConfig(method='equal',deadband=0,max_step=1)
        ledger=risk_simulate(self.market,np.ones(len(self.market)),[cfg]*len(self.market),
            self.regimes,WearConfig(),np.full(len(self.market),.001))
        self.assertTrue(ledger.halted.any())
        self.assertTrue(ledger.loc[ledger.halted,'proposal'].eq(0).all())
        switches=np.flatnonzero(ledger.halted.ne(ledger.halted.shift(fill_value=False)).to_numpy())
        for begin,end in zip(switches[::2],switches[1::2]):
            self.assertGreaterEqual(end-begin,19)
            self.assertEqual(ledger.regime.iloc[end],2)

    def test_context_update_attributes_return_to_previous_close_state(self):
        targets=pd.DataFrame({'risk':[0.,1.,1.],'cash':[0.,0.,0.]})
        rewards=pd.DataFrame({'risk':[0.,.01,-.01],'cash':[0.,0.,0.]})
        _,weights=mix_path(targets,rewards,np.array([0,2,0]),
            AllocationConfig(method='context',eta=.3),1)
        self.assertAlmostEqual(weights[1,0],.5)
        self.assertGreater(weights[2,0],.5)

if __name__=='__main__':
    unittest.main()
