import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from quant.research.domestic_portfolio import load_panel,capped_weights,matured_labels,simulate,CODES,Panel
from quant.research.byd_wear import WearConfig


class PortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.panel=load_panel()

    def test_independent_feature_rebuild_has_no_future_information(self):
        p=self.panel;short=load_panel(end_date='2020-01-01');n=len(short.dates)
        np.testing.assert_allclose(short.features,p.features[:n])
        for key in p.data:np.testing.assert_allclose(short.data[key],p.data[key][:n])

    def test_weight_constraints(self):
        w=capped_weights(np.ones(7));self.assertLessEqual(w.sum(),1+1e-12)
        self.assertLessEqual(w[0],.10);self.assertLessEqual(w[5],.15)
        self.assertLessEqual(w[[0,1,2,5,6]].sum(),.70+1e-12)

    def test_no_prelisting_broker_position(self):
        p=self.panel;w=np.zeros_like(p.data['close']);w[:,5]=.15
        ledger,orders=simulate(p,w)
        self.assertTrue(ledger.loc[ledger.date.lt('2016-08-08'),'shares_512880.SH'].eq(0).all())
        if len(orders):self.assertTrue(orders.date.ge('2016-08-08').all())

    def test_etf_no_stock_stamp_or_transfer(self):
        p=self.panel;w=np.zeros_like(p.data['close']);w[:,3]=.5;w[2000:,3]=0
        ledger,orders=simulate(p,w)
        self.assertGreater(len(orders),0);self.assertTrue(orders.stamp.eq(0).all());self.assertTrue(orders.transfer.eq(0).all())
        self.assertGreaterEqual(ledger.cash.min(),-1e-6)
        self.assertTrue((orders.loc[orders.shares.gt(0),'shares']%100).eq(0).all())

    def test_cash_is_shared_not_multiple_accounts(self):
        p=self.panel;w=np.tile([.1,.2,.15,.3,.15,.05,.05],(len(p.dates),1))
        ledger,_=simulate(p,w)
        self.assertGreaterEqual(ledger.cash.min(),-1e-6)
        self.assertTrue((ledger.position<=1+1e-10).all())
        self.assertTrue(ledger.filter(like='shares_').ge(0).all().all())

    def test_reverse_split_marks_nonquoted_day(self):
        p=self.panel;i=p.dates.get_loc('2015-04-14');a=2
        self.assertAlmostEqual(p.data['split'][i,a],.28032483)
        self.assertAlmostEqual(p.data['close'][i,a]*p.data['split'][i,a],p.data['close'][i-1,a])
        self.assertFalse(p.data['tradable'][i,a])

    def test_label_needs_complete_horizon(self):
        p=self.panel;y=matured_labels(p,21)
        self.assertTrue(np.isnan(y[-22:]).all())
        first=np.flatnonzero(np.isfinite(y[:,5]))[0]
        self.assertTrue(p.data['eligible'][first,5])
        value=p.data['adj_open'][first+22,5]/p.data['adj_open'][first+1,5]-1-.0016
        self.assertAlmostEqual(y[first,5],np.clip(value,-.5,.5))

    def test_feature_and_account_prefix(self):
        p=self.panel;cut=p.dates.get_loc('2020-01-02')
        short=Panel(p.dates[:cut],{k:v[:cut].copy() for k,v in p.data.items()},p.features[:cut],p.feature_names,
            [e for e in p.dividends if e['ex_index']<cut],p.coverage)
        targets=np.tile([0,.2,0,.3,.5,0,0],(len(p.dates),1))
        full,_=simulate(p,targets);prefix,_=simulate(short,targets[:cut])
        pd.testing.assert_frame_equal(full.iloc[:len(prefix)].reset_index(drop=True),prefix)

    def test_dividend_receivable_excluded_from_spendable_cash(self):
        p=self.panel;w=np.zeros_like(p.data['close']);w[:,6]=.35
        ledger,_=simulate(p,w)
        self.assertGreater(ledger.receivable.max(),0)
        self.assertGreaterEqual(ledger.cash.min(),0)

if __name__=='__main__':unittest.main()
