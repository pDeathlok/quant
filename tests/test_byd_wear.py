from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from quant.research.byd_allocation import AllocationConfig
from quant.research.byd_mwu import _Account, market_arrays
from quant.research.byd_wear import (
    COST_COLUMNS, WearConfig, commission_rate, impact_rate, long_allocation_path, past_score,
    same_orders_gross, stamp_rate, transfer_rate, wear_cost_target, wear_metrics, wear_simulate, wear_trade_day,
)
from test_byd_mwu import market_frame

GROSS = WearConfig(commission=0, minimum_commission=0, spread=0, impact_eta=0,
                   historical_transfer=False, fixed_transfer=0, charge_stamp=False,
                   dividend_tax_reserve=0)
CFG = AllocationConfig(deadband=0, max_step=1)


class WearContractTests(unittest.TestCase):
    def test_historical_transfer_and_stamp_boundaries(self) -> None:
        wear = WearConfig()
        self.assertEqual(transfer_rate(20150731, wear), .0000255)
        self.assertEqual(transfer_rate(20150801, wear), .00002)
        self.assertEqual(transfer_rate(20220428, wear), .00002)
        self.assertEqual(transfer_rate(20220429, wear), .00001)
        self.assertEqual(stamp_rate(20230827, wear), .001)
        self.assertEqual(stamp_rate(20230828, wear), .0005)
        self.assertEqual(stamp_rate(20230828, GROSS), 0)

    def test_square_root_impact_increases_with_size_and_volatility(self) -> None:
        wear = WearConfig(impact_eta=.5)
        small = impact_rate(100, 100_000, .02, wear)
        self.assertAlmostEqual(impact_rate(400, 100_000, .02, wear), 2*small)
        self.assertAlmostEqual(impact_rate(100, 100_000, .04, wear), 2*small)
        self.assertAlmostEqual(impact_rate(100, 400_000, .02, wear), small/2)

    def test_broker_history_stress_is_separate_from_base_contract(self) -> None:
        base = WearConfig()
        stress = replace(base, historical_commission=True)
        self.assertEqual(commission_rate(20140101, base), .0003)
        self.assertEqual(commission_rate(20140101, stress), .001)
        self.assertEqual(commission_rate(20150101, stress), .0005)
        self.assertEqual(commission_rate(20200101, stress), .0003)

    def test_calendar_cagr_uses_elapsed_dates_not_quote_count(self) -> None:
        dates = pd.to_datetime(["2020-01-01", "2020-12-31"])
        market = market_frame(dates, np.array([100.,110.]))
        ledger = wear_simulate(market, np.ones(2), [CFG]*2, GROSS, start="2020-12-31")
        # Buying at the second open earns no prior-year price gap.
        ledger.loc[0, "return"] = .10
        m = wear_metrics(ledger)
        self.assertAlmostEqual(m["calendar_years"], 365/365.25)
        self.assertAlmostEqual(m["cagr"], 1.1**(365.25/365)-1)

    def test_fees_components_and_gross_same_orders_identity(self) -> None:
        market = market_frame(pd.date_range("2023-08-25", periods=6), np.full(6, 100.))
        targets = np.array([.5, .5, 0., .5, 0., 0.])
        wear = WearConfig(initial_cash=100_000, impact_eta=0)
        net = wear_simulate(market, targets, [CFG]*6, wear, start="2023-08-26")
        np.testing.assert_allclose(net.cost, net[list(COST_COLUMNS)].sum(axis=1), atol=1e-9)
        gross = same_orders_gross(market, net, wear)
        self.assertAlmostEqual(gross.equity.iloc[-1]-net.equity.iloc[-1], net.cost.sum(), places=7)
        np.testing.assert_array_equal(gross.shares, net.shares)
        self.assertGreater(net.stamp_cost.sum(), 0)

    def test_capacity_uses_prior_volume_not_today_volume(self) -> None:
        market = market_frame(pd.date_range("2012-08-09", periods=3), np.full(3, 100.))
        market.loc[0, "volume_shares"] = 10_000
        market.loc[1, "volume_shares"] = 100_000_000
        ledger = wear_simulate(market, np.ones(3), [CFG]*3, replace(GROSS, initial_cash=100_000), start="2012-08-10")
        self.assertEqual(ledger.trade_shares.iloc[0], 100)
        self.assertTrue(ledger.capacity_limited.iloc[0])
        self.assertGreater(ledger.unfilled_shares.iloc[0], 0)

    def test_same_day_close_cannot_change_open_fill_impact(self) -> None:
        dates = pd.date_range("2012-07-15", periods=30)
        closes = 100+np.sin(np.arange(30))*2
        market = market_frame(dates, closes)
        market["open"] = 100.
        altered = market.copy()
        altered.loc[25, "close"] = 200
        a = wear_simulate(market, np.ones(30)*.5, [CFG]*30, start=str(dates[25].date()))
        b = wear_simulate(altered, np.ones(30)*.5, [CFG]*30, start=str(dates[25].date()))
        self.assertEqual(a.fill_price.iloc[0], b.fill_price.iloc[0])
        self.assertEqual(a.modeled_impact_rate.iloc[0], b.modeled_impact_rate.iloc[0])

    def test_full_position_keeps_cash_positive_and_buy_lots(self) -> None:
        market = market_frame(pd.date_range("2012-08-09", periods=4), np.full(4, 33.))
        wear = WearConfig(initial_cash=100_000, spread=.02, impact_eta=1)
        ledger = wear_simulate(market, np.ones(4), [CFG]*4, wear, start="2012-08-10")
        self.assertTrue((ledger.cash >= -1e-6).all())
        self.assertTrue((ledger.shares % 100 == 0).all())
        self.assertTrue(ledger.cash_limited.any())

    def test_cost_no_trade_preserves_shares_during_gap_and_corporate_actions(self) -> None:
        market = market_frame(pd.date_range("2012-08-09", periods=4), np.array([100.,100.,150.,50.]))
        market.loc[3, "split"] = 3
        proposals = np.array([.5,.5,.6,.6])
        cfg = replace(CFG, cost_gamma=4)
        ledger = wear_simulate(market, proposals, [CFG,cfg,cfg,cfg], replace(GROSS, initial_cash=100_000), start="2012-08-10")
        np.testing.assert_array_equal(ledger.shares, [500,500,1500])
        self.assertEqual(ledger.trade_shares.iloc[1], 0)

    def test_deadband_keeps_odd_inventory_without_float_rounding_order(self) -> None:
        market = market_frame(pd.date_range("2012-08-09", periods=2), np.full(2, 30.))
        arrays = market_arrays(market)
        arrays["known_sigma"] = np.zeros(2)
        account = _Account(30_000., shares=2333, equity=99_990.)
        result = wear_trade_day(account, .7, 1, arrays, GROSS, replace(CFG, deadband=.03))
        self.assertEqual(result["trade_shares"], 0)
        self.assertEqual(result["shares"], 2333)

    def test_early_online_learning_and_ledger_prefix_are_causal(self) -> None:
        market = market_frame(pd.date_range("2012-08-09", periods=40), np.full(40, 100.))
        targets = pd.DataFrame({"a": np.ones(40), "b": np.zeros(40)})
        rewards = pd.DataFrame({"a": np.ones(40)*.02, "b": np.zeros(40)})
        features = pd.DataFrame({"distance_120": np.zeros(40), "vol20": np.zeros(40)})
        path, _ = long_allocation_path(market, targets, rewards, features, CFG)
        short, _ = long_allocation_path(market.iloc[:30], targets.iloc[:30], rewards.iloc[:30], features.iloc[:30], CFG)
        np.testing.assert_allclose(path[:30], short)
        self.assertGreater(path[2], path[0])  # Learning starts in 2012, not hard-coded 2015.
        full = wear_simulate(market, path, [CFG]*40)
        prefix = wear_simulate(market.iloc[:30], short, [CFG]*30)
        pd.testing.assert_frame_equal(full.iloc[:len(prefix)], prefix)

    def test_annual_selection_score_ignores_future_years(self) -> None:
        dates = pd.bdate_range("2013-01-01", "2017-12-31")
        ledger = pd.DataFrame({"date": dates, "return": np.sin(np.arange(len(dates)))*.01,
                               "position": .3, "turnover": 0., "trade_shares": 0, "cost": 0.})
        score = past_score(ledger, 2016)
        ledger.loc[ledger.date.dt.year.ge(2016), "return"] = .1
        self.assertEqual(score, past_score(ledger, 2016))

    def test_controller_accounts_for_size_dependent_wear(self) -> None:
        cfg = replace(CFG, cost_gamma=4)
        a = wear_cost_target(.9,.5,100_000,100,100_000,.001,20260101,cfg,replace(GROSS,impact_eta=.1))
        b = wear_cost_target(.9,.5,100_000_000,100,100_000,.001,20260101,cfg,replace(GROSS,impact_eta=1))
        self.assertLess(b,a)


if __name__ == "__main__":
    unittest.main()
