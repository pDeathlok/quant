from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from quant.research.byd_allocation import (
    AllocationConfig, OnlineAllocator, allocation_path, bayesian_search, cost_target,
    expanded_experts, inner_score, past_pruned_mask, project_simplex, signal_from_snapshot,
    simulate_path, state_features,
)
from quant.research.byd_mwu import ExecutionConfig
from test_byd_mwu import ZERO_FEES, market_frame


class AllocationContractTests(unittest.TestCase):
    def test_all_methods_keep_finite_simplex_and_reward_good_expert(self) -> None:
        for method in ("equal", "mwu", "share", "discount", "adah2011", "squint_grid", "multiwindow", "context", "omd", "ftrl"):
            learner = OnlineAllocator(3, AllocationConfig(method=method, share=.02, forget=.99))
            for _ in range(80):
                learner.observe(np.array([.01, -.01, 0]), 2)
            weights = learner.weights(2)
            with self.subTest(method=method):
                self.assertTrue(np.isfinite(weights).all())
                self.assertTrue((weights >= 0).all())
                self.assertAlmostEqual(weights.sum(), 1)
                if method != "equal":
                    self.assertGreater(weights[0], weights[1])

    def test_adah2011_first_update_matches_paper(self) -> None:
        learner = OnlineAllocator(2, AllocationConfig(method="adah2011"))
        learner.observe(np.array([.02, -.02]))
        self.assertAlmostEqual(learner.weights()[0], 1 / (1 + np.exp(-1)))
        self.assertGreaterEqual(learner.gap, 0)

    def test_squint_first_update_matches_discrete_prior_integral(self) -> None:
        learner = OnlineAllocator(2, AllocationConfig(method="squint_grid"))
        learner.observe(np.array([.02, -.02]))
        masses = np.array([
            np.sum(learner.etas * np.exp(learner.etas * r - learner.etas ** 2 * .25))
            for r in (.5, -.5)
        ])
        np.testing.assert_allclose(learner.weights(), masses / masses.sum(), atol=1e-12)

    def test_simplex_projection_closed_form(self) -> None:
        np.testing.assert_allclose(project_simplex(np.array([2., 1., -3.])), [1., 0., 0.])

    def test_context_feedback_only_updates_preceding_order_state(self) -> None:
        learner = OnlineAllocator(2, AllocationConfig(method="context"))
        learner.observe(np.array([.02, -.02]), 1)
        self.assertGreater(learner.weights(1)[0], .5)
        np.testing.assert_allclose(learner.weights(5), [.5, .5])

    def test_cost_controller_retains_small_trade_and_moves_for_large_signal(self) -> None:
        cfg = AllocationConfig(cost_gamma=4)
        execution = ExecutionConfig()
        self.assertEqual(cost_target(.5001, .5, 100_000, .001, 20260101, cfg, execution), .5)
        self.assertGreater(cost_target(.9, .5, 100_000, .001, 20260101, cfg, execution), .8)
        robust = replace(cfg, risk=1000)
        self.assertLess(cost_target(.9, .5, 100_000, .001, 20260101, robust, execution), .5)

    def test_feedback_cannot_change_same_day_order_and_prefix_is_invariant(self) -> None:
        market = market_frame(pd.date_range("2014-12-30", periods=30), np.full(30, 100.))
        targets = pd.DataFrame({"a": np.ones(30), "b": np.zeros(30)})
        features = pd.DataFrame({"distance_120": np.zeros(30), "vol20": np.zeros(30)})
        rewards = pd.DataFrame({"a": np.full(30, .02), "b": np.full(30, -.02)})
        for method in ("mwu", "adah2011", "squint_grid", "multiwindow", "context", "omd", "ftrl"):
            cfg = AllocationConfig(method=method, deadband=0, max_step=1)
            path, _ = allocation_path(market, targets, rewards, features, cfg)
            short, _ = allocation_path(market.iloc[:20], targets.iloc[:20], rewards.iloc[:20], features.iloc[:20], cfg)
            np.testing.assert_allclose(short, path[:20], atol=1e-12)
            ledger = simulate_path(market, path, [cfg] * len(market), ZERO_FEES)
            self.assertEqual(ledger.requested_target.iloc[0], .5)
            changed = rewards.copy()
            changed.loc[2] = [-.02, .02]
            alt, _ = allocation_path(market, targets, changed, features, cfg)
            other = simulate_path(market, alt, [cfg] * len(market), ZERO_FEES)
            self.assertEqual(ledger.requested_target.iloc[0], other.requested_target.iloc[0])

    def test_cost_no_trade_keeps_inventory_across_overnight_gap_and_split(self) -> None:
        market = market_frame(pd.date_range("2014-12-31", periods=4), np.array([100., 100., 150., 50.]))
        market.loc[3, "split"] = 3
        path = np.array([.5, .5, .6, .6])
        cfg = AllocationConfig(cost_gamma=4, deadband=0, max_step=1)
        configs = [replace(cfg, cost_gamma=0), cfg, cfg, cfg]
        ledger = simulate_path(market, path, configs, ZERO_FEES)
        self.assertEqual(ledger.shares.iloc[0], 500)
        self.assertEqual(ledger.shares.iloc[1], 500)
        self.assertEqual(ledger.shares.iloc[2], 1500)
        self.assertEqual(ledger.trade_shares.iloc[1], 0)
        self.assertEqual(ledger.trade_shares.iloc[2], 0)
        self.assertEqual(ledger.blocked.iloc[1], "cost_no_trade")

    def test_horizon_models_and_rules_prefix_with_mature_labels(self) -> None:
        rng = np.random.default_rng(20261013)
        n = 725
        closes = 100 * np.exp(np.cumsum(rng.normal(0, .015, n)))
        market = market_frame(pd.bdate_range("2011-01-01", periods=n), closes)
        market["high"], market["low"] = market.close * 1.01, market.close * .99
        signal = signal_from_snapshot(market)
        signal["high"], signal["low"] = signal.close * 1.01, signal.close * .99
        index = pd.DataFrame({"trade_date": market.date.dt.strftime("%Y%m%d"), "close": closes * 10})
        legacy = pd.DataFrame({"original_range": np.full(n, .35)})
        full, features, audit = expanded_experts(signal, index, legacy)
        short, short_features, _ = expanded_experts(signal.iloc[:700], index.iloc[:700], legacy.iloc[:700])
        pd.testing.assert_frame_equal(full.iloc[:700], short)
        pd.testing.assert_frame_equal(features.iloc[:700], short_features)
        self.assertTrue((pd.to_datetime(audit.last_label_end) <= pd.to_datetime(audit.fit_date)).all())
        self.assertEqual(set(audit.horizon), {1, 21, 63})

    def test_annual_pruning_and_inner_score_ignore_future(self) -> None:
        dates = pd.bdate_range("2017-01-01", "2021-12-31")
        n = len(dates)
        market = market_frame(dates, np.full(n, 100.))
        targets = pd.DataFrame({"a": np.linspace(.2, .8, n), "b": np.linspace(.2, .8, n), "cash": np.zeros(n)})
        rewards = pd.DataFrame({"a": np.sin(np.arange(n)) * .01, "b": np.sin(np.arange(n)) * .01, "cash": np.zeros(n)})
        mask, audit = past_pruned_mask(market, targets, rewards)
        changed = rewards.copy()
        changed.loc[market.date.dt.year.ge(2020), "b"] = .1
        altered, _ = past_pruned_mask(market, targets, changed)
        np.testing.assert_array_equal(mask[market.date.dt.year.le(2020)], altered[market.date.dt.year.le(2020)])
        self.assertTrue(all(pd.Timestamp(row["cutoff"]).year < row["year"] for row in audit))
        ledger = pd.DataFrame({"date": dates, "return": rewards.a, "position": .3, "turnover": 0., "cost": 0., "trade_shares": 0})
        score = inner_score(ledger, 2020)
        ledger.loc[ledger.date.dt.year.ge(2020), "return"] = .1
        self.assertEqual(score, inner_score(ledger, 2020))

    def test_bayesian_search_obeys_finite_budget_and_observed_best(self) -> None:
        values = np.linspace(0, 1, 25)[:, None]
        oracle = -(values[:, 0] - .7) ** 2
        best, seen = bayesian_search(values, oracle, budget=9)
        self.assertEqual(len(seen), 9)
        self.assertEqual(len(set(seen)), 9)
        self.assertIn(best, seen)
        self.assertEqual(oracle[best], max(oracle[seen]))


if __name__ == "__main__":
    unittest.main()
