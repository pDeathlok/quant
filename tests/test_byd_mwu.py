"""Research contract tests runnable with unittest or the project pytest runner."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from quant.research.byd_mwu import (
    ExecutionConfig, MixerConfig, _RidgeExpert, build_experts, metrics, rolling_model_experts,
    simulate, update_weights,
)


def market_frame(dates: pd.DatetimeIndex, closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates, "open": closes, "close": closes,
        "pre_close": np.r_[closes[0], closes[:-1]],
        "split": 1, "cash_dividend": 0.0, "bonus_shares": 0.0,
        "volume_shares": 1_000_000.0, "up_limit": 1_000_000.0,
        "down_limit": 0.01, "is_suspended": False,
    })


ZERO_FEES = ExecutionConfig(
    initial_cash=100_000, commission=0, minimum_commission=0,
    transfer_fee=0, slippage=0, dividend_tax_reserve=0,
)


class MwuContractTests(unittest.TestCase):
    def test_ridge_matches_closed_form_with_standardized_feature(self) -> None:
        x = np.array([[-1.], [0.], [1.]])
        model = _RidgeExpert()
        model.fit(x, 2 * x[:, 0])
        predicted = model.predict(np.array([[1.]]))[0]
        self.assertAlmostEqual(predicted, 6 / 53, places=12)

    def test_update_rewards_relative_performance_and_retains_share_floor(self) -> None:
        config = MixerConfig(experts=("good", "bad"), eta=0.3, share=0.10)
        weights = np.array([0.5, 0.5])
        for _ in range(100):
            weights = update_weights(weights, np.array([0.02, -0.02]), config)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertGreaterEqual(weights[1], 0.05)

    def test_extreme_observations_remain_finite(self) -> None:
        config = MixerConfig(experts=("a", "b"), eta=100)
        weights = update_weights(np.array([1e-250, 1.0]), np.array([1e100, -1e100]), config)
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_open_entry_does_not_earn_prior_overnight_gap(self) -> None:
        market = market_frame(pd.date_range("2014-12-31", periods=3), np.array([100., 200., 200.]))
        targets = pd.DataFrame({"a": np.ones(3)})
        ledger = simulate(market, targets, ZERO_FEES, fixed_target=targets.a.to_numpy())
        self.assertAlmostEqual(ledger["return"].iloc[0], 0.0)
        self.assertEqual(ledger.shares.iloc[0], 500)
        self.assertEqual(ledger.signal_date.iloc[0], pd.Timestamp("2014-12-31"))

    def test_today_reward_cannot_change_today_order(self) -> None:
        market = market_frame(pd.date_range("2014-12-31", periods=4), np.full(4, 100.))
        targets = pd.DataFrame({"a": np.ones(4), "b": np.zeros(4)})
        rewards = pd.DataFrame({"a": [0., .02, .02, 0.], "b": [0., -.02, -.02, 0.]})
        config = MixerConfig(experts=("a", "b"), eta=1)
        ledger = simulate(market, targets, ZERO_FEES, config, rewards)
        self.assertAlmostEqual(ledger.requested_target.iloc[0], .5)
        self.assertGreater(ledger.requested_target.iloc[1], .5)
        self.assertAlmostEqual(ledger.weight_a.iloc[0], .5)

    def test_split_and_dividend_do_not_create_false_crash(self) -> None:
        market = market_frame(pd.date_range("2014-12-31", periods=3), np.array([100., 100., 30.]))
        market.loc[2, "split"] = 3
        market.loc[2, "cash_dividend"] = 10
        targets = pd.DataFrame({"a": [1., .9, .9]})
        ledger = simulate(market, targets, ZERO_FEES, fixed_target=targets.a.to_numpy())
        self.assertEqual(ledger.shares.iloc[-1], 3000)
        self.assertAlmostEqual(ledger.equity.iloc[-1], 100_000)
        self.assertAlmostEqual(ledger["return"].iloc[-1], 0)

    def test_cash_lot_cost_and_limit_invariants(self) -> None:
        market = market_frame(pd.date_range("2014-12-31", periods=4), np.full(4, 100.))
        market.loc[1, "up_limit"] = 100
        target = np.ones(4)
        ledger = simulate(market, pd.DataFrame({"a": target}), replace(ZERO_FEES, commission=.0003, minimum_commission=5), fixed_target=target)
        self.assertEqual(ledger.trade_shares.iloc[0], 0)
        self.assertEqual(ledger.blocked.iloc[0], "buy_at_upper_limit")
        self.assertTrue((ledger.cash >= 0).all())
        self.assertTrue((ledger.shares % 100 == 0).all())
        self.assertGreater(ledger.cost.sum(), 0)

    def test_capacity_uses_previous_volume(self) -> None:
        market = market_frame(pd.date_range("2014-12-31", periods=3), np.full(3, 100.))
        market.loc[0, "volume_shares"] = 10_000
        market.loc[1, "volume_shares"] = 100_000_000
        targets = pd.DataFrame({"a": np.ones(3)})
        ledger = simulate(market, targets, ZERO_FEES, fixed_target=targets.a.to_numpy())
        self.assertEqual(ledger.trade_shares.iloc[0], 100)

    def test_max_drawdown_includes_initial_capital(self) -> None:
        ledger = pd.DataFrame({"return": [-.1, .01], "position": [.5, .5], "turnover": [1., 0.], "trade_shares": [100, 0], "cost": [0., 0.]})
        self.assertAlmostEqual(metrics(ledger)["max_drawdown"], -.1)

    def test_future_data_cannot_change_past_mixer_or_trades(self) -> None:
        dates = pd.bdate_range("2014-12-01", periods=90)
        closes = 100 + np.sin(np.arange(90) / 8) * 5
        market = market_frame(dates, closes)
        targets = pd.DataFrame({"a": .8, "b": .2}, index=market.index)
        rewards = pd.DataFrame({"a": np.sin(np.arange(90)) * .01, "b": .0})
        config = MixerConfig(experts=("a", "b"), eta=.3, share=.01)
        full = simulate(market, targets, ZERO_FEES, config, rewards)
        prefix = simulate(market.iloc[:65], targets.iloc[:65], ZERO_FEES, config, rewards.iloc[:65])
        pd.testing.assert_frame_equal(full.iloc[:len(prefix)].reset_index(drop=True), prefix)

    def test_rule_features_are_prefix_invariant_and_fix_direction(self) -> None:
        dates = pd.bdate_range("2011-01-03", periods=580)
        close = 100 + np.arange(580) * .02 + np.sin(np.arange(580) / 20) * 25
        signal = pd.DataFrame({"date": dates, "symbol": "002594.SZ", "open": close * .999, "high": close * 1.01, "low": close * .99, "close": close, "volume": 1e6 * (1 + .2 * np.sin(np.arange(580) / 11)), "pct_change": pd.Series(close).pct_change().fillna(0) * 100})
        full, _ = build_experts(signal, include_models=False)
        prefix, _ = build_experts(signal.iloc[:500], include_models=False)
        pd.testing.assert_frame_equal(full.iloc[:500], prefix)
        self.assertTrue(full.original_score.ne(full.corrected_score).any())

    def test_model_retraining_uses_only_mature_labels_and_is_prefix_invariant(self) -> None:
        dates = pd.bdate_range("2011-01-03", periods=570)
        close = 100 + np.arange(570) * .05 + np.sin(np.arange(570) / 7) * 2
        signal = pd.DataFrame({"date": dates, "open": close, "high": close * 1.01, "low": close * .99, "close": close, "volume": 1e6})
        full, audit = rolling_model_experts(signal)
        prefix, _ = rolling_model_experts(signal.iloc[:550])
        pd.testing.assert_frame_equal(full.iloc[:550], prefix)
        self.assertTrue((pd.to_datetime(audit.last_label_end) <= pd.to_datetime(audit.fit_date)).all())
        self.assertTrue((pd.to_datetime(audit.last_feature_date) < pd.to_datetime(audit.fit_date)).all())


if __name__ == "__main__":
    unittest.main()
