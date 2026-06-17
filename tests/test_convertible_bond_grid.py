import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest


def test_conservative_grid_only_builds_when_low_position_confirmed():
    from quant.strategies.convertible_bond.grid import ConservativeGridConfig, ConservativeGridStrategy

    daily = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "trade_date": "20260616",
                "close": 108.0,
                "bond_over_rate": 8.0,
                "amount": 8000.0,
                "pct_chg": 0.0,
                "price_position_252": 0.10,
                "drawdown_from_252_high": -0.22,
            },
            {
                "ts_code": "123001.SZ",
                "trade_date": "20260616",
                "close": 108.0,
                "bond_over_rate": 8.0,
                "amount": 8000.0,
                "pct_chg": 0.0,
                "price_position_252": 0.80,
                "drawdown_from_252_high": -0.02,
            },
        ]
    )
    basic = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "conv_start_date": "20200101",
            },
            {
                "ts_code": "123001.SZ",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "conv_start_date": "20200101",
            },
        ]
    )

    target = ConservativeGridStrategy(ConservativeGridConfig()).target_portfolio(
        daily=daily,
        basic=basic,
        call=pd.DataFrame(),
    )

    assert list(target["ts_code"]) == ["110001.SH"]
    assert target["target_weight"].iloc[0] > 0.10


def test_add_low_position_features_creates_price_position():
    from quant.strategies.convertible_bond.grid import add_low_position_features

    daily = pd.DataFrame(
        {
            "ts_code": ["110001.SH"] * 30,
            "trade_date": pd.date_range("2026-01-01", periods=30, freq="D").strftime("%Y%m%d"),
            "close": list(range(100, 130)),
            "bond_over_rate": [10.0] * 30,
            "amount": [1000.0] * 30,
        }
    )

    featured = add_low_position_features(daily, window=20)

    assert "price_position_252" in featured.columns
    assert featured["price_position_252"].dropna().iloc[-1] == 1.0


def test_holding_grid_keeps_existing_until_exit():
    from quant.strategies.convertible_bond.grid import HoldingGridConfig, HoldingGridStrategy

    strategy = HoldingGridStrategy(
        HoldingGridConfig(exit_price=130.0, exit_premium_rate=50.0, exit_double_low=180.0)
    )
    daily = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "trade_date": "20260616",
                "close": 120.0,
                "bond_over_rate": 12.0,
                "amount": 8000.0,
                "pct_chg": 0.0,
                "price_position_252": 0.50,
                "drawdown_from_252_high": -0.05,
            }
        ]
    )
    basic = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "conv_start_date": "20200101",
            }
        ]
    )

    weights, entries, target = strategy.target_with_existing(
        daily=daily,
        basic=basic,
        current_weights={"110001.SH": 0.12},
        entry_prices={"110001.SH": 108.0},
        call=pd.DataFrame(),
    )

    assert weights == {"110001.SH": 0.12}
    assert entries == {"110001.SH": 108.0}
    assert bool(target["is_existing"].iloc[0]) is True


def test_holding_grid_can_filter_new_entries_by_trend_signal():
    from quant.strategies.convertible_bond.grid import HoldingGridConfig, HoldingGridStrategy

    strategy = HoldingGridStrategy(
        HoldingGridConfig(
            top_n=None,
            max_entry_price=112.0,
            max_double_low=130.0,
            max_price_position_252=0.35,
            min_drawdown_from_252_high=0.05,
            min_entry_trend_strength=75.0,
            min_entry_six_sword=4,
            min_entry_return_5d=0.0,
        )
    )
    daily = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "trade_date": "20260616",
                "close": 108.0,
                "bond_over_rate": 8.0,
                "amount": 8000.0,
                "pct_chg": 0.0,
                "price_position_252": 0.10,
                "drawdown_from_252_high": -0.20,
                "trend_strength": 100.0,
                "six_sword_daily": 5,
                "return_5d": 2.0,
            },
            {
                "ts_code": "110002.SH",
                "trade_date": "20260616",
                "close": 108.0,
                "bond_over_rate": 8.0,
                "amount": 8000.0,
                "pct_chg": 0.0,
                "price_position_252": 0.10,
                "drawdown_from_252_high": -0.20,
                "trend_strength": 25.0,
                "six_sword_daily": 2,
                "return_5d": -1.0,
            },
        ]
    )
    basic = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "conv_start_date": "20200101",
            },
            {
                "ts_code": "110002.SH",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "conv_start_date": "20200101",
            },
        ]
    )

    target = strategy.target_portfolio(daily=daily, basic=basic, call=pd.DataFrame())

    assert list(target["ts_code"]) == ["110001.SH"]


def test_holding_grid_adds_on_lower_grid_and_updates_entry_price():
    from quant.strategies.convertible_bond.grid import HoldingGridConfig, HoldingGridStrategy

    strategy = HoldingGridStrategy(
        HoldingGridConfig(
            max_position_weight=0.20,
            initial_entry_fraction=0.50,
            add_on_drawdown_step=0.04,
            add_position_fraction=0.25,
            max_grid_position_fraction=1.00,
            max_entry_price=116.0,
            max_premium_rate=24.0,
            max_double_low=138.0,
            max_price_position_252=0.35,
            min_drawdown_from_252_high=0.07,
        )
    )
    daily = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "trade_date": "20260616",
                "close": 104.0,
                "bond_over_rate": 8.0,
                "amount": 8000.0,
                "pct_chg": 0.0,
                "price_position_252": 0.10,
                "drawdown_from_252_high": -0.18,
            }
        ]
    )
    basic = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "conv_start_date": "20200101",
            }
        ]
    )

    weights, entries, _ = strategy.target_with_existing(
        daily=daily,
        basic=basic,
        current_weights={"110001.SH": 0.10},
        entry_prices={"110001.SH": 110.0},
        call=pd.DataFrame(),
    )

    assert weights["110001.SH"] == pytest.approx(0.15)
    assert entries["110001.SH"] == pytest.approx(108.0)


def test_holding_grid_takes_partial_profit_before_full_exit():
    from quant.strategies.convertible_bond.grid import HoldingGridConfig, HoldingGridStrategy

    strategy = HoldingGridStrategy(
        HoldingGridConfig(
            max_position_weight=0.20,
            take_profit_1=0.08,
            take_profit_1_keep_fraction=0.40,
            take_profit_2=0.20,
            exit_price=150.0,
            exit_premium_rate=80.0,
            exit_double_low=220.0,
            exit_price_position_252=0.95,
        )
    )
    daily = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "trade_date": "20260616",
                "close": 120.0,
                "bond_over_rate": 12.0,
                "amount": 8000.0,
                "pct_chg": 0.0,
                "price_position_252": 0.50,
                "drawdown_from_252_high": -0.05,
            }
        ]
    )
    basic = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "conv_start_date": "20200101",
            }
        ]
    )

    weights, entries, _ = strategy.target_with_existing(
        daily=daily,
        basic=basic,
        current_weights={"110001.SH": 0.16},
        entry_prices={"110001.SH": 110.0},
        call=pd.DataFrame(),
    )

    assert weights["110001.SH"] == pytest.approx(0.08)
    assert entries["110001.SH"] == 110.0


def test_holding_grid_exits_on_floor_price():
    from quant.strategies.convertible_bond.grid import HoldingGridConfig, HoldingGridStrategy

    strategy = HoldingGridStrategy(
        HoldingGridConfig(
            exit_floor_price=98.0,
            exit_price=150.0,
            exit_premium_rate=80.0,
            exit_double_low=220.0,
            exit_price_position_252=0.95,
        )
    )
    row = pd.Series(
        {
            "close": 96.5,
            "premium_rate": 5.0,
            "double_low": 101.5,
            "price_position_252": 0.10,
            "call_risk": False,
        }
    )

    assert strategy.should_exit(row, entry_price=110.0) is True


def test_conservative_grid_top_n_none_keeps_all_qualified_candidates():
    from quant.strategies.convertible_bond.grid import ConservativeGridConfig, ConservativeGridStrategy

    daily = pd.DataFrame(
        [
            {
                "ts_code": f"11000{idx}.SH",
                "trade_date": "20260616",
                "close": 108.0 + idx * 0.1,
                "bond_over_rate": 6.0,
                "amount": 8000.0,
                "pct_chg": 0.0,
                "price_position_252": 0.10,
                "drawdown_from_252_high": -0.20,
            }
            for idx in range(3)
        ]
    )
    basic = pd.DataFrame(
        [
            {
                "ts_code": f"11000{idx}.SH",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "conv_start_date": "20200101",
            }
            for idx in range(3)
        ]
    )

    target = ConservativeGridStrategy(
        ConservativeGridConfig(top_n=None, max_entry_price=112.0, max_double_low=125.0)
    ).target_portfolio(daily=daily, basic=basic, call=pd.DataFrame())

    assert len(target) == 3


def test_dynamic_grid_assigns_wider_grid_and_smaller_weight_to_high_risk():
    from quant.strategies.convertible_bond.grid import HoldingGridConfig, HoldingGridStrategy

    strategy = HoldingGridStrategy(
        HoldingGridConfig(
            top_n=None,
            dynamic_grid=True,
            max_position_weight=0.10,
            max_entry_price=118.0,
            max_double_low=144.0,
            max_price_position_252=0.40,
            min_drawdown_from_252_high=0.06,
            min_credit_rating="A+",
        )
    )
    daily = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "trade_date": "20260616",
                "close": 108.0,
                "bond_over_rate": 6.0,
                "amount": 9000.0,
                "pct_chg": 0.0,
                "price_position_252": 0.08,
                "drawdown_from_252_high": -0.20,
            },
            {
                "ts_code": "110002.SH",
                "trade_date": "20260616",
                "close": 116.0,
                "bond_over_rate": 22.0,
                "amount": 2000.0,
                "pct_chg": 0.0,
                "price_position_252": 0.32,
                "drawdown_from_252_high": -0.08,
            },
        ]
    )
    basic = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "conv_start_date": "20200101",
            },
            {
                "ts_code": "110002.SH",
                "remain_size": 5.0,
                "newest_rating": "A+",
                "conv_start_date": "20200101",
            },
        ]
    )

    target = strategy.target_portfolio(daily=daily, basic=basic, call=pd.DataFrame())
    by_code = target.set_index("ts_code")

    assert by_code.loc["110001.SH", "risk_level"] == "low"
    assert by_code.loc["110002.SH", "risk_level"] == "high"
    assert by_code.loc["110001.SH", "grid_step_pct"] == pytest.approx(0.03)
    assert by_code.loc["110002.SH", "grid_step_pct"] == pytest.approx(0.055)
    assert by_code.loc["110002.SH", "target_weight"] < by_code.loc["110001.SH", "target_weight"]
