from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.portfolio import (
    PortfolioConstraints,
    PortfolioConstructor,
    target_weights_to_orders,
)


def test_equal_weight_respects_position_and_industry_caps() -> None:
    constructor = PortfolioConstructor(
        PortfolioConstraints(
            max_weight=0.40,
            max_industry_weight=0.50,
            cash_buffer=0.0,
        )
    )
    candidates = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"],
            "industry": ["银行", "银行", "消费"],
        }
    )

    result = constructor.equal_weight(candidates)

    assert result.target_weights.sum() == pytest.approx(5 / 6)
    assert result.target_weights.loc[["000001.SZ", "000002.SZ"]].sum() == pytest.approx(0.5)
    assert result.target_weights.max() <= 0.40
    assert result.cash_weight == pytest.approx(1 / 6)


def test_score_weight_ignores_ineligible_and_non_positive_scores() -> None:
    constructor = PortfolioConstructor(
        PortfolioConstraints(max_weight=0.60, max_industry_weight=1.0, cash_buffer=0.0)
    )
    candidates = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "score": [3.0, 1.0, -2.0, 100.0],
            "eligible": [True, True, True, False],
            "industry": ["X", "Y", "Z", "X"],
        }
    )

    result = constructor.score_weight(candidates)

    assert result.target_weights.to_dict() == pytest.approx({"A": 0.6, "B": 0.25})
    assert result.cash_weight == pytest.approx(0.15)


def test_inverse_volatility_allocates_more_to_lower_volatility() -> None:
    returns = pd.DataFrame(
        {
            "LOW": [0.001, -0.001, 0.001, -0.001, 0.001],
            "HIGH": [0.04, -0.03, 0.05, -0.04, 0.02],
        }
    )
    constructor = PortfolioConstructor(
        PortfolioConstraints(max_weight=0.80, max_industry_weight=1.0, cash_buffer=0.0)
    )

    result = constructor.inverse_volatility(returns)

    assert result.target_weights["LOW"] > result.target_weights["HIGH"]
    assert result.target_weights.sum() <= 1.0 + 1e-12
    assert result.target_weights.max() <= 0.80 + 1e-12


def test_minimum_variance_returns_feasible_weights() -> None:
    rng = np.random.default_rng(20260801)
    returns = pd.DataFrame(
        rng.normal(0, [0.01, 0.02, 0.03], size=(200, 3)),
        columns=["A", "B", "C"],
    )
    constructor = PortfolioConstructor(
        PortfolioConstraints(max_weight=0.60, max_industry_weight=1.0, cash_buffer=0.05)
    )

    result = constructor.minimum_variance(returns)

    assert result.target_weights.sum() == pytest.approx(0.95, abs=1e-6)
    assert result.target_weights.min() >= 0
    assert result.target_weights.max() <= 0.60 + 1e-8
    assert result.diagnostics["method"] == "minimum_variance"


def test_turnover_limit_interpolates_from_current_weights() -> None:
    constructor = PortfolioConstructor(
        PortfolioConstraints(
            max_weight=1.0,
            max_industry_weight=1.0,
            max_turnover=0.20,
            cash_buffer=0.0,
        )
    )
    candidates = pd.DataFrame(
        {"symbol": ["A", "B"], "score": [0.0, 1.0], "industry": ["X", "Y"]}
    )

    result = constructor.score_weight(
        candidates,
        current_weights=pd.Series({"A": 1.0, "B": 0.0}),
    )

    assert result.turnover == pytest.approx(0.20)
    assert result.target_weights.to_dict() == pytest.approx({"A": 0.9, "B": 0.1})


def test_target_weights_to_orders_uses_round_lots_and_sells_first() -> None:
    orders = target_weights_to_orders(
        target_weights=pd.Series({"A": 0.25, "B": 0.50}),
        prices=pd.Series({"A": 10.0, "B": 20.0}),
        current_positions={"A": 5000, "B": 0},
        total_equity=100_000.0,
        available_cash=50_000.0,
        lot_size=100,
    )

    assert orders[["symbol", "side", "quantity"]].to_dict("records") == [
        {"symbol": "A", "side": "sell", "quantity": 2500},
        {"symbol": "B", "side": "buy", "quantity": 2500},
    ]
    assert (orders["quantity"] % 100 == 0).all()
