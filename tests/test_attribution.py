from __future__ import annotations

import pandas as pd
import pytest

from quant.analysis import AttributionAnalyzer


def test_attribution_groups_contribution_and_factor_exposure() -> None:
    positions = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "industry": ["银行", "银行", "消费"],
            "weight": [0.2, 0.3, 0.5],
            "return": [0.10, -0.02, 0.04],
            "value_factor": [1.0, 2.0, -1.0],
        }
    )
    analyzer = AttributionAnalyzer(pd.Series(dtype=float), positions)

    by_symbol = analyzer.by_symbol()
    by_industry = analyzer.by_industry()
    exposure = analyzer.factor_exposure(["value_factor"])

    assert by_symbol.loc["A", "attribution"] == pytest.approx(0.02)
    assert by_industry.loc["银行", "attribution"] == pytest.approx(0.014)
    assert exposure["value_factor"] == pytest.approx(0.3)


def test_brinson_attribution_reconciles_active_return() -> None:
    portfolio = pd.DataFrame(
        {
            "group": ["X", "Y"],
            "weight": [0.75, 0.25],
            "return": [0.12, -0.02],
        }
    )
    benchmark = pd.DataFrame(
        {
            "group": ["X", "Y"],
            "weight": [0.50, 0.50],
            "return": [0.10, 0.00],
        }
    )

    result = AttributionAnalyzer.brinson_attribution(portfolio, benchmark)

    assert result["allocation"].sum() == pytest.approx(0.025)
    assert result["selection"].sum() == pytest.approx(0.0)
    assert result["interaction"].sum() == pytest.approx(0.01)
    assert result["total_effect"].sum() == pytest.approx(0.035)
