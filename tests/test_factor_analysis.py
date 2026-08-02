from __future__ import annotations

import pandas as pd
import pytest

from quant.analysis import FactorAnalyzer


def test_add_forward_returns_uses_future_prices_within_symbol() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03"] * 2
            ),
            "symbol": ["A"] * 3 + ["B"] * 3,
            "close": [10.0, 11.0, 12.0, 20.0, 18.0, 21.0],
        }
    )

    enriched = FactorAnalyzer.add_forward_returns(prices, periods=(1, 2))

    a_rows = enriched.loc[enriched["symbol"].eq("A")].reset_index(drop=True)
    assert a_rows.loc[0, "forward_return_1"] == pytest.approx(0.10)
    assert a_rows.loc[1, "forward_return_1"] == pytest.approx(12 / 11 - 1)
    assert a_rows.loc[0, "forward_return_2"] == pytest.approx(0.20)
    assert pd.isna(a_rows.loc[2, "forward_return_1"])


def test_information_coefficient_and_quantile_returns() -> None:
    rows = []
    for date in pd.to_datetime(["2026-01-02", "2026-01-05"]):
        for index in range(1, 7):
            rows.append(
                {
                    "date": date,
                    "symbol": f"S{index}",
                    "factor": float(index),
                    "forward_return": float(index) / 100,
                }
            )
    data = pd.DataFrame(rows)
    analyzer = FactorAnalyzer(data, minimum_cross_section=5)

    pearson = analyzer.information_coefficient("factor", "forward_return")
    rank = analyzer.information_coefficient(
        "factor", "forward_return", method="spearman"
    )
    quantiles = analyzer.quantile_returns(
        "factor", "forward_return", quantiles=3
    )

    assert pearson["ic"].tolist() == pytest.approx([1.0, 1.0])
    assert rank["ic"].tolist() == pytest.approx([1.0, 1.0])
    assert quantiles.loc[quantiles["quantile"].eq(3), "mean_return"].tolist() == pytest.approx(
        [0.055, 0.055]
    )
    assert quantiles.loc[quantiles["quantile"].eq("long_short"), "mean_return"].tolist() == pytest.approx(
        [0.04, 0.04]
    )


def test_factor_turnover_tracks_quantile_membership_changes() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"] * 4 + ["2026-01-05"] * 4),
            "symbol": ["A", "B", "C", "D"] * 2,
            "factor": [1, 2, 3, 4, 4, 2, 3, 1],
            "forward_return": [0.0] * 8,
        }
    )
    analyzer = FactorAnalyzer(data, minimum_cross_section=4)

    turnover = analyzer.quantile_turnover("factor", quantiles=2)

    second_date = turnover.loc[turnover["date"].eq(pd.Timestamp("2026-01-05"))]
    assert second_date["turnover"].tolist() == pytest.approx([0.5, 0.5])
