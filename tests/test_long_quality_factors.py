from __future__ import annotations

import numpy as np
import pandas as pd

from quant.features.long_quality_factors import (
    add_enhanced_long_scores,
    build_annual_quality_events,
    merge_annual_quality_asof,
)


def _annual_rows(values: list[tuple[str, str, float]], column: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "end_date": end_date,
                "ann_date": ann_date,
                "report_type": "1",
                column: value,
            }
            for end_date, ann_date, value in values
        ]
    )


def test_annual_quality_uses_first_published_values_and_asof_dates() -> None:
    periods = [
        ("20181231", "20190401", 10.0),
        ("20191231", "20200401", 12.0),
        ("20201231", "20210401", 15.0),
    ]
    fina = _annual_rows(periods, "roe").drop(columns="report_type")
    fina["netprofit_margin"] = 10.0
    fina["debt_to_assets"] = 30.0
    fina["ar_turn"] = 5.0
    fina["inv_turn"] = 4.0
    income = _annual_rows(
        [(end, ann, value * 10.0) for end, ann, value in periods], "n_income_attr_p"
    )
    income["revenue"] = [500.0, 600.0, 750.0]
    cashflow = _annual_rows(
        [
            ("20181231", "20190401", 120.0),
            ("20191231", "20200401", 130.0),
            # Cash flow is published after the other annual statements.  The
            # combined feature event must wait for the last required source.
            ("20201231", "20210410", 200.0),
            # A correction disclosed later must not rewrite the original row.
            ("20201231", "20210601", 9999.0),
        ],
        "n_cashflow_act",
    )
    cashflow["c_pay_acq_const_fiolta"] = 20.0
    balance = _annual_rows(periods, "total_assets")
    balance["total_assets"] = [1000.0, 1100.0, 1300.0]
    balance["goodwill"] = 10.0
    balance["inventories"] = 100.0

    events = build_annual_quality_events(fina, income, cashflow, balance)
    latest = events.loc[events["period_end"].eq(pd.Timestamp("2020-12-31"))].iloc[0]
    expected = (120.0 + 130.0 + 200.0) / (100.0 + 120.0 + 150.0)
    assert latest["cashflow_quality_3y"] == expected
    assert latest["annual_quality_available_at"] == pd.Timestamp("2021-04-10")

    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-04-09", "2021-04-10"]),
            "ts_code": ["000001.SZ", "000001.SZ"],
        }
    )
    merged = merge_annual_quality_asof(signals, events)
    assert merged.loc[0, "period_end"] == pd.Timestamp("2019-12-31")
    assert merged.loc[1, "period_end"] == pd.Timestamp("2020-12-31")


def test_enhanced_scores_reward_cash_conversion_and_build_explicit_gates() -> None:
    rows = []
    for index in range(6):
        strong = index >= 3
        rows.append(
            {
                "date": pd.Timestamp("2025-06-30"),
                "ts_code": f"00000{index}.SZ",
                "industry": "制造",
                "profitability_score": 60.0 + index,
                "fundamental_growth_score": 55.0 + index,
                "balance_sheet_score": 60.0,
                "business_stability_score": 65.0,
                "cashflow_quality_3y": 1.2 if strong else 0.2,
                "free_cashflow_margin_3y": 0.15 if strong else -0.10,
                "cfo_positive_share_5y": 1.0 if strong else 0.4,
                "accruals_to_assets_3y": -0.05 if strong else 0.20,
                "profit_positive_share_5y": 1.0 if strong else 0.6,
                "roe_mean_5y": 18.0 if strong else 8.0,
                "roe_std_5y": 2.0 if strong else 8.0,
                "revenue_growth_positive_share_5y": 0.8 if strong else 0.4,
                "annual_goodwill_to_assets": 0.05 if strong else 0.35,
                "fina_ar_turn": 8.0 if strong else 2.0,
                "fina_inv_turn": 6.0 if strong else 1.5,
                "fina_debt_to_assets": 35.0 if strong else 75.0,
                "annual_history_years": 5.0,
                "pe_ttm_industry_pct": (index + 1) / 6,
                "pb_industry_pct": (index + 1) / 6,
                "pr_industry_pct": (index + 1) / 6,
                "historical_value_score_5y": 60.0,
                "close_to_ma120": 0.02,
                "return_120d_cross_section_pct": 0.5,
                "downside_volatility_60d": 0.15 + index * 0.01,
            }
        )
    scored = add_enhanced_long_scores(pd.DataFrame(rows))
    assert scored["enhanced_good_stock_score"].between(0, 100).all()
    assert scored.loc[5, "cashflow_quality_score"] > scored.loc[0, "cashflow_quality_score"]
    assert bool(scored.loc[5, "cashflow_gate_08"])
    assert not bool(scored.loc[0, "cashflow_gate_08"])
    assert bool(scored.loc[5, "durability_gate"])
    assert not bool(scored.loc[0, "durability_gate"])
    assert np.isfinite(scored["rule_long_model_score"]).all()
    assert scored["rule_long_model_score"].between(0, 100).all()

    without_industry_value = pd.DataFrame(rows).drop(
        columns=["pe_ttm_industry_pct", "pb_industry_pct", "pr_industry_pct"]
    )
    fallback = add_enhanced_long_scores(without_industry_value)
    assert fallback["blended_value_score"].eq(60.0).all()
