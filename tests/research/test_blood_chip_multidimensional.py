from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.blood_chip_multidimensional import (
    INDUSTRY_DIAGNOSTIC_POLICIES,
    SELECTION_ELIGIBLE_POLICIES,
    MultidimensionalGateConfig,
    add_current_industry_repair_features,
    add_market_repair_features,
    apply_multidimensional_gates,
    merge_capital_pressure_asof,
    merge_daily_basic_on_signal_date,
    merge_financial_survival_asof,
    signals_for_policy,
)


def _signals(dates: list[str], codes: list[str] | None = None) -> pd.DataFrame:
    symbols = codes or ["000001.SZ"] * len(dates)
    return pd.DataFrame(
        {
            "ts_code": symbols,
            "signal_date": pd.to_datetime(dates),
            "signal_score": np.linspace(0.4, 0.6, len(dates)),
        }
    )


def test_financial_events_are_invisible_before_announcement() -> None:
    signals = _signals(["2021-04-09", "2021-04-10"])
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "period_end": pd.to_datetime(["2019-12-31", "2020-12-31"]),
            "annual_quality_available_at": pd.to_datetime(["2020-04-01", "2021-04-10"]),
            "annual_history_years": [3.0, 4.0],
            "income_n_income_attr_p": [10.0, 20.0],
        }
    )

    merged = merge_financial_survival_asof(signals, events)

    assert merged.loc[0, "period_end"] == pd.Timestamp("2019-12-31")
    assert merged.loc[1, "period_end"] == pd.Timestamp("2020-12-31")
    assert merged.loc[1, "financial_age_days"] == 100


def test_daily_basic_requires_exact_signal_close() -> None:
    signals = _signals(["2022-01-04", "2022-01-05"])
    basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20220105"],
            "pb": [1.2],
            "total_mv": [200_000.0],
        }
    )

    merged = merge_daily_basic_on_signal_date(signals, basic)

    assert not bool(merged.loc[0, "daily_basic_coverage"])
    assert bool(merged.loc[1, "daily_basic_coverage"])
    assert merged.loc[1, "basic_pb"] == 1.2


def test_pledge_same_day_is_not_visible_and_future_holder_trade_is_excluded() -> None:
    signals = _signals(["2022-06-01", "2022-06-02"])
    pledge = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20220531", "20220601"],
            "pledge_ratio": [12.0, 55.0],
        }
    )
    holder = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "ann_date": ["20211101", "20220531", "20220603"],
            "in_de": ["DE", "IN", "DE"],
            "change_ratio": [9.0, 1.5, 7.0],
        }
    )

    merged = merge_capital_pressure_asof(signals, pledge, holder)

    assert merged.loc[0, "pledge_ratio"] == 12.0
    assert merged.loc[1, "pledge_ratio"] == 55.0
    assert merged.loc[0, "holder_net_change_ratio_180d"] == 1.5
    assert merged.loc[1, "holder_net_change_ratio_180d"] == 1.5
    assert merged["holder_event_count_180d"].eq(1).all()


def test_market_features_do_not_change_when_future_rows_are_appended() -> None:
    dates = pd.bdate_range("2020-01-01", periods=400)
    benchmark = pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y%m%d"),
            "close": np.linspace(3000.0, 4200.0, len(dates)),
        }
    )
    signal_date = dates[300]
    signals = _signals([signal_date.strftime("%Y-%m-%d")])
    original = add_market_repair_features(signals, benchmark.iloc[:350])
    future = benchmark.iloc[350:].copy()
    future["close"] = future["close"] * 10.0
    appended = add_market_repair_features(signals, pd.concat([benchmark.iloc[:350], future]))

    columns = ["market_return_20d", "market_return_120d", "market_close_to_ma250"]
    pd.testing.assert_series_equal(original.loc[0, columns], appended.loc[0, columns])


def test_current_industry_repair_is_marked_as_biased_diagnostic() -> None:
    signals = _signals(["2022-01-05"], ["000001.SZ"])
    features = pd.DataFrame(
        {
            "ts_code": [f"00000{i}.SZ" for i in range(1, 7)],
            "date": pd.Timestamp("2022-01-05"),
            "return_20d": [-0.02, 0.01, 0.03, -0.01, 0.02, 0.04],
        }
    )
    stock_basic = pd.DataFrame(
        {
            "ts_code": [f"00000{i}.SZ" for i in range(1, 7)],
            "industry": ["制造"] * 6,
        }
    )

    enriched = add_current_industry_repair_features(signals, features, stock_basic)

    assert enriched.loc[0, "current_industry_constituents"] == 6
    assert bool(enriched.loc[0, "current_industry_mapping_bias"])
    assert "current_industry_diagnostic" in INDUSTRY_DIAGNOSTIC_POLICIES
    assert "current_industry_diagnostic" not in SELECTION_ELIGIBLE_POLICIES


def _fully_enriched_signal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "signal_date": pd.to_datetime(["2022-06-30"]),
            "signal_score": [0.55],
            "annual_quality_available_at": pd.to_datetime(["2022-04-01"]),
            "financial_age_days": [181.0],
            "annual_history_years": [5.0],
            "profit_positive_share_5y": [0.8],
            "cfo_positive_share_5y": [0.8],
            "income_n_income_attr_p": [100.0],
            "cashflow_n_cashflow_act": [120.0],
            "daily_basic_coverage": [True],
            "basic_total_mv": [300_000.0],
            "basic_pb": [1.5],
            "basic_ps_ttm": [2.0],
            "basic_turnover_rate_f": [2.0],
            "pledge_ratio": [10.0],
            "holder_net_change_ratio_180d": [0.2],
            "holder_event_count_180d": [1],
            "market_coverage": [True],
            "market_return_20d": [0.01],
            "market_return_120d": [-0.02],
            "market_close_to_ma250": [1.01],
            "current_industry_return_20d": [0.01],
            "current_industry_positive_share_20d": [0.55],
            "current_industry_constituents": [20],
        }
    )


def test_missing_financials_cannot_pass_survival_gate() -> None:
    config = MultidimensionalGateConfig()
    complete = apply_multidimensional_gates(_fully_enriched_signal(), config)
    missing = _fully_enriched_signal().drop(
        columns=[
            "annual_quality_available_at",
            "financial_age_days",
            "annual_history_years",
            "profit_positive_share_5y",
            "cfo_positive_share_5y",
            "income_n_income_attr_p",
            "cashflow_n_cashflow_act",
        ]
    )
    missing = apply_multidimensional_gates(missing, config)

    assert bool(complete.loc[0, "survival_gate"])
    assert not bool(missing.loc[0, "survival_gate"])
    assert complete.loc[0, "survival_score"] > missing.loc[0, "survival_score"]


def test_policy_filter_uses_only_registered_gates_and_labels_diagnostics() -> None:
    config = MultidimensionalGateConfig()
    enriched = apply_multidimensional_gates(_fully_enriched_signal(), config)

    combined = signals_for_policy(enriched, "auditable_combined", config)
    diagnostic = signals_for_policy(enriched, "current_industry_diagnostic", config)

    assert len(combined) == 1
    assert bool(combined.loc[0, "selection_eligible_policy"])
    assert not bool(combined.loc[0, "current_industry_mapping_bias"])
    assert len(diagnostic) == 1
    assert not bool(diagnostic.loc[0, "selection_eligible_policy"])
    assert bool(diagnostic.loc[0, "current_industry_mapping_bias"])
    assert combined.loc[0, "signal_score"] != combined.loc[0, "price_signal_score"]
