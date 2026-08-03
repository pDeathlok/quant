from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


RESEARCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "research"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from backtest_tea_master_long import add_historical_valuation_features  # noqa: E402
import backtest_long_dividend_quality as dividend_module  # noqa: E402
from backtest_long_good_stock_price import (  # noqa: E402
    price_score_band_payload,
    summarize_price_score_bands,
)
from backtest_long_good_price_weekly_absolute import (  # noqa: E402
    absolute_rule_masks,
    historical_rule_masks,
)


def test_dual_pr_formulas_and_type_weights_are_preserved() -> None:
    dates = pd.date_range("2023-01-31", periods=24, freq="ME")
    frame = pd.DataFrame(
        {
            "ts_code": ["LIGHT"] * 24 + ["HEAVY"] * 24,
            "date": list(dates) * 2,
            "industry": ["软件服务"] * 24 + ["钢铁"] * 24,
            "roe": [10.0] * 48,
            "pe_ttm": [20.0] * 48,
            "pb": [2.0] * 48,
        }
    )

    result = add_historical_valuation_features(frame).groupby("ts_code").tail(1).set_index("ts_code")

    assert result.loc["LIGHT", "pr_pe"] == pytest.approx(2.0)
    assert result.loc["LIGHT", "pr_pb"] == pytest.approx(2.0)
    assert result.loc["LIGHT", "pr_pe_weight"] == pytest.approx(0.70)
    assert result.loc["LIGHT", "pr_pb_weight"] == pytest.approx(0.30)
    assert result.loc["HEAVY", "pr_pe_weight"] == pytest.approx(0.30)
    assert result.loc["HEAVY", "pr_pb_weight"] == pytest.approx(0.70)
    assert result.loc["LIGHT", "valuation_history_points"] == 24


def test_history_percentile_uses_only_selected_recent_months() -> None:
    dates = pd.date_range("2019-01-31", periods=72, freq="ME")
    frame = pd.DataFrame(
        {
            "ts_code": ["A"] * 72,
            "date": dates,
            "industry": ["通用设备"] * 72,
            "roe": list(range(72, 0, -1)),
            "pe_ttm": list(range(72, 0, -1)),
            "pb": [value / 10 for value in range(72, 0, -1)],
        }
    )

    result = add_historical_valuation_features(
        frame,
        window_months=60,
        minimum_months=24,
    ).iloc[-1]

    assert result["valuation_history_points"] == 60
    assert result["pe_hist_percentile"] == pytest.approx(100 / 60)
    assert result["pb_hist_percentile"] == pytest.approx(100 / 60)
    assert result["roe_hist_percentile"] == pytest.approx(100 / 60)
    assert result["roe_history_points"] == 60


def test_history_percentiles_stay_aligned_for_interleaved_symbols() -> None:
    dates = pd.date_range("2024-01-31", periods=24, freq="ME")
    rows = []
    for index, date in enumerate(dates):
        rows.extend(
            [
                {
                    "ts_code": "DOWN",
                    "date": date,
                    "industry": "通用设备",
                    "roe": 10.0,
                    "pe_ttm": 24 - index,
                    "pb": (24 - index) / 10,
                },
                {
                    "ts_code": "UP",
                    "date": date,
                    "industry": "通用设备",
                    "roe": 10.0,
                    "pe_ttm": index + 1,
                    "pb": (index + 1) / 10,
                },
            ]
        )

    latest = (
        add_historical_valuation_features(pd.DataFrame(rows))
        .groupby("ts_code")
        .tail(1)
        .set_index("ts_code")
    )

    assert latest.loc["DOWN", "pe_hist_percentile"] == pytest.approx(100 / 24)
    assert latest.loc["UP", "pe_hist_percentile"] == pytest.approx(100.0)


def test_price_score_is_independent_of_same_day_cross_section() -> None:
    dates = pd.date_range("2024-01-31", periods=30, freq="ME")
    primary = pd.DataFrame(
        {
            "ts_code": "PRIMARY",
            "date": dates,
            "industry": "通用设备",
            "roe": 12.0,
            "pe_ttm": list(range(30, 0, -1)),
            "pb": [value / 10 for value in range(30, 0, -1)],
        }
    )
    distraction = pd.DataFrame(
        {
            "ts_code": "DISTRACTION",
            "date": dates,
            "industry": "通用设备",
            "roe": 8.0,
            "pe_ttm": list(range(101, 131)),
            "pb": [value / 10 for value in range(101, 131)],
        }
    )

    standalone = add_historical_valuation_features(primary).iloc[-1]
    combined = add_historical_valuation_features(
        pd.concat([primary, distraction], ignore_index=True)
    )
    with_cross_section = combined[combined["ts_code"] == "PRIMARY"].iloc[-1]

    assert with_cross_section["historical_value_score"] == pytest.approx(
        standalone["historical_value_score"]
    )


def test_production_window_retains_at_most_seven_years_of_month_ends() -> None:
    dates = pd.date_range("2017-01-31", periods=108, freq="ME")
    frame = pd.DataFrame(
        {
            "ts_code": ["A"] * len(dates),
            "date": dates,
            "industry": ["通用设备"] * len(dates),
            "roe": [10.0] * len(dates),
            "pe_ttm": list(range(len(dates), 0, -1)),
            "pb": [value / 10 for value in range(len(dates), 0, -1)],
        }
    )

    result = add_historical_valuation_features(
        frame,
        window_months=84,
        minimum_months=24,
    ).iloc[-1]

    assert result["valuation_history_points"] == 84
    assert result["pe_hist_percentile"] == pytest.approx(100 / 84)


def test_three_year_eps_forecast_mean_and_sample_variance_are_point_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = pd.DataFrame(
        [
            {
                "source": "datayes_consensus",
                "ts_code": "000001.SZ",
                "report_date": date,
                "report_title": "一致预期",
                "org_name": "datayes_consensus",
                "author_name": None,
                "forecast_year": year,
                "eps": eps,
                "pe": pe,
                "target_price": None,
                "net_profit": None,
                "revenue": None,
                "report_count": 10,
            }
            for date, year, eps, pe in [
                (pd.Timestamp("2026-06-30"), 2026, 1.0, 10.0),
                (pd.Timestamp("2026-07-30"), 2026, 1.2, 10.0),
                (pd.Timestamp("2026-07-30"), 2027, 1.5, 10.0),
                (pd.Timestamp("2026-07-30"), 2028, 2.0, 10.0),
                (pd.Timestamp("2026-07-30"), 2029, 3.0, 10.0),
                (pd.Timestamp("2026-08-01"), 2027, 9.9, 99.0),
            ]
        ]
        + [
            {
                "source": "akshare_em_research",
                "ts_code": "000001.SZ",
                "report_date": pd.Timestamp("2026-07-29"),
                "report_title": "机构预测",
                "org_name": "机构A",
                "author_name": "分析师甲",
                "forecast_year": 2027,
                "eps": 1.7,
                "pe": 12.0,
                "target_price": None,
                "net_profit": None,
                "revenue": None,
                "report_count": None,
            }
        ]
    )
    monkeypatch.setattr(dividend_module, "load_raw_analyst_reports", lambda: reports)
    features = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "date": [pd.Timestamp("2026-07-31")],
            "close": [10.0],
        }
    )

    result = dividend_module.load_analyst_forecast_asof(features).iloc[0]

    assert result["analyst_forward_eps_3y_mean_180d"] == pytest.approx(1.6)
    assert result["analyst_forward_eps_3y_variance_180d"] == pytest.approx(0.34 / 3)
    assert result["analyst_forward_eps_3y_years_180d"] == 3
    assert result["analyst_forward_eps_3y_estimate_count_180d"] == 4
    assert result["analyst_forward_y0_year"] == 2026
    assert result["analyst_forward_y0_eps_mean_180d"] == pytest.approx(1.2)
    assert pd.isna(result["analyst_forward_y0_eps_std_180d"])
    assert result["analyst_forward_y0_price_mean_180d"] == pytest.approx(12.0)
    assert result["analyst_forward_y1_year"] == 2027
    assert result["analyst_forward_y1_eps_mean_180d"] == pytest.approx(1.6)
    assert result["analyst_forward_y1_eps_std_180d"] == pytest.approx(0.1 * 2**0.5)
    assert result["analyst_forward_y1_price_mean_180d"] == pytest.approx(17.7)
    assert result["analyst_forward_y1_price_std_180d"] == pytest.approx(5.4 / 2**0.5)
    assert result["analyst_forward_y1_eps_estimate_count_180d"] == 2
    assert result["analyst_forward_y1_price_estimate_count_180d"] == 2
    assert result["analyst_forward_y2_year"] == 2028
    assert result["analyst_forward_y2_eps_mean_180d"] == pytest.approx(2.0)
    assert result["analyst_forward_y2_price_mean_180d"] == pytest.approx(20.0)


def test_three_year_forward_aggregates_exclude_actual_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = pd.DataFrame(
        [
            {
                "source": "datayes_consensus",
                "ts_code": "000001.SZ",
                "report_date": pd.Timestamp("2026-07-30"),
                "report_title": "实际值",
                "org_name": "consensus",
                "author_name": None,
                "forecast_year": 2026,
                "eps": 100.0,
                "revenue": 100_000.0,
                "net_profit": 10_000.0,
                "pe": 10.0,
                "target_price": None,
                "report_count": 10,
                "is_predict": False,
            },
            {
                "source": "akshare_em_research",
                "ts_code": "000001.SZ",
                "report_date": pd.Timestamp("2026-07-30"),
                "report_title": "预测值",
                "org_name": "机构A",
                "author_name": "分析师甲",
                "forecast_year": 2026,
                "eps": 2.0,
                "revenue": 200.0,
                "net_profit": 20.0,
                "pe": 8.0,
                "target_price": None,
                "report_count": None,
                "is_predict": True,
            },
        ]
    )
    monkeypatch.setattr(dividend_module, "load_raw_analyst_reports", lambda: reports)
    features = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "date": [pd.Timestamp("2026-07-31")], "close": [10.0]}
    )

    result = dividend_module.load_analyst_forecast_asof(features).iloc[0]

    assert result["analyst_forward_eps_3y_mean_180d"] == pytest.approx(2.0)
    assert result["analyst_forward_revenue_3y_mean_180d"] == pytest.approx(200.0)
    assert result["analyst_forward_net_profit_3y_mean_180d"] == pytest.approx(20.0)


def test_absolute_pr_bands_are_exclusive_at_one_and_two() -> None:
    frame = pd.DataFrame(
        {
            "pe_ttm": [9.0, 10.0, 15.0, 21.0],
            "pb": [0.9, 1.0, 1.5, 2.1],
            "pr": [0.9, 1.0, 2.0, 2.1],
            "pr_pe": [0.8, 1.0, 1.9, 2.2],
            "pr_pb": [0.9, 0.9, 2.0, 2.1],
            "valuation_profile": ["balanced"] * 4,
            "close": [10.0] * 4,
            "ma_120": [10.0] * 4,
            "ma_120_slope_20d": [0.0] * 4,
        }
    )

    masks = absolute_rule_masks(frame)

    assert masks["band_pr_lt_1"].tolist() == [True, False, False, False]
    assert masks["band_pr_1_to_2"].tolist() == [False, True, True, False]
    assert masks["band_pr_gt_2"].tolist() == [False, False, False, True]
    assert masks["abs_both_pr_lt_1"].tolist() == [True, False, False, False]


def test_profile_adapted_pr_uses_earnings_and_asset_formulas_separately() -> None:
    frame = pd.DataFrame(
        {
            "pe_ttm": [20.0, 20.0, 20.0],
            "pb": [2.0, 2.0, 2.0],
            "pr": [1.4, 1.4, 0.8],
            "pr_pe": [0.8, 2.0, 2.0],
            "pr_pb": [2.0, 0.8, 2.0],
            "valuation_profile": ["earnings_based", "asset_based", "balanced"],
            "close": [10.0] * 3,
            "ma_120": [10.0] * 3,
            "ma_120_slope_20d": [0.0] * 3,
        }
    )

    masks = absolute_rule_masks(frame)

    assert masks["abs_profile_pr_lt_1"].tolist() == [True, True, True]
    assert masks["abs_both_pr_lt_1"].tolist() == [False, False, False]


def test_weekly_historical_rule_requires_minimum_observation_years() -> None:
    frame = pd.DataFrame(
        {
            "valuation_history_points": [103, 104, 156],
            "pe_hist_percentile": [10.0, 10.0, 70.0],
            "pb_hist_percentile": [10.0, 10.0, 70.0],
            "pr_hist_percentile": [10.0, 10.0, 70.0],
            "historical_value_score": [80.0, 80.0, 40.0],
            "pr": [0.8, 0.8, 0.8],
            "close": [10.0] * 3,
            "ma_120": [10.0] * 3,
            "ma_120_slope_20d": [0.0] * 3,
        }
    )

    masks = historical_rule_masks(frame, minimum_weeks=2 * 52)

    assert masks["hist_composite60"].tolist() == [False, True, False]
    assert masks["hybrid_hist60_abs1"].tolist() == [False, True, False]
    assert masks["hybrid_hist60_or_abs1"].tolist() == [True, True, True]


def test_price_score_band_backtest_uses_disjoint_production_ranges() -> None:
    rows = []
    for date, score, stock_return in [
        ("2021-01-31", 80.0, 0.10),
        ("2021-01-31", 79.999, -0.10),
        ("2024-01-31", 60.0, 0.20),
        ("2024-01-31", 59.999, 0.30),
        ("2024-02-29", 20.0, -0.20),
        ("2024-02-29", 19.999, -0.30),
    ]:
        rows.append(
            {
                "date": pd.Timestamp(date),
                "ts_code": f"S{len(rows)}",
                "history_window_months": 84,
                "rule": "all_good_stocks",
                "historical_value_score": score,
                "return_12m": stock_return,
                "excess_return_12m": stock_return - 0.05,
                "mae_12m": min(stock_return, -0.10),
            }
        )

    summary = summarize_price_score_bands(pd.DataFrame(rows))
    payload = price_score_band_payload(summary)
    bands = {item["key"]: item for item in payload["bands"]}

    assert bands["80_100"]["validation"]["signals"] == 1
    assert bands["60_80"]["validation"]["signals"] == 1
    assert bands["60_80"]["test"]["signals"] == 1
    assert bands["40_60"]["test"]["signals"] == 1
    assert bands["20_40"]["test"]["signals"] == 1
    assert bands["0_20"]["test"]["signals"] == 1
    assert payload["execution"] == "next_trading_day_close"
