"""Regression tests for the industry market-size reconciliation contract."""

from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import reconcile_market_size as sizing  # noqa: E402


def market_input() -> dict[str, object]:
    return {
        "markets": [
            {
                "id": "C1",
                "name": "Example Component",
                "geography": "China demand",
                "base_year": 2025,
                "unit": "CNY_billion",
                "basis": "manufacturer_revenue",
                "estimates": [
                    {
                        "label": "official",
                        "method": "top_down",
                        "value": "100",
                        "weight": "2",
                    },
                    {
                        "label": "volume_x_asp",
                        "method": "bottom_up",
                        "value": "108",
                        "weight": "1",
                    },
                ],
                "forecast": {
                    "end_year": 2030,
                    "end_value": "161.051",
                    "cagr": "0.10",
                },
                "flow_balance": {
                    "production": "120",
                    "imports": "15",
                    "exports": "30",
                    "inventory_increase": "2",
                    "stated_demand": "103",
                },
            }
        ]
    }


class EstimateSummaryTests(unittest.TestCase):
    def test_reconciles_range_centers_and_alignment(self) -> None:
        result = sizing.calculate(market_input())

        summary = result["markets"][0]["estimate_summary"]

        self.assertEqual(summary["minimum"], Decimal("100"))
        self.assertEqual(summary["maximum"], Decimal("108"))
        self.assertEqual(summary["median"], Decimal("104"))
        self.assertEqual(summary["weighted_median"], Decimal("100"))
        self.assertEqual(
            summary["weighted_mean"], Decimal("308") / Decimal("3")
        )
        self.assertEqual(summary["alignment"], "aligned")

    def test_single_estimate_is_not_presented_as_cross_validation(self) -> None:
        data = market_input()
        market = data["markets"][0]
        assert isinstance(market, dict)
        estimates = market["estimates"]
        assert isinstance(estimates, list)
        market["estimates"] = estimates[:1]

        result = sizing.calculate(data)

        summary = result["markets"][0]["estimate_summary"]
        self.assertEqual(summary["alignment"], "single_estimate")
        self.assertEqual(summary["count"], 1)

    def test_equal_weights_average_the_two_middle_estimates(self) -> None:
        data = market_input()
        market = data["markets"][0]
        assert isinstance(market, dict)
        estimates = market["estimates"]
        assert isinstance(estimates, list)
        first = estimates[0]
        assert isinstance(first, dict)
        first["weight"] = "1"

        result = sizing.calculate(data)

        summary = result["markets"][0]["estimate_summary"]
        self.assertEqual(summary["weighted_median"], Decimal("104"))

    def test_duplicate_estimate_labels_are_rejected(self) -> None:
        data = market_input()
        market = data["markets"][0]
        assert isinstance(market, dict)
        estimates = market["estimates"]
        assert isinstance(estimates, list)
        second = estimates[1]
        assert isinstance(second, dict)
        second["label"] = "official"

        with self.assertRaisesRegex(sizing.InputError, "duplicate label"):
            sizing.calculate(data)


class ForecastTests(unittest.TestCase):
    def test_matching_end_value_and_cagr_are_consistent(self) -> None:
        result = sizing.calculate(market_input())

        forecast = result["markets"][0]["forecast"]
        assert isinstance(forecast, dict)

        self.assertEqual(forecast["implied_end_from_cagr"], Decimal("161.05100"))
        self.assertEqual(forecast["consistency"], "consistent")
        self.assertLess(forecast["gap_ratio"], Decimal("0.000001"))

    def test_material_forecast_gap_is_flagged(self) -> None:
        data = market_input()
        market = data["markets"][0]
        assert isinstance(market, dict)
        forecast = market["forecast"]
        assert isinstance(forecast, dict)
        forecast["end_value"] = "200"

        result = sizing.calculate(data)

        output = result["markets"][0]["forecast"]
        assert isinstance(output, dict)
        self.assertEqual(output["consistency"], "inconsistent")
        self.assertGreater(output["gap_ratio"], Decimal("0.01"))

    def test_forecast_requires_future_end_year(self) -> None:
        data = market_input()
        market = data["markets"][0]
        assert isinstance(market, dict)
        forecast = market["forecast"]
        assert isinstance(forecast, dict)
        forecast["end_year"] = 2025

        with self.assertRaisesRegex(sizing.InputError, "later than base_year"):
            sizing.calculate(data)


class FlowBalanceTests(unittest.TestCase):
    def test_reconciles_apparent_demand(self) -> None:
        result = sizing.calculate(market_input())

        flow = result["markets"][0]["flow_balance"]
        assert isinstance(flow, dict)

        self.assertEqual(flow["implied_demand"], Decimal("103"))
        self.assertEqual(flow["demand_gap"], Decimal("0"))
        self.assertEqual(flow["gap_ratio"], Decimal("0"))
        self.assertFalse(flow["inventory_assumed_zero"])

    def test_missing_inventory_defaults_to_zero_and_is_disclosed(self) -> None:
        data = market_input()
        market = data["markets"][0]
        assert isinstance(market, dict)
        flow = market["flow_balance"]
        assert isinstance(flow, dict)
        flow.pop("inventory_increase")
        flow["stated_demand"] = "105"

        result = sizing.calculate(data)

        output = result["markets"][0]["flow_balance"]
        assert isinstance(output, dict)
        self.assertEqual(output["implied_demand"], Decimal("105"))
        self.assertTrue(output["inventory_assumed_zero"])

    def test_negative_implied_demand_is_rejected(self) -> None:
        data = market_input()
        market = data["markets"][0]
        assert isinstance(market, dict)
        flow = market["flow_balance"]
        assert isinstance(flow, dict)
        flow["exports"] = "500"

        with self.assertRaisesRegex(sizing.InputError, "negative demand"):
            sizing.calculate(data)


class RootContractTests(unittest.TestCase):
    def test_duplicate_market_ids_are_rejected(self) -> None:
        data = market_input()
        markets = data["markets"]
        assert isinstance(markets, list)
        markets.append(dict(markets[0]))

        with self.assertRaisesRegex(sizing.InputError, "duplicate market ids"):
            sizing.calculate(data)

    def test_markdown_output_discloses_method_limit(self) -> None:
        output = sizing.render_markdown(sizing.calculate(market_input()), precision=2)

        self.assertIn("# 市场规模复核", output)
        self.assertIn("算术工作", output)
        self.assertIn("加权中位数", output)


if __name__ == "__main__":
    unittest.main()
