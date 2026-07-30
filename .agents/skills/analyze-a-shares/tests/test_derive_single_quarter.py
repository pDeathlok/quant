"""Regression tests for cumulative-to-standalone quarter derivation."""

from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import derive_single_quarter as quarter  # noqa: E402


class DeriveSingleQuarterTests(unittest.TestCase):
    def test_full_cumulative_series_derives_q1_through_q4(self) -> None:
        data = {
            "year": 2025,
            "unit": "CNY_million",
            "metrics": {
                "revenue": {"q1": "100", "h1": "230", "q3_ytd": "390", "fy": "580"},
                "net_profit": {"q1": "10", "h1": "27", "q3_ytd": "45", "fy": "70"},
            },
        }

        results, warnings = quarter.derive_quarters(data)

        self.assertEqual(
            results["revenue"],
            {
                "q1": Decimal("100"),
                "q2": Decimal("130"),
                "q3": Decimal("160"),
                "q4": Decimal("190"),
            },
        )
        self.assertEqual(
            results["net_profit"],
            {
                "q1": Decimal("10"),
                "q2": Decimal("17"),
                "q3": Decimal("18"),
                "q4": Decimal("25"),
            },
        )
        self.assertEqual(warnings, [])

    def test_missing_cumulative_pair_returns_none_and_warning(self) -> None:
        data = {
            "year": 2025,
            "unit": "CNY_million",
            "metrics": {"revenue": {"q1": "100", "q3_ytd": "390"}},
        }

        results, warnings = quarter.derive_quarters(data)

        self.assertEqual(
            results["revenue"],
            {"q1": Decimal("100"), "q2": None, "q3": None, "q4": None},
        )
        self.assertTrue(any("Q2" in warning for warning in warnings))
        self.assertTrue(any("Q3" in warning for warning in warnings))
        self.assertTrue(any("Q4" in warning for warning in warnings))

    def test_negative_derived_quarter_is_preserved_and_warned(self) -> None:
        data = {
            "year": 2025,
            "unit": "CNY_million",
            "metrics": {"net_profit": {"q1": "10", "h1": "8"}},
        }

        results, warnings = quarter.derive_quarters(data)

        self.assertEqual(results["net_profit"]["q2"], Decimal("-2"))
        self.assertTrue(any("negative" in warning for warning in warnings))

    def test_unknown_period_and_missing_unit_are_rejected(self) -> None:
        with self.assertRaisesRegex(quarter.InputError, "unsupported keys"):
            quarter.derive_quarters(
                {"year": 2025, "unit": "CNY", "metrics": {"x": {"q5": 1}}}
            )
        with self.assertRaisesRegex(quarter.InputError, "unit"):
            quarter.derive_quarters({"year": 2025, "metrics": {"x": {"q1": 1}}})


if __name__ == "__main__":
    unittest.main()
