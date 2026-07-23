"""Regression tests for the A-share scenario valuation input contract."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import scenario_valuation as valuation  # noqa: E402


def mixed_scenarios() -> dict[str, dict[str, object]]:
    """Return one valid scenario for each arithmetic method except target_price."""

    return {
        "bear": {
            "method": "per_share_multiple",
            "metric_name": "diluted EPS",
            "metric_period": "FY2027E",
            "bridge_as_of": "2026-06-30",
            "metric_unit": "CNY_per_share",
            "metric_per_share": "0.6",
            "multiple": "10",
            "multiple_basis": "peer median less risk discount",
        },
        "base": {
            "method": "enterprise_multiple",
            "metric_name": "EBITDA",
            "metric_period": "FY2027E",
            "bridge_as_of": "2026-06-30",
            "metric_total": "1200",
            "multiple": "8",
            "multiple_basis": "same-period peer median",
            "total_value_unit": "CNY_million",
            "debt": "2400",
            "cash": "800",
            "minority_interest": "100",
            "preferred_equity": "0",
            "non_operating_assets": "200",
            "diluted_shares": "720",
            "shares_unit": "million_shares",
            "shares_period": "FY2027E",
        },
        "bull": {
            "method": "equity_value",
            "metric_period": "FY2027E DCF",
            "bridge_as_of": "2026-06-30",
            "equity_value": "12000",
            "total_value_unit": "CNY_million",
            "diluted_shares": "700",
            "shares_unit": "million_shares",
            "shares_period": "FY2027E",
            "method_note": "five-year FCFF DCF",
        },
    }


def target_price_scenario(target_price: str = "12") -> dict[str, object]:
    return {
        "method": "target_price",
        "metric_period": "FY2027E SOTP",
        "bridge_as_of": "2026-06-30",
        "target_price": target_price,
        "target_price_unit": "CNY_per_share",
        "method_note": "externally reviewed working paper",
        "model_type": "sotp",
        "model_reference": "valuation-workbook.xlsx#SOTP",
        "independent_check": "recomputed division by diluted shares and bridge totals",
    }


def valuation_input() -> dict[str, object]:
    return {
        "company": "Example A Share",
        "ticker": "600000.SH",
        "currency": "CNY",
        "as_of_date": "2026-07-17",
        "analysis_cutoff": "2026-07-17T15:30:00+08:00",
        "price_as_of": "2026-07-17T15:00:00+08:00",
        "current_price": "10",
        "current_price_source": "exchange close",
        "price_basis": "unadjusted",
        "target_date": "2027-07-17",
        "scenarios_exhaustive": False,
        "scenarios": mixed_scenarios(),
    }


def add_probabilities(
    data: dict[str, object], values: tuple[str, ...] = ("0.2", "0.5", "0.3")
) -> None:
    scenarios = data["scenarios"]
    assert isinstance(scenarios, dict)
    for key, probability in zip(valuation.SCENARIO_ORDER, values):
        scenario = scenarios[key]
        assert isinstance(scenario, dict)
        scenario["probability"] = probability


class ScenarioMethodContractTests(unittest.TestCase):
    def calculate(
        self, data: dict[str, object]
    ) -> tuple[valuation.RootContext, list[valuation.ScenarioResult], list[str]]:
        context, warnings = valuation.validate_root(data)
        results = valuation.calculate_scenarios(data, context, warnings)
        return context, results, warnings

    def test_three_calculated_methods_return_decimal_targets(self) -> None:
        _, results, _ = self.calculate(valuation_input())

        self.assertEqual(
            [result.target_price for result in results],
            [Decimal("6.0"), Decimal("11.25"), Decimal(120) / Decimal(7)],
        )
        self.assertTrue(all(isinstance(result.target_price, Decimal) for result in results))

    def test_target_price_method_returns_decimal_target(self) -> None:
        data = valuation_input()
        data["scenarios"] = {
            key: target_price_scenario(price)
            for key, price in zip(valuation.SCENARIO_ORDER, ("8", "12", "16"))
        }

        context, results, warnings = self.calculate(data)
        payload = json.loads(valuation.render_json(data, context, results, warnings))

        self.assertEqual(
            [result.target_price for result in results],
            [Decimal("8"), Decimal("12"), Decimal("16")],
        )
        self.assertEqual(
            payload["scenarios"]["base"]["external_model_audit"],
            {
                "model_type": "sotp",
                "model_reference": "valuation-workbook.xlsx#SOTP",
                "independent_check": (
                    "recomputed division by diluted shares and bridge totals"
                ),
                "recalculated_by_script": False,
            },
        )

    def test_plain_cny_and_shares_pair_is_unambiguous(self) -> None:
        scenario = {
            "method": "equity_value",
            "metric_period": "FY2027E DCF",
            "bridge_as_of": "2026-06-30",
            "equity_value": "12000000000",
            "total_value_unit": "CNY",
            "diluted_shares": "700000000",
            "shares_unit": "shares",
            "shares_period": "FY2027E",
            "method_note": "five-year FCFF DCF",
        }

        _, target, calculation, _, _ = valuation.calculate_target(scenario, "scenario")

        self.assertEqual(target, Decimal(120) / Decimal(7))
        self.assertIn("12000000000×1 CNY", calculation)
        self.assertIn("700000000×1 shares", calculation)

    def test_million_unit_pair_shows_both_conversion_scales(self) -> None:
        scenario = mixed_scenarios()["base"]

        _, target, calculation, _, _ = valuation.calculate_target(scenario, "scenario")

        self.assertEqual(target, Decimal("11.25"))
        self.assertIn("×1000000 CNY", calculation)
        self.assertIn("×1000000 shares", calculation)

    def test_incompatible_total_and_share_units_are_rejected(self) -> None:
        scenario = mixed_scenarios()["base"]
        scenario["shares_unit"] = "shares"

        with self.assertRaisesRegex(valuation.InputError, "compatible pair"):
            valuation.calculate_target(scenario, "scenario")

    def test_legacy_or_unsupported_total_units_are_rejected(self) -> None:
        for field, value, expected in (
            ("value_unit", "CNY_million", "total_value_unit"),
            ("total_value_unit", "CNY_100_million", "total_value_unit"),
        ):
            with self.subTest(field=field, value=value):
                scenario = mixed_scenarios()["base"]
                scenario.pop("total_value_unit")
                scenario[field] = value

                with self.assertRaisesRegex(valuation.InputError, expected):
                    valuation.calculate_target(scenario, "scenario")

    def test_every_method_requires_metric_period_and_bridge_date(self) -> None:
        scenarios = [
            mixed_scenarios()["bear"],
            mixed_scenarios()["base"],
            mixed_scenarios()["bull"],
            target_price_scenario(),
        ]
        for required_field in ("metric_period", "bridge_as_of"):
            for scenario in scenarios:
                with self.subTest(method=scenario["method"], field=required_field):
                    invalid = copy.deepcopy(scenario)
                    invalid.pop(required_field)

                    with self.assertRaisesRegex(valuation.InputError, required_field):
                        valuation.calculate_target(invalid, "scenario")

    def test_method_specific_missing_field_is_rejected(self) -> None:
        scenario = mixed_scenarios()["base"]
        scenario.pop("debt")

        with self.assertRaisesRegex(valuation.InputError, "debt is required"):
            valuation.calculate_target(scenario, "scenario")

    def test_external_target_requires_model_audit_fields(self) -> None:
        for field in ("model_type", "model_reference", "independent_check"):
            with self.subTest(field=field):
                scenario = target_price_scenario()
                scenario.pop(field)

                with self.assertRaisesRegex(valuation.InputError, field):
                    valuation.calculate_target(scenario, "scenario")

    def test_external_target_rejects_unknown_model_type(self) -> None:
        scenario = target_price_scenario()
        scenario["model_type"] = "magic_model"

        with self.assertRaisesRegex(valuation.InputError, "model_type"):
            valuation.calculate_target(scenario, "scenario")


class RootContractTests(unittest.TestCase):
    def test_all_three_a_share_exchange_suffixes_are_accepted(self) -> None:
        for ticker in ("600519.SH", "000001.SZ", "920000.BJ"):
            with self.subTest(ticker=ticker):
                data = valuation_input()
                data["ticker"] = ticker

                context, _ = valuation.validate_root(data)

                self.assertEqual(context.currency, "CNY")

    def test_malformed_or_non_a_share_tickers_are_rejected(self) -> None:
        for ticker in (
            "60000.SH",
            "6000000.SH",
            "600000SH",
            "600000.HK",
            "600000.sh",
        ):
            with self.subTest(ticker=ticker):
                data = valuation_input()
                data["ticker"] = ticker

                with self.assertRaisesRegex(valuation.InputError, "ticker"):
                    valuation.validate_root(data)

    def test_cny_and_rmb_are_accepted_and_output_is_canonical_cny(self) -> None:
        for input_currency in ("CNY", "RMB"):
            with self.subTest(currency=input_currency):
                data = valuation_input()
                data["currency"] = input_currency
                context, warnings = valuation.validate_root(data)
                results = valuation.calculate_scenarios(data, context, warnings)

                json_payload = json.loads(
                    valuation.render_json(data, context, results, warnings)
                )
                markdown = valuation.render_markdown(data, context, results, warnings, 2)

                self.assertEqual(context.currency, "CNY")
                self.assertEqual(json_payload["currency"], "CNY")
                self.assertIn("基准价格：CNY", markdown)

    def test_other_currency_is_rejected(self) -> None:
        data = valuation_input()
        data["currency"] = "USD"

        with self.assertRaisesRegex(valuation.InputError, "currency"):
            valuation.validate_root(data)

    def test_as_of_and_target_dates_require_extended_iso_date(self) -> None:
        for field, value in (
            ("as_of_date", "20260717"),
            ("as_of_date", "2026-02-30"),
            ("target_date", "2027-07-17T00:00:00+08:00"),
        ):
            with self.subTest(field=field, value=value):
                data = valuation_input()
                data[field] = value

                with self.assertRaisesRegex(valuation.InputError, field):
                    valuation.validate_root(data)

    def test_analysis_cutoff_and_price_timestamp_require_timezone(self) -> None:
        for field in ("analysis_cutoff", "price_as_of"):
            with self.subTest(field=field):
                data = valuation_input()
                data[field] = "2026-07-17T15:00:00"

                with self.assertRaisesRegex(valuation.InputError, "UTC offset"):
                    valuation.validate_root(data)

    def test_z_timezone_is_accepted(self) -> None:
        data = valuation_input()
        data["analysis_cutoff"] = "2026-07-17T07:30:00Z"
        data["price_as_of"] = "2026-07-17T07:00:00Z"

        context, _ = valuation.validate_root(data)

        self.assertEqual(context.analysis_cutoff.utcoffset().total_seconds(), 0)

    def test_price_as_of_cannot_be_after_cutoff_across_offsets(self) -> None:
        data = valuation_input()
        data["analysis_cutoff"] = "2026-07-17T15:30:00+08:00"
        data["price_as_of"] = "2026-07-17T08:00:00+00:00"

        with self.assertRaisesRegex(valuation.InputError, "cannot be later"):
            valuation.validate_root(data)

    def test_as_of_date_must_match_cutoff_local_date(self) -> None:
        data = valuation_input()
        data["as_of_date"] = "2026-07-16"

        with self.assertRaisesRegex(valuation.InputError, "local calendar date"):
            valuation.validate_root(data)

    def test_target_date_must_be_after_as_of_date(self) -> None:
        for target_date in ("2026-07-16", "2026-07-17"):
            with self.subTest(target_date=target_date):
                data = valuation_input()
                data["target_date"] = target_date

                with self.assertRaisesRegex(valuation.InputError, "must be later"):
                    valuation.validate_root(data)


class ScenarioSetAndProbabilityTests(unittest.TestCase):
    def calculate(self, data: dict[str, object]) -> list[valuation.ScenarioResult]:
        context, warnings = valuation.validate_root(data)
        return valuation.calculate_scenarios(data, context, warnings)

    def test_results_use_bear_base_bull_order_regardless_of_input_order(self) -> None:
        data = valuation_input()
        scenarios = data["scenarios"]
        assert isinstance(scenarios, dict)
        data["scenarios"] = {
            "bull": scenarios["bull"],
            "bear": scenarios["bear"],
            "base": scenarios["base"],
        }

        results = self.calculate(data)

        self.assertEqual([result.key for result in results], ["bear", "base", "bull"])

    def test_non_monotonic_prices_are_rejected(self) -> None:
        data = valuation_input()
        scenarios = data["scenarios"]
        assert isinstance(scenarios, dict)
        bull = scenarios["bull"]
        assert isinstance(bull, dict)
        bull["equity_value"] = "1000"

        with self.assertRaisesRegex(valuation.InputError, "bear <= base <= bull"):
            self.calculate(data)

    def test_scenario_set_must_have_exactly_three_named_members(self) -> None:
        data = valuation_input()
        scenarios = data["scenarios"]
        assert isinstance(scenarios, dict)
        scenarios.pop("bear")

        with self.assertRaisesRegex(valuation.InputError, "exactly bear, base and bull"):
            self.calculate(data)

    def test_probability_is_rejected_without_exhaustive_gate(self) -> None:
        data = valuation_input()
        add_probabilities(data)

        with self.assertRaisesRegex(valuation.InputError, "scenarios_exhaustive=true"):
            self.calculate(data)

    def test_probability_requires_nonempty_basis(self) -> None:
        for probability_basis in (None, "   "):
            with self.subTest(probability_basis=probability_basis):
                data = valuation_input()
                data["scenarios_exhaustive"] = True
                if probability_basis is not None:
                    data["probability_basis"] = probability_basis
                add_probabilities(data)

                with self.assertRaisesRegex(valuation.InputError, "probability_basis"):
                    self.calculate(data)

    def test_probability_requires_all_three_values(self) -> None:
        data = valuation_input()
        data["scenarios_exhaustive"] = True
        data["probability_basis"] = "mutually exclusive order-conversion event tree"
        add_probabilities(data, ("0.2", "0.8"))

        with self.assertRaisesRegex(valuation.InputError, "all three"):
            self.calculate(data)

    def test_probability_sum_must_equal_one_exactly(self) -> None:
        data = valuation_input()
        data["scenarios_exhaustive"] = True
        data["probability_basis"] = "mutually exclusive order-conversion event tree"
        add_probabilities(data, ("0.2", "0.5", "0.30001"))

        with self.assertRaisesRegex(valuation.InputError, "sum to 1.0"):
            self.calculate(data)

    def test_valid_probability_contract_calculates_weighted_price(self) -> None:
        data = valuation_input()
        data["scenarios_exhaustive"] = True
        data["probability_basis"] = "mutually exclusive order-conversion event tree"
        add_probabilities(data)
        context, warnings = valuation.validate_root(data)
        results = valuation.calculate_scenarios(data, context, warnings)

        weighted = valuation.weighted_summary(results, context.current_price)

        expected = (
            Decimal("6.0") * Decimal("0.2")
            + Decimal("11.25") * Decimal("0.5")
            + (Decimal(120) / Decimal(7)) * Decimal("0.3")
        )
        self.assertIsNotNone(weighted)
        assert weighted is not None
        self.assertEqual(weighted["target_price"], expected)


if __name__ == "__main__":
    unittest.main()
