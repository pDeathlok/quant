"""Regression tests for the point-in-time price-volume snapshot."""

from __future__ import annotations

import copy
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import price_volume_snapshot as snapshot  # noqa: E402


def make_bars(count: int = 120, *, start: float = 10.0, step: float = 0.05) -> list[dict[str, object]]:
    first_date = date(2026, 1, 1)
    bars: list[dict[str, object]] = []
    for index in range(count):
        close = start + index * step
        open_price = close - step / 2 if step >= 0 else close - step / 2
        bars.append(
            {
                "date": (first_date + timedelta(days=index)).isoformat(),
                "open": round(open_price, 4),
                "high": round(max(open_price, close) + 0.1, 4),
                "low": round(min(open_price, close) - 0.1, 4),
                "close": round(close, 4),
                "volume": 1000,
            }
        )
    return bars


def replace_bar(
    bars: list[dict[str, object]],
    index: int,
    *,
    close: float,
    open_price: float,
    volume: float,
) -> None:
    original_date = bars[index]["date"]
    bars[index] = {
        "date": original_date,
        "open": open_price,
        "high": max(open_price, close) + 0.2,
        "low": min(open_price, close) - 0.2,
        "close": close,
        "volume": volume,
    }


def valid_input(bars: list[dict[str, object]] | None = None) -> dict[str, object]:
    bars = bars or make_bars()
    last_date = str(bars[-1]["date"])
    return {
        "ticker": "600000.SH",
        "as_of_date": last_date,
        "analysis_cutoff": f"{last_date}T18:00:00+08:00",
        "last_bar_available_at": f"{last_date}T16:00:00+08:00",
        "source": "verified daily OHLCV fixture",
        "price_basis": "unadjusted",
        "volume_unit": "shares",
        "bars": bars,
    }


class PriceVolumeSignalTests(unittest.TestCase):
    def test_expanded_volume_breakout_returns_constructive_observations(self) -> None:
        bars = make_bars()
        prior_high = max(float(bar["high"]) for bar in bars[-21:-1])
        replace_bar(
            bars,
            -1,
            close=prior_high + 1.0,
            open_price=prior_high,
            volume=2500,
        )

        result = snapshot.analyze_price_volume(valid_input(bars))

        self.assertEqual(result["trend"]["state"], "constructive")
        self.assertTrue(result["signals"]["breakout_on_expanded_volume"])
        self.assertTrue(result["signals"]["high_volume_bullish_key_bar"])
        self.assertGreater(len(result["supportive_observations"]), 0)

    def test_expanded_volume_breakdown_returns_cautious_observations(self) -> None:
        bars = make_bars(start=20, step=-0.05)
        prior_low = min(float(bar["low"]) for bar in bars[-21:-1])
        replace_bar(
            bars,
            -1,
            close=prior_low - 1.0,
            open_price=prior_low,
            volume=2500,
        )

        result = snapshot.analyze_price_volume(valid_input(bars))

        self.assertEqual(result["trend"]["state"], "cautious")
        self.assertTrue(result["signals"]["breakdown_on_expanded_volume"])
        self.assertTrue(result["signals"]["high_volume_bearish_key_bar"])
        self.assertGreater(len(result["risk_observations"]), 0)

    def test_contracting_pullback_requires_a_prior_positive_volume_impulse(self) -> None:
        bars = make_bars(count=80)
        previous_close = float(bars[69]["close"])
        replace_bar(
            bars,
            70,
            close=previous_close + 1.0,
            open_price=previous_close + 0.2,
            volume=2500,
        )
        for index, close in zip(range(71, 75), (14.30, 14.20, 14.10, 14.00)):
            replace_bar(bars, index, close=close, open_price=close + 0.03, volume=900)
        for index, close in zip(range(75, 80), (13.95, 13.90, 13.85, 13.80, 13.75)):
            replace_bar(bars, index, close=close, open_price=close + 0.03, volume=500)

        result = snapshot.analyze_price_volume(valid_input(bars))

        self.assertTrue(
            result["signals"]["pullback_on_contracting_volume_after_impulse"]
        )
        self.assertEqual(result["volume_price"]["quadrant_5d"], "price_down_volume_down")

    def test_contracting_decline_without_prior_impulse_is_not_labeled_pullback_setup(self) -> None:
        bars = make_bars(count=80)
        for index, close in zip(range(75, 80), (13.90, 13.85, 13.80, 13.75, 13.70)):
            replace_bar(bars, index, close=close, open_price=close + 0.03, volume=500)

        result = snapshot.analyze_price_volume(valid_input(bars))

        self.assertFalse(
            result["signals"]["pullback_on_contracting_volume_after_impulse"]
        )

    def test_sixty_bars_omit_ma120_and_disclose_the_gap(self) -> None:
        result = snapshot.analyze_price_volume(valid_input(make_bars(count=60)))

        self.assertIsNone(result["trend"]["ma120"])
        self.assertIn("少于120根日线", result["warnings"][0])

    def test_zero_volume_is_preserved_and_warned_instead_of_filled(self) -> None:
        bars = make_bars()
        bars[-1]["volume"] = 0

        result = snapshot.analyze_price_volume(valid_input(bars))

        self.assertTrue(any("零成交量" in warning for warning in result["warnings"]))
        self.assertEqual(result["volume_price"]["current_volume_ratio_to_prior_20d"], 0.0)


class PriceVolumeInputContractTests(unittest.TestCase):
    def test_bar_dates_must_be_unique_and_strictly_increasing(self) -> None:
        data = valid_input()
        bars = data["bars"]
        assert isinstance(bars, list)
        bars[-1]["date"] = bars[-2]["date"]

        with self.assertRaisesRegex(snapshot.InputError, "strictly increasing"):
            snapshot.analyze_price_volume(data)

    def test_ohlc_invariants_are_rejected(self) -> None:
        data = valid_input()
        bars = data["bars"]
        assert isinstance(bars, list)
        bars[-1]["high"] = float(bars[-1]["close"]) - 1

        with self.assertRaisesRegex(snapshot.InputError, "high"):
            snapshot.analyze_price_volume(data)

    def test_last_bar_availability_cannot_be_after_cutoff(self) -> None:
        data = valid_input()
        data["last_bar_available_at"] = f"{data['as_of_date']}T19:00:00+08:00"

        with self.assertRaisesRegex(snapshot.InputError, "cannot be later"):
            snapshot.analyze_price_volume(data)

    def test_adjusted_series_requires_nonfuture_reference_date(self) -> None:
        for reference_date in (None, "2027-01-01"):
            with self.subTest(reference_date=reference_date):
                data = valid_input()
                data["price_basis"] = "forward_adjusted"
                if reference_date is not None:
                    data["adjustment_reference_date"] = reference_date

                with self.assertRaisesRegex(
                    snapshot.InputError, "adjustment_reference_date"
                ):
                    snapshot.analyze_price_volume(data)

    def test_valid_adjusted_series_discloses_comparison_warning(self) -> None:
        data = valid_input()
        data["price_basis"] = "forward_adjusted"
        data["adjustment_reference_date"] = data["as_of_date"]

        result = snapshot.analyze_price_volume(data)

        self.assertEqual(result["price_basis"], "forward_adjusted")
        self.assertTrue(any("复权桥" in warning for warning in result["warnings"]))

    def test_adjustment_reference_cannot_precede_last_bar(self) -> None:
        data = valid_input()
        data["price_basis"] = "backward_adjusted"
        data["adjustment_reference_date"] = "2026-01-01"

        with self.assertRaisesRegex(snapshot.InputError, "earlier than the last bar"):
            snapshot.analyze_price_volume(data)

    def test_last_bar_date_cannot_be_after_analysis_date(self) -> None:
        data = valid_input()
        data["as_of_date"] = "2026-04-29"
        data["analysis_cutoff"] = "2026-04-29T18:00:00+08:00"
        data["last_bar_available_at"] = "2026-04-29T17:00:00+08:00"

        with self.assertRaisesRegex(snapshot.InputError, "last bar date"):
            snapshot.analyze_price_volume(data)

    def test_unknown_or_invalid_thresholds_are_rejected(self) -> None:
        invalid_thresholds = (
            {"unknown": 1},
            {"volume_expansion_ratio": 1},
            {"volume_contraction_ratio": 1},
            {"breakout_buffer_pct": -0.1},
        )
        for thresholds in invalid_thresholds:
            with self.subTest(thresholds=thresholds):
                data = valid_input()
                data["thresholds"] = thresholds

                with self.assertRaises(snapshot.InputError):
                    snapshot.analyze_price_volume(data)

    def test_fewer_than_sixty_bars_are_rejected(self) -> None:
        with self.assertRaisesRegex(snapshot.InputError, "at least 60"):
            snapshot.analyze_price_volume(valid_input(make_bars(count=59)))

    def test_markdown_labels_output_as_nonprescriptive_technical_aid(self) -> None:
        result = snapshot.analyze_price_volume(valid_input())

        markdown = snapshot.render_markdown(result, 4)

        self.assertIn("技术辅助层", markdown)
        self.assertIn("不构成交易指令", markdown)

    def test_validation_does_not_mutate_caller_input(self) -> None:
        data = valid_input()
        original = copy.deepcopy(data)

        snapshot.analyze_price_volume(data)

        self.assertEqual(data, original)


if __name__ == "__main__":
    unittest.main()
