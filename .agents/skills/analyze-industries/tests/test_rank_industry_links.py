"""Regression tests for transparent industry-link potential scoring."""

from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import rank_industry_links as ranking  # noqa: E402


def link(
    link_id: str,
    name: str,
    score: str,
    *,
    confidence: str = "0.75",
) -> dict[str, object]:
    return {
        "id": link_id,
        "name": name,
        "scores": {dimension: score for dimension in ranking.DIMENSIONS},
        "confidence": confidence,
        "evidence_note": f"Evidence for {name}",
    }


def ranking_input() -> dict[str, object]:
    return {
        "perspective": "listed_equity",
        "links": [
            link("L1", "High Link", "5", confidence="0.60"),
            link("L2", "Medium Link", "3", confidence="0.90"),
            link("L3", "Low Link", "1", confidence="0.95"),
        ],
    }


class ScoreCalculationTests(unittest.TestCase):
    def test_default_weights_map_uniform_scores_to_expected_scale(self) -> None:
        result = ranking.calculate(ranking_input())
        by_id = {item["id"]: item for item in result["links"]}

        self.assertEqual(by_id["L1"]["potential_score"], Decimal("100"))
        self.assertEqual(by_id["L1"]["tier"], "高")
        self.assertEqual(by_id["L2"]["potential_score"], Decimal("50"))
        self.assertEqual(by_id["L2"]["tier"], "中")
        self.assertEqual(by_id["L3"]["potential_score"], Decimal("0"))
        self.assertEqual(by_id["L3"]["tier"], "低")

    def test_custom_weights_are_normalized(self) -> None:
        data = ranking_input()
        data["weights"] = {dimension: "1" for dimension in ranking.DIMENSIONS}

        result = ranking.calculate(data)

        self.assertTrue(
            all(weight == Decimal("0.2") for weight in result["weights"].values())
        )

    def test_non_uniform_scores_apply_explicit_default_weights(self) -> None:
        data = {
            "perspective": "listed_equity",
            "links": [
                {
                    "id": "L1",
                    "name": "Mixed Link",
                    "scores": {
                        "market_space": "5",
                        "profit_pool": "4",
                        "structural_quality": "3",
                        "china_opportunity": "2",
                        "realization_certainty": "1",
                    },
                    "confidence": "0.7",
                    "evidence_note": "Mixed evidence",
                }
            ],
        }

        result = ranking.calculate(data)

        item = result["links"][0]
        self.assertEqual(item["weighted_score_1_to_5"], Decimal("3.35"))
        self.assertEqual(item["potential_score"], Decimal("58.75"))
        self.assertEqual(item["tier"], "中")

    def test_ranking_uses_score_then_confidence(self) -> None:
        data = ranking_input()
        links = data["links"]
        assert isinstance(links, list)
        links.append(link("L4", "Another High", "5", confidence="0.80"))

        result = ranking.calculate(data)

        self.assertEqual(
            [item["id"] for item in result["links"]],
            ["L4", "L1", "L2", "L3"],
        )


class ValidationTests(unittest.TestCase):
    def test_score_outside_one_to_five_is_rejected(self) -> None:
        data = ranking_input()
        links = data["links"]
        assert isinstance(links, list)
        first = links[0]
        assert isinstance(first, dict)
        scores = first["scores"]
        assert isinstance(scores, dict)
        scores["market_space"] = "6"

        with self.assertRaisesRegex(ranking.InputError, "between 1 and 5"):
            ranking.calculate(data)

    def test_confidence_outside_zero_to_one_is_rejected(self) -> None:
        data = ranking_input()
        links = data["links"]
        assert isinstance(links, list)
        first = links[0]
        assert isinstance(first, dict)
        first["confidence"] = "1.1"

        with self.assertRaisesRegex(ranking.InputError, "between 0 and 1"):
            ranking.calculate(data)

    def test_missing_dimension_is_rejected(self) -> None:
        data = ranking_input()
        links = data["links"]
        assert isinstance(links, list)
        first = links[0]
        assert isinstance(first, dict)
        scores = first["scores"]
        assert isinstance(scores, dict)
        scores.pop("profit_pool")

        with self.assertRaisesRegex(ranking.InputError, "exactly the five"):
            ranking.calculate(data)

    def test_zero_weight_sum_is_rejected(self) -> None:
        data = ranking_input()
        data["weights"] = {dimension: "0" for dimension in ranking.DIMENSIONS}

        with self.assertRaisesRegex(ranking.InputError, "positive"):
            ranking.calculate(data)

    def test_duplicate_link_id_is_rejected(self) -> None:
        data = ranking_input()
        links = data["links"]
        assert isinstance(links, list)
        links.append(link("L1", "Duplicate", "4"))

        with self.assertRaisesRegex(ranking.InputError, "duplicate id"):
            ranking.calculate(data)


class OutputTests(unittest.TestCase):
    def test_markdown_output_keeps_confidence_separate(self) -> None:
        output = ranking.render_markdown(
            ranking.calculate(ranking_input()), precision=1
        )

        self.assertIn("# 产业环节潜力排序", output)
        self.assertIn("置信度", output)
        self.assertIn("不", output)
        self.assertIn("单独显示", output)


if __name__ == "__main__":
    unittest.main()
