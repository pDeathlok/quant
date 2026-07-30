#!/usr/bin/env python3
"""Recalculate transparent five-dimension potential scores for industry links."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """Raised when the potential-ranking input contract is invalid."""


DIMENSIONS = (
    "market_space",
    "profit_pool",
    "structural_quality",
    "china_opportunity",
    "realization_certainty",
)
DIMENSION_LABELS = {
    "market_space": "市场空间",
    "profit_pool": "利润池",
    "structural_quality": "结构质量",
    "china_opportunity": "中国机会",
    "realization_certainty": "兑现确定性",
}
DEFAULT_WEIGHTS = {
    "market_space": Decimal("0.25"),
    "profit_pool": Decimal("0.25"),
    "structural_quality": Decimal("0.20"),
    "china_opportunity": Decimal("0.20"),
    "realization_certainty": Decimal("0.10"),
}


@dataclass(frozen=True)
class LinkScore:
    """Validated score inputs for one industry-chain link."""

    link_id: str
    name: str
    scores: dict[str, Decimal]
    confidence: Decimal
    evidence_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and recalculate five-dimension industry-link potential scores. "
            "Scores remain analyst judgments; this tool only applies explicit weights "
            "and produces a consistent tier and ranking. Uses only the Python standard library."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Required JSON root:\n"
            "  perspective, links.\n"
            "Optional weights: all five dimensions; defaults to listed-equity weights.\n\n"
            "Each link requires:\n"
            "  id, name, scores (all five dimensions, each 1-5),\n"
            "  confidence (0-1), evidence_note.\n\n"
            "Dimensions:\n"
            "  market_space, profit_pool, structural_quality,\n"
            "  china_opportunity, realization_certainty."
        ),
    )
    parser.add_argument("input", help="Input JSON path, or '-' for standard input.")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="Displayed score decimal places from 0 to 3 (default: 1).",
    )
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    try:
        raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot read valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError("The JSON root must be an object.")
    return data


def require_text(mapping: dict[str, Any], key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{path}.{key} must be a non-empty string.")
    return value.strip()


def as_decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise InputError(f"{field} must be a finite number, not {value!r}.")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise InputError(f"{field} must be a finite number, not {value!r}.") from exc
    if not number.is_finite():
        raise InputError(f"{field} must be finite.")
    return number


def parse_weights(raw: Any) -> dict[str, Decimal]:
    if raw is None:
        return dict(DEFAULT_WEIGHTS)
    if not isinstance(raw, dict):
        raise InputError("root.weights must be an object.")
    unknown = set(raw) - set(DIMENSIONS)
    missing = set(DIMENSIONS) - set(raw)
    if unknown or missing:
        raise InputError(
            "root.weights must contain exactly the five supported dimensions; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    weights = {
        dimension: as_decimal(raw[dimension], f"root.weights.{dimension}")
        for dimension in DIMENSIONS
    }
    if any(weight < 0 for weight in weights.values()):
        raise InputError("root.weights values must be non-negative.")
    total = sum(weights.values(), Decimal(0))
    if total <= 0:
        raise InputError("root.weights must include at least one positive value.")
    return {dimension: weight / total for dimension, weight in weights.items()}


def parse_links(raw: Any) -> list[LinkScore]:
    if not isinstance(raw, list) or not raw:
        raise InputError("root.links must be a non-empty array.")
    links: list[LinkScore] = []
    ids: set[str] = set()
    for index, item in enumerate(raw):
        path = f"root.links[{index}]"
        if not isinstance(item, dict):
            raise InputError(f"{path} must be an object.")
        link_id = require_text(item, "id", path)
        if link_id in ids:
            raise InputError(f"root.links contains duplicate id {link_id!r}.")
        ids.add(link_id)
        name = require_text(item, "name", path)
        raw_scores = item.get("scores")
        if not isinstance(raw_scores, dict):
            raise InputError(f"{path}.scores must be an object.")
        unknown = set(raw_scores) - set(DIMENSIONS)
        missing = set(DIMENSIONS) - set(raw_scores)
        if unknown or missing:
            raise InputError(
                f"{path}.scores must contain exactly the five supported dimensions; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        scores = {
            dimension: as_decimal(
                raw_scores[dimension], f"{path}.scores.{dimension}"
            )
            for dimension in DIMENSIONS
        }
        for dimension, score in scores.items():
            if score < 1 or score > 5:
                raise InputError(
                    f"{path}.scores.{dimension} must be between 1 and 5."
                )
        if "confidence" not in item:
            raise InputError(f"{path}.confidence is required.")
        confidence = as_decimal(item["confidence"], f"{path}.confidence")
        if confidence < 0 or confidence > 1:
            raise InputError(f"{path}.confidence must be between 0 and 1.")
        evidence_note = require_text(item, "evidence_note", path)
        links.append(
            LinkScore(
                link_id=link_id,
                name=name,
                scores=scores,
                confidence=confidence,
                evidence_note=evidence_note,
            )
        )
    return links


def tier_for(score: Decimal) -> str:
    if score >= 80:
        return "高"
    if score >= 65:
        return "中高"
    if score >= 45:
        return "中"
    if score >= 30:
        return "中低"
    return "低"


def confidence_label(confidence: Decimal) -> str:
    if confidence >= Decimal("0.8"):
        return "高"
    if confidence >= Decimal("0.5"):
        return "中"
    return "低"


def calculate(data: dict[str, Any]) -> dict[str, Any]:
    perspective = require_text(data, "perspective", "root")
    weights = parse_weights(data.get("weights"))
    links = parse_links(data.get("links"))
    ranked: list[dict[str, Any]] = []
    for link in links:
        weighted_score = sum(
            (link.scores[dimension] * weights[dimension] for dimension in DIMENSIONS),
            Decimal(0),
        )
        normalized_score = (weighted_score - Decimal(1)) / Decimal(4) * Decimal(100)
        ranked.append(
            {
                "id": link.link_id,
                "name": link.name,
                "scores": link.scores,
                "weighted_score_1_to_5": weighted_score,
                "potential_score": normalized_score,
                "tier": tier_for(normalized_score),
                "confidence": link.confidence,
                "confidence_label": confidence_label(link.confidence),
                "evidence_note": link.evidence_note,
            }
        )
    ranked.sort(
        key=lambda item: (
            -item["potential_score"],
            -item["confidence"],
            item["id"],
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return {
        "perspective": perspective,
        "weights": weights,
        "links": ranked,
        "method_note": (
            "潜力分是透明的分析判断输入，不是可观测事实或证券回报预测。"
            "置信度单独显示，不会暗中乘入潜力分。"
        ),
    }


def serialize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    return value


def format_number(value: Decimal, precision: int) -> str:
    quantizer = Decimal(1).scaleb(-precision)
    return f"{value.quantize(quantizer):f}"


def format_percent(value: Decimal, precision: int = 0) -> str:
    return f"{format_number(value * Decimal(100), precision)}%"


def render_markdown(result: dict[str, Any], precision: int) -> str:
    sections = [
        "# 产业环节潜力排序",
        "",
        f"- 视角：`{result['perspective']}`",
        "- 权重："
        + "；".join(
            f"{DIMENSION_LABELS[dimension]} {format_percent(weight)}"
            for dimension, weight in result["weights"].items()
        ),
        "",
        "| 排名 | 节点 | 市场空间 | 利润池 | 结构质量 | 中国机会 | "
        "兑现确定性 | 潜力分 | 档位 | 置信度 | 证据说明 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in result["links"]:
        sections.append(
            "| {rank} | {id} {name} | {market_space} | {profit_pool} | "
            "{structural_quality} | {china_opportunity} | "
            "{realization_certainty} | {potential_score} | {tier} | "
            "{confidence} | {note} |".format(
                rank=item["rank"],
                id=item["id"],
                name=item["name"],
                market_space=format_number(item["scores"]["market_space"], 1),
                profit_pool=format_number(item["scores"]["profit_pool"], 1),
                structural_quality=format_number(
                    item["scores"]["structural_quality"], 1
                ),
                china_opportunity=format_number(
                    item["scores"]["china_opportunity"], 1
                ),
                realization_certainty=format_number(
                    item["scores"]["realization_certainty"], 1
                ),
                potential_score=format_number(item["potential_score"], precision),
                tier=item["tier"],
                confidence=(
                    f"{item['confidence_label']} "
                    f"({format_number(item['confidence'], 2)})"
                ),
                note=item["evidence_note"].replace("|", "/"),
            )
        )
    sections.extend(["", f"> {result['method_note']}"])
    return "\n".join(sections)


def main() -> int:
    args = parse_args()
    if not 0 <= args.precision <= 3:
        print("error: --precision must be between 0 and 3.", file=sys.stderr)
        return 2
    try:
        result = calculate(load_json(args.input))
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(serialize(result), ensure_ascii=False, indent=2))
    else:
        print(render_markdown(result, args.precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
