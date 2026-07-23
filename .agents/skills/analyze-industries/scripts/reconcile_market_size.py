#!/usr/bin/env python3
"""Validate and reconcile unit-compatible industry market-size estimates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """Raised when the market-size input contract is invalid."""


@dataclass(frozen=True)
class Estimate:
    """One independently sourced or calculated market-size estimate."""

    label: str
    method: str
    value: Decimal
    weight: Decimal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate same-basis market-size estimates and recalculate ranges, "
            "weighted centers, forecast CAGR consistency, and optional China "
            "production/import/export demand balance. Uses only the Python standard library."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input JSON root:\n"
            "  {\"markets\": [{...}]}\n\n"
            "Required per market:\n"
            "  id, name, geography, base_year, unit, basis, estimates.\n"
            "Each estimate requires label, method, value; weight defaults to 1.\n\n"
            "Optional forecast:\n"
            "  end_year and at least one of end_value or cagr (decimal, e.g. 0.10).\n"
            "Optional flow_balance (same unit as market):\n"
            "  production, imports, exports; inventory_increase defaults to 0;\n"
            "  stated_demand is optional.\n\n"
            "Important: the script cannot determine whether sources are independent or "
            "whether their definitions are genuinely compatible. Confirm those judgments first."
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
        default=2,
        help="Displayed decimal places from 0 to 6 (default: 2).",
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


def require_int(mapping: dict[str, Any], key: str, path: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{path}.{key} must be an integer.")
    return value


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


def require_decimal(mapping: dict[str, Any], key: str, path: str) -> Decimal:
    if key not in mapping:
        raise InputError(f"{path}.{key} is required.")
    return as_decimal(mapping[key], f"{path}.{key}")


def require_nonnegative(mapping: dict[str, Any], key: str, path: str) -> Decimal:
    number = require_decimal(mapping, key, path)
    if number < 0:
        raise InputError(f"{path}.{key} must be non-negative.")
    return number


def optional_decimal(
    mapping: dict[str, Any], key: str, path: str
) -> Decimal | None:
    if key not in mapping:
        return None
    return as_decimal(mapping[key], f"{path}.{key}")


def decimal_median(values: list[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def weighted_median(estimates: list[Estimate]) -> Decimal:
    ordered = sorted(estimates, key=lambda item: item.value)
    total_weight = sum((item.weight for item in ordered), Decimal(0))
    midpoint = total_weight / Decimal(2)
    cumulative = Decimal(0)
    for index, estimate in enumerate(ordered):
        cumulative += estimate.weight
        if cumulative == midpoint and index + 1 < len(ordered):
            return (estimate.value + ordered[index + 1].value) / Decimal(2)
        if cumulative >= midpoint:
            return estimate.value
    raise AssertionError("A positive weighted estimate set must have a median.")


def alignment_label(count: int, relative_spread: Decimal | None) -> str:
    if count == 1:
        return "single_estimate"
    if relative_spread is None:
        return "undefined_zero_median"
    if relative_spread <= Decimal("0.15"):
        return "aligned"
    if relative_spread <= Decimal("0.35"):
        return "mixed"
    return "divergent"


def parse_estimates(raw: Any, path: str) -> list[Estimate]:
    if not isinstance(raw, list) or not raw:
        raise InputError(f"{path} must be a non-empty array.")
    estimates: list[Estimate] = []
    labels: set[str] = set()
    for index, item in enumerate(raw):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            raise InputError(f"{item_path} must be an object.")
        label = require_text(item, "label", item_path)
        if label in labels:
            raise InputError(f"{path} contains duplicate label {label!r}.")
        labels.add(label)
        method = require_text(item, "method", item_path)
        value = require_nonnegative(item, "value", item_path)
        weight = (
            as_decimal(item["weight"], f"{item_path}.weight")
            if "weight" in item
            else Decimal(1)
        )
        if weight <= 0:
            raise InputError(f"{item_path}.weight must be greater than zero.")
        estimates.append(
            Estimate(label=label, method=method, value=value, weight=weight)
        )
    return estimates


def summarize_estimates(estimates: list[Estimate]) -> dict[str, Any]:
    values = [estimate.value for estimate in estimates]
    minimum = min(values)
    maximum = max(values)
    median = decimal_median(values)
    total_weight = sum((estimate.weight for estimate in estimates), Decimal(0))
    weighted_mean = (
        sum(
            (estimate.value * estimate.weight for estimate in estimates),
            Decimal(0),
        )
        / total_weight
    )
    center = weighted_median(estimates)
    relative_spread = (
        (maximum - minimum) / median
        if median != 0
        else (Decimal(0) if maximum == minimum else None)
    )
    return {
        "count": len(estimates),
        "minimum": minimum,
        "maximum": maximum,
        "median": median,
        "weighted_mean": weighted_mean,
        "weighted_median": center,
        "relative_spread": relative_spread,
        "alignment": alignment_label(len(estimates), relative_spread),
        "estimates": [
            {
                "label": estimate.label,
                "method": estimate.method,
                "value": estimate.value,
                "weight": estimate.weight,
            }
            for estimate in estimates
        ],
    }


def calculate_forecast(
    raw: Any, *, base_year: int, base_value: Decimal, path: str
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InputError(f"{path} must be an object.")
    end_year = require_int(raw, "end_year", path)
    if end_year <= base_year:
        raise InputError(f"{path}.end_year must be later than base_year.")
    years = end_year - base_year
    end_value = optional_decimal(raw, "end_value", path)
    cagr = optional_decimal(raw, "cagr", path)
    if end_value is None and cagr is None:
        raise InputError(f"{path} requires end_value, cagr, or both.")
    if end_value is not None and end_value < 0:
        raise InputError(f"{path}.end_value must be non-negative.")
    if cagr is not None and cagr <= Decimal(-1):
        raise InputError(f"{path}.cagr must be greater than -1.")
    if base_value <= 0:
        raise InputError(
            f"{path} cannot calculate CAGR from a non-positive weighted median."
        )

    implied_end = (
        base_value * (Decimal(1) + cagr) ** years if cagr is not None else None
    )
    implied_cagr = (
        (end_value / base_value) ** (Decimal(1) / Decimal(years)) - Decimal(1)
        if end_value is not None and end_value > 0
        else (Decimal(-1) if end_value == 0 else None)
    )

    gap_ratio: Decimal | None = None
    consistency = "derived"
    if end_value is not None and implied_end is not None:
        denominator = max(abs(end_value), abs(implied_end))
        gap_ratio = (
            abs(end_value - implied_end) / denominator
            if denominator != 0
            else Decimal(0)
        )
        consistency = "consistent" if gap_ratio <= Decimal("0.01") else "inconsistent"

    return {
        "end_year": end_year,
        "years": years,
        "stated_end_value": end_value,
        "stated_cagr": cagr,
        "implied_end_from_cagr": implied_end,
        "implied_cagr_from_end": implied_cagr,
        "gap_ratio": gap_ratio,
        "consistency": consistency,
    }


def calculate_flow_balance(raw: Any, path: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InputError(f"{path} must be an object.")
    production = require_nonnegative(raw, "production", path)
    imports = require_nonnegative(raw, "imports", path)
    exports = require_nonnegative(raw, "exports", path)
    inventory_increase = (
        as_decimal(raw["inventory_increase"], f"{path}.inventory_increase")
        if "inventory_increase" in raw
        else Decimal(0)
    )
    stated_demand = optional_decimal(raw, "stated_demand", path)
    if stated_demand is not None and stated_demand < 0:
        raise InputError(f"{path}.stated_demand must be non-negative.")
    implied_demand = production + imports - exports - inventory_increase
    if implied_demand < 0:
        raise InputError(
            f"{path} implies negative demand; check units, signs, and trade data."
        )
    demand_gap = (
        stated_demand - implied_demand if stated_demand is not None else None
    )
    gap_ratio = (
        abs(demand_gap) / max(stated_demand, implied_demand)
        if demand_gap is not None and max(stated_demand, implied_demand) != 0
        else (Decimal(0) if demand_gap == 0 else None)
    )
    return {
        "production": production,
        "imports": imports,
        "exports": exports,
        "inventory_increase": inventory_increase,
        "inventory_assumed_zero": "inventory_increase" not in raw,
        "implied_demand": implied_demand,
        "stated_demand": stated_demand,
        "demand_gap": demand_gap,
        "gap_ratio": gap_ratio,
    }


def calculate_market(raw: Any, index: int) -> dict[str, Any]:
    path = f"root.markets[{index}]"
    if not isinstance(raw, dict):
        raise InputError(f"{path} must be an object.")
    market_id = require_text(raw, "id", path)
    name = require_text(raw, "name", path)
    geography = require_text(raw, "geography", path)
    base_year = require_int(raw, "base_year", path)
    unit = require_text(raw, "unit", path)
    basis = require_text(raw, "basis", path)
    estimates = parse_estimates(raw.get("estimates"), f"{path}.estimates")
    estimate_summary = summarize_estimates(estimates)
    base_value = estimate_summary["weighted_median"]
    assert isinstance(base_value, Decimal)
    forecast = calculate_forecast(
        raw.get("forecast"),
        base_year=base_year,
        base_value=base_value,
        path=f"{path}.forecast",
    )
    flow_balance = calculate_flow_balance(
        raw.get("flow_balance"), f"{path}.flow_balance"
    )
    return {
        "id": market_id,
        "name": name,
        "geography": geography,
        "base_year": base_year,
        "unit": unit,
        "basis": basis,
        "estimate_summary": estimate_summary,
        "forecast": forecast,
        "flow_balance": flow_balance,
    }


def calculate(data: dict[str, Any]) -> dict[str, Any]:
    markets = data.get("markets")
    if not isinstance(markets, list) or not markets:
        raise InputError("root.markets must be a non-empty array.")
    results = [calculate_market(market, index) for index, market in enumerate(markets)]
    ids = [result["id"] for result in results]
    if len(ids) != len(set(ids)):
        raise InputError("root.markets contains duplicate market ids.")
    return {
        "markets": results,
        "method_note": (
            "加权中心只是算术工作底稿，不代表真实市场规模。来源是否独立、"
            "口径是否兼容仍需分析者复核。"
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


def format_number(value: Decimal | None, precision: int) -> str:
    if value is None:
        return "—"
    quantizer = Decimal(1).scaleb(-precision)
    return f"{value.quantize(quantizer):f}"


def format_percent(value: Decimal | None, precision: int = 1) -> str:
    if value is None:
        return "—"
    return f"{format_number(value * Decimal(100), precision)}%"


def render_markdown(result: dict[str, Any], precision: int) -> str:
    sections: list[str] = ["# 市场规模复核"]
    for market in result["markets"]:
        summary = market["estimate_summary"]
        sections.extend(
            [
                "",
                f"## {market['id']} {market['name']}",
                "",
                (
                    f"- 口径：{market['geography']}｜{market['base_year']}｜"
                    f"{market['basis']}｜单位 {market['unit']}"
                ),
                f"- 收敛状态：`{summary['alignment']}`",
                "",
                "| 估算 | 方法 | 数值 | 权重 |",
                "|---|---|---:|---:|",
            ]
        )
        for estimate in summary["estimates"]:
            sections.append(
                "| {label} | {method} | {value} | {weight} |".format(
                    label=estimate["label"],
                    method=estimate["method"],
                    value=format_number(estimate["value"], precision),
                    weight=format_number(estimate["weight"], 2),
                )
            )
        sections.extend(
            [
                "",
                "| 最小 | 最大 | 中位数 | 加权均值 | 加权中位数 | 相对分歧 |",
                "|---:|---:|---:|---:|---:|---:|",
                "| {minimum} | {maximum} | {median} | {weighted_mean} | "
                "{weighted_median} | {relative_spread} |".format(
                    minimum=format_number(summary["minimum"], precision),
                    maximum=format_number(summary["maximum"], precision),
                    median=format_number(summary["median"], precision),
                    weighted_mean=format_number(summary["weighted_mean"], precision),
                    weighted_median=format_number(
                        summary["weighted_median"], precision
                    ),
                    relative_spread=format_percent(summary["relative_spread"]),
                ),
            ]
        )
        forecast = market["forecast"]
        if forecast is not None:
            sections.extend(
                [
                    "",
                    "### 预测一致性",
                    "",
                    "| 终点年 | 陈述终值 | 陈述CAGR | CAGR隐含终值 | 终值隐含CAGR | 差异 | 状态 |",
                    "|---:|---:|---:|---:|---:|---:|---|",
                    "| {end_year} | {end_value} | {cagr} | {implied_end} | "
                    "{implied_cagr} | {gap} | {status} |".format(
                        end_year=forecast["end_year"],
                        end_value=format_number(
                            forecast["stated_end_value"], precision
                        ),
                        cagr=format_percent(forecast["stated_cagr"]),
                        implied_end=format_number(
                            forecast["implied_end_from_cagr"], precision
                        ),
                        implied_cagr=format_percent(
                            forecast["implied_cagr_from_end"]
                        ),
                        gap=format_percent(forecast["gap_ratio"]),
                        status=forecast["consistency"],
                    ),
                ]
            )
        flow = market["flow_balance"]
        if flow is not None:
            inventory_note = "（缺省按0）" if flow["inventory_assumed_zero"] else ""
            sections.extend(
                [
                    "",
                    "### 供需平衡",
                    "",
                    "| 生产 | 进口 | 出口 | 库存增加 | 隐含需求 | 陈述需求 | 差额 | 相对差异 |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|",
                    "| {production} | {imports} | {exports} | {inventory}{note} | "
                    "{implied} | {stated} | {gap} | {ratio} |".format(
                        production=format_number(flow["production"], precision),
                        imports=format_number(flow["imports"], precision),
                        exports=format_number(flow["exports"], precision),
                        inventory=format_number(
                            flow["inventory_increase"], precision
                        ),
                        note=inventory_note,
                        implied=format_number(flow["implied_demand"], precision),
                        stated=format_number(flow["stated_demand"], precision),
                        gap=format_number(flow["demand_gap"], precision),
                        ratio=format_percent(flow["gap_ratio"]),
                    ),
                ]
            )
    sections.extend(["", f"> {result['method_note']}"])
    return "\n".join(sections)


def main() -> int:
    args = parse_args()
    if not 0 <= args.precision <= 6:
        print("error: --precision must be between 0 and 6.", file=sys.stderr)
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
