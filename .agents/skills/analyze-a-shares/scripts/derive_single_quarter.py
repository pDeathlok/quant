#!/usr/bin/env python3
"""Derive standalone quarters from cumulative A-share filing values."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """Raised when the input contract is invalid."""


PERIOD_KEYS = ("q1", "h1", "q3_ytd", "fy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Q1, H1, Q3 year-to-date and full-year cumulative flow "
            "values into Q1-Q4 standalone values."
        )
    )
    parser.add_argument(
        "input",
        help="Input JSON path, or '-' to read JSON from standard input.",
    )
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
        help="Displayed decimal places, from 0 to 8 (default: 2).",
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


def derive_quarters(
    data: dict[str, Any],
) -> tuple[dict[str, dict[str, Decimal | None]], list[str]]:
    year = data.get("year")
    if isinstance(year, bool) or not isinstance(year, (int, str)) or not str(year).strip():
        raise InputError("'year' must be a non-empty year label or integer.")
    unit = data.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        raise InputError("'unit' must be a non-empty string such as 'CNY million'.")

    metrics = data.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise InputError("'metrics' must be a non-empty object.")

    results: dict[str, dict[str, Decimal | None]] = {}
    warnings: list[str] = []

    for metric_name, cumulative in metrics.items():
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise InputError("Every metric name must be a non-empty string.")
        if not isinstance(cumulative, dict):
            raise InputError(f"metrics.{metric_name} must be an object.")
        if not cumulative:
            raise InputError(f"metrics.{metric_name} must contain at least one cumulative value.")
        unknown = set(cumulative) - set(PERIOD_KEYS)
        if unknown:
            raise InputError(
                f"metrics.{metric_name} has unsupported keys: {sorted(unknown)}. "
                f"Allowed keys: {list(PERIOD_KEYS)}."
            )
        values = {
            key: as_decimal(value, f"metrics.{metric_name}.{key}")
            for key, value in cumulative.items()
        }

        q1 = values.get("q1")
        q2 = (
            values["h1"] - values["q1"]
            if "h1" in values and "q1" in values
            else None
        )
        q3 = (
            values["q3_ytd"] - values["h1"]
            if "q3_ytd" in values and "h1" in values
            else None
        )
        q4 = (
            values["fy"] - values["q3_ytd"]
            if "fy" in values and "q3_ytd" in values
            else None
        )
        results[metric_name] = {"q1": q1, "q2": q2, "q3": q3, "q4": q4}

        if q1 is None:
            warnings.append(f"{metric_name}: cannot show Q1; missing cumulative field q1.")

        required_pairs = {
            "Q2": ("q1", "h1"),
            "Q3": ("h1", "q3_ytd"),
            "Q4": ("q3_ytd", "fy"),
        }
        for quarter, pair in required_pairs.items():
            missing = [key for key in pair if key not in values]
            if missing:
                warnings.append(
                    f"{metric_name}: cannot derive {quarter}; missing cumulative "
                    f"field(s) {', '.join(missing)}."
                )

        for quarter, value in results[metric_name].items():
            if value is not None and value < 0:
                warnings.append(
                    f"{metric_name}: derived {quarter.upper()} is negative ({value}); "
                    "this may be valid for profit/cash flow, but verify source periods and restatements."
                )

    return results, warnings


def decimal_as_json(value: Decimal | None) -> int | float | None:
    if value is None:
        return None
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def render_json(
    data: dict[str, Any],
    results: dict[str, dict[str, Decimal | None]],
    warnings: list[str],
) -> str:
    payload = {
        "year": data.get("year"),
        "unit": data.get("unit"),
        "metrics": {
            metric: {quarter: decimal_as_json(value) for quarter, value in quarters.items()}
            for metric, quarters in results.items()
        },
        "warnings": warnings,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_decimal(value: Decimal | None, precision: int) -> str:
    if value is None:
        return "—"
    quantum = Decimal(1).scaleb(-precision)
    return f"{value.quantize(quantum):,.{precision}f}"


def render_markdown(
    data: dict[str, Any],
    results: dict[str, dict[str, Decimal | None]],
    warnings: list[str],
    precision: int,
) -> str:
    year = data.get("year", "未提供")
    unit = data.get("unit", "未提供")
    lines = [
        "# 单季度还原结果",
        "",
        f"- 会计年度：{year}",
        f"- 单位：{unit}",
        "- 公式：Q2=H1-Q1；Q3=Q3累计-H1；Q4=FY-Q3累计",
        "",
        "| 指标 | Q1 | Q2 | Q3 | Q4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, quarters in results.items():
        lines.append(
            "| "
            + " | ".join(
                [metric]
                + [format_decimal(quarters[key], precision) for key in ("q1", "q2", "q3", "q4")]
            )
            + " |"
        )
    if warnings:
        lines.extend(["", "## 校验提示", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not 0 <= args.precision <= 8:
        print("error: --precision must be between 0 and 8.", file=sys.stderr)
        return 2
    try:
        data = load_json(args.input)
        results, warnings = derive_quarters(data)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(render_json(data, results, warnings))
    else:
        print(render_markdown(data, results, warnings, args.precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
