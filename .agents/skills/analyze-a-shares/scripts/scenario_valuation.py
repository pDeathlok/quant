#!/usr/bin/env python3
"""Validate and recalculate bear/base/bull A-share price scenarios."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """Raised when the valuation input contract is invalid."""


SCENARIO_ORDER = ("bear", "base", "bull")
SCENARIO_LABELS = {"bear": "悲观", "base": "中性", "bull": "乐观"}
METHOD_LABELS = {
    "per_share_multiple": "每股指标×倍数",
    "enterprise_multiple": "企业价值倍数",
    "equity_value": "股权总价值/股数",
    "target_price": "外部模型目标价（仅审计契约）",
}
EXTERNAL_MODEL_TYPES = {
    "dcf",
    "sotp",
    "residual_income",
    "risk_adjusted_npv",
    "other",
}
CANONICAL_CURRENCY = "CNY"
CURRENCY_ALIASES = {"CNY": CANONICAL_CURRENCY, "RMB": CANONICAL_CURRENCY}
TOTAL_VALUE_UNIT_SCALE = {
    "CNY": Decimal(1),
    "CNY_million": Decimal(1_000_000),
}
SHARE_UNIT_SCALE = {
    "shares": Decimal(1),
    "million_shares": Decimal(1_000_000),
}
COMPATIBLE_TOTAL_SHARE_UNITS = {
    ("CNY", "shares"),
    ("CNY_million", "million_shares"),
}
PER_SHARE_UNIT = "CNY_per_share"
TICKER_PATTERN = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class RootContext:
    current_price: Decimal
    currency: str
    as_of_date: date
    analysis_cutoff: datetime
    price_as_of: datetime
    target_date: date
    scenarios_exhaustive: bool


@dataclass(frozen=True)
class ScenarioResult:
    key: str
    method: str
    target_price: Decimal
    price_return: Decimal
    dividend_per_share: Decimal | None
    total_return: Decimal | None
    probability: Decimal | None
    calculation: str
    metric_period: str
    bridge_as_of: date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and recalculate bear/base/bull A-share target prices from "
            "explicit, unit-aware valuation inputs. Uses only the Python standard library."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Required root fields:\n"
            "  company, ticker (000001.SZ), as_of_date and target_date (YYYY-MM-DD),\n"
            "  analysis_cutoff and price_as_of (ISO 8601 with timezone),\n"
            "  current_price, current_price_source, price_basis='unadjusted',\n"
            "  currency='CNY' or 'RMB',\n"
            "  scenarios_exhaustive, scenarios.bear/base/bull.\n\n"
            "Every scenario requires method, metric_period and bridge_as_of (YYYY-MM-DD).\n\n"
            "Method-specific fields:\n"
            "  per_share_multiple: metric_name,\n"
            "    metric_unit='CNY_per_share', metric_per_share, multiple, multiple_basis.\n"
            "  enterprise_multiple: metric_name, metric_total, multiple, multiple_basis,\n"
            "    total_value_unit, debt, cash, minority_interest,\n"
            "    preferred_equity, non_operating_assets, diluted_shares, shares_unit,\n"
            "    shares_period. All totals use the one declared total_value_unit.\n"
            "  equity_value: equity_value, total_value_unit, diluted_shares, shares_unit,\n"
            "    shares_period, method_note.\n"
            "  target_price: target_price, target_price_unit='CNY_per_share',\n"
            "    method_note, model_type, model_reference, independent_check.\n"
            "    This method validates the audit trail but does not recalculate the model.\n\n"
            "Allowed total/share unit pairs: (CNY, shares) or\n"
            "  (CNY_million, million_shares). Probabilities require all three values,\n"
            "  scenarios_exhaustive=true, a non-empty root probability_basis, and sum=1."
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
        help="Displayed decimal places, from 0 to 4 (default: 2).",
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


def require_bool(mapping: dict[str, Any], key: str, path: str) -> bool:
    if key not in mapping or not isinstance(mapping[key], bool):
        raise InputError(f"{path}.{key} must be true or false.")
    return mapping[key]


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


def require_positive(mapping: dict[str, Any], key: str, path: str) -> Decimal:
    number = require_decimal(mapping, key, path)
    if number <= 0:
        raise InputError(f"{path}.{key} must be greater than zero.")
    return number


def require_enum(
    mapping: dict[str, Any], key: str, path: str, allowed: set[str]
) -> str:
    value = require_text(mapping, key, path)
    if value not in allowed:
        raise InputError(f"{path}.{key} must be one of {sorted(allowed)}, not {value!r}.")
    return value


def parse_iso_timestamp(value: str, field: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise InputError(f"{field} must be an ISO 8601 timestamp with timezone.") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise InputError(f"{field} must include an explicit UTC offset, such as +08:00.")
    return timestamp


def parse_iso_date(value: str, field: str) -> date:
    if not ISO_DATE_PATTERN.fullmatch(value):
        raise InputError(f"{field} must use YYYY-MM-DD.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise InputError(f"{field} must use YYYY-MM-DD.") from exc


def require_compatible_total_share_units(
    mapping: dict[str, Any], path: str
) -> tuple[str, str]:
    total_value_unit = require_enum(
        mapping, "total_value_unit", path, set(TOTAL_VALUE_UNIT_SCALE)
    )
    shares_unit = require_enum(mapping, "shares_unit", path, set(SHARE_UNIT_SCALE))
    if (total_value_unit, shares_unit) not in COMPATIBLE_TOTAL_SHARE_UNITS:
        raise InputError(
            f"{path}.total_value_unit and {path}.shares_unit must be a compatible "
            "pair: (CNY, shares) or (CNY_million, million_shares)."
        )
    return total_value_unit, shares_unit


def validate_root(data: dict[str, Any]) -> tuple[RootContext, list[str]]:
    require_text(data, "company", "root")
    ticker = require_text(data, "ticker", "root")
    if not TICKER_PATTERN.fullmatch(ticker):
        raise InputError("root.ticker must look like 600519.SH, 000001.SZ or 920000.BJ.")
    input_currency = require_enum(data, "currency", "root", set(CURRENCY_ALIASES))
    currency = CURRENCY_ALIASES[input_currency]
    require_enum(data, "price_basis", "root", {"unadjusted"})
    require_text(data, "current_price_source", "root")
    current_price = require_positive(data, "current_price", "root")

    as_of_date = parse_iso_date(
        require_text(data, "as_of_date", "root"), "root.as_of_date"
    )
    analysis_cutoff = parse_iso_timestamp(
        require_text(data, "analysis_cutoff", "root"), "root.analysis_cutoff"
    )
    if as_of_date != analysis_cutoff.date():
        raise InputError(
            "root.as_of_date must equal the local calendar date of "
            "root.analysis_cutoff."
        )
    price_as_of = parse_iso_timestamp(
        require_text(data, "price_as_of", "root"), "root.price_as_of"
    )
    if price_as_of.astimezone(timezone.utc) > analysis_cutoff.astimezone(timezone.utc):
        raise InputError("root.price_as_of cannot be later than root.analysis_cutoff.")

    target_date = parse_iso_date(
        require_text(data, "target_date", "root"), "root.target_date"
    )
    if target_date <= as_of_date:
        raise InputError("root.target_date must be later than root.as_of_date.")

    scenarios_exhaustive = require_bool(data, "scenarios_exhaustive", "root")
    warnings: list[str] = []
    price_age_days = (
        analysis_cutoff.astimezone(timezone.utc) - price_as_of.astimezone(timezone.utc)
    ).total_seconds() / 86_400
    if price_age_days > 7:
        warnings.append(
            f"基准价格比分析截止时点早 {price_age_days:.1f} 天；确认停牌、节假日或数据缺失。"
        )

    return (
        RootContext(
            current_price=current_price,
            currency=currency,
            as_of_date=as_of_date,
            analysis_cutoff=analysis_cutoff,
            price_as_of=price_as_of,
            target_date=target_date,
            scenarios_exhaustive=scenarios_exhaustive,
        ),
        warnings,
    )


def calculate_target(
    scenario: dict[str, Any], path: str
) -> tuple[str, Decimal, str, str, date]:
    """Validate one method contract and return its deterministic target price."""

    method = require_enum(scenario, "method", path, set(METHOD_LABELS))
    metric_period = require_text(scenario, "metric_period", path)
    bridge_as_of = parse_iso_date(
        require_text(scenario, "bridge_as_of", path), f"{path}.bridge_as_of"
    )

    if method == "per_share_multiple":
        metric_name = require_text(scenario, "metric_name", path)
        require_enum(scenario, "metric_unit", path, {PER_SHARE_UNIT})
        metric = require_nonnegative(scenario, "metric_per_share", path)
        multiple = require_nonnegative(scenario, "multiple", path)
        multiple_basis = require_text(scenario, "multiple_basis", path)
        target = metric * multiple
        calculation = (
            f"{metric_name} {metric} {PER_SHARE_UNIT} ({metric_period}) × {multiple} "
            f"[{multiple_basis}] = {target} CNY/share; bridge={bridge_as_of}"
        )

    elif method == "enterprise_multiple":
        metric_name = require_text(scenario, "metric_name", path)
        metric = require_nonnegative(scenario, "metric_total", path)
        multiple = require_nonnegative(scenario, "multiple", path)
        multiple_basis = require_text(scenario, "multiple_basis", path)
        total_value_unit, shares_unit = require_compatible_total_share_units(
            scenario, path
        )
        debt = require_nonnegative(scenario, "debt", path)
        cash = require_nonnegative(scenario, "cash", path)
        minority = require_nonnegative(scenario, "minority_interest", path)
        preferred = require_nonnegative(scenario, "preferred_equity", path)
        non_operating = require_nonnegative(scenario, "non_operating_assets", path)
        shares = require_positive(scenario, "diluted_shares", path)
        shares_period = require_text(scenario, "shares_period", path)

        value_scale = TOTAL_VALUE_UNIT_SCALE[total_value_unit]
        share_scale = SHARE_UNIT_SCALE[shares_unit]
        enterprise_value = metric * multiple
        common_equity_value = (
            enterprise_value
            - debt
            - minority
            - preferred
            + cash
            + non_operating
        )
        if common_equity_value < 0:
            raise InputError(
                f"{path} implies negative common equity value. Use an explicit "
                "distress/restructuring model instead of publishing a negative target."
            )
        target = (common_equity_value * value_scale) / (shares * share_scale)
        calculation = (
            f"EV={metric_name} {metric} {total_value_unit}×{multiple}="
            f"{enterprise_value} {total_value_unit}; common equity="
            f"{enterprise_value}-{debt}-{minority}-{preferred}+{cash}+{non_operating}="
            f"{common_equity_value} {total_value_unit}; target=("
            f"{common_equity_value}×{value_scale} CNY)/({shares}×{share_scale} shares)="
            f"{target} CNY/share; "
            f"metric={metric_period}, bridge={bridge_as_of}, shares={shares_period}, "
            f"multiple basis={multiple_basis}"
        )

    elif method == "equity_value":
        equity_value = require_nonnegative(scenario, "equity_value", path)
        total_value_unit, shares_unit = require_compatible_total_share_units(
            scenario, path
        )
        shares = require_positive(scenario, "diluted_shares", path)
        shares_period = require_text(scenario, "shares_period", path)
        method_note = require_text(scenario, "method_note", path)
        value_scale = TOTAL_VALUE_UNIT_SCALE[total_value_unit]
        share_scale = SHARE_UNIT_SCALE[shares_unit]
        target = (equity_value * value_scale) / (shares * share_scale)
        calculation = (
            f"equity value=({equity_value}×{value_scale} CNY)/("
            f"{shares}×{share_scale} shares)={target} CNY/share; "
            f"metric={metric_period}, bridge={bridge_as_of}, shares={shares_period}, "
            f"method={method_note}"
        )

    else:
        require_enum(scenario, "target_price_unit", path, {PER_SHARE_UNIT})
        target = require_nonnegative(scenario, "target_price", path)
        method_note = require_text(scenario, "method_note", path)
        model_type = require_enum(
            scenario, "model_type", path, EXTERNAL_MODEL_TYPES
        )
        model_reference = require_text(scenario, "model_reference", path)
        independent_check = require_text(scenario, "independent_check", path)
        calculation = (
            f"external {model_type} target {target} CNY/share (not recalculated); "
            f"metric={metric_period}, bridge={bridge_as_of}, method={method_note}, "
            f"model reference={model_reference}, independent check={independent_check}"
        )

    return method, target, calculation, metric_period, bridge_as_of


def calculate_scenarios(
    data: dict[str, Any], context: RootContext, warnings: list[str]
) -> list[ScenarioResult]:
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, dict):
        raise InputError("root.scenarios must be an object.")
    missing = set(SCENARIO_ORDER) - set(scenarios)
    extra = set(scenarios) - set(SCENARIO_ORDER)
    if missing or extra:
        raise InputError(
            "root.scenarios must contain exactly bear, base and bull; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )

    results: list[ScenarioResult] = []
    for key in SCENARIO_ORDER:
        scenario = scenarios[key]
        path = f"root.scenarios.{key}"
        if not isinstance(scenario, dict):
            raise InputError(f"{path} must be an object.")
        method, target, calculation, metric_period, bridge_as_of = calculate_target(
            scenario, path
        )
        price_return = target / context.current_price - Decimal(1)

        dividend: Decimal | None = None
        total_return: Decimal | None = None
        if "dividend_per_share" in scenario:
            require_enum(scenario, "dividend_unit", path, {PER_SHARE_UNIT})
            require_text(scenario, "dividend_period", path)
            dividend = require_nonnegative(scenario, "dividend_per_share", path)
            total_return = (target + dividend) / context.current_price - Decimal(1)
        else:
            warnings.append(
                f"{SCENARIO_LABELS[key]}情景未提供 dividend_per_share；不计算含息总回报。"
            )

        probability: Decimal | None = None
        if "probability" in scenario:
            probability = require_decimal(scenario, "probability", path)
            if not Decimal(0) <= probability <= Decimal(1):
                raise InputError(f"{path}.probability must be between 0 and 1.")

        results.append(
            ScenarioResult(
                key=key,
                method=method,
                target_price=target,
                price_return=price_return,
                dividend_per_share=dividend,
                total_return=total_return,
                probability=probability,
                calculation=calculation,
                metric_period=metric_period,
                bridge_as_of=bridge_as_of,
            )
        )

    prices = [result.target_price for result in results]
    if not prices[0] <= prices[1] <= prices[2]:
        raise InputError(
            "Scenario target prices must satisfy bear <= base <= bull. "
            f"Calculated values: {prices}."
        )

    probabilities = [result.probability for result in results]
    if any(value is not None for value in probabilities):
        if not all(value is not None for value in probabilities):
            raise InputError("Provide probability for all three scenarios or for none.")
        if not context.scenarios_exhaustive:
            raise InputError(
                "Probabilities require root.scenarios_exhaustive=true; representative "
                "scenarios must remain unweighted."
            )
        require_text(data, "probability_basis", "root")
        probability_sum = sum(
            (value for value in probabilities if value is not None), Decimal(0)
        )
        if probability_sum != Decimal(1):
            raise InputError(
                f"Scenario probabilities must sum to 1.0; got {probability_sum}."
            )
    elif context.scenarios_exhaustive:
        warnings.append(
            "scenarios_exhaustive=true 但未提供概率；不计算概率加权值。"
        )
    return results


def decimal_as_json(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def weighted_summary(
    results: list[ScenarioResult], current_price: Decimal
) -> dict[str, Decimal] | None:
    weighted_price = Decimal(0)
    for result in results:
        if result.probability is None:
            return None
        weighted_price += result.target_price * result.probability
    summary = {
        "target_price": weighted_price,
        "price_return": weighted_price / current_price - Decimal(1),
    }
    if all(result.dividend_per_share is not None for result in results):
        weighted_dividend = Decimal(0)
        for result in results:
            if result.dividend_per_share is None or result.probability is None:
                return summary
            weighted_dividend += result.dividend_per_share * result.probability
        summary["dividend_per_share"] = weighted_dividend
        summary["total_return"] = (
            weighted_price + weighted_dividend
        ) / current_price - Decimal(1)
    return summary


def render_json(
    data: dict[str, Any],
    context: RootContext,
    results: list[ScenarioResult],
    warnings: list[str],
) -> str:
    weighted = weighted_summary(results, context.current_price)
    horizon_days = (context.target_date - context.as_of_date).days
    payload = {
        "company": data["company"],
        "ticker": data["ticker"],
        "currency": context.currency,
        "as_of_date": context.as_of_date.isoformat(),
        "analysis_cutoff": data["analysis_cutoff"],
        "price_as_of": data["price_as_of"],
        "current_price": decimal_as_json(context.current_price),
        "current_price_source": data["current_price_source"],
        "price_basis": data["price_basis"],
        "target_date": data["target_date"],
        "horizon_days": horizon_days,
        "scenarios_exhaustive": context.scenarios_exhaustive,
        "probability_basis": data.get("probability_basis"),
        "scenarios": {
            result.key: {
                "label": SCENARIO_LABELS[result.key],
                "method": result.method,
                "metric_period": result.metric_period,
                "bridge_as_of": result.bridge_as_of.isoformat(),
                "target_price": decimal_as_json(result.target_price),
                "price_return": decimal_as_json(result.price_return),
                "dividend_per_share": decimal_as_json(result.dividend_per_share),
                "total_return": decimal_as_json(result.total_return),
                "probability": decimal_as_json(result.probability),
                "calculation": result.calculation,
                "external_model_audit": (
                    {
                        "model_type": data["scenarios"][result.key]["model_type"],
                        "model_reference": data["scenarios"][result.key]["model_reference"],
                        "independent_check": data["scenarios"][result.key][
                            "independent_check"
                        ],
                        "recalculated_by_script": False,
                    }
                    if result.method == "target_price"
                    else None
                ),
            }
            for result in results
        },
        "probability_weighted": (
            {key: decimal_as_json(value) for key, value in weighted.items()}
            if weighted is not None
            else None
        ),
        "warnings": warnings,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def quantized(value: Decimal, precision: int) -> str:
    quantum = Decimal(1).scaleb(-precision)
    return f"{value.quantize(quantum):,.{precision}f}"


def percent(value: Decimal | None) -> str:
    if value is None:
        return "—"
    return f"{value * Decimal(100):.1f}%"


def render_markdown(
    data: dict[str, Any],
    context: RootContext,
    results: list[ScenarioResult],
    warnings: list[str],
    precision: int,
) -> str:
    currency = context.currency
    horizon_days = (context.target_date - context.as_of_date).days
    lines = [
        f"# {data['company']}（{data['ticker']}）三情景估值复算",
        "",
        f"- 分析基准日：{context.as_of_date.isoformat()}",
        f"- 分析截止：{data['analysis_cutoff']}",
        f"- 基准价格：{currency} {quantized(context.current_price, precision)} "
        f"（{data['price_as_of']}；{data['current_price_source']}；未复权）",
        f"- 目标日期：{data['target_date']}（{horizon_days} 天）",
        "- 说明：目标价不含股息；只有显式提供股息时才计算含息总回报。",
        "",
        "| 情景 | 算术模型 | 指标期 | 桥接日 | 目标价 | 价格涨跌 | 股息/股 | 含息总回报 | 概率 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        dividend = (
            quantized(result.dividend_per_share, precision)
            if result.dividend_per_share is not None
            else "—"
        )
        lines.append(
            f"| {SCENARIO_LABELS[result.key]} | {METHOD_LABELS[result.method]} | "
            f"{result.metric_period} | {result.bridge_as_of.isoformat()} | {currency} "
            f"{quantized(result.target_price, precision)} | {percent(result.price_return)} | "
            f"{dividend} | {percent(result.total_return)} | {percent(result.probability)} |"
        )

    lines.extend(["", "## 计算链", ""])
    lines.extend(
        f"- {SCENARIO_LABELS[result.key]}：`{result.calculation}`" for result in results
    )

    weighted = weighted_summary(results, context.current_price)
    if weighted is not None:
        lines.extend(
            [
                "",
                "## 概率加权（输入已声明情景穷尽且提供概率依据）",
                "",
                f"- 概率依据：{data['probability_basis']}",
                f"- 加权目标价：{currency} {quantized(weighted['target_price'], precision)}",
                f"- 加权价格涨跌：{percent(weighted['price_return'])}",
            ]
        )
        if "total_return" in weighted:
            lines.extend(
                [
                    f"- 加权股息/股：{quantized(weighted['dividend_per_share'], precision)}",
                    f"- 加权含息总回报：{percent(weighted['total_return'])}",
                ]
            )
    if warnings:
        lines.extend(["", "## 校验提示", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "> 本脚本只复核输入契约和可执行算术，不判断经营假设、倍数、概率或来源是否合理；"
            "`target_price` 方法只校验外部模型审计字段，不会重算模型。",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not 0 <= args.precision <= 4:
        print("error: --precision must be between 0 and 4.", file=sys.stderr)
        return 2
    try:
        data = load_json(args.input)
        context, warnings = validate_root(data)
        results = calculate_scenarios(data, context, warnings)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(render_json(data, context, results, warnings))
    else:
        print(render_markdown(data, context, results, warnings, args.precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
