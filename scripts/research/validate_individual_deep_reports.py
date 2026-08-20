#!/usr/bin/env python3
"""Validate individually researched A-share Deep Markdown reports."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONTRACT = Path("config/good_company_deep_report_contract_v1.json")
RECORD_ID_PATTERN = re.compile(r"\d{8}T\d{6}[pm]\d{4}-[0-9a-f]{10}")


@dataclass(frozen=True)
class ReportContract:
    minimum_characters: int
    minimum_lines: int
    minimum_evidence_rows: int
    minimum_mcp_actual_calls: int
    minimum_mcp_successful_calls: int
    required_headings: tuple[str, ...]
    required_score_ids: tuple[str, ...]
    required_scenarios: tuple[str, ...]
    blocked_placeholders: tuple[str, ...]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


def load_contract(path: Path) -> ReportContract:
    """Load the versioned JSON contract and validate its boundary types."""

    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("contract schema_version must be 1")
    integer_fields = (
        "minimum_characters",
        "minimum_lines",
        "minimum_evidence_rows",
        "minimum_mcp_actual_calls",
        "minimum_mcp_successful_calls",
    )
    for field in integer_fields:
        if not isinstance(payload.get(field), int) or payload[field] < 0:
            raise ValueError(f"contract {field} must be a non-negative integer")
    list_fields = (
        "required_headings",
        "required_score_ids",
        "required_scenarios",
        "blocked_placeholders",
    )
    for field in list_fields:
        value = payload.get(field)
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ValueError(f"contract {field} must be a non-empty string array")
    return ReportContract(
        minimum_characters=payload["minimum_characters"],
        minimum_lines=payload["minimum_lines"],
        minimum_evidence_rows=payload["minimum_evidence_rows"],
        minimum_mcp_actual_calls=payload["minimum_mcp_actual_calls"],
        minimum_mcp_successful_calls=payload["minimum_mcp_successful_calls"],
        required_headings=tuple(payload["required_headings"]),
        required_score_ids=tuple(payload["required_score_ids"]),
        required_scenarios=tuple(payload["required_scenarios"]),
        blocked_placeholders=tuple(payload["blocked_placeholders"]),
    )


def _section(text: str, start_heading: str, end_heading: str | None) -> str:
    start = text.find(start_heading)
    if start < 0:
        return ""
    if end_heading is None:
        return text[start:]
    end = text.find(end_heading, start + len(start_heading))
    return text[start:] if end < 0 else text[start:end]


def _extract_mcp_execution(text: str) -> dict[str, Any] | None:
    section = _section(
        text,
        "## 13. MCP执行审计、冲突与局限",
        "## 14. 历史归档",
    )
    for candidate in re.findall(r"```json\s*(\{.*?\})\s*```", section, re.DOTALL):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        execution = payload.get("mcp_execution")
        if isinstance(execution, dict):
            return execution
    return None


def validate_report_text(text: str, contract: ReportContract) -> list[ValidationIssue]:
    """Return all contract violations for one report body."""

    issues: list[ValidationIssue] = []
    lines = text.splitlines()
    if len(text) < contract.minimum_characters:
        issues.append(
            ValidationIssue(
                "minimum_characters",
                f"正文仅{len(text)}字符，要求至少{contract.minimum_characters}",
            )
        )
    if len(lines) < contract.minimum_lines:
        issues.append(
            ValidationIssue(
                "minimum_lines",
                f"正文仅{len(lines)}行，要求至少{contract.minimum_lines}",
            )
        )

    heading_positions = [text.find(heading) for heading in contract.required_headings]
    missing_headings = [
        heading
        for heading, position in zip(contract.required_headings, heading_positions)
        if position < 0
    ]
    if missing_headings:
        issues.append(
            ValidationIssue("missing_headings", f"缺少章节：{missing_headings}")
        )
    present_positions = [position for position in heading_positions if position >= 0]
    if present_positions != sorted(present_positions):
        issues.append(ValidationIssue("heading_order", "章节顺序不符合Deep合同"))

    score_ids = re.findall(r"^\|\s*([A-G]\d)\s*\|", text, re.MULTILINE)
    missing_score_ids = sorted(set(contract.required_score_ids) - set(score_ids))
    duplicate_score_ids = sorted(
        score_id for score_id in set(score_ids) if score_ids.count(score_id) > 1
    )
    if missing_score_ids:
        issues.append(
            ValidationIssue("missing_score_ids", f"缺少评分项：{missing_score_ids}")
        )
    if duplicate_score_ids:
        issues.append(
            ValidationIssue("duplicate_score_ids", f"重复评分项：{duplicate_score_ids}")
        )

    scenario_section = _section(
        text,
        "## 5. 三情景价格预期",
        "## 6. 投资逻辑与反证",
    )
    for scenario in contract.required_scenarios:
        if not re.search(rf"^\|\s*{re.escape(scenario)}\s*\|", scenario_section, re.MULTILINE):
            issues.append(
                ValidationIssue("missing_scenario", f"缺少{scenario}情景表行")
            )

    evidence_section = _section(text, "## 12. 证据账本", "## 13. MCP执行审计、冲突与局限")
    evidence_rows = re.findall(r"^\|\s*\d+\s*\|", evidence_section, re.MULTILINE)
    if len(evidence_rows) < contract.minimum_evidence_rows:
        issues.append(
            ValidationIssue(
                "evidence_rows",
                f"证据账本仅{len(evidence_rows)}行，要求至少{contract.minimum_evidence_rows}",
            )
        )
    if "](" not in evidence_section:
        issues.append(ValidationIssue("evidence_links", "证据账本缺少可点击链接"))

    execution = _extract_mcp_execution(text)
    if execution is None:
        issues.append(ValidationIssue("mcp_audit", "缺少可解析的MCP执行审计JSON"))
    else:
        actual_calls = execution.get("actual_call_count")
        successful_calls = execution.get("successful_call_count")
        tools = execution.get("tools")
        if not isinstance(actual_calls, int) or actual_calls < contract.minimum_mcp_actual_calls:
            issues.append(ValidationIssue("mcp_actual_calls", "MCP实际调用数不足"))
        accepted_fallback_failures = {
            "provider_credits_exhausted",
            "authentication_failed",
            "permission_denied",
            "rate_limited",
            "service_unavailable",
            "runtime_not_loaded",
        }
        fallback_accepted = (
            isinstance(actual_calls, int)
            and actual_calls >= contract.minimum_mcp_actual_calls
            and successful_calls == 0
            and execution.get("status") == "failed_fallback_used"
            and execution.get("failure_class") in accepted_fallback_failures
            and isinstance(execution.get("fallback"), str)
            and bool(execution["fallback"].strip())
        )
        if (
            not isinstance(successful_calls, int)
            or (
                successful_calls < contract.minimum_mcp_successful_calls
                and not fallback_accepted
            )
        ):
            issues.append(ValidationIssue("mcp_successful_calls", "MCP成功调用数不足"))
        if isinstance(actual_calls, int) and isinstance(tools, list) and len(tools) != actual_calls:
            issues.append(
                ValidationIssue(
                    "mcp_tool_count",
                    f"MCP工具审计{len(tools)}行，与actual_call_count={actual_calls}不一致",
                )
            )

    archive_section = _section(text, "## 14. 历史归档", None)
    record_ids = RECORD_ID_PATTERN.findall(archive_section)
    if len(record_ids) < 2:
        issues.append(
            ValidationIssue("history_record_ids", "归档章节必须包含基线和本次两个record_id")
        )
    for placeholder in contract.blocked_placeholders:
        if placeholder in text:
            issues.append(
                ValidationIssue("placeholder", f"仍含未完成占位符：{placeholder}")
            )
    return issues


def validate_report(path: Path, contract: ReportContract) -> list[ValidationIssue]:
    """Read and validate one Markdown report."""

    return validate_report_text(path.read_text(encoding="utf-8"), contract)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_contract(args.contract)
    failure_count = 0
    for report_path in args.reports:
        issues = validate_report(report_path, contract)
        if issues:
            failure_count += 1
            print(f"FAIL {report_path}")
            for issue in issues:
                print(f"  [{issue.code}] {issue.message}")
        else:
            print(f"PASS {report_path}")
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
