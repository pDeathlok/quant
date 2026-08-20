from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/research/validate_individual_deep_reports.py"
SPEC = importlib.util.spec_from_file_location("deep_report_validator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


def test_haomai_deep_report_only_has_expected_pending_archive_issues() -> None:
    contract = VALIDATOR.load_contract(
        ROOT / "config/good_company_deep_report_contract_v1.json"
    )
    report = (
        ROOT
        / "reports/good_company_deep_20260809/individual_reports/002595.SZ.md"
    )

    issues = VALIDATOR.validate_report(report, contract)

    assert {issue.code for issue in issues} <= {"history_record_ids", "placeholder"}


def test_contract_rejects_thin_batch_proxy_report() -> None:
    contract = VALIDATOR.load_contract(
        ROOT / "config/good_company_deep_report_contract_v1.json"
    )

    issues = VALIDATOR.validate_report_text("# 简报\n\n只有一句结论。", contract)

    codes = {issue.code for issue in issues}
    assert "minimum_characters" in codes
    assert "missing_headings" in codes
    assert "missing_score_ids" in codes
    assert "mcp_audit" in codes
