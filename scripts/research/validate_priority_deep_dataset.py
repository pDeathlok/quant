#!/usr/bin/env python3
"""Validate the normalized priority Deep dataset before HTML generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "reports/good_company_deep_20260809"
MODULE_MAX = {
    "a_customer_business": 10,
    "b_scarcity_moat": 20,
    "c_growth_reinvestment": 10,
    "d_returns_profitability": 20,
    "e_cash_accounting": 15,
    "f_resilience_risk": 10,
    "g_governance_allocation": 15,
}
VALID_CLASSIFICATIONS = {"卓越复利候选", "优质公司", "潜力公司", "普通公司"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPORT_ROOT / "company_evaluations_deep_final.json",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=REPORT_ROOT / "deep_research_progress.csv",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=PROJECT_ROOT / "reports/a_shares",
    )
    parser.add_argument("--expected", type=int, default=112)
    return parser.parse_args()


def close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=tolerance)


def main() -> None:
    args = parse_args()
    data: list[dict[str, Any]] = json.loads(args.data.read_text(encoding="utf-8"))
    with args.progress.open(encoding="utf-8", newline="") as handle:
        progress = {
            row["ts_code"]: row
            for row in csv.DictReader(handle)
            if row["status"] == "completed"
        }
    issues: list[str] = []
    if len(data) != args.expected:
        issues.append(f"dataset count {len(data)} != {args.expected}")
    if len(progress) != args.expected:
        issues.append(f"progress completed count {len(progress)} != {args.expected}")
    tickers = [item.get("identity", {}).get("ts_code") for item in data]
    if len(set(tickers)) != len(tickers):
        issues.append("duplicate tickers")
    if set(tickers) != set(progress):
        issues.append("dataset tickers differ from completed progress tickers")

    for item in data:
        ticker = item["identity"]["ts_code"]
        gqs = item["gqs"]
        module_sum = 0.0
        for key, maximum in MODULE_MAX.items():
            value = gqs.get(key)
            if not isinstance(value, (int, float)) or not 0 <= value <= maximum:
                issues.append(f"{ticker}: invalid module {key}={value}")
                continue
            module_sum += float(value)
        if not close(module_sum, float(gqs["gqs_r"]), 0.011):
            issues.append(
                f"{ticker}: module sum {module_sum:.4f} != GQS-R {gqs['gqs_r']}"
            )
        expected_forward = round(
            float(gqs["gqs_r"]) + float(gqs["forward_adjustment"]), 2
        )
        if not close(expected_forward, float(gqs["gqs_f"]), 0.011):
            issues.append(
                f"{ticker}: GQS-R + forward {expected_forward} != GQS-F {gqs['gqs_f']}"
            )
        classification = gqs.get("classification")
        if classification not in VALID_CLASSIFICATIONS:
            issues.append(f"{ticker}: invalid classification {classification}")
        if classification in {"优质公司", "卓越复利候选"}:
            hard_gates = {
                "b_scarcity_moat": 14,
                "d_returns_profitability": 14,
                "e_cash_accounting": 9,
                "g_governance_allocation": 10,
            }
            for key, threshold in hard_gates.items():
                if float(gqs.get(key, -1)) < threshold:
                    issues.append(
                        f"{ticker}: {classification} fails hard gate {key}>={threshold}"
                    )
            if float(gqs.get("gqs_f", -1)) < 75:
                issues.append(f"{ticker}: {classification} has GQS-F below 75")
        if classification == "卓越复利候选":
            excellent_gates = {
                "b_scarcity_moat": 16,
                "d_returns_profitability": 16,
                "e_cash_accounting": 11,
                "g_governance_allocation": 12,
            }
            for key, threshold in excellent_gates.items():
                if float(gqs.get(key, -1)) < threshold:
                    issues.append(
                        f"{ticker}: excellent candidate fails {key}>={threshold}"
                    )
            if float(gqs.get("gqs_f", -1)) < 85:
                issues.append(f"{ticker}: excellent candidate has GQS-F below 85")
        coverage = gqs.get("coverage_ratio")
        if not isinstance(coverage, (int, float)) or not 0 <= float(coverage) <= 1:
            issues.append(f"{ticker}: invalid coverage ratio {coverage}")
        cutoff = item.get("cutoff", {})
        target_date = cutoff.get("target_date")
        try:
            analysis_date = datetime.fromisoformat(cutoff["analysis_cutoff"]).date()
            parsed_target = date.fromisoformat(str(target_date)[:10])
            if parsed_target <= analysis_date:
                issues.append(f"{ticker}: target_date is not after analysis cutoff")
        except (KeyError, TypeError, ValueError):
            issues.append(f"{ticker}: invalid or missing target_date {target_date}")
        price = item.get("market", {}).get("current_price")
        if not isinstance(price, (int, float)) or price <= 0:
            issues.append(f"{ticker}: invalid current price {price}")
            continue
        targets = []
        for name in ("bear", "base", "bull"):
            scenario = item.get("valuation", {}).get(name)
            if not isinstance(scenario, dict):
                issues.append(f"{ticker}: missing {name} scenario")
                continue
            target = scenario.get("target_price")
            upside = scenario.get("price_upside")
            if not isinstance(target, (int, float)) or target <= 0:
                issues.append(f"{ticker}: invalid {name} target {target}")
                continue
            targets.append(float(target))
            if not isinstance(upside, (int, float)) or not close(
                float(upside), float(target) / float(price) - 1, 1e-3
            ):
                issues.append(
                    f"{ticker}: {name} upside does not reconcile target/current price"
                )
            dividend = scenario.get("dividend_per_share")
            total_return = scenario.get("total_return")
            if isinstance(dividend, (int, float)) and isinstance(total_return, (int, float)):
                expected_total = (float(target) + float(dividend)) / float(price) - 1
                if not close(float(total_return), expected_total, 1e-3):
                    issues.append(
                        f"{ticker}: {name} total return does not reconcile target/dividend/current"
                    )
        if len(targets) == 3 and not targets[0] <= targets[1] <= targets[2]:
            issues.append(f"{ticker}: scenario targets are not bear <= base <= bull")
        price_source = item.get("market", {}).get("price_source")
        if price_source in (None, "missing", "scenario_reconciliation_fallback"):
            issues.append(f"{ticker}: current price lacks an explicit source ({price_source})")
        sources = item.get("evidence", {}).get("sources", [])
        if not sources or not any(source.get("url") or source.get("path") for source in sources):
            issues.append(f"{ticker}: evidence sources have no usable URL/path")
        if any(not source.get("label") or source.get("label") == "研究证据" for source in sources):
            issues.append(f"{ticker}: evidence source label is missing/default")
        if not item.get("research", {}).get("falsifiers"):
            issues.append(f"{ticker}: falsifiers are empty")

        progress_row = progress.get(ticker)
        if progress_row is None:
            continue
        report = PROJECT_ROOT / progress_row["deep_report_path"]
        if not report.exists():
            issues.append(f"{ticker}: missing report {report}")
        latest_path = args.archive_root / ticker / "latest.json"
        if not latest_path.exists():
            issues.append(f"{ticker}: missing latest.json")
            continue
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        if latest.get("record_id") != progress_row["record_id"]:
            issues.append(
                f"{ticker}: progress record {progress_row['record_id']} != latest {latest.get('record_id')}"
            )
        bundle_path = REPORT_ROOT / "workpapers" / ticker / "research_bundle.json"
        if bundle_path.exists():
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            if bundle.get("analysis_cutoff") != cutoff.get("analysis_cutoff"):
                issues.append(f"{ticker}: bundle cutoff differs from archive cutoff")
            bundle_record = bundle.get("latest_record_id") or bundle.get("record_id")
            if bundle_record and bundle_record != progress_row["record_id"]:
                issues.append(f"{ticker}: bundle record_id differs from progress/latest")

    if issues:
        print(f"FAIL: {len(issues)} issue(s)")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "status": "PASS",
                "companies": len(data),
                "valuations_available": sum(
                    item["valuation"]["status"] == "available" for item in data
                ),
                "reports_present": len(data),
                "latest_records_match": len(data),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
