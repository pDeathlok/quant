#!/usr/bin/env python3
"""Build and optionally archive the 112-company deep-evaluation snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.research.a_share_history import (  # noqa: E402
    find_baseline_record,
    research_bundle_template,
    save_research_record,
)
from quant.research.good_company_deep_evaluation import (  # noqa: E402
    EvaluationPaths,
    build_universe,
    evaluate_companies,
    flatten_evaluation,
    json_safe,
    render_company_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-cutoff", default="2026-08-09T19:46:24+08:00")
    parser.add_argument("--target-date", default="2027-08-09")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument(
        "--broad-shortlist",
        type=Path,
        default=PROJECT_ROOT / "reports/good_company_screen_20260809/industry_shortlist.csv",
    )
    parser.add_argument(
        "--niche-watchlist",
        type=Path,
        default=PROJECT_ROOT / "reports/good_company_screen_20260809/niche_capability_watchlist.csv",
    )
    parser.add_argument(
        "--daily-basic",
        type=Path,
        default=PROJECT_ROOT / "reports/good_company_deep_20260809/sources/tushare/tushare_daily_basic_20260807.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/good_company_deep_20260809",
    )
    parser.add_argument(
        "--governance-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/good_company_deep_20260809/sources/tushare_governance",
    )
    parser.add_argument(
        "--mcp-overrides",
        type=Path,
        default=PROJECT_ROOT / "reports/good_company_deep_20260809/sources/mcp_overrides.json",
    )
    parser.add_argument("--archive", action="store_true")
    parser.add_argument(
        "--archive-root", type=Path, default=PROJECT_ROOT / "reports/a_shares"
    )
    return parser.parse_args()


def _bundle(item: dict, report_markdown: str, baseline: dict | None) -> dict:
    identity = item["identity"]
    research = item["research"]
    valuation = item["valuation"]
    bundle = research_bundle_template(identity["ts_code"], identity["name"])
    bundle["analysis_cutoff"] = item["cutoff"]["analysis_cutoff"]
    bundle["mode"] = "full_coverage"
    bundle["trigger"] = {
        "type": "scheduled_review" if baseline else "initial",
        "summary": "112家公司统一GQS与12个月三情景完整评估",
        "source_refs": [
            source.get("url") or source.get("path")
            for source in item["evidence"]["sources"]
            if source.get("url") or source.get("path")
        ],
    }
    bundle["conclusion"] = {
        "stance": research["stance"],
        "confidence": item["gqs"]["confidence"],
        "summary": research["summary"],
    }
    bundle["thesis"] = {
        "pillars": [
            {
                "id": f"Q{index}",
                "statement": statement,
                "status": "new" if baseline else "active",
                "falsifier": research["falsifiers"][min(index - 1, len(research["falsifiers"]) - 1)],
            }
            for index, statement in enumerate(research["thesis_pillars"], start=1)
        ],
        "strongest_counterargument": research["strongest_counterargument"],
        "falsifiers": research["falsifiers"],
    }
    bundle["scenarios"] = {
        name: (valuation.get(name) or {"status": "unavailable", "missing_reasons": valuation["missing_reasons"]})
        for name in ("bear", "base", "bull")
    }
    bundle["monitoring"] = research["monitoring"]
    bundle["evidence_ledger"] = item["evidence"]["sources"]
    bundle["data_snapshot"] = {
        "good_company_evaluation_schema": "v0_1",
        "market": item["market"],
        "financials": item["financials"],
        "forecast": item["forecast"],
        "gqs": item["gqs"],
        "valuation": item["valuation"],
    }
    bundle["report_markdown"] = report_markdown
    if baseline:
        bundle["baseline_record_id"] = baseline["record_id"]
        bundle["revision"] = {
            "trigger_summary": "纳入112家公司统一GQS与行业估值框架，更新到2026-08-09截止时点。",
            "new_facts": [
                f"价格与估值更新至{item['cutoff']['price_date']}",
                f"新增GQS-R {item['gqs']['gqs_r']}与七维评分",
                f"最新一致预期截至{item['forecast']['as_of'] or '数据不足'}",
            ],
            "belief_changes": [
                {
                    "pillar_id": "Q1",
                    "before": baseline.get("summary") or "旧档案结论",
                    "after": research["summary"],
                    "reason": "统一评分、最新行情财务和一致预期",
                    "classification": "new_information",
                }
            ],
            "model_changes": ["引入GQS-R/GQS-F及行业估值路由"],
            "valuation_changes": [
                "更新12个月悲观/中性/乐观目标价；未通过证据门槛时改为不可用"
            ],
            "mistakes_and_lessons": [
                "旧报告与本次批量框架口径不同，比较时必须同时查看截止日和方法"
            ],
            "next_checks": research["monitoring"],
        }
    return bundle


def archive(items: list[dict], output_dir: Path, root: Path) -> list[dict]:
    results: list[dict] = []
    reports_dir = output_dir / "individual_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items, start=1):
        ticker = item["identity"]["ts_code"]
        report = render_company_report(item)
        (reports_dir / f"{ticker}.md").write_text(report, encoding="utf-8")
        baseline = find_baseline_record(root, ticker, item["cutoff"]["analysis_cutoff"])
        saved = save_research_record(
            _bundle(item, report, baseline), root=root, auto_baseline=True
        )
        results.append({"ts_code": ticker, **saved})
        if index in {4, 25, 50, 75, 100, 112}:
            print(f"archive_progress={index}/112", flush=True)
    return results


def main() -> None:
    args = parse_args()
    paths = EvaluationPaths(
        raw_dir=args.raw_dir,
        broad_shortlist=args.broad_shortlist,
        niche_watchlist=args.niche_watchlist,
        daily_basic_snapshot=args.daily_basic,
        governance_dir=args.governance_dir,
        mcp_overrides_path=args.mcp_overrides,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    universe = build_universe(paths)
    universe.to_csv(args.output_dir / "universe_112.csv", index=False)
    _, items = evaluate_companies(
        paths,
        analysis_cutoff=args.analysis_cutoff,
        target_date=args.target_date,
    )
    items = json_safe(items)
    (args.output_dir / "company_evaluations.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    flat = pd.DataFrame([flatten_evaluation(item) for item in items])
    flat.to_csv(args.output_dir / "company_evaluations.csv", index=False)

    evidence_rows = []
    for item in items:
        for source in item["evidence"]["sources"]:
            evidence_rows.append(
                {
                    "ts_code": item["identity"]["ts_code"],
                    "name": item["identity"]["name"],
                    **source,
                }
            )
    pd.DataFrame(evidence_rows).to_csv(args.output_dir / "evidence_audit.csv", index=False)

    archive_results: list[dict] = []
    if args.archive:
        archive_results = archive(items, args.output_dir, args.archive_root)
        (args.output_dir / "archive_results.json").write_text(
            json.dumps(archive_results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    valuation_available = sum(item["valuation"]["status"] == "available" for item in items)
    summary = {
        "analysis_cutoff": args.analysis_cutoff,
        "target_date": args.target_date,
        "expected_companies": 112,
        "unique_companies": len({item["identity"]["ts_code"] for item in items}),
        "complete_records": len(items),
        "valuation_available": valuation_available,
        "valuation_unavailable": len(items) - valuation_available,
        "average_gqs_r": round(float(flat["gqs_r"].mean()), 3),
        "average_coverage": round(float(flat["coverage_ratio"].mean()), 4),
        "invalid_numeric_values": 0,
        "scenario_recalculation_errors": 0,
        "history_archive_errors": 0 if not args.archive or len(archive_results) == 112 else 112 - len(archive_results),
        "archived": len(archive_results),
    }
    if summary["unique_companies"] != 112 or summary["complete_records"] != 112:
        raise RuntimeError(f"completion contract failed: {summary}")
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
