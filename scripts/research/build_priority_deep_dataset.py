#!/usr/bin/env python3
"""Normalize completed immutable Deep records into the dashboard data contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "reports/good_company_deep_20260809"
MODULE_KEYS = {
    "A": "a_customer_business",
    "B": "b_scarcity_moat",
    "C": "c_growth_reinvestment",
    "D": "d_returns_profitability",
    "E": "e_cash_accounting",
    "F": "f_resilience_risk",
    "G": "g_governance_allocation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--legacy-universe",
        type=Path,
        default=REPORT_ROOT / "company_evaluations.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "company_evaluations_deep_final.json",
    )
    parser.add_argument("--expected", type=int, default=112)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def canonical_classification(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("卓越"):
        return "卓越复利候选"
    if cleaned.startswith("优质"):
        return "优质公司"
    if cleaned.startswith("潜力"):
        return "潜力公司"
    if cleaned.startswith("普通"):
        return "普通公司"
    if "卓越" in cleaned:
        return "卓越复利候选"
    if "潜力" in cleaned:
        return "潜力公司"
    if "优质" in cleaned:
        return "优质公司"
    return "普通公司"


def gqs_sources(record: dict[str, Any], workpaper: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    snapshot = record.get("data_snapshot", {})
    for key in ("gqs", "score"):
        value = snapshot.get(key)
        if isinstance(value, dict):
            sources.append(value)
    scorecard = workpaper / "gqs_scorecard.json"
    if scorecard.exists():
        sources.append(load_json(scorecard))
    bundle_path = workpaper / "research_bundle.json"
    if bundle_path.exists():
        bundle = load_json(bundle_path)
        bundle_snapshot = bundle.get("data_snapshot", {})
        for key in ("gqs", "score"):
            value = bundle_snapshot.get(key)
            if isinstance(value, dict):
                sources.append(value)
    return sources


def extract_modules(sources: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for source in sources:
        module_scores = source.get("module_scores")
        if isinstance(module_scores, dict):
            for letter in MODULE_KEYS:
                value = module_scores.get(letter)
                if isinstance(value, (int, float)):
                    result.setdefault(letter, float(value))
        modules = source.get("modules")
        if isinstance(modules, dict):
            for letter in MODULE_KEYS:
                value = modules.get(letter)
                if isinstance(value, dict):
                    value = value.get("score", value.get("realized"))
                if isinstance(value, (int, float)):
                    result.setdefault(letter, float(value))
        dimensions = source.get("dimensions_realized")
        if isinstance(dimensions, dict):
            for letter in MODULE_KEYS:
                value = dimensions.get(letter)
                if isinstance(value, (int, float)):
                    result.setdefault(letter, float(value))
        realized_points = source.get("module_realized_points")
        if isinstance(realized_points, dict):
            for letter in MODULE_KEYS:
                value = realized_points.get(letter)
                if isinstance(value, (int, float)):
                    result.setdefault(letter, float(value))
        for letter, long_key in MODULE_KEYS.items():
            value = source.get(long_key, source.get(letter))
            if letter == "F" and value is None:
                value = source.get("f_resilience")
            if isinstance(value, (int, float)):
                result.setdefault(letter, float(value))
    return result


def extract_gqs(
    progress: dict[str, str], record: dict[str, Any], workpaper: Path
) -> dict[str, Any]:
    sources = gqs_sources(record, workpaper)
    modules = extract_modules(sources)
    if set(modules) != set(MODULE_KEYS):
        missing = sorted(set(MODULE_KEYS) - set(modules))
        raise ValueError(f"{progress['ts_code']} missing GQS modules: {missing}")

    def source_value(*keys: str) -> Any:
        for source in sources:
            for key in keys:
                if source.get(key) is not None:
                    return source[key]
        return None

    classification_detail = str(
        source_value("classification") or progress["classification"] or "普通公司"
    )
    gqs_r = first_number(source_value("gqs_r"), float(progress["gqs_r"]))
    gqs_f = first_number(source_value("gqs_f"), float(progress["gqs_f"]))
    realized_coverage = first_number(
        source_value("realized_coverage", "coverage_ratio", "coverage"), 1.0
    )
    forward_coverage = first_number(
        source_value("forward_coverage", "forward_coverage_ratio"), 0.0
    )
    confidence = source_value("confidence") or record.get("conclusion", {}).get(
        "confidence", "未标注"
    )
    return {
        **{MODULE_KEYS[letter]: round(modules[letter], 2) for letter in MODULE_KEYS},
        "gqs_r": gqs_r,
        "gqs_f": gqs_f,
        "forward_adjustment": round((gqs_f or 0) - (gqs_r or 0), 2),
        "classification": canonical_classification(classification_detail),
        "classification_detail": classification_detail,
        "coverage_ratio": realized_coverage,
        "forward_coverage_ratio": forward_coverage,
        "confidence": str(confidence),
        "score_limitations": [
            "GQS衡量公司质量，不包含估值与价格。",
            "实现评分与forward桥分轨；三情景是条件估算。",
        ],
    }


def normalize_scenario(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    target = first_number(value.get("target_price"))
    upside = first_number(value.get("price_upside"), value.get("price_return"))
    total = first_number(value.get("total_return"))
    if target is None:
        return None
    return {
        "target_price": target,
        "price_upside": upside,
        "total_return": total,
        "dividend_per_share": first_number(value.get("dividend_per_share")),
        "conditions": value.get("conditions", value.get("story", "见个股Deep报告")),
    }


def extract_price(
    record: dict[str, Any],
    bundle: dict[str, Any],
    scenarios: dict[str, Any],
    workpaper: Path,
) -> tuple[float | None, str]:
    for payload in (record, bundle):
        snapshot = payload.get("data_snapshot", {})
        for value in (snapshot.get("price"), snapshot.get("market"), payload.get("market")):
            if isinstance(value, dict):
                direct = first_number(
                    value.get("current_price"),
                    value.get("price"),
                    value.get("price_cny"),
                    value.get("close"),
                )
                if direct is not None:
                    return direct, "archive_or_bundle_snapshot"
            elif isinstance(value, (int, float)):
                return float(value), "archive_or_bundle_snapshot"
    for name in ("scenario_valuation_input.json", "scenario_valuation_output.json"):
        path = workpaper / name
        if not path.exists():
            continue
        payload = load_json(path)
        direct = first_number(
            payload.get("current_price"),
            payload.get("price"),
            payload.get("price_cny"),
            payload.get("market", {}).get("current_price")
            if isinstance(payload.get("market"), dict)
            else None,
        )
        if direct is not None:
            return direct, name
    for scenario in scenarios.values():
        if not scenario:
            continue
        target = scenario.get("target_price")
        upside = scenario.get("price_upside")
        if isinstance(target, (int, float)) and isinstance(upside, (int, float)):
            if abs(1 + upside) > 1e-9:
                return target / (1 + upside), "scenario_reconciliation_fallback"
    return None, "missing"


def load_optional_bundle(workpaper: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = workpaper / "research_bundle.json"
    if not path.exists():
        return record
    bundle = load_json(path)
    record_ticker = record.get("ticker")
    if record_ticker is None and isinstance(record.get("identity"), dict):
        record_ticker = record["identity"].get("ts_code")
    if record_ticker and bundle.get("ticker") not in (None, record_ticker):
        return record
    if bundle.get("analysis_cutoff") != record.get("analysis_cutoff"):
        return record
    return bundle


def extract_target_date(
    record: dict[str, Any], bundle: dict[str, Any], workpaper: Path
) -> str | None:
    for payload in (record, bundle):
        for value in payload.get("scenarios", {}).values():
            if isinstance(value, dict) and value.get("target_date"):
                return str(value["target_date"])
        cutoff = payload.get("cutoff")
        if isinstance(cutoff, dict) and cutoff.get("target_date"):
            return str(cutoff["target_date"])
    for name in ("scenario_valuation_output.json", "scenario_valuation_input.json"):
        path = workpaper / name
        if not path.exists():
            continue
        payload = load_json(path)
        for key in ("target_date", "valuation_target_date"):
            if payload.get(key):
                return str(payload[key])
        calculation = payload.get("calculation")
        if isinstance(calculation, dict) and calculation.get("target_date"):
            return str(calculation["target_date"])
    return None


def extract_price_date(record: dict[str, Any], bundle: dict[str, Any]) -> str | None:
    for payload in (record, bundle):
        snapshot = payload.get("data_snapshot", {})
        price = snapshot.get("price")
        if isinstance(price, dict):
            value = price.get("date", price.get("as_of", price.get("price_as_of")))
            if value:
                return str(value)
        market = snapshot.get("market")
        if isinstance(market, dict):
            value = market.get("price_date", market.get("price_as_of", market.get("as_of")))
            if value:
                return str(value)
    return None


def normalize_monitoring(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        if isinstance(value, str):
            normalized.append(value)
            continue
        if not isinstance(value, dict):
            normalized.append(str(value))
            continue
        title = value.get("metric", value.get("indicator", value.get("name", "监测项")))
        details = []
        for key, label in (
            ("latest", "当前"),
            ("threshold", "阈值"),
            ("frequency", "频率"),
            ("next_check", "复核"),
        ):
            if value.get(key) is not None:
                details.append(f"{label}：{value[key]}")
        normalized.append(f"{title}｜{'；'.join(details)}" if details else str(title))
    return normalized


def extract_technical(workpaper: Path, record: dict[str, Any]) -> str:
    labels = {
        "cautious": "谨慎",
        "constructive": "建设性偏强",
        "mixed": "震荡/混合",
    }
    for name in (
        "technical_summary.json",
        "price_volume_output.json",
        "technical_output.json",
    ):
        path = workpaper / name
        if not path.exists():
            continue
        payload = load_json(path)
        state = payload.get("trend", {}).get("state") or payload.get("technical_state")
        if state:
            return labels.get(str(state), str(state))
    snapshot = record.get("data_snapshot", {})
    technical = snapshot.get("technical")
    if isinstance(technical, dict):
        state = str(technical.get("trend", {}).get("state") or "已核验")
        return labels.get(state, state)
    return "已核验"


def normalize_company(
    progress: dict[str, str],
    archive_root: Path,
    legacy: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ticker = progress["ts_code"]
    workpaper = REPORT_ROOT / "workpapers" / ticker
    record_path = archive_root / ticker / "records" / progress["record_id"] / "record.json"
    record = load_json(record_path)
    bundle = load_optional_bundle(workpaper, record)
    gqs = extract_gqs(progress, record, workpaper)
    scenarios = {
        key: normalize_scenario(record.get("scenarios", {}).get(key))
        for key in ("bear", "base", "bull")
    }
    price, price_source = extract_price(record, bundle, scenarios, workpaper)
    valuation_available = all(value is not None for value in scenarios.values())
    legacy_item = legacy.get(ticker, {})
    legacy_identity = legacy_item.get("identity", {})
    thesis = bundle.get("thesis", record.get("thesis", {}))
    pillars = thesis.get("pillars", []) if isinstance(thesis, dict) else []
    pillar_texts = []
    pillar_falsifiers = []
    for pillar in pillars:
        if isinstance(pillar, dict):
            text = pillar.get("statement", pillar.get("thesis", pillar.get("name")))
            falsifier = pillar.get("falsifier", pillar.get("falsifiers"))
            if isinstance(falsifier, list):
                pillar_falsifiers.extend(str(value) for value in falsifier if value)
            elif falsifier:
                pillar_falsifiers.append(str(falsifier))
        else:
            text = pillar
        if text:
            pillar_texts.append(str(text))
    evidence_ledger = bundle.get("evidence_ledger", record.get("evidence_ledger", []))
    sources = []
    for source in evidence_ledger[:12]:
        if not isinstance(source, dict):
            continue
        source_url = source.get("url")
        source_path = source.get("path", source.get("source_ref"))
        raw_source = source.get("source")
        if source_url is None and isinstance(raw_source, str) and raw_source.startswith(
            ("http://", "https://")
        ):
            source_url = raw_source
        if source_url is None and isinstance(source_path, str) and source_path.startswith(
            ("http://", "https://")
        ):
            source_url, source_path = source_path, None
        if isinstance(source_url, str) and not source_url.startswith(("http://", "https://")):
            source_path = source_path or source_url
            source_url = None
        source_label = source.get("label", source.get("title"))
        if not source_label and isinstance(raw_source, str) and not raw_source.startswith(
            ("http://", "https://")
        ):
            source_label = raw_source
        if not source_label:
            source_label = source.get("fact", "研究证据")
        source_label = str(source_label)
        if len(source_label) > 80:
            source_label = source_label[:77] + "…"
        sources.append(
            {
                "label": source_label,
                "kind": source.get("kind", "evidence"),
                "path": source_path,
                "url": source_url,
                "available_at": source.get("available_at"),
            }
        )
    top_falsifiers = thesis.get("falsifiers", []) if isinstance(thesis, dict) else []
    if isinstance(top_falsifiers, str):
        top_falsifiers = [top_falsifiers]
    elif not isinstance(top_falsifiers, list):
        top_falsifiers = []
    falsifiers = [str(value) for value in top_falsifiers if value]
    for value in pillar_falsifiers:
        if value not in falsifiers:
            falsifiers.append(value)
    price_date = extract_price_date(record, bundle)
    conclusion = record.get("conclusion", {})
    return {
        "identity": {
            "ts_code": ticker,
            "name": progress["name"],
            "industry": legacy_identity.get("industry") or progress["broad_industry"],
            "broad_industry": progress["broad_industry"],
        },
        "cutoff": {
            "analysis_cutoff": record["analysis_cutoff"],
            "target_date": extract_target_date(record, bundle, workpaper),
            "price_date": price_date or "2026-08-07",
        },
        "market": {
            "current_price": price,
            "price_source": price_source,
            "technical_state": extract_technical(workpaper, record),
        },
        "gqs": gqs,
        "valuation": {
            "status": "available" if valuation_available else "unavailable",
            "method_primary": "条件式三情景估值",
            "method_crosscheck": "详见个股MD估值桥与反向估值",
            "forecast_basis": "目标年完全摊薄盈利/现金流",
            "bear": scenarios["bear"],
            "base": scenarios["base"],
            "bull": scenarios["bull"],
            "missing_reasons": [] if valuation_available else ["归档情景字段不完整"],
        },
        "research": {
            "stance": conclusion.get("stance", "中性观察"),
            "summary": conclusion.get("summary", progress["notes"]),
            "thesis_pillars": pillar_texts,
            "strongest_counterargument": thesis.get("strongest_counterargument", "见个股Deep报告") if isinstance(thesis, dict) else "见个股Deep报告",
            "falsifiers": falsifiers,
            "catalysts": [],
            "risks": [],
            "monitoring": normalize_monitoring(
                bundle.get("monitoring", record.get("monitoring", []))
            ),
        },
        "evidence": {"sources": sources, "data_conflicts": []},
        "archive": {
            "record_id": progress["record_id"],
            "report_path": progress["deep_report_path"],
            "validated_at": progress["validated_at"],
        },
    }


def main() -> None:
    args = parse_args()
    with args.progress.open(encoding="utf-8", newline="") as handle:
        progress = [row for row in csv.DictReader(handle) if row["status"] == "completed"]
    if len(progress) != args.expected:
        raise SystemExit(f"expected {args.expected} completed companies, got {len(progress)}")
    legacy_items = load_json(args.legacy_universe)
    legacy = {item["identity"]["ts_code"]: item for item in legacy_items}
    data = [normalize_company(row, args.archive_root, legacy) for row in progress]
    if len({item["identity"]["ts_code"] for item in data}) != len(data):
        raise SystemExit("duplicate ticker in normalized Deep dataset")
    args.output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "companies": len(data),
                "valuations_available": sum(
                    item["valuation"]["status"] == "available" for item in data
                ),
                "bytes": args.output.stat().st_size,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
