"""Immutable local history for A-share research and belief updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant.data.atomic_io import atomic_write_json
from quant.data.source_merge import normalize_ts_code
from quant.routine.paths import PROJECT_ROOT


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_RESEARCH_ROOT = PROJECT_ROOT / "reports/a_shares"
SCHEMA_VERSION = 1
RECORD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._+-]+$")
UPDATE_TRIGGER_TYPES = {
    "financial_report",
    "company_event",
    "industry_event",
    "policy_event",
    "market_event",
    "scheduled_review",
    "historical_replay",
    "other",
}
BELIEF_CHANGE_CLASSIFICATIONS = {
    "new_information",
    "prior_error",
    "model_limitation",
    "noise",
    "unresolved",
}
THESIS_STATUSES = {
    "active",
    "strengthened",
    "unchanged",
    "weakened",
    "invalidated",
    "new",
}
REVISION_LIST_FIELDS = (
    "new_facts",
    "belief_changes",
    "model_changes",
    "valuation_changes",
    "mistakes_and_lessons",
    "next_checks",
)


class ResearchHistoryError(ValueError):
    """Raised when an archive request violates the immutable history contract."""


def _parse_aware_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ResearchHistoryError(f"{field} must be a non-empty ISO 8601 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ResearchHistoryError(
            f"{field} must be an ISO 8601 timestamp with timezone"
        ) from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ResearchHistoryError(f"{field} must include an explicit UTC offset")
    return timestamp.astimezone(SHANGHAI_TZ)


def _normalize_ticker(value: Any) -> str:
    if not isinstance(value, str):
        raise ResearchHistoryError("ticker must be a string")
    ticker = normalize_ts_code(value).upper()
    if (
        len(ticker) != 9
        or ticker[6] != "."
        or not ticker[:6].isdigit()
        or ticker[7:] not in {"SH", "SZ", "BJ"}
    ):
        raise ResearchHistoryError("ticker must resolve to a six-digit .SH, .SZ, or .BJ code")
    return ticker


def _require_mapping(bundle: dict[str, Any], field: str) -> dict[str, Any]:
    value = bundle.get(field)
    if not isinstance(value, dict):
        raise ResearchHistoryError(f"{field} must be an object")
    return value


def _require_list(bundle: dict[str, Any], field: str) -> list[Any]:
    value = bundle.get(field)
    if not isinstance(value, list):
        raise ResearchHistoryError(f"{field} must be an array")
    return value


def _record_directory(root: Path, ticker: str, record_id: str) -> Path:
    if not RECORD_ID_PATTERN.fullmatch(record_id):
        raise ResearchHistoryError(f"invalid record_id: {record_id}")
    return root / ticker / "records" / record_id


def _index_path(root: Path, ticker: str) -> Path:
    return root / ticker / "index.json"


def _read_index(root: Path, ticker: str) -> dict[str, Any]:
    path = _index_path(root, ticker)
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "ticker": ticker, "records": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchHistoryError(f"cannot read research index {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ResearchHistoryError(f"research index has an invalid structure: {path}")
    return payload


def list_research_records(root: Path, ticker: str) -> list[dict[str, Any]]:
    """List one ticker's records from newest to oldest."""

    normalized_ticker = _normalize_ticker(ticker)
    records = list(_read_index(root, normalized_ticker)["records"])
    return sorted(
        records,
        key=lambda item: (str(item.get("analysis_cutoff", "")), str(item.get("created_at", ""))),
        reverse=True,
    )


def find_baseline_record(
    root: Path,
    ticker: str,
    before: str | datetime,
) -> dict[str, Any] | None:
    """Return the latest immutable record strictly before a new cutoff."""

    normalized_ticker = _normalize_ticker(ticker)
    if isinstance(before, datetime):
        if before.tzinfo is None or before.utcoffset() is None:
            raise ResearchHistoryError("before must include an explicit UTC offset")
        cutoff = before.astimezone(SHANGHAI_TZ)
    else:
        cutoff = _parse_aware_timestamp(before, "before")
    for entry in list_research_records(root, normalized_ticker):
        entry_cutoff = _parse_aware_timestamp(entry.get("analysis_cutoff"), "analysis_cutoff")
        if entry_cutoff < cutoff:
            return entry
    return None


def load_research_record(
    root: Path,
    ticker: str,
    record_id: str = "latest",
) -> dict[str, Any]:
    """Load a record and restore its report Markdown into the returned bundle."""

    normalized_ticker = _normalize_ticker(ticker)
    if record_id == "latest":
        records = list_research_records(root, normalized_ticker)
        if not records:
            raise ResearchHistoryError(f"no research history found for {normalized_ticker}")
        record_id = str(records[0]["record_id"])
    record_dir = _record_directory(root, normalized_ticker, record_id)
    record_path = record_dir / "record.json"
    report_path = record_dir / "report.md"
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        payload["report_markdown"] = report_path.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchHistoryError(f"cannot load research record {record_id}: {exc}") from exc
    return payload


def _validate_revision(bundle: dict[str, Any], *, is_update: bool) -> None:
    revision = bundle.get("revision")
    if not is_update:
        if revision is not None and not isinstance(revision, dict):
            raise ResearchHistoryError("revision must be an object or null")
        return
    if not isinstance(revision, dict):
        raise ResearchHistoryError("updated research must include a revision object")
    trigger_summary = revision.get("trigger_summary")
    if not isinstance(trigger_summary, str) or not trigger_summary.strip():
        raise ResearchHistoryError("revision.trigger_summary must be a non-empty string")
    for field in REVISION_LIST_FIELDS:
        value = revision.get(field)
        if not isinstance(value, list):
            raise ResearchHistoryError(f"revision.{field} must be an array")
    for index, change in enumerate(revision["belief_changes"]):
        if not isinstance(change, dict):
            raise ResearchHistoryError(
                f"revision.belief_changes[{index}] must be an object"
            )
        pillar_id = change.get("pillar_id")
        if not isinstance(pillar_id, str) or not pillar_id.strip():
            raise ResearchHistoryError(
                f"revision.belief_changes[{index}].pillar_id must be non-empty"
            )
        classification = change.get("classification")
        if classification not in BELIEF_CHANGE_CLASSIFICATIONS:
            raise ResearchHistoryError(
                f"revision.belief_changes[{index}].classification must be one of "
                f"{sorted(BELIEF_CHANGE_CLASSIFICATIONS)}"
            )


def _validate_thesis(thesis: dict[str, Any]) -> None:
    pillars = thesis.get("pillars")
    if not isinstance(pillars, list):
        raise ResearchHistoryError("thesis.pillars must be an array")
    seen_ids: set[str] = set()
    for index, pillar in enumerate(pillars):
        if not isinstance(pillar, dict):
            raise ResearchHistoryError(f"thesis.pillars[{index}] must be an object")
        pillar_id = pillar.get("id")
        if not isinstance(pillar_id, str) or not pillar_id.strip():
            raise ResearchHistoryError(f"thesis.pillars[{index}].id must be non-empty")
        if pillar_id in seen_ids:
            raise ResearchHistoryError(f"duplicate thesis pillar id: {pillar_id}")
        seen_ids.add(pillar_id)
        statement = pillar.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise ResearchHistoryError(
                f"thesis.pillars[{index}].statement must be non-empty"
            )
        status = pillar.get("status")
        if status not in THESIS_STATUSES:
            raise ResearchHistoryError(
                f"thesis.pillars[{index}].status must be one of "
                f"{sorted(THESIS_STATUSES)}"
            )


def validate_research_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one research record before persistence."""

    if not isinstance(bundle, dict):
        raise ResearchHistoryError("research bundle must be a JSON object")
    normalized = deepcopy(bundle)
    schema_version = normalized.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ResearchHistoryError(f"schema_version must be {SCHEMA_VERSION}")
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["ticker"] = _normalize_ticker(normalized.get("ticker"))

    company_name = normalized.get("company_name")
    if not isinstance(company_name, str) or not company_name.strip():
        raise ResearchHistoryError("company_name must be a non-empty string")
    normalized["company_name"] = company_name.strip()
    cutoff = _parse_aware_timestamp(normalized.get("analysis_cutoff"), "analysis_cutoff")
    normalized["analysis_cutoff"] = cutoff.isoformat(timespec="seconds")

    mode = normalized.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        raise ResearchHistoryError("mode must be a non-empty string")
    normalized["mode"] = mode.strip()
    conclusion = _require_mapping(normalized, "conclusion")
    for field in ("stance", "confidence", "summary"):
        value = conclusion.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ResearchHistoryError(f"conclusion.{field} must be a non-empty string")
    thesis = _require_mapping(normalized, "thesis")
    _validate_thesis(thesis)
    scenarios = _require_mapping(normalized, "scenarios")
    for name in ("bear", "base", "bull"):
        if not isinstance(scenarios.get(name), dict):
            raise ResearchHistoryError(f"scenarios.{name} must be an object")
    _require_list(normalized, "monitoring")
    _require_list(normalized, "evidence_ledger")

    report_markdown = normalized.get("report_markdown")
    if not isinstance(report_markdown, str) or not report_markdown.strip():
        raise ResearchHistoryError("report_markdown must be a non-empty string")

    trigger = _require_mapping(normalized, "trigger")
    trigger_type = trigger.get("type")
    if not isinstance(trigger_type, str) or not trigger_type.strip():
        raise ResearchHistoryError("trigger.type must be a non-empty string")
    trigger["type"] = trigger_type.strip()
    trigger_summary = trigger.get("summary")
    if not isinstance(trigger_summary, str) or not trigger_summary.strip():
        raise ResearchHistoryError("trigger.summary must be a non-empty string")

    baseline_record_id = normalized.get("baseline_record_id")
    if baseline_record_id is not None and (
        not isinstance(baseline_record_id, str)
        or not RECORD_ID_PATTERN.fullmatch(baseline_record_id)
    ):
        raise ResearchHistoryError("baseline_record_id is invalid")
    is_update = baseline_record_id is not None or trigger["type"] in UPDATE_TRIGGER_TYPES
    _validate_revision(normalized, is_update=is_update)
    return normalized


def _canonical_content(bundle: dict[str, Any]) -> tuple[str, str]:
    content = deepcopy(bundle)
    content.pop("created_at", None)
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _record_id(cutoff: datetime, content_sha256: str) -> str:
    offset = cutoff.strftime("%z")
    safe_offset = f"{'p' if offset.startswith('+') else 'm'}{offset[1:]}"
    return f"{cutoff.strftime('%Y%m%dT%H%M%S')}{safe_offset}-{content_sha256[:10]}"


def _atomic_write_text(text: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def _build_index_entry(
    persisted: dict[str, Any],
    *,
    root: Path,
    ticker: str,
    record_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    conclusion = persisted["conclusion"]
    trigger = persisted["trigger"]
    root_path = root.resolve()
    return {
        "record_id": persisted["record_id"],
        "ticker": ticker,
        "company_name": persisted["company_name"],
        "analysis_cutoff": persisted["analysis_cutoff"],
        "created_at": persisted["created_at"],
        "mode": persisted["mode"],
        "trigger_type": trigger.get("type"),
        "trigger_summary": trigger.get("summary"),
        "baseline_record_id": persisted.get("baseline_record_id"),
        "stance": conclusion.get("stance"),
        "confidence": conclusion.get("confidence"),
        "summary": conclusion.get("summary"),
        "record_path": str(record_path.resolve().relative_to(root_path)),
        "report_path": str(report_path.resolve().relative_to(root_path)),
        "content_sha256": persisted["content_sha256"],
    }


def _update_index(root: Path, ticker: str, entry: dict[str, Any]) -> None:
    index = _read_index(root, ticker)
    records = [
        item
        for item in index["records"]
        if item.get("record_id") != entry["record_id"]
    ]
    records.append(entry)
    records.sort(
        key=lambda item: (
            str(item.get("analysis_cutoff", "")),
            str(item.get("created_at", "")),
        )
    )
    index["records"] = records
    atomic_write_json(index, _index_path(root, ticker))
    atomic_write_json(entry, root / ticker / "latest.json")


def save_research_record(
    bundle: dict[str, Any],
    *,
    root: Path = DEFAULT_RESEARCH_ROOT,
    auto_baseline: bool = True,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Persist a new immutable research record and update its ticker index."""

    working = deepcopy(bundle)
    ticker = _normalize_ticker(working.get("ticker"))
    cutoff = _parse_aware_timestamp(working.get("analysis_cutoff"), "analysis_cutoff")
    baseline = find_baseline_record(root, ticker, cutoff.isoformat())
    if auto_baseline and working.get("baseline_record_id") is None and baseline is not None:
        working["baseline_record_id"] = baseline["record_id"]

    normalized = validate_research_bundle(working)
    baseline_record_id = normalized.get("baseline_record_id")
    if baseline_record_id is not None:
        baseline_bundle = load_research_record(root, ticker, baseline_record_id)
        baseline_cutoff = _parse_aware_timestamp(
            baseline_bundle["analysis_cutoff"], "baseline.analysis_cutoff"
        )
        if baseline_cutoff >= cutoff:
            raise ResearchHistoryError(
                "baseline_record_id must refer to a record before analysis_cutoff"
            )

    _, content_sha256 = _canonical_content(normalized)
    record_id = _record_id(cutoff, content_sha256)
    record_dir = _record_directory(root, ticker, record_id)
    record_path = record_dir / "record.json"
    report_path = record_dir / "report.md"

    if record_path.exists() and report_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("content_sha256") != content_sha256:
            raise ResearchHistoryError(f"immutable record collision: {record_id}")
        entry = _build_index_entry(
            existing,
            root=root,
            ticker=ticker,
            record_path=record_path,
            report_path=report_path,
        )
        _update_index(root, ticker, entry)
        return {
            "status": "existing",
            "record_id": record_id,
            "record_path": str(record_path),
            "report_path": str(report_path),
            "baseline_record_id": existing.get("baseline_record_id"),
        }

    saved_at = created_at or datetime.now(tz=SHANGHAI_TZ)
    if saved_at.tzinfo is None or saved_at.utcoffset() is None:
        raise ResearchHistoryError("created_at must include timezone")
    report_markdown = normalized.pop("report_markdown")
    persisted = {
        **normalized,
        "record_id": record_id,
        "created_at": saved_at.astimezone(SHANGHAI_TZ).isoformat(timespec="seconds"),
        "content_sha256": content_sha256,
        "report_path": "report.md",
    }
    records_root = record_dir.parent
    records_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = records_root / f".{record_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    temporary_dir.mkdir()
    try:
        _atomic_write_text(report_markdown, temporary_dir / "report.md")
        atomic_write_json(persisted, temporary_dir / "record.json")
        os.replace(temporary_dir, record_dir)
    finally:
        if temporary_dir.exists():
            for child in temporary_dir.iterdir():
                child.unlink(missing_ok=True)
            temporary_dir.rmdir()

    entry = _build_index_entry(
        persisted,
        root=root,
        ticker=ticker,
        record_path=record_path,
        report_path=report_path,
    )
    _update_index(root, ticker, entry)
    return {
        "status": "saved",
        "record_id": record_id,
        "record_path": str(record_path),
        "report_path": str(report_path),
        "baseline_record_id": persisted.get("baseline_record_id"),
    }


def research_bundle_template(ticker: str, company_name: str) -> dict[str, Any]:
    """Return the executable archive schema used by the skill."""

    normalized_ticker = _normalize_ticker(ticker)
    now = datetime.now(tz=SHANGHAI_TZ).isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "ticker": normalized_ticker,
        "company_name": company_name,
        "analysis_cutoff": now,
        "mode": "full_coverage",
        "trigger": {
            "type": "initial",
            "summary": "首次建立研究基线",
            "source_refs": [],
        },
        "baseline_record_id": None,
        "conclusion": {
            "stance": "neutral",
            "confidence": "low",
            "summary": "",
        },
        "thesis": {
            "pillars": [],
            "strongest_counterargument": "",
            "falsifiers": [],
        },
        "scenarios": {
            "bear": {},
            "base": {},
            "bull": {},
        },
        "monitoring": [],
        "evidence_ledger": [],
        "data_snapshot": {},
        "revision": None,
        "report_markdown": "# 待填写研究报告",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save and retrieve immutable A-share research baselines for financial-report "
            "updates, event reviews, and belief iteration."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_RESEARCH_ROOT,
        help=f"Archive root (default: {DEFAULT_RESEARCH_ROOT}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser("template", help="Print a record template.")
    template_parser.add_argument("--ticker", required=True)
    template_parser.add_argument("--company-name", required=True)

    save_parser = subparsers.add_parser("save", help="Save one immutable JSON bundle.")
    save_parser.add_argument("input", type=Path)
    save_parser.add_argument(
        "--no-auto-baseline",
        action="store_true",
        help="Do not attach the latest prior record automatically.",
    )

    list_parser = subparsers.add_parser("list", help="List saved records.")
    list_parser.add_argument("--ticker", required=True)

    show_parser = subparsers.add_parser("show", help="Load one saved record.")
    show_parser.add_argument("--ticker", required=True)
    show_parser.add_argument("--record-id", default="latest")

    baseline_parser = subparsers.add_parser(
        "baseline", help="Find the latest record strictly before a cutoff."
    )
    baseline_parser.add_argument("--ticker", required=True)
    baseline_parser.add_argument("--before", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "template":
            result: Any = research_bundle_template(args.ticker, args.company_name)
        elif args.command == "save":
            payload = json.loads(args.input.read_text(encoding="utf-8"))
            result = save_research_record(
                payload,
                root=args.root,
                auto_baseline=not args.no_auto_baseline,
            )
        elif args.command == "list":
            result = {
                "ticker": _normalize_ticker(args.ticker),
                "records": list_research_records(args.root, args.ticker),
            }
        elif args.command == "show":
            result = load_research_record(
                args.root,
                args.ticker,
                args.record_id,
            )
        else:
            baseline = find_baseline_record(args.root, args.ticker, args.before)
            result = {
                "ticker": _normalize_ticker(args.ticker),
                "before": parse_analysis_time_for_output(args.before),
                "found": baseline is not None,
                "baseline": baseline,
            }
    except (OSError, json.JSONDecodeError, ResearchHistoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parse_analysis_time_for_output(value: str) -> str:
    """Normalize a CLI cutoff for stable JSON output."""

    return _parse_aware_timestamp(value, "before").isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
