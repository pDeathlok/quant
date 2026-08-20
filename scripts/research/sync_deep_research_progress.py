#!/usr/bin/env python3
"""Initialize or update the 112-company individual Deep research ledger."""

from __future__ import annotations

import argparse
import csv
import os
import uuid
from pathlib import Path


FIELDNAMES = (
    "sequence",
    "ts_code",
    "name",
    "broad_industry",
    "status",
    "deep_report_path",
    "record_id",
    "analysis_cutoff",
    "gqs_r",
    "gqs_f",
    "classification",
    "base_upside",
    "validated_at",
    "notes",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read UTF-8 CSV rows or return an empty ledger."""

    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sync_rows(
    universe_path: Path,
    progress_path: Path,
    update: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Reconcile the ledger to universe order and optionally update one ticker."""

    existing = {row["ts_code"]: row for row in read_rows(progress_path)}
    universe = read_rows(universe_path)
    rows: list[dict[str, str]] = []
    for sequence, company in enumerate(universe, start=1):
        ticker = company["ts_code"]
        prior = existing.get(ticker, {})
        row = {field: prior.get(field, "") for field in FIELDNAMES}
        row.update(
            {
                "sequence": str(sequence),
                "ts_code": ticker,
                "name": company["name"],
                "broad_industry": company["broad_industry"],
                "status": prior.get("status") or "pending",
            }
        )
        if update is not None and ticker == update["ts_code"]:
            row.update(update)
        rows.append(row)
    if update is not None and not any(
        row["ts_code"] == update["ts_code"] for row in rows
    ):
        raise ValueError(f"ticker not found in universe: {update['ts_code']}")
    return rows


def atomic_write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    """Atomically replace the progress CSV after a complete write."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--ticker")
    parser.add_argument("--status", choices=("pending", "in_progress", "completed"))
    parser.add_argument("--report-path", default="")
    parser.add_argument("--record-id", default="")
    parser.add_argument("--analysis-cutoff", default="")
    parser.add_argument("--gqs-r", default="")
    parser.add_argument("--gqs-f", default="")
    parser.add_argument("--classification", default="")
    parser.add_argument("--base-upside", default="")
    parser.add_argument("--validated-at", default="")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.ticker) != bool(args.status):
        raise ValueError("--ticker and --status must be provided together")
    update = None
    if args.ticker:
        update = {
            "ts_code": args.ticker,
            "status": args.status,
            "deep_report_path": args.report_path,
            "record_id": args.record_id,
            "analysis_cutoff": args.analysis_cutoff,
            "gqs_r": args.gqs_r,
            "gqs_f": args.gqs_f,
            "classification": args.classification,
            "base_upside": args.base_upside,
            "validated_at": args.validated_at,
            "notes": args.notes,
        }
    rows = sync_rows(args.universe, args.progress, update)
    atomic_write_rows(rows, args.progress)
    completed = sum(row["status"] == "completed" for row in rows)
    print(f"completed={completed} total={len(rows)} progress={args.progress}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
