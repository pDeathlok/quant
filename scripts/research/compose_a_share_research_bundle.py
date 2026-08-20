#!/usr/bin/env python3
"""Compose one history bundle from JSON metadata and a Markdown report."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any


def compose_bundle(metadata_path: Path, report_path: Path) -> dict[str, Any]:
    """Load metadata and inject the exact Markdown body for immutable archiving."""

    payload: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "report_markdown" in payload:
        raise ValueError("metadata must not contain report_markdown")
    report = report_path.read_text(encoding="utf-8")
    if not report.strip():
        raise ValueError("report must not be empty")
    payload["report_markdown"] = report
    return payload


def atomic_write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Atomically persist UTF-8 JSON without partial output files."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    atomic_write_json(compose_bundle(args.metadata, args.report), args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
