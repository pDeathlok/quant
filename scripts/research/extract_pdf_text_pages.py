#!/usr/bin/env python3
"""Extract a PDF into UTF-8 text while retaining explicit one-based page markers."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reader = PdfReader(args.input)
    parts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        parts.append(f"\n===== PDF PAGE {page_number} =====\n{text.rstrip()}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(parts), encoding="utf-8")
    print(f"{args.input}: pages={len(reader.pages)} chars={sum(map(len, parts))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
