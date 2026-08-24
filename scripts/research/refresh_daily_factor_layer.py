#!/usr/bin/env python
"""Refresh the versioned daily factor layer shared by all selector rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.features.daily_factor_layer import DEFAULT_FACTOR_ROOT, refresh_daily_factor_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-dir", type=Path, default=PROJECT_ROOT / "data/raw/daily")
    parser.add_argument("--factor-root", type=Path, default=PROJECT_ROOT / DEFAULT_FACTOR_ROOT)
    parser.add_argument("--incremental-start-date", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="override configs/factors/governance.json calculator worker policy",
    )
    parser.add_argument(
        "--executor",
        choices=["threads", "processes"],
        default=None,
        help="override configs/factors/governance.json calculator executor",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = refresh_daily_factor_layer(
        daily_dir=args.daily_dir,
        factor_root=args.factor_root,
        incremental_start_date=args.incremental_start_date,
        workers=args.workers,
        executor_type=args.executor,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
