#!/usr/bin/env python
"""Build the PIT weekly external-factor cache for long-entry research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.features.long_external_factors import build_weekly_external_factor_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/features/long_entry/weekly_training_v1.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/features/long_entry/weekly_external_v1.parquet",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signals = pd.read_parquet(args.dataset, columns=["date", "ts_code", "close"])
    factors, manifest = build_weekly_external_factor_cache(
        signals,
        raw_dir=PROJECT_ROOT / "data/raw",
        cache_path=args.output,
        manifest_path=args.output.with_suffix(".manifest.json"),
        force=args.force,
    )
    print(json.dumps({**manifest, "rows": len(factors), "path": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
