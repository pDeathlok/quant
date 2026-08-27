"""Build historical selector factor archives without changing active latest snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
from quant.application.selector_ranking import DEFAULT_SELECTOR_RANKING_CONFIG
from quant.data.atomic_io import atomic_write_json
from quant.routine.left_side_unified_production import build_left_side_production_features
from quant.routine.right_side_unified_production import build_right_side_unified_production_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", required=True)
    args = parser.parse_args()
    target = pd.Timestamp(args.target_date).normalize()
    configs = {"left": DEFAULT_LEFT_SIDE_RANKING_CONFIG, "right": DEFAULT_SELECTOR_RANKING_CONFIG}
    before = {}
    for config in configs.values():
        latest = pd.read_parquet(config.paths.feature_output, columns=["date"])
        if latest.empty or pd.to_datetime(latest["date"]).min().normalize() <= target:
            raise RuntimeError("archive backfill must precede both active feature snapshots")
        for path in (config.paths.feature_output, config.paths.feature_manifest):
            before[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    results = {}
    builders = {"left": build_left_side_production_features, "right": build_right_side_unified_production_features}
    for side, config in configs.items():
        archive = config.paths.feature_output.with_name(f"{target:%Y%m%d}_features.parquet")
        historical = replace(config, paths=replace(
            config.paths, feature_output=archive, feature_manifest=archive.with_suffix(".json"),
        ))
        print(f"building {side} archive for {target.date()}", flush=True)
        results[side] = builders[side](target.date().isoformat(), config=historical)
    changed = [str(path) for path, digest in before.items() if hashlib.sha256(path.read_bytes()).hexdigest() != digest]
    report = {
        "status": "success" if not changed else "failed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_date": target.date().isoformat(), "latest_files_changed": changed,
        "results": results,
    }
    atomic_write_json(report, PROJECT_ROOT / "reports/production/selector_factor_archives" / f"{target:%Y%m%d}.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if changed:
        raise RuntimeError(f"active latest snapshots changed during backfill: {changed}")


if __name__ == "__main__":
    main()
