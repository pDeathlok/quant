#!/usr/bin/env python
"""Score the seven live legacy Z-skill artifacts on canonical-event overlap.

This is an optional compatibility baseline.  It deliberately reads factors
from ``z_skill_model_dataset.parquet`` because the artifacts are incompatible
with the current causal project-factor schema used by the unified models.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from quant.research.right_side_legacy_artifact_baseline import (
    LEGACY_LABEL_CONTRACT,
    LEGACY_Z_ARTIFACT_SIGNALS,
    aggregate_legacy_event_predictions,
    build_legacy_overlap_rows,
    evaluate_legacy_event_predictions,
    legacy_overlap_coverage,
    load_legacy_z_artifacts,
    required_legacy_features,
    score_legacy_overlap_rows,
)


DEFAULT_EVENTS = PROJECT_ROOT / "data/research/right_side_unified/unified_right_side_dataset.parquet"
DEFAULT_LABELS = PROJECT_ROOT / "data/research/right_side_unified/unified_right_side_labels.parquet"
DEFAULT_LEGACY_FACTORS = PROJECT_ROOT / "data/features/z_skill_model_dataset.parquet"
DEFAULT_MODELS = PROJECT_ROOT / "models/research/z_skill"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/research/right_side_unified/legacy_artifact_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--legacy-factors", type=Path, default=DEFAULT_LEGACY_FACTORS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--entry-mode", choices=["next_open", "next_close"], default="next_open")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["hit_up5", "hit_up8", "good_path5"],
    )
    parser.add_argument("--top-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models, contracts = load_legacy_z_artifacts(args.model_dir)
    events = pd.read_parquet(
        args.events,
        columns=["symbol", "date", *LEGACY_Z_ARTIFACT_SIGNALS],
    )
    import pyarrow.parquet as pq

    label_schema = set(pq.ParquetFile(args.labels).schema.names)
    labels = pd.read_parquet(
        args.labels,
        columns=[
            column
            for column in (
                "symbol",
                "date",
                "entry_mode",
                "horizon",
                "mature",
                "locked_limit_up",
                *args.targets,
                "entry_date",
                "terminal_return",
                "mfe",
                "mae",
            )
            if column in label_schema
        ],
    )
    legacy_schema = set(pq.ParquetFile(args.legacy_factors).schema.names)
    legacy_factors = pd.read_parquet(
        args.legacy_factors,
        columns=[
            column
            for column in (
                "symbol",
                "date",
                *LEGACY_Z_ARTIFACT_SIGNALS,
                *required_legacy_features(models),
            )
            if column in legacy_schema
        ],
    )
    overlap = build_legacy_overlap_rows(
        events,
        labels,
        legacy_factors,
        models,
        entry_mode=args.entry_mode,
        horizon=args.horizon,
        targets=args.targets,
    )
    scored = score_legacy_overlap_rows(overlap, models)
    event_predictions = aggregate_legacy_event_predictions(scored)
    coverage = legacy_overlap_coverage(scored)
    metrics = evaluate_legacy_event_predictions(
        event_predictions,
        targets=args.targets,
        top_fraction=args.top_fraction,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_contract_path = atomic_write_csv(
        contracts,
        args.output_dir / "artifact_contracts.csv",
        index=False,
    )
    coverage_path = atomic_write_csv(
        coverage,
        args.output_dir / "overlap_coverage.csv",
        index=False,
    )
    long_path = atomic_write_parquet(
        scored,
        args.output_dir / "signal_overlap_predictions.parquet",
        index=False,
        compression="zstd",
    )
    event_path = atomic_write_parquet(
        event_predictions,
        args.output_dir / "event_predictions.parquet",
        index=False,
        compression="zstd",
    )
    metrics_path = atomic_write_csv(
        metrics,
        args.output_dir / "metrics.csv",
        index=False,
    )
    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": "right-side-legacy-artifact-overlap-v1",
        "scope": {
            "entry_mode": args.entry_mode,
            "horizon": args.horizon,
            "targets": args.targets,
            "top_fraction": args.top_fraction,
        },
        "inputs": {
            "events": str(args.events),
            "labels": str(args.labels),
            "legacy_factors": str(args.legacy_factors),
            "model_dir": str(args.model_dir),
        },
        "outputs": {
            "artifact_contracts": str(artifact_contract_path),
            "overlap_coverage": str(coverage_path),
            "signal_overlap_predictions": str(long_path),
            "event_predictions": str(event_path),
            "metrics": str(metrics_path),
        },
        "rows": {
            "canonical_event_signal_rows": int(len(overlap)),
            "legacy_factor_overlap_rows": int(overlap["legacy_factor_available"].sum()),
            "scored_event_signal_rows": int(scored["legacy_scored"].sum()),
            "scored_unique_events": int(len(event_predictions)),
        },
        "legacy_label_contract": dict(LEGACY_LABEL_CONTRACT),
        "interpretation_limits": [
            "Artifacts live under models/research/z_skill and have no immutable production manifest; 'legacy production baseline' means current selector-consumed research artifacts.",
            "Only exact symbol/date overlap with persisted legacy-factor rows is scored; coverage is not a full canonical-event replay.",
            "Historical and canonical signal predicates/timing differ, and the historical cache imposed a five-session cooldown.",
            "up5/up8 are only near-equivalent to next_open horizon-5 hit labels after new maturity/tradability gates; down3 is a different target.",
            "2024 participated in fit/early-stop; 2025+ was training OOT but was later inspected to choose playbooks. No period is an untouched final test for artifact selection.",
            "The fixed quality score is descriptive and is not a newly tuned trading threshold.",
        ],
    }
    manifest_path = atomic_write_json(manifest, args.output_dir / "manifest.json")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "overlap_rows": len(overlap),
                "scored_rows": int(scored["legacy_scored"].sum()),
                "events": len(event_predictions),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
