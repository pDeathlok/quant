#!/usr/bin/env python
"""Evaluate one A/B-frozen right-side candidate on C-only CSV artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json
from quant.research.right_side_c_confirmation import (
    c_confirmation_payload,
    evaluate_c_confirmation,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-decision", type=Path, required=True)
    parser.add_argument("--paired-c", type=Path, required=True)
    parser.add_argument("--metrics-c", type=Path, required=True)
    parser.add_argument("--signal-metrics-c", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = (args.paired_c, args.metrics_c, args.signal_metrics_c)
    if any(not path.is_file() for path in (args.frozen_decision, *input_paths)):
        raise FileNotFoundError("all frozen/C confirmation inputs must exist")
    # Operational preregistration guard: the frozen file must predate every
    # C-only extract.  The payload policy validation is the semantic guard.
    frozen_mtime = args.frozen_decision.stat().st_mtime_ns
    if any(frozen_mtime >= path.stat().st_mtime_ns for path in input_paths):
        raise ValueError("frozen A/B decision must predate every C-only input")

    frozen_payload = json.loads(args.frozen_decision.read_text(encoding="utf-8"))
    evaluation = evaluate_c_confirmation(
        frozen_payload,
        pd.read_csv(args.paired_c),
        pd.read_csv(args.metrics_c),
        pd.read_csv(args.signal_metrics_c),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks_path = atomic_write_csv(
        evaluation.checks,
        args.output_dir / "c_confirmation_checks.csv",
        index=False,
    )
    signal_path = atomic_write_csv(
        evaluation.signal_deltas,
        args.output_dir / "c_confirmation_signal_deltas.csv",
        index=False,
    )
    payload = c_confirmation_payload(evaluation)
    payload["evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["inputs"] = {
        "frozen_decision": str(args.frozen_decision),
        "paired_c": str(args.paired_c),
        "metrics_c": str(args.metrics_c),
        "signal_metrics_c": str(args.signal_metrics_c),
    }
    payload["input_sha256"] = {
        path.name: _sha256(path) for path in (args.frozen_decision, *input_paths)
    }
    payload["outputs"] = {
        "checks": str(checks_path),
        "signal_deltas": str(signal_path),
    }
    output = atomic_write_json(
        payload,
        args.output_dir / "c_confirmation_decision.json",
    )
    print(json.dumps({"decision": evaluation.decision, "output": str(output)}))


if __name__ == "__main__":
    main()
