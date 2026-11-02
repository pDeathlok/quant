"""Add point-in-time 90-stock context without changing existing labels or gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant.research.overnight_context import add_watchlist_context

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "reports/research/overnight_limit_exit_dataset_v1"
    output = ROOT / "reports/research/overnight_context_dataset_v1"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["smoke_only"] or not (source / "complete.json").exists():
        raise ValueError("Complete full dataset required")
    frame = pd.read_parquet(source / "dataset.parquet")
    result = add_watchlist_context(frame)
    pd.testing.assert_frame_equal(frame, result[frame.columns])
    output.mkdir(exist_ok=False)
    manifest.update(parent_dataset_sha256=hashlib.sha256((source / "dataset.parquet").read_bytes()).hexdigest(),
        context="same-date 14:55 available watchlist inputs, all observed stocks regardless of entry gate;"
        " no labels, full-day market marks or future quality gates; NOT whole-market breadth",
        new_features=[c for c in result if c.startswith("x_ctx_")])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    result.to_parquet(output / "dataset.parquet", index=False)
    summary = {"rows": len(result), "stocks": result.symbol.nunique(),
               "new_features": len(manifest["new_features"]), "original_columns_unchanged": True}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
