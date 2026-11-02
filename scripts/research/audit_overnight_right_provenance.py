"""Compare current right-side family flags without reading outcome labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant.core.paths import PROJECT_ROOT
from quant.research.right_side_unified_signals import (
    B2_FAMILY_SOURCE_COLUMNS,
    B3_FAMILY_SOURCE_COLUMNS,
    FAMILY_DIRECT_SOURCE_COLUMNS,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    family_path = PROJECT_ROOT / "data/features/b1/b1_family_rule_candidates.parquet"
    right_path = PROJECT_ROOT / (
        "data/research/right_side_unified_current_v6_20260824/"
        "unified_right_side_dataset.parquet"
    )
    output = PROJECT_ROOT / "reports/research/overnight_right_provenance_v1"
    if output.exists():
        raise FileExistsError(output)
    sources = {str(path): sha256(path) for path in (family_path, right_path)}
    mapping = {
        "B2": list(B2_FAMILY_SOURCE_COLUMNS),
        "B3": list(B3_FAMILY_SOURCE_COLUMNS),
        **{key: [value] for key, value in FAMILY_DIRECT_SOURCE_COLUMNS.items()},
    }
    keys = ["symbol", "date"]
    family = pd.read_parquet(
        family_path, columns=keys + [col for cols in mapping.values() for col in cols]
    )
    right = pd.read_parquet(right_path, columns=keys + list(mapping))
    frames = []
    for frame in (family, right):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.loc[frame["date"].between("2025-07-01", "2025-12-31")].copy()
        if frame.duplicated(keys).any() or frame[keys].isna().any().any():
            raise ValueError("Duplicate or missing source keys")
        frames.append(frame)
    family, right = frames
    flags = family[keys].copy()
    for signal, columns in mapping.items():
        flags[signal] = family[columns].fillna(False).astype(bool).any(axis=1)
    merged = flags.merge(right, on=keys, how="outer", suffixes=("_family", "_right"))
    differences = []
    counts = []
    for signal in mapping:
        left = merged[signal + "_family"].fillna(False).astype(bool)
        cached = merged[signal + "_right"].fillna(False).astype(bool)
        different = left.ne(cached)
        detail = merged.loc[different, keys].copy()
        detail["signal"] = signal
        detail["family_flag"] = left.loc[different]
        detail["right_flag"] = cached.loc[different]
        differences.append(detail)
        counts.append({
            "signal": signal,
            "family": int(left.sum()),
            "right": int(cached.sum()),
            "family_only": int((left & ~cached).sum()),
            "right_only": int((cached & ~left).sum()),
        })
    for path, original in sources.items():
        if sha256(Path(path)) != original:
            raise RuntimeError(f"Source changed during audit: {path}")
    differences_frame = pd.concat(differences, ignore_index=True)
    result = {
        "scope": "Four family-sourced right-side signals, 2025 H2, no labels read",
        "sources_sha256": sources,
        "family_rows": len(family),
        "right_rows": len(right),
        "counts": counts,
        "different_signal_keys": len(differences_frame),
        "different_event_keys": len(differences_frame[keys].drop_duplicates()),
        "consistent": differences_frame.empty,
        "limitations": [
            "Current family cache is not proven identical to the build-time cache.",
            "Differences do not by themselves establish lookahead or faulty formulas.",
            "Legacy family builder can silently skip failed symbols; coverage unproven.",
            "Agreement would not prove point-in-time correctness or complete coverage.",
            "Ten directly computed z signals require a separate prefix/cache replay.",
        ],
        "next_action": "Replay differing symbols from canonical daily history before backtesting.",
    }
    output.mkdir(parents=True)
    differences_frame.to_csv(output / "differences.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "counts.csv", index=False)
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
