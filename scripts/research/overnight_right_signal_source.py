"""Require a complete, hash-verified current-day right signal rebuild."""

import json
from pathlib import Path

import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.research.right_side_unified_signals import CANONICAL_RIGHT_SIDE_SIGNALS


def load_events(source: Path) -> tuple[pd.DataFrame, dict]:
    completion = source.with_name("complete.json")
    if not completion.exists():
        raise ValueError("Current-day right signal rebuild unfinished")
    summary = json.loads(completion.read_text())
    if sha256(source) != summary["dataset_sha256"]:
        raise ValueError("Right signal dataset hash mismatch")
    frame = pd.read_parquet(source)
    signals = list(CANONICAL_RIGHT_SIDE_SIGNALS)
    if len(frame) != summary["rows"] or frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Right signal dataset row/key mismatch")
    counts = {s: int(frame[s].sum()) for s in signals}
    if counts != summary["signals"]:
        raise ValueError("Right signal counts mismatch")
    audit = {"all_daily_rows": len(frame), "signal_unavailable": {s: int(frame[s].isna().sum()) for s in signals},
             "unavailable_policy": "Unknown signal is not selected; counts remain explicit, not false observations"}
    events = frame.loc[frame[signals].eq(True).any(axis=1), ["symbol", "date", *signals]].copy()
    return events, audit
