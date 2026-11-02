"""Audit sell-at-close ambiguity without changing historical buy decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from train_overnight_direct import select

ROOT = Path(__file__).resolve().parents[2]


def exit_flags(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    price = np.ceil(frame.entry_close * (1 + threshold) * 100 - 1e-9) / 100
    reached = frame.next_high.ge(price + frame.entry_close * .001)
    known = frame[["entry_close", "next_high", "next_close"]].gt(0).all(axis=1)
    fallback = ~reached & known
    unknown_limit = fallback & ~frame.down_limit.gt(0)
    at_lower = fallback & frame.down_limit.gt(0) & frame.next_close.le(frame.down_limit + .0051)
    return pd.DataFrame({"close_fallback": fallback, "fallback_limit_unknown": unknown_limit,
                         "fallback_at_lower": at_lower}, index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refreshed-limits", action="store_true")
    parser.add_argument("--rebuilt-predictions", action="store_true")
    args = parser.parse_args()
    source = ROOT / "reports/research/overnight_same_day_limit_gated_v1"
    output = ROOT / ("reports/research/overnight_close_exit_limit_audit_v2" if args.refreshed_limits
                     else "reports/research/overnight_close_exit_limit_audit_v1")
    cache = ROOT / ("reports/research/overnight_exit_limit_history_2025q4_v1" if args.refreshed_limits
                    else "reports/research/overnight_limit_history")
    if args.rebuilt_predictions:
        if not args.refreshed_limits:
            raise ValueError("Rebuilt prediction audit requires refreshed exit limits")
        source = ROOT / "reports/research/overnight_same_day_rebuilt_limit_gated_v1"
        output = ROOT / "reports/research/overnight_close_exit_limit_audit_v3"
    if not (source / "complete.json").exists():
        raise ValueError("Prediction run incomplete")
    if not (cache / "complete.json").exists():
        raise ValueError("Limit cache incomplete")
    output.mkdir(exist_ok=False)
    label_path = source / "limit_label_source.parquet"
    labels = pd.read_parquet(label_path)
    hashes = {str(label_path): hashlib.sha256(label_path.read_bytes()).hexdigest()}
    parts = []
    for path in sorted(cache.glob("*.parquet")):
        parts.append(pd.read_parquet(path, columns=["ts_code", "trade_date", "down_limit"]))
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    limits = pd.concat(parts, ignore_index=True).rename(columns={"ts_code": "symbol", "trade_date": "exit_date"})
    limits["exit_date"] = pd.to_datetime(limits.exit_date)
    labels = labels.merge(limits, on=["symbol", "exit_date"], how="left", validate="many_to_one")
    rows = []
    for name, threshold in [("limit05", .005), ("limit1", .01)]:
        flags = exit_flags(labels, threshold)
        audit = pd.concat([labels[["symbol", "date"]], flags], axis=1)
        path = source / f"{name}_predictions.parquet"
        pred = pd.read_parquet(path)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Fixed, unfiltered daily Top1/Top5 only; no new probability search.
        for top_k in [1, 5]:
            picks = select(pred, 0., -np.inf, top_k)
            joined = picks.merge(audit, on=["symbol", "date"], how="left", validate="one_to_one")
            pd.testing.assert_frame_equal(picks.reset_index(drop=True), joined[picks.columns])
            if joined[flags.columns].isna().any().any():
                raise ValueError("Missing selected-row audit")
            joined.to_parquet(output / f"{name}_top{top_k}.parquet", index=False)
            ambiguous = joined.fallback_at_lower | joined.fallback_limit_unknown
            rows.append({"target": name, "top_k": top_k, "signals": len(joined),
                         "dates": int(joined.date.nunique()), "unknown_labels": int(joined.net.isna().sum()),
                         **{c: int(joined[c].sum()) for c in flags},
                         "original_mean_net20": float(joined.net.mean()),
                         "ambiguous_known_mean_net20": float(joined.loc[ambiguous, "net"].mean()),
                         "warning": "Known daily marks at/without lower limit are not verified sell fills; "
                                    "keep candidates, do not drop these losses to improve win rate"})
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Audit source changed")
    (output / "manifest.json").write_text(json.dumps({"sources": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "postdecision execution audit; no retraining, no outcome-based candidate removal"}, indent=2))
    (output / "complete.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
