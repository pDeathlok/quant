"""Freeze locked-model predictions and picks before constructing exit outcomes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from threadpoolctl import threadpool_limits
from train_overnight_direct import select

from quant.core.paths import PROJECT_ROOT


def entry_gate(frame: pd.DataFrame) -> pd.DataFrame:
    prices = frame[[f"today_raw_{c}" for c in ["open", "high", "low", "close", "pre_close", "vol"]]]
    valid = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
    valid &= frame.today_raw_high.ge(frame[["today_raw_open", "today_raw_close"]].max(axis=1) - 1e-8)
    valid &= frame.today_raw_low.le(frame[["today_raw_open", "today_raw_close"]].min(axis=1) + 1e-8)
    valid &= np.isfinite(frame.up_limit) & frame.up_limit.gt(0)
    result = frame[["symbol", "date", "up_limit", "down_limit"]].copy()
    result["unknown_entry"] = ~valid
    result["at_upper"] = frame.today_raw_close.ge(frame.up_limit - .0051) & valid
    result["eligible"] = valid & ~result.at_upper
    return result


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    lock_path = root / "overnight_event_extension_lock_v1/protocol.json"
    lock = json.loads(lock_path.read_text())
    hashes = {**lock["sources"], str(lock_path): sha256(lock_path)}
    input_path = root / "overnight_event_extension_inputs_v1/dataset.parquet"
    input_proof = input_path.parent / "complete.json"
    proof = json.loads(input_proof.read_text())
    if sha256(input_path) != proof["dataset_sha256"] or proof["different_cells"]:
        raise ValueError("Unverified inputs")
    hashes.update({str(p): sha256(p) for p in [input_path, input_proof]})
    frame = pd.read_parquet(input_path)
    frame = frame.loc[frame.date.between(lock["entry_start"], lock["entry_end"])].reset_index(drop=True)
    limit_root = root / "overnight_event_extension_limits_v1"
    limit_complete = limit_root / "complete.json"
    if json.loads(limit_complete.read_text())["missing_keys"]:
        raise ValueError("Missing entry limits")
    hashes[str(limit_complete)] = sha256(limit_complete)
    coverage_path = limit_root / "coverage.csv"
    hashes[str(coverage_path)] = sha256(coverage_path)
    coverage = pd.read_csv(coverage_path, dtype={"date": str}).set_index("date")
    parts = []
    for date in sorted(frame.date.unique()):
        day = pd.Timestamp(date).strftime("%Y%m%d")
        path = limit_root / f"{day}.parquet"
        hashes[str(path)] = coverage.loc[day, "sha256"]
        part = pd.read_parquet(path)
        if not part.trade_date.astype(str).eq(day).all() or part.ts_code.duplicated().any():
            raise ValueError("Invalid limit keys")
        part["date"] = pd.Timestamp(date)
        parts.append(part.rename(columns={"ts_code": "symbol"})[["symbol", "date", "up_limit", "down_limit"]])
    frame = frame.merge(pd.concat(parts), on=["symbol", "date"], how="left", validate="one_to_one")
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Changed locked source: {path}")
    model = joblib.load(lock["model"])
    if model["columns"] != lock["features"] or proof["features"] != lock["features"]:
        raise ValueError("Locked input order changed")
    gate = entry_gate(frame)
    x = frame.loc[gate.eligible, model["columns"]].astype("float32").replace([np.inf, -np.inf], np.nan)
    predictions = frame.loc[gate.eligible, ["symbol", "date"]].copy()
    with threadpool_limits(limits=4):
        predictions["probability"] = model["calibrator"].predict(model["classifier"].predict_proba(x)[:, 1])
        predictions["expected_net"] = model["regressor"].predict(x)
        predictions["risk"] = model["risk_calibrator"].predict(model["risk"].predict_proba(x)[:, 1])
    rule = lock["rule"]
    if rule["minimum_expected_net"] is not None:
        raise ValueError("Unexpected rule")
    picks = select(predictions, rule["probability"], -np.inf, rule["top_k"], rule["maximum_risk"])
    output = root / "overnight_event_extension_predictions_v1"
    output.mkdir(exist_ok=False)
    gate.to_parquet(output / "entry_gate.parquet", index=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    picks.to_parquet(output / "picks.parquet", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed during prediction: {path}")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    summary = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "candidate_rows": len(frame),
               "eligible": int(gate.eligible.sum()), "unknown_entry": int(gate.unknown_entry.sum()),
               "at_upper": int(gate.at_upper.sum()), "picks": len(picks),
               "predictions_sha256": sha256(output / "predictions.parquet"),
               "picks_sha256": sha256(output / "picks.parquet"), "script_sha256": sha256(Path(__file__)),
               "status": "Frozen without reading next-session outcomes; not independent confirmation"}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
