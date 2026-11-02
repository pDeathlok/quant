"""Retrain sequence tasks on redesigned samples and emit forward-only variables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from audit_overnight_right_provenance import sha256
from train_right_t012_full_registry import sample_weights
from train_right_t012_sequence import fit_sequence, sequence_predict, sequence_windows

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    output = root / "right_t012_full_sequence_v1"
    source = root / "right_t012_full_dataset_v4/dataset.parquet"
    check = json.loads(source.with_name("complete.json").read_text())
    if sha256(source) != check["dataset_sha256"]:
        raise ValueError("Unverified full dataset")
    audit = json.loads((root / "right_t012_full_dataset_audit_v1/complete.json").read_text())
    if audit["dataset_sha256"] != check["dataset_sha256"]:
        raise ValueError("Dataset admission audit mismatch")
    columns = ["symbol", "date", "planned_exit_date", "episode_age", "entry_close", "next_open", "next_high", "next_low", "next_close",
               "net_close_net_after_exit_audit", "limit1_net_after_exit_audit", "limit1_penetrated"]
    frame = pd.read_parquet(source, columns=columns)
    bars_path = root / "overnight_unfiltered_universe_v1/daily_bars.parquet"
    bars_digest = sha256(bars_path)
    bars = pd.read_parquet(bars_path)
    calendar = pd.DatetimeIndex(bars.date.unique()).sort_values()
    output.mkdir(exist_ok=False)
    folds = [("2024H2", "2024-07-01", "2025-01-01"), ("2025H1", "2025-01-01", "2025-07-01"),
             ("2025H2_2026", "2025-07-01", "2026-09-04")]
    protocol = dict(dataset_sha256=check["dataset_sha256"], bars_sha256=bars_digest, folds=folds,
                    architecture="Same32-session5-input GRU32, constrained OHLC+3event+5bucket outputs; six epochs; no volume prediction",
                    weighting="New within-date normalized inverse sqrt causal episode age",
                    prediction="Every prediction uses model trained only on exits before its entry segment; prediction readiness ignores future labels",
                    limits="No2024H1OOF prediction; downstream matched controls must exclude missing sequence rows; historical research only",
                    script_sha256=sha256(Path(__file__)), implementation_sha256=sha256(PROJECT_ROOT / "scripts/research/train_right_t012_sequence.py"))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    torch.set_num_threads(4)
    windows, ready = sequence_windows(bars, frame, calendar)
    del bars
    frame[["symbol", "date"]].assign(sequence_ready=ready).to_parquet(output / "window_coverage.parquet", index=False)
    print(f"Sequence windows: {ready.sum()}/{len(frame)} ready", flush=True)
    prices = frame[["next_open", "next_high", "next_low", "next_close"]].div(frame.entry_close, axis=0).sub(1).to_numpy(np.float32)
    close_net = frame.net_close_net_after_exit_audit
    limit_net = frame.limit1_net_after_exit_audit
    labels = np.column_stack([close_net.gt(0), frame.limit1_penetrated.fillna(False), close_net.lt(-.02)]).astype(np.float32)
    buckets = np.digitize(prices[:, 3], [-.03, -.01, .01, .03]).astype(np.int64)
    known = ready & close_net.notna().to_numpy() & limit_net.notna().to_numpy() & np.isfinite(prices).all(axis=1)
    values = np.full((len(frame), 12), np.nan, dtype=np.float32)
    records = []
    for name, start, end in folds:
        fit = known & frame.date.lt(start).to_numpy() & frame.planned_exit_date.lt(start).to_numpy()
        predict = ready & frame.date.ge(start).to_numpy() & frame.date.lt(end).to_numpy()
        if frame.loc[fit, "planned_exit_date"].max() >= frame.loc[predict, "date"].min():
            raise ValueError("Forward-fold timing violation")
        weights = np.ones(len(frame), dtype=np.float32)
        weights[fit] = sample_weights(frame.loc[fit]).astype(np.float32)
        model, losses = fit_sequence(windows, prices, labels, buckets, weights, fit)
        torch.save(model.state_dict(), output / f"{name}.pt")
        values[predict] = sequence_predict(model, windows[predict])
        records.append(dict(fold=name, fit=int(fit.sum()), predictions=int(predict.sum()),
                            max_fit_exit=str(frame.loc[fit, "planned_exit_date"].max()),
                            min_prediction_entry=str(frame.loc[predict, "date"].min()), losses=losses))
        (output / "folds.json").write_text(json.dumps(records, indent=2))
        print(f"Sequence fold {name} finished", flush=True)
    names = [f"seq_return_{p}" for p in ["open", "high", "low", "close"]] + ["seq_prob_profit", "seq_prob_hit1", "seq_prob_loss2"] + [f"seq_bucket_{i}" for i in range(5)]
    result = pd.concat([frame[["symbol", "date"]], pd.DataFrame(values, columns=names)], axis=1)
    for price in ["open", "high", "low", "close"]:
        result[f"seq_price_{price}"] = frame.entry_close * (1 + result[f"seq_return_{price}"])
    result.to_parquet(output / "sequence_variables.parquet", index=False)
    if sha256(source) != check["dataset_sha256"] or sha256(bars_path) != bars_digest:
        raise ValueError("Source changed during training")
    (output / "complete.json").write_text(json.dumps(dict(rows=len(result), predicted=int(result.seq_return_close.notna().sum()),
        features=names, dataset_sha256=check["dataset_sha256"], variables_sha256=sha256(output / "sequence_variables.parquet"),
        status="Retrained forward sequence variables ready; downstream fusion still requires training"), indent=2))


if __name__ == "__main__":
    main()
