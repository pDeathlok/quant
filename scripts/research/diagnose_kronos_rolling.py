#!/usr/bin/env python
"""Compare each stored rolling head's train fit and subsequent predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from quant.research.kronos_multiscale import SequenceSubscores, predict_subscores
from quant.research.kronos_rolling import mature_window


def auc(frame, probability):
    target = frame.net_return.gt(0)
    return (
        float(roc_auc_score(target, probability))
        if len(frame) and target.nunique() == 2
        else float("nan")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source
    output = args.output
    torch.set_num_threads(4)
    embeddings = np.load(source / "embeddings.npy", mmap_mode="r")
    audit = pd.read_csv(output / "refit_audit.csv", parse_dates=["refit_date"])
    records = []
    for side in ["left", "right"]:
        frame = pd.read_parquet(source / f"{side}_scored_events.parquet")
        for frequency in ["monthly", "weekly"]:
            prediction = pd.read_parquet(output / f"{side}_{frequency}_predictions.parquet")
            for event in audit[audit.side.eq(side) & audit.frequency.eq(frequency)].itertuples():
                train = mature_window(frame, event.refit_date)
                path = output / "models" / side / frequency / f"{event.refit_date:%Y%m%d}_head.pt"
                if not path.exists():
                    continue
                head = SequenceSubscores(embeddings.shape[1])
                head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                head.eval()
                fitted = predict_subscores(head, embeddings[train.embedding_row])[:, 1]
                future = prediction[prediction.refit_date.eq(event.refit_date)]
                records.append(
                    {
                        "side": side,
                        "frequency": frequency,
                        "refit_date": event.refit_date,
                        "train_rows": len(train),
                        "forecast_rows": len(future),
                        "train_auc": auc(train, fitted),
                        "forecast_auc": auc(future, future.rolling_sequence),
                        "train_win_rate": float(train.net_return.gt(0).mean()),
                        "forecast_win_rate": float(future.net_return.gt(0).mean()),
                    }
                )
    result = pd.DataFrame(records)
    result.to_csv(output / "head_fit_diagnostic.csv", index=False)
    print(
        result.groupby(["side", "frequency"])[
            ["train_auc", "forecast_auc", "train_rows", "forecast_rows"]
        ]
        .median()
        .to_string()
    )


if __name__ == "__main__":
    main()
