"""Keep fold A's sequence learner frozen before its calibration/selection data."""

from __future__ import annotations

import json

import pandas as pd
import torch
from audit_overnight_right_provenance import sha256
from train_right_t012_sequence import PriceSequence, sequence_predict, sequence_windows

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    sequence_root = root / "right_t012_full_sequence_v1"
    output = root / "right_t012_full_sequence_fold_a_v1"
    check = json.loads((sequence_root / "complete.json").read_text())
    original = sequence_root / "sequence_variables.parquet"
    if sha256(original) != check["variables_sha256"]:
        raise ValueError("Sequence source changed")
    dataset = root / "right_t012_full_dataset_v4/dataset.parquet"
    if sha256(dataset) != check["dataset_sha256"]:
        raise ValueError("Dataset changed")
    folds = json.loads((sequence_root / "folds.json").read_text())
    fold = next(f for f in folds if f["fold"] == "2025H1")
    if pd.Timestamp(fold["max_fit_exit"]) >= pd.Timestamp("2025-01-01"):
        raise ValueError("Fold A feature learner trained beyond2024")
    checkpoint = sequence_root / "2025H1.pt"
    checkpoint_hash = sha256(checkpoint)
    frame = pd.read_parquet(dataset, columns=["symbol", "date", "entry_close"])
    requested = frame.loc[frame.date.ge("2025-07-01") & frame.date.lt("2026-01-01")].reset_index(drop=True)
    bars = pd.read_parquet(root / "overnight_unfiltered_universe_v1/daily_bars.parquet")
    calendar = pd.DatetimeIndex(bars.date.unique()).sort_values()
    windows, ready = sequence_windows(bars, requested, calendar)
    torch.set_num_threads(4)
    model = PriceSequence()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    values = sequence_predict(model, windows[ready])
    result = pd.read_parquet(original).set_index(["symbol", "date"])
    keys = pd.MultiIndex.from_frame(requested.loc[ready, ["symbol", "date"]])
    result.loc[keys, check["features"]] = values
    entry = frame.set_index(["symbol", "date"]).entry_close
    for price in ["open", "high", "low", "close"]:
        result[f"seq_price_{price}"] = entry * (1 + result[f"seq_return_{price}"])
    output.mkdir(exist_ok=False)
    result.reset_index().to_parquet(output / "sequence_variables.parquet", index=False)
    if sha256(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed")
    (output / "complete.json").write_text(json.dumps(dict(dataset_sha256=check["dataset_sha256"],
        variables_sha256=sha256(output / "sequence_variables.parquet"), original_sha256=check["variables_sha256"],
        features=check["features"], replaced_predictions=int(ready.sum()), learner_max_exit=fold["max_fit_exit"],
        checkpoint_sha256=checkpoint_hash,
        contract="For2025H2 outer fold only: feature learner frozen at2024 for all2025 calibration/selection/diagnostic dates;2026 rows unused"), indent=2))
    print(f"Fold A frozen2024 checkpoint: {ready.sum()} H2 predictions", flush=True)


if __name__ == "__main__":
    main()
