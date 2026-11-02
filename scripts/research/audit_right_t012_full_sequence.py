"""Measure forward OHLC quality against a no-price-change reference."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    directory = root / "right_t012_full_sequence_v1"
    check = json.loads((directory / "complete.json").read_text())
    path = directory / "sequence_variables.parquet"
    assert sha256(path) == check["variables_sha256"]
    dataset = root / "right_t012_full_dataset_v4/dataset.parquet"
    assert sha256(dataset) == check["dataset_sha256"]
    keys = ["symbol", "date"]
    base = pd.read_parquet(dataset, columns=[*keys, "entry_close", "next_open", "next_high", "next_low", "next_close"])
    seq = pd.read_parquet(path).drop(columns=["entry_close"], errors="ignore")
    frame = base.merge(seq, on=keys, validate="one_to_one")
    frame["quarter"] = frame.date.dt.to_period("Q").astype(str)
    records = []
    for quarter, rows in frame.groupby("quarter"):
        for price in ["open", "high", "low", "close"]:
            actual = rows[f"next_{price}"] / rows.entry_close - 1
            forecast = rows[f"seq_return_{price}"]
            valid = np.isfinite(actual) & np.isfinite(forecast)
            actual, forecast = actual[valid], forecast[valid]
            if not len(actual):
                continue
            assert np.allclose(rows.loc[valid, f"seq_price_{price}"], rows.loc[valid, "entry_close"] * (1 + forecast))
            records.append(dict(quarter=quarter, price=price, rows=len(actual), mae_return=float((forecast - actual).abs().mean()), unchanged_mae_return=float(actual.abs().mean()), mean_forecast=float(forecast.mean()), mean_actual=float(actual.mean()), direction_accuracy=float(np.sign(actual).eq(np.sign(forecast)).mean()), always_positive_accuracy=float(actual.gt(0).mean()), always_negative_accuracy=float(actual.lt(0).mean())))
    pd.DataFrame(records).to_csv(directory / "price_diagnostics.csv", index=False)
    (directory / "price_audit_complete.json").write_text(json.dumps(dict(variables_sha256=check["variables_sha256"], groups=len(records), scope="Forward variable price identity and descriptive quarterly accuracy; no threshold selection, raw nominal price target"), indent=2))
    print(f"Checked {len(records)} quarter/price groups")


if __name__ == "__main__":
    main()
