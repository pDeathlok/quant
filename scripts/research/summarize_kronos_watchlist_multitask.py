"""Paired date-block ablation deltas on already completed, identical test samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant.research.kronos_watchlist_multitask import EVENTS, event_arrays


def summarize(root: Path) -> pd.DataFrame:
    comparisons = [("B", "A", "auxiliary"), ("C", "B", "calendar_without_factors"),
                   ("D", "B", "factors_without_calendar"), ("E", "D", "calendar_with_factors"),
                   ("E", "C", "factors_with_calendar"), ("E", "F", "Kronos_increment")]
    records = []
    for first in sorted((root / "predictions").glob("*_A_test.parquet")):
        fold = first.name.removesuffix("_A_test.parquet")
        groups = {v: pd.read_parquet(root / "predictions" / f"{fold}_{v}_test.parquet").sort_values(["origin_date", "symbol"]).reset_index(drop=True)
                  for v in "ABCDEF"}
        reference = groups["A"]
        days = pd.DatetimeIndex(reference.origin_date.unique()).sort_values()
        block = reference.origin_date.map({d: i // 5 for i, d in enumerate(days)}).to_numpy()
        nblocks = int(block.max()) + 1
        counts = np.bincount(block, minlength=nblocks)
        draws = np.random.default_rng(20260912).multinomial(nblocks, np.full(nblocks, 1 / nblocks), size=1000)
        for variant, baseline, meaning in comparisons:
            left, right = groups[variant], groups[baseline]
            pd.testing.assert_frame_equal(left[["symbol", "origin_date", "target_date"]], right[["symbol", "origin_date", "target_date"]])
            differences = {}
            for field in ("open", "high", "low", "close"):
                differences[f"mae_{field}_pct"] = 100 * ((left[f"pred_{field}"] - left[f"actual_{field}"]).abs() -
                                                        (right[f"pred_{field}"] - right[f"actual_{field}"]).abs()) / left.current_close
            if baseline != "A":
                economic = 100 * (left[["actual_open", "actual_close"]].to_numpy() / left.target_pre_close.to_numpy()[:, None] - 1)
                probabilities = [np.stack([group[[f"p_{field}_{b}" for b in range(7)]].to_numpy() for field in ("open", "close")], axis=1)
                                 for group in (left, right)]
                for event in EVENTS:
                    p, y = event_arrays(probabilities[0], economic, event)
                    q, _ = event_arrays(probabilities[1], economic, event)
                    differences[f"{event}_brier"] = (p - y) ** 2 - (q - y) ** 2
            for metric, difference in differences.items():
                sums = np.bincount(block, weights=difference, minlength=nblocks)
                bootstrap = (draws @ sums) / (draws @ counts)
                records.append({"fold": fold, "variant": variant, "baseline": baseline, "meaning": meaning,
                                "metric": metric, "delta": float(np.mean(difference)), "low90": float(np.quantile(bootstrap, .05)),
                                "high90": float(np.quantile(bootstrap, .95)), "n": len(difference), "dates": len(days)})
    result = pd.DataFrame(records)
    result.to_csv(root / "ablation_deltas.csv", index=False)
    print(result.loc[result.metric.isin(["mae_open_pct", "mae_close_pct"])].to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/research/kronos_watchlist_multitask_v1"))
    summarize(parser.parse_args().root)
