"""Fixed no-intraday-stop policy comparison, preserving all pre-entry inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.research.overnight_minutes import close_check_exit

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "reports/research/overnight_minute_dataset_v1"
    minutes = ROOT / "reports/research/overnight_baostock_2024_2026"
    output = ROOT / "reports/research/overnight_no_stop_dataset_v1"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["smoke_only"] or not (source / "complete.json").exists():
        raise ValueError("Complete full dataset required")
    for path, digest in manifest["files"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Source changed since original experiment")
    frame = pd.read_parquet(source / "dataset.parquet")
    output.mkdir(exist_ok=False)
    manifest.update(parent_dataset_sha256=hashlib.sha256((source / "dataset.parquet").read_bytes()).hexdigest(),
        policy_comparison="same 0.5%/1% completed-close profit triggers, next active bar open;"
        " disable intraday stop, precommitted next-session close fallback; uncapped losses",
        candidate_population="unchanged; unknown outcomes retained")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    labels = []
    for number, (symbol, part) in enumerate(frame.groupby("symbol", sort=True), 1):
        bars = pd.concat([pd.read_parquet(minutes / symbol / f"{year}.parquet") for year in (2024, 2025, 2026)])
        bars["trade_time"] = pd.to_datetime(bars.trade_time)
        groups = {day: group for day, group in bars.groupby(bars.trade_time.dt.normalize())}
        for row in part.itertuples():
            record = {"index": row.Index}
            future = groups.get(row.target_date)
            for name, threshold in (("tp05_nostop", .005), ("tp1_nostop", .01)):
                result = close_check_exit(future, row.entry_close_label, threshold, .01, use_stop=False) if (
                    future is not None and np.isfinite(row.entry_close_label) and row.entry_close_label > 0
                ) else {"status": "missing_entry_or_next_session"}
                record[f"target_net_{name}"] = result.get("net", np.nan) if row.future_quality == "ok" else np.nan
                record[f"net_proxy_{name}"] = result.get("net", np.nan)
                record[f"exit_status_{name}"] = result["status"]
            labels.append(record)
        print(f"no-stop labels {number}/90 {symbol}", flush=True)
    result = pd.concat([frame, pd.DataFrame(labels).set_index("index").reindex(frame.index)], axis=1)
    pd.testing.assert_frame_equal(frame, result[frame.columns])
    result.to_parquet(output / "dataset.parquet", index=False)
    comparisons = []
    validation = result[result.decision_eligible & result.date.ge("2025-10-01")
                        & result.target_date.lt("2026-01-01")]
    for target in ("tp05", "tp1"):
        a, b = validation[f"target_net_{target}"], validation[f"target_net_{target}_nostop"]
        paired = a.notna() & b.notna()
        comparisons.append({"target": target, "signals": len(validation), "paired_known": int(paired.sum()),
            "stop_unknown": int(a.isna().sum()), "nostop_unknown": int(b.isna().sum()),
            "stop_known_win": float(a[paired].gt(0).mean()), "nostop_known_win": float(b[paired].gt(0).mean()),
            "stop_known_mean": float(a[paired].mean()), "nostop_known_mean": float(b[paired].mean()),
            "stop_known_q01": float(a[paired].quantile(.01)), "nostop_known_q01": float(b[paired].quantile(.01)),
            "loss_to_win": int((a[paired].le(0) & b[paired].gt(0)).sum()),
            "win_to_loss": int((a[paired].gt(0) & b[paired].le(0)).sum())})
    (output / "policy_diagnostic.json").write_text(json.dumps(comparisons, indent=2))
    (output / "complete.json").write_text(json.dumps({"rows": len(result), "stocks": result.symbol.nunique()}))
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
