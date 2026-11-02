"""Add fixed pre-14:55 intraday path descriptors to the frozen minute dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.research.overnight_minutes import session_grid

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ("10:00", "10:30", "11:30", "13:30", "14:00", "14:30")


def path_features(bars: pd.DataFrame, day: pd.Timestamp) -> dict | None:
    """Only completed bars with a conservative five-minute publication delay."""
    day = pd.Timestamp(day).normalize()
    cutoff = day + pd.Timedelta(hours=14, minutes=55)
    times = pd.to_datetime(bars.trade_time)
    f = bars[times.dt.normalize().eq(day) & (times + pd.Timedelta(minutes=5)).le(cutoff)].copy()
    f = f.sort_values("trade_time")
    expected = session_grid(day, include_auction=False)
    expected = expected[expected + pd.Timedelta(minutes=5) <= cutoff]
    if not pd.DatetimeIndex(f.trade_time).equals(expected):
        return None
    values = f[["close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or not f.close.gt(0).all() or not f.volume.ge(0).all() or f.volume.sum() <= 0:
        return None
    close = f.set_index("trade_time").close
    volume = f.set_index("trade_time").volume
    result = {}
    for stamp in CHECKPOINTS:
        point = day + pd.Timedelta(hours=int(stamp[:2]), minutes=int(stamp[3:]))
        suffix = stamp.replace(":", "")
        result[f"x_min_path_return_{suffix}"] = float(close.loc[point] / close.iloc[0] - 1)
        result[f"x_min_path_volume_fraction_{suffix}"] = float(volume.loc[:point].sum() / volume.sum())
    returns = close.pct_change().dropna()
    result["x_min_path_max_drawdown"] = float((close / close.cummax() - 1).min())
    result["x_min_path_max_runup"] = float((close / close.cummin() - 1).max())
    total = returns.abs().sum()
    result["x_min_path_efficiency"] = float(abs(close.iloc[-1] / close.iloc[0] - 1) / total) if total > 0 else 0.
    result["x_min_path_positive_fraction"] = float(returns.gt(0).mean())
    return result


def main() -> None:
    source = ROOT / "reports/research/overnight_context_dataset_v1"
    minute_root = ROOT / "reports/research/overnight_baostock_2024_2026"
    output = ROOT / "reports/research/overnight_intraday_path_dataset_v1"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["smoke_only"] or not (source / "complete.json").exists() or not (minute_root / "complete.json").exists():
        raise ValueError("Complete parents required")
    output.mkdir(exist_ok=False)
    frame = pd.read_parquet(source / "dataset.parquet")
    coverage = pd.read_csv(minute_root / "coverage.csv")
    hashes, records = [], []
    for number, symbol in enumerate(sorted(frame.symbol.unique()), 1):
        parts = []
        for year in (2024, 2025, 2026):
            path = minute_root / symbol / f"{year}.parquet"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            match = coverage[coverage.symbol.eq(symbol) & coverage.year.eq(year)]
            if len(match) != 1 or match.sha256.iloc[0] != digest:
                raise ValueError(f"Minute cache changed: {symbol}/{year}")
            hashes.append({"path": str(path), "sha256": digest})
            f = pd.read_parquet(path)
            if len(f):
                parts.append(f[["trade_time", "close", "volume"]])
        if not parts:
            continue
        bars = pd.concat(parts, ignore_index=True)
        for day, group in bars.groupby(bars.trade_time.dt.normalize()):
            features = path_features(group, day)
            if features is not None:
                records.append({"symbol": symbol, "date": day, **features})
        if number % 15 == 0:
            print(f"path features: {number}/90 stocks", flush=True)
    result = frame.merge(pd.DataFrame(records), on=["symbol", "date"], how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(frame, result[frame.columns])
    cols = [c for c in result if c.startswith("x_min_path_")]
    if len(cols) != 16:
        raise ValueError("Path feature schema drift")
    manifest.update(parent_dataset_sha256=hashlib.sha256((source / "dataset.parquet").read_bytes()).hexdigest(),
        intraday_path="16 fixed close/volume path features; only bars through14:50 at14:55 cutoff",
        new_path_features=cols, original_columns_unchanged=True,
        path_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "minute_hashes.json").write_text(json.dumps(hashes, indent=2))
    result.to_parquet(output / "dataset.parquet", index=False)
    summary = {"rows": len(result), "new_features": len(cols), "eligible": int(result.decision_eligible.sum()),
               "eligible_path_complete": int(result.loc[result.decision_eligible, cols].notna().all(axis=1).sum()),
               "original_columns_unchanged": True}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
