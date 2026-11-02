"""Exact-calendar 3/5-day moneyflow histories, with no compressed missing dates."""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from audit_overnight_moneyflow_coverage import AMOUNTS
from audit_overnight_right_provenance import sha256
from build_overnight_moneyflow_lag import ALIAS_DATE, FEATURES, ratios

from quant.core.paths import PROJECT_ROOT


def fetch_extra_day() -> None:
    """One audited missing day; isolated cache, never replace a downloaded file."""
    output = PROJECT_ROOT / "reports/research/overnight_moneyflow_rolling_source_v1"
    output.mkdir(exist_ok=True)
    path = output / "day_20260825.parquet"
    if path.exists():
        raise ValueError("Already downloaded; inspect before any retry")
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        for line in (PROJECT_ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not token:
        raise RuntimeError("No Tushare credential")
    fields = ["ts_code", "trade_date", *AMOUNTS, "net_mf_amount"]
    try:
        with httpx.Client(timeout=25) as client:
            response = client.post("https://api.tushare.pro", json={"api_name": "moneyflow", "token": token,
                "params": {"trade_date": "20260825"}, "fields": ",".join(fields)})
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError):
        raise RuntimeError("Provider transport failed; credential not logged") from None
    if body.get("code") != 0 or not body.get("data"):
        raise RuntimeError(f"Provider error code {body.get('code')}; no quota retry")
    raw = pd.DataFrame(body["data"]["items"], columns=body["data"]["fields"])
    if (not 0 < len(raw) < 6000 or not set(fields).issubset(raw.columns)
            or raw.ts_code.isna().any() or raw.ts_code.duplicated().any()
            or not raw.trade_date.astype(str).eq("20260825").all()):
        raise ValueError("Unexpected provider date, keys, or truncation")
    raw.to_parquet(path, index=False)
    (output / "complete.json").write_text(json.dumps({"path": str(path), "sha256": sha256(path), "rows": len(raw)}, indent=2))
    print(f"Downloaded {len(raw)} moneyflow rows for audited missing day", flush=True)


def window_means(values: np.ndarray, window: int) -> np.ndarray:
    """Input chronological exact trading-day slots; NaN invalidates that feature window."""
    if values.ndim != 3 or values.shape[0] < window:
        raise ValueError("Insufficient exact-calendar slots")
    return values[-window:].mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-extra-day", action="store_true")
    args = parser.parse_args()
    if args.fetch_extra_day:
        fetch_extra_day()
        return
    root = PROJECT_ROOT / "reports/research"
    source = root / "overnight_moneyflow_lag_v1/dataset.parquet"
    calendar_path = root / "overnight_same_day_market_limit_gated_v2/market_context.parquet"
    old = pd.read_parquet(source)
    calendar = pd.read_parquet(calendar_path, columns=["date"]).sort_values("date").date.tolist()
    hashes = {str(source): sha256(source), str(calendar_path): sha256(calendar_path)}
    output = root / "overnight_moneyflow_rolling_v1"
    output.mkdir(exist_ok=False)

    @lru_cache(maxsize=8)
    def load_day(date: pd.Timestamp) -> pd.DataFrame:
        choices = [PROJECT_ROOT / f"data/raw/moneyflow/tushare_moneyflow_{date:%Y%m%d}.parquet",
                   root / f"overnight_moneyflow_gap_fill_v1/day_{date:%Y%m%d}.parquet",
                   root / f"overnight_moneyflow_rolling_source_v1/day_{date:%Y%m%d}.parquet"]
        path = next((p for p in choices if p.exists()), None)
        if path is None:
            raise ValueError(f"Missing market date {date}; never compress window")
        hashes[str(path)] = sha256(path)
        raw = pd.read_parquet(path, columns=["ts_code", "trade_date", *AMOUNTS, "net_mf_amount"])
        if raw.ts_code.isna().any() or raw.ts_code.duplicated().any() or not raw.trade_date.astype(str).eq(date.strftime("%Y%m%d")).all():
            raise ValueError("Invalid source keys")
        values = ratios(raw)
        values.index = raw.ts_code
        return values

    records = []
    added = [f"{c}_mean{w}" for w in (3, 5) for c in FEATURES]
    added += [f"{c}_delta5" for c in FEATURES]
    for date, group in old.groupby("date", sort=True):
        position = calendar.index(date)
        if position < 5:
            raise ValueError("Insufficient calendar history")
        dates = calendar[position-5:position]
        snapshots = []
        for flow_date in dates:
            symbols = group.symbol.copy()
            if flow_date < ALIAS_DATE:
                symbols = symbols.replace({"302132.SZ": "300114.SZ"})
            snapshots.append(load_day(flow_date).reindex(symbols)[FEATURES].to_numpy())
        history = np.stack(snapshots)
        np.testing.assert_allclose(history[-1], group[FEATURES], equal_nan=True, rtol=0, atol=0)
        current = group[["symbol", "date", "flow_date", *FEATURES]].copy()
        current["window_start"] = dates[0]
        for window in (3, 5):
            current[[f"{c}_mean{window}" for c in FEATURES]] = window_means(history, window)
        delta = history[-1] - history[0]
        delta[~np.isfinite(history).all(axis=0)] = np.nan
        current[[f"{c}_delta5" for c in FEATURES]] = delta
        records.append(current)
    frame = pd.concat(records).sort_values(["date", "symbol"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[["symbol", "date", *FEATURES]], old[["symbol", "date", *FEATURES]])
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source changed during construction")
    frame.to_parquet(output / "dataset.parquet", index=False)
    summary = {"rows": len(frame), "features": [*FEATURES, *added],
               "complete_windows": {str(w): int(frame[[f"{c}_mean{w}" for c in FEATURES]].notna().all(axis=1).sum()) for w in (3, 5)},
               "dataset_sha256": sha256(output / "dataset.parquet"), "script_sha256": sha256(Path(__file__)),
               "scope": "24 features, exact T-5 through T-1 market slots; simple daily ratio means; no missing-to-zero or forward fill; no model result"}
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
