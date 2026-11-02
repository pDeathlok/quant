"""Rebuild 14 current-day signals from raw bars, independently of outcome labels."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from audit_overnight_right_provenance import sha256
from overnight_market_context import ASHARE
from replay_overnight_right_differences import fresh_flags

from quant.core.paths import PROJECT_ROOT
from quant.data.source_merge import normalize_tushare_daily
from quant.research.right_side_unified_signals import (
    CANONICAL_RIGHT_SIDE_SIGNALS,
    compute_canonical_z_signal_flags,
)


def compute(history: pd.DataFrame, symbol: str, *, start: str = "2025-07-01",
            end: str = "2025-12-31") -> pd.DataFrame:
    if history.duplicated("trade_date").any():
        raise ValueError(f"Duplicate history: {symbol}")
    daily = normalize_tushare_daily(history, symbol).sort_values("date").reset_index(drop=True)
    valid = daily[["open", "high", "low", "close", "pre_close", "volume"]].gt(0).all(axis=1)
    valid &= daily.high.ge(daily[["open", "close"]].max(axis=1) - 1e-8)
    valid &= daily.low.le(daily[["open", "close"]].min(axis=1) + 1e-8)
    result = daily[["symbol", "date"]].copy()
    result["valid_bar"] = valid
    result["valid_history_count"] = valid.cumsum()
    usable = daily.loc[valid].reset_index(drop=True)
    if usable.empty:
        for signal in CANONICAL_RIGHT_SIDE_SIGNALS:
            result[signal] = pd.NA
    else:
        family = fresh_flags(usable, symbol).reset_index()
        z_flags = compute_canonical_z_signal_flags(usable)
        signals = z_flags.merge(family, on="date", validate="one_to_one")
        if len(signals) != len(usable):
            raise ValueError(f"Signal dates lost: {symbol}")
        result = result.merge(signals, on=["symbol", "date"], how="left", validate="one_to_one")
    # The old family entry point required 160 total rows, potentially including future rows.
    # Here readiness is determined independently on each date from its available history.
    result["family_ready"] = result.valid_bar & result.valid_history_count.ge(160)
    for signal in CANONICAL_RIGHT_SIDE_SIGNALS:
        result[signal] = result[signal].astype("boolean")
    for signal in ("B2", "B3", "VEGAS", "TRIPLE_VOLUME_BREAKOUT"):
        result.loc[~result.family_ready, signal] = pd.NA
    return result.loc[result.date.between(start, end)].reset_index(drop=True)


def main() -> None:
    output = PROJECT_ROOT / "reports/research/overnight_current_right_signals_v2"
    output.mkdir(exist_ok=False)
    shards = output / "by_symbol"
    shards.mkdir()
    paths = [p for p in sorted((PROJECT_ROOT / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
             if "201401" <= p.parent.name.split("=")[1] <= "202512"]
    hashes = {str(p): sha256(p) for p in paths}
    dataset = ds.dataset([str(p) for p in paths], format="parquet")
    recent = dataset.to_table(columns=["ts_code", "trade_date"],
                              filter=ds.field("trade_date") >= "20250701").to_pandas()
    recent = recent.loc[recent.ts_code.str.fullmatch(ASHARE)]
    if recent.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Duplicate market keys")
    symbols = sorted(recent.ts_code.unique())
    manifest = {
        "raw_sha256": hashes, "script_sha256": sha256(Path(__file__)),
        "history_start": "2014-01-01", "history_end": "2025-12-31",
        "output_start": "2025-07-01", "output_end": "2025-12-31",
        "symbols": symbols, "expected_rows": len(recent), "signals": list(CANONICAL_RIGHT_SIDE_SIGNALS),
        "method": "Fresh factors and 14 flags; no cached signals, labels or future tradability filters",
        "missing_policy": "Retain invalid daily keys with unavailable flags; only positive valid bars enter history",
        "readiness": "Four family flags require 160 valid bars as of each signal date",
        "universe": "All locally observed A-share keys, no current watchlist or historical tail-pool intersection",
        "limitations": ["Historical ST names and raw vendor revisions not point-in-time audited",
                        "14:30 incomplete day is a deployment approximation, not tested here",
                        "Q3/Q4 2025 already researched; not fresh independent confirmation"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    recent.to_parquet(output / "expected_keys.parquet", index=False)
    completed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(symbols), 32):
            batch = symbols[start:start + 32]
            history = dataset.to_table(
                columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
                         "vol", "volume", "amount", "pct_chg", "change"],
                filter=ds.field("ts_code").isin(batch),
            ).to_pandas()
            groups = {symbol: group for symbol, group in history.groupby("ts_code")}
            futures = [(symbol, pool.submit(compute, groups[symbol], symbol)) for symbol in batch]
            for symbol, future in futures:
                frame = future.result()
                path = shards / f"{symbol}.parquet"
                frame.to_parquet(path, index=False)
                completed.append({"symbol": symbol, "rows": len(frame), "sha256": sha256(path)})
            pd.DataFrame(completed).to_csv(output / "coverage.csv", index=False)
            print(f"Built {len(completed)}/{len(symbols)} stocks", flush=True)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError("Raw input changed during build")
    all_flags = pd.concat([pd.read_parquet(shards / f"{s}.parquet") for s in symbols], ignore_index=True)
    observed = all_flags[["symbol", "date"]].sort_values(["symbol", "date"]).reset_index(drop=True)
    expected = recent.rename(columns={"ts_code": "symbol", "trade_date": "date"})
    expected["date"] = pd.to_datetime(expected.date)
    expected = expected.sort_values(["symbol", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected)
    all_flags.to_parquet(output / "all_daily_signals.parquet", index=False)
    summary = {"stocks": len(symbols), "rows": len(all_flags),
               "invalid_rows": int((~all_flags.valid_bar).sum()),
               "family_unready_rows": int((~all_flags.family_ready).sum()),
               "signals": {s: int(all_flags[s].sum()) for s in CANONICAL_RIGHT_SIDE_SIGNALS},
               "dataset_sha256": sha256(output / "all_daily_signals.parquet")}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
