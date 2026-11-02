"""Complete current-day candidate factors with the shared daily-basic formulas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.features.variable_library import (
    DAILY_BASIC_PROJECT_FACTOR_COLUMNS,
    DAILY_BASIC_SOURCE_COLUMNS,
    PROJECT_FACTOR_COLUMNS,
    derive_daily_basic_features,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unfiltered", action="store_true", help="Complete expanded daily keys using the unchanged basic formulas")
    parser.add_argument("--event-extension", action="store_true", help="Complete locked September extension and boundary keys")
    args = parser.parse_args()
    if args.unfiltered and args.event_extension:
        parser.error("Choose one candidate universe")
    market = ROOT / "reports/research/overnight_current_market_factors_v1"
    output = ROOT / "reports/research/overnight_current_full_factors_v1"
    source = ROOT / "reports/research/tail_overnight/dataset.parquet"
    if args.unfiltered:
        market = ROOT / "reports/research/overnight_unfiltered_market_factors_v1"
        output = ROOT / "reports/research/overnight_unfiltered_full_factors_v1"
        source = ROOT / "reports/research/overnight_unfiltered_universe_v1/daily_bars.parquet"
    if args.event_extension:
        market = ROOT / "reports/research/overnight_event_extension_market_factors_v1"
        output = ROOT / "reports/research/overnight_event_extension_full_factors_v1"
        source = ROOT / "reports/research/overnight_event_extension_pool_v1/entry_pool.parquet"
    if not (market / "market_complete.json").exists():
        raise ValueError("Market stage must finish first")
    output.mkdir(exist_ok=False)
    keys = pd.read_parquet(source, columns=["symbol", "date"])
    market_manifest = json.loads((market / "manifest.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != market_manifest["candidate_sha256"]:
        raise ValueError("Candidate source changed since market build")
    if args.event_extension:
        boundary_path = source.parent / "boundary_comparison.parquet"
        if hashlib.sha256(boundary_path.read_bytes()).hexdigest() != market_manifest["extra_candidate_sources"][str(boundary_path)]:
            raise ValueError("Boundary keys changed since market build")
        boundary = pd.read_parquet(boundary_path)
        if not boundary._merge.eq("both").all():
            raise ValueError("Unaudited boundary membership")
        keys = pd.concat([boundary[["symbol", "date"]], keys], ignore_index=True)
        if keys.duplicated(["symbol", "date"]).any():
            raise ValueError("Duplicate extension keys")
    symbols = set(keys.symbol)
    start, end = keys.date.min().strftime("%Y%m%d"), keys.date.max().strftime("%Y%m%d")
    files = sorted((ROOT / "data/raw/daily_basic").glob("*.parquet"))
    if any(len(p.stem) != 8 or not p.stem.isdigit() for p in files):
        raise ValueError("Unrecognized daily-basic file dates")
    hashes: dict[str, str] = {}

    def read(path: Path, selected: set[str]) -> pd.DataFrame:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Some old files omit optional source columns; formulas preserve NaNs.
        raw = pd.read_parquet(path)
        if not {"ts_code", "trade_date"} <= set(raw):
            raise ValueError(f"Missing daily-basic keys: {path}")
        raw = raw.loc[raw.ts_code.isin(selected), [c for c in DAILY_BASIC_SOURCE_COLUMNS if c in raw]].copy()
        raw["trade_date"] = pd.to_datetime(raw.trade_date.astype(str), format="mixed").dt.strftime("%Y%m%d")
        if not raw.trade_date.eq(path.stem).all() or raw.ts_code.duplicated().any():
            raise ValueError(f"Unexpected daily-basic dates or duplicate symbols: {path}")
        return raw

    frames = []
    for index, path in enumerate(p for p in files if start <= p.stem <= end):
        frames.append(read(path, symbols))
        if index % 100 == 0:
            print(f"daily-basic target-range files: {index + 1}", flush=True)
    # Read backwards only retaining stocks still needing a rolling warmup.
    # Requiring finite MV values at the oldest retained row also preserves the
    # shared pct_change implementation's historical forward-fill semantics.
    remaining = set(symbols)
    counts = {s: 0 for s in symbols}
    for path in reversed([p for p in files if p.stem < start]):
        if not remaining:
            break
        prior = read(path, remaining)
        frames.append(prior)
        for row in prior.itertuples(index=False):
            symbol = row.ts_code
            counts[symbol] += 1
            if counts[symbol] >= 20 and all(pd.notna(getattr(row, c, np.nan)) for c in ["total_mv", "circ_mv"]):
                remaining.remove(symbol)
    history = pd.concat(frames, ignore_index=True)
    del frames
    if history.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Duplicate daily-basic history")
    print(f"daily-basic retained history: {len(history)} rows; source files: {len(hashes)}", flush=True)
    groups = {s: g for s, g in history.groupby("ts_code", sort=False)}
    market_coverage = pd.read_csv(market / "coverage.csv").set_index("symbol")
    results = []
    basic_cols = list(DAILY_BASIC_PROJECT_FACTOR_COLUMNS)
    for index, (symbol, target) in enumerate(keys.groupby("symbol", sort=True)):
        path = market / "by_symbol" / f"{symbol}.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != market_coverage.loc[symbol, "sha256"]:
            raise ValueError(f"Market shard changed: {symbol}")
        factors = pd.read_parquet(path)
        basic = groups.get(symbol)
        if basic is not None:
            basic = derive_daily_basic_features(basic).rename(columns={"ts_code": "symbol", "trade_date": "date"})
            basic["date"] = pd.to_datetime(basic.date, format="%Y%m%d")
            basic["basic_row_present"] = True
            basic = basic.reindex(columns=["symbol", "date", "basic_row_present", *basic_cols])
            factors = factors.merge(basic, on=["symbol", "date"], how="left", validate="one_to_one")
        else:
            factors[basic_cols] = np.nan
            factors["basic_row_present"] = False
        if set(PROJECT_FACTOR_COLUMNS) - set(factors):
            raise ValueError(f"Incomplete feature schema: {symbol}")
        if len(factors) != len(target):
            raise ValueError("Lost candidate keys")
        factors[list(PROJECT_FACTOR_COLUMNS)] = factors[list(PROJECT_FACTOR_COLUMNS)].astype("float32")
        results.append(factors)
        if index % 500 == 0:
            print(f"full factor joins: {index + 1}/{len(symbols)}", flush=True)
    result = pd.concat(results, ignore_index=True)
    if len(result) != len(keys) or result.duplicated(["symbol", "date"]).any():
        raise ValueError("Full candidate key mismatch")
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Daily-basic source changed during build")
    result.to_parquet(output / "dataset.parquet", index=False)
    result[list(PROJECT_FACTOR_COLUMNS)].notna().mean().rename("non_null_fraction").to_csv(output / "feature_coverage.csv")
    manifest = {"candidate_sha256": market_manifest["candidate_sha256"], "daily_basic_files": hashes,
                "market_manifest": market_manifest, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "formulas_sha256": hashlib.sha256((ROOT / "src/quant/features/variable_library.py").read_bytes()).hexdigest(),
                "warmup_observations": counts, "warmup_unresolved_after_all_local_history": sorted(remaining),
                "timing": "current EOD; 14:30 deployment approximation, no minute inputs",
                "missing_policy": "exact key left joins; no fabricated prices or future filling"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = {"rows": len(result), "stocks": int(result.symbol.nunique()), "features": len(PROJECT_FACTOR_COLUMNS),
               "market_rows": int(result.market_row_present.eq(True).sum()),
               "basic_rows": int(result.basic_row_present.eq(True).sum()),
               "dataset_sha256": hashlib.sha256((output / "dataset.parquet").read_bytes()).hexdigest()}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
