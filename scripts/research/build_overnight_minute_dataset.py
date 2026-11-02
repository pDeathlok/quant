"""Build causal pre-close inputs; retain untrusted or unavailable future labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from quant.features.project_factor_layer import calculate_project_factor_frame
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS, load_daily_basic_features
from quant.research.overnight_minutes import close_check_exit, lagged_factor_row, preclose_snapshot

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    research = ROOT / "reports/research"
    source = research / "overnight_baostock_2024_2026"
    limit_source = research / "overnight_watchlist_limits_2024_2026"
    snapshot = research / "kronos_watchlist_multitask_v1/watchlist_snapshot.json"
    symbols = sorted(set(json.loads(snapshot.read_text())["symbols"]))
    ready = [s for s in symbols if all((source / s / f"{y}.parquet").exists() for y in (2024, 2025, 2026))]
    if not ready or (not args.smoke and (ready != symbols or not (source / "complete.json").exists())):
        raise RuntimeError("Full minute coverage required; --smoke is pipeline validation only")
    if not (limit_source / "complete.json").exists():
        raise RuntimeError("Complete watchlist limit cache required")
    # Freeze files before processing; completion order must not become strategy selection.
    paths = [source / s / f"{y}.parquet" for s in ready for y in (2024, 2025, 2026)]
    paths += [limit_source / f"{s}.parquet" for s in ready]
    args.output.mkdir(exist_ok=False)
    manifest = {"smoke_only": args.smoke, "symbols": ready,
                "not_downloaded": sorted(set(symbols) - set(ready)),
                "decision": "14:55, minute stamps <=14:50; daily factors strictly before T",
                "entry": "T official close price proxy, not guaranteed execution",
                "exit": "completed close trigger; next active bar open proxy; otherwise precommitted close",
                "cost_bps": 20, "evaluation": "exploratory; no untouched historical confirmation claimed",
                "future_quality": "label metadata only, never a pre-entry selection gate",
                "watchlist_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    columns = ["ts_code", "trade_date", "date", "symbol", "open", "high", "low", "close",
               "volume", "amount", "pre_close", "pct_chg", "change", "vol"]
    raw = ds.dataset(ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
    daily = raw.to_table(columns=columns, filter=ds.field("ts_code").isin(ready)
                         & (ds.field("trade_date") <= "20260911")).to_pandas()
    daily["date"] = pd.to_datetime(daily.trade_date)
    if daily.duplicated(["ts_code", "date"]).any():
        raise ValueError("Duplicate canonical daily keys")
    basic = load_daily_basic_features(ROOT / "data/raw/daily_basic", target_keys=daily[["ts_code", "trade_date"]])
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(
        research / "kronos_watchlist_multitask_v1/market_calendar.csv").date)).sort_values().unique()
    next_session = dict(zip(calendar[:-1], calendar[1:]))
    records = []
    for number, symbol in enumerate(ready, 1):
        history = daily[daily.ts_code.eq(symbol)].sort_values("date")
        factors = calculate_project_factor_frame(history, symbol,
            daily_basic_features=basic[basic.ts_code.eq(symbol)],
            factor_schema_version="project-v5-canonical-alias-free")
        indexed = history.set_index("date")
        previous = indexed.close.shift(1)
        minutes = pd.concat([pd.read_parquet(source / symbol / f"{y}.parquet") for y in (2024, 2025, 2026)])
        minutes["trade_time"] = pd.to_datetime(minutes.trade_time)
        minutes = minutes.sort_values("trade_time")
        groups = {day: part for day, part in minutes.groupby(minutes.trade_time.dt.normalize())}
        limits = pd.read_parquet(limit_source / f"{symbol}.parquet")
        limits["date"] = pd.to_datetime(limits.trade_date)
        if limits.date.duplicated().any():
            raise ValueError("Duplicate limit keys")
        limits = limits.set_index("date")
        # All market dates, including missing stock bars: do not select by next-day availability.
        for day in calendar[calendar >= pd.Timestamp("2024-01-01")]:
            target = next_session.get(day, pd.NaT)
            cutoff = day + pd.Timedelta(hours=14, minutes=55)
            row = {"symbol": symbol, "date": day, "target_date": target, "cutoff": cutoff,
                   "input_available": False, "decision_eligible": False}
            prior = float(previous.get(day, np.nan))
            frame = groups.get(day)
            features = preclose_snapshot(frame, prior, cutoff, include_auction=False) if (
                frame is not None and np.isfinite(prior) and prior > 0) else None
            lag = lagged_factor_row(factors, day)
            if features is not None and lag is not None:
                row.update({f"x_min_{key}": value for key, value in features.items()})
                row.update({f"x_lag_{key}": lag[key] for key in PROJECT_FACTOR_COLUMNS})
                row["factor_date"] = pd.Timestamp(lag.date)
                row["x_factor_age_days"] = (day - pd.Timestamp(lag.date)).days
                row["input_available"] = True
                upper = float(limits.loc[day, "up_limit"]) if day in limits.index else np.nan
                row["decision_eligible"] = bool(np.isfinite(upper) and features["snapshot_price"] < upper - .0051)
            entry = float(indexed.loc[day, "close"]) if day in indexed.index else np.nan
            row["entry_close_label"] = entry
            row["entry_final_at_limit"] = bool(day in limits.index and np.isfinite(entry)
                and entry >= float(limits.loc[day, "up_limit"]) - .0051)
            future = groups.get(target)
            quality = "ok"
            if not np.isfinite(entry) or entry <= 0:
                quality = "missing_entry"
            elif future is None or target not in indexed.index:
                quality = "missing_next_session"
            elif not np.isfinite(indexed.loc[target, ["pre_close", "open", "close"]].to_numpy(float)).all():
                quality = "invalid_next_daily_prices"
            elif abs(float(indexed.loc[target, "pre_close"]) - entry) > .011:
                quality = "corporate_action_unresolved"
            elif (abs(float(future.open.iloc[0]) - float(indexed.loc[target, "open"])) > .011
                  or abs(float(future.close.iloc[-1]) - float(indexed.loc[target, "close"])) > .011
                  or frame is None or abs(float(frame.close.iloc[-1]) - entry) > .011):
                quality = "source_price_conflict"
            row["future_quality"] = quality
            for name, threshold in (("tp05", .005), ("tp1", .01)):
                result = close_check_exit(future, entry, threshold, .01) if (
                    future is not None and np.isfinite(entry) and entry > 0) else {"status": quality}
                row[f"exit_status_{name}"] = result["status"]
                row[f"net_proxy_{name}"] = result.get("net", np.nan)
                row[f"target_net_{name}"] = result.get("net", np.nan) if quality == "ok" else np.nan
            records.append(row)
        print(f"minute dataset {number}/{len(ready)} {symbol}", flush=True)
    if any(not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest
           for p, digest in manifest["files"].items()):
        raise RuntimeError("Source changed while building; discard this incomplete output and rerun")
    result = pd.DataFrame(records)
    result.to_parquet(args.output / "dataset.parquet", index=False)
    summary = {"rows": len(result), "stocks": len(ready),
               "inputs": int(result.input_available.sum()), "eligible": int(result.decision_eligible.sum()),
               "quality_counts": result.future_quality.value_counts().to_dict()}
    (args.output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
