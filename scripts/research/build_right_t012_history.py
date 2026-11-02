"""Rebuild long pure-right signal inputs without legacy future-label screening."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic

import pandas as pd
import pyarrow.dataset as ds
from audit_overnight_right_provenance import sha256
from build_overnight_current_right_signals import compute

from quant.core.paths import PROJECT_ROOT
from quant.features.factor_registry import registry_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = PROJECT_ROOT
    output = root / "reports/research/right_t012_history_v1"
    if args.resume:
        if not (output / "protocol.json").exists() or (output / "complete.json").exists():
            raise ValueError("Only an incomplete protocol-backed run may resume")
    else:
        output.mkdir(exist_ok=False)
    shards = output / "by_symbol"
    shards.mkdir(exist_ok=args.resume)
    registry = registry_frame()
    if not args.resume:
        registry.to_csv(output / "registry_inventory.csv", index=False)
    paths = [p for p in sorted((root / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
             if "201401" <= p.parent.name.split("=")[1] <= "202609"]
    bars_path = root / "reports/research/overnight_unfiltered_universe_v1/daily_bars.parquet"
    expected = pd.read_parquet(bars_path, columns=["symbol", "date"])
    hashes = {str(p): sha256(p) for p in [*paths, bars_path,
        root / "scripts/research/build_overnight_current_right_signals.py",
        root / "scripts/research/replay_overnight_right_differences.py",
        root / "src/quant/research/right_side_unified_signals.py",
        root / "src/quant/features/factor_registry.py"]}
    manifest = dict(sources=hashes, start="2024-01-01", end="2026-09-04", warmup="2014-01-01",
                    candidate_source="All observed A-share bars, no old pool or future label filters",
                    numeric_registry_candidates=int(registry.role.isin(["feature", "research_feature"]).sum()),
                    signal_timing="T screen; next global session entry; second global session exit",
                    sample_design="All pure-right events retained; onset/repeated episodes tracked causally; no daily hash cap or five-day mature filter",
                    status="History rebuilding; not training completion", script_sha256=sha256(Path(__file__)))
    completed = []
    if args.resume:
        old = json.loads((output / "protocol.json").read_text())
        if hashes != old["sources"]:
            raise ValueError("Resume source hashes differ from original protocol")
        completed = pd.read_csv(output / "coverage.csv").to_dict("records")
        for item in completed:
            if sha256(shards / f"{item['symbol']}.parquet") != item["sha256"]:
                raise ValueError("Completed shard changed")
        (output / "resume_processpool.json").write_text(json.dumps(dict(
            verified_completed=len(completed), resumed_script_sha256=sha256(Path(__file__)), workers=4,
            change="Execution engine only: same compute function, sources, date range and key contract"), indent=2))
    else:
        (output / "protocol.json").write_text(json.dumps(manifest, indent=2))
    raw = ds.dataset([str(p) for p in paths], format="parquet")
    symbols = sorted(expected.symbol.unique())
    done_symbols = {r["symbol"] for r in completed}
    pending_symbols = [s for s in symbols if s not in done_symbols]
    started = monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(pending_symbols), 64):
            batch = pending_symbols[offset:offset + 64]
            history = raw.to_table(columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
                "vol", "volume", "amount", "pct_chg", "change"],
                filter=ds.field("ts_code").isin(batch) & (ds.field("trade_date") <= "20260904")).to_pandas()
            # Raw daily universe already uses the vendor's canonical302132 history.
            # Preserve its keys; old-code aliases are needed only by other sources.
            groups = {s: g for s, g in history.groupby("ts_code", sort=False)}
            futures = [(s, pool.submit(compute, groups[s], s, start="2024-01-01", end="2026-09-04"))
                       for s in batch if s in groups]
            for symbol, future in futures:
                frame = future.result()
                path = shards / f"{symbol}.parquet"
                frame.to_parquet(path, index=False)
                completed.append(dict(symbol=symbol, rows=len(frame), sha256=sha256(path)))
            pd.DataFrame(completed).to_csv(output / "coverage.csv", index=False)
            print(f"Signals {len(completed)}/{len(symbols)} elapsed={monotonic() - started:.0f}s", flush=True)
    signals = pd.concat([pd.read_parquet(shards / f"{s}.parquet") for s in symbols], ignore_index=True)
    # Exact daily-universe keys are authoritative; extra invalid/raw alias rows stay in shards.
    signals = expected.merge(signals, on=["symbol", "date"], how="left", validate="one_to_one", indicator=True)
    if not signals._merge.eq("both").all():
        raise ValueError("Missing expected signal keys")
    signals = signals.drop(columns="_merge")
    signals.to_parquet(output / "all_daily_signals.parquet", index=False)
    for name, digest in hashes.items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Source changed: {name}")
    (output / "complete.json").write_text(json.dumps(dict(rows=len(signals), stocks=signals.symbol.nunique(),
        dataset_sha256=sha256(output / "all_daily_signals.parquet"), status="History complete; factors/samples/training pending"), indent=2))


if __name__ == "__main__":
    main()
