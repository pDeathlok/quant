"""Replay discrepant family flags with fresh factors and signal-date prefixes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT
from quant.data.source_merge import normalize_tushare_daily
from quant.features.daily_factor_layer import calculate_daily_base_factors
from quant.research.b1_family_rules import compute_signal_flags
from quant.research.right_side_unified_signals import (
    B2_FAMILY_SOURCE_COLUMNS,
    B3_FAMILY_SOURCE_COLUMNS,
)
from quant.strategies.custom.triple_volume_breakout import add_triple_volume_strategy_pool_signals
from quant.strategies.custom.vegas_tunnel import (
    OPTIMIZED_VEGAS_TUNNEL_PARAMS,
    add_vegas_tunnel_signals,
)


def fresh_flags(daily: pd.DataFrame, symbol: str) -> pd.DataFrame:
    factors = calculate_daily_base_factors(daily, symbol)
    columns = [c for c in factors if c not in daily and c != "factor_version"]
    attached = daily.merge(factors[["date", *columns]], on="date", validate="one_to_one")
    flags = compute_signal_flags(attached, factors_attached=True)
    # Call directly: the family wrapper otherwise silently converts strategy errors to false.
    vegas = add_vegas_tunnel_signals(daily, **OPTIMIZED_VEGAS_TUNNEL_PARAMS)
    tvb = add_triple_volume_strategy_pool_signals(daily)
    result = flags[["date"]].copy()
    result["B2"] = flags[list(B2_FAMILY_SOURCE_COLUMNS)].any(axis=1)
    result["B3"] = flags[list(B3_FAMILY_SOURCE_COLUMNS)].any(axis=1)
    result = result.set_index("date")
    for name, source, column in (
        ("VEGAS", vegas, "signal_vegas_tunnel"),
        ("TRIPLE_VOLUME_BREAKOUT", tvb, "signal_tvb_merged"),
    ):
        source = source.set_index("date")
        if not result.index.equals(source.index):
            raise ValueError(f"{symbol}: {name} dropped or reordered daily dates")
        result[name] = source[column].fillna(False).astype(bool)
    return result


def main() -> None:
    source = PROJECT_ROOT / "reports/research/overnight_right_provenance_v1/differences.csv"
    output = PROJECT_ROOT / "reports/research/overnight_right_difference_replay_v2"
    output.mkdir(exist_ok=False)
    differences = pd.read_csv(source, parse_dates=["date"])
    paths = [p for p in sorted((PROJECT_ROOT / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
             if "201401" <= p.parent.name.split("=")[1] <= "202608"]
    hashes = {str(p): sha256(p) for p in paths}
    manifest = {
        "difference_source_sha256": sha256(source), "raw_sha256": hashes,
        "history_start": "2014-01-01", "history_end": "2026-08-22",
        "scope": "All 77 differing events in 68 symbols; not a full-universe validation",
        "method": "Fresh uncached factors; compare full history with prefix ending at each signal date",
        "limitations": ["Raw provider revisions and historical name status not resolved",
                        "History origin may differ from original cache build"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    raw = ds.dataset([str(p) for p in paths], format="parquet").to_table(
        columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
                 "vol", "volume", "amount", "pct_chg", "change"],
        filter=ds.field("ts_code").isin(differences.symbol.unique())
        & (ds.field("trade_date") <= "20260822"),
    ).to_pandas()
    if raw.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Duplicate raw keys")
    rows = []
    for number, (symbol, events) in enumerate(differences.groupby("symbol"), 1):
        daily = normalize_tushare_daily(raw.loc[raw.ts_code.eq(symbol)].copy(), symbol)
        daily = daily.sort_values("date").reset_index(drop=True)
        full = fresh_flags(daily, symbol)
        for event in events.itertuples(index=False):
            prefix = fresh_flags(daily.loc[daily.date.le(event.date)].copy(), symbol)
            rows.append({"symbol": symbol, "date": event.date, "signal": event.signal,
                         "family_flag": event.family_flag, "right_flag": event.right_flag,
                         "fresh_full": bool(full.loc[event.date, event.signal]),
                         "fresh_prefix": bool(prefix.loc[event.date, event.signal])})
        pd.DataFrame(rows).to_csv(output / "replay.csv", index=False)
        print(f"Replayed {number}/68 symbols, {len(rows)}/77 events", flush=True)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError("Raw data changed during replay")
    if sha256(source) != manifest["difference_source_sha256"]:
        raise RuntimeError("Difference source changed")
    frame = pd.DataFrame(rows)
    summary = {"events": len(frame), "symbols": int(frame.symbol.nunique()),
               "prefix_full_disagreements": int(frame.fresh_full.ne(frame.fresh_prefix).sum()),
               "prefix_matches_family": int(frame.fresh_prefix.eq(frame.family_flag).sum()),
               "prefix_matches_right": int(frame.fresh_prefix.eq(frame.right_flag).sum())}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
