"""Fixed T-1 signal strata with next-session sell-limit price proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from quant.research.left_side_unified_features import LEFT_SIDE_SIGNALS
from quant.research.overnight_minutes import daily_limit_exit
from quant.research.right_side_unified_features import RIGHT_SIDE_SIGNALS

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["left", "right"], default="left")
    args = parser.parse_args()
    signals = LEFT_SIDE_SIGNALS if args.side == "left" else RIGHT_SIDE_SIGNALS
    source = ROOT / "reports/research/tail_overnight/dataset.parquet"
    event_path = ROOT / ("data/research/left_side_unified_v3_group4_input_parity/events.parquet"
                        if args.side == "left" else
                        "data/research/right_side_unified_current_v6_20260824/unified_right_side_dataset.parquet")
    output = ROOT / ("reports/research/overnight_signal_strata_v1" if args.side == "left" else
                     "reports/research/overnight_right_signal_strata_v1")
    output.mkdir(exist_ok=False)
    protocol = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "events_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "side": args.side, "strata": list(signals), "periods": ["2025Q3", "2025Q4"],
        "take_profit": [.005, .01], "cost_bps": [20, 40], "penetration_bps": 10,
        "decision": "Exact previous market day signal; no same-day final-data gate; all stratum members",
        "exit": "Upward cent-rounded sell limit; require next high >= limit + entry*0.001; else close",
        "caveats": ["Descriptive reused validation; no independent confirmation",
                    "No opening improvement, no stop, no assumed queue fill",
                    "Old entry feasibility flags and future prices never select individual rows",
                    "Missing data and corporate actions retained; down-limit execution not established",
                    "Signal groups overlap; no summed independent opportunity count",
                    "Right family-cache full point-in-time provenance remains unaudited"],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    f = pd.read_parquet(source, columns=["symbol", "date", "feature_date", "exit_date", "entry_policy_eligible"])
    f = f[f.date.ge("2025-07-01") & f.date.lt("2026-01-01") & f.exit_date.lt("2026-01-01")].copy()
    if f.duplicated(["symbol", "date"]).any() or not f.feature_date.lt(f.date).all():
        raise ValueError("Sample time/key contract violated")
    events = pd.read_parquet(event_path, columns=["symbol", "date", *signals])
    events = events.rename(columns={"date": "feature_date"})
    f = f.merge(events, on=["symbol", "feature_date"], how="left", validate="many_to_one")
    # Membership uses only observed previous-day flags; no later outcome filter.
    f = f[f[list(signals)].eq(True).any(axis=1)].reset_index(drop=True)
    raw = ds.dataset(ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
    bars = raw.to_table(columns=["ts_code", "trade_date", "high", "close", "pre_close", "vol"],
        filter=(ds.field("trade_date") >= "20250701") & (ds.field("trade_date") < "20260101")).to_pandas()
    bars["date"] = pd.to_datetime(bars.trade_date)
    bars = bars.rename(columns={"ts_code": "symbol"}).drop(columns="trade_date")
    f = f.merge(bars[["symbol", "date", "close", "vol"]].rename(
        columns={"close": "entry_close", "vol": "entry_vol"}), on=["symbol", "date"], how="left", validate="many_to_one")
    nxt = bars.rename(columns={"date": "exit_date", **{c: f"next_{c}" for c in ["high", "close", "pre_close", "vol"]}})
    f = f.merge(nxt, on=["symbol", "exit_date"], how="left", validate="many_to_one")
    f.to_parquet(output / "source_snapshot.parquet", index=False)
    rows = []
    for tp in protocol["take_profit"]:
        result = pd.DataFrame([daily_limit_exit(r.entry_close, r.next_pre_close, r.next_high, r.next_close, tp)
                               for r in f.itertuples()])
        valid_volume = f.entry_vol.gt(0) & f.next_vol.gt(0)
        result.loc[~valid_volume, "status"] = "no_observed_volume"
        result.loc[~valid_volume, "net"] = np.nan
        record = pd.concat([f[["symbol", "date", "feature_date", "exit_date", "entry_policy_eligible", *signals]], result], axis=1)
        record.to_parquet(output / f"tp{int(tp * 10000)}_outcomes.parquet", index=False)
        for period in protocol["periods"]:
            dates = record.date.dt.to_period("Q").astype(str).eq(period)
            for signal in signals:
                picks = record[dates & record[signal].eq(True)]
                for cost in protocol["cost_bps"]:
                    net = picks.net - (cost - 20) / 10000
                    daily = net.groupby(picks.date).mean()
                    rows.append({"period": period, "signal": signal, "take_profit": tp, "cost_bps": cost,
                        "signals": len(picks), "dates": picks.date.nunique(), "unknown": int(net.isna().sum()),
                        "known_win": float(net.dropna().gt(0).mean()), "win_missing_as_failure": float(net.gt(0).mean()),
                        "known_mean_net": float(net.mean()), "known_daily_mean": float(daily.mean()),
                        "known_daily_score": float(daily.mean() - 2 * daily.std() / np.sqrt(len(daily))),
                        "postdecision_old_entry_failures": int((~picks.entry_policy_eligible).sum())})
    table = pd.DataFrame(rows)
    table.to_csv(output / "strata.csv", index=False)
    summary = {"rows": len(f), "stocks": int(f.symbol.nunique()), "dates": int(f.date.nunique()),
               "strata_tests": len(table), "status": "diagnostic_only_not_goal_completion"}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    print(table[table.cost_bps.eq(40)].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
