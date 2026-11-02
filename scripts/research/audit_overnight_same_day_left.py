"""Current-day left signals; full event population, Q3 selection/Q4 checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from overnight_right_signal_source import load_events
from train_overnight_direct import metrics

from quant.research.left_side_unified_features import LEFT_SIDE_SIGNALS
from quant.research.overnight_minutes import daily_limit_exit
from quant.research.right_side_unified_signals import CANONICAL_RIGHT_SIDE_SIGNALS

ROOT = Path(__file__).resolve().parents[2]


def signal_metrics(picks: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    result = metrics(picks, days, 1)
    daily = picks.net.fillna(-.10).groupby(picks.date).mean().reindex(days, fill_value=0)
    result.update(daily_penalized_mean=float(daily.mean()),
                  daily_score=float(daily.mean() - 2 * daily.std() / np.sqrt(len(days))),
                  allocation=float(picks.date.nunique() / len(days)))
    return result


def quarter_mask(frame: pd.DataFrame, quarter: str) -> pd.Series:
    period = pd.Period(quarter, freq="Q")
    end = period.end_time.normalize() + pd.Timedelta(days=1)
    return frame.date.ge(period.start_time) & frame.date.lt(end) & frame.exit_date.lt(end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right-same-day", action="store_true")
    args = parser.parse_args()
    signals = LEFT_SIDE_SIGNALS
    source = ROOT / "data/research/left_side_unified_v3_group4_input_parity/events.parquet"
    cache = ROOT / "reports/research/overnight_left_same_day_limits_2025h2_v1"
    output = ROOT / "reports/research/overnight_same_day_left_v2"
    if args.right_same_day:
        signals = CANONICAL_RIGHT_SIDE_SIGNALS
        source = ROOT / "reports/research/overnight_current_right_signals_v2/all_daily_signals.parquet"
        cache = ROOT / "reports/research/overnight_right_same_day_limits_2025h2_v1"
        output = ROOT / "reports/research/overnight_same_day_right_v1"
    if not (cache / "complete.json").exists():
        raise ValueError("Limit collection unfinished")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    availability = {}
    if args.right_same_day:
        f, availability = load_events(source)
        limit_manifest = json.loads((cache / "manifest.json").read_text())
        if limit_manifest["source_sha256"] != digest:
            raise ValueError("Limits collected for a different right signal source")
    else:
        manifest = json.loads(source.with_name("manifest.json").read_text())
        if manifest["status"] != "success" or digest != manifest["event_dataset_sha256"]:
            raise ValueError("Unverified event cache")
        f = pd.read_parquet(source, columns=["symbol", "date", *signals])
    output.mkdir(exist_ok=False)
    calendar_path = ROOT / "reports/research/overnight_same_day_market_limit_gated_v2/market_context.parquet"
    calendar = pd.read_parquet(calendar_path, columns=["date"]).sort_values("date")
    calendar["exit_date"] = calendar.date.shift(-1)
    f = f.merge(calendar, on="date", how="left", validate="many_to_one")
    f = f[f.date.ge("2025-07-01") & f.date.lt("2026-01-01") & f.exit_date.lt("2026-01-01")].reset_index(drop=True)
    if f.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate events")
    protocol = {"source_sha256": digest, "signals": list(signals), "targets": [.005, .01],
                "availability_audit": availability,
                "costs": [20, 40], "membership": "all currentT signal events, no old candidate-pool intersection",
                "entry": "T close, actual upper-limit gate; unknown entry prices/limits are reported separately",
                "labels": "next market day; missing/corporate-action/zero-volume outcomes retained unknown",
                "selection": "Q3 label exit strictly beforeOct1; >=50 trades,>=20 dates,unknown=0,win>=80%,"
                             "40bps daily score>0,zero ambiguous exits; only then Q4 check with exit beforeJan1",
                "limitations": "reused research periods; no independent confirmation; raw signal groups without cooldown;"
                               " equal weight all same-day members; unknown=-10% scoring penalty, not loss bound;"
                               " no ranking or position cap; EOD marks not fills",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    paths = [p for p in sorted((ROOT / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
             if "202507" <= p.parent.name.split("=")[1] <= "202512"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, calendar_path, *paths]}
    b = ds.dataset([str(p) for p in paths], format="parquet").to_table(
        columns=["ts_code", "trade_date", "high", "close", "pre_close", "vol", "volume"]).to_pandas()
    b["vol"] = b.vol.fillna(b.volume)
    b["date"] = pd.to_datetime(b.trade_date)
    b = b.rename(columns={"ts_code": "symbol"})
    f = f.merge(b[["symbol", "date", "close", "vol"]].rename(columns={"close": "entry_close", "vol": "entry_vol"}),
                on=["symbol", "date"], how="left", validate="one_to_one")
    nxt = b[["symbol", "date", "high", "close", "pre_close", "vol"]].rename(
        columns={"date": "exit_date", **{c: f"next_{c}" for c in ["high", "close", "pre_close", "vol"]}})
    f = f.merge(nxt, on=["symbol", "exit_date"], how="left", validate="many_to_one")
    parts = []
    for path in sorted(cache.glob("*.parquet")):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        parts.append(pd.read_parquet(path))
    limits = pd.concat(parts, ignore_index=True).rename(columns={"ts_code": "symbol"})
    limits["date"] = pd.to_datetime(limits.trade_date)
    f = f.merge(limits[["symbol", "date", "up_limit"]], on=["symbol", "date"], how="left", validate="one_to_one")
    f = f.merge(limits[["symbol", "date", "down_limit"]].rename(columns={"date": "exit_date"}),
                on=["symbol", "exit_date"], how="left", validate="many_to_one")
    f["unknown_entry"] = ~(f.entry_close.gt(0) & f.entry_vol.gt(0) & f.up_limit.gt(0))
    f["at_upper"] = ~f.unknown_entry & f.entry_close.ge(f.up_limit - .0051)
    f.to_parquet(output / "all_events_snapshot.parquet", index=False)
    summary = {"all_events": len(f), "stocks": int(f.symbol.nunique()), "unknown_entry": int(f.unknown_entry.sum()),
               "at_upper": int(f.at_upper.sum())}
    f = f[~f.unknown_entry & ~f.at_upper].reset_index(drop=True)
    records, checks = [], []
    for tp in [.005, .01]:
        result = pd.DataFrame([daily_limit_exit(r.entry_close, r.next_pre_close, r.next_high, r.next_close, tp)
                               for r in f.itertuples()])
        result.loc[~f.next_vol.gt(0), "net"] = np.nan
        picks = pd.concat([f, result[["net"]]], axis=1)
        price = np.ceil(picks.entry_close * (1 + tp) * 100 - 1e-9) / 100
        fallback = ~picks.next_high.ge(price + picks.entry_close * .001)
        picks["exit_ambiguous"] = fallback & (~picks.down_limit.gt(0) | picks.next_close.le(picks.down_limit + .0051))
        picks.to_parquet(output / f"tp{int(tp*10000)}_outcomes.parquet", index=False)
        for signal in signals:
            eligible = False
            for quarter in ["2025Q3", "2025Q4"]:
                if quarter == "2025Q4" and not eligible:
                    continue
                mask = quarter_mask(picks, quarter)
                selected = picks[mask & picks[signal].eq(True)]
                days = pd.DatetimeIndex(calendar.loc[quarter_mask(calendar, quarter), "date"])
                stats = signal_metrics(selected, days)
                stress = signal_metrics(selected.assign(net=selected.net - .002), days)
                record = {"signal": signal, "take_profit": tp, "quarter": quarter, "cost20": stats,
                          "cost40": stress, "exit_ambiguous": int(selected.exit_ambiguous.sum())}
                eligible = bool(stats["signals"] >= 50 and stats["dates"] >= 20 and stats["unknown"] == 0
                    and stress["known_win"] >= .8 and stress["daily_score"] > 0 and not selected.exit_ambiguous.any())
                record["eligible"] = eligible
                (records if quarter == "2025Q3" else checks).append(record)
    for path, old_hash in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != old_hash:
            raise ValueError("Source changed during signal audit")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "q3_rules.json").write_text(json.dumps(records, indent=2))
    (output / "q4_locked_checks.json").write_text(json.dumps(checks, indent=2))
    summary.update(q3_rules=len(records), q3_eligible=sum(r["eligible"] for r in records), q4_checks=len(checks))
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
