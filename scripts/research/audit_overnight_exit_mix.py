"""Paired fixed split-exit marks, without final-open-conditioned same-open fills."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]


def exit_marks(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed weights in the original position; no optimal hindsight switch."""
    entry = frame.entry_close
    close = (frame.next_close / entry - 1 - .002).where(frame.net.notna())
    opened = (frame.next_open / entry - 1 - .002).where(
        frame.net.notna() & np.isfinite(frame.next_open) & frame.next_open.gt(0))
    return pd.DataFrame({"full_limit": frame.net, "half_limit_half_close": .5 * frame.net + .5 * close,
                         "half_open_half_close": .5 * opened + .5 * close, "full_close": close})


def stats(net: pd.Series, dates: pd.Series, days: pd.DatetimeIndex) -> dict:
    daily = net.groupby(dates).mean().reindex(days, fill_value=0)
    return {"signals": len(net), "dates": int(dates.nunique()), "unknown": int(net.isna().sum()),
            "known_win": float(net.dropna().gt(0).mean()), "win_missing_as_failure": float(net.gt(0).mean()),
            "known_mean_net": float(net.mean()), "loss_mean": float(net[net.le(0)].mean()),
            "known_daily_score": float(daily.mean() - 2 * daily.std() / np.sqrt(len(days)))}


def main() -> None:
    output = ROOT / "reports/research/overnight_exit_mix_v1"
    output.mkdir(exist_ok=False)
    protocol = {
        "policies": ["full_limit", "half_limit_half_close", "half_open_half_close", "full_close"],
        "position_weights": "Fixed at entry: 1, or 0.5/0.5 of original position; not compounded",
        "take_profit_bps": [50, 100], "cost_bps": [20, 40],
        "selection": "Q3:>=50 signals,>=20 dates,unknown=0,40bps win>=80%,daily score>0,zero old entry failures",
        "check": "Only eligible Q3 rules checked in Q4; both are previously examined, not independent",
        "execution": "Precommitted open/close marks only, no final-open-conditioned same-open trades;"
                     " no guaranteed auctions/limit fills; minimum commissions on split orders not modelled",
        "inherited": "Parent signal provenance and execution uncertainties persist; no per-row future gate",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    raw = ds.dataset(ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
    bars = raw.to_table(columns=["ts_code", "trade_date", "open"],
        filter=(ds.field("trade_date") >= "20250701") & (ds.field("trade_date") < "20260101")).to_pandas()
    bars["exit_date"] = pd.to_datetime(bars.trade_date)
    bars = bars.rename(columns={"ts_code": "symbol", "open": "next_open"}).drop(columns="trade_date")
    rows, paired, checks, provenance = [], [], [], []
    for name in ("overnight_signal_strata_v1", "overnight_right_signal_strata_v1"):
        parent = ROOT / "reports/research" / name
        if not (parent / "complete.json").exists():
            raise ValueError("Incomplete parent")
        signals = json.loads((parent / "protocol.json").read_text())["strata"]
        snapshot_path = parent / "source_snapshot.parquet"
        snapshot = pd.read_parquet(snapshot_path, columns=["symbol", "date", "entry_close", "next_close"])
        provenance.append({"path": str(snapshot_path), "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest()})
        for tp in protocol["take_profit_bps"]:
            source = parent / f"tp{tp}_outcomes.parquet"
            provenance.append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
            f = pd.read_parquet(source).merge(snapshot, on=["symbol", "date"], how="left", validate="one_to_one")
            f = f.merge(bars, on=["symbol", "exit_date"], how="left", validate="many_to_one")
            if not f.feature_date.lt(f.date).all():
                raise ValueError("Non-lagged signal date")
            marks = exit_marks(f)
            pd.testing.assert_series_equal(marks.full_limit, f.net, check_names=False)
            pd.concat([f, marks.drop(columns="full_limit")], axis=1).to_parquet(
                output / f"{name}_tp{tp}_paired.parquet", index=False)
            q3 = f.date.lt("2025-10-01")
            days = pd.DatetimeIndex(f.loc[q3, "date"].unique())
            for signal in signals:
                mask = f[signal].eq(True) & q3
                for policy in protocol["policies"]:
                    # Open/close-only marks do not depend on take-profit; do not count them twice.
                    if tp == 100 and policy in ("half_open_half_close", "full_close"):
                        continue
                    net = marks.loc[mask, policy] - .002
                    identity = {"parent": name, "target_bps": tp if "limit" in policy else 0,
                                "signal": signal, "policy": policy}
                    row = {**identity, **stats(net, f.loc[mask, "date"], days),
                           "postdecision_entry_failures": int((~f.loc[mask, "entry_policy_eligible"]).sum())}
                    row["cost20_mean"] = float(marks.loc[mask, policy].mean())
                    row["eligible"] = bool(row["signals"] >= 50 and row["dates"] >= 20 and row["unknown"] == 0
                        and row["known_win"] >= .8 and row["known_daily_score"] > 0 and row["postdecision_entry_failures"] == 0)
                    rows.append(row)
                    common = mask & marks[policy].notna() & marks.full_limit.notna()
                    diff = marks.loc[common, policy] - marks.loc[common, "full_limit"]
                    paired.append({**identity, "common_rows": int(common.sum()), "mean_delta": float(diff.mean()),
                        "win_to_loss": int(((marks.loc[common, "full_limit"] > .002) & (marks.loc[common, policy] <= .002)).sum()),
                        "loss_to_win": int(((marks.loc[common, "full_limit"] <= .002) & (marks.loc[common, policy] > .002)).sum())})
                    if row["eligible"]:
                        later = f[signal].eq(True) & ~q3
                        checks.append({**identity, **stats(marks.loc[later, policy] - .002, f.loc[later, "date"],
                            pd.DatetimeIndex(f.loc[~q3, "date"].unique()))})
    table = pd.DataFrame(rows)
    table.to_csv(output / "q3_rules.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "q3_paired_changes.csv", index=False)
    (output / "parent_hashes.json").write_text(json.dumps(provenance, indent=2))
    (output / "q4_locked_checks.json").write_text(json.dumps(checks, indent=2))
    summary = {"rules": len(table), "eligible": len(checks), "status": "diagnostic_only"}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    print(table[table.signals.ge(50) & table.dates.ge(20)].sort_values("known_mean_net", ascending=False).head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
