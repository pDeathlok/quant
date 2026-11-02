"""Predeclared lagged breadth/gap-risk strata; reused-period diagnostic only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]


def summarize(picks: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    """Unknowns are retained; date means are marks, not executable returns."""
    net = picks.net - .002  # Parent uses 20bps; this experiment uses 40bps.
    daily = net.groupby(picks.date).mean().reindex(days, fill_value=0)
    return {"signals": len(picks), "dates": int(picks.date.nunique()),
            "unknown": int(net.isna().sum()), "known_win": float(net.dropna().gt(0).mean()),
            "win_missing_as_failure": float(net.gt(0).mean()), "known_mean_net": float(net.mean()),
            "known_daily_score": float(daily.mean() - 2 * daily.std() / np.sqrt(len(days))),
            "postdecision_entry_failures": int((~picks.entry_policy_eligible).sum())}


def main() -> None:
    output = ROOT / "reports/research/overnight_regime_strata_v1"
    output.mkdir(exist_ok=False)
    parent_roots = [ROOT / f"reports/research/{n}" for n in
                    ("overnight_signal_strata_v1", "overnight_right_signal_strata_v1")]
    lag_path = ROOT / "reports/research/tail_overnight/dataset.parquet"
    protocol = {
        "breadth": {"weak": "<0.4", "middle": "[0.4,0.6]", "strong": ">0.6", "all": "no gate"},
        "gap_risk": {"low": "previous 20 observations gap std <=0.01", "high": ">0.01", "all": "no gate"},
        "breadth_population": "all locally observed daily stocks with positive close, pre_close, vol; no current watchlist",
        "selection": "2025Q3: >=50 signals,>=20 dates,unknown=0,win>=80%,40bps daily score>0,"
                     "zero postdecision old entry failures; reject whole rule, never remove individual failures",
        "check": "Only Q3 eligible rules evaluated as locked in Q4; both periods already researched, not independent",
        "labels": "Unchanged parent rounded-limit/10bps penetration/close-fallback marks, targets 0.5% and 1%",
        "limitations": "No execution proof; inherited signal-cache provenance incomplete; all trials recorded",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lag_source_sha256": hashlib.sha256(lag_path.read_bytes()).hexdigest(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    raw = ds.dataset(ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
    bars = raw.to_table(columns=["ts_code", "trade_date", "close", "pre_close", "vol"],
        filter=(ds.field("trade_date") >= "20250601") & (ds.field("trade_date") < "20260101")).to_pandas()
    if bars.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Duplicate market bars")
    bars = bars[bars[["close", "pre_close", "vol"]].gt(0).all(axis=1)].copy()
    bars["positive"] = bars.close.gt(bars.pre_close)
    breadth = bars.groupby("trade_date").agg(breadth=("positive", "mean"), stocks=("ts_code", "nunique")).reset_index()
    if breadth.stocks.lt(1000).any():
        raise ValueError("Incomplete daily market universe")
    breadth["feature_date"] = pd.to_datetime(breadth.trade_date)
    breadth.to_parquet(output / "market_breadth_source.parquet", index=False)
    lag = pd.read_parquet(lag_path, columns=["symbol", "date", "feature_date", "lag1_daily_gap_std_20"])
    rows, locked = [], []
    provenance = []
    for parent in parent_roots:
        info = json.loads((parent / "protocol.json").read_text())
        if not (parent / "complete.json").exists():
            raise ValueError("Incomplete parent")
        for target in (50, 100):
            path = parent / f"tp{target}_outcomes.parquet"
            provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            frame = pd.read_parquet(path).merge(lag, on=["symbol", "date", "feature_date"], how="left", validate="one_to_one")
            frame = frame.merge(breadth[["feature_date", "breadth"]], on="feature_date", how="left", validate="many_to_one")
            if not frame.feature_date.lt(frame.date).all() or frame.breadth.isna().any():
                raise ValueError("Regime time/coverage contract violated")
            frame.to_parquet(output / f"{parent.name}_tp{target}_inputs_and_labels.parquet", index=False)
            market_masks = {"all": pd.Series(True, index=frame.index), "weak": frame.breadth.lt(.4),
                            "middle": frame.breadth.between(.4, .6), "strong": frame.breadth.gt(.6)}
            risk_masks = {"all": pd.Series(True, index=frame.index), "low": frame.lag1_daily_gap_std_20.le(.01),
                          "high": frame.lag1_daily_gap_std_20.gt(.01)}
            q3 = frame.date.lt("2025-10-01")
            for signal in info["strata"]:
                for market, mm in market_masks.items():
                    for risk, rm in risk_masks.items():
                        mask = frame[signal].eq(True) & mm & rm
                        identity = {"parent": parent.name, "target_bps": target, "signal": signal, "market": market, "risk": risk}
                        stats = summarize(frame[mask & q3], pd.DatetimeIndex(frame.loc[q3, "date"].unique()))
                        row = {**identity, **stats}
                        row["eligible"] = bool(stats["signals"] >= 50 and stats["dates"] >= 20 and stats["unknown"] == 0
                            and stats["known_win"] >= .8 and stats["known_daily_score"] > 0
                            and stats["postdecision_entry_failures"] == 0)
                        rows.append(row)
                        if row["eligible"]:
                            locked.append({**identity, **summarize(frame[mask & ~q3],
                                pd.DatetimeIndex(frame.loc[~q3, "date"].unique()))})
    table = pd.DataFrame(rows)
    table.to_csv(output / "q3_all_rules.csv", index=False)
    (output / "parent_hashes.json").write_text(json.dumps(provenance, indent=2))
    (output / "q4_locked_checks.json").write_text(json.dumps(locked, indent=2))
    result = {"q3_rules": len(table), "eligible": len(locked), "status": "diagnostic_only"}
    (output / "complete.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    sufficient = table[table.signals.ge(50) & table.dates.ge(20)]
    print(sufficient.sort_values("win_missing_as_failure", ascending=False).head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
