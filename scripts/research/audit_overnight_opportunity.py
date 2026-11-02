"""Audit a pre-existing high-hit-rate signal, without retuning its threshold."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def exit_returns(
    frame: pd.DataFrame, *, cost_bps: float, penetration_bps: float, opening_improvement: bool
) -> pd.Series:
    """Price-only marks: no fill guarantee, no intraday stop-loss ordering."""
    known = np.isfinite(frame[["y_open", "y_high", "y_close"]]).all(axis=1)
    touched = frame.y_high.ge(0.01 + penetration_bps / 10000)
    value = frame.y_close.mask(touched, 0.01)
    if opening_improvement:
        value = value.mask(frame.y_open.ge(0.01 + penetration_bps / 10000), frame.y_open)
    return (value - cost_bps / 10000).where(known)


def summarize(frame: pd.DataFrame, net: pd.Series) -> dict:
    known = net.notna()
    counts = frame.groupby("date").size().sort_values(ascending=False)
    # Unknown exits are NOT cash. Only complete dates enter known-only daily means.
    daily = pd.DataFrame({"date": frame.date, "net": net}).groupby("date").net
    daily_mean = daily.mean()
    complete = daily.count().eq(daily.size())
    return {
        "signals": len(frame),
        "known_outcomes": int(known.sum()),
        "unresolved_outcomes": int((~known).sum()),
        "dates": len(counts),
        "largest_two_dates_share": float(counts.head(2).sum() / len(frame)),
        "hit_1pct_known": float(frame.loc[known, "y_high"].ge(0.01).mean()),
        "net_win_known": float(net[known].gt(0).mean()),
        "net_win_missing_as_failure": float(net.gt(0).sum() / len(frame)),
        "mean_net_known": float(net.mean()),
        "worst_net_known": float(net.min()),
        "complete_dates": int(complete.sum()),
        "equal_date_mean_complete_only": float(daily_mean[complete].mean()),
        "mean_net_excluding_largest_two_dates": float(net[~frame.date.isin(counts.head(2).index)].mean()),
        "corporate_action_flags": int(frame.corporate_action_gap.sum()),
        "legacy_open_exit_flags": int((~frame.exit_open_executable).sum()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source = ROOT / "reports/research/tail_overnight/dataset.parquet"
    forecasts = ROOT / "reports/research/holding_event_probability_v2/refit_probabilities.parquet"
    output = ROOT / "reports/research/overnight_opportunity_v1"
    output.mkdir(exist_ok=False)
    protocol = {
        "purpose": "audit existing signal, not independent confirmation or executable PnL",
        "signal": "high_up1>=0.8 and abs(tail_return)<0.095",
        "exit": "1% limit target, otherwise close; opening improvement sensitivity",
        "known_limitation": "T close and full-day values used as tail proxy; selected historical pools",
        "missing": "retain and report; exclude only from known-outcome statistics",
        "cost_bps": [20, 40],
        "penetration_bps": [0, 5, 10],
        "sources": {str(p): sha256(p) for p in (source, forecasts)},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    columns = [
        "symbol", "date", "exit_date", "y_open", "y_high", "y_close", "tail_return",
        "corporate_action_gap", "outcome_observed", "entry_executable",
        "entry_policy_eligible", "exit_open_executable", "pool_short", "pool_good", "pool_long",
    ]
    frame = pd.read_parquet(forecasts).merge(
        pd.read_parquet(source, columns=columns), on=["symbol", "date"], validate="one_to_one"
    )
    picks = frame[frame.high_up1.ge(0.8) & frame.tail_return.abs().lt(0.095)].copy()
    if picks.empty:
        raise ValueError("No selected rows for the fixed existing rule")
    rows = []
    for cost in protocol["cost_bps"]:
        for penetration in protocol["penetration_bps"]:
            for improvement in (True, False):
                net = exit_returns(picks, cost_bps=cost, penetration_bps=penetration,
                                   opening_improvement=improvement)
                rows.append({"cost_bps": cost, "penetration_bps": penetration,
                             "opening_improvement": improvement, **summarize(picks, net)})
    results = pd.DataFrame(rows)
    results.to_csv(output / "sensitivity.csv", index=False)
    picks["net_proxy"] = exit_returns(picks, cost_bps=20, penetration_bps=0, opening_improvement=True)
    picks.to_parquet(output / "selected.parquet", index=False)
    daily = picks.groupby("date").agg(
        signals=("symbol", "size"), known=("net_proxy", "count"), mean_net=("net_proxy", "mean"),
        mean_probability=("high_up1", "mean"), mean_entry_return=("tail_return", "mean"),
    )
    daily["unresolved"] = daily.signals - daily.known
    daily.to_csv(output / "daily.csv")
    print(results.to_string(index=False))
    print(f"Saved exploratory audit: {output}")


if __name__ == "__main__":
    main()
