"""Audit an unchanged historical signal using date-block resampling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from audit_overnight_opportunity import exit_returns

from quant.research.overnight_uncertainty import date_block_audit

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "reports/research/overnight_opportunity_v1/selected.parquet"
    forecast_source = ROOT / "reports/research/holding_event_probability_v2/refit_probabilities.parquet"
    calendar_source = ROOT / "reports/research/kronos_watchlist_multitask_v1/market_calendar.csv"
    picks = pd.read_parquet(source)
    forecasts = pd.read_parquet(forecast_source, columns=["date"])
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(calendar_source).date)).sort_values().unique()
    calendar = calendar[(calendar >= forecasts.date.min()) & (calendar <= forecasts.date.max())]
    args.output.mkdir(exist_ok=False)
    protocol = {"scope": "unchanged old signal, not independent confirmation",
                "start": str(calendar.min()), "end": str(calendar.max()), "market_dates": len(calendar),
                "blocks": [5, 20], "draws": 2000, "seed": 20260912,
                "missing": "failure for hit fraction, unresolved returns never imputed",
                "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (source, forecast_source, calendar_source)}}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    for name, cost, penetration, improvement in (("original20", 20, 0, True), ("stress40", 40, 10, False)):
        selected = picks[["date"]].copy()
        selected["net"] = exit_returns(picks, cost_bps=cost, penetration_bps=penetration, opening_improvement=improvement)
        for block in protocol["blocks"]:
            rows.append({"scenario": name, **date_block_audit(selected, calendar, block_days=block)})
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
