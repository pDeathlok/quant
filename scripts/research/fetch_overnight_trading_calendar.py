"""Fetch a complete daily calendar once, preserving existing caches."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    output = PROJECT_ROOT / "reports/research/overnight_trading_calendar_v1"
    output.mkdir(exist_ok=False)
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        for line in (PROJECT_ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not token:
        raise RuntimeError("Credential unavailable")
    params = {"exchange": "SSE", "start_date": "20231201", "end_date": "20260930"}
    fields = ["exchange", "cal_date", "is_open", "pretrade_date"]
    with httpx.Client(timeout=25) as client:
        try:
            response = client.post("https://api.tushare.pro", json={"api_name": "trade_cal", "token": token,
                "params": params, "fields": ",".join(fields)})
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError):
            raise RuntimeError("Calendar transport failed; credential not logged; no retry") from None
    if body.get("code") != 0 or body.get("data") is None:
        raise RuntimeError(f"Provider code {body.get('code')}; no quota retry")
    raw = pd.DataFrame(body["data"]["items"], columns=body["data"]["fields"])
    if not set(fields).issubset(raw) or not raw.exchange.eq("SSE").all() or raw.cal_date.duplicated().any():
        raise ValueError("Unexpected calendar keys/schema")
    raw["date"] = pd.to_datetime(raw.cal_date.astype(str), format="%Y%m%d")
    raw["is_open"] = pd.to_numeric(raw.is_open, errors="raise")
    raw = raw.sort_values("date").reset_index(drop=True)
    expected = pd.date_range("2023-12-01", "2026-09-30")
    if not pd.DatetimeIndex(raw.date).equals(expected) or not raw.is_open.isin([0, 1]).all():
        raise ValueError("Incomplete calendar; missing closed days or truncated response")
    opened = raw.loc[raw.is_open.eq(1)].copy()
    previous = pd.to_datetime(opened.pretrade_date.astype(str), format="%Y%m%d")
    if not previous.iloc[1:].reset_index(drop=True).equals(opened.date.shift(1).iloc[1:].reset_index(drop=True)):
        raise ValueError("Provider previous-session links disagree")
    source = PROJECT_ROOT / "reports/research/overnight_unfiltered_universe_v1/daily_bars.parquet"
    source_hash = sha256(source)
    observed = pd.DatetimeIndex(pd.read_parquet(source, columns=["date"]).date.unique()).sort_values()
    market = pd.DatetimeIndex(opened.loc[opened.date.between(observed.min(), observed.max()), "date"])
    if not market.equals(observed):
        raise ValueError("Calendar and observed market dates differ; investigate before feature generation")
    path = output / "calendar.parquet"
    raw.to_parquet(path, index=False)
    if sha256(source) != source_hash:
        raise ValueError("Universe changed")
    summary = {"params": params, "calendar_rows": len(raw), "open_days": len(opened),
               "observed_market_dates_matched": len(observed), "dataset_sha256": sha256(path),
               "source_sha256": source_hash, "script_sha256": sha256(Path(__file__)),
               "retrieved_at": datetime.now(timezone.utc).isoformat(),
               "reference": "https://tushare.pro/document/2?doc_id=26",
               "limitation": "Current vendor calendar snapshot; historical publication/revision versions not proven; no price or outcome data fetched"}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
