"""Read-only isolated index history collection for registry factor expansion."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import httpx
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT
from quant.features.active_market_value import KEY_INDEX_SPECS


def main() -> None:
    output = PROJECT_ROOT / "reports/research/right_t012_index_sources_v1"
    output.mkdir(exist_ok=False)
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        for line in (PROJECT_ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not token:
        raise RuntimeError("Credential unavailable")
    records = []
    with httpx.Client(timeout=25) as client:
        for spec in KEY_INDEX_SPECS:
            for endpoint in ["index_daily", "index_dailybasic"]:
                params = dict(ts_code=spec.ts_code, start_date="20180101", end_date="20260904")
                try:
                    response = client.post("https://api.tushare.pro", json=dict(api_name=endpoint, token=token, params=params))
                    response.raise_for_status()
                    body = response.json()
                except (httpx.HTTPError, ValueError):
                    records.append(dict(index=spec.ts_code, endpoint=endpoint, status="transport_failed_no_retry"))
                    continue
                if body.get("code") != 0 or body.get("data") is None:
                    records.append(dict(index=spec.ts_code, endpoint=endpoint, status="provider_unavailable", code=body.get("code")))
                    continue
                frame = pd.DataFrame(body["data"]["items"], columns=body["data"]["fields"])
                if frame.empty:
                    records.append(dict(index=spec.ts_code, endpoint=endpoint, status="no_rows"))
                    continue
                if frame.duplicated(["ts_code", "trade_date"]).any() or not frame.ts_code.eq(spec.ts_code).all():
                    raise ValueError("Unexpected index keys")
                path = output / f"{spec.ts_code}_{endpoint}.parquet"
                frame.to_parquet(path, index=False)
                records.append(dict(index=spec.ts_code, endpoint=endpoint, status="downloaded_not_coverage_verified",
                                    rows=len(frame), first=str(frame.trade_date.min()), last=str(frame.trade_date.max()), sha256=sha256(path)))
                print(f"{endpoint} {spec.ts_code}: {len(frame)} rows", flush=True)
    (output / "collection.json").write_text(json.dumps(dict(retrieved_at=datetime.now(timezone.utc).isoformat(),
        records=records, status="Collection only; coverage/units and PIT availability still require checks"), indent=2))


if __name__ == "__main__":
    main()
