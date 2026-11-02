"""Audit exact previous-market-day moneyflow availability without outcome labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT

AMOUNTS = [f"{side}_{size}_amount" for size in ("sm", "md", "lg", "elg") for side in ("buy", "sell")]


def main() -> None:
    output = PROJECT_ROOT / "reports/research/overnight_moneyflow_coverage_v1"
    output.mkdir(exist_ok=False)
    source = PROJECT_ROOT / "reports/research/tail_overnight/dataset.parquet"
    calendar_path = PROJECT_ROOT / "reports/research/overnight_same_day_market_limit_gated_v2/market_context.parquet"
    hashes = {str(p): sha256(p) for p in (source, calendar_path)}
    keys = pd.read_parquet(source, columns=["symbol", "date"])
    calendar = pd.read_parquet(calendar_path, columns=["date"]).sort_values("date")
    calendar["flow_date"] = calendar.date.shift(1)
    keys = keys.merge(calendar, on="date", how="left", validate="many_to_one")
    if keys.flow_date.isna().any() or not keys.flow_date.lt(keys.date).all():
        raise ValueError("Previous-market-day mapping unavailable")
    root = PROJECT_ROOT / "data/raw/moneyflow"
    output_rows, day_audit = [], []
    for date, group in keys.groupby("flow_date", sort=True):
        path = root / f"tushare_moneyflow_{date:%Y%m%d}.parquet"
        current = group.copy()
        current["status"] = "missing_file"
        raw_rows = 0
        if path.exists():
            hashes[str(path)] = sha256(path)
            raw = pd.read_parquet(path, columns=["ts_code", "trade_date", *AMOUNTS, "net_mf_amount"])
            raw_rows = len(raw)
            if (raw.ts_code.isna().any() or raw.ts_code.duplicated().any()
                    or not raw.trade_date.astype(str).eq(date.strftime("%Y%m%d")).all()):
                raise ValueError(f"Duplicate or unexpected moneyflow keys: {path.name}")
            values = raw[[*AMOUNTS, "net_mf_amount"]].apply(pd.to_numeric, errors="coerce")
            valid = np.isfinite(values).all(axis=1) & values[AMOUNTS].ge(0).all(axis=1)
            gross = values[AMOUNTS].sum(axis=1, min_count=len(AMOUNTS))
            raw["status"] = np.select([~valid, gross.le(0)], ["invalid_fields", "zero_gross"], default="ok")
            status = raw.set_index("ts_code").status
            current["status"] = current.symbol.map(status).fillna("missing_stock_row")
        current["exchange"] = current.symbol.str.rsplit(".", n=1).str[-1]
        output_rows.append(current)
        day_audit.append({"flow_date": str(date.date()), "decision_date": str(group.date.iloc[0].date()),
                          "file_present": path.exists(), "source_rows": raw_rows, "candidate_rows": len(current),
                          **{status: int(current.status.eq(status).sum()) for status in
                             ("ok", "missing_file", "missing_stock_row", "invalid_fields", "zero_gross")}})
    frame = pd.concat(output_rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    if len(frame) != len(keys) or frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Candidate keys changed")
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source changed during audit")
    frame.to_parquet(output / "key_availability.parquet", index=False)
    pd.DataFrame(day_audit).to_csv(output / "daily_coverage.csv", index=False)
    summary = {"candidate_rows": len(frame), "decision_days": int(frame.date.nunique()),
               "status": {str(k): int(v) for k, v in frame.status.value_counts().items()},
               "by_exchange": frame.groupby(["exchange", "status"]).size().reset_index(name="rows").to_dict("records"),
               "missing_file_dates": [row["flow_date"] for row in day_audit if not row["file_present"]],
               "timing": "Exact previous market day; no stale carry or missing-to-zero fill",
               "provider_docs": ["https://tushare.pro/document/2?doc_id=170", "https://tushare.pro/document/1?doc_id=108"],
               "scope": "Local source audit, not historical publication timestamp verification or a model result"}
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
