#!/usr/bin/env python3
"""Fetch point-in-time governance and capital-allocation evidence for 112 stocks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_json, atomic_write_parquet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        type=Path,
        default=PROJECT_ROOT / "reports/good_company_deep_20260809/universe_112.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/good_company_deep_20260809/sources/tushare_governance",
    )
    parser.add_argument("--cutoff-date", default="20260809")
    parser.add_argument("--price-date", default="20260807")
    parser.add_argument("--sleep", type=float, default=0.34)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def fetch_with_retry(call, *, retries: int, sleep_seconds: float) -> pd.DataFrame:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            frame = call()
            return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:  # remote service errors need a bounded retry
            error = exc
            time.sleep(sleep_seconds * (2 ** attempt))
    raise RuntimeError(str(error) if error else "unknown Tushare error")


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN is required")
    pro = ts.pro_api(token)
    universe = pd.read_csv(args.universe)[["ts_code", "name"]].sort_values("ts_code")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audits: list[pd.DataFrame] = []
    pledges: list[pd.DataFrame] = []
    dividends: list[pd.DataFrame] = []
    repurchases: list[pd.DataFrame] = []
    errors: list[dict] = []

    for index, row in enumerate(universe.itertuples(index=False), start=1):
        code = str(row.ts_code)
        calls = {
            "audit": lambda: pro.fina_audit(
                ts_code=code, start_date="20250101", end_date=args.cutoff_date
            ),
            "pledge": lambda: pro.pledge_stat(ts_code=code),
            "dividend": lambda: pro.dividend(ts_code=code),
            "repurchase": lambda: pro.repurchase(
                ts_code=code, start_date="20250101", end_date=args.cutoff_date
            ),
        }
        targets = {
            "audit": audits,
            "pledge": pledges,
            "dividend": dividends,
            "repurchase": repurchases,
        }
        for kind, call in calls.items():
            try:
                frame = fetch_with_retry(
                    call, retries=args.retries, sleep_seconds=args.sleep
                )
                if not frame.empty:
                    frame = frame.copy()
                    frame["queried_at_cutoff"] = args.cutoff_date
                    targets[kind].append(frame)
            except Exception as exc:
                errors.append(
                    {"ts_code": code, "name": row.name, "kind": kind, "error": str(exc)}
                )
            time.sleep(args.sleep)
        if index in {4, 25, 50, 75, 100, 112}:
            print(f"governance_progress={index}/112 errors={len(errors)}", flush=True)

    def persist(frames: list[pd.DataFrame], name: str) -> pd.DataFrame:
        output = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        if not output.empty:
            atomic_write_parquet(output, args.output_dir / f"{name}.parquet", index=False)
        return output

    audit = persist(audits, "fina_audit")
    pledge = persist(pledges, "pledge_stat")
    dividend = persist(dividends, "dividend")
    repurchase = persist(repurchases, "repurchase")
    manifest = {
        "source": "Tushare Pro",
        "source_policy": "point_in_time_governance_and_capital_allocation",
        "cutoff_date": args.cutoff_date,
        "price_date": args.price_date,
        "companies": len(universe),
        "audit_stocks": int(audit["ts_code"].nunique()) if not audit.empty else 0,
        "pledge_stocks": int(pledge["ts_code"].nunique()) if not pledge.empty else 0,
        "dividend_stocks": int(dividend["ts_code"].nunique()) if not dividend.empty else 0,
        "repurchase_stocks": int(repurchase["ts_code"].nunique()) if not repurchase.empty else 0,
        "errors": errors,
    }
    atomic_write_json(manifest, args.output_dir / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
