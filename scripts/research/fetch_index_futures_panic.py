"""Snapshot AKShare QVIX and probe Tushare option access without logging secrets."""
import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import akshare
import akshare.index.index_option_qvix as qvix
import pandas as pd
import requests
from dotenv import load_dotenv

OUT = Path("data/research/index_futures/panic_factors/sources")
SERIES = ("50etf", "300etf", "500etf", "300index", "1000index", "50index", "cyb", "kcb")


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = dict(retrieved_at=datetime.now(timezone.utc).isoformat(), akshare_version=akshare.__version__, qvix=[], tushare=[])
    raw_path = OUT / "optbbs_k.csv"
    if not raw_path.exists():
        for scheme in ("https", "http"):
            url = f"{scheme}://1.optbbs.com/d/csv/d/k.csv"
            try:
                response = requests.get(url, timeout=(15, 150))
                response.raise_for_status()
                raw = pd.read_csv(io.BytesIO(response.content), encoding="gbk")
                if raw.shape[1] < 83 or len(raw) < 100:
                    raise ValueError("Unexpected QVIX source dimensions")
                raw_path.write_bytes(response.content)
                audit["qvix_url"] = url
                break
            except (requests.RequestException, ValueError) as exc:
                audit["qvix"].append(dict(url=url, status=type(exc).__name__))
    if raw_path.exists():
        raw = pd.read_csv(raw_path, encoding="gbk")
        audit["qvix_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        # Use installed AKShare's own column mapping, with a bounded downloader.
        original = qvix.__get_optbbs_daily
        qvix.__get_optbbs_daily = lambda: raw.copy()
        try:
            for series in SERIES:
                frame = getattr(qvix, f"index_option_{series}_qvix")()
                frame["trade_date"] = pd.to_datetime(frame.date).dt.strftime("%Y%m%d")
                frame = frame.dropna(subset=["trade_date", "close"])
                frame = frame[frame.close.gt(0)].sort_values("trade_date")
                duplicates = frame.trade_date.duplicated(keep=False)
                duplicate_dates = sorted(frame.loc[duplicates, "trade_date"].unique())
                # Conflicting source dates cannot be resolved by choosing the last row.
                frame.loc[duplicates].to_csv(OUT / f"quarantine_{series}.csv", index=False)
                frame = frame.loc[~duplicates].copy()
                frame.to_parquet(OUT / f"qvix_{series}.parquet", index=False)
                audit["qvix"].append(dict(series=series, rows=len(frame), first=frame.trade_date.min(), last=frame.trade_date.max(), quarantined_dates=duplicate_dates))
                print("QVIX", audit["qvix"][-1], flush=True)
        finally:
            qvix.__get_optbbs_daily = original
    load_dotenv()
    token = os.environ.get("TUSHARE_TOKEN")
    probes = [("opt_basic", dict(exchange="SSE", limit=1000)),
              ("opt_basic", dict(exchange="CFFEX", limit=1000)),
              ("opt_daily", dict(exchange="SSE", trade_date="20261106")),
              ("opt_daily", dict(exchange="CFFEX", trade_date="20261106"))]
    for api, params in probes:
        record = dict(api=api, params=params)
        if not token:
            record["status"] = "token_unavailable"
        else:
            try:
                response = requests.post("https://api.tushare.pro", json=dict(api_name=api, token=token, params=params), timeout=30)
                response.raise_for_status()
                payload = response.json()
                record["code"] = payload.get("code")
                if payload.get("code") == 0:
                    frame = pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])
                    record.update(status="ok", rows=len(frame), fields=list(frame.columns))
                    frame.to_parquet(OUT / f"tushare_{api}_{params['exchange']}_probe.parquet", index=False)
                else:
                    record["status"] = "api_rejected"
                    record["message"] = str(payload.get("msg", "")).replace(token, "[redacted]")[:250]
            except (requests.RequestException, ValueError) as exc:
                record["status"] = type(exc).__name__
        audit["tushare"].append(record)
        print("Tushare", record, flush=True)
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    fetch()
