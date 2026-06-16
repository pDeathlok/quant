"""Refresh missing Tushare financial datasets by stock symbol.

This is a research/data-maintenance helper for long-horizon strategies. It
appends fetched rows into data/raw/*.parquet and writes an audit CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import tushare as ts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/raw"
AUDIT_ROOT = RAW_DIR / "source_audit"


DATASETS = {
    "fina_indicator": {
        "path": RAW_DIR / "fina_indicator.parquet",
        "method": "fina_indicator",
        "fields": (
            "ts_code,ann_date,end_date,eps,roe,roe_waa,roe_dt,roa,"
            "netprofit_margin,grossprofit_margin,debt_to_assets,current_ratio,"
            "quick_ratio,ar_turn,inv_turn,assets_turn,profit_to_gr,basic_eps_yoy,or_yoy"
        ),
        "dedupe": ["ts_code", "ann_date", "end_date"],
    },
    "income": {
        "path": RAW_DIR / "income.parquet",
        "method": "income",
        "fields": (
            "ts_code,ann_date,end_date,report_type,revenue,operate_profit,"
            "total_profit,n_income,n_income_attr_p,total_revenue,income_tax,minority_gain"
        ),
        "dedupe": ["ts_code", "ann_date", "end_date", "report_type"],
    },
    "cashflow": {
        "path": RAW_DIR / "cashflow.parquet",
        "method": "cashflow",
        "fields": (
            "ts_code,ann_date,end_date,report_type,n_cashflow_act,"
            "c_pay_acq_const_fiolta,net_profit"
        ),
        "dedupe": ["ts_code", "ann_date", "end_date", "report_type"],
    },
    "report_rc": {
        "path": RAW_DIR / "report_rc.parquet",
        "method": "report_rc",
        "fields": (
            "ts_code,name,report_date,report_title,report_type,org_name,"
            "author_name,quarter,op_rt,op_pr,tp,np,eps,pe,rd"
        ),
        "dedupe": ["ts_code", "report_date", "org_name", "author_name", "quarter"],
        "sort": ["ts_code", "report_date"],
        "date_column": "report_date",
    },
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_symbols() -> list[str]:
    path = RAW_DIR / "stock_basic.parquet"
    if not path.exists():
        raise RuntimeError(f"stock_basic not found: {path}")
    frame = pd.read_parquet(path, columns=["ts_code"])
    return sorted(frame["ts_code"].dropna().astype(str).unique())


def existing_symbols(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_parquet(path, columns=["ts_code"])
    return set(frame["ts_code"].dropna().astype(str).unique())


def fetch_symbol(pro, dataset: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    spec = DATASETS[dataset]
    method = getattr(pro, spec["method"])
    return method(
        ts_code=symbol,
        start_date=start_date,
        end_date=end_date,
        fields=spec["fields"],
    )


def fetch_report_rc_page(pro, start_date: str, end_date: str, page_size: int, offset: int) -> pd.DataFrame:
    spec = DATASETS["report_rc"]
    return pro.report_rc(
        start_date=start_date,
        end_date=end_date,
        limit=page_size,
        offset=offset,
        fields=spec["fields"],
    )


def sort_dataset_frame(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    sort_columns = spec.get("sort") or ["ts_code", "ann_date", "end_date"]
    existing = [column for column in sort_columns if column in frame.columns]
    if not existing:
        return frame
    ascending = [True] + [False] * (len(existing) - 1)
    return frame.sort_values(existing, ascending=ascending)


def rate_limit_sleep_seconds(error: str | None, default_sleep: float) -> float | None:
    if not error or "频率超限" not in error:
        return None
    if "1次/小时" in error:
        return 3700.0
    if "1次/分钟" in error:
        return max(65.0, default_sleep)
    return max(65.0, default_sleep)


def merge_and_write_dataset(spec: dict, new_frame: pd.DataFrame) -> pd.DataFrame:
    if spec["path"].exists():
        old_frame = pd.read_parquet(spec["path"])
        combined = pd.concat([old_frame, new_frame], ignore_index=True) if not new_frame.empty else old_frame
    else:
        combined = new_frame
    if not combined.empty:
        combined = combined.drop_duplicates(spec["dedupe"], keep="last")
        combined = sort_dataset_frame(combined, spec)
        spec["path"].parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(spec["path"], index=False)
    return combined


def refresh_dataset(
    dataset: str,
    start_date: str,
    end_date: str,
    missing_only: bool,
    sleep_seconds: float,
    retries: int,
    limit: int | None,
    flush_every: int,
) -> dict:
    spec = DATASETS[dataset]
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    pro = ts.pro_api(token)

    symbols = load_symbols()
    if missing_only:
        have = existing_symbols(spec["path"])
        symbols = [symbol for symbol in symbols if symbol not in have]
    if limit:
        symbols = symbols[:limit]

    fetched: list[pd.DataFrame] = []
    audits: list[dict] = []
    flushed_rows = 0
    for index, symbol in enumerate(symbols, start=1):
        status = "failed"
        rows = 0
        error = None
        for attempt in range(1, retries + 2):
            try:
                frame = fetch_symbol(pro, dataset, symbol, start_date, end_date)
                if frame is None:
                    frame = pd.DataFrame()
                rows = len(frame)
                if not frame.empty:
                    fetched.append(frame)
                status = "success"
                break
            except Exception as exc:
                error = str(exc)
                wait_seconds = rate_limit_sleep_seconds(error, sleep_seconds)
                if wait_seconds is not None and attempt <= retries:
                    print(
                        f"{dataset} rate limited for {symbol}; sleeping {wait_seconds:.0f}s before retry",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                    continue
                if attempt > retries:
                    break
                time.sleep(min(30.0, sleep_seconds * (2 ** attempt)))
        if status == "success" and fetched and flush_every > 0 and len(fetched) >= flush_every:
            flushed = pd.concat(fetched, ignore_index=True)
            merge_and_write_dataset(spec, flushed)
            flushed_rows += len(flushed)
            fetched = []
        audits.append({"ts_code": symbol, "status": status, "rows": rows, "error": error})
        if index % 100 == 0 or index == len(symbols):
            ok = sum(1 for item in audits if item["status"] == "success")
            failed = sum(1 for item in audits if item["status"] == "failed")
            print(f"{dataset} progress: {index}/{len(symbols)} success={ok} failed={failed}", flush=True)
        time.sleep(sleep_seconds)

    new_frame = pd.concat(fetched, ignore_index=True) if fetched else pd.DataFrame()
    combined = merge_and_write_dataset(spec, new_frame)

    audit_dir = AUDIT_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset}"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{dataset}_audit.csv"
    pd.DataFrame(audits).to_csv(audit_path, index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "start_date": start_date,
        "end_date": end_date,
        "missing_only": missing_only,
        "symbols_requested": len(symbols),
        "success": sum(1 for item in audits if item["status"] == "success"),
        "failed": sum(1 for item in audits if item["status"] == "failed"),
        "new_rows": int(flushed_rows + len(new_frame)),
        "total_rows": int(len(combined)),
        "total_symbols": int(combined["ts_code"].nunique()) if not combined.empty else 0,
        "output_path": str(spec["path"]),
        "audit_path": str(audit_path),
    }
    (audit_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def refresh_report_rc_by_pages(
    start_date: str,
    end_date: str,
    sleep_seconds: float,
    retries: int,
    page_size: int,
    start_offset: int,
    max_pages: int | None,
) -> dict:
    spec = DATASETS["report_rc"]
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    pro = ts.pro_api(token)

    audits: list[dict] = []
    total_new_rows = 0
    offset = start_offset
    page = 0
    combined = pd.read_parquet(spec["path"]) if spec["path"].exists() else pd.DataFrame()
    while True:
        if max_pages is not None and page >= max_pages:
            break
        page += 1
        status = "failed"
        rows = 0
        error = None
        frame = pd.DataFrame()
        for attempt in range(1, retries + 2):
            try:
                frame = fetch_report_rc_page(pro, start_date, end_date, page_size, offset)
                if frame is None:
                    frame = pd.DataFrame()
                rows = len(frame)
                status = "success"
                break
            except Exception as exc:
                error = str(exc)
                wait_seconds = rate_limit_sleep_seconds(error, sleep_seconds)
                if wait_seconds is not None and attempt <= retries:
                    print(
                        f"report_rc page offset={offset} rate limited; sleeping {wait_seconds:.0f}s before retry",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                    continue
                if attempt > retries:
                    break
                time.sleep(min(30.0, sleep_seconds * (2 ** attempt)))
        audits.append({"page": page, "offset": offset, "status": status, "rows": rows, "error": error})
        print(f"report_rc page={page} offset={offset} status={status} rows={rows}", flush=True)
        if status != "success":
            break
        if frame.empty:
            break
        combined = merge_and_write_dataset(spec, frame)
        total_new_rows += rows
        if rows < page_size:
            break
        offset += page_size
        time.sleep(sleep_seconds)

    audit_dir = AUDIT_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_report_rc_pages"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "report_rc_pages_audit.csv"
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "report_rc",
        "mode": "pages",
        "start_date": start_date,
        "end_date": end_date,
        "page_size": page_size,
        "start_offset": start_offset,
        "pages_requested": len(audits),
        "success": sum(1 for item in audits if item["status"] == "success"),
        "failed": sum(1 for item in audits if item["status"] == "failed"),
        "new_rows_before_dedupe": int(total_new_rows),
        "total_rows": int(len(combined)),
        "total_symbols": int(combined["ts_code"].nunique()) if not combined.empty else 0,
        "output_path": str(spec["path"]),
        "audit_path": str(audit_path),
    }
    (audit_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Tushare financial datasets by symbol.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--all", action="store_true", help="Fetch all symbols instead of only missing symbols.")
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=None)
    parser.add_argument("--by-symbol", action="store_true", help="For report_rc, use old per-symbol fallback mode.")
    parser.add_argument("--page-size", type=int, default=3000)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    if args.dataset == "report_rc" and not args.by_symbol:
        result = refresh_report_rc_by_pages(
            start_date=args.start,
            end_date=args.end,
            sleep_seconds=args.sleep if args.sleep is not None else 1.0,
            retries=args.retries,
            page_size=args.page_size,
            start_offset=args.start_offset,
            max_pages=args.max_pages,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = refresh_dataset(
        dataset=args.dataset,
        start_date=args.start,
        end_date=args.end,
        missing_only=not args.all,
        sleep_seconds=args.sleep if args.sleep is not None else (65.0 if args.dataset == "report_rc" else 0.12),
        retries=args.retries,
        limit=args.limit,
        flush_every=args.flush_every if args.flush_every is not None else (1 if args.dataset == "report_rc" else 500),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
