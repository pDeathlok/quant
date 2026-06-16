"""Refresh analyst forecast data from supplemental research sources.

The canonical historical source is Tushare report_rc. AkShare is used here as a
research supplement:

- akshare_em_snapshot fetches a current all-market consensus snapshot in a few
  paged calls. It is not historical, so report_date is the snapshot date.
- akshare_em_research fetches per-symbol Eastmoney research reports with
  publish dates and EPS/PE forecasts, so it can be used with as-of rules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/raw"
AUDIT_ROOT = RAW_DIR / "source_audit"
OUTPUT_PATH = RAW_DIR / "analyst_forecasts.parquet"


def load_symbols(limit: int | None = None) -> list[str]:
    path = RAW_DIR / "stock_basic.parquet"
    if not path.exists():
        raise RuntimeError(f"stock_basic not found: {path}")
    frame = pd.read_parquet(path, columns=["ts_code"])
    symbols = sorted(frame["ts_code"].dropna().astype(str).unique())
    if limit is not None:
        symbols = symbols[:limit]
    return symbols


def find_latest_datayes_audit() -> Path | None:
    paths = sorted(AUDIT_ROOT.glob("*_datayes_consensus/datayes_consensus_audit.csv"))
    return paths[-1] if paths else None


def load_retryable_datayes_symbols(
    audit_path: Path,
    error_pattern: str,
    include_no_data: bool = False,
) -> list[str]:
    audit = pd.read_csv(audit_path)
    required_columns = {"ts_code", "status", "error"}
    missing_columns = required_columns - set(audit.columns)
    if missing_columns:
        raise ValueError(f"{audit_path} missing required columns: {sorted(missing_columns)}")

    failed = audit[audit["status"].astype(str) != "success"].copy()
    if failed.empty:
        return []

    errors = failed["error"].fillna("").astype(str)
    retryable = errors.str.contains(error_pattern, flags=re.IGNORECASE, regex=True, na=False)
    if not include_no_data:
        retryable &= ~errors.str.contains(r"\bno data\b", flags=re.IGNORECASE, regex=True, na=False)

    symbols = failed.loc[retryable, "ts_code"].dropna().astype(str).unique().tolist()
    return sorted(symbols)


def normalize_a_code(ts_code: str) -> str:
    return ts_code.split(".", 1)[0]


def load_symbol_names() -> dict[str, str]:
    path = RAW_DIR / "stock_basic.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, columns=["ts_code", "name"])
    return dict(zip(frame["ts_code"].astype(str), frame["name"].astype(str), strict=False))


def to_ts_code(code: str) -> str:
    code = str(code).zfill(6)
    suffix = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{suffix}"


def parse_forecast_year(column: str) -> int | None:
    head = str(column).split("-", 1)[0]
    if len(head) == 4 and head.isdigit():
        return int(head)
    return None


def parse_number(value) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--", "None", "nan"}:
        return None
    multiplier = 1.0
    if text.endswith("亿"):
        multiplier = 100000000.0
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 10000.0
        text = text[:-1]
    if text.endswith("%"):
        multiplier = 0.01
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def append_and_dedupe(new_frame: pd.DataFrame) -> pd.DataFrame:
    if OUTPUT_PATH.exists():
        old_frame = pd.read_parquet(OUTPUT_PATH)
        combined = pd.concat([old_frame, new_frame], ignore_index=True)
    else:
        combined = new_frame
    if combined.empty:
        return combined
    combined = combined.drop_duplicates(
        ["source", "ts_code", "report_date", "org_name", "author_name", "forecast_year"],
        keep="last",
    )
    combined = combined.sort_values(["ts_code", "report_date", "forecast_year", "source"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    return combined


def refresh_akshare_em_snapshot() -> dict:
    import akshare as ak

    snapshot_date = pd.Timestamp(datetime.now().date())
    frame = ak.stock_profit_forecast_em(symbol="")
    rows: list[dict] = []
    for _, item in frame.iterrows():
        ts_code = to_ts_code(item["代码"])
        report_count = parse_number(item.get("研报数"))
        rating_buy = parse_number(item.get("机构投资评级(近六个月)-买入"))
        rating_overweight = parse_number(item.get("机构投资评级(近六个月)-增持"))
        for column in frame.columns:
            if "预测每股收益" not in str(column):
                continue
            year = str(column).split("预测", 1)[0]
            if not year.isdigit():
                continue
            rows.append(
                {
                    "source": "akshare_em_snapshot",
                    "ts_code": ts_code,
                    "name": item.get("名称"),
                    "report_date": snapshot_date,
                    "report_title": "东方财富全市场盈利预测快照",
                    "org_name": "eastmoney_consensus",
                    "author_name": None,
                    "forecast_year": int(year),
                    "quarter": f"{year}Q4",
                    "eps": parse_number(item.get(column)),
                    "pe": None,
                    "revenue": None,
                    "net_profit": None,
                    "target_price": None,
                    "report_count": report_count,
                    "rating_buy": rating_buy,
                    "rating_overweight": rating_overweight,
                    "snapshot_only": True,
                }
            )
    new_frame = pd.DataFrame(rows)
    combined = append_and_dedupe(new_frame)
    return {
        "source": "akshare_em_snapshot",
        "new_rows": int(len(new_frame)),
        "total_rows": int(len(combined)),
        "total_symbols": int(combined["ts_code"].nunique()) if not combined.empty else 0,
        "output_path": str(OUTPUT_PATH),
    }


def normalize_eastmoney_research(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, item in frame.iterrows():
        report_date = pd.to_datetime(item.get("日期"), errors="coerce")
        if pd.isna(report_date):
            continue
        for column in frame.columns:
            column_text = str(column)
            year = parse_forecast_year(column_text)
            if year is None or "盈利预测" not in column_text:
                continue
            if "收益" in column_text:
                metric = "eps"
            elif "市盈率" in column_text:
                metric = "pe"
            else:
                continue
            rows.append(
                {
                    "source": "akshare_em_research",
                    "ts_code": symbol,
                    "name": item.get("股票简称"),
                    "report_date": report_date,
                    "report_title": item.get("报告名称"),
                    "org_name": item.get("机构"),
                    "author_name": None,
                    "forecast_year": year,
                    "quarter": f"{year}Q4",
                    metric: parse_number(item.get(column)),
                    "industry": item.get("行业"),
                    "rating": item.get("东财评级"),
                    "pdf_url": item.get("报告PDF链接"),
                    "snapshot_only": False,
                }
            )
    if not rows:
        return pd.DataFrame()
    long_frame = pd.DataFrame(rows)
    id_columns = [
        "source",
        "ts_code",
        "name",
        "report_date",
        "report_title",
        "org_name",
        "author_name",
        "forecast_year",
        "quarter",
        "industry",
        "rating",
        "pdf_url",
        "snapshot_only",
    ]
    return (
        long_frame.groupby(id_columns, dropna=False, as_index=False)
        .agg({"eps": "first", "pe": "first"})
        .assign(revenue=None, net_profit=None, target_price=None, report_count=None)
    )


def refresh_akshare_em_research(limit: int | None, sleep_seconds: float, retries: int) -> dict:
    import akshare as ak

    existing_symbols: set[str] = set()
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH, columns=["source", "ts_code"])
        existing_symbols = set(
            existing.loc[existing["source"] == "akshare_em_research", "ts_code"].dropna().astype(str)
        )

    symbols = [symbol for symbol in load_symbols(limit=None) if symbol not in existing_symbols]
    if limit is not None:
        symbols = symbols[:limit]

    fetched: list[pd.DataFrame] = []
    audits: list[dict] = []
    for index, symbol in enumerate(symbols, start=1):
        status = "failed"
        rows = 0
        error = None
        for attempt in range(1, retries + 2):
            try:
                raw = ak.stock_research_report_em(symbol=normalize_a_code(symbol))
                normalized = normalize_eastmoney_research(symbol, raw)
                rows = len(normalized)
                if not normalized.empty:
                    fetched.append(normalized)
                status = "success"
                break
            except Exception as exc:
                error = str(exc)
                if attempt > retries:
                    break
                time.sleep(min(30.0, sleep_seconds * (2 ** attempt)))
        audits.append({"ts_code": symbol, "status": status, "rows": rows, "error": error})
        if index % 50 == 0 or index == len(symbols):
            ok = sum(1 for item in audits if item["status"] == "success")
            failed = sum(1 for item in audits if item["status"] == "failed")
            print(f"akshare_em_research progress: {index}/{len(symbols)} success={ok} failed={failed}", flush=True)
        time.sleep(sleep_seconds)

    new_frame = pd.concat(fetched, ignore_index=True) if fetched else pd.DataFrame()
    combined = append_and_dedupe(new_frame) if not new_frame.empty else (
        pd.read_parquet(OUTPUT_PATH) if OUTPUT_PATH.exists() else pd.DataFrame()
    )
    audit_dir = AUDIT_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_akshare_em_research"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "akshare_em_research_audit.csv"
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    return {
        "source": "akshare_em_research",
        "symbols_requested": len(symbols),
        "success": sum(1 for item in audits if item["status"] == "success"),
        "failed": sum(1 for item in audits if item["status"] == "failed"),
        "new_rows": int(len(new_frame)),
        "total_rows": int(len(combined)),
        "total_symbols": int(combined["ts_code"].nunique()) if not combined.empty else 0,
        "output_path": str(OUTPUT_PATH),
        "audit_path": str(audit_path),
    }


def refresh_akshare_cninfo_rating(
    start_date: str,
    end_date: str,
    sleep_seconds: float,
    retries: int,
    limit_days: int | None,
) -> dict:
    import akshare as ak

    start = pd.to_datetime(start_date, format="%Y%m%d", errors="raise")
    end = pd.to_datetime(end_date, format="%Y%m%d", errors="raise")
    dates: list[pd.Timestamp] = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    if limit_days is not None:
        dates = dates[:limit_days]

    fetched: list[pd.DataFrame] = []
    audits: list[dict] = []
    for index, date in enumerate(dates, start=1):
        date_text = date.strftime("%Y%m%d")
        status = "failed"
        rows = 0
        error = None
        for attempt in range(1, retries + 2):
            try:
                raw = ak.stock_rank_forecast_cninfo(date=date_text)
                if raw is None or raw.empty:
                    normalized = pd.DataFrame()
                else:
                    normalized = pd.DataFrame(
                        {
                            "source": "akshare_cninfo_rating",
                            "ts_code": raw["证券代码"].astype(str).map(to_ts_code),
                            "name": raw["证券简称"],
                            "report_date": pd.to_datetime(raw["发布日期"], errors="coerce"),
                            "report_title": "巨潮投资评级",
                            "org_name": raw["研究机构简称"],
                            "author_name": raw["研究员名称"],
                            "forecast_year": pd.NA,
                            "quarter": pd.NA,
                            "eps": None,
                            "pe": None,
                            "revenue": None,
                            "net_profit": None,
                            "target_price": pd.to_numeric(raw[["目标价格-下限", "目标价格-上限"]].mean(axis=1), errors="coerce"),
                            "report_count": None,
                            "rating_buy": None,
                            "rating_overweight": None,
                            "rating": raw["投资评级"],
                            "rating_change": raw["评级变化"],
                            "snapshot_only": False,
                        }
                    ).dropna(subset=["report_date"])
                rows = len(normalized)
                if not normalized.empty:
                    fetched.append(normalized)
                status = "success"
                break
            except Exception as exc:
                error = str(exc)
                if attempt > retries:
                    break
                time.sleep(min(30.0, sleep_seconds * (2 ** attempt)))
        audits.append({"date": date_text, "status": status, "rows": rows, "error": error})
        if index % 30 == 0 or index == len(dates):
            ok = sum(1 for item in audits if item["status"] == "success")
            failed = sum(1 for item in audits if item["status"] == "failed")
            print(f"akshare_cninfo_rating progress: {index}/{len(dates)} success={ok} failed={failed}", flush=True)
        time.sleep(sleep_seconds)

    new_frame = pd.concat(fetched, ignore_index=True) if fetched else pd.DataFrame()
    combined = append_and_dedupe(new_frame) if not new_frame.empty else (
        pd.read_parquet(OUTPUT_PATH) if OUTPUT_PATH.exists() else pd.DataFrame()
    )
    audit_dir = AUDIT_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_akshare_cninfo_rating"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "akshare_cninfo_rating_audit.csv"
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    return {
        "source": "akshare_cninfo_rating",
        "dates_requested": len(dates),
        "success": sum(1 for item in audits if item["status"] == "success"),
        "failed": sum(1 for item in audits if item["status"] == "failed"),
        "new_rows": int(len(new_frame)),
        "total_rows": int(len(combined)),
        "total_symbols": int(combined["ts_code"].nunique()) if not combined.empty else 0,
        "output_path": str(OUTPUT_PATH),
        "audit_path": str(audit_path),
    }


def parse_datayes_timestamp(value) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    try:
        return (
            pd.to_datetime(int(value), unit="ms", utc=True)
            .tz_convert("Asia/Shanghai")
            .tz_localize(None)
            .normalize()
        )
    except (TypeError, ValueError, OverflowError):
        return None


def playwright_cli_path() -> str:
    configured = os.environ.get("PWCLI")
    if configured:
        return configured
    codex_home = os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
    return str(Path(codex_home) / "skills/playwright/scripts/playwright_cli.sh")


def fetch_datayes_batch(symbols: list[str], timeout_seconds: int, concurrency: int) -> list[dict]:
    """Fetch Datayes analyst consensus through the logged-in Playwright page.

    The request intentionally runs in the browser context so that Datayes
    receives the same authenticated, same-origin cookies as the visible page.
    """

    tickers = [normalize_a_code(symbol) for symbol in symbols]
    concurrency = max(1, min(concurrency, len(tickers)))
    js_code = f"""
async () => {{
  const tickers = {json.dumps(tickers, ensure_ascii=False)};
  const out = [];
  let cursor = 0;
  async function fetchTicker(ticker) {{
    const item = {{ticker}};
    try {{
      const consensusResp = await fetch(
        `https://gw.datayes.com/rrp_adventure/web/stockModel/v3/consensus?ticker=${{ticker}}`,
        {{credentials: 'include'}}
      );
      item.consensus = await consensusResp.json();
      const summaryResp = await fetch(
        `https://gw.datayes.com/rrp_adventure/web/stockModel/v3/authorEvaluateSummary?ticker=${{ticker}}`,
        {{credentials: 'include'}}
      );
      item.summary = await summaryResp.json();
    }} catch (error) {{
      item.error = String(error && error.message ? error.message : error);
    }}
    out.push(item);
  }}
  async function worker() {{
    while (cursor < tickers.length) {{
      const ticker = tickers[cursor++];
      await fetchTicker(ticker);
    }}
  }}
  await Promise.all(Array.from({{length: {concurrency}}}, () => worker()));
  return JSON.stringify(out);
}}
"""
    result = subprocess.run(
        [playwright_cli_path(), "--raw", "eval", js_code],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    payload = json.loads(result.stdout)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected datayes payload: {type(payload).__name__}")
    return payload


def normalize_datayes_consensus_item(
    item: dict,
    symbol: str,
    name: str | None,
) -> tuple[pd.DataFrame, dict]:
    consensus = item.get("consensus") or {}
    summary = item.get("summary") or {}
    error = item.get("error")
    if error:
        return pd.DataFrame(), {"ts_code": symbol, "status": "failed", "rows": 0, "error": error}
    if consensus.get("code") != 1:
        return pd.DataFrame(), {
            "ts_code": symbol,
            "status": "failed",
            "rows": 0,
            "error": f"consensus code={consensus.get('code')} message={consensus.get('message')}",
        }
    rows: list[dict] = []
    summary_data = summary.get("data") if summary.get("code") == 1 else {}
    for raw in consensus.get("data") or []:
        report_date = parse_datayes_timestamp(raw.get("conDate"))
        forecast_year = raw.get("year")
        if report_date is None or forecast_year is None:
            continue
        rows.append(
            {
                "source": "datayes_consensus",
                "ts_code": symbol,
                "name": name,
                "report_date": report_date,
                "report_title": "萝卜投研分析师一致预期",
                "org_name": "datayes_consensus",
                "author_name": None,
                "forecast_year": int(forecast_year),
                "quarter": f"{int(forecast_year)}Q4",
                "eps": parse_number(raw.get("predictEps")),
                "pe": parse_number(raw.get("predictPe")),
                "revenue": parse_number(raw.get("predictIncome")),
                "net_profit": parse_number(raw.get("predictProfit")),
                "target_price": None,
                "report_count": parse_number(summary_data.get("reportNum")),
                "analyst_count": parse_number(summary_data.get("analystNum")),
                "evaluate_count": parse_number(summary_data.get("evaluateNum")),
                "actual_eps": parse_number(raw.get("eps")),
                "actual_revenue": parse_number(raw.get("income")),
                "actual_net_profit": parse_number(raw.get("profit")),
                "actual_pe": parse_number(raw.get("pe")),
                "is_predict": bool(raw.get("isPredict")),
                "snapshot_only": False,
            }
        )
    frame = pd.DataFrame(rows)
    return frame, {"ts_code": symbol, "status": "success", "rows": len(frame), "error": None}


def refresh_datayes_consensus(
    limit: int | None,
    batch_size: int,
    concurrency: int,
    sleep_seconds: float,
    retries: int,
    timeout_seconds: int,
    retry_failed_from: str | None = None,
    retry_error_pattern: str = r"Internal Server Error|timeout|missing response|ECONN|Target page|code=-24",
    include_no_data: bool = False,
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    retry_audit_path: Path | None = None
    if retry_failed_from:
        retry_audit_path = find_latest_datayes_audit() if retry_failed_from == "latest" else Path(retry_failed_from)
        if retry_audit_path is None or not retry_audit_path.exists():
            raise FileNotFoundError(f"Datayes audit file not found: {retry_failed_from}")
        symbols = load_retryable_datayes_symbols(retry_audit_path, retry_error_pattern, include_no_data)
    else:
        existing_symbols: set[str] = set()
        if OUTPUT_PATH.exists():
            existing = pd.read_parquet(OUTPUT_PATH, columns=["source", "ts_code"])
            existing_symbols = set(
                existing.loc[existing["source"] == "datayes_consensus", "ts_code"].dropna().astype(str)
            )
        symbols = [symbol for symbol in load_symbols(limit=None) if symbol not in existing_symbols]
    if limit is not None:
        symbols = symbols[:limit]
    names = load_symbol_names()

    fetched: list[pd.DataFrame] = []
    audits: list[dict] = []
    for offset in range(0, len(symbols), batch_size):
        batch_symbols = symbols[offset : offset + batch_size]
        batch_audits: list[dict] = []
        batch_frames: list[pd.DataFrame] = []
        last_error: str | None = None
        for attempt in range(1, retries + 2):
            try:
                payload = fetch_datayes_batch(batch_symbols, timeout_seconds, concurrency)
                by_ticker = {str(item.get("ticker")).zfill(6): item for item in payload}
                for symbol in batch_symbols:
                    item = by_ticker.get(normalize_a_code(symbol), {"error": "missing response"})
                    frame, audit = normalize_datayes_consensus_item(item, symbol, names.get(symbol))
                    batch_audits.append(audit)
                    if not frame.empty:
                        batch_frames.append(frame)
                break
            except Exception as exc:
                last_error = str(exc)
                if attempt > retries:
                    batch_audits = [
                        {"ts_code": symbol, "status": "failed", "rows": 0, "error": last_error}
                        for symbol in batch_symbols
                    ]
                    break
                time.sleep(min(30.0, sleep_seconds * (2 ** attempt)))
        audits.extend(batch_audits)
        if batch_frames:
            batch_frame = pd.concat(batch_frames, ignore_index=True)
            fetched.append(batch_frame)
            append_and_dedupe(batch_frame)
        done = min(offset + len(batch_symbols), len(symbols))
        ok = sum(1 for item in audits if item["status"] == "success")
        failed = sum(1 for item in audits if item["status"] == "failed")
        print(f"datayes_consensus progress: {done}/{len(symbols)} success={ok} failed={failed}", flush=True)
        time.sleep(sleep_seconds)

    new_frame = pd.concat(fetched, ignore_index=True) if fetched else pd.DataFrame()
    combined = pd.read_parquet(OUTPUT_PATH) if OUTPUT_PATH.exists() else pd.DataFrame()
    audit_dir = AUDIT_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_datayes_consensus"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "datayes_consensus_audit.csv"
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    return {
        "source": "datayes_consensus",
        "mode": "retry_failed" if retry_failed_from else "missing_symbols",
        "retry_failed_from": str(retry_audit_path) if retry_audit_path else None,
        "retry_error_pattern": retry_error_pattern if retry_failed_from else None,
        "include_no_data": include_no_data if retry_failed_from else None,
        "symbols_requested": len(symbols),
        "success": sum(1 for item in audits if item["status"] == "success"),
        "failed": sum(1 for item in audits if item["status"] == "failed"),
        "new_rows": int(len(new_frame)),
        "total_rows": int(len(combined)),
        "total_symbols": int(combined["ts_code"].nunique()) if not combined.empty else 0,
        "output_path": str(OUTPUT_PATH),
        "audit_path": str(audit_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh analyst forecast supplement data.")
    parser.add_argument(
        "--source",
        choices=["akshare_em_snapshot", "akshare_em_research", "akshare_cninfo_rating", "datayes_consensus"],
        required=True,
    )
    parser.add_argument("--start", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--retry-failed-from",
        default=None,
        help="For datayes_consensus, retry failed symbols from an audit CSV path, or 'latest'.",
    )
    parser.add_argument(
        "--retry-error-pattern",
        default=r"Internal Server Error|timeout|missing response|ECONN|Target page|code=-24",
        help="Regex for retryable Datayes audit errors. Defaults to transient/server failures.",
    )
    parser.add_argument(
        "--include-no-data",
        action="store_true",
        help="Also retry Datayes no-data failures. Off by default because source no-data is usually terminal.",
    )
    args = parser.parse_args()

    if args.source == "akshare_em_snapshot":
        result = refresh_akshare_em_snapshot()
    elif args.source == "akshare_em_research":
        result = refresh_akshare_em_research(args.limit, args.sleep, args.retries)
    elif args.source == "akshare_cninfo_rating":
        result = refresh_akshare_cninfo_rating(args.start, args.end, args.sleep, args.retries, args.limit_days)
    else:
        result = refresh_datayes_consensus(
            args.limit,
            args.batch_size,
            args.concurrency,
            args.sleep,
            args.retries,
            args.timeout,
            args.retry_failed_from,
            args.retry_error_pattern,
            args.include_no_data,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
